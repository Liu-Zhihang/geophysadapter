#!/usr/bin/env python3
"""Matched Prithvi-EO-2.0 visual/Terrain-v2 trainer for Sen12 LOGO-5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_sen12_xdomain_geophysadapter as protocol  # noqa: E402
from sen12_prithvi_v2 import (  # noqa: E402
    BoundedTerrainAdapterV2,
    PrithviEO2ChangeModel,
    load_prithvi_encoder,
)
from sen12_terrain_v2 import (  # noqa: E402
    CURRENT_SCALE_GROUPS,
    NATIVE_TERRAIN_V2_NAMES,
    NATIVE_TERRAIN_V2_SCALE_GROUPS,
    SupportOnlyMultiScaleTerrainPyramid,
    TerrainScaleGroups,
)


BASE_H5 = PROJECT_ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5"
OPTICAL_H5 = PROJECT_ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_prithvi_4t6b_p128.h5"
TERRAIN_H5 = PROJECT_ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_native_terrain_v2_p128.h5"
SPLIT_CSV = PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv"
OUT_ROOT = PROJECT_ROOT / "experiments/revision2026/sen12_prithvi_terrain_v2"

PILD_COMMON_TERRAIN9_NAMES = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "curvature_laplacian",
    "tpi_90m",
    "tpi_300m",
    "roughness_90m",
    "local_relief_300m",
)
SEN12_COMMON_TERRAIN9_NAMES = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "curvature_laplacian",
    "tpi_90m",
    "tpi_300m",
    "roughness_30m",
    "local_relief_300m",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def resolve_terrain_contract(
    terrain_h5: Path,
) -> tuple[tuple[str, ...], TerrainScaleGroups, str]:
    """Resolve only audited Terrain schemas; unknown channel layouts are fatal."""
    with h5py.File(terrain_h5, "r") as handle:
        if "terrain_names" not in handle:
            raise KeyError(f"terrain_names missing from {terrain_h5}")
        names = tuple(decode(handle["terrain_names"][:]))
        channels = int(handle["terrain"].shape[1])
    if channels != len(names):
        raise RuntimeError(
            f"Terrain channel/name mismatch: channels={channels}, names={len(names)}"
        )
    if names == tuple(NATIVE_TERRAIN_V2_NAMES):
        return names, NATIVE_TERRAIN_V2_SCALE_GROUPS, "sen12_native_terrain17_v2"
    if names == PILD_COMMON_TERRAIN9_NAMES:
        return names, CURRENT_SCALE_GROUPS, "pild_common_terrain9_v2"
    if names == SEN12_COMMON_TERRAIN9_NAMES:
        return names, CURRENT_SCALE_GROUPS, "sen12_common_terrain9_v1"
    raise RuntimeError(f"unsupported Terrain contract in {terrain_h5}: {names}")


def stratified_fraction_subset(
    sample_ids: Sequence[str],
    rows: dict[str, dict[str, str]],
    fraction: float,
    seed: int,
) -> list[str]:
    """Return nested, region-stratified train subsets without reading labels."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if fraction == 1.0:
        return list(sample_ids)
    grouped: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        group = rows[sample_id]["spatial_supergroup"]
        grouped.setdefault(group, []).append(sample_id)
    selected = []
    for group, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda value: hashlib.sha256(
                f"{seed}|sen12-data-scaling-v1|{group}|{value}".encode()
            ).hexdigest(),
        )
        keep = max(1, int(math.ceil(len(ordered) * fraction)))
        selected.extend(ordered[:keep])
    return sorted(selected)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != expected:
        raise RuntimeError(
            f"trainable state mismatch: missing={sorted(expected-set(state))}, extra={sorted(set(state)-expected)}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def tensor_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def module_parameter_sha256(module: nn.Module) -> str:
    return tensor_dict_sha256(
        {name: parameter.detach() for name, parameter in module.named_parameters()}
    )


def validate_sidecars(base_h5: Path, optical_h5: Path, terrain_h5: Path) -> tuple[list[str], list[str]]:
    with h5py.File(base_h5, "r") as base, h5py.File(optical_h5, "r") as optical, h5py.File(terrain_h5, "r") as terrain:
        base_ids = decode(base["sample_id"][:])
        optical_ids = decode(optical["sample_id"][:])
        terrain_ids = decode(terrain["sample_id"][:])
        if not (base_ids == optical_ids == terrain_ids):
            raise RuntimeError("base/Prithvi/Terrain-v2 sample identity or ordering differs")
        if int(optical.attrs.get("complete", 0)) != 1 or int(terrain.attrs.get("complete", 0)) != 1:
            raise RuntimeError("Prithvi or Terrain-v2 H5 is not complete")
        if optical["optical"].shape != (len(base_ids), 6, 4, 128, 128):
            raise RuntimeError(f"unexpected optical shape {optical['optical'].shape}")
        terrain_names, _, _ = resolve_terrain_contract(terrain_h5)
        if terrain["terrain"].shape != (len(base_ids), len(terrain_names), 128, 128):
            raise RuntimeError(f"unexpected Terrain-v2 shape {terrain['terrain'].shape}")
        event_ids = decode(base["physical_event_id"][:])
    return base_ids, event_ids


def estimate_terrain_stats(
    terrain_h5: Path, all_ids: Sequence[str], train_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    index = {sample_id: position for position, sample_id in enumerate(all_ids)}
    with h5py.File(terrain_h5, "r") as handle:
        channels = int(handle["terrain"].shape[1])
        sums = np.zeros(channels, dtype=np.float64)
        squares = np.zeros(channels, dtype=np.float64)
        counts = np.zeros(channels, dtype=np.float64)
        for sample_id in train_ids:
            position = index[sample_id]
            value = np.asarray(handle["terrain"][position], dtype=np.float64)
            valid = np.asarray(handle["terrain_valid"][position], dtype=bool)
            keep = np.broadcast_to(valid, value.shape)
            safe = np.where(keep, value, 0.0)
            sums += safe.sum(axis=(1, 2))
            squares += np.square(safe).sum(axis=(1, 2))
            counts += keep.sum(axis=(1, 2))
    mean = sums / np.maximum(counts, 1.0)
    variance = squares / np.maximum(counts, 1.0) - np.square(mean)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)


class PrithviTerrainDataset(Dataset):
    def __init__(
        self,
        base_h5: Path,
        optical_h5: Path,
        terrain_h5: Path,
        all_ids: Sequence[str],
        event_ids: Sequence[str],
        rows: dict[str, dict[str, str]],
        sample_ids: Sequence[str],
        terrain_mean: np.ndarray,
        terrain_std: np.ndarray,
        seed: int,
        donor_pool: Sequence[str],
        include_terrain: bool,
    ) -> None:
        self.paths = (base_h5, optical_h5, terrain_h5)
        self.files: tuple[h5py.File, h5py.File, h5py.File] | None = None
        self.rows = rows
        self.sample_ids = list(sample_ids)
        self.index = {sample_id: position for position, sample_id in enumerate(all_ids)}
        self.indices = [self.index[sample_id] for sample_id in self.sample_ids]
        self.event_ids = [event_ids[self.index[sample_id]] for sample_id in self.sample_ids]
        self.regions = [rows[sample_id]["spatial_supergroup"] for sample_id in self.sample_ids]
        self.mean = terrain_mean[:, None, None]
        self.std = terrain_std[:, None, None]
        self.include_terrain = bool(include_terrain)
        donor_pool = sorted(donor_pool)
        self.donor_ids = []
        self.donor_indices = []
        self.donor_regions = []
        for sample_id, region in zip(self.sample_ids, self.regions, strict=True):
            if not self.include_terrain:
                self.donor_ids.append("")
                self.donor_indices.append(self.index[sample_id])
                self.donor_regions.append("")
                continue
            candidates = [item for item in donor_pool if rows[item]["spatial_supergroup"] != region]
            if not candidates:
                raise RuntimeError(f"no other-region donor for {region}")
            token = hashlib.sha256(f"{seed}|{sample_id}|terrain-v2-donor".encode()).digest()
            donor = candidates[int.from_bytes(token[:8], "big") % len(candidates)]
            self.donor_ids.append(donor)
            self.donor_indices.append(self.index[donor])
            self.donor_regions.append(rows[donor]["spatial_supergroup"])

    def _open(self) -> tuple[h5py.File, h5py.File, h5py.File]:
        if self.files is None:
            self.files = tuple(h5py.File(path, "r") for path in self.paths)  # type: ignore[assignment]
        return self.files

    def __len__(self) -> int:
        return len(self.indices)

    def _terrain(self, handle: h5py.File, index: int) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(handle["terrain"][index], dtype=np.float32)
        valid = np.asarray(handle["terrain_valid"][index], dtype=np.float32)
        normalized = ((value - self.mean) / self.std) * valid
        quality = valid
        if "q_T" in handle:
            source_quality = np.asarray(handle["q_T"][index], dtype=np.float32)
            if source_quality.ndim == 0:
                source_quality = np.full_like(valid, float(source_quality))
            elif source_quality.ndim == 2:
                source_quality = source_quality[None]
            if source_quality.shape != valid.shape:
                raise RuntimeError(
                    f"q_T shape {source_quality.shape} does not match terrain_valid {valid.shape}"
                )
            quality = valid * np.clip(source_quality, 0.0, 1.0)
        return normalized.astype(np.float32), quality.astype(np.float32)

    def __getitem__(self, position: int) -> dict[str, Any]:
        base, optical, terrain = self._open()
        index = self.indices[position]
        donor_index = self.donor_indices[position]
        optical_value = np.asarray(optical["optical"][index], dtype=np.float32)
        coordinates = np.concatenate(
            (
                np.asarray(optical["temporal_coords"][index], dtype=np.float32).reshape(-1),
                np.asarray(optical["location_coords"][index], dtype=np.float32),
            )
        )
        mask = np.asarray(base["mask"][index], dtype=np.float32)
        visual_valid = np.asarray(base["valid_mask"][index], dtype=np.float32)
        optical_valid = np.asarray(optical["optical_valid"][index], dtype=np.float32)
        valid = visual_valid * optical_valid
        item = {
            "pre": torch.from_numpy(optical_value),
            "post": torch.from_numpy(coordinates),
            "mask": torch.from_numpy(mask),
            "valid": torch.from_numpy(valid),
            "sample_id": self.sample_ids[position],
            "event_id": self.event_ids[position],
            "region": self.regions[position],
            "donor_sample_id": self.donor_ids[position],
            "donor_region": self.donor_regions[position],
        }
        if self.include_terrain:
            terrain_value, q_t = self._terrain(terrain, index)
            donor_value, donor_q_t = self._terrain(terrain, donor_index)
            item.update(
                {
                    "terrain": torch.from_numpy(terrain_value),
                    "q_t": torch.from_numpy(q_t),
                    "donor_terrain": torch.from_numpy(donor_value),
                    "donor_q_t": torch.from_numpy(donor_q_t),
                }
            )
        return item


def split_coordinates(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 2 or value.shape[1] != 10:
        raise ValueError(f"expected flattened coordinates [B,10], got {tuple(value.shape)}")
    return value[:, :8].reshape(-1, 4, 2), value[:, 8:]


class PrithviVisualCompat(nn.Module):
    def __init__(self, model: PrithviEO2ChangeModel) -> None:
        super().__init__()
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, optical: torch.Tensor, coordinates: torch.Tensor):
        temporal, location = split_coordinates(coordinates)
        output = self.model(optical, temporal, location)
        return output["logits"], output["visual_feature"]


class PrithviTerrainCompat(nn.Module):
    def __init__(
        self,
        visual: PrithviVisualCompat,
        alpha_max: float,
        uncertainty_cutoff: float,
        uncertainty_temperature: float,
        terrain_channels: int,
        terrain_scale_groups: TerrainScaleGroups,
    ) -> None:
        super().__init__()
        self.visual = visual
        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        self.adapter = BoundedTerrainAdapterV2(
            terrain_channels,
            128,
            terrain_scale_groups,
            alpha_max=alpha_max,
            uncertainty_cutoff=uncertainty_cutoff,
            uncertainty_temperature=uncertainty_temperature,
        )
        self.correction_scale = 1.0

    def set_correction_scale(self, scale: float) -> None:
        if not 0.0 <= scale <= 1.0:
            raise ValueError("correction scale must be in [0, 1]")
        self.correction_scale = float(scale)

    @staticmethod
    def uncertainty(logits: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits.detach())
        return 1.0 - 2.0 * torch.abs(probability - 0.5)

    def support_forward(self, visual_logits, visual_feature, uncertainty, terrain, q_t):
        _, raw = self.adapter(visual_logits, visual_feature, uncertainty, terrain, q_t)
        correction = raw["correction"] * self.correction_scale
        logits = visual_logits + correction
        diagnostics = {
            "visual_logits": visual_logits,
            "visual_feature": visual_feature,
            "uncertainty": uncertainty,
            "gate": raw["visual_reliability_gate"],
            "bounded_residual": raw["bounded_terrain_direction"],
            "correction": correction,
            "q_t": raw["q_t"],
        }
        return logits, diagnostics

    def forward(self, optical, coordinates, terrain, q_t):
        with torch.no_grad():
            visual_logits, visual_feature = self.visual(optical, coordinates)
        return self.support_forward(
            visual_logits, visual_feature, self.uncertainty(visual_logits), terrain, q_t
        )

    def forward_controls(self, optical, coordinates, terrain, q_t, donor_terrain, donor_q_t):
        with torch.no_grad():
            visual_logits, visual_feature = self.visual(optical, coordinates)
        uncertainty = self.uncertainty(visual_logits)
        zero = torch.zeros_like(visual_logits)
        base = {
            "visual_logits": visual_logits,
            "visual_feature": visual_feature,
            "uncertainty": uncertainty,
            "gate": zero,
            "bounded_residual": zero,
            "correction": zero,
            "q_t": zero,
        }
        outputs = {"visual_anchor": (visual_logits, base)}
        controls = {
            "aligned": (terrain, q_t),
            "zero": (torch.zeros_like(terrain), torch.ones_like(q_t)),
            "roll32": (torch.roll(terrain, (32, 32), (-2, -1)), torch.roll(q_t, (32, 32), (-2, -1))),
            "roll64": (torch.roll(terrain, (64, 64), (-2, -1)), torch.roll(q_t, (64, 64), (-2, -1))),
            "other_region_donor": (donor_terrain, donor_q_t),
        }
        for name, (support, quality) in controls.items():
            outputs[name] = self.support_forward(
                visual_logits, visual_feature, uncertainty, support, quality
            )
        return outputs


class PrithviTerrainVetoCompat(nn.Module):
    """Terrain-only FP veto with an explicit TP-preservation contract."""

    def __init__(
        self,
        visual: PrithviVisualCompat,
        alpha_max: float,
        decision_threshold: float,
        terrain_channels: int,
        terrain_scale_groups: TerrainScaleGroups,
        interaction_gate: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.visual = visual
        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        self.terrain = SupportOnlyMultiScaleTerrainPyramid(
            terrain_channels, terrain_scale_groups
        )
        self.alpha_max = float(alpha_max)
        self.decision_threshold = float(decision_threshold)
        self.interaction_gate = bool(interaction_gate)
        self.bidirectional = bool(bidirectional)
        if self.interaction_gate:
            self.visual_projection = nn.Sequential(
                nn.Conv2d(128, 32, 1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
            )
            self.terrain_projection = nn.Sequential(
                nn.Conv2d(48, 32, 1, bias=False),
                nn.GroupNorm(8, 32, affine=False),
                nn.GELU(),
            )
            # Four explicit optical-change descriptors expose burn/bare-soil-like
            # confusions to the error-state head.  They never open the correction
            # gate by themselves: terrain_presence and q_t remain mandatory.
            self.veto_head = nn.Sequential(
                nn.Conv2d(101, 96, 3, padding=1),
                nn.GroupNorm(12, 96),
                nn.GELU(),
                nn.Conv2d(96, 96, 3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(12, 96),
                nn.GELU(),
                nn.Conv2d(96, 2 if self.bidirectional else 1, 1),
            )
            nn.init.zeros_(self.veto_head[-1].weight)
            nn.init.constant_(self.veto_head[-1].bias, -4.0)
        self.correction_scale = 1.0

    def set_correction_scale(self, scale: float) -> None:
        if not 0.0 <= scale <= 1.0:
            raise ValueError("correction scale must be in [0, 1]")
        self.correction_scale = float(scale)

    @staticmethod
    def uncertainty(logits: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits.detach())
        return 1.0 - 2.0 * torch.abs(probability - 0.5)

    @staticmethod
    def spectral_change_features(optical: torch.Tensor) -> torch.Tensor:
        """Label-free S2 change descriptors; bands are B02,B03,B04,B8A,B11,B12."""
        reflectance = optical.float() / 10000.0
        before = reflectance[:, :, :2].mean(dim=2)
        after = reflectance[:, :, 2:].mean(dim=2)
        eps = 1e-4

        def normalized_difference(value: torch.Tensor, left: int, right: int) -> torch.Tensor:
            return (value[:, left : left + 1] - value[:, right : right + 1]) / (
                value[:, left : left + 1] + value[:, right : right + 1] + eps
            )

        nbr_before = normalized_difference(before, 3, 5)
        nbr_after = normalized_difference(after, 3, 5)
        ndvi_before = normalized_difference(before, 3, 2)
        ndvi_after = normalized_difference(after, 3, 2)
        # dNBR/dNDVI expose strong vegetation-loss confusions, while post-event
        # NBR and SWIR contrast distinguish their resulting spectral state.
        swir_contrast = normalized_difference(after, 5, 3)
        return torch.cat(
            (nbr_before - nbr_after, ndvi_before - ndvi_after, nbr_after, swir_contrast),
            dim=1,
        ).clamp(-2.0, 2.0)

    def support_forward(
        self, visual_logits, visual_feature, uncertainty, terrain, q_t, optical=None
    ):
        raw_veto, terrain_features = self.terrain(terrain)
        zero_veto, _ = self.terrain(torch.zeros_like(terrain))
        # Positive support means "veto this visual-positive candidate". ReLU(tanh)
        # gives an exact zero action for zero support and permits learned abstention.
        if self.interaction_gate:
            if optical is None:
                raise ValueError("interaction veto requires optical input")
            terrain_feature = terrain_features["terrain_pyramid_feature"]
            visual_projected = self.visual_projection(visual_feature.detach())
            terrain_projected = self.terrain_projection(terrain_feature)
            terrain_presence = torch.tanh(
                terrain_projected.abs().mean(dim=1, keepdim=True)
            )
            interaction = visual_projected * terrain_projected
            spectral_change = self.spectral_change_features(optical).detach()
            error_probability = torch.sigmoid(
                self.veto_head(torch.cat(
                    (
                        interaction,
                        visual_projected,
                        terrain_projected,
                        uncertainty.detach(),
                        spectral_change,
                    ),
                    dim=1,
                ))
            )
            veto_strength = terrain_presence * error_probability[:, :1]
            rescue_strength = (
                terrain_presence * error_probability[:, 1:2]
                if self.bidirectional
                else torch.zeros_like(veto_strength)
            )
        else:
            veto_strength = torch.relu(torch.tanh(raw_veto - zero_veto))
            rescue_strength = torch.zeros_like(veto_strength)
        visual_positive = (
            torch.sigmoid(visual_logits.detach()) >= self.decision_threshold
        ).to(visual_logits.dtype)
        q_t = torch.clamp(q_t, 0.0, 1.0)
        if q_t.ndim == 1:
            q_t = q_t[:, None, None, None]
        # The interaction head already observes uncertainty.  Multiplying it into
        # the action a second time would make confident visual false positives
        # impossible to correct, which is precisely the failure mode this branch
        # is intended to test.  The Terrain-only compatibility branch retains the
        # conservative uncertainty permission used by the frozen experiment.
        reliability_permission = (
            torch.ones_like(uncertainty)
            if self.interaction_gate
            else uncertainty.detach()
        )
        visual_negative = 1.0 - visual_positive
        veto_gate = q_t * visual_positive * reliability_permission * veto_strength
        # Rescue is deliberately more conservative: a confident visual negative
        # is not overturned solely by static susceptibility support.
        rescue_permission = uncertainty.detach() if self.bidirectional else reliability_permission
        rescue_gate = q_t * visual_negative * rescue_permission * rescue_strength
        gate = veto_gate + rescue_gate
        bounded_residual = self.alpha_max * (rescue_strength - veto_strength)
        correction = self.alpha_max * (rescue_gate - veto_gate) * self.correction_scale
        logits = visual_logits + correction
        return logits, {
            "visual_logits": visual_logits,
            "visual_feature": visual_feature,
            "uncertainty": uncertainty,
            "gate": gate,
            "veto_gate": veto_gate,
            "rescue_gate": rescue_gate,
            "veto_strength": veto_strength,
            "rescue_strength": rescue_strength,
            "visual_positive_permission": visual_positive,
            "bounded_residual": bounded_residual,
            "correction": correction,
            "q_t": q_t,
            **terrain_features,
        }

    def forward(self, optical, coordinates, terrain, q_t):
        with torch.no_grad():
            visual_logits, visual_feature = self.visual(optical, coordinates)
        uncertainty = self.uncertainty(visual_logits)
        return self.support_forward(
            visual_logits, visual_feature, uncertainty, terrain, q_t, optical
        )

    def forward_controls(self, optical, coordinates, terrain, q_t, donor_terrain, donor_q_t):
        with torch.no_grad():
            visual_logits, visual_feature = self.visual(optical, coordinates)
        uncertainty = self.uncertainty(visual_logits)
        zero = torch.zeros_like(visual_logits)
        base = {
            "visual_logits": visual_logits,
            "visual_feature": visual_feature,
            "uncertainty": uncertainty,
            "gate": zero,
            "veto_gate": zero,
            "rescue_gate": zero,
            "veto_strength": zero,
            "rescue_strength": zero,
            "visual_positive_permission": zero,
            "bounded_residual": zero,
            "correction": zero,
            "q_t": zero,
        }
        outputs = {"visual_anchor": (visual_logits, base)}
        controls = {
            "aligned": (terrain, q_t),
            "zero": (torch.zeros_like(terrain), torch.ones_like(q_t)),
            "roll32": (torch.roll(terrain, (32, 32), (-2, -1)), torch.roll(q_t, (32, 32), (-2, -1))),
            "roll64": (torch.roll(terrain, (64, 64), (-2, -1)), torch.roll(q_t, (64, 64), (-2, -1))),
            "other_region_donor": (donor_terrain, donor_q_t),
        }
        for name, (support, quality) in controls.items():
            outputs[name] = self.support_forward(
                visual_logits, visual_feature, uncertainty, support, quality, optical
            )
        return outputs


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def train_model(
    model, mode, train_loader, val_loader, args, pos_weight, decision_threshold, log
):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    positive = torch.tensor([pos_weight], device=args.device).view(1, 1, 1, 1)
    best_key, best_epoch, best_state = None, 0, None
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; seen = 0
        for batch in train_loader:
            optical = batch["pre"].to(args.device, non_blocking=True)
            coordinates = batch["post"].to(args.device, non_blocking=True)
            target = batch["mask"].to(args.device, non_blocking=True)
            valid = batch["valid"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with protocol.autocast_context(args.amp):
                if mode == "visual":
                    logits, _ = model(optical, coordinates); diagnostics = None
                else:
                    logits, diagnostics = model(
                        optical, coordinates,
                        batch["terrain"].to(args.device, non_blocking=True),
                        batch["q_t"].to(args.device, non_blocking=True),
                    )
                if (
                    diagnostics is not None
                    and "rescue_strength" in diagnostics
                    and model.bidirectional
                    and args.adapter_objective == "correction"
                ):
                    visual_logits = diagnostics["visual_logits"].detach()
                    visual_prediction = torch.sigmoid(visual_logits) >= decision_threshold
                    positive_mask = target >= 0.5
                    false_positive = (visual_prediction & ~positive_mask).to(valid.dtype) * valid
                    false_negative = (~visual_prediction & positive_mask).to(valid.dtype) * valid
                    true_positive = (visual_prediction & positive_mask).to(valid.dtype) * valid
                    true_negative = (~visual_prediction & ~positive_mask).to(valid.dtype) * valid
                    veto_error = (diagnostics["veto_strength"] - false_positive).square()
                    rescue_error = (diagnostics["rescue_strength"] - false_negative).square()
                    gate_supervision = 0.25 * (
                        masked_mean(veto_error, false_positive)
                        + masked_mean(veto_error, true_positive)
                        + masked_mean(rescue_error, false_negative)
                        + masked_mean(rescue_error, true_negative)
                    )
                    segmentation_bce = F.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"
                    )
                    correction_loss = (
                        args.fp_correction_weight
                        * masked_mean(segmentation_bce, false_positive)
                        + args.fn_correction_weight
                        * masked_mean(segmentation_bce, false_negative)
                    )
                    preservation_loss = (
                        args.tp_preserve_weight
                        * masked_mean(diagnostics["correction"].square(), true_positive)
                        + args.tn_preserve_weight
                        * masked_mean(diagnostics["correction"].square(), true_negative)
                    )
                    balanced_segmentation_bce = F.binary_cross_entropy_with_logits(
                        logits, target, pos_weight=positive, reduction="none"
                    )
                    full_loss = masked_mean(balanced_segmentation_bce, valid)
                    loss = (
                        correction_loss
                        + args.preserve_weight * preservation_loss
                        + args.gate_supervision_weight * gate_supervision
                        + args.adapter_full_bce_weight * full_loss
                        + args.adapter_dice_weight
                        * protocol.dice_loss_per_sample(logits, target, valid).mean()
                    )
                elif (
                    diagnostics is not None
                    and "veto_strength" in diagnostics
                    and not model.bidirectional
                ):
                    visual_logits = diagnostics["visual_logits"].detach()
                    visual_prediction = torch.sigmoid(visual_logits) >= decision_threshold
                    positive = target >= 0.5
                    false_positive = (visual_prediction & ~positive).to(valid.dtype) * valid
                    true_positive = (visual_prediction & positive).to(valid.dtype) * valid
                    candidate = false_positive + true_positive
                    veto_strength = diagnostics["veto_strength"]
                    veto_target = false_positive
                    veto_error = (veto_strength - veto_target).square()
                    # Balance FP correction against explicit TP preservation, independent
                    # of their pixel prevalence within the visual-positive candidate set.
                    veto_supervision = 0.5 * (
                        masked_mean(veto_error, false_positive)
                        + masked_mean(veto_error, true_positive)
                    )
                    segmentation_bce = F.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"
                    )
                    candidate_segmentation = masked_mean(segmentation_bce, candidate)
                    tp_preservation = masked_mean(
                        diagnostics["correction"].square(), true_positive
                    )
                    full_loss = masked_mean(segmentation_bce, valid)
                    loss = (
                        args.gate_supervision_weight * veto_supervision
                        + candidate_segmentation
                        + args.tp_preserve_weight * tp_preservation
                        + args.adapter_full_bce_weight * full_loss
                        + args.adapter_dice_weight
                        * protocol.dice_loss_per_sample(logits, target, valid).mean()
                    )
                elif diagnostics is not None and args.adapter_objective == "correction":
                    visual_logits = diagnostics["visual_logits"].detach()
                    visual_prediction = torch.sigmoid(visual_logits) >= decision_threshold
                    visual_wrong = torch.logical_xor(
                        visual_prediction, target >= 0.5
                    ).to(valid.dtype) * valid
                    visual_correct = (valid - visual_wrong).clamp_min(0.0)
                    positive = (target >= 0.5).to(valid.dtype)
                    false_negative = visual_wrong * positive
                    false_positive = visual_wrong * (1.0 - positive)
                    true_positive = visual_correct * positive
                    true_negative = visual_correct * (1.0 - positive)
                    unweighted_bce = F.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"
                    )
                    if args.correction_stratification == "state":
                        correction_loss = (
                            args.fn_correction_weight
                            * masked_mean(unweighted_bce, false_negative)
                            + args.fp_correction_weight
                            * masked_mean(unweighted_bce, false_positive)
                        )
                        preservation_loss = (
                            args.tp_preserve_weight
                            * masked_mean(diagnostics["correction"].square(), true_positive)
                            + args.tn_preserve_weight
                            * masked_mean(diagnostics["correction"].square(), true_negative)
                        )
                    else:
                        correction_loss = masked_mean(unweighted_bce, visual_wrong)
                        preservation_loss = masked_mean(
                            diagnostics["correction"].square(), visual_correct
                        )
                    full_loss = masked_mean(unweighted_bce, valid)
                    gate_target = visual_wrong * diagnostics["uncertainty"].detach()
                    gate_error = (diagnostics["gate"] - gate_target).square()
                    if args.correction_stratification == "state":
                        gate_supervision = 0.25 * (
                            masked_mean(gate_error, false_negative)
                            + masked_mean(gate_error, false_positive)
                            + masked_mean(gate_error, true_positive)
                            + masked_mean(gate_error, true_negative)
                        )
                    else:
                        gate_supervision = masked_mean(gate_error, valid)
                    loss = (
                        correction_loss
                        + args.preserve_weight * preservation_loss
                        + args.adapter_full_bce_weight * full_loss
                        + args.gate_supervision_weight * gate_supervision
                        + args.adapter_dice_weight
                        * protocol.dice_loss_per_sample(logits, target, valid).mean()
                    )
                else:
                    bce = F.binary_cross_entropy_with_logits(
                        logits, target, pos_weight=positive, reduction="none"
                    )
                    bce = (bce * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
                    loss = (
                        bce + args.dice_weight * protocol.dice_loss_per_sample(logits, target, valid)
                    ).mean()
                if diagnostics is not None:
                    loss = loss + args.gate_l1 * diagnostics["gate"].mean()
                    loss = loss + args.residual_l1 * diagnostics["bounded_residual"].abs().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer); scaler.update()
            total_loss += float(loss.detach()) * optical.shape[0]; seen += optical.shape[0]; step += 1
            if args.max_steps and step >= args.max_steps: break
        val_ap, val_histogram = protocol.score_validation_ap(
            model, val_loader, mode, args.device, args.amp
        )
        val_errors = None
        if mode == "adapter":
            val_counts = val_histogram.counts_at(decision_threshold)
            val_errors = int(val_counts["fp"] + val_counts["fn"])
            val_iou = float(protocol.metrics_from_counts(val_counts)["iou"])
            if args.adapter_architecture in {
                "terrain_veto", "terrain_interaction_veto", "terrain_interaction_residual"
            }:
                selection_key = (-val_iou, val_errors, -val_ap)
            else:
                selection_key = (val_errors, -val_ap)
        else:
            val_iou = None
            selection_key = (-val_ap,)
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": total_loss / max(seen, 1),
            "val_ap": val_ap,
            "val_iou_at_visual_threshold": val_iou,
            "val_errors_at_visual_threshold": val_errors,
        }
        history.append(row)
        log(
            f"[epoch] {epoch}/{args.epochs} step={step} loss={row['train_loss']:.6f} "
            f"val_ap={val_ap:.6f} val_iou={val_iou} val_errors={val_errors}"
        )
        if best_key is None or selection_key < best_key:
            best_key, best_epoch, best_state = selection_key, epoch, trainable_state(model)
        if args.max_steps and step >= args.max_steps: break
    if best_state is None: raise RuntimeError("training produced no checkpoint")
    return best_state, history, best_epoch


@torch.no_grad()
def calibrate_correction_scale(model, val_loader, threshold, args, log):
    scales = sorted({float(value) for value in args.correction_scales.split(",")})
    if not scales or scales[0] < 0.0 or scales[-1] > 1.0:
        raise ValueError("--correction-scales must be a nonempty comma-separated subset of [0, 1]")
    if 0.0 not in scales:
        raise ValueError("--correction-scales must include 0 as the abstention option")
    rows = []
    for scale in scales:
        model.set_correction_scale(scale)
        average_precision, histogram = protocol.score_validation_ap(
            model, val_loader, "adapter", args.device, args.amp
        )
        counts = histogram.counts_at(threshold)
        errors = int(counts["fp"] + counts["fn"])
        row = {
            "correction_scale": scale,
            "errors": errors,
            "average_precision": float(average_precision),
            "brier": float(histogram.brier),
            **protocol.metrics_from_counts(counts),
        }
        rows.append(row)
        log(
            f"[calibration] scale={scale:.4f} errors={errors} "
            f"iou={row['iou']:.6f} ap={average_precision:.6f} brier={histogram.brier:.6f}"
        )
    if args.adapter_architecture in {
        "terrain_veto", "terrain_interaction_veto", "terrain_interaction_residual"
    }:
        selected = min(
            rows,
            key=lambda row: (
                -row["iou"],
                row["errors"],
                -row["average_precision"],
                row["correction_scale"],
            ),
        )
    else:
        selected = min(
            rows,
            key=lambda row: (
                row["errors"],
                -row["average_precision"],
                row["brier"],
                row["correction_scale"],
            ),
        )
    model.set_correction_scale(selected["correction_scale"])
    log(
        f"[calibration-selected] scale={selected['correction_scale']:.4f} "
        f"errors={selected['errors']} iou={selected['iou']:.6f}"
    )
    return selected, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, default=BASE_H5)
    parser.add_argument("--optical-h5", type=Path, default=OPTICAL_H5)
    parser.add_argument("--terrain-h5", type=Path, default=TERRAIN_H5)
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    parser.add_argument("--mode", choices=("visual", "adapter"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--visual-checkpoint", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument(
        "--adapter-objective", choices=("segmentation", "correction"), default="correction"
    )
    parser.add_argument(
        "--adapter-architecture",
        choices=(
            "bounded_residual",
            "terrain_veto",
            "terrain_interaction_veto",
            "terrain_interaction_residual",
        ),
        default="bounded_residual",
    )
    parser.add_argument(
        "--correction-stratification", choices=("aggregate", "state"), default="aggregate"
    )
    parser.add_argument("--preserve-weight", type=float, default=10.0)
    parser.add_argument("--fn-correction-weight", type=float, default=1.0)
    parser.add_argument("--fp-correction-weight", type=float, default=1.0)
    parser.add_argument("--tp-preserve-weight", type=float, default=20.0)
    parser.add_argument("--tn-preserve-weight", type=float, default=1.0)
    parser.add_argument("--adapter-full-bce-weight", type=float, default=0.1)
    parser.add_argument("--gate-supervision-weight", type=float, default=1.0)
    parser.add_argument("--adapter-dice-weight", type=float, default=0.05)
    parser.add_argument("--gate-l1", type=float, default=1e-3)
    parser.add_argument("--residual-l1", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--uncertainty-cutoff", type=float, default=0.5)
    parser.add_argument("--uncertainty-temperature", type=float, default=0.1)
    parser.add_argument(
        "--correction-scales",
        default="0,0.0625,0.125,0.25,0.5,0.75,1",
        help="Validation-only residual dosage grid; 0 is required as the abstention option.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Nested label-free fraction of each training spatial supergroup.",
    )
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--evaluation-splits",
        default="val,test",
        help="Comma-separated final evaluation splits. Development sweeps must use val only.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_splits = tuple(
        item.strip() for item in args.evaluation_splits.split(",") if item.strip()
    )
    if not evaluation_splits or any(item not in {"val", "test"} for item in evaluation_splits):
        raise ValueError("--evaluation-splits must contain val and/or test")
    args.outdir = args.outdir or OUT_ROOT / f"fold{args.fold}_seed{args.seed}" / args.mode
    if args.mode == "adapter" and args.visual_checkpoint is None:
        raise ValueError("adapter requires --visual-checkpoint")
    args.outdir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    (args.outdir / "command.txt").write_text(command + "\n", encoding="utf-8")
    log_path = args.outdir / "run.log"; log_path.write_text(command + "\n", encoding="utf-8")
    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream: stream.write(message + "\n")
    started = time.time(); set_seed(args.seed)
    all_ids, event_ids = validate_sidecars(args.base_h5, args.optical_h5, args.terrain_h5)
    terrain_names, terrain_scale_groups, terrain_schema = resolve_terrain_contract(
        args.terrain_h5
    )
    terrain_channels = len(terrain_names)
    rows, roles, split_regions = protocol.load_logo_rows(args.split_csv, args.fold)
    allowed = set(all_ids)
    roles = {role:[sample for sample in values if sample in allowed] for role,values in roles.items()}
    fraction_train_ids = stratified_fraction_subset(
        roles["train"], rows, args.train_fraction, args.seed
    )
    train_ids = protocol.deterministic_subset(
        fraction_train_ids, args.max_train_samples, args.seed, "train"
    )
    val_ids = protocol.deterministic_subset(roles["val"], args.max_eval_samples, args.seed, "val")
    test_ids = protocol.deterministic_subset(roles["test"], args.max_eval_samples, args.seed, "test")
    if args.mode == "adapter":
        mean, std = estimate_terrain_stats(args.terrain_h5, all_ids, train_ids)
    else:
        mean = np.zeros(terrain_channels, dtype=np.float32)
        std = np.ones(terrain_channels, dtype=np.float32)
    datasets = {
        role: PrithviTerrainDataset(
            args.base_h5, args.optical_h5, args.terrain_h5, all_ids, event_ids, rows, ids,
            mean, std, args.seed, train_ids, args.mode == "adapter",
        )
        for role, ids in (("train",train_ids),("val",val_ids),("test",test_ids))
    }
    loaders = {role:protocol.make_loader(data,args,shuffle=role=="train") for role,data in datasets.items()}
    current_data_identity = {
        "base_h5": file_signature(args.base_h5),
        "optical_h5": file_signature(args.optical_h5),
        "terrain_h5": file_signature(args.terrain_h5),
        "split_csv_sha256": protocol.sha256_file(args.split_csv),
        "sample_sha256": {
            role: protocol.sha256_strings(data.sample_ids) for role, data in datasets.items()
        },
    }
    encoder, provenance = load_prithvi_encoder()
    core = PrithviEO2ChangeModel(encoder, decoder_width=128, freeze_encoder=True)
    visual = PrithviVisualCompat(core)
    parent = None
    if args.mode == "visual":
        model: nn.Module = visual
    else:
        parent = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "mode": "visual",
            "fold": args.fold,
            "seed": args.seed,
            "prithvi_checkpoint_sha256": provenance["checkpoint_sha256"],
            **current_data_identity,
        }
        mismatch = {key:(value,parent["identity"].get(key)) for key,value in expected.items() if parent["identity"].get(key)!=value}
        if mismatch: raise RuntimeError(f"visual parent identity mismatch: {mismatch}")
        load_trainable_state(visual, parent["trainable_state_dict"])
        if args.adapter_architecture in {
            "terrain_veto", "terrain_interaction_veto", "terrain_interaction_residual"
        }:
            model = PrithviTerrainVetoCompat(
                visual,
                args.alpha_max,
                float(parent["threshold"]),
                terrain_channels,
                terrain_scale_groups,
                interaction_gate=args.adapter_architecture in {
                    "terrain_interaction_veto", "terrain_interaction_residual"
                },
                bidirectional=args.adapter_architecture == "terrain_interaction_residual",
            )
        else:
            model = PrithviTerrainCompat(
                visual,
                args.alpha_max,
                args.uncertainty_cutoff,
                args.uncertainty_temperature,
                terrain_channels,
                terrain_scale_groups,
            )
    model = model.to(args.device)
    frozen_visual_sha256_before = module_parameter_sha256(visual.decoder)
    pos_weight = protocol.estimate_pos_weight(datasets["train"])
    decision_threshold = float(parent["threshold"]) if parent else None
    best_state, history, best_epoch = train_model(
        model,
        args.mode,
        loaders["train"],
        loaders["val"],
        args,
        pos_weight,
        decision_threshold,
        log,
    )
    load_trainable_state(model,best_state)
    frozen_visual_sha256_after = module_parameter_sha256(visual.decoder)
    if args.mode == "adapter" and frozen_visual_sha256_after != frozen_visual_sha256_before:
        raise RuntimeError("frozen Prithvi visual anchor changed during Terrain-v2 training")
    calibration_selected = None
    calibration_grid = None
    if args.mode == "adapter":
        calibration_selected, calibration_grid = calibrate_correction_scale(
            model, loaders["val"], float(parent["threshold"]), args, log
        )
    val_ap,val_hist = protocol.score_validation_ap(model,loaders["val"],args.mode,args.device,args.amp)
    if args.mode=="visual": threshold,threshold_metrics=protocol.choose_threshold(val_hist)
    else:
        threshold=float(parent["threshold"]); threshold_metrics=protocol.metrics_from_counts(val_hist.counts_at(threshold))
    sample_rows=[];region_rows=[];event_rows=[];corpus_rows=[];audits=[]
    for split in evaluation_splits:
        samples,regions,events,corpus,audit=protocol.evaluate(model,args.mode,loaders[split],threshold,split,args)
        sample_rows+=samples;region_rows+=regions;event_rows+=events;corpus_rows+=corpus["rows"];audits.append(audit)
    identity={
        "mode":args.mode,"fold":args.fold,"seed":args.seed,
        "prithvi_checkpoint_sha256":provenance["checkpoint_sha256"],
        **current_data_identity,
        "parent_visual_checkpoint":str(args.visual_checkpoint.resolve()) if args.visual_checkpoint else None,
        "parent_visual_trainable_sha256":parent.get("trainable_sha256") if parent else None,
    }
    config={"contract":"Prithvi EO2 + audited multiscale Terrain support",
            "mode":args.mode,"fold":args.fold,"seed":args.seed,"identity":identity,
            "terrain_schema":terrain_schema,"terrain_names":terrain_names,
            "terrain_mean":mean.tolist(),"terrain_std":std.tolist(),"split_regions":split_regions,
            "prithvi_provenance":provenance,"args":vars(args),"command":command,
            "correction_calibration": {
                "selection_split": "validation",
                "selection_objective": (
                    "maximum IoU at the frozen matched-visual threshold"
                    if args.adapter_architecture in {
                        "terrain_veto", "terrain_interaction_veto", "terrain_interaction_residual"
                    }
                    else "minimum fp+fn at the frozen matched-visual threshold"
                ),
                "selected": calibration_selected,
                "grid": calibration_grid,
            }}
    protocol.write_csv(sample_rows,args.outdir/"per_sample.csv");protocol.write_csv(region_rows,args.outdir/"per_region.csv");protocol.write_csv(event_rows,args.outdir/"per_event.csv")
    state=trainable_state(model); state_hash=tensor_dict_sha256(state)
    threshold_source = "visual_validation" if args.mode == "visual" else "loaded_matched_visual_checkpoint"
    torch.save({"identity":identity,"trainable_state_dict":state,"trainable_sha256":state_hash,
                "threshold":threshold,"threshold_source":threshold_source,
                "correction_scale": model.correction_scale if args.mode == "adapter" else None,
                "best_epoch":best_epoch,"history":history},args.outdir/"checkpoint.pt")
    result={"identity":identity,"best_epoch":best_epoch,"history":history,"pos_weight":pos_weight,"threshold":threshold,
            "validation_ap":val_ap,"validation_threshold_metrics":threshold_metrics,"corpus_metrics":corpus_rows,
            "correction_calibration": config["correction_calibration"],
            "identity_and_control_audits":audits,"trainable_sha256":state_hash,"elapsed_seconds":time.time()-started}
    result["frozen_visual_sha256_before"] = frozen_visual_sha256_before
    result["frozen_visual_sha256_after"] = frozen_visual_sha256_after
    (args.outdir/"config.json").write_text(json.dumps(protocol.json_safe(config),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (args.outdir/"result.json").write_text(json.dumps(protocol.json_safe(result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    done={"status":"complete","mode":args.mode,"fold":args.fold,"seed":args.seed,
          "trainable_sha256":state_hash,
          "result_sha256":protocol.sha256_file(args.outdir/"result.json"),
          "checkpoint_sha256":protocol.sha256_file(args.outdir/"checkpoint.pt")}
    (args.outdir/"DONE.json").write_text(json.dumps(done,indent=2)+"\n",encoding="utf-8")
    log(f"[done] mode={args.mode} fold={args.fold} seed={args.seed} best_epoch={best_epoch} val_ap={val_ap:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
