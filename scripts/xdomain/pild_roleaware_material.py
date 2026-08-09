#!/usr/bin/env python3
"""Role-pure Terrain x Material susceptibility interaction.

Material is a 21-feature sample context. It may change the magnitude of an
existing Terrain residual through bounded, low-rank FiLM coefficients, but it
cannot create a dense direction. All spatial maps are deterministic functions
of detached Terrain responses and the detached Terrain residual.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MATERIAL_FEATURE_NAMES = (
    "awc_0_10_footprint_mean_mm",
    "awc_10_30_footprint_mean_mm",
    "awc_30_60_footprint_mean_mm",
    "awc_60_100_footprint_mean_mm",
    "awc_100_200_footprint_mean_mm",
    "soil_clay_0_5cm_mean_raw",
    "soil_clay_5_15cm_mean_raw",
    "soil_sand_0_5cm_mean_raw",
    "soil_sand_5_15cm_mean_raw",
    "soil_silt_0_5cm_mean_raw",
    "soil_silt_5_15cm_mean_raw",
    "soil_cec_0_5cm_mean_raw",
    "soil_cec_5_15cm_mean_raw",
    "soil_soc_0_5cm_mean_raw",
    "soil_soc_5_15cm_mean_raw",
    "soil_bdod_0_5cm_mean_raw",
    "soil_bdod_5_15cm_mean_raw",
    "soil_cfvo_0_5cm_mean_raw",
    "soil_cfvo_5_15cm_mean_raw",
    "soil_phh2o_0_5cm_mean_raw",
    "soil_phh2o_5_15cm_mean_raw",
)
MATERIAL_FEATURE_COUNT = len(MATERIAL_FEATURE_NAMES)

CONTEXT_ALIGNED = "aligned"
CONTEXT_SHUFFLED = "within-source/event-shuffle"
CONTEXT_ZERO_Q = "zero-q"
CONTEXT_ABSTAIN = "ABSTAIN"
ALLOWED_CONTEXTS = (
    CONTEXT_ALIGNED,
    CONTEXT_SHUFFLED,
    CONTEXT_ZERO_Q,
    CONTEXT_ABSTAIN,
)

DEFAULT_TERRAIN_RESPONSE_GROUPS = {
    "slope": (0,),
    "curvature": (1,),
    "relief": (2,),
}


def _validate_identity(
    sample_ids: Sequence[str],
    source_ids: Sequence[str],
    event_ids: Sequence[str],
    expected: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    identities = tuple(str(value) for value in sample_ids)
    sources = tuple(str(value) for value in source_ids)
    events = tuple(str(value) for value in event_ids)
    if not (len(identities) == len(sources) == len(events) == expected):
        raise ValueError("sample/source/event identity lengths must match Material rows")
    if len(set(identities)) != len(identities) or any(not value for value in identities):
        raise ValueError("sample_ids must be non-empty and unique")
    if any(not value for value in sources) or any(not value for value in events):
        raise ValueError("source_ids and event_ids must be non-empty")
    return identities, sources, events


def _hierarchical_source_event_weights(
    source_ids: np.ndarray, event_ids: np.ndarray
) -> np.ndarray:
    """Equal source -> event -> sample weights for shortcut-resistant fitting."""

    weights = np.zeros(len(source_ids), dtype=np.float64)
    sources = sorted(set(source_ids.tolist()))
    for source in sources:
        source_positions = np.flatnonzero(source_ids == source)
        events = sorted(set(event_ids[source_positions].tolist()))
        for event in events:
            positions = source_positions[event_ids[source_positions] == event]
            weights[positions] = 1.0 / (
                len(sources) * len(events) * len(positions)
            )
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("source/event balancing weights do not sum to one")
    return weights


@dataclass(frozen=True)
class OuterTrainMaterialNormalizer:
    """Frozen Material normalization fitted from outer-training identities only.

    Features with variance that is effectively only between sources are
    neutralized. Source IDs are used only for this fit-time guard and never
    become model inputs.
    """

    feature_names: tuple[str, ...]
    impute_mean: np.ndarray
    scale: np.ndarray
    fit_counts: np.ndarray
    within_source_fraction: np.ndarray
    shortcut_blocked: np.ndarray
    outer_train_sample_sha256: str
    z_clip: float
    shortcut_min_within_fraction: float

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        sample_ids: Sequence[str],
        source_ids: Sequence[str],
        event_ids: Sequence[str],
        outer_train_ids: Sequence[str],
        *,
        feature_names: Sequence[str] = MATERIAL_FEATURE_NAMES,
        z_clip: float = 5.0,
        shortcut_min_within_fraction: float = 0.01,
    ) -> "OuterTrainMaterialNormalizer":
        names = tuple(str(value) for value in feature_names)
        expected_features = len(names)
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != expected_features:
            raise ValueError(
                f"Material values must have shape [N,{expected_features}]"
            )
        if not names or len(set(names)) != expected_features:
            raise ValueError("feature_names must contain at least one unique name")
        if z_clip <= 0.0:
            raise ValueError("z_clip must be positive")
        if not 0.0 <= shortcut_min_within_fraction <= 1.0:
            raise ValueError("shortcut_min_within_fraction must be in [0,1]")

        identities, sources, events = _validate_identity(
            sample_ids, source_ids, event_ids, len(array)
        )
        train_ids = tuple(str(value) for value in outer_train_ids)
        if not train_ids or len(train_ids) != len(set(train_ids)):
            raise ValueError("outer_train_ids must be non-empty and unique")
        position = {sample_id: index for index, sample_id in enumerate(identities)}
        missing = sorted(set(train_ids) - set(position))
        if missing:
            raise ValueError(f"outer_train_ids contain unknown samples: {missing[:3]}")
        train_positions = np.asarray([position[value] for value in train_ids], dtype=np.int64)
        train = array[train_positions]
        train_sources = np.asarray(sources, dtype=object)[train_positions]
        train_events = np.asarray(events, dtype=object)[train_positions]
        weights = _hierarchical_source_event_weights(train_sources, train_events)

        means = np.zeros(expected_features, dtype=np.float64)
        scales = np.ones(expected_features, dtype=np.float64)
        counts = np.zeros(expected_features, dtype=np.int64)
        within_fraction = np.ones(expected_features, dtype=np.float64)
        blocked = np.zeros(expected_features, dtype=bool)
        multiple_sources = len(set(train_sources.tolist())) > 1

        for column in range(expected_features):
            finite = np.isfinite(train[:, column])
            counts[column] = int(finite.sum())
            if not finite.any():
                blocked[column] = True
                continue
            valid_weights = weights[finite]
            valid_weights = valid_weights / valid_weights.sum()
            column_weights = np.zeros_like(weights)
            column_weights[finite] = valid_weights
            valid_values = train[finite, column]
            mean = float(np.sum(valid_weights * valid_values))
            variance = float(np.sum(valid_weights * np.square(valid_values - mean)))
            means[column] = mean
            scales[column] = math.sqrt(variance) if variance > 1e-12 else 1.0

            if multiple_sources and variance > 1e-12:
                within_variance = 0.0
                for source in sorted(set(train_sources.tolist())):
                    source_mask = finite & (train_sources == source)
                    if not source_mask.any():
                        continue
                    source_weights = column_weights[source_mask]
                    source_mass = float(source_weights.sum())
                    source_weights = source_weights / source_mass
                    source_values = train[source_mask, column]
                    source_mean = float(np.sum(source_weights * source_values))
                    within_variance += source_mass * float(
                        np.sum(source_weights * np.square(source_values - source_mean))
                    )
                within_fraction[column] = within_variance / variance
                blocked[column] = (
                    within_fraction[column] < shortcut_min_within_fraction
                )

        digest = hashlib.sha256()
        for sample_id in train_ids:
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\n")
        return cls(
            feature_names=names,
            impute_mean=means.astype(np.float32),
            scale=scales.astype(np.float32),
            fit_counts=counts,
            within_source_fraction=within_fraction.astype(np.float32),
            shortcut_blocked=blocked,
            outer_train_sample_sha256=digest.hexdigest(),
            z_clip=float(z_clip),
            shortcut_min_within_fraction=float(shortcut_min_within_fraction),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        expected_features = len(self.feature_names)
        if array.ndim != 2 or array.shape[1] != expected_features:
            raise ValueError(
                f"Material values must have shape [N,{expected_features}]"
            )
        filled = np.where(np.isfinite(array), array, self.impute_mean)
        normalized = np.clip(
            (filled - self.impute_mean) / self.scale, -self.z_clip, self.z_clip
        ).astype(np.float32)
        normalized[:, self.shortcut_blocked] = 0.0
        return normalized

    def audit(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "fit_scope": "outer-train-only",
            "balance": "equal-source/equal-event/equal-sample",
            "outer_train_sample_sha256": self.outer_train_sample_sha256,
            "impute_mean": self.impute_mean.tolist(),
            "scale": self.scale.tolist(),
            "fit_counts": self.fit_counts.tolist(),
            "within_source_fraction": self.within_source_fraction.tolist(),
            "shortcut_blocked": self.shortcut_blocked.tolist(),
            "source_id_is_model_feature": False,
            "z_clip": self.z_clip,
        }


@dataclass(frozen=True)
class MaterialContextBatch:
    """Material tensors plus immutable recipient and explicit donor identity."""

    name: str
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    donor_sample_ids: tuple[str, ...]
    donor_event_ids: tuple[str, ...]
    material: torch.Tensor
    q_m: torch.Tensor
    abstain: torch.Tensor


def _stable_index(token: str, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot select from an empty donor set")
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % size


def build_material_contexts(
    material: torch.Tensor,
    q_m: torch.Tensor,
    sample_ids: Sequence[str],
    source_ids: Sequence[str],
    event_ids: Sequence[str],
    *,
    abstain: torch.Tensor | None = None,
    seed: int = 20260722,
) -> dict[str, MaterialContextBatch]:
    """Build aligned, same-source cross-event shuffle, and zero-q paths."""

    if material.ndim != 2 or material.shape[1] != MATERIAL_FEATURE_COUNT:
        raise ValueError(
            f"material must have shape [B,{MATERIAL_FEATURE_COUNT}]"
        )
    batch = material.shape[0]
    identities, sources, events = _validate_identity(
        sample_ids, source_ids, event_ids, batch
    )
    quality = q_m.to(device=material.device, dtype=material.dtype).reshape(-1)
    if quality.shape != (batch,):
        raise ValueError("q_m must contain one scalar per sample")
    if abstain is None:
        abstain_mask = torch.zeros(batch, dtype=torch.bool, device=material.device)
    else:
        abstain_mask = abstain.to(device=material.device, dtype=torch.bool).reshape(-1)
        if abstain_mask.shape != (batch,):
            raise ValueError("abstain must contain one flag per sample")

    finite_material = torch.isfinite(material).all(dim=1)
    finite_quality = torch.isfinite(quality)
    aligned_abstain = abstain_mask | ~finite_material | ~finite_quality
    aligned_material = torch.where(torch.isfinite(material), material, torch.zeros_like(material))
    aligned_q = torch.where(
        aligned_abstain,
        torch.zeros_like(quality),
        torch.clamp(quality, 0.0, 1.0),
    )

    eligible = ((aligned_q > 0.0) & ~aligned_abstain).detach().cpu().numpy()
    donor_indices = np.arange(batch, dtype=np.int64)
    shuffle_abstain = np.ones(batch, dtype=bool)

    for source in sorted(set(sources)):
        supported_events = sorted(
            {
                events[index]
                for index in range(batch)
                if sources[index] == source and eligible[index]
            }
        )
        if len(supported_events) < 2:
            continue
        shift = 1 + _stable_index(f"{seed}|{source}|event-shift", len(supported_events) - 1)
        event_donor = {
            event: supported_events[(index + shift) % len(supported_events)]
            for index, event in enumerate(supported_events)
        }
        for index in range(batch):
            if sources[index] != source or not eligible[index]:
                continue
            donor_event = event_donor[events[index]]
            candidates = [
                donor
                for donor in range(batch)
                if sources[donor] == source
                and events[donor] == donor_event
                and eligible[donor]
            ]
            selected = candidates[
                _stable_index(f"{seed}|{identities[index]}|material-donor", len(candidates))
            ]
            donor_indices[index] = selected
            shuffle_abstain[index] = False

    donor_tensor = torch.as_tensor(donor_indices, device=material.device, dtype=torch.long)
    shuffled_material = aligned_material.index_select(0, donor_tensor)
    shuffled_q = aligned_q.index_select(0, donor_tensor)
    shuffled_abstain = torch.as_tensor(
        shuffle_abstain, device=material.device, dtype=torch.bool
    )
    shuffled_q = torch.where(shuffled_abstain, torch.zeros_like(shuffled_q), shuffled_q)

    def make(
        name: str,
        values: torch.Tensor,
        quality_values: torch.Tensor,
        abstain_values: torch.Tensor,
        donors: np.ndarray,
    ) -> MaterialContextBatch:
        return MaterialContextBatch(
            name=name,
            sample_ids=identities,
            source_ids=sources,
            event_ids=events,
            donor_sample_ids=tuple(identities[index] for index in donors),
            donor_event_ids=tuple(events[index] for index in donors),
            material=values,
            q_m=quality_values,
            abstain=abstain_values,
        )

    identity_donors = np.arange(batch, dtype=np.int64)
    return {
        CONTEXT_ALIGNED: make(
            CONTEXT_ALIGNED,
            aligned_material,
            aligned_q,
            aligned_abstain,
            identity_donors,
        ),
        CONTEXT_SHUFFLED: make(
            CONTEXT_SHUFFLED,
            shuffled_material,
            shuffled_q,
            shuffled_abstain,
            donor_indices,
        ),
        CONTEXT_ZERO_Q: make(
            CONTEXT_ZERO_Q,
            aligned_material,
            torch.zeros_like(aligned_q),
            aligned_abstain,
            identity_donors,
        ),
    }


class LowRankMaterialFiLMHead(nn.Module):
    """Map the 21 Material features to low-rank Terrain-response weights."""

    def __init__(
        self,
        response_count: int,
        *,
        hidden_dim: int = 32,
        rank: int = 4,
    ) -> None:
        super().__init__()
        if response_count <= 0 or hidden_dim <= 0 or rank <= 0:
            raise ValueError("response_count, hidden_dim, and rank must be positive")
        self.encoder = nn.Sequential(
            nn.Linear(MATERIAL_FEATURE_COUNT, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank),
        )
        self.response = nn.Linear(rank, response_count, bias=False)
        nn.init.zeros_(self.response.weight)

    def forward(self, material: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.response(self.encoder(material)))


class RoleAwareMaterialInteraction(nn.Module):
    """Bounded Material FiLM over detached Terrain response and residual maps."""

    def __init__(
        self,
        response_groups: Mapping[str, Sequence[int]] = DEFAULT_TERRAIN_RESPONSE_GROUPS,
        *,
        hidden_dim: int = 32,
        rank: int = 4,
        modulation_bound: float = 0.25,
    ) -> None:
        super().__init__()
        groups = {
            str(name): tuple(int(index) for index in indices)
            for name, indices in response_groups.items()
        }
        if not groups or any(not indices for indices in groups.values()):
            raise ValueError("every Terrain response group must be non-empty")
        if any(index < 0 for indices in groups.values() for index in indices):
            raise ValueError("Terrain response indices must be non-negative")
        lowered = " ".join(groups).lower()
        for required in ("slope", "curvature", "relief"):
            if required not in lowered:
                raise ValueError(f"Terrain response groups must include {required}")
        if not 0.0 < modulation_bound < 1.0:
            raise ValueError("modulation_bound must be in (0,1)")
        self.response_names = tuple(groups)
        self.response_groups = tuple(groups.values())
        self.modulation_bound = float(modulation_bound)
        self.interaction_head = LowRankMaterialFiLMHead(
            len(groups), hidden_dim=hidden_dim, rank=rank
        )

    def _terrain_basis(
        self, terrain_feature: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        if terrain_feature.ndim != 4:
            raise ValueError("terrain_feature must have shape [B,C,H,W]")
        required_channels = 1 + max(
            index for indices in self.response_groups for index in indices
        )
        if terrain_feature.shape[1] < required_channels:
            raise ValueError(
                f"terrain_feature needs at least {required_channels} channels"
            )
        detached = terrain_feature.detach()
        responses = [
            detached[:, indices].mean(dim=1, keepdim=True)
            for indices in self.response_groups
        ]
        basis = torch.cat(responses, dim=1)
        if basis.shape[-2:] != output_size:
            basis = F.interpolate(
                basis, size=output_size, mode="bilinear", align_corners=False
            )
        return torch.tanh(basis)

    @staticmethod
    def _quality(q_m: torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        quality = q_m.to(device=device, dtype=dtype).reshape(-1)
        if quality.shape != (batch,):
            raise ValueError("q_m must contain one scalar per sample")
        return quality

    def forward(
        self,
        terrain_feature: torch.Tensor,
        terrain_residual: torch.Tensor,
        material: torch.Tensor,
        q_m: torch.Tensor,
        *,
        abstain: torch.Tensor | None = None,
        context: str = CONTEXT_ALIGNED,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if context not in ALLOWED_CONTEXTS:
            raise ValueError(f"context must be one of {ALLOWED_CONTEXTS}")
        if terrain_residual.ndim != 4 or terrain_residual.shape[1] != 1:
            raise ValueError("terrain_residual must have shape [B,1,H,W]")
        batch = terrain_residual.shape[0]
        if material.shape != (batch, MATERIAL_FEATURE_COUNT):
            raise ValueError(
                f"material must have shape [B,{MATERIAL_FEATURE_COUNT}]"
            )
        if terrain_feature.shape[0] != batch:
            raise ValueError("terrain_feature batch differs from terrain_residual")
        if not torch.isfinite(terrain_feature).all() or not torch.isfinite(terrain_residual).all():
            raise ValueError("Terrain inputs must be finite")

        device, dtype = terrain_residual.device, terrain_residual.dtype
        material = material.to(device=device, dtype=dtype).detach()
        quality = self._quality(q_m, batch, device, dtype).detach()
        if abstain is None:
            abstain_mask = torch.zeros(batch, device=device, dtype=torch.bool)
        else:
            abstain_mask = abstain.to(device=device, dtype=torch.bool).reshape(-1)
            if abstain_mask.shape != (batch,):
                raise ValueError("abstain must contain one flag per sample")

        finite_material = torch.isfinite(material).all(dim=1)
        finite_quality = torch.isfinite(quality)
        quality = torch.where(finite_quality, quality.clamp(0.0, 1.0), torch.zeros_like(quality))
        force_inactive = context in (CONTEXT_ZERO_Q, CONTEXT_ABSTAIN)
        active = finite_material & finite_quality & ~abstain_mask & (quality > 0.0)
        if force_inactive:
            active = torch.zeros_like(active)
        effective_q = torch.where(active, quality, torch.zeros_like(quality))

        clean_material = torch.where(
            torch.isfinite(material), material, torch.zeros_like(material)
        )
        basis = self._terrain_basis(terrain_feature, terrain_residual.shape[-2:])
        coefficients = self.interaction_head(clean_material)
        raw_map = torch.sum(
            coefficients[:, :, None, None] * basis, dim=1, keepdim=True
        ) / math.sqrt(len(self.response_names))
        delta = effective_q[:, None, None, None] * self.modulation_bound * torch.tanh(raw_map)
        ones = torch.ones_like(terrain_residual)
        active_map = active[:, None, None, None]
        multiplier = torch.where(active_map, ones + delta, ones)

        base = terrain_residual.detach()
        candidate = base * multiplier
        conditioned = torch.where(active_map, candidate, base)
        interaction = torch.where(
            active_map, candidate - base, torch.zeros_like(base)
        )
        audit_coefficients = torch.where(
            active[:, None], coefficients, torch.zeros_like(coefficients)
        )
        return conditioned, {
            "context": context,
            "terrain_response_names": self.response_names,
            "terrain_response_basis_maps": basis,
            "material_response_coefficients": audit_coefficients,
            "material_multiplier_map": multiplier,
            "material_interaction_map": interaction,
            "conditioned_terrain_residual": conditioned,
            "q_M_effective": effective_q[:, None, None, None],
            "material_active": active[:, None, None, None],
            "modulation_bounds": (
                1.0 - self.modulation_bound,
                1.0 + self.modulation_bound,
            ),
            "material_dense_direction": False,
        }

    def forward_context(
        self,
        terrain_feature: torch.Tensor,
        terrain_residual: torch.Tensor,
        context_batch: MaterialContextBatch,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self(
            terrain_feature,
            terrain_residual,
            context_batch.material,
            context_batch.q_m,
            abstain=context_batch.abstain,
            context=context_batch.name,
        )
