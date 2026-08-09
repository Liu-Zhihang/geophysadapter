#!/usr/bin/env python3
"""Strict matched Sen12Landslides LOGO-5 trainer for GeoPhysAdapter.

Each invocation trains exactly one mode/fold/seed run.  The visual mode uses a
frozen, shared-weight Hiera-MAE twin encoder and trains only a matched FPN
decoder over pre/post/absolute-difference features.  The adapter mode loads the
matched visual checkpoint, freezes the entire visual model, and learns a
bounded Terrain correction.  Material and Trigger are deliberately abstained
from until auditable Sen12 support is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shlex
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_baseline_ladder_unet import ConvBlock  # noqa: E402
from run_support_adapter_timmfm import TimmFeatureEncoder, TimmFPNHead  # noqa: E402


H5_DEFAULT = PROJECT_ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5"
SPLIT_DEFAULT = PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv"
OUT_DEFAULT = PROJECT_ROOT / "experiments/revision2026/sen12_xdomain_geophysadapter"
CONTROLS = ("aligned", "zero", "roll32", "roll64", "other_region_donor")
ADAPTER_EVAL_CONTROLS = ("visual_anchor", *CONTROLS)
COUNT_KEYS = ("tp", "fp", "fn", "tn")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def decode_strings(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in values]


def as_chw(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 2:
        return value[None]
    if value.ndim != 3:
        raise ValueError(f"{name} must be 2D/3D per sample, got shape={value.shape}")
    if value.shape[0] <= 64 and value.shape[1] > 16 and value.shape[2] > 16:
        return value
    if value.shape[-1] <= 64 and value.shape[0] > 16 and value.shape[1] > 16:
        return np.moveaxis(value, -1, 0)
    raise ValueError(f"Cannot infer CHW layout for {name}, shape={value.shape}")


def binary_counts(target: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    y = target.astype(bool)
    p = prediction.astype(bool)
    return {
        "tp": int(np.logical_and(y, p).sum()),
        "fp": int(np.logical_and(~y, p).sum()),
        "fn": int(np.logical_and(y, ~p).sum()),
        "tn": int(np.logical_and(~y, ~p).sum()),
    }


def add_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key in COUNT_KEYS:
        dst[key] += int(src[key])


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (counts[k] for k in COUNT_KEYS)
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


class ProbabilityHistogram:
    """Streaming, deterministic probability metrics without retaining all pixels."""

    def __init__(self, bins: int = 4096) -> None:
        self.bins = int(bins)
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)
        self.brier_sum = 0.0
        self.n = 0

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        p = np.clip(np.asarray(probability, dtype=np.float64).reshape(-1), 0.0, 1.0)
        y = np.asarray(target).reshape(-1) > 0.5
        indices = np.minimum((p * self.bins).astype(np.int64), self.bins - 1)
        self.positive += np.bincount(indices[y], minlength=self.bins)
        self.negative += np.bincount(indices[~y], minlength=self.bins)
        self.brier_sum += float(np.square(p - y.astype(np.float64)).sum())
        self.n += int(p.size)

    @property
    def average_precision(self) -> float:
        total_positive = int(self.positive.sum())
        if total_positive == 0:
            return 0.0
        tp = np.cumsum(self.positive[::-1], dtype=np.int64)
        fp = np.cumsum(self.negative[::-1], dtype=np.int64)
        precision = tp / np.maximum(tp + fp, 1)
        recall_increment = self.positive[::-1] / total_positive
        return float(np.sum(precision * recall_increment))

    @property
    def brier(self) -> float:
        return self.brier_sum / max(self.n, 1)

    def counts_at(self, threshold: float) -> dict[str, int]:
        first_positive = min(max(int(math.ceil(threshold * self.bins)), 0), self.bins)
        tp = int(self.positive[first_positive:].sum())
        fp = int(self.negative[first_positive:].sum())
        fn = int(self.positive[:first_positive].sum())
        tn = int(self.negative[:first_positive].sum())
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_threshold(histogram: ProbabilityHistogram) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_iou = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = metrics_from_counts(histogram.counts_at(float(threshold)))
        if metrics["iou"] > best_iou:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_iou = metrics["iou"]
    assert best_metrics is not None
    return best_threshold, best_metrics


@dataclass(frozen=True)
class H5Schema:
    pre_key: str
    post_key: str
    terrain_key: str
    mask_key: str
    valid_mask_key: str | None
    sample_id_key: str
    q_t_key: str | None
    pre_channels: tuple[int, ...] | None = None
    post_channels: tuple[int, ...] | None = None
    terrain_channels: tuple[int, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


def first_existing(h5: h5py.File, aliases: Sequence[str]) -> str | None:
    return next((name for name in aliases if name in h5), None)


def discover_h5_schema(path: Path) -> H5Schema:
    with h5py.File(path, "r") as h5:
        pre = first_existing(h5, ("pre_rgb", "x_pre", "pre", "visual_pre", "image_pre"))
        post = first_existing(h5, ("post_rgb", "x_post", "post", "visual_post", "image_post"))
        terrain = first_existing(h5, ("terrain", "x_terrain", "terrain_dense", "dem_features", "dem"))
        mask = first_existing(h5, ("mask", "y", "label", "labels"))
        valid = first_existing(h5, ("valid_mask", "valid", "pixel_valid"))
        sample_id = first_existing(h5, ("sample_id", "patch_id", "id"))
        q_t = first_existing(h5, ("q_T", "q_t", "terrain_support", "terrain_valid", "terrain_quality"))
        if mask is None or sample_id is None:
            raise KeyError(f"H5 requires mask and sample IDs; keys={sorted(h5.keys())}")
        if pre and post and terrain:
            return H5Schema(pre, post, terrain, mask, valid, sample_id, q_t)

        if "obs" in h5 and terrain:
            if h5["obs"].ndim != 4 or h5["obs"].shape[1] != 6:
                raise KeyError(f"obs cache contract requires [N,6,H,W], got {h5['obs'].shape}")
            return H5Schema(
                "obs",
                "obs",
                terrain,
                mask,
                valid,
                sample_id,
                q_t,
                (0, 1, 2),
                (3, 4, 5),
                None,
            )

        if "x" not in h5 or "channel_names" not in h5:
            raise KeyError(
                "H5 requires explicit pre/post/terrain arrays, or x plus channel_names; "
                f"keys={sorted(h5.keys())}"
            )
        names = [name.lower() for name in decode_strings(h5["channel_names"][:])]

        def ordered_rgb(prefix: str) -> tuple[int, ...]:
            ordered: list[int] = []
            for aliases in (("red", "b04"), ("green", "b03"), ("blue", "b02")):
                matches = [
                    index
                    for index, name in enumerate(names)
                    if name.startswith(prefix) and any(alias in name for alias in aliases)
                ]
                if len(matches) != 1:
                    return ()
                ordered.append(matches[0])
            return tuple(ordered)

        pre_idx = ordered_rgb("pre_")
        post_idx = ordered_rgb("post_")
        terrain_idx = tuple(i for i, name in enumerate(names) if any(token in name for token in ("dem", "slope", "aspect", "curvature", "roughness", "terrain")))
        if len(pre_idx) != 3 or len(post_idx) != 3 or not terrain_idx:
            raise KeyError(
                "Could not identify exactly three pre/post RGB channels and at least one Terrain channel from "
                f"channel_names={names}"
            )
        return H5Schema(
            "x", "x", "x", mask, valid, sample_id, q_t, pre_idx, post_idx, terrain_idx
        )


def load_logo_rows(path: Path, fold: int) -> tuple[dict[str, dict[str, str]], dict[str, list[str]], dict[str, list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "outer_fold", "role", "region_group", "spatial_supergroup"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"LOGO CSV missing columns={sorted(missing)}")
        rows = [row for row in reader if int(row["outer_fold"]) == fold]
    if not rows:
        raise ValueError(f"No rows for outer_fold={fold}")
    by_id: dict[str, dict[str, str]] = {}
    roles: dict[str, list[str]] = {role: [] for role in ("train", "val", "test")}
    regions: dict[str, list[str]] = {role: [] for role in roles}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in by_id:
            raise ValueError(f"Duplicate sample_id={sample_id} within fold={fold}")
        role = row["role"]
        if role not in roles:
            raise ValueError(f"Unknown role={role!r}")
        by_id[sample_id] = row
        roles[role].append(sample_id)
        regions[role].append(row["spatial_supergroup"])
    for role, values in roles.items():
        if not values:
            raise ValueError(f"Empty {role} split for fold={fold}")
    split_sets = {role: set(values) for role, values in roles.items()}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Sample leakage across train/val/test")
    region_sets = {role: set(values) for role, values in regions.items()}
    if any(region_sets[a] & region_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError(f"Spatial-supergroup leakage: {region_sets}")
    return by_id, roles, regions


class Sen12H5Dataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        schema: H5Schema,
        rows_by_id: dict[str, dict[str, str]],
        sample_ids: Sequence[str],
        reflectance_scale: float,
        terrain_mean: np.ndarray,
        terrain_std: np.ndarray,
        seed: int,
        donor_sample_ids: Sequence[str] | None = None,
    ) -> None:
        self.h5_path = h5_path
        self.schema = schema
        self.rows_by_id = rows_by_id
        self.reflectance_scale = float(reflectance_scale)
        self.terrain_mean = np.asarray(terrain_mean, dtype=np.float32)[:, None, None]
        self.terrain_std = np.asarray(terrain_std, dtype=np.float32)[:, None, None]
        self._h5: h5py.File | None = None
        with h5py.File(h5_path, "r") as h5:
            all_ids = decode_strings(h5[schema.sample_id_key][:])
            if "physical_event_id" not in h5:
                raise KeyError("H5 cache must contain physical_event_id for event-level inference")
            all_event_ids = decode_strings(h5["physical_event_id"][:])
        if len(all_event_ids) != len(all_ids):
            raise ValueError("H5 physical_event_id length differs from sample_id length")
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("H5 sample IDs are not unique")
        index_by_id = {sample_id: idx for idx, sample_id in enumerate(all_ids)}
        missing = sorted(set(sample_ids) - set(index_by_id))
        if missing:
            raise KeyError(f"{len(missing)} LOGO samples missing from H5, examples={missing[:8]}")
        self.sample_ids = list(sample_ids)
        self.indices = [index_by_id[sample_id] for sample_id in self.sample_ids]
        self.regions = [rows_by_id[sample_id]["spatial_supergroup"] for sample_id in self.sample_ids]
        self.event_ids = [all_event_ids[index_by_id[sample_id]] for sample_id in self.sample_ids]
        self.index_by_id = index_by_id
        self.donor_pool_ids = list(donor_sample_ids or self.sample_ids)
        missing_donors = sorted(set(self.donor_pool_ids) - set(index_by_id))
        if missing_donors:
            raise KeyError(f"Donor pool samples missing from H5: {missing_donors[:8]}")
        self.donor_ids, self.donor_indices, self.donor_regions = self._build_donors(seed)

    def _build_donors(self, seed: int) -> tuple[list[str], list[int], list[str]]:
        available = sorted(self.donor_pool_ids)
        donor_ids: list[str] = []
        donor_indices: list[int] = []
        donor_regions: list[str] = []
        for sample_id, region in zip(self.sample_ids, self.regions):
            candidates = [
                candidate
                for candidate in available
                if self.rows_by_id[candidate]["spatial_supergroup"] != region
            ]
            if not candidates:
                raise ValueError(f"No other-region Terrain donor is available for region={region}")
            token = hashlib.sha256(f"{seed}|{sample_id}|donor".encode("utf-8")).digest()
            donor_id = candidates[int.from_bytes(token[:8], "big") % len(candidates)]
            donor_ids.append(donor_id)
            donor_indices.append(self.index_by_id[donor_id])
            donor_regions.append(self.rows_by_id[donor_id]["spatial_supergroup"])
        return donor_ids, donor_indices, donor_regions

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def _read_channels(self, h5: h5py.File, key: str, index: int, channels: tuple[int, ...] | None, name: str) -> np.ndarray:
        value = as_chw(np.asarray(h5[key][index]), name)
        return value if channels is None else value[list(channels)]

    def _read_terrain(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        h5 = self._file()
        terrain = self._read_channels(h5, self.schema.terrain_key, index, self.schema.terrain_channels, "terrain").astype(np.float32)
        finite = np.all(np.isfinite(terrain), axis=0, keepdims=True)
        terrain = np.nan_to_num(terrain, nan=0.0, posinf=0.0, neginf=0.0)
        terrain = (terrain - self.terrain_mean) / self.terrain_std
        if self.schema.q_t_key is None:
            q_t = finite.astype(np.float32)
        else:
            raw_q_t = np.asarray(h5[self.schema.q_t_key][index])
            if raw_q_t.ndim == 0:
                q_t = np.full((1, terrain.shape[-2], terrain.shape[-1]), float(raw_q_t), dtype=np.float32)
            else:
                q_t = as_chw(raw_q_t, "q_T").astype(np.float32)
            if q_t.shape[0] != 1:
                q_t = np.min(q_t, axis=0, keepdims=True)
            q_t = np.clip(np.nan_to_num(q_t, nan=0.0), 0.0, 1.0) * finite.astype(np.float32)
        return terrain, q_t

    def __getitem__(self, position: int) -> dict[str, Any]:
        h5 = self._file()
        index = self.indices[position]
        pre = self._read_channels(h5, self.schema.pre_key, index, self.schema.pre_channels, "pre_rgb").astype(np.float32)
        post = self._read_channels(h5, self.schema.post_key, index, self.schema.post_channels, "post_rgb").astype(np.float32)
        if pre.shape[0] != 3 or post.shape[0] != 3:
            raise ValueError(f"Hiera RGB contract requires 3+3 channels, got pre={pre.shape}, post={post.shape}")
        pre = np.clip(np.nan_to_num(pre / self.reflectance_scale, nan=0.0), 0.0, 1.0)
        post = np.clip(np.nan_to_num(post / self.reflectance_scale, nan=0.0), 0.0, 1.0)
        terrain, q_t = self._read_terrain(index)
        donor_terrain, donor_q_t = self._read_terrain(self.donor_indices[position])
        mask = as_chw(np.asarray(h5[self.schema.mask_key][index]), "mask").astype(np.float32)
        if mask.shape[0] != 1:
            mask = np.max(mask, axis=0, keepdims=True)
        mask = (mask > 0.5).astype(np.float32)
        if self.schema.valid_mask_key is None:
            valid = np.ones_like(mask, dtype=np.float32)
        else:
            valid = as_chw(
                np.asarray(h5[self.schema.valid_mask_key][index]), "valid_mask"
            ).astype(np.float32)
            if valid.shape[0] != 1:
                valid = np.min(valid, axis=0, keepdims=True)
            valid = (valid > 0.5).astype(np.float32)
        sample_id = self.sample_ids[position]
        donor_sample_id = self.donor_ids[position]
        return {
            "pre": torch.from_numpy(pre),
            "post": torch.from_numpy(post),
            "terrain": torch.from_numpy(terrain),
            "q_t": torch.from_numpy(q_t),
            "mask": torch.from_numpy(mask),
            "valid": torch.from_numpy(valid),
            "sample_id": sample_id,
            "event_id": self.event_ids[position],
            "region": self.regions[position],
            "donor_terrain": torch.from_numpy(donor_terrain),
            "donor_q_t": torch.from_numpy(donor_q_t),
            "donor_sample_id": donor_sample_id,
            "donor_region": self.donor_regions[position],
        }


def deterministic_subset(sample_ids: list[str], limit: int, seed: int, role: str) -> list[str]:
    if limit <= 0 or limit >= len(sample_ids):
        return list(sample_ids)
    ordered = sorted(sample_ids, key=lambda value: hashlib.sha256(f"{seed}|{role}|{value}".encode()).hexdigest())
    return ordered[:limit]


def inspect_reflectance_scale(path: Path, schema: H5Schema, sample_ids: Sequence[str]) -> float:
    with h5py.File(path, "r") as h5:
        all_ids = decode_strings(h5[schema.sample_id_key][:])
        index_by_id = {sample_id: idx for idx, sample_id in enumerate(all_ids)}
        maxima: list[float] = []
        for sample_id in list(sample_ids)[: min(32, len(sample_ids))]:
            idx = index_by_id[sample_id]
            for key, channels in ((schema.pre_key, schema.pre_channels), (schema.post_key, schema.post_channels)):
                value = as_chw(np.asarray(h5[key][idx]), key)
                if channels is not None:
                    value = value[list(channels)]
                finite = value[np.isfinite(value)]
                if finite.size:
                    maxima.append(float(np.percentile(finite, 99.5)))
    if not maxima:
        raise ValueError("No finite RGB values available to infer reflectance scale")
    return 10000.0 if float(np.median(maxima)) > 2.0 else 1.0


def estimate_terrain_stats(
    path: Path,
    schema: H5Schema,
    sample_ids: Sequence[str],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = list(sample_ids)[: max_samples if max_samples > 0 else len(sample_ids)]
    sums: np.ndarray | None = None
    squared: np.ndarray | None = None
    counts: np.ndarray | None = None
    with h5py.File(path, "r") as h5:
        all_ids = decode_strings(h5[schema.sample_id_key][:])
        index_by_id = {sample_id: idx for idx, sample_id in enumerate(all_ids)}
        for sample_id in selected:
            value = as_chw(np.asarray(h5[schema.terrain_key][index_by_id[sample_id]]), "terrain")
            if schema.terrain_channels is not None:
                value = value[list(schema.terrain_channels)]
            value = value.astype(np.float64)
            finite = np.isfinite(value)
            safe = np.where(finite, value, 0.0)
            channel_sum = safe.sum(axis=(1, 2))
            channel_sq = np.square(safe).sum(axis=(1, 2))
            channel_count = finite.sum(axis=(1, 2)).astype(np.float64)
            if sums is None:
                sums = np.zeros_like(channel_sum)
                squared = np.zeros_like(channel_sq)
                counts = np.zeros_like(channel_count)
            sums += channel_sum
            squared += channel_sq
            counts += channel_count
    assert sums is not None and squared is not None and counts is not None
    mean = sums / np.maximum(counts, 1.0)
    variance = squared / np.maximum(counts, 1.0) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class TwinHieraVisual(nn.Module):
    """Shared Hiera encoder with per-scale pre/post/change fusion and FPN."""

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        image_size: int,
        out_indices: tuple[int, ...],
        hidden: int,
    ) -> None:
        super().__init__()
        self.encoder = TimmFeatureEncoder(
            backbone,
            in_chans=3,
            pretrained=pretrained,
            img_size=image_size,
            out_indices=out_indices,
            freeze_backbone=True,
        )
        self.decoder = TimmFPNHead([3 * channels for channels in self.encoder.channels], hidden=hidden)
        self.image_size = int(image_size)
        self.hidden = int(hidden)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=True)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=True)
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "TwinHieraVisual":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = (pre - self.image_mean) / self.image_std
        post = (post - self.image_mean) / self.image_std
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        fused = [torch.cat([a, b, torch.abs(b - a)], dim=1) for a, b in zip(pre_features, post_features)]
        return self.decoder(fused, pre.shape[-2:])


class TerrainCorrectionAdapter(nn.Module):
    """Visual-only gate plus a centered, bounded Terrain correction direction."""

    def __init__(
        self,
        visual: TwinHieraVisual,
        terrain_channels: int,
        hidden: int,
        terrain_base: int,
        alpha_max: float,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.alpha_max = float(alpha_max)
        self.terrain_encoder = nn.Sequential(
            ConvBlock(terrain_channels, terrain_base),
            ConvBlock(terrain_base, terrain_base),
        )
        self.terrain_direction = nn.Sequential(
            nn.Conv2d(terrain_base, hidden, 3, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 4, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, 3, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 4, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.terrain_direction[-1].weight)
        nn.init.zeros_(self.terrain_direction[-1].bias)
        self.freeze_visual()

    def freeze_visual(self) -> None:
        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)
        self.visual.eval()

    def train(self, mode: bool = True) -> "TerrainCorrectionAdapter":
        super().train(mode)
        self.visual.eval()
        return self

    def visual_forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, feature = self.visual(pre, post)
        probability = torch.sigmoid(logits.detach())
        uncertainty = 1.0 - torch.abs(2.0 * probability - 1.0)
        return logits.detach(), feature.detach(), uncertainty.detach()

    def support_forward(
        self,
        visual_logits: torch.Tensor,
        visual_feature: torch.Tensor,
        uncertainty: torch.Tensor,
        terrain: torch.Tensor,
        q_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate = torch.sigmoid(self.gate_head(torch.cat([visual_feature.detach(), uncertainty.detach()], dim=1)))
        gate = gate * uncertainty.detach()
        direction = self.terrain_direction(self.terrain_encoder(terrain))
        zero_direction = self.terrain_direction(self.terrain_encoder(torch.zeros_like(terrain)))
        bounded_residual = self.alpha_max * torch.tanh(direction - zero_direction)
        q_t = torch.clamp(q_t, 0.0, 1.0)
        correction = q_t * gate * bounded_residual
        return visual_logits + correction, {
            "visual_logits": visual_logits,
            "visual_feature": visual_feature,
            "uncertainty": uncertainty,
            "gate": gate,
            "bounded_residual": bounded_residual,
            "correction": correction,
            "q_t": q_t,
        }

    def forward(self, pre: torch.Tensor, post: torch.Tensor, terrain: torch.Tensor, q_t: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        visual_logits, visual_feature, uncertainty = self.visual_forward(pre, post)
        return self.support_forward(visual_logits, visual_feature, uncertainty, terrain, q_t)

    def forward_controls(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        terrain: torch.Tensor,
        q_t: torch.Tensor,
        donor_terrain: torch.Tensor,
        donor_q_t: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        visual_logits, visual_feature, uncertainty = self.visual_forward(pre, post)
        zeros = torch.zeros_like(visual_logits)
        variants = {
            "aligned": (terrain, q_t),
            "zero": (torch.zeros_like(terrain), torch.ones_like(q_t)),
            "roll32": (torch.roll(terrain, shifts=(32, 32), dims=(-2, -1)), torch.roll(q_t, shifts=(32, 32), dims=(-2, -1))),
            "roll64": (torch.roll(terrain, shifts=(64, 64), dims=(-2, -1)), torch.roll(q_t, shifts=(64, 64), dims=(-2, -1))),
            "other_region_donor": (donor_terrain, donor_q_t),
        }
        outputs = {
            name: self.support_forward(visual_logits, visual_feature, uncertainty, support, quality)
            for name, (support, quality) in variants.items()
        }
        outputs["visual_anchor"] = (
            visual_logits,
            {
                "visual_logits": visual_logits,
                "visual_feature": visual_feature,
                "uncertainty": uncertainty,
                "gate": zeros,
                "bounded_residual": zeros,
                "correction": zeros,
                "q_t": zeros,
            },
        )
        return outputs


def dice_loss_per_sample(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    probability = probability * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def estimate_pos_weight(dataset: Dataset, max_weight: float = 40.0) -> float:
    positive = 0.0
    total = 0
    for index in range(len(dataset)):
        mask = dataset[index]["mask"]
        valid = dataset[index]["valid"]
        positive += float((mask * valid).sum())
        total += int(valid.sum())
    return float(min(max_weight, max(1.0, (total - positive) / max(positive, 1.0))))


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = None
    if shuffle:
        event_counts = Counter(dataset.event_ids)
        weights = torch.tensor(
            [1.0 / event_counts[event_id] for event_id in dataset.event_ids],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def autocast_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


@torch.no_grad()
def score_validation_ap(model: nn.Module, loader: DataLoader, mode: str, device: torch.device, amp: bool) -> tuple[float, ProbabilityHistogram]:
    model.eval()
    histogram = ProbabilityHistogram()
    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        with autocast_context(amp):
            if mode == "visual":
                logits, _ = model(pre, post)
            else:
                logits, _ = model(
                    pre,
                    post,
                    batch["terrain"].to(device, non_blocking=True),
                    batch["q_t"].to(device, non_blocking=True),
                )
        probability = torch.sigmoid(logits).float().cpu().numpy()
        target = batch["mask"].numpy()
        valid = batch["valid"].numpy() > 0.5
        histogram.update(probability[valid], target[valid])
    return histogram.average_precision, histogram


def train_model(
    model: nn.Module,
    mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    pos_weight: float,
    log,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    pos_weight_tensor = torch.tensor([pos_weight], device=args.device)
    best_ap = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            pre = batch["pre"].to(args.device, non_blocking=True)
            post = batch["post"].to(args.device, non_blocking=True)
            target = batch["mask"].to(args.device, non_blocking=True)
            valid = batch["valid"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args.amp):
                if mode == "visual":
                    logits, _ = model(pre, post)
                    diagnostics = None
                else:
                    logits, diagnostics = model(
                        pre,
                        post,
                        batch["terrain"].to(args.device, non_blocking=True),
                        batch["q_t"].to(args.device, non_blocking=True),
                    )
                bce = F.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    pos_weight=pos_weight_tensor.view(1, 1, 1, 1),
                    reduction="none",
                )
                bce = (bce * valid).flatten(1).sum(dim=1) / valid.flatten(1).sum(dim=1).clamp_min(1.0)
                loss = (
                    bce + args.dice_weight * dice_loss_per_sample(logits, target, valid)
                ).mean()
                if diagnostics is not None:
                    loss = loss + args.gate_l1 * diagnostics["gate"].mean()
                    loss = loss + args.residual_l1 * diagnostics["bounded_residual"].abs().mean()
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * pre.shape[0]
            seen += pre.shape[0]
            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        val_ap, _ = score_validation_ap(model, val_loader, mode, args.device, args.amp)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(seen, 1),
            "val_average_precision": val_ap,
        }
        history.append(row)
        log(
            f"[epoch] {epoch}/{args.epochs} step={global_step} "
            f"loss={row['train_loss']:.6f} val_ap={val_ap:.6f}"
        )
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    return best_state, history, best_epoch


def sample_probability_metrics(probability: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    histogram = ProbabilityHistogram(bins=1024)
    histogram.update(probability, target)
    return histogram.average_precision, histogram.brier


@torch.no_grad()
def evaluate(
    model: nn.Module,
    mode: str,
    loader: DataLoader,
    threshold: float,
    split: str,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    model.eval()
    controls = ("visual",) if mode == "visual" else ADAPTER_EVAL_CONTROLS
    global_counts = {control: {key: 0 for key in COUNT_KEYS} for control in controls}
    global_hist = {control: ProbabilityHistogram() for control in controls}
    region_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {control: {key: 0 for key in COUNT_KEYS} for control in controls}
    )
    region_hist: dict[str, dict[str, ProbabilityHistogram]] = defaultdict(
        lambda: {control: ProbabilityHistogram() for control in controls}
    )
    event_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {control: {key: 0 for key in COUNT_KEYS} for control in controls}
    )
    event_hist: dict[str, dict[str, ProbabilityHistogram]] = defaultdict(
        lambda: {control: ProbabilityHistogram() for control in controls}
    )
    sample_rows: list[dict[str, Any]] = []
    ids_by_control: dict[str, list[str]] = {control: [] for control in controls}
    max_zero_delta = 0.0
    max_q0_delta = 0.0
    donor_region_violations = 0
    for batch in loader:
        pre = batch["pre"].to(args.device, non_blocking=True)
        post = batch["post"].to(args.device, non_blocking=True)
        with autocast_context(args.amp):
            if mode == "visual":
                visual_logits, visual_feature = model(pre, post)
                outputs = {
                    "visual": (
                        visual_logits,
                        {
                            "visual_logits": visual_logits,
                            "gate": torch.zeros_like(visual_logits),
                            "bounded_residual": torch.zeros_like(visual_logits),
                            "correction": torch.zeros_like(visual_logits),
                            "q_t": torch.zeros_like(visual_logits),
                        },
                    )
                }
            else:
                outputs = model.forward_controls(
                    pre,
                    post,
                    batch["terrain"].to(args.device, non_blocking=True),
                    batch["q_t"].to(args.device, non_blocking=True),
                    batch["donor_terrain"].to(args.device, non_blocking=True),
                    batch["donor_q_t"].to(args.device, non_blocking=True),
                )
                visual_logits = outputs["aligned"][1]["visual_logits"]
                visual_feature = None
                q0_logits, _ = model.support_forward(
                    visual_logits,
                    outputs["aligned"][1]["visual_feature"],
                    outputs["aligned"][1]["uncertainty"],
                    batch["terrain"].to(args.device, non_blocking=True),
                    torch.zeros_like(batch["q_t"].to(args.device, non_blocking=True)),
                )
                max_q0_delta = max(max_q0_delta, float((q0_logits - visual_logits).abs().max().float().cpu()))
                max_zero_delta = max(
                    max_zero_delta,
                    float((outputs["zero"][0] - visual_logits).abs().max().float().cpu()),
                )
        target = batch["mask"].numpy() > 0.5
        valid = batch["valid"].numpy() > 0.5
        visual_probability = torch.sigmoid(visual_logits).float().cpu().numpy()
        visual_prediction = visual_probability >= threshold
        for control, (logits, diagnostics) in outputs.items():
            if torch.equal(logits, visual_logits):
                probability = visual_probability
                prediction = visual_prediction
            else:
                probability = torch.sigmoid(logits).float().cpu().numpy()
                prediction = probability >= threshold
            gate = diagnostics["gate"].float().cpu().numpy()
            residual = diagnostics["bounded_residual"].float().cpu().numpy()
            correction = diagnostics["correction"].float().cpu().numpy()
            q_t = diagnostics["q_t"].float().cpu().numpy()
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                region = str(batch["region"][index])
                event_id = str(batch["event_id"][index])
                keep = valid[index, 0]
                if not keep.any():
                    raise RuntimeError(f"Sample has no valid evaluation pixels: {sample_id}")
                target_i = target[index, 0][keep]
                prediction_i = prediction[index, 0][keep]
                probability_i = probability[index, 0][keep]
                visual_prediction_i = visual_prediction[index, 0][keep]
                ids_by_control[control].append(sample_id)
                counts = binary_counts(target_i, prediction_i)
                add_counts(global_counts[control], counts)
                add_counts(region_counts[region][control], counts)
                add_counts(event_counts[event_id][control], counts)
                global_hist[control].update(probability_i, target_i)
                region_hist[region][control].update(probability_i, target_i)
                event_hist[event_id][control].update(probability_i, target_i)
                visual_wrong = visual_prediction_i != target_i
                adapted_wrong = prediction_i != target_i
                corrected = int(np.logical_and(visual_wrong, ~adapted_wrong).sum())
                harmed = int(np.logical_and(~visual_wrong, adapted_wrong).sum())
                visual_errors = int(visual_wrong.sum())
                adapter_errors = int(adapted_wrong.sum())
                sample_ap, sample_brier = sample_probability_metrics(probability_i, target_i)
                row = {
                    "mode": mode,
                    "fold": args.fold,
                    "seed": args.seed,
                    "split": split,
                    "control": control,
                    "sample_id": sample_id,
                    "region_group": region,
                    "physical_event_id": event_id,
                    "threshold": threshold,
                    "valid_pixels": int(keep.sum()),
                    "positive_fraction": float(target_i.mean()),
                    "prediction_fraction": float(prediction_i.mean()),
                    "average_precision": sample_ap,
                    "brier": sample_brier,
                    "visual_errors": visual_errors,
                    "adapter_errors": adapter_errors,
                    "corrected": corrected,
                    "harmed": harmed,
                    "net_corrected": corrected - harmed,
                    "rer": (visual_errors - adapter_errors) / max(visual_errors, 1),
                    "gate_mean": float(gate[index, 0][keep].mean()),
                    "abs_bounded_residual_mean": float(np.abs(residual[index, 0][keep]).mean()),
                    "abs_correction_mean": float(np.abs(correction[index, 0][keep]).mean()),
                    "q_t_mean": float(q_t[index, 0][keep].mean()),
                    "donor_sample_id": str(batch["donor_sample_id"][index]) if control == "other_region_donor" else "",
                    "donor_region_group": str(batch["donor_region"][index]) if control == "other_region_donor" else "",
                    **counts,
                    **metrics_from_counts(counts),
                }
                if control == "other_region_donor" and row["donor_region_group"] == region:
                    donor_region_violations += 1
                sample_rows.append(row)

    region_rows: list[dict[str, Any]] = []
    for region in sorted(region_counts):
        for control in controls:
            counts = region_counts[region][control]
            region_rows.append(
                {
                    "mode": mode,
                    "fold": args.fold,
                    "seed": args.seed,
                    "split": split,
                    "control": control,
                    "region_group": region,
                    "average_precision": region_hist[region][control].average_precision,
                    "brier": region_hist[region][control].brier,
                    **counts,
                    **metrics_from_counts(counts),
                }
            )

    event_rows: list[dict[str, Any]] = []
    for event_id in sorted(event_counts):
        for control in controls:
            counts = event_counts[event_id][control]
            event_rows.append(
                {
                    "mode": mode,
                    "fold": args.fold,
                    "seed": args.seed,
                    "split": split,
                    "control": control,
                    "physical_event_id": event_id,
                    "average_precision": event_hist[event_id][control].average_precision,
                    "brier": event_hist[event_id][control].brier,
                    **counts,
                    **metrics_from_counts(counts),
                }
            )

    corpus_rows: list[dict[str, Any]] = []
    for control in controls:
        counts = global_counts[control]
        selected_regions = [row for row in region_rows if row["control"] == control]
        selected_events = [row for row in event_rows if row["control"] == control]
        visual_errors = counts["fp"] + counts["fn"] if mode == "visual" else global_counts[control]["fp"] + global_counts[control]["fn"]
        if mode == "adapter":
            visual_error_total = sum(int(row["visual_errors"]) for row in sample_rows if row["control"] == control)
            adapter_error_total = sum(int(row["adapter_errors"]) for row in sample_rows if row["control"] == control)
            corrected_total = sum(int(row["corrected"]) for row in sample_rows if row["control"] == control)
            harmed_total = sum(int(row["harmed"]) for row in sample_rows if row["control"] == control)
        else:
            visual_error_total = visual_errors
            adapter_error_total = visual_errors
            corrected_total = 0
            harmed_total = 0
        corpus_rows.append(
            {
                "mode": mode,
                "fold": args.fold,
                "seed": args.seed,
                "split": split,
                "control": control,
                "threshold": threshold,
                "n_samples": len(ids_by_control[control]),
                "n_regions": len(selected_regions),
                "n_physical_events": len(selected_events),
                "average_precision": global_hist[control].average_precision,
                "average_precision_method": f"streaming_histogram_{global_hist[control].bins}_bins",
                "brier": global_hist[control].brier,
                "region_macro_iou": float(np.mean([row["iou"] for row in selected_regions])),
                "region_macro_average_precision": float(np.mean([row["average_precision"] for row in selected_regions])),
                "region_macro_brier": float(np.mean([row["brier"] for row in selected_regions])),
                "event_macro_iou": float(np.mean([row["iou"] for row in selected_events])),
                "event_macro_average_precision": float(np.mean([row["average_precision"] for row in selected_events])),
                "event_macro_brier": float(np.mean([row["brier"] for row in selected_events])),
                "visual_errors": visual_error_total,
                "adapter_errors": adapter_error_total,
                "corrected": corrected_total,
                "harmed": harmed_total,
                "net_corrected": corrected_total - harmed_total,
                "rer": (visual_error_total - adapter_error_total) / max(visual_error_total, 1),
                **counts,
                **metrics_from_counts(counts),
            }
        )
    sample_hashes = {control: sha256_strings(ids_by_control[control]) for control in controls}
    identity_audit = {
        "split": split,
        "n_samples_by_control": {control: len(ids_by_control[control]) for control in controls},
        "sample_order_sha256_by_control": sample_hashes,
        "same_sample_identity_and_order": len(set(sample_hashes.values())) == 1,
        "zero_terrain_max_abs_logit_delta_from_visual": max_zero_delta,
        "q_t_zero_max_abs_logit_delta_from_visual": max_q0_delta,
        "zero_terrain_exact_fallback": max_zero_delta == 0.0,
        "q_t_zero_exact_fallback": max_q0_delta == 0.0,
        "other_region_donor_violations": donor_region_violations,
        "controls": {
            "visual_anchor": "frozen matched visual logits without physical correction",
            "aligned": "unchanged aligned Terrain and q_T",
            "zero": "normalized Terrain set to zero; q_T kept one to test centered identity",
            "roll32": "Terrain and q_T circularly rolled by (+32,+32) pixels",
            "roll64": "Terrain and q_T circularly rolled by (+64,+64) pixels",
            "other_region_donor": "deterministic outer-train donor from a different spatial_supergroup",
        } if mode == "adapter" else {"visual": "matched visual anchor"},
    }
    if not identity_audit["same_sample_identity_and_order"]:
        raise RuntimeError("Control identity audit failed")
    if mode == "adapter" and (not identity_audit["zero_terrain_exact_fallback"] or not identity_audit["q_t_zero_exact_fallback"]):
        raise RuntimeError(f"Centered physical identity audit failed: {identity_audit}")
    if donor_region_violations:
        raise RuntimeError(f"other-region donor audit failed for {donor_region_violations} samples")
    return sample_rows, region_rows, event_rows, {"rows": corpus_rows}, identity_audit


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def parse_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indices:
        raise argparse.ArgumentTypeError("at least one output index is required")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=H5_DEFAULT)
    parser.add_argument("--split-csv", type=Path, default=SPLIT_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=None, help="Atomic run directory; default encodes mode/fold/seed.")
    parser.add_argument("--mode", choices=("visual", "adapter"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--visual-checkpoint", type=Path, default=None, help="Required in adapter mode.")
    parser.add_argument("--backbone", default="hiera_small_224.mae")
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out-indices", type=parse_indices, default=parse_indices("0,1,2,3"))
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--terrain-base", type=int, default=32)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0, help="0 selects 3e-4 visual or 1e-3 adapter.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--gate-l1", type=float, default=1e-3)
    parser.add_argument("--residual-l1", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--reflectance-scale", default="auto", help="auto, 1, or 10000.")
    parser.add_argument("--terrain-stat-samples", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="Global optimizer-step cap for smoke runs.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        raise ValueError("CUDA is mandatory; CPU execution and silent CPU fallback are forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Requested {device}, available CUDA devices={torch.cuda.device_count()}")
    args.device = device
    if not args.h5.is_file():
        raise FileNotFoundError(args.h5)
    if not args.split_csv.is_file():
        raise FileNotFoundError(args.split_csv)
    if args.mode == "adapter" and (args.visual_checkpoint is None or not args.visual_checkpoint.is_file()):
        raise FileNotFoundError("adapter mode requires an existing --visual-checkpoint")
    if args.mode == "visual" and args.visual_checkpoint is not None:
        raise ValueError("--visual-checkpoint is only valid in adapter mode")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.alpha_max <= 0:
        raise ValueError("--alpha-max must be positive")
    args.lr = args.lr if args.lr > 0 else (3e-4 if args.mode == "visual" else 1e-3)
    if args.outdir is None:
        args.outdir = OUT_DEFAULT / f"{args.mode}_fold{args.fold}_seed{args.seed}"
    args.outdir = args.outdir.resolve()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to mix artifacts in non-empty run directory: {args.outdir}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    done_path = args.outdir / "DONE.json"
    done_path.unlink(missing_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    (args.outdir / "command.txt").write_text(command + "\n", encoding="utf-8")
    log_path = args.outdir / "run.log"
    log_path.write_text(command + "\n", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    started = time.time()
    try:
        set_seed(args.seed)
        schema = discover_h5_schema(args.h5)
        rows_by_id, roles, split_regions = load_logo_rows(args.split_csv, args.fold)
        with h5py.File(args.h5, "r") as handle:
            cache_sample_ids = set(decode_strings(handle[schema.sample_id_key][:]))
        logo_sample_ids = set().union(*(set(values) for values in roles.values()))
        unexpected_cache_ids = sorted(cache_sample_ids - logo_sample_ids)
        if unexpected_cache_ids:
            raise RuntimeError(
                f"H5 contains samples outside the frozen LOGO protocol: {unexpected_cache_ids[:20]}"
            )
        excluded_sample_ids = sorted(logo_sample_ids - cache_sample_ids)
        roles = {
            role: [sample_id for sample_id in values if sample_id in cache_sample_ids]
            for role, values in roles.items()
        }
        if any(not values for values in roles.values()):
            raise RuntimeError("Eligibility filtering emptied at least one train/val/test role")
        train_ids = deterministic_subset(roles["train"], args.max_train_samples, args.seed, "train")
        val_ids = deterministic_subset(roles["val"], args.max_eval_samples, args.seed, "val")
        test_ids = deterministic_subset(roles["test"], args.max_eval_samples, args.seed, "test")
        if args.reflectance_scale == "auto":
            reflectance_scale = inspect_reflectance_scale(args.h5, schema, train_ids)
        else:
            reflectance_scale = float(args.reflectance_scale)
            if reflectance_scale <= 0:
                raise ValueError("--reflectance-scale must be positive")
        stats_ids = deterministic_subset(train_ids, args.terrain_stat_samples, args.seed, "terrain_stats")
        terrain_mean, terrain_std = estimate_terrain_stats(args.h5, schema, stats_ids, 0)
        train_dataset = Sen12H5Dataset(
            args.h5,
            schema,
            rows_by_id,
            train_ids,
            reflectance_scale,
            terrain_mean,
            terrain_std,
            args.seed,
            donor_sample_ids=train_ids,
        )
        val_dataset = Sen12H5Dataset(
            args.h5,
            schema,
            rows_by_id,
            val_ids,
            reflectance_scale,
            terrain_mean,
            terrain_std,
            args.seed,
            donor_sample_ids=train_ids,
        )
        test_dataset = Sen12H5Dataset(
            args.h5,
            schema,
            rows_by_id,
            test_ids,
            reflectance_scale,
            terrain_mean,
            terrain_std,
            args.seed,
            donor_sample_ids=train_ids,
        )
        train_loader = make_loader(train_dataset, args, shuffle=True)
        val_loader = make_loader(val_dataset, args, shuffle=False)
        test_loader = make_loader(test_dataset, args, shuffle=False)
        terrain_channels = int(train_dataset[0]["terrain"].shape[0])
        split_csv_sha = sha256_file(args.split_csv)
        h5_signature = {
            "path": str(args.h5.resolve()),
            "size": args.h5.stat().st_size,
            "mtime_ns": args.h5.stat().st_mtime_ns,
        }
        config = {
            "contract": "Sen12 LOGO-5 matched visual-to-Terrain-adapter v1",
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "h5_signature": h5_signature,
            "split_csv": str(args.split_csv.resolve()),
            "split_csv_sha256": split_csv_sha,
            "schema": schema.as_dict(),
            "split_counts": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
            "change_view_excluded_count": len(excluded_sample_ids),
            "change_view_excluded_sample_ids": excluded_sample_ids,
            "training_sampler": "physical_event_balanced_weighted_resampling",
            "split_regions": {role: sorted(set(values)) for role, values in split_regions.items()},
            "sample_identity_sha256": {
                "train": sha256_strings(train_dataset.sample_ids),
                "val": sha256_strings(val_dataset.sample_ids),
                "test": sha256_strings(test_dataset.sample_ids),
            },
            "backbone": args.backbone,
            "pretrained_backbone": args.pretrained_backbone,
            "backbone_frozen": True,
            "visual_train_scope": "decoder_only" if args.mode == "visual" else "fully_frozen_matched_checkpoint",
            "twin_encoder_shared_weights": True,
            "per_scale_fusion": "concat(pre,post,abs(post-pre))",
            "reflectance_scale": reflectance_scale,
            "terrain_channels": terrain_channels,
            "terrain_mean": terrain_mean.tolist(),
            "terrain_std": terrain_std.tolist(),
            "terrain_role": "only_dense_correction_direction",
            "gate_inputs": "detached_visual_feature_and_detached_visual_uncertainty_only",
            "center_physical_correction": True,
            "material_multiplier_m_M": 1.0,
            "trigger_multiplier_tau_R": 1.0,
            "material_trigger_status": "abstained_not_fabricated",
            "evaluation_controls": list(ADAPTER_EVAL_CONTROLS) if args.mode == "adapter" else ["visual"],
            "terrain_donor_policy": "deterministic outer-train donor from a different spatial_supergroup; donor labels are never used",
            "threshold_policy": "selected_once_on_validation_after_AP_checkpoint_selection_then_frozen_for_test; adapter uses matched visual threshold",
            "args": vars(args),
            "command": command,
        }

        visual = TwinHieraVisual(
            args.backbone,
            pretrained=args.pretrained_backbone if args.mode == "visual" else False,
            image_size=args.image_size,
            out_indices=args.out_indices,
            hidden=args.hidden,
        )
        matched_visual_threshold: float | None = None
        visual_checkpoint_identity: dict[str, Any] | None = None
        if args.mode == "visual":
            model: nn.Module = visual
            for parameter in model.encoder.parameters():
                if parameter.requires_grad:
                    raise RuntimeError("Visual backbone must remain frozen")
        else:
            checkpoint = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
            identity = checkpoint.get("identity", {})
            required_identity = {
                "mode": "visual",
                "fold": args.fold,
                "seed": args.seed,
                "backbone": args.backbone,
                "split_csv_sha256": split_csv_sha,
                "h5_signature": h5_signature,
                "sample_identity_sha256": config["sample_identity_sha256"],
                "reflectance_scale": reflectance_scale,
                "image_size": args.image_size,
                "out_indices": list(args.out_indices),
                "hidden": args.hidden,
                "pretrained_backbone": args.pretrained_backbone,
            }
            mismatches = {
                key: {"expected": value, "actual": identity.get(key)}
                for key, value in required_identity.items()
                if identity.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"Matched visual checkpoint identity failed: {mismatches}")
            missing, unexpected = visual.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if missing or unexpected:
                raise RuntimeError(f"Visual state mismatch: missing={missing}, unexpected={unexpected}")
            matched_visual_threshold = float(checkpoint["threshold"])
            visual_checkpoint_identity = identity
            model = TerrainCorrectionAdapter(
                visual,
                terrain_channels=terrain_channels,
                hidden=args.hidden,
                terrain_base=args.terrain_base,
                alpha_max=args.alpha_max,
            )
        model = model.to(args.device)
        visual_module = model if args.mode == "visual" else model.visual
        visual_hash_before = state_sha256(visual_module)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if args.mode == "visual":
            decoder_parameters = sum(parameter.numel() for parameter in model.decoder.parameters())
            if trainable_parameters != decoder_parameters:
                raise RuntimeError(f"Visual mode must train decoder only: trainable={trainable_parameters}, decoder={decoder_parameters}")
        else:
            if any(parameter.requires_grad for parameter in model.visual.parameters()):
                raise RuntimeError("Adapter mode must freeze the entire visual model")
        config.update(
            {
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "visual_state_sha256_before": visual_hash_before,
                "visual_checkpoint": str(args.visual_checkpoint.resolve()) if args.visual_checkpoint else None,
                "visual_checkpoint_identity": visual_checkpoint_identity,
            }
        )
        (args.outdir / "config.json").write_text(json.dumps(json_safe(config), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        log(
            f"[setup] mode={args.mode} fold={args.fold} seed={args.seed} device={args.device} "
            f"train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} "
            f"trainable={trainable_parameters:,}/{total_parameters:,}"
        )
        pos_weight = estimate_pos_weight(train_dataset)
        best_state, history, best_epoch = train_model(model, args.mode, train_loader, val_loader, args, pos_weight, log)
        model.load_state_dict(best_state, strict=True)
        visual_hash_after = state_sha256(visual_module)
        if args.mode == "adapter" and visual_hash_after != visual_hash_before:
            raise RuntimeError("Frozen visual state changed during adapter training")
        val_ap, val_histogram = score_validation_ap(model, val_loader, args.mode, args.device, args.amp)
        if args.mode == "visual":
            threshold, threshold_metrics = choose_threshold(val_histogram)
            threshold_source = "matched_visual_validation"
        else:
            assert matched_visual_threshold is not None
            threshold = matched_visual_threshold
            threshold_metrics = metrics_from_counts(val_histogram.counts_at(threshold))
            threshold_source = "loaded_matched_visual_checkpoint"
        sample_rows: list[dict[str, Any]] = []
        region_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        corpus_rows: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for split, loader in (("val", val_loader), ("test", test_loader)):
            split_samples, split_regions_rows, split_event_rows, split_corpus, audit = evaluate(
                model, args.mode, loader, threshold, split, args
            )
            sample_rows.extend(split_samples)
            region_rows.extend(split_regions_rows)
            event_rows.extend(split_event_rows)
            corpus_rows.extend(split_corpus["rows"])
            audits.append(audit)
            for row in split_corpus["rows"]:
                log(
                    f"[eval] split={split} control={row['control']} iou={row['iou']:.6f} "
                    f"region_macro_iou={row['region_macro_iou']:.6f} "
                    f"event_macro_iou={row['event_macro_iou']:.6f} ap={row['average_precision']:.6f} "
                    f"brier={row['brier']:.6f} rer={row['rer']:.6f}"
                )
        checkpoint_identity = {
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "backbone": args.backbone,
            "split_csv_sha256": split_csv_sha,
            "h5_signature": h5_signature,
            "sample_identity_sha256": config["sample_identity_sha256"],
            "reflectance_scale": reflectance_scale,
            "image_size": args.image_size,
            "out_indices": list(args.out_indices),
            "hidden": args.hidden,
            "pretrained_backbone": args.pretrained_backbone,
            "visual_state_sha256": visual_hash_after,
        }
        checkpoint_payload = {
            "identity": checkpoint_identity,
            "model_state_dict": model.state_dict(),
            "threshold": threshold,
            "threshold_source": threshold_source,
            "best_epoch": best_epoch,
            "history": history,
            "config": config,
        }
        torch.save(checkpoint_payload, args.outdir / "checkpoint.pt")
        write_csv(sample_rows, args.outdir / "per_sample.csv")
        write_csv(region_rows, args.outdir / "per_region.csv")
        write_csv(event_rows, args.outdir / "per_event.csv")
        result = {
            "identity": checkpoint_identity,
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "history": history,
            "pos_weight": pos_weight,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "validation_checkpoint_average_precision": val_ap,
            "validation_threshold_metrics": threshold_metrics,
            "corpus_metrics": corpus_rows,
            "identity_and_control_audits": audits,
            "material_trigger_abstention": {
                "m_M": 1.0,
                "tau_R": 1.0,
                "reason": "Sen12 dense cache currently provides auditable Terrain only; Material/Trigger effects are not fabricated.",
            },
            "visual_state_sha256_before": visual_hash_before,
            "visual_state_sha256_after": visual_hash_after,
            "elapsed_seconds": time.time() - started,
            "artifact_contract": [
                "config.json",
                "command.txt",
                "run.log",
                "result.json",
                "per_sample.csv",
                "per_region.csv",
                "per_event.csv",
                "checkpoint.pt",
                "DONE.json",
            ],
        }
        (args.outdir / "result.json").write_text(
            json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        required = [
            "config.json",
            "command.txt",
            "run.log",
            "result.json",
            "per_sample.csv",
            "per_region.csv",
            "per_event.csv",
            "checkpoint.pt",
        ]
        missing_artifacts = [name for name in required if not (args.outdir / name).is_file() or (args.outdir / name).stat().st_size == 0]
        if missing_artifacts:
            raise RuntimeError(f"Artifact gate failed: {missing_artifacts}")
        done = {
            "status": "complete",
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "completed_unix": time.time(),
            "result_sha256": sha256_file(args.outdir / "result.json"),
            "checkpoint_sha256": sha256_file(args.outdir / "checkpoint.pt"),
        }
        done_path.write_text(json.dumps(done, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        log(f"[done] {args.outdir}")
        return 0
    except Exception:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("[FAILED]\n")
            handle.write(traceback.format_exc())
        done_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
