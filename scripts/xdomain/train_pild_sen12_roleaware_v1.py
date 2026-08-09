#!/usr/bin/env python3
"""Unified role-aware Prithvi training for the PILD + Sen12 protocol.

The five variants are hierarchical rather than symmetric feature fusions:

``V``
    A frozen Prithvi-EO-2.0 encoder with a trainable visual decoder.
``VT``
    The frozen visual anchor plus a bounded, audited Terrain residual.
``VTM``
    ``VT`` plus a 21-D Material context that may only modulate the existing
    Terrain residual.
``VTR``
    ``VT`` plus an event-scalar Trigger prior applied only where the frozen
    visual prediction is uncertain.
``VTMR``
    The two role-pure Material and Trigger interventions above.

Unsupported Material/Trigger rows are exact identities.  Non-visual variants
load a matched parent checkpoint and freeze every upstream component.  A run
is first assembled in a same-filesystem staging directory and is published by
one atomic rename only after all required evidence artifacts are complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pild_roleaware_material import (  # noqa: E402
    CONTEXT_ALIGNED as MATERIAL_ALIGNED,
    CONTEXT_SHUFFLED as MATERIAL_SHUFFLE,
    CONTEXT_ZERO_Q as MATERIAL_ZERO_Q,
    MATERIAL_FEATURE_COUNT,
    MATERIAL_FEATURE_NAMES,
    OuterTrainMaterialNormalizer,
    RoleAwareMaterialInteraction,
)
from pild_roleaware_trigger import (  # noqa: E402
    CONTEXT_ALIGNED as TRIGGER_ALIGNED,
    CONTEXT_EVENT_SHUFFLE as TRIGGER_EVENT_SHUFFLE,
    CONTEXT_WRONG_TIME as TRIGGER_WRONG_TIME,
    CONTEXT_ZERO_Q as TRIGGER_ZERO_Q,
    PILDRoleAwareTrigger,
    TriggerGateConfig,
    build_wrong_time_features,
)
from pild_sen12_training_loader_v2 import (  # noqa: E402
    DatasetEventPatchBalancedSampler,
    NaturalPatchSampler,
    REQUIRED_MANIFEST_COLUMNS,
    ROLE_MATERIAL_FEATURE_NAMES,
    SourceEventPatchBalancedSampler,
    TemperedDatasetEventPatchSampler,
    UnifiedPILDSen12Dataset,
    load_protocol_summary,
    sha256_file,
)
from sen12_prithvi_v2 import (  # noqa: E402
    PrithviEO2ChangeModel,
    load_prithvi_encoder,
)
from sen12_terrain_v2 import (  # noqa: E402
    BoundedTerrainAdapterV2,
    NATIVE_TERRAIN_V2_NAMES,
    NATIVE_TERRAIN_V2_SCALE_GROUPS,
    TerrainScaleGroups,
)


DEFAULT_METADATA = PROJECT_ROOT / "metadata/pild_sen12_training_v2"
DEFAULT_MANIFEST = DEFAULT_METADATA / "unified_sample_manifest_v2.csv"
DEFAULT_SUMMARY = DEFAULT_METADATA / "protocol_summary_v2.json"
DEFAULT_SPLIT = DEFAULT_METADATA / "event_isolated_split_v2.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "experiments/revision2026/pild_sen12_roleaware_v1"

VARIANTS = ("V", "VT", "VTM", "VTR", "VTMR")
COMMON_TERRAIN9_NAMES = (
    "elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "laplacian_curvature",
    "tpi_90m",
    "tpi_300m",
    "ruggedness_90m",
    "local_relief_300m",
)
# This contract is local and immutable.  In particular, it must never inherit
# NATIVE_TERRAIN_V2_SCALE_GROUPS, whose indices address a 17-channel cache.
COMMON_TERRAIN9_SCALE_GROUPS = TerrainScaleGroups(
    fine=(1, 2, 3, 4, 7),       # slope, aspect, curvature, ruggedness
    meso=(5, 6),                # 90 m and 300 m TPI
    macro=(0, 8),               # elevation and 300 m local relief
)
COMMON_TERRAIN9_SCALE_GROUPS.validate(len(COMMON_TERRAIN9_NAMES))
MATERIAL_CONTEXTS = (MATERIAL_ALIGNED, MATERIAL_SHUFFLE, MATERIAL_ZERO_Q)
TRIGGER_CONTEXTS = (
    TRIGGER_ALIGNED,
    TRIGGER_WRONG_TIME,
    TRIGGER_EVENT_SHUFFLE,
    TRIGGER_ZERO_Q,
)
PARENT_VARIANT = {"VT": "V", "VTM": "VT", "VTR": "VT", "VTMR": "VT"}

TERRAIN_ALIGNED = "aligned"
TERRAIN_ZERO = "terrain-zero"
TERRAIN_SHIFT32 = "terrain-shift32-zero-pad"
TERRAIN_ROLL64 = "terrain-roll64-circular"
TERRAIN_DONOR = "terrain-other-source-or-event-donor"
TERRAIN_CONTEXTS = (
    TERRAIN_ALIGNED,
    TERRAIN_ZERO,
    TERRAIN_SHIFT32,
    TERRAIN_ROLL64,
    TERRAIN_DONOR,
)


def resolve_terrain_contract(
    names: Sequence[str],
) -> tuple[tuple[str, ...], TerrainScaleGroups, str]:
    """Accept only the two audited PILD Terrain schemas."""

    resolved = tuple(str(name) for name in names)
    if resolved == tuple(COMMON_TERRAIN9_NAMES):
        return resolved, COMMON_TERRAIN9_SCALE_GROUPS, "pild_sen12_common_terrain9_v2"
    if resolved == tuple(NATIVE_TERRAIN_V2_NAMES):
        return resolved, NATIVE_TERRAIN_V2_SCALE_GROUPS, "pild_native_terrain17_v1"
    raise RuntimeError(f"unsupported unified Terrain schema: {resolved}")


def material_terrain_groups(names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Map Material modulation to physically named Terrain channel groups."""

    resolved = tuple(str(name) for name in names)
    slope = tuple(index for index, name in enumerate(resolved) if name == "slope_deg")
    curvature = tuple(
        index for index, name in enumerate(resolved) if "curvature" in name
    )
    relief = tuple(
        index
        for index, name in enumerate(resolved)
        if name.startswith(
            (
                "tpi_",
                "local_std_",
                "local_relief_",
                "valley_depth_",
                "ridge_height_",
                "ruggedness_",
            )
        )
    )
    groups = {"slope": slope, "curvature": curvature, "relief": relief}
    if any(not values for values in groups.values()):
        raise RuntimeError(f"Terrain schema cannot support Material interaction: {groups}")
    return groups


@dataclass(frozen=True)
class EvaluationContext:
    """One inference-only intervention evaluated from the selected checkpoint."""

    name: str
    terrain: str = TERRAIN_ALIGNED
    material: str = MATERIAL_ALIGNED
    trigger: str = TRIGGER_ALIGNED


def evaluation_contexts_for_variant(variant: str) -> tuple[EvaluationContext, ...]:
    """Return the frozen, deliberately small negative-control set per variant."""

    if variant == "V":
        return (EvaluationContext("aligned"),)
    if variant == "VT":
        return (
            EvaluationContext("aligned"),
            EvaluationContext("terrain-zero", terrain=TERRAIN_ZERO),
            EvaluationContext("terrain-shift32-zero-pad", terrain=TERRAIN_SHIFT32),
            EvaluationContext("terrain-roll64-circular", terrain=TERRAIN_ROLL64),
            EvaluationContext("terrain-other-source-or-event-donor", terrain=TERRAIN_DONOR),
        )
    if variant == "VTM":
        return (
            EvaluationContext("aligned"),
            EvaluationContext("material-aligned", material=MATERIAL_ALIGNED),
            EvaluationContext("material-shuffle", material=MATERIAL_SHUFFLE),
            EvaluationContext("material-zero-q", material=MATERIAL_ZERO_Q),
        )
    if variant == "VTR":
        return (
            EvaluationContext("aligned"),
            EvaluationContext("trigger-aligned", trigger=TRIGGER_ALIGNED),
            EvaluationContext("trigger-wrong-time", trigger=TRIGGER_WRONG_TIME),
            EvaluationContext("trigger-event-shuffle", trigger=TRIGGER_EVENT_SHUFFLE),
            EvaluationContext("trigger-zero-q", trigger=TRIGGER_ZERO_Q),
        )
    if variant == "VTMR":
        return (
            EvaluationContext("aligned"),
            EvaluationContext("material-shuffle", material=MATERIAL_SHUFFLE),
            EvaluationContext("material-zero-q", material=MATERIAL_ZERO_Q),
            EvaluationContext("trigger-wrong-time", trigger=TRIGGER_WRONG_TIME),
            EvaluationContext("trigger-event-shuffle", trigger=TRIGGER_EVENT_SHUFFLE),
            EvaluationContext("trigger-zero-q", trigger=TRIGGER_ZERO_Q),
            EvaluationContext(
                "material-trigger-both-zero-q",
                material=MATERIAL_ZERO_Q,
                trigger=TRIGGER_ZERO_Q,
            ),
        )
    raise ValueError(f"unknown variant {variant!r}")


def condition_for_evaluation(variant: str, context: EvaluationContext) -> str:
    if context.name == "aligned":
        return variant
    mapping = {
        "terrain-zero": "T_zero",
        "terrain-shift32-zero-pad": "T_shift",
        "terrain-roll64-circular": "T_roll",
        "terrain-other-source-or-event-donor": "T_donor",
        "material-aligned": "M_aligned",
        "material-shuffle": "M_shuffle",
        "material-zero-q": "M_zero_q",
        "trigger-aligned": "R_aligned",
        "trigger-wrong-time": "R_wrong_time",
        "trigger-event-shuffle": "R_event_shuffle",
        "trigger-zero-q": "R_zero_q",
    }
    return mapping.get(context.name, f"{variant}_{context.name}")


def reference_condition_for_variant(variant: str) -> str:
    if variant == "V":
        return "V"
    if variant == "VT":
        return "V"
    return "VT"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(json_safe(value), indent=2, allow_nan=False, sort_keys=True) + "\n",
    )


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames = sorted({str(key) for row in rows for key in row})
    if not fieldnames:
        raise RuntimeError(f"refusing to write empty evidence CSV: {path.name}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})
    os.replace(temporary, path)


def stable_index(token: str, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot choose from an empty deterministic pool")
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def deterministic_terrain_donors(
    sample_ids: Sequence[str],
    source_ids: Sequence[str],
    event_ids: Sequence[str],
    seed: int,
) -> np.ndarray:
    """Choose a non-self donor in the same split, preferring same-source events."""

    if not (len(sample_ids) == len(source_ids) == len(event_ids)):
        raise ValueError("Terrain donor identity arrays must have equal lengths")
    donors = np.full(len(sample_ids), -1, dtype=np.int64)
    for index, (source, event) in enumerate(zip(source_ids, event_ids, strict=True)):
        preferred = [
            candidate
            for candidate, (other_source, other_event) in enumerate(
                zip(source_ids, event_ids, strict=True)
            )
            if candidate != index and other_source == source and other_event != event
        ]
        fallback = [
            candidate
            for candidate, other_event in enumerate(event_ids)
            if candidate != index and other_event != event
        ]
        candidates = preferred or fallback
        if candidates:
            donors[index] = candidates[
                stable_index(f"{seed}|T-donor|{sample_ids[index]}", len(candidates))
            ]
    return donors


def zero_pad_spatial_shift(value: torch.Tensor, pixels: int) -> torch.Tensor:
    """Shift down/right without wraparound; vacated pixels are exactly zero."""

    if pixels < 0:
        raise ValueError("pixels must be nonnegative")
    if pixels == 0:
        return value
    shifted = torch.zeros_like(value)
    height, width = value.shape[-2:]
    if pixels < height and pixels < width:
        shifted[..., pixels:, pixels:] = value[..., : height - pixels, : width - pixels]
    return shifted


def tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def state_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    observed = set(state)
    if expected != observed:
        raise RuntimeError(
            "trainable checkpoint mismatch: "
            f"missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


@dataclass(frozen=True)
class OuterTrainTriggerNormalizer:
    """Event-balanced Trigger normalization fitted on supported train events."""

    mean: np.ndarray
    scale: np.ndarray
    event_ids: tuple[str, ...]
    z_clip: float = 5.0

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        q_r: np.ndarray,
        event_ids: Sequence[str],
        *,
        z_clip: float = 5.0,
    ) -> "OuterTrainTriggerNormalizer":
        array = np.asarray(values, dtype=np.float64)
        quality = np.asarray(q_r, dtype=np.float64).reshape(-1)
        if array.ndim != 2 or array.shape[1] != 3 or len(array) != len(quality):
            raise ValueError("Trigger fit requires values [N,3] and q_R [N]")
        if len(event_ids) != len(array):
            raise ValueError("Trigger event identity length mismatch")
        rows_by_event: dict[str, list[np.ndarray]] = defaultdict(list)
        for index, event_id in enumerate(event_ids):
            if quality[index] > 0 and np.isfinite(array[index]).all():
                rows_by_event[str(event_id)].append(array[index])
        if not rows_by_event:
            return cls(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), tuple(), z_clip)
        ordered = tuple(sorted(rows_by_event))
        event_values = np.stack(
            [np.median(np.stack(rows_by_event[event]), axis=0) for event in ordered]
        )
        mean = event_values.mean(axis=0)
        scale = event_values.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32), ordered, float(z_clip))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("Trigger transform requires [N,3]")
        clean = np.where(np.isfinite(array), array, self.mean)
        return np.clip((clean - self.mean) / self.scale, -self.z_clip, self.z_clip).astype(np.float32)

    def audit(self) -> dict[str, Any]:
        return {
            "fit_scope": "outer-train-supported-events-only",
            "weighting": "one-row-per-canonical-event",
            "n_events": len(self.event_ids),
            "event_sha256": hashlib.sha256("\n".join(self.event_ids).encode()).hexdigest(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "z_clip": self.z_clip,
        }


def aggregate_trigger_by_event(
    values: np.ndarray,
    q_r: np.ndarray,
    event_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Median-aggregate supported Trigger rows per canonical event and broadcast."""

    array = np.asarray(values, dtype=np.float32)
    quality = np.asarray(q_r, dtype=np.float32).reshape(-1)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Trigger aggregation requires [N,3]")
    if len(array) != len(quality) or len(event_ids) != len(array):
        raise ValueError("Trigger aggregation identity length mismatch")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, event_id in enumerate(event_ids):
        grouped[str(event_id)].append(index)
    output = np.zeros_like(array)
    effective_q = np.zeros_like(quality)
    for indices in grouped.values():
        supported = [
            index
            for index in indices
            if quality[index] > 0 and np.isfinite(array[index]).all()
        ]
        if not supported:
            continue
        median = np.median(array[supported], axis=0).astype(np.float32)
        output[indices] = median
        effective_q[indices] = 1.0
    return output, effective_q


def validate_protocol_schema(
    manifest_path: Path,
    summary_path: Path,
    split_path: Path,
    fold_id: str,
) -> dict[str, Any]:
    summary = load_protocol_summary(summary_path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False, nrows=32)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    terrain = summary.get("terrain_contract", {})
    names = tuple(terrain.get("names", ()))
    names, _, terrain_schema_id = resolve_terrain_contract(names)
    if tuple(ROLE_MATERIAL_FEATURE_NAMES) != tuple(MATERIAL_FEATURE_NAMES):
        raise RuntimeError("loader and role-aware Material 21-D feature order differs")
    split = pd.read_csv(split_path, keep_default_na=False)
    required_split = {"fold_id", "sample_id", "canonical_event_id", "role"}
    if required_split - set(split.columns):
        raise ValueError(f"split missing columns: {sorted(required_split-set(split.columns))}")
    selected = split[split["fold_id"].astype(str).eq(str(fold_id))].copy()
    if selected.empty:
        available = sorted(split["fold_id"].astype(str).unique())
        raise ValueError(f"unknown fold_id={fold_id!r}; available={available}")
    if selected["sample_id"].duplicated().any():
        raise RuntimeError("selected fold repeats sample_id")
    allowed_roles = {"train", "val", "test", "excluded"}
    unknown_roles = sorted(set(selected["role"].astype(str)) - allowed_roles)
    if unknown_roles:
        raise RuntimeError(f"selected fold contains unknown split roles: {unknown_roles}")
    # LODO uses explicit `excluded` rows for cross-source aliases of a held-out
    # physical event. They are intentionally absent from every loader and must
    # not be counted as an active split role when checking event isolation.
    active = selected[selected["role"].isin(("train", "val", "test"))].copy()
    event_roles = active.groupby("canonical_event_id")["role"].nunique()
    if int(event_roles.max()) != 1:
        raise RuntimeError("canonical event leakage across train/val/test")
    roles = active["role"].value_counts().to_dict()
    for role in ("train", "val", "test"):
        if roles.get(role, 0) == 0:
            raise RuntimeError(f"selected fold has no {role} samples")
    outputs = summary.get("outputs", {})
    expected_hash = outputs.get("manifest", {}).get("sha256")
    if expected_hash and sha256_file(manifest_path) != expected_hash:
        raise RuntimeError("manifest SHA-256 differs from protocol summary")
    return {
        "status": "PASS",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "summary": str(summary_path.resolve()),
        "split": str(split_path.resolve()),
        "split_sha256": sha256_file(split_path),
        "fold_id": str(fold_id),
        "split_counts": {str(key): int(value) for key, value in roles.items()},
        "excluded_samples": int(selected["role"].eq("excluded").sum()),
        "terrain_channels": list(names),
        "terrain_schema_id": terrain_schema_id,
        "role_material_features": list(ROLE_MATERIAL_FEATURE_NAMES),
        "trigger_feature_count": 3,
        "readiness": summary.get("readiness", {}),
    }


def _support_arrays(dataset: UnifiedPILDSen12Dataset) -> dict[str, Any]:
    material, q_m, trigger, q_r = [], [], [], []
    for index in range(len(dataset)):
        material_row = dataset.support_row(index, "material")
        trigger_row = dataset.support_row(index, "trigger")
        from pild_sen12_training_loader_v2 import (  # local to keep schema ownership explicit
            harmonize_role_material,
            harmonize_trigger,
        )

        m_value, m_quality = harmonize_role_material(material_row)
        r_value, r_quality = harmonize_trigger(trigger_row)
        material.append(m_value)
        q_m.append(m_quality)
        trigger.append(r_value)
        q_r.append(r_quality)
    frame = dataset.frame
    event_ids = tuple(frame["canonical_event_id"].astype(str))
    trigger_array, trigger_quality = aggregate_trigger_by_event(
        np.asarray(trigger, dtype=np.float32),
        np.asarray(q_r, dtype=np.float32),
        event_ids,
    )
    return {
        "material": np.asarray(material, dtype=np.float32),
        "q_m": np.asarray(q_m, dtype=np.float32),
        "trigger": trigger_array,
        "q_r": trigger_quality,
        "sample_ids": tuple(frame["sample_id"].astype(str)),
        "source_ids": tuple(frame["source_id"].astype(str)),
        "event_ids": event_ids,
    }


def fit_outer_normalizers(
    train_dataset: UnifiedPILDSen12Dataset,
) -> tuple[OuterTrainMaterialNormalizer, OuterTrainTriggerNormalizer, dict[str, Any]]:
    arrays = _support_arrays(train_dataset)
    fit_material = arrays["material"].copy()
    fit_material[arrays["q_m"] <= 0] = np.nan
    material = OuterTrainMaterialNormalizer.fit(
        fit_material,
        arrays["sample_ids"],
        arrays["source_ids"],
        arrays["event_ids"],
        arrays["sample_ids"],
    )
    trigger = OuterTrainTriggerNormalizer.fit(
        arrays["trigger"], arrays["q_r"], arrays["event_ids"]
    )
    return material, trigger, arrays


def estimate_terrain_stats(dataset: UnifiedPILDSen12Dataset) -> tuple[np.ndarray, np.ndarray]:
    if len(dataset) == 0:
        raise ValueError("cannot estimate Terrain statistics from an empty dataset")
    channels = int(dataset[0]["terrain"].shape[0])
    sums = np.zeros(channels, dtype=np.float64)
    squares = np.zeros(channels, dtype=np.float64)
    counts = np.zeros(channels, dtype=np.float64)
    for index in range(len(dataset)):
        item = dataset[index]
        value = item["terrain"].numpy().astype(np.float64)
        valid = item["terrain_valid"].numpy().astype(bool)
        if valid.ndim == 2:
            valid = valid[None]
        if valid.shape[0] == 1:
            valid = np.broadcast_to(valid, value.shape)
        elif valid.shape != value.shape:
            raise RuntimeError(f"unexpected terrain_valid shape {valid.shape}")
        safe = np.where(valid, value, 0.0)
        sums += safe.sum(axis=(1, 2))
        squares += np.square(safe).sum(axis=(1, 2))
        counts += valid.sum(axis=(1, 2))
    mean = sums / np.maximum(counts, 1.0)
    variance = squares / np.maximum(counts, 1.0) - np.square(mean)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)


class NormalizedRoleDataset(Dataset[dict[str, Any]]):
    """Apply outer-train normalization and deterministic negative controls."""

    def __init__(
        self,
        base: UnifiedPILDSen12Dataset,
        terrain_mean: np.ndarray,
        terrain_std: np.ndarray,
        material_normalizer: OuterTrainMaterialNormalizer,
        trigger_normalizer: OuterTrainTriggerNormalizer,
        *,
        seed: int,
    ) -> None:
        self.base = base
        self.frame = base.frame
        self.terrain_mean = np.asarray(terrain_mean, dtype=np.float32)[:, None, None]
        self.terrain_std = np.asarray(terrain_std, dtype=np.float32)[:, None, None]
        arrays = _support_arrays(base)
        self.sample_ids = arrays["sample_ids"]
        self.source_ids = arrays["source_ids"]
        self.event_ids = arrays["event_ids"]
        self.q_m = arrays["q_m"]
        self.q_r = arrays["q_r"]
        self.material = material_normalizer.transform(arrays["material"])
        self.trigger = trigger_normalizer.transform(arrays["trigger"])
        raw_wrong = build_wrong_time_features(torch.from_numpy(arrays["trigger"])).numpy()
        self.trigger_wrong_time = trigger_normalizer.transform(raw_wrong)
        self.terrain_donor = deterministic_terrain_donors(
            self.sample_ids, self.source_ids, self.event_ids, seed
        )
        self.material_donor = self._material_donors(seed)
        self.trigger_donor = self._trigger_donors(seed)
        self.trigger_shuffle_q = np.asarray(
            [
                self.q_r[index] if int(donor) != index else 0.0
                for index, donor in enumerate(self.trigger_donor)
            ],
            dtype=np.float32,
        )

    def _material_donors(self, seed: int) -> np.ndarray:
        donors = np.arange(len(self), dtype=np.int64)
        pools: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, (source, event) in enumerate(zip(self.source_ids, self.event_ids, strict=True)):
            if self.q_m[index] > 0:
                pools[(source, event)].append(index)
        events_by_source: dict[str, list[str]] = defaultdict(list)
        for source, event in pools:
            events_by_source[source].append(event)
        for index, (source, event) in enumerate(zip(self.source_ids, self.event_ids, strict=True)):
            if self.q_m[index] <= 0:
                continue
            events = sorted(set(events_by_source[source]) - {event})
            if not events:
                continue
            donor_event = events[stable_index(f"{seed}|M-event|{self.sample_ids[index]}", len(events))]
            candidates = pools[(source, donor_event)]
            donors[index] = candidates[
                stable_index(f"{seed}|M-sample|{self.sample_ids[index]}", len(candidates))
            ]
        return donors

    def _trigger_donors(self, seed: int) -> np.ndarray:
        donors = np.arange(len(self), dtype=np.int64)
        first_supported: dict[str, int] = {}
        event_source: dict[str, str] = {}
        for index, (source, event) in enumerate(zip(self.source_ids, self.event_ids, strict=True)):
            event_source.setdefault(event, source)
            if self.q_r[index] > 0:
                first_supported.setdefault(event, index)
        supported = sorted(first_supported)
        by_source: dict[str, list[str]] = defaultdict(list)
        for event in supported:
            by_source[event_source[event]].append(event)
        donor_event: dict[str, str] = {}
        for event in supported:
            same_source = sorted(set(by_source[event_source[event]]) - {event})
            candidates = same_source or sorted(set(supported) - {event})
            if candidates:
                donor_event[event] = candidates[
                    stable_index(f"{seed}|R-event|{event}", len(candidates))
                ]
        for index, event in enumerate(self.event_ids):
            if self.q_r[index] > 0 and event in donor_event:
                donors[index] = first_supported[donor_event[event]]
        return donors

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        terrain, valid = self._normalized_terrain(item)
        item["terrain"] = terrain
        item["q_t"] = valid
        terrain_donor = int(self.terrain_donor[index])
        if terrain_donor >= 0:
            donor_item = self.base[terrain_donor]
            donor_terrain, donor_valid = self._normalized_terrain(donor_item)
            donor_sample_id = self.sample_ids[terrain_donor]
            donor_event_id = self.event_ids[terrain_donor]
        else:
            donor_terrain = torch.zeros_like(terrain)
            donor_valid = torch.zeros_like(valid)
            donor_sample_id = ""
            donor_event_id = ""
        material_donor = int(self.material_donor[index])
        trigger_donor = int(self.trigger_donor[index])
        item.update(
            {
                "terrain_donor": donor_terrain,
                "terrain_donor_q": donor_valid,
                "terrain_donor_sample_id": donor_sample_id,
                "terrain_donor_event_id": donor_event_id,
                "role_material_features": torch.from_numpy(self.material[index]),
                "material_shuffle_features": torch.from_numpy(self.material[material_donor]),
                "material_shuffle_q": torch.tensor(
                    self.q_m[material_donor] if material_donor != index else 0.0,
                    dtype=torch.float32,
                ),
                "material_donor_sample_id": self.sample_ids[material_donor],
                "trigger_features": torch.from_numpy(self.trigger[index]),
                "trigger_wrong_time_features": torch.from_numpy(self.trigger_wrong_time[index]),
                "trigger_shuffle_features": torch.from_numpy(self.trigger[trigger_donor]),
                "trigger_shuffle_q": torch.tensor(
                    self.trigger_shuffle_q[index], dtype=torch.float32
                ),
                "trigger_donor_event_id": self.event_ids[trigger_donor],
            }
        )
        return item

    def _normalized_terrain(
        self, item: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        terrain = item["terrain"].numpy().astype(np.float32)
        valid = item["terrain_valid"].numpy().astype(np.float32)
        if valid.ndim == 2:
            valid = valid[None]
        if valid.shape[0] != 1:
            valid = np.all(valid > 0, axis=0, keepdims=True).astype(np.float32)
        normalized = ((terrain - self.terrain_mean) / self.terrain_std) * valid
        return (
            torch.from_numpy(normalized.astype(np.float32, copy=False)),
            torch.from_numpy(valid.astype(np.float32, copy=False)),
        )


class RoleAwareGeoPhysAdapter(nn.Module):
    """Composable role-pure model used by both production and CPU tests."""

    def __init__(
        self,
        visual: nn.Module,
        variant: str,
        *,
        visual_channels: int = 128,
        alpha_max: float = 2.0,
        terrain_names: Sequence[str] = COMMON_TERRAIN9_NAMES,
        terrain_scale_groups: TerrainScaleGroups = COMMON_TERRAIN9_SCALE_GROUPS,
        terrain_adapter: BoundedTerrainAdapterV2 | None = None,
        material_module: RoleAwareMaterialInteraction | None = None,
        trigger_module: PILDRoleAwareTrigger | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.visual = visual
        self.terrain_names = tuple(str(name) for name in terrain_names)
        terrain_scale_groups.validate(len(self.terrain_names))
        interaction_groups = material_terrain_groups(self.terrain_names)
        needs_terrain = variant != "V"
        self.terrain_adapter = terrain_adapter or (
            BoundedTerrainAdapterV2(
                len(self.terrain_names),
                visual_channels,
                terrain_scale_groups,
                alpha_max=alpha_max,
            )
            if needs_terrain
            else None
        )
        self.material_module = material_module or (
            RoleAwareMaterialInteraction(interaction_groups)
            if "M" in variant
            else None
        )
        self.trigger_module = trigger_module or (
            PILDRoleAwareTrigger(TriggerGateConfig(feature_dim=3))
            if "R" in variant
            else None
        )
        self._set_role_trainability()

    def _set_role_trainability(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.variant == "V":
            target = getattr(self.visual, "decoder", self.visual)
            for parameter in target.parameters():
                parameter.requires_grad = True
        elif self.variant == "VT":
            assert self.terrain_adapter is not None
            for parameter in self.terrain_adapter.parameters():
                parameter.requires_grad = True
        else:
            if self.material_module is not None:
                for parameter in self.material_module.parameters():
                    parameter.requires_grad = True
            if self.trigger_module is not None:
                for parameter in self.trigger_module.parameters():
                    parameter.requires_grad = True

    @staticmethod
    def visual_uncertainty(logits: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits.detach())
        return 1.0 - torch.abs(2.0 * probability - 1.0)

    @staticmethod
    def _material_inputs(
        batch: Mapping[str, Any], context: str
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        if context == MATERIAL_ALIGNED:
            return batch["role_material_features"], batch["q_material"], MATERIAL_ALIGNED
        if context == MATERIAL_SHUFFLE:
            return batch["material_shuffle_features"], batch["material_shuffle_q"], MATERIAL_SHUFFLE
        if context == MATERIAL_ZERO_Q:
            return batch["role_material_features"], torch.zeros_like(batch["q_material"]), MATERIAL_ZERO_Q
        raise ValueError(f"unknown Material context {context!r}")

    @staticmethod
    def _terrain_inputs(
        batch: Mapping[str, Any], context: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context == TERRAIN_ALIGNED:
            return batch["terrain"], batch["q_t"]
        if context == TERRAIN_ZERO:
            return torch.zeros_like(batch["terrain"]), torch.zeros_like(batch["q_t"])
        if context == TERRAIN_SHIFT32:
            return (
                zero_pad_spatial_shift(batch["terrain"], 32),
                zero_pad_spatial_shift(batch["q_t"], 32),
            )
        if context == TERRAIN_ROLL64:
            return (
                torch.roll(batch["terrain"], shifts=(64, 64), dims=(-2, -1)),
                torch.roll(batch["q_t"], shifts=(64, 64), dims=(-2, -1)),
            )
        if context == TERRAIN_DONOR:
            return batch["terrain_donor"], batch["terrain_donor_q"]
        raise ValueError(f"unknown Terrain context {context!r}")

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        terrain_context: str = TERRAIN_ALIGNED,
        material_context: str = MATERIAL_ALIGNED,
        trigger_context: str = TRIGGER_ALIGNED,
    ) -> dict[str, Any]:
        optical = batch["optical"] * 10_000.0  # unified loader stores reflectance in [0,1]
        visual_context = torch.enable_grad() if self.variant == "V" else torch.no_grad()
        with visual_context:
            visual = self.visual(
                optical,
                batch["temporal_coords"],
                batch["location_coords"],
            )
        visual_logits = visual["logits"]
        output: dict[str, Any] = {
            "logits": visual_logits,
            "visual_logits": visual_logits.detach(),
            "reference_logits": visual_logits.detach(),
            "visual_feature": visual["visual_feature"].detach(),
            "terrain_correction": torch.zeros_like(visual_logits),
            "material_correction": torch.zeros_like(visual_logits),
            "trigger_correction": torch.zeros_like(visual_logits),
        }
        if self.variant == "V":
            return output

        assert self.terrain_adapter is not None
        terrain, q_t = self._terrain_inputs(batch, terrain_context)
        terrain_grad_context = torch.enable_grad() if self.variant == "VT" else torch.no_grad()
        with terrain_grad_context:
            _, terrain_audit = self.terrain_adapter(
                visual_logits.detach(),
                visual["visual_feature"].detach(),
                self.visual_uncertainty(visual_logits),
                terrain,
                q_t,
            )
        terrain_correction = terrain_audit["correction"]
        terrain_reference_logits = visual_logits.detach() + terrain_correction
        conditioned_terrain = terrain_correction
        material_audit = None
        if self.material_module is not None:
            material, q_m, resolved_context = self._material_inputs(batch, material_context)
            conditioned_terrain, material_audit = self.material_module(
                terrain,
                terrain_correction,
                material,
                q_m,
                context=resolved_context,
            )

        trigger_audit = None
        trigger_delta = torch.zeros_like(visual_logits)
        if self.trigger_module is not None:
            kwargs: dict[str, Any] = {}
            trigger_q = batch["q_trigger"]
            if trigger_context == TRIGGER_WRONG_TIME:
                kwargs["wrong_time_features"] = batch["trigger_wrong_time_features"]
            elif trigger_context == TRIGGER_EVENT_SHUFFLE:
                kwargs["event_shuffled_features"] = batch["trigger_shuffle_features"]
                trigger_q = batch["trigger_shuffle_q"]
            trigger_audit = self.trigger_module(
                visual_logits,
                batch["trigger_features"],
                trigger_q,
                batch["canonical_event_id"],
                context=trigger_context,
                **kwargs,
            )
            trigger_delta = trigger_audit["logit_delta"]

        logits = visual_logits.detach() + conditioned_terrain + trigger_delta
        output.update(
            {
                "logits": logits,
                "reference_logits": (
                    visual_logits.detach()
                    if self.variant == "VT"
                    else terrain_reference_logits.detach()
                ),
                "terrain_correction": terrain_correction,
                "material_correction": conditioned_terrain - terrain_correction.detach(),
                "trigger_correction": trigger_delta,
                "terrain_audit": terrain_audit,
                "material_audit": material_audit,
                "trigger_audit": trigger_audit,
            }
        )
        return output


def component_state(model: RoleAwareGeoPhysAdapter) -> dict[str, dict[str, torch.Tensor]]:
    state = {"visual_decoder": state_to_cpu(getattr(model.visual, "decoder", model.visual))}
    # A fine-tuned encoder can no longer be rebuilt from the pristine Prithvi snapshot,
    # so the whole visual module is stored alongside the decoder. Consumers that only
    # know about `visual_decoder` keep working because the extra key is additive.
    if not getattr(model.visual, "freeze_encoder", True):
        state["visual_full"] = state_to_cpu(model.visual)
    if model.terrain_adapter is not None:
        state["terrain_adapter"] = state_to_cpu(model.terrain_adapter)
    if model.material_module is not None:
        state["material_module"] = state_to_cpu(model.material_module)
    if model.trigger_module is not None:
        state["trigger_module"] = state_to_cpu(model.trigger_module)
    return state


def load_parent_components(
    model: RoleAwareGeoPhysAdapter,
    checkpoint: Mapping[str, Any],
    *,
    expected_variant: str,
    identity: Mapping[str, Any],
) -> None:
    if checkpoint.get("variant") != expected_variant:
        raise RuntimeError(
            f"parent variant must be {expected_variant}; got {checkpoint.get('variant')}"
        )
    parent_identity = checkpoint.get("identity", {})
    keys = ("manifest_sha256", "split_sha256", "fold_id", "seed", "prithvi_checkpoint_sha256")
    mismatch = {
        key: (identity.get(key), parent_identity.get(key))
        for key in keys
        if identity.get(key) != parent_identity.get(key)
    }
    if mismatch:
        raise RuntimeError(f"parent identity mismatch: {mismatch}")
    components = checkpoint.get("components", {})
    getattr(model.visual, "decoder", model.visual).load_state_dict(
        components["visual_decoder"], strict=True
    )
    if expected_variant == "VT":
        assert model.terrain_adapter is not None
        model.terrain_adapter.load_state_dict(components["terrain_adapter"], strict=True)


class BinaryHistogram:
    def __init__(self, bins: int = 2048) -> None:
        self.bins = int(bins)
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def update(self, probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        keep = valid.detach().bool().cpu().numpy().reshape(-1)
        probability_np = probability.detach().float().cpu().numpy().reshape(-1)[keep]
        target_np = target.detach().cpu().numpy().reshape(-1)[keep] >= 0.5
        indices = np.minimum((probability_np * self.bins).astype(np.int64), self.bins - 1)
        self.positive += np.bincount(indices[target_np], minlength=self.bins)
        self.negative += np.bincount(indices[~target_np], minlength=self.bins)

    def average_precision(self) -> float:
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        total = int(self.positive.sum())
        if total == 0:
            return 0.0
        precision = tp / np.maximum(tp + fp, 1)
        recall_increment = self.positive[::-1] / total
        return float(np.sum(precision * recall_increment))

    def counts(self, threshold: float) -> dict[str, int]:
        index = min(max(int(math.floor(threshold * self.bins)), 0), self.bins - 1)
        tp = int(self.positive[index:].sum())
        fp = int(self.negative[index:].sum())
        fn = int(self.positive[:index].sum())
        tn = int(self.negative[:index].sum())
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "errors": float(fp + fn),
    }


def choose_threshold(histogram: BinaryHistogram) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.05, 0.95, 181)
    rows = [(float(value), metrics_from_counts(histogram.counts(float(value)))) for value in candidates]
    return max(rows, key=lambda item: (item[1]["iou"], -item[1]["errors"]))


def choose_fixed_fpr_threshold(
    histogram: BinaryHistogram, target_fpr: float = 0.05
) -> float:
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie in (0,1)")
    candidates = np.linspace(0.01, 0.99, 197)
    feasible: list[tuple[float, float]] = []
    for threshold in candidates:
        counts = histogram.counts(float(threshold))
        fpr = counts["fp"] / max(counts["fp"] + counts["tn"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        if fpr <= target_fpr:
            feasible.append((recall, float(threshold)))
    if feasible:
        return max(feasible, key=lambda item: (item[0], -item[1]))[1]
    return 0.99


def sample_average_precision(
    probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> float:
    histogram = BinaryHistogram()
    histogram.update(probability[None], target[None], valid[None])
    return histogram.average_precision()


def ensure_map(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value[:, None]
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"expected [B,1,H,W] or [B,H,W], got {tuple(value.shape)}")
    return value


def move_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def segmentation_loss(
    output: Mapping[str, Any],
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: torch.Tensor,
    variant: str,
    decision_threshold: float,
) -> torch.Tensor:
    logits = output["logits"]
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits)
    intersection = (probability * target * valid).flatten(1).sum(1)
    denominator = ((probability + target) * valid).flatten(1).sum(1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    loss = bce + 0.5 * dice
    if variant != "V":
        reference_prediction = (
            torch.sigmoid(output["reference_logits"]) >= decision_threshold
        )
        reference_correct = (reference_prediction == (target >= 0.5)).to(valid.dtype) * valid
        total_correction = logits - output["reference_logits"]
        preserve = (total_correction.square() * reference_correct).sum() / reference_correct.sum().clamp_min(1.0)
        loss = loss + 0.1 * preserve + 1e-3 * total_correction.abs().mean()
    return loss


@torch.no_grad()
def validation_histogram(
    model: RoleAwareGeoPhysAdapter,
    loader: DataLoader,
    device: str,
) -> BinaryHistogram:
    """Validate only the aligned intervention used for checkpoint selection."""

    model.eval()
    histogram = BinaryHistogram()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            batch,
            terrain_context=TERRAIN_ALIGNED,
            material_context=MATERIAL_ALIGNED,
            trigger_context=TRIGGER_ALIGNED,
        )
        target = ensure_map(batch["mask"])
        valid = ensure_map(batch["valid_mask"])
        histogram.update(torch.sigmoid(output["logits"]), target, valid)
    return histogram


def train_model(
    model: RoleAwareGeoPhysAdapter,
    train_loader: DataLoader,
    train_sampler: (
        DatasetEventPatchBalancedSampler
        | SourceEventPatchBalancedSampler
        | TemperedDatasetEventPatchSampler
        | NaturalPatchSampler
    ),
    val_loader: DataLoader,
    args: argparse.Namespace,
    pos_weight: float,
    fixed_threshold: float | None,
    log: Any,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise RuntimeError("selected variant has no trainable parameters")
    # Pretrained encoder weights move on a much smaller step than a freshly initialised
    # decoder, otherwise a few epochs on 55 events destroy the pretrained representation.
    encoder_prefix = "visual.encoder."
    encoder_group = [
        value
        for name, value in model.named_parameters()
        if value.requires_grad and name.startswith(encoder_prefix)
    ]
    if encoder_group:
        encoder_ids = {id(value) for value in encoder_group}
        head_group = [value for value in parameters if id(value) not in encoder_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": head_group, "lr": args.lr},
                {
                    "params": encoder_group,
                    "lr": args.lr * float(args.encoder_lr_scale),
                },
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    positive = torch.tensor([pos_weight], device=args.device).reshape(1, 1, 1, 1)
    best_key: tuple[float, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_threshold = fixed_threshold
    history: list[dict[str, Any]] = []
    step = 0
    if fixed_threshold is not None:
        identity_histogram = validation_histogram(model, val_loader, args.device)
        identity_ap = identity_histogram.average_precision()
        identity_metrics = metrics_from_counts(identity_histogram.counts(fixed_threshold))
        best_key = (-identity_metrics["iou"], identity_metrics["errors"], -identity_ap)
        best_state = trainable_state(model)
        best_threshold = fixed_threshold
        history.append(
            {
                "epoch": 0,
                "steps": 0,
                "train_loss": None,
                "val_ap": identity_ap,
                "val_threshold": fixed_threshold,
                "selection_role": "exact_parent_identity",
                **{f"val_{key}": value for key, value in identity_metrics.items()},
            }
        )
        log(
            f"[epoch] 0/{args.epochs} exact_parent_identity "
            f"val_iou={identity_metrics['iou']:.6f} val_ap={identity_ap:.6f} "
            f"threshold={fixed_threshold:.3f}"
        )
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = move_batch(batch, args.device)
            target = ensure_map(batch["mask"])
            valid = ensure_map(batch["valid_mask"])
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch,
                terrain_context=TERRAIN_ALIGNED,
                material_context=MATERIAL_ALIGNED,
                trigger_context=TRIGGER_ALIGNED,
            )
            loss = segmentation_loss(
                output,
                target,
                valid,
                positive,
                args.variant,
                fixed_threshold if fixed_threshold is not None else 0.5,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        histogram = validation_histogram(model, val_loader, args.device)
        ap = histogram.average_precision()
        if fixed_threshold is None:
            threshold, metrics = choose_threshold(histogram)
            key = (-ap, -metrics["iou"], metrics["errors"])
        else:
            threshold = fixed_threshold
            metrics = metrics_from_counts(histogram.counts(threshold))
            key = (-metrics["iou"], metrics["errors"], -ap)
        row = {
            "epoch": epoch,
            "steps": step,
            "train_loss": total_loss / max(seen, 1),
            "val_ap": ap,
            "val_threshold": threshold,
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        log(
            f"[epoch] {epoch}/{args.epochs} loss={row['train_loss']:.6f} "
            f"val_iou={metrics['iou']:.6f} val_ap={ap:.6f} threshold={threshold:.3f}"
        )
        if best_key is None or key < best_key:
            best_key = key
            best_state = trainable_state(model)
            best_epoch = epoch
            best_threshold = threshold
        if args.max_steps and step >= args.max_steps:
            break
    if best_state is None or best_threshold is None:
        raise RuntimeError("training produced no selected checkpoint")
    return best_state, history, best_epoch, float(best_threshold)


def estimate_pos_weight(dataset: Dataset[dict[str, Any]]) -> float:
    positive, negative = 0.0, 0.0
    for index in range(len(dataset)):
        item = dataset[index]
        target = item["mask"].float()
        valid = item["valid_mask"].float()
        positive += float((target * valid).sum())
        negative += float(((1.0 - target) * valid).sum())
    return float(np.clip(negative / max(positive, 1.0), 1.0, 50.0))


@torch.no_grad()
def evaluate(
    model: RoleAwareGeoPhysAdapter,
    loader: DataLoader,
    device: str,
    threshold: float,
    fixed_fpr_threshold: float,
    split: str,
    evaluation_context: EvaluationContext,
    seed: int,
    checkpoint_sha256: str,
    component_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    sample_rows: list[dict[str, Any]] = []
    histogram = BinaryHistogram()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            batch,
            terrain_context=evaluation_context.terrain,
            material_context=evaluation_context.material,
            trigger_context=evaluation_context.trigger,
        )
        target = ensure_map(batch["mask"])
        valid = ensure_map(batch["valid_mask"]).bool()
        probability = torch.sigmoid(output["logits"])
        visual_probability = torch.sigmoid(output["visual_logits"])
        reference_probability = torch.sigmoid(output["reference_logits"])
        histogram.update(probability, target, valid)
        prediction = probability >= threshold
        visual_prediction = visual_probability >= threshold
        reference_prediction = reference_probability >= threshold
        fixed_prediction = probability >= fixed_fpr_threshold
        truth = target >= 0.5
        _, effective_q_t_map = model._terrain_inputs(batch, evaluation_context.terrain)
        effective_q_t = effective_q_t_map.float().flatten(1).mean(1)
        if evaluation_context.material == MATERIAL_SHUFFLE:
            effective_q_m = batch["material_shuffle_q"]
        elif evaluation_context.material == MATERIAL_ZERO_Q:
            effective_q_m = torch.zeros_like(batch["q_material"])
        else:
            effective_q_m = batch["q_material"]
        if evaluation_context.trigger == TRIGGER_EVENT_SHUFFLE:
            effective_q_r = batch["trigger_shuffle_q"]
        elif evaluation_context.trigger == TRIGGER_ZERO_Q:
            effective_q_r = torch.zeros_like(batch["q_trigger"])
        else:
            effective_q_r = batch["q_trigger"]
        condition = condition_for_evaluation(model.variant, evaluation_context)
        reference_condition = reference_condition_for_variant(model.variant)
        for index in range(target.shape[0]):
            keep = valid[index]
            pred = prediction[index] & keep
            vis = visual_prediction[index] & keep
            reference = reference_prediction[index] & keep
            fixed = fixed_prediction[index] & keep
            actual = truth[index] & keep
            tp = int((pred & actual).sum())
            fp = int((pred & ~actual & keep).sum())
            fn = int((~pred & actual & keep).sum())
            tn = int((~pred & ~actual & keep).sum())
            visual_wrong = torch.logical_xor(vis, actual) & keep
            reference_wrong = torch.logical_xor(reference, actual) & keep
            final_wrong = torch.logical_xor(pred, actual) & keep
            corrected = int((reference_wrong & ~final_wrong).sum())
            harmed = int((~reference_wrong & final_wrong & keep).sum())
            visual_errors = int(visual_wrong.sum())
            reference_errors = int(reference_wrong.sum())
            positive = actual
            negative = ~actual & keep
            fixed_tp = int((fixed & positive).sum())
            fixed_fn = int((~fixed & positive & keep).sum())
            fixed_fp = int((fixed & negative).sum())
            fixed_tn = int((~fixed & negative & keep).sum())
            valid_pixels = int(keep.sum())
            target_positive = int(positive.sum())
            probability_i = probability[index]
            target_i = target[index].float()
            clipped = probability_i.clamp(1e-7, 1.0 - 1e-7)
            brier_sum = float(((probability_i - target_i).square() * keep).sum().cpu())
            nll_sum = float(
                (-target_i * torch.log(clipped) - (1.0 - target_i) * torch.log1p(-clipped))[
                    keep
                ].sum().cpu()
            )
            soft_area_error = float((probability_i[keep].sum() - target_i[keep].sum()).cpu())
            q_t_value = float(effective_q_t[index].detach().cpu())
            q_m_value = float(effective_q_m[index].detach().cpu())
            q_r_value = float(effective_q_r[index].detach().cpu())
            if evaluation_context.name.startswith("terrain-"):
                effective_q = q_t_value
                applicable = float(batch["q_t"][index].float().mean().cpu()) > 0
                if evaluation_context.terrain == TERRAIN_DONOR:
                    applicable = applicable and q_t_value > 0
            elif evaluation_context.name.startswith("material-"):
                effective_q = q_m_value
                applicable = float(batch["q_material"][index].detach().cpu()) > 0
                if evaluation_context.material == MATERIAL_SHUFFLE:
                    applicable = applicable and q_m_value > 0
            elif evaluation_context.name.startswith("trigger-"):
                effective_q = q_r_value
                applicable = float(batch["q_trigger"][index].detach().cpu()) > 0
                if evaluation_context.trigger == TRIGGER_EVENT_SHUFFLE:
                    applicable = applicable and q_r_value > 0
            elif evaluation_context.name == "material-trigger-both-zero-q":
                effective_q = 0.0
                applicable = (
                    float(batch["q_material"][index].detach().cpu()) > 0
                    or float(batch["q_trigger"][index].detach().cpu()) > 0
                )
            else:
                role_values = [q_t_value]
                if "M" in model.variant:
                    role_values.append(q_m_value)
                if "R" in model.variant:
                    role_values.append(q_r_value)
                effective_q = float(min(role_values)) if role_values else 1.0
                applicable = True
            sample_rows.append(
                {
                    "split": split,
                    "sample_id": batch["sample_id"][index],
                    "dataset_id": batch["dataset_id"][index],
                    "source_id": batch["source_id"][index],
                    "source": batch["source_id"][index],
                    "canonical_event_id": batch["canonical_event_id"][index],
                    "variant": model.variant,
                    "condition": condition,
                    "seed": int(seed),
                    "reference_condition": reference_condition,
                    "evaluation_context": evaluation_context.name,
                    "terrain_context": evaluation_context.terrain,
                    "material_context": evaluation_context.material,
                    "trigger_context": evaluation_context.trigger,
                    "checkpoint_sha256": checkpoint_sha256,
                    "component_sha256_json": json.dumps(
                        dict(sorted(component_hashes.items())), sort_keys=True
                    ),
                    "q_material": float(batch["q_material"][index].detach().cpu()),
                    "q_trigger": float(batch["q_trigger"][index].detach().cpu()),
                    "effective_q": effective_q,
                    "effective_q_terrain": q_t_value,
                    "effective_q_material": q_m_value,
                    "effective_q_trigger": q_r_value,
                    "control_applicable": int(applicable),
                    "terrain_donor_sample_id": batch.get(
                        "terrain_donor_sample_id", [""] * target.shape[0]
                    )[index],
                    "terrain_donor_event_id": batch.get(
                        "terrain_donor_event_id", [""] * target.shape[0]
                    )[index],
                    "material_donor_sample_id": batch.get(
                        "material_donor_sample_id", [""] * target.shape[0]
                    )[index],
                    "trigger_donor_event_id": batch.get(
                        "trigger_donor_event_id", [""] * target.shape[0]
                    )[index],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "iou": tp / max(tp + fp + fn, 1),
                    "ap": sample_average_precision(
                        probability_i, target_i, keep
                    ),
                    "visual_errors": visual_errors,
                    "reference_errors": reference_errors,
                    "final_errors": fp + fn,
                    "corrected": corrected,
                    "harmed": harmed,
                    "net_error_reduction": corrected - harmed,
                    "rer": (corrected - harmed) / max(reference_errors, 1),
                    "brier_sum": brier_sum,
                    "nll_sum": nll_sum,
                    "soft_area_error": soft_area_error,
                    "fixed_fpr_tp": fixed_tp,
                    "fixed_fpr_fn": fixed_fn,
                    "fixed_fpr_fp": fixed_fp,
                    "fixed_fpr_tn": fixed_tn,
                    "valid_pixel_count": valid_pixels,
                    "target_positive_count": target_positive,
                    "terrain_delta_abs_mean": float(output["terrain_correction"][index].abs().mean().cpu()),
                    "material_delta_abs_mean": float(output["material_correction"][index].abs().mean().cpu()),
                    "trigger_delta_abs_mean": float(output["trigger_correction"][index].abs().mean().cpu()),
                }
            )
    event_rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(sample_rows)
    for (event, dataset_id), group in frame.groupby(["canonical_event_id", "dataset_id"], sort=True):
        sums = group[["tp", "fp", "fn", "tn", "visual_errors", "reference_errors", "final_errors", "corrected", "harmed"]].sum()
        donor_receipts = {
            column: ";".join(sorted({str(value) for value in group[column] if str(value)}))
            for column in (
                "terrain_donor_sample_id",
                "terrain_donor_event_id",
                "material_donor_sample_id",
                "trigger_donor_event_id",
            )
        }
        event_rows.append(
            {
                "split": split,
                "canonical_event_id": event,
                "dataset_id": dataset_id,
                "variant": model.variant,
                "condition": condition,
                "seed": int(seed),
                "reference_condition": reference_condition,
                "evaluation_context": evaluation_context.name,
                "terrain_context": evaluation_context.terrain,
                "material_context": evaluation_context.material,
                "trigger_context": evaluation_context.trigger,
                "checkpoint_sha256": checkpoint_sha256,
                "component_sha256_json": json.dumps(
                    dict(sorted(component_hashes.items())), sort_keys=True
                ),
                "n_samples": len(group),
                "effective_q": float(group["effective_q"].mean()),
                "effective_q_terrain": float(group["effective_q_terrain"].mean()),
                "effective_q_material": float(group["effective_q_material"].mean()),
                "effective_q_trigger": float(group["effective_q_trigger"].mean()),
                "n_control_applicable": int(group["control_applicable"].sum()),
                **donor_receipts,
                **{key: int(sums[key]) for key in sums.index},
                "iou": int(sums.tp) / max(int(sums.tp + sums.fp + sums.fn), 1),
                "rer": (int(sums.corrected) - int(sums.harmed)) / max(int(sums.reference_errors), 1),
            }
        )
    counts = histogram.counts(threshold)
    total_visual_errors = int(frame["visual_errors"].sum())
    total_reference_errors = int(frame["reference_errors"].sum())
    corrected = int(frame["corrected"].sum())
    harmed = int(frame["harmed"].sum())
    corpus = {
        "split": split,
        "variant": model.variant,
        "condition": condition,
        "seed": int(seed),
        "reference_condition": reference_condition,
        "evaluation_context": evaluation_context.name,
        "terrain_context": evaluation_context.terrain,
        "material_context": evaluation_context.material,
        "trigger_context": evaluation_context.trigger,
        "checkpoint_sha256": checkpoint_sha256,
        "component_sha256": dict(sorted(component_hashes.items())),
        "threshold": threshold,
        "fixed_fpr_threshold": fixed_fpr_threshold,
        "average_precision": histogram.average_precision(),
        **metrics_from_counts(counts),
        "visual_errors": total_visual_errors,
        "reference_errors": total_reference_errors,
        "corrected": corrected,
        "harmed": harmed,
        "net_error_reduction": corrected - harmed,
        "rer": (corrected - harmed) / max(total_reference_errors, 1),
        "n_samples": len(frame),
        "n_events": int(frame["canonical_event_id"].nunique()),
    }
    return sample_rows, event_rows, corpus


def unfreeze_encoder_tail(visual: PrithviEO2ChangeModel, blocks: int) -> dict[str, Any]:
    """Open the last transformer blocks for the V variant only.

    The Terrain-v2 contract pre-registers this as an optional second stage: the encoder
    stays frozen by default so every existing checkpoint reproduces bit for bit, and when
    it is opened the same setting must be applied to the visual and the physics-augmented
    arms alike. Adapter stages never call this, so a physics gain can never be confused
    with extra visual fine-tuning.
    """
    if blocks <= 0:
        return {"unfrozen_blocks": 0, "unfrozen_parameters": 0}
    encoder = visual.encoder
    container = None
    for name in ("blocks", "layers", "transformer"):
        candidate = getattr(encoder, name, None)
        if candidate is not None and len(list(candidate)) > 0:
            container = candidate
            break
    if container is None:
        raise RuntimeError("could not locate transformer blocks on the Prithvi encoder")
    depth = len(list(container))
    opened = min(int(blocks), depth)
    visual.freeze_encoder = False
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    count = 0
    for block in list(container)[depth - opened :]:
        for parameter in block.parameters():
            parameter.requires_grad = True
            count += int(parameter.numel())
    norm = getattr(encoder, "norm", None)
    if norm is not None:
        for parameter in norm.parameters():
            parameter.requires_grad = True
            count += int(parameter.numel())
    return {
        "encoder_depth": int(depth),
        "unfrozen_blocks": int(opened),
        "unfrozen_parameters": int(count),
    }


def build_model(args: argparse.Namespace, identity: dict[str, Any]) -> tuple[RoleAwareGeoPhysAdapter, dict[str, Any], dict[str, Any] | None]:
    encoder, provenance = load_prithvi_encoder(args.prithvi_snapshot)
    visual = PrithviEO2ChangeModel(
        encoder, decoder_width=args.decoder_width, freeze_encoder=True
    )
    terrain_names, terrain_scale_groups, _ = resolve_terrain_contract(
        args.terrain_names
    )
    model = RoleAwareGeoPhysAdapter(
        visual,
        args.variant,
        visual_channels=args.decoder_width,
        alpha_max=args.alpha_max,
        terrain_names=terrain_names,
        terrain_scale_groups=terrain_scale_groups,
    )
    if args.variant == "VT":
        assert model.terrain_adapter is not None
        nn.init.zeros_(model.terrain_adapter.terrain.output.weight)
    unfreeze_blocks = int(getattr(args, "unfreeze_encoder_blocks", 0) or 0)
    if unfreeze_blocks > 0:
        if args.variant != "V":
            raise ValueError("encoder unfreezing is only allowed for the V variant")
        identity["encoder_unfreeze"] = unfreeze_encoder_tail(visual, unfreeze_blocks)
    else:
        identity["encoder_unfreeze"] = {"unfrozen_blocks": 0, "unfrozen_parameters": 0}
    identity["prithvi_checkpoint_sha256"] = provenance["checkpoint_sha256"]
    parent = None
    if args.variant != "V":
        if args.parent_checkpoint is None:
            raise ValueError(f"{args.variant} requires --parent-checkpoint")
        parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
        load_parent_components(
            model,
            parent,
            expected_variant=PARENT_VARIANT[args.variant],
            identity=identity,
        )
    return model, provenance, parent


def make_loaders(
    args: argparse.Namespace,
) -> tuple[
    dict[str, NormalizedRoleDataset],
    dict[str, DataLoader],
    DatasetEventPatchBalancedSampler
    | SourceEventPatchBalancedSampler
    | TemperedDatasetEventPatchSampler
    | NaturalPatchSampler,
    dict[str, Any],
]:
    base = {
        role: UnifiedPILDSen12Dataset(
            args.manifest,
            args.protocol_summary,
            split_path=args.split,
            fold_id=args.fold_id,
            role=role,
            readiness="core",
        )
        for role in ("train", "val", "test")
    }
    terrain_mean, terrain_std = estimate_terrain_stats(base["train"])
    material_normalizer, trigger_normalizer, _ = fit_outer_normalizers(base["train"])
    datasets = {
        role: NormalizedRoleDataset(
            value,
            terrain_mean,
            terrain_std,
            material_normalizer,
            trigger_normalizer,
            seed=args.seed,
        )
        for role, value in base.items()
    }
    epoch_samples = args.epoch_samples or len(base["train"])
    if args.sampling_mode == "tempered":
        sampler = TemperedDatasetEventPatchSampler(
            base["train"].frame,
            num_samples=epoch_samples,
            seed=args.seed,
            dataset_temperature=args.dataset_temperature,
            event_temperature=args.event_temperature,
        )
    else:
        sampler_class = {
            "natural": NaturalPatchSampler,
            "balanced": SourceEventPatchBalancedSampler,
            "dataset_balanced": DatasetEventPatchBalancedSampler,
        }[args.sampling_mode]
        sampler = sampler_class(
            base["train"].frame,
            num_samples=epoch_samples,
            seed=args.seed,
        )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
    }
    normalization = {
        "fit_scope": "outer-train-only",
        "terrain_mean": terrain_mean.tolist(),
        "terrain_std": terrain_std.tolist(),
        "material": material_normalizer.audit(),
        "trigger": trigger_normalizer.audit(),
    }
    return datasets, loaders, sampler, normalization


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", default="event_isolated")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--prithvi-snapshot", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epoch-samples", type=int, default=0)
    parser.add_argument(
        "--sampling-mode",
        choices=("balanced", "dataset_balanced", "tempered", "natural"),
        default="balanced",
        help=(
            "natural preserves patch proportions; balanced uses source/event; "
            "dataset_balanced uses dataset/event hierarchy"
        ),
    )
    parser.add_argument("--dataset-temperature", type=float, default=0.75)
    parser.add_argument("--event-temperature", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument(
        "--unfreeze-encoder-blocks",
        type=int,
        default=0,
        help=(
            "open the last N transformer blocks for the V variant; 0 keeps the frozen "
            "encoder used by every existing checkpoint"
        ),
    )
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.1,
        help="learning-rate multiplier for unfrozen encoder parameters",
    )
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive and num-workers nonnegative")
    if args.epoch_samples < 0 or args.max_steps < 0:
        parser.error("epoch-samples and max-steps must be nonnegative")
    if not 0 < args.dataset_temperature <= 1 or not 0 < args.event_temperature <= 1:
        parser.error("sampling temperatures must be in (0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    schema = validate_protocol_schema(
        args.manifest, args.protocol_summary, args.split, args.fold_id
    )
    args.terrain_names = tuple(schema["terrain_channels"])
    if args.validate_only:
        schema["variant"] = args.variant
        schema["parent_checkpoint_required"] = args.variant != "V"
        schema["cache_opened"] = False
        print(json.dumps(json_safe(schema), indent=2, allow_nan=False))
        return 0

    if args.variant != "V" and args.parent_checkpoint is None:
        raise ValueError(f"{args.variant} requires --parent-checkpoint")
    outdir = args.outdir or (
        DEFAULT_OUT_ROOT
        / str(args.fold_id)
        / f"seed{args.seed}"
        / args.variant
    )
    outdir = outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    stage = outdir.parent / f".{outdir.name}.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    command = shlex.join([sys.executable, *(sys.argv if argv is None else [__file__, *argv])])
    log_path = stage / "run.log"
    log_path.write_text(command + "\n", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    started = time.time()
    set_seed(args.seed)
    identity = {
        "manifest_sha256": schema["manifest_sha256"],
        "split_sha256": schema["split_sha256"],
        "fold_id": str(args.fold_id),
        "seed": int(args.seed),
    }
    datasets, loaders, train_sampler, normalization = make_loaders(args)
    model, provenance, parent = build_model(args, identity)
    model = model.to(args.device)
    pos_weight = estimate_pos_weight(datasets["train"])
    fixed_threshold = None if parent is None else float(parent["threshold"])
    config = {
        "schema_version": "pild_sen12_roleaware_config.v1",
        "contract": "PILD+Sen12 role-aware GeoPhysAdapter v1",
        "variant": args.variant,
        "condition": args.variant,
        "seed": int(args.seed),
        "terrain_channel_order": list(args.terrain_names),
        "terrain_schema_id": schema["terrain_schema_id"],
        "material_interaction_groups": {
            key: list(value)
            for key, value in material_terrain_groups(args.terrain_names).items()
        },
        "roles": {
            "Terrain": (
                f"{len(args.terrain_names)}-channel aligned bounded dense residual"
            ),
            "Material": "21-D context; bounded modulation of existing Terrain residual only",
            "Trigger": "event scalar; visual-uncertainty calibration only",
        },
        "sampling": {
            "mode": args.sampling_mode,
            "epoch_samples": int(args.epoch_samples or len(datasets["train"])),
        },
        "training_context": {
            "terrain": TERRAIN_ALIGNED,
            "material": MATERIAL_ALIGNED,
            "trigger": TRIGGER_ALIGNED,
        },
        "evaluation_contexts": [
            {
                "name": context.name,
                "terrain": context.terrain,
                "material": context.material,
                "trigger": context.trigger,
            }
            for context in evaluation_contexts_for_variant(args.variant)
        ],
        "identity": identity,
        "schema_validation": schema,
        "normalization": normalization,
        "prithvi_provenance": provenance,
        "parent_checkpoint": str(args.parent_checkpoint.resolve()) if args.parent_checkpoint else None,
        "args": vars(args),
        "command": command,
    }
    atomic_write_json(stage / "config.json", config)
    best_state, history, best_epoch, threshold = train_model(
        model,
        loaders["train"],
        train_sampler,
        loaders["val"],
        args,
        pos_weight,
        fixed_threshold,
        log,
    )
    load_trainable_state(model, best_state)
    if parent is None:
        fixed_fpr_threshold = choose_fixed_fpr_threshold(
            validation_histogram(model, loaders["val"], args.device)
        )
    else:
        if "fixed_fpr_threshold" not in parent:
            raise RuntimeError("parent checkpoint lacks validation_visual_only fixed-FPR threshold")
        fixed_fpr_threshold = float(parent["fixed_fpr_threshold"])
    components = component_state(model)
    component_hashes = {name: tensor_sha256(value) for name, value in components.items()}
    checkpoint = {
        "variant": args.variant,
        "identity": identity,
        "components": components,
        "component_sha256": component_hashes,
        "threshold": threshold,
        "threshold_source": "visual_validation" if args.variant == "V" else "matched_V_parent",
        "fixed_fpr_threshold": fixed_fpr_threshold,
        "fixed_fpr_threshold_source": "validation_visual_only",
        "best_epoch": best_epoch,
        "history": history,
        "normalization": normalization,
        "training_context": {
            "terrain": TERRAIN_ALIGNED,
            "material": MATERIAL_ALIGNED,
            "trigger": TRIGGER_ALIGNED,
        },
    }
    checkpoint_tmp = stage / f".checkpoint.pt.tmp-{os.getpid()}"
    torch.save(checkpoint, checkpoint_tmp)
    os.replace(checkpoint_tmp, stage / "checkpoint.pt")
    checkpoint_sha256 = sha256_file(stage / "checkpoint.pt")
    per_sample: list[dict[str, Any]] = []
    per_event: list[dict[str, Any]] = []
    corpus = []
    for split in ("val", "test"):
        for evaluation_context in evaluation_contexts_for_variant(args.variant):
            sample_rows, event_rows, split_metrics = evaluate(
                model,
                loaders[split],
                args.device,
                threshold,
                fixed_fpr_threshold,
                split,
                evaluation_context,
                args.seed,
                checkpoint_sha256,
                component_hashes,
            )
            per_sample.extend(sample_rows)
            per_event.extend(event_rows)
            corpus.append(split_metrics)
    atomic_write_csv(stage / "per_sample.csv", per_sample)
    atomic_write_csv(stage / "per_event.csv", per_event)
    analyzer_rows = [
        row
        for row in per_sample
        if row["split"] == "test" and row["condition"] == args.variant
    ]
    atomic_write_csv(stage / "per_sample_metrics.csv", analyzer_rows)
    result = {
        "schema_version": "pild_sen12_roleaware_run.v1",
        "status": "complete",
        "variant": args.variant,
        "condition": args.variant,
        "seed": int(args.seed),
        "evaluation_split": "test",
        "identity": identity,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "fixed_fpr_threshold": fixed_fpr_threshold,
        "fixed_fpr_threshold_source": "validation_visual_only",
        "pos_weight": pos_weight,
        "history": history,
        "corpus_metrics": corpus,
        "checkpoint_sha256": checkpoint_sha256,
        "component_sha256": component_hashes,
        "n_per_sample_rows": len(per_sample),
        "n_per_event_rows": len(per_event),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(stage / "result.json", result)
    required = (
        "config.json",
        "run.log",
        "result.json",
        "per_sample.csv",
        "per_event.csv",
        "per_sample_metrics.csv",
        "checkpoint.pt",
    )
    if any(not (stage / name).is_file() or (stage / name).stat().st_size == 0 for name in required):
        raise RuntimeError("artifact completeness gate failed")
    done = {
        "schema_version": "pild_sen12_roleaware_done.v1",
        "status": "complete",
        "variant": args.variant,
        "condition": args.variant,
        "fold_id": args.fold_id,
        "seed": args.seed,
        "config_sha256": sha256_file(stage / "config.json"),
        "result_sha256": sha256_file(stage / "result.json"),
        "per_sample_metrics_sha256": sha256_file(stage / "per_sample_metrics.csv"),
        "artifacts": {
            name: {"sha256": sha256_file(stage / name), "size": (stage / name).stat().st_size}
            for name in required
        },
    }
    atomic_write_json(stage / "DONE.json", done)
    os.replace(stage, outdir)
    print(f"[done] atomically published {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
