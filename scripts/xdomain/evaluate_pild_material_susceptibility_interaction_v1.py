#!/usr/bin/env python3
"""Event-OOF test of a role-constrained Terrain x Material susceptibility model.

Protocol assumptions (frozen before looking at outcomes)
--------------------------------------------------------
1. This is a susceptibility probe, not a visual-segmentation experiment. It
   intentionally needs no optical cache.
2. Terrain is the only dense direction. The T-only model uses local slope,
   Laplacian curvature, and 300 m local relief.
3. Material has no independent pixel-level main effect in the confirmatory
   T x M model. It may only modulate Terrain through a fixed low-dimensional
   interaction basis: AWC, shallow-AWC fraction, clay, sand, SOC, and CEC.
4. Material interactions are multiplied by the recipient sample's q_M_full.
   Therefore q_M_full == 0 makes every interaction exactly zero.
5. Five folds isolate canonical events. A canonical event represented by more
   than one source remains wholly in one fold.
6. Training pixels are selected deterministically and weighted to balance
   source -> event -> class. Held events are evaluated on all valid pixels.
7. Robust centering/scaling is fit from training events only. sample_id,
   event_id, source_id, and any labels from held events are never model inputs.
8. T x M test-shuffled is an inference-only intervention on the exact fitted
   aligned model. Material donors come from another held event within the same
   source; the recipient q_M_full is retained.
9. M-only is a leakage diagnostic and must not be reported as a physical
   segmentation contribution.
10. Feature sets, regularization, folds, and controls are fixed in this file.
    A negative or inconclusive result is reported without outcome-driven
    retuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "metadata/pild_sen12_training_v2/unified_sample_manifest_v2.csv"
)
DEFAULT_OUTDIR = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_material_susceptibility_interaction_v1"
)

TERRAIN_FEATURES = ("slope", "curvature", "relief_300m")
MATERIAL_FEATURES = (
    "awc_total",
    "awc_shallow_fraction",
    "clay_shallow",
    "sand_shallow",
    "soc_shallow",
    "cec_shallow",
)
TRAIN_CONDITIONS = (
    "T_ONLY",
    "TXM_ALIGNED",
    "M_ONLY_DIAGNOSTIC",
)
EVALUATION_CONDITIONS = (
    "T_ONLY",
    "TXM_ALIGNED",
    "TXM_TEST_SHUFFLED_SAME_MODEL",
    "M_ONLY_DIAGNOSTIC",
)


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        clean = np.where(np.isfinite(values), values, self.center)
        return np.clip((clean - self.center) / self.scale, -5.0, 5.0)


@dataclass(frozen=True)
class PixelRef:
    manifest_index: int
    flat_index: int
    source_id: str
    event_id: str
    class_id: int
    priority: int


@dataclass(frozen=True)
class FixedOffsetInteractionModel:
    """Material interaction residual on a frozen Terrain-only logit."""

    terrain_model: LogisticRegression
    interaction_coef: np.ndarray
    n_terrain_features: int
    n_iter: int
    optimization_success: bool

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        terrain = values[:, : self.n_terrain_features]
        interactions = values[:, self.n_terrain_features :]
        return (
            self.terrain_model.decision_function(terrain)
            + interactions @ self.interaction_coef
        )

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        positive = expit(self.decision_function(values))
        return np.column_stack([1.0 - positive, positive])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pixels-per-event-class", type=int, default=1024)
    parser.add_argument("--per-sample-class-cap", type=int, default=64)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def robust_scaler(values: np.ndarray) -> RobustScaler:
    values = np.asarray(values, dtype=np.float64)
    center = np.nanmedian(values, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    q25 = np.nanpercentile(values, 25.0, axis=0)
    q75 = np.nanpercentile(values, 75.0, axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return RobustScaler(center.astype(np.float64), scale.astype(np.float64))


def weighted_scaler(values: np.ndarray, weights: np.ndarray) -> RobustScaler:
    """Use the fixed robust estimator; weights are accepted for testability.

    Source/event/class balance is enforced in estimator sample weights. Robust
    train-only scaling intentionally does not inspect source or event IDs.
    """

    if len(values) != len(weights):
        raise ValueError("values and weights have different lengths")
    return robust_scaler(values)


def first_numeric(row: pd.Series, names: Sequence[str]) -> float:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return float("nan")


def material_vector(row: pd.Series) -> tuple[np.ndarray, float]:
    awc_total = first_numeric(
        row,
        ("awc_0_200_aligned_mm", "awc_0_200_footprint_mean_mm", "awc_layer_sum_0_200_mm"),
    )
    awc_0_10 = first_numeric(row, ("awc_0_10_aligned_mm", "awc_0_10_footprint_mean_mm"))
    awc_10_30 = first_numeric(row, ("awc_10_30_aligned_mm", "awc_10_30_footprint_mean_mm"))
    shallow_fraction = (
        (awc_0_10 + awc_10_30) / awc_total
        if np.isfinite(awc_total) and awc_total > 1e-8 and np.isfinite(awc_0_10 + awc_10_30)
        else float("nan")
    )
    vector = np.asarray(
        [
            awc_total,
            shallow_fraction,
            first_numeric(row, ("soil_clay_0_5cm_mean_raw", "soil_clay_0_5cm_center_raw")),
            first_numeric(row, ("soil_sand_0_5cm_mean_raw", "soil_sand_0_5cm_center_raw")),
            first_numeric(row, ("soil_soc_0_5cm_mean_raw", "soil_soc_0_5cm_center_raw")),
            first_numeric(row, ("soil_cec_0_5cm_mean_raw", "soil_cec_0_5cm_center_raw")),
        ],
        dtype=np.float64,
    )
    q_m = first_numeric(row, ("q_M_full",))
    if not np.isfinite(q_m):
        q_m = 0.0
    return vector, float(np.clip(q_m, 0.0, 1.0))


def interaction_features(
    terrain_z: np.ndarray,
    material_z: np.ndarray,
    q_m: np.ndarray,
) -> np.ndarray:
    terrain_z = np.asarray(terrain_z, dtype=np.float64)
    material_z = np.asarray(material_z, dtype=np.float64)
    q_m = np.asarray(q_m, dtype=np.float64).reshape(-1, 1, 1)
    if terrain_z.ndim != 2 or material_z.ndim != 2:
        raise ValueError("terrain_z and material_z must be matrices")
    if len(terrain_z) != len(material_z):
        raise ValueError("terrain and material row counts differ")
    interactions = terrain_z[:, :, None] * material_z[:, None, :] * q_m
    return interactions.reshape(len(terrain_z), -1)


def condition_matrix(
    condition: str,
    terrain_z: np.ndarray,
    material_z: np.ndarray,
    q_m: np.ndarray,
) -> np.ndarray:
    if condition == "T_ONLY":
        return terrain_z
    if condition in {"TXM_ALIGNED", "TXM_TEST_SHUFFLED_SAME_MODEL"}:
        return np.concatenate(
            [terrain_z, interaction_features(terrain_z, material_z, q_m)], axis=1
        )
    if condition == "M_ONLY_DIAGNOSTIC":
        return material_z * np.asarray(q_m).reshape(-1, 1)
    raise KeyError(condition)


def assign_event_folds(manifest: pd.DataFrame, folds: int, seed: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    presence = (
        manifest[["canonical_event_id", "source_id"]]
        .drop_duplicates()
        .groupby("canonical_event_id")["source_id"]
        .apply(lambda values: tuple(sorted(map(str, values))))
    )
    sources = sorted(manifest["source_id"].astype(str).unique())
    source_index = {source: index for index, source in enumerate(sources)}
    counts = np.zeros((folds, len(sources)), dtype=np.int64)
    total = np.zeros(folds, dtype=np.int64)
    assignment: dict[str, int] = {}
    events = sorted(
        presence.index.astype(str),
        key=lambda event: (-len(presence[event]), stable_u64(seed, "event-fold-order", event)),
    )
    for event in events:
        vector = np.zeros(len(sources), dtype=np.int64)
        for source in presence[event]:
            vector[source_index[source]] = 1
        candidates = []
        for fold in range(folds):
            projected = counts.copy()
            projected[fold] += vector
            imbalance = float(np.sum(np.var(projected, axis=0)))
            candidates.append((imbalance, int(total[fold]), stable_u64(seed, event, fold), fold))
        selected = min(candidates)[-1]
        assignment[event] = selected
        counts[selected] += vector
        total[selected] += 1
    return assignment


def terrain_indices(names: Sequence[str]) -> tuple[int, int, int]:
    normalized = [str(name).lower() for name in names]

    def find(aliases: Sequence[str]) -> int:
        for alias in aliases:
            if alias in normalized:
                return normalized.index(alias)
        raise KeyError(f"terrain channel missing; aliases={aliases}; names={normalized}")

    return (
        find(("slope", "slope_deg")),
        find(("curvature_laplacian", "laplacian_curvature")),
        find(("local_relief_300m",)),
    )


class AssetReader:
    def __init__(self, manifest: pd.DataFrame):
        self.manifest = manifest.set_index("manifest_index", drop=False)
        self.h5: dict[str, h5py.File] = {}
        self.terrain_channel_cache: dict[str, tuple[int, int, int]] = {}
        self.material_tables: dict[str, pd.DataFrame] = {}
        self.material_cache: dict[int, tuple[np.ndarray, float]] = {}

    def close(self) -> None:
        for handle in self.h5.values():
            handle.close()

    def _h5(self, path: str) -> h5py.File:
        if path not in self.h5:
            self.h5[path] = h5py.File(path, "r")
        return self.h5[path]

    def _terrain_indices(self, path: str) -> tuple[int, int, int]:
        if path not in self.terrain_channel_cache:
            handle = self._h5(path)
            names = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["terrain_names"][:]
            ]
            self.terrain_channel_cache[path] = terrain_indices(names)
        return self.terrain_channel_cache[path]

    def mask_valid(self, manifest_index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.manifest.loc[manifest_index]
        handle = self._h5(str(row.base_h5_path))
        index = int(row.base_h5_index)
        mask = np.asarray(handle["mask"][index]).squeeze().astype(bool)
        valid = np.asarray(handle["valid_mask"][index]).squeeze().astype(bool)
        terrain_handle = self._h5(str(row.terrain_h5_path))
        terrain_index = int(row.terrain_h5_index)
        if "terrain_valid" in terrain_handle:
            valid &= np.asarray(terrain_handle["terrain_valid"][terrain_index]).squeeze().astype(bool)
        return mask, valid

    def terrain_pixels(self, manifest_index: int, flat_indices: np.ndarray) -> np.ndarray:
        row = self.manifest.loc[manifest_index]
        path = str(row.terrain_h5_path)
        handle = self._h5(path)
        index = int(row.terrain_h5_index)
        channel_ids = self._terrain_indices(path)
        terrain = np.asarray(handle["terrain"][index], dtype=np.float32)
        flat = terrain.reshape(terrain.shape[0], -1)
        return flat[np.asarray(channel_ids)][:, flat_indices].T.astype(np.float64)

    def material(self, manifest_index: int) -> tuple[np.ndarray, float]:
        if manifest_index in self.material_cache:
            return self.material_cache[manifest_index]
        row = self.manifest.loc[manifest_index]
        path = str(row.material_registry_path)
        if path not in self.material_tables:
            self.material_tables[path] = pd.read_csv(path, low_memory=False)
        material_index = int(row.material_registry_index)
        table = self.material_tables[path]
        if not 0 <= material_index < len(table):
            raise IndexError(f"material index {material_index} outside {path}")
        material_row = table.iloc[material_index]
        if str(material_row["sample_id"]) != str(row.sample_id):
            raise RuntimeError(
                f"material sample mismatch: manifest={row.sample_id}, registry={material_row['sample_id']}"
            )
        result = material_vector(material_row)
        self.material_cache[manifest_index] = result
        return result


def deterministic_pixel_refs(
    manifest: pd.DataFrame,
    reader: AssetReader,
    train_events: set[str],
    seed: int,
    per_sample_class_cap: int,
    pixels_per_event_class: int,
) -> list[PixelRef]:
    buckets: dict[tuple[str, str, int], list[PixelRef]] = defaultdict(list)
    active = manifest[manifest["canonical_event_id"].astype(str).isin(train_events)]
    for row in active.itertuples(index=False):
        mask, valid = reader.mask_valid(int(row.manifest_index))
        flat_mask = mask.reshape(-1)
        flat_valid = valid.reshape(-1)
        for class_id in (0, 1):
            candidates = np.flatnonzero(flat_valid & (flat_mask == bool(class_id)))
            if not len(candidates):
                continue
            local_seed = stable_u64(seed, "pixel-candidates", row.sample_id, class_id)
            rng = np.random.default_rng(local_seed)
            if len(candidates) > per_sample_class_cap:
                candidates = rng.choice(candidates, size=per_sample_class_cap, replace=False)
            key = (str(row.source_id), str(row.canonical_event_id), class_id)
            for flat_index in candidates:
                buckets[key].append(
                    PixelRef(
                        manifest_index=int(row.manifest_index),
                        flat_index=int(flat_index),
                        source_id=str(row.source_id),
                        event_id=str(row.canonical_event_id),
                        class_id=class_id,
                        priority=stable_u64(
                            seed, "event-class-pixel", row.sample_id, int(flat_index), class_id
                        ),
                    )
                )
    selected: list[PixelRef] = []
    for key, refs in sorted(buckets.items()):
        ordered = sorted(refs, key=lambda ref: ref.priority)
        selected.extend(ordered[:pixels_per_event_class])
    return selected


def balanced_weights(refs: Sequence[PixelRef]) -> np.ndarray:
    sources = sorted({ref.source_id for ref in refs})
    events_by_source: dict[str, set[str]] = defaultdict(set)
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for ref in refs:
        events_by_source[ref.source_id].add(ref.event_id)
        counts[(ref.source_id, ref.event_id, ref.class_id)] += 1
    weights = []
    for ref in refs:
        denominator = (
            len(sources)
            * len(events_by_source[ref.source_id])
            * 2
            * counts[(ref.source_id, ref.event_id, ref.class_id)]
        )
        weights.append(1.0 / denominator)
    output = np.asarray(weights, dtype=np.float64)
    return output / output.mean()


def donor_manifest_map(manifest: pd.DataFrame, seed: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for source, source_frame in manifest.groupby("source_id", sort=True):
        events = sorted(
            source_frame["canonical_event_id"].astype(str).unique(),
            key=lambda event: stable_u64(seed, "M-event-order", source, event),
        )
        if len(events) < 2:
            raise RuntimeError(f"source {source} has fewer than two Material donor events")
        event_donor = {event: events[(index + 1) % len(events)] for index, event in enumerate(events)}
        grouped = {
            str(event): sorted(group.manifest_index.astype(int).tolist())
            for event, group in source_frame.groupby("canonical_event_id", sort=False)
        }
        for row in source_frame.itertuples(index=False):
            donor_event = event_donor[str(row.canonical_event_id)]
            candidates = grouped[donor_event]
            position = stable_u64(seed, "M-sample-donor", row.sample_id) % len(candidates)
            mapping[int(row.manifest_index)] = int(candidates[position])
    return mapping


def assemble_training_arrays(
    refs: Sequence[PixelRef],
    reader: AssetReader,
) -> dict[str, np.ndarray]:
    terrain = np.zeros((len(refs), len(TERRAIN_FEATURES)), dtype=np.float64)
    aligned_m = np.zeros((len(refs), len(MATERIAL_FEATURES)), dtype=np.float64)
    q_m = np.zeros(len(refs), dtype=np.float64)
    labels = np.asarray([ref.class_id for ref in refs], dtype=np.uint8)
    by_sample: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_index, ref in enumerate(refs):
        by_sample[ref.manifest_index].append((row_index, ref.flat_index))
    for manifest_index, locations in by_sample.items():
        rows = np.asarray([row for row, _ in locations], dtype=np.int64)
        pixels = np.asarray([pixel for _, pixel in locations], dtype=np.int64)
        terrain[rows] = reader.terrain_pixels(manifest_index, pixels)
        m_value, q_value = reader.material(manifest_index)
        aligned_m[rows] = m_value
        q_m[rows] = q_value
    return {
        "terrain": terrain,
        "material_aligned": aligned_m,
        "q_m": q_m,
        "labels": labels,
        "weights": balanced_weights(refs),
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores)) if np.any(labels == 1) else float("nan")


def metric_row(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "n_pixels": int(len(labels)),
        "n_positive": int(np.sum(labels == 1)),
        "positive_fraction": float(np.mean(labels == 1)) if len(labels) else float("nan"),
        "auc": safe_auc(labels, scores),
        "ap": safe_ap(labels, scores),
    }


def event_support_map(sample_frame: pd.DataFrame) -> dict[tuple[int, str], float]:
    """Collapse cross-source aliases to the canonical-event support contract."""

    return {
        (int(fold), str(event)): float(value)
        for (fold, event), value in sample_frame.groupby(
            ["fold", "canonical_event_id"], sort=True
        )["q_M_full"].max().items()
    }


def fit_model(x: np.ndarray, labels: np.ndarray, weights: np.ndarray, c_value: float, max_iter: int) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=0,
    )
    model.fit(x, labels, sample_weight=weights)
    return model


def fit_fixed_offset_interaction(
    terrain_model: LogisticRegression,
    terrain: np.ndarray,
    interactions: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    c_value: float,
    max_iter: int,
) -> FixedOffsetInteractionModel:
    """Fit only T x M residual coefficients; Terrain logits stay immutable."""

    terrain = np.asarray(terrain, dtype=np.float64)
    interactions = np.asarray(interactions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if interactions.ndim != 2 or len(interactions) != len(labels):
        raise ValueError("interaction matrix and labels are inconsistent")
    if c_value <= 0:
        raise ValueError("logistic C must be positive")
    base_logit = np.asarray(terrain_model.decision_function(terrain), dtype=np.float64)
    normalized_weight = weights / np.sum(weights)
    ridge = 1.0 / (c_value * max(len(labels), 1))

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        logit = base_logit + interactions @ coef
        probability = expit(logit)
        data_loss = np.sum(
            normalized_weight * (np.logaddexp(0.0, logit) - labels * logit)
        )
        loss = data_loss + 0.5 * ridge * float(coef @ coef)
        gradient = interactions.T @ (normalized_weight * (probability - labels))
        gradient = gradient + ridge * coef
        return float(loss), np.asarray(gradient, dtype=np.float64)

    initial = np.zeros(interactions.shape[1], dtype=np.float64)
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-10, "gtol": 1e-7},
    )
    if not np.all(np.isfinite(fitted.x)):
        raise RuntimeError("fixed-offset Material optimizer produced non-finite coefficients")
    return FixedOffsetInteractionModel(
        terrain_model=terrain_model,
        interaction_coef=np.asarray(fitted.x, dtype=np.float64),
        n_terrain_features=terrain.shape[1],
        n_iter=int(fitted.nit),
        optimization_success=bool(fitted.success),
    )


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def protocol_markdown(args: argparse.Namespace) -> str:
    return f"""# PILD Material susceptibility interaction v1 - frozen assumptions

- Scope: 56 canonical events from the unified PILD/Sen12 manifest; no optical cache.
- Split: {args.folds}-fold canonical-event OOF; cross-source aliases remain in one fold.
- Terrain dense direction: `{', '.join(TERRAIN_FEATURES)}`.
- Material moderators: `{', '.join(MATERIAL_FEATURES)}`.
- Confirmatory fitted models: `T_ONLY`, `TXM_ALIGNED`.
- Attribution control: the exact fitted `TXM_ALIGNED` model is evaluated with
  held-event, same-source Material donors (`TXM_TEST_SHUFFLED_SAME_MODEL`).
- Diagnostic only: `M_ONLY_DIAGNOSTIC`.
- Material contract: no Material main effect in T x M; q_M_full=0 makes all interactions exactly zero.
- Training sampling: deterministic source -> event -> class balance, up to {args.pixels_per_event_class} pixels/event/class.
- Evaluation: all valid pixels in held canonical events; sample, event, and fold AUC/AP.
- Preprocessing: train-only median/IQR robust standardization.
- Estimator: fixed L2 logistic regression, C={args.logistic_c}; no outcome-driven tuning.
- Primary contrasts: aligned minus T-only and aligned minus event-shuffled at paired-event level.
- Interpretation: this tests susceptibility interaction, not visual segmentation or causal effect.
"""


def validate_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "manifest_index",
        "source_id",
        "canonical_event_id",
        "sample_id",
        "base_h5_path",
        "base_h5_index",
        "terrain_h5_path",
        "terrain_h5_index",
        "material_registry_path",
        "material_registry_index",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"manifest missing columns: {sorted(missing)}")
    if manifest["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("sample_id must be unique")
    if manifest["canonical_event_id"].nunique() != 56:
        raise RuntimeError(
            f"expected 56 canonical events, found {manifest['canonical_event_id'].nunique()}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.manifest = args.manifest.resolve()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "protocol_assumptions.md").write_text(protocol_markdown(args), encoding="utf-8")
    (args.outdir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    manifest = pd.read_csv(args.manifest, low_memory=False)
    validate_manifest(manifest)
    if args.smoke:
        keep_events = []
        for source, group in manifest.groupby("source_id", sort=True):
            keep_events.extend(sorted(group.canonical_event_id.astype(str).unique())[: min(5, group.canonical_event_id.nunique())])
        manifest = manifest[manifest.canonical_event_id.astype(str).isin(set(keep_events))].copy()
        args.folds = min(args.folds, max(2, manifest.canonical_event_id.nunique() // 2))

    event_folds = assign_event_folds(manifest, args.folds, args.seed)
    manifest["fold"] = manifest["canonical_event_id"].astype(str).map(event_folds)
    reader = AssetReader(manifest)
    fold_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    try:
        for fold in range(args.folds):
            test_frame = manifest[manifest.fold == fold].copy()
            train_frame = manifest[manifest.fold != fold].copy()
            test_donor_map = donor_manifest_map(test_frame, args.seed + fold)
            train_events = set(train_frame.canonical_event_id.astype(str))
            refs = deterministic_pixel_refs(
                manifest,
                reader,
                train_events,
                args.seed + fold,
                args.per_sample_class_cap,
                args.pixels_per_event_class,
            )
            if not refs:
                raise RuntimeError(f"fold {fold}: no training pixels")
            arrays = assemble_training_arrays(refs, reader)
            terrain_scaler = weighted_scaler(arrays["terrain"], arrays["weights"])
            material_scaler = robust_scaler(arrays["material_aligned"])
            terrain_z = terrain_scaler.transform(arrays["terrain"])
            aligned_z = material_scaler.transform(arrays["material_aligned"])

            terrain_model = fit_model(
                terrain_z,
                arrays["labels"],
                arrays["weights"],
                args.logistic_c,
                args.max_iter,
            )
            material_only_x = condition_matrix(
                "M_ONLY_DIAGNOSTIC", terrain_z, aligned_z, arrays["q_m"]
            )
            material_only_model = fit_model(
                material_only_x,
                arrays["labels"],
                arrays["weights"],
                args.logistic_c,
                args.max_iter,
            )
            interactions = interaction_features(terrain_z, aligned_z, arrays["q_m"])
            interaction_model = fit_fixed_offset_interaction(
                terrain_model,
                terrain_z,
                interactions,
                arrays["labels"],
                arrays["weights"],
                args.logistic_c,
                args.max_iter,
            )
            models: dict[str, Any] = {
                "T_ONLY": terrain_model,
                "TXM_ALIGNED": interaction_model,
                "M_ONLY_DIAGNOSTIC": material_only_model,
            }
            for condition in TRAIN_CONDITIONS:
                model = models[condition]
                if isinstance(model, FixedOffsetInteractionModel):
                    n_features = len(TERRAIN_FEATURES) + len(model.interaction_coef)
                    n_iter = model.n_iter
                    intercept = 0.0
                    coefficient = model.interaction_coef
                    fit_role = "fixed_T_logit_plus_no_intercept_TxM_residual"
                    success = model.optimization_success
                else:
                    n_features = int(model.coef_.shape[1])
                    n_iter = int(np.max(model.n_iter_))
                    intercept = float(model.intercept_[0])
                    coefficient = model.coef_[0]
                    fit_role = "standalone_logistic"
                    success = True
                model_rows.append(
                    {
                        "fold": fold,
                        "condition": condition,
                        "fit_role": fit_role,
                        "optimization_success": success,
                        "n_features": n_features,
                        "n_iter": n_iter,
                        "intercept": intercept,
                        "coef_json": json.dumps(coefficient.tolist()),
                    }
                )

            fold_scores: dict[str, list[np.ndarray]] = {
                condition: [] for condition in EVALUATION_CONDITIONS
            }
            fold_labels: list[np.ndarray] = []
            event_scores: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
            event_labels: dict[str, list[np.ndarray]] = defaultdict(list)

            for row in test_frame.itertuples(index=False):
                mask, valid = reader.mask_valid(int(row.manifest_index))
                pixels = np.flatnonzero(valid.reshape(-1))
                if not len(pixels):
                    continue
                labels = mask.reshape(-1)[pixels].astype(np.uint8)
                terrain_raw = reader.terrain_pixels(int(row.manifest_index), pixels)
                terrain_eval = terrain_scaler.transform(terrain_raw)
                material_raw, q_value = reader.material(int(row.manifest_index))
                donor_raw, _ = reader.material(test_donor_map[int(row.manifest_index)])
                aligned_eval = material_scaler.transform(material_raw[None, :]).repeat(len(pixels), axis=0)
                shuffled_eval = material_scaler.transform(donor_raw[None, :]).repeat(len(pixels), axis=0)
                q_eval = np.full(len(pixels), q_value, dtype=np.float64)
                fold_labels.append(labels)
                event_labels[str(row.canonical_event_id)].append(labels)
                for condition in EVALUATION_CONDITIONS:
                    fitted_condition = (
                        "TXM_ALIGNED"
                        if condition == "TXM_TEST_SHUFFLED_SAME_MODEL"
                        else condition
                    )
                    model = models[fitted_condition]
                    material_eval = (
                        shuffled_eval
                        if condition == "TXM_TEST_SHUFFLED_SAME_MODEL"
                        else aligned_eval
                    )
                    x_eval = condition_matrix(condition, terrain_eval, material_eval, q_eval)
                    scores = model.predict_proba(x_eval)[:, 1].astype(np.float32)
                    fold_scores[condition].append(scores)
                    event_scores[(str(row.canonical_event_id), condition)].append(scores)
                    sample_rows.append(
                        {
                            "fold": fold,
                            "condition": condition,
                            "fitted_model_condition": fitted_condition,
                            "source_id": str(row.source_id),
                            "canonical_event_id": str(row.canonical_event_id),
                            "sample_id": str(row.sample_id),
                            "q_M_full": q_value,
                            **metric_row(labels, scores),
                        }
                    )

            fold_y = np.concatenate(fold_labels)
            for condition in EVALUATION_CONDITIONS:
                scores = np.concatenate(fold_scores[condition])
                fold_rows.append(
                    {
                        "fold": fold,
                        "condition": condition,
                        "fitted_model_condition": (
                            "TXM_ALIGNED"
                            if condition == "TXM_TEST_SHUFFLED_SAME_MODEL"
                            else condition
                        ),
                        "n_train_pixels": len(refs),
                        "n_train_events": len(train_events),
                        "n_test_events": int(test_frame.canonical_event_id.nunique()),
                        **metric_row(fold_y, scores),
                    }
                )
            for event in sorted(event_labels):
                event_y = np.concatenate(event_labels[event])
                source_signature = "+".join(
                    sorted(test_frame.loc[test_frame.canonical_event_id.astype(str) == event, "source_id"].astype(str).unique())
                )
                for condition in EVALUATION_CONDITIONS:
                    scores = np.concatenate(event_scores[(event, condition)])
                    event_rows.append(
                        {
                            "fold": fold,
                            "condition": condition,
                            "fitted_model_condition": (
                                "TXM_ALIGNED"
                                if condition == "TXM_TEST_SHUFFLED_SAME_MODEL"
                                else condition
                            ),
                            "source_id": source_signature,
                            "canonical_event_id": event,
                            **metric_row(event_y, scores),
                        }
                    )
            print(
                f"[fold {fold}] train_events={len(train_events)} test_events={test_frame.canonical_event_id.nunique()} train_pixels={len(refs)}",
                flush=True,
            )
    finally:
        reader.close()

    sample_frame = pd.DataFrame(sample_rows)
    event_frame = pd.DataFrame(event_rows)
    fold_frame = pd.DataFrame(fold_rows)
    model_frame = pd.DataFrame(model_rows)
    sample_frame.to_csv(args.outdir / "sample_metrics.csv", index=False)
    event_frame.to_csv(args.outdir / "event_metrics.csv", index=False)
    fold_frame.to_csv(args.outdir / "fold_metrics.csv", index=False)
    model_frame.to_csv(args.outdir / "model_coefficients.csv", index=False)

    pivot = event_frame.pivot(
        index=["fold", "source_id", "canonical_event_id"],
        columns="condition",
        values=["auc", "ap"],
    )
    event_q_m = event_support_map(sample_frame)
    paired_rows: list[dict[str, Any]] = []
    contrasts = (
        ("TXM_ALIGNED_MINUS_T_ONLY", "TXM_ALIGNED", "T_ONLY"),
        (
            "TXM_ALIGNED_MINUS_TEST_SHUFFLED_SAME_MODEL",
            "TXM_ALIGNED",
            "TXM_TEST_SHUFFLED_SAME_MODEL",
        ),
    )
    for contrast, left, right in contrasts:
        for metric in ("auc", "ap"):
            values = (pivot[(metric, left)] - pivot[(metric, right)]).rename("delta")
            for index, delta in values.items():
                paired_rows.append(
                    {
                        "contrast": contrast,
                        "metric": metric,
                        "fold": index[0],
                        "source_id": index[1],
                        "canonical_event_id": index[2],
                        "q_M_full": float(event_q_m[(index[0], index[2])]),
                        "delta": delta,
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.outdir / "event_paired_deltas.csv", index=False)

    contrast_summary: list[dict[str, Any]] = []
    for (contrast, metric), group in paired.groupby(["contrast", "metric"], sort=True):
        scopes = {
            "all_events": group,
            "material_supported_events": group[group["q_M_full"] > 0],
        }
        for scope, scoped in scopes.items():
            values = scoped.delta.to_numpy(dtype=np.float64)
            lower, upper = bootstrap_mean_ci(
                values,
                args.bootstrap_reps,
                stable_u64(args.seed, contrast, metric, scope),
            )
            contrast_summary.append(
                {
                    "scope": scope,
                    "contrast": contrast,
                    "metric": metric,
                    "n_events": int(np.isfinite(values).sum()),
                    "mean_delta": float(np.nanmean(values)),
                    "median_delta": float(np.nanmedian(values)),
                    "positive_events": int(np.sum(values > 0)),
                    "ci95_low": lower,
                    "ci95_high": upper,
                }
            )

    fold_assignment = pd.DataFrame(
        [{"canonical_event_id": event, "fold": fold} for event, fold in sorted(event_folds.items())]
    )
    fold_assignment.to_csv(args.outdir / "event_fold_assignment.csv", index=False)
    summary = {
        "status": "complete",
        "interpretation": "susceptibility probe; not visual segmentation",
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        "n_samples": int(len(manifest)),
        "n_canonical_events": int(manifest.canonical_event_id.nunique()),
        "n_sources": int(manifest.source_id.nunique()),
        "folds": args.folds,
        "terrain_features": TERRAIN_FEATURES,
        "material_features": MATERIAL_FEATURES,
        "train_conditions": TRAIN_CONDITIONS,
        "evaluation_conditions": EVALUATION_CONDITIONS,
        "q_m_zero_contract": "all T x M interactions exactly zero",
        "contrasts": contrast_summary,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# PILD Material susceptibility interaction v1",
        "",
        "This is an event-OOF susceptibility probe without optical inputs. Material is only a Terrain interaction moderator.",
        "",
        "| Scope | Contrast | Metric | Events | Mean delta | 95% bootstrap CI | Positive events |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in contrast_summary:
        lines.append(
            f"| {item['scope']} | {item['contrast']} | {item['metric']} | {item['n_events']} | "
            f"{item['mean_delta']:.6f} | [{item['ci95_low']:.6f}, {item['ci95_high']:.6f}] | "
            f"{item['positive_events']} |"
        )
    lines.extend(
        [
            "",
            "M_ONLY_DIAGNOSTIC is retained only to expose source/event leakage and is not a contribution claim.",
        ]
    )
    (args.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.outdir / "DONE.json").write_text(
        json.dumps({"status": "complete", "summary": "summary.json"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
