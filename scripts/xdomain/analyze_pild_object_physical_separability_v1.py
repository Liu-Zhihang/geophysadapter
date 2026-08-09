#!/usr/bin/env python3
"""G0 diagnostic: are predicted landslide bodies separable by physical plausibility?

The pixel-level gates explored so far act on the visual decision boundary, which is
exactly where true positives are most fragile. This diagnostic asks whether the
decision can instead be made per connected component, using first-principles
gravity constraints computed from the native Terrain pyramid.

Three pre-registered questions:

1. Addressable FP mass  : what share of all false-positive pixels sits inside
                          near-pure false-positive components? This bounds what
                          object-level rejection can ever remove without harming TP.
2. Physical separability: can gravity-consistency features distinguish
                          false-positive components from true-positive ones, with
                          event-grouped out-of-fold validation?
3. Confidence placement : are false-positive components visually confident? If so,
                          uncertainty-gated pixel correction cannot reach them.

Labels are used here only to define the diagnostic target, never to select samples,
thresholds or model parameters for any downstream model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Channel order of the frozen pild_native_terrain17_v1 contract.
TERRAIN_INDEX = {
    "elevation": 0,
    "slope_deg": 1,
    "aspect_sin": 2,
    "aspect_cos": 3,
    "profile_curvature": 4,
    "plan_curvature": 5,
    "laplacian_curvature": 6,
    "tpi_90m": 7,
    "tpi_300m": 8,
    "tpi_900m": 9,
    "local_std_90m": 10,
    "local_std_300m": 11,
    "local_relief_300m": 12,
    "local_relief_900m": 13,
    "valley_depth_900m": 14,
    "ridge_height_900m": 15,
    "ruggedness_90m": 16,
}

# Physical thresholds are fixed before looking at any separability outcome.
RIDGE_TPI_M = 5.0
VALLEY_TPI_M = -5.0
FLAT_SLOPE_DEG = 5.0
STEEP_SLOPE_DEG = 25.0
PIXEL_METRES = 10.0

PHYSICAL_FEATURES = (
    "area_px",
    "log_area",
    "mean_slope",
    "p10_slope",
    "p90_slope",
    "flat_fraction",
    "steep_fraction",
    "elev_range",
    "relative_relief",
    "aspect_coherence",
    "elongation",
    "downslope_alignment",
    "descent_consistency",
    "slope_decline",
    "divide_straddle",
    "tpi900_range",
    "mean_tpi_90m",
    "mean_tpi_300m",
    "mean_tpi_900m",
    "valley_bottom_fraction",
    "mean_valley_depth",
    "mean_ridge_height",
    "mean_ruggedness",
    "mean_local_relief_300m",
    "mean_plan_curvature",
    "mean_profile_curvature",
    "compactness",
)
CONFIDENCE_FEATURES = ("mean_probability", "max_probability", "p90_probability")


def circular_coherence(sin_values: np.ndarray, cos_values: np.ndarray) -> float:
    """Resultant length of the aspect vectors: 1 means one single slope facet."""
    mean_sin = float(np.mean(sin_values))
    mean_cos = float(np.mean(cos_values))
    return float(np.hypot(mean_sin, mean_cos))


def shape_axes(rows: np.ndarray, cols: np.ndarray) -> tuple[float, np.ndarray]:
    """Return elongation and the unit major axis from the pixel scatter matrix."""
    if rows.size < 3:
        return 1.0, np.asarray([0.0, 1.0], dtype=float)
    coords = np.stack([rows.astype(float), cols.astype(float)], axis=1)
    centred = coords - coords.mean(axis=0, keepdims=True)
    covariance = np.cov(centred, rowvar=False)
    if not np.all(np.isfinite(covariance)):
        return 1.0, np.asarray([0.0, 1.0], dtype=float)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 1e-9, None)
    major = vectors[:, order[0]]
    return float(np.sqrt(values[0] / values[1])), major / (np.linalg.norm(major) + 1e-12)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3:
        return 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std < 1e-9 or right_std < 1e-9:
        return 0.0
    value = float(np.mean((left - left.mean()) * (right - right.mean())) / (left_std * right_std))
    return value if np.isfinite(value) else 0.0


def component_features(
    rows: np.ndarray,
    cols: np.ndarray,
    terrain: np.ndarray,
    probability: np.ndarray,
    component_mask: np.ndarray,
) -> dict[str, float]:
    """Physical descriptors of one candidate landslide body."""
    area = int(rows.size)
    elevation = terrain[TERRAIN_INDEX["elevation"], rows, cols].astype(np.float32)
    slope = terrain[TERRAIN_INDEX["slope_deg"], rows, cols].astype(np.float32)
    aspect_sin = terrain[TERRAIN_INDEX["aspect_sin"], rows, cols].astype(np.float32)
    aspect_cos = terrain[TERRAIN_INDEX["aspect_cos"], rows, cols].astype(np.float32)
    tpi900 = terrain[TERRAIN_INDEX["tpi_900m"], rows, cols].astype(np.float32)

    elongation, major = shape_axes(rows, cols)

    # Aspect points downslope. In array space, row increases southward, so the
    # downslope unit vector is (-aspect_cos, aspect_sin) for (row, col).
    mean_sin = float(np.mean(aspect_sin))
    mean_cos = float(np.mean(aspect_cos))
    downslope = np.asarray([-mean_cos, mean_sin], dtype=float)
    norm = float(np.linalg.norm(downslope))
    downslope = downslope / norm if norm > 1e-9 else np.asarray([0.0, 0.0])

    centred_rows = rows.astype(float) - float(rows.mean())
    centred_cols = cols.astype(float) - float(cols.mean())
    projection = centred_rows * downslope[0] + centred_cols * downslope[1]

    ridge_fraction = float(np.mean(tpi900 > RIDGE_TPI_M))
    valley_fraction = float(np.mean(tpi900 < VALLEY_TPI_M))

    # Boundary pixels give an isoperimetric compactness proxy.
    eroded = ndimage.binary_erosion(component_mask, border_value=0)
    perimeter = float(np.count_nonzero(component_mask & ~eroded))
    compactness = (
        float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    )

    features: dict[str, float] = {
        "area_px": float(area),
        "log_area": float(np.log10(area + 1.0)),
        "mean_slope": float(np.mean(slope)),
        "p10_slope": float(np.percentile(slope, 10)),
        "p90_slope": float(np.percentile(slope, 90)),
        "flat_fraction": float(np.mean(slope < FLAT_SLOPE_DEG)),
        "steep_fraction": float(np.mean(slope > STEEP_SLOPE_DEG)),
        "elev_range": float(np.max(elevation) - np.min(elevation)),
        "relative_relief": float(
            (np.max(elevation) - np.min(elevation))
            / (np.sqrt(max(area, 1)) * PIXEL_METRES)
        ),
        "aspect_coherence": circular_coherence(aspect_sin, aspect_cos),
        "elongation": elongation,
        "downslope_alignment": float(abs(float(np.dot(major, downslope)))),
        # Real bodies lose elevation downslope, so the correlation is negative;
        # the sign flip makes larger values mean "more gravity-consistent".
        "descent_consistency": -safe_correlation(projection, elevation),
        # Source steep, deposit gentle: slope should also decline downslope.
        "slope_decline": -safe_correlation(projection, slope),
        "divide_straddle": float(min(ridge_fraction, valley_fraction)),
        "tpi900_range": float(
            np.percentile(tpi900, 95) - np.percentile(tpi900, 5)
        ),
        "mean_tpi_90m": float(np.mean(terrain[TERRAIN_INDEX["tpi_90m"], rows, cols])),
        "mean_tpi_300m": float(np.mean(terrain[TERRAIN_INDEX["tpi_300m"], rows, cols])),
        "mean_tpi_900m": float(np.mean(tpi900)),
        "valley_bottom_fraction": float(
            np.mean((slope < FLAT_SLOPE_DEG) & (tpi900 < VALLEY_TPI_M))
        ),
        "mean_valley_depth": float(
            np.mean(terrain[TERRAIN_INDEX["valley_depth_900m"], rows, cols])
        ),
        "mean_ridge_height": float(
            np.mean(terrain[TERRAIN_INDEX["ridge_height_900m"], rows, cols])
        ),
        "mean_ruggedness": float(
            np.mean(terrain[TERRAIN_INDEX["ruggedness_90m"], rows, cols])
        ),
        "mean_local_relief_300m": float(
            np.mean(terrain[TERRAIN_INDEX["local_relief_300m"], rows, cols])
        ),
        "mean_plan_curvature": float(
            np.mean(terrain[TERRAIN_INDEX["plan_curvature"], rows, cols])
        ),
        "mean_profile_curvature": float(
            np.mean(terrain[TERRAIN_INDEX["profile_curvature"], rows, cols])
        ),
        "compactness": compactness,
    }
    values = probability[rows, cols].astype(np.float32)
    features["mean_probability"] = float(np.mean(values))
    features["max_probability"] = float(np.max(values))
    features["p90_probability"] = float(np.percentile(values, 90))
    return features


def apply_terrain_condition(
    terrain: np.ndarray,
    terrain_valid: np.ndarray,
    dataset_id: np.ndarray,
    event_id: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scramble the physical evidence while leaving candidate identity untouched.

    The visual prediction, and therefore every component, is unchanged by these
    interventions; only the Terrain used to judge each component moves. Any gain that
    survives a control is not attributable to correctly aligned physics.
    """
    if condition == "aligned":
        return terrain, terrain_valid
    if condition == "zero":
        return np.zeros_like(terrain), np.zeros_like(terrain_valid)
    if condition in {"shift32", "roll64"}:
        shift = 32 if condition == "shift32" else 64
        return (
            np.roll(terrain, shift=(shift, shift), axis=(-2, -1)),
            np.roll(terrain_valid, shift=(shift, shift), axis=(-2, -1)),
        )
    if condition == "donor":
        # Same source family, different physical event, so marginal Terrain statistics
        # stay realistic while the location-specific evidence becomes wrong.
        rng = np.random.default_rng(seed)
        order = np.arange(len(terrain))
        donor = order.copy()
        for dataset in np.unique(dataset_id):
            members = np.nonzero(dataset_id == dataset)[0]
            events = event_id[members]
            for index_position, index in enumerate(members):
                candidates = members[events != events[index_position]]
                if candidates.size == 0:
                    candidates = members[members != index]
                if candidates.size == 0:
                    continue
                donor[index] = int(rng.choice(candidates))
        return terrain[donor], terrain_valid[donor]
    raise ValueError(f"unsupported terrain condition: {condition}")


def analyse_fold(
    cache_path: Path,
    *,
    threshold: float,
    min_area: int,
    condition: str = "aligned",
    seed: int = 20260725,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Build one row per predicted component and accumulate pixel-level totals."""
    with np.load(cache_path, allow_pickle=False) as handle:
        sample_id = handle["sample_id"]
        dataset_id = handle["dataset_id"]
        event_id = handle["canonical_event_id"]
        probability = handle["visual_probability"]
        target = handle["target"]
        valid = handle["valid"]
        terrain = handle["terrain"]
        terrain_valid = handle["terrain_valid"]
    terrain, terrain_valid = apply_terrain_condition(
        terrain, terrain_valid, dataset_id, event_id, condition, seed
    )

    rows_out: list[dict[str, Any]] = []
    totals = {
        "fp_pixels": 0.0,
        "tp_pixels": 0.0,
        "fn_pixels": 0.0,
        "fp_in_components": 0.0,
        "tp_in_components": 0.0,
        "components": 0.0,
        "skipped_small_components": 0.0,
        "fp_in_skipped": 0.0,
    }
    structure = ndimage.generate_binary_structure(2, 2)

    for index in range(len(sample_id)):
        keep = valid[index].astype(bool)
        truth = target[index].astype(bool) & keep
        predicted = (probability[index].astype(np.float32) >= threshold) & keep
        totals["fp_pixels"] += float(np.count_nonzero(predicted & ~truth))
        totals["tp_pixels"] += float(np.count_nonzero(predicted & truth))
        totals["fn_pixels"] += float(np.count_nonzero(~predicted & truth))
        if not predicted.any():
            continue
        labels, count = ndimage.label(predicted, structure=structure)
        if count == 0:
            continue
        objects = ndimage.find_objects(labels)
        sample_terrain = terrain[index]
        sample_probability = probability[index].astype(np.float32)
        sample_support = terrain_valid[index].astype(bool)
        for label_value in range(1, count + 1):
            window = objects[label_value - 1]
            local = labels[window] == label_value
            area = int(np.count_nonzero(local))
            local_truth = truth[window] & local
            intersection = int(np.count_nonzero(local_truth))
            if area < min_area:
                totals["skipped_small_components"] += 1.0
                totals["fp_in_skipped"] += float(area - intersection)
                continue
            rows_local, cols_local = np.nonzero(local)
            rows_global = rows_local + window[0].start
            cols_global = cols_local + window[1].start
            support_fraction = float(
                np.mean(sample_support[rows_global, cols_global])
            )
            features = component_features(
                rows_global,
                cols_global,
                sample_terrain,
                sample_probability,
                local,
            )
            purity = intersection / area
            totals["components"] += 1.0
            totals["fp_in_components"] += float(area - intersection)
            totals["tp_in_components"] += float(intersection)
            rows_out.append(
                {
                    "sample_id": str(sample_id[index]),
                    "dataset_id": str(dataset_id[index]),
                    "canonical_event_id": str(event_id[index]),
                    "component_id": int(label_value),
                    "purity": float(purity),
                    "intersection_px": intersection,
                    "false_px": int(area - intersection),
                    "terrain_support_fraction": support_fraction,
                    **features,
                }
            )
    return rows_out, totals


def addressable_mass(frame: pd.DataFrame, totals: dict[str, float], limit: float) -> dict[str, float]:
    """False-positive pixels that live inside near-pure false-positive bodies."""
    pure = frame[frame.purity <= limit]
    fp_total = max(totals["fp_pixels"], 1.0)
    tp_total = max(totals["tp_pixels"], 1.0)
    return {
        "purity_limit": limit,
        "components": int(len(pure)),
        "component_share": float(len(pure) / max(len(frame), 1)),
        "fp_pixels_addressable": float(pure.false_px.sum()),
        "fp_mass_share": float(pure.false_px.sum() / fp_total),
        "tp_pixels_at_risk": float(pure.intersection_px.sum()),
        "tp_mass_at_risk": float(pure.intersection_px.sum() / tp_total),
    }


def grouped_auc(
    frame: pd.DataFrame,
    features: list[str],
    *,
    fp_limit: float,
    tp_limit: float,
    splits: int,
    seed: int,
) -> dict[str, Any]:
    """Event-grouped out-of-fold AUC, so separability is not an in-sample artefact."""
    labelled = frame[(frame.purity <= fp_limit) | (frame.purity >= tp_limit)].copy()
    labelled["is_true_body"] = (labelled.purity >= tp_limit).astype(int)
    groups = labelled.canonical_event_id.to_numpy()
    x = labelled[features].to_numpy(dtype=float)
    y = labelled.is_true_body.to_numpy()
    if len(np.unique(y)) < 2:
        return {"status": "degenerate_single_class", "n_rows": int(len(labelled))}
    n_groups = int(len(np.unique(groups)))
    folds = int(min(splits, n_groups))
    predictions = np.full(len(labelled), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for train_index, test_index in splitter.split(x, y, groups=groups):
        if len(np.unique(y[train_index])) < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(x[train_index], y[train_index])
        predictions[test_index] = model.predict_proba(x[test_index])[:, 1]
    mask = np.isfinite(predictions)
    result = {
        "status": "ok",
        "n_rows": int(len(labelled)),
        "n_true_bodies": int(y.sum()),
        "n_false_bodies": int(len(y) - y.sum()),
        "n_event_groups": n_groups,
        "n_group_folds": folds,
        "oof_auc": float(roc_auc_score(y[mask], predictions[mask])),
        "features": list(features),
    }
    labelled = labelled.assign(oof_score=predictions)
    per_dataset = {}
    for dataset, part in labelled[mask].groupby("dataset_id"):
        if part.is_true_body.nunique() < 2:
            per_dataset[str(dataset)] = None
            continue
        per_dataset[str(dataset)] = float(
            roc_auc_score(part.is_true_body, part.oof_score)
        )
    result["oof_auc_by_dataset"] = per_dataset
    return result


def single_feature_auc(
    frame: pd.DataFrame, features: list[str], *, fp_limit: float, tp_limit: float
) -> list[dict[str, Any]]:
    labelled = frame[(frame.purity <= fp_limit) | (frame.purity >= tp_limit)]
    y = (labelled.purity >= tp_limit).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return []
    rows = []
    for name in features:
        values = labelled[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or float(np.std(values)) < 1e-12:
            continue
        auc = float(roc_auc_score(y, values))
        rows.append(
            {
                "feature": name,
                "auc": auc,
                "directed_auc": max(auc, 1.0 - auc),
                "favours": "true_body" if auc >= 0.5 else "false_body",
            }
        )
    return sorted(rows, key=lambda row: row["directed_auc"], reverse=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1/separability_v1",
    )
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--fp-purity-limit", type=float, default=0.10)
    parser.add_argument("--tp-purity-limit", type=float, default=0.50)
    parser.add_argument("--group-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--terrain-condition",
        choices=("aligned", "zero", "shift32", "roll64", "donor"),
        default="aligned",
        help="mismatch control applied to Terrain only; components never change",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    receipts = sorted(cache_dir.glob("*_oof_cache_receipt.json"))
    if not receipts:
        raise FileNotFoundError(f"no OOF cache receipts under {cache_dir}")

    all_rows: list[dict[str, Any]] = []
    totals = {
        key: 0.0
        for key in (
            "fp_pixels",
            "tp_pixels",
            "fn_pixels",
            "fp_in_components",
            "tp_in_components",
            "components",
            "skipped_small_components",
            "fp_in_skipped",
        )
    }
    fold_rows: list[dict[str, Any]] = []
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        cache_path = Path(receipt["cache_path"])
        if not cache_path.is_file():
            cache_path = cache_dir / cache_path.name
        rows, fold_totals = analyse_fold(
            cache_path,
            threshold=float(receipt["threshold"]),
            min_area=args.min_area,
            condition=args.terrain_condition,
            seed=args.seed,
        )
        for row in rows:
            row["fold_id"] = receipt["fold_id"]
        all_rows.extend(rows)
        for key, value in fold_totals.items():
            totals[key] += value
        fold_rows.append(
            {
                "fold_id": receipt["fold_id"],
                "threshold": float(receipt["threshold"]),
                "n_samples": int(receipt["n_samples"]),
                "n_components": int(fold_totals["components"]),
                **{key: value for key, value in fold_totals.items() if key != "components"},
            }
        )
        print(
            f"[fold] {receipt['fold_id']}: {int(fold_totals['components'])} components, "
            f"fp={int(fold_totals['fp_pixels'])}",
            flush=True,
        )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(outdir / "component_features.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_totals.csv", index=False)

    mass = [
        addressable_mass(frame, totals, limit)
        for limit in (0.0, 0.05, 0.10, 0.20, 0.30)
    ]
    physical = grouped_auc(
        frame,
        list(PHYSICAL_FEATURES),
        fp_limit=args.fp_purity_limit,
        tp_limit=args.tp_purity_limit,
        splits=args.group_splits,
        seed=args.seed,
    )
    with_confidence = grouped_auc(
        frame,
        list(PHYSICAL_FEATURES) + list(CONFIDENCE_FEATURES),
        fp_limit=args.fp_purity_limit,
        tp_limit=args.tp_purity_limit,
        splits=args.group_splits,
        seed=args.seed,
    )
    confidence_only = grouped_auc(
        frame,
        list(CONFIDENCE_FEATURES),
        fp_limit=args.fp_purity_limit,
        tp_limit=args.tp_purity_limit,
        splits=args.group_splits,
        seed=args.seed,
    )
    per_feature = single_feature_auc(
        frame,
        list(PHYSICAL_FEATURES) + list(CONFIDENCE_FEATURES),
        fp_limit=args.fp_purity_limit,
        tp_limit=args.tp_purity_limit,
    )
    pd.DataFrame(per_feature).to_csv(outdir / "single_feature_auc.csv", index=False)

    false_bodies = frame[frame.purity <= args.fp_purity_limit]
    true_bodies = frame[frame.purity >= args.tp_purity_limit]
    confidence = {
        "false_body_mean_probability": float(false_bodies.mean_probability.mean())
        if len(false_bodies)
        else None,
        "true_body_mean_probability": float(true_bodies.mean_probability.mean())
        if len(true_bodies)
        else None,
        "false_body_median_max_probability": float(false_bodies.max_probability.median())
        if len(false_bodies)
        else None,
        "true_body_median_max_probability": float(true_bodies.max_probability.median())
        if len(true_bodies)
        else None,
        "fp_mass_in_high_confidence_bodies": float(
            false_bodies[false_bodies.mean_probability >= 0.9].false_px.sum()
            / max(totals["fp_pixels"], 1.0)
        )
        if len(false_bodies)
        else None,
    }

    main_mass = next(item for item in mass if item["purity_limit"] == args.fp_purity_limit)
    verdict = {
        "addressable_fp_mass_pass": bool(main_mass["fp_mass_share"] >= 0.40),
        "separability_pass": bool(
            physical.get("oof_auc", 0.0) >= 0.70 if physical.get("status") == "ok" else False
        ),
        "stop_rule_triggered": bool(
            (physical.get("oof_auc", 0.0) < 0.60 if physical.get("status") == "ok" else True)
            or main_mass["fp_mass_share"] < 0.25
        ),
    }
    verdict["decision"] = (
        "GO_OBJECT_LEVEL"
        if verdict["addressable_fp_mass_pass"] and verdict["separability_pass"]
        else ("STOP" if verdict["stop_rule_triggered"] else "PARTIAL")
    )

    summary = {
        "schema_version": "pild_object_physical_separability.v1",
        "purpose": "G0 falsifiable test of the scale-matching hypothesis",
        "terrain_condition": args.terrain_condition,
        "pixel_totals": totals,
        "component_count": int(len(frame)),
        "addressable_mass": mass,
        "physical_only": physical,
        "physical_plus_confidence": with_confidence,
        "confidence_only": confidence_only,
        "confidence_placement": confidence,
        "top_features": per_feature[:12],
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )

    print("\n=== G0 verdict ===")
    print(f"components               : {len(frame)}")
    print(f"total FP pixels          : {int(totals['fp_pixels'])}")
    print(
        f"addressable FP mass      : {main_mass['fp_mass_share']:.1%} "
        f"(TP at risk {main_mass['tp_mass_at_risk']:.2%})"
    )
    if physical.get("status") == "ok":
        print(f"physical OOF AUC         : {physical['oof_auc']:.4f}")
        print(f"  by dataset             : {physical['oof_auc_by_dataset']}")
    if confidence_only.get("status") == "ok":
        print(f"confidence-only OOF AUC  : {confidence_only['oof_auc']:.4f}")
    if with_confidence.get("status") == "ok":
        print(f"physical+confidence AUC  : {with_confidence['oof_auc']:.4f}")
    print(f"FP body mean probability : {confidence['false_body_mean_probability']}")
    print(f"TP body mean probability : {confidence['true_body_mean_probability']}")
    print(f"DECISION                 : {verdict['decision']}")
    print(f"\nartifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
