#!/usr/bin/env python3
"""G4: interpretability evidence for the object-level physical operator.

Interpretability here is not a post-hoc explanation layer. Once the decision unit matches
the native support of each input, every variable carries exactly one named function, so
the question "why was this body rejected" has a physical answer. This script produces the
evidence that the named functions really are what the model uses.

Four blocks:

1. Role-group permutation importance, grouped by physical meaning rather than by column,
   with event-grouped out-of-fold scoring so an importance cannot come from memorising an
   event.
2. Direction consistency: for each descriptor, whether the empirical relationship with
   body purity has the sign that gravity-driven failure implies.
3. Material as a threshold setter: the slope level that best separates true from false
   bodies is estimated inside Material strata. Weak, water-retentive ground should fail on
   gentler slopes, which is a statement about a threshold, not about mean IoU.
4. Trigger as an event dose: the per-event share of bodies worth removing is compared with
   the event's excess antecedent rainfall, and against the same-place wrong-time control.

Blocks 3 and 4 exist because global IoU is the wrong instrument for Material and Trigger.
Their coverage does not overlap the addressable error mass, so an averaged accuracy delta
dilutes them; their function is to calibrate the threshold and the dose, and that function
can be tested directly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from analyze_pild_object_physical_separability_v1 import (  # noqa: E402
    CONFIDENCE_FEATURES,
    PHYSICAL_FEATURES,
)
from evaluate_pild_object_role_hierarchy_v1 import (  # noqa: E402
    TRIGGER_NAMES,
    load_role_context,
)

# Grouping by physical meaning, which is the unit a reader reasons about.
ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "relief_and_slope": (
        "mean_slope",
        "p10_slope",
        "p90_slope",
        "flat_fraction",
        "steep_fraction",
        "elev_range",
        "relative_relief",
        "mean_local_relief_300m",
        "mean_ruggedness",
    ),
    "gravity_geometry": (
        "aspect_coherence",
        "downslope_alignment",
        "descent_consistency",
        "slope_decline",
        "elongation",
    ),
    "hillslope_position": (
        "divide_straddle",
        "tpi900_range",
        "mean_tpi_90m",
        "mean_tpi_300m",
        "mean_tpi_900m",
        "valley_bottom_fraction",
        "mean_valley_depth",
        "mean_ridge_height",
    ),
    "surface_curvature": (
        "mean_plan_curvature",
        "mean_profile_curvature",
    ),
    "body_shape": ("area_px", "log_area", "compactness"),
    "support_quality": ("terrain_support_fraction",),
    "visual_confidence": tuple(CONFIDENCE_FEATURES),
}

# Sign that gravity-driven failure implies for the correlation with body purity.
EXPECTED_DIRECTION: dict[str, int] = {
    "mean_slope": +1,
    "p90_slope": +1,
    "steep_fraction": +1,
    "elev_range": +1,
    "relative_relief": +1,
    "mean_local_relief_300m": +1,
    "mean_ruggedness": +1,
    "flat_fraction": -1,
    "valley_bottom_fraction": -1,
    "aspect_coherence": +1,
    "downslope_alignment": +1,
    "descent_consistency": +1,
    "slope_decline": +1,
    "divide_straddle": -1,
}


def all_features() -> list[str]:
    return list(PHYSICAL_FEATURES) + ["terrain_support_fraction"] + list(
        CONFIDENCE_FEATURES
    )


def oof_predictions(
    frame: pd.DataFrame, features: list[str], *, splits: int, seed: int
) -> tuple[np.ndarray, list[Any]]:
    x = frame[features].to_numpy(dtype=float)
    y = frame.purity.to_numpy(dtype=float)
    weight = frame.area_px.to_numpy(dtype=float)
    weight = np.clip(weight / max(float(weight.mean()), 1e-9), 0.0, 50.0)
    groups = frame.canonical_event_id.to_numpy()
    prediction = np.full(len(frame), np.nan, dtype=float)
    models: list[Any] = []
    folds: list[np.ndarray] = []
    splitter = GroupKFold(n_splits=int(min(splits, np.unique(groups).size)))
    for train_index, test_index in splitter.split(x, y, groups=groups):
        model = HistGradientBoostingRegressor(
            max_depth=4,
            max_iter=350,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(x[train_index], y[train_index], sample_weight=weight[train_index])
        prediction[test_index] = model.predict(x[test_index])
        models.append(model)
        folds.append(test_index)
    return prediction, list(zip(models, folds, strict=True))


def weighted_correlation(
    prediction: np.ndarray, truth: np.ndarray, weight: np.ndarray
) -> float:
    mask = np.isfinite(prediction)
    p, t, w = prediction[mask], truth[mask], weight[mask]
    w = w / w.sum()
    pm = float(np.sum(w * p))
    tm = float(np.sum(w * t))
    cov = float(np.sum(w * (p - pm) * (t - tm)))
    vp = float(np.sum(w * (p - pm) ** 2))
    vt = float(np.sum(w * (t - tm) ** 2))
    if vp <= 0 or vt <= 0:
        return 0.0
    return cov / np.sqrt(vp * vt)


def group_permutation_importance(
    frame: pd.DataFrame,
    features: list[str],
    fitted: list[Any],
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Permute whole physical groups inside each held-out event set."""
    x = frame[features].to_numpy(dtype=float)
    y = frame.purity.to_numpy(dtype=float)
    weight = np.clip(
        frame.area_px.to_numpy(dtype=float)
        / max(float(frame.area_px.mean()), 1e-9),
        0.0,
        50.0,
    )
    index_of = {name: position for position, name in enumerate(features)}
    rng = np.random.default_rng(seed)

    base = np.full(len(frame), np.nan, dtype=float)
    for model, test_index in fitted:
        base[test_index] = model.predict(x[test_index])
    reference = weighted_correlation(base, y, weight)

    rows: list[dict[str, Any]] = []
    for group, members in ROLE_GROUPS.items():
        columns = [index_of[name] for name in members if name in index_of]
        if not columns:
            continue
        drops = []
        for _ in range(repeats):
            permuted = np.full(len(frame), np.nan, dtype=float)
            for model, test_index in fitted:
                block = x[test_index].copy()
                order = rng.permutation(block.shape[0])
                block[:, columns] = block[np.ix_(order, columns)]
                permuted[test_index] = model.predict(block)
            drops.append(reference - weighted_correlation(permuted, y, weight))
        rows.append(
            {
                "role_group": group,
                "n_features": len(columns),
                "reference_correlation": reference,
                "mean_correlation_drop": float(np.mean(drops)),
                "std_correlation_drop": float(np.std(drops)),
                "relative_drop": float(np.mean(drops) / max(reference, 1e-9)),
            }
        )
    return sorted(rows, key=lambda row: row["mean_correlation_drop"], reverse=True)


def direction_consistency(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Compare the empirical sign against what gravity-driven failure implies."""
    truth = (frame.purity > 0.179).astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for name, expected in EXPECTED_DIRECTION.items():
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or float(np.std(values)) < 1e-12:
            continue
        auc = float(roc_auc_score(truth, values))
        observed = 1 if auc > 0.5 else -1
        rows.append(
            {
                "feature": name,
                "expected_sign": expected,
                "observed_sign": observed,
                "auc_true_body": auc,
                "directed_auc": max(auc, 1.0 - auc),
                "consistent": bool(observed == expected),
            }
        )
    return sorted(rows, key=lambda row: row["directed_auc"], reverse=True)


def material_threshold_shift(
    frame: pd.DataFrame, *, strata: int, seed: int
) -> dict[str, Any]:
    """Does weaker, more water-retentive ground lower the separating slope threshold?"""
    material_columns = [c for c in frame.columns if c.startswith("material_")]
    covered = frame[(frame.q_material > 0) & frame[material_columns].notna().all(axis=1)]
    if len(covered) < 500:
        return {"status": "insufficient_material_coverage", "n": int(len(covered))}

    # One interpretable axis: the leading component of the standardized Material block.
    values = covered[material_columns].to_numpy(dtype=float)
    centre = values.mean(axis=0)
    spread = values.std(axis=0)
    spread[spread < 1e-9] = 1.0
    standardized = np.clip((values - centre) / spread, -8.0, 8.0)
    centred = standardized - standardized.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    axis = centred @ components[0]

    quantiles = np.quantile(axis, np.linspace(0.0, 1.0, strata + 1))
    quantiles[0] -= 1e-6
    labels = np.digitize(axis, quantiles[1:-1])
    slope = covered.mean_slope.to_numpy(dtype=float)
    truth = (covered.purity.to_numpy(dtype=float) > 0.179).astype(int)
    grid = np.arange(2.0, 45.0, 0.5)

    rows = []
    for stratum in range(strata):
        mask = labels == stratum
        if mask.sum() < 100 or truth[mask].sum() < 20:
            continue
        best_threshold, best_score = float("nan"), -np.inf
        positive = truth[mask] == 1
        n_pos = int(positive.sum())
        n_neg = int((~positive).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        for candidate in grid:
            predicted = slope[mask] >= candidate
            if predicted.all() or not predicted.any():
                continue
            # Youden's J normalises within class, so a threshold cannot win simply by
            # predicting almost nothing.
            true_positive_rate = float((predicted & positive).sum() / n_pos)
            false_positive_rate = float((predicted & ~positive).sum() / n_neg)
            score = true_positive_rate - false_positive_rate
            if score > best_score:
                best_score, best_threshold = score, float(candidate)
        rows.append(
            {
                "stratum": int(stratum),
                "n": int(mask.sum()),
                "material_axis_mean": float(axis[mask].mean()),
                "true_body_rate": float(truth[mask].mean()),
                "separating_slope_threshold": best_threshold,
                "youden_j": float(best_score),
                "mean_slope": float(slope[mask].mean()),
            }
        )
    if len(rows) < 3:
        return {"status": "insufficient_strata", "rows": rows}
    axis_values = np.asarray([row["material_axis_mean"] for row in rows])
    thresholds = np.asarray([row["separating_slope_threshold"] for row in rows])
    correlation = float(np.corrcoef(axis_values, thresholds)[0, 1])
    return {
        "status": "ok",
        "strata": rows,
        "axis_threshold_correlation": correlation,
        "threshold_range_deg": float(np.nanmax(thresholds) - np.nanmin(thresholds)),
        "interpretation": (
            "a non-zero correlation means the slope level that separates true from false "
            "bodies moves with Material, which is the threshold-setting role"
        ),
    }


def trigger_dose_response(frame: pd.DataFrame) -> dict[str, Any]:
    """Do wetter events genuinely need fewer removals than drier ones?"""
    covered = frame[frame.q_trigger > 0]
    if covered.canonical_event_id.nunique() < 6:
        return {
            "status": "insufficient_trigger_coverage",
            "n_events": int(covered.canonical_event_id.nunique()),
        }
    rows = []
    for event, part in covered.groupby("canonical_event_id"):
        rows.append(
            {
                "canonical_event_id": str(event),
                "dataset_id": str(part.dataset_id.iloc[0]),
                "n_bodies": int(len(part)),
                "removal_share": float((part.purity <= 0.179).mean()),
                "excess_rain_mm": float(part["rain_d7_case_minus_wrongtime_mm"].iloc[0]),
                "wrongtime_rain_mm": float(part["rain_d7_wrongtime_median_mm"].iloc[0]),
            }
        )
    table = pd.DataFrame(rows)
    case = float(np.corrcoef(table.excess_rain_mm, table.removal_share)[0, 1])
    control = float(np.corrcoef(table.wrongtime_rain_mm, table.removal_share)[0, 1])
    return {
        "status": "ok",
        "n_events": int(len(table)),
        "case_correlation": case,
        "wrongtime_correlation": control,
        "case_minus_wrongtime": case - control,
        "expected_sign": -1,
        "case_sign_consistent": bool(case < 0),
        "beats_wrongtime_control": bool(abs(case) > abs(control)),
        "events": rows,
        "interpretation": (
            "a negative case correlation means a wetter event needs fewer removals, "
            "which is the event-dose role; the wrong-time window is the matched control"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_interpretability_v1",
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--material-strata", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    diagnostic = args.diagnostic_root.resolve()
    components = pd.read_csv(diagnostic / "separability_v1" / "component_features.csv")
    folds = sorted(components.fold_id.unique())
    context = load_role_context(args.cache_dir.resolve(), folds)
    components = components.merge(
        context, on="sample_id", how="left", validate="many_to_one"
    ).reset_index(drop=True)

    features = all_features()
    prediction, fitted = oof_predictions(
        components, features, splits=args.splits, seed=args.seed
    )
    importance = group_permutation_importance(
        components, features, fitted, repeats=args.repeats, seed=args.seed
    )
    pd.DataFrame(importance).to_csv(outdir / "role_group_importance.csv", index=False)

    direction = direction_consistency(components)
    pd.DataFrame(direction).to_csv(outdir / "direction_consistency.csv", index=False)

    material = material_threshold_shift(
        components, strata=args.material_strata, seed=args.seed
    )
    trigger = trigger_dose_response(components)

    consistent = sum(1 for row in direction if row["consistent"])
    summary = {
        "schema_version": "pild_object_interpretability.v1",
        "evidence_status": "development: event-grouped out-of-fold on already-opened folds",
        "n_components": int(len(components)),
        "oof_purity_correlation": float(
            np.corrcoef(
                prediction[np.isfinite(prediction)],
                components.purity.to_numpy()[np.isfinite(prediction)],
            )[0, 1]
        ),
        "role_group_importance": importance,
        "direction_consistency": direction,
        "direction_consistent_count": consistent,
        "direction_total": len(direction),
        "material_threshold_role": material,
        "trigger_dose_role": trigger,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )

    print("=== role-group permutation importance (event-grouped OOF) ===")
    for row in importance:
        print(
            f"  {row['role_group']:20s} drop={row['mean_correlation_drop']:+.4f} "
            f"+-{row['std_correlation_drop']:.4f}  relative={row['relative_drop']:6.1%}"
        )
    print(f"\n=== direction consistency: {consistent}/{len(direction)} ===")
    for row in direction[:10]:
        flag = "ok " if row["consistent"] else "NO "
        print(
            f"  {flag}{row['feature']:24s} expected={row['expected_sign']:+d} "
            f"observed={row['observed_sign']:+d} directed_auc={row['directed_auc']:.4f}"
        )
    print("\n=== Material as threshold setter ===")
    if material.get("status") == "ok":
        print(
            f"  axis-threshold correlation={material['axis_threshold_correlation']:+.4f} "
            f"threshold range={material['threshold_range_deg']:.1f} deg"
        )
        for row in material["strata"]:
            print(
                f"    stratum {row['stratum']}: n={row['n']:6d} axis={row['material_axis_mean']:+7.3f} "
                f"threshold={row['separating_slope_threshold']:5.1f} deg "
                f"true_rate={row['true_body_rate']:.3f}"
            )
    else:
        print(f"  {material}")
    print("\n=== Trigger as event dose ===")
    if trigger.get("status") == "ok":
        print(
            f"  events={trigger['n_events']} case_corr={trigger['case_correlation']:+.4f} "
            f"wrongtime_corr={trigger['wrongtime_correlation']:+.4f} "
            f"sign_ok={trigger['case_sign_consistent']} beats_control={trigger['beats_wrongtime_control']}"
        )
    else:
        print(f"  {trigger}")
    print(f"\nartifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
