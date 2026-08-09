#!/usr/bin/env python3
"""G3: give Material and Trigger their native-scale roles at the object level.

At pixel scale these two inputs are provably uninformative. A 250 m Material cell spans
625 prediction pixels and cannot say which of them failed; a 5 km event rainfall value is
constant inside a 1.28 km patch and therefore has no within-patch variance. That is why
every previous matrix contracted them to zero, and it is a property of the decision unit
rather than of the data.

The object unit removes both obstacles, which turns the scale-matching hypothesis into a
falsifiable prediction: one candidate body sits in one Material context, so Material can
set that body's failure threshold, and one body belongs to one event, so Trigger can set
how permissive that event should be. This script tests exactly that prediction.

Roles are constrained rather than free. Material and Trigger never enter the purity model
as extra covariates, because unconstrained event-level covariates are learned as source
identity: the patch-context experiment lost 0.005 IoU that way. Instead each acts through
one named function on top of a frozen Terrain-based purity estimate.

    Material  bounded correction to predicted purity, fitted on out-of-fold residuals
              of the Terrain model, so it must add information beyond terrain geometry
    Trigger   event-level shift of the removal threshold, so a wet event keeps more
              candidate bodies and a dry event discards more

Promotion requires beating the matched mismatch control, not merely beating the base:
Material against event-shuffled Material, Trigger against the same-place wrong-time
window that its own case-crossover control provides.
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from analyze_pild_object_physical_separability_v1 import (  # noqa: E402
    CONFIDENCE_FEATURES,
    PHYSICAL_FEATURES,
)

DEFAULT_CACHE = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
DEFAULT_DIAGNOSTIC = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1"
)
TERRAIN_FEATURES = tuple(PHYSICAL_FEATURES) + ("terrain_support_fraction",) + tuple(
    CONFIDENCE_FEATURES
)
TRIGGER_NAMES = (
    "rain_d7_antecedent_case_mm",
    "rain_d7_wrongtime_median_mm",
    "rain_d7_case_minus_wrongtime_mm",
)


def load_role_context(cache_dir: Path, folds: list[str]) -> pd.DataFrame:
    """Per-sample Material and Trigger context with their availability flags."""
    frames = []
    for fold_id in folds:
        path = cache_dir / f"{fold_id}_optical_cache.npz"
        with np.load(path, allow_pickle=False) as handle:
            material = handle["material_features"]
            trigger = handle["trigger_features"]
            frame = pd.DataFrame(
                {
                    "sample_id": [str(item) for item in handle["sample_id"]],
                    "q_material": handle["q_material"].astype(float),
                    "q_trigger": handle["q_trigger"].astype(float),
                }
            )
        for index in range(material.shape[1]):
            frame[f"material_{index:02d}"] = material[:, index].astype(float)
        for index, name in enumerate(TRIGGER_NAMES[: trigger.shape[1]]):
            frame[name] = trigger[:, index].astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def pooled_gain(
    purity_hat: np.ndarray, area: np.ndarray, *, tp: float, fp: float, fn: float
) -> np.ndarray:
    denominator = tp + fp + fn
    baseline = tp / denominator
    purity_hat = np.clip(purity_hat, 0.0, 1.0)
    return (tp - purity_hat * area) / np.clip(
        denominator - (1.0 - purity_hat) * area, 1.0, None
    ) - baseline


def evaluate(
    frame: pd.DataFrame, remove: np.ndarray, *, tp: float, fp: float, fn: float
) -> dict[str, float]:
    removed = frame[remove]
    lost = float(removed.intersection_px.sum())
    cleared = float(removed.false_px.sum())
    baseline = tp / (tp + fp + fn)
    adapted = (tp - lost) / max(tp + fp + fn - cleared, 1.0)
    return {
        "n_removed": int(remove.sum()),
        "delta_iou": float(adapted - baseline),
        "rer": float((cleared - lost) / max(fp + fn, 1.0)),
        "cleared_fp": cleared,
        "lost_tp": lost,
        "corrected_to_harmed": float(cleared / max(lost, 1.0)),
        "fp_mass_captured": float(cleared / max(fp, 1.0)),
        "tp_mass_lost": float(lost / max(tp, 1.0)),
    }


def best_cut(
    score: np.ndarray,
    false_px: np.ndarray,
    intersection_px: np.ndarray,
    *,
    tp: float,
    fp: float,
    fn: float,
) -> tuple[float, float]:
    denominator = tp + fp + fn
    baseline = tp / denominator
    order = np.argsort(-score, kind="stable")
    curve = (tp - np.cumsum(intersection_px[order])) / np.clip(
        denominator - np.cumsum(false_px[order]), 1.0, None
    ) - baseline
    best = int(np.argmax(curve))
    if float(curve[best]) <= 0.0:
        return float(np.nextafter(float(score.max()), np.inf)), 0.0
    return float(score[order][best]), float(curve[best])


def shuffle_within_dataset(
    values: np.ndarray, dataset: np.ndarray, event: np.ndarray, seed: int
) -> np.ndarray:
    """Reassign each event's context to a different event of the same source family."""
    rng = np.random.default_rng(seed)
    output = values.copy()
    for name in np.unique(dataset):
        members = np.nonzero(dataset == name)[0]
        events = np.unique(event[members])
        if events.size < 2:
            continue
        donor = {}
        permuted = rng.permutation(events)
        for original, replacement in zip(events, permuted, strict=True):
            donor[original] = replacement if replacement != original else events[
                (np.where(events == original)[0][0] + 1) % events.size
            ]
        for index in members:
            source = np.nonzero(event == donor[event[index]])[0]
            if source.size:
                output[index] = values[source[0]]
    return output


def fit_material_correction(
    residual: np.ndarray,
    features: np.ndarray,
    weight: np.ndarray,
    *,
    alpha: float,
    bound: float,
) -> tuple[Ridge, float]:
    """Bounded linear correction so Material can only nudge, never dominate."""
    model = Ridge(alpha=alpha)
    model.fit(features, residual, sample_weight=weight)
    return model, float(bound)


def run_condition(
    components: pd.DataFrame,
    *,
    use_material: bool,
    use_trigger: bool,
    material_control: str,
    trigger_control: str,
    tp: float,
    fp: float,
    fn: float,
    outer_splits: int,
    inner_splits: int,
    beta_material_grid: tuple[float, ...],
    beta_trigger_grid: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    x = components[list(TERRAIN_FEATURES)].to_numpy(dtype=float)
    purity = components.purity.to_numpy(dtype=float)
    area = components.area_px.to_numpy(dtype=float)
    false_px = components.false_px.to_numpy(dtype=float)
    intersection_px = components.intersection_px.to_numpy(dtype=float)
    groups = components.canonical_event_id.to_numpy()
    datasets = components.dataset_id.to_numpy()
    weight = area / float(area.mean())

    material_columns = [c for c in components.columns if c.startswith("material_")]
    material = components[material_columns].to_numpy(dtype=float)
    q_material = components.q_material.to_numpy(dtype=float)
    if material_control == "shuffle":
        material = shuffle_within_dataset(material, datasets, groups, seed + 11)
    elif material_control == "zero":
        q_material = np.zeros_like(q_material)

    trigger_case = components["rain_d7_case_minus_wrongtime_mm"].to_numpy(dtype=float)
    if trigger_control == "wrong_time":
        # The same-place wrong-time window is the control the Trigger audit already uses.
        trigger_case = components["rain_d7_wrongtime_median_mm"].to_numpy(dtype=float)
    elif trigger_control == "shuffle":
        trigger_case = shuffle_within_dataset(
            trigger_case.reshape(-1, 1), datasets, groups, seed + 23
        ).ravel()
    q_trigger = components.q_trigger.to_numpy(dtype=float)
    if trigger_control == "zero":
        q_trigger = np.zeros_like(q_trigger)

    purity_hat = np.full(len(components), np.nan, dtype=float)
    applied = np.full(len(components), np.nan, dtype=float)
    selections: list[dict[str, Any]] = []

    unique_groups = np.unique(groups)
    outer = GroupKFold(n_splits=int(min(outer_splits, unique_groups.size)))
    for fold_index, (train_index, test_index) in enumerate(
        outer.split(x, purity, groups=groups)
    ):
        train_groups = groups[train_index]
        inner_folds = int(min(inner_splits, np.unique(train_groups).size))
        inner_purity = np.full(train_index.size, np.nan, dtype=float)
        inner = GroupKFold(n_splits=inner_folds)
        for inner_train, inner_test in inner.split(
            x[train_index], purity[train_index], groups=train_groups
        ):
            base = HistGradientBoostingRegressor(
                max_depth=4,
                max_iter=350,
                learning_rate=0.06,
                l2_regularization=1.0,
                random_state=seed + fold_index,
            )
            base.fit(
                x[train_index][inner_train],
                purity[train_index][inner_train],
                sample_weight=weight[train_index][inner_train],
            )
            inner_purity[inner_test] = base.predict(x[train_index][inner_test])

        finite = np.isfinite(inner_purity)
        inner_rows = np.nonzero(finite)[0]
        inner_global = train_index[inner_rows]

        # Material is fitted on the terrain model's own out-of-fold residual, so it can
        # only be promoted if it explains something terrain geometry missed.
        material_model = None
        material_centre = material_spread = None
        if use_material:
            residual = purity[inner_global] - inner_purity[finite]
            active = q_material[inner_global] > 0
            if active.sum() >= 200:
                material_centre = material[inner_global][active].mean(axis=0)
                material_spread = material[inner_global][active].std(axis=0)
                material_spread[material_spread < 1e-6] = 1.0
                standardized = (
                    material[inner_global][active] - material_centre
                ) / material_spread
                material_model, _ = fit_material_correction(
                    residual[active],
                    np.clip(standardized, -8.0, 8.0),
                    weight[inner_global][active],
                    alpha=50.0,
                    bound=1.0,
                )

        def adjust(rows: np.ndarray, base_purity: np.ndarray, beta: float) -> np.ndarray:
            if material_model is None or beta <= 0.0:
                return base_purity
            standardized = np.clip(
                (material[rows] - material_centre) / material_spread, -8.0, 8.0
            )
            correction = material_model.predict(standardized)
            bounded = beta * np.tanh(correction / max(beta, 1e-6))
            return np.clip(base_purity + bounded * (q_material[rows] > 0), 0.0, 1.0)

        def event_shift(rows: np.ndarray, beta: float) -> np.ndarray:
            if beta <= 0.0:
                return np.zeros(rows.size, dtype=float)
            values = trigger_case[rows]
            active = q_trigger[rows] > 0
            centre = float(np.median(values[active])) if active.any() else 0.0
            spread = float(np.std(values[active])) if active.any() else 1.0
            spread = spread if spread > 1e-6 else 1.0
            z = np.clip((values - centre) / spread, -3.0, 3.0)
            return beta * z * active

        best = {"beta_material": 0.0, "beta_trigger": 0.0, "threshold": np.inf, "inner_delta": 0.0}
        base_inner_gain = pooled_gain(
            inner_purity[finite], area[inner_global], tp=tp, fp=fp, fn=fn
        )
        gain_scale = float(np.std(base_inner_gain)) or 1.0
        for beta_m in (beta_material_grid if use_material else (0.0,)):
            adjusted = adjust(inner_global, inner_purity[finite], beta_m)
            gain = pooled_gain(adjusted, area[inner_global], tp=tp, fp=fp, fn=fn)
            for beta_r in (beta_trigger_grid if use_trigger else (0.0,)):
                score = gain - event_shift(inner_global, beta_r) * gain_scale
                cut, delta = best_cut(
                    score,
                    false_px[inner_global],
                    intersection_px[inner_global],
                    tp=tp,
                    fp=fp,
                    fn=fn,
                )
                if delta > best["inner_delta"]:
                    best = {
                        "beta_material": float(beta_m),
                        "beta_trigger": float(beta_r),
                        "threshold": float(cut),
                        "inner_delta": float(delta),
                    }

        final = HistGradientBoostingRegressor(
            max_depth=4,
            max_iter=350,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed + fold_index,
        )
        final.fit(x[train_index], purity[train_index], sample_weight=weight[train_index])
        predicted = final.predict(x[test_index])
        predicted = adjust(test_index, predicted, best["beta_material"])
        purity_hat[test_index] = predicted
        test_gain = pooled_gain(predicted, area[test_index], tp=tp, fp=fp, fn=fn)
        applied[test_index] = (
            test_gain - event_shift(test_index, best["beta_trigger"]) * gain_scale
        ) - best["threshold"]
        selections.append({"outer_fold": fold_index, **best})

    decision = np.isfinite(applied) & (applied >= 0.0)
    outcome = evaluate(components, decision, tp=tp, fp=fp, fn=fn)
    finite = np.isfinite(purity_hat)
    outcome["purity_correlation"] = float(
        np.corrcoef(purity_hat[finite], purity[finite])[0, 1]
    )
    outcome["selected_beta_material"] = float(
        np.mean([item["beta_material"] for item in selections])
    )
    outcome["selected_beta_trigger"] = float(
        np.mean([item["beta_trigger"] for item in selections])
    )
    outcome["material_abstained_folds"] = int(
        sum(1 for item in selections if item["beta_material"] == 0.0)
    )
    outcome["trigger_abstained_folds"] = int(
        sum(1 for item in selections if item["beta_trigger"] == 0.0)
    )
    outcome["selections"] = selections
    outcome["decision"] = decision
    outcome["purity_hat"] = purity_hat
    return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_role_hierarchy_v1",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    diagnostic = args.diagnostic_root.resolve()
    components = pd.read_csv(diagnostic / "separability_v1" / "component_features.csv")
    totals = json.loads(
        (diagnostic / "separability_v1" / "summary.json").read_text(encoding="utf-8")
    )["pixel_totals"]
    tp, fp, fn = (
        float(totals["tp_pixels"]),
        float(totals["fp_pixels"]),
        float(totals["fn_pixels"]),
    )
    folds = sorted(components.fold_id.unique())
    context = load_role_context(args.cache_dir.resolve(), folds)
    components = components.merge(context, on="sample_id", how="left", validate="many_to_one")
    missing = components.q_material.isna().sum()
    if missing:
        raise RuntimeError(f"{missing} components lack role context")
    components = components.reset_index(drop=True)

    coverage = {
        "n_components": int(len(components)),
        "material_available_fraction": float((components.q_material > 0).mean()),
        "trigger_available_fraction": float((components.q_trigger > 0).mean()),
        "material_available_by_dataset": components.groupby("dataset_id")
        .q_material.apply(lambda s: float((s > 0).mean()))
        .to_dict(),
        "trigger_available_by_dataset": components.groupby("dataset_id")
        .q_trigger.apply(lambda s: float((s > 0).mean()))
        .to_dict(),
    }
    print(json.dumps(coverage, indent=2, ensure_ascii=False), flush=True)

    beta_material_grid = (0.0, 0.02, 0.05, 0.10, 0.20)
    beta_trigger_grid = (0.0, 0.05, 0.10, 0.25, 0.50)
    arms = [
        ("T", False, False, "aligned", "aligned"),
        ("T_M", True, False, "aligned", "aligned"),
        ("T_R", False, True, "aligned", "aligned"),
        ("T_M_R", True, True, "aligned", "aligned"),
        ("T_M_shuffle", True, False, "shuffle", "aligned"),
        ("T_R_wrongtime", False, True, "aligned", "wrong_time"),
        ("T_M_R_controls", True, True, "shuffle", "wrong_time"),
    ]

    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for name, use_material, use_trigger, material_control, trigger_control in arms:
        outcome = run_condition(
            components,
            use_material=use_material,
            use_trigger=use_trigger,
            material_control=material_control,
            trigger_control=trigger_control,
            tp=tp,
            fp=fp,
            fn=fn,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            beta_material_grid=beta_material_grid,
            beta_trigger_grid=beta_trigger_grid,
            seed=args.seed,
        )
        decision = outcome.pop("decision")
        purity_hat = outcome.pop("purity_hat")
        selections = outcome.pop("selections")
        rows.append({"arm": name, **outcome})
        detail[name] = {"selections": selections}
        if name == "T_M_R":
            components.assign(
                purity_hat=purity_hat, removed=decision
            ).to_csv(outdir / "decisions_T_M_R.csv", index=False)
        if name == "T":
            components.assign(
                purity_hat=purity_hat, removed=decision
            ).to_csv(outdir / "decisions_T.csv", index=False)
        print(
            f"[{name:16s}] dIoU={outcome['delta_iou']:+.5f} RER={outcome['rer']:+.2%} "
            f"c/h={outcome['corrected_to_harmed']:5.1f} corr={outcome['purity_correlation']:.4f} "
            f"beta_M={outcome['selected_beta_material']:.3f} beta_R={outcome['selected_beta_trigger']:.3f} "
            f"abstain M/R={outcome['material_abstained_folds']}/{outcome['trigger_abstained_folds']}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "role_arms.csv", index=False)
    base = next(row for row in rows if row["arm"] == "T")
    lookup = {row["arm"]: row for row in rows}
    verdict = {
        "base_delta_iou": base["delta_iou"],
        "material_increment_over_T": lookup["T_M"]["delta_iou"] - base["delta_iou"],
        "trigger_increment_over_T": lookup["T_R"]["delta_iou"] - base["delta_iou"],
        "joint_increment_over_T": lookup["T_M_R"]["delta_iou"] - base["delta_iou"],
        "material_minus_shuffle": lookup["T_M"]["delta_iou"]
        - lookup["T_M_shuffle"]["delta_iou"],
        "trigger_minus_wrongtime": lookup["T_R"]["delta_iou"]
        - lookup["T_R_wrongtime"]["delta_iou"],
        "material_promoted": bool(
            lookup["T_M"]["delta_iou"] > base["delta_iou"]
            and lookup["T_M"]["delta_iou"] > lookup["T_M_shuffle"]["delta_iou"]
        ),
        "trigger_promoted": bool(
            lookup["T_R"]["delta_iou"] > base["delta_iou"]
            and lookup["T_R"]["delta_iou"] > lookup["T_R_wrongtime"]["delta_iou"]
        ),
    }
    summary = {
        "schema_version": "pild_object_role_hierarchy.v1",
        "evidence_status": "development: nested event-grouped cross-validation on already-opened folds",
        "hypothesis": "object scale is the first setting where Material and Trigger can carry information",
        "coverage": coverage,
        "arms": rows,
        "verdict": verdict,
        "detail": detail,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )
    print("\n=== G3 verdict ===")
    for key, value in verdict.items():
        print(f"  {key} = {value}")
    print(f"\nartifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
