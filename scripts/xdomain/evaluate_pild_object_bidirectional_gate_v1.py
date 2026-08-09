#!/usr/bin/env python3
"""G6b: bidirectional object correction from a single purity criterion.

Veto and rescue are the same decision seen from two sides. For a body holding ``i`` true
and ``f`` false pixels the pooled score becomes ``(TP-i)/(D-f)`` if it is removed and
``(TP+i)/(D+f)`` if it is promoted, so both are governed by one comparison:

    remove  when  i / f  <  IoU_baseline
    promote when  i / f  >  IoU_baseline

One purity model therefore drives both directions, and the operator stays interpretable:
bodies that do not look like gravity-driven mass movements are discarded, bodies that do
but sit just under the visual threshold are recovered. Rejected proposals in either
direction restore the frozen visual prediction exactly.

Thresholds, the rescue probability level and both decision cuts are selected inside the
outer-training events only, and the Terrain mismatch controls apply unchanged.
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
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from analyze_pild_object_physical_separability_v1 import (  # noqa: E402
    CONFIDENCE_FEATURES,
    PHYSICAL_FEATURES,
)

VETO_FEATURES = tuple(PHYSICAL_FEATURES) + ("terrain_support_fraction",) + tuple(
    CONFIDENCE_FEATURES
)
# A landslide is spatially contiguous, so a sub-threshold body that touches a detection is
# usually the missing tail of that same failure: 80 percent of the recoverable area sits in
# touching bodies, and their useful rate is roughly twice that of isolated ones.
ADJACENCY_FEATURES = (
    "touches_detection",
    "contact_pixels",
    "contact_fraction",
    "log_neighbour_area",
    "distance_to_detection",
    "neighbour_mean_probability",
)
RESCUE_FEATURES = (
    tuple(PHYSICAL_FEATURES) + tuple(CONFIDENCE_FEATURES) + ADJACENCY_FEATURES
)


def prepare_veto(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["true_px"] = out.intersection_px
    out["false_px_body"] = out.false_px
    return out


def prepare_rescue(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = frame[frame.rescue_threshold == threshold].copy()
    out["true_px"] = out.recovered_px
    out["false_px_body"] = out.added_false_px
    return out


def fit_purity(
    frame: pd.DataFrame, features: list[str], seed: int
) -> HistGradientBoostingRegressor:
    x = frame[features].to_numpy(dtype=float)
    y = frame.purity.to_numpy(dtype=float)
    weight = frame.area_px.to_numpy(dtype=float)
    weight = weight / max(float(weight.mean()), 1e-9)
    model = HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=350,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=np.clip(weight, 0.0, 50.0))
    return model


def apply_counts(
    veto: pd.DataFrame,
    veto_remove: np.ndarray,
    rescue: pd.DataFrame,
    rescue_promote: np.ndarray,
    *,
    tp: float,
    fp: float,
    fn: float,
) -> dict[str, float]:
    lost = float(veto[veto_remove].true_px.sum())
    cleared = float(veto[veto_remove].false_px_body.sum())
    gained = float(rescue[rescue_promote].true_px.sum())
    added = float(rescue[rescue_promote].false_px_body.sum())
    new_tp = tp - lost + gained
    new_fp = fp - cleared + added
    new_fn = fn + lost - gained
    baseline = tp / (tp + fp + fn)
    adapted = new_tp / max(new_tp + new_fp + new_fn, 1.0)
    return {
        "n_removed": int(veto_remove.sum()),
        "n_promoted": int(rescue_promote.sum()),
        "cleared_fp": cleared,
        "lost_tp": lost,
        "recovered_tp": gained,
        "added_fp": added,
        "baseline_iou": float(baseline),
        "adapted_iou": float(adapted),
        "delta_iou": float(adapted - baseline),
        "rer": float(((fp + fn) - (new_fp + new_fn)) / max(fp + fn, 1.0)),
        "corrected_to_harmed": float((cleared + gained) / max(lost + added, 1.0)),
        "fp_mass_captured": float(cleared / max(fp, 1.0)),
        "fn_mass_recovered": float(gained / max(fn, 1.0)),
    }


def sweep_two_sided(
    veto_score: np.ndarray,
    veto_true: np.ndarray,
    veto_false: np.ndarray,
    rescue_score: np.ndarray,
    rescue_true: np.ndarray,
    rescue_false: np.ndarray,
    *,
    tp: float,
    fp: float,
    fn: float,
) -> tuple[float, float, float]:
    """Choose both cuts by sweeping each side on its own accumulated mass."""
    denominator = tp + fp + fn
    baseline = tp / denominator

    order = np.argsort(veto_score, kind="stable")  # lowest predicted purity first
    lost = np.cumsum(veto_true[order])
    cleared = np.cumsum(veto_false[order])
    veto_curve = (tp - lost) / np.clip(denominator - cleared, 1.0, None) - baseline
    veto_best = int(np.argmax(veto_curve)) if veto_curve.size else -1
    if veto_best < 0 or veto_curve[veto_best] <= 0:
        veto_cut, veto_gain, veto_lost, veto_cleared = -np.inf, 0.0, 0.0, 0.0
    else:
        veto_cut = float(veto_score[order][veto_best])
        veto_gain = float(veto_curve[veto_best])
        veto_lost = float(lost[veto_best])
        veto_cleared = float(cleared[veto_best])

    # Rescue is evaluated on the corpus already corrected by the veto side.
    tp_mid = tp - veto_lost
    fp_mid = fp - veto_cleared
    fn_mid = fn + veto_lost
    mid_denominator = tp_mid + fp_mid + fn_mid
    mid_baseline = tp_mid / max(mid_denominator, 1.0)

    order_r = np.argsort(-rescue_score, kind="stable")  # highest purity first
    gained = np.cumsum(rescue_true[order_r])
    added = np.cumsum(rescue_false[order_r])
    rescue_curve = (tp_mid + gained) / np.clip(
        mid_denominator + added, 1.0, None
    ) - mid_baseline
    rescue_best = int(np.argmax(rescue_curve)) if rescue_curve.size else -1
    if rescue_best < 0 or rescue_curve[rescue_best] <= 0:
        rescue_cut = np.inf
        total = veto_gain
    else:
        rescue_cut = float(rescue_score[order_r][rescue_best])
        total = veto_gain + float(rescue_curve[rescue_best])
    return veto_cut, rescue_cut, total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1",
    )
    parser.add_argument(
        "--rescue-root",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_rescue_capacity_v2",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_bidirectional_gate_v1",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["aligned", "zero", "shift32", "roll64", "donor"]
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--rescue-thresholds", type=float, nargs="+", default=[0.5, 0.3, 0.15]
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def run_condition(
    veto_all: pd.DataFrame,
    rescue_all: pd.DataFrame,
    *,
    tp: float,
    fp: float,
    fn: float,
    rescue_thresholds: list[float],
    outer_splits: int,
    inner_splits: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, dict[float, np.ndarray]]:
    veto_features = list(VETO_FEATURES)
    rescue_features = list(RESCUE_FEATURES)
    groups = veto_all.canonical_event_id.to_numpy()
    unique_groups = np.unique(groups)
    outer = GroupKFold(n_splits=int(min(outer_splits, unique_groups.size)))

    veto_decision = np.zeros(len(veto_all), dtype=bool)
    rescue_decisions = {
        threshold: np.zeros(int((rescue_all.rescue_threshold == threshold).sum()), dtype=bool)
        for threshold in rescue_thresholds
    }
    rescue_index = {
        threshold: np.nonzero((rescue_all.rescue_threshold == threshold).to_numpy())[0]
        for threshold in rescue_thresholds
    }
    receipts: list[dict[str, Any]] = []
    chosen_threshold_per_fold: list[float] = []

    for fold_index, (train_rows, test_rows) in enumerate(
        outer.split(veto_all[veto_features], veto_all.purity, groups=groups)
    ):
        train_events = set(np.unique(groups[train_rows]).tolist())
        test_events = set(np.unique(groups[test_rows]).tolist())
        veto_train = veto_all.iloc[train_rows]
        veto_test = veto_all.iloc[test_rows]

        inner_groups = groups[train_rows]
        inner_folds = int(min(inner_splits, len(train_events)))
        inner_purity = np.full(train_rows.size, np.nan, dtype=float)
        inner = GroupKFold(n_splits=inner_folds)
        for inner_train, inner_test in inner.split(
            veto_train[veto_features], veto_train.purity, groups=inner_groups
        ):
            model = fit_purity(veto_train.iloc[inner_train], veto_features, seed + fold_index)
            inner_purity[inner_test] = model.predict(
                veto_train.iloc[inner_test][veto_features].to_numpy(dtype=float)
            )

        best = {"threshold": rescue_thresholds[0], "total": -np.inf}
        for rescue_threshold in rescue_thresholds:
            pool = rescue_all[rescue_all.rescue_threshold == rescue_threshold]
            pool_train = pool[pool.canonical_event_id.isin(train_events)]
            if len(pool_train) < 200:
                continue
            inner_rescue = np.full(len(pool_train), np.nan, dtype=float)
            pool_groups = pool_train.canonical_event_id.to_numpy()
            pool_inner = GroupKFold(
                n_splits=int(min(inner_splits, np.unique(pool_groups).size))
            )
            for inner_train, inner_test in pool_inner.split(
                pool_train[rescue_features], pool_train.purity, groups=pool_groups
            ):
                model = fit_purity(
                    pool_train.iloc[inner_train], rescue_features, seed + fold_index
                )
                inner_rescue[inner_test] = model.predict(
                    pool_train.iloc[inner_test][rescue_features].to_numpy(dtype=float)
                )
            finite_v = np.isfinite(inner_purity)
            finite_r = np.isfinite(inner_rescue)
            veto_cut, rescue_cut, total = sweep_two_sided(
                inner_purity[finite_v],
                veto_train.true_px.to_numpy(dtype=float)[finite_v],
                veto_train.false_px_body.to_numpy(dtype=float)[finite_v],
                inner_rescue[finite_r],
                pool_train.true_px.to_numpy(dtype=float)[finite_r],
                pool_train.false_px_body.to_numpy(dtype=float)[finite_r],
                tp=tp,
                fp=fp,
                fn=fn,
            )
            if total > best["total"]:
                best = {
                    "threshold": float(rescue_threshold),
                    "veto_cut": float(veto_cut),
                    "rescue_cut": float(rescue_cut),
                    "total": float(total),
                }
        chosen_threshold_per_fold.append(best["threshold"])

        veto_model = fit_purity(veto_train, veto_features, seed + fold_index)
        veto_prediction = veto_model.predict(
            veto_test[veto_features].to_numpy(dtype=float)
        )
        veto_decision[test_rows] = veto_prediction <= best.get("veto_cut", -np.inf)

        pool = rescue_all[rescue_all.rescue_threshold == best["threshold"]]
        pool_train = pool[pool.canonical_event_id.isin(train_events)]
        pool_test_mask = pool.canonical_event_id.isin(test_events).to_numpy()
        if pool_test_mask.any() and len(pool_train) >= 200:
            rescue_model = fit_purity(pool_train, rescue_features, seed + fold_index)
            prediction = rescue_model.predict(
                pool[pool_test_mask][rescue_features].to_numpy(dtype=float)
            )
            local = np.zeros(len(pool), dtype=bool)
            local[np.nonzero(pool_test_mask)[0]] = prediction >= best.get(
                "rescue_cut", np.inf
            )
            rescue_decisions[best["threshold"]] |= local
        receipts.append({"outer_fold": fold_index, **best})

    promote = np.zeros(len(rescue_all), dtype=bool)
    for threshold, local in rescue_decisions.items():
        promote[rescue_index[threshold]] = local
    outcome = apply_counts(
        veto_all, veto_decision, rescue_all, promote, tp=tp, fp=fp, fn=fn
    )
    outcome["rescue_thresholds_selected"] = chosen_threshold_per_fold
    outcome["receipts"] = receipts
    return outcome, veto_decision, {"promote": promote}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rescue_raw = pd.read_csv(args.rescue_root.resolve() / "rescue_candidates.csv")
    totals = json.loads(
        (args.diagnostic_root.resolve() / "separability_v1" / "summary.json").read_text(
            encoding="utf-8"
        )
    )["pixel_totals"]
    tp, fp, fn = (
        float(totals["tp_pixels"]),
        float(totals["fp_pixels"]),
        float(totals["fn_pixels"]),
    )

    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        directory = args.diagnostic_root.resolve() / (
            "separability_v1" if condition == "aligned" else f"separability_{condition}"
        )
        path = directory / "component_features.csv"
        if not path.is_file():
            print(f"[skip] {condition}", flush=True)
            continue
        veto_all = prepare_veto(pd.read_csv(path)).reset_index(drop=True)
        # Rescue candidates are derived from the frozen prediction, so the Terrain
        # condition only changes the physical descriptors of a body, never its extent.
        rescue_all = pd.concat(
            [prepare_rescue(rescue_raw, threshold) for threshold in args.rescue_thresholds],
            ignore_index=True,
        )
        outcome, _, _ = run_condition(
            veto_all,
            rescue_all,
            tp=tp,
            fp=fp,
            fn=fn,
            rescue_thresholds=list(args.rescue_thresholds),
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            seed=args.seed,
        )
        receipts = outcome.pop("receipts")
        rows.append({"condition": condition, **outcome})
        print(
            f"[{condition:8s}] removed={outcome['n_removed']:6d} promoted={outcome['n_promoted']:6d} "
            f"dIoU={outcome['delta_iou']:+.5f} RER={outcome['rer']:+.2%} "
            f"c/h={outcome['corrected_to_harmed']:5.2f} "
            f"FPcap={outcome['fp_mass_captured']:.1%} FNrec={outcome['fn_mass_recovered']:.1%}",
            flush=True,
        )
        if condition == "aligned":
            (outdir / "aligned_receipts.json").write_text(
                json.dumps(receipts, indent=2, default=float) + "\n", encoding="utf-8"
            )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "condition_summary.csv", index=False)
    aligned = next((row for row in rows if row["condition"] == "aligned"), None)
    controls = [row for row in rows if row["condition"] != "aligned"]
    verdict: dict[str, Any] = {}
    if aligned is not None:
        best_control = max((row["delta_iou"] for row in controls), default=0.0)
        verdict = {
            "aligned_delta_iou": aligned["delta_iou"],
            "aligned_rer": aligned["rer"],
            "best_control_delta_iou": best_control,
            "aligned_minus_best_control": aligned["delta_iou"] - best_control,
            "reaches_target_delta_iou": bool(aligned["delta_iou"] >= 0.030),
            "reaches_target_rer": bool(aligned["rer"] >= 0.10),
        }
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_bidirectional_gate.v1",
                "evidence_status": "development: nested event-grouped cross-validation on already-opened folds",
                "decision_rule": "single purity criterion; remove below and promote above the baseline IoU odds",
                "conditions": rows,
                "verdict": verdict,
                "elapsed_seconds": round(time.time() - started, 2),
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n=== G6b verdict ===")
    for key, value in verdict.items():
        print(f"  {key} = {value}")
    print(f"\nartifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
