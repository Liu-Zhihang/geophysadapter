#!/usr/bin/env python3
"""G1: deployable object-level physical veto for landslide candidates.

The decision unit is a connected component of the frozen visual prediction, not a
pixel. A component is rejected only when label-free physical evidence says it is not
a plausible gravity-driven mass movement; rejection restores the visual negative
exactly, and every retained component keeps the visual prediction bit for bit.

Exact removal criterion
-----------------------
Removing a component with ``i`` true and ``f`` false pixels changes the pooled score
to ``(TP-i)/(D-f)`` where ``D = TP+FP+FN``. That is an improvement precisely when

    i / f  <  TP / D  =  IoU_baseline

so the utility target is analytic rather than a tuned heuristic, and each component is
weighted by the pooled IoU change its removal would cause. This makes the gate optimise
captured false-positive *mass* instead of component count.

Evaluation discipline
---------------------
Nested event-grouped cross-validation: the outer loop holds out whole physical events,
the inner loop selects the decision threshold using only outer-training events. No
event contributes labels to its own decision, and no threshold is chosen on the events
it is scored on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from analyze_pild_object_physical_separability_v1 import (  # noqa: E402
    CONFIDENCE_FEATURES,
    PHYSICAL_FEATURES,
)

SUPPORT_FEATURES = ("terrain_support_fraction",)


def pooled_counts(totals: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(totals["tp_pixels"]),
        float(totals["fp_pixels"]),
        float(totals["fn_pixels"]),
    )


def removal_gain(
    frame: pd.DataFrame, *, tp: float, fp: float, fn: float
) -> np.ndarray:
    """Pooled IoU change caused by removing each component on its own."""
    denominator = tp + fp + fn
    baseline = tp / denominator
    i = frame.intersection_px.to_numpy(dtype=float)
    f = frame.false_px.to_numpy(dtype=float)
    return (tp - i) / np.clip(denominator - f, 1.0, None) - baseline


def evaluate_decision(
    frame: pd.DataFrame,
    remove: np.ndarray,
    *,
    tp: float,
    fp: float,
    fn: float,
) -> dict[str, float]:
    """Apply a rejection vector and recompute the pooled confusion counts."""
    removed = frame[remove]
    lost_tp = float(removed.intersection_px.sum())
    cleared_fp = float(removed.false_px.sum())
    new_tp = tp - lost_tp
    new_fp = fp - cleared_fp
    new_fn = fn + lost_tp
    baseline_iou = tp / (tp + fp + fn)
    adapted_iou = new_tp / max(new_tp + new_fp + new_fn, 1.0)
    baseline_errors = fp + fn
    adapted_errors = new_fp + new_fn
    return {
        "n_removed": int(remove.sum()),
        "removal_rate": float(remove.mean()) if remove.size else 0.0,
        "baseline_iou": float(baseline_iou),
        "adapted_iou": float(adapted_iou),
        "delta_iou": float(adapted_iou - baseline_iou),
        "cleared_fp": cleared_fp,
        "lost_tp": lost_tp,
        "corrected_to_harmed": float(cleared_fp / max(lost_tp, 1.0)),
        "fp_mass_captured": float(cleared_fp / max(fp, 1.0)),
        "tp_mass_lost": float(lost_tp / max(tp, 1.0)),
        "rer": float((baseline_errors - adapted_errors) / max(baseline_errors, 1.0)),
    }


def best_cut(
    score: np.ndarray,
    false_px: np.ndarray,
    intersection_px: np.ndarray,
    *,
    tp: float,
    fp: float,
    fn: float,
) -> tuple[float, float, np.ndarray]:
    """Pick the score cut that maximises pooled IoU, evaluated at every cut point.

    Sorting once and accumulating removed mass turns the threshold search into a
    single sweep, which is both exact over all achievable operating points and fast
    enough to repeat inside nested cross-validation.
    """
    denominator = tp + fp + fn
    baseline = tp / denominator
    order = np.argsort(-score, kind="stable")
    cleared = np.cumsum(false_px[order])
    lost = np.cumsum(intersection_px[order])
    curve = (tp - lost) / np.clip(denominator - cleared, 1.0, None) - baseline
    best_index = int(np.argmax(curve))
    if float(curve[best_index]) <= 0.0:
        # Abstention: a threshold above every score removes nothing.
        return float(np.nextafter(score.max(), np.inf)), 0.0, curve
    return float(score[order][best_index]), float(curve[best_index]), curve


def nested_gate_scores(
    frame: pd.DataFrame,
    features: list[str],
    *,
    tp: float,
    fp: float,
    fn: float,
    outer_splits: int,
    inner_splits: int,
    threshold_grid: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return out-of-fold scores, per-row applied thresholds and fold receipts."""
    gain = removal_gain(frame, tp=tp, fp=fp, fn=fn)
    target = (gain > 0).astype(int)
    weight = np.abs(gain)
    weight = weight / max(float(weight.mean()), 1e-12)
    x = frame[features].to_numpy(dtype=float)
    groups = frame.canonical_event_id.to_numpy()

    scores = np.full(len(frame), np.nan, dtype=float)
    applied = np.full(len(frame), np.nan, dtype=float)
    receipts: list[dict[str, Any]] = []

    unique_groups = np.unique(groups)
    outer = GroupKFold(n_splits=int(min(outer_splits, unique_groups.size)))
    for fold_index, (train_index, test_index) in enumerate(
        outer.split(x, target, groups=groups)
    ):
        train_groups = groups[train_index]
        inner_folds = int(min(inner_splits, np.unique(train_groups).size))
        inner_scores = np.full(train_index.size, np.nan, dtype=float)
        inner = GroupKFold(n_splits=inner_folds)
        for inner_train, inner_test in inner.split(
            x[train_index], target[train_index], groups=train_groups
        ):
            if len(np.unique(target[train_index][inner_train])) < 2:
                continue
            model = HistGradientBoostingClassifier(
                max_depth=4,
                max_iter=300,
                learning_rate=0.06,
                l2_regularization=1.0,
                random_state=seed + fold_index,
            )
            model.fit(
                x[train_index][inner_train],
                target[train_index][inner_train],
                sample_weight=weight[train_index][inner_train],
            )
            inner_scores[inner_test] = model.predict_proba(
                x[train_index][inner_test]
            )[:, 1]

        # Choose the threshold that maximises pooled IoU on inner predictions only.
        inner_frame = frame.iloc[train_index]
        finite = np.isfinite(inner_scores)
        if not finite.any():
            best_threshold, best_delta = 1.01, 0.0
        else:
            best_threshold, best_delta, _ = best_cut(
                inner_scores[finite],
                inner_frame.false_px.to_numpy(dtype=float)[finite],
                inner_frame.intersection_px.to_numpy(dtype=float)[finite],
                tp=tp,
                fp=fp,
                fn=fn,
            )

        final = HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=300,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed + fold_index,
        )
        final.fit(
            x[train_index], target[train_index], sample_weight=weight[train_index]
        )
        scores[test_index] = final.predict_proba(x[test_index])[:, 1]
        applied[test_index] = best_threshold
        receipts.append(
            {
                "outer_fold": fold_index,
                "n_train_components": int(train_index.size),
                "n_test_components": int(test_index.size),
                "n_train_events": int(np.unique(train_groups).size),
                "n_test_events": int(np.unique(groups[test_index]).size),
                "selected_threshold": best_threshold,
                "inner_best_delta_iou": float(best_delta),
            }
        )
    return scores, applied, receipts


def stratified_report(
    frame: pd.DataFrame,
    remove: np.ndarray,
    *,
    key: str,
    tp: float,
    fp: float,
    fn: float,
) -> list[dict[str, Any]]:
    """Per-source or per-event outcome using that stratum's own pixel totals."""
    rows: list[dict[str, Any]] = []
    for name, part in frame.groupby(key):
        mask = remove[part.index.to_numpy()]
        # Stratum-local totals: predicted-side counts are exact, FN is not available
        # per stratum from the component table, so IoU here is a component-side proxy
        # and the authoritative endpoint remains the pooled corpus number.
        local_tp = float(part.intersection_px.sum())
        local_fp = float(part.false_px.sum())
        cleared = float(part[mask].false_px.sum())
        lost = float(part[mask].intersection_px.sum())
        rows.append(
            {
                key: str(name),
                "n_components": int(len(part)),
                "n_removed": int(mask.sum()),
                "predicted_tp": local_tp,
                "predicted_fp": local_fp,
                "cleared_fp": cleared,
                "lost_tp": lost,
                "fp_mass_captured": float(cleared / max(local_fp, 1.0)),
                "tp_mass_lost": float(lost / max(local_tp, 1.0)),
                "corrected_to_harmed": float(cleared / max(lost, 1.0)),
                "net_error_reduction": cleared - lost,
            }
        )
    return rows


def condition_directory(root: Path, condition: str) -> Path:
    return root / (
        "separability_v1" if condition == "aligned" else f"separability_{condition}"
    )


def load_condition(root: Path, condition: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load one condition's component table.

    Pixel totals come from the aligned run because they are derived from the frozen
    visual prediction alone; Terrain interventions never change them.
    """
    directory = condition_directory(root, condition)
    frame = pd.read_csv(directory / "component_features.csv")
    totals_path = condition_directory(root, "aligned") / "summary.json"
    summary = json.loads(totals_path.read_text(encoding="utf-8"))
    return frame.reset_index(drop=True), summary["pixel_totals"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_veto_gate_v1",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["aligned", "zero", "shift32", "roll64", "donor"],
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--feature-set",
        choices=("physical", "physical_support", "physical_support_confidence"),
        default="physical_support",
    )
    return parser


def resolve_features(feature_set: str) -> list[str]:
    features = list(PHYSICAL_FEATURES)
    if feature_set in {"physical_support", "physical_support_confidence"}:
        features += list(SUPPORT_FEATURES)
    if feature_set == "physical_support_confidence":
        features += list(CONFIDENCE_FEATURES)
    return features


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.diagnostic_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    features = resolve_features(args.feature_set)
    threshold_grid = np.round(np.arange(0.05, 0.96, 0.025), 4)

    condition_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for condition in args.conditions:
        try:
            frame, totals = load_condition(root, condition)
        except FileNotFoundError:
            print(f"[skip] condition not available: {condition}", flush=True)
            continue
        tp, fp, fn = pooled_counts(totals)
        scores, applied, receipts = nested_gate_scores(
            frame,
            features,
            tp=tp,
            fp=fp,
            fn=fn,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            threshold_grid=threshold_grid,
            seed=args.seed,
        )
        decision = np.isfinite(scores) & np.isfinite(applied) & (scores >= applied)
        outcome = evaluate_decision(frame, decision, tp=tp, fp=fp, fn=fn)
        row = {"condition": condition, "n_components": int(len(frame)), **outcome}
        condition_rows.append(row)
        detail[condition] = {
            "fold_receipts": receipts,
            "outcome": outcome,
            "threshold_grid": [float(value) for value in threshold_grid],
        }
        if condition == "aligned":
            frame = frame.assign(gate_score=scores, gate_threshold=applied, removed=decision)
            frame.to_csv(outdir / "aligned_component_decisions.csv", index=False)
            pd.DataFrame(
                stratified_report(
                    frame, decision, key="dataset_id", tp=tp, fp=fp, fn=fn
                )
            ).to_csv(outdir / "aligned_by_dataset.csv", index=False)
            pd.DataFrame(
                stratified_report(
                    frame, decision, key="canonical_event_id", tp=tp, fp=fp, fn=fn
                )
            ).to_csv(outdir / "aligned_by_event.csv", index=False)
        print(
            f"[{condition:8s}] removed {outcome['n_removed']:6d} "
            f"dIoU={outcome['delta_iou']:+.5f} RER={outcome['rer']:+.2%} "
            f"c/h={outcome['corrected_to_harmed']:.1f} "
            f"FPmass={outcome['fp_mass_captured']:.1%} TPloss={outcome['tp_mass_lost']:.2%}",
            flush=True,
        )

    table = pd.DataFrame(condition_rows)
    table.to_csv(outdir / "condition_summary.csv", index=False)
    aligned = next((row for row in condition_rows if row["condition"] == "aligned"), None)
    controls = [row for row in condition_rows if row["condition"] != "aligned"]
    verdict: dict[str, Any] = {}
    if aligned is not None:
        best_control = max((row["delta_iou"] for row in controls), default=0.0)
        verdict = {
            "aligned_delta_iou": aligned["delta_iou"],
            "best_control_delta_iou": best_control,
            "aligned_minus_best_control": aligned["delta_iou"] - best_control,
            "gate_g1_threshold": 0.015,
            "passes_g1": bool(aligned["delta_iou"] >= 0.015),
            "beats_all_controls": bool(aligned["delta_iou"] > best_control),
            "reaches_target_delta_iou": bool(aligned["delta_iou"] >= 0.030),
            "reaches_target_rer": bool(aligned["rer"] >= 0.10),
        }

    summary = {
        "schema_version": "pild_object_veto_gate.v1",
        "evidence_status": "development: nested event-grouped cross-validation on already-opened folds",
        "decision_unit": "connected component of the frozen visual prediction",
        "removal_criterion": "analytic pooled-IoU gain (i/f < baseline IoU), weighted by that gain",
        "feature_set": args.feature_set,
        "features": features,
        "conditions": condition_rows,
        "verdict": verdict,
        "detail": detail,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )
    print("\n=== G1 verdict ===")
    for key, value in verdict.items():
        print(f"  {key} = {value}")
    print(f"\nartifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
