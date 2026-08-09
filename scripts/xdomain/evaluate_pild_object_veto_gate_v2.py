#!/usr/bin/env python3
"""G1b: decision-theoretic object veto driven by predicted landslide purity.

The v1 classifier answered "is removing this body beneficial?" as a binary question,
which treats a tiny mistake and a catastrophic one alike. Its false removals were
concentrated in large, genuinely positive bodies: 4,235 wrong removals destroyed
196,635 true pixels, and applying only the correct removals would have reached
delta IoU +0.0397 instead of +0.0176.

This version predicts the purity of each candidate body, then converts the prediction
into the analytic pooled-IoU change that removal would cause. Because area is known at
inference time, the expected cost of a mistake scales with the body actually at stake,
so large bodies must clear a much higher evidential bar than small ones.

    predicted true pixels   i_hat = purity_hat * area
    predicted false pixels  f_hat = (1 - purity_hat) * area
    expected gain           (TP - i_hat) / (D - f_hat) - TP / D

The decision threshold on expected gain, optionally per source, is selected inside the
outer-training events only. Rejected bodies restore the visual prediction exactly.
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
from evaluate_pild_object_veto_gate_v1 import (  # noqa: E402
    SUPPORT_FEATURES,
    evaluate_decision,
    load_condition,
    pooled_counts,
    resolve_features,
    stratified_report,
)


def expected_gain(
    purity_hat: np.ndarray,
    area: np.ndarray,
    *,
    tp: float,
    fp: float,
    fn: float,
) -> np.ndarray:
    """Pooled IoU change implied by removing a body of known area and predicted purity."""
    denominator = tp + fp + fn
    baseline = tp / denominator
    purity_hat = np.clip(purity_hat, 0.0, 1.0)
    true_hat = purity_hat * area
    false_hat = (1.0 - purity_hat) * area
    return (tp - true_hat) / np.clip(denominator - false_hat, 1.0, None) - baseline


def sweep_best_threshold(
    score: np.ndarray,
    false_px: np.ndarray,
    intersection_px: np.ndarray,
    *,
    tp: float,
    fp: float,
    fn: float,
) -> tuple[float, float]:
    """Exact best cut on a ranking score, evaluated at every achievable cut point."""
    denominator = tp + fp + fn
    baseline = tp / denominator
    order = np.argsort(-score, kind="stable")
    cleared = np.cumsum(false_px[order])
    lost = np.cumsum(intersection_px[order])
    curve = (tp - lost) / np.clip(denominator - cleared, 1.0, None) - baseline
    best = int(np.argmax(curve))
    if float(curve[best]) <= 0.0:
        return float(np.nextafter(score.max(), np.inf)), 0.0
    return float(score[order][best]), float(curve[best])


def fit_purity_model(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, seed: int
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=4,
        max_iter=350,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=weight)
    return model


def nested_purity_gate(
    frame: pd.DataFrame,
    features: list[str],
    *,
    tp: float,
    fp: float,
    fn: float,
    outer_splits: int,
    inner_splits: int,
    per_dataset: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x = frame[features].to_numpy(dtype=float)
    purity = frame.purity.to_numpy(dtype=float)
    area = frame.area_px.to_numpy(dtype=float)
    false_px = frame.false_px.to_numpy(dtype=float)
    intersection_px = frame.intersection_px.to_numpy(dtype=float)
    groups = frame.canonical_event_id.to_numpy()
    datasets = frame.dataset_id.to_numpy()
    # Mass-aware weighting: a body's influence on the corpus score scales with its area.
    weight = area / float(area.mean())

    purity_hat = np.full(len(frame), np.nan, dtype=float)
    gain_hat = np.full(len(frame), np.nan, dtype=float)
    applied = np.full(len(frame), np.nan, dtype=float)
    receipts: list[dict[str, Any]] = []

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
            model = fit_purity_model(
                x[train_index][inner_train],
                purity[train_index][inner_train],
                weight[train_index][inner_train],
                seed + fold_index,
            )
            inner_purity[inner_test] = model.predict(x[train_index][inner_test])

        finite = np.isfinite(inner_purity)
        inner_gain = expected_gain(
            inner_purity[finite], area[train_index][finite], tp=tp, fp=fp, fn=fn
        )
        thresholds: dict[str, float] = {}
        if per_dataset:
            inner_datasets = datasets[train_index][finite]
            for name in np.unique(datasets):
                members = inner_datasets == name
                if members.sum() < 20:
                    thresholds[str(name)] = float("inf")
                    continue
                cut, _ = sweep_best_threshold(
                    inner_gain[members],
                    false_px[train_index][finite][members],
                    intersection_px[train_index][finite][members],
                    tp=tp,
                    fp=fp,
                    fn=fn,
                )
                thresholds[str(name)] = cut
        else:
            cut, _ = sweep_best_threshold(
                inner_gain,
                false_px[train_index][finite],
                intersection_px[train_index][finite],
                tp=tp,
                fp=fp,
                fn=fn,
            )
            thresholds = {"__global__": cut}

        final = fit_purity_model(
            x[train_index], purity[train_index], weight[train_index], seed + fold_index
        )
        predicted = final.predict(x[test_index])
        purity_hat[test_index] = predicted
        gain_hat[test_index] = expected_gain(
            predicted, area[test_index], tp=tp, fp=fp, fn=fn
        )
        if per_dataset:
            applied[test_index] = [
                thresholds.get(str(name), float("inf")) for name in datasets[test_index]
            ]
        else:
            applied[test_index] = thresholds["__global__"]
        receipts.append(
            {
                "outer_fold": fold_index,
                "n_train_components": int(train_index.size),
                "n_test_components": int(test_index.size),
                "n_train_events": int(np.unique(train_groups).size),
                "n_test_events": int(np.unique(groups[test_index]).size),
                "selected_thresholds": thresholds,
            }
        )
    return purity_hat, gain_hat, applied, receipts


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
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_veto_gate_v2",
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
        default="physical_support_confidence",
    )
    parser.add_argument(
        "--per-dataset-threshold",
        action="store_true",
        help="select one expected-gain threshold per source family on inner events",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.diagnostic_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    features = resolve_features(args.feature_set)

    condition_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for condition in args.conditions:
        directory = root / (
            "separability_v1" if condition == "aligned" else f"separability_{condition}"
        )
        if not (directory / "component_features.csv").is_file():
            print(f"[skip] condition unavailable: {condition}", flush=True)
            continue
        frame, totals = load_condition(root, condition)
        tp, fp, fn = pooled_counts(totals)
        purity_hat, gain_hat, applied, receipts = nested_purity_gate(
            frame,
            features,
            tp=tp,
            fp=fp,
            fn=fn,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            per_dataset=args.per_dataset_threshold,
            seed=args.seed,
        )
        decision = (
            np.isfinite(gain_hat) & np.isfinite(applied) & (gain_hat >= applied)
        )
        outcome = evaluate_decision(frame, decision, tp=tp, fp=fp, fn=fn)
        correct = decision & (frame.purity.to_numpy() <= 0.179)
        wrong = decision & (frame.purity.to_numpy() > 0.179)
        outcome["removal_precision"] = float(
            correct.sum() / max(decision.sum(), 1)
        )
        outcome["wrong_removals"] = int(wrong.sum())
        outcome["tp_lost_to_wrong_removals"] = float(
            frame[wrong].intersection_px.sum()
        )
        row = {"condition": condition, "n_components": int(len(frame)), **outcome}
        condition_rows.append(row)
        detail[condition] = {"fold_receipts": receipts, "outcome": outcome}

        if condition == "aligned":
            decided = frame.assign(
                purity_hat=purity_hat,
                expected_gain=gain_hat,
                gate_threshold=applied,
                removed=decision,
            )
            decided.to_csv(outdir / "aligned_component_decisions.csv", index=False)
            pd.DataFrame(
                stratified_report(
                    decided, decision, key="dataset_id", tp=tp, fp=fp, fn=fn
                )
            ).to_csv(outdir / "aligned_by_dataset.csv", index=False)
            pd.DataFrame(
                stratified_report(
                    decided, decision, key="canonical_event_id", tp=tp, fp=fp, fn=fn
                )
            ).to_csv(outdir / "aligned_by_event.csv", index=False)

        print(
            f"[{condition:8s}] removed {outcome['n_removed']:6d} "
            f"dIoU={outcome['delta_iou']:+.5f} RER={outcome['rer']:+.2%} "
            f"c/h={outcome['corrected_to_harmed']:6.1f} "
            f"FPmass={outcome['fp_mass_captured']:.1%} TPloss={outcome['tp_mass_lost']:.2%} "
            f"precision={outcome['removal_precision']:.3f}",
            flush=True,
        )

    pd.DataFrame(condition_rows).to_csv(outdir / "condition_summary.csv", index=False)
    aligned = next((row for row in condition_rows if row["condition"] == "aligned"), None)
    controls = [row for row in condition_rows if row["condition"] != "aligned"]
    verdict: dict[str, Any] = {}
    if aligned is not None:
        best_control = max((row["delta_iou"] for row in controls), default=0.0)
        verdict = {
            "aligned_delta_iou": aligned["delta_iou"],
            "aligned_rer": aligned["rer"],
            "best_control_delta_iou": best_control,
            "aligned_minus_best_control": aligned["delta_iou"] - best_control,
            "beats_all_controls": bool(aligned["delta_iou"] > best_control),
            "reaches_target_delta_iou": bool(aligned["delta_iou"] >= 0.030),
            "reaches_target_rer": bool(aligned["rer"] >= 0.10),
        }

    summary = {
        "schema_version": "pild_object_veto_gate.v2",
        "evidence_status": "development: nested event-grouped cross-validation on already-opened folds",
        "decision_rule": "remove when expected pooled-IoU gain from predicted purity exceeds an inner-selected threshold",
        "per_dataset_threshold": bool(args.per_dataset_threshold),
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
    print("\n=== G1b verdict ===")
    for key, value in verdict.items():
        print(f"  {key} = {value}")
    print(f"\nartifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
