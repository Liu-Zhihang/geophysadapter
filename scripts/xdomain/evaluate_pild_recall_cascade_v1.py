#!/usr/bin/env python3
"""Recall-first cascade: lower the visual operating point, then apply object review.

Keeps the visual optimum (IoU = 0.21819) as the reference baseline and reports
ΔIoU / error reduction relative to that fixed baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
FOLD_IDS = [f"source_stratified_{i}" for i in range(4)]

CONFIDENCE = ["mean_probability", "max_probability", "p90_probability"]
TERRAIN = [
    "area_px", "log_area", "mean_slope", "p10_slope", "p90_slope", "flat_fraction",
    "steep_fraction", "elev_range", "relative_relief", "aspect_coherence", "elongation",
    "downslope_alignment", "descent_consistency", "slope_decline", "divide_straddle",
    "tpi900_range", "mean_tpi_90m", "mean_tpi_300m", "mean_tpi_900m",
    "valley_bottom_fraction", "mean_valley_depth", "mean_ridge_height", "mean_ruggedness",
    "mean_local_relief_300m", "mean_plan_curvature", "mean_profile_curvature", "compactness",
]
KEYS = ["sample_id", "component_id"]

REFERENCE_IOU = 0.21819164482792633
REFERENCE_TP = 1823755.0
REFERENCE_FP = 3942700.0
REFERENCE_FN = 2592046.0


def pooled_pixel_counts(cache_dir: Path, threshold: float) -> tuple[float, float, float]:
    """在给定掩膜阈值下统计整池 TP/FP/FN，必须来自像元而非对象表。"""
    tp = fp = fn = 0.0
    for fold_id in FOLD_IDS:
        with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
            probability = handle["visual_probability"]
            target = handle["target"]
            valid = handle["valid"]
            for index in range(probability.shape[0]):
                keep = valid[index].astype(bool)
                truth = (target[index] > 0) & keep
                predicted = (probability[index].astype(np.float32) >= threshold) & keep
                tp += float(np.count_nonzero(predicted & truth))
                fp += float(np.count_nonzero(predicted & ~truth))
                fn += float(np.count_nonzero(~predicted & truth))
    return tp, fp, fn


def oof_scores(x, y, groups, n_splits, seed):
    out = np.zeros(len(y), dtype=float)
    for train, test in GroupKFold(n_splits).split(x, y, groups=groups):
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(x[train], y[train])
        out[test] = model.predict(x[test])
    return np.clip(out, 0.0, 1.0)


def outcome(remove, frame, tp, fp, fn):
    """相对视觉最优参照系报告结果。"""
    i_px = frame.intersection_px.to_numpy(dtype=float)
    f_px = frame.false_px.to_numpy(dtype=float)
    lost = float(i_px[remove].sum())
    cleared = float(f_px[remove].sum())
    new_iou = (tp - lost) / (tp + fp + fn - cleared)
    new_err = (fp - cleared) + (fn + lost)
    reference_err = REFERENCE_FP + REFERENCE_FN
    purity = frame.purity.to_numpy()
    return {
        "n_units": int(len(frame)),
        "n_removed": int(remove.sum()),
        "iou_after": float(new_iou),
        "delta_iou_vs_reference": float(new_iou - REFERENCE_IOU),
        "rer_vs_reference": float((reference_err - new_err) / reference_err),
        "lost_tp": lost,
        "cleared_fp": cleared,
        "removal_precision": float(
            1.0 - (remove & (purity >= REFERENCE_IOU)).sum() / max(remove.sum(), 1)
        ),
    }


def event_macro(remove, frame, threshold_iou):
    work = frame.assign(_rm=remove)
    deltas = []
    for _, block in work.groupby("canonical_event_id"):
        e_tp = float(block.intersection_px.sum())
        e_fp = float(block.false_px.sum())
        if e_tp <= 0:
            continue
        e_fn = max(e_tp / threshold_iou - e_tp - e_fp, 0.0)
        lost = float(block.intersection_px[block._rm].sum())
        cleared = float(block.false_px[block._rm].sum())
        denom = e_tp + e_fp + e_fn
        deltas.append((e_tp - lost) / max(denom - cleared, 1.0) - e_tp / denom)
    return np.asarray(deltas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--configs", nargs="+",
        default=["0.80:experiments/revision2026/pild_recall_units_t080",
                 "0.70:experiments/revision2026/pild_recall_units_t070"],
        help="每项形如 阈值:单元目录",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_recall_cascade_v1",
    )
    args = parser.parse_args()
    started = time.time()

    print(
        f"参照系：视觉最优 IoU={REFERENCE_IOU:.5f}，"
        f"总错误={REFERENCE_FP + REFERENCE_FN:,.0f} px\n"
    )

    rows = []
    for config in args.configs:
        threshold_text, directory = config.split(":", 1)
        threshold = float(threshold_text)
        units_dir = Path(directory)
        if not units_dir.is_absolute():
            units_dir = PROJECT_ROOT / units_dir

        tp, fp, fn = pooled_pixel_counts(args.cache, threshold)
        threshold_iou = tp / (tp + fp + fn)
        cut = REFERENCE_IOU / (1.0 + REFERENCE_IOU)
        print(
            f"=== 掩膜阈值 {threshold:.2f}：TP={tp:,.0f} FP={fp:,.0f} FN={fn:,.0f} "
            f"未审查 IoU={threshold_iou:.5f}（Δ vs 参照 {threshold_iou - REFERENCE_IOU:+.5f}）"
        )

        whole = pd.read_parquet(units_dir / "units_whole.parquet").reset_index(drop=True)
        spec_cols = [c for c in whole.columns if c.startswith("spec_")]
        joint = TERRAIN + CONFIDENCE + spec_cols

        whole_score = oof_scores(
            whole[joint].to_numpy(dtype=float),
            whole.purity.to_numpy(dtype=float),
            whole.canonical_event_id.to_numpy(),
            args.n_splits, args.seed,
        )
        whole_remove = whole_score < cut
        res = outcome(whole_remove, whole, tp, fp, fn)
        macro = event_macro(whole_remove, whole, threshold_iou)
        res.update({"threshold": threshold, "arm": "whole_joint",
                    "event_macro_delta_iou": float(macro.mean()),
                    "event_positive_fraction": float((macro > 0).mean())})
        rows.append(res)
        print(
            f"  whole_joint                 Δ={res['delta_iou_vs_reference']:+.5f}  "
            f"RER={res['rer_vs_reference']:+.4f}  精度={res['removal_precision']:.3f}  "
            f"移除 {res['n_removed']:,}/{res['n_units']:,}"
        )
        removed_parents = set(map(tuple, whole.loc[whole_remove, KEYS].to_numpy().tolist()))

        parent_table = whole[KEYS + joint].rename(columns={c: f"par_{c}" for c in joint})
        sub = pd.read_parquet(units_dir / "units_geomorphic.parquet").reset_index(drop=True)
        sub = sub.merge(parent_table, on=KEYS, how="left")
        with_parent = joint + [f"par_{c}" for c in joint]
        sub_score = oof_scores(
            sub[with_parent].to_numpy(dtype=float),
            sub.purity.to_numpy(dtype=float),
            sub.canonical_event_id.to_numpy(),
            args.n_splits, args.seed,
        )
        sub_remove = sub_score < cut
        parent_removed = np.asarray(
            [tuple(k) in removed_parents for k in sub[KEYS].to_numpy().tolist()]
        )
        for label, mask in (
            ("sub_with_parent", sub_remove),
            ("sub_with_parent+cascade", parent_removed | sub_remove),
        ):
            res = outcome(mask, sub, tp, fp, fn)
            macro = event_macro(mask, sub, threshold_iou)
            res.update({"threshold": threshold, "arm": label,
                        "event_macro_delta_iou": float(macro.mean()),
                        "event_positive_fraction": float((macro > 0).mean())})
            rows.append(res)
            print(
                f"  {label:26s} Δ={res['delta_iou_vs_reference']:+.5f}  "
                f"RER={res['rer_vs_reference']:+.4f}  精度={res['removal_precision']:.3f}  "
                f"移除 {res['n_removed']:,}/{res['n_units']:,}"
            )
        print()

    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "arms.csv", index=False)
    best = table.loc[table.delta_iou_vs_reference.idxmax()]
    verdict = {
        "best_threshold": float(best.threshold),
        "best_arm": str(best.arm),
        "best_delta_iou": float(best.delta_iou_vs_reference),
        "best_rer": float(best.rer_vs_reference),
        "reaches_target_delta_iou": bool(best.delta_iou_vs_reference >= 0.03),
    }
    print("裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))
    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_recall_cascade.v1",
                "evidence_status": "development: event-grouped OOF on already-opened folds",
                "reference_iou": REFERENCE_IOU,
                "arms": rows,
                "verdict": verdict,
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n写出 {args.outdir}")


if __name__ == "__main__":
    main()
