#!/usr/bin/env python3
"""Parent-context and coarse-to-fine object review.

Modes:
    sub_with_parent  sub-object descriptors plus parent-body descriptors
    cascade          whole-body veto first, then sub-object review on survivors

Pixel accounting is unchanged so ΔIoU remains comparable to the main protocol.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_UNITS = PROJECT_ROOT / "experiments/revision2026/pild_subobject_units_v1"

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


def pooled(remove, i_px, f_px, tp, fp, fn, base_iou):
    lost = float(i_px[remove].sum())
    cleared = float(f_px[remove].sum())
    denom = tp + fp + fn
    base_err = fp + fn
    new_err = (fp - cleared) + (fn + lost)
    return {
        "n_removed": int(remove.sum()),
        "deployed_delta_iou": float((tp - lost) / (denom - cleared) - base_iou),
        "deployed_rer": float((base_err - new_err) / base_err),
        "lost_tp": lost,
        "cleared_fp": cleared,
    }


def ranking_ceiling(score, i_px, f_px, tp, fp, fn, base_iou):
    order = np.argsort(score)
    denom = tp + fp + fn
    curve = np.concatenate(
        [[base_iou], (tp - np.cumsum(i_px[order])) / (denom - np.cumsum(f_px[order]))]
    )
    return float(curve.max() - base_iou)


def event_macro(remove, frame, base_iou):
    work = frame.assign(_rm=remove)
    deltas = []
    for _, block in work.groupby("canonical_event_id"):
        e_tp = float(block.intersection_px.sum())
        e_fp = float(block.false_px.sum())
        if e_tp <= 0:
            continue
        e_fn = max(e_tp / base_iou - e_tp - e_fp, 0.0)
        lost = float(block.intersection_px[block._rm].sum())
        cleared = float(block.false_px[block._rm].sum())
        denom = e_tp + e_fp + e_fn
        deltas.append((e_tp - lost) / max(denom - cleared, 1.0) - e_tp / denom)
    return np.asarray(deltas)


def report(name, remove, score, frame, tp, fp, fn, base_iou):
    i_px = frame.intersection_px.to_numpy(dtype=float)
    f_px = frame.false_px.to_numpy(dtype=float)
    res = pooled(remove, i_px, f_px, tp, fp, fn, base_iou)
    macro = event_macro(remove, frame, base_iou)
    purity = frame.purity.to_numpy()
    res.update(
        {
            "arm": name,
            "n_units": int(len(frame)),
            "removal_precision": float(
                1.0 - (remove & (purity >= base_iou)).sum() / max(remove.sum(), 1)
            ),
            "event_macro_delta_iou": float(macro.mean()),
            "event_positive_fraction": float((macro > 0).mean()),
        }
    )
    if score is not None:
        res["spearman"] = float(pd.Series(score).corr(pd.Series(purity), method="spearman"))
        big = frame.area_px.to_numpy() >= 200
        res["spearman_big"] = float(
            pd.Series(score[big]).corr(pd.Series(purity[big]), method="spearman")
        )
        res["ranking_ceiling"] = ranking_ceiling(score, i_px, f_px, tp, fp, fn, base_iou)
    print(
        f"{name:34s} Δ={res['deployed_delta_iou']:+.5f}  RER={res['deployed_rer']:+.4f}  "
        f"精度={res['removal_precision']:.3f}  事件宏观Δ={res['event_macro_delta_iou']:+.5f} "
        f"({res['event_positive_fraction']:.0%})"
        + (
            f"  rho={res['spearman']:.3f}/{res['spearman_big']:.3f}"
            f"  上界={res['ranking_ceiling']:+.5f}"
            if score is not None
            else ""
        )
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--modes", nargs="+", default=["geomorphic", "material", "material_shuffled"])
    parser.add_argument("--baseline-iou", type=float, default=0.21819164482792633)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_hierarchical_veto_v1",
    )
    args = parser.parse_args()
    started = time.time()

    whole = pd.read_parquet(args.units / "units_whole.parquet").reset_index(drop=True)
    spec_cols = [c for c in whole.columns if c.startswith("spec_")]
    joint_cols = TERRAIN + CONFIDENCE + spec_cols

    tp = float(whole.intersection_px.sum())
    fp = float(whole.false_px.sum())
    fn = tp / args.baseline_iou - tp - fp
    base_iou = args.baseline_iou
    cut = base_iou / (1.0 + base_iou)
    print(f"整池 TP={tp:,.0f} FP={fp:,.0f} FN={fn:,.0f} baseline={base_iou:.5f} 判据 cut={cut:.5f}\n")

    rows = []
    # --- 整块基准 ---
    whole_score = oof_scores(
        whole[joint_cols].to_numpy(dtype=float),
        whole.purity.to_numpy(dtype=float),
        whole.canonical_event_id.to_numpy(),
        args.n_splits, args.seed,
    )
    whole_remove = whole_score < cut
    rows.append(report("whole_joint(基准)", whole_remove, whole_score, whole, tp, fp, fn, base_iou))
    removed_parents = set(
        map(tuple, whole.loc[whole_remove, KEYS].to_numpy().tolist())
    )
    print()

    parent_cols = {c: f"par_{c}" for c in joint_cols}
    parent_table = whole[KEYS + joint_cols].rename(columns=parent_cols)

    for mode in args.modes:
        sub = pd.read_parquet(args.units / f"units_{mode}.parquet").reset_index(drop=True)
        sub = sub.merge(parent_table, on=KEYS, how="left")
        if sub[[f"par_{c}" for c in joint_cols]].isna().all(axis=1).any():
            raise RuntimeError(f"{mode}: 存在无法匹配母体的子单元")
        sub_only = joint_cols
        with_parent = joint_cols + [f"par_{c}" for c in joint_cols]
        y = sub.purity.to_numpy(dtype=float)
        groups = sub.canonical_event_id.to_numpy()

        for label, cols in (("sub_only", sub_only), ("sub_with_parent", with_parent)):
            score = oof_scores(
                sub[cols].to_numpy(dtype=float), y, groups, args.n_splits, args.seed
            )
            remove = score < cut
            res = report(f"{mode}/{label}", remove, score, sub, tp, fp, fn, base_iou)
            res["mode"] = mode
            rows.append(res)

            # --- 级联：母体已被整块清除的，其子单元一并清除；其余按子单元判决 ---
            parent_removed = np.asarray(
                [tuple(k) in removed_parents for k in sub[KEYS].to_numpy().tolist()]
            )
            cascade = parent_removed | remove
            res_c = report(f"{mode}/{label}+cascade", cascade, None, sub, tp, fp, fn, base_iou)
            res_c["mode"] = mode
            rows.append(res_c)
        print()

    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "arms.csv", index=False)
    best = table.loc[table.deployed_delta_iou.idxmax()]
    verdict = {
        "whole_joint_baseline": float(table.iloc[0].deployed_delta_iou),
        "best_arm": str(best.arm),
        "best_delta_iou": float(best.deployed_delta_iou),
        "best_rer": float(best.deployed_rer),
        "reaches_target_delta_iou": bool(best.deployed_delta_iou >= 0.03),
    }
    print("裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))
    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_hierarchical_veto.v1",
                "evidence_status": "development: event-grouped OOF on already-opened folds",
                "baseline_iou": base_iou,
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
