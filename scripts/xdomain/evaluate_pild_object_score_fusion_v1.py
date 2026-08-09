#!/usr/bin/env python3
"""G11：多种子集成与整块-子单元分数融合。

G9 结束时排序上界已贴住可部署值，说明判据形式与操作点都已榨干，只剩排序本身。
在不引入新信息源的前提下，还有两处纯统计的余量：

    集成   单模型的纯度预测带有可观的种子方差；对多个种子取平均可降低排序噪声。
    融合   整块分数与子单元分数刻画的是不同尺度的证据，硬性级联（或运算）只用到
           二者的极端组合，软融合可能优于硬阈值的并集。

融合形式一律先验固定、不看测试标签：
    hard_or       parent < cut 或 sub < cut 即移除（G9 的级联）
    mean          0.5*parent + 0.5*sub 低于 cut 即移除
    min           两者较小值低于 cut 即移除（等价于 hard_or）
    geometric     sqrt(parent*sub) 低于 cut 即移除
    parent_gated  只有当 parent 未被整块清除时才启用 sub 判决（保守级联）
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
REFERENCE_IOU = 0.21819164482792633


def ensemble_scores(x, y, groups, n_splits, seeds):
    """多种子平均的事件分组 OOF 纯度预测。"""
    total = np.zeros(len(y), dtype=float)
    for seed in seeds:
        out = np.zeros(len(y), dtype=float)
        for train, test in GroupKFold(n_splits).split(x, y, groups=groups):
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=seed,
            )
            model.fit(x[train], y[train])
            out[test] = model.predict(x[test])
        total += np.clip(out, 0.0, 1.0)
    return total / len(seeds)


def outcome(remove, frame, tp, fp, fn, base_iou):
    i_px = frame.intersection_px.to_numpy(dtype=float)
    f_px = frame.false_px.to_numpy(dtype=float)
    lost = float(i_px[remove].sum())
    cleared = float(f_px[remove].sum())
    purity = frame.purity.to_numpy()
    base_err = fp + fn
    new_err = (fp - cleared) + (fn + lost)
    return {
        "n_removed": int(remove.sum()),
        "delta_iou": float((tp - lost) / (tp + fp + fn - cleared) - base_iou),
        "rer": float((base_err - new_err) / base_err),
        "removal_precision": float(
            1.0 - (remove & (purity >= base_iou)).sum() / max(remove.sum(), 1)
        ),
        "lost_tp": lost,
        "cleared_fp": cleared,
    }


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


def bootstrap_ci(deltas, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--mode", default="geomorphic")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260725, 7, 101, 2029, 55555])
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_score_fusion_v1",
    )
    args = parser.parse_args()
    started = time.time()

    whole = pd.read_parquet(args.units / "units_whole.parquet").reset_index(drop=True)
    spec_cols = [c for c in whole.columns if c.startswith("spec_")]
    joint = TERRAIN + CONFIDENCE + spec_cols
    tp = float(whole.intersection_px.sum())
    fp = float(whole.false_px.sum())
    fn = tp / REFERENCE_IOU - tp - fp
    base_iou = REFERENCE_IOU
    cut = base_iou / (1.0 + base_iou)
    print(f"整池 TP={tp:,.0f} FP={fp:,.0f} FN={fn:,.0f}  cut={cut:.5f}  种子 {args.seeds}\n")

    rows = []
    # --- 整块：单种子 vs 集成 ---
    xw = whole[joint].to_numpy(dtype=float)
    yw = whole.purity.to_numpy(dtype=float)
    gw = whole.canonical_event_id.to_numpy()
    single = ensemble_scores(xw, yw, gw, args.n_splits, args.seeds[:1])
    ens = ensemble_scores(xw, yw, gw, args.n_splits, args.seeds)
    for name, score in (("whole_single", single), ("whole_ensemble", ens)):
        res = outcome(score < cut, whole, tp, fp, fn, base_iou)
        macro = event_macro(score < cut, whole, base_iou)
        res.update(
            {
                "arm": name,
                "spearman": float(pd.Series(score).corr(pd.Series(yw), method="spearman")),
                "event_macro_delta_iou": float(macro.mean()),
                "event_positive_fraction": float((macro > 0).mean()),
            }
        )
        rows.append(res)
        print(
            f"{name:34s} Δ={res['delta_iou']:+.5f}  RER={res['rer']:+.4f}  "
            f"rho={res['spearman']:.3f}  事件宏观Δ={res['event_macro_delta_iou']:+.5f}"
        )

    parent_removed_keys = set(map(tuple, whole.loc[ens < cut, KEYS].to_numpy().tolist()))
    parent_score_map = dict(zip(map(tuple, whole[KEYS].to_numpy().tolist()), ens))

    # --- 子单元：集成 + 各种融合 ---
    sub = pd.read_parquet(args.units / f"units_{args.mode}.parquet").reset_index(drop=True)
    parent_table = whole[KEYS + joint].rename(columns={c: f"par_{c}" for c in joint})
    sub = sub.merge(parent_table, on=KEYS, how="left")
    with_parent = joint + [f"par_{c}" for c in joint]
    ys = sub.purity.to_numpy(dtype=float)
    gs = sub.canonical_event_id.to_numpy()
    sub_score = ensemble_scores(
        sub[with_parent].to_numpy(dtype=float), ys, gs, args.n_splits, args.seeds
    )
    keys_sub = list(map(tuple, sub[KEYS].to_numpy().tolist()))
    parent_score = np.asarray([parent_score_map[k] for k in keys_sub])
    parent_removed = np.asarray([k in parent_removed_keys for k in keys_sub])

    fusions = {
        "sub_ensemble": sub_score < cut,
        "hard_or": parent_removed | (sub_score < cut),
        "mean": (0.5 * parent_score + 0.5 * sub_score) < cut,
        "geometric": np.sqrt(np.clip(parent_score * sub_score, 0.0, None)) < cut,
        "parent_gated": parent_removed | (~parent_removed & (sub_score < cut)),
    }
    print()
    for name, remove in fusions.items():
        res = outcome(remove, sub, tp, fp, fn, base_iou)
        macro = event_macro(remove, sub, base_iou)
        lo, hi = bootstrap_ci(macro)
        res.update(
            {
                "arm": f"{args.mode}/{name}",
                "event_macro_delta_iou": float(macro.mean()),
                "event_macro_ci_low": lo,
                "event_macro_ci_high": hi,
                "event_positive_fraction": float((macro > 0).mean()),
            }
        )
        rows.append(res)
        print(
            f"{args.mode}/{name:24s} Δ={res['delta_iou']:+.5f}  RER={res['rer']:+.4f}  "
            f"精度={res['removal_precision']:.3f}  "
            f"事件宏观Δ={res['event_macro_delta_iou']:+.5f} "
            f"[{lo:+.5f}, {hi:+.5f}] ({res['event_positive_fraction']:.0%})"
        )

    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "arms.csv", index=False)
    best = table.loc[table.delta_iou.idxmax()]
    verdict = {
        "best_arm": str(best.arm),
        "best_delta_iou": float(best.delta_iou),
        "best_rer": float(best.rer),
        "ensemble_gain_whole": float(
            table[table.arm == "whole_ensemble"].delta_iou.iloc[0]
            - table[table.arm == "whole_single"].delta_iou.iloc[0]
        ),
        "reaches_target_delta_iou": bool(best.delta_iou >= 0.03),
    }
    print("\n裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))
    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_score_fusion.v1",
                "evidence_status": "development: event-grouped OOF on already-opened folds",
                "seeds": args.seeds,
                "mode": args.mode,
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
