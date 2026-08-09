#!/usr/bin/env python3
"""Object review under alternative unit partitions from export_pild_subobject_units_v1.

Partitions: whole, geomorphic, material, material_shuffled.
Feature arms: terrain, spectral, joint. Pixel-level TP/FP/FN mass is fixed across
partitions, so ΔIoU is directly comparable.
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
MODES = ["whole", "geomorphic", "material", "material_shuffled"]


def oof_scores(x, y, groups, n_splits, seed):
    """事件分组交叉预测的单元纯度。"""
    out = np.zeros(len(y), dtype=float)
    for train, test in GroupKFold(n_splits).split(x, y, groups=groups):
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(x[train], y[train])
        out[test] = model.predict(x[test])
    return np.clip(out, 0.0, 1.0)


def evaluate(score, frame, tp, fp, fn, base_iou):
    """Ranking quality, oracle cutoffs, and analytic deployable outcomes."""
    i_px = frame.intersection_px.to_numpy(dtype=float)
    f_px = frame.false_px.to_numpy(dtype=float)
    denom = tp + fp + fn

    order = np.argsort(score)
    curve = np.concatenate(
        [[base_iou], (tp - np.cumsum(i_px[order])) / (denom - np.cumsum(f_px[order]))]
    )
    k = int(np.argmax(curve))

    cut = base_iou / (1.0 + base_iou)
    remove = score < cut
    lost = float(i_px[remove].sum())
    cleared = float(f_px[remove].sum())
    base_err = fp + fn
    new_err = (fp - cleared) + (fn + lost)
    big = frame.area_px.to_numpy() >= 200
    purity = frame.purity.to_numpy()
    return {
        "n_units": int(len(frame)),
        "spearman": float(pd.Series(score).corr(pd.Series(purity), method="spearman")),
        "spearman_big": float(
            pd.Series(score[big]).corr(pd.Series(purity[big]), method="spearman")
        ) if big.sum() > 10 else float("nan"),
        "ranking_best_delta_iou": float(curve[k] - base_iou),
        "deployed_delta_iou": float((tp - lost) / (denom - cleared) - base_iou),
        "deployed_rer": float((base_err - new_err) / base_err),
        "deployed_n_removed": int(remove.sum()),
        "removal_precision": float(
            1.0 - (remove & (purity >= base_iou)).sum() / max(remove.sum(), 1)
        ),
        "lost_tp": lost,
        "cleared_fp": cleared,
        "oracle_delta_iou": float(
            np.max(
                np.concatenate(
                    [
                        [base_iou],
                        (tp - np.cumsum(i_px[np.argsort(purity)]))
                        / (denom - np.cumsum(f_px[np.argsort(purity)])),
                    ]
                )
            )
            - base_iou
        ),
    }


def event_macro(score, frame, base_iou):
    """逐事件重算 ΔIoU，用于事件级稳定性与 bootstrap。"""
    cut = base_iou / (1.0 + base_iou)
    work = frame.assign(_rm=score < cut)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--baseline-iou", type=float, default=0.21819164482792633)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_subobject_veto_v1",
    )
    args = parser.parse_args()
    started = time.time()

    # 整池混淆矩阵以 whole 模式为准，四种模式共享同一批预测像元
    reference = pd.read_parquet(args.units / "units_whole.parquet")
    tp = float(reference.intersection_px.sum())
    fp = float(reference.false_px.sum())
    fn = tp / args.baseline_iou - tp - fp
    base_iou = args.baseline_iou
    print(f"整池 TP={tp:,.0f} FP={fp:,.0f} FN={fn:,.0f} baseline IoU={base_iou:.5f}\n")

    rows = []
    for mode in MODES:
        frame = pd.read_parquet(args.units / f"units_{mode}.parquet").reset_index(drop=True)
        drift_tp = abs(float(frame.intersection_px.sum()) - tp) / tp
        drift_fp = abs(float(frame.false_px.sum()) - fp) / fp
        if max(drift_tp, drift_fp) > 0.02:
            raise RuntimeError(
                f"{mode}: 像素账目漂移过大 TP {drift_tp:.3%} FP {drift_fp:.3%}"
            )
        spec_cols = [c for c in frame.columns if c.startswith("spec_")]
        y = frame.purity.to_numpy(dtype=float)
        groups = frame.canonical_event_id.to_numpy()
        arms = {
            "terrain": TERRAIN + CONFIDENCE,
            "spectral": spec_cols + CONFIDENCE,
            "joint": TERRAIN + CONFIDENCE + spec_cols,
        }
        for arm, cols in arms.items():
            x = frame[cols].to_numpy(dtype=float)
            score = oof_scores(x, y, groups, args.n_splits, args.seed)
            res = evaluate(score, frame, tp, fp, fn, base_iou)
            macro = event_macro(score, frame, base_iou)
            res.update(
                {
                    "mode": mode,
                    "arm": arm,
                    "event_macro_delta_iou": float(macro.mean()),
                    "event_positive_fraction": float((macro > 0).mean()),
                    "n_events": int(len(macro)),
                }
            )
            rows.append(res)
            print(
                f"{mode:18s} {arm:9s} rho={res['spearman']:.3f} "
                f"rho_big={res['spearman_big']:.3f}  "
                f"可部署Δ={res['deployed_delta_iou']:+.5f}  RER={res['deployed_rer']:+.4f}  "
                f"精度={res['removal_precision']:.3f}  "
                f"事件宏观Δ={res['event_macro_delta_iou']:+.5f} "
                f"({res['event_positive_fraction']:.0%})  "
                f"oracle={res['oracle_delta_iou']:+.5f}"
            )
        print()

    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "arms.csv", index=False)

    def pick(mode, arm):
        return table[(table["mode"] == mode) & (table.arm == arm)].iloc[0]

    verdict = {
        "whole_terrain": float(pick("whole", "terrain").deployed_delta_iou),
        "geomorphic_joint": float(pick("geomorphic", "joint").deployed_delta_iou),
        "material_joint": float(pick("material", "joint").deployed_delta_iou),
        "material_shuffled_joint": float(pick("material_shuffled", "joint").deployed_delta_iou),
        "unit_change_gain": float(
            pick("geomorphic", "joint").deployed_delta_iou
            - pick("whole", "terrain").deployed_delta_iou
        ),
        "material_increment_over_geomorphic": float(
            pick("material", "joint").deployed_delta_iou
            - pick("geomorphic", "joint").deployed_delta_iou
        ),
        "material_minus_shuffled": float(
            pick("material", "joint").deployed_delta_iou
            - pick("material_shuffled", "joint").deployed_delta_iou
        ),
        "best_delta_iou": float(table.deployed_delta_iou.max()),
        "best_config": table.loc[table.deployed_delta_iou.idxmax(), ["mode", "arm"]].to_dict(),
        "reaches_target_delta_iou": bool(table.deployed_delta_iou.max() >= 0.03),
    }
    print("裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))

    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_subobject_veto.v1",
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
