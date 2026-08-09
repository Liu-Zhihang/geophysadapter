"""Object-level veto headroom analysis (CPU only).

Reports (i) best global-threshold ΔIoU under the current ranking,
(ii) oracle-purity upper bound, and (iii) where wrong-removal mass concentrates.
Reads aligned_component_decisions.csv; no GPU required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DECISIONS = Path(
    "experiments/revision2026/pild_object_veto_gate_v2/aligned_component_decisions.csv"
)


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["intersection_px"] = frame["intersection_px"].astype(float)
    frame["false_px"] = frame["false_px"].astype(float)
    return frame


def pooled_iou(tp: float, fp: float, fn: float) -> float:
    """整池 IoU。移除对象时 TP 转为 FN，FP 直接消失。"""
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def base_counts(frame: pd.DataFrame, baseline_iou: float) -> tuple[float, float, float]:
    """由对象表与已知 baseline IoU 反解整池 TP/FP/FN。"""
    tp = float(frame.intersection_px.sum())
    fp = float(frame.false_px.sum())
    total = tp / baseline_iou
    fn = total - tp - fp
    return tp, fp, fn


def evaluate_removal(
    frame: pd.DataFrame, mask: np.ndarray, tp: float, fp: float, fn: float
) -> dict:
    """给定移除掩码，返回移除后的整池指标。"""
    lost_tp = float(frame.intersection_px.to_numpy()[mask].sum())
    cleared_fp = float(frame.false_px.to_numpy()[mask].sum())
    new_tp = tp - lost_tp
    new_fp = fp - cleared_fp
    new_fn = fn + lost_tp
    base = pooled_iou(tp, fp, fn)
    new = pooled_iou(new_tp, new_fp, new_fn)
    base_err = fp + fn
    new_err = new_fp + new_fn
    return {
        "n_removed": int(mask.sum()),
        "delta_iou": new - base,
        "iou": new,
        "rer": (base_err - new_err) / base_err if base_err > 0 else 0.0,
        "cleared_fp": cleared_fp,
        "lost_tp": lost_tp,
    }


def sweep_threshold(
    frame: pd.DataFrame, score: np.ndarray, tp: float, fp: float, fn: float, n_grid: int = 400
) -> pd.DataFrame:
    """按分数从低到高逐步移除，扫描最优操作点。"""
    order = np.argsort(score)
    tp_arr = frame.intersection_px.to_numpy()[order]
    fp_arr = frame.false_px.to_numpy()[order]
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)
    n = len(order)
    idx = np.unique(np.linspace(0, n, n_grid).astype(int))
    rows = []
    base = pooled_iou(tp, fp, fn)
    base_err = fp + fn
    for k in idx:
        lost = float(cum_tp[k - 1]) if k > 0 else 0.0
        cleared = float(cum_fp[k - 1]) if k > 0 else 0.0
        new = pooled_iou(tp - lost, fp - cleared, fn + lost)
        new_err = (fp - cleared) + (fn + lost)
        rows.append(
            {
                "k_removed": int(k),
                "iou": new,
                "delta_iou": new - base,
                "rer": (base_err - new_err) / base_err,
                "lost_tp": lost,
                "cleared_fp": cleared,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--baseline-iou", type=float, default=0.21819164482792633)
    args = parser.parse_args()

    frame = load(args.decisions)
    tp, fp, fn = base_counts(frame, args.baseline_iou)
    print(f"整池 TP={tp:,.0f}  FP={fp:,.0f}  FN={fn:,.0f}  baseline IoU={args.baseline_iou:.5f}")
    print(f"对象数 {len(frame):,}，其中真阳性像元为 0 的纯 FP 体 {int((frame.intersection_px == 0).sum()):,}")

    # --- 1. 当前部署决策 ---
    removed = frame.removed.to_numpy().astype(bool)
    cur = evaluate_removal(frame, removed, tp, fp, fn)
    print("\n[1] 当前部署（nested 阈值）")
    print(f"    移除 {cur['n_removed']:,}  ΔIoU={cur['delta_iou']:+.5f}  RER={cur['rer']:+.4f}")

    # --- 2. 同一排序，事后最优全局阈值（阈值选择上限）---
    score = frame.purity_hat.to_numpy()
    sweep = sweep_threshold(frame, score, tp, fp, fn)
    best = sweep.loc[sweep.delta_iou.idxmax()]
    print("\n[2] 同一 purity_hat 排序 + 事后最优截断（阈值上限，非可部署）")
    print(
        f"    移除 {int(best.k_removed):,}  ΔIoU={best.delta_iou:+.5f}  RER={best.rer:+.4f}"
    )

    # --- 3. 完美排序（oracle purity）---
    oracle_score = frame.purity.to_numpy()
    sweep_o = sweep_threshold(frame, oracle_score, tp, fp, fn)
    best_o = sweep_o.loc[sweep_o.delta_iou.idxmax()]
    print("\n[3] oracle purity 排序 + 最优截断（排序上限）")
    print(
        f"    移除 {int(best_o.k_removed):,}  ΔIoU={best_o.delta_iou:+.5f}  RER={best_o.rer:+.4f}"
    )

    # --- 4. 误删归因：当前移除中哪些是错的 ---
    wrong = removed & (frame.purity.to_numpy() >= args.baseline_iou)
    right = removed & ~wrong
    only_right = evaluate_removal(frame, right, tp, fp, fn)
    print("\n[4] 若剔除全部误删（保留正确移除）")
    print(
        f"    误删 {int(wrong.sum()):,} 个体，占移除 {wrong.sum() / max(removed.sum(), 1):.1%}，"
        f"带走 TP {frame.intersection_px.to_numpy()[wrong].sum():,.0f}"
    )
    print(
        f"    仅正确移除：ΔIoU={only_right['delta_iou']:+.5f}  RER={only_right['rer']:+.4f}"
    )

    # --- 5. 误删质量按体量分层 ---
    print("\n[5] 误删 TP 质量按对象面积分层")
    bins = [0, 10, 50, 200, 1000, 5000, np.inf]
    labels = ["<10", "10-50", "50-200", "200-1k", "1k-5k", ">5k"]
    area_bin = pd.cut(frame.area_px, bins=bins, labels=labels, right=False)
    tab = (
        pd.DataFrame(
            {
                "area_bin": area_bin,
                "wrong": wrong,
                "removed": removed,
                "tp": frame.intersection_px,
            }
        )
        .assign(wrong_tp=lambda d: d.tp * d.wrong)
        .groupby("area_bin", observed=False)
        .agg(n=("wrong", "size"), n_wrong=("wrong", "sum"), wrong_tp=("wrong_tp", "sum"))
    )
    tab["wrong_tp_share"] = tab.wrong_tp / tab.wrong_tp.sum()
    print(tab.to_string())

    # --- 6. 排序质量：按质量加权的 AUC 视角 ---
    print("\n[6] 排序质量诊断")
    beneficial = (frame.purity.to_numpy() < args.baseline_iou).astype(int)
    weight = frame.area_px.to_numpy()
    print(
        f"    未加权 Spearman(purity_hat, purity) = "
        f"{pd.Series(score).corr(pd.Series(frame.purity), method='spearman'):.4f}"
    )
    big = frame.area_px.to_numpy() >= 200
    print(
        f"    仅大体（area>=200，n={big.sum():,}）Spearman = "
        f"{pd.Series(score[big]).corr(pd.Series(frame.purity.to_numpy()[big]), method='spearman'):.4f}"
    )
    print(
        f"    大体承载 TP 质量占比 = {frame.intersection_px.to_numpy()[big].sum() / tp:.1%}，"
        f"FP 质量占比 = {frame.false_px.to_numpy()[big].sum() / fp:.1%}"
    )
    share = weight[beneficial == 1].sum() / weight.sum()
    print(f"    可移除体（purity<baseline）占总面积质量 = {share:.1%}")


if __name__ == "__main__":
    main()
