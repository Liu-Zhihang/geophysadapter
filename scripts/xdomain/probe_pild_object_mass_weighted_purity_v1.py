"""探针：对象级 veto 的决策论加权重排。

动机：整池 IoU 对每个对象的敏感度不是等权的。移除一个对象带来的 IoU 变化量为
    gain ≈ (IoU_base * f - i) / D
其中 i 为该对象内真阳性像元数，f 为假阳性像元数，D 为整池 (TP+FP+FN)。
因此对象在目标函数中的权重是 |IoU_base * f - i|，而不是 1。
当前 purity 回归对所有对象等权训练，导致模型把容量花在数万个几像元的小体上，
而 92% 的误删损失来自 50-5000 px 的中大体。

本探针在不改特征、不改折划分的前提下，对比：
    A 无权重 purity 回归（复现当前口径）
    B 决策论质量加权 purity 回归
    C 决策论质量加权的“移除是否有益”二分类
    D 同 C，但只在大体上加权（稳健性检查）

评价一律用解析判据 i/f < IoU_base 折算的整池 ΔIoU / RER，事件分组沿用原 fold_id。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

DEFAULT_DECISIONS = Path(
    "experiments/revision2026/pild_object_veto_gate_v2/aligned_component_decisions.csv"
)

# 与 veto_gate v2 保持一致的特征集合
FEATURES = [
    "area_px", "log_area", "mean_slope", "p10_slope", "p90_slope", "flat_fraction",
    "steep_fraction", "elev_range", "relative_relief", "aspect_coherence", "elongation",
    "downslope_alignment", "descent_consistency", "slope_decline", "divide_straddle",
    "tpi900_range", "mean_tpi_90m", "mean_tpi_300m", "mean_tpi_900m",
    "valley_bottom_fraction", "mean_valley_depth", "mean_ridge_height", "mean_ruggedness",
    "mean_local_relief_300m", "mean_plan_curvature", "mean_profile_curvature", "compactness",
    "terrain_support_fraction", "mean_probability", "max_probability", "p90_probability",
]


def pooled_metrics(frame: pd.DataFrame, remove: np.ndarray, tp: float, fp: float, fn: float) -> dict:
    """把对象级移除动作折算回整池像素混淆矩阵。"""
    lost_tp = float(frame.intersection_px.to_numpy()[remove].sum())
    cleared_fp = float(frame.false_px.to_numpy()[remove].sum())
    base_iou = tp / (tp + fp + fn)
    new_iou = (tp - lost_tp) / (tp - lost_tp + fp - cleared_fp + fn + lost_tp)
    base_err, new_err = fp + fn, (fp - cleared_fp) + (fn + lost_tp)
    wrong = remove & (frame.purity.to_numpy() >= base_iou)
    return {
        "n_removed": int(remove.sum()),
        "delta_iou": new_iou - base_iou,
        "rer": (base_err - new_err) / base_err,
        "lost_tp": lost_tp,
        "cleared_fp": cleared_fp,
        "removal_precision": 1.0 - wrong.sum() / max(remove.sum(), 1),
        "wrong_tp": float(frame.intersection_px.to_numpy()[wrong].sum()),
    }


def sweep_best(
    frame: pd.DataFrame, score: np.ndarray, tp: float, fp: float, fn: float, base_iou: float
) -> dict:
    """按分数升序逐步移除，返回排序本身能达到的最优操作点（排序质量上界）。"""
    order = np.argsort(score)
    cum_tp = np.cumsum(frame.intersection_px.to_numpy()[order])
    cum_fp = np.cumsum(frame.false_px.to_numpy()[order])
    denom_total = tp + fp + fn
    ious = (tp - cum_tp) / (denom_total - cum_fp)
    k = int(np.argmax(ious))
    return {
        "best_k": k + 1,
        "best_delta_iou": float(ious[k] - base_iou),
        "best_cut_score": float(score[order][k]),
    }


def run_arm(
    frame: pd.DataFrame,
    name: str,
    mode: str,
    weight: np.ndarray | None,
    tp: float,
    fp: float,
    fn: float,
    base_iou: float,
    seed: int = 20260725,
) -> dict:
    """按原 fold_id 做留折预测，返回整池指标。"""
    x = frame[FEATURES].to_numpy(dtype=float)
    purity = frame.purity.to_numpy(dtype=float)
    # 解析判据：移除有益 <=> purity < IoU / (1 + IoU)
    purity_cut = base_iou / (1.0 + base_iou)
    beneficial = (purity < purity_cut).astype(int)
    folds = frame.fold_id.to_numpy()
    remove = np.zeros(len(frame), dtype=bool)
    score = np.zeros(len(frame), dtype=float)

    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        w = None if weight is None else weight[train]
        if mode == "regress":
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=seed,
            )
            model.fit(x[train], purity[train], sample_weight=w)
            pred = model.predict(x[test])
            remove[test] = pred < purity_cut
            score[test] = pred
        else:
            model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=seed,
            )
            model.fit(x[train], beneficial[train], sample_weight=w)
            pred = model.predict_proba(x[test])[:, 1]
            remove[test] = pred > 0.5
            score[test] = -pred  # 统一为“分数越低越该移除”

    out = pooled_metrics(frame, remove, tp, fp, fn)
    out.update(sweep_best(frame, score, tp, fp, fn, base_iou))
    out["spearman_vs_purity"] = float(
        pd.Series(score).corr(pd.Series(frame.purity.to_numpy()), method="spearman")
    )
    big = frame.area_px.to_numpy() >= 200
    out["spearman_big"] = float(
        pd.Series(score[big]).corr(pd.Series(frame.purity.to_numpy()[big]), method="spearman")
    )
    out["arm"] = name
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--baseline-iou", type=float, default=0.21819164482792633)
    parser.add_argument("--outdir", type=Path, default=Path(
        "experiments/revision2026/pild_object_mass_weighted_probe_v1"))
    args = parser.parse_args()

    frame = pd.read_csv(args.decisions)
    tp = float(frame.intersection_px.sum())
    fp = float(frame.false_px.sum())
    fn = tp / args.baseline_iou - tp - fp
    base_iou = args.baseline_iou

    i = frame.intersection_px.to_numpy(dtype=float)
    f = frame.false_px.to_numpy(dtype=float)
    # 决策论权重：该对象被正确/错误处理时对整池 IoU 的影响量级
    mass = np.abs(base_iou * f - i)
    mass_norm = mass / mass.mean()
    big = frame.area_px.to_numpy() >= 50
    mass_big = np.where(big, mass_norm, 1.0)

    arms = [
        ("A_unweighted_regress", "regress", None),
        ("B_mass_weighted_regress", "regress", mass_norm),
        ("C_mass_weighted_classify", "classify", mass_norm),
        ("D_bigobject_weighted_classify", "classify", mass_big),
    ]

    rows = []
    for name, mode, weight in arms:
        res = run_arm(frame, name, mode, weight, tp, fp, fn, base_iou)
        rows.append(res)
        print(
            f"{name:32s} n_rm={res['n_removed']:6d}  ΔIoU={res['delta_iou']:+.5f}  "
            f"RER={res['rer']:+.4f}  prec={res['removal_precision']:.4f}  "
            f"wrongTP={res['wrong_tp']:,.0f}  | 排序上界 ΔIoU={res['best_delta_iou']:+.5f}"
            f" @k={res['best_k']:,}  rho={res['spearman_vs_purity']:.3f}"
            f" rho_big={res['spearman_big']:.3f}"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / "arms.csv", index=False)
    print(f"\n写出 {args.outdir / 'arms.csv'}")
    print(f"参考：当前部署 ΔIoU=+0.01913；达 +0.03 需把误删 TP 从 242,527 降到约 170,000")


if __name__ == "__main__":
    main()
