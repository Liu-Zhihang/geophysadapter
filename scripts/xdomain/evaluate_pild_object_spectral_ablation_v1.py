#!/usr/bin/env python3
"""G8-b：光谱描述子与物理描述子的对象级消融。

要回答审稿人必问的一个问题：对象级物理审查的收益，是否只是"更好的视觉后处理"？
为此必须给视觉侧一个公平且强的对手——不是 3 个概率标量，而是完整的光谱与变化描述子。

同时这也是当前 ΔIoU 缺口的主攻方向。设计图（probe_pild_recall_first_cascade_v1）显示
事件外排序 Spearman 从 0.374 提到 0.45 即可跨过 +0.03，而排序质量是唯一起作用的旋钮。

臂：
    confidence_only  仅 3 个概率标量（原视觉基线）
    spectral_only    仅光谱与变化描述子
    terrain_only     仅地形几何 + 概率标量（现行部署口径）
    terrain_spectral 两者合并
    spectral_shift   光谱描述子在数据源内按事件打乱（错配控制）

一律事件分组 OOF，报告排序相关、事后最优截断的 ΔIoU（排序质量），以及用解析判据
i/f < IoU_base 的可部署 ΔIoU。
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
DEFAULT_DECISIONS = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_veto_gate_v2/aligned_component_decisions.csv"
)
DEFAULT_SPECTRAL = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_spectral_features_v1/object_spectral_features.parquet"
)

CONFIDENCE = ["mean_probability", "max_probability", "p90_probability"]
TERRAIN = [
    "area_px", "log_area", "mean_slope", "p10_slope", "p90_slope", "flat_fraction",
    "steep_fraction", "elev_range", "relative_relief", "aspect_coherence", "elongation",
    "downslope_alignment", "descent_consistency", "slope_decline", "divide_straddle",
    "tpi900_range", "mean_tpi_90m", "mean_tpi_300m", "mean_tpi_900m",
    "valley_bottom_fraction", "mean_valley_depth", "mean_ridge_height", "mean_ruggedness",
    "mean_local_relief_300m", "mean_plan_curvature", "mean_profile_curvature", "compactness",
    "terrain_support_fraction",
]


def oof_scores(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> np.ndarray:
    """事件分组交叉预测的对象纯度。"""
    out = np.zeros(len(y), dtype=float)
    for train, test in GroupKFold(n_splits).split(x, y, groups=groups):
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(x[train], y[train])
        out[test] = model.predict(x[test])
    return np.clip(out, 0.0, 1.0)


def summarize(
    score: np.ndarray, frame: pd.DataFrame, tp: float, fp: float, fn: float, base_iou: float
) -> dict:
    """排序质量上界 + 解析判据下的可部署结果。"""
    i_px = frame.intersection_px.to_numpy(dtype=float)
    f_px = frame.false_px.to_numpy(dtype=float)
    denom = tp + fp + fn

    order = np.argsort(score)
    ious = np.concatenate(
        [[base_iou], (tp - np.cumsum(i_px[order])) / (denom - np.cumsum(f_px[order]))]
    )
    k = int(np.argmax(ious))

    # 可部署判据：预测纯度低于 IoU/(1+IoU) 即移除
    cut = base_iou / (1.0 + base_iou)
    remove = score < cut
    lost = float(i_px[remove].sum())
    cleared = float(f_px[remove].sum())
    deployed = (tp - lost) / (denom - cleared)
    base_err = fp + fn
    new_err = (fp - cleared) + (fn + lost)
    big = frame.area_px.to_numpy() >= 200
    return {
        "spearman": float(pd.Series(score).corr(frame.purity, method="spearman")),
        "spearman_big": float(
            pd.Series(score[big]).corr(pd.Series(frame.purity.to_numpy()[big]), method="spearman")
        ),
        "ranking_best_delta_iou": float(ious[k] - base_iou),
        "ranking_best_k": k,
        "deployed_delta_iou": float(deployed - base_iou),
        "deployed_rer": float((base_err - new_err) / base_err),
        "deployed_n_removed": int(remove.sum()),
        "deployed_removal_precision": float(
            1.0 - (remove & (frame.purity.to_numpy() >= base_iou)).sum() / max(remove.sum(), 1)
        ),
    }


def shuffle_within_dataset(
    values: np.ndarray, datasets: np.ndarray, events: np.ndarray, seed: int
) -> np.ndarray:
    """源内按事件整体置换特征块，破坏对象与其光谱证据的对应关系。"""
    rng = np.random.default_rng(seed)
    out = values.copy()
    for ds in np.unique(datasets):
        rows = np.where(datasets == ds)[0]
        uniq = np.unique(events[rows])
        if len(uniq) < 2:
            continue
        mapping = dict(zip(uniq, rng.permutation(uniq)))
        donor_pool = {e: np.where(events == e)[0] for e in uniq}
        for e in uniq:
            target_rows = np.where((events == e) & (datasets == ds))[0]
            donor = donor_pool[mapping[e]]
            pick = rng.integers(0, len(donor), size=len(target_rows))
            out[target_rows] = values[donor[pick]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--spectral", type=Path, default=DEFAULT_SPECTRAL)
    parser.add_argument("--baseline-iou", type=float, default=0.21819164482792633)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_spectral_ablation_v1",
    )
    args = parser.parse_args()
    started = time.time()

    frame = pd.read_csv(args.decisions)
    spectral = pd.read_parquet(args.spectral)
    merged = frame.merge(spectral, on=["sample_id", "component_id"], how="inner")
    if len(merged) != len(frame):
        raise RuntimeError(f"对齐失败：决策表 {len(frame)} 行，合并后 {len(merged)} 行")
    merged = merged.reset_index(drop=True)

    spec_cols = [c for c in merged.columns if c.startswith("spec_")]
    print(f"对象 {len(merged):,}  光谱描述子 {len(spec_cols)}  地形描述子 {len(TERRAIN)}")

    tp = float(merged.intersection_px.sum())
    fp = float(merged.false_px.sum())
    fn = tp / args.baseline_iou - tp - fp
    base_iou = args.baseline_iou
    y = merged.purity.to_numpy(dtype=float)
    groups = merged.canonical_event_id.to_numpy()
    datasets = merged.dataset_id.to_numpy()

    spec_shuffled = shuffle_within_dataset(
        merged[spec_cols].to_numpy(dtype=float), datasets, groups, args.seed + 13
    )

    arms = {
        "confidence_only": merged[CONFIDENCE].to_numpy(dtype=float),
        "spectral_only": merged[spec_cols].to_numpy(dtype=float),
        "spectral_confidence": merged[spec_cols + CONFIDENCE].to_numpy(dtype=float),
        "terrain_only": merged[TERRAIN + CONFIDENCE].to_numpy(dtype=float),
        "terrain_spectral": merged[TERRAIN + CONFIDENCE + spec_cols].to_numpy(dtype=float),
        "spectral_shuffled": np.hstack(
            [merged[TERRAIN + CONFIDENCE].to_numpy(dtype=float), spec_shuffled]
        ),
    }

    rows = []
    for name, x in arms.items():
        score = oof_scores(x, y, groups, args.n_splits, args.seed)
        res = summarize(score, merged, tp, fp, fn, base_iou)
        res["arm"] = name
        res["n_features"] = int(x.shape[1])
        rows.append(res)
        print(
            f"{name:22s} rho={res['spearman']:.3f} rho_big={res['spearman_big']:.3f}  "
            f"排序上界Δ={res['ranking_best_delta_iou']:+.5f}  "
            f"可部署Δ={res['deployed_delta_iou']:+.5f}  RER={res['deployed_rer']:+.4f}  "
            f"精度={res['deployed_removal_precision']:.3f}"
        )

    table = pd.DataFrame(rows).set_index("arm")
    terrain = table.loc["terrain_only"]
    joint = table.loc["terrain_spectral"]
    spec = table.loc["spectral_confidence"]
    shuffled = table.loc["spectral_shuffled"]
    verdict = {
        "terrain_only_deployed": float(terrain.deployed_delta_iou),
        "spectral_confidence_deployed": float(spec.deployed_delta_iou),
        "terrain_spectral_deployed": float(joint.deployed_delta_iou),
        "physics_increment_over_spectral": float(
            joint.deployed_delta_iou - spec.deployed_delta_iou
        ),
        "spectral_increment_over_terrain": float(
            joint.deployed_delta_iou - terrain.deployed_delta_iou
        ),
        "beats_spectral_shuffled": bool(
            joint.deployed_delta_iou > shuffled.deployed_delta_iou
        ),
        "joint_spearman": float(joint.spearman),
        "reaches_target_delta_iou": bool(joint.deployed_delta_iou >= 0.03),
    }
    print("\n裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))

    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "arms.csv")
    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_spectral_ablation.v1",
                "evidence_status": "development: event-grouped OOF on already-opened folds",
                "n_objects": int(len(merged)),
                "baseline_iou": base_iou,
                "spectral_features": spec_cols,
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
