#!/usr/bin/env python3
"""L4 探针：召回优先级联的可行性。

当前对象候选由 0.92 的掩膜阈值生成，这是视觉基线自身的 IoU 最优点，代价是 FN 质量
（2.59 M px）超过 TP 质量（1.82 M px）。若物理对象审查能高精度清除整块假阳性，
就可以让视觉侧运行在更高召回的操作点，用审查买回精度。

这与已关闭的"救援"方向机制不同：救援试图把邻接的漏检区域重新加回来，排序太弱；
这里是直接在更低阈值上生成候选体，让漏检区域本身成为可审查的对象。

本探针只算上界与带噪估计，不训练模型：
    oracle    ：完美纯度排序下的可达 IoU
    noisy     ：把纯度加噪至与实测事件外排序质量（Spearman≈0.37）相当后的可达 IoU
参照系一律是当前基线 IoU = 0.21819（阈值 0.92 处的视觉最优）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
FOLD_IDS = [f"source_stratified_{i}" for i in range(4)]
STRUCTURE = np.ones((3, 3), dtype=int)  # 8 邻接，与导出脚本一致


def scan_threshold(cache_dir: Path, threshold: float) -> dict:
    """在给定掩膜阈值下统计整池混淆矩阵与逐对象 TP/FP。"""
    inter, false, event_ids = [], [], []
    total_tp = total_fp = total_fn = 0.0
    for fold_id in FOLD_IDS:
        path = cache_dir / f"{fold_id}_oof_cache.npz"
        with np.load(path, allow_pickle=False) as handle:
            prob = handle["visual_probability"]
            target = handle["target"]
            valid = handle["valid"]
            events = handle["canonical_event_id"]
            for k in range(prob.shape[0]):
                v = valid[k].astype(bool)
                y = (target[k] > 0) & v
                p = (prob[k].astype(np.float32) >= threshold) & v
                total_tp += float((p & y).sum())
                total_fp += float((p & ~y).sum())
                total_fn += float((~p & y).sum())
                if not p.any():
                    continue
                labels, n = ndimage.label(p, structure=STRUCTURE)
                if n == 0:
                    continue
                idx = np.arange(1, n + 1)
                i_px = ndimage.sum(y & p, labels, index=idx)
                a_px = ndimage.sum(p, labels, index=idx)
                inter.append(np.asarray(i_px, dtype=np.float64))
                false.append(np.asarray(a_px - i_px, dtype=np.float64))
                event_ids.append(np.repeat(str(events[k]), n))
    return {
        "threshold": threshold,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "iou": total_tp / max(total_tp + total_fp + total_fn, 1.0),
        "intersection": np.concatenate(inter) if inter else np.zeros(0),
        "false": np.concatenate(false) if false else np.zeros(0),
        "event": np.concatenate(event_ids) if event_ids else np.zeros(0, dtype=str),
    }


def best_removal_iou(
    score: np.ndarray, i_px: np.ndarray, f_px: np.ndarray, tp: float, fp: float, fn: float
) -> tuple[float, int]:
    """按分数升序逐步移除对象，返回可达的最优整池 IoU 与移除数。"""
    order = np.argsort(score)
    cum_i = np.cumsum(i_px[order])
    cum_f = np.cumsum(f_px[order])
    denom = tp + fp + fn
    ious = np.concatenate([[tp / denom], (tp - cum_i) / (denom - cum_f)])
    k = int(np.argmax(ious))
    return float(ious[k]), k


def noisy_purity(
    purity: np.ndarray, target_rho: float, seed: int, tries: int = 24
) -> np.ndarray:
    """给真纯度加高斯噪声，使其与真值的 Spearman 相关接近实测的事件外排序质量。"""
    rng = np.random.default_rng(seed)
    lo, hi = 0.0, 4.0
    best = purity.copy()
    for _ in range(tries):
        mid = 0.5 * (lo + hi)
        cand = purity + rng.normal(0.0, mid, size=purity.shape)
        rho = pd.Series(cand).corr(pd.Series(purity), method="spearman")
        if rho > target_rho:
            lo = mid
        else:
            hi = mid
        best = cand
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.30, 0.50, 0.70, 0.80, 0.88, 0.92, 0.95],
    )
    parser.add_argument("--reference-iou", type=float, default=0.21819164482792633)
    parser.add_argument("--target-rho", type=float, default=0.374)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_recall_first_cascade_probe_v1",
    )
    args = parser.parse_args()

    rows = []
    for threshold in args.thresholds:
        stat = scan_threshold(args.cache, threshold)
        i_px, f_px = stat["intersection"], stat["false"]
        area = i_px + f_px
        purity = np.divide(i_px, np.maximum(area, 1.0))
        tp, fp, fn = stat["tp"], stat["fp"], stat["fn"]

        oracle_iou, oracle_k = best_removal_iou(purity, i_px, f_px, tp, fp, fn)
        noisy = noisy_purity(purity, args.target_rho, args.seed)
        noisy_iou, noisy_k = best_removal_iou(noisy, i_px, f_px, tp, fp, fn)

        row = {
            "threshold": threshold,
            "n_objects": int(len(i_px)),
            "pure_fp_objects": int((i_px == 0).sum()),
            "baseline_iou": stat["iou"],
            "tp": tp, "fp": fp, "fn": fn,
            "oracle_veto_iou": oracle_iou,
            "oracle_removed": oracle_k,
            "noisy_veto_iou": noisy_iou,
            "noisy_removed": noisy_k,
            "delta_vs_reference_oracle": oracle_iou - args.reference_iou,
            "delta_vs_reference_noisy": noisy_iou - args.reference_iou,
        }
        rows.append(row)
        print(
            f"thr={threshold:.2f}  基线IoU={stat['iou']:.5f}  对象={len(i_px):6,}  "
            f"纯FP体={int((i_px == 0).sum()):6,}  |  oracle后IoU={oracle_iou:.5f} "
            f"(Δ vs 0.2182 = {oracle_iou - args.reference_iou:+.5f})  |  "
            f"带噪后IoU={noisy_iou:.5f} (Δ={noisy_iou - args.reference_iou:+.5f})"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / "threshold_sweep.csv", index=False)
    best_noisy = table.loc[table.delta_vs_reference_noisy.idxmax()]
    payload = {
        "schema_version": "pild_recall_first_cascade_probe.v1",
        "evidence_status": "development: upper bound and noise-matched estimate, no model trained",
        "reference_iou": args.reference_iou,
        "target_rho": args.target_rho,
        "best_noisy_threshold": float(best_noisy.threshold),
        "best_noisy_delta": float(best_noisy.delta_vs_reference_noisy),
        "rows": rows,
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n带噪估计最优阈值 {best_noisy.threshold:.2f}，"
        f"ΔIoU={best_noisy.delta_vs_reference_noisy:+.5f}（参照 0.21819）"
    )
    print(f"写出 {args.outdir}")


if __name__ == "__main__":
    main()
