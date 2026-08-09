#!/usr/bin/env python3
"""对任意视觉锚点重跑完整的对象级物理审查流水线。

用于回答"对象级机制是否依赖特定视觉锚点"。每个锚点各自：

1. 复用同一份光学缓存（影像与锚点无关，软链接即可）；
2. 用该锚点自己的 OOF 概率与逐折阈值重建连通分量，重算地形几何与光谱描述子；
3. 重算水文拓扑描述子（分量变了，必须重算）；
4. 用该锚点**自己的**整池基线 IoU 作为参照系与解析判据，跑最终配置评估。

第 4 步是关键：不同锚点的基线 IoU 不同，若沿用 Prithvi 的 0.21819 会同时污染
参照系与移除判据 `purity < IoU/(1+IoU)`，把锚点强弱混进物理效应里。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PRITHVI_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
FOLD_IDS = [f"source_stratified_{index}" for index in range(4)]
PYTHON = sys.executable


def link_optical_caches(anchor_cache: Path, donor: Path) -> None:
    """光学影像与视觉锚点无关，软链接复用，避免重复占用磁盘。"""
    for fold_id in FOLD_IDS:
        for suffix in ("_optical_cache.npz", "_optical_cache_receipt.json"):
            target = donor / f"{fold_id}{suffix}"
            link = anchor_cache / f"{fold_id}{suffix}"
            if not target.is_file():
                raise FileNotFoundError(target)
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(target)


def pooled_baseline_iou(anchor_cache: Path) -> dict[str, float]:
    """由该锚点自己的逐折阈值统计整池 TP/FP/FN。"""
    tp = fp = fn = 0.0
    thresholds = {}
    for fold_id in FOLD_IDS:
        receipt = json.loads(
            (anchor_cache / f"{fold_id}_oof_cache_receipt.json").read_text(encoding="utf-8")
        )
        threshold = float(receipt["threshold"])
        thresholds[fold_id] = threshold
        with np.load(anchor_cache / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
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
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "baseline_iou": tp / (tp + fp + fn),
        "thresholds": thresholds,
    }


def run(command: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(command[-6:])}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{label} 失败，退出码 {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, help="锚点标识，如 hiera_small_mae_fpn")
    parser.add_argument("--anchor-root", type=Path, default=None)
    parser.add_argument("--donor-cache", type=Path, default=PRITHVI_CACHE)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    started = time.time()

    root = args.anchor_root or (
        PROJECT_ROOT / f"experiments/revision2026/pild_alt_anchor_{args.anchor}_v1"
    )
    cache = root / "oof_cache"
    if not cache.is_dir():
        raise FileNotFoundError(cache)

    link_optical_caches(cache, args.donor_cache.resolve())
    counts = pooled_baseline_iou(cache)
    print(
        f"[{args.anchor}] 逐折阈值 "
        + " ".join(f"{k.split('_')[-1]}:{v:.3f}" for k, v in counts["thresholds"].items())
        + f"\n[{args.anchor}] 整池 TP={counts['tp']:,.0f} FP={counts['fp']:,.0f} "
        f"FN={counts['fn']:,.0f}  基线 IoU={counts['baseline_iou']:.5f}",
        flush=True,
    )

    units = root / "units"
    hydrology = root / "hydrology"
    if not args.skip_export:
        run(
            [PYTHON, "scripts/xdomain/export_pild_subobject_units_v1.py",
             "--cache", str(cache), "--modes", "whole", "--outdir", str(units)],
            "units",
        )
        run(
            [PYTHON, "scripts/xdomain/export_pild_object_hydrology_features_v1.py",
             "--cache", str(cache), "--outdir", str(hydrology)],
            "hydrology",
        )

    run(
        [PYTHON, "scripts/xdomain/evaluate_pild_object_veto_final_v1.py",
         "--units", str(units),
         "--hydrology", str(hydrology / "object_hydrology_features.parquet"),
         "--baseline-iou", f"{counts['baseline_iou']:.10f}",
         "--anchor", args.anchor,
         "--outdir", str(root / "object_veto_final")],
        "veto",
    )

    (root / "anchor_baseline.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_anchor_object_veto.v1",
                "anchor": args.anchor,
                **counts,
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[done] {args.anchor} 对象级流水线完成 -> {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
