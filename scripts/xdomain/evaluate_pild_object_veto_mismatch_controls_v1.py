#!/usr/bin/env python3
"""最终配置下的地形错配控制。

需要回答的是：ΔIoU +0.03095 里有多少来自"地形与候选体在空间上正确对齐"，
而不是来自描述子的边缘统计、光谱证据、或数据源身份。

干预只作用于用来判断候选体的地形栈，候选体本身完全不动：
    aligned   正确对齐
    shift32   地形整体平移 32 px（约 320 m），形态统计不变、位置对应被破坏
    roll64    平移 64 px（约 640 m），破坏更彻底
    donor     换成同源但不同事件的地形，边缘分布真实、位置证据完全错误

地形与水文描述子在每种条件下重算（水文量由 elevation 派生，必须跟着动），
光谱与置信度描述子不依赖地形，保持不变。评估协议与最终配置逐字一致：
92 维描述子 + 源独热、五种子集成、解析判据、事件分组 OOF。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_pild_object_physical_separability_v1 import (  # noqa: E402
    apply_terrain_condition,
    component_features,
)
from export_pild_object_hydrology_features_v1 import (  # noqa: E402
    component_hydrology,
    hydrology_stack,
)

DEFAULT_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
DEFAULT_SPECTRAL = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_spectral_features_v1/object_spectral_features.parquet"
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
REFERENCE_IOU = 0.21819164482792633
MIN_AREA = 4
RING_RADIUS = 5


def build_condition_table(
    cache_dir: Path, condition: str, seed: int, min_area: int, ring_radius: int
) -> pd.DataFrame:
    """在给定地形干预下，重算每个候选体的地形与水文描述子。"""
    structure = ndimage.generate_binary_structure(2, 2)
    rows: list[dict] = []
    for fold_id in FOLD_IDS:
        receipt = json.loads(
            (cache_dir / f"{fold_id}_oof_cache_receipt.json").read_text(encoding="utf-8")
        )
        threshold = float(receipt["threshold"])
        with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
            sample_id = [str(item) for item in handle["sample_id"]]
            dataset_id = handle["dataset_id"]
            event_id = handle["canonical_event_id"]
            probability_all = handle["visual_probability"]
            target_all = handle["target"]
            valid_all = handle["valid"]
            terrain_all = handle["terrain"]
            terrain_valid_all = handle["terrain_valid"]
        terrain_all, terrain_valid_all = apply_terrain_condition(
            terrain_all, terrain_valid_all, dataset_id, event_id, condition, seed
        )
        for index in range(len(sample_id)):
            keep = valid_all[index].astype(bool)
            truth = target_all[index].astype(bool) & keep
            probability = probability_all[index].astype(np.float32)
            predicted = (probability >= threshold) & keep
            if not predicted.any():
                continue
            labels, count = ndimage.label(predicted, structure=structure)
            if count == 0:
                continue
            terrain = terrain_all[index].astype(np.float32)
            support = terrain_valid_all[index].astype(bool)
            stack = hydrology_stack(terrain[0], keep)
            windows = ndimage.find_objects(labels)
            for label_value in range(1, count + 1):
                window = windows[label_value - 1]
                local = labels[window] == label_value
                area = int(local.sum())
                if area < min_area:
                    continue
                rows_local, cols_local = np.nonzero(local)
                rows_global = rows_local + window[0].start
                cols_global = cols_local + window[1].start
                row = component_features(
                    rows_global, cols_global, terrain, probability, local
                )
                row.update(component_hydrology(local, window, stack, keep, ring_radius))
                row["terrain_support_fraction"] = float(
                    np.mean(support[rows_global, cols_global])
                )
                intersection = int(np.count_nonzero(truth[rows_global, cols_global]))
                row.update(
                    {
                        "sample_id": sample_id[index],
                        "dataset_id": str(dataset_id[index]),
                        "canonical_event_id": str(event_id[index]),
                        "component_id": int(label_value),
                        "purity": intersection / area,
                        "intersection_px": float(intersection),
                        "false_px": float(area - intersection),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def ensemble_oof(x, y, groups, seeds, n_splits, max_iter, max_leaf_nodes):
    total = np.zeros(len(y), dtype=float)
    for seed in seeds:
        out = np.zeros(len(y), dtype=float)
        for train, test in GroupKFold(n_splits).split(x, y, groups=groups):
            model = HistGradientBoostingRegressor(
                max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, learning_rate=0.06,
                l2_regularization=1.0, random_state=seed,
            )
            model.fit(x[train], y[train])
            out[test] = model.predict(x[test])
        total += np.clip(out, 0.0, 1.0)
    return total / len(seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--spectral", type=Path, default=DEFAULT_SPECTRAL)
    parser.add_argument(
        "--conditions", nargs="+", default=["aligned", "shift32", "roll64", "donor"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 101, 2029, 55555])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--control-seed", type=int, default=20260725)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_veto_mismatch_v1",
    )
    args = parser.parse_args()
    started = time.time()

    spectral = pd.read_parquet(args.spectral)
    spec_cols = [c for c in spectral.columns if c.startswith("spec_")]

    rows = []
    for condition in args.conditions:
        stage = time.time()
        frame = build_condition_table(
            args.cache, condition, args.control_seed, MIN_AREA, RING_RADIUS
        )
        frame = frame.merge(spectral, on=["sample_id", "component_id"], how="inner")
        hyd_cols = [c for c in frame.columns if c.startswith("hyd_")]
        feature_cols = TERRAIN + CONFIDENCE + spec_cols + hyd_cols

        tp = float(frame.intersection_px.sum())
        fp = float(frame.false_px.sum())
        fn = tp / REFERENCE_IOU - tp - fp
        cut = REFERENCE_IOU / (1.0 + REFERENCE_IOU)
        y = frame.purity.to_numpy(dtype=float)
        groups = frame.canonical_event_id.to_numpy()
        x = np.hstack(
            [
                frame[feature_cols].to_numpy(dtype=float),
                pd.get_dummies(frame.dataset_id).to_numpy(dtype=float),
            ]
        )
        score = ensemble_oof(
            x, y, groups, args.seeds, args.n_splits, args.max_iter, args.max_leaf_nodes
        )
        remove = score < cut
        i_px = frame.intersection_px.to_numpy(dtype=float)
        f_px = frame.false_px.to_numpy(dtype=float)
        lost = float(i_px[remove].sum())
        cleared = float(f_px[remove].sum())
        base_err = fp + fn
        new_err = (fp - cleared) + (fn + lost)
        res = {
            "condition": condition,
            "n_units": int(len(frame)),
            "n_removed": int(remove.sum()),
            "delta_iou": float((tp - lost) / (tp + fp + fn - cleared) - REFERENCE_IOU),
            "rer": float((base_err - new_err) / base_err),
            "spearman": float(pd.Series(score).corr(pd.Series(y), method="spearman")),
            "corrected_to_harmed": float(cleared / max(lost, 1.0)),
            "removal_precision": float(
                1.0 - (remove & (y >= REFERENCE_IOU)).sum() / max(remove.sum(), 1)
            ),
            "fp_mass_captured": float(cleared / fp),
            "tp_mass_lost": float(lost / tp),
            "elapsed_seconds": round(time.time() - stage, 1),
        }
        rows.append(res)
        print(
            f"{condition:9s} ΔIoU={res['delta_iou']:+.5f}  RER={res['rer']:+.4f}  "
            f"rho={res['spearman']:.3f}  c/h={res['corrected_to_harmed']:.2f}  "
            f"精度={res['removal_precision']:.3f}  ({res['elapsed_seconds']:.0f}s)"
        )

    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "conditions.csv", index=False)
    aligned = table[table.condition == "aligned"].iloc[0]
    controls = table[table.condition != "aligned"]
    verdict = {
        "aligned_delta_iou": float(aligned.delta_iou),
        "best_control_delta_iou": float(controls.delta_iou.max()),
        "aligned_minus_best_control": float(aligned.delta_iou - controls.delta_iou.max()),
        "beats_all_controls": bool((aligned.delta_iou > controls.delta_iou).all()),
        "aligned_rer": float(aligned.rer),
        "best_control_rer": float(controls.rer.max()),
    }
    print("\n裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))
    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_veto_mismatch.v1",
                "evidence_status": "development: event-grouped OOF on already-opened folds",
                "reference_iou": REFERENCE_IOU,
                "conditions": rows,
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
