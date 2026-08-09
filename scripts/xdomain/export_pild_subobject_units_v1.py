#!/usr/bin/env python3
"""G9-a：把连通分量切成地貌一致的子对象。

诊断依据（见 对象级增益上限拆账与MR角色重定位_攻坚方案_20260725.md §6bis）：
纯度落在 0.05–0.60 的模糊体只占 11.0% 的对象，却承载 26.3% 的真阳性与 29.1% 的假阳性；
面积 200–5000 px 的体中三分之一以上是模糊体。这些体整块保留或整块移除都必然大量出错。
这不是排序问题，而是决策单元问题：视觉掩膜的连通分量不是地貌单元。

切分依据是三种边界证据的加权和（各自在样本内做鲁棒归一化后等权）：
    divide       正的平面曲率，即分水线与坡面分隔
    aspect_break 坡向单位矢量场的空间梯度，即坡面朝向的突变
    slope_break  坡度穿越临界角处的带状响应 exp(-((slope - theta_c)/delta)^2)

其中 theta_c 是 Material 进入决策的第三种功能位置：不参与对象排序，也不设定移除阈值，
而是决定"在哪里断开"。土力学上砂质、含砾越高内摩擦角越大，黏粒与持水量越高越弱，因此

    theta_c = 22 deg + 16 deg * rank01( z(sand) + 0.5 z(cfvo) - z(clay) - 0.5 z(awc) )

22–38 度是土体内摩擦角的常规区间，映射为先验固定、不看任何标签，因此不存在泄漏。
无 Material 支持的样本回退到固定 25 度（与既有 STEEP_SLOPE_DEG 一致）。

三种切分模式一次导出：
    whole              不切分，复现现有 62,203 个对象（对照基准）
    geomorphic         切分，theta_c 固定 25 度（Material 不参与）
    material           切分，theta_c 由 Material 调制
    material_shuffled  切分，theta_c 由源内按事件打乱后的 Material 调制（错配控制）
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
from skimage.segmentation import watershed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_pild_object_physical_separability_v1 import (  # noqa: E402
    TERRAIN_INDEX,
    component_features,
)
from export_pild_object_spectral_features_v1 import (  # noqa: E402
    component_spectral_features,
    spectral_index_stack,
)

DEFAULT_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
FOLD_IDS = [f"source_stratified_{i}" for i in range(4)]

# ROLE_MATERIAL_FEATURE_NAMES 的列序：5 个 AWC 深度，随后 (clay,sand,silt,cec,soc,bdod,cfvo,phh2o) x (0_5cm,5_15cm)
AWC_SLICE = slice(0, 5)
CLAY_IDX = (5, 6)
SAND_IDX = (7, 8)
CFVO_IDX = (17, 18)

THETA_MIN_DEG = 22.0
THETA_SPAN_DEG = 16.0
THETA_FALLBACK_DEG = 25.0
SLOPE_BREAK_WIDTH_DEG = 3.0
MARKER_QUANTILE = 0.35
MIN_AREA = 4


def robust_unit(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """用掩膜内的 95 分位做鲁棒归一化，输出裁剪到 [0, 1]。"""
    if not mask.any():
        return np.zeros_like(values, dtype=np.float32)
    scale = float(np.percentile(np.abs(values[mask]), 95))
    if scale < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def rank01(values: np.ndarray) -> np.ndarray:
    """秩归一化到 [0, 1]，只用特征本身，不涉及任何标签。"""
    order = np.argsort(np.argsort(values))
    return order / max(len(values) - 1, 1)


def critical_angle_from_material(
    material: np.ndarray, q_material: np.ndarray
) -> np.ndarray:
    """由土壤质地推出逐样本临界坡度，先验固定映射到 22–38 度。"""
    def z(column: np.ndarray) -> np.ndarray:
        std = float(np.std(column))
        return (column - float(np.mean(column))) / (std if std > 1e-9 else 1.0)

    sand = material[:, SAND_IDX[0] : SAND_IDX[1] + 1].mean(axis=1)
    clay = material[:, CLAY_IDX[0] : CLAY_IDX[1] + 1].mean(axis=1)
    cfvo = material[:, CFVO_IDX[0] : CFVO_IDX[1] + 1].mean(axis=1)
    awc = material[:, AWC_SLICE].mean(axis=1)
    index = z(sand) + 0.5 * z(cfvo) - z(clay) - 0.5 * z(awc)
    theta = THETA_MIN_DEG + THETA_SPAN_DEG * rank01(index)
    return np.where(q_material > 0, theta, THETA_FALLBACK_DEG).astype(np.float32)


def boundary_strength(
    terrain: np.ndarray, valid: np.ndarray, theta_c: float
) -> np.ndarray:
    """样本级边界强度图：分水线 + 坡向突变 + 临界坡度断裂，三者等权。"""
    plan = terrain[TERRAIN_INDEX["plan_curvature"]].astype(np.float32)
    slope = terrain[TERRAIN_INDEX["slope_deg"]].astype(np.float32)
    aspect_sin = terrain[TERRAIN_INDEX["aspect_sin"]].astype(np.float32)
    aspect_cos = terrain[TERRAIN_INDEX["aspect_cos"]].astype(np.float32)

    divide = robust_unit(np.clip(plan, 0.0, None), valid)
    grad_sin = np.hypot(*np.gradient(aspect_sin))
    grad_cos = np.hypot(*np.gradient(aspect_cos))
    aspect_break = robust_unit(np.hypot(grad_sin, grad_cos), valid)
    slope_break = np.exp(
        -(((slope - theta_c) / SLOPE_BREAK_WIDTH_DEG) ** 2)
    ).astype(np.float32)
    return (divide + aspect_break + slope_break) / 3.0


def absorb_small_units(labels: np.ndarray, mask: np.ndarray, min_area: int) -> np.ndarray:
    """把面积不足的子单元并入最近的合格单元，保证像素账目不丢失。"""
    if labels.max() <= 1:
        return labels
    sizes = np.bincount(labels.ravel())
    small = {i for i in range(1, len(sizes)) if 0 < sizes[i] < min_area}
    if not small:
        return labels
    keep = np.isin(labels, [i for i in range(1, len(sizes)) if sizes[i] >= min_area])
    if not keep.any():
        return np.where(mask, 1, 0).astype(labels.dtype)
    # 最近合格单元的标签由距离变换的索引给出
    _, indices = ndimage.distance_transform_edt(~keep, return_indices=True)
    filled = labels[tuple(indices)]
    return np.where(mask & ~keep, filled, labels)


def split_component(
    local_mask: np.ndarray, local_boundary: np.ndarray, min_area: int
) -> np.ndarray:
    """在单个连通体内做分水岭切分，返回从 1 开始编号的子单元标签图。"""
    if int(local_mask.sum()) < 2 * min_area:
        return local_mask.astype(np.int32)
    inside = local_boundary[local_mask]
    cut = float(np.quantile(inside, MARKER_QUANTILE))
    seeds = local_mask & (local_boundary <= cut)
    markers, count = ndimage.label(seeds, structure=np.ones((3, 3), int))
    if count < 2:
        return local_mask.astype(np.int32)
    labels = watershed(local_boundary, markers, mask=local_mask)
    labels = absorb_small_units(labels, local_mask, min_area)
    # 重新压缩标签编号
    unique = np.unique(labels[labels > 0])
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, len(unique) + 1)
    return remap[labels]


def process_fold(
    cache_dir: Path,
    fold_id: str,
    modes: dict[str, np.ndarray],
    min_area: int,
    ring_radius: int,
    threshold_override: float | None = None,
) -> dict[str, list[dict]]:
    """对该折逐样本重建连通分量并按各模式切分，逐子单元计算全部描述子。

    threshold_override 用于召回优先级联：把掩膜阈值下调后重新生成候选体，
    由物理审查买回精度。留空则沿用该折 receipt 中的视觉最优阈值。
    """
    receipt = json.loads(
        (cache_dir / f"{fold_id}_oof_cache_receipt.json").read_text(encoding="utf-8")
    )
    threshold = float(receipt["threshold"]) if threshold_override is None else threshold_override
    with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
        sample_id = [str(item) for item in handle["sample_id"]]
        dataset_id = [str(item) for item in handle["dataset_id"]]
        event_id = [str(item) for item in handle["canonical_event_id"]]
        probability_all = handle["visual_probability"]
        target_all = handle["target"]
        valid_all = handle["valid"]
        terrain_all = handle["terrain"]
    with np.load(cache_dir / f"{fold_id}_optical_cache.npz", allow_pickle=False) as handle:
        pre_all = handle["optical_pre"]
        post_all = handle["optical_post"]

    structure = ndimage.generate_binary_structure(2, 2)
    out: dict[str, list[dict]] = {mode: [] for mode in modes}

    for index in range(len(sample_id)):
        keep = valid_all[index].astype(bool)
        truth = target_all[index].astype(bool) & keep
        probability = probability_all[index].astype(np.float32)
        predicted = (probability >= threshold) & keep
        if not predicted.any():
            continue
        component_labels, count = ndimage.label(predicted, structure=structure)
        if count == 0:
            continue
        terrain = terrain_all[index].astype(np.float32)
        pre_cube = pre_all[index].astype(np.float32)
        post_cube = post_all[index].astype(np.float32)
        pre_idx = spectral_index_stack(pre_cube)
        post_idx = spectral_index_stack(post_cube)
        windows = ndimage.find_objects(component_labels)

        boundary_cache: dict[float, np.ndarray] = {}
        for mode, theta_all in modes.items():
            if mode == "whole":
                continue
            theta = float(theta_all[0]) if theta_all.ndim == 0 else float(theta_all[index])
            if theta not in boundary_cache:
                boundary_cache[theta] = boundary_strength(terrain, keep, theta)

        for label_value in range(1, count + 1):
            window = windows[label_value - 1]
            local = component_labels[window] == label_value
            if int(local.sum()) < min_area:
                continue
            for mode, theta_all in modes.items():
                if mode == "whole":
                    units = local.astype(np.int32)
                else:
                    theta = (
                        float(theta_all[0]) if theta_all.ndim == 0 else float(theta_all[index])
                    )
                    units = split_component(
                        local, boundary_cache[theta][window], min_area
                    )
                for unit_value in range(1, int(units.max()) + 1):
                    unit = units == unit_value
                    area = int(unit.sum())
                    if area < min_area:
                        continue
                    rows_local, cols_local = np.nonzero(unit)
                    rows = rows_local + window[0].start
                    cols = cols_local + window[1].start
                    intersection = int(np.count_nonzero(truth[rows, cols]))
                    row = component_features(rows, cols, terrain, probability, unit)
                    row.update(
                        component_spectral_features(
                            unit, window, pre_idx, post_idx, pre_cube, post_cube,
                            keep, ring_radius,
                        )
                    )
                    row.update(
                        {
                            "sample_id": sample_id[index],
                            "dataset_id": dataset_id[index],
                            "canonical_event_id": event_id[index],
                            "component_id": int(label_value),
                            "unit_id": int(unit_value),
                            "parent_area_px": float(local.sum()),
                            "purity": intersection / area,
                            "intersection_px": float(intersection),
                            "false_px": float(area - intersection),
                        }
                    )
                    out[mode].append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--min-area", type=int, default=MIN_AREA)
    parser.add_argument("--ring-radius", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--modes", nargs="+", default=["whole", "geomorphic", "material", "material_shuffled"],
        help="需要导出的单元划分模式",
    )
    parser.add_argument(
        "--threshold-override", type=float, default=None,
        help="召回优先级联用：统一下调掩膜阈值重新生成候选体",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_subobject_units_v1",
    )
    args = parser.parse_args()
    started = time.time()

    # --- 先在全语料上算 Material 临界角，保证秩归一化口径一致 ---
    materials, q_materials, sample_ids, events, datasets = [], [], [], [], []
    for fold_id in FOLD_IDS:
        with np.load(args.cache / f"{fold_id}_optical_cache.npz", allow_pickle=False) as handle:
            materials.append(handle["material_features"])
            q_materials.append(handle["q_material"])
            sample_ids.extend(str(item) for item in handle["sample_id"])
        with np.load(args.cache / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
            events.extend(str(item) for item in handle["canonical_event_id"])
            datasets.extend(str(item) for item in handle["dataset_id"])
    material = np.concatenate(materials).astype(np.float64)
    q_material = np.concatenate(q_materials).astype(np.float64)
    theta = critical_angle_from_material(material, q_material)

    # 错配控制：源内按事件整体置换 Material 行
    rng = np.random.default_rng(args.seed + 31)
    events_arr = np.asarray(events)
    datasets_arr = np.asarray(datasets)
    donor = np.arange(len(material))
    for ds in np.unique(datasets_arr):
        rows = np.where(datasets_arr == ds)[0]
        uniq = np.unique(events_arr[rows])
        if len(uniq) < 2:
            continue
        mapping = dict(zip(uniq, rng.permutation(uniq)))
        for e in uniq:
            target_rows = np.where((events_arr == e) & (datasets_arr == ds))[0]
            pool = np.where(events_arr == mapping[e])[0]
            donor[target_rows] = pool[rng.integers(0, len(pool), size=len(target_rows))]
    theta_shuffled = theta[donor]

    print(
        f"临界坡度：中位 {np.median(theta):.1f}°  "
        f"IQR [{np.percentile(theta, 25):.1f}°, {np.percentile(theta, 75):.1f}°]  "
        f"有 Material 支持 {float((q_material > 0).mean()):.1%}"
    )

    offset = 0
    collected: dict[str, list[pd.DataFrame]] = {}
    for fold_id in FOLD_IDS:
        with np.load(args.cache / f"{fold_id}_optical_cache.npz", allow_pickle=False) as handle:
            n = len(handle["sample_id"])
        available = {
            "whole": np.asarray(THETA_FALLBACK_DEG),
            "geomorphic": np.full(n, THETA_FALLBACK_DEG, dtype=np.float32),
            "material": theta[offset : offset + n],
            "material_shuffled": theta_shuffled[offset : offset + n],
        }
        modes = {name: available[name] for name in args.modes}
        offset += n
        result = process_fold(
            args.cache, fold_id, modes, args.min_area, args.ring_radius,
            threshold_override=args.threshold_override,
        )
        for mode, rows in result.items():
            collected.setdefault(mode, []).append(pd.DataFrame(rows))
        print(
            f"{fold_id}: "
            + "  ".join(f"{mode}={len(rows):,}" for mode, rows in result.items())
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for mode, frames in collected.items():
        table = pd.concat(frames, ignore_index=True)
        table.to_parquet(args.outdir / f"units_{mode}.parquet", index=False)
        mixed = (table.purity > 0.05) & (table.purity < 0.60)
        stats[mode] = {
            "n_units": int(len(table)),
            "median_area": float(table.area_px.median()),
            "mixed_fraction": float(mixed.mean()),
            "mixed_tp_share": float(
                table.intersection_px[mixed].sum() / table.intersection_px.sum()
            ),
            "mixed_fp_share": float(table.false_px[mixed].sum() / table.false_px.sum()),
            "tp_total": float(table.intersection_px.sum()),
            "fp_total": float(table.false_px.sum()),
        }
        print(
            f"{mode:18s} 单元 {len(table):7,}  中位面积 {table.area_px.median():6.1f}  "
            f"模糊体占比 {mixed.mean():.1%}  模糊体承载 TP {stats[mode]['mixed_tp_share']:.1%} / "
            f"FP {stats[mode]['mixed_fp_share']:.1%}"
        )

    (args.outdir / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_subobject_units.v1",
                "threshold_override": args.threshold_override,
                "min_area": args.min_area,
                "ring_radius": args.ring_radius,
                "marker_quantile": MARKER_QUANTILE,
                "theta_range_deg": [THETA_MIN_DEG, THETA_MIN_DEG + THETA_SPAN_DEG],
                "theta_fallback_deg": THETA_FALLBACK_DEG,
                "slope_break_width_deg": SLOPE_BREAK_WIDTH_DEG,
                "stats": stats,
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
