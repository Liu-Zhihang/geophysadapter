#!/usr/bin/env python3
"""G10：对象级水文与坡位描述子。

缺口来源：现有 17 通道地形栈是 elevation / slope / aspect / 曲率 / TPI(90,300,900) /
local_std / local_relief / valley_depth / ridge_height / ruggedness，全部是**局部形态**量。
其中没有任何**流域拓扑**量——没有汇流累积，没有 HAND（河道以上高度），没有到河道的距离。

这恰好是主要假阳性类别的判别式。河床、漫滩、谷底道路、河漫滩耕地在局部形态上可以
不平坦、不低洼，因而 TPI 与 valley_depth 未必抓得住；但它们在流域拓扑上无一例外地
位于极高汇流累积、极低 HAND 的位置。反过来，真实滑坡体的物源区位于坡体中上部，
具有中等 HAND 与低汇流累积，并沿坡向延伸到较低的坡位。

本脚本用缓存中的 elevation 通道，逐样本计算：
    D8 汇流累积       按高程降序扫描，每格把自身与上游流量交给最陡下坡邻居
    河道掩膜         汇流累积超过 100 格（1 ha）视为河道
    HAND             按高程升序沿受水方向上溯传播，得到该格相对其下游河道的高差
    到河道距离        河道掩膜的欧氏距离变换
    归一化坡位        (elev - 样本内最低) / (样本内高差)

再对每个候选体聚合，并给出体与其外扩 5 px 环带的对比。
"""

from __future__ import annotations

import argparse
import json
import time
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

ELEVATION_CHANNEL = 0
PIXEL_METRES = 10.0
CHANNEL_ACCUM_CELLS = 100.0
MIN_AREA = 4
RING_RADIUS = 5

# D8 的 8 个邻居偏移与对应距离
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGHBOUR_DIST = np.asarray(
    [np.hypot(dr, dc) for dr, dc in NEIGHBOURS], dtype=np.float32
) * PIXEL_METRES


def d8_receivers(elevation: np.ndarray) -> np.ndarray:
    """每格的最陡下坡邻居的扁平索引；无下坡邻居（洼地/出口）时指向自身。"""
    height, width = elevation.shape
    flat = elevation.ravel()
    index = np.arange(flat.size).reshape(height, width)
    best_slope = np.zeros_like(elevation, dtype=np.float32)
    receiver = index.copy()

    for k, (dr, dc) in enumerate(NEIGHBOURS):
        shifted = np.full_like(elevation, np.inf, dtype=np.float32)
        shifted_index = np.full((height, width), -1, dtype=np.int64)
        r0, r1 = max(dr, 0), height + min(dr, 0)
        c0, c1 = max(dc, 0), width + min(dc, 0)
        shifted[r0:r1, c0:c1] = elevation[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
        shifted_index[r0:r1, c0:c1] = index[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
        drop = (elevation - shifted) / NEIGHBOUR_DIST[k]
        better = (drop > best_slope) & (shifted_index >= 0)
        best_slope = np.where(better, drop, best_slope)
        receiver = np.where(better, shifted_index, receiver)
    return receiver.ravel()


def topological_levels(receiver: np.ndarray) -> list[np.ndarray]:
    """Kahn 分层：第 0 层是无上游的源格，逐层向下游推进。

    受水关系构成一片森林，故层数等于最长流路长度（约 O(边长)），
    每层内部可以整体向量化，避免逐格 Python 循环。
    """
    size = receiver.size
    cells = np.arange(size)
    downstream = receiver != cells
    indegree = np.bincount(receiver[downstream], minlength=size)
    frontier = cells[indegree == 0]
    levels = []
    while frontier.size:
        levels.append(frontier)
        moving = frontier[receiver[frontier] != frontier]
        if moving.size == 0:
            break
        targets = receiver[moving]
        decrement = np.bincount(targets, minlength=size)
        indegree = indegree - decrement
        touched = np.unique(targets)
        frontier = touched[indegree[touched] == 0]
    return levels


def flow_accumulation(receiver: np.ndarray, levels: list[np.ndarray]) -> np.ndarray:
    """逐层把流量交给下游，得到以格数为单位的汇流累积。"""
    accumulation = np.ones(receiver.size, dtype=np.float32)
    for level in levels:
        moving = level[receiver[level] != level]
        if moving.size:
            np.add.at(accumulation, receiver[moving], accumulation[moving])
    return accumulation


def height_above_drainage(
    elevation_flat: np.ndarray,
    receiver: np.ndarray,
    channel_flat: np.ndarray,
    levels: list[np.ndarray],
) -> np.ndarray:
    """由下游向上游传播所属河道的高程，差值即 HAND。

    逆着拓扑层序处理：某格的受水格必定位于更高层，因此先解算高层即可整层向量化。
    """
    drain = np.where(channel_flat, elevation_flat, np.nan).astype(np.float32)
    for level in reversed(levels):
        need = level[~channel_flat[level]]
        if need.size == 0:
            continue
        targets = receiver[need]
        outlet = targets == need
        drain[need[outlet]] = elevation_flat[need[outlet]]
        inner = need[~outlet]
        if inner.size:
            drain[inner] = drain[receiver[inner]]
    hand = elevation_flat - drain
    return np.clip(np.where(np.isfinite(hand), hand, 0.0), 0.0, None)


def hydrology_stack(elevation: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """单样本的流域拓扑量。"""
    surface = elevation.astype(np.float32).copy()
    if valid.any():
        surface[~valid] = float(surface[valid].max()) + 1.0
    receiver = d8_receivers(surface)
    levels = topological_levels(receiver)
    accumulation = flow_accumulation(receiver, levels).reshape(surface.shape)
    channel = accumulation >= CHANNEL_ACCUM_CELLS
    hand = height_above_drainage(
        surface.ravel(), receiver, channel.ravel(), levels
    ).reshape(surface.shape)
    if channel.any():
        distance = ndimage.distance_transform_edt(~channel) * PIXEL_METRES
    else:
        distance = np.full(surface.shape, 128.0 * PIXEL_METRES, dtype=np.float32)
    if valid.any():
        low = float(surface[valid].min())
        span = float(surface[valid].max() - low)
    else:
        low, span = 0.0, 1.0
    position = (surface - low) / (span if span > 1e-6 else 1.0)
    return {
        "log_accum": np.log10(accumulation + 1.0).astype(np.float32),
        "hand": hand.astype(np.float32),
        "channel_distance": distance.astype(np.float32),
        "slope_position": np.clip(position, 0.0, 1.0).astype(np.float32),
        "channel": channel,
    }


def component_hydrology(
    mask: np.ndarray, window: tuple[slice, slice], stack: dict[str, np.ndarray],
    valid: np.ndarray, ring_radius: int,
) -> dict[str, float]:
    """单个候选体的水文描述子，含外扩环带对比。"""
    row_slice, col_slice = window
    r0 = max(row_slice.start - ring_radius, 0)
    r1 = min(row_slice.stop + ring_radius, valid.shape[0])
    c0 = max(col_slice.start - ring_radius, 0)
    c1 = min(col_slice.stop + ring_radius, valid.shape[1])
    big = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    big[row_slice.start - r0 : row_slice.stop - r0, col_slice.start - c0 : col_slice.stop - c0] = mask
    ring = ndimage.binary_dilation(big, iterations=ring_radius) & ~big & valid[r0:r1, c0:c1]
    sub = (slice(r0, r1), slice(c0, c1))

    out: dict[str, float] = {}
    for key in ("log_accum", "hand", "channel_distance", "slope_position"):
        values = stack[key][sub][big]
        out[f"hyd_mean_{key}"] = float(values.mean())
        out[f"hyd_p10_{key}"] = float(np.percentile(values, 10))
        out[f"hyd_p90_{key}"] = float(np.percentile(values, 90))
        out[f"hyd_range_{key}"] = float(out[f"hyd_p90_{key}"] - out[f"hyd_p10_{key}"])
        if ring.any():
            out[f"hyd_contrast_{key}"] = float(
                values.mean() - stack[key][sub][ring].mean()
            )
        else:
            out[f"hyd_contrast_{key}"] = np.nan
    channel_local = stack["channel"][sub][big]
    out["hyd_channel_fraction"] = float(channel_local.mean())
    # 真实滑坡体应跨越坡位：物源区高、堆积区低
    out["hyd_position_span"] = float(
        np.percentile(stack["slope_position"][sub][big], 95)
        - np.percentile(stack["slope_position"][sub][big], 5)
    )
    # 低 HAND 且高汇流：典型的河床或漫滩假阳性
    out["hyd_floodplain_fraction"] = float(
        np.mean((stack["hand"][sub][big] < 5.0) & (stack["log_accum"][sub][big] > 2.0))
    )
    return out


def process_fold(cache_dir: Path, fold_id: str, min_area: int, ring_radius: int) -> list[dict]:
    receipt = json.loads(
        (cache_dir / f"{fold_id}_oof_cache_receipt.json").read_text(encoding="utf-8")
    )
    threshold = float(receipt["threshold"])
    with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
        sample_id = [str(item) for item in handle["sample_id"]]
        probability_all = handle["visual_probability"]
        valid_all = handle["valid"]
        terrain_all = handle["terrain"]

    structure = ndimage.generate_binary_structure(2, 2)
    rows: list[dict] = []
    for index in range(len(sample_id)):
        keep = valid_all[index].astype(bool)
        predicted = (probability_all[index].astype(np.float32) >= threshold) & keep
        if not predicted.any():
            continue
        labels, count = ndimage.label(predicted, structure=structure)
        if count == 0:
            continue
        stack = hydrology_stack(terrain_all[index][ELEVATION_CHANNEL], keep)
        windows = ndimage.find_objects(labels)
        for label_value in range(1, count + 1):
            window = windows[label_value - 1]
            local = labels[window] == label_value
            if int(local.sum()) < min_area:
                continue
            row = component_hydrology(local, window, stack, keep, ring_radius)
            row["sample_id"] = sample_id[index]
            row["component_id"] = int(label_value)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--min-area", type=int, default=MIN_AREA)
    parser.add_argument("--ring-radius", type=int, default=RING_RADIUS)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_hydrology_features_v1",
    )
    args = parser.parse_args()
    started = time.time()

    frames = []
    for fold_id in FOLD_IDS:
        rows = process_fold(args.cache, fold_id, args.min_area, args.ring_radius)
        print(f"{fold_id}: {len(rows):,} 个对象")
        frames.append(pd.DataFrame(rows))
    table = pd.concat(frames, ignore_index=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.outdir / "object_hydrology_features.parquet", index=False)
    feature_cols = [c for c in table.columns if c.startswith("hyd_")]
    (args.outdir / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_hydrology_features.v1",
                "channel_accum_cells": CHANNEL_ACCUM_CELLS,
                "pixel_metres": PIXEL_METRES,
                "min_area": args.min_area,
                "ring_radius": args.ring_radius,
                "n_objects": int(len(table)),
                "features": feature_cols,
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n共 {len(table):,} 个对象，{len(feature_cols)} 个水文描述子")
    print(f"写出 {args.outdir}")


if __name__ == "__main__":
    main()
