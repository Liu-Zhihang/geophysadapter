#!/usr/bin/env python3
"""Build native-scale, buffered CopDEM geomorphometry for Sen12 Terrain-v2.

Unlike the legacy cache, derivatives are computed on a projected 30 m grid
with a multi-kilometre buffer. Only the completed derivative rasters are
reprojected to the 10 m prediction grid. Hydrological accumulation variables
are deliberately excluded until a watershed-complete flow-routing product is
available; patch-local TWI would violate the physical support contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_NAMES = (
    "elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "profile_curvature",
    "plan_curvature",
    "laplacian_curvature",
    "tpi_90m",
    "tpi_300m",
    "tpi_900m",
    "local_std_90m",
    "local_std_300m",
    "local_relief_300m",
    "local_relief_900m",
    "valley_depth_900m",
    "ridge_height_900m",
    "ruggedness_90m",
)
SCALE_ROLES = (
    "macro",
    "fine",
    "fine",
    "fine",
    "fine",
    "fine",
    "fine",
    "fine",
    "meso",
    "macro",
    "fine",
    "meso",
    "meso",
    "macro",
    "macro",
    "macro",
    "fine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--base-h5", type=Path, default=None)
    parser.add_argument("--dem-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--buffer-m", type=float, default=3000.0)
    parser.add_argument("--native-resolution-m", type=float, default=30.0)
    parser.add_argument("--target-resolution-m", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def tile_name(latitude: int, longitude: int) -> str:
    lat = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}_00"
    lon = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}_00"
    return f"Copernicus_DSM_COG_10_{lat}_{lon}_DEM.tif"


def tile_names_for_bounds(bounds: tuple[float, float, float, float]) -> list[str]:
    left, bottom, right, top = bounds
    return [
        tile_name(latitude, longitude)
        for latitude in range(math.floor(bottom), math.ceil(top))
        for longitude in range(math.floor(left), math.ceil(right))
    ]


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "crs", "min_x", "min_y", "max_x", "max_y", "width", "height"}
    if not rows:
        raise RuntimeError("empty Sen12 registry")
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"registry missing columns: {sorted(missing)}")
    output = {row["sample_id"]: row for row in rows}
    if len(output) != len(rows):
        raise RuntimeError("duplicate sample_id in Sen12 registry")
    return output


def fill_nearest(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        raise RuntimeError("native DEM window has no valid pixels")
    if valid.all():
        return array.astype(np.float32, copy=False)
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.asarray(array, dtype=np.float32)[tuple(indices)]


def odd_window(distance_m: float, resolution_m: float) -> int:
    pixels = max(1, int(round(distance_m / resolution_m)))
    return pixels if pixels % 2 else pixels + 1


def local_mean(array: np.ndarray, window: int) -> np.ndarray:
    return ndimage.uniform_filter(array, size=window, mode="reflect")


def local_std(array: np.ndarray, window: int) -> np.ndarray:
    mean = local_mean(array, window)
    mean_square = local_mean(np.square(array), window)
    return np.sqrt(np.maximum(mean_square - np.square(mean), 0.0))


def derive_features(dem: np.ndarray, valid: np.ndarray, resolution_m: float) -> np.ndarray:
    elevation = fill_nearest(dem, valid)
    smooth = ndimage.gaussian_filter(elevation, sigma=0.75, mode="reflect")
    dz_dy, dz_dx = np.gradient(smooth, resolution_m, resolution_m)
    gradient_sq = np.square(dz_dx) + np.square(dz_dy)
    gradient = np.sqrt(gradient_sq)
    slope = np.degrees(np.arctan(gradient))
    aspect = np.arctan2(-dz_dx, dz_dy)
    aspect_sin = np.sin(aspect)
    aspect_cos = np.cos(aspect)
    aspect_sin[gradient < 1e-7] = 0.0
    aspect_cos[gradient < 1e-7] = 0.0

    d2z_dx2 = np.gradient(dz_dx, resolution_m, axis=1)
    d2z_dy2 = np.gradient(dz_dy, resolution_m, axis=0)
    d2z_dxdy = np.gradient(dz_dx, resolution_m, axis=0)
    eps = 1e-8
    profile = -(
        d2z_dx2 * np.square(dz_dx)
        + 2.0 * d2z_dxdy * dz_dx * dz_dy
        + d2z_dy2 * np.square(dz_dy)
    ) / ((gradient_sq + eps) * np.power(1.0 + gradient_sq, 1.5))
    plan = (
        d2z_dx2 * np.square(dz_dy)
        - 2.0 * d2z_dxdy * dz_dx * dz_dy
        + d2z_dy2 * np.square(dz_dx)
    ) / (np.power(gradient_sq + eps, 1.5))
    laplacian = d2z_dx2 + d2z_dy2
    profile = np.clip(profile, -0.5, 0.5)
    plan = np.clip(plan, -0.5, 0.5)
    laplacian = np.clip(laplacian, -0.5, 0.5)

    w90 = odd_window(90.0, resolution_m)
    w300 = odd_window(300.0, resolution_m)
    w900 = odd_window(900.0, resolution_m)
    mean90 = local_mean(elevation, w90)
    mean300 = local_mean(elevation, w300)
    mean900 = local_mean(elevation, w900)
    min300 = ndimage.minimum_filter(elevation, size=w300, mode="reflect")
    max300 = ndimage.maximum_filter(elevation, size=w300, mode="reflect")
    min900 = ndimage.minimum_filter(elevation, size=w900, mode="reflect")
    max900 = ndimage.maximum_filter(elevation, size=w900, mode="reflect")
    ruggedness90 = local_mean(np.abs(elevation - mean90), w90)

    output = np.stack(
        (
            elevation,
            slope,
            aspect_sin,
            aspect_cos,
            profile,
            plan,
            laplacian,
            elevation - mean90,
            elevation - mean300,
            elevation - mean900,
            local_std(elevation, w90),
            local_std(elevation, w300),
            max300 - min300,
            max900 - min900,
            elevation - min900,
            max900 - elevation,
            ruggedness90,
        ),
        axis=0,
    ).astype(np.float32)
    if output.shape[0] != len(FEATURE_NAMES) or not np.isfinite(output).all():
        raise RuntimeError("invalid native Terrain-v2 feature stack")
    output[:, ~valid] = 0.0
    return output


def target_transform(row: dict[str, str], target_resolution_m: float) -> rasterio.Affine:
    width = int(row["width"])
    height = int(row["height"])
    min_x, min_y, max_x, max_y = (float(row[key]) for key in ("min_x", "min_y", "max_x", "max_y"))
    expected_width = (max_x - min_x) / max(width - 1, 1)
    expected_height = (max_y - min_y) / max(height - 1, 1)
    if not (abs(expected_width - target_resolution_m) <= 1.0 and abs(expected_height - target_resolution_m) <= 1.0):
        raise RuntimeError(f"unexpected target resolution: x={expected_width}, y={expected_height}")
    return from_bounds(
        min_x - expected_width / 2.0,
        min_y - expected_height / 2.0,
        max_x + expected_width / 2.0,
        max_y + expected_height / 2.0,
        width,
        height,
    )


def build_sample(
    row: dict[str, str],
    sources: dict[str, rasterio.DatasetReader],
    buffer_m: float,
    native_resolution_m: float,
    target_resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_crs = row["crs"]
    min_x, min_y, max_x, max_y = (float(row[key]) for key in ("min_x", "min_y", "max_x", "max_y"))
    projected_bounds = (
        min_x - buffer_m,
        min_y - buffer_m,
        max_x + buffer_m,
        max_y + buffer_m,
    )
    geographic_bounds = transform_bounds(target_crs, "EPSG:4326", *projected_bounds, densify_pts=21)
    names = tile_names_for_bounds(geographic_bounds)
    missing = [name for name in names if name not in sources]
    if missing:
        raise FileNotFoundError(f"missing buffered CopDEM tiles: {missing}")
    mosaic, mosaic_transform = merge(
        [sources[name] for name in names],
        bounds=geographic_bounds,
        masked=True,
        resampling=Resampling.bilinear,
    )
    source_dem = np.asarray(mosaic[0].filled(np.nan), dtype=np.float32)

    width_native = max(4, int(math.ceil((projected_bounds[2] - projected_bounds[0]) / native_resolution_m)))
    height_native = max(4, int(math.ceil((projected_bounds[3] - projected_bounds[1]) / native_resolution_m)))
    native_transform = from_origin(
        projected_bounds[0], projected_bounds[3], native_resolution_m, native_resolution_m
    )
    native_dem = np.full((height_native, width_native), np.nan, dtype=np.float32)
    reproject(
        source=source_dem,
        destination=native_dem,
        src_transform=mosaic_transform,
        src_crs="EPSG:4326",
        dst_transform=native_transform,
        dst_crs=target_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    native_valid = np.isfinite(native_dem) & (native_dem > -500.0) & (native_dem < 9000.0)
    features_native = derive_features(native_dem, native_valid, native_resolution_m)

    height = int(row["height"])
    width = int(row["width"])
    transform = target_transform(row, target_resolution_m)
    output = np.zeros((len(FEATURE_NAMES), height, width), dtype=np.float32)
    for channel in range(len(FEATURE_NAMES)):
        reproject(
            source=features_native[channel],
            destination=output[channel],
            src_transform=native_transform,
            src_crs=target_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            src_nodata=0.0,
            dst_nodata=0.0,
            resampling=Resampling.bilinear,
        )
    target_valid = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=native_valid.astype(np.uint8),
        destination=target_valid,
        src_transform=native_transform,
        src_crs=target_crs,
        dst_transform=transform,
        dst_crs=target_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    output[:, target_valid == 0] = 0.0
    return output, target_valid[None], names


_WORKER_SOURCES: dict[str, rasterio.DatasetReader] = {}


def initialize_worker(dem_dir: str) -> None:
    global _WORKER_SOURCES
    _WORKER_SOURCES = {
        path.name: rasterio.open(path)
        for path in sorted(Path(dem_dir).glob("Copernicus_DSM_COG_10_*_DEM.tif"))
    }
    if not _WORKER_SOURCES:
        raise RuntimeError(f"worker found no CopDEM tiles in {dem_dir}")


def process_sample_task(
    task: tuple[int, str, dict[str, str], float, float, float],
) -> tuple[int, str, np.ndarray, np.ndarray, list[str]]:
    index, sample_id, row, buffer_m, native_resolution_m, target_resolution_m = task
    terrain, valid, names = build_sample(
        row,
        _WORKER_SOURCES,
        buffer_m,
        native_resolution_m,
        target_resolution_m,
    )
    return index, sample_id, terrain, valid, names


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    registry_path = (
        args.registry.resolve()
        if args.registry
        else root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
    )
    base_h5 = (
        args.base_h5.resolve()
        if args.base_h5
        else root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5"
    )
    dem_dir = (
        args.dem_dir.resolve()
        if args.dem_dir
        else root / "raw_fullcopy/static/copdem_glo30_2021"
    )
    output_path = (
        args.out.resolve()
        if args.out
        else root / "processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_native_terrain_v2_p128.h5"
    )
    if args.buffer_m < 1000.0:
        raise ValueError("Terrain-v2 requires at least 1000 m native context")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output_path}")

    registry = read_registry(registry_path)
    with h5py.File(base_h5, "r") as handle:
        sample_ids = decode_strings(handle["sample_id"][:])
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]
    missing_rows = [sample_id for sample_id in sample_ids if sample_id not in registry]
    if missing_rows:
        raise RuntimeError(f"base cache samples absent from registry: {missing_rows[:5]}")

    tile_paths = sorted(dem_dir.glob("Copernicus_DSM_COG_10_*_DEM.tif"))
    if not tile_paths:
        raise RuntimeError(f"no CopDEM tiles found in {dem_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    string_type = h5py.string_dtype("utf-8")

    with h5py.File(temporary, "w") as output:
            output.attrs.update(
                {
                    "complete": 0,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_registry": str(registry_path),
                    "source_registry_sha256": sha256_file(registry_path),
                    "base_h5": str(base_h5),
                    "base_h5_sha256": sha256_file(base_h5),
                    "terrain_source": "Copernicus DEM GLO-30 DSM, native approximately 30 m",
                    "derivative_policy": "native projected 30 m, Gaussian sigma 0.75, 3 km buffered context, then bilinear reprojection",
                    "hydrology_policy": "TWI/flow accumulation/HAND excluded until watershed-complete routing is available",
                    "native_resolution_m": args.native_resolution_m,
                    "target_resolution_m": args.target_resolution_m,
                    "buffer_m": args.buffer_m,
                }
            )
            output.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=object), dtype=string_type)
            output.create_dataset("terrain_names", data=np.asarray(FEATURE_NAMES, dtype=object), dtype=string_type)
            output.create_dataset("terrain_scale_roles", data=np.asarray(SCALE_ROLES, dtype=object), dtype=string_type)
            terrain_ds = output.create_dataset(
                "terrain",
                shape=(len(sample_ids), len(FEATURE_NAMES), 128, 128),
                dtype="float16",
                chunks=(1, len(FEATURE_NAMES), 128, 128),
                compression="lzf",
            )
            valid_ds = output.create_dataset(
                "terrain_valid", shape=(len(sample_ids), 1, 128, 128), dtype="uint8", chunks=(1, 1, 128, 128), compression="lzf"
            )
            q_ds = output.create_dataset("q_T", shape=(len(sample_ids),), dtype="float32")
            tile_ds = output.create_dataset("source_tiles", shape=(len(sample_ids),), dtype=string_type)

            tasks = [
                (
                    index,
                    sample_id,
                    registry[sample_id],
                    args.buffer_m,
                    args.native_resolution_m,
                    args.target_resolution_m,
                )
                for index, sample_id in enumerate(sample_ids)
            ]

            def commit(result: tuple[int, str, np.ndarray, np.ndarray, list[str]], completed: int) -> None:
                index, sample_id, terrain, valid, names = result
                terrain_ds[index] = terrain.astype(np.float16)
                valid_ds[index] = valid
                q_ds[index] = float(valid.mean())
                tile_ds[index] = ";".join(names)
                output.attrs["completed_samples"] = completed
                if completed % max(1, args.flush_every) == 0 or completed == len(sample_ids):
                    output.flush()
                    print(f"[terrain-v2] {completed}/{len(sample_ids)} {sample_id}", flush=True)

            if args.workers <= 1:
                with ExitStack() as stack:
                    sources = {
                        path.name: stack.enter_context(rasterio.open(path)) for path in tile_paths
                    }
                    for completed, task in enumerate(tasks, start=1):
                        index, sample_id, row, buffer_m, native_resolution_m, target_resolution_m = task
                        terrain, valid, names = build_sample(
                            row,
                            sources,
                            buffer_m,
                            native_resolution_m,
                            target_resolution_m,
                        )
                        commit((index, sample_id, terrain, valid, names), completed)
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.workers,
                    initializer=initialize_worker,
                    initargs=(str(dem_dir),),
                ) as pool:
                    futures = [pool.submit(process_sample_task, task) for task in tasks]
                    for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                        commit(future.result(), completed)
            output.attrs["complete"] = 1
            output.attrs["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            output.flush()
    temporary.replace(output_path)
    print(json.dumps({"output": str(output_path), "samples": len(sample_ids), "features": len(FEATURE_NAMES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
