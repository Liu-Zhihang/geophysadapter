#!/usr/bin/env python3
"""Build buffered native17 Terrain caches for DLR from CopDEM or FABDEM."""

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

import h5py
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject

from build_sen12_native_terrain_v2_cache import (
    FEATURE_NAMES,
    SCALE_ROLES,
    derive_features,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BASE = (
    PROJECT_ROOT
    / "processed/hybrid_pinn/pild_core_geo_v2_1_native30_raw/dlr_postrgb_terrain_raw_p128.h5"
)
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "processed/hybrid_pinn/pild_core_geo_v2_1_native30_raw/dlr_window_registry_v2.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source", choices=("copdem", "fabdem"), required=True)
    parser.add_argument("--dem-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--buffer-m", type=float, default=3000.0)
    parser.add_argument("--native-resolution-m", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def parse_affine(text: str) -> Affine:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 6:
        raise ValueError(f"invalid target_transform: {text}")
    transform = Affine(*values)
    if abs(transform.a - 10.0) > 1e-3 or abs(transform.e + 10.0) > 1e-3:
        raise RuntimeError(f"unexpected DLR target grid: {transform}")
    return transform


def tile_stem(latitude: int, longitude: int) -> str:
    lat = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}"
    lon = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}"
    return f"{lat}{lon}"


def tile_filename(source: str, latitude: int, longitude: int) -> str:
    stem = tile_stem(latitude, longitude)
    if source == "fabdem":
        return f"{stem}_FABDEM_V1-2.tif"
    lat, lon = stem[:3], stem[3:]
    return f"Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif"


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "target_crs", "target_transform"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"invalid DLR window registry: {path}")
    output = {row["sample_id"]: row for row in rows}
    if len(output) != len(rows):
        raise RuntimeError("duplicate sample_id in DLR window registry")
    return output


def target_bounds(row: dict[str, str]) -> tuple[float, float, float, float]:
    transform = parse_affine(row["target_transform"])
    corners = (
        transform * (0, 0),
        transform * (128, 0),
        transform * (0, 128),
        transform * (128, 128),
    )
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def geographic_bounds(
    projected_bounds: tuple[float, float, float, float], target_crs: str
) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    points = [
        transformer.transform(x, y)
        for x, y in (
            (projected_bounds[0], projected_bounds[1]),
            (projected_bounds[0], projected_bounds[3]),
            (projected_bounds[2], projected_bounds[1]),
            (projected_bounds[2], projected_bounds[3]),
        )
    ]
    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def required_names(
    source: str, bounds: tuple[float, float, float, float]
) -> list[str]:
    left, bottom, right, top = bounds
    return [
        tile_filename(source, latitude, longitude)
        for latitude in range(math.floor(bottom), math.ceil(top))
        for longitude in range(math.floor(left), math.ceil(right))
    ]


def build_sample(
    row: dict[str, str],
    source_kind: str,
    sources: dict[str, rasterio.DatasetReader],
    buffer_m: float,
    native_resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_crs = row["target_crs"]
    target = target_bounds(row)
    projected = (
        target[0] - buffer_m,
        target[1] - buffer_m,
        target[2] + buffer_m,
        target[3] + buffer_m,
    )
    geographic = geographic_bounds(projected, target_crs)
    names = required_names(source_kind, geographic)
    missing = [name for name in names if name not in sources]
    if missing:
        raise FileNotFoundError(f"missing buffered {source_kind} tiles: {missing}")
    mosaic, mosaic_transform = merge(
        [sources[name] for name in names],
        bounds=geographic,
        masked=True,
        resampling=Resampling.bilinear,
    )
    source_dem = np.asarray(mosaic[0].filled(np.nan), dtype=np.float32)
    width_native = max(4, int(math.ceil((projected[2] - projected[0]) / native_resolution_m)))
    height_native = max(4, int(math.ceil((projected[3] - projected[1]) / native_resolution_m)))
    native_transform = from_origin(
        projected[0], projected[3], native_resolution_m, native_resolution_m
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
    features = derive_features(native_dem, native_valid, native_resolution_m)
    output = np.zeros((len(FEATURE_NAMES), 128, 128), dtype=np.float32)
    target_transform = parse_affine(row["target_transform"])
    for channel in range(len(FEATURE_NAMES)):
        reproject(
            source=features[channel],
            destination=output[channel],
            src_transform=native_transform,
            src_crs=target_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            src_nodata=0.0,
            dst_nodata=0.0,
            resampling=Resampling.bilinear,
        )
    valid = np.zeros((128, 128), dtype=np.uint8)
    reproject(
        source=native_valid.astype(np.uint8),
        destination=valid,
        src_transform=native_transform,
        src_crs=target_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    output[:, valid == 0] = 0.0
    return output, valid[None], names


_WORKER_SOURCES: dict[str, rasterio.DatasetReader] = {}
_WORKER_SOURCE_KIND = ""


def initialize_worker(dem_dir: str, source_kind: str) -> None:
    global _WORKER_SOURCES, _WORKER_SOURCE_KIND
    _WORKER_SOURCE_KIND = source_kind
    pattern = "*_FABDEM_V1-2.tif" if source_kind == "fabdem" else "Copernicus_DSM_COG_10_*_DEM.tif"
    _WORKER_SOURCES = {
        path.name: rasterio.open(path) for path in sorted(Path(dem_dir).glob(pattern))
    }
    if not _WORKER_SOURCES:
        raise RuntimeError(f"no {source_kind} tiles in {dem_dir}")


def process_sample(
    task: tuple[int, str, dict[str, str], float, float],
) -> tuple[int, str, np.ndarray, np.ndarray, list[str]]:
    index, sample_id, row, buffer_m, native_resolution_m = task
    terrain, valid, names = build_sample(
        row,
        _WORKER_SOURCE_KIND,
        _WORKER_SOURCES,
        buffer_m,
        native_resolution_m,
    )
    return index, sample_id, terrain, valid, names


def main() -> int:
    args = parse_args()
    if args.buffer_m < 1000.0:
        raise ValueError("native17 Terrain requires at least 1000 m buffered context")
    dem_dir = args.dem_dir
    if dem_dir is None:
        dem_dir = (
            PROJECT_ROOT / "raw_fullcopy/static/fabdem_v1_2_dlr"
            if args.source == "fabdem"
            else PROJECT_ROOT / "raw_fullcopy/static/copdem_glo30_2021"
        )
    registry = read_registry(args.registry)
    with h5py.File(args.base_h5, "r") as handle:
        sample_ids = decode(handle["sample_id"][:])
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    missing_rows = [sample_id for sample_id in sample_ids if sample_id not in registry]
    if missing_rows:
        raise RuntimeError(f"base samples absent from DLR registry: {missing_rows[:5]}")
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    string_type = h5py.string_dtype("utf-8")
    tasks = [
        (index, sample_id, registry[sample_id], args.buffer_m, args.native_resolution_m)
        for index, sample_id in enumerate(sample_ids)
    ]
    with h5py.File(temporary, "w") as output:
        output.attrs.update(
            {
                "complete": 0,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "terrain_source": (
                    "FABDEM V1-2 approximately 30 m DTM-like surface"
                    if args.source == "fabdem"
                    else "Copernicus DEM GLO-30 approximately 30 m DSM"
                ),
                "source_kind": args.source,
                "source_registry": str(args.registry.resolve()),
                "source_registry_sha256": sha256_file(args.registry),
                "base_h5": str(args.base_h5.resolve()),
                "derivative_policy": (
                    "native projected 30 m, Gaussian sigma 0.75, 3 km buffered "
                    "context, then bilinear alignment to the DLR 10 m grid"
                ),
                "hydrology_policy": (
                    "TWI/flow accumulation/HAND excluded until watershed-complete routing"
                ),
                "buffer_m": args.buffer_m,
                "native_resolution_m": args.native_resolution_m,
                "target_resolution_m": 10.0,
            }
        )
        output.create_dataset(
            "sample_id", data=np.asarray(sample_ids, dtype=object), dtype=string_type
        )
        output.create_dataset(
            "terrain_names", data=np.asarray(FEATURE_NAMES, dtype=object), dtype=string_type
        )
        output.create_dataset(
            "terrain_scale_roles",
            data=np.asarray(SCALE_ROLES, dtype=object),
            dtype=string_type,
        )
        terrain_ds = output.create_dataset(
            "terrain",
            shape=(len(sample_ids), len(FEATURE_NAMES), 128, 128),
            dtype="float16",
            chunks=(1, len(FEATURE_NAMES), 128, 128),
            compression="lzf",
        )
        valid_ds = output.create_dataset(
            "terrain_valid",
            shape=(len(sample_ids), 1, 128, 128),
            dtype="uint8",
            chunks=(1, 1, 128, 128),
            compression="lzf",
        )
        q_ds = output.create_dataset("q_T", shape=(len(sample_ids),), dtype="float32")
        tiles_ds = output.create_dataset(
            "source_tiles", shape=(len(sample_ids),), dtype=string_type
        )

        def commit(result: tuple[int, str, np.ndarray, np.ndarray, list[str]], completed: int) -> None:
            index, sample_id, terrain, valid, names = result
            terrain_ds[index] = terrain.astype(np.float16)
            valid_ds[index] = valid
            q_ds[index] = float(valid.mean())
            tiles_ds[index] = ";".join(names)
            output.attrs["completed_samples"] = completed
            if completed % max(args.flush_every, 1) == 0 or completed == len(tasks):
                output.flush()
                print(
                    f"[dlr-terrain-v3:{args.source}] {completed}/{len(tasks)} {sample_id}",
                    flush=True,
                )

        if args.workers <= 1:
            pattern = (
                "*_FABDEM_V1-2.tif"
                if args.source == "fabdem"
                else "Copernicus_DSM_COG_10_*_DEM.tif"
            )
            with ExitStack() as stack:
                sources = {
                    path.name: stack.enter_context(rasterio.open(path))
                    for path in sorted(dem_dir.glob(pattern))
                }
                for completed, task in enumerate(tasks, start=1):
                    index, sample_id, row, buffer_m, native_resolution_m = task
                    terrain, valid, names = build_sample(
                        row, args.source, sources, buffer_m, native_resolution_m
                    )
                    commit((index, sample_id, terrain, valid, names), completed)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=initialize_worker,
                initargs=(str(dem_dir), args.source),
            ) as pool:
                futures = [pool.submit(process_sample, task) for task in tasks]
                for completed, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    commit(future.result(), completed)
        output.attrs["complete"] = 1
        output.attrs["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.flush()
    os.replace(temporary, args.out)
    print(
        json.dumps(
            {
                "status": "complete",
                "source": args.source,
                "out": str(args.out.resolve()),
                "samples": len(sample_ids),
                "features": len(FEATURE_NAMES),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
