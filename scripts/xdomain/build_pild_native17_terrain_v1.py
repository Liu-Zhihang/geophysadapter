#!/usr/bin/env python3
"""Build the same buffered native17 Terrain contract for any georeferenced PILD member."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import rasterio

import build_dlr_native_terrain_v3_cache as shared
from build_sen12_native_terrain_v2_cache import FEATURE_NAMES, SCALE_ROLES


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DEM_DIR = PROJECT_ROOT / "raw_fullcopy/static/copdem_glo30_2021"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dem-dir", type=Path, default=DEFAULT_DEM_DIR)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--buffer-m", type=float, default=3000.0)
    parser.add_argument("--native-resolution-m", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.buffer_m < 1000.0:
        raise ValueError("native17 Terrain requires at least 1000 m buffered context")
    if not args.base_h5.is_file():
        raise FileNotFoundError(args.base_h5)
    if not args.registry.is_file():
        raise FileNotFoundError(args.registry)
    if not args.dem_dir.is_dir():
        raise FileNotFoundError(args.dem_dir)

    registry = shared.read_registry(args.registry)
    with h5py.File(args.base_h5, "r") as handle:
        sample_ids = shared.decode(handle["sample_id"][:])
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    missing_rows = [sample_id for sample_id in sample_ids if sample_id not in registry]
    if missing_rows:
        raise RuntimeError(
            f"{args.dataset_id}: {len(missing_rows)} base samples absent from registry; "
            f"examples={missing_rows[:5]}"
        )

    available_tiles = {
        path.name for path in args.dem_dir.glob("Copernicus_DSM_COG_10_*_DEM.tif")
    }
    required_tiles: set[str] = set()
    missing_tiles: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        row = registry[sample_id]
        target = shared.target_bounds(row)
        projected = (
            target[0] - args.buffer_m,
            target[1] - args.buffer_m,
            target[2] + args.buffer_m,
            target[3] + args.buffer_m,
        )
        names = shared.required_names(
            "copdem", shared.geographic_bounds(projected, row["target_crs"])
        )
        required_tiles.update(names)
        missing = [name for name in names if name not in available_tiles]
        if missing:
            missing_tiles[sample_id] = missing
    if missing_tiles:
        unique_missing = sorted(
            {name for names in missing_tiles.values() for name in names}
        )
        raise FileNotFoundError(
            f"{args.dataset_id}: buffered native17 coverage misses {unique_missing}; "
            f"affected_samples={len(missing_tiles)}"
        )

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    string_type = h5py.string_dtype("utf-8")
    tasks = [
        (
            index,
            sample_id,
            registry[sample_id],
            args.buffer_m,
            args.native_resolution_m,
        )
        for index, sample_id in enumerate(sample_ids)
    ]

    with h5py.File(temporary, "w") as output:
        output.attrs.update(
            {
                "complete": 0,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_id": args.dataset_id,
                "terrain_contract": "pild_native17_v1",
                "terrain_source": "Copernicus DEM GLO-30 approximately 30 m DSM",
                "source_kind": "copdem",
                "source_registry": str(args.registry.resolve()),
                "source_registry_sha256": shared.sha256_file(args.registry),
                "base_h5": str(args.base_h5.resolve()),
                "base_h5_sha256": shared.sha256_file(args.base_h5),
                "derivative_policy": (
                    "derive 17 variables on native projected 30 m DEM with 3 km "
                    "buffered context, then bilinearly align to the 10 m prediction grid"
                ),
                "hydrology_policy": (
                    "TWI/flow accumulation/HAND excluded until watershed-complete routing"
                ),
                "buffer_m": args.buffer_m,
                "native_resolution_m": args.native_resolution_m,
                "target_resolution_m": 10.0,
                "required_source_tiles": len(required_tiles),
            }
        )
        output.create_dataset(
            "sample_id", data=np.asarray(sample_ids, dtype=object), dtype=string_type
        )
        output.create_dataset(
            "terrain_names",
            data=np.asarray(FEATURE_NAMES, dtype=object),
            dtype=string_type,
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

        def commit(result, completed: int) -> None:
            index, sample_id, terrain, valid, names = result
            terrain_ds[index] = terrain.astype(np.float16)
            valid_ds[index] = valid
            q_ds[index] = float(valid.mean())
            tiles_ds[index] = ";".join(names)
            output.attrs["completed_samples"] = completed
            if completed % max(args.flush_every, 1) == 0 or completed == len(tasks):
                output.flush()
                print(
                    f"[pild-native17:{args.dataset_id}] "
                    f"{completed}/{len(tasks)} {sample_id}",
                    flush=True,
                )

        if args.workers <= 1:
            with ExitStack() as stack:
                sources = {
                    path.name: stack.enter_context(rasterio.open(path))
                    for path in sorted(
                        args.dem_dir.glob("Copernicus_DSM_COG_10_*_DEM.tif")
                    )
                    if path.name in required_tiles
                }
                for completed, task in enumerate(tasks, start=1):
                    index, sample_id, row, buffer_m, native_resolution_m = task
                    terrain, valid, names = shared.build_sample(
                        row,
                        "copdem",
                        sources,
                        buffer_m,
                        native_resolution_m,
                    )
                    commit((index, sample_id, terrain, valid, names), completed)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=shared.initialize_worker,
                initargs=(str(args.dem_dir), "copdem"),
            ) as pool:
                futures = [pool.submit(shared.process_sample, task) for task in tasks]
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
                "dataset_id": args.dataset_id,
                "out": str(args.out.resolve()),
                "samples": len(sample_ids),
                "features": len(FEATURE_NAMES),
                "required_source_tiles": len(required_tiles),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
