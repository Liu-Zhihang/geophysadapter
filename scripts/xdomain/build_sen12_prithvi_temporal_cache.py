#!/usr/bin/env python3
"""Build the four-date, six-band Sen12 sidecar for Prithvi-EO-2.0.

The cache follows the already frozen 4,979-sample H5 ordering. It stores only
visual inputs and coordinates; labels and Terrain remain in independently
auditable H5 files. Two observations immediately before the event span and two
at/after it are selected. Boundary repetition is explicit when only one valid
observation exists on a side.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import netCDF4
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
CLOUD_CODES = (3, 8, 9, 10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--base-h5", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def read_registry(path: Path, root: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "sample_id", "relative_path", "event_date_start", "event_date_end",
        "center_lat", "center_lon", "change_view_eligible",
    }
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"registry missing fields: {sorted(required - set(rows[0] if rows else []))}")
    output = {}
    for row in rows:
        row["source_path"] = str((root.parent / row["relative_path"]).resolve())
        output[row["sample_id"]] = row
    if len(output) != len(rows):
        raise RuntimeError("duplicate sample_id in registry")
    return output


def iso_date(value: Any) -> str:
    return f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}"


def choose_four_dates(dates: list[str], event_start: str, event_end: str) -> tuple[list[int], list[int]]:
    pre = [index for index, date in enumerate(dates) if date < event_start]
    post = [index for index, date in enumerate(dates) if date >= event_end]
    if not pre or not post:
        raise RuntimeError(
            f"event span is not bracketed: start={event_start}, end={event_end}, dates={dates}"
        )
    pre_selected = pre[-2:] if len(pre) >= 2 else [pre[-1], pre[-1]]
    post_selected = post[:2] if len(post) >= 2 else [post[0], post[0]]
    indices = pre_selected + post_selected
    duplicated = [
        int(pre_selected[0] == pre_selected[1]),
        int(pre_selected[0] == pre_selected[1]),
        int(post_selected[0] == post_selected[1]),
        int(post_selected[0] == post_selected[1]),
    ]
    return indices, duplicated


def read_time_stack(variable: netCDF4.Variable, indices: list[int]) -> np.ndarray:
    dimensions = list(variable.dimensions)
    if set(dimensions) != {"time", "x", "y"} or len(dimensions) != 3:
        raise RuntimeError(
            f"{variable.name} must have time/x/y dimensions, got {dimensions}"
        )
    selection: list[Any] = [slice(None)] * 3
    selection[dimensions.index("time")] = indices
    value = variable[tuple(selection)]
    if np.ma.isMaskedArray(value):
        value = value.astype(np.float32).filled(np.nan)
    array = np.asarray(value)
    current_dimensions = ["time", *[name for name in dimensions if name != "time"]]
    if dimensions.index("time") != 0:
        array = np.moveaxis(array, dimensions.index("time"), 0)
    array = np.transpose(
        array,
        (
            current_dimensions.index("time"),
            current_dimensions.index("y"),
            current_dimensions.index("x"),
        ),
    )
    if array.shape != (4, 128, 128):
        raise RuntimeError(f"{variable.name} has unexpected selected shape {array.shape}")
    return array


def process_task(task: tuple[int, str, dict[str, str]]) -> tuple[int, str, dict[str, Any]]:
    index, sample_id, row = task
    path = Path(row["source_path"])
    with netCDF4.Dataset(path, "r") as dataset:
        missing = set(BANDS + ("SCL", "time")) - set(dataset.variables)
        if missing:
            raise RuntimeError(f"{sample_id}: missing NetCDF variables {sorted(missing)}")
        time = dataset.variables["time"]
        calendar = getattr(time, "calendar", "standard")
        dates = [iso_date(value) for value in netCDF4.num2date(time[:], time.units, calendar)]
        indices, duplicated = choose_four_dates(
            dates, row["event_date_start"], row["event_date_end"]
        )
        optical = np.stack(
            [
                np.asarray(read_time_stack(dataset.variables[band], indices), dtype=np.float32)
                for band in BANDS
            ],
            axis=0,
        )
        if optical.shape[1:] != (4, 128, 128):
            raise RuntimeError(f"{sample_id}: unexpected optical shape {optical.shape}")
        finite = np.isfinite(optical) & (optical >= 0.0) & (optical <= 10_000.0)
        valid = finite.all(axis=(0, 1)).astype(np.uint8)[None]
        optical = np.clip(np.nan_to_num(optical, nan=0.0), 0.0, 10_000.0).astype(np.uint16)
        scl = np.asarray(read_time_stack(dataset.variables["SCL"], indices), dtype=np.int16)
        scl = np.clip(scl, 0, 255).astype(np.uint8)
        selected_dates = [dates[item] for item in indices]
        temporal = np.asarray(
            [
                [int(date[:4]), datetime.fromisoformat(date).timetuple().tm_yday]
                for date in selected_dates
            ],
            dtype=np.int16,
        )
    return index, sample_id, {
        "optical": optical,
        "valid": valid,
        "scl": scl,
        "indices": np.asarray(indices, dtype=np.int16),
        "duplicated": np.asarray(duplicated, dtype=np.uint8),
        "temporal": temporal,
        "location": np.asarray([float(row["center_lat"]), float(row["center_lon"])], dtype=np.float32),
        "dates": ";".join(selected_dates),
        "cloud_fraction": np.isin(scl, CLOUD_CODES).mean(axis=(1, 2)).astype(np.float32),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    registry_path = (args.registry or root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv").resolve()
    base_h5 = (args.base_h5 or root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5").resolve()
    output_path = (
        args.out
        or root / "processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_prithvi_4t6b_p128.h5"
    ).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output_path}")
    registry = read_registry(registry_path, root)
    with h5py.File(base_h5, "r") as handle:
        sample_ids = decode_strings(handle["sample_id"][:])
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    if any(sample_id not in registry for sample_id in sample_ids):
        raise RuntimeError("frozen base H5 contains sample IDs absent from registry")
    if any(registry[sample_id]["change_view_eligible"] != "1" for sample_id in sample_ids):
        raise RuntimeError("base H5 includes non-eligible change-view sample")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    text_type = h5py.string_dtype("utf-8")
    tasks = [(index, sample_id, registry[sample_id]) for index, sample_id in enumerate(sample_ids)]
    with h5py.File(temporary, "w") as output:
        output.attrs.update(
            {
                "complete": 0,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_registry": str(registry_path),
                "base_h5": str(base_h5),
                "selection_contract": "two closest observations before event span plus two earliest at/after event span; explicit boundary repetition",
                "bands": ";".join(BANDS),
                "scaling": "raw S2 surface reflectance clipped to [0,10000] and stored uint16",
                "coordinate_contract": "temporal=[year,day_of_year], location=[latitude,longitude]",
            }
        )
        output.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=object), dtype=text_type)
        optical_ds = output.create_dataset(
            "optical", shape=(len(sample_ids), 6, 4, 128, 128), dtype="uint16",
            chunks=(1, 6, 1, 128, 128), compression="lzf",
        )
        valid_ds = output.create_dataset(
            "optical_valid", shape=(len(sample_ids), 1, 128, 128), dtype="uint8",
            chunks=(1, 1, 128, 128), compression="lzf",
        )
        scl_ds = output.create_dataset(
            "scl", shape=(len(sample_ids), 4, 128, 128), dtype="uint8",
            chunks=(1, 4, 128, 128), compression="lzf",
        )
        temporal_ds = output.create_dataset("temporal_coords", shape=(len(sample_ids), 4, 2), dtype="int16")
        location_ds = output.create_dataset("location_coords", shape=(len(sample_ids), 2), dtype="float32")
        index_ds = output.create_dataset("selected_indices", shape=(len(sample_ids), 4), dtype="int16")
        duplicate_ds = output.create_dataset("duplicated_observation", shape=(len(sample_ids), 4), dtype="uint8")
        cloud_ds = output.create_dataset("cloud_fraction", shape=(len(sample_ids), 4), dtype="float32")
        date_ds = output.create_dataset("selected_dates", shape=(len(sample_ids),), dtype=text_type)

        def commit(result: tuple[int, str, dict[str, Any]], completed: int) -> None:
            index, sample_id, values = result
            optical_ds[index] = values["optical"]
            valid_ds[index] = values["valid"]
            scl_ds[index] = values["scl"]
            temporal_ds[index] = values["temporal"]
            location_ds[index] = values["location"]
            index_ds[index] = values["indices"]
            duplicate_ds[index] = values["duplicated"]
            cloud_ds[index] = values["cloud_fraction"]
            date_ds[index] = values["dates"]
            output.attrs["completed_samples"] = completed
            if completed % max(1, args.flush_every) == 0 or completed == len(tasks):
                output.flush()
                print(f"[prithvi-cache] {completed}/{len(tasks)} {sample_id}", flush=True)

        if args.workers <= 1:
            for completed, task in enumerate(tasks, start=1):
                commit(process_task(task), completed)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process_task, task) for task in tasks]
                for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    commit(future.result(), completed)
        output.attrs["complete"] = 1
        output.attrs["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.flush()
    os.replace(temporary, output_path)
    print(json.dumps({"output": str(output_path), "samples": len(sample_ids), "shape": [len(sample_ids), 6, 4, 128, 128]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
