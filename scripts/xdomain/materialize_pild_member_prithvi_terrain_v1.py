#!/usr/bin/env python3
"""Materialize one PILD member for the matched Prithvi-plus-Terrain protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "metadata/pild_sen12_training_v2/unified_sample_manifest_v2.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def decode(values: Iterable[Any]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def copy_attrs(source: h5py.File, target: h5py.File) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def copy_indexed_dataset(
    source: h5py.File,
    target: h5py.File,
    key: str,
    indices: list[int],
    *,
    chunk_rows: int = 32,
) -> None:
    source_dataset = source[key]
    if source_dataset.ndim == 0:
        target.create_dataset(key, data=source_dataset[()])
        return
    if source_dataset.shape[0] <= max(indices, default=-1):
        raise IndexError(f"{key}: source index exceeds shape {source_dataset.shape}")
    shape = (len(indices), *source_dataset.shape[1:])
    if h5py.check_string_dtype(source_dataset.dtype) is not None:
        output = target.create_dataset(key, shape=shape, dtype=h5py.string_dtype("utf-8"))
    else:
        chunks = (1, *source_dataset.shape[1:]) if source_dataset.ndim > 1 else True
        output = target.create_dataset(key, shape=shape, dtype=source_dataset.dtype, chunks=chunks)
    for key_attr, value in source_dataset.attrs.items():
        output.attrs[key_attr] = value
    for start in range(0, len(indices), chunk_rows):
        stop = min(start + chunk_rows, len(indices))
        source_indices = indices[start:stop]
        if source_indices != sorted(source_indices):
            values = np.stack([source_dataset[index] for index in source_indices])
        else:
            values = source_dataset[source_indices]
        if h5py.check_string_dtype(source_dataset.dtype) is not None:
            values = np.asarray(decode(values), dtype=object)
        output[start:stop] = values


def load_manifest(path: Path, dataset_id: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset_id"] == dataset_id]
    if not rows:
        raise ValueError(f"dataset_id={dataset_id!r} is absent from {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"dataset_id={dataset_id!r} has duplicate sample IDs")
    for support in ("core_assets_ready",):
        bad = [row["sample_id"] for row in rows if row[support] != "1"]
        if bad:
            raise RuntimeError(f"{len(bad)} rows fail {support}, examples={bad[:5]}")
    return rows


def require_single_path(rows: list[dict[str, str]], column: str) -> Path:
    values = {row[column] for row in rows}
    if len(values) != 1:
        raise RuntimeError(f"{column} is not unique: {sorted(values)}")
    path = Path(next(iter(values)))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def materialize_h5_files(
    rows: list[dict[str, str]], outdir: Path
) -> tuple[Path, Path, Path]:
    base_source = require_single_path(rows, "base_h5_path")
    optical_source = require_single_path(rows, "optical_h5_path")
    terrain_source = require_single_path(rows, "terrain_h5_path")
    base_indices = [int(row["base_h5_index"]) for row in rows]
    optical_indices = [int(row["optical_h5_index"]) for row in rows]
    terrain_indices = [int(row["terrain_h5_index"]) for row in rows]
    paths = (
        outdir / "base_p128.h5",
        outdir / "prithvi_4t6b_p128.h5",
        outdir / "common_terrain9_p128.h5",
    )
    temporary = tuple(path.with_suffix(path.suffix + ".partial") for path in paths)
    for path in temporary:
        path.unlink(missing_ok=True)

    with (
        h5py.File(base_source, "r") as source,
        h5py.File(temporary[0], "w") as target,
    ):
        copy_attrs(source, target)
        for key in (
            "sample_id",
            "event_uid",
            "physical_event_id",
            "source_scene_id",
            "mask",
            "valid_mask",
        ):
            copy_indexed_dataset(source, target, key, base_indices)
        target.attrs["complete"] = 1
        target.attrs["materialization_contract"] = "frozen_manifest_indices_only; no label filtering"
        target.attrs["source_h5"] = str(base_source)

    with (
        h5py.File(optical_source, "r") as source,
        h5py.File(temporary[1], "w") as target,
    ):
        copy_attrs(source, target)
        for key in (
            "sample_id",
            "optical",
            "optical_valid",
            "temporal_coords",
            "location_coords",
        ):
            copy_indexed_dataset(source, target, key, optical_indices)
        target.attrs["complete"] = 1
        target.attrs["selected_samples"] = len(rows)
        target.attrs["materialization_contract"] = "frozen_manifest_indices_only; no label filtering"
        target.attrs["source_h5"] = str(optical_source)

    with (
        h5py.File(terrain_source, "r") as source,
        h5py.File(temporary[2], "w") as target,
    ):
        copy_attrs(source, target)
        copy_indexed_dataset(source, target, "sample_id", terrain_indices)
        copy_indexed_dataset(source, target, "terrain", terrain_indices)
        copy_indexed_dataset(source, target, "terrain_valid", terrain_indices)
        if "q_T" in source:
            copy_indexed_dataset(source, target, "q_T", terrain_indices)
        target.create_dataset(
            "terrain_names",
            data=np.asarray(decode(source["terrain_names"][:]), dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        target.attrs["complete"] = 1
        target.attrs["materialization_contract"] = "frozen_manifest_indices_only; no label filtering"
        target.attrs["source_h5"] = str(terrain_source)

    for partial, final in zip(temporary, paths, strict=True):
        partial.replace(final)
    return paths


def parse_fold(fold_id: str) -> int:
    match = re.search(r"_fold(\d+)$", fold_id)
    if match is None:
        raise ValueError(f"cannot parse fold from {fold_id!r}")
    return int(match.group(1))


def convert_split(
    source_path: Path,
    output_path: Path,
    dataset_id: str,
    expected_ids: set[str],
) -> dict[str, Any]:
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset_id"] == dataset_id]
    if not rows:
        raise ValueError(f"no {dataset_id} rows in {source_path}")
    output_rows = []
    fold_ids: set[int] = set()
    for row in rows:
        fold = parse_fold(row["fold_id"])
        fold_ids.add(fold)
        event_id = row["canonical_event_id"]
        output_rows.append(
            {
                "protocol_id": row["protocol_id"],
                "outer_fold": fold,
                "sample_id": row["sample_id"],
                "dataset_id": row["dataset_id"],
                "source_id": row["source_id"],
                "source_event_id": event_id,
                "region_group": event_id,
                "spatial_supergroup": event_id,
                "role": row["role"],
                "role_reason": row["role_reason"],
            }
        )
    for fold in sorted(fold_ids):
        fold_rows = [row for row in output_rows if row["outer_fold"] == fold]
        fold_samples = {row["sample_id"] for row in fold_rows}
        if fold_samples != expected_ids:
            raise RuntimeError(
                f"fold={fold} sample mismatch: missing={len(expected_ids - fold_samples)}, "
                f"extra={len(fold_samples - expected_ids)}"
            )
        role_events = {
            role: {
                row["spatial_supergroup"]
                for row in fold_rows
                if row["role"] == role
            }
            for role in ("train", "val", "test")
        }
        if any(not values for values in role_events.values()):
            raise RuntimeError(f"fold={fold} has an empty role: {role_events}")
        pairs = (("train", "val"), ("train", "test"), ("val", "test"))
        if any(role_events[left] & role_events[right] for left, right in pairs):
            raise RuntimeError(f"fold={fold} event leakage: {role_events}")
    fieldnames = list(output_rows[0])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return {"folds": sorted(fold_ids), "rows": len(output_rows)}


def validate_outputs(
    paths: tuple[Path, Path, Path], expected_ids: list[str]
) -> dict[str, Any]:
    observed = []
    shapes: dict[str, Any] = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            ids = decode(handle["sample_id"][:])
            if int(handle.attrs.get("complete", 0)) != 1:
                raise RuntimeError(f"{path} is not marked complete")
            observed.append(ids)
            shapes[path.name] = {
                key: list(handle[key].shape)
                for key in handle
                if key in {"mask", "optical", "terrain"}
            }
    if observed != [expected_ids, expected_ids, expected_ids]:
        raise RuntimeError("materialized base/optical/terrain sample identities differ")
    return shapes


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    sentinel = args.outdir / "materialization_summary.json"
    outputs = (
        args.outdir / "base_p128.h5",
        args.outdir / "prithvi_4t6b_p128.h5",
        args.outdir / "common_terrain9_p128.h5",
    )
    split_output = args.outdir / "event_isolated_splits.csv"
    if not args.overwrite and sentinel.exists() and all(path.exists() for path in outputs):
        print(sentinel.read_text(encoding="utf-8"), end="")
        return 0
    rows = load_manifest(args.manifest, args.dataset_id)
    sample_ids = [row["sample_id"] for row in rows]
    paths = materialize_h5_files(rows, args.outdir)
    split = convert_split(args.split_csv, split_output, args.dataset_id, set(sample_ids))
    shapes = validate_outputs(paths, sample_ids)
    payload = {
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": args.dataset_id,
        "n_samples": len(rows),
        "n_events": len({row["canonical_event_id"] for row in rows}),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "source_split": str(args.split_csv.resolve()),
        "source_split_sha256": sha256(args.split_csv),
        "split": split,
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (*paths, split_output)
        },
        "shapes": shapes,
        "selection_contract": "all dataset rows from frozen manifest; no label- or outcome-based filtering",
    }
    sentinel.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
