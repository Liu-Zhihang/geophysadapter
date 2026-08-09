#!/usr/bin/env python3
"""Merge completed PILD Prithvi temporal-cache shards into one audited cache.

The merger never reads label data. It validates each shard against the frozen
readiness and availability registries, restores the original readiness order,
and publishes the merged HDF5 only after all identities and source manifests
have passed strict checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BANDS = "B02;B03;B04;B8A;B11;B12"
REQUIRED_DATASETS = {
    "sample_id",
    "acquisition_unit_id",
    "optical",
    "scl",
    "optical_valid",
    "selected_dates",
    "selected_item_ids",
    "temporal_coords",
    "location_coords",
    "cloud_fraction",
    "coverage_fraction",
    "q_visual_temporal",
    "completed",
    "failure_reason",
}
DYNAMIC_ATTRIBUTES = {
    "complete",
    "completed_at_utc",
    "completed_samples",
    "created_at_utc",
    "last_completed_unit",
    "last_updated_at_utc",
    "q_visual_temporal_0",
    "q_visual_temporal_1",
    "sample_identity_sha256",
    "selected_samples",
    "selected_units",
    "source_manifest",
    "source_manifest_sha256",
}


@dataclass(frozen=True)
class Shard:
    path: Path
    marker_path: Path
    manifest_path: Path
    sample_ids: tuple[str, ...]
    acquisition_unit_ids: tuple[str, ...]
    readiness_sha256: str
    availability_sha256: str


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        action="append",
        type=Path,
        required=True,
        help="Completed shard HDF5; repeat once per shard.",
    )
    parser.add_argument("--readiness", type=Path, default=base / "pild_window_readiness.csv")
    parser.add_argument(
        "--availability", type=Path, default=base / "acquisition_availability_v1.csv"
    )
    parser.add_argument("--out", type=Path, default=base / "pild_prithvi_4t6b_p128.h5")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=2937,
        help="Production invariant is 2937; override only for synthetic smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values)


def marker_path_for(shard: Path) -> Path:
    return shard.with_name(f"{shard.name}.complete.json")


def json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def normalize_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [normalize_attr(item) for item in value.tolist()]
    return value


def dataset_signature(dataset: h5py.Dataset) -> dict[str, Any]:
    string = h5py.check_string_dtype(dataset.dtype)
    dtype = (
        {"kind": "string", "encoding": string.encoding, "length": string.length}
        if string is not None
        else {"kind": "numeric", "dtype": dataset.dtype.str}
    )
    return {
        "dtype": dtype,
        "tail_shape": list(dataset.shape[1:]),
        "chunks": list(dataset.chunks) if dataset.chunks is not None else None,
        "compression": dataset.compression,
        "compression_opts": dataset.compression_opts,
        "shuffle": bool(dataset.shuffle),
        "fletcher32": bool(dataset.fletcher32),
        "scaleoffset": dataset.scaleoffset,
    }


def schema_for(handle: h5py.File) -> dict[str, dict[str, Any]]:
    keys = set(handle.keys())
    missing = REQUIRED_DATASETS - keys
    unexpected = keys - REQUIRED_DATASETS
    if missing or unexpected:
        raise RuntimeError(
            f"HDF5 dataset schema mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return {name: dataset_signature(handle[name]) for name in sorted(keys)}


def validate_builder_schema(handle: h5py.File, path: Path) -> None:
    if int(handle.attrs.get("schema_version", -1)) != 1:
        raise RuntimeError(f"{path}: schema_version must be 1")
    if normalize_attr(handle.attrs.get("bands", "")) != BANDS:
        raise RuntimeError(f"{path}: unexpected bands attribute")
    expected = {
        "optical": ((6, 4, 128, 128), np.dtype("uint16")),
        "scl": ((4, 128, 128), np.dtype("uint8")),
        "optical_valid": ((1, 128, 128), np.dtype("uint8")),
        "temporal_coords": ((4, 2), np.dtype("int16")),
        "location_coords": ((2,), np.dtype("float32")),
        "cloud_fraction": ((4,), np.dtype("float32")),
        "coverage_fraction": ((4, 7), np.dtype("float32")),
        "q_visual_temporal": ((), np.dtype("uint8")),
        "completed": ((), np.dtype("uint8")),
    }
    for name, (tail, dtype) in expected.items():
        dataset = handle[name]
        if dataset.shape[1:] != tail or dataset.dtype != dtype:
            raise RuntimeError(
                f"{path}: {name} has shape/dtype {dataset.shape}/{dataset.dtype}, "
                f"expected (N,{','.join(map(str, tail))})/{dtype}"
            )


def resolve_manifest(marker: dict[str, Any], shard: Path) -> Path:
    raw = marker.get("source_manifest")
    if not raw:
        raise RuntimeError(f"{shard}: completion marker lacks source_manifest")
    manifest = Path(str(raw)).expanduser()
    if not manifest.is_absolute():
        manifest = shard.parent / manifest
    return manifest.resolve()


def validate_shard(
    path: Path,
    readiness_hash: str,
    availability_hash: str,
    reference_schema: dict[str, dict[str, Any]] | None,
    reference_static_attrs: dict[str, Any] | None,
) -> tuple[Shard, dict[str, dict[str, Any]], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"shard does not exist: {path}")
    marker_path = marker_path_for(path)
    if not marker_path.is_file():
        raise FileNotFoundError(f"shard completion marker is missing: {marker_path}")
    marker = json_object(marker_path)
    if marker.get("complete") is not True:
        raise RuntimeError(f"{marker_path}: complete is not true")
    actual_hash = sha256_file(path)
    if marker.get("output_sha256") != actual_hash:
        raise RuntimeError(f"{path}: marker output_sha256 mismatch")
    manifest_path = resolve_manifest(marker, path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest is missing: {manifest_path}")
    manifest_hash = sha256_file(manifest_path)
    if marker.get("source_manifest_sha256") != manifest_hash:
        raise RuntimeError(f"{manifest_path}: marker source_manifest_sha256 mismatch")
    if marker.get("source_readiness_sha256") != readiness_hash:
        raise RuntimeError(f"{path}: marker readiness hash differs from frozen registry")
    if marker.get("source_availability_sha256") != availability_hash:
        raise RuntimeError(f"{path}: marker availability hash differs from frozen registry")

    with h5py.File(path, "r") as handle:
        schema = schema_for(handle)
        validate_builder_schema(handle, path)
        if int(handle.attrs.get("complete", 0)) != 1:
            raise RuntimeError(f"{path}: HDF5 complete attribute is not 1")
        if normalize_attr(handle.attrs.get("source_readiness_sha256")) != readiness_hash:
            raise RuntimeError(f"{path}: HDF5 readiness hash mismatch")
        if normalize_attr(handle.attrs.get("source_availability_sha256")) != availability_hash:
            raise RuntimeError(f"{path}: HDF5 availability hash mismatch")
        if normalize_attr(handle.attrs.get("source_manifest_sha256")) != manifest_hash:
            raise RuntimeError(f"{path}: HDF5 source manifest hash mismatch")
        n_samples = int(handle["sample_id"].shape[0])
        if n_samples < 1 or any(handle[name].shape[0] != n_samples for name in handle.keys()):
            raise RuntimeError(f"{path}: inconsistent or empty first dimensions")
        if int(marker.get("n_samples", -1)) != n_samples:
            raise RuntimeError(f"{path}: marker n_samples mismatch")
        if int(handle.attrs.get("selected_samples", -1)) != n_samples:
            raise RuntimeError(f"{path}: selected_samples attribute mismatch")
        sample_ids = decode_strings(handle["sample_id"][:])
        unit_ids = decode_strings(handle["acquisition_unit_id"][:])
        if len(set(sample_ids)) != n_samples:
            raise RuntimeError(f"{path}: duplicate sample_id inside shard")
        if normalize_attr(handle.attrs.get("sample_identity_sha256")) != sha256_text(sample_ids):
            raise RuntimeError(f"{path}: sample_identity_sha256 mismatch")
        if not np.asarray(handle["completed"][:]).astype(bool).all():
            raise RuntimeError(f"{path}: not every sample is committed")
        if int(handle.attrs.get("label_content_accessed", -1)) != 0:
            raise RuntimeError(f"{path}: label_content_accessed is not zero")
        static_attrs = {
            key: normalize_attr(value)
            for key, value in handle.attrs.items()
            if key not in DYNAMIC_ATTRIBUTES
        }
    if reference_schema is not None and schema != reference_schema:
        raise RuntimeError(f"{path}: dtype/chunk/compression schema differs from first shard")
    if reference_static_attrs is not None and static_attrs != reference_static_attrs:
        differing = sorted(
            key
            for key in set(static_attrs) | set(reference_static_attrs)
            if static_attrs.get(key) != reference_static_attrs.get(key)
        )
        raise RuntimeError(f"{path}: static HDF5 attributes differ: {differing}")
    shard = Shard(
        path=path,
        marker_path=marker_path,
        manifest_path=manifest_path,
        sample_ids=sample_ids,
        acquisition_unit_ids=unit_ids,
        readiness_sha256=readiness_hash,
        availability_sha256=availability_hash,
    )
    return shard, schema, static_attrs


def validate_unsigned_manifest_record(record: dict[str, Any], path: Path, line_no: int) -> None:
    required = {"acquisition_unit_id", "collection", "item_id", "datetime", "bbox", "assets"}
    missing = required - set(record)
    if missing:
        raise RuntimeError(f"{path}:{line_no}: source manifest missing {sorted(missing)}")
    assets = record.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise RuntimeError(f"{path}:{line_no}: source manifest assets must be non-empty")
    for asset_name, details in assets.items():
        if not isinstance(details, dict) or not details.get("href"):
            raise RuntimeError(f"{path}:{line_no}: malformed asset {asset_name}")
        href = str(details["href"])
        parsed = urlsplit(href)
        if parsed.query or parsed.fragment:
            raise RuntimeError(f"{path}:{line_no}: signed/query-bearing href is forbidden")


def merge_manifests(shards: list[Shard], output: Path) -> int:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for shard in shards:
        shard_manifest_units: set[str] = set()
        for line_no, line in enumerate(
            shard.manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{shard.manifest_path}:{line_no}: invalid JSON") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"{shard.manifest_path}:{line_no}: expected JSON object")
            validate_unsigned_manifest_record(record, shard.manifest_path, line_no)
            key = (str(record["acquisition_unit_id"]), str(record["item_id"]))
            shard_manifest_units.add(key[0])
            if key in records and records[key] != record:
                raise RuntimeError(f"conflicting source manifest records for {key}")
            records[key] = record
        expected_units = set(shard.acquisition_unit_ids)
        if shard_manifest_units != expected_units:
            raise RuntimeError(
                f"{shard.manifest_path}: manifest/unit mismatch; "
                f"missing={sorted(expected_units - shard_manifest_units)[:5]}, "
                f"unexpected={sorted(shard_manifest_units - expected_units)[:5]}"
            )
    with output.open("w", encoding="utf-8") as stream:
        for key in sorted(records):
            stream.write(json.dumps(records[key], sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return len(records)


def dataset_create_kwargs(source: h5py.Dataset) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"dtype": source.dtype}
    if source.chunks is not None:
        kwargs["chunks"] = source.chunks
    if source.compression is not None:
        kwargs["compression"] = source.compression
        kwargs["compression_opts"] = source.compression_opts
    if source.shuffle:
        kwargs["shuffle"] = True
    if source.fletcher32:
        kwargs["fletcher32"] = True
    if source.scaleoffset is not None:
        kwargs["scaleoffset"] = source.scaleoffset
    if h5py.check_string_dtype(source.dtype) is None and source.fillvalue is not None:
        kwargs["fillvalue"] = source.fillvalue
    return kwargs


def contiguous_runs(source_indices: list[int], destination_indices: list[int]) -> Iterator[tuple[int, int, int]]:
    if not source_indices:
        return
    src_start = source_indices[0]
    dst_start = destination_indices[0]
    length = 1
    for src, dst, previous_src, previous_dst in zip(
        source_indices[1:],
        destination_indices[1:],
        source_indices[:-1],
        destination_indices[:-1],
    ):
        if src == previous_src + 1 and dst == previous_dst + 1:
            length += 1
        else:
            yield src_start, dst_start, length
            src_start, dst_start, length = src, dst, 1
    yield src_start, dst_start, length


def create_output(
    path: Path,
    first_shard: Shard,
    static_attrs: dict[str, Any],
    sample_ids: list[str],
    unit_ids: list[str],
    readiness: Path,
    availability: Path,
    readiness_hash: str,
    availability_hash: str,
    manifest: Path,
    manifest_hash: str,
) -> h5py.File:
    handle = h5py.File(path, "w")
    with h5py.File(first_shard.path, "r") as source:
        for name in sorted(source.keys()):
            source_dataset = source[name]
            shape = (len(sample_ids), *source_dataset.shape[1:])
            output_dataset = handle.create_dataset(
                name, shape=shape, **dataset_create_kwargs(source_dataset)
            )
            for key, value in source_dataset.attrs.items():
                output_dataset.attrs[key] = value
    for key, value in static_attrs.items():
        handle.attrs[key] = value
    handle.attrs.update(
        {
            "complete": 0,
            "created_at_utc": utc_now(),
            "source_readiness": str(readiness),
            "source_availability": str(availability),
            "source_readiness_sha256": readiness_hash,
            "source_availability_sha256": availability_hash,
            "source_manifest": str(manifest),
            "source_manifest_sha256": manifest_hash,
            "sample_identity_sha256": sha256_text(sample_ids),
            "selected_units": len(set(unit_ids)),
            "selected_samples": len(sample_ids),
            "completed_samples": 0,
            "merge_contract": "completed shards reordered exactly to frozen readiness registry; labels forbidden",
            "label_content_accessed": 0,
        }
    )
    handle.flush()
    return handle


def copy_shards(
    handle: h5py.File,
    shards: list[Shard],
    destination_index: dict[str, int],
) -> None:
    for shard_position, shard in enumerate(shards, start=1):
        destinations = [destination_index[sample_id] for sample_id in shard.sample_ids]
        source_indices = list(range(len(shard.sample_ids)))
        runs = list(contiguous_runs(source_indices, destinations))
        with h5py.File(shard.path, "r") as source:
            for name in sorted(source.keys()):
                for src_start, dst_start, length in runs:
                    handle[name][dst_start : dst_start + length] = source[name][
                        src_start : src_start + length
                    ]
        handle.attrs["completed_samples"] = int(
            handle.attrs.get("completed_samples", 0)
        ) + len(shard.sample_ids)
        handle.attrs["last_merged_shard"] = str(shard.path)
        handle.attrs["last_updated_at_utc"] = utc_now()
        handle.flush()
        print(
            f"[{shard_position}/{len(shards)}] merged {shard.path.name}: "
            f"{len(shard.sample_ids)} samples",
            flush=True,
        )


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    args.readiness = args.readiness.resolve()
    args.availability = args.availability.resolve()
    args.out = args.out.resolve()
    manifest = (
        args.manifest or args.out.with_name(f"{args.out.stem}.source_manifest.jsonl")
    ).resolve()
    marker = marker_path_for(args.out)
    if args.expected_samples < 1:
        raise ValueError("--expected-samples must be positive")
    if not args.readiness.is_file() or not args.availability.is_file():
        raise FileNotFoundError("frozen readiness and availability registries are required")
    if len({path.resolve() for path in args.shard}) != len(args.shard):
        raise RuntimeError("the same shard path was supplied more than once")
    for target in (args.out, manifest, marker):
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {target}")

    readiness = pd.read_csv(
        args.readiness,
        usecols=["sample_id", "dataset_id", "source_scene_id", "physical_event_id"],
    )
    # Match build_pild_prithvi_temporal_cache_v1.py exactly. The persisted
    # readiness acquisition_unit_id predates physical-event disambiguation and
    # must not be used as the shard identity contract.
    readiness["acquisition_unit_id"] = (
        readiness["dataset_id"].astype(str)
        + "::"
        + readiness["source_scene_id"].astype(str)
        + "::"
        + readiness["physical_event_id"].astype(str)
    )
    expected_ids = readiness["sample_id"].astype(str).tolist()
    expected_units = readiness["acquisition_unit_id"].astype(str).tolist()
    if len(expected_ids) != args.expected_samples:
        raise RuntimeError(
            f"readiness must contain exactly {args.expected_samples} samples, got {len(expected_ids)}"
        )
    if len(set(expected_ids)) != len(expected_ids):
        raise RuntimeError("readiness registry contains duplicate sample_id")
    expected_by_id = dict(zip(expected_ids, expected_units))
    readiness_hash = sha256_file(args.readiness)
    availability_hash = sha256_file(args.availability)
    availability = pd.read_csv(
        args.availability,
        usecols=[
            "acquisition_unit_id",
            "n_windows",
            "status",
            "prithvi_temporal_ready",
        ],
    )
    if availability["acquisition_unit_id"].astype(str).duplicated().any():
        raise RuntimeError("availability registry contains duplicate acquisition_unit_id")
    available = availability[
        availability["status"].astype(str).eq("ready")
        & availability["prithvi_temporal_ready"].astype(int).eq(1)
    ].copy()
    expected_unit_set = set(expected_units)
    available_unit_set = set(available["acquisition_unit_id"].astype(str))
    if available_unit_set != expected_unit_set:
        raise RuntimeError(
            "availability/readiness unit mismatch; "
            f"missing={sorted(expected_unit_set - available_unit_set)[:5]}, "
            f"unexpected={sorted(available_unit_set - expected_unit_set)[:5]}"
        )
    readiness_counts = pd.Series(expected_units).value_counts().to_dict()
    availability_counts = available.set_index("acquisition_unit_id")["n_windows"].astype(int)
    count_mismatches = {
        unit_id: (int(readiness_counts[unit_id]), int(availability_counts[unit_id]))
        for unit_id in sorted(expected_unit_set)
        if int(readiness_counts[unit_id]) != int(availability_counts[unit_id])
    }
    if count_mismatches:
        raise RuntimeError(
            f"availability/readiness n_windows mismatch: {dict(list(count_mismatches.items())[:5])}"
        )

    shards: list[Shard] = []
    reference_schema: dict[str, dict[str, Any]] | None = None
    reference_static_attrs: dict[str, Any] | None = None
    for raw_path in args.shard:
        shard, schema, static_attrs = validate_shard(
            raw_path,
            readiness_hash,
            availability_hash,
            reference_schema,
            reference_static_attrs,
        )
        reference_schema = schema if reference_schema is None else reference_schema
        reference_static_attrs = static_attrs if reference_static_attrs is None else reference_static_attrs
        shards.append(shard)

    input_paths = {
        path
        for shard in shards
        for path in (shard.path, shard.marker_path, shard.manifest_path)
    }
    collisions = sorted(
        str(path) for path in (args.out, marker, manifest) if path in input_paths
    )
    if collisions:
        raise RuntimeError(f"output paths collide with shard inputs: {collisions}")

    observed: dict[str, tuple[Path, int]] = {}
    for shard in shards:
        for index, (sample_id, unit_id) in enumerate(
            zip(shard.sample_ids, shard.acquisition_unit_ids)
        ):
            if sample_id not in expected_by_id:
                raise RuntimeError(f"{shard.path}: unknown sample_id {sample_id}")
            if sample_id in observed:
                previous, previous_index = observed[sample_id]
                raise RuntimeError(
                    f"duplicate sample_id {sample_id}: {previous}[{previous_index}] and {shard.path}[{index}]"
                )
            if unit_id != expected_by_id[sample_id]:
                raise RuntimeError(
                    f"{shard.path}: acquisition_unit_id mismatch for {sample_id}: "
                    f"{unit_id} != {expected_by_id[sample_id]}"
                )
            observed[sample_id] = (shard.path, index)
    missing = [sample_id for sample_id in expected_ids if sample_id not in observed]
    if missing or len(observed) != len(expected_ids):
        raise RuntimeError(
            f"shards do not exactly cover readiness: missing={len(missing)} "
            f"examples={missing[:5]}, observed={len(observed)}, expected={len(expected_ids)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    h5_partial = args.out.with_name(f".{args.out.name}.inprogress.{os.getpid()}")
    manifest_partial = manifest.with_name(f".{manifest.name}.inprogress.{os.getpid()}")
    h5_partial.unlink(missing_ok=True)
    manifest_partial.unlink(missing_ok=True)
    try:
        n_manifest_records = merge_manifests(shards, manifest_partial)
        manifest_hash = sha256_file(manifest_partial)
        handle = create_output(
            h5_partial,
            shards[0],
            reference_static_attrs or {},
            expected_ids,
            expected_units,
            args.readiness,
            args.availability,
            readiness_hash,
            availability_hash,
            manifest,
            manifest_hash,
        )
        try:
            copy_shards(handle, shards, {sample_id: index for index, sample_id in enumerate(expected_ids)})
            merged_ids = decode_strings(handle["sample_id"][:])
            merged_units = decode_strings(handle["acquisition_unit_id"][:])
            if list(merged_ids) != expected_ids or list(merged_units) != expected_units:
                raise RuntimeError("merged HDF5 identity/order differs from readiness registry")
            if not np.asarray(handle["completed"][:]).astype(bool).all():
                raise RuntimeError("merged HDF5 contains uncommitted samples")
            if int(handle.attrs.get("completed_samples", -1)) != len(expected_ids):
                raise RuntimeError("merged completed_samples attribute mismatch")
            handle.attrs["complete"] = 1
            handle.attrs["completed_at_utc"] = utc_now()
            handle.attrs["q_visual_temporal_1"] = int(handle["q_visual_temporal"][:].sum())
            handle.attrs["q_visual_temporal_0"] = int(
                len(expected_ids) - handle["q_visual_temporal"][:].sum()
            )
            handle.flush()
        finally:
            handle.close()

        os.replace(manifest_partial, manifest)
        os.replace(h5_partial, args.out)
        output_hash = sha256_file(args.out)
        completion = {
            "schema_version": 1,
            "complete": True,
            "completed_at_utc": utc_now(),
            "output": str(args.out),
            "output_sha256": output_hash,
            "source_manifest": str(manifest),
            "source_manifest_sha256": sha256_file(manifest),
            "source_readiness": str(args.readiness),
            "source_readiness_sha256": readiness_hash,
            "source_availability": str(args.availability),
            "source_availability_sha256": availability_hash,
            "n_shards": len(shards),
            "n_units": len(set(expected_units)),
            "n_samples": len(expected_ids),
            "shape": [len(expected_ids), 6, 4, 128, 128],
            "sample_identity_sha256": sha256_text(expected_ids),
            "n_source_manifest_records": n_manifest_records,
            "shards": [
                {
                    "path": str(shard.path),
                    "sha256": sha256_file(shard.path),
                    "n_samples": len(shard.sample_ids),
                    "marker": str(shard.marker_path),
                }
                for shard in shards
            ],
            "label_content_accessed": 0,
        }
        write_json_atomic(marker, completion)
    except Exception:
        h5_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise

    print(json.dumps(completion, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
