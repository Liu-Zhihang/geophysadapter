#!/usr/bin/env python3
"""Resume the PILD PC mirror as four disjoint Blob shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mirror_pild_prithvi_assets_v1 as legacy
import resume_pild_planetary_blob_assets_v1 as blob


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    base = ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, default=base / "acquisition_availability_v1.csv")
    parser.add_argument("--mirror-root", type=Path, default=base / "sentinel2_asset_mirror_v1")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers-per-shard", type=int, default=16)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--download-retries", type=int, default=100)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def partition_item_ids(item_ids: list[str], shards: int) -> list[list[str]]:
    if shards < 1:
        raise ValueError("--shards must be positive")
    partitions = [item_ids[index::shards] for index in range(shards)]
    if any(not partition for partition in partitions):
        raise RuntimeError("shard count exceeds frozen item count")
    flat = [item for partition in partitions for item in partition]
    if len(flat) != len(set(flat)) or set(flat) != set(item_ids):
        raise RuntimeError("Blob recovery partitions overlap or omit frozen items")
    return partitions


def merge_strict(
    destination: dict[tuple[str, str], dict[str, Any]],
    source: dict[tuple[str, str], dict[str, Any]],
    label: str,
) -> None:
    for key, record in source.items():
        prior = destination.get(key)
        if prior is not None and prior != record:
            raise RuntimeError(f"conflicting {label} records for {key}")
        destination[key] = record


def load_legacy_seed_records(
    mirror_root: Path, availability_sha256: str
) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    manifests = [mirror_root / "asset_mirror_manifest_v1.jsonl"]
    manifests.extend(sorted(mirror_root.glob("asset_mirror_manifest_shard*_v1.jsonl")))
    for manifest in manifests:
        if manifest.exists():
            merge_strict(records, legacy.load_manifest(manifest, availability_sha256), manifest.name)
    return records


def owner_by_item(partitions: list[list[str]]) -> dict[str, int]:
    return {item_id: shard for shard, values in enumerate(partitions) for item_id in values}


def atomic_json(path: Path, value: Any) -> None:
    legacy.atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    args = parse_args()
    if args.shards != 4:
        raise ValueError("full Blob recovery is frozen to exactly 4 disjoint shards")
    if not 8 <= args.workers_per_shard <= 64:
        raise ValueError("--workers-per-shard must be in [8, 64]")
    availability = args.availability.resolve()
    mirror_root = args.mirror_root.resolve()
    availability_hash = legacy.sha256_file(availability)
    item_ids = legacy.read_selected_item_ids(availability)
    partitions = partition_item_ids(item_ids, args.shards)
    expected = {(item_id, asset) for item_id in item_ids for asset in blob.ASSETS}
    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolver": "planetary-computer-azure-blob-listing",
        "stac_api_used": False,
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "mirror_root": str(mirror_root),
        "n_items": len(item_ids),
        "n_assets": len(expected),
        "shards": args.shards,
        "workers_per_shard": args.workers_per_shard,
        "total_worker_threads": args.shards * args.workers_per_shard,
        "item_counts": [len(values) for values in partitions],
        "disjoint_item_partitions": True,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.plan_only:
        return 0

    mirror_root.mkdir(parents=True, exist_ok=True)
    main_manifest = mirror_root / "asset_mirror_manifest_v1.jsonl"
    main_marker = legacy.complete_marker_path(main_manifest)
    main_marker.unlink(missing_ok=True)
    seed_records = load_legacy_seed_records(mirror_root, availability_hash)
    children: list[tuple[int, subprocess.Popen[bytes], Any, Path, Path, Path]] = []
    script = Path(blob.__file__).resolve()
    python = Path(sys.executable).resolve()
    for shard in range(args.shards):
        shard_manifest = mirror_root / f"asset_mirror_blob_shard{shard:02d}_v1.jsonl"
        resolution_manifest = mirror_root / f"planetary_blob_resolution_shard{shard:02d}_v1.jsonl"
        shard_log = mirror_root / f"planetary_blob_shard{shard:02d}.log"
        existing = legacy.load_manifest(shard_manifest, availability_hash)
        records = dict(seed_records)
        merge_strict(records, existing, shard_manifest.name)
        legacy.write_manifest(shard_manifest, records)
        command = [
            str(python), str(script),
            "--availability", str(availability),
            "--mirror-root", str(mirror_root),
            "--manifest", str(shard_manifest),
            "--resolution-manifest", str(resolution_manifest),
            "--workers", str(args.workers_per_shard),
            "--retries", str(args.retries),
            "--download-retries", str(args.download_retries),
            "--shards", str(args.shards),
            "--shard-index", str(shard),
        ]
        handle = shard_log.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(
            (shard, process, handle, shard_manifest, resolution_manifest, shard_log)
        )

    atomic_json(
        mirror_root / "planetary_blob_sharded_download_pids_v1.json",
        {
            **plan,
            "orchestrator_pid": os.getpid(),
            "children": [
                {
                    "shard": shard,
                    "pid": process.pid,
                    "manifest": str(manifest),
                    "resolution_manifest": str(resolution),
                    "log": str(log),
                }
                for shard, process, _, manifest, resolution, log in children
            ],
        },
    )
    failures = []
    for shard, process, handle, _, _, log in children:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append({"shard": shard, "return_code": return_code, "log": str(log)})
    if failures:
        raise RuntimeError(f"Planetary Blob recovery shards failed: {failures}")

    owners = owner_by_item(partitions)
    shard_records = {
        shard: legacy.load_manifest(manifest, availability_hash)
        for shard, _, _, manifest, _, _ in children
    }
    shard_resolution = {
        shard: blob.load_resolution_manifest(resolution, availability_hash)
        for shard, _, _, _, resolution, _ in children
    }
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    merged_resolution: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(expected):
        shard = owners[key[0]]
        if key not in shard_records[shard] or key not in shard_resolution[shard]:
            raise RuntimeError(f"owner shard {shard} lacks evidence for {key}")
        merged[key] = shard_records[shard][key]
        merged_resolution[key] = shard_resolution[shard][key]
        path = mirror_root / str(merged[key]["local_path"])
        if not legacy.validate_completed_record(merged[key], path):
            raise RuntimeError(f"final byte/SHA verification failed: {key}")
        if merged_resolution[key].get("acquisition_identity_match") is not True:
            raise RuntimeError(f"final identity verification failed: {key}")
    if len(merged) != 2016 or len(merged_resolution) != 2016:
        raise RuntimeError(
            f"full completion requires exactly 2016 asset records, got "
            f"{len(merged)} and {len(merged_resolution)}"
        )

    global_resolution = mirror_root / "planetary_blob_resolution_manifest_v1.jsonl"
    legacy.write_manifest(main_manifest, merged)
    blob.write_jsonl(global_resolution, merged_resolution)
    completion = {
        "schema_version": legacy.MANIFEST_SCHEMA_VERSION,
        "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolver": "planetary-computer-azure-blob-listing",
        "stac_api_used": False,
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "manifest": str(main_manifest),
        "manifest_sha256": legacy.sha256_file(main_manifest),
        "resolution_manifest": str(global_resolution),
        "resolution_manifest_sha256": legacy.sha256_file(global_resolution),
        "asset_names": list(blob.ASSETS),
        "item_ids": item_ids,
        "item_identity_sha256": legacy.sha256_text(item_ids),
        "scope": "full_availability",
        "n_items": 288,
        "n_assets": 2016,
        "n_resolution_records": 2016,
        "download_topology": {
            "shards": args.shards,
            "workers_per_shard": args.workers_per_shard,
            "disjoint_by": "frozen original item_id",
        },
    }
    legacy.atomic_write_text(
        main_marker,
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
