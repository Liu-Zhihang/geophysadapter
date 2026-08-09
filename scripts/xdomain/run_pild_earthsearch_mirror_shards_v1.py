#!/usr/bin/env python3
"""Run disjoint Earth Search mirror shards and publish one verified manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mirror_pild_earthsearch_assets_v1 as mirror


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    base = ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, default=base / "acquisition_availability_v1.csv")
    parser.add_argument(
        "--mirror-root",
        type=Path,
        default=base / "sentinel2_asset_mirror_earthsearch_v1",
    )
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers-per-shard", type=int, default=16)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--download-retries", type=int, default=100)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    mirror.safeio.atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def partition_item_ids(item_ids: list[str], shards: int) -> list[list[str]]:
    if shards < 1:
        raise ValueError("--shards must be positive")
    partitions = [item_ids[index::shards] for index in range(shards)]
    if any(not values for values in partitions):
        raise RuntimeError("shard count exceeds available item count")
    flat = [item for partition in partitions for item in partition]
    if len(flat) != len(set(flat)) or set(flat) != set(item_ids):
        raise RuntimeError("Earth Search shard partitions overlap or omit items")
    return partitions


def merge_records(
    destination: dict[tuple[str, str], dict[str, Any]],
    source: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for key, record in source.items():
        prior = destination.get(key)
        if prior is not None and prior != record:
            raise RuntimeError(f"conflicting Earth Search shard records for {key}")
        destination[key] = record


def main() -> int:
    args = parse_args()
    if args.shards < 2 or args.workers_per_shard < 1:
        raise ValueError("--shards must be >=2 and --workers-per-shard must be positive")
    availability = args.availability.resolve()
    mirror_root = args.mirror_root.resolve()
    mirror.assert_independent_mirror_root(mirror_root)
    availability_hash = mirror.safeio.sha256_file(availability)
    item_ids = mirror.safeio.read_selected_item_ids(availability)
    partitions = partition_item_ids(item_ids, args.shards)
    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mirror_source": "earth-search",
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "mirror_root": str(mirror_root),
        "selection_uses_labels": False,
        "n_items": len(item_ids),
        "n_assets": len(item_ids) * len(mirror.ASSETS),
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
    main_manifest = mirror_root / "asset_mirror_manifest_earthsearch_v1.jsonl"
    main_marker = mirror.complete_marker_path(main_manifest)
    main_marker.unlink(missing_ok=True)
    seed_records = mirror.load_manifest(main_manifest, availability_hash)
    children: list[tuple[int, subprocess.Popen[bytes], Any, Path, Path]] = []
    script = Path(mirror.__file__).resolve()
    python = Path(sys.executable).resolve()
    for shard in range(args.shards):
        shard_manifest = mirror_root / f"asset_mirror_manifest_earthsearch_shard{shard:02d}_v1.jsonl"
        shard_log = mirror_root / f"earthsearch_shard{shard:02d}.log"
        existing = mirror.load_manifest(shard_manifest, availability_hash)
        records: dict[tuple[str, str], dict[str, Any]] = {}
        merge_records(records, seed_records)
        merge_records(records, existing)
        mirror.write_manifest(shard_manifest, records)
        command = [
            str(python),
            str(script),
            "--availability", str(availability),
            "--mirror-root", str(mirror_root),
            "--manifest", str(shard_manifest),
            "--workers", str(args.workers_per_shard),
            "--retries", str(args.retries),
            "--download-retries", str(args.download_retries),
            "--shards", str(args.shards),
            "--shard-index", str(shard),
        ]
        log_handle = shard_log.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((shard, process, log_handle, shard_manifest, shard_log))

    atomic_json(
        mirror_root / "earthsearch_sharded_download_pids_v1.json",
        {
            **plan,
            "orchestrator_pid": os.getpid(),
            "children": [
                {
                    "shard": shard,
                    "pid": process.pid,
                    "manifest": str(manifest),
                    "log": str(log),
                }
                for shard, process, _, manifest, log in children
            ],
        },
    )

    failures = []
    for shard, process, handle, _, log in children:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append({"shard": shard, "return_code": return_code, "log": str(log)})
    if failures:
        raise RuntimeError(f"Earth Search mirror shards failed: {failures}")

    merged = dict(seed_records)
    for _, _, _, shard_manifest, _ in children:
        merge_records(merged, mirror.load_manifest(shard_manifest, availability_hash))
    expected = {(item_id, asset) for item_id in item_ids for asset in mirror.ASSETS}
    missing = sorted(expected - set(merged))
    if missing:
        raise RuntimeError(f"merged Earth Search manifest lacks assets: {missing[:5]}")
    for key in sorted(expected):
        record = merged[key]
        path = mirror_root / str(record["local_path"])
        if record.get("acquisition_identity_match") is not True:
            raise RuntimeError(f"merged record lacks identity match: {key}")
        if not mirror.safeio.validate_completed_record(record, path):
            raise RuntimeError(f"post-shard verification failed: {key}")

    mirror.write_manifest(main_manifest, merged)
    completion = {
        "schema_version": mirror.MANIFEST_SCHEMA_VERSION,
        "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "mirror_source": "earth-search",
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "manifest": str(main_manifest),
        "manifest_sha256": mirror.safeio.sha256_file(main_manifest),
        "asset_names": list(mirror.ASSETS),
        "original_item_ids": item_ids,
        "original_item_identity_sha256": mirror.safeio.sha256_text(item_ids),
        "selection_uses_labels": False,
        "acquisition_identity_match": True,
        "acquisition_identity_policy": (
            "exact satellite+sensing_timestamp+relative_orbit+MGRS_tile; "
            "processing baseline and generation may differ"
        ),
        "scope": "full_availability",
        "n_items": len(item_ids),
        "n_assets": len(expected),
        "content_length_bytes": sum(int(merged[key]["content_length"]) for key in expected),
        "download_topology": {
            "shards": args.shards,
            "workers_per_shard": args.workers_per_shard,
            "disjoint_by": "frozen original STAC item ID",
        },
    }
    mirror.safeio.atomic_write_text(
        main_marker,
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
