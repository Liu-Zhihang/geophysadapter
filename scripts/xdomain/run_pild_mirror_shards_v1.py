#!/usr/bin/env python3
"""Run disjoint PILD Sentinel-2 mirror shards and publish one verified manifest.

Each STAC item belongs to exactly one child process, so child processes never
write the same asset or ``.part`` file.  Per-shard manifests are merged only
after every child succeeds and every expected asset passes size/SHA256 checks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mirror_pild_prithvi_assets_v1 as mirror


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    base = ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, default=base / "acquisition_availability_v1.csv")
    parser.add_argument("--mirror-root", type=Path, default=base / "sentinel2_asset_mirror_v1")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers-per-shard", type=int, default=32)
    parser.add_argument("--download-retries", type=int, default=100)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    mirror.atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def main() -> int:
    args = parse_args()
    if args.shards < 2 or args.workers_per_shard < 1:
        raise ValueError("--shards must be >=2 and --workers-per-shard must be positive")

    availability = args.availability.resolve()
    mirror_root = args.mirror_root.resolve()
    mirror_root.mkdir(parents=True, exist_ok=True)
    main_manifest = mirror_root / "asset_mirror_manifest_v1.jsonl"
    main_marker = mirror.complete_marker_path(main_manifest)
    availability_hash = mirror.sha256_file(availability)
    item_ids = mirror.read_selected_item_ids(availability)
    partitions = [item_ids[index::args.shards] for index in range(args.shards)]
    if any(not values for values in partitions):
        raise RuntimeError("shard count exceeds available item count")

    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "mirror_root": str(mirror_root),
        "n_items": len(item_ids),
        "n_assets": len(item_ids) * len(mirror.ASSETS),
        "shards": args.shards,
        "workers_per_shard": args.workers_per_shard,
        "total_worker_threads": args.shards * args.workers_per_shard,
        "item_counts": [len(values) for values in partitions],
        "disjoint_item_partitions": len(set().union(*map(set, partitions))) == len(item_ids),
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.plan_only:
        return 0

    main_marker.unlink(missing_ok=True)
    seed_records = mirror.load_manifest(main_manifest, availability_hash)
    children: list[tuple[int, subprocess.Popen[bytes], Any, Path, Path]] = []
    script = Path(mirror.__file__).resolve()
    python = Path(sys.executable).resolve()
    for shard, selected in enumerate(partitions):
        shard_manifest = mirror_root / f"asset_mirror_manifest_shard{shard:02d}_v1.jsonl"
        shard_log = mirror_root / f"shard{shard:02d}.log"
        existing = mirror.load_manifest(shard_manifest, availability_hash) if shard_manifest.exists() else {}
        records = dict(seed_records)
        for key, record in existing.items():
            prior = records.get(key)
            if prior is not None and prior != record:
                raise RuntimeError(f"conflicting seed/shard manifest record for {key}")
            records[key] = record
        mirror.write_manifest(shard_manifest, records)
        command = [
            str(python), str(script),
            "--availability", str(availability),
            "--mirror-root", str(mirror_root),
            "--manifest", str(shard_manifest),
            "--workers", str(args.workers_per_shard),
            "--download-retries", str(args.download_retries),
        ]
        for item_id in selected:
            command.extend(("--item-id", item_id))
        log_handle = shard_log.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((shard, process, log_handle, shard_manifest, shard_log))

    pid_receipt = {
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
    }
    atomic_json(mirror_root / "sharded_download_pids_v1.json", pid_receipt)

    failures = []
    for shard, process, log_handle, _, shard_log in children:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append({"shard": shard, "return_code": return_code, "log": str(shard_log)})
    if failures:
        raise RuntimeError(f"mirror shards failed: {failures}")

    merged = dict(seed_records)
    for _, _, _, shard_manifest, _ in children:
        for key, record in mirror.load_manifest(shard_manifest, availability_hash).items():
            prior = merged.get(key)
            if prior is not None and prior != record:
                raise RuntimeError(f"conflicting shard records for {key}")
            merged[key] = record

    expected = {(item_id, asset) for item_id in item_ids for asset in mirror.ASSETS}
    missing = sorted(expected - set(merged))
    if missing:
        raise RuntimeError(f"merged manifest lacks {len(missing)} expected assets: {missing[:5]}")
    for key in sorted(expected):
        record = merged[key]
        destination = mirror_root / str(record["local_path"])
        if not mirror.validate_completed_record(record, destination):
            raise RuntimeError(f"post-shard verification failed: {key}")

    mirror.write_manifest(main_manifest, merged)
    completion = {
        "schema_version": mirror.MANIFEST_SCHEMA_VERSION,
        "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "manifest": str(main_manifest),
        "manifest_sha256": mirror.sha256_file(main_manifest),
        "asset_names": list(mirror.ASSETS),
        "item_ids": item_ids,
        "item_identity_sha256": mirror.sha256_text(item_ids),
        "scope": "full_availability",
        "n_items": len(item_ids),
        "n_assets": len(expected),
        "n_manifest_assets": len(merged),
        "content_length_bytes": sum(int(merged[key]["content_length"]) for key in expected),
        "download_topology": {
            "shards": args.shards,
            "workers_per_shard": args.workers_per_shard,
            "disjoint_by": "STAC item_id",
        },
    }
    mirror.atomic_write_text(
        main_marker,
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
