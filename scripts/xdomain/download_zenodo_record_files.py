#!/usr/bin/env python3
"""Download selected files from a pinned Zenodo record with checksums."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path

import requests


def md5sum(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def download(item: dict[str, object], out_dir: Path, attempts: int) -> dict[str, object]:
    name = str(item["key"])
    size = int(item["size"])
    expected = str(item["checksum"]).split(":", 1)[1]
    target = out_dir / name
    part = target.with_name(target.name + ".part")
    if target.is_file() and target.stat().st_size == size and md5sum(target) == expected:
        return {"name": name, "status": "existing", "bytes": size}
    if target.exists():
        target.unlink()
    if part.exists() and part.stat().st_size > size:
        part.unlink()
    url = str(item["links"]["self"])
    error = ""
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            [
                "curl", "-fL", "-C", "-", "--connect-timeout", "30",
                "--speed-time", "180", "--speed-limit", "1024",
                "--retry", "6", "--retry-all-errors", "--retry-delay", "5",
                "--silent", "--show-error", url, "-o", str(part),
            ],
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0 and part.is_file() and part.stat().st_size == size:
            actual = md5sum(part)
            if actual == expected:
                part.replace(target)
                return {"name": name, "status": "downloaded", "bytes": size}
            error = f"md5 mismatch: {actual} != {expected}"
            part.unlink(missing_ok=True)
        else:
            actual_size = part.stat().st_size if part.exists() else 0
            error = f"curl={completed.returncode}, bytes={actual_size}/{size}, {completed.stderr[-300:]}"
            if completed.returncode == 33:
                part.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"{name}: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_id")
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    record_url = f"https://zenodo.org/api/records/{args.record_id}"
    response = requests.get(record_url, timeout=60)
    response.raise_for_status()
    record = response.json()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"zenodo_record_{args.record_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    by_name = {item["key"]: item for item in record["files"]}
    missing = set(args.file) - by_name.keys()
    if missing:
        raise SystemExit(f"Files absent from record {args.record_id}: {sorted(missing)}")
    selected = [by_name[name] for name in args.file]
    results: list[dict[str, object]] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, item, args.out_dir, args.attempts): item for item in selected}
        for future in concurrent.futures.as_completed(futures):
            name = str(futures[future]["key"])
            try:
                result = future.result()
                results.append(result)
                print(f"[{str(result['status']).upper()}] {name}", flush=True)
            except Exception as error:
                failures.append(f"{name}: {error}")
                print(f"[FAIL] {name}: {error}", flush=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(
            {
                "record_id": args.record_id,
                "doi": record.get("doi", ""),
                "results": sorted(results, key=lambda item: str(item["name"])),
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
