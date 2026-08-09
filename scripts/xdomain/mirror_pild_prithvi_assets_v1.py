#!/usr/bin/env python3
"""Freeze the label-free PILD Sentinel-2 assets into a local mirror.

The availability registry is the only selection input. Each completed asset is
recorded with its exact byte length and SHA256 after an atomic ``.part`` rename.
The JSONL manifest is checkpointed after every asset, while the complete marker
is emitted only after the requested item scope is fully present and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
ASSETS = ("B02", "B03", "B04", "B8A", "B11", "B12", "SCL")
AUTH_FAILURE_CODES = {401, 403, 409}
MANIFEST_SCHEMA_VERSION = 1
CHUNK_SIZE = 8 * 1024 * 1024
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class DownloadError(RuntimeError):
    """A retryable or terminal asset download failure."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AssetSpec:
    item_id: str
    item_datetime: str
    collection: str
    asset_name: str
    unsigned_href: str
    media_type: str | None
    roles: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = PROJECT_ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    parser.add_argument("--availability", type=Path, default=base / "acquisition_availability_v1.csv")
    parser.add_argument("--mirror-root", type=Path, default=base / "sentinel2_asset_mirror_v1")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL manifest; must be inside --mirror-root (default: asset_mirror_manifest_v1.jsonl).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument(
        "--download-retries",
        type=int,
        default=100,
        help="Per-asset reconnect budget; successful partial progress resumes immediately.",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument(
        "--chunk-size-mib",
        type=float,
        default=0.25,
        help="Streaming write chunk in MiB; small chunks preserve progress across broken responses.",
    )
    parser.add_argument("--item-id", action="append", default=[])
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
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


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def split_item_ids(value: str) -> list[str]:
    groups = [[item.strip() for item in part.split(";") if item.strip()] for part in str(value).split("|")]
    if len(groups) != 4 or any(not group for group in groups):
        raise RuntimeError(f"expected four non-empty observations, got {value!r}")
    return [item for group in groups for item in group]


def read_selected_item_ids(path: Path) -> list[str]:
    availability = pd.read_csv(
        path,
        usecols=["selected_item_ids", "status", "prithvi_temporal_ready", "selection_uses_labels"],
    )
    if (availability["selection_uses_labels"].astype(int) != 0).any():
        raise RuntimeError("availability registry reports label-dependent image selection")
    ready = availability[
        (availability["status"].astype(str) == "ready")
        & (availability["prithvi_temporal_ready"].astype(int) == 1)
    ]
    if ready.empty:
        raise RuntimeError("availability registry has no Prithvi-ready acquisitions")
    item_ids = sorted({item for value in ready["selected_item_ids"] for item in split_item_ids(value)})
    for item_id in item_ids:
        if not SAFE_ID.fullmatch(item_id):
            raise RuntimeError(f"unsafe STAC item ID: {item_id!r}")
    return item_ids


def select_item_ids(all_item_ids: list[str], requested: list[str], max_items: int) -> list[str]:
    if requested:
        unknown = sorted(set(requested) - set(all_item_ids))
        if unknown:
            raise RuntimeError(f"requested item IDs are absent from availability: {unknown}")
        selected = list(dict.fromkeys(requested))
    else:
        selected = list(all_item_ids)
    if max_items:
        if max_items < 1:
            raise ValueError("--max-items must be positive")
        selected = selected[:max_items]
    if not selected:
        raise RuntimeError("no items selected")
    return selected


class PlanetaryComputerAssets:
    """Thread-safe signed-URL resolver with forced refresh after auth expiry."""

    def __init__(self, item_ids: list[str], retries: int):
        self.retries = retries
        self._lock = threading.RLock()
        self._client: Any = None
        self._raw: dict[str, Any] = {}
        self._signed: dict[str, Any] = {}
        self._fetch(item_ids)

    def _catalog(self):
        if self._client is None:
            from pystac_client import Client

            self._client = Client.open(CATALOG_URL)
        return self._client

    def _fetch(self, item_ids: list[str]) -> None:
        import planetary_computer

        wanted = sorted(set(item_ids))
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                items = list(
                    self._catalog()
                    .search(collections=[COLLECTION], ids=wanted, max_items=len(wanted))
                    .items()
                )
                found = {item.id: item for item in items}
                missing = sorted(set(wanted) - set(found))
                if missing:
                    raise RuntimeError(f"STAC did not return item IDs: {missing}")
                for item_id, item in found.items():
                    absent = sorted(set(ASSETS) - set(item.assets))
                    if absent:
                        raise RuntimeError(f"{item_id} lacks required assets: {absent}")
                    if item.datetime is None:
                        raise RuntimeError(f"{item_id} has no STAC datetime")
                    self._raw[item_id] = item
                    self._signed[item_id] = planetary_computer.sign(item)
                return
            except Exception as error:
                last_error = error
                self._client = None
                if attempt + 1 < self.retries:
                    time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"unable to fetch STAC items after {self.retries} attempts: {last_error}")

    def specs(self, item_ids: list[str]) -> list[AssetSpec]:
        result = []
        with self._lock:
            for item_id in item_ids:
                item = self._raw[item_id]
                for asset_name in ASSETS:
                    asset = item.assets[asset_name]
                    result.append(
                        AssetSpec(
                            item_id=item_id,
                            item_datetime=item.datetime.isoformat(),
                            collection=str(item.collection_id),
                            asset_name=asset_name,
                            unsigned_href=asset.href.split("?", 1)[0],
                            media_type=asset.media_type,
                            roles=tuple(asset.roles or ()),
                        )
                    )
        return result

    def href(self, item_id: str, asset_name: str, refresh: bool = False) -> str:
        with self._lock:
            if refresh:
                self._fetch([item_id])
            return str(self._signed[item_id].assets[asset_name].href)


def load_manifest(path: Path, availability_sha256: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported manifest schema at line {line_number}")
        if record.get("availability_sha256") != availability_sha256:
            raise RuntimeError("manifest was created from a different availability registry")
        key = (str(record["item_id"]), str(record["asset_name"]))
        if key in records and records[key] != record:
            raise RuntimeError(f"conflicting manifest records for {key}")
        records[key] = record
    return records


def write_manifest(path: Path, records: dict[tuple[str, str], dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(records[key], sort_keys=True, allow_nan=False) + "\n" for key in sorted(records)
    )
    atomic_write_text(path, content)


def validate_completed_record(record: dict[str, Any], destination: Path) -> bool:
    try:
        expected_size = int(record["content_length"])
        expected_hash = str(record["sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None
        and sha256_file(destination) == expected_hash
    )


def response_total_length(response: requests.Response, offset: int) -> tuple[int, str]:
    if response.status_code == 206:
        value = response.headers.get("Content-Range", "")
        match = CONTENT_RANGE.fullmatch(value)
        if not match:
            raise DownloadError(f"invalid Content-Range: {value!r}")
        start, end, total = map(int, match.groups())
        if start != offset or end < start or end >= total:
            raise DownloadError(f"unexpected Content-Range for offset {offset}: {value!r}")
        return total, "ab"
    if response.status_code == 200:
        value = response.headers.get("Content-Length")
        if value is None:
            raise DownloadError("source response lacks Content-Length")
        return int(value), "wb"
    raise DownloadError(
        f"HTTP {response.status_code}: {response.reason}", status_code=response.status_code
    )


def transfer_once(
    href: str,
    part_path: Path,
    timeout: tuple[float, float],
    chunk_size: int,
) -> int:
    offset = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with requests.get(href, headers=headers, stream=True, timeout=timeout) as response:
        if response.status_code in AUTH_FAILURE_CODES:
            raise DownloadError(
                f"signed URL rejected with HTTP {response.status_code}",
                status_code=response.status_code,
            )
        if response.status_code == 416:
            value = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes \*/(\d+)", value)
            if offset and match and int(match.group(1)) == offset:
                return offset
            raise DownloadError(f"range rejected for partial size {offset}: {value!r}", status_code=416)
        total, mode = response_total_length(response, offset)
        if mode == "wb":
            offset = 0
        part_path.parent.mkdir(parents=True, exist_ok=True)
        with part_path.open(mode) as stream:
            for block in response.iter_content(chunk_size=chunk_size):
                if block:
                    stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
    actual = part_path.stat().st_size
    if actual != total:
        raise DownloadError(f"incomplete response: expected {total} bytes, have {actual}")
    return total


def download_asset(
    spec: AssetSpec,
    destination: Path,
    relative_path: str,
    prior_record: dict[str, Any] | None,
    resolver: PlanetaryComputerAssets,
    availability_sha256: str,
    retries: int,
    timeout: tuple[float, float],
    chunk_size: int,
) -> tuple[dict[str, Any], bool]:
    if prior_record is not None and validate_completed_record(prior_record, destination):
        return prior_record, True

    part_path = destination.with_name(f"{destination.name}.part")
    last_error: Exception | None = None
    refresh = False
    for attempt in range(retries):
        previous_size = part_path.stat().st_size if part_path.exists() else 0
        try:
            href = resolver.href(spec.item_id, spec.asset_name, refresh=refresh)
            refresh = False
            content_length = transfer_once(href, part_path, timeout, chunk_size)
            if part_path.stat().st_size != content_length:
                raise DownloadError("partial file length changed before commit")
            digest = sha256_file(part_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part_path, destination)
            fsync_directory(destination.parent)
            record = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "availability_sha256": availability_sha256,
                "collection": spec.collection,
                "item_id": spec.item_id,
                "item_datetime": spec.item_datetime,
                "asset_name": spec.asset_name,
                "unsigned_href": spec.unsigned_href,
                "media_type": spec.media_type,
                "roles": list(spec.roles),
                "local_path": relative_path,
                "content_length": content_length,
                "sha256": digest,
                "completed_at_utc": utc_now(),
            }
            return record, False
        except (DownloadError, requests.RequestException, OSError) as error:
            last_error = error
            if isinstance(error, DownloadError) and error.status_code in AUTH_FAILURE_CODES:
                refresh = True
            if attempt + 1 < retries:
                current_size = part_path.stat().st_size if part_path.exists() else 0
                if current_size <= previous_size:
                    time.sleep(min(5, 2 ** min(attempt, 3)))
    raise RuntimeError(
        f"{spec.item_id}/{spec.asset_name} failed after {retries} attempts: {last_error}"
    )


def complete_marker_path(manifest: Path) -> Path:
    return manifest.with_name(f"{manifest.stem}.complete.json")


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.retries < 1 or args.download_retries < 1:
        raise ValueError("--workers, --retries, and --download-retries must be positive")
    if args.connect_timeout <= 0 or args.read_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if args.chunk_size_mib <= 0:
        raise ValueError("--chunk-size-mib must be positive")
    chunk_size = int(args.chunk_size_mib * 1024 * 1024)
    if chunk_size < 1:
        raise ValueError("--chunk-size-mib is too small")

    availability = args.availability.resolve()
    mirror_root = args.mirror_root.resolve()
    manifest = (args.manifest or mirror_root / "asset_mirror_manifest_v1.jsonl").resolve()
    if manifest.parent != mirror_root:
        raise ValueError("--manifest must be directly inside --mirror-root")
    mirror_root.mkdir(parents=True, exist_ok=True)

    availability_hash = sha256_file(availability)
    all_item_ids = read_selected_item_ids(availability)
    item_ids = select_item_ids(all_item_ids, args.item_id, args.max_items)
    resolver = PlanetaryComputerAssets(item_ids, args.retries)
    specs = resolver.specs(item_ids)
    records = load_manifest(manifest, availability_hash)
    marker = complete_marker_path(manifest)
    marker.unlink(missing_ok=True)

    tasks = []
    for spec in specs:
        relative_path = f"assets/{spec.item_id}/{spec.asset_name}.tif"
        destination = mirror_root / relative_path
        tasks.append((spec, destination, relative_path, records.get((spec.item_id, spec.asset_name))))

    failures = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_tasks = {
            pool.submit(
                download_asset,
                spec,
                destination,
                relative_path,
                prior,
                resolver,
                availability_hash,
                args.download_retries,
                (args.connect_timeout, args.read_timeout),
                chunk_size,
            ): (spec, destination)
            for spec, destination, relative_path, prior in tasks
        }
        for position, future in enumerate(as_completed(future_tasks), start=1):
            spec, _ = future_tasks[future]
            try:
                record, was_skipped = future.result()
                records[(spec.item_id, spec.asset_name)] = record
                skipped += int(was_skipped)
                write_manifest(manifest, records)
                action = "verified" if was_skipped else "downloaded"
                print(
                    f"[{position}/{len(tasks)}] {action} {spec.item_id}/{spec.asset_name} "
                    f"{int(record['content_length']) / (1024 ** 2):.1f} MiB",
                    flush=True,
                )
            except Exception as error:
                failures.append(f"{spec.item_id}/{spec.asset_name}: {error}")
                print(f"[{position}/{len(tasks)}] FAILED {failures[-1]}", flush=True)

    if failures:
        raise RuntimeError(f"mirror incomplete with {len(failures)} failed assets: {failures[:5]}")

    scope_keys = {(item_id, asset) for item_id in item_ids for asset in ASSETS}
    missing = sorted(scope_keys - set(records))
    if missing:
        raise RuntimeError(f"mirror scope lacks completed manifest records: {missing[:5]}")
    for item_id, asset_name in sorted(scope_keys):
        record = records[(item_id, asset_name)]
        path = mirror_root / str(record["local_path"])
        if not validate_completed_record(record, path):
            raise RuntimeError(f"post-download verification failed: {item_id}/{asset_name}")

    write_manifest(manifest, records)
    completion = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "complete": True,
        "completed_at_utc": utc_now(),
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "asset_names": list(ASSETS),
        "item_ids": item_ids,
        "item_identity_sha256": sha256_text(item_ids),
        "scope": "full_availability" if item_ids == all_item_ids else "selected_items",
        "n_items": len(item_ids),
        "n_assets": len(scope_keys),
        "n_manifest_assets": len(records),
        "content_length_bytes": sum(int(records[key]["content_length"]) for key in scope_keys),
        "verified_existing_assets": skipped,
    }
    atomic_write_text(marker, json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(completion, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
