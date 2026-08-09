#!/usr/bin/env python3
"""Resume the PILD Planetary Computer mirror through Azure Blob listing.

The frozen availability registry supplies the exact original item IDs.  This
resolver does not call a STAC API: it enumerates the corresponding Azure SAFE
folder with a short-lived Planetary Computer SAS token, validates acquisition
identity, resolves the seven required COGs, and resumes the existing ``.part``
files in the original mirror root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import requests

import mirror_pild_prithvi_assets_v1 as safeio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "sentinel2l2a01"
CONTAINER = "sentinel2-l2"
CONTAINER_URL = f"https://{ACCOUNT}.blob.core.windows.net/{CONTAINER}"
SAS_ENDPOINT = (
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/"
    f"{ACCOUNT}/{CONTAINER}"
)
ASSET_LAYOUT = {
    "B02": ("R10m", "B02", "10m"),
    "B03": ("R10m", "B03", "10m"),
    "B04": ("R10m", "B04", "10m"),
    "B8A": ("R20m", "B8A", "20m"),
    "B11": ("R20m", "B11", "20m"),
    "B12": ("R20m", "B12", "20m"),
    "SCL": ("R20m", "SCL", "20m"),
}
ASSETS = tuple(ASSET_LAYOUT)
ITEM_RE = re.compile(
    r"^(?P<satellite>S2[AB])_MSIL2A_(?P<sensing>\d{8}T\d{6})_"
    r"R(?P<orbit>\d{3})_T(?P<tile>\d{2}[A-Z]{3})_"
    r"(?P<generation>\d{8}T\d{6})$"
)
SAFE_RE = re.compile(
    r"^(?P<satellite>S2[AB])_MSIL2A_(?P<sensing>\d{8}T\d{6})_"
    r"N(?P<baseline>\d{4})_R(?P<orbit>\d{3})_"
    r"T(?P<tile>\d{2}[A-Z]{3})_(?P<generation>\d{8}T\d{6})\.SAFE$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESOLUTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ItemIdentity:
    item_id: str
    satellite: str
    sensing: str
    orbit: str
    tile: str
    generation: str

    @property
    def date_parts(self) -> tuple[str, str, str]:
        return self.sensing[:4], self.sensing[4:6], self.sensing[6:8]

    @property
    def tile_parts(self) -> tuple[str, str, str]:
        return self.tile[:2], self.tile[2], self.tile[3:]


@dataclass(frozen=True)
class BlobEntry:
    name: str
    content_length: int
    etag: str | None
    content_md5: str | None


@dataclass(frozen=True)
class BlobAssetSpec:
    item_id: str
    safe_name: str
    granule_name: str
    processing_baseline: str
    asset_name: str
    resolution: str
    blob_name: str
    unsigned_href: str
    content_length: int
    etag: str | None
    content_md5: str | None


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, default=base / "acquisition_availability_v1.csv")
    parser.add_argument("--mirror-root", type=Path, default=base / "sentinel2_asset_mirror_v1")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--resolution-manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--download-retries", type=int, default=100)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--chunk-size-mib", type=float, default=0.25)
    parser.add_argument("--item-id", action="append", default=[])
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def parse_item_id(item_id: str) -> ItemIdentity:
    match = ITEM_RE.fullmatch(str(item_id))
    if match is None:
        raise RuntimeError(f"unsupported frozen Sentinel-2 item ID: {item_id!r}")
    return ItemIdentity(item_id=item_id, **match.groupdict())


def blob_date_prefix(identity: ItemIdentity) -> str:
    zone, band, square = identity.tile_parts
    year, month, day = identity.date_parts
    return f"{zone}/{band}/{square}/{year}/{month}/{day}/"


def blob_product_prefix(identity: ItemIdentity) -> str:
    return (
        f"{blob_date_prefix(identity)}"
        f"{identity.satellite}_MSIL2A_{identity.sensing}_"
    )


def append_query(url: str, token: str) -> str:
    clean = str(token).strip().lstrip("?").lstrip("&")
    if not clean:
        raise RuntimeError("empty SAS token")
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{clean}"


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class SASTokenProvider:
    """Thread-safe SAS cache with expiry and forced refresh support."""

    def __init__(
        self,
        endpoint: str = SAS_ENDPOINT,
        retries: int = 6,
        timeout: tuple[float, float] = (30.0, 60.0),
        refresh_margin_seconds: float = 120.0,
    ):
        self.endpoint = endpoint
        self.retries = retries
        self.timeout = timeout
        self.refresh_margin_seconds = refresh_margin_seconds
        self._lock = threading.RLock()
        self._token: str | None = None
        self._expiry: datetime | None = None
        self.refresh_count = 0

    def _valid(self) -> bool:
        if self._token is None or self._expiry is None:
            return False
        return (self._expiry - datetime.now(timezone.utc)).total_seconds() > self.refresh_margin_seconds

    def refresh(self, force: bool = False) -> str:
        with self._lock:
            if not force and self._valid():
                return str(self._token)
            last_error: Exception | None = None
            for attempt in range(self.retries):
                try:
                    response = requests.get(self.endpoint, timeout=self.timeout)
                    response.raise_for_status()
                    payload = response.json()
                    token = str(payload.get("token", "")).lstrip("?")
                    expiry = parse_utc(str(payload.get("msft:expiry", "")))
                    if not token:
                        raise RuntimeError("SAS endpoint returned an empty token")
                    if expiry <= datetime.now(timezone.utc):
                        raise RuntimeError("SAS endpoint returned an expired token")
                    self._token = token
                    self._expiry = expiry
                    self.refresh_count += 1
                    return token
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    last_error = error
                    if attempt + 1 < self.retries:
                        time.sleep(min(10, 2 ** attempt))
            raise RuntimeError(f"unable to refresh Planetary Computer SAS token: {last_error}")

    def sign(self, unsigned_url: str, force_refresh: bool = False) -> str:
        return append_query(unsigned_url, self.refresh(force=force_refresh))


def _local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element.iter():
        if _local_tag(child) == name:
            return child.text
    return None


def parse_blob_listing_xml(content: bytes) -> tuple[list[BlobEntry], str]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise RuntimeError(f"invalid Azure Blob listing XML: {error}") from error
    entries: list[BlobEntry] = []
    next_marker = ""
    for element in root.iter():
        tag = _local_tag(element)
        if tag == "Blob":
            name = _child_text(element, "Name")
            length_text = _child_text(element, "Content-Length")
            if not name or length_text is None:
                raise RuntimeError("Azure Blob listing entry lacks Name or Content-Length")
            length = int(length_text)
            if length < 1:
                raise RuntimeError(f"Azure Blob listing reports non-positive size for {name}")
            entries.append(
                BlobEntry(
                    name=name,
                    content_length=length,
                    etag=_child_text(element, "Etag"),
                    content_md5=_child_text(element, "Content-MD5"),
                )
            )
        elif tag == "NextMarker":
            next_marker = (element.text or "").strip()
    return entries, next_marker


def list_blobs(
    prefix: str,
    provider: SASTokenProvider,
    retries: int,
    timeout: tuple[float, float],
    max_results: int = 5000,
) -> list[BlobEntry]:
    if retries < 1 or max_results < 1:
        raise ValueError("listing retry and page-size values must be positive")
    marker = ""
    records: dict[str, BlobEntry] = {}
    seen_markers: set[str] = set()
    while True:
        query = {"restype": "container", "comp": "list", "prefix": prefix, "maxresults": max_results}
        if marker:
            query["marker"] = marker
        unsigned_url = f"{CONTAINER_URL}?{urlencode(query)}"
        last_error: Exception | None = None
        response_content: bytes | None = None
        force_refresh = False
        for attempt in range(retries):
            try:
                response = requests.get(
                    provider.sign(unsigned_url, force_refresh=force_refresh),
                    timeout=timeout,
                )
                if response.status_code in {401, 403}:
                    force_refresh = True
                    raise safeio.DownloadError(
                        f"Azure listing SAS rejected with HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                response.raise_for_status()
                response_content = response.content
                break
            except (requests.RequestException, safeio.DownloadError) as error:
                last_error = error
                if attempt + 1 < retries:
                    time.sleep(min(5, 2 ** min(attempt, 3)))
        if response_content is None:
            raise RuntimeError(f"Azure Blob listing failed for prefix {prefix!r}: {last_error}")
        page, next_marker = parse_blob_listing_xml(response_content)
        for entry in page:
            if not entry.name.startswith(prefix):
                raise RuntimeError(f"Azure listing escaped requested prefix: {entry.name}")
            prior = records.get(entry.name)
            if prior is not None and prior != entry:
                raise RuntimeError(f"conflicting Azure Blob listing entry: {entry.name}")
            records[entry.name] = entry
        if not next_marker:
            break
        if next_marker in seen_markers:
            raise RuntimeError(f"Azure Blob listing repeated continuation marker: {next_marker}")
        seen_markers.add(next_marker)
        marker = next_marker
    return [records[name] for name in sorted(records)]


def validate_safe_identity(identity: ItemIdentity, safe_name: str) -> str:
    match = SAFE_RE.fullmatch(safe_name)
    if match is None:
        raise RuntimeError(f"invalid SAFE name under frozen prefix: {safe_name}")
    values = match.groupdict()
    expected = {
        "satellite": identity.satellite,
        "sensing": identity.sensing,
        "orbit": identity.orbit,
        "tile": identity.tile,
        "generation": identity.generation,
    }
    failed = sorted(key for key, value in expected.items() if values[key] != value)
    if failed:
        raise RuntimeError(f"SAFE identity mismatch for {identity.item_id}: {failed}")
    return values["baseline"]


def resolve_blob_assets(identity: ItemIdentity, entries: list[BlobEntry]) -> list[BlobAssetSpec]:
    date_prefix = blob_date_prefix(identity)
    safe_names: set[str] = set()
    for entry in entries:
        if not entry.name.startswith(blob_product_prefix(identity)):
            raise RuntimeError(f"Blob escaped frozen product prefix: {entry.name}")
        relative = entry.name.removeprefix(date_prefix)
        components = relative.split("/")
        if components and components[0].endswith(".SAFE"):
            safe_names.add(components[0])
    safe_names = sorted(safe_names)
    exact: list[tuple[str, str]] = []
    mismatches: list[str] = []
    for safe_name in safe_names:
        try:
            exact.append((safe_name, validate_safe_identity(identity, safe_name)))
        except RuntimeError as error:
            mismatches.append(str(error))
    if len(exact) != 1:
        raise RuntimeError(
            f"expected one exact SAFE for {identity.item_id}, found {len(exact)}; "
            f"candidates={safe_names}; mismatches={mismatches[:3]}"
        )
    safe_name, baseline = exact[0]
    safe_prefix = f"{date_prefix}{safe_name}/"
    by_name = {entry.name: entry for entry in entries if entry.name.startswith(safe_prefix)}
    resolved: list[BlobAssetSpec] = []
    granules: set[str] = set()
    for asset_name, (resolution, band, size) in ASSET_LAYOUT.items():
        suffix = f"/IMG_DATA/{resolution}/T{identity.tile}_{identity.sensing}_{band}_{size}.tif"
        matches = [entry for name, entry in by_name.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {asset_name} {resolution} blob for {identity.item_id}, "
                f"found {len(matches)}"
            )
        entry = matches[0]
        relative = entry.name.removeprefix(safe_prefix)
        parts = relative.split("/")
        if len(parts) < 5 or parts[0] != "GRANULE":
            raise RuntimeError(f"unexpected GRANULE path for {identity.item_id}: {entry.name}")
        granule = parts[1]
        granules.add(granule)
        resolved.append(
            BlobAssetSpec(
                item_id=identity.item_id,
                safe_name=safe_name,
                granule_name=granule,
                processing_baseline=baseline,
                asset_name=asset_name,
                resolution=resolution,
                blob_name=entry.name,
                unsigned_href=f"{CONTAINER_URL}/{quote(entry.name, safe='/')}",
                content_length=entry.content_length,
                etag=entry.etag,
                content_md5=entry.content_md5,
            )
        )
    if len(granules) != 1:
        raise RuntimeError(f"required assets span multiple GRANULE folders for {identity.item_id}: {granules}")
    return resolved


def resolve_item(
    item_id: str,
    provider: SASTokenProvider,
    retries: int,
    timeout: tuple[float, float],
) -> list[BlobAssetSpec]:
    identity = parse_item_id(item_id)
    entries = list_blobs(blob_product_prefix(identity), provider, retries, timeout)
    return resolve_blob_assets(identity, entries)


def resolution_record(spec: BlobAssetSpec, availability_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": RESOLUTION_SCHEMA_VERSION,
        "availability_sha256": availability_sha256,
        "resolver": "planetary-computer-azure-blob-listing",
        "stac_api_used": False,
        "sas_endpoint": SAS_ENDPOINT,
        "account": ACCOUNT,
        "container": CONTAINER,
        "item_id": spec.item_id,
        "acquisition_identity_match": True,
        "identity_policy": "exact satellite+sensing+orbit+tile+generation; baseline may vary",
        "safe_name": spec.safe_name,
        "processing_baseline": spec.processing_baseline,
        "granule_name": spec.granule_name,
        "asset_name": spec.asset_name,
        "resolution": spec.resolution,
        "blob_name": spec.blob_name,
        "unsigned_href": spec.unsigned_href,
        "remote_content_length": spec.content_length,
        "etag": spec.etag,
        "content_md5": spec.content_md5,
    }


def load_resolution_manifest(
    path: Path, availability_sha256: str
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("schema_version", -1)) != RESOLUTION_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported Blob resolution schema at line {line_number}")
        if record.get("availability_sha256") != availability_sha256:
            raise RuntimeError("Blob resolution manifest uses another availability registry")
        if record.get("stac_api_used") is not False or record.get("acquisition_identity_match") is not True:
            raise RuntimeError("Blob resolution manifest contains unverified provenance")
        key = (str(record["item_id"]), str(record["asset_name"]))
        if key in records and records[key] != record:
            raise RuntimeError(f"conflicting Blob resolution records for {key}")
        records[key] = record
    return records


def write_jsonl(path: Path, records: dict[tuple[str, str], dict[str, Any]]) -> None:
    safeio.atomic_write_text(
        path,
        "".join(
            json.dumps(records[key], sort_keys=True, allow_nan=False) + "\n"
            for key in sorted(records)
        ),
    )


def validate_prior_record(
    record: dict[str, Any], spec: BlobAssetSpec, relative_path: str
) -> None:
    expected = {
        "item_id": spec.item_id,
        "asset_name": spec.asset_name,
        "local_path": relative_path,
        "unsigned_href": spec.unsigned_href,
        "content_length": spec.content_length,
    }
    failed = sorted(key for key, value in expected.items() if record.get(key) != value)
    if failed:
        raise RuntimeError(f"existing mirror record conflicts with Blob resolution: {failed}")
    digest = str(record.get("sha256", ""))
    if SHA256_RE.fullmatch(digest) is None:
        raise RuntimeError("existing mirror record has invalid SHA256")


def make_asset_record(
    spec: BlobAssetSpec,
    relative_path: str,
    availability_sha256: str,
    digest: str,
    adopted_existing: bool,
) -> dict[str, Any]:
    identity = parse_item_id(spec.item_id)
    item_datetime = datetime.strptime(identity.sensing, "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    ).isoformat()
    return {
        "schema_version": safeio.MANIFEST_SCHEMA_VERSION,
        "availability_sha256": availability_sha256,
        "collection": safeio.COLLECTION,
        "item_id": spec.item_id,
        "item_datetime": item_datetime,
        "asset_name": spec.asset_name,
        "unsigned_href": spec.unsigned_href,
        "media_type": "image/tiff; application=geotiff; profile=cloud-optimized",
        "roles": ["data"],
        "local_path": relative_path,
        "content_length": spec.content_length,
        "sha256": digest,
        "completed_at_utc": safeio.utc_now(),
        "resolver": "planetary-computer-azure-blob-listing",
        "stac_api_used": False,
        "safe_name": spec.safe_name,
        "granule_name": spec.granule_name,
        "processing_baseline": spec.processing_baseline,
        "blob_etag": spec.etag,
        "adopted_existing_file": adopted_existing,
    }


def download_asset(
    spec: BlobAssetSpec,
    destination: Path,
    relative_path: str,
    prior_record: dict[str, Any] | None,
    provider: SASTokenProvider,
    availability_sha256: str,
    retries: int,
    timeout: tuple[float, float],
    chunk_size: int,
) -> tuple[dict[str, Any], bool]:
    if prior_record is not None:
        validate_prior_record(prior_record, spec, relative_path)
        if safeio.validate_completed_record(prior_record, destination):
            return prior_record, True
    if destination.exists():
        if destination.stat().st_size != spec.content_length:
            raise RuntimeError(f"existing final file has wrong length: {destination}")
        digest = safeio.sha256_file(destination)
        return make_asset_record(
            spec, relative_path, availability_sha256, digest, adopted_existing=True
        ), True

    part_path = destination.with_name(f"{destination.name}.part")
    if part_path.exists() and part_path.stat().st_size > spec.content_length:
        raise RuntimeError(f"partial file exceeds remote Content-Length: {part_path}")
    last_error: Exception | None = None
    force_refresh = False
    for attempt in range(retries):
        previous_size = part_path.stat().st_size if part_path.exists() else 0
        try:
            signed_href = provider.sign(spec.unsigned_href, force_refresh=force_refresh)
            force_refresh = False
            content_length = safeio.transfer_once(signed_href, part_path, timeout, chunk_size)
            if content_length != spec.content_length:
                raise safeio.DownloadError(
                    f"listing/response Content-Length mismatch: {spec.content_length} != {content_length}"
                )
            digest = safeio.sha256_file(part_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part_path, destination)
            safeio.fsync_directory(destination.parent)
            return make_asset_record(
                spec, relative_path, availability_sha256, digest, adopted_existing=False
            ), False
        except (safeio.DownloadError, requests.RequestException, OSError) as error:
            last_error = error
            if isinstance(error, safeio.DownloadError) and error.status_code in {401, 403}:
                force_refresh = True
            if attempt + 1 < retries:
                current_size = part_path.stat().st_size if part_path.exists() else 0
                if current_size <= previous_size:
                    time.sleep(min(5, 2 ** min(attempt, 3)))
    raise RuntimeError(f"{spec.item_id}/{spec.asset_name} failed after {retries} attempts: {last_error}")


def partition_item_ids(item_ids: list[str], shards: int, shard_index: int) -> list[str]:
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("--shards must be positive and --shard-index must be in range")
    return item_ids[shard_index::shards]


def complete_marker_path(manifest: Path) -> Path:
    return safeio.complete_marker_path(manifest)


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.retries < 1 or args.download_retries < 1:
        raise ValueError("workers and retry counts must be positive")
    if args.connect_timeout <= 0 or args.read_timeout <= 0 or args.chunk_size_mib <= 0:
        raise ValueError("timeouts and chunk size must be positive")
    availability = args.availability.resolve()
    mirror_root = args.mirror_root.resolve()
    mirror_root.mkdir(parents=True, exist_ok=True)
    default_manifest = (
        mirror_root / "asset_mirror_manifest_v1.jsonl"
        if args.shards == 1
        else mirror_root / f"asset_mirror_blob_shard{args.shard_index:02d}_v1.jsonl"
    )
    manifest = (args.manifest or default_manifest).resolve()
    resolution_manifest = (
        args.resolution_manifest
        or mirror_root / f"planetary_blob_resolution_{manifest.stem}.jsonl"
    ).resolve()
    if manifest.parent != mirror_root or resolution_manifest.parent != mirror_root:
        raise ValueError("manifest files must be directly inside --mirror-root")
    marker = complete_marker_path(manifest)
    marker.unlink(missing_ok=True)

    availability_hash = safeio.sha256_file(availability)
    all_item_ids = safeio.read_selected_item_ids(availability)
    requested = safeio.select_item_ids(all_item_ids, args.item_id, args.max_items)
    item_ids = partition_item_ids(requested, args.shards, args.shard_index)
    if not item_ids:
        raise RuntimeError("selected Blob recovery shard is empty")
    provider = SASTokenProvider(
        retries=args.retries,
        timeout=(args.connect_timeout, min(args.read_timeout, 60.0)),
    )
    resolution_records = load_resolution_manifest(resolution_manifest, availability_hash)
    specs: list[BlobAssetSpec] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                resolve_item,
                item_id,
                provider,
                args.retries,
                (args.connect_timeout, args.read_timeout),
            ): item_id
            for item_id in item_ids
        }
        for position, future in enumerate(as_completed(futures), start=1):
            item_specs = future.result()
            specs.extend(item_specs)
            for spec in item_specs:
                key = (spec.item_id, spec.asset_name)
                record = resolution_record(spec, availability_hash)
                prior = resolution_records.get(key)
                if prior is not None and prior != record:
                    raise RuntimeError(f"conflicting rerun Blob resolution for {key}")
                resolution_records[key] = record
            write_jsonl(resolution_manifest, resolution_records)
            print(f"[resolve {position}/{len(item_ids)}] {item_specs[0].item_id}", flush=True)

    records = safeio.load_manifest(manifest, availability_hash)
    tasks = []
    for spec in specs:
        relative = f"assets/{spec.item_id}/{spec.asset_name}.tif"
        tasks.append((spec, mirror_root / relative, relative, records.get((spec.item_id, spec.asset_name))))
    skipped = 0
    chunk_size = int(args.chunk_size_mib * 1024 * 1024)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_asset,
                spec,
                destination,
                relative,
                prior,
                provider,
                availability_hash,
                args.download_retries,
                (args.connect_timeout, args.read_timeout),
                chunk_size,
            ): spec
            for spec, destination, relative, prior in tasks
        }
        for position, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            record, was_skipped = future.result()
            records[(spec.item_id, spec.asset_name)] = record
            skipped += int(was_skipped)
            safeio.write_manifest(manifest, records)
            action = "verified" if was_skipped else "downloaded"
            print(f"[{position}/{len(tasks)}] {action} {spec.item_id}/{spec.asset_name}", flush=True)

    expected = {(item_id, asset) for item_id in item_ids for asset in ASSETS}
    if expected - set(records) or expected - set(resolution_records):
        raise RuntimeError("Blob recovery scope lacks manifest or resolution records")
    for key in sorted(expected):
        if not safeio.validate_completed_record(
            records[key], mirror_root / str(records[key]["local_path"])
        ):
            raise RuntimeError(f"post-recovery verification failed: {key}")
    safeio.write_manifest(manifest, records)
    write_jsonl(resolution_manifest, resolution_records)
    completion = {
        "schema_version": 1,
        "complete": True,
        "completed_at_utc": safeio.utc_now(),
        "resolver": "planetary-computer-azure-blob-listing",
        "stac_api_used": False,
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "manifest": str(manifest),
        "manifest_sha256": safeio.sha256_file(manifest),
        "resolution_manifest": str(resolution_manifest),
        "resolution_manifest_sha256": safeio.sha256_file(resolution_manifest),
        "asset_names": list(ASSETS),
        "item_ids": item_ids,
        "item_identity_sha256": safeio.sha256_text(item_ids),
        "scope": "full_availability" if item_ids == all_item_ids else "selected_items_or_shard",
        "shards": args.shards,
        "shard_index": args.shard_index,
        "n_items": len(item_ids),
        "n_assets": len(expected),
        "verified_existing_assets": skipped,
        "sas_refresh_count": provider.refresh_count,
    }
    safeio.atomic_write_text(
        marker, json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(completion, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
