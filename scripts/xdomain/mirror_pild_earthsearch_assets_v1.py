#!/usr/bin/env python3
"""Mirror frozen PILD Sentinel-2 acquisitions from the Earth Search catalog.

This is an independent fallback for the Planetary Computer mirror.  The frozen
availability registry remains the sole acquisition-selection input; Earth
Search is used only to resolve an equivalent public product and its COG assets.
No file or manifest from the Planetary Computer mirror is read or reused.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests

import mirror_pild_prithvi_assets_v1 as safeio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
ASSET_MAP = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}
ASSETS = tuple(ASSET_MAP)
MANIFEST_SCHEMA_VERSION = 1
MAX_OCCURRENCE = 3

PC_ITEM_RE = re.compile(
    r"^(?P<satellite>S2[AB])_MSIL2A_"
    r"(?P<sensing>\d{8}T\d{6})_R(?P<orbit>\d{3})_"
    r"T(?P<tile>\d{2}[A-Z]{3})_(?P<generation>\d{8}T\d{6})$"
)
EARTHSEARCH_ITEM_RE = re.compile(
    r"^(?P<satellite>S2[AB])_(?P<tile>\d{2}[A-Z]{3})_"
    r"(?P<date>\d{8})_(?P<occurrence>[0-3])_L2A$"
)
PRODUCT_URI_RE = re.compile(
    r"^(?P<satellite>S2[AB])_MSIL2A_(?P<sensing>\d{8}T\d{6})_"
    r"N(?P<baseline>\d{4})_R(?P<orbit>\d{3})_"
    r"T(?P<tile>\d{2}[A-Z]{3})_(?P<generation>\d{8}T\d{6})\.SAFE$"
)
PLATFORM = {"S2A": "sentinel-2a", "S2B": "sentinel-2b"}


@dataclass(frozen=True)
class AcquisitionIdentity:
    satellite: str
    platform: str
    sensing: str
    sensing_date: str
    orbit: str
    tile: str
    original_generation: str


@dataclass(frozen=True)
class EarthSearchItem:
    original_item_id: str
    earthsearch_item_id: str
    earthsearch_product_uri: str
    item_datetime: str
    acquisition_identity_match: bool
    assets: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class AssetSpec:
    original_item_id: str
    earthsearch_item_id: str
    earthsearch_product_uri: str
    item_datetime: str
    acquisition_identity_match: bool
    asset_name: str
    earthsearch_asset_name: str
    href: str
    media_type: str | None
    roles: tuple[str, ...]


class AcquisitionIdentityMismatch(RuntimeError):
    """A catalog candidate exists but is not the frozen acquisition."""


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, default=base / "acquisition_availability_v1.csv")
    parser.add_argument(
        "--mirror-root",
        type=Path,
        default=base / "sentinel2_asset_mirror_earthsearch_v1",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
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


def parse_pc_item_id(item_id: str) -> AcquisitionIdentity:
    match = PC_ITEM_RE.fullmatch(str(item_id))
    if match is None:
        raise RuntimeError(f"unsupported Planetary Computer Sentinel-2 item ID: {item_id!r}")
    values = match.groupdict()
    satellite = values["satellite"]
    return AcquisitionIdentity(
        satellite=satellite,
        platform=PLATFORM[satellite],
        sensing=values["sensing"],
        sensing_date=values["sensing"][:8],
        orbit=values["orbit"],
        tile=values["tile"],
        original_generation=values["generation"],
    )


def candidate_earthsearch_ids(item_id: str, max_occurrence: int = MAX_OCCURRENCE) -> list[str]:
    if not 0 <= max_occurrence <= MAX_OCCURRENCE:
        raise ValueError(f"max_occurrence must be in [0, {MAX_OCCURRENCE}]")
    identity = parse_pc_item_id(item_id)
    return [
        f"{identity.satellite}_{identity.tile}_{identity.sensing_date}_{index}_L2A"
        for index in range(max_occurrence + 1)
    ]


def normalize_mgrs_tile(properties: dict[str, Any]) -> str:
    grid_code = str(properties.get("grid:code", ""))
    if grid_code.startswith("MGRS-"):
        return grid_code.removeprefix("MGRS-")
    zone = properties.get("mgrs:utm_zone")
    band = properties.get("mgrs:latitude_band")
    square = properties.get("mgrs:grid_square")
    if zone is None or not band or not square:
        return ""
    return f"{int(zone):02d}{str(band).upper()}{str(square).upper()}"


def validate_earthsearch_item(original_item_id: str, item: dict[str, Any]) -> EarthSearchItem:
    expected = parse_pc_item_id(original_item_id)
    earthsearch_item_id = str(item.get("id", ""))
    item_match = EARTHSEARCH_ITEM_RE.fullmatch(earthsearch_item_id)
    if item_match is None:
        raise RuntimeError(f"invalid Earth Search item ID for {original_item_id}: {earthsearch_item_id!r}")

    properties = item.get("properties")
    if not isinstance(properties, dict):
        raise RuntimeError(f"Earth Search item {earthsearch_item_id} lacks properties")
    product_uri = str(properties.get("s2:product_uri", ""))
    product_match = PRODUCT_URI_RE.fullmatch(product_uri)
    if product_match is None:
        raise RuntimeError(f"Earth Search item {earthsearch_item_id} has invalid s2:product_uri")

    item_values = item_match.groupdict()
    product_values = product_match.groupdict()
    actual_platform = str(properties.get("platform", "")).lower()
    actual_tile = normalize_mgrs_tile(properties)
    actual_datetime = str(properties.get("datetime", ""))
    actual_date = actual_datetime[:10].replace("-", "")
    checks = {
        "platform": actual_platform == expected.platform,
        "item_satellite": item_values["satellite"] == expected.satellite,
        "item_tile": item_values["tile"] == expected.tile,
        "item_date": item_values["date"] == expected.sensing_date,
        "property_tile": actual_tile == expected.tile,
        "property_date": actual_date == expected.sensing_date,
        "product_satellite": product_values["satellite"] == expected.satellite,
        "product_sensing": product_values["sensing"] == expected.sensing,
        "product_orbit": product_values["orbit"] == expected.orbit,
        "product_tile": product_values["tile"] == expected.tile,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AcquisitionIdentityMismatch(
            f"Earth Search acquisition identity mismatch for {original_item_id} -> "
            f"{earthsearch_item_id}: {failed}"
        )

    assets = item.get("assets")
    if not isinstance(assets, dict):
        raise RuntimeError(f"Earth Search item {earthsearch_item_id} lacks assets")
    missing = sorted(set(ASSET_MAP.values()) - set(assets))
    if missing:
        raise RuntimeError(f"Earth Search item {earthsearch_item_id} lacks assets: {missing}")
    for name in ASSET_MAP.values():
        asset = assets[name]
        href = str(asset.get("href", "")) if isinstance(asset, dict) else ""
        parsed = urlparse(href)
        if parsed.scheme != "https" or not parsed.netloc:
            raise RuntimeError(f"Earth Search item {earthsearch_item_id}/{name} has unsafe href")

    return EarthSearchItem(
        original_item_id=original_item_id,
        earthsearch_item_id=earthsearch_item_id,
        earthsearch_product_uri=product_uri,
        item_datetime=actual_datetime,
        acquisition_identity_match=True,
        assets=assets,
    )


def fetch_earthsearch_item(
    original_item_id: str,
    retries: int,
    timeout: tuple[float, float],
) -> EarthSearchItem:
    errors: list[str] = []
    for candidate in candidate_earthsearch_ids(original_item_id):
        url = f"{EARTHSEARCH_URL}/collections/{COLLECTION}/items/{quote(candidate, safe='')}"
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 404:
                    break
                response.raise_for_status()
                return validate_earthsearch_item(original_item_id, response.json())
            except AcquisitionIdentityMismatch as error:
                errors.append(f"{candidate}: {error}")
                break
            except (requests.RequestException, ValueError, RuntimeError) as error:
                errors.append(f"{candidate}: {error}")
                if isinstance(error, RuntimeError):
                    # Malformed metadata or unsafe assets must not be searched around.
                    raise
                if attempt + 1 < retries:
                    time.sleep(min(10, 2 ** attempt))
    raise RuntimeError(
        f"no identity-matched Earth Search occurrence 0..{MAX_OCCURRENCE} for "
        f"{original_item_id}; last errors: {errors[-3:]}"
    )


def item_to_specs(item: EarthSearchItem) -> list[AssetSpec]:
    specs = []
    for output_name, earth_name in ASSET_MAP.items():
        asset = item.assets[earth_name]
        specs.append(
            AssetSpec(
                original_item_id=item.original_item_id,
                earthsearch_item_id=item.earthsearch_item_id,
                earthsearch_product_uri=item.earthsearch_product_uri,
                item_datetime=item.item_datetime,
                acquisition_identity_match=item.acquisition_identity_match,
                asset_name=output_name,
                earthsearch_asset_name=earth_name,
                href=str(asset["href"]),
                media_type=asset.get("type"),
                roles=tuple(asset.get("roles") or ()),
            )
        )
    return specs


def load_manifest(path: Path, availability_sha256: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported Earth Search manifest schema at line {line_number}")
        if record.get("mirror_source") != "earth-search":
            raise RuntimeError("manifest contains non-Earth-Search assets")
        if record.get("availability_sha256") != availability_sha256:
            raise RuntimeError("manifest was created from a different availability registry")
        if record.get("acquisition_identity_match") is not True:
            raise RuntimeError("manifest contains an unverified acquisition identity")
        validate_manifest_identity_record(record)
        key = (str(record["original_item_id"]), str(record["asset_name"]))
        if key in records and records[key] != record:
            raise RuntimeError(f"conflicting Earth Search manifest records for {key}")
        records[key] = record
    return records


def validate_manifest_identity_record(record: dict[str, Any]) -> None:
    original_item_id = str(record.get("original_item_id", ""))
    earthsearch_item_id = str(record.get("earthsearch_item_id", ""))
    product_uri = str(record.get("earthsearch_product_uri", ""))
    expected = parse_pc_item_id(original_item_id)
    item_match = EARTHSEARCH_ITEM_RE.fullmatch(earthsearch_item_id)
    product_match = PRODUCT_URI_RE.fullmatch(product_uri)
    if item_match is None or product_match is None:
        raise RuntimeError(f"manifest contains invalid Earth Search identity for {original_item_id}")
    item_values = item_match.groupdict()
    product_values = product_match.groupdict()
    if (
        item_values["satellite"] != expected.satellite
        or item_values["tile"] != expected.tile
        or item_values["date"] != expected.sensing_date
        or product_values["satellite"] != expected.satellite
        or product_values["sensing"] != expected.sensing
        or product_values["orbit"] != expected.orbit
        or product_values["tile"] != expected.tile
    ):
        raise RuntimeError(f"manifest acquisition identity mismatch for {original_item_id}")
    asset_name = str(record.get("asset_name", ""))
    earthsearch_asset_name = str(record.get("earthsearch_asset_name", ""))
    if asset_name not in ASSET_MAP or ASSET_MAP[asset_name] != earthsearch_asset_name:
        raise RuntimeError(f"manifest asset mapping mismatch for {original_item_id}/{asset_name}")
    expected_path = f"assets/{original_item_id}/{asset_name}.tif"
    if record.get("local_path") != expected_path:
        raise RuntimeError(f"manifest has non-canonical local path for {original_item_id}/{asset_name}")
    href = urlparse(str(record.get("public_href", "")))
    if href.scheme != "https" or not href.netloc:
        raise RuntimeError(f"manifest has unsafe public href for {original_item_id}/{asset_name}")


def write_manifest(path: Path, records: dict[tuple[str, str], dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(records[key], sort_keys=True, allow_nan=False) + "\n"
        for key in sorted(records)
    )
    safeio.atomic_write_text(path, content)


def write_label_free_receipt(
    path: Path,
    availability: Path,
    availability_sha256: str,
    all_item_ids: list[str],
) -> None:
    receipt = {
        "schema_version": 1,
        "selection_uses_labels": False,
        "selection_source": str(availability),
        "selection_source_sha256": availability_sha256,
        "selection_column": "selected_item_ids",
        "selection_filter": "status=ready and prithvi_temporal_ready=1",
        "catalog_role": "identity-preserving asset resolution only",
        "catalog": "Earth Search",
        "collection": COLLECTION,
        "n_frozen_original_item_ids": len(all_item_ids),
        "frozen_original_item_identity_sha256": safeio.sha256_text(all_item_ids),
    }
    content = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError("conflicting label-free selection receipt")
    safeio.atomic_write_text(path, content)


def assert_independent_mirror_root(mirror_root: Path) -> None:
    base = PROJECT_ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    pc_root = (base / "sentinel2_asset_mirror_v1").resolve()
    if mirror_root.resolve() == pc_root:
        raise RuntimeError("Earth Search mirror root must not equal the Planetary Computer mirror root")
    pc_artifacts = (
        mirror_root / "asset_mirror_manifest_v1.jsonl",
        mirror_root / "asset_mirror_manifest_v1.complete.json",
        mirror_root / "sharded_download_pids_v1.json",
    )
    if any(path.exists() for path in pc_artifacts):
        raise RuntimeError("Earth Search mirror root contains Planetary Computer mirror artifacts")


def download_asset(
    spec: AssetSpec,
    destination: Path,
    relative_path: str,
    prior_record: dict[str, Any] | None,
    availability_sha256: str,
    retries: int,
    timeout: tuple[float, float],
    chunk_size: int,
) -> tuple[dict[str, Any], bool]:
    if prior_record is not None:
        validate_prior_record(spec, prior_record, relative_path)
        if safeio.validate_completed_record(prior_record, destination):
            return prior_record, True
    part_path = destination.with_name(f"{destination.name}.part")
    last_error: Exception | None = None
    for attempt in range(retries):
        previous_size = part_path.stat().st_size if part_path.exists() else 0
        try:
            content_length = safeio.transfer_once(spec.href, part_path, timeout, chunk_size)
            if part_path.stat().st_size != content_length:
                raise safeio.DownloadError("partial file length changed before commit")
            digest = safeio.sha256_file(part_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part_path, destination)
            safeio.fsync_directory(destination.parent)
            return {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "mirror_source": "earth-search",
                "availability_sha256": availability_sha256,
                "collection": COLLECTION,
                "original_item_id": spec.original_item_id,
                "earthsearch_item_id": spec.earthsearch_item_id,
                "earthsearch_product_uri": spec.earthsearch_product_uri,
                "acquisition_identity_policy": (
                    "exact satellite+sensing_timestamp+relative_orbit+MGRS_tile; "
                    "processing baseline and generation may differ"
                ),
                "acquisition_identity_match": spec.acquisition_identity_match,
                "item_datetime": spec.item_datetime,
                "asset_name": spec.asset_name,
                "earthsearch_asset_name": spec.earthsearch_asset_name,
                "public_href": spec.href,
                "media_type": spec.media_type,
                "roles": list(spec.roles),
                "local_path": relative_path,
                "content_length": content_length,
                "sha256": digest,
                "completed_at_utc": safeio.utc_now(),
            }, False
        except (safeio.DownloadError, requests.RequestException, OSError) as error:
            last_error = error
            if attempt + 1 < retries:
                current_size = part_path.stat().st_size if part_path.exists() else 0
                if current_size <= previous_size:
                    time.sleep(min(5, 2 ** min(attempt, 3)))
    raise RuntimeError(
        f"{spec.original_item_id}/{spec.asset_name} failed after {retries} attempts: {last_error}"
    )


def validate_prior_record(
    spec: AssetSpec,
    record: dict[str, Any],
    relative_path: str,
) -> None:
    expected = {
        "mirror_source": "earth-search",
        "original_item_id": spec.original_item_id,
        "earthsearch_item_id": spec.earthsearch_item_id,
        "earthsearch_product_uri": spec.earthsearch_product_uri,
        "acquisition_identity_match": True,
        "asset_name": spec.asset_name,
        "earthsearch_asset_name": spec.earthsearch_asset_name,
        "public_href": spec.href,
        "local_path": relative_path,
    }
    mismatches = sorted(key for key, value in expected.items() if record.get(key) != value)
    if mismatches:
        raise RuntimeError(
            f"prior Earth Search manifest conflicts with resolved asset "
            f"{spec.original_item_id}/{spec.asset_name}: {mismatches}"
        )


def complete_marker_path(manifest: Path) -> Path:
    return manifest.with_name(f"{manifest.stem}.complete.json")


def partition_item_ids(item_ids: list[str], shards: int, shard_index: int) -> list[str]:
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("--shards must be positive and --shard-index must be in [0, shards)")
    return item_ids[shard_index::shards]


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.retries < 1 or args.download_retries < 1:
        raise ValueError("worker and retry counts must be positive")
    if args.connect_timeout <= 0 or args.read_timeout <= 0 or args.chunk_size_mib <= 0:
        raise ValueError("timeouts and chunk size must be positive")

    availability = args.availability.resolve()
    mirror_root = args.mirror_root.resolve()
    default_manifest_name = (
        "asset_mirror_manifest_earthsearch_v1.jsonl"
        if args.shards == 1
        else f"asset_mirror_manifest_earthsearch_shard{args.shard_index:02d}_v1.jsonl"
    )
    manifest = (args.manifest or mirror_root / default_manifest_name).resolve()
    if manifest.parent != mirror_root:
        raise ValueError("--manifest must be directly inside --mirror-root")
    assert_independent_mirror_root(mirror_root)
    mirror_root.mkdir(parents=True, exist_ok=True)
    availability_hash = safeio.sha256_file(availability)
    all_item_ids = safeio.read_selected_item_ids(availability)
    requested = safeio.select_item_ids(all_item_ids, args.item_id, args.max_items)
    item_ids = partition_item_ids(requested, args.shards, args.shard_index)
    if not item_ids:
        raise RuntimeError("selected Earth Search shard is empty")
    write_label_free_receipt(
        mirror_root / "label_free_selection_receipt_v1.json",
        availability,
        availability_hash,
        all_item_ids,
    )
    marker = complete_marker_path(manifest)
    marker.unlink(missing_ok=True)
    resolution_failures_path = manifest.with_name(f"{manifest.stem}.resolution_failures.json")

    resolved: list[EarthSearchItem] = []
    resolution_failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                fetch_earthsearch_item,
                item_id,
                args.retries,
                (args.connect_timeout, args.read_timeout),
            ): item_id
            for item_id in item_ids
        }
        for position, future in enumerate(as_completed(futures), start=1):
            original_item_id = futures[future]
            try:
                item = future.result()
                resolved.append(item)
                print(
                    f"[resolve {position}/{len(item_ids)}] {item.original_item_id} -> "
                    f"{item.earthsearch_item_id}",
                    flush=True,
                )
            except Exception as error:
                resolution_failures.append(
                    {"original_item_id": original_item_id, "error": str(error)}
                )
                print(
                    f"[resolve {position}/{len(item_ids)}] FAILED "
                    f"{original_item_id}: {error}",
                    flush=True,
                )

    if resolution_failures:
        safeio.atomic_write_text(
            resolution_failures_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at_utc": safeio.utc_now(),
                    "mirror_source": "earth-search",
                    "availability_sha256": availability_hash,
                    "n_requested_items": len(item_ids),
                    "n_resolved_items": len(resolved),
                    "n_failed_items": len(resolution_failures),
                    "failures": sorted(
                        resolution_failures,
                        key=lambda value: value["original_item_id"],
                    ),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
    else:
        resolution_failures_path.unlink(missing_ok=True)
    if not resolved:
        raise RuntimeError("no Earth Search items resolved; see resolution failure receipt")

    specs = [spec for item in resolved for spec in item_to_specs(item)]
    records = load_manifest(manifest, availability_hash)
    tasks = []
    for spec in specs:
        relative = f"assets/{spec.original_item_id}/{spec.asset_name}.tif"
        tasks.append(
            (
                spec,
                mirror_root / relative,
                relative,
                records.get((spec.original_item_id, spec.asset_name)),
            )
        )

    failures: list[str] = []
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
                availability_hash,
                args.download_retries,
                (args.connect_timeout, args.read_timeout),
                chunk_size,
            ): spec
            for spec, destination, relative, prior in tasks
        }
        for position, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            try:
                record, was_skipped = future.result()
                records[(spec.original_item_id, spec.asset_name)] = record
                skipped += int(was_skipped)
                write_manifest(manifest, records)
                action = "verified" if was_skipped else "downloaded"
                print(
                    f"[{position}/{len(tasks)}] {action} "
                    f"{spec.original_item_id}/{spec.asset_name}",
                    flush=True,
                )
            except Exception as error:
                failures.append(f"{spec.original_item_id}/{spec.asset_name}: {error}")
    if failures:
        raise RuntimeError(f"Earth Search mirror incomplete: {failures[:5]}")

    resolved_item_ids = sorted(item.original_item_id for item in resolved)
    resolved_expected = {(item_id, asset) for item_id in resolved_item_ids for asset in ASSETS}
    missing = sorted(resolved_expected - set(records))
    if missing:
        raise RuntimeError(f"Earth Search manifest lacks expected assets: {missing[:5]}")
    for key in sorted(resolved_expected):
        record = records[key]
        if record.get("acquisition_identity_match") is not True:
            raise RuntimeError(f"identity receipt missing for {key}")
        if not safeio.validate_completed_record(record, mirror_root / str(record["local_path"])):
            raise RuntimeError(f"post-download verification failed: {key}")

    write_manifest(manifest, records)
    if resolution_failures:
        raise RuntimeError(
            f"Earth Search mirror downloaded {len(resolved)} resolvable items but remains "
            f"incomplete for {len(resolution_failures)} items; no complete marker was written; "
            f"see {resolution_failures_path}"
        )
    expected = {(item_id, asset) for item_id in item_ids for asset in ASSETS}
    completion = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "complete": True,
        "completed_at_utc": safeio.utc_now(),
        "mirror_source": "earth-search",
        "availability": str(availability),
        "availability_sha256": availability_hash,
        "manifest": str(manifest),
        "manifest_sha256": safeio.sha256_file(manifest),
        "asset_names": list(ASSETS),
        "original_item_ids": item_ids,
        "original_item_identity_sha256": safeio.sha256_text(item_ids),
        "selection_uses_labels": False,
        "acquisition_identity_match": True,
        "acquisition_identity_policy": (
            "exact satellite+sensing_timestamp+relative_orbit+MGRS_tile; "
            "processing baseline and generation may differ"
        ),
        "scope": "full_availability" if item_ids == all_item_ids else "selected_items_or_shard",
        "shards": args.shards,
        "shard_index": args.shard_index,
        "n_items": len(item_ids),
        "n_assets": len(expected),
        "content_length_bytes": sum(int(records[key]["content_length"]) for key in expected),
        "verified_existing_assets": skipped,
    }
    safeio.atomic_write_text(
        marker,
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(completion, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
