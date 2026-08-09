#!/usr/bin/env python3
"""Build an auditable four-date Sentinel-2 cache for PILD Prithvi runs.

The acquisition registry freezes image selection before this script runs. This
builder reads no label raster and uses only frozen window geometry to crop the
selected Sentinel-2 L2A assets. A physical acquisition unit is processed as one
transaction so that same-date, cross-tile observations are mosaicked under a
single spatial contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
ASSETS = BANDS + ("SCL",)
CLOUD_CODES = (3, 8, 9, 10, 11)
HEIGHT = 128
WIDTH = 128
# Nearby windows are read as small mosaics. Keeping blocks narrow avoids broad
# WarpedVRT reads across remote tile boundaries while still coalescing requests.
READ_BLOCK_TOPLEFT_SPAN = 256
READ_BLOCK_MAX_WINDOWS = 2
READINESS_COLUMNS = (
    "sample_id",
    "dataset_id",
    "source_scene_id",
    "physical_event_id",
    "target_crs",
    "target_transform",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "bbox_top",
    "target_gsd_m",
    "window_selection_uses_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = PROJECT_ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
    parser.add_argument("--readiness", type=Path, default=base / "pild_window_readiness.csv")
    parser.add_argument(
        "--availability", type=Path, default=base / "acquisition_availability_v1.csv"
    )
    parser.add_argument("--out", type=Path, default=base / "pild_prithvi_4t6b_p128.h5")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--asset-mirror-manifest",
        type=Path,
        default=None,
        help="Completed local asset JSONL manifest. Local mode is strict and never falls back to STAC.",
    )
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument(
        "--unit-id",
        action="append",
        default=[],
        help="Exact acquisition_unit_id to process; repeat for multiple units.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-window-coverage", type=float, default=0.999)
    parser.add_argument("--max-window-cloud-fraction", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--read-timeout-seconds",
        type=int,
        default=90,
        help="Hard timeout for one mosaicked band/block read; timed-out blocks fall back to windows.",
    )
    parser.add_argument("--flush-every", type=int, default=8)
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


def parse_transform(value: str) -> Affine:
    numbers = [float(item.strip()) for item in str(value).split(",")]
    if len(numbers) != 6:
        raise ValueError(f"target_transform must have six values, got {value!r}")
    transform = Affine(*numbers)
    if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9:
        raise ValueError(f"rotated target grids are unsupported: {value!r}")
    if transform.a <= 0 or transform.e >= 0:
        raise ValueError(f"expected north-up target transform: {value!r}")
    return transform


def split_observations(value: str) -> list[list[str]]:
    observations = [part.split(";") for part in str(value).split("|")]
    observations = [[item.strip() for item in group if item.strip()] for group in observations]
    if len(observations) != 4 or any(not group for group in observations):
        raise ValueError(f"expected four non-empty observations, got {value!r}")
    return observations


def split_dates(value: str) -> list[str]:
    dates = [part.strip()[:10] for part in str(value).split("|")]
    if len(dates) != 4 or any(len(date) != 10 for date in dates):
        raise ValueError(f"expected four ISO datetimes, got {value!r}")
    return dates


def temporal_coordinates(dates: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [int(date[:4]), datetime.fromisoformat(date).timetuple().tm_yday]
            for date in dates
        ],
        dtype=np.int16,
    )


def read_inputs(readiness_path: Path, availability_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    readiness = pd.read_csv(readiness_path, usecols=list(READINESS_COLUMNS))
    availability = pd.read_csv(availability_path)
    required_availability = {
        "acquisition_unit_id",
        "dataset_id",
        "source_scene_id",
        "physical_event_id",
        "n_windows",
        "selected_item_ids",
        "selected_datetimes",
        "status",
        "prithvi_temporal_ready",
        "selection_uses_labels",
    }
    missing = required_availability - set(availability.columns)
    if missing:
        raise RuntimeError(f"availability registry missing fields: {sorted(missing)}")
    if readiness["sample_id"].duplicated().any():
        raise RuntimeError("readiness registry contains duplicate sample_id")
    if (readiness["window_selection_uses_label"].astype(int) != 0).any():
        raise RuntimeError("window registry reports label-dependent selection")
    if (availability["selection_uses_labels"].astype(int) != 0).any():
        raise RuntimeError("acquisition registry reports label-dependent image selection")
    readiness["acquisition_unit_id"] = (
        readiness["dataset_id"].astype(str)
        + "::"
        + readiness["source_scene_id"].astype(str)
        + "::"
        + readiness["physical_event_id"].astype(str)
    )
    availability = availability[
        (availability["status"].astype(str) == "ready")
        & (availability["prithvi_temporal_ready"].astype(int) == 1)
    ].copy()
    availability = availability.sort_values("acquisition_unit_id").reset_index(drop=True)
    if availability["acquisition_unit_id"].duplicated().any():
        raise RuntimeError("availability registry contains duplicate acquisition_unit_id")
    available_units = set(availability["acquisition_unit_id"])
    missing_units = sorted(set(readiness["acquisition_unit_id"]) - available_units)
    if missing_units:
        raise RuntimeError(f"readiness windows lack ready acquisitions: {missing_units[:5]}")
    counts = readiness.groupby("acquisition_unit_id")["sample_id"].size()
    expected = availability.set_index("acquisition_unit_id")["n_windows"].astype(int)
    mismatches = {
        unit: (int(counts.get(unit, 0)), int(value))
        for unit, value in expected.items()
        if int(counts.get(unit, 0)) != int(value)
    }
    if mismatches:
        raise RuntimeError(f"unit/window identity mismatch: {dict(list(mismatches.items())[:5])}")
    return readiness, availability


def select_units(availability: pd.DataFrame, unit_ids: list[str], max_units: int) -> pd.DataFrame:
    selected = availability
    if unit_ids:
        unknown = sorted(set(unit_ids) - set(availability["acquisition_unit_id"]))
        if unknown:
            raise RuntimeError(f"unknown or unavailable unit IDs: {unknown}")
        order = {unit: index for index, unit in enumerate(unit_ids)}
        selected = availability[availability["acquisition_unit_id"].isin(unit_ids)].copy()
        selected["_order"] = selected["acquisition_unit_id"].map(order)
        selected = selected.sort_values("_order").drop(columns="_order")
    if max_units:
        if max_units < 1:
            raise ValueError("--max-units must be positive")
        selected = selected.head(max_units)
    if selected.empty:
        raise RuntimeError("no acquisition units selected")
    return selected.reset_index(drop=True)


def open_catalog(retries: int):
    from pystac_client import Client

    error: Exception | None = None
    for attempt in range(retries):
        try:
            return Client.open(CATALOG_URL)
        except Exception as caught:  # transient network/TLS failures
            error = caught
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"unable to open STAC catalog after {retries} attempts: {error}")


def fetch_items(client, item_ids: list[str], retries: int) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            items = list(
                client.search(
                    collections=[COLLECTION], ids=sorted(set(item_ids)), max_items=len(set(item_ids))
                ).items()
            )
            result = {item.id: item for item in items}
            missing = sorted(set(item_ids) - set(result))
            if missing:
                raise RuntimeError(f"STAC did not return item IDs: {missing}")
            return result
        except Exception as caught:  # transient network/TLS failures
            error = caught
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"unable to fetch STAC items after {retries} attempts: {error}")


def unsigned_manifest_record(unit_id: str, item: Any) -> dict[str, Any]:
    return {
        "acquisition_unit_id": unit_id,
        "collection": item.collection_id,
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "bbox": list(item.bbox) if item.bbox else None,
        "assets": {
            asset_name: {
                "href": item.assets[asset_name].href.split("?")[0],
                "media_type": item.assets[asset_name].media_type,
                "roles": item.assets[asset_name].roles,
            }
            for asset_name in ASSETS
        },
    }


def signed_items(items: dict[str, Any]) -> dict[str, Any]:
    import planetary_computer

    output = {}
    for item_id, item in items.items():
        missing = sorted(set(ASSETS) - set(item.assets))
        if missing:
            raise RuntimeError(f"{item_id} lacks required assets: {missing}")
        output[item_id] = planetary_computer.sign(item)
    return output


def validate_item_dates(
    observations: list[list[str]], dates: list[str], items: dict[str, Any]
) -> None:
    for expected_date, item_ids in zip(dates, observations):
        for item_id in item_ids:
            item = items[item_id]
            if item.datetime is None:
                raise RuntimeError(f"{item_id} has no STAC datetime")
            actual_date = item.datetime.date().isoformat()
            if actual_date != expected_date:
                raise RuntimeError(
                    f"{item_id} date {actual_date} differs from frozen date {expected_date}"
                )


def canonicalize_manifest(path: Path) -> None:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["acquisition_unit_id"]), str(record["item_id"]))
        if key in records and records[key] != record:
            raise RuntimeError(f"conflicting source manifest records for {key}")
        records[key] = record
    with path.open("w", encoding="utf-8") as stream:
        for key in sorted(records):
            stream.write(json.dumps(records[key], sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True)
class LocalAsset:
    href: str
    media_type: str | None
    roles: list[str]


@dataclass(frozen=True)
class LocalItem:
    id: str
    collection_id: str
    datetime: datetime
    bbox: list[float] | None
    assets: dict[str, LocalAsset]


def asset_mirror_marker_path(manifest: Path) -> Path:
    return manifest.with_name(f"{manifest.stem}.complete.json")


def load_local_asset_items(
    manifest: Path,
    required_item_ids: set[str],
    availability_sha256: str,
) -> tuple[dict[str, LocalItem], Path, str]:
    """Load and byte-verify a complete local mirror without remote fallback."""
    marker_path = asset_mirror_marker_path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"asset mirror manifest is missing: {manifest}")
    if not marker_path.is_file():
        raise FileNotFoundError(f"asset mirror complete marker is missing: {marker_path}")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest)
    if marker.get("complete") is not True:
        raise RuntimeError("asset mirror marker does not report complete=true")
    if int(marker.get("schema_version", -1)) != 1:
        raise RuntimeError("unsupported asset mirror marker schema")
    if marker.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("asset mirror manifest hash differs from complete marker")
    if marker.get("availability_sha256") != availability_sha256:
        raise RuntimeError("asset mirror was built from a different availability registry")
    if tuple(marker.get("asset_names", ())) != ASSETS:
        raise RuntimeError("asset mirror marker does not freeze the required seven assets")
    marker_items = {str(value) for value in marker.get("item_ids", [])}
    missing_marker_items = sorted(required_item_ids - marker_items)
    if missing_marker_items:
        raise RuntimeError(
            f"asset mirror complete scope lacks required items: {missing_marker_items[:5]}"
        )

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("schema_version", -1)) != 1:
            raise RuntimeError(f"unsupported asset mirror schema at line {line_number}")
        if record.get("availability_sha256") != availability_sha256:
            raise RuntimeError(f"asset mirror availability hash mismatch at line {line_number}")
        item_id = str(record.get("item_id", ""))
        asset_name = str(record.get("asset_name", ""))
        if asset_name not in ASSETS:
            raise RuntimeError(f"unexpected mirrored asset at line {line_number}: {asset_name!r}")
        key = (item_id, asset_name)
        if key in records:
            raise RuntimeError(f"duplicate asset mirror record for {key}")
        records[key] = record

    missing_records = sorted(
        (item_id, asset_name)
        for item_id in required_item_ids
        for asset_name in ASSETS
        if (item_id, asset_name) not in records
    )
    if missing_records:
        raise RuntimeError(f"asset mirror lacks required item/assets: {missing_records[:5]}")

    items: dict[str, LocalItem] = {}
    mirror_root = manifest.parent.resolve()
    for item_id in sorted(required_item_ids):
        assets: dict[str, LocalAsset] = {}
        item_datetimes: list[datetime] = []
        collections: set[str] = set()
        for asset_name in ASSETS:
            record = records[(item_id, asset_name)]
            expected_relative = Path("assets") / item_id / f"{asset_name}.tif"
            relative = Path(str(record.get("local_path", "")))
            if relative != expected_relative or relative.is_absolute():
                raise RuntimeError(
                    f"{item_id}/{asset_name} has non-canonical local path: {relative}"
                )
            path = (mirror_root / relative).resolve()
            if mirror_root not in path.parents:
                raise RuntimeError(f"asset path escapes mirror root: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"mirrored asset is missing: {path}")
            try:
                expected_size = int(record["content_length"])
                expected_hash = str(record["sha256"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"invalid size/hash record for {item_id}/{asset_name}") from error
            if expected_size <= 0 or path.stat().st_size != expected_size:
                raise RuntimeError(f"mirrored asset length mismatch: {item_id}/{asset_name}")
            if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
                raise RuntimeError(f"invalid mirrored asset SHA256: {item_id}/{asset_name}")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise RuntimeError(f"mirrored asset SHA256 mismatch: {item_id}/{asset_name}")
            raw_datetime = str(record.get("item_datetime", ""))
            try:
                item_datetimes.append(
                    datetime.fromisoformat(raw_datetime.replace("Z", "+00:00"))
                )
            except ValueError as error:
                raise RuntimeError(
                    f"invalid item datetime for {item_id}: {raw_datetime!r}"
                ) from error
            collections.add(str(record.get("collection", "")))
            assets[asset_name] = LocalAsset(
                href=str(path),
                media_type=record.get("media_type"),
                roles=[str(value) for value in record.get("roles", [])],
            )
        # Planetary Computer metadata and native granule metadata may serialize
        # the same acquisition time with or without fractional seconds. A
        # sub-second difference is not an item-identity conflict; larger drift
        # still fails closed.
        datetime_span = max(item_datetimes) - min(item_datetimes)
        if datetime_span.total_seconds() >= 1.0 or len(collections) != 1:
            raise RuntimeError(f"inconsistent item metadata in asset mirror: {item_id}")
        collection = next(iter(collections))
        if collection != COLLECTION:
            raise RuntimeError(f"unexpected collection for {item_id}: {collection!r}")
        item_datetime = min(item_datetimes)
        items[item_id] = LocalItem(
            id=item_id,
            collection_id=collection,
            datetime=item_datetime,
            bbox=None,
            assets=assets,
        )
    return items, marker_path, manifest_hash


def transforms_match(left: Affine, right: Affine, tolerance: float = 1e-5) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(tuple(left), tuple(right)))


@dataclass(frozen=True)
class WindowSlice:
    row: dict[str, Any]
    row_offset: int
    col_offset: int


@dataclass(frozen=True)
class MosaicBlock:
    transform: Affine
    height: int
    width: int
    windows: tuple[WindowSlice, ...]


@contextmanager
def hard_timeout(seconds: int):
    """Interrupt a stalled GDAL read without weakening the unit transaction."""
    if seconds <= 0:
        yield
        return

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"remote raster read exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def read_mosaic(
    sources: list[WarpedVRT],
    target_transform: Affine,
    height: int,
    width: int,
    resampling: Resampling,
    timeout_seconds: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    bounds = array_bounds(height, width, target_transform)
    with hard_timeout(timeout_seconds):
        data, output_transform = merge(
            sources,
            bounds=bounds,
            res=(abs(target_transform.a), abs(target_transform.e)),
            nodata=0,
            dtype="float32",
            resampling=resampling,
            masked=True,
            target_aligned_pixels=False,
        )
    if data.shape != (1, height, width):
        raise RuntimeError(f"mosaic returned shape {data.shape}, expected (1,{height},{width})")
    if not transforms_match(output_transform, target_transform):
        raise RuntimeError(
            f"mosaic transform {tuple(output_transform)[:6]} != target {tuple(target_transform)[:6]}"
        )
    mask = np.ma.getmaskarray(data)[0]
    values = np.asarray(np.ma.filled(data[0], 0.0), dtype=np.float32)
    valid = (~mask) & np.isfinite(values)
    return values, valid, float(valid.mean())


def read_mosaic_window(
    sources: list[WarpedVRT],
    target_transform: Affine,
    resampling: Resampling,
    timeout_seconds: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    return read_mosaic(
        sources,
        target_transform,
        HEIGHT,
        WIDTH,
        resampling,
        timeout_seconds,
    )


def build_mosaic_blocks(unit_windows: pd.DataFrame) -> list[MosaicBlock]:
    """Plan bounded reads on the exact frozen target grid for one unit."""
    rows = unit_windows.to_dict("records")
    transforms = [parse_transform(row["target_transform"]) for row in rows]
    reference = transforms[0]
    for transform in transforms[1:]:
        if any(
            abs(left - right) > 1e-9
            for left, right in zip(
                (transform.a, transform.b, transform.d, transform.e),
                (reference.a, reference.b, reference.d, reference.e),
            )
        ):
            raise RuntimeError("unit windows do not share one target pixel grid")

    origin_left = min(transform.c for transform in transforms)
    origin_top = max(transform.f for transform in transforms)
    unit_transform = Affine(
        reference.a,
        reference.b,
        origin_left,
        reference.d,
        reference.e,
        origin_top,
    )
    grouped: dict[tuple[int, int], list[tuple[dict[str, Any], int, int, Affine]]] = {}
    for row, transform in zip(rows, transforms):
        col_float = (transform.c - origin_left) / reference.a
        row_float = (origin_top - transform.f) / abs(reference.e)
        col_offset = int(round(col_float))
        row_offset = int(round(row_float))
        if abs(col_float - col_offset) > 1e-5 or abs(row_float - row_offset) > 1e-5:
            raise RuntimeError("unit window target transforms are not pixel-aligned")
        expected = unit_transform * Affine.translation(col_offset, row_offset)
        if not transforms_match(expected, transform):
            raise RuntimeError("unit window target transform differs from planned frozen grid")
        key = (
            row_offset // READ_BLOCK_TOPLEFT_SPAN,
            col_offset // READ_BLOCK_TOPLEFT_SPAN,
        )
        grouped.setdefault(key, []).append((row, row_offset, col_offset, transform))

    blocks = []
    for cell_members in grouped.values():
        cell_members.sort(key=lambda member: (member[1], member[2]))
        for start in range(0, len(cell_members), READ_BLOCK_MAX_WINDOWS):
            members = cell_members[start : start + READ_BLOCK_MAX_WINDOWS]
            min_row = min(member[1] for member in members)
            min_col = min(member[2] for member in members)
            max_row = max(member[1] + HEIGHT for member in members)
            max_col = max(member[2] + WIDTH for member in members)
            block_transform = unit_transform * Affine.translation(min_col, min_row)
            windows = tuple(
                WindowSlice(
                    row=member[0],
                    row_offset=member[1] - min_row,
                    col_offset=member[2] - min_col,
                )
                for member in members
            )
            blocks.append(
                MosaicBlock(
                    transform=block_transform,
                    height=max_row - min_row,
                    width=max_col - min_col,
                    windows=windows,
                )
            )
    return blocks


def quality_failure(
    coverage: np.ndarray,
    cloud_fraction: np.ndarray,
    all_valid: np.ndarray,
    args: argparse.Namespace,
) -> str:
    min_coverage = float(coverage.min())
    max_cloud = float(cloud_fraction.max())
    reasons = []
    if min_coverage < args.min_window_coverage:
        reasons.append(f"coverage {min_coverage:.6f} < {args.min_window_coverage:.6f}")
    if max_cloud > args.max_window_cloud_fraction:
        reasons.append(f"cloud {max_cloud:.6f} > {args.max_window_cloud_fraction:.6f}")
    if not all_valid.any():
        reasons.append("no jointly valid optical/SCL pixels")
    return "; ".join(reasons)


def read_window_payload(
    row: dict[str, Any],
    opened: dict[tuple[int, str], list[WarpedVRT]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    optical = np.zeros((6, 4, HEIGHT, WIDTH), dtype=np.uint16)
    scl = np.zeros((4, HEIGHT, WIDTH), dtype=np.uint8)
    all_valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    coverage = np.zeros((4, 7), dtype=np.float32)
    cloud_fraction = np.ones(4, dtype=np.float32)
    failure = ""
    read_failed = False
    try:
        transform = parse_transform(row["target_transform"])
        for time_index in range(4):
            for band_index, band in enumerate(BANDS):
                values, valid, fraction = read_mosaic_window(
                    opened[(time_index, band)],
                    transform,
                    Resampling.bilinear,
                    args.read_timeout_seconds,
                )
                coverage[time_index, band_index] = fraction
                all_valid &= valid
                optical[band_index, time_index] = np.clip(
                    np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0, 10_000
                ).astype(np.uint16)
            scl_values, scl_valid, scl_coverage = read_mosaic_window(
                opened[(time_index, "SCL")],
                transform,
                Resampling.nearest,
                args.read_timeout_seconds,
            )
            coverage[time_index, 6] = scl_coverage
            all_valid &= scl_valid
            scl[time_index] = np.clip(
                np.nan_to_num(scl_values, nan=0.0), 0, 255
            ).astype(np.uint8)
            valid_scl = scl_valid & (scl[time_index] > 0)
            cloud_fraction[time_index] = (
                float(np.isin(scl[time_index][valid_scl], CLOUD_CODES).mean())
                if valid_scl.any()
                else 1.0
            )
        failure = quality_failure(coverage, cloud_fraction, all_valid, args)
    except Exception as error:
        read_failed = True
        failure = f"{type(error).__name__}: {error}"

    if read_failed:
        # An incomplete read must never masquerade as a real observation.
        optical.fill(0)
        scl.fill(0)
        all_valid.fill(False)
    return {
        "optical": optical,
        "scl": scl,
        "all_valid": all_valid,
        "coverage": coverage,
        "cloud_fraction": cloud_fraction,
        "failure": failure,
    }


def read_block_payloads(
    block: MosaicBlock,
    opened: dict[tuple[int, str], list[WarpedVRT]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Read each temporal asset once for a block, then cut exact frozen windows."""
    count = len(block.windows)
    optical = np.zeros((count, 6, 4, HEIGHT, WIDTH), dtype=np.uint16)
    scl = np.zeros((count, 4, HEIGHT, WIDTH), dtype=np.uint8)
    all_valid = np.ones((count, HEIGHT, WIDTH), dtype=bool)
    coverage = np.zeros((count, 4, 7), dtype=np.float32)
    cloud_fraction = np.ones((count, 4), dtype=np.float32)

    for time_index in range(4):
        for band_index, band in enumerate(BANDS):
            values, valid, _ = read_mosaic(
                opened[(time_index, band)],
                block.transform,
                block.height,
                block.width,
                Resampling.bilinear,
                args.read_timeout_seconds,
            )
            for window_index, window in enumerate(block.windows):
                row_slice = slice(window.row_offset, window.row_offset + HEIGHT)
                col_slice = slice(window.col_offset, window.col_offset + WIDTH)
                window_values = values[row_slice, col_slice]
                window_valid = valid[row_slice, col_slice]
                coverage[window_index, time_index, band_index] = float(window_valid.mean())
                all_valid[window_index] &= window_valid
                optical[window_index, band_index, time_index] = np.clip(
                    np.nan_to_num(window_values, nan=0.0, posinf=0.0, neginf=0.0),
                    0,
                    10_000,
                ).astype(np.uint16)

        scl_values, scl_valid, _ = read_mosaic(
            opened[(time_index, "SCL")],
            block.transform,
            block.height,
            block.width,
            Resampling.nearest,
            args.read_timeout_seconds,
        )
        for window_index, window in enumerate(block.windows):
            row_slice = slice(window.row_offset, window.row_offset + HEIGHT)
            col_slice = slice(window.col_offset, window.col_offset + WIDTH)
            window_values = scl_values[row_slice, col_slice]
            window_valid = scl_valid[row_slice, col_slice]
            coverage[window_index, time_index, 6] = float(window_valid.mean())
            all_valid[window_index] &= window_valid
            scl[window_index, time_index] = np.clip(
                np.nan_to_num(window_values, nan=0.0), 0, 255
            ).astype(np.uint8)
            valid_scl = window_valid & (scl[window_index, time_index] > 0)
            cloud_fraction[window_index, time_index] = (
                float(
                    np.isin(
                        scl[window_index, time_index][valid_scl],
                        CLOUD_CODES,
                    ).mean()
                )
                if valid_scl.any()
                else 1.0
            )

    return [
        {
            "optical": optical[index],
            "scl": scl[index],
            "all_valid": all_valid[index],
            "coverage": coverage[index],
            "cloud_fraction": cloud_fraction[index],
            "failure": quality_failure(
                coverage[index], cloud_fraction[index], all_valid[index], args
            ),
        }
        for index in range(count)
    ]


def create_cache(
    path: Path,
    windows: pd.DataFrame,
    args: argparse.Namespace,
    readiness_hash: str,
    availability_hash: str,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    text_type = h5py.string_dtype("utf-8")
    sample_ids = windows["sample_id"].astype(str).tolist()
    handle.attrs.update(
        {
            "schema_version": 1,
            "complete": 0,
            "created_at_utc": utc_now(),
            "source_readiness": str(args.readiness.resolve()),
            "source_availability": str(args.availability.resolve()),
            "source_readiness_sha256": readiness_hash,
            "source_availability_sha256": availability_hash,
            "sample_identity_sha256": sha256_text(sample_ids),
            "selection_contract": "four observations frozen by acquisition_availability_v1; labels forbidden",
            "bands": ";".join(BANDS),
            "scaling": "Sentinel-2 L2A surface reflectance clipped to [0,10000] and stored uint16",
            "coordinate_contract": "temporal=[year,day_of_year], location=[latitude,longitude]",
            "spatial_contract": "per-window target_crs and target_transform; WarpedVRT plus same-date mosaic",
            "min_window_coverage": float(args.min_window_coverage),
            "max_window_cloud_fraction": float(args.max_window_cloud_fraction),
            "label_content_accessed": 0,
            "selected_units": int(windows["acquisition_unit_id"].nunique()),
            "selected_samples": len(windows),
        }
    )
    if args.asset_mirror_manifest is not None:
        handle.attrs.update(
            {
                "source_asset_mirror_manifest": str(args.asset_mirror_manifest),
                "source_asset_mirror_manifest_sha256": args.asset_mirror_manifest_sha256,
                "source_asset_mirror_complete_marker": str(
                    args.asset_mirror_complete_marker
                ),
                "source_asset_mode": "strict_local_no_remote_fallback",
            }
        )
    handle.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=object), dtype=text_type)
    handle.create_dataset(
        "acquisition_unit_id",
        data=np.asarray(windows["acquisition_unit_id"].astype(str), dtype=object),
        dtype=text_type,
    )
    handle.create_dataset(
        "optical",
        shape=(len(windows), 6, 4, HEIGHT, WIDTH),
        dtype="uint16",
        chunks=(1, 6, 1, HEIGHT, WIDTH),
        compression="lzf",
        fillvalue=0,
    )
    handle.create_dataset(
        "scl",
        shape=(len(windows), 4, HEIGHT, WIDTH),
        dtype="uint8",
        chunks=(1, 4, HEIGHT, WIDTH),
        compression="lzf",
        fillvalue=0,
    )
    handle.create_dataset(
        "optical_valid",
        shape=(len(windows), 1, HEIGHT, WIDTH),
        dtype="uint8",
        chunks=(1, 1, HEIGHT, WIDTH),
        compression="lzf",
        fillvalue=0,
    )
    handle.create_dataset("selected_dates", shape=(len(windows),), dtype=text_type)
    handle.create_dataset("selected_item_ids", shape=(len(windows),), dtype=text_type)
    handle.create_dataset("temporal_coords", shape=(len(windows), 4, 2), dtype="int16")
    handle.create_dataset("location_coords", shape=(len(windows), 2), dtype="float32")
    handle.create_dataset("cloud_fraction", shape=(len(windows), 4), dtype="float32")
    handle.create_dataset("coverage_fraction", shape=(len(windows), 4, 7), dtype="float32")
    handle.create_dataset("q_visual_temporal", shape=(len(windows),), dtype="uint8")
    handle.create_dataset("completed", shape=(len(windows),), dtype="uint8")
    handle.create_dataset("failure_reason", shape=(len(windows),), dtype=text_type)
    handle.flush()
    return handle


def validate_resume(handle: h5py.File, windows: pd.DataFrame, args: argparse.Namespace) -> None:
    expected_ids = windows["sample_id"].astype(str).tolist()
    current_ids = [item.decode() if isinstance(item, bytes) else str(item) for item in handle["sample_id"][:]]
    if current_ids != expected_ids:
        raise RuntimeError("resume cache sample identity/order differs from selected registry")
    if handle.attrs.get("source_readiness_sha256") != sha256_file(args.readiness):
        raise RuntimeError("readiness registry changed since partial cache creation")
    if handle.attrs.get("source_availability_sha256") != sha256_file(args.availability):
        raise RuntimeError("availability registry changed since partial cache creation")
    if int(handle.attrs.get("complete", 0)) != 0:
        raise RuntimeError("partial cache unexpectedly carries complete=1")
    if args.asset_mirror_manifest is not None:
        if (
            handle.attrs.get("source_asset_mirror_manifest_sha256")
            != args.asset_mirror_manifest_sha256
        ):
            raise RuntimeError("asset mirror manifest changed since partial cache creation")
        if handle.attrs.get("source_asset_mode") != "strict_local_no_remote_fallback":
            raise RuntimeError("partial cache was not created in strict local asset mode")
    elif "source_asset_mirror_manifest_sha256" in handle.attrs:
        raise RuntimeError("partial cache requires --asset-mirror-manifest to resume")


def write_window_payload(
    handle: h5py.File,
    row: dict[str, Any],
    payload: dict[str, Any],
    dates: list[str],
    selected_item_ids: str,
) -> int:
    cache_index = int(row["_cache_index"])
    q_visual = int(not payload["failure"])
    handle["optical"][cache_index] = payload["optical"]
    handle["scl"][cache_index] = payload["scl"]
    handle["optical_valid"][cache_index, 0] = payload["all_valid"].astype(np.uint8)
    handle["selected_dates"][cache_index] = ";".join(dates)
    handle["selected_item_ids"][cache_index] = selected_item_ids
    handle["temporal_coords"][cache_index] = temporal_coordinates(dates)
    handle["location_coords"][cache_index] = np.asarray(
        [
            (float(row["bbox_bottom"]) + float(row["bbox_top"])) / 2,
            (float(row["bbox_left"]) + float(row["bbox_right"])) / 2,
        ],
        dtype=np.float32,
    )
    handle["cloud_fraction"][cache_index] = payload["cloud_fraction"]
    handle["coverage_fraction"][cache_index] = payload["coverage"]
    handle["q_visual_temporal"][cache_index] = q_visual
    handle["failure_reason"][cache_index] = payload["failure"]
    return q_visual


def process_unit(
    handle: h5py.File,
    unit_row: dict[str, Any],
    unit_windows: pd.DataFrame,
    items: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    unit_id = str(unit_row["acquisition_unit_id"])
    observations = split_observations(unit_row["selected_item_ids"])
    dates = split_dates(unit_row["selected_datetimes"])
    target_crs_values = unit_windows["target_crs"].astype(str).unique().tolist()
    if len(target_crs_values) != 1:
        raise RuntimeError(f"{unit_id}: unit has multiple target CRS values: {target_crs_values}")
    target_crs = target_crs_values[0]
    indices = unit_windows["_cache_index"].astype(int).tolist()
    already = handle["completed"][indices].astype(bool)
    if already.all():
        return {"acquisition_unit_id": unit_id, "status": "already_complete", "n_windows": len(indices)}
    if already.any():
        raise RuntimeError(f"{unit_id}: partially committed unit violates unit transaction contract")

    vrt_options = {
        "crs": target_crs,
        "nodata": 0,
        "warp_mem_limit": 256,
    }
    opened: dict[tuple[int, str], list[WarpedVRT]] = {}
    source_open_failures: list[dict[str, str]] = []
    with ExitStack() as stack:
        for time_index, item_group in enumerate(observations):
            for asset_name in ASSETS:
                resampling = Resampling.nearest if asset_name == "SCL" else Resampling.bilinear
                sources = []
                for item_id in item_group:
                    asset = items[item_id].assets[asset_name]
                    try:
                        with hard_timeout(args.read_timeout_seconds):
                            source = stack.enter_context(rasterio.open(asset.href))
                            vrt = stack.enter_context(
                                WarpedVRT(source, resampling=resampling, **vrt_options)
                            )
                        sources.append(vrt)
                    except Exception as error:
                        source_open_failures.append(
                            {
                                "time_index": str(time_index),
                                "item_id": item_id,
                                "asset": asset_name,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                opened[(time_index, asset_name)] = sources

        if args.asset_mirror_manifest is not None and source_open_failures:
            first = source_open_failures[0]
            raise RuntimeError(
                "strict local asset open failed for "
                f"{first['item_id']}/{first['asset']}: {first['error']}"
            )

        unit_failures = 0
        unit_valid = 0
        position = 0
        try:
            blocks = build_mosaic_blocks(unit_windows)
        except Exception:
            # Preserve per-window error and abstention semantics for malformed grids.
            blocks = []

        if blocks:
            batches = ((block.windows, block) for block in blocks)
        else:
            batches = (
                ((WindowSlice(row=row, row_offset=0, col_offset=0),), None)
                for row in unit_windows.to_dict("records")
            )

        for windows, block in batches:
            if block is None:
                payloads = [read_window_payload(windows[0].row, opened, args)]
            else:
                try:
                    payloads = read_block_payloads(block, opened, args)
                except Exception:
                    # A block timeout or remote failure must not widen the affected
                    # sample set. Retry through the original window-level path.
                    payloads = [
                        read_window_payload(window.row, opened, args) for window in windows
                    ]
            for window, payload in zip(windows, payloads):
                position += 1
                q_visual = write_window_payload(
                    handle,
                    window.row,
                    payload,
                    dates,
                    str(unit_row["selected_item_ids"]),
                )
                if q_visual:
                    unit_valid += 1
                else:
                    unit_failures += 1
                if position % max(1, args.flush_every) == 0:
                    handle.flush()

        # Commit markers are written only after every window payload is present.
        handle["completed"][indices] = 1
        handle.attrs["completed_samples"] = int(handle["completed"][:].sum())
        handle.attrs["last_completed_unit"] = unit_id
        handle.attrs["last_updated_at_utc"] = utc_now()
        handle.flush()
    return {
        "acquisition_unit_id": unit_id,
        "status": "complete",
        "n_windows": len(indices),
        "n_q_visual_temporal_1": unit_valid,
        "n_q_visual_temporal_0": unit_failures,
        "n_source_open_failures": len(source_open_failures),
        "source_open_failures": source_open_failures,
    }


def main() -> int:
    args = parse_args()
    args.readiness = args.readiness.resolve()
    args.availability = args.availability.resolve()
    args.out = args.out.resolve()
    if args.asset_mirror_manifest is not None:
        args.asset_mirror_manifest = args.asset_mirror_manifest.resolve()
    if not 0.0 <= args.min_window_coverage <= 1.0:
        raise ValueError("--min-window-coverage must be in [0,1]")
    if not 0.0 <= args.max_window_cloud_fraction <= 1.0:
        raise ValueError("--max-window-cloud-fraction must be in [0,1]")
    if args.read_timeout_seconds < 0:
        raise ValueError("--read-timeout-seconds must be non-negative")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    readiness, availability = read_inputs(args.readiness, args.availability)
    selected_units = select_units(availability, args.unit_id, args.max_units)
    selected_ids = selected_units["acquisition_unit_id"].astype(str).tolist()
    # Preserve the frozen readiness-registry order for sidecar identity alignment.
    windows = readiness[readiness["acquisition_unit_id"].isin(selected_ids)].copy().reset_index(drop=True)
    windows["_cache_index"] = np.arange(len(windows), dtype=np.int64)
    if not args.max_units and not args.unit_id and len(windows) != 2937:
        raise RuntimeError(f"full build must contain 2937 windows, got {len(windows)}")

    readiness_hash = sha256_file(args.readiness)
    availability_hash = sha256_file(args.availability)
    local_items: dict[str, LocalItem] | None = None
    if args.asset_mirror_manifest is not None:
        required_item_ids = {
            item_id
            for value in selected_units["selected_item_ids"].astype(str)
            for group in split_observations(value)
            for item_id in group
        }
        local_items, mirror_marker, mirror_manifest_hash = load_local_asset_items(
            args.asset_mirror_manifest,
            required_item_ids,
            availability_hash,
        )
        args.asset_mirror_complete_marker = mirror_marker
        args.asset_mirror_manifest_sha256 = mirror_manifest_hash

    partial = args.out.with_name(f".{args.out.name}.inprogress")
    marker = args.out.with_name(f"{args.out.name}.complete.json")
    manifest = (args.manifest or args.out.with_name(f"{args.out.stem}.source_manifest.jsonl")).resolve()
    if args.asset_mirror_manifest is not None and manifest == args.asset_mirror_manifest:
        raise ValueError("--manifest output must differ from --asset-mirror-manifest input")
    if args.overwrite:
        args.out.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
    if args.out.exists():
        raise FileExistsError(f"completed output exists: {args.out}")
    if partial.exists() and not args.resume:
        raise FileExistsError(f"partial output exists; pass --resume: {partial}")

    if partial.exists():
        if not manifest.exists():
            raise FileNotFoundError(f"resume source manifest is missing: {manifest}")
        handle = h5py.File(partial, "r+")
        validate_resume(handle, windows, args)
    else:
        handle = create_cache(partial, windows, args, readiness_hash, availability_hash)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.unlink(missing_ok=True)

    client = None if local_items is not None else open_catalog(args.retries)
    summaries = []
    try:
        for unit_position, unit_row in enumerate(selected_units.to_dict("records"), start=1):
            unit_id = str(unit_row["acquisition_unit_id"])
            unit_windows = windows[windows["acquisition_unit_id"] == unit_id]
            indices = unit_windows["_cache_index"].astype(int).tolist()
            if handle["completed"][indices].astype(bool).all():
                summaries.append(
                    {"acquisition_unit_id": unit_id, "status": "already_complete", "n_windows": len(indices)}
                )
                continue
            observations = split_observations(unit_row["selected_item_ids"])
            item_ids = sorted({item for group in observations for item in group})
            raw_items = (
                {item_id: local_items[item_id] for item_id in item_ids}
                if local_items is not None
                else fetch_items(client, item_ids, args.retries)
            )
            validate_item_dates(
                observations, split_dates(unit_row["selected_datetimes"]), raw_items
            )
            records = [unsigned_manifest_record(unit_id, raw_items[item_id]) for item_id in item_ids]
            with manifest.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            items = raw_items if local_items is not None else signed_items(raw_items)
            with rasterio.Env(
                GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_HTTP_CONNECTTIMEOUT="30",
                GDAL_HTTP_TIMEOUT="60",
                GDAL_HTTP_IPRESOLVE="V4",
                GDAL_HTTP_MAX_RETRY="4",
                GDAL_HTTP_RETRY_DELAY="2",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                GDAL_CACHEMAX=512,
            ):
                summary = process_unit(handle, unit_row, unit_windows, items, args)
            summaries.append(summary)
            print(
                f"[{unit_position}/{len(selected_units)}] {unit_id}: "
                f"q=1 {summary.get('n_q_visual_temporal_1', len(indices))}/"
                f"{summary.get('n_windows', len(indices))}",
                flush=True,
            )
    finally:
        if handle.id.valid:
            handle.flush()

    completed = handle["completed"][:].astype(bool)
    if not completed.all():
        handle.close()
        raise RuntimeError(
            f"build stopped with {int(completed.sum())}/{len(completed)} samples complete; resume {partial}"
        )
    handle.attrs["complete"] = 1
    handle.attrs["completed_at_utc"] = utc_now()
    canonicalize_manifest(manifest)
    handle.attrs["source_manifest"] = str(manifest)
    handle.attrs["source_manifest_sha256"] = sha256_file(manifest)
    handle.attrs["q_visual_temporal_1"] = int(handle["q_visual_temporal"][:].sum())
    handle.attrs["q_visual_temporal_0"] = int(len(windows) - handle["q_visual_temporal"][:].sum())
    handle.flush()
    handle.close()
    os.replace(partial, args.out)
    output_hash = sha256_file(args.out)
    completion = {
        "schema_version": 1,
        "complete": True,
        "completed_at_utc": utc_now(),
        "output": str(args.out),
        "output_sha256": output_hash,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "source_readiness_sha256": readiness_hash,
        "source_availability_sha256": availability_hash,
        "n_units": len(selected_units),
        "n_samples": len(windows),
        "shape": [len(windows), 6, 4, HEIGHT, WIDTH],
        "unit_summaries": summaries,
    }
    if args.asset_mirror_manifest is not None:
        completion.update(
            {
                "source_asset_mirror_manifest": str(args.asset_mirror_manifest),
                "source_asset_mirror_manifest_sha256": args.asset_mirror_manifest_sha256,
                "source_asset_mirror_complete_marker": str(
                    args.asset_mirror_complete_marker
                ),
                "source_asset_mode": "strict_local_no_remote_fallback",
            }
        )
    marker.write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(completion, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
