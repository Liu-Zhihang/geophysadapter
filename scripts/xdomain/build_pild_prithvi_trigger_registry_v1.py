#!/usr/bin/env python3
"""Build the label-free PILD-2937 event-first CHIRPS Trigger registry.

The event is the statistical unit. Each physical event receives one median
sample-footprint center, one strict D-7..D-1 case window, and four shifted
windows (-56, -28, +28, +56 days). Event values are then broadcast unchanged
to all member samples. Any non-unique date, non-rainfall mechanism, or
incomplete case/control coverage fails closed to q_R=0.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SAMPLE_COUNT = 2937
EXPECTED_EVENT_COUNT = 42
SCHEMA_VERSION = 1
STRICT_LAGS = tuple(range(1, 8))
ANCHOR_SHIFTS = {
    "case": 0,
    "wrong_m56": -56,
    "wrong_m28": -28,
    "wrong_p28": 28,
    "wrong_p56": 56,
}
CONTROL_ROLES = tuple(role for role in ANCHOR_SHIFTS if role != "case")
RAINFALL_CANONICAL_FAMILY = "hydrometeorological"

READINESS_COLUMNS = [
    "sample_id",
    "event_uid",
    "physical_event_id",
    "dataset_id",
    "event_date",
    "event_date_valid",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "bbox_top",
    "window_selection_uses_label",
    "locked_retrospective",
]
EVENT_COLUMNS = [
    "physical_event_id",
    "canonical_date",
    "physical_trigger_family",
    "event_uids",
    "registry_dataset_ids",
]
EVENT_BROADCAST_COLUMNS = [
    "registry_build_id",
    "event_record_sha256",
    "canonical_event_date",
    "readiness_event_dates",
    "n_unique_readiness_dates",
    "date_unique_and_canonical",
    "physical_trigger_family",
    "rainfall_mechanism",
    "event_center_lon",
    "event_center_lat",
    "rain_d7_case_mm",
    "rain_d7_wrong_m56_mm",
    "rain_d7_wrong_m28_mm",
    "rain_d7_wrong_p28_mm",
    "rain_d7_wrong_p56_mm",
    "rain_d7_wrongtime_median_mm",
    "rain_d7_case_minus_wrongtime_mm",
    "case_days_available",
    "wrong_m56_days_available",
    "wrong_m28_days_available",
    "wrong_p28_days_available",
    "wrong_p56_days_available",
    "available_days_total",
    "required_days_total",
    "chirps_coverage_complete",
    "q_R",
    "q_R_reason",
]
PROHIBITED_OUTPUT_TOKENS = (
    "label",
    "mask",
    "prediction",
    "logit",
    "checkpoint",
    "model_result",
    "metric",
    "iou",
)

CONTRACT = {
    "registry_unit": "physical_event_id, then unchanged broadcast to member sample_id rows",
    "event_coordinate": "median of readiness sample bbox-center longitude and latitude",
    "rainfall_source": "CHIRPS v2.0 daily global 0.05-degree GeoTIFF gzip",
    "spatial_statistic": "daily median of valid native 3x3 CHIRPS cells at the event coordinate",
    "case_window": "strict D-7..D-1 relative to the unique canonical event date; D0 excluded",
    "wrong_time_shifts_days": [-56, -28, 28, 56],
    "wrong_time_window": "strict shifted-anchor D-7..D-1; D0 of each shifted anchor excluded",
    "control_statistic": "median of the four complete wrong-time D7 totals",
    "rainfall_mechanism": (
        "physical_trigger_family == hydrometeorological; the canonical PILD registry "
        "normalizes source rainfall/storm mechanisms to this family"
    ),
    "q_R": (
        "1 only when the readiness event date is valid, unique, equal to canonical_date, "
        "the canonical mechanism is rainfall, and all 35 case/control daily values are valid"
    ),
    "fail_closed": True,
    "imputation": "none",
    "label_and_model_result_use": "forbidden; only explicit whitelisted readiness/event fields are loaded",
    "expected_samples": EXPECTED_SAMPLE_COUNT,
}


class RegistryValidationError(RuntimeError):
    """Raised when an input or output violates the frozen registry contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--readiness",
        type=Path,
        default=Path("processed/hybrid_pinn/pild_prithvi_integration_v1/pild_window_readiness.csv"),
    )
    parser.add_argument(
        "--event-registry",
        type=Path,
        default=Path("metadata/pild_core_v2/event_registry_v2.csv"),
    )
    parser.add_argument(
        "--chirps-root",
        type=Path,
        default=Path("raw_fullcopy/weather/chirps_daily_global"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("processed/hybrid_pinn/pild_prithvi_integration_v1"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate committed outputs and hashes without writing files.",
    )
    parser.add_argument(
        "--skip-chirps-rehash",
        action="store_true",
        help="In validate-only mode, trust recorded per-raster hashes after output validation.",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(
        json_safe(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    atomic_write_bytes(path, encoded)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def strict_iso_date(value: Any, field: str) -> date:
    text = str(value).strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise RegistryValidationError(f"Invalid {field} ISO date: {value!r}") from exc
    if parsed.isoformat() != text:
        raise RegistryValidationError(f"Non-canonical {field} ISO date: {value!r}")
    return parsed


def strict_window_dates(anchor: date, shift_days: int) -> tuple[date, ...]:
    shifted = anchor + timedelta(days=shift_days)
    return tuple(shifted - timedelta(days=lag) for lag in STRICT_LAGS)


def chirps_path(root: Path, day: date) -> Path:
    return root / f"{day.year:04d}" / f"chirps-v2.0.{day:%Y.%m.%d}.tif.gz"


def output_paths(outdir: Path) -> dict[str, Path]:
    return {
        "event_registry": outdir / "pild_trigger_event_registry_v1.csv",
        "sample_registry": outdir / "pild_trigger_sample_registry_v1.csv",
        "chirps_manifest": outdir / "pild_trigger_chirps_manifest_v1.csv",
        "audit": outdir / "pild_trigger_audit_v1.json",
        "hash_manifest": outdir / "pild_trigger_hash_manifest_v1.json",
    }


def read_inputs(readiness_path: Path, event_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    readiness = pd.read_csv(readiness_path, usecols=READINESS_COLUMNS, low_memory=False)
    events = pd.read_csv(event_path, usecols=EVENT_COLUMNS, low_memory=False)
    if set(readiness.columns) != set(READINESS_COLUMNS):
        raise RegistryValidationError("Readiness whitelist columns changed or are missing")
    if set(events.columns) != set(EVENT_COLUMNS):
        raise RegistryValidationError("Event-registry whitelist columns changed or are missing")
    return readiness[READINESS_COLUMNS], events[EVENT_COLUMNS]


def validate_readiness(readiness: pd.DataFrame, events: pd.DataFrame) -> None:
    if list(readiness.columns) != READINESS_COLUMNS:
        raise RegistryValidationError("Readiness whitelist columns changed or are missing")
    if len(readiness) != EXPECTED_SAMPLE_COUNT:
        raise RegistryValidationError(
            f"PILD readiness must contain exactly {EXPECTED_SAMPLE_COUNT} rows; observed {len(readiness)}"
        )
    if readiness["sample_id"].isna().any() or readiness["sample_id"].duplicated().any():
        raise RegistryValidationError("readiness sample_id values must be non-null and unique")
    if readiness["physical_event_id"].isna().any():
        raise RegistryValidationError("readiness physical_event_id contains null values")
    if int(readiness["physical_event_id"].nunique()) != EXPECTED_EVENT_COUNT:
        raise RegistryValidationError(
            f"Expected {EXPECTED_EVENT_COUNT} PILD physical events; observed "
            f"{readiness['physical_event_id'].nunique()}"
        )
    for column in ("event_date_valid", "window_selection_uses_label", "locked_retrospective"):
        readiness[column] = pd.to_numeric(readiness[column], errors="raise").astype(int)
    if not readiness["window_selection_uses_label"].eq(0).all():
        raise RegistryValidationError("Trigger input rejected: readiness window selection used labels")
    if not readiness["locked_retrospective"].eq(1).all():
        raise RegistryValidationError("Trigger input rejected: readiness rows are not locked retrospective")
    coordinate_columns = ("bbox_left", "bbox_bottom", "bbox_right", "bbox_top")
    for column in coordinate_columns:
        readiness[column] = pd.to_numeric(readiness[column], errors="raise")
        if not np.isfinite(readiness[column].to_numpy(dtype=np.float64)).all():
            raise RegistryValidationError(f"Non-finite readiness coordinate: {column}")
    if (readiness["bbox_left"] > readiness["bbox_right"]).any() or (
        readiness["bbox_bottom"] > readiness["bbox_top"]
    ).any():
        raise RegistryValidationError("Readiness contains inverted bounding boxes")
    centers_lon = (readiness["bbox_left"] + readiness["bbox_right"]) / 2.0
    centers_lat = (readiness["bbox_bottom"] + readiness["bbox_top"]) / 2.0
    if not centers_lon.between(-180.0, 180.0).all() or not centers_lat.between(-90.0, 90.0).all():
        raise RegistryValidationError("Readiness sample centers are outside geographic bounds")

    if events["physical_event_id"].isna().any() or events["physical_event_id"].duplicated().any():
        raise RegistryValidationError("Canonical event registry physical_event_id must be unique")
    readiness_ids = set(readiness["physical_event_id"].astype(str))
    event_ids = set(events["physical_event_id"].astype(str))
    missing = sorted(readiness_ids - event_ids)
    if missing:
        raise RegistryValidationError(f"Canonical event registry misses PILD events: {missing}")


def q_r_reason(
    all_dates_valid: bool,
    unique_date_count: int,
    canonical_matches: bool,
    rainfall_mechanism: bool,
    coverage_complete: bool,
) -> str:
    if not all_dates_valid:
        return "event_date_invalid"
    if unique_date_count != 1:
        return "event_date_not_unique"
    if not canonical_matches:
        return "event_date_canonical_mismatch"
    if not rainfall_mechanism:
        return "mechanism_not_rainfall"
    if not coverage_complete:
        return "incomplete_chirps_coverage"
    return "rainfall_strict_d7_complete"


def prepare_event_frame(readiness: pd.DataFrame, canonical_events: pd.DataFrame) -> pd.DataFrame:
    source = readiness.copy()
    source["sample_center_lon"] = (source["bbox_left"] + source["bbox_right"]) / 2.0
    source["sample_center_lat"] = (source["bbox_bottom"] + source["bbox_top"]) / 2.0
    event_lookup = canonical_events.set_index("physical_event_id", drop=False)
    rows: list[dict[str, Any]] = []
    for physical_event_id, group in source.groupby("physical_event_id", sort=True):
        physical_event_id = str(physical_event_id)
        metadata = event_lookup.loc[physical_event_id]
        dates = sorted({str(value).strip() for value in group["event_date"].dropna() if str(value).strip()})
        canonical_text = str(metadata["canonical_date"]).strip()
        canonical_valid = True
        try:
            canonical = strict_iso_date(canonical_text, "canonical_date")
        except RegistryValidationError:
            canonical_valid = False
            canonical = None
        readiness_dates_valid = True
        for value in dates:
            try:
                strict_iso_date(value, "readiness event_date")
            except RegistryValidationError:
                readiness_dates_valid = False
        all_dates_valid = bool(group["event_date_valid"].eq(1).all() and readiness_dates_valid)
        canonical_matches = bool(canonical_valid and len(dates) == 1 and dates[0] == canonical_text)
        family = str(metadata["physical_trigger_family"]).strip().lower()
        rainfall = family == RAINFALL_CANONICAL_FAMILY
        date_gate = all_dates_valid and len(dates) == 1 and canonical_matches
        row: dict[str, Any] = {
            "physical_event_id": physical_event_id,
            "canonical_event_date": canonical_text,
            "readiness_event_dates": "|".join(dates),
            "n_unique_readiness_dates": len(dates),
            "all_readiness_dates_valid": int(all_dates_valid),
            "date_unique_and_canonical": int(date_gate),
            "physical_trigger_family": family,
            "rainfall_mechanism": int(rainfall),
            "dataset_ids": "|".join(sorted(set(group["dataset_id"].astype(str)))),
            "event_uids": "|".join(sorted(set(group["event_uid"].astype(str)))),
            "n_samples": len(group),
            "event_center_lon": float(np.median(group["sample_center_lon"].to_numpy(dtype=float))),
            "event_center_lat": float(np.median(group["sample_center_lat"].to_numpy(dtype=float))),
            "event_center_contract": "median_readiness_sample_bbox_centers",
            "case_window_start": (
                (canonical - timedelta(days=7)).isoformat() if canonical is not None else ""
            ),
            "case_window_end": (
                (canonical - timedelta(days=1)).isoformat() if canonical is not None else ""
            ),
        }
        for role in ANCHOR_SHIFTS:
            row[f"rain_d7_{role}_mm"] = np.nan
            row[f"{role}_days_available"] = 0
        row["rain_d7_wrongtime_median_mm"] = np.nan
        row["rain_d7_case_minus_wrongtime_mm"] = np.nan
        row["available_days_total"] = 0
        row["required_days_total"] = len(ANCHOR_SHIFTS) * len(STRICT_LAGS)
        row["chirps_coverage_complete"] = 0
        row["q_R"] = 0
        row["q_R_reason"] = q_r_reason(
            all_dates_valid, len(dates), canonical_matches, rainfall, False
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("physical_event_id").reset_index(drop=True)


def validate_raster_contract(source: rasterio.io.DatasetReader, path: Path) -> dict[str, Any]:
    if source.count != 1 or source.crs is None or not source.crs.is_geographic:
        raise RegistryValidationError(f"Invalid CHIRPS raster contract: {path}")
    resolution = [abs(float(source.transform.a)), abs(float(source.transform.e))]
    if not all(math.isclose(value, 0.05, abs_tol=1e-8) for value in resolution):
        raise RegistryValidationError(f"Unexpected CHIRPS resolution {resolution}: {path}")
    return {
        "raster_width": int(source.width),
        "raster_height": int(source.height),
        "raster_crs": str(source.crs),
        "resolution_lon_degrees": resolution[0],
        "resolution_lat_degrees": resolution[1],
    }


def sample_event_3x3(
    source: rasterio.io.DatasetReader, lon: float, lat: float
) -> tuple[float | None, int, str]:
    try:
        row, column = source.index(lon, lat)
    except (ValueError, OverflowError):
        return None, 0, "outside_raster"
    if row < 0 or row >= source.height or column < 0 or column >= source.width:
        return None, 0, "outside_raster"
    row_start = max(0, row - 1)
    row_stop = min(source.height, row + 2)
    column_start = max(0, column - 1)
    column_stop = min(source.width, column + 2)
    block = source.read(
        1,
        window=Window(column_start, row_start, column_stop - column_start, row_stop - row_start),
        out_dtype="float64",
    )
    valid = np.isfinite(block) & (block >= 0.0)
    if source.nodata is not None and math.isfinite(float(source.nodata)):
        valid &= ~np.isclose(block, float(source.nodata), rtol=0.0, atol=1e-8)
    values = block[valid]
    if not values.size:
        return None, 0, "no_valid_3x3_cells"
    return float(np.median(values)), int(values.size), "valid"


def build_chirps_requests(event_frame: pd.DataFrame) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    candidates = event_frame[
        event_frame["date_unique_and_canonical"].eq(1)
        & event_frame["rainfall_mechanism"].eq(1)
    ]
    for event in candidates.itertuples(index=False):
        anchor = strict_iso_date(event.canonical_event_date, "canonical_event_date")
        for role, shift in ANCHOR_SHIFTS.items():
            for lag, day in zip(STRICT_LAGS, strict_window_dates(anchor, shift)):
                requests.append(
                    {
                        "physical_event_id": str(event.physical_event_id),
                        "anchor_role": role,
                        "anchor_shift_days": shift,
                        "antecedent_lag_days": lag,
                        "chirps_date": day.isoformat(),
                        "event_center_lon": float(event.event_center_lon),
                        "event_center_lat": float(event.event_center_lat),
                    }
                )
    return requests


def extract_chirps(
    requests: Sequence[dict[str, Any]], chirps_root: Path
) -> pd.DataFrame:
    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        by_date[strict_iso_date(request["chirps_date"], "chirps_date")].append(request)
    manifest_rows: list[dict[str, Any]] = []
    for index, day in enumerate(sorted(by_date), start=1):
        path = chirps_path(chirps_root, day)
        base: dict[str, Any] = {
            "source_path": str(path),
            "source_bytes": 0,
            "source_sha256": "",
            "raster_width": 0,
            "raster_height": 0,
            "raster_crs": "",
            "resolution_lon_degrees": np.nan,
            "resolution_lat_degrees": np.nan,
        }
        if not path.is_file():
            for request in by_date[day]:
                manifest_rows.append(
                    {**request, **base, "rainfall_mm": np.nan, "valid_3x3_cells": 0, "status": "missing_file"}
                )
            continue
        try:
            compressed = path.read_bytes()
            base["source_bytes"] = len(compressed)
            base["source_sha256"] = sha256_bytes(compressed)
            uncompressed = gzip.decompress(compressed)
            with MemoryFile(uncompressed) as memory:
                with memory.open() as source:
                    base.update(validate_raster_contract(source, path))
                    for request in by_date[day]:
                        value, valid_cells, status = sample_event_3x3(
                            source,
                            float(request["event_center_lon"]),
                            float(request["event_center_lat"]),
                        )
                        manifest_rows.append(
                            {
                                **request,
                                **base,
                                "rainfall_mm": value if value is not None else np.nan,
                                "valid_3x3_cells": valid_cells,
                                "status": status,
                            }
                        )
        except (OSError, EOFError, rasterio.errors.RasterioError, RegistryValidationError) as exc:
            for request in by_date[day]:
                manifest_rows.append(
                    {
                        **request,
                        **base,
                        "rainfall_mm": np.nan,
                        "valid_3x3_cells": 0,
                        "status": f"invalid_raster:{type(exc).__name__}",
                    }
                )
        if index % 100 == 0 or index == len(by_date):
            print(f"[chirps] validated {index}/{len(by_date)} unique dates", flush=True)
    columns = [
        "physical_event_id",
        "anchor_role",
        "anchor_shift_days",
        "antecedent_lag_days",
        "chirps_date",
        "event_center_lon",
        "event_center_lat",
        "source_path",
        "source_bytes",
        "source_sha256",
        "raster_width",
        "raster_height",
        "raster_crs",
        "resolution_lon_degrees",
        "resolution_lat_degrees",
        "rainfall_mm",
        "valid_3x3_cells",
        "status",
    ]
    return pd.DataFrame(manifest_rows, columns=columns).sort_values(
        ["physical_event_id", "anchor_shift_days", "antecedent_lag_days"]
    ).reset_index(drop=True)


def apply_event_rainfall(event_frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    output = event_frame.copy()
    for position, event in output.iterrows():
        event_id = str(event["physical_event_id"])
        rows = manifest[manifest["physical_event_id"].astype(str).eq(event_id)]
        available_total = 0
        role_values: dict[str, float | None] = {}
        for role in ANCHOR_SHIFTS:
            role_rows = rows[rows["anchor_role"].eq(role)]
            valid_rows = role_rows[
                role_rows["status"].eq("valid") & role_rows["rainfall_mm"].notna()
            ]
            available = int(len(valid_rows))
            output.loc[position, f"{role}_days_available"] = available
            available_total += available
            complete = len(role_rows) == len(STRICT_LAGS) and available == len(STRICT_LAGS)
            value = float(valid_rows["rainfall_mm"].sum()) if complete else None
            role_values[role] = value
            output.loc[position, f"rain_d7_{role}_mm"] = value if value is not None else np.nan
        output.loc[position, "available_days_total"] = available_total
        complete_coverage = bool(
            event["date_unique_and_canonical"] == 1
            and event["rainfall_mechanism"] == 1
            and all(role_values[role] is not None for role in ANCHOR_SHIFTS)
            and available_total == len(ANCHOR_SHIFTS) * len(STRICT_LAGS)
        )
        output.loc[position, "chirps_coverage_complete"] = int(complete_coverage)
        if complete_coverage:
            controls = [float(role_values[role]) for role in CONTROL_ROLES]
            control_median = float(np.median(controls))
            output.loc[position, "rain_d7_wrongtime_median_mm"] = control_median
            output.loc[position, "rain_d7_case_minus_wrongtime_mm"] = (
                float(role_values["case"]) - control_median
            )
        reason = q_r_reason(
            bool(event["all_readiness_dates_valid"]),
            int(event["n_unique_readiness_dates"]),
            bool(event["date_unique_and_canonical"]),
            bool(event["rainfall_mechanism"]),
            complete_coverage,
        )
        output.loc[position, "q_R_reason"] = reason
        output.loc[position, "q_R"] = int(reason == "rainfall_strict_d7_complete")
    for column in [
        "all_readiness_dates_valid",
        "date_unique_and_canonical",
        "rainfall_mechanism",
        *[f"{role}_days_available" for role in ANCHOR_SHIFTS],
        "available_days_total",
        "required_days_total",
        "chirps_coverage_complete",
        "q_R",
    ]:
        output[column] = output[column].astype(int)
    return output


def event_record_hash(row: Mapping[str, Any]) -> str:
    excluded = {"event_record_sha256"}
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded:
            continue
        if pd.isna(value):
            payload[key] = None
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            payload[key] = (
                str(int(numeric)) if numeric.is_integer() else format(numeric, ".12g")
            )
        else:
            payload[key] = str(value)
    return sha256_bytes(canonical_json_bytes(payload))


def chirps_source_set_sha256(manifest: pd.DataFrame) -> str:
    sources = []
    unique = manifest[["source_path", "source_bytes", "source_sha256"]].drop_duplicates()
    for row in unique.sort_values("source_path").itertuples(index=False):
        sources.append(
            {
                "source_path": str(row.source_path),
                "source_bytes": int(row.source_bytes),
                "source_sha256": "" if pd.isna(row.source_sha256) else str(row.source_sha256),
            }
        )
    return sha256_bytes(canonical_json_bytes(sources))


def add_build_identity(
    event_frame: pd.DataFrame,
    input_hashes: Mapping[str, str],
    chirps_sources_sha256: str,
) -> tuple[pd.DataFrame, str]:
    build_id = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": CONTRACT,
                "input_hashes": dict(sorted(input_hashes.items())),
                "chirps_source_set_sha256": chirps_sources_sha256,
            }
        )
    )
    output = event_frame.copy()
    output.insert(0, "registry_build_id", build_id)
    output.insert(1, "registry_schema_version", SCHEMA_VERSION)
    output["event_record_sha256"] = ""
    output["event_record_sha256"] = [
        event_record_hash(row) for row in output.to_dict("records")
    ]
    columns = list(output.columns)
    columns.insert(2, columns.pop(columns.index("event_record_sha256")))
    return output[columns], build_id


def build_sample_frame(readiness: pd.DataFrame, event_frame: pd.DataFrame) -> pd.DataFrame:
    sample = readiness[
        ["sample_id", "event_uid", "physical_event_id", "dataset_id", "event_date", "event_date_valid"]
    ].copy()
    sample = sample.merge(
        event_frame[["physical_event_id", *EVENT_BROADCAST_COLUMNS]],
        on="physical_event_id",
        how="left",
        validate="many_to_one",
    )
    if sample["registry_build_id"].isna().any():
        raise RegistryValidationError("Sample broadcast missed one or more physical events")
    sample.insert(1, "registry_schema_version", SCHEMA_VERSION)
    sample.insert(2, "broadcast_from_event_registry", 1)
    return sample.sort_values("sample_id").reset_index(drop=True)


def scalar_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10)
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


def validate_frames(
    readiness: pd.DataFrame,
    canonical_events: pd.DataFrame,
    event_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
    chirps_manifest: pd.DataFrame,
) -> dict[str, Any]:
    expected_samples = set(readiness["sample_id"].astype(str))
    observed_samples = set(sample_frame["sample_id"].astype(str))
    if len(sample_frame) != EXPECTED_SAMPLE_COUNT or sample_frame["sample_id"].duplicated().any():
        raise RegistryValidationError("Sample registry is not 2937 unique rows")
    if expected_samples != observed_samples:
        raise RegistryValidationError(
            f"sample_id coverage mismatch: missing={len(expected_samples-observed_samples)} "
            f"extra={len(observed_samples-expected_samples)}"
        )
    expected_events = set(readiness["physical_event_id"].astype(str))
    observed_events = set(event_frame["physical_event_id"].astype(str))
    if len(event_frame) != EXPECTED_EVENT_COUNT or event_frame["physical_event_id"].duplicated().any():
        raise RegistryValidationError("Event registry is not 42 unique physical events")
    if expected_events != observed_events:
        raise RegistryValidationError("Event registry physical_event_id coverage mismatch")
    for frame_name, frame in (("event", event_frame), ("sample", sample_frame)):
        prohibited = sorted(
            column
            for column in frame.columns
            if any(token in column.lower() for token in PROHIBITED_OUTPUT_TOKENS)
        )
        if prohibited:
            raise RegistryValidationError(f"{frame_name} registry has prohibited columns: {prohibited}")
    if not sample_frame["broadcast_from_event_registry"].eq(1).all():
        raise RegistryValidationError("Sample rows are not marked as event broadcasts")
    if not event_frame["q_R"].isin([0, 1]).all() or not sample_frame["q_R"].isin([0, 1]).all():
        raise RegistryValidationError("q_R must be binary")
    blocked = event_frame[
        event_frame["date_unique_and_canonical"].ne(1)
        | event_frame["rainfall_mechanism"].ne(1)
        | event_frame["chirps_coverage_complete"].ne(1)
    ]
    if not blocked["q_R"].eq(0).all():
        raise RegistryValidationError("A blocked event has q_R=1")
    admitted = event_frame[event_frame["q_R"].eq(1)]
    if not admitted[
        ["date_unique_and_canonical", "rainfall_mechanism", "chirps_coverage_complete"]
    ].eq(1).all().all():
        raise RegistryValidationError("An admitted event violates a fail-closed gate")
    if not event_frame["n_samples"].sum() == EXPECTED_SAMPLE_COUNT:
        raise RegistryValidationError("Event n_samples does not sum to 2937")

    event_lookup = event_frame.set_index("physical_event_id", drop=False)
    for event_id, group in sample_frame.groupby("physical_event_id", sort=True):
        event = event_lookup.loc[str(event_id)]
        if len(group) != int(event["n_samples"]):
            raise RegistryValidationError(f"Broadcast count mismatch for {event_id}")
        for column in EVENT_BROADCAST_COLUMNS:
            if not all(scalar_equal(value, event[column]) for value in group[column]):
                raise RegistryValidationError(f"Broadcast mismatch for {event_id}.{column}")

    for row in event_frame.to_dict("records"):
        if event_record_hash(row) != str(row["event_record_sha256"]):
            raise RegistryValidationError(
                f"event_record_sha256 mismatch for {row['physical_event_id']}"
            )

    candidate_count = int(
        (
            event_frame["date_unique_and_canonical"].eq(1)
            & event_frame["rainfall_mechanism"].eq(1)
        ).sum()
    )
    expected_manifest_rows = candidate_count * len(ANCHOR_SHIFTS) * len(STRICT_LAGS)
    if len(chirps_manifest) != expected_manifest_rows:
        raise RegistryValidationError(
            f"CHIRPS manifest row count mismatch: expected={expected_manifest_rows}, "
            f"observed={len(chirps_manifest)}"
        )
    key_columns = ["physical_event_id", "anchor_role", "antecedent_lag_days"]
    if chirps_manifest.duplicated(key_columns).any():
        raise RegistryValidationError("CHIRPS manifest has duplicate event-role-lag usages")
    for row in chirps_manifest.itertuples(index=False):
        event = event_lookup.loc[str(row.physical_event_id)]
        anchor = strict_iso_date(event["canonical_event_date"], "canonical_event_date")
        expected_day = anchor + timedelta(
            days=int(row.anchor_shift_days) - int(row.antecedent_lag_days)
        )
        if row.chirps_date != expected_day.isoformat():
            raise RegistryValidationError(
                f"Non-strict CHIRPS date for {row.physical_event_id}/{row.anchor_role}/D-{row.antecedent_lag_days}"
            )
    canonical_map = canonical_events.set_index("physical_event_id")["physical_trigger_family"]
    for row in event_frame.itertuples(index=False):
        if str(row.physical_trigger_family) != str(canonical_map.loc[row.physical_event_id]).lower():
            raise RegistryValidationError(f"Mechanism drift for {row.physical_event_id}")

    return {
        "sample_rows": len(sample_frame),
        "sample_ids_unique": int(sample_frame["sample_id"].nunique()),
        "physical_events": len(event_frame),
        "candidate_rainfall_events": candidate_count,
        "q_R_positive_events": int(event_frame["q_R"].sum()),
        "q_R_positive_samples": int(sample_frame["q_R"].sum()),
        "chirps_manifest_rows": len(chirps_manifest),
        "chirps_status_counts": {
            str(key): int(value) for key, value in chirps_manifest["status"].value_counts().items()
        },
        "q_R_reason_event_counts": {
            str(key): int(value) for key, value in event_frame["q_R_reason"].value_counts().items()
        },
        "q_R_reason_sample_counts": {
            str(key): int(value) for key, value in sample_frame["q_R_reason"].value_counts().items()
        },
    }


def artifact_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def build_audit(
    build_id: str,
    input_paths: Mapping[str, Path],
    input_hashes: Mapping[str, str],
    chirps_sources_sha256: str,
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "registry_build_id": build_id,
        "contract": CONTRACT,
        "input_policy": {
            "readiness_columns_loaded": READINESS_COLUMNS,
            "event_columns_loaded": EVENT_COLUMNS,
            "segmentation_labels_loaded": False,
            "model_results_loaded": False,
        },
        "inputs": {
            key: {"path": str(input_paths[key]), "sha256": input_hashes[key]}
            for key in sorted(input_paths)
        },
        "chirps_source_set_sha256": chirps_sources_sha256,
        "counts": dict(counts),
        "invariants": {
            "sample_id_exact_2937_coverage": True,
            "sample_id_unique": True,
            "event_first_then_broadcast": True,
            "strict_d7_to_d1_only": True,
            "wrong_time_shifts_exact": list(ANCHOR_SHIFTS.values())[1:],
            "non_unique_or_mismatched_date_q_R_zero": True,
            "non_rainfall_mechanism_q_R_zero": True,
            "incomplete_chirps_coverage_q_R_zero": True,
            "no_imputation": True,
            "label_free_and_model_result_free": True,
        },
        "validation": {
            "command": "python scripts/xdomain/build_pild_prithvi_trigger_registry_v1.py --validate-only",
            "hash_manifest_is_commit_marker": True,
        },
    }


def verify_chirps_source_hashes(chirps_manifest: pd.DataFrame) -> int:
    valid = chirps_manifest[chirps_manifest["source_sha256"].fillna("").astype(str).ne("")]
    sources = valid[["source_path", "source_sha256"]].drop_duplicates()
    for row in sources.itertuples(index=False):
        path = Path(str(row.source_path))
        if not path.is_file():
            raise RegistryValidationError(f"Recorded CHIRPS source is missing: {path}")
        observed = sha256_file(path)
        if observed != str(row.source_sha256):
            raise RegistryValidationError(f"Recorded CHIRPS source hash changed: {path}")
    return len(sources)


def validate_committed(
    readiness_path: Path,
    event_path: Path,
    paths: Mapping[str, Path],
    verify_chirps: bool,
) -> dict[str, Any]:
    for path in paths.values():
        if not path.is_file():
            raise RegistryValidationError(f"Required committed artifact is missing: {path}")
    hash_manifest = json.loads(paths["hash_manifest"].read_text(encoding="ascii"))
    if hash_manifest.get("schema_version") != SCHEMA_VERSION:
        raise RegistryValidationError("Hash manifest schema version mismatch")
    expected_contract_hash = sha256_bytes(canonical_json_bytes(CONTRACT))
    if hash_manifest.get("contract_sha256") != expected_contract_hash:
        raise RegistryValidationError("Hash manifest contract hash mismatch")
    input_lookup = {"event_registry": event_path, "readiness": readiness_path}
    for key, path in input_lookup.items():
        recorded = hash_manifest["inputs"][key]
        if str(path) != recorded["path"] or sha256_file(path) != recorded["sha256"]:
            raise RegistryValidationError(f"Committed input hash mismatch: {key}")
    for key, record in hash_manifest["outputs"].items():
        path = paths[key]
        if str(path) != record["path"] or sha256_file(path) != record["sha256"]:
            raise RegistryValidationError(f"Committed output hash mismatch: {key}")

    readiness, canonical_events = read_inputs(readiness_path, event_path)
    validate_readiness(readiness, canonical_events)
    event_frame = pd.read_csv(paths["event_registry"], low_memory=False)
    sample_frame = pd.read_csv(paths["sample_registry"], low_memory=False)
    chirps_manifest = pd.read_csv(paths["chirps_manifest"], low_memory=False)
    observed_chirps_fingerprint = chirps_source_set_sha256(chirps_manifest)
    if hash_manifest.get("chirps_source_set_sha256") != observed_chirps_fingerprint:
        raise RegistryValidationError("CHIRPS source-set fingerprint mismatch")
    counts = validate_frames(
        readiness, canonical_events, event_frame, sample_frame, chirps_manifest
    )
    if verify_chirps:
        counts["unique_chirps_sources_rehashed"] = verify_chirps_source_hashes(chirps_manifest)
    audit = json.loads(paths["audit"].read_text(encoding="ascii"))
    if audit.get("status") != "passed" or audit.get("registry_build_id") != hash_manifest.get(
        "registry_build_id"
    ):
        raise RegistryValidationError("Audit and hash manifest build identities disagree")
    if audit.get("chirps_source_set_sha256") != observed_chirps_fingerprint:
        raise RegistryValidationError("Audit CHIRPS source-set fingerprint mismatch")
    return counts


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    readiness_path = resolve(root, args.readiness)
    event_path = resolve(root, args.event_registry)
    chirps_root = resolve(root, args.chirps_root)
    outdir = resolve(root, args.outdir)
    paths = output_paths(outdir)

    if args.validate_only:
        counts = validate_committed(
            readiness_path,
            event_path,
            paths,
            verify_chirps=not args.skip_chirps_rehash,
        )
        print(json.dumps({"status": "passed", **counts}, sort_keys=True))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    readiness, canonical_events = read_inputs(readiness_path, event_path)
    validate_readiness(readiness, canonical_events)
    input_paths = {"readiness": readiness_path, "event_registry": event_path}
    input_hashes = {key: sha256_file(path) for key, path in input_paths.items()}

    event_frame = prepare_event_frame(readiness, canonical_events)
    requests = build_chirps_requests(event_frame)
    chirps_manifest = extract_chirps(requests, chirps_root)
    event_frame = apply_event_rainfall(event_frame, chirps_manifest)
    chirps_sources_sha256 = chirps_source_set_sha256(chirps_manifest)
    event_frame, build_id = add_build_identity(
        event_frame, input_hashes, chirps_sources_sha256
    )
    sample_frame = build_sample_frame(readiness, event_frame)
    counts = validate_frames(
        readiness, canonical_events, event_frame, sample_frame, chirps_manifest
    )
    audit = build_audit(
        build_id, input_paths, input_hashes, chirps_sources_sha256, counts
    )

    atomic_write_csv(paths["event_registry"], event_frame)
    atomic_write_csv(paths["sample_registry"], sample_frame)
    atomic_write_csv(paths["chirps_manifest"], chirps_manifest)
    atomic_write_json(paths["audit"], audit)
    rows = {
        "event_registry": len(event_frame),
        "sample_registry": len(sample_frame),
        "chirps_manifest": len(chirps_manifest),
    }
    hash_manifest = {
        "schema_version": SCHEMA_VERSION,
        "registry_build_id": build_id,
        "algorithm": "sha256",
        "contract_sha256": sha256_bytes(canonical_json_bytes(CONTRACT)),
        "chirps_source_set_sha256": chirps_sources_sha256,
        "inputs": {
            key: artifact_record(path) for key, path in sorted(input_paths.items())
        },
        "outputs": {
            key: artifact_record(path, rows.get(key))
            for key, path in paths.items()
            if key != "hash_manifest"
        },
    }
    atomic_write_json(paths["hash_manifest"], hash_manifest)

    committed_counts = validate_committed(
        readiness_path, event_path, paths, verify_chirps=False
    )
    print(
        f"[done] build={build_id[:12]} samples={committed_counts['sample_rows']} "
        f"events={committed_counts['physical_events']} "
        f"q_R_events={committed_counts['q_R_positive_events']} "
        f"q_R_samples={committed_counts['q_R_positive_samples']}",
        flush=True,
    )
    print(f"[done] {paths['hash_manifest']}", flush=True)


if __name__ == "__main__":
    main()
