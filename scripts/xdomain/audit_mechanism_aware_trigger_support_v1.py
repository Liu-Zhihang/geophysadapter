#!/usr/bin/env python3
"""Audit mechanism-aware Trigger support without reading model outcomes.

The audit preserves all 57 source events (42 PILD + 15 Sen12) while also
reporting alias-collapsed canonical-event counts. Earthquakes are matched only
against the official USGS ComCat/ShakeMap services. Snowmelt is limited to an
ERA5-Land variable/time-window availability audit; no ERA5 data are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_STEM = Path("metadata/pild_sen12_training_v2/mechanism_aware_trigger_support_v1")
USGS_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_DETAIL_TEMPLATE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/{event_id}.geojson"
USGS_DOC = "https://earthquake.usgs.gov/fdsnws/event/1/"
SHAKEMAP_DOC = "https://earthquake.usgs.gov/data/shakemap/"
ERA5_LAND_DOC = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview"

EARTHQUAKE_RADIUS_KM = 300.0
EARTHQUAKE_MIN_MAGNITUDE = 4.5
EARTHQUAKE_DATE_TOLERANCE_DAYS = 1
GRID_FIELDS = {"pga", "pgv", "mmi"}
ERA5_SNOWMELT_VARIABLES = (
    "2m_temperature",
    "snow_cover",
    "snow_depth",
    "snow_depth_water_equivalent",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def parse_date(value: Any) -> date | None:
    if value is None or pd.isna(value) or str(value).strip() in {"", "undated", "nan", "NaT"}:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    lon1r, lat1r, lon2r, lat2r = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2r - lon1r, lat2r - lat1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def read_json_url(
    url: str,
    cache_path: Path,
    *,
    offline: bool,
    refresh: bool,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    if cache_path.is_file() and not refresh:
        return json.loads(cache_path.read_text())
    if offline:
        raise RuntimeError(f"offline cache missing: {cache_path}")
    request = Request(url, headers={"User-Agent": "GeoPhysAdapter-trigger-audit/1.0"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with opener(request, timeout=60) as response:
                payload = response.read()
            data = json.loads(payload)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
            return data
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"official API request failed: {url}: {last_error}")


def read_grid_header_fields(
    url: str,
    cache_path: Path,
    *,
    offline: bool,
    refresh: bool,
    opener: Callable[..., Any] = urlopen,
) -> set[str]:
    if cache_path.is_file() and not refresh:
        text = cache_path.read_text(errors="ignore")
    else:
        if offline:
            raise RuntimeError(f"offline grid-header cache missing: {cache_path}")
        request = Request(
            url,
            headers={"User-Agent": "GeoPhysAdapter-trigger-audit/1.0", "Range": "bytes=0-262143"},
        )
        with opener(request, timeout=90) as response:
            payload = response.read(262144)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
        text = payload.decode("utf-8", errors="ignore")
    return {
        name.lower()
        for name in re.findall(r"<grid_field\b[^>]*\bname=[\"']([^\"']+)[\"']", text, flags=re.I)
    }


def build_source_events(
    pild: pd.DataFrame,
    sen12: pd.DataFrame,
    pild_material: pd.DataFrame,
    sen12_material: pd.DataFrame,
    aliases: pd.DataFrame,
    declared_usgs: pd.DataFrame,
) -> pd.DataFrame:
    alias_keys = aliases.set_index(["source_collection", "source_event_id"], drop=False)
    declared = declared_usgs.set_index("physical_event_cluster_id", drop=False)
    pild_m = pild_material.set_index("physical_event_id", drop=False)
    sen12_m = sen12_material.set_index("physical_event_cluster_id", drop=False)
    rows: list[dict[str, Any]] = []

    def alias_row(collection: str, event_id: str) -> pd.Series:
        key = (collection, event_id)
        if key not in alias_keys.index:
            raise RuntimeError(f"event absent from frozen alias registry: {key}")
        value = alias_keys.loc[key]
        if isinstance(value, pd.DataFrame):
            raise RuntimeError(f"duplicate alias rows: {key}")
        return value

    for item in pild.itertuples(index=False):
        alias = alias_row("PILD", str(item.physical_event_id))
        if str(item.physical_event_id) not in pild_m.index:
            raise RuntimeError(f"PILD event absent from material registry: {item.physical_event_id}")
        material = pild_m.loc[str(item.physical_event_id)]
        family = str(item.physical_trigger_family)
        if family == "hydrometeorological":
            family = "rainfall"
        rows.append({
            "source_collection": "PILD",
            "source_event_id": str(item.physical_event_id),
            "canonical_event_id": str(alias.canonical_physical_event_id),
            "dataset_ids": str(item.dataset_ids),
            "event_aliases": str(item.event_uids),
            "mechanism_family": family,
            "event_date": item.canonical_event_date,
            "date_reliable": int(item.date_unique_and_canonical),
            "center_lon": item.event_center_lon,
            "center_lat": item.event_center_lat,
            "bbox_left": alias.source_bbox_left,
            "bbox_bottom": alias.source_bbox_bottom,
            "bbox_right": alias.source_bbox_right,
            "bbox_top": alias.source_bbox_top,
            "n_samples": int(item.n_samples),
            "existing_q_R": int(item.q_R),
            "existing_q_R_reason": str(item.q_R_reason),
            "material_q_M_mean": material.q_M_mean,
            "material_q_M_full_fraction": material.q_M_full_mean,
            "material_lithology_classes": material.lithology_classes,
            "declared_usgs_event_id": "",
            "alias_decision": str(alias.alias_decision),
            "alias_evidence": str(alias.decision_evidence),
        })

    for item in sen12.itertuples(index=False):
        event_id = str(item.physical_event_id)
        alias = alias_row("Sen12Landslides", event_id)
        if event_id not in sen12_m.index:
            raise RuntimeError(f"Sen12 event absent from material registry: {event_id}")
        material = sen12_m.loc[event_id]
        declared_id = ""
        if event_id in declared.index:
            declared_id = str(declared.loc[event_id, "usgs_event_id"])
        event_date = item.trigger_anchor_date
        if pd.isna(event_date):
            event_date = alias.source_event_date
        rows.append({
            "source_collection": "Sen12Landslides",
            "source_event_id": event_id,
            "canonical_event_id": str(alias.canonical_physical_event_id),
            "dataset_ids": "SEN12LS_HARMONIZED",
            "event_aliases": str(alias.source_event_names),
            "mechanism_family": str(item.mechanism_family),
            "event_date": event_date,
            "date_reliable": int(alias.source_date_reliable),
            "center_lon": alias.source_center_lon,
            "center_lat": alias.source_center_lat,
            "bbox_left": alias.source_bbox_left,
            "bbox_bottom": alias.source_bbox_bottom,
            "bbox_right": alias.source_bbox_right,
            "bbox_top": alias.source_bbox_top,
            "n_samples": int(item.n_samples),
            "existing_q_R": int(item.n_q_R_positive > 0),
            "existing_q_R_reason": str(item.gate_reason),
            "material_q_M_mean": material.q_M_mean,
            "material_q_M_full_fraction": material.q_M_full_fraction,
            "material_lithology_classes": material.lithology_classes,
            "declared_usgs_event_id": declared_id,
            "alias_decision": str(alias.alias_decision),
            "alias_evidence": str(alias.decision_evidence),
        })

    frame = pd.DataFrame(rows)
    if len(frame) != 57 or frame[["source_collection", "source_event_id"]].duplicated().any():
        raise RuntimeError(f"expected 57 unique source events, found {len(frame)}")
    return frame.sort_values(["source_collection", "source_event_id"]).reset_index(drop=True)


def candidate_rows(payload: dict[str, Any], event_date: date, lon: float, lat: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) < 3:
            continue
        timestamp = pd.to_datetime(props.get("time"), unit="ms", utc=True, errors="coerce")
        magnitude = finite_float(props.get("mag"))
        if pd.isna(timestamp) or magnitude is None:
            continue
        distance = haversine_km(lon, lat, float(coords[0]), float(coords[1]))
        delta_days = abs((timestamp.date() - event_date).days)
        strict = (
            delta_days <= EARTHQUAKE_DATE_TOLERANCE_DAYS
            and distance <= EARTHQUAKE_RADIUS_KM
            and magnitude >= EARTHQUAKE_MIN_MAGNITUDE
        )
        rows.append({
            "event_id": str(feature.get("id", "")),
            "time_utc": timestamp.isoformat(),
            "event_lon": float(coords[0]),
            "event_lat": float(coords[1]),
            "date_delta_days": int(delta_days),
            "distance_km": distance,
            "magnitude": magnitude,
            "depth_km": finite_float(coords[2]),
            "place": str(props.get("place", "")),
            "strict_candidate": bool(strict),
        })
    return sorted(rows, key=lambda row: (not row["strict_candidate"], row["date_delta_days"], row["distance_km"], -row["magnitude"]))


def choose_shakemap_product(detail: dict[str, Any]) -> dict[str, Any] | None:
    products = detail.get("properties", {}).get("products", {}).get("shakemap", [])
    eligible = [p for p in products if "download/grid.xml" in p.get("contents", {})]
    if not eligible:
        return None

    def quality(product: dict[str, Any]) -> tuple[int, float]:
        props = product.get("properties", {})
        reviewed = str(props.get("review-status", "")).lower() == "reviewed"
        released = str(props.get("map-status", "")).lower() == "released"
        return (2 if reviewed else 1 if released else 0, float(product.get("preferredWeight", 0)))

    return max(eligible, key=quality)


def audit_earthquake(
    row: dict[str, Any],
    raw_dir: Path,
    *,
    offline: bool,
    refresh: bool,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    event_date = parse_date(row["event_date"])
    lon, lat = finite_float(row["center_lon"]), finite_float(row["center_lat"])
    if event_date is None or not int(row["date_reliable"]):
        return {"support_status": "needs_review", "support_reason": "earthquake_date_not_unique_or_reliable"}
    if lon is None or lat is None:
        return {"support_status": "unsupported", "support_reason": "earthquake_center_missing"}

    params = {
        "format": "geojson",
        "starttime": (event_date - timedelta(days=1)).isoformat(),
        "endtime": (event_date + timedelta(days=2)).isoformat(),
        "latitude": f"{lat:.8f}",
        "longitude": f"{lon:.8f}",
        "maxradiuskm": str(int(EARTHQUAKE_RADIUS_KM)),
        "minmagnitude": str(EARTHQUAKE_MIN_MAGNITUDE),
        "orderby": "time",
    }
    query_url = f"{USGS_QUERY}?{urlencode(params)}"
    cache_prefix = f"{row['source_collection']}_{row['source_event_id']}".replace("/", "_")
    try:
        payload = read_json_url(
            query_url, raw_dir / f"{cache_prefix}_query.json", offline=offline, refresh=refresh, opener=opener
        )
    except RuntimeError as exc:
        return {
            "support_status": "needs_review",
            "support_reason": "usgs_query_unavailable",
            "api_error": str(exc),
            "usgs_query_url": query_url,
        }
    candidates = candidate_rows(payload, event_date, lon, lat)
    strict = [candidate for candidate in candidates if candidate["strict_candidate"]]
    declared = str(row.get("declared_usgs_event_id", "")).strip()
    chosen: dict[str, Any] | None = None
    match_basis = ""
    if declared:
        chosen = next((candidate for candidate in strict if candidate["event_id"] == declared), None)
        if chosen is None:
            return {
                "support_status": "needs_review",
                "support_reason": "preregistered_usgs_id_not_in_strict_candidates",
                "usgs_query_url": query_url,
                "n_usgs_candidates": len(candidates),
                "n_strict_candidates": len(strict),
                "candidate_event_ids": json.dumps([c["event_id"] for c in strict]),
            }
        match_basis = "preregistered_usgs_event_id_verified"
    elif len(strict) == 1:
        chosen = strict[0]
        match_basis = "unique_strict_date_distance_candidate"
    elif len(strict) > 1:
        return {
            "support_status": "needs_review",
            "support_reason": "multiple_strict_usgs_candidates_require_manual_review",
            "usgs_query_url": query_url,
            "n_usgs_candidates": len(candidates),
            "n_strict_candidates": len(strict),
            "candidate_event_ids": json.dumps([c["event_id"] for c in strict]),
            "candidate_summary_json": json.dumps(json_safe(strict), separators=(",", ":")),
        }
    else:
        return {
            "support_status": "unsupported",
            "support_reason": "no_strict_usgs_candidate",
            "usgs_query_url": query_url,
            "n_usgs_candidates": len(candidates),
            "n_strict_candidates": 0,
        }

    event_id = chosen["event_id"]
    detail_url = USGS_DETAIL_TEMPLATE.format(event_id=event_id)
    try:
        detail = read_json_url(
            detail_url, raw_dir / f"{event_id}_detail.json", offline=offline, refresh=refresh, opener=opener
        )
        product = choose_shakemap_product(detail)
        if product is None:
            return {
                "support_status": "unsupported",
                "support_reason": "usgs_event_has_no_shakemap_grid_product",
                "usgs_event_id": event_id,
                "match_basis": match_basis,
            }
        grid_url = product["contents"]["download/grid.xml"]["url"]
        fields = read_grid_header_fields(
            grid_url, raw_dir / f"{event_id}_grid_header.xml", offline=offline, refresh=refresh, opener=opener
        )
    except (RuntimeError, HTTPError, URLError, TimeoutError) as exc:
        return {
            "support_status": "needs_review",
            "support_reason": "usgs_detail_or_shakemap_grid_unavailable",
            "usgs_event_id": event_id,
            "match_basis": match_basis,
            "api_error": str(exc),
        }

    props = product.get("properties", {})
    complete = GRID_FIELDS.issubset(fields)
    return {
        "support_status": "supported" if complete else "unsupported",
        "support_reason": "official_usgs_shakemap_pga_pgv_mmi_complete" if complete else "shakemap_grid_missing_required_fields",
        "usgs_query_url": query_url,
        "usgs_detail_url": detail_url,
        "usgs_event_id": event_id,
        "match_basis": match_basis,
        "n_usgs_candidates": len(candidates),
        "n_strict_candidates": len(strict),
        "candidate_event_ids": json.dumps([c["event_id"] for c in strict]),
        "earthquake_magnitude": chosen["magnitude"],
        "earthquake_depth_km": chosen["depth_km"],
        "earthquake_event_lon": chosen["event_lon"],
        "earthquake_event_lat": chosen["event_lat"],
        "earthquake_distance_km": chosen["distance_km"],
        "earthquake_time_utc": chosen["time_utc"],
        "shakemap_product_source": product.get("source", ""),
        "shakemap_product_status": product.get("status", ""),
        "shakemap_map_status": props.get("map-status", ""),
        "shakemap_review_status": props.get("review-status", ""),
        "shakemap_update_time": product.get("updateTime"),
        "shakemap_grid_url": grid_url,
        "shakemap_has_pga": int("pga" in fields),
        "shakemap_has_pgv": int("pgv" in fields),
        "shakemap_has_mmi": int("mmi" in fields),
        "shakemap_grid_fields": "|".join(sorted(fields)),
    }


def audit_frame(events: pd.DataFrame, raw_dir: Path, *, offline: bool, refresh: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in events.to_dict(orient="records"):
        family = record["mechanism_family"]
        result: dict[str, Any]
        if family == "rainfall":
            supported = bool(record["existing_q_R"])
            result = {
                "support_status": "supported" if supported else "unsupported",
                "support_reason": "existing_strict_rainfall_qR" if supported else "rainfall_registry_gate_not_passed",
            }
        elif family == "earthquake":
            result = audit_earthquake(record, raw_dir, offline=offline, refresh=refresh)
        elif family == "snowmelt":
            event_date = parse_date(record["event_date"])
            result = {
                "support_status": "needs_review" if event_date else "unsupported",
                "support_reason": "era5_land_variables_available_download_pending" if event_date else "snowmelt_event_date_missing",
                "era5_land_variables": "|".join(ERA5_SNOWMELT_VARIABLES),
                "era5_case_window": f"{event_date - timedelta(days=30)}..{event_date}" if event_date else "",
                "era5_reference_window": f"{event_date - timedelta(days=60)}..{event_date - timedelta(days=31)}" if event_date else "",
                "era5_availability_scope": "global_hourly_1950_to_present",
                "era5_downloaded": 0,
            }
        elif family == "complex":
            result = {"support_status": "unsupported", "support_reason": "complex_mechanism_not_decomposed"}
        else:
            result = {"support_status": "unsupported", "support_reason": "mechanism_unknown"}
        record.update(result)
        rows.append(record)
        print(
            f"[{record['source_collection']}] {record['source_event_id']} {family}: "
            f"{record['support_status']} ({record['support_reason']})",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values(["source_collection", "source_event_id"]).reset_index(drop=True)


CANONICAL_OFFICIAL_COLUMNS = (
    "usgs_detail_url",
    "usgs_event_id",
    "earthquake_magnitude",
    "earthquake_depth_km",
    "earthquake_event_lon",
    "earthquake_event_lat",
    "earthquake_time_utc",
    "shakemap_product_source",
    "shakemap_product_status",
    "shakemap_map_status",
    "shakemap_review_status",
    "shakemap_update_time",
    "shakemap_grid_url",
    "shakemap_has_pga",
    "shakemap_has_pgv",
    "shakemap_has_mmi",
    "shakemap_grid_fields",
)


def propagate_canonical_earthquake_support(frame: pd.DataFrame) -> pd.DataFrame:
    """Propagate verified official support across non-conflicting source aliases."""
    output = frame.copy()
    for column in CANONICAL_OFFICIAL_COLUMNS:
        if column not in output:
            output[column] = pd.NA
    output["pre_canonical_support_status"] = output["support_status"]
    output["pre_canonical_support_reason"] = output["support_reason"]
    output["canonical_support_action"] = "none"
    output["canonical_support_source_record"] = ""

    for canonical_id, group in output.groupby("canonical_event_id", sort=False):
        if len(group) < 2:
            continue
        supported = group[
            group["mechanism_family"].eq("earthquake")
            & group["support_status"].eq("supported")
            & group["usgs_event_id"].notna()
            & group["usgs_event_id"].astype(str).ne("")
        ]
        if supported.empty:
            continue

        indices = group.index
        official_ids = set(supported["usgs_event_id"].astype(str))
        conflict_reason = ""
        if not group["mechanism_family"].eq("earthquake").all():
            conflict_reason = "canonical_alias_mechanism_conflict_requires_review"
        elif len(official_ids) != 1:
            conflict_reason = "canonical_alias_usgs_event_id_conflict_requires_review"
        else:
            official_dates = [parse_date(value) for value in supported["earthquake_time_utc"]]
            official_dates = [value for value in official_dates if value is not None]
            source_dates = [parse_date(value) for value in group["event_date"]]
            dates_reliable = group["date_reliable"].fillna(0).astype(int).eq(1).all()
            if len(set(official_dates)) != 1:
                conflict_reason = "canonical_alias_official_event_date_conflict_requires_review"
            elif not dates_reliable or any(value is None for value in source_dates):
                conflict_reason = "canonical_alias_source_date_not_reliable_requires_review"
            elif any(
                abs((value - official_dates[0]).days) > EARTHQUAKE_DATE_TOLERANCE_DAYS
                for value in source_dates
            ):
                conflict_reason = "canonical_alias_source_date_conflict_requires_review"

        if conflict_reason:
            output.loc[indices, "support_status"] = "needs_review"
            output.loc[indices, "support_reason"] = conflict_reason
            output.loc[indices, "canonical_support_action"] = "conflict_blocked"
            continue

        representative = supported.sort_values(["source_collection", "source_event_id"]).iloc[0]
        source_record = f"{representative['source_collection']}::{representative['source_event_id']}"
        event_lon = finite_float(representative["earthquake_event_lon"])
        event_lat = finite_float(representative["earthquake_event_lat"])
        for index in indices:
            if output.at[index, "support_status"] == "supported":
                output.at[index, "canonical_support_action"] = "verified_origin"
                output.at[index, "canonical_support_source_record"] = source_record
                continue
            for column in CANONICAL_OFFICIAL_COLUMNS:
                output.at[index, column] = representative[column]
            center_lon = finite_float(output.at[index, "center_lon"])
            center_lat = finite_float(output.at[index, "center_lat"])
            if None not in (center_lon, center_lat, event_lon, event_lat):
                output.at[index, "earthquake_distance_km"] = haversine_km(
                    center_lon, center_lat, event_lon, event_lat
                )
            output.at[index, "support_status"] = "supported"
            output.at[index, "support_reason"] = (
                "official_usgs_shakemap_support_propagated_via_verified_canonical_alias"
            )
            output.at[index, "match_basis"] = "canonical_alias_verified_usgs_event_id_propagated"
            output.at[index, "canonical_support_action"] = "propagated_from_verified_alias"
            output.at[index, "canonical_support_source_record"] = source_record
    return output


def write_outputs(frame: pd.DataFrame, csv_path: Path, json_path: Path, md_path: Path, inputs: dict[str, Path]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    counts = frame["support_status"].value_counts().to_dict()
    mechanism_counts = frame["mechanism_family"].value_counts().to_dict()
    rainfall_existing = frame[(frame["mechanism_family"] == "rainfall") & (frame["existing_q_R"] == 1)]
    earthquake_supported = frame[(frame["mechanism_family"] == "earthquake") & (frame["support_status"] == "supported")]
    snow_pending = frame[(frame["mechanism_family"] == "snowmelt") & (frame["support_status"] == "needs_review")]
    propagated = frame[frame["canonical_support_action"] == "propagated_from_verified_alias"]
    conflicts = frame[frame["canonical_support_action"] == "conflict_blocked"]
    summary = {
        "schema_version": 1,
        "scope": "label_free_data_support_audit_only",
        "n_source_events": len(frame),
        "n_canonical_events_after_alias_collapse": int(frame["canonical_event_id"].nunique()),
        "mechanism_counts_source_events": mechanism_counts,
        "status_counts_source_events": counts,
        "n_existing_rainfall_supported_source_events": len(rainfall_existing),
        "n_new_earthquake_supported_source_events": len(earthquake_supported),
        "n_new_earthquake_supported_canonical_events": int(earthquake_supported["canonical_event_id"].nunique()),
        "n_canonical_alias_propagated_source_events": len(propagated),
        "n_canonical_alias_conflict_source_events": len(conflicts),
        "n_snowmelt_availability_confirmed_download_pending": len(snow_pending),
        "supported_canonical_events_total": int(
            frame.loc[frame["support_status"] == "supported", "canonical_event_id"].nunique()
        ),
        "status_is_not_model_evidence": True,
        "earthquake_match_contract": {
            "date_tolerance_days": EARTHQUAKE_DATE_TOLERANCE_DAYS,
            "max_distance_km": EARTHQUAKE_RADIUS_KM,
            "min_magnitude": EARTHQUAKE_MIN_MAGNITUDE,
            "ambiguous_candidates": "needs_review unless a preregistered USGS event ID is exactly verified",
            "required_shakemap_grid_fields": sorted(GRID_FIELDS),
        },
        "snowmelt_contract": {
            "variables": list(ERA5_SNOWMELT_VARIABLES),
            "action": "availability audit only; no ERA5-Land data downloaded",
        },
        "official_sources": {"usgs_event_api": USGS_DOC, "usgs_shakemap": SHAKEMAP_DOC, "era5_land": ERA5_LAND_DOC},
        "input_sha256": {str(path): sha256(path) for path in inputs.values()},
        "outputs": {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)},
    }
    json_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")

    lines = [
        "# Mechanism-aware Trigger support audit v1",
        "",
        "## Scope and guardrails",
        "",
        "This is a label-free data-support audit. It does not read model predictions, segmentation labels, IoU, AP, RER, or any training artifact.",
        "The table preserves 57 source events and separately reports alias-collapsed canonical events.",
        "",
        "## Coverage",
        "",
        f"- Source events: **{len(frame)}**; canonical events after alias collapse: **{frame['canonical_event_id'].nunique()}**.",
        f"- Mechanisms: `{json.dumps(mechanism_counts, sort_keys=True)}`.",
        f"- Status: `{json.dumps(counts, sort_keys=True)}`.",
        f"- Existing strict rainfall support: **{len(rainfall_existing)} source events**.",
        f"- Newly supported earthquake records: **{len(earthquake_supported)} source / {earthquake_supported['canonical_event_id'].nunique()} canonical events**.",
        f"- Canonical-alias propagation: **{len(propagated)} source records**; conflict-blocked: **{len(conflicts)} records**.",
        f"- Snowmelt availability confirmed but download pending: **{len(snow_pending)} events**.",
        "",
        "## Decision contract",
        "",
        "- Earthquake: official USGS ComCat only; date and distance gates are mandatory. Multiple strict candidates require manual review unless a preregistered USGS event ID is exactly verified.",
        "- Earthquake support additionally requires an official ShakeMap grid containing PGA, PGV, and MMI fields.",
        "- Verified earthquake support propagates across a canonical alias only when mechanism and reliable source dates agree; conflicts downgrade the whole alias group to needs_review.",
        "- Snowmelt: ERA5-Land variable/time-window availability only. No support is promoted until the data are downloaded and audited.",
        "- Complex and unknown mechanisms remain unsupported in v1.",
        "",
        "## Event-level decisions",
        "",
        "| collection | event | mechanism | date | status | reason | official event | M | distance km |",
        "|---|---|---:|---:|---|---|---|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.source_collection} | {row.source_event_id} | {row.mechanism_family} | "
            f"{json_safe(row.event_date) or ''} | {row.support_status} | {row.support_reason} | "
            f"{json_safe(getattr(row, 'usgs_event_id', '')) or ''} | "
            f"{json_safe(getattr(row, 'earthquake_magnitude', '')) or ''} | "
            f"{json_safe(getattr(row, 'earthquake_distance_km', '')) or ''} |"
        )
    lines.extend([
        "",
        "## Re-run",
        "",
        "```bash",
        "conda run -n dpl python scripts/xdomain/audit_mechanism_aware_trigger_support_v1.py",
        "# Cached/offline reproduction:",
        "conda run -n dpl python scripts/xdomain/audit_mechanism_aware_trigger_support_v1.py --offline",
        "```",
        "",
        "## Official sources",
        "",
        f"- USGS Event Web Service: {USGS_DOC}",
        f"- USGS ShakeMap: {SHAKEMAP_DOC}",
        f"- ERA5-Land: {ERA5_LAND_DOC}",
    ])
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--offline", action="store_true", help="Use only cached official API responses")
    parser.add_argument("--refresh", action="store_true", help="Refresh official API caches")
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = {
        "pild_trigger": root / "processed/hybrid_pinn/pild_prithvi_integration_v1/pild_trigger_event_registry_v1.csv",
        "sen12_trigger": root / "processed/hybrid_pinn/sen12_context_v1/trigger_event_registry_v1.csv",
        "pild_material": root / "processed/hybrid_pinn/pild_prithvi_integration_v1/material_event_registry_v1.csv",
        "sen12_material": root / "processed/hybrid_pinn/sen12_context_v1/material_event_registry.csv",
        "aliases": root / "processed/hybrid_pinn/pild_prithvi_integration_v1/pild_sen12_event_aliases_v1.csv",
        "declared_usgs": root / "metadata/pild_xdomain_v1/sen12_earthquake_event_registry_v1.csv",
        "unified_manifest": root / "metadata/pild_sen12_training_v2/unified_sample_manifest_v2.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required registries missing: {missing}")
    # The unified registry is read only to freeze the current 56 canonical-event identity.
    unified = pd.read_csv(inputs["unified_manifest"], low_memory=False)
    if unified["canonical_event_id"].nunique() != 56:
        raise RuntimeError("expected 56 alias-collapsed canonical events in unified manifest")
    events = build_source_events(
        pd.read_csv(inputs["pild_trigger"], low_memory=False),
        pd.read_csv(inputs["sen12_trigger"], low_memory=False),
        pd.read_csv(inputs["pild_material"], low_memory=False),
        pd.read_csv(inputs["sen12_material"], low_memory=False),
        pd.read_csv(inputs["aliases"], low_memory=False),
        pd.read_csv(inputs["declared_usgs"], low_memory=False),
    )
    stem = root / DEFAULT_OUT_STEM
    raw_dir = stem.parent / f"{stem.name}_raw"
    frame = propagate_canonical_earthquake_support(
        audit_frame(events, raw_dir, offline=args.offline, refresh=args.refresh)
    )
    write_outputs(
        frame,
        stem.with_suffix(".csv"),
        stem.with_suffix(".json"),
        stem.with_suffix(".md"),
        inputs,
    )


if __name__ == "__main__":
    main()
