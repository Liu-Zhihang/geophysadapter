#!/usr/bin/env python3
"""Build an event-level PILD-XDomain registry without inflating patch power."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio


def slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "unknown"


def joined(values: pd.Series) -> str:
    return ";".join(sorted({str(value).strip() for value in values.dropna() if str(value).strip()}))


def iso_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def valid_bounds(frame) -> tuple[float, float, float, float]:
    bounds = frame.total_bounds
    if len(bounds) != 4 or not np.isfinite(bounds).all():
        return (math.nan, math.nan, math.nan, math.nan)
    return tuple(float(value) for value in bounds)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if not all(math.isfinite(value) for value in (lon1, lat1, lon2, lat2)):
        return math.inf
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def canonical_trigger(value: object) -> str:
    text = str(value).lower()
    if any(token in text for token in ("rain", "downpour", "storm", "cyclone", "hurricane", "typhoon")):
        return "rainfall"
    if any(token in text for token in ("earthquake", "seismic", "seismogenic")):
        return "earthquake"
    if "volcan" in text:
        return "volcanic"
    return "unknown"


def trigger_event_id(trigger: str, date: str, name: str) -> str:
    storm_match = re.search(r"(maria|freddy|fiona|mangkhut|nalgae|doksuri)", name.lower())
    event_name = storm_match.group(1) if storm_match else slug(name)[:48]
    year = date[:4] if date else "unknown"
    return f"TRG_{trigger}_{year}_{event_name}"


def read_existing_pild(root: Path) -> pd.DataFrame:
    registry_path = root / "metadata/pild_core_v2/event_registry_v2.csv"
    master_path = root / "data_external/event_master.csv"
    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    master = pd.read_csv(master_path, dtype=str).fillna("").set_index("event_uid", drop=False)
    rows: list[dict[str, object]] = []
    for item in registry.to_dict("records"):
        event_uids = [value for value in item["event_uids"].split(";") if value]
        matched = master.loc[[value for value in event_uids if value in master.index]] if event_uids else master.iloc[0:0]
        numeric = {}
        for field in ("min_lon", "min_lat", "max_lon", "max_lat"):
            values = pd.to_numeric(matched[field], errors="coerce") if field in matched else pd.Series(dtype=float)
            numeric[field] = float(values.min() if field.startswith("min_") else values.max()) if values.notna().any() else math.nan
        min_lon, min_lat, max_lon, max_lat = (
            numeric["min_lon"], numeric["min_lat"], numeric["max_lon"], numeric["max_lat"]
        )
        rows.append(
            {
                "record_id": f"PILD_{item['physical_event_id']}",
                "source_id": "PILD_CORE_V2",
                "source_record_id": item["physical_event_id"],
                "event_name": item["physical_event_id"],
                "event_date_start": item["canonical_date"],
                "event_date_end": item["canonical_date"],
                "date_quality": "existing_registry",
                "trigger_family": item["physical_trigger_family"] or "unknown",
                "trigger_event_id": trigger_event_id(
                    item["physical_trigger_family"] or "unknown", item["canonical_date"], item["physical_event_id"]
                ),
                "geographic_region_id": f"PILD_{slug(item['physical_event_id'])}",
                "n_labels": int(float(item["n_supervised_samples_qc"] or 0)),
                "label_geometry_type": "standardized_window",
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "center_lon": (min_lon + max_lon) / 2 if math.isfinite(min_lon + max_lon) else math.nan,
                "center_lat": (min_lat + max_lat) / 2 if math.isfinite(min_lat + max_lat) else math.nan,
                "source_confidence": 1.0,
                "label_ready": int(item["in_supervised_core"] == "1"),
                "imagery_ready": int(item["in_supervised_core"] == "1"),
                "segmentation_ready": int(item["in_supervised_core"] == "1"),
                "trigger_context_ready": int(
                    bool(iso_date(item["canonical_date"]))
                    and (item["physical_trigger_family"] or "unknown") != "unknown"
                ),
                "inventory_role": "existing_standardized_core",
                "notes": f"event_uids={item['event_uids']}",
            }
        )
    return pd.DataFrame(rows)


def read_sen12(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "data_raw/08_Sen12Landslides/inventories/inventories.shp"
    data = pyogrio.read_dataframe(path)
    for field in ("pre_date", "event_date", "post_date"):
        data[field] = pd.to_datetime(data[field], errors="coerce", utc=True)
    data["high_date_confidence"] = data["event_conf"].fillna(0).ge(0.95)
    summaries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for location, group in data.groupby("location", dropna=False):
        min_lon, min_lat, max_lon, max_lat = valid_bounds(group)
        high = group[group["high_date_confidence"] & group["event_date"].notna()]
        low = group.drop(high.index)
        summaries.append(
            {
                "location": location,
                "n_polygons": len(group),
                "n_high_confidence_polygons": len(high),
                "n_low_confidence_polygons": len(low),
                "n_distinct_dates_raw": group["event_date"].nunique(),
                "event_confidence_mean": float(group["event_conf"].mean()),
                "event_confidence_median": float(group["event_conf"].median()),
                "event_types": joined(group["event_type"]),
                "date_min": iso_date(group["event_date"].min()),
                "date_max": iso_date(group["event_date"].max()),
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            }
        )
        for date, event_group in high.groupby("event_date"):
            b0, b1, b2, b3 = valid_bounds(event_group)
            date_text = iso_date(date)
            name = f"Sen12Landslides {location} {date_text}"
            trigger = canonical_trigger(joined(event_group["event_type"]))
            rows.append(
                {
                    "record_id": f"SEN12_{slug(location)}_{date_text.replace('-', '')}",
                    "source_id": "SEN12LS_HARMONIZED",
                    "source_record_id": f"{location}|{date_text}",
                    "event_name": name,
                    "event_date_start": date_text,
                    "event_date_end": date_text,
                    "date_quality": "inventory_event_confidence_ge_0.95",
                    "trigger_family": trigger,
                    "trigger_event_id": trigger_event_id(trigger, date_text, str(location)),
                    "geographic_region_id": f"SEN12_REGION_{slug(location)}",
                    "n_labels": len(event_group),
                    "label_geometry_type": "polygon",
                    "min_lon": b0,
                    "min_lat": b1,
                    "max_lon": b2,
                    "max_lat": b3,
                    "center_lon": (b0 + b2) / 2,
                    "center_lat": (b1 + b3) / 2,
                    "source_confidence": float(event_group["event_conf"].median()),
                    "label_ready": 1,
                    "imagery_ready": 0,
                    "segmentation_ready": 0,
                    "trigger_context_ready": int(trigger != "unknown"),
                    "inventory_role": "high_confidence_event_candidate",
                    "notes": f"polygon_ids={event_group['id'].min()}..{event_group['id'].max()}",
                }
            )
        if len(low):
            b0, b1, b2, b3 = valid_bounds(low)
            rows.append(
                {
                    "record_id": f"SEN12_{slug(location)}_LOWCONF_REGION",
                    "source_id": "SEN12LS_HARMONIZED",
                    "source_record_id": f"{location}|low_confidence_region",
                    "event_name": f"Sen12Landslides {location} low-confidence inventory region",
                    "event_date_start": iso_date(low["event_date"].min()),
                    "event_date_end": iso_date(low["event_date"].max()),
                    "date_quality": "estimated_not_independent_event",
                    "trigger_family": canonical_trigger(joined(low["event_type"])),
                    "trigger_event_id": f"TRG_UNRESOLVED_{slug(location)}",
                    "geographic_region_id": f"SEN12_REGION_{slug(location)}",
                    "n_labels": len(low),
                    "label_geometry_type": "polygon",
                    "min_lon": b0,
                    "min_lat": b1,
                    "max_lon": b2,
                    "max_lat": b3,
                    "center_lon": (b0 + b2) / 2,
                    "center_lat": (b1 + b3) / 2,
                    "source_confidence": float(low["event_conf"].median()),
                    "label_ready": 1,
                    "imagery_ready": 0,
                    "segmentation_ready": 0,
                    "trigger_context_ready": 0,
                    "inventory_role": "regional_inventory_not_event_power",
                    "notes": "Estimated dates must not be counted as independent trigger events.",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def read_nasa(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = (
        root
        / "data_raw/09_NASA_COOLR_Rainfall_Events/extracted/nasa_coolr_new_events/"
        "nasa_coolr_new_events.gdb"
    )
    data = pyogrio.read_dataframe(path, layer="nasa_coolr_events_new_point")
    data["ev_date"] = pd.to_datetime(data["ev_date"], errors="coerce", utc=True)
    data["event_key"] = (
        data["ev_title"].map(slug) + "|" + data["ev_date"].map(iso_date) + "|" + data["ctry_name"].map(slug)
    )
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for key, group in data.groupby("event_key"):
        min_lon, min_lat, max_lon, max_lat = valid_bounds(group)
        date_text = iso_date(group["ev_date"].iloc[0])
        title = str(group["ev_title"].iloc[0])
        trigger = canonical_trigger(joined(group["ls_trig"]))
        source_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        row = {
            "record_id": f"NASA_COOLR_{source_id}",
            "source_id": "NASA_COOLR_RAIN22",
            "source_record_id": key,
            "event_name": title,
            "event_date_start": date_text,
            "event_date_end": date_text,
            "date_quality": "reported_event_date",
            "trigger_family": trigger,
            "trigger_event_id": trigger_event_id(trigger, date_text, title),
            "geographic_region_id": f"NASA_{slug(joined(group['ctry_name']))}_{source_id}",
            "n_labels": len(group),
            "label_geometry_type": "point",
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "center_lon": (min_lon + max_lon) / 2,
            "center_lat": (min_lat + max_lat) / 2,
            "source_confidence": 1.0,
            "label_ready": 1,
            "imagery_ready": 0,
            "segmentation_ready": 0,
            "trigger_context_ready": 1,
            "inventory_role": "event_trigger_and_imagery_acquisition_candidate",
            "notes": (
                f"countries={joined(group['ctry_name'])}; methods={joined(group['method'])}; "
                f"image_types={joined(group['img_type'])}; pre={joined(group['sat_date_b'])}; "
                f"post={joined(group['sat_date_a'])}"
            ),
        }
        rows.append(row)
        summaries.append({**row, "point_ids": f"{group['ev_id'].min()}..{group['ev_id'].max()}"})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def read_usgs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "data_raw/07_USGS_Inventory_v3/extracted/US_Landslide_v3_gpkg/US_Landslide_v3.gpkg"
    data = pyogrio.read_dataframe(path, layer="us_ls_v3_poly")
    data["Date_Min"] = pd.to_datetime(data["Date_Min"], errors="coerce", utc=True)
    data["Date_Max"] = pd.to_datetime(data["Date_Max"], errors="coerce", utc=True)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for inventory, group in data.groupby("Inventory", dropna=False):
        min_lon, min_lat, max_lon, max_lat = valid_bounds(group)
        date_start = group["Date_Min"].min()
        date_end = group["Date_Max"].max()
        date_fraction = float((group["Date_Min"].notna() & group["Date_Max"].notna()).mean())
        span_days = (date_end - date_start).days if pd.notna(date_start) and pd.notna(date_end) else math.inf
        confidence = float(group["Confidence"].median())
        name = str(inventory)
        explicit_event_name = bool(
            re.search(
                r"\b(jan|july|nov|dec|hurricane|montecito|walpert|crow creek|front range|east sf bay)\b",
                name.lower(),
            )
        )
        bounded_event = date_fraction >= 0.8 and span_days <= 120
        single_date_event = explicit_event_name and pd.notna(date_start) and pd.isna(date_end)
        event_like = bool((bounded_event or single_date_event) and confidence >= 5 and len(group) >= 10)
        if single_date_event:
            date_end = date_start
        date_text = iso_date(date_start)
        trigger = canonical_trigger(name)
        record_id = f"USGS_{slug(name)}"
        row = {
            "record_id": record_id,
            "source_id": "USGS_NLSI_V3",
            "source_record_id": name,
            "event_name": name,
            "event_date_start": date_text,
            "event_date_end": iso_date(date_end),
            "date_quality": "bounded_inventory_window" if event_like else "heterogeneous_or_missing",
            "trigger_family": trigger,
            "trigger_event_id": trigger_event_id(trigger, date_text, name),
            "geographic_region_id": f"USGS_REGION_{slug(name)}",
            "n_labels": len(group),
            "label_geometry_type": "multipolygon",
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "center_lon": (min_lon + max_lon) / 2,
            "center_lat": (min_lat + max_lat) / 2,
            "source_confidence": confidence / 8.0,
            "label_ready": 1,
            "imagery_ready": 0,
            "segmentation_ready": 0,
            "trigger_context_ready": int(event_like and bool(date_text)),
            "inventory_role": "acute_event_imagery_candidate" if event_like else "context_inventory_only",
            "notes": (
                f"date_fraction={date_fraction:.3f}; span_days={span_days}; "
                f"confidence_median={confidence:.1f}; types={joined(group['LS_Type'])}"
            ),
        }
        rows.append(row)
        summaries.append(row)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def read_uglc(root: Path) -> pd.DataFrame:
    path = root / "metadata/pild_xdomain_v1/uglc_event_groups_v1.csv"
    if not path.is_file():
        return pd.DataFrame()
    data = pd.read_csv(path)
    data = data[data["sentinel2_event_candidate"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    for item in data.to_dict("records"):
        reliability = float(item.get("reliability_median", 10))
        date_start = iso_date(item.get("event_date_start", ""))
        date_end = iso_date(item.get("event_date_end", ""))
        rows.append(
            {
                "record_id": item["record_id"],
                "source_id": "UGLC",
                "source_record_id": item["event_key"],
                "event_name": f"{item['native_dataset']} {date_start}",
                "event_date_start": date_start,
                "event_date_end": date_end,
                "date_quality": item["date_quality"],
                "trigger_family": item["trigger_family"],
                "trigger_event_id": item["trigger_event_id"],
                "geographic_region_id": item["geographic_region_id"],
                "n_labels": int(item["n_polygons"]),
                "label_geometry_type": "polygon",
                "min_lon": float(item["min_lon"]),
                "min_lat": float(item["min_lat"]),
                "max_lon": float(item["max_lon"]),
                "max_lat": float(item["max_lat"]),
                "center_lon": float(item["center_lon"]),
                "center_lat": float(item["center_lat"]),
                "source_confidence": max(0.0, min(1.0, 1.0 - (reliability - 1.0) / 9.0)),
                "label_ready": 1,
                "imagery_ready": 0,
                "segmentation_ready": 0,
                "trigger_context_ready": int(item["trigger_family"] != "unknown"),
                "inventory_role": "acute_event_imagery_candidate",
                "notes": f"native_dataset={item['native_dataset']}; reliability_class={reliability}",
            }
        )
    return pd.DataFrame(rows)


def mark_probable_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["overlap_status"] = "not_evaluated"
    result["overlap_with_record_id"] = ""
    result["overlap_distance_km"] = np.nan
    result["overlap_date_days"] = np.nan
    existing = result[result["source_id"] == "PILD_CORE_V2"]
    for index, row in result[result["source_id"] != "PILD_CORE_V2"].iterrows():
        row_date = pd.to_datetime(row["event_date_start"], errors="coerce", utc=True)
        best: tuple[float, float, str] | None = None
        for _, candidate in existing.iterrows():
            candidate_date = pd.to_datetime(candidate["event_date_start"], errors="coerce", utc=True)
            date_days = abs((row_date - candidate_date).days) if pd.notna(row_date) and pd.notna(candidate_date) else math.inf
            distance = haversine_km(
                float(row["center_lon"]), float(row["center_lat"]),
                float(candidate["center_lon"]), float(candidate["center_lat"]),
            )
            score = distance + min(date_days, 3650)
            if best is None or score < best[0] + best[1]:
                best = (distance, float(date_days), str(candidate["record_id"]))
        if best is None:
            continue
        distance, date_days, record_id = best
        result.at[index, "overlap_distance_km"] = distance
        result.at[index, "overlap_date_days"] = date_days if math.isfinite(date_days) else np.nan
        result.at[index, "overlap_with_record_id"] = record_id
        if distance <= 75 and date_days <= 30:
            result.at[index, "overlap_status"] = "probable_existing_overlap"
        elif distance <= 150 and date_days <= 90:
            result.at[index, "overlap_status"] = "manual_review"
        else:
            result.at[index, "overlap_status"] = "no_close_existing_match"
    result.loc[result["source_id"] == "PILD_CORE_V2", "overlap_status"] = "existing_reference"
    return result


class UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def boxes_intersect(left: pd.Series, right: pd.Series) -> bool:
    values = [left.get(key) for key in ("min_lon", "min_lat", "max_lon", "max_lat")]
    values += [right.get(key) for key in ("min_lon", "min_lat", "max_lon", "max_lat")]
    try:
        parsed = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in parsed):
        return False
    l0, l1, l2, l3, r0, r1, r2, r3 = parsed
    return not (l2 < r0 or r2 < l0 or l3 < r1 or r3 < l1)


def assign_event_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    keys = result["record_id"].astype(str).tolist()
    union_find = UnionFind(keys)
    eligible = result[
        result["inventory_role"].isin(
            [
                "existing_standardized_core",
                "high_confidence_event_candidate",
                "event_trigger_and_imagery_acquisition_candidate",
                "acute_event_imagery_candidate",
            ]
        )
    ]
    rows = list(eligible.iterrows())
    for position, (_, left) in enumerate(rows):
        left_date = pd.to_datetime(left["event_date_start"], errors="coerce", utc=True)
        if pd.isna(left_date):
            continue
        for _, right in rows[position + 1 :]:
            right_date = pd.to_datetime(right["event_date_start"], errors="coerce", utc=True)
            if pd.isna(right_date) or abs((left_date - right_date).days) > 30:
                continue
            same_source = left["source_id"] == right["source_id"]
            same_named_trigger = (
                str(left["trigger_event_id"]) == str(right["trigger_event_id"])
                and not str(left["trigger_event_id"]).startswith("TRG_UNRESOLVED")
            )
            same_uglc_date = (
                same_source
                and left["source_id"] == "UGLC"
                and left_date.date() == right_date.date()
            )
            if same_source and not same_named_trigger and not same_uglc_date:
                continue
            triggers = {str(left["trigger_family"]), str(right["trigger_family"])} - {"unknown", "nan"}
            if len(triggers) > 1:
                continue
            distance = haversine_km(
                float(left["center_lon"]), float(left["center_lat"]),
                float(right["center_lon"]), float(right["center_lat"]),
            )
            if boxes_intersect(left, right) or distance <= 100:
                union_find.union(str(left["record_id"]), str(right["record_id"]))

    members: dict[str, list[str]] = {}
    for key in keys:
        members.setdefault(union_find.find(key), []).append(key)
    cluster_by_record: dict[str, str] = {}
    cluster_rows: list[dict[str, object]] = []
    indexed = result.set_index("record_id", drop=False)
    for member_ids in sorted(members.values(), key=lambda values: min(values)):
        stable = hashlib.sha256(";".join(sorted(member_ids)).encode("utf-8")).hexdigest()[:14]
        cluster_id = f"XEV_{stable}"
        subset = indexed.loc[member_ids]
        if isinstance(subset, pd.Series):
            subset = subset.to_frame().T
        for record_id in member_ids:
            cluster_by_record[record_id] = cluster_id
        cluster_rows.append(
            {
                "physical_event_cluster_id": cluster_id,
                "n_records": len(member_ids),
                "n_sources": subset["source_id"].nunique(),
                "source_ids": ";".join(sorted(subset["source_id"].astype(str).unique())),
                "record_ids": ";".join(sorted(member_ids)),
                "event_date_start": min(filter(None, subset["event_date_start"].astype(str)), default=""),
                "event_date_end": max(filter(None, subset["event_date_end"].astype(str)), default=""),
                "trigger_families": ";".join(sorted(subset["trigger_family"].astype(str).unique())),
                "n_labels": int(pd.to_numeric(subset["n_labels"], errors="coerce").fillna(0).sum()),
            }
        )
    result["physical_event_cluster_id"] = result["record_id"].map(cluster_by_record)
    return result, pd.DataFrame(cluster_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root / "metadata/pild_xdomain_v1").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = read_existing_pild(root)
    sen12, sen12_summary = read_sen12(root)
    nasa, nasa_summary = read_nasa(root)
    usgs, usgs_summary = read_usgs(root)
    uglc = read_uglc(root)
    registry = mark_probable_overlap(pd.concat([existing, sen12, nasa, usgs, uglc], ignore_index=True))
    registry, clusters = assign_event_clusters(registry)
    registry = registry.sort_values(["source_id", "event_date_start", "record_id"]).reset_index(drop=True)
    registry.to_csv(out_dir / "candidate_event_registry_v1.csv", index=False)
    clusters.sort_values(["n_sources", "n_labels"], ascending=[False, False]).to_csv(
        out_dir / "cross_source_event_clusters_v1.csv", index=False
    )
    sen12_summary.to_csv(out_dir / "sen12_location_summary_v1.csv", index=False)
    nasa_summary.to_csv(out_dir / "nasa_coolr_event_summary_v1.csv", index=False)
    usgs_summary.to_csv(out_dir / "usgs_inventory_summary_v1.csv", index=False)

    source_summary = (
        registry.groupby("source_id", dropna=False)
        .agg(
            n_records=("record_id", "size"),
            n_labels=("n_labels", "sum"),
            n_regions=("geographic_region_id", "nunique"),
            n_trigger_groups=("trigger_event_id", "nunique"),
            n_label_ready=("label_ready", "sum"),
            n_segmentation_ready=("segmentation_ready", "sum"),
            n_trigger_context_ready=("trigger_context_ready", "sum"),
            n_probable_existing_overlap=("overlap_status", lambda x: int((x == "probable_existing_overlap").sum())),
        )
        .reset_index()
    )
    source_summary.to_csv(out_dir / "source_usability_summary_v1.csv", index=False)
    print(source_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
