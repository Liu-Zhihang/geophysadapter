#!/usr/bin/env python3
"""Extract conservative event groups and AOIs from selected UGLC sources."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import duckdb
import pandas as pd


SOURCES = (
    "ETGFI - Earthquake Triggered Ground Failure Inventories (USGS)",
    "Cooperative Open Online Landslide Repository (NASA) - report and event polygons",
    "Philippines inventories of landslides triggered by the 2019 Cotabato - Davao del Sur seismic sequence",
    "Haiti Landslide Inventories",
    "Malesian Earthquake induced landslides",
    "Asia summer moonsoon triggered landslides",
)


def slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def trigger_family(text: object) -> str:
    value = str(text).lower()
    if "seismic" in value or "earthquake" in value:
        return "earthquake"
    if any(token in value for token in ("rain", "climate", "storm", "cyclone", "hurricane")):
        return "rainfall"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-polygons", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root / "physics_informed_landslide_dataset/metadata/pild_xdomain_v1").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = root / "data_raw/10_UGLC/UGLC_poly.csv"

    source_literals = ",".join("'" + value.replace("'", "''") + "'" for value in SOURCES)
    connection = duckdb.connect()
    connection.execute("INSTALL spatial")
    connection.execute("LOAD spatial")
    connection.execute("PRAGMA threads=12")
    connection.execute("PRAGMA memory_limit='32GB'")
    escaped = str(source_path).replace("'", "''")
    query = f"""
    WITH selected AS (
        SELECT
            \"OLD DATASET\" AS native_dataset,
            COUNTRY AS country,
            try_cast(\"START DATE\" AS DATE) AS start_date,
            try_cast(\"END DATE\" AS DATE) AS end_date,
            concat_ws(';', nullif(TRIGGER, ''), nullif(\"PHYSICAL FACTORS\", '')) AS trigger_text,
            try_cast(RELIABILITY AS DOUBLE) AS reliability,
            try_cast(ACCURACY AS DOUBLE) AS accuracy,
            \"RECORD TYPE\" AS record_type,
            ST_GeomFromText(WKT_GEOM) AS geometry
        FROM read_csv('{escaped}', delim='|', header=true, all_varchar=true)
        WHERE \"OLD DATASET\" IN ({source_literals})
    ), grouped AS (
        SELECT
            *,
            CASE
                WHEN native_dataset LIKE 'ETGFI%' THEN concat(native_dataset, '|', cast(start_date AS VARCHAR))
                WHEN native_dataset LIKE 'Cooperative Open Online%' AND start_date = end_date
                    THEN concat(native_dataset, '|', cast(start_date AS VARCHAR))
                ELSE native_dataset
            END AS event_key
        FROM selected
    )
    SELECT
        event_key,
        native_dataset,
        count(*) AS n_polygons,
        count(DISTINCT country) AS n_countries,
        string_agg(DISTINCT country, ';' ORDER BY country) AS countries,
        min(start_date) AS event_date_start,
        max(end_date) AS event_date_end,
        avg(CASE WHEN start_date = end_date AND start_date > DATE '1678-01-01' THEN 1.0 ELSE 0.0 END)
            AS exact_date_fraction,
        median(reliability) AS reliability_median,
        median(accuracy) AS accuracy_median_m,
        string_agg(DISTINCT trigger_text, ';' ORDER BY trigger_text) AS trigger_terms,
        string_agg(DISTINCT record_type, ';' ORDER BY record_type) AS record_types,
        avg(CASE WHEN ST_IsValid(geometry) THEN 1.0 ELSE 0.0 END) AS valid_geometry_fraction,
        ST_XMin(ST_Extent_Agg(geometry)) AS min_lon,
        ST_YMin(ST_Extent_Agg(geometry)) AS min_lat,
        ST_XMax(ST_Extent_Agg(geometry)) AS max_lon,
        ST_YMax(ST_Extent_Agg(geometry)) AS max_lat
    FROM grouped
    GROUP BY event_key, native_dataset
    ORDER BY event_date_start, native_dataset
    """
    groups = connection.execute(query).df()
    groups["record_id"] = groups["event_key"].map(
        lambda value: "UGLC_" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:14]
    )
    groups["trigger_family"] = groups["trigger_terms"].map(trigger_family)
    groups["date_quality"] = "coarse_native_range"
    groups.loc[groups["exact_date_fraction"].ge(0.99), "date_quality"] = "exact_native_event_date"
    groups.loc[groups["native_dataset"].eq("Haiti Landslide Inventories"), "date_quality"] = (
        "known_event_start_with_coarse_mapping_end"
    )
    groups.loc[
        groups["native_dataset"].str.startswith("Philippines inventories", na=False), "date_quality"
    ] = "known_earthquake_sequence_window"
    groups["center_lon"] = (groups["min_lon"] + groups["max_lon"]) / 2
    groups["center_lat"] = (groups["min_lat"] + groups["max_lat"]) / 2
    start = pd.to_datetime(groups["event_date_start"], errors="coerce")
    modern_sensor = start.ge(pd.Timestamp("2015-06-23"))
    date_usable = groups["date_quality"].ne("coarse_native_range")
    groups["sentinel2_event_candidate"] = (
        groups["n_polygons"].ge(args.min_polygons)
        & modern_sensor
        & date_usable
        & groups["valid_geometry_fraction"].ge(0.99)
        & groups["trigger_family"].ne("unknown")
    ).astype(int)
    groups["requires_imagery_acquisition"] = groups["sentinel2_event_candidate"]
    groups["source_id"] = "UGLC"
    groups["source_record_id"] = groups["event_key"]
    groups["geographic_region_id"] = groups.apply(
        lambda row: f"UGLC_REGION_{slug(row['countries'])}_{row['record_id'][-6:]}", axis=1
    )
    groups["trigger_event_id"] = groups.apply(
        lambda row: f"TRG_{row['trigger_family']}_{str(row['event_date_start'])[:10]}_{row['record_id'][-8:]}", axis=1
    )
    groups.to_csv(out_dir / "uglc_event_groups_v1.csv", index=False)
    selected = groups[groups["sentinel2_event_candidate"].eq(1)].copy()
    selected.to_csv(out_dir / "uglc_sentinel2_event_candidates_v1.csv", index=False)
    print(
        f"UGLC event groups={len(groups)}, Sentinel-2 candidates={len(selected)}, "
        f"candidate polygons={int(selected['n_polygons'].sum()) if len(selected) else 0}"
    )
    if len(selected):
        print(
            selected[
                [
                    "record_id", "event_date_start", "trigger_family", "countries", "n_polygons",
                    "min_lon", "min_lat", "max_lon", "max_lat", "native_dataset",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
