#!/usr/bin/env python3
"""Summarize UGLC by native inventory without treating it as homogeneous."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


POINT_COLUMNS = """
    \"OLD DATASET\" AS native_dataset,
    'point' AS geometry_kind,
    COUNTRY AS country,
    try_cast(ACCURACY AS DOUBLE) AS accuracy,
    try_cast(\"START DATE\" AS DATE) AS start_date,
    try_cast(\"END DATE\" AS DATE) AS end_date,
    TYPE AS landslide_type,
    \"PHYSICAL FACTORS\" AS trigger_text,
    try_cast(RELIABILITY AS DOUBLE) AS reliability,
    \"RECORD TYPE\" AS record_type
"""

POLY_COLUMNS = """
    \"OLD DATASET\" AS native_dataset,
    'polygon' AS geometry_kind,
    COUNTRY AS country,
    try_cast(ACCURACY AS DOUBLE) AS accuracy,
    try_cast(\"START DATE\" AS DATE) AS start_date,
    try_cast(\"END DATE\" AS DATE) AS end_date,
    TYPE AS landslide_type,
    concat_ws(';', nullif(TRIGGER, ''), nullif(\"PHYSICAL FACTORS\", '')) AS trigger_text,
    try_cast(RELIABILITY AS DOUBLE) AS reliability,
    \"RECORD TYPE\" AS record_type
"""


def source_sql(path: Path, columns: str) -> str:
    escaped = str(path).replace("'", "''")
    return (
        f"SELECT {columns} FROM read_csv('{escaped}', delim='|', header=true, "
        "sample_size=100000, all_varchar=true)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root / "metadata/pild_xdomain_v1").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    point_path = root / "data_raw/10_UGLC/UGLC_point.csv"
    poly_path = root / "data_raw/10_UGLC/UGLC_poly.csv"
    if not point_path.is_file() or not poly_path.is_file():
        raise SystemExit("UGLC point/poly CSV files are incomplete")

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=12")
    connection.execute("PRAGMA memory_limit='24GB'")
    union_sql = f"{source_sql(point_path, POINT_COLUMNS)} UNION ALL {source_sql(poly_path, POLY_COLUMNS)}"
    summary = connection.execute(
        f"""
        WITH records AS ({union_sql})
        SELECT
            native_dataset,
            geometry_kind,
            count(*) AS n_records,
            count(DISTINCT country) AS n_countries,
            string_agg(DISTINCT country, ';' ORDER BY country) AS countries,
            min(start_date) AS date_start,
            max(end_date) AS date_end,
            avg(CASE WHEN start_date = end_date AND start_date > DATE '1678-01-01' THEN 1.0 ELSE 0.0 END)
                AS exact_date_fraction,
            avg(CASE WHEN start_date > DATE '1678-01-01' AND end_date < DATE '2023-12-31' THEN 1.0 ELSE 0.0 END)
                AS bounded_date_fraction,
            median(accuracy) AS accuracy_median_m,
            quantile_cont(accuracy, 0.9) AS accuracy_p90_m,
            median(reliability) AS reliability_median,
            quantile_cont(reliability, 0.1) AS reliability_p10,
            string_agg(DISTINCT record_type, ';' ORDER BY record_type) AS record_types,
            string_agg(DISTINCT landslide_type, ';' ORDER BY landslide_type) AS landslide_types,
            string_agg(DISTINCT trigger_text, ';' ORDER BY trigger_text) AS trigger_terms
        FROM records
        GROUP BY native_dataset, geometry_kind
        ORDER BY geometry_kind, n_records DESC
        """
    ).df()
    for field in ("exact_date_fraction", "bounded_date_fraction", "reliability_median", "accuracy_median_m"):
        summary[field] = pd.to_numeric(summary[field], errors="coerce")
    trigger_known = ~summary["trigger_terms"].fillna("").str.lower().isin(["", "nd"])
    summary["native_event_source_candidate"] = (
        summary["geometry_kind"].eq("polygon")
        & summary["n_records"].ge(10)
        & summary["bounded_date_fraction"].ge(0.5)
        & summary["reliability_median"].le(4)
        & summary["accuracy_median_m"].le(1000)
        & summary["record_types"].fillna("").str.contains("event", case=False)
        & trigger_known
    ).astype(int)
    existing_tokens = "usgs|nasa|sen12|dlr|gld4cd|glad4cd"
    summary["probable_existing_source_overlap"] = (
        summary["native_dataset"].fillna("").str.lower().str.contains(existing_tokens, regex=True)
    ).astype(int)
    summary.to_csv(out_dir / "uglc_native_dataset_summary_v1.csv", index=False)
    candidates = summary[summary["native_event_source_candidate"].eq(1)].copy()
    candidates.to_csv(out_dir / "uglc_native_event_sources_v1.csv", index=False)
    print(
        f"UGLC summaries={len(summary)}, native event sources={len(candidates)}, "
        f"candidate records={int(candidates['n_records'].sum()) if len(candidates) else 0}"
    )
    if len(candidates):
        print(
            candidates[
                [
                    "native_dataset", "n_records", "countries", "date_start", "date_end",
                    "exact_date_fraction", "accuracy_median_m", "reliability_median", "trigger_terms",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
