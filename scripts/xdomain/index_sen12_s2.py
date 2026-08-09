#!/usr/bin/env python3
"""Index Sen12Landslides S2 NetCDF samples and their event/geospatial metadata."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


def parse_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_mappings(value: object) -> list[dict[str, int]]:
    """Parse the upstream comma-joined sequence of pre/post dictionaries."""
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(f"[{text}]")
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid pre_post_dates={value!r}") from error
    mappings = []
    for item in parsed:
        if not isinstance(item, dict) or not {"pre", "post"} <= set(item):
            raise ValueError(f"Invalid pre/post item in {value!r}")
        mappings.append({"pre": int(item["pre"]), "post": int(item["post"])})
    return mappings


def split_values(value: object, *, drop_none: bool = False) -> list[str]:
    values = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if drop_none:
        values = [item for item in values if item.lower() not in {"none", "nan", "nat"}]
    return values


def split_ids(value: object) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def index_one(path: Path, root: Path) -> dict[str, object]:
    with netCDF4.Dataset(path, "r") as dataset:
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        variables = sorted(dataset.variables)
        dimensions = {name: len(value) for name, value in dataset.dimensions.items()}
        crs_text = str(attrs.get("crs", ""))
        crs = CRS.from_user_input(crs_text)
        x = np.asarray(dataset.variables["x"][:], dtype=np.float64)
        y = np.asarray(dataset.variables["y"][:], dtype=np.float64)
        min_x, max_x = float(np.nanmin(x)), float(np.nanmax(x))
        min_y, max_y = float(np.nanmin(y)), float(np.nanmax(y))
        transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        corners = [
            transformer.transform(min_x, min_y),
            transformer.transform(min_x, max_y),
            transformer.transform(max_x, min_y),
            transformer.transform(max_x, max_y),
        ]
        lons, lats = zip(*corners)
        mappings = parse_mappings(attrs.get("pre_post_dates", ""))
        unique_pairs = sorted({(item["pre"], item["post"]) for item in mappings})
        pre_index = min((item["pre"] for item in mappings), default=-1)
        post_index = max((item["post"] for item in mappings), default=-1)
        mask = np.asarray(dataset.variables["MASK"][:])
        mask_first = mask[0] > 0 if mask.ndim == 3 else mask > 0
        mask_union = np.any(mask > 0, axis=0) if mask.ndim == 3 else mask_first
        dem = np.asarray(dataset.variables["DEM"][0] if dataset.variables["DEM"].ndim == 3 else dataset.variables["DEM"][:])
        scl = dataset.variables.get("SCL")
        post_cloud_fraction = math.nan
        if scl is not None and 0 <= post_index < scl.shape[0]:
            post_scl = np.asarray(scl[post_index])
            post_cloud_fraction = float(np.isin(post_scl, [3, 8, 9, 10, 11]).mean())
        times = np.asarray(dataset.variables["time"][:])
        time_units = getattr(dataset.variables["time"], "units", "")
        time_calendar = getattr(dataset.variables["time"], "calendar", "standard")
        decoded_times = netCDF4.num2date(times, units=time_units, calendar=time_calendar)
        date_strings = [value.strftime("%Y-%m-%d") for value in decoded_times]

    stem = path.stem
    region = stem.split("_s2_", 1)[0]
    patch_id = stem.replace("_s2_", "_", 1)
    annotated = parse_bool(attrs.get("annotated", False))
    event_date_values = split_values(attrs.get("event_date", ""), drop_none=True)
    event_dates = sorted(
        {
            value.strftime("%Y-%m-%d")
            for value in pd.to_datetime(event_date_values, errors="coerce")
            if pd.notna(value)
        }
    )
    confidence_values = [
        float(value) for value in split_values(attrs.get("date_confidence", ""))
    ]
    date_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
    event_date = event_dates[0] if len(event_dates) == 1 else ""
    if not annotated:
        date_quality = "unannotated"
    elif len(event_dates) == 1 and date_confidence >= 0.95:
        date_quality = "high_single_event"
    elif len(event_dates) == 1:
        date_quality = "estimated_single_event"
    elif len(event_dates) > 1:
        date_quality = "multi_event_mixed"
    else:
        date_quality = "undated_annotated"
    event_span_bracketed = bool(
        event_dates
        and 0 <= pre_index < len(date_strings)
        and 0 <= post_index < len(date_strings)
        and pd.Timestamp(date_strings[pre_index]) < pd.Timestamp(event_dates[0])
        and pd.Timestamp(date_strings[post_index]) >= pd.Timestamp(event_dates[-1])
    )
    change_view_eligible = bool(
        annotated and event_span_bracketed and pre_index < post_index
    )
    if change_view_eligible:
        change_view_exclusion_reason = ""
    elif not annotated:
        change_view_exclusion_reason = "unannotated"
    elif not event_dates:
        change_view_exclusion_reason = "missing_event_date"
    elif pre_index >= post_index:
        change_view_exclusion_reason = "no_distinct_pre_post_pair"
    else:
        change_view_exclusion_reason = "event_span_not_bracketed"
    if not annotated:
        event_group = f"SEN12_UNLABELED_REGION_{region}"
    elif date_quality == "high_single_event":
        event_group = f"SEN12_EVENT_{region}_{event_date.replace('-', '')}"
    else:
        event_group = f"SEN12_REGION_{region}_LOWCONF"
    annotation_ids = split_ids(attrs.get("ann_id", ""))
    return {
        "sample_id": f"SEN12_S2_{patch_id}",
        "patch_id": patch_id,
        "source_id": "SEN12LS_HARMONIZED",
        "relative_path": str(path.relative_to(root)),
        "filename": path.name,
        "region": region,
        "physical_event_group": event_group,
        "event_date": event_date,
        "event_dates": ";".join(event_dates),
        "event_date_start": event_dates[0] if event_dates else "",
        "event_date_end": event_dates[-1] if event_dates else "",
        "n_event_dates": len(event_dates),
        "date_confidence": date_confidence,
        "date_confidence_values": ";".join(map(str, confidence_values)),
        "date_quality": date_quality,
        "annotated": int(annotated),
        "annotation_ids": ";".join(annotation_ids),
        "n_annotation_ids": len(annotation_ids),
        "pre_index": pre_index,
        "post_index": post_index,
        "pre_indices": ";".join(str(item["pre"]) for item in mappings),
        "post_indices": ";".join(str(item["post"]) for item in mappings),
        "n_pre_post_pairs": len(mappings),
        "time_selection_contract": (
            "single_pair"
            if len(unique_pairs) == 1
            else "event_span_pair"
        ),
        "event_span_bracketed": int(event_span_bracketed),
        "change_view_eligible": int(change_view_eligible),
        "change_view_exclusion_reason": change_view_exclusion_reason,
        "pre_date": date_strings[pre_index] if 0 <= pre_index < len(date_strings) else "",
        "post_date": date_strings[post_index] if 0 <= post_index < len(date_strings) else "",
        "n_times": dimensions.get("time", 0),
        "height": dimensions.get("y", dimensions.get("x", 0)),
        "width": dimensions.get("x", dimensions.get("y", 0)),
        "variables": ";".join(variables),
        "crs": crs.to_string(),
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "min_lon": min(lons),
        "min_lat": min(lats),
        "max_lon": max(lons),
        "max_lat": max(lats),
        "center_lon": (min(lons) + max(lons)) / 2,
        "center_lat": (min(lats) + max(lats)) / 2,
        "positive_pixels": int(mask_first.sum()),
        "positive_fraction": float(mask_first.mean()),
        "positive_pixels_union": int(mask_union.sum()),
        "dem_finite_fraction": float(np.isfinite(dem).mean()),
        "dem_std_m": float(np.nanstd(dem)),
        "post_cloud_fraction": post_cloud_fraction,
        "terrain_source": "Copernicus_DEM_resampled_10m",
        "terrain_native_resolution_m": 30,
        "trigger_context_ready": int(date_quality == "high_single_event"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = root / "data_raw/08_Sen12Landslides/extracted"
    paths = sorted(source_root.rglob("*.nc"))
    if len(paths) != 13628:
        raise SystemExit(f"Expected 13628 Sen12 S2 files, found {len(paths)}")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(index_one, path, root): path for path in paths}
        for position, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            path = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(f"{path}: {error}")
            if position % 500 == 0:
                print(f"[INDEX] {position}/{len(paths)} failures={len(failures)}", flush=True)
    out_dir = root / "metadata/pild_xdomain_v1"
    if failures:
        (out_dir / "sen12_s2_index_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
        raise SystemExit(f"Failed to index {len(failures)} files")
    frame = pd.DataFrame(rows).sort_values("sample_id")
    candidate_path = out_dir / "candidate_event_registry_v1.csv"
    if not candidate_path.is_file():
        raise SystemExit("Cross-source candidate event registry is missing")
    candidates = pd.read_csv(candidate_path)
    candidates = candidates[candidates["source_id"] == "SEN12LS_HARMONIZED"].copy()
    candidates["join_key"] = candidates["source_record_id"].astype(str).str.lower()
    if candidates["join_key"].duplicated().any():
        raise SystemExit("Duplicate Sen12 source_record_id values in candidate event registry")
    event_map = candidates.set_index("join_key")[
        ["physical_event_cluster_id", "geographic_region_id"]
    ]
    frame["event_join_key"] = frame.apply(
        lambda row: (
            f"{row['region']}|{row['event_date']}"
            if row["date_quality"] == "high_single_event"
            else f"{row['region']}|low_confidence_region"
        ).lower(),
        axis=1,
    )
    frame = frame.join(event_map, on="event_join_key")
    missing_join = frame["physical_event_cluster_id"].isna()
    missing_high_confidence = missing_join & frame["date_quality"].eq("high_single_event")
    if missing_high_confidence.any():
        missing = sorted(frame.loc[missing_high_confidence, "event_join_key"].unique())
        raise SystemExit(f"High-confidence Sen12 event-cluster join is incomplete: {missing[:20]}")
    if missing_join.any():
        for region in sorted(frame.loc[missing_join, "region"].unique()):
            region_mask = missing_join & frame["region"].eq(region)
            token = hashlib.sha1(region.encode("utf-8")).hexdigest()[:14]
            annotated_region = region_mask & frame["annotated"].eq(1)
            unannotated_region = region_mask & frame["annotated"].eq(0)
            frame.loc[annotated_region, "physical_event_cluster_id"] = f"XREG_{token}"
            frame.loc[annotated_region, "geographic_region_id"] = f"SEN12_REGION_{region}"
            frame.loc[unannotated_region, "physical_event_cluster_id"] = f"XBG_{token}"
            frame.loc[unannotated_region, "geographic_region_id"] = f"SEN12_UNLABELED_{region}"
    frame.to_csv(out_dir / "sen12_s2_sample_registry_v1.csv", index=False)
    (out_dir / "sen12_s2_index_failures.json").unlink(missing_ok=True)
    event_summary = (
        frame.groupby(
            [
                "physical_event_cluster_id",
                "physical_event_group",
                "geographic_region_id",
                "region",
                "date_quality",
                "event_date",
            ],
            dropna=False,
        )
        .agg(
            n_samples=("sample_id", "size"),
            n_annotated=("annotated", "sum"),
            n_positive=("positive_pixels", lambda value: int((value > 0).sum())),
            positive_pixels=("positive_pixels", "sum"),
            min_lon=("min_lon", "min"),
            min_lat=("min_lat", "min"),
            max_lon=("max_lon", "max"),
            max_lat=("max_lat", "max"),
            cloud_fraction_median=("post_cloud_fraction", "median"),
            dem_std_median=("dem_std_m", "median"),
        )
        .reset_index()
    )
    event_summary.to_csv(out_dir / "sen12_s2_event_summary_v1.csv", index=False)
    print(
        f"[DONE] samples={len(frame)}, event_groups={len(event_summary)}, "
        f"regions={frame['region'].nunique()}, annotated={int(frame['annotated'].sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
