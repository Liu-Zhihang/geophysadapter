#!/usr/bin/env python3
"""Build and optionally audit the deduplicated PILD Sentinel-2 acquisition plan."""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd


TARGET_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
REQUIRED_ASSETS = set(TARGET_BANDS) | {"SCL"}
CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness",
        type=Path,
        default=root / "processed/hybrid_pinn/pild_prithvi_integration_v1/pild_window_readiness.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "processed/hybrid_pinn/pild_prithvi_integration_v1",
    )
    parser.add_argument("--query-stac", action="store_true")
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--cloud-cover-max", type=float, default=90.0)
    return parser.parse_args()


def bbox_union_coverage(item_bboxes: list[list[float]], target_bbox: list[float]) -> float:
    rectangles = []
    x_edges = {float(target_bbox[0]), float(target_bbox[2])}
    for item_bbox in item_bboxes:
        left = max(float(item_bbox[0]), float(target_bbox[0]))
        bottom = max(float(item_bbox[1]), float(target_bbox[1]))
        right = min(float(item_bbox[2]), float(target_bbox[2]))
        top = min(float(item_bbox[3]), float(target_bbox[3]))
        if right > left and top > bottom:
            rectangles.append((left, bottom, right, top))
            x_edges.update((left, right))
    intersection = 0.0
    x_values = sorted(x_edges)
    for x0, x1 in zip(x_values[:-1], x_values[1:]):
        intervals = sorted(
            (bottom, top)
            for left, bottom, right, top in rectangles
            if left < x1 and right > x0
        )
        covered_y = 0.0
        if intervals:
            current_bottom, current_top = intervals[0]
            for bottom, top in intervals[1:]:
                if bottom > current_top:
                    covered_y += current_top - current_bottom
                    current_bottom, current_top = bottom, top
                else:
                    current_top = max(current_top, top)
            covered_y += current_top - current_bottom
        intersection += (x1 - x0) * covered_y
    target_area = max(1e-12, (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1]))
    return min(1.0, intersection / target_area)


def select_side(
    items: list, event_date: pd.Timestamp, side: str, target_bbox: list[float]
) -> list[dict]:
    by_day: dict[str, list] = {}
    for item in items:
        acquired = pd.Timestamp(item.datetime).tz_localize(None)
        offset = int((acquired.normalize() - event_date.normalize()).days)
        if side == "pre" and not (-180 <= offset <= -7):
            continue
        if side == "post" and not (7 <= offset <= 180):
            continue
        if REQUIRED_ASSETS.issubset(item.assets):
            by_day.setdefault(acquired.date().isoformat(), []).append(item)
    candidates = []
    for day, day_items in by_day.items():
        acquired = pd.Timestamp(day)
        coverage = bbox_union_coverage([list(item.bbox) for item in day_items], target_bbox)
        if coverage < 0.999:
            continue
        clouds = [float(item.properties.get("eo:cloud_cover", 100.0)) for item in day_items]
        candidates.append({
            "item_ids": ";".join(sorted(item.id for item in day_items)),
            "datetime": acquired.isoformat(),
            "offset_days": int((acquired - event_date.normalize()).days),
            "cloud_cover": max(clouds),
            "n_tiles": len(day_items),
            "aoi_bbox_coverage": coverage,
        })
    candidates.sort(
        key=lambda row: (row["cloud_cover"], abs(row["offset_days"]), row["item_ids"])
    )
    selected = []
    for row in candidates:
        day = pd.Timestamp(row["datetime"])
        if all(abs((day - pd.Timestamp(old["datetime"])).days) >= 10 for old in selected):
            selected.append(row)
        if len(selected) == 2:
            break
    selected.sort(key=lambda row: row["datetime"])
    return selected


def query_unit(client, row: dict, cloud_max: float) -> dict:
    event = pd.Timestamp(row["event_date"])
    start = event - timedelta(days=180)
    end = event + timedelta(days=180)
    bbox = [row["bbox_left"], row["bbox_bottom"], row["bbox_right"], row["bbox_top"]]
    last_error = None
    for attempt in range(4):
        try:
            search = client.search(
                collections=[COLLECTION],
                bbox=bbox,
                datetime=f"{start.date().isoformat()}/{end.date().isoformat()}",
                query={"eo:cloud_cover": {"lt": cloud_max}},
                max_items=500,
            )
            items = list(search.items())
            break
        except Exception as error:  # network service retry
            last_error = error
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"STAC query failed after retries: {last_error}")
    pre = select_side(items, event, "pre", bbox)
    post = select_side(items, event, "post", bbox)
    result = dict(row)
    result.update({
        "n_catalog_items": len(items),
        "n_pre_selected": len(pre),
        "n_post_selected": len(post),
        "selected_item_ids": "|".join(x["item_ids"] for x in pre + post),
        "selected_datetimes": "|".join(x["datetime"] for x in pre + post),
        "selected_offsets_days": "|".join(str(x["offset_days"]) for x in pre + post),
        "selected_cloud_cover": "|".join(f"{x['cloud_cover']:.6g}" for x in pre + post),
        "selected_aoi_bbox_coverage": "|".join(
            f"{x['aoi_bbox_coverage']:.6g}" for x in pre + post
        ),
        "selected_tile_counts": "|".join(str(x["n_tiles"]) for x in pre + post),
        "prithvi_temporal_ready": int(len(pre) == 2 and len(post) == 2),
        "status": "ready" if len(pre) == 2 and len(post) == 2 else "insufficient_temporal_support",
    })
    return result


def open_catalog():
    from pystac_client import Client

    last_error = None
    for attempt in range(6):
        try:
            return Client.open(CATALOG_URL)
        except Exception as error:  # transient STAC/TLS failure
            last_error = error
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Unable to open STAC catalog after retries: {last_error}")


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.readiness)
    required = {
        "sample_id", "dataset_id", "source_scene_id", "physical_event_id", "event_date",
        "bbox_left", "bbox_bottom", "bbox_right", "bbox_top",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Readiness registry missing fields: {sorted(missing)}")
    units = (
        frame.groupby(["dataset_id", "source_scene_id", "physical_event_id", "event_date"], as_index=False)
        .agg(
            n_windows=("sample_id", "size"),
            bbox_left=("bbox_left", "min"),
            bbox_bottom=("bbox_bottom", "min"),
            bbox_right=("bbox_right", "max"),
            bbox_top=("bbox_top", "max"),
        )
        .sort_values(["dataset_id", "source_scene_id", "physical_event_id"])
        .reset_index(drop=True)
    )
    units.insert(
        0,
        "acquisition_unit_id",
        units["dataset_id"].astype(str) + "::" + units["source_scene_id"].astype(str)
        + "::" + units["physical_event_id"].astype(str),
    )
    units["pre_window_start"] = (pd.to_datetime(units["event_date"]) - pd.Timedelta(days=180)).dt.date.astype(str)
    units["pre_window_end"] = (pd.to_datetime(units["event_date"]) - pd.Timedelta(days=7)).dt.date.astype(str)
    units["post_window_start"] = (pd.to_datetime(units["event_date"]) + pd.Timedelta(days=7)).dt.date.astype(str)
    units["post_window_end"] = (pd.to_datetime(units["event_date"]) + pd.Timedelta(days=180)).dt.date.astype(str)
    units["target_bands"] = ",".join(TARGET_BANDS)
    units["observations_per_side"] = 2
    units["minimum_same_side_separation_days"] = 10
    units["selection_uses_labels"] = 0
    units["status"] = "metadata_query_pending"
    args.outdir.mkdir(parents=True, exist_ok=True)
    plan_path = args.outdir / "acquisition_request_registry_v1.csv"
    units.to_csv(plan_path, index=False)

    audit_path = args.outdir / "acquisition_availability_v1.csv"
    if args.query_stac:
        try:
            import pystac_client  # noqa: F401
        except ImportError as error:
            raise RuntimeError("pystac-client is required for --query-stac") from error
        query = units.head(args.max_units) if args.max_units else units
        client = open_catalog()
        audited = []
        for position, row in enumerate(query.to_dict("records"), start=1):
            result = query_unit(client, row, args.cloud_cover_max)
            audited.append(result)
            print(
                f"[{position}/{len(query)}] {row['acquisition_unit_id']} "
                f"pre={result['n_pre_selected']} post={result['n_post_selected']} "
                f"ready={result['prithvi_temporal_ready']}",
                flush=True,
            )
        audit = pd.DataFrame(audited)
        audit.to_csv(audit_path, index=False)
    else:
        audit = pd.DataFrame()

    summary = {
        "schema_version": 1,
        "catalog_url": CATALOG_URL,
        "collection": COLLECTION,
        "n_acquisition_units": len(units),
        "n_windows": int(units["n_windows"].sum()),
        "n_events": int(units["physical_event_id"].nunique()),
        "n_units_queried": len(audit),
        "n_units_ready": int(audit["prithvi_temporal_ready"].sum()) if len(audit) else 0,
        "required_assets": sorted(REQUIRED_ASSETS),
        "selection_contract": "two pre and two post observations, >=10 d same-side separation, cloud then event-offset ordering; labels forbidden",
        "request_registry": str(plan_path.resolve()),
        "availability_registry": str(audit_path.resolve()) if len(audit) else None,
        "warning": "Availability is not model evidence; imagery must still pass coverage, SCL and crop identity audits.",
    }
    (args.outdir / "acquisition_summary_v1.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
