#!/usr/bin/env python3
"""Audit Sentinel-2 L2A pre/post availability for UGLC event candidates."""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pystac_client import Client


CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
REQUIRED_ASSETS = {"B02", "B03", "B04", "SCL"}


def query_items(client: Client, bbox: list[float], start: pd.Timestamp, end: pd.Timestamp) -> list:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            search = client.search(
                collections=[COLLECTION],
                bbox=bbox,
                datetime=f"{start.date().isoformat()}/{end.date().isoformat()}",
                query={"eo:cloud_cover": {"lt": 90}},
                max_items=300,
            )
            return [item for item in search.items() if REQUIRED_ASSETS.issubset(item.assets)]
        except Exception as error:  # network service retry
            last_error = error
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"STAC query failed after retries: {last_error}")


def summarize(items: list, target: pd.Timestamp) -> dict[str, object]:
    if not items:
        return {
            "n_items": 0,
            "cloud_min": None,
            "cloud_median": None,
            "selected_item_id": "",
            "selected_datetime": "",
            "selected_cloud": None,
            "selected_target_offset_days": None,
        }
    rows = []
    for item in items:
        acquired = pd.Timestamp(item.datetime).tz_localize(None)
        cloud = float(item.properties.get("eo:cloud_cover", np.nan))
        offset = abs((acquired - target).total_seconds()) / 86400.0
        score = offset + 0.5 * (cloud if np.isfinite(cloud) else 100.0)
        rows.append((score, item.id, acquired, cloud, offset))
    rows.sort(key=lambda row: (row[0], row[1]))
    selected = rows[0]
    clouds = np.asarray([row[3] for row in rows], dtype=np.float64)
    clouds = clouds[np.isfinite(clouds)]
    return {
        "n_items": len(items),
        "cloud_min": float(clouds.min()) if clouds.size else None,
        "cloud_median": float(np.median(clouds)) if clouds.size else None,
        "selected_item_id": selected[1],
        "selected_datetime": selected[2].isoformat(),
        "selected_cloud": selected[3] if np.isfinite(selected[3]) else None,
        "selected_target_offset_days": selected[4],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--pre-days", type=int, default=90)
    parser.add_argument("--post-days", type=int, default=90)
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = root / "metadata/pild_xdomain_v1"
    source_path = metadata / "uglc_sentinel2_event_candidates_v1.csv"
    frame = pd.read_csv(source_path)
    if args.max_events:
        frame = frame.head(args.max_events)
    client = Client.open(CATALOG_URL)
    output = []
    for position, row in enumerate(frame.to_dict(orient="records"), start=1):
        start = pd.Timestamp(row["event_date_start"])
        end = pd.Timestamp(row["event_date_end"])
        bbox = [float(row[key]) for key in ("min_lon", "min_lat", "max_lon", "max_lat")]
        pre_start = start - timedelta(days=args.pre_days)
        pre_end = start - timedelta(days=7)
        post_start = end + timedelta(days=1)
        post_end = end + timedelta(days=args.post_days)
        pre = summarize(query_items(client, bbox, pre_start, pre_end), start - timedelta(days=14))
        post = summarize(query_items(client, bbox, post_start, post_end), end + timedelta(days=14))
        item = {
            **row,
            "pre_window_start": pre_start.date().isoformat(),
            "pre_window_end": pre_end.date().isoformat(),
            "post_window_start": post_start.date().isoformat(),
            "post_window_end": post_end.date().isoformat(),
            **{f"pre_{key}": value for key, value in pre.items()},
            **{f"post_{key}": value for key, value in post.items()},
            "s2_prepost_ready": int(pre["n_items"] > 0 and post["n_items"] > 0),
        }
        output.append(item)
        print(
            f"[{position}/{len(frame)}] {row['record_id']} "
            f"pre={pre['n_items']} post={post['n_items']} ready={item['s2_prepost_ready']}",
            flush=True,
        )
    result = pd.DataFrame(output)
    csv_path = metadata / "uglc_sentinel2_availability_v1.csv"
    result.to_csv(csv_path, index=False)
    summary = {
        "catalog_url": CATALOG_URL,
        "collection": COLLECTION,
        "required_assets": sorted(REQUIRED_ASSETS),
        "n_events": len(result),
        "n_prepost_ready": int(result["s2_prepost_ready"].sum()),
        "pre_days": args.pre_days,
        "post_days": args.post_days,
        "selection_score": "absolute_days_from_14_day_target + 0.5 * eo_cloud_cover",
        "metadata_only": True,
        "output_csv": str(csv_path),
    }
    (metadata / "uglc_sentinel2_availability_summary_v1.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
