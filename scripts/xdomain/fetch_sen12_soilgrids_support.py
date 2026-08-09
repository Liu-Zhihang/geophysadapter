#!/usr/bin/env python3
"""Fetch checksum-verified SoilGrids tiles intersecting frozen Sen12 regions."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
import fetch_dlr_soilgrids_support as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--properties", default=",".join(base.DEFAULT_PROPERTIES))
    parser.add_argument("--depths", default=",".join(base.DEFAULT_DEPTHS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/sen12_soilgrids_support_fetch_v1"),
    )
    return parser.parse_args()


def sen12_region_bounds(root: Path) -> dict[str, tuple[float, float, float, float, str]]:
    registry_path = root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
    cache_path = root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv"
    registry = pd.read_csv(registry_path, low_memory=False)
    cache = pd.read_csv(cache_path, usecols=["sample_id"])
    frame = cache.merge(
        registry[["sample_id", "region", "min_lon", "min_lat", "max_lon", "max_lat"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    required = ["region", "min_lon", "min_lat", "max_lon", "max_lat"]
    if len(frame) != 4979 or not frame[required].notna().all().all():
        raise RuntimeError("Frozen Sen12 cache lacks complete geographic footprints")
    output: dict[str, tuple[float, float, float, float, str]] = {}
    for region, group in frame.groupby("region", sort=True):
        output[str(region)] = (
            float(group["min_lon"].min()),
            float(group["min_lat"].min()),
            float(group["max_lon"].max()),
            float(group["max_lat"].max()),
            "EPSG:4326",
        )
    if len(output) != 13:
        raise RuntimeError(f"Expected 13 Sen12 regions, found {len(output)}")
    return output


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    properties = tuple(value.strip() for value in args.properties.split(",") if value.strip())
    depths = tuple(value.strip() for value in args.depths.split(",") if value.strip())
    if not properties or not depths or args.workers < 1:
        raise ValueError("properties/depths must be non-empty and workers positive")
    unknown = sorted(set(properties) - set(base.DEFAULT_PROPERTIES))
    if unknown:
        raise ValueError(f"Unsupported SoilGrids properties: {unknown}")

    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    vrt_cache = root / "metadata/cache/soilgrids_vrt"
    soil_root = root / "raw_fullcopy/static/soilgrids"
    regions = sen12_region_bounds(root)

    region_rows: list[dict[str, object]] = []
    pair_sources: dict[tuple[str, str], dict[str, list[str]]] = {}
    pair_vrts = {
        (prop, depth): (
            f"{base.REMOTE_BASE}/{prop}/{prop}_{depth}_mean.vrt",
            vrt_cache / f"{prop}_{depth}_mean.vrt",
        )
        for prop in properties
        for depth in depths
    }
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                base.ensure_remote_file, url, path, args.retries, args.timeout, args.force
            )
            for url, path in pair_vrts.values()
        ]
        for future in as_completed(futures):
            future.result()

    for prop in properties:
        for depth in depths:
            stem = f"{prop}_{depth}_mean"
            _, vrt_path = pair_vrts[(prop, depth)]
            vrt_crs, geotransform, sources = base.parse_vrt(vrt_path)
            by_region: dict[str, list[str]] = {}
            for region, (*bounds, source_crs) in regions.items():
                window = base.pixel_window(tuple(bounds), source_crs, vrt_crs, geotransform)
                hits = base.intersecting_sources(window, sources)
                if not hits:
                    raise RuntimeError(f"No {stem} source tile intersects region={region}")
                by_region[region] = hits
                region_rows.append(
                    {
                        "region": region,
                        "property": prop,
                        "depth": depth,
                        "n_tiles": len(hits),
                        "tiles": ";".join(hits),
                    }
                )
            pair_sources[(prop, depth)] = by_region

    source_records: list[tuple[str, str, str, tuple[str, ...], tuple[str, str, str]]] = []
    checksum_urls: dict[tuple[str, str, str], str] = {}
    for (prop, depth), by_region in pair_sources.items():
        stem = f"{prop}_{depth}_mean"
        reverse: dict[str, set[str]] = {}
        for region, paths in by_region.items():
            for path in paths:
                reverse.setdefault(path, set()).add(region)
        for relative_path, region_names in sorted(reverse.items()):
            rel = Path(relative_path)
            if len(rel.parts) < 3 or rel.parts[0] != stem:
                raise RuntimeError(f"Unexpected VRT source path: {relative_path}")
            parent = rel.parent.name
            cache_key = (prop, stem, parent)
            checksum_urls[cache_key] = (
                f"{base.REMOTE_BASE}/{prop}/{stem}/{parent}/checksum.sha256.txt"
            )
            source_records.append(
                (prop, depth, relative_path, tuple(sorted(region_names)), cache_key)
            )

    checksum_cache: dict[tuple[str, str, str], dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(base.fetch_bytes, url, args.retries, args.timeout): key
            for key, url in checksum_urls.items()
        }
        for future in as_completed(future_map):
            checksum_cache[future_map[future]] = base.parse_checksums(future.result())

    items: list[base.DownloadItem] = []
    for prop, depth, relative_path, region_names, cache_key in source_records:
        rel = Path(relative_path)
        expected = checksum_cache[cache_key].get(rel.name)
        if expected is None:
            raise RuntimeError(f"Official checksum missing for {relative_path}")
        items.append(
            base.DownloadItem(
                property_name=prop,
                depth=depth,
                relative_path=relative_path,
                remote_url=f"{base.REMOTE_BASE}/{prop}/{relative_path}",
                local_path=soil_root / prop / relative_path,
                expected_sha256=expected,
                events=region_names,
            )
        )

    plan_rows = [
        {
            "property": item.property_name,
            "depth": item.depth,
            "relative_path": item.relative_path,
            "regions": ";".join(item.events),
            "remote_url": item.remote_url,
            "local_path": str(item.local_path),
            "expected_sha256": item.expected_sha256,
            "already_present": int(item.local_path.is_file()),
            "existing_bytes": item.local_path.stat().st_size if item.local_path.is_file() else 0,
        }
        for item in items
    ]
    base.write_csv(outdir / "region_tile_map.csv", region_rows)
    base.write_csv(outdir / "download_plan.csv", plan_rows)

    run_rows: list[dict[str, object]] = []
    if not args.plan_only:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    base.download_one, item, args.retries, args.timeout, args.force
                ): item
                for item in items
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                item = future_map[future]
                result = future.result()
                run_rows.append(
                    {
                        "property": item.property_name,
                        "depth": item.depth,
                        "relative_path": item.relative_path,
                        "regions": ";".join(item.events),
                        **result,
                    }
                )
                if index % 25 == 0 or index == len(items):
                    print(f"[download] {index}/{len(items)}", flush=True)
        run_rows.sort(
            key=lambda row: (str(row["property"]), str(row["depth"]), str(row["relative_path"]))
        )
        base.write_csv(outdir / "download_results.csv", run_rows)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "13 frozen Sen12 regions; exact official SoilGrids VRT intersections",
        "remote_base": base.REMOTE_BASE,
        "properties": list(properties),
        "depths": list(depths),
        "regions": len(regions),
        "download_items": len(items),
        "already_present_items": int(sum(int(row["already_present"]) for row in plan_rows)),
        "missing_items": int(sum(1 - int(row["already_present"]) for row in plan_rows)),
        "plan_only": bool(args.plan_only),
        "all_downloads_verified": bool(run_rows) and len(run_rows) == len(items),
        "status_counts": (
            pd.Series([str(row["status"]) for row in run_rows]).value_counts().sort_index().to_dict()
            if run_rows
            else {}
        ),
        "download_bytes": int(sum(int(row["bytes"]) for row in run_rows)),
        "data_boundary": (
            "SoilGrids has approximately 250 m native prediction support. Reprojection to a "
            "segmentation grid does not create native 10 m Material measurements."
        ),
    }
    base.write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
