#!/usr/bin/env python3
"""Register, download, and validate native CopDEM GLO-30 tiles for Sen12.

The harmonized Sen12 NetCDF files contain a GLO-30 surface resampled to the
10 m Sentinel-2 grid. Terrain-v2 instead derives geomorphometry on the native
GLO-30 grid with spatial context, then reprojects the derived layers. This
script freezes the required source tiles from the sample registry before any
Terrain-v2 outcome is produced.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio


BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--dem-dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--buffer-degrees",
        type=float,
        default=0.03,
        help="Conservative geographic buffer for native-scale derivatives (about 3 km).",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tile_name(latitude: int, longitude: int) -> str:
    lat = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}_00"
    lon = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}_00"
    return f"Copernicus_DSM_COG_10_{lat}_{lon}_DEM.tif"


def tile_url(name: str) -> str:
    return f"{BASE_URL}/{name.removesuffix('.tif')}/{name}"


def validate(path: Path) -> tuple[bool, str]:
    try:
        with rasterio.open(path) as source:
            if str(source.crs) != "EPSG:4326" or source.count != 1:
                return False, "invalid_crs_or_band_count"
            if source.width < 100 or source.height < 100:
                return False, "invalid_dimensions"
            probe = source.read(
                1,
                window=((0, min(16, source.height)), (0, min(16, source.width))),
                masked=True,
            )
            if probe.size == 0:
                return False, "empty_probe"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def acquire(row: dict[str, Any], retries: int, timeout: int) -> dict[str, Any]:
    path = Path(str(row["path"]))
    valid, error = validate(path) if path.is_file() else (False, "missing")
    if valid:
        return {
            **row,
            "status": "existing",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "error": "",
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    last_error = error
    for attempt in range(1, max(1, retries) + 1):
        try:
            temporary.unlink(missing_ok=True)
            request = urllib.request.Request(
                str(row["url"]), headers={"User-Agent": "GeoPhysAdapter-revision/terrain-v2"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
                expected = int(response.headers.get("Content-Length", "0") or 0)
                while block := response.read(4 * 1024 * 1024):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if expected and temporary.stat().st_size != expected:
                raise RuntimeError(f"truncated:{temporary.stat().st_size}!={expected}")
            valid, validation_error = validate(temporary)
            if not valid:
                raise RuntimeError(validation_error)
            temporary.replace(path)
            return {
                **row,
                "status": "downloaded",
                "attempt": attempt,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "error": "",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 16))
    return {**row, "status": "failed", "size_bytes": 0, "sha256": "", "error": last_error}


def read_registry(path: Path, buffer_degrees: float) -> tuple[list[dict[str, Any]], int]:
    tiles: defaultdict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"regions": set(), "events": set(), "samples": set()}
    )
    eligible_samples = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "sample_id",
            "region",
            "physical_event_group",
            "change_view_eligible",
            "min_lon",
            "min_lat",
            "max_lon",
            "max_lat",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"registry missing columns: {sorted(missing)}")
        for row in reader:
            if row["change_view_eligible"] != "1":
                continue
            eligible_samples += 1
            left = float(row["min_lon"]) - buffer_degrees
            bottom = float(row["min_lat"]) - buffer_degrees
            right = float(row["max_lon"]) + buffer_degrees
            top = float(row["max_lat"]) + buffer_degrees
            for latitude in range(math.floor(bottom), math.ceil(top)):
                for longitude in range(math.floor(left), math.ceil(right)):
                    name = tile_name(latitude, longitude)
                    tiles[name]["regions"].add(row["region"])
                    tiles[name]["events"].add(row["physical_event_group"])
                    tiles[name]["samples"].add(row["sample_id"])

    rows = []
    for name, membership in sorted(tiles.items()):
        rows.append(
            {
                "tile": name,
                "url": tile_url(name),
                "regions": ";".join(sorted(membership["regions"])),
                "physical_events": ";".join(sorted(membership["events"])),
                "n_samples": len(membership["samples"]),
            }
        )
    return rows, eligible_samples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    registry = (
        args.registry.resolve()
        if args.registry
        else root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
    )
    dem_dir = (
        args.dem_dir.resolve()
        if args.dem_dir
        else root / "raw_fullcopy/static/copdem_glo30_2021"
    )
    outdir = (
        args.outdir.resolve()
        if args.outdir
        else root / "metadata/pild_xdomain_v1/terrain_v2_copdem"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    dem_dir.mkdir(parents=True, exist_ok=True)

    if args.buffer_degrees < 0:
        raise ValueError("--buffer-degrees must be non-negative")
    rows, eligible_samples = read_registry(registry, args.buffer_degrees)
    for row in rows:
        row["path"] = str(dem_dir / str(row["tile"]))
    write_csv(outdir / "required_tiles.csv", rows)

    if args.download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(acquire, row, args.retries, args.timeout) for row in rows]
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{result['status']}] {result['tile']} {result.get('error', '')}", flush=True)
        results.sort(key=lambda row: str(row["tile"]))
    else:
        results = []
        for row in rows:
            path = Path(str(row["path"]))
            valid, error = validate(path) if path.is_file() else (False, "missing")
            results.append(
                {
                    **row,
                    "status": "existing" if valid else "missing_or_invalid",
                    "size_bytes": path.stat().st_size if valid else 0,
                    "sha256": sha256_file(path) if valid else "",
                    "error": error,
                }
            )

    write_csv(outdir / "acquisition_manifest.csv", results)
    complete = all(row["status"] in {"existing", "downloaded"} for row in results)
    status_counts: dict[str, int] = {}
    for row in results:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(registry),
        "source_registry_sha256": sha256_file(registry),
        "dem_dir": str(dem_dir),
        "native_source": "Copernicus DEM GLO-30 DSM, approximately 30 m",
        "derivative_contract": "derive on native buffered mosaic before reprojection to prediction grid",
        "geographic_buffer_degrees": args.buffer_degrees,
        "n_eligible_samples": eligible_samples,
        "n_required_tiles": len(results),
        "status_counts": status_counts,
        "all_complete": complete,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
