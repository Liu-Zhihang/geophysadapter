#!/usr/bin/env python3
"""Download only the FABDEM one-degree tiles required by buffered DLR windows."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from huggingface_hub import hf_hub_download
from pyproj import Transformer
from rasterio.transform import Affine
from remotezip import RemoteZip


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "processed/hybrid_pinn/pild_core_geo_v2_1_native30_raw/dlr_window_registry_v2.csv"
)
DEFAULT_OUTDIR = PROJECT_ROOT / "raw_fullcopy/static/fabdem_v1_2_dlr"
BASE_URL = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--buffer-m", type=float, default=3000.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--provider",
        choices=("huggingface", "bristol"),
        default="huggingface",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_affine(text: str) -> Affine:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 6:
        raise ValueError(f"invalid target_transform: {text}")
    return Affine(*values)


def tile_stem(latitude: int, longitude: int) -> str:
    lat = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}"
    lon = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}"
    return f"{lat}{lon}"


def block_name(latitude: int, longitude: int) -> str:
    south = math.floor(latitude / 10) * 10
    west = math.floor(longitude / 10) * 10
    east = west + 10
    if east == 180:
        east = -180
    return (
        f"{tile_stem(south, west)}-{tile_stem(south + 10, east)}"
        "_FABDEM_V1-2.zip"
    )


def buffered_geographic_bounds(row: dict[str, str], buffer_m: float) -> tuple[float, ...]:
    transform = parse_affine(row["target_transform"])
    width = 128
    height = 128
    corners = (
        transform * (0, 0),
        transform * (width, 0),
        transform * (0, height),
        transform * (width, height),
    )
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    transformer = Transformer.from_crs(row["target_crs"], "EPSG:4326", always_xy=True)
    projected = (
        min(xs) - buffer_m,
        min(ys) - buffer_m,
        max(xs) + buffer_m,
        max(ys) + buffer_m,
    )
    geographic = [
        transformer.transform(x, y)
        for x, y in (
            (projected[0], projected[1]),
            (projected[0], projected[3]),
            (projected[2], projected[1]),
            (projected[2], projected[3]),
        )
    ]
    longitudes = [point[0] for point in geographic]
    latitudes = [point[1] for point in geographic]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def required_members(registry: Path, buffer_m: float) -> dict[str, set[str]]:
    with registry.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty DLR registry: {registry}")
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        left, bottom, right, top = buffered_geographic_bounds(row, buffer_m)
        for latitude in range(math.floor(bottom), math.ceil(top)):
            for longitude in range(math.floor(left), math.ceil(right)):
                member = f"{tile_stem(latitude, longitude)}_FABDEM_V1-2.tif"
                grouped[block_name(latitude, longitude)].add(member)
    return grouped


def validate_tile(path: Path) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        if dataset.crs is None or dataset.width < 100 or dataset.height < 100:
            raise RuntimeError(f"invalid FABDEM tile: {path}")
        return {
            "filename": path.name,
            "width": dataset.width,
            "height": dataset.height,
            "crs": str(dataset.crs),
            "bounds": list(dataset.bounds),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def download_block(
    block: str,
    members: set[str],
    outdir: Path,
    overwrite: bool,
    retries: int,
    provider: str,
) -> list[dict[str, object]]:
    pending = [
        member
        for member in sorted(members)
        if overwrite or not (outdir / member).exists()
    ]
    if pending:
        block_stem = block.removesuffix(".zip")
        url = (
            f"https://huggingface.co/datasets/links-ads/fabdem-v12/tree/main/tiles/{block_stem}"
            if provider == "huggingface"
            else f"{BASE_URL}/{block}"
        )
        last_error: Exception | None = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                if provider == "huggingface":
                    for member in pending:
                        cached = hf_hub_download(
                            repo_id="links-ads/fabdem-v12",
                            repo_type="dataset",
                            filename=f"tiles/{block_stem}/{member}",
                        )
                        destination = outdir / member
                        temporary = destination.with_suffix(
                            destination.suffix + f".tmp.{os.getpid()}"
                        )
                        temporary.unlink(missing_ok=True)
                        shutil.copyfile(cached, temporary)
                        os.replace(temporary, destination)
                else:
                    with RemoteZip(url) as archive:
                        available = set(archive.namelist())
                        missing = set(pending) - available
                        if missing:
                            raise FileNotFoundError(
                                f"{block} lacks required members: {sorted(missing)}"
                            )
                        for member in pending:
                            destination = outdir / member
                            temporary = destination.with_suffix(
                                destination.suffix + f".tmp.{os.getpid()}"
                            )
                            temporary.unlink(missing_ok=True)
                            with archive.open(member) as source, temporary.open(
                                "wb"
                            ) as target:
                                shutil.copyfileobj(
                                    source, target, length=8 * 1024 * 1024
                                )
                            os.replace(temporary, destination)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt < retries:
                    time.sleep(min(3 * attempt, 15))
        if last_error is not None:
            raise RuntimeError(
                f"FABDEM block failed after {retries} attempts: {block}"
            ) from last_error
    return [validate_tile(outdir / member) for member in sorted(members)]


def main() -> int:
    args = parse_args()
    if args.buffer_m < 1000:
        raise ValueError("buffer must be at least 1000 m for native Terrain derivatives")
    args.outdir.mkdir(parents=True, exist_ok=True)
    grouped = required_members(args.registry, args.buffer_m)
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_block,
                block,
                members,
                args.outdir,
                args.overwrite,
                args.retries,
                args.provider,
            ): block
            for block, members in grouped.items()
        }
        for future in concurrent.futures.as_completed(futures):
            block = futures[future]
            block_records = future.result()
            for record in block_records:
                record["source_block"] = block
                record["source_url"] = (
                    f"https://huggingface.co/datasets/links-ads/fabdem-v12/tree/main/"
                    f"tiles/{block.removesuffix('.zip')}"
                    if args.provider == "huggingface"
                    else f"{BASE_URL}/{block}"
                )
            records.extend(block_records)
            print(
                json.dumps(
                    {"block": block, "tiles": len(block_records), "status": "complete"}
                ),
                flush=True,
            )
    records.sort(key=lambda record: str(record["filename"]))
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "FABDEM V1-2",
        "license": "CC BY-NC-SA 4.0",
        "doi": "10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn",
        "download_provider": args.provider,
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "buffer_m": args.buffer_m,
        "n_blocks": len(grouped),
        "n_tiles": len(records),
        "tiles": records,
    }
    (args.outdir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.outdir / "DONE.json").write_text(
        json.dumps(
            {"status": "complete", "n_tiles": len(records)},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(args.outdir), "tiles": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
