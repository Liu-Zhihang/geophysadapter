#!/usr/bin/env python3
"""Build event/sample physics vectors for strict_t2 supervised-ready subsets.

This script targets the current multi-source supervised-ready strict_t2 pool
(`CAS + DLR + GDCLD + GLaD4CD v1 labeled subset`). It keeps the vector design
explicit:

- Terrain is represented as event-level local-window proxies sampled from
  CopDEM around the event centroid.
- Material variables come from WorldCover, SoilGrids, and LiMW/GLiM.
- Trigger variables come from CHIRPS, SMAP, and optional DLR-only ERA5-Land.
- Physics vectors are computed at the event level and then broadcast to every
  supervised-ready sample of that event.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gzip
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.windows import Window
from shapely.geometry import Point

TRIGGER_NAMES = ["rainfall", "earthquake", "storm", "snowmelt", "complex", "unknown"]
WORLD_COVER_TREE = {10}
WORLD_COVER_CROP = {40}
WORLD_COVER_BARE = {60}
SOIL_VARS = ["clay", "sand", "silt", "cec", "soc"]
ERA5_ELEMS = {"TP", "SRO", "SWVL1"}
SOIL_INDEX_CACHE = "soilgrids_tile_index_clay_0_5cm_v1.json"
LITH_LAYER = "GLiM_export"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict_t2 supervised-ready event/sample physics vectors")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument(
        "--sample-manifest",
        default="",
        help="default: metadata/manifests/strict_t2_supervised_ready_native_v1.csv",
    )
    p.add_argument(
        "--sample-manifest-post-rgb",
        default="",
        help="default: metadata/manifests/strict_t2_supervised_ready_post_rgb_v1.csv",
    )
    p.add_argument(
        "--sample-manifest-change-rgb",
        default="",
        help="default: metadata/manifests/strict_t2_supervised_ready_change_rgb_v1.csv",
    )
    p.add_argument(
        "--event-summary",
        default="",
        help="default: metadata/manifests/strict_t2_supervised_ready_event_summary_v1.csv",
    )
    p.add_argument(
        "--event-index",
        default="",
        help="default: metadata/manifests/event_index_v1_strict_t2.csv",
    )
    p.add_argument(
        "--outdir",
        default="",
        help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1",
    )
    p.add_argument("--limit", type=int, default=0, help="optional debug limit on number of events")
    p.add_argument("--skip-soil", action="store_true")
    p.add_argument("--skip-lithology", action="store_true")
    p.add_argument("--skip-chirps", action="store_true")
    p.add_argument("--skip-smap", action="store_true")
    p.add_argument("--skip-era5", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detect_raw_root(root: Path) -> Path:
    for cand in (root / "raw_fullcopy", root / "raw"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"unable to find raw dataset root under {root}")


def build_tile_index(files: Iterable[Path]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for path in sorted(files):
        if path.suffix.lower() != ".tif":
            continue
        with rasterio.open(path) as src:
            out.append({"path": path, "bounds": tuple(float(v) for v in src.bounds)})
    return out


def point_in(bounds: tuple[float, float, float, float], lon: float, lat: float) -> bool:
    return bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]


def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _fmt_wc_coord(v: int, is_lat: bool) -> str:
    hemi = ("N" if v >= 0 else "S") if is_lat else ("E" if v >= 0 else "W")
    width = 2 if is_lat else 3
    return f"{hemi}{abs(v):0{width}d}"


def worldcover_paths_for_bbox(worldcover_dir: Path, bbox_wgs84: tuple[float, float, float, float]) -> list[Path]:
    left, bottom, right, top = bbox_wgs84
    lon0 = int(math.floor(left / 3.0) * 3)
    lon1 = int(math.floor((right - 1e-9) / 3.0) * 3)
    lat0 = int(math.floor(bottom / 3.0) * 3)
    lat1 = int(math.floor((top - 1e-9) / 3.0) * 3)
    paths: list[Path] = []
    for lat in range(lat0, lat1 + 1, 3):
        for lon in range(lon0, lon1 + 1, 3):
            name = f"ESA_WorldCover_10m_2021_v200_{_fmt_wc_coord(lat, True)}{_fmt_wc_coord(lon, False)}_Map.tif"
            path = worldcover_dir / name
            if path.exists():
                paths.append(path)
    return sorted(set(paths))


def bbox_values_from_paths(files: Iterable[Path], bbox_wgs84: tuple[float, float, float, float]) -> np.ndarray:
    vals: list[np.ndarray] = []
    for path in files:
        with rasterio.open(path) as src:
            try:
                window = src.window(*bbox_wgs84)
                arr = src.read(1, window=window, masked=True)
            except Exception:
                continue
            data = arr.compressed()
            if data.size:
                vals.append(data.astype(np.float32))
    if not vals:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(vals)


def sample_worldcover_fractions_bbox(worldcover_dir: Path, bbox_wgs84: tuple[float, float, float, float]) -> tuple[float, float, float]:
    paths = worldcover_paths_for_bbox(worldcover_dir, bbox_wgs84)
    vals = bbox_values_from_paths(paths, bbox_wgs84)
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    tree = float(np.isin(vals, list(WORLD_COVER_TREE)).mean())
    crop = float(np.isin(vals, list(WORLD_COVER_CROP)).mean())
    bare = float(np.isin(vals, list(WORLD_COVER_BARE)).mean())
    return tree, crop, bare


def build_soil_tile_index(sample_var_dir: Path, cache_path: Path) -> tuple[str, list[dict[str, object]]]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return str(payload["crs"]), list(payload["tiles"])

    tif_files = sorted(sample_var_dir.glob("*_4-4.tif"))
    tiles: list[dict[str, object]] = []
    crs_wkt = ""
    for path in tif_files:
        with rasterio.open(path) as src:
            if src.crs is None:
                continue
            if not crs_wkt:
                crs_wkt = src.crs.to_wkt()
            tiles.append(
                {
                    "name": path.name,
                    "bounds": [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)],
                }
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"crs": crs_wkt, "tiles": tiles}, indent=2), encoding="utf-8")
    return crs_wkt, tiles


def find_soil_tile_name(soil_tiles: list[dict[str, object]], soil_crs_wkt: str, lon: float, lat: float) -> str | None:
    transformer = Transformer.from_crs(4326, soil_crs_wkt, always_xy=True)
    x, y = transformer.transform(lon, lat)
    for tile in soil_tiles:
        left, bottom, right, top = tile["bounds"]  # type: ignore[index]
        if left <= x <= right and bottom <= y <= top:
            return str(tile["name"])
    return None


def sample_soil_vars(soil_root: Path, tile_name: str | None, lon: float, lat: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for var in SOIL_VARS:
        if not tile_name:
            out[f"soil_{var}_raw"] = float("nan")
            continue
        path = soil_root / var / f"{var}_0-5cm_mean" / tile_name
        if not path.exists():
            out[f"soil_{var}_raw"] = float("nan")
            continue
        with rasterio.open(path) as src:
            try:
                val = next(src.sample([(lon, lat)]))[0]
            except Exception:
                val = float("nan")
            out[f"soil_{var}_raw"] = float(val) if np.isfinite(val) else float("nan")
    return out


def get_lithology_crs(path: Path):
    info = pyogrio.read_info(path, layer=LITH_LAYER)
    crs = info.get("crs")
    if not crs:
        raise RuntimeError(f"lithology CRS missing: {path}")
    return crs


def stable_unit_value(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / float(0xFFFFFFFF)


def sample_lithology_value(path: Path, lith_crs, lon: float, lat: float) -> float:
    transformer = Transformer.from_crs(4326, lith_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    pad = 50000.0
    subset = pyogrio.read_dataframe(path, layer=LITH_LAYER, bbox=(x - pad, y - pad, x + pad, y + pad), columns=["xx"])
    if subset.empty:
        return 0.0
    if subset.crs is None:
        return 0.0
    if str(subset.crs).upper() != "EPSG:4326":
        subset = subset.to_crs(4326)
    pt = Point(lon, lat)
    hit = subset[subset.geometry.contains(pt)]
    if hit.empty:
        hit = subset[subset.geometry.intersects(pt)]
    code = str((hit if not hit.empty else subset).iloc[0].get("xx", "unknown"))
    return float(stable_unit_value(code))


def build_smap_index(smap_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(smap_dir.glob("SMAP_L3_SM_P_*.h5")):
        parts = path.stem.split("_")
        if len(parts) >= 5:
            out[parts[4]] = path
    return out


def nearest_smap_rc(sample_file: Path, lon: float, lat: float) -> tuple[int, int]:
    with h5py.File(sample_file, "r") as f:
        lats = np.asarray(f["Soil_Moisture_Retrieval_Data_AM/latitude"], dtype=np.float32)
        lons = np.asarray(f["Soil_Moisture_Retrieval_Data_AM/longitude"], dtype=np.float32)
    dist = np.square(lats - lat) + np.square(lons - lon)
    idx = int(np.nanargmin(dist))
    rows, cols = lats.shape
    return divmod(idx, cols)


def read_smap_value(path: Path, row: int, col: int) -> float:
    vals: list[float] = []
    with h5py.File(path, "r") as f:
        for ds_name in [
            "Soil_Moisture_Retrieval_Data_AM/soil_moisture",
            "Soil_Moisture_Retrieval_Data_PM/soil_moisture_dca_pm",
        ]:
            if ds_name not in f:
                continue
            arr = f[ds_name]
            val = float(arr[row, col])
            if np.isfinite(val) and 0.0 <= val <= 1.0:
                vals.append(val)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def aggregate_smap(smap_index: dict[str, Path], event_dt: datetime, row: int, col: int) -> float:
    vals: list[float] = []
    for offset in range(-3, 4):
        key = (event_dt + timedelta(days=offset)).strftime("%Y%m%d")
        path = smap_index.get(key)
        if path is None:
            continue
        val = read_smap_value(path, row, col)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def build_chirps_index(chirps_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(chirps_dir.glob("*/*.tif.gz")):
        name = path.name
        if len(name) >= 21:
            key = name[11:21].replace(".", "")
            out[key] = path
    return out


@functools.lru_cache(maxsize=16)
def chirps_sample_value(path_str: str, lon: float, lat: float) -> float:
    path = Path(path_str)
    with gzip.open(path, "rb") as f:
        data = f.read()
    with MemoryFile(data) as mem:
        with mem.open() as src:
            try:
                val = next(src.sample([(lon, lat)]))[0]
            except Exception:
                return float("nan")
    if not np.isfinite(val):
        return float("nan")
    return float(val)


def aggregate_chirps(chirps_index: dict[str, Path], event_dt: datetime, lon: float, lat: float, days: int = 7) -> float:
    vals: list[float] = []
    for offset in range(-(days - 1), 1):
        key = (event_dt + timedelta(days=offset)).strftime("%Y%m%d")
        path = chirps_index.get(key)
        if path is None:
            continue
        val = chirps_sample_value(str(path), lon, lat)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan")
    return float(np.sum(vals))


def aggregate_api(chirps_index: dict[str, Path], event_dt: datetime, lon: float, lat: float, days: int = 14, decay: float = 0.85) -> float:
    vals: list[float] = []
    weights: list[float] = []
    for lag in range(1, days + 1):
        key = (event_dt - timedelta(days=lag)).strftime("%Y%m%d")
        path = chirps_index.get(key)
        if path is None:
            continue
        val = chirps_sample_value(str(path), lon, lat)
        if not np.isfinite(val):
            continue
        vals.append(float(val))
        weights.append(float(decay ** (lag - 1)))
    if not vals:
        return float("nan")
    v = np.asarray(vals, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    return float(np.sum(v * w))


def aggregate_era5_event(era5_dir: Path, event_dt: datetime, lon: float, lat: float) -> dict[str, float]:
    metrics = {"TP": [], "SRO": [], "SWVL1": []}
    start = (event_dt - timedelta(days=3)).date()
    end = event_dt.date()
    for path in sorted(era5_dir.glob("*.grib")):
        with rasterio.open(path) as src:
            for band in range(1, src.count + 1):
                tags = src.tags(band)
                elem = tags.get("GRIB_ELEMENT", "")
                if elem not in ERA5_ELEMS:
                    continue
                valid_time = tags.get("GRIB_VALID_TIME")
                if not valid_time:
                    continue
                valid_dt = datetime.fromtimestamp(int(valid_time), tz=timezone.utc).date()
                if valid_dt < start or valid_dt > end:
                    continue
                try:
                    val = next(src.sample([(lon, lat)], indexes=band))[0]
                except Exception:
                    continue
                if np.isfinite(val):
                    metrics[elem].append(float(val))
    tp = float(np.mean(metrics["TP"])) if metrics["TP"] else float("nan")
    sro = float(np.mean(metrics["SRO"])) if metrics["SRO"] else float("nan")
    swvl1 = float(np.mean(metrics["SWVL1"])) if metrics["SWVL1"] else float("nan")
    hydro = float(np.nanmean([sro, swvl1])) if (np.isfinite(sro) or np.isfinite(swvl1)) else float("nan")
    return {"era5_tp_raw": tp, "era5_sro_raw": sro, "era5_swvl1_raw": swvl1, "era5_hydro_raw": hydro}


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def safe_float(v: object) -> float:
    try:
        if v is None or str(v).strip() == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def event_bbox(meta: dict[str, str]) -> tuple[float, float, float, float]:
    left = safe_float(meta.get("min_lon"))
    bottom = safe_float(meta.get("min_lat"))
    right = safe_float(meta.get("max_lon"))
    top = safe_float(meta.get("max_lat"))
    if np.isfinite(left) and np.isfinite(bottom) and np.isfinite(right) and np.isfinite(top) and right > left and top > bottom:
        return (left, bottom, right, top)
    lon = safe_float(meta.get("lon"))
    lat = safe_float(meta.get("lat"))
    if not (np.isfinite(lon) and np.isfinite(lat)):
        raise RuntimeError(f"missing bbox and point for event {meta.get('event_uid')}")
    pad = 0.05
    return (lon - pad, lat - pad, lon + pad, lat + pad)


def event_centroid(meta: dict[str, str]) -> tuple[float, float]:
    lon = safe_float(meta.get("lon"))
    lat = safe_float(meta.get("lat"))
    if np.isfinite(lon) and np.isfinite(lat):
        return (lon, lat)
    bbox = event_bbox(meta)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def find_dem_tile(tile_index: list[dict[str, object]], lon: float, lat: float) -> Path:
    for tile in tile_index:
        bounds = tile["bounds"]
        assert isinstance(bounds, tuple)
        if point_in(bounds, lon, lat):
            return Path(tile["path"])
    raise FileNotFoundError(f"no CopDEM tile covers lon={lon}, lat={lat}")


def local_dem_window(dem_path: Path, lon: float, lat: float, bbox_wgs84: tuple[float, float, float, float]) -> tuple[np.ndarray, float, float, tuple[float, float, float, float]]:
    with rasterio.open(dem_path) as src:
        row, col = src.index(lon, lat)
        xres = abs(float(src.transform.a))
        yres = abs(float(src.transform.e))
        bbox_w = abs(bbox_wgs84[2] - bbox_wgs84[0])
        bbox_h = abs(bbox_wgs84[3] - bbox_wgs84[1])
        inferred = int(max(bbox_w / max(xres, 1e-6), bbox_h / max(yres, 1e-6)) / 2.0)
        # Keep the local terrain context bounded; larger windows make the
        # D8-based TWI proxy disproportionately slow without adding much signal.
        radius = min(max(inferred + 16, 48), 96)
        row_off = max(0, row - radius)
        col_off = max(0, col - radius)
        height = min(src.height - row_off, radius * 2 + 1)
        width = min(src.width - col_off, radius * 2 + 1)
        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
        arr = src.read(1, window=window, masked=True).astype(np.float32)
        bounds = src.window_bounds(window)
    data = np.asarray(arr.filled(np.nan), dtype=np.float32)
    return data, xres, yres, tuple(float(v) for v in bounds)


def _fill_dem(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    finite = np.isfinite(out)
    fill = float(np.nanmedian(out)) if finite.any() else 0.0
    out[~finite] = fill
    return out


def d8_flow_accumulation(dem: np.ndarray, xres: float, yres: float) -> np.ndarray:
    nrows, ncols = dem.shape
    offsets = [
        (-1, -1, math.hypot(xres, yres)),
        (-1, 0, yres),
        (-1, 1, math.hypot(xres, yres)),
        (0, -1, xres),
        (0, 1, xres),
        (1, -1, math.hypot(xres, yres)),
        (1, 0, yres),
        (1, 1, math.hypot(xres, yres)),
    ]
    receiver = np.full((nrows, ncols), -1, dtype=np.int32)
    flat_idx = lambda rr, cc: rr * ncols + cc
    for rr in range(nrows):
        for cc in range(ncols):
            z = dem[rr, cc]
            best = -1
            best_drop = 0.0
            for dr, dc, dist in offsets:
                nr = rr + dr
                nc = cc + dc
                if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                    continue
                drop = (z - dem[nr, nc]) / max(dist, 1e-6)
                if np.isfinite(drop) and drop > best_drop:
                    best_drop = drop
                    best = flat_idx(nr, nc)
            receiver[rr, cc] = best
    order = np.argsort(-dem.ravel())
    acc = np.ones(nrows * ncols, dtype=np.float32)
    rec = receiver.ravel()
    for idx in order:
        nxt = rec[idx]
        if nxt >= 0:
            acc[nxt] += acc[idx]
    return acc.reshape((nrows, ncols))


def terrain_metrics_from_dem(dem_raw: np.ndarray, xres: float, yres: float) -> dict[str, float]:
    dem = _fill_dem(dem_raw)
    gy, gx = np.gradient(dem, yres, xres)
    slope_rad = np.arctan(np.sqrt(np.square(gx) + np.square(gy)))
    slope_deg = np.degrees(slope_rad)
    dxx = np.gradient(gx, xres, axis=1)
    dyy = np.gradient(gy, yres, axis=0)
    curvature = dxx + dyy
    acc = d8_flow_accumulation(dem, xres, yres)
    twi = np.log(((acc + 1.0) * ((xres + yres) / 2.0)) / (np.tan(slope_rad) + 1e-6))

    cy = dem.shape[0] // 2
    cx = dem.shape[1] // 2
    sl = slice(max(0, cy - 5), min(dem.shape[0], cy + 6))
    sc = slice(max(0, cx - 5), min(dem.shape[1], cx + 6))

    return {
        "terrain_dem_raw": float(np.nanmedian(dem)),
        "terrain_slope_raw": float(np.nanmedian(slope_deg[sl, sc])),
        "terrain_curvature_raw": float(np.nanmedian(curvature[sl, sc])),
        "terrain_twi_raw": float(np.nanmedian(twi[sl, sc])),
    }


def minmax_fill(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        vals = out[col].astype(float).replace([np.inf, -np.inf], np.nan)
        fill = float(vals.median()) if vals.notna().any() else 0.0
        vals = vals.fillna(fill)
        lo = float(vals.min())
        hi = float(vals.max())
        if hi <= lo + 1e-12:
            out[col] = 0.0
        else:
            out[col] = (vals - lo) / (hi - lo)
    return out


def keep_existing_order(rows: list[dict[str, str]], extra_cols: list[str]) -> list[str]:
    if not rows:
        return extra_cols
    return list(rows[0].keys()) + extra_cols


def dataframe_from_rows(rows: list[dict[str, str]], fallback_columns: list[str]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=fallback_columns)


def build_report(event_df: pd.DataFrame, sample_df: pd.DataFrame, outpath: Path) -> None:
    ext_available = int(event_df["era5_tp_raw"].notna().sum()) if "era5_tp_raw" in event_df else 0
    lines = [
        "# strict_t2 Supervised-Ready Physics Vectors",
        "",
        "This report summarizes event/sample physics vectors materialized for the strict_t2 supervised-ready pool.",
        "",
        "## Summary",
        "",
        f"- Events: `{len(event_df)}`",
        f"- Samples: `{len(sample_df)}`",
        f"- Terrain dim: `4`",
        f"- Material dim: `9`",
        f"- Trigger common dim: `9`",
        f"- Trigger DLR-ext dim: `2`",
        f"- DLR events with ERA5 ext values: `{ext_available}`",
        "",
        "## Dataset Coverage",
        "",
        "| dataset | events | samples |",
        "|---|---:|---:|",
    ]
    event_counts = event_df.groupby("dataset_id").size().to_dict()
    sample_counts = sample_df.groupby("dataset_id").size().to_dict()
    for dataset in sorted(sample_counts):
        lines.append(f"| {dataset} | {event_counts.get(dataset, 0)} | {sample_counts.get(dataset, 0)} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Terrain metrics are local CopDEM proxies sampled from a centroid-centered window and normalized across the 49 supervised-ready events.",
            "- Material vectors combine local WorldCover fractions, point-sampled SoilGrids, and LiMW/GLiM lithology.",
            "- Common trigger vectors use trigger one-hot + CHIRPS + API + SMAP.",
            "- DLR-only ERA5 values are kept in `trigger_ext_0/1` instead of polluting the common strict_t2 trigger channels.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    raw_root = detect_raw_root(root)

    sample_manifest = Path(args.sample_manifest) if args.sample_manifest.strip() else root / "metadata" / "manifests" / "strict_t2_supervised_ready_native_v1.csv"
    sample_manifest_post = Path(args.sample_manifest_post_rgb) if args.sample_manifest_post_rgb.strip() else root / "metadata" / "manifests" / "strict_t2_supervised_ready_post_rgb_v1.csv"
    sample_manifest_change = Path(args.sample_manifest_change_rgb) if args.sample_manifest_change_rgb.strip() else root / "metadata" / "manifests" / "strict_t2_supervised_ready_change_rgb_v1.csv"
    event_summary = Path(args.event_summary) if args.event_summary.strip() else root / "metadata" / "manifests" / "strict_t2_supervised_ready_event_summary_v1.csv"
    event_index = Path(args.event_index) if args.event_index.strip() else root / "metadata" / "manifests" / "event_index_v1_strict_t2.csv"
    outdir = Path(args.outdir) if args.outdir.strip() else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    sample_rows = read_csv(sample_manifest)
    sample_rows_post = read_csv(sample_manifest_post)
    sample_rows_change = read_csv(sample_manifest_change)
    event_rows = read_csv(event_summary)
    index_rows = read_csv(event_index)
    master_rows = read_csv(raw_root / "indexes" / "event_master.csv")

    index_map = {row["event_uid"]: row for row in index_rows}
    master_map = {row["event_uid"]: row for row in master_rows}
    event_map = {row["event_uid"]: row for row in event_rows}
    event_uids = [row["event_uid"] for row in event_rows]
    if args.limit > 0:
        event_uids = event_uids[: args.limit]
        sample_rows = [row for row in sample_rows if row["event_uid"] in set(event_uids)]
        sample_rows_post = [row for row in sample_rows_post if row["event_uid"] in set(event_uids)]
        sample_rows_change = [row for row in sample_rows_change if row["event_uid"] in set(event_uids)]

    static_root = raw_root / "static"
    weather_root = raw_root / "weather"
    worldcover_dir = static_root / "worldcover_v200_2021"
    soil_root = static_root / "soilgrids"
    lithology_path = static_root / "lithology" / "LiMW_GIS 2015.gdb"
    chirps_dir = weather_root / "chirps_daily_global"
    smap_dir = weather_root / "smap_spl3smp"
    era5_root = weather_root / "era5_land" / "DLR_Landslide_Ref_2025"
    copdem_dir = static_root / "copdem_glo30_2021"

    dem_tiles = build_tile_index(copdem_dir.glob("*.tif"))
    soil_crs_wkt = ""
    soil_tiles: list[dict[str, object]] = []
    if not args.skip_soil:
        soil_index_cache = root / "metadata" / "cache" / SOIL_INDEX_CACHE
        soil_crs_wkt, soil_tiles = build_soil_tile_index(soil_root / "clay" / "clay_0-5cm_mean", soil_index_cache)
    lith_crs = None if args.skip_lithology else get_lithology_crs(lithology_path)
    chirps_index = {} if args.skip_chirps else build_chirps_index(chirps_dir)
    smap_index = {} if args.skip_smap else build_smap_index(smap_dir)
    first_smap = next(iter(smap_index.values())) if smap_index else None

    rows: list[dict[str, object]] = []
    for event_uid in event_uids:
        meta = dict(index_map.get(event_uid, {}))
        meta.update(master_map.get(event_uid, {}))
        meta.update(event_map.get(event_uid, {}))
        bbox = event_bbox(meta)
        lon, lat = event_centroid(meta)
        event_dt = parse_date(str(meta["event_date"]))
        terrain_available = 1
        terrain_source = "copdem_local_window"
        try:
            dem_path = find_dem_tile(dem_tiles, lon, lat)
            dem_window, xres, yres, local_bounds = local_dem_window(dem_path, lon, lat, bbox)
            terrain = terrain_metrics_from_dem(dem_window, xres, yres)
        except FileNotFoundError:
            terrain_available = 0
            terrain_source = "missing_copdem_tile"
            local_bounds = bbox
            terrain = {
                "terrain_dem_raw": float("nan"),
                "terrain_slope_raw": float("nan"),
                "terrain_curvature_raw": float("nan"),
                "terrain_twi_raw": float("nan"),
            }
        wc_tree, wc_crop, wc_bare = sample_worldcover_fractions_bbox(worldcover_dir, local_bounds)

        if args.skip_soil:
            soil_raw = {f"soil_{var}_raw": float("nan") for var in SOIL_VARS}
        else:
            soil_tile_name = find_soil_tile_name(soil_tiles, soil_crs_wkt, lon, lat)
            soil_raw = sample_soil_vars(soil_root, soil_tile_name, lon, lat)

        lith_raw = float("nan") if args.skip_lithology else sample_lithology_value(lithology_path, lith_crs, lon, lat)
        chirps_event = float("nan") if args.skip_chirps else aggregate_chirps(chirps_index, event_dt, lon, lat)
        api_rain = float("nan") if args.skip_chirps else aggregate_api(chirps_index, event_dt, lon, lat)
        if args.skip_smap or not first_smap:
            smap_raw = float("nan")
        else:
            smap_rc = nearest_smap_rc(first_smap, lon, lat)
            smap_raw = aggregate_smap(smap_index, event_dt, smap_rc[0], smap_rc[1])

        if args.skip_era5 or str(meta.get("dataset_id")) != "DLR_Landslide_Ref_2025":
            era5 = {"era5_tp_raw": float("nan"), "era5_sro_raw": float("nan"), "era5_swvl1_raw": float("nan"), "era5_hydro_raw": float("nan")}
        else:
            era5_dir = era5_root / event_uid
            era5 = aggregate_era5_event(era5_dir, event_dt, lon, lat) if era5_dir.exists() else {
                "era5_tp_raw": float("nan"),
                "era5_sro_raw": float("nan"),
                "era5_swvl1_raw": float("nan"),
                "era5_hydro_raw": float("nan"),
            }

        trigger_type = str(meta.get("trigger_type", "unknown")).strip().lower() or "unknown"
        trig_vec = [1.0 if trigger_type == name else 0.0 for name in TRIGGER_NAMES]
        if sum(trig_vec) == 0.0:
            trig_vec[-1] = 1.0

        row: dict[str, object] = {
            "event_uid": event_uid,
            "dataset_id": str(meta.get("dataset_id", "")),
            "role": str(meta.get("role", "")),
            "event_date": str(meta.get("event_date", "")),
            "trigger_type": trigger_type,
            "sample_count": int(meta.get("sample_count", 0) or 0),
            "terrain_available": terrain_available,
            "terrain_source": terrain_source,
            "bbox_left": bbox[0],
            "bbox_bottom": bbox[1],
            "bbox_right": bbox[2],
            "bbox_top": bbox[3],
            "centroid_lon": lon,
            "centroid_lat": lat,
            "local_bbox_left": local_bounds[0],
            "local_bbox_bottom": local_bounds[1],
            "local_bbox_right": local_bounds[2],
            "local_bbox_top": local_bounds[3],
            "wc_tree_frac": wc_tree,
            "wc_crop_frac": wc_crop,
            "wc_bare_frac": wc_bare,
            "chirps_event_raw": chirps_event,
            "api_rain_raw": api_rain,
            "smap_sm_raw": smap_raw,
            "lithology_raw": lith_raw,
            **terrain,
            **soil_raw,
            **era5,
        }
        for i, v in enumerate(trig_vec):
            row[f"trigger_onehot_{i}"] = v
        rows.append(row)
        print(f"[event] {len(rows)}/{len(event_uids)} {event_uid} terrain_available={terrain_available}", flush=True)

    event_df = pd.DataFrame(rows).sort_values("event_uid").reset_index(drop=True)
    raw_norm_cols = [
        "terrain_dem_raw",
        "terrain_slope_raw",
        "terrain_curvature_raw",
        "terrain_twi_raw",
        "soil_clay_raw",
        "soil_sand_raw",
        "soil_silt_raw",
        "soil_cec_raw",
        "soil_soc_raw",
        "lithology_raw",
        "chirps_event_raw",
        "api_rain_raw",
        "smap_sm_raw",
    ]
    event_df = minmax_fill(event_df, raw_norm_cols)

    era5_valid = event_df["era5_tp_raw"].notna() | event_df["era5_hydro_raw"].notna()
    if era5_valid.any():
        era5_tmp = minmax_fill(event_df.loc[era5_valid].copy(), ["era5_tp_raw", "era5_hydro_raw"])
        event_df["trigger_ext_0"] = 0.0
        event_df["trigger_ext_1"] = 0.0
        event_df.loc[era5_valid, "trigger_ext_0"] = era5_tmp["era5_tp_raw"].to_numpy()
        event_df.loc[era5_valid, "trigger_ext_1"] = era5_tmp["era5_hydro_raw"].to_numpy()
    else:
        event_df["trigger_ext_0"] = 0.0
        event_df["trigger_ext_1"] = 0.0

    event_df["terrain_0"] = event_df["terrain_dem_raw"]
    event_df["terrain_1"] = event_df["terrain_slope_raw"]
    event_df["terrain_2"] = event_df["terrain_curvature_raw"]
    event_df["terrain_3"] = event_df["terrain_twi_raw"]

    event_df["material_0"] = event_df["wc_tree_frac"]
    event_df["material_1"] = event_df["wc_crop_frac"]
    event_df["material_2"] = event_df["wc_bare_frac"]
    event_df["material_3"] = event_df["soil_clay_raw"]
    event_df["material_4"] = event_df["soil_sand_raw"]
    event_df["material_5"] = event_df["soil_silt_raw"]
    event_df["material_6"] = event_df["soil_cec_raw"]
    event_df["material_7"] = event_df["soil_soc_raw"]
    event_df["material_8"] = event_df["lithology_raw"]

    for i in range(len(TRIGGER_NAMES)):
        event_df[f"trigger_{i}"] = event_df[f"trigger_onehot_{i}"]
    event_df["trigger_6"] = event_df["chirps_event_raw"]
    event_df["trigger_7"] = event_df["api_rain_raw"]
    event_df["trigger_8"] = event_df["smap_sm_raw"]
    event_df["hydro_proxy"] = np.clip(0.45 * event_df["trigger_8"] + 0.35 * event_df["trigger_7"] + 0.20 * event_df["trigger_6"], 0.0, 1.0)
    event_df["stability_proxy"] = np.clip(
        0.35 * event_df["terrain_1"]
        + 0.15 * event_df["terrain_2"]
        + 0.15 * event_df["terrain_3"]
        + 0.10 * event_df["material_3"]
        + 0.10 * event_df["material_2"]
        + 0.10 * event_df["material_8"]
        - 0.10 * event_df["material_4"],
        0.0,
        1.0,
    )

    sample_columns = list(sample_rows[0].keys()) if sample_rows else []
    sample_df = dataframe_from_rows(sample_rows, sample_columns)
    sample_post_df = dataframe_from_rows(sample_rows_post, sample_columns)
    sample_change_df = dataframe_from_rows(sample_rows_change, sample_columns)
    sample_vector_cols = [
        "event_uid",
        "sample_count",
        "terrain_available",
        "terrain_source",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
        "bbox_top",
        "centroid_lon",
        "centroid_lat",
        "local_bbox_left",
        "local_bbox_bottom",
        "local_bbox_right",
        "local_bbox_top",
        "terrain_0",
        "terrain_1",
        "terrain_2",
        "terrain_3",
        "material_0",
        "material_1",
        "material_2",
        "material_3",
        "material_4",
        "material_5",
        "material_6",
        "material_7",
        "material_8",
        "trigger_0",
        "trigger_1",
        "trigger_2",
        "trigger_3",
        "trigger_4",
        "trigger_5",
        "trigger_6",
        "trigger_7",
        "trigger_8",
        "trigger_ext_0",
        "trigger_ext_1",
        "hydro_proxy",
        "stability_proxy",
        "terrain_dem_raw",
        "terrain_slope_raw",
        "terrain_curvature_raw",
        "terrain_twi_raw",
        "wc_tree_frac",
        "wc_crop_frac",
        "wc_bare_frac",
        "soil_clay_raw",
        "soil_sand_raw",
        "soil_silt_raw",
        "soil_cec_raw",
        "soil_soc_raw",
        "lithology_raw",
        "chirps_event_raw",
        "api_rain_raw",
        "smap_sm_raw",
        "era5_tp_raw",
        "era5_sro_raw",
        "era5_swvl1_raw",
        "era5_hydro_raw",
    ]
    sample_out_df = sample_df.merge(event_df[sample_vector_cols], on="event_uid", how="left")
    sample_post_out_df = sample_post_df.merge(event_df[sample_vector_cols], on="event_uid", how="left")
    sample_change_out_df = sample_change_df.merge(event_df[sample_vector_cols], on="event_uid", how="left")

    event_out = outdir / "event_physics_vectors_v1.csv"
    sample_out = outdir / "sample_physics_vectors_v1.csv"
    sample_post_out = outdir / "sample_physics_vectors_post_rgb_v1.csv"
    sample_change_out = outdir / "sample_physics_vectors_change_rgb_v1.csv"
    report_out = outdir / "strict_t2_event_physics_report.md"
    summary_out = outdir / "event_physics_vectors_v1_summary.json"

    event_df.to_csv(event_out, index=False, encoding="utf-8")
    sample_out_df.to_csv(sample_out, index=False, encoding="utf-8")
    sample_post_out_df.to_csv(sample_post_out, index=False, encoding="utf-8")
    sample_change_out_df.to_csv(sample_change_out, index=False, encoding="utf-8")
    write_csv(outdir / "sample_manifest.csv", sample_rows, keep_existing_order(sample_rows, []))
    write_csv(outdir / "sample_manifest_post_rgb.csv", sample_rows_post, keep_existing_order(sample_rows_post, []))
    write_csv(outdir / "sample_manifest_change_rgb.csv", sample_rows_change, keep_existing_order(sample_rows_change, []))
    write_csv(outdir / "event_manifest.csv", event_rows, keep_existing_order(event_rows, []))
    build_report(event_df, sample_out_df, report_out)

    summary = {
        "num_events": int(len(event_df)),
        "num_samples": int(len(sample_out_df)),
        "num_samples_post_rgb": int(len(sample_post_out_df)),
        "num_samples_change_rgb": int(len(sample_change_out_df)),
        "terrain_dim": 4,
        "material_dim": 9,
        "trigger_dim_common": 9,
        "trigger_dim_ext": 2,
        "datasets": dict(sorted(Counter(sample_out_df["dataset_id"]).items())),
        "notes": [
            "Vectors are event-level and broadcast to every supervised-ready sample of that event.",
            "Terrain uses CopDEM local-window proxies around the event centroid.",
            "Common strict_t2 triggers are one-hot + CHIRPS + API + SMAP.",
            "DLR-only ERA5 values are stored in trigger_ext_0/1.",
        ],
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"outdir": str(outdir), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
