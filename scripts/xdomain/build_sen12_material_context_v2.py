#!/usr/bin/env python3
"""Build a footprint-scale, label-free Material registry for frozen Sen12.

The builder reads only frozen sample identities and georeferencing. It never
opens segmentation labels, predictions, checkpoints, or evaluation metrics.
Continuous Material sources are summarized over each actual ~1.28 km patch
footprint at their native resolution. LiMW/GLiM polygons are intersected with
the footprint and represented by area proportions, not a center-point class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SAMPLES = 4979
SCHEMA_VERSION = "2.0"
MIN_VALID_COVERAGE = 0.80
MIN_NATIVE_CELLS = 4
NONCONSTANT_EPS = 1e-8

AWC_SPECS = (
    ("awc_0_10", "0..10cm", True),
    ("awc_10_30", "10..30cm", True),
    ("awc_30_60", "30..60cm", True),
    ("awc_60_100", "60..100cm", True),
    ("awc_100_200", "100..200cm", True),
    ("awc_0_200", "0..200cm", False),
)

SOIL_UNITS = {
    "bdod": (100.0, "kg_dm3"),
    "cec": (10.0, "cmolc_kg"),
    "cfvo": (10.0, "percent_volume"),
    "clay": (10.0, "g_kg"),
    "phh2o": (10.0, "pH"),
    "sand": (10.0, "g_kg"),
    "silt": (10.0, "g_kg"),
    "soc": (10.0, "g_kg"),
}

LITHOLOGY_BROAD_CLASSES = (
    "ev", "ig", "mt", "nd", "pa", "pb", "pi", "py",
    "sc", "sm", "ss", "su", "va", "vb", "vi", "wb",
)

GEOREF_COLUMNS = (
    "sample_id",
    "source_id",
    "region",
    "physical_event_cluster_id",
    "crs",
    "min_x",
    "min_y",
    "max_x",
    "max_y",
    "min_lon",
    "min_lat",
    "max_lon",
    "max_lat",
    "center_lon",
    "center_lat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("processed/hybrid_pinn/sen12_context_v2"),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        help="Optional newline-delimited frozen sample IDs for a deterministic smoke run.",
    )
    parser.add_argument("--skip-lithology", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_frozen_samples(
    root: Path, max_samples: int = 0, sample_ids_file: Path | None = None
) -> tuple[pd.DataFrame, dict[str, Path]]:
    cache_path = require_file(
        root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv"
    )
    registry_path = require_file(
        root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
    )
    cache = pd.read_csv(
        cache_path,
        usecols=["sample_id", "physical_event_id", "region_group"],
        low_memory=False,
    )
    registry = pd.read_csv(registry_path, usecols=list(GEOREF_COLUMNS), low_memory=False)
    if cache["sample_id"].duplicated().any() or registry["sample_id"].duplicated().any():
        raise RuntimeError("Frozen cache and georeference registry require unique sample_id")
    frame = cache.merge(registry, on="sample_id", how="left", validate="one_to_one")
    required = [column for column in GEOREF_COLUMNS if column != "sample_id"]
    if frame[required].isna().any().any():
        missing = frame.loc[frame[required].isna().any(axis=1), "sample_id"].head().tolist()
        raise RuntimeError(f"Frozen samples lack complete georeferencing: {missing}")
    mismatch = frame["physical_event_id"].astype(str) != frame[
        "physical_event_cluster_id"
    ].astype(str)
    if mismatch.any():
        raise RuntimeError("Cache and geographic registry disagree on physical event identity")

    if sample_ids_file is not None:
        requested = [line.strip() for line in sample_ids_file.read_text().splitlines() if line.strip()]
        if len(requested) != len(set(requested)):
            raise RuntimeError("sample-ids-file contains duplicate IDs")
        indexed = frame.set_index("sample_id", drop=False)
        missing = sorted(set(requested) - set(indexed.index.astype(str)))
        if missing:
            raise RuntimeError(f"Unknown frozen sample IDs: {missing[:5]}")
        frame = indexed.loc[requested].reset_index(drop=True)
    elif max_samples > 0:
        frame = frame.sort_values("sample_id").head(max_samples).reset_index(drop=True)
    else:
        if len(frame) != EXPECTED_SAMPLES:
            raise RuntimeError(f"Expected {EXPECTED_SAMPLES} frozen samples, found {len(frame)}")
        frame = frame.reset_index(drop=True)
    return frame, {"cache_index": cache_path, "georeference_registry": registry_path}


def build_footprints(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    geometries = []
    for item in frame.itertuples(index=False):
        source_crs = CRS.from_user_input(str(item.crs))
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        xs = [float(item.min_x), float(item.max_x), float(item.max_x), float(item.min_x)]
        ys = [float(item.min_y), float(item.min_y), float(item.max_y), float(item.max_y)]
        lon, lat = transformer.transform(xs, ys)
        polygon = shapely.Polygon(list(zip(lon, lat)))
        if not polygon.is_valid:
            polygon = shapely.make_valid(polygon)
        if polygon.is_empty or polygon.area <= 0:
            raise RuntimeError(f"Invalid footprint geometry for {item.sample_id}")
        geometries.append(polygon)
    output = gpd.GeoDataFrame(frame.copy(), geometry=geometries, crs="EPSG:4326")
    return output


def awc_paths(root: Path) -> dict[str, Path]:
    source_root = require_file(
        root
        / "raw_external/openlandmap_material_v612_250m_zenodo2629148_2784001"
        / "sol_available.water.capacity_usda.mm_m_250m_0..10cm_1950..2017_v0.1.tif"
    ).parent
    paths: dict[str, Path] = {}
    for key, token, _ in AWC_SPECS:
        matches = sorted(source_root.glob(f"*available.water.capacity*_{token}_*.tif"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one OpenLandMap AWC raster for {key}: {matches}")
        paths[key] = matches[0]
    return paths


def soilgrids_paths(root: Path) -> dict[str, Path]:
    vrt_root = require_file(
        root
        / "metadata/pild_xdomain_v1/tmr_support_audit_v1/vrts"
        / "soilgrids_clay_0-5cm_mean.vrt"
    ).parent
    pattern = re.compile(r"soilgrids_([a-z0-9]+)_([0-9]+)-([0-9]+)cm_mean\.vrt$")
    paths: dict[str, Path] = {}
    for path in sorted(vrt_root.glob("soilgrids_*cm_mean.vrt")):
        match = pattern.match(path.name)
        if not match:
            continue
        prop, top, bottom = match.groups()
        if prop not in SOIL_UNITS:
            continue
        paths[f"soil_{prop}_{top}_{bottom}cm"] = path
    expected = len(SOIL_UNITS) * 2
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} SoilGrids VRTs, found {len(paths)}")
    return paths


def intersecting_window(source: rasterio.io.DatasetReader, geometry: Any) -> Window | None:
    left, bottom, right, top = geometry.bounds
    raw = from_bounds(left, bottom, right, top, transform=source.transform)
    col0 = max(0, int(math.floor(raw.col_off)))
    row0 = max(0, int(math.floor(raw.row_off)))
    col1 = min(source.width, int(math.ceil(raw.col_off + raw.width)))
    row1 = min(source.height, int(math.ceil(raw.row_off + raw.height)))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def footprint_stats(
    source: rasterio.io.DatasetReader, geometry: Any
) -> tuple[float, float, float, int, int]:
    window = intersecting_window(source, geometry)
    if window is None:
        return math.nan, math.nan, 0.0, 0, 0
    data = source.read(1, window=window, masked=True)
    transform = source.window_transform(window)
    inside = geometry_mask(
        [mapping(geometry)],
        out_shape=data.shape,
        transform=transform,
        all_touched=True,
        invert=True,
    )
    footprint_count = int(inside.sum())
    if footprint_count == 0:
        return math.nan, math.nan, 0.0, 0, 0
    values = np.asarray(data.data, dtype=np.float64)
    valid = inside & ~np.ma.getmaskarray(data) & np.isfinite(values)
    if source.nodata is not None:
        valid &= values != float(source.nodata)
    valid_values = values[valid]
    valid_count = int(valid_values.size)
    coverage = valid_count / footprint_count
    if valid_count == 0:
        return math.nan, math.nan, coverage, 0, footprint_count
    return (
        float(valid_values.mean()),
        float(valid_values.std(ddof=0)),
        float(coverage),
        valid_count,
        footprint_count,
    )


def transform_footprints(footprints: gpd.GeoDataFrame, target_crs: Any) -> list[Any]:
    return list(footprints.geometry.to_crs(target_crs))


def summarize_raster(
    footprints: gpd.GeoDataFrame,
    path: Path,
    prefix: str,
    divisor: float,
    unit: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with rasterio.open(path) as source:
        if source.crs is None:
            raise RuntimeError(f"Raster has no CRS: {path}")
        projected = transform_footprints(footprints, source.crs)
        ordering = sorted(
            range(len(projected)),
            key=lambda index: (
                rasterio.transform.rowcol(
                    source.transform,
                    projected[index].centroid.x,
                    projected[index].centroid.y,
                )[0],
                index,
            ),
        )
        rows: list[tuple[float, float, float, int, int] | None] = [None] * len(projected)
        for index in ordering:
            rows[index] = footprint_stats(source, projected[index])
        array = np.asarray(rows, dtype=np.float64)
        output = pd.DataFrame(
            {
                f"{prefix}_mean_{unit}": array[:, 0] / divisor,
                f"{prefix}_std_{unit}": array[:, 1] / divisor,
                f"{prefix}_valid_coverage": array[:, 2],
                f"{prefix}_native_cell_count": array[:, 3].astype(np.int64),
                f"{prefix}_footprint_cell_count": array[:, 4].astype(np.int64),
            }
        )
        metadata = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path) if path.suffix.lower() != ".vrt" else sha256(path),
            "crs": str(source.crs),
            "resolution": [float(abs(source.res[0])), float(abs(source.res[1]))],
            "nodata": source.nodata,
            "dtype": source.dtypes[0],
            "unit": unit,
            "raw_divisor": divisor,
            "source_valid": bool(source.width > 0 and source.height > 0),
        }
    return output, metadata


def summarize_continuous_sources(
    root: Path, footprints: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str]]:
    output = pd.DataFrame(index=np.arange(len(footprints)))
    metadata: dict[str, Any] = {}
    model_features: list[str] = []
    audit_features: list[str] = []

    awc = awc_paths(root)
    for key, _, model_eligible in AWC_SPECS:
        print(f"[material-v2] footprint raster {key} ({len(footprints)} samples)", flush=True)
        stats, info = summarize_raster(footprints, awc[key], key, 1.0, "mm")
        output = pd.concat([output, stats], axis=1)
        metadata[key] = {**info, "family": "OpenLandMap_AWC", "model_eligible": model_eligible}
        mean = f"{key}_mean_mm"
        std = f"{key}_std_mm"
        (model_features if model_eligible else audit_features).extend([mean, std])
        audit_features.extend(
            [
                f"{key}_valid_coverage",
                f"{key}_native_cell_count",
                f"{key}_footprint_cell_count",
            ]
        )

    soil = soilgrids_paths(root)
    for key, path in soil.items():
        print(f"[material-v2] footprint raster {key} ({len(footprints)} samples)", flush=True)
        prop = key.split("_")[1]
        divisor, unit = SOIL_UNITS[prop]
        stats, info = summarize_raster(footprints, path, key, divisor, unit)
        output = pd.concat([output, stats], axis=1)
        metadata[key] = {**info, "family": "SoilGrids_2.0", "model_eligible": True}
        model_features.extend([f"{key}_mean_{unit}", f"{key}_std_{unit}"])
        audit_features.extend(
            [
                f"{key}_valid_coverage",
                f"{key}_native_cell_count",
                f"{key}_footprint_cell_count",
            ]
        )
    return output, metadata, model_features, audit_features


def lithology_source_info(path: Path) -> dict[str, Any]:
    info = pyogrio.read_info(path, layer="GLiM_export")
    fields = set(map(str, info.get("fields", [])))
    valid = (
        path.is_file()
        and int(info.get("features", 0)) > 0
        and "Litho" in fields
        and info.get("crs") is not None
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256(path) if path.is_file() else None,
        "layer": "GLiM_export",
        "features": int(info.get("features", 0)),
        "crs": info.get("crs"),
        "fields": sorted(fields),
        "source_valid": bool(valid),
        "family": "LiMW_GLiM",
    }


def summarize_lithology(
    footprints: gpd.GeoDataFrame, path: Path
) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str]]:
    info = lithology_source_info(path)
    if not info["source_valid"]:
        raise RuntimeError(f"LiMW/GLiM source failed validation: {info}")
    projected = footprints[["sample_id", "region", "geometry"]].to_crs(info["crs"])
    n = len(projected)
    areas_by_class: list[defaultdict[str, float]] = [defaultdict(float) for _ in range(n)]
    detailed_by_class: list[defaultdict[str, float]] = [defaultdict(float) for _ in range(n)]
    candidate_counts = np.zeros(n, dtype=np.int64)
    patch_areas = projected.geometry.area.to_numpy(dtype=np.float64)

    for _, region_group in projected.groupby("region", sort=True):
        print(
            f"[material-v2] lithology region={region_group['region'].iloc[0]} "
            f"samples={len(region_group)}",
            flush=True,
        )
        region_indices = region_group.index.to_numpy(dtype=np.int64)
        bounds = tuple(map(float, region_group.total_bounds))
        geology = pyogrio.read_dataframe(
            path,
            layer="GLiM_export",
            bbox=bounds,
            columns=["Litho"],
        )
        if geology.empty:
            continue
        geology = geology[geology.geometry.notna() & ~geology.geometry.is_empty].reset_index(drop=True)
        if geology.empty:
            continue
        invalid = ~geology.geometry.is_valid
        if invalid.any():
            geology.loc[invalid, "geometry"] = geology.loc[invalid, "geometry"].map(
                shapely.make_valid
            )
        lith_geometries = geology.geometry.to_numpy()
        tree = STRtree(lith_geometries)
        patch_geometries = region_group.geometry.to_numpy()
        pairs = tree.query(patch_geometries, predicate="intersects")
        if pairs.size == 0:
            continue
        local_patch_indices, lith_indices = pairs
        intersections = shapely.intersection(
            patch_geometries[local_patch_indices], lith_geometries[lith_indices]
        )
        intersection_areas = shapely.area(intersections)
        lith_codes = geology["Litho"].fillna("nd____").astype(str).to_numpy()
        for local_patch, lith_index, area in zip(
            local_patch_indices, lith_indices, intersection_areas
        ):
            if not np.isfinite(area) or area <= 0:
                continue
            global_index = int(region_indices[int(local_patch)])
            code = lith_codes[int(lith_index)]
            broad = code[:2].lower()
            if broad not in LITHOLOGY_BROAD_CLASSES:
                broad = "nd"
            areas_by_class[global_index][broad] += float(area)
            detailed_by_class[global_index][code] += float(area)
            candidate_counts[global_index] += 1

    rows: list[dict[str, Any]] = []
    for index in range(n):
        broad_areas = areas_by_class[index]
        detailed_areas = detailed_by_class[index]
        covered_area = float(sum(broad_areas.values()))
        patch_area = float(patch_areas[index])
        coverage = min(1.0, covered_area / patch_area) if patch_area > 0 else 0.0
        total = covered_area
        fractions = {
            code: (broad_areas.get(code, 0.0) / total if total > 0 else 0.0)
            for code in LITHOLOGY_BROAD_CLASSES
        }
        positive = np.asarray([value for value in fractions.values() if value > 0], dtype=float)
        entropy = float(-(positive * np.log(positive)).sum()) if len(positive) else math.nan
        normalized_entropy = (
            entropy / math.log(len(positive)) if len(positive) > 1 else 0.0 if len(positive) == 1 else math.nan
        )
        dominant = max(fractions, key=fractions.get) if total > 0 else ""
        detailed = {
            code: area / total
            for code, area in sorted(detailed_areas.items())
            if total > 0 and area > 0
        }
        row: dict[str, Any] = {
            "limw_valid_coverage": coverage,
            "limw_polygon_candidate_count": int(candidate_counts[index]),
            "limw_broad_class_count": int(len(positive)),
            "limw_dominant_broad_class": dominant,
            "limw_entropy": entropy,
            "limw_normalized_entropy": normalized_entropy,
            "limw_detailed_proportions_json": json.dumps(detailed, sort_keys=True),
        }
        row.update({f"limw_frac_{code}": value for code, value in fractions.items()})
        rows.append(row)
    output = pd.DataFrame(rows)
    model_features = [f"limw_frac_{code}" for code in LITHOLOGY_BROAD_CLASSES]
    model_features.append("limw_normalized_entropy")
    audit_features = [
        "limw_valid_coverage",
        "limw_polygon_candidate_count",
        "limw_broad_class_count",
        "limw_dominant_broad_class",
        "limw_entropy",
        "limw_detailed_proportions_json",
    ]
    return output, info, model_features, audit_features


def add_quality_gates(
    frame: pd.DataFrame,
    raster_metadata: dict[str, Any],
    lithology_available: bool,
) -> None:
    awc_primary = [key for key, _, eligible in AWC_SPECS if eligible]
    soil_keys = sorted(key for key in raster_metadata if key.startswith("soil_"))
    awc_coverage = frame[[f"{key}_valid_coverage" for key in awc_primary]].min(axis=1)
    awc_cells = frame[[f"{key}_native_cell_count" for key in awc_primary]].min(axis=1)
    awc_variable = (
        frame[[f"{key}_std_mm" for key in awc_primary]].fillna(0).abs() > NONCONSTANT_EPS
    ).sum(axis=1)
    soil_coverage = frame[[f"{key}_valid_coverage" for key in soil_keys]].min(axis=1)
    soil_cells = frame[[f"{key}_native_cell_count" for key in soil_keys]].min(axis=1)
    soil_std = [column for column in frame if column.startswith("soil_") and "_std_" in column]
    soil_variable = (frame[soil_std].fillna(0).abs() > NONCONSTANT_EPS).sum(axis=1)

    awc_source_valid = all(raster_metadata[key]["source_valid"] for key in awc_primary)
    soil_source_valid = all(raster_metadata[key]["source_valid"] for key in soil_keys)
    frame["awc_min_valid_coverage"] = awc_coverage
    frame["awc_min_native_cell_count"] = awc_cells.astype(int)
    frame["awc_nonconstant_feature_count"] = awc_variable.astype(int)
    frame["soil_min_valid_coverage"] = soil_coverage
    frame["soil_min_native_cell_count"] = soil_cells.astype(int)
    frame["soil_nonconstant_feature_count"] = soil_variable.astype(int)
    frame["q_M_awc"] = (
        awc_source_valid
        & (awc_coverage >= MIN_VALID_COVERAGE)
        & (awc_cells >= MIN_NATIVE_CELLS)
        & (awc_variable >= 1)
    ).astype(float)
    frame["q_M_soilgrids"] = (
        soil_source_valid
        & (soil_coverage >= MIN_VALID_COVERAGE)
        & (soil_cells >= MIN_NATIVE_CELLS)
        & (soil_variable >= 1)
    ).astype(float)
    if lithology_available:
        frame["q_M_lithology"] = (
            (frame["limw_valid_coverage"] >= MIN_VALID_COVERAGE)
            & (frame["limw_polygon_candidate_count"] >= 1)
            & (frame["limw_broad_class_count"] >= 1)
        ).astype(float)
    else:
        frame["q_M_lithology"] = 0.0
    frame["q_M_continuous"] = np.minimum(frame["q_M_awc"], frame["q_M_soilgrids"])
    frame["q_M"] = np.minimum(frame["q_M_continuous"], frame["q_M_lithology"])
    reasons = []
    for row in frame.itertuples(index=False):
        failed = []
        if row.q_M_awc <= 0:
            failed.append("AWC")
        if row.q_M_soilgrids <= 0:
            failed.append("SOILGRIDS")
        if row.q_M_lithology <= 0:
            failed.append("LITHOLOGY")
        reasons.append("PASS" if not failed else "FAIL_" + "+".join(failed))
    frame["q_M_status"] = reasons


def feature_schema(
    frame: pd.DataFrame,
    model_features: list[str],
    audit_features: list[str],
    source_metadata: dict[str, Any],
    lithology_blocker: str | None,
) -> dict[str, Any]:
    model_features = list(dict.fromkeys(model_features))
    audit_features = list(dict.fromkeys(audit_features))
    missing_model = sorted(set(model_features) - set(frame.columns))
    if missing_model:
        raise RuntimeError(f"Model feature schema references missing columns: {missing_model}")
    applicability_fields = sorted(
        dict.fromkeys(
            [
            column
            for column in frame.columns
            if (
                column.endswith("_valid_coverage")
                or column.endswith("_native_cell_count")
                or "_std_" in column
            )
        ]
        + [
            "limw_valid_coverage",
            "limw_polygon_candidate_count",
            "limw_broad_class_count",
            "limw_entropy",
            "limw_normalized_entropy",
            "awc_nonconstant_feature_count",
            "soil_nonconstant_feature_count",
            ]
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": (
            "Footprint-scale Material moderator of Terrain support; not a dense boundary expert"
        ),
        "identity_columns": [
            "sample_id",
            "source_id",
            "region",
            "physical_event_id",
            "physical_event_cluster_id",
        ],
        "model_eligible_features": model_features,
        "model_eligible_dimension": len(model_features),
        "audit_only_features": audit_features,
        "quality_columns": [
            "q_M_awc",
            "q_M_soilgrids",
            "q_M_lithology",
            "q_M_continuous",
            "q_M",
            "q_M_status",
        ],
        "normalization_contract": "Fit numeric transforms on outer-training events only",
        "fail_closed_contract": {
            "minimum_valid_coverage": MIN_VALID_COVERAGE,
            "minimum_native_cells": MIN_NATIVE_CELLS,
            "continuous_nonconstant_epsilon": NONCONSTANT_EPS,
            "q_M": "min(q_M_awc,q_M_soilgrids,q_M_lithology)",
            "missing_source": "q_M=0; never impute source support",
        },
        "local_conditional_effect_contract": {
            "q_M_equals_1": (
                "The patch footprint has jointly valid continuous Material support "
                "(OpenLandMap AWC and SoilGrids) and LiMW/GLiM lithology support under "
                "the registered coverage, native-cell, non-constant, and source-validity gates."
            ),
            "q_M_does_not_mean": (
                "q_M=1 is not evidence that Material improves a prediction, identifies a "
                "landslide, or has a positive local treatment effect."
            ),
            "q_M_equals_0": (
                "Fail closed: downstream Material modulation must be exactly neutral "
                "(m_M=1) and cannot create or redirect a spatial correction."
            ),
            "pre_registered_applicability_fields": applicability_fields,
            "subset_rule": (
                "Any applicability subset must be frozen from these label-free support fields "
                "before reading segmentation labels, predictions, checkpoints, or metrics."
            ),
            "forbidden_selection": (
                "No sample may be retained, removed, weighted, or called applicable using its "
                "segmentation label, prediction error, IoU, AP, uncertainty, or downstream result."
            ),
            "effect_test": (
                "A local conditional effect is a separate downstream aligned-versus-mismatched "
                "intervention test; registry availability alone cannot establish it."
            ),
        },
        "prohibitions": [
            "No segmentation labels or predictions are read",
            "No Material source is upsampled into a 10 m boundary map",
            "No center-point Material value is represented as a footprint mean",
        ],
        "source_metadata": source_metadata,
        "lithology_blocker": lithology_blocker,
    }


def variability_audit(
    frame: pd.DataFrame, model_features: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_column = "physical_event_id"
    source_column = "source_id"
    for feature in model_features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        finite_values = values[finite]
        event_variable = []
        for _, group in frame.assign(_value=values).groupby(event_column, sort=True):
            observed = group["_value"].dropna().to_numpy(dtype=float)
            event_variable.append(bool(len(observed) > 1 and np.nanstd(observed) > NONCONSTANT_EPS))
        source_variable = []
        for _, group in frame.assign(_value=values).groupby(source_column, sort=True):
            observed = group["_value"].dropna().to_numpy(dtype=float)
            source_variable.append(bool(len(observed) > 1 and np.nanstd(observed) > NONCONSTANT_EPS))
        rows.append(
            {
                "feature": feature,
                "finite_fraction": float(finite.mean()),
                "global_mean": float(finite_values.mean()) if len(finite_values) else math.nan,
                "global_std": float(finite_values.std(ddof=0)) if len(finite_values) else math.nan,
                "global_unique_count": int(finite_values.nunique()),
                "within_event_variable_fraction": float(np.mean(event_variable)),
                "within_source_variable_fraction": float(np.mean(source_variable)),
                "eligible_for_model": True,
                "availability_decision": (
                    "MODEL_ELIGIBLE"
                    if finite.mean() >= MIN_VALID_COVERAGE
                    and len(finite_values) > 1
                    and finite_values.std(ddof=0) > NONCONSTANT_EPS
                    else "AUDIT_ONLY_INSUFFICIENT_VARIATION_OR_COVERAGE"
                ),
            }
        )
    audit = pd.DataFrame(rows)
    summary = {
        "features": len(audit),
        "features_with_nonzero_global_variation": int(
            (audit["global_std"].fillna(0) > NONCONSTANT_EPS).sum()
        ),
        "features_model_available": int(audit["availability_decision"].eq("MODEL_ELIGIBLE").sum()),
        "median_within_event_variable_fraction": float(
            audit["within_event_variable_fraction"].median()
        ),
        "median_within_source_variable_fraction": float(
            audit["within_source_variable_fraction"].median()
        ),
    }
    return audit, summary


def source_manifest(inputs: dict[str, Path], metadata: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for role, path in inputs.items():
        rows.append(
            {
                "source_family": "Sen12_frozen_support",
                "asset": role,
                "local_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_valid": True,
                "role": "identity_or_georeference_only",
            }
        )
    for key, info in metadata.items():
        rows.append(
            {
                "source_family": info.get("family"),
                "asset": key,
                "local_path": info.get("path"),
                "bytes": info.get("bytes"),
                "sha256": info.get("sha256"),
                "source_valid": info.get("source_valid"),
                "role": "footprint_material_context",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    outdir = resolve(root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample_ids_file = resolve(root, args.sample_ids_file) if args.sample_ids_file else None
    frame, inputs = load_frozen_samples(root, args.max_samples, sample_ids_file)
    footprints = build_footprints(frame)

    continuous, raster_metadata, model_features, audit_features = summarize_continuous_sources(
        root, footprints
    )
    frame = pd.concat([frame.reset_index(drop=True), continuous.reset_index(drop=True)], axis=1)

    lithology_path = root / "raw_fullcopy/static/lithology/limw.gpkg"
    lithology_blocker: str | None = None
    lithology_available = False
    source_metadata: dict[str, Any] = dict(raster_metadata)
    if args.skip_lithology:
        lithology_blocker = "Explicitly skipped by --skip-lithology; q_M is fail-closed to zero"
    elif not lithology_path.is_file():
        lithology_blocker = f"Missing local LiMW/GLiM asset: {lithology_path}"
    else:
        try:
            lithology, lith_info, lith_model, lith_audit = summarize_lithology(
                footprints, lithology_path
            )
            frame = pd.concat([frame, lithology], axis=1)
            source_metadata["limw_glim"] = lith_info
            model_features.extend(lith_model)
            audit_features.extend(lith_audit)
            lithology_available = True
        except Exception as error:
            lithology_blocker = f"{type(error).__name__}: {error}"
    if not lithology_available:
        for code in LITHOLOGY_BROAD_CLASSES:
            frame[f"limw_frac_{code}"] = math.nan
        frame["limw_normalized_entropy"] = math.nan
        audit_features.extend([f"limw_frac_{code}" for code in LITHOLOGY_BROAD_CLASSES])
        audit_features.append("limw_normalized_entropy")

    add_quality_gates(frame, raster_metadata, lithology_available)
    audit_features.extend(
        [
            "crs", "min_x", "min_y", "max_x", "max_y",
            "min_lon", "min_lat", "max_lon", "max_lat", "center_lon", "center_lat",
            "awc_min_valid_coverage", "awc_min_native_cell_count",
            "awc_nonconstant_feature_count", "soil_min_valid_coverage",
            "soil_min_native_cell_count", "soil_nonconstant_feature_count",
        ]
    )
    schema = feature_schema(
        frame, model_features, audit_features, source_metadata, lithology_blocker
    )
    candidate_model_features = schema["model_eligible_features"]
    variation, variation_summary = variability_audit(frame, candidate_model_features)
    available_features = variation.loc[
        variation["availability_decision"].eq("MODEL_ELIGIBLE"), "feature"
    ].astype(str).tolist()
    unavailable_features = variation.loc[
        ~variation["availability_decision"].eq("MODEL_ELIGIBLE"), "feature"
    ].astype(str).tolist()
    schema["candidate_model_features"] = candidate_model_features
    schema["candidate_model_dimension"] = len(candidate_model_features)
    schema["model_eligible_features"] = available_features
    schema["model_eligible_dimension"] = len(available_features)
    schema["audit_only_features"] = list(
        dict.fromkeys([*schema["audit_only_features"], *unavailable_features])
    )
    schema["availability_downgraded_features"] = unavailable_features

    sample_path = outdir / "material_sample_registry_v2.csv"
    schema_path = outdir / "material_feature_schema_v2.json"
    variation_path = outdir / "material_variability_audit_v2.csv"
    source_path = outdir / "material_source_manifest_v2.csv"
    summary_path = outdir / "summary_v2.json"
    atomic_write_csv(sample_path, frame)
    atomic_write_json(schema_path, schema)
    atomic_write_csv(variation_path, variation)
    manifest = source_manifest(inputs, source_metadata)
    atomic_write_csv(source_path, manifest)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen Sen12 footprint-scale, label-free Material registry",
        "n_samples": len(frame),
        "full_registry": bool(len(frame) == EXPECTED_SAMPLES and args.max_samples <= 0 and sample_ids_file is None),
        "n_sources": int(frame["source_id"].nunique()),
        "n_regions": int(frame["region"].nunique()),
        "n_events": int(frame["physical_event_id"].nunique()),
        "label_data_opened": False,
        "prediction_or_metric_data_opened": False,
        "candidate_model_dimension": len(candidate_model_features),
        "model_eligible_dimension": len(available_features),
        "coverage": {
            "q_M_positive_fraction": float((frame["q_M"] > 0).mean()),
            "q_M_continuous_positive_fraction": float((frame["q_M_continuous"] > 0).mean()),
            "q_M_awc_positive_fraction": float((frame["q_M_awc"] > 0).mean()),
            "q_M_soilgrids_positive_fraction": float((frame["q_M_soilgrids"] > 0).mean()),
            "q_M_lithology_positive_fraction": float((frame["q_M_lithology"] > 0).mean()),
            "q_M_status_counts": frame["q_M_status"].value_counts().to_dict(),
        },
        "native_cell_statistics": {
            "awc_min_native_cell_count": frame["awc_min_native_cell_count"].describe().to_dict(),
            "soil_min_native_cell_count": frame["soil_min_native_cell_count"].describe().to_dict(),
        },
        "variation": variation_summary,
        "lithology": {
            "available": lithology_available,
            "blocker": lithology_blocker,
            "contract": "footprint area proportions over 16 broad GLiM classes",
        },
        "local_conditional_effect_contract": schema["local_conditional_effect_contract"],
        "artifacts": {},
    }
    for name, path in {
        "sample_registry": sample_path,
        "feature_schema": schema_path,
        "variability_audit": variation_path,
        "source_manifest": source_path,
    }.items():
        summary["artifacts"][name] = {"path": str(path), "sha256": sha256(path)}
    atomic_write_json(summary_path, summary)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
