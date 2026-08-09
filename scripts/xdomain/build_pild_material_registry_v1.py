#!/usr/bin/env python3
"""Build and validate the label-free Material registry for 2,937 PILD windows.

The only PILD sample input is the frozen readiness table, read through an
explicit identity/geography whitelist. Segmentation labels, model outputs,
metrics, and visual pixels are never opened. Material remains a coarse
sample/event context moderator and is never represented as a dense boundary
expert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from shapely import make_valid
from shapely.errors import GEOSException


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SAMPLES = 2937
CONTROL_SEED = 20260722
SCHEMA_VERSION = "1.0"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

READINESS_COLUMNS = (
    "sample_id",
    "physical_event_id",
    "dataset_id",
    "source_scene_id",
    "target_crs",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "bbox_top",
    "target_gsd_m",
    "footprint_m",
)
FORBIDDEN_PATH_TOKENS = (
    "label",
    "mask",
    "prediction",
    "metric",
    "checkpoint",
    "logit",
)

AWC_LAYERS = (
    ("0_10", "0..10cm", 100.0),
    ("10_30", "10..30cm", 200.0),
    ("30_60", "30..60cm", 300.0),
    ("60_100", "60..100cm", 400.0),
    ("100_200", "100..200cm", 1000.0),
)
AWC_TOTAL = ("0_200", "0..200cm", 2000.0)
AWC_TOLERANCE_MM = 6.0
SOIL_PROPERTIES = ("clay", "sand", "silt", "cec", "soc", "bdod", "cfvo", "phh2o")
SOIL_DEPTHS = ("0-5cm", "5-15cm")

ARTIFACT_NAMES = {
    "sample_registry": "material_sample_registry_v1.csv",
    "event_registry": "material_event_registry_v1.csv",
    "variation_audit": "material_variation_audit_v1.csv",
    "shuffle_controls": "material_event_shuffle_controls_v1.csv",
    "source_manifest": "material_source_manifest_v1.csv",
    "source_asset_hashes": "material_source_asset_hashes_v1.csv",
    "summary": "material_summary_v1.json",
    "done": "material_DONE_v1.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--readiness",
        type=Path,
        default=Path("processed/hybrid_pinn/pild_prithvi_integration_v1/pild_window_readiness.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("processed/hybrid_pinn/pild_prithvi_integration_v1"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing v1 artifacts without rebuilding or opening source rasters/vectors.",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_safe_file(path: Path) -> Path:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"Refusing prohibited label/model artifact path: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_readiness(path: Path) -> pd.DataFrame:
    require_safe_file(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(READINESS_COLUMNS) - set(header))
    if missing:
        raise RuntimeError(f"Readiness is missing required identity/geography columns: {missing}")
    frame = pd.read_csv(path, usecols=list(READINESS_COLUMNS), keep_default_na=False)
    if tuple(frame.columns) != READINESS_COLUMNS:
        frame = frame.loc[:, READINESS_COLUMNS]
    if len(frame) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLES:,} readiness rows, found {len(frame):,}")
    if frame["sample_id"].eq("").any() or frame["sample_id"].duplicated().any():
        raise RuntimeError("PILD readiness sample_id must be non-empty and unique")
    required_text = ("physical_event_id", "dataset_id", "source_scene_id", "target_crs")
    if frame.loc[:, required_text].eq("").any().any():
        raise RuntimeError("PILD readiness identity fields must be complete")
    numeric = ("bbox_left", "bbox_bottom", "bbox_right", "bbox_top", "target_gsd_m", "footprint_m")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    valid_geometry = (
        frame["bbox_left"].between(-180, 180)
        & frame["bbox_right"].between(-180, 180)
        & frame["bbox_bottom"].between(-90, 90)
        & frame["bbox_top"].between(-90, 90)
        & frame["bbox_right"].gt(frame["bbox_left"])
        & frame["bbox_top"].gt(frame["bbox_bottom"])
        & frame["target_gsd_m"].gt(0)
        & frame["footprint_m"].gt(0)
    )
    if not valid_geometry.all():
        raise RuntimeError(f"Invalid PILD geographic rows: {int((~valid_geometry).sum())}")
    frame["center_lon"] = (frame["bbox_left"] + frame["bbox_right"]) / 2.0
    frame["center_lat"] = (frame["bbox_bottom"] + frame["bbox_top"]) / 2.0
    if any(any(token in column.lower() for token in FORBIDDEN_PATH_TOKENS) for column in frame):
        raise RuntimeError("A prohibited label/model column entered the in-memory readiness frame")
    return frame


def awc_paths(root: Path) -> dict[str, Path]:
    source_root = root / "raw_external/openlandmap_material_v612_250m_zenodo2629148_2784001"
    output: dict[str, Path] = {}
    for key, token, _ in (*AWC_LAYERS, AWC_TOTAL):
        matches = sorted(source_root.glob(f"*available.water.capacity*_{token}_*.tif"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one OpenLandMap AWC raster for {key}, found {matches}")
        output[key] = require_safe_file(matches[0])
    return output


def soil_vrt_paths(root: Path) -> dict[str, Path]:
    vrt_root = root / "metadata/pild_xdomain_v1/tmr_support_audit_v1/vrts"
    output: dict[str, Path] = {}
    for prop in SOIL_PROPERTIES:
        for depth in SOIL_DEPTHS:
            key = f"{prop}_{depth.replace('-', '_')}"
            output[key] = require_safe_file(vrt_root / f"soilgrids_{prop}_{depth}_mean.vrt")
    return output


def projected_footprints(frame: pd.DataFrame, target_crs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    left = frame["bbox_left"].to_numpy(float)
    bottom = frame["bbox_bottom"].to_numpy(float)
    right = frame["bbox_right"].to_numpy(float)
    top = frame["bbox_top"].to_numpy(float)
    corner_lon = np.concatenate([left, left, right, right])
    corner_lat = np.concatenate([bottom, top, bottom, top])
    xs, ys = transformer.transform(corner_lon, corner_lat)
    xs = np.asarray(xs, dtype=float).reshape(4, len(frame)).T
    ys = np.asarray(ys, dtype=float).reshape(4, len(frame)).T
    bounds = np.column_stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)])
    center_x, center_y = transformer.transform(
        frame["center_lon"].to_numpy(float), frame["center_lat"].to_numpy(float)
    )
    return bounds, np.asarray(center_x, dtype=float), np.asarray(center_y, dtype=float)


def footprint_windows(source: rasterio.io.DatasetReader, bounds: np.ndarray) -> list[tuple[int, int, int, int] | None]:
    output: list[tuple[int, int, int, int] | None] = []
    for left, bottom, right, top in bounds:
        raw = from_bounds(left, bottom, right, top, transform=source.transform)
        col0 = max(0, int(math.floor(raw.col_off)))
        row0 = max(0, int(math.floor(raw.row_off)))
        col1 = min(source.width, int(math.ceil(raw.col_off + raw.width)))
        row1 = min(source.height, int(math.ceil(raw.row_off + raw.height)))
        output.append(None if row1 <= row0 or col1 <= col0 else (row0, row1, col0, col1))
    return output


def raster_footprint_stats(
    frame: pd.DataFrame,
    path: Path,
    prefix: str,
    valid_range: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute exact native-cell statistics for every all-touched footprint window."""
    require_safe_file(path)
    with rasterio.open(path) as source:
        bounds, center_x, center_y = projected_footprints(frame, source.crs)
        windows = footprint_windows(source, bounds)
        center_rows, center_cols = rasterio.transform.rowcol(source.transform, center_x, center_y)
        cache: dict[tuple[int, int, int, int], tuple[float, float, float, int, int, int]] = {}

        def summarize(key: tuple[int, int, int, int]) -> tuple[float, float, float, int, int, int]:
            if key in cache:
                return cache[key]
            row0, row1, col0, col1 = key
            array = source.read(1, window=Window(col0, row0, col1 - col0, row1 - row0), masked=True)
            values = np.asarray(array.data, dtype=float)
            valid = ~np.ma.getmaskarray(array) & np.isfinite(values)
            if source.nodata is not None:
                valid &= values != float(source.nodata)
            if valid_range is not None:
                valid &= (values >= valid_range[0]) & (values <= valid_range[1])
            selected = values[valid]
            candidate = int(values.size)
            if selected.size:
                result = (
                    float(selected.mean()),
                    float(selected.std(ddof=0)),
                    float(selected.max() - selected.min()),
                    int(np.unique(selected).size),
                    int(selected.size),
                    candidate,
                )
            else:
                result = (math.nan, math.nan, math.nan, 0, 0, candidate)
            cache[key] = result
            return result

        stats = [
            (math.nan, math.nan, math.nan, 0, 0, 0) if key is None else summarize(key)
            for key in windows
        ]
        values = np.asarray(stats, dtype=float)
        center_coordinates = np.column_stack([center_x, center_y])
        center = np.asarray([item[0] for item in source.sample(center_coordinates)], dtype=float)
        center_valid = np.isfinite(center)
        if source.nodata is not None:
            center_valid &= center != float(source.nodata)
        if valid_range is not None:
            center_valid &= (center >= valid_range[0]) & (center <= valid_range[1])
        center = np.where(center_valid, center, np.nan)
        candidate = values[:, 5]
        output = pd.DataFrame(
            {
                f"{prefix}_center_raw": center,
                f"{prefix}_mean_raw": values[:, 0],
                f"{prefix}_native_cell_std_raw": values[:, 1],
                f"{prefix}_native_cell_range_raw": values[:, 2],
                f"{prefix}_native_cell_unique_count": values[:, 3].astype(int),
                f"{prefix}_valid_native_cell_count": values[:, 4].astype(int),
                f"{prefix}_candidate_native_cell_count": candidate.astype(int),
                f"{prefix}_valid_fraction": np.divide(
                    values[:, 4], candidate, out=np.zeros(len(frame), dtype=float), where=candidate > 0
                ),
            }
        )
        metadata = {
            "path": str(path),
            "crs": str(source.crs),
            "native_resolution": [float(abs(source.res[0])), float(abs(source.res[1]))],
            "native_resolution_units": "degree" if source.crs.is_geographic else "metre",
            "nodata": source.nodata,
            "dtype": source.dtypes[0],
            "unique_footprint_windows": len(cache),
        }
        geometry = pd.DataFrame(
            {
                f"{prefix}_native_row": np.asarray(center_rows, dtype=int),
                f"{prefix}_native_col": np.asarray(center_cols, dtype=int),
                f"{prefix}_native_cell_id": [
                    f"r{row}_c{col}" for row, col in zip(center_rows, center_cols)
                ],
            }
        )
    return pd.concat([geometry, output], axis=1), metadata


def sample_awc(frame: pd.DataFrame, paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for layer_index, (key, _, maximum) in enumerate((*AWC_LAYERS, AWC_TOTAL)):
        sampled, source_metadata = raster_footprint_stats(
            frame, paths[key], f"awc_{key}", valid_range=(0.0, maximum)
        )
        if layer_index:
            sampled = sampled.drop(columns=[f"awc_{key}_native_row", f"awc_{key}_native_col", f"awc_{key}_native_cell_id"])
        else:
            sampled = sampled.rename(
                columns={
                    f"awc_{key}_native_row": "awc_native_row",
                    f"awc_{key}_native_col": "awc_native_col",
                    f"awc_{key}_native_cell_id": "awc_native_cell_id",
                }
            )
        sampled[f"awc_{key}_aligned_mm"] = sampled[f"awc_{key}_center_raw"]
        sampled[f"awc_{key}_footprint_mean_mm"] = sampled[f"awc_{key}_mean_raw"]
        sampled = sampled.drop(columns=[f"awc_{key}_center_raw", f"awc_{key}_mean_raw"])
        parts.append(sampled)
        metadata[key] = source_metadata
    output = pd.concat(parts, axis=1)
    layer_sum = sum(output[f"awc_{key}_footprint_mean_mm"].to_numpy(float) for key, _, _ in AWC_LAYERS)
    total = output[f"awc_{AWC_TOTAL[0]}_footprint_mean_mm"].to_numpy(float)
    output["awc_layer_sum_0_200_mm"] = layer_sum
    output["awc_total_abs_error_mm"] = np.abs(layer_sum - total)
    coverage = np.min(
        np.column_stack(
            [output[f"awc_{key}_valid_fraction"].to_numpy(float) for key, _, _ in (*AWC_LAYERS, AWC_TOTAL)]
        ),
        axis=1,
    )
    consistency = np.isfinite(layer_sum) & np.isfinite(total) & (np.abs(layer_sum - total) <= AWC_TOLERANCE_MM)
    output["q_M_awc_coverage"] = coverage
    output["q_M_awc"] = coverage * consistency.astype(float)
    output["awc_native_resolution_x_degrees"] = metadata[AWC_LAYERS[0][0]]["native_resolution"][0]
    output["awc_native_resolution_y_degrees"] = metadata[AWC_LAYERS[0][0]]["native_resolution"][1]
    return output, metadata


def sample_soilgrids(
    frame: pd.DataFrame, paths: dict[str, Path]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for layer_index, (key, path) in enumerate(paths.items()):
        prefix = f"soil_{key}"
        sampled, source_metadata = raster_footprint_stats(frame, path, prefix)
        if layer_index:
            sampled = sampled.drop(columns=[f"{prefix}_native_row", f"{prefix}_native_col", f"{prefix}_native_cell_id"])
        else:
            sampled = sampled.rename(
                columns={
                    f"{prefix}_native_row": "soilgrids_native_row",
                    f"{prefix}_native_col": "soilgrids_native_col",
                    f"{prefix}_native_cell_id": "soilgrids_native_cell_id",
                }
            )
        parts.append(sampled)
        metadata[key] = source_metadata
    output = pd.concat(parts, axis=1)
    coverage_columns = [f"soil_{key}_valid_fraction" for key in paths]
    output["q_M_soil"] = output[coverage_columns].min(axis=1)
    output["q_M_soilgrids"] = output["q_M_soil"]
    first = metadata[next(iter(paths))]
    output["soilgrids_native_resolution_x_m"] = first["native_resolution"][0]
    output["soilgrids_native_resolution_y_m"] = first["native_resolution"][1]
    return output, metadata


def sample_lithology(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_safe_file(path)
    layers = pyogrio.list_layers(path)
    if len(layers) != 1:
        raise RuntimeError(f"Expected one GLiM layer, found {layers.tolist()}")
    layer = str(layers[0, 0])
    info = pyogrio.read_info(path, layer=layer)
    target_crs = info["crs"]
    output = pd.DataFrame(index=frame.index)
    output["lithology_class"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    output["lithology_candidate_count"] = 0
    output["lithology_native_polygon_count"] = 0
    output["lithology_class_count"] = 0
    output["lithology_valid_fraction"] = 0.0
    output["lithology_dominant_fraction"] = 0.0
    output["lithology_native_cell_variation"] = 0

    for event_id, indices in frame.groupby("physical_event_id", sort=True).groups.items():
        windows = gpd.GeoDataFrame(
            {"row_index": list(indices)},
            geometry=[
                box(row.bbox_left, row.bbox_bottom, row.bbox_right, row.bbox_top)
                for row in frame.loc[indices].itertuples()
            ],
            crs="EPSG:4326",
        ).to_crs(target_crs)
        min_x, min_y, max_x, max_y = windows.total_bounds
        polygons = pyogrio.read_dataframe(
            path,
            layer=layer,
            columns=["Litho"],
            bbox=(min_x - 1.0, min_y - 1.0, max_x + 1.0, max_y + 1.0),
        )
        polygons = polygons.loc[
            polygons["Litho"].notna() & polygons.geometry.notna() & ~polygons.geometry.is_empty
        ].copy()
        invalid = ~polygons.geometry.is_valid
        if invalid.any():
            polygons.loc[invalid, "geometry"] = polygons.loc[invalid, "geometry"].map(make_valid)
            polygons = polygons.loc[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()
        if polygons.empty:
            continue
        joined = gpd.sjoin(
            windows[["row_index", "geometry"]],
            polygons[["Litho", "geometry"]],
            how="left",
            predicate="intersects",
        ).dropna(subset=["index_right", "Litho"])
        areas_by_row: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        polygons_by_row: dict[int, set[int]] = defaultdict(set)
        window_geometry = windows.set_index("row_index").geometry
        for item in joined.itertuples():
            row_index = int(item.row_index)
            polygon_index = int(item.index_right)
            window_shape = window_geometry.loc[row_index]
            polygon_shape = polygons.loc[polygon_index].geometry
            try:
                intersection_area = float(window_shape.intersection(polygon_shape).area)
            except GEOSException:
                intersection_area = float(make_valid(window_shape).intersection(make_valid(polygon_shape)).area)
            if intersection_area <= 0:
                continue
            areas_by_row[row_index][str(item.Litho)] += intersection_area
            polygons_by_row[row_index].add(polygon_index)
        for row_index, class_areas in areas_by_row.items():
            footprint_area = float(window_geometry.loc[row_index].area)
            dominant_class, dominant_area = max(class_areas.items(), key=lambda item: item[1])
            covered_area = min(footprint_area, sum(class_areas.values()))
            output.at[row_index, "lithology_class"] = dominant_class
            output.at[row_index, "lithology_candidate_count"] = len(polygons_by_row[row_index])
            output.at[row_index, "lithology_native_polygon_count"] = len(polygons_by_row[row_index])
            output.at[row_index, "lithology_class_count"] = len(class_areas)
            output.at[row_index, "lithology_valid_fraction"] = covered_area / max(footprint_area, 1e-12)
            output.at[row_index, "lithology_dominant_fraction"] = dominant_area / max(footprint_area, 1e-12)
            output.at[row_index, "lithology_native_cell_variation"] = int(len(class_areas) > 1)
        print(
            f"[GLiM] event={event_id} samples={len(indices)} candidates={len(polygons)}",
            flush=True,
        )
    for column in (
        "lithology_candidate_count",
        "lithology_native_polygon_count",
        "lithology_class_count",
        "lithology_native_cell_variation",
    ):
        output[column] = output[column].astype(int)
    output["q_M_lithology"] = output["lithology_valid_fraction"].clip(0, 1)
    output["q_M_geology"] = output["q_M_lithology"]
    output["lithology_native_scale"] = "GLiM_polygon_map_approximately_1_to_1M"
    metadata = {
        "path": str(path),
        "layer": layer,
        "crs": str(target_crs),
        "feature_count": int(info["features"]),
        "native_scale": "categorical polygon map, approximately 1:1,000,000 target scale",
    }
    return output, metadata


def apply_material_quality(frame: pd.DataFrame) -> None:
    frame["q_M_hydraulic"] = np.minimum(frame["q_M_awc"], frame["q_M_soilgrids"])
    frame["q_M"] = np.maximum(frame["q_M_hydraulic"], frame["q_M_geology"])
    frame["q_M_full"] = np.minimum(frame["q_M_hydraulic"], frame["q_M_geology"])
    frame["material_multiplier_neutral"] = 1.0
    frame["material_multiplier_min_allowed"] = 0.75
    frame["material_multiplier_max_allowed"] = 1.25
    frame["material_scientific_role"] = "context_moderator_only"


def event_shuffle_controls(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(CONTROL_SEED)
    rows: list[dict[str, Any]] = []
    groups = {name: group for name, group in frame.groupby("dataset_id", sort=True)}
    for item in frame.itertuples(index=False):
        candidates = groups[str(item.dataset_id)]
        candidates = candidates[candidates["physical_event_id"].astype(str) != str(item.physical_event_id)]
        if candidates.empty:
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "event_shuffle_status": "ABSTAIN_NO_WITHIN_DATASET_ALTERNATE_EVENT",
                    "donor_sample_id": pd.NA,
                    "donor_event_id": pd.NA,
                }
            )
            continue
        donor = candidates.iloc[int(rng.integers(0, len(candidates)))]
        rows.append(
            {
                "sample_id": item.sample_id,
                "event_shuffle_status": "VALID_WITHIN_DATASET_CROSS_EVENT",
                "donor_sample_id": donor["sample_id"],
                "donor_event_id": donor["physical_event_id"],
            }
        )
    return pd.DataFrame(rows)


def grouped_source_accuracy(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    values = frame[columns].to_numpy(dtype=float)
    labels = frame["dataset_id"].astype(str).to_numpy()
    event_ids = frame["physical_event_id"].astype(str).to_numpy()
    folds = np.asarray(
        [int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 5 for value in event_ids], dtype=int
    )
    predictions = np.empty(len(frame), dtype=object)
    for held_out in range(5):
        train = folds != held_out
        test = ~train
        if not test.any():
            continue
        median = np.nanmedian(values[train], axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled_train = np.where(np.isfinite(values[train]), values[train], median)
        filled_test = np.where(np.isfinite(values[test]), values[test], median)
        q25, q75 = np.nanpercentile(filled_train, [25, 75], axis=0)
        scale = np.where(np.isfinite(q75 - q25) & (q75 - q25 > 1e-8), q75 - q25, 1.0)
        train_z = (filled_train - median) / scale
        test_z = (filled_test - median) / scale
        classes = sorted(set(labels[train]))
        centroids = np.stack([train_z[labels[train] == label].mean(axis=0) for label in classes])
        distance = ((test_z[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
        predictions[test] = np.asarray(classes)[np.argmin(distance, axis=1)]
    accuracy = float(np.mean(predictions == labels))
    recalls = [float(np.mean(predictions[labels == label] == label)) for label in sorted(set(labels))]
    baseline = float(pd.Series(labels).value_counts(normalize=True).max())
    return {
        "event_grouped_fivefold_accuracy": accuracy,
        "event_grouped_fivefold_balanced_accuracy": float(np.mean(recalls)),
        "majority_source_accuracy": baseline,
    }


def variation_audit(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        total_variance = float(values.var(ddof=0))
        dataset_means = frame.assign(_value=values).groupby("dataset_id")["_value"].transform("mean")
        event_means = frame.assign(_value=values).groupby("physical_event_id")["_value"].transform("mean")
        rows.append(
            {
                "feature": column,
                "coverage_fraction": float(values.notna().mean()),
                "overall_std": float(values.std(ddof=0)),
                "unique_values": int(values.nunique(dropna=True)),
                "native_cell_nonzero_variation_fraction": float(
                    (pd.to_numeric(frame[column.replace("_mean_raw", "_native_cell_std_raw")], errors="coerce") > 0).mean()
                    if column.endswith("_mean_raw")
                    and column.replace("_mean_raw", "_native_cell_std_raw") in frame
                    else math.nan
                ),
                "variable_dataset_fraction": float(
                    (frame.assign(_value=values).groupby("dataset_id")["_value"].nunique(dropna=True) > 1).mean()
                ),
                "variable_event_fraction": float(
                    (frame.assign(_value=values).groupby("physical_event_id")["_value"].nunique(dropna=True) > 1).mean()
                ),
                "between_dataset_variance_fraction": (
                    float(dataset_means.var(ddof=0) / total_variance) if total_variance > 0 else math.nan
                ),
                "between_event_variance_fraction": (
                    float(event_means.var(ddof=0) / total_variance) if total_variance > 0 else math.nan
                ),
            }
        )
    audit = pd.DataFrame(rows)
    finite_dataset = audit["between_dataset_variance_fraction"].replace([np.inf, -np.inf], np.nan).dropna()
    summary = {
        "features": len(audit),
        "features_with_nonzero_variation": int((audit["overall_std"] > 0).sum()),
        "median_native_cell_nonzero_variation_fraction": float(
            audit["native_cell_nonzero_variation_fraction"].median()
        ),
        "median_variable_event_fraction": float(audit["variable_event_fraction"].median()),
        "median_between_dataset_variance_fraction": float(finite_dataset.median()),
    }
    return audit, summary


def build_event_registry(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_id, group in frame.groupby("physical_event_id", sort=True):
        rows.append(
            {
                "physical_event_id": event_id,
                "dataset_id": ";".join(sorted(group["dataset_id"].astype(str).unique())),
                "source_scene_ids": int(group["source_scene_id"].nunique()),
                "n_samples": len(group),
                "q_M_mean": float(group["q_M"].mean()),
                "q_M_positive_fraction": float(group["q_M"].gt(0).mean()),
                "q_M_full_mean": float(group["q_M_full"].mean()),
                "awc_center_native_cells": int(group["awc_native_cell_id"].nunique()),
                "soilgrids_center_native_cells": int(group["soilgrids_native_cell_id"].nunique()),
                "lithology_classes": int(group["lithology_class"].nunique(dropna=True)),
                "lithology_boundary_crossing_fraction": float(
                    group["lithology_native_cell_variation"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def file_list_for_vrt(vrt: Path) -> list[Path]:
    list_path = vrt.with_suffix(".files.txt")
    require_safe_file(list_path)
    paths = [Path(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths:
        raise RuntimeError(f"Empty source list for {vrt}")
    return [require_safe_file(path) for path in paths]


def hash_asset_rows(paths: Iterable[tuple[str, str, Path, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_family, asset_group, path, verification in paths:
        rows.append(
            {
                "source_family": source_family,
                "asset_group": asset_group,
                "local_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "verification": verification,
            }
        )
    return pd.DataFrame(rows)


def aggregate_hash(rows: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for item in rows.sort_values("local_path").itertuples(index=False):
        digest.update(f"{item.local_path}\t{item.bytes}\t{item.sha256}\n".encode())
    return digest.hexdigest()


def build_source_manifests(
    root: Path,
    readiness_path: Path,
    awc: dict[str, Path],
    awc_metadata: dict[str, Any],
    soil: dict[str, Path],
    soil_metadata: dict[str, Any],
    lithology_path: Path,
    lithology_metadata: dict[str, Any],
    outdir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fetch_path = require_safe_file(
        root
        / "metadata/reports/dlr_openlandmap_hydraulic_v6_1_2_fetch_20260717/source_file_manifest.csv"
    )
    fetch = pd.read_csv(fetch_path, low_memory=False)
    source_rows: list[dict[str, Any]] = []
    asset_frames: list[pd.DataFrame] = []
    for key, path in awc.items():
        matched = fetch[fetch["local_path"].astype(str) == str(path)]
        if len(matched) != 1:
            raise RuntimeError(f"No unique verified OpenLandMap source row for {path}")
        item = matched.iloc[0]
        if int(item["bytes"]) != path.stat().st_size or not HASH_RE.fullmatch(str(item["local_sha256"])):
            raise RuntimeError(f"OpenLandMap manifest mismatch for {path}")
        source_rows.append(
            {
                "source_family": "OpenLandMap_AWC",
                "role": "primary_hydraulic_context" if key != AWC_TOTAL[0] else "profile_diagnostic",
                "asset": key,
                "native_resolution": "0.002083333 degrees, approximately 250 m",
                "valid_coverage_field": f"awc_{key}_valid_fraction",
                "variation_field": f"awc_{key}_native_cell_std_raw",
                "local_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": str(item["local_sha256"]),
                "hash_basis": "verified_download_manifest_and_size_recheck",
                "license": "CC-BY-NC-SA-4.0",
                "metadata": json.dumps(awc_metadata[key], sort_keys=True),
            }
        )
        asset_frames.append(
            pd.DataFrame(
                [
                    {
                        "source_family": "OpenLandMap_AWC",
                        "asset_group": key,
                        "local_path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": str(item["local_sha256"]),
                        "verification": "verified_download_manifest_and_size_recheck",
                    }
                ]
            )
        )

    for key, vrt in soil.items():
        tile_rows = hash_asset_rows(
            ("SoilGrids_2.0", key, path, "local_full_file_sha256") for path in file_list_for_vrt(vrt)
        )
        asset_frames.append(tile_rows)
        source_rows.append(
            {
                "source_family": "SoilGrids_2.0",
                "role": "soil_property_context",
                "asset": key,
                "native_resolution": "250 m",
                "valid_coverage_field": f"soil_{key}_valid_fraction",
                "variation_field": f"soil_{key}_native_cell_std_raw",
                "local_path": str(vrt),
                "bytes": int(tile_rows["bytes"].sum()),
                "sha256": aggregate_hash(tile_rows),
                "hash_basis": f"aggregate_of_{len(tile_rows)}_per_tile_sha256_hashes",
                "license": "CC-BY-4.0",
                "metadata": json.dumps(
                    {
                        **soil_metadata[key],
                        "vrt_sha256": sha256(vrt),
                        "file_list_sha256": sha256(vrt.with_suffix(".files.txt")),
                    },
                    sort_keys=True,
                ),
            }
        )

    direct_assets = hash_asset_rows(
        [
            ("LiMW_GLiM", "limw_glim", lithology_path, "local_full_file_sha256"),
            ("PILD_readiness", "identity_geography_whitelist", readiness_path, "local_full_file_sha256"),
            ("OpenLandMap_AWC", "verified_fetch_manifest", fetch_path, "local_full_file_sha256"),
        ]
    )
    asset_frames.append(direct_assets)
    lithology_asset = direct_assets[direct_assets["asset_group"].eq("limw_glim")].iloc[0]
    readiness_asset = direct_assets[
        direct_assets["asset_group"].eq("identity_geography_whitelist")
    ].iloc[0]
    source_rows.extend(
        [
            {
                "source_family": "LiMW_GLiM",
                "role": "polygon_lithology_context",
                "asset": "limw.gpkg",
                "native_resolution": "polygon map, approximately 1:1,000,000 target scale",
                "valid_coverage_field": "lithology_valid_fraction",
                "variation_field": "lithology_native_cell_variation",
                "local_path": str(lithology_path),
                "bytes": int(lithology_asset["bytes"]),
                "sha256": lithology_asset["sha256"],
                "hash_basis": "local_full_file_sha256",
                "license": "source_terms_apply",
                "metadata": json.dumps(lithology_metadata, sort_keys=True),
            },
            {
                "source_family": "PILD_readiness",
                "role": "sample_identity_and_geography_only",
                "asset": readiness_path.name,
                "native_resolution": "frozen_128x128_window_registry",
                "valid_coverage_field": "not_applicable",
                "variation_field": "not_applicable",
                "local_path": str(readiness_path),
                "bytes": int(readiness_asset["bytes"]),
                "sha256": readiness_asset["sha256"],
                "hash_basis": "local_full_file_sha256",
                "license": "project_internal_provenance",
                "metadata": "Only READINESS_COLUMNS identity/geography whitelist was loaded",
            },
        ]
    )
    source_manifest = pd.DataFrame(source_rows)
    asset_hashes = pd.concat(asset_frames, ignore_index=True)
    atomic_csv(source_manifest, outdir / ARTIFACT_NAMES["source_manifest"])
    atomic_csv(asset_hashes, outdir / ARTIFACT_NAMES["source_asset_hashes"])
    return source_manifest, asset_hashes


def validate_frame_contract(readiness: pd.DataFrame, frame: pd.DataFrame) -> None:
    if len(frame) != EXPECTED_SAMPLES or frame["sample_id"].duplicated().any():
        raise RuntimeError("Material sample registry is not one-to-one for 2,937 PILD samples")
    if frame["sample_id"].astype(str).tolist() != readiness["sample_id"].astype(str).tolist():
        raise RuntimeError("Material sample_id order/set does not exactly match PILD readiness")
    required = {
        "q_M_awc",
        "q_M_soilgrids",
        "q_M_geology",
        "q_M_hydraulic",
        "q_M",
        "q_M_full",
        "awc_native_resolution_x_degrees",
        "soilgrids_native_resolution_x_m",
        "lithology_native_scale",
        "material_scientific_role",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Material registry missing contract columns: {missing}")
    forbidden = [
        column for column in frame.columns if any(token in column.lower() for token in FORBIDDEN_PATH_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"Material registry contains prohibited label/model columns: {forbidden}")
    quality_columns = ["q_M_awc", "q_M_soilgrids", "q_M_geology", "q_M_hydraulic", "q_M", "q_M_full"]
    quality = frame[quality_columns].apply(pd.to_numeric, errors="raise")
    if not ((quality >= -1e-12) & (quality <= 1 + 1e-12)).all().all():
        raise RuntimeError("Material quality values must stay in [0, 1]")
    hydraulic = np.minimum(quality["q_M_awc"], quality["q_M_soilgrids"])
    expected_q = np.maximum(hydraulic, quality["q_M_geology"])
    expected_full = np.minimum(hydraulic, quality["q_M_geology"])
    if not np.allclose(quality["q_M_hydraulic"], hydraulic, atol=1e-12):
        raise RuntimeError("q_M_hydraulic formula mismatch")
    if not np.allclose(quality["q_M"], expected_q, atol=1e-12):
        raise RuntimeError("q_M formula mismatch")
    if not np.allclose(quality["q_M_full"], expected_full, atol=1e-12):
        raise RuntimeError("q_M_full formula mismatch")
    if not frame["material_scientific_role"].eq("context_moderator_only").all():
        raise RuntimeError("Material role must remain context_moderator_only")
    if not np.allclose(frame["awc_native_resolution_x_degrees"], 0.002083333, atol=1e-9):
        raise RuntimeError("Unexpected OpenLandMap native resolution")
    if not np.allclose(frame["soilgrids_native_resolution_x_m"], 250.0, atol=1e-9):
        raise RuntimeError("Unexpected SoilGrids native resolution")


def validate_outputs(root: Path, readiness_path: Path, outdir: Path) -> dict[str, Any]:
    readiness = load_readiness(readiness_path)
    paths = {key: outdir / name for key, name in ARTIFACT_NAMES.items()}
    for key, path in paths.items():
        if key != "done" or path.exists():
            require_safe_file(path)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    done = json.loads(paths["done"].read_text(encoding="utf-8"))
    frame = pd.read_csv(paths["sample_registry"], low_memory=False)
    validate_frame_contract(readiness, frame)
    events = pd.read_csv(paths["event_registry"], low_memory=False)
    variation = pd.read_csv(paths["variation_audit"], low_memory=False)
    shuffle = pd.read_csv(paths["shuffle_controls"], low_memory=False)
    sources = pd.read_csv(paths["source_manifest"], low_memory=False)
    assets = pd.read_csv(paths["source_asset_hashes"], low_memory=False)
    if events["n_samples"].sum() != EXPECTED_SAMPLES or events["physical_event_id"].duplicated().any():
        raise RuntimeError("Event registry does not partition all PILD samples exactly once")
    if shuffle["sample_id"].astype(str).tolist() != readiness["sample_id"].astype(str).tolist():
        raise RuntimeError("Event-shuffle controls do not match PILD sample identity/order")
    identity = frame.set_index("sample_id")[["dataset_id", "physical_event_id"]]
    valid_shuffle = shuffle["event_shuffle_status"].eq("VALID_WITHIN_DATASET_CROSS_EVENT")
    donors = shuffle.loc[valid_shuffle, "donor_sample_id"].astype(str)
    if not donors.isin(identity.index.astype(str)).all():
        raise RuntimeError("Event-shuffle controls reference unknown donor samples")
    recipients = identity.loc[shuffle.loc[valid_shuffle, "sample_id"].astype(str)]
    donor_identity = identity.loc[donors]
    if not np.array_equal(
        recipients["dataset_id"].astype(str).to_numpy(),
        donor_identity["dataset_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Event-shuffle donor must come from the same dataset")
    if np.any(
        recipients["physical_event_id"].astype(str).to_numpy()
        == donor_identity["physical_event_id"].astype(str).to_numpy()
    ):
        raise RuntimeError("Event-shuffle donor must come from a different physical event")
    if variation.empty or variation["feature"].duplicated().any():
        raise RuntimeError("Variation audit is empty or duplicated")
    required_families = {"OpenLandMap_AWC", "SoilGrids_2.0", "LiMW_GLiM", "PILD_readiness"}
    if not required_families.issubset(set(sources["source_family"])):
        raise RuntimeError("Source manifest is missing a required Material source family")
    if not sources["sha256"].astype(str).map(lambda value: bool(HASH_RE.fullmatch(value))).all():
        raise RuntimeError("Source manifest contains an invalid SHA-256")
    if not assets["sha256"].astype(str).map(lambda value: bool(HASH_RE.fullmatch(value))).all():
        raise RuntimeError("Source asset hash registry contains an invalid SHA-256")
    if assets["local_path"].duplicated().any():
        raise RuntimeError("Source asset hash registry contains duplicate local paths")
    for item in assets.itertuples(index=False):
        asset_path = require_safe_file(Path(item.local_path))
        if asset_path.stat().st_size != int(item.bytes):
            raise RuntimeError(f"Source asset byte-size drift: {asset_path}")
    for item in sources.itertuples(index=False):
        grouped_assets = assets[assets["asset_group"].astype(str).eq(str(item.asset))]
        if item.source_family == "SoilGrids_2.0":
            if grouped_assets.empty or aggregate_hash(grouped_assets) != str(item.sha256):
                raise RuntimeError(f"SoilGrids aggregate hash registry mismatch: {item.asset}")
        elif item.source_family == "OpenLandMap_AWC":
            if len(grouped_assets) != 1 or grouped_assets.iloc[0]["sha256"] != str(item.sha256):
                raise RuntimeError(f"OpenLandMap hash registry mismatch: {item.asset}")
    if any(
        any(token in str(path).lower() for token in FORBIDDEN_PATH_TOKENS)
        for path in sources["local_path"]
    ):
        raise RuntimeError("Source manifest references prohibited label/model artifacts")
    if summary.get("n_samples") != EXPECTED_SAMPLES:
        raise RuntimeError("Summary sample count mismatch")
    if summary.get("label_data_opened") is not False or summary.get("prediction_or_metric_data_opened") is not False:
        raise RuntimeError("Summary does not certify the label/model isolation boundary")
    if summary.get("material_contract", {}).get("scientific_role") != "context_moderator_only":
        raise RuntimeError("Summary Material role contract mismatch")
    if summary.get("sample_identity_contract", {}).get("source_sha256") != sha256(readiness_path):
        raise RuntimeError("Readiness identity source hash drift")
    for key in (
        "sample_registry",
        "event_registry",
        "variation_audit",
        "shuffle_controls",
        "source_manifest",
        "source_asset_hashes",
    ):
        expected = summary["artifacts"][f"{key}_sha256"]
        observed = sha256(paths[key])
        if expected != observed:
            raise RuntimeError(f"Artifact hash mismatch for {key}: {observed} != {expected}")
    if done.get("status") != "complete" or done.get("summary_sha256") != sha256(paths["summary"]):
        raise RuntimeError("DONE sentinel is incomplete or stale")
    return {
        "status": "PASS",
        "samples": len(frame),
        "events": len(events),
        "source_manifest_rows": len(sources),
        "source_asset_hash_rows": len(assets),
        "q_M_positive_fraction": float(frame["q_M"].gt(0).mean()),
        "q_M_full_positive_fraction": float(frame["q_M_full"].gt(0).mean()),
        "summary_sha256": sha256(paths["summary"]),
    }


def build(root: Path, readiness_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_readiness(readiness_path)
    awc_source_paths = awc_paths(root)
    soil_source_paths = soil_vrt_paths(root)
    lithology_path = require_safe_file(root / "raw_fullcopy/static/lithology/limw.gpkg")

    awc, awc_metadata = sample_awc(frame, awc_source_paths)
    frame = pd.concat([frame.reset_index(drop=True), awc.reset_index(drop=True)], axis=1)
    soil, soil_metadata = sample_soilgrids(frame, soil_source_paths)
    frame = pd.concat([frame, soil.reset_index(drop=True)], axis=1)
    lithology, lithology_metadata = sample_lithology(frame, lithology_path)
    frame = pd.concat([frame, lithology.reset_index(drop=True)], axis=1)
    apply_material_quality(frame)
    validate_frame_contract(load_readiness(readiness_path), frame)

    numeric_features = [f"awc_{key}_footprint_mean_mm" for key, _, _ in AWC_LAYERS]
    numeric_features.extend(f"soil_{key}_mean_raw" for key in soil_source_paths)
    variation, variation_summary = variation_audit(frame, numeric_features)
    shortcut = grouped_source_accuracy(frame, numeric_features)
    shortcut_risk = (
        "HIGH"
        if shortcut["event_grouped_fivefold_balanced_accuracy"] >= 0.8
        or variation_summary["median_between_dataset_variance_fraction"] >= 0.5
        else "MODERATE"
    )
    shuffle = event_shuffle_controls(frame)
    events = build_event_registry(frame)

    sample_path = outdir / ARTIFACT_NAMES["sample_registry"]
    event_path = outdir / ARTIFACT_NAMES["event_registry"]
    variation_path = outdir / ARTIFACT_NAMES["variation_audit"]
    shuffle_path = outdir / ARTIFACT_NAMES["shuffle_controls"]
    atomic_csv(frame, sample_path)
    atomic_csv(events, event_path)
    atomic_csv(variation, variation_path)
    atomic_csv(shuffle, shuffle_path)
    source_manifest, asset_hashes = build_source_manifests(
        root,
        readiness_path,
        awc_source_paths,
        awc_metadata,
        soil_source_paths,
        soil_metadata,
        lithology_path,
        lithology_metadata,
        outdir,
    )

    paths = {key: outdir / name for key, name in ARTIFACT_NAMES.items()}
    coverage_pass = bool(frame["q_M"].gt(0).mean() >= 0.9)
    variation_pass = bool(
        variation_summary["features_with_nonzero_variation"] == len(numeric_features)
        and variation_summary["median_variable_event_fraction"] >= 0.8
    )
    deployment = "PASS_CONTEXT_ONLY" if coverage_pass and variation_pass else "ABSTAIN"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen 2,937-sample PILD readiness; label-free native-scale Material registration",
        "n_samples": len(frame),
        "n_datasets": int(frame["dataset_id"].nunique()),
        "n_events": int(frame["physical_event_id"].nunique()),
        "sample_identity_contract": {
            "source": str(readiness_path),
            "source_sha256": sha256(readiness_path),
            "loaded_columns": list(READINESS_COLUMNS),
            "ordered_one_to_one": True,
        },
        "label_data_opened": False,
        "prediction_or_metric_data_opened": False,
        "native_scale_contract": {
            "openlandmap_awc": "0.002083333 degrees, approximately 250 m",
            "soilgrids": "250 m Interrupted Goode Homolosine native cells",
            "limw_glim": "categorical polygons, approximately 1:1,000,000 target scale",
            "footprint_rule": "all native cells touched by each frozen readiness bbox",
            "prohibition": "No Material source is upsampled or represented as a dense boundary map",
        },
        "coverage": {
            "q_M_positive_fraction": float(frame["q_M"].gt(0).mean()),
            "q_M_full_positive_fraction": float(frame["q_M_full"].gt(0).mean()),
            "awc_mean_quality": float(frame["q_M_awc"].mean()),
            "soilgrids_mean_quality": float(frame["q_M_soilgrids"].mean()),
            "geology_mean_quality": float(frame["q_M_geology"].mean()),
        },
        "variation": variation_summary,
        "source_shortcut": {
            "risk": shortcut_risk,
            **shortcut,
            "interpretation": (
                "Material may identify source geography. Downstream normalization and encoders must be "
                "fit on outer-training events only, with source-balanced and mismatched-context controls."
            ),
        },
        "event_shuffle_control": {
            "seed": CONTROL_SEED,
            "valid_fraction": float(
                shuffle["event_shuffle_status"].eq("VALID_WITHIN_DATASET_CROSS_EVENT").mean()
            ),
            "interpretation": "Label-free support-mismatch control; not an outcome association estimate.",
        },
        "case_background_gate": {
            "decision": "ABSTAIN",
            "reason": "Labels and model outputs are prohibited; no Material outcome association is estimated.",
        },
        "gates": {
            "coverage": "PASS" if coverage_pass else "FAIL",
            "within_event_native_cell_variation": "PASS" if variation_pass else "FAIL",
            "source_shortcut": f"WARN_{shortcut_risk}",
            "outcome_association": "ABSTAIN",
            "deployment": deployment,
        },
        "material_contract": {
            "scientific_role": "context_moderator_only",
            "quality": "q_M=max(min(q_M_awc,q_M_soilgrids),q_M_geology)",
            "full_support": "q_M_full=min(q_M_awc,q_M_soilgrids,q_M_geology)",
            "neutral_multiplier": "m_M=1 when q_M=0 or Material branch abstains",
            "future_learned_multiplier": "m_M=1+q_M*clip(delta_M,-0.25,0.25)",
            "allowed_range": [0.75, 1.25],
            "normalization": "Fit on outer-training events only",
        },
        "source_manifest_rows": len(source_manifest),
        "source_asset_hash_rows": len(asset_hashes),
        "artifacts": {
            f"{key}_sha256": sha256(paths[key])
            for key in (
                "sample_registry",
                "event_registry",
                "variation_audit",
                "shuffle_controls",
                "source_manifest",
                "source_asset_hashes",
            )
        },
    }
    atomic_json(summary, paths["summary"])
    atomic_json(
        {
            "status": "complete",
            "summary": str(paths["summary"]),
            "summary_sha256": sha256(paths["summary"]),
            "samples": len(frame),
            "deployment_gate": deployment,
        },
        paths["done"],
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    readiness_path = resolve(root, args.readiness)
    outdir = resolve(root, args.outdir)
    if args.validate_only:
        result = validate_outputs(root, readiness_path, outdir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    build(root, readiness_path, outdir)
    result = validate_outputs(root, readiness_path, outdir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
