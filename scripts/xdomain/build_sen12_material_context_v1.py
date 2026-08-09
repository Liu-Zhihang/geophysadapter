#!/usr/bin/env python3
"""Build a label-free, native-scale Material registry for frozen Sen12 samples.

Material is registered as sample/event context. The script never opens a
segmentation mask, prediction, checkpoint, or metric artifact, and it never
upsamples Material into a 10 m boundary map. OpenLandMap AWC and SoilGrids are
approximately 250 m sources; LiMW/GLiM is polygon-scale geology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SAMPLES = 4979
CONTROL_SEED = 20260722

AWC_LAYERS = (
    ("0_10", "0..10cm", 100.0),
    ("10_30", "10..30cm", 200.0),
    ("30_60", "30..60cm", 300.0),
    ("60_100", "60..100cm", 400.0),
    ("100_200", "100..200cm", 1000.0),
)
AWC_TOTAL = ("0_200", "0..200cm", 2000.0)
AWC_TOLERANCE_MM = 6.0
CARDINALS = ("north", "south", "east", "west")
TERRAIN_SHIFT_PIXELS = 25  # 250 m on the frozen 10 m target grid.

REGISTRY_COLUMNS = (
    "sample_id",
    "region",
    "physical_event_cluster_id",
    "crs",
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
        default=Path("processed/hybrid_pinn/sen12_context_v1"),
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def awc_paths(source_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for key, token, _ in (*AWC_LAYERS, AWC_TOTAL):
        matches = sorted(source_root.glob(f"*available.water.capacity*_{token}_*.tif"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one OpenLandMap AWC raster for {key}, found {matches}")
        output[key] = matches[0]
    return output


def load_frozen_samples(root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    cache_path = require_file(
        root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv"
    )
    registry_path = require_file(
        root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
    )
    cache = pd.read_csv(cache_path, usecols=["sample_id"])
    registry = pd.read_csv(registry_path, usecols=list(REGISTRY_COLUMNS), low_memory=False)
    if cache["sample_id"].duplicated().any() or registry["sample_id"].duplicated().any():
        raise RuntimeError("sample_id must be unique in frozen cache and geographic registry")
    frame = cache.merge(registry, on="sample_id", how="left", validate="one_to_one")
    geographic = [column for column in REGISTRY_COLUMNS if column != "sample_id"]
    if len(frame) != EXPECTED_SAMPLES or not frame[geographic].notna().all().all():
        raise RuntimeError("Frozen 4,979-sample cache lacks complete geography/CRS")
    return frame, {"cache_index": cache_path, "sample_registry": registry_path}


def load_existing_material(root: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    path = require_file(
        root
        / "metadata/pild_xdomain_v1/tmr_support_audit_v1/"
        "sen12_material_support_audit_v1.csv"
    )
    header = pd.read_csv(path, nrows=0).columns.tolist()
    soil_columns = sorted(
        column
        for column in header
        if column.startswith("soil_")
        and (
            column.endswith("_mean_raw")
            or column.endswith("_local_std_raw")
            or column.endswith("_valid_fraction")
        )
    )
    allowed = [
        "sample_id",
        *soil_columns,
        "lithology_class",
        "lithology_candidate_count",
        "q_M_soil",
        "q_M_lithology",
    ]
    material = pd.read_csv(path, usecols=allowed, low_memory=False)
    if len(material) != EXPECTED_SAMPLES or material["sample_id"].duplicated().any():
        raise RuntimeError("Existing label-free Material audit is not one-to-one with Sen12")
    output = frame.merge(material, on="sample_id", how="left", validate="one_to_one")
    if output[soil_columns].isna().all(axis=1).any():
        raise RuntimeError("At least one frozen sample has no SoilGrids observation")
    return output, path


def raster_cell_geometry(frame: pd.DataFrame, raster_path: Path, prefix: str) -> pd.DataFrame:
    """Register center native cell and a conservative footprint cell count."""
    output = pd.DataFrame(index=frame.index)
    with rasterio.open(raster_path) as source:
        centers_x = frame["center_lon"].to_numpy(dtype=float)
        centers_y = frame["center_lat"].to_numpy(dtype=float)
        if source.crs.to_epsg() != 4326:
            from pyproj import Transformer

            transformer = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
            centers_x, centers_y = transformer.transform(centers_x, centers_y)
        rows, cols = rasterio.transform.rowcol(source.transform, centers_x, centers_y)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        output[f"{prefix}_native_row"] = rows
        output[f"{prefix}_native_col"] = cols
        output[f"{prefix}_native_cell_id"] = [f"r{row}_c{col}" for row, col in zip(rows, cols)]

        counts: list[int] = []
        for item in frame.itertuples(index=False):
            bounds = (float(item.min_lon), float(item.min_lat), float(item.max_lon), float(item.max_lat))
            if source.crs.to_epsg() != 4326:
                bounds = transform_bounds("EPSG:4326", source.crs, *bounds, densify_pts=21)
            left, bottom, right, top = bounds
            r0, c0 = rasterio.transform.rowcol(source.transform, left, top)
            r1, c1 = rasterio.transform.rowcol(source.transform, right, bottom)
            row_count = max(1, abs(int(r1) - int(r0)) + 1)
            col_count = max(1, abs(int(c1) - int(c0)) + 1)
            counts.append(row_count * col_count)
        output[f"{prefix}_footprint_native_cell_count"] = counts
        output.attrs = {
            "crs": str(source.crs),
            "resolution": [float(abs(source.res[0])), float(abs(source.res[1]))],
            "nodata": source.nodata,
        }
    return output


def sample_sorted(source: rasterio.io.DatasetReader, coordinates: np.ndarray) -> np.ndarray:
    rows, cols = rasterio.transform.rowcol(
        source.transform, coordinates[:, 0], coordinates[:, 1]
    )
    order = np.lexsort((np.asarray(cols), np.asarray(rows)))
    ordered = coordinates[order]
    sampled = np.asarray([value[0] for value in source.sample(ordered)], dtype=np.float64)
    restored = np.empty_like(sampled)
    restored[order] = sampled
    valid = np.isfinite(restored)
    if source.nodata is not None:
        valid &= restored != float(source.nodata)
    return np.where(valid, restored, np.nan)


def sample_awc(
    frame: pd.DataFrame, paths: dict[str, Path]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sample aligned and one-native-cell cardinal AWC without resampling."""
    reference = paths[AWC_LAYERS[0][0]]
    geometry = raster_cell_geometry(frame, reference, "awc")
    base_rows = geometry["awc_native_row"].to_numpy(dtype=np.int64)
    base_cols = geometry["awc_native_col"].to_numpy(dtype=np.int64)
    deltas = ((0, 0), (-1, 0), (1, 0), (0, 1), (0, -1))
    positions = ("aligned", *CARDINALS)
    output = geometry.copy()
    source_metadata: dict[str, Any] = {}

    for key, path in paths.items():
        with rasterio.open(path) as source:
            if source.crs.to_epsg() != 4326:
                raise RuntimeError(f"OpenLandMap AWC must be EPSG:4326, found {source.crs}")
            coordinates: list[tuple[float, float]] = []
            for dr, dc in deltas:
                xs, ys = rasterio.transform.xy(
                    source.transform,
                    base_rows + dr,
                    base_cols + dc,
                    offset="center",
                )
                coordinates.extend(zip(xs, ys))
            values = sample_sorted(source, np.asarray(coordinates, dtype=np.float64)).reshape(
                len(deltas), len(frame)
            )
            for position, array in zip(positions, values):
                output[f"awc_{key}_{position}_mm"] = array
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                output[f"awc_{key}_cardinal_std_mm"] = np.nanstd(values[1:], axis=0)
            source_metadata[key] = {
                "path": str(path),
                "crs": str(source.crs),
                "native_resolution_degrees": [float(abs(source.res[0])), float(abs(source.res[1]))],
                "nodata": source.nodata,
                "dtype": source.dtypes[0],
            }

    primary_valid = np.ones(len(frame), dtype=bool)
    for key, _, maximum in AWC_LAYERS:
        values = output[f"awc_{key}_aligned_mm"].to_numpy(dtype=float)
        primary_valid &= np.isfinite(values) & (values >= 0) & (values <= maximum)
    total = output[f"awc_{AWC_TOTAL[0]}_aligned_mm"].to_numpy(dtype=float)
    layer_sum = sum(output[f"awc_{key}_aligned_mm"].to_numpy(dtype=float) for key, _, _ in AWC_LAYERS)
    total_valid = np.isfinite(total) & (total >= 0) & (total <= AWC_TOTAL[2])
    consistent = np.isfinite(layer_sum) & total_valid & (np.abs(layer_sum - total) <= AWC_TOLERANCE_MM)
    output["awc_layer_sum_0_200_mm"] = layer_sum
    output["awc_total_abs_error_mm"] = np.abs(layer_sum - total)
    output["q_M_awc"] = (primary_valid & consistent).astype(float)
    return output, source_metadata


def load_terrain_controls(root: Path, sample_ids: pd.Series) -> tuple[pd.DataFrame, Path]:
    """Read only Terrain arrays; the mask dataset in the file is never opened."""
    path = require_file(
        root
        / "processed/hybrid_pinn/sen12_s2_xdomain_v2/"
        "sen12_native_terrain_v2_p128.h5"
    )
    with h5py.File(path, "r") as handle:
        h5_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["sample_id"][:]]
        if h5_ids != sample_ids.astype(str).tolist():
            raise RuntimeError("Terrain H5 order does not match frozen sample order")
        names = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["terrain_names"][:]]
        selected = [names.index(name) for name in ("elevation", "slope_deg", "local_relief_300m")]
        terrain = handle["terrain"]
        n = len(sample_ids)
        positions = {
            "aligned": (64, 64),
            "north": (64 - TERRAIN_SHIFT_PIXELS, 64),
            "south": (64 + TERRAIN_SHIFT_PIXELS, 64),
            "east": (64, 64 + TERRAIN_SHIFT_PIXELS),
            "west": (64, 64 - TERRAIN_SHIFT_PIXELS),
        }
        vectors = {key: np.empty((n, len(selected)), dtype=np.float64) for key in positions}
        for start in range(0, n, 128):
            stop = min(start + 128, n)
            block = np.asarray(terrain[start:stop, selected, :, :], dtype=np.float32)
            for direction, (row, col) in positions.items():
                patch = block[:, :, row - 2 : row + 3, col - 2 : col + 3]
                vectors[direction][start:stop] = np.nanmean(patch, axis=(2, 3))

    all_values = np.concatenate(list(vectors.values()), axis=0)
    scales = np.nanpercentile(all_values, 75, axis=0) - np.nanpercentile(all_values, 25, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0)
    distances = np.column_stack(
        [
            np.sqrt(np.nanmean(((vectors[direction] - vectors["aligned"]) / scales) ** 2, axis=1))
            for direction in CARDINALS
        ]
    )
    choice = np.nanargmin(np.where(np.isfinite(distances), distances, np.inf), axis=1)
    output = pd.DataFrame(
        {
            "terrain_matched_shift_direction": np.asarray(CARDINALS)[choice],
            "terrain_matched_shift_distance": distances[np.arange(len(choice)), choice],
        }
    )
    for feature_index, name in enumerate(("elevation", "slope_deg", "local_relief_300m")):
        output[f"terrain_{name}_aligned"] = vectors["aligned"][:, feature_index]
    return output, path


def apply_terrain_matched_awc_control(frame: pd.DataFrame) -> None:
    directions = frame["terrain_matched_shift_direction"].astype(str).to_numpy()
    for key, _, _ in (*AWC_LAYERS, AWC_TOTAL):
        shifted = np.full(len(frame), np.nan, dtype=float)
        for direction in CARDINALS:
            mask = directions == direction
            shifted[mask] = frame.loc[mask, f"awc_{key}_{direction}_mm"].to_numpy(dtype=float)
        aligned = frame[f"awc_{key}_aligned_mm"].to_numpy(dtype=float)
        frame[f"awc_{key}_terrain_matched_shift_mm"] = shifted
        frame[f"awc_{key}_aligned_minus_shift_mm"] = aligned - shifted


def event_shuffle_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Create within-region, cross-event donors only; otherwise abstain."""
    rng = np.random.default_rng(CONTROL_SEED)
    rows: list[dict[str, Any]] = []
    by_region = {region: group.copy() for region, group in frame.groupby("region", sort=True)}
    for item in frame.itertuples(index=False):
        candidates = by_region[str(item.region)]
        candidates = candidates[
            candidates["physical_event_cluster_id"].astype(str)
            != str(item.physical_event_cluster_id)
        ]
        if candidates.empty:
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "event_shuffle_status": "ABSTAIN_NO_WITHIN_REGION_ALTERNATE_EVENT",
                    "donor_sample_id": pd.NA,
                    "donor_event_id": pd.NA,
                }
            )
            continue
        donor = candidates.iloc[int(rng.integers(0, len(candidates)))]
        rows.append(
            {
                "sample_id": item.sample_id,
                "event_shuffle_status": "VALID_WITHIN_REGION_CROSS_EVENT",
                "donor_sample_id": donor["sample_id"],
                "donor_event_id": donor["physical_event_cluster_id"],
            }
        )
    return pd.DataFrame(rows)


def nearest_centroid_region_accuracy(frame: pd.DataFrame, columns: list[str]) -> float:
    values = frame[columns].to_numpy(dtype=float)
    labels = frame["region"].astype(str).to_numpy()
    fold = np.asarray(
        [int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 5 for value in frame["sample_id"].astype(str)]
    )
    predictions = np.empty(len(frame), dtype=object)
    for held_out in range(5):
        train = fold != held_out
        test = ~train
        median = np.nanmedian(values[train], axis=0)
        filled_train = np.where(np.isfinite(values[train]), values[train], median)
        filled_test = np.where(np.isfinite(values[test]), values[test], median)
        q25, q75 = np.nanpercentile(filled_train, [25, 75], axis=0)
        scale = np.where(q75 - q25 > 1e-8, q75 - q25, 1.0)
        train_z = (filled_train - median) / scale
        test_z = (filled_test - median) / scale
        classes = sorted(set(labels[train]))
        centroids = np.stack([train_z[labels[train] == label].mean(axis=0) for label in classes])
        distance = ((test_z[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
        predictions[test] = np.asarray(classes)[np.argmin(distance, axis=1)]
    return float(np.mean(predictions == labels))


def variation_audit(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = frame[column].astype(float)
        total_variance = float(values.var(ddof=0))
        region_means = frame.assign(_value=values).groupby("region")["_value"].transform("mean")
        event_means = frame.assign(_value=values).groupby("physical_event_cluster_id")["_value"].transform("mean")
        region_fraction = float(region_means.var(ddof=0) / total_variance) if total_variance > 0 else math.nan
        event_fraction = float(event_means.var(ddof=0) / total_variance) if total_variance > 0 else math.nan
        region_variable = frame.assign(_value=values).groupby("region")["_value"].nunique(dropna=True) > 1
        event_variable = (
            frame.assign(_value=values)
            .groupby("physical_event_cluster_id")["_value"]
            .nunique(dropna=True)
            > 1
        )
        rows.append(
            {
                "feature": column,
                "coverage_fraction": float(values.notna().mean()),
                "overall_std": float(values.std(ddof=0)),
                "unique_values": int(values.nunique(dropna=True)),
                "variable_region_fraction": float(region_variable.mean()),
                "variable_event_fraction": float(event_variable.mean()),
                "between_region_variance_fraction": region_fraction,
                "between_event_variance_fraction": event_fraction,
            }
        )
    audit = pd.DataFrame(rows)
    valid_between = audit["between_region_variance_fraction"].replace([np.inf, -np.inf], np.nan).dropna()
    summary = {
        "features": len(audit),
        "features_with_nonzero_variation": int((audit["overall_std"] > 0).sum()),
        "median_variable_region_fraction": float(audit["variable_region_fraction"].median()),
        "median_variable_event_fraction": float(audit["variable_event_fraction"].median()),
        "median_between_region_variance_fraction": float(valid_between.median()),
    }
    return audit, summary


def build_event_registry(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_id, group in frame.groupby("physical_event_cluster_id", sort=True):
        row: dict[str, Any] = {
            "physical_event_cluster_id": event_id,
            "region": ";".join(sorted(group["region"].astype(str).unique())),
            "n_samples": len(group),
            "q_M_mean": float(group["q_M"].mean()),
            "q_M_full_fraction": float((group["q_M_full"] > 0).mean()),
            "awc_native_cells": int(group["awc_native_cell_id"].nunique()),
            "soilgrids_native_cells": int(group["soilgrids_native_cell_id"].nunique()),
            "lithology_classes": int(group["lithology_class"].nunique(dropna=True)),
        }
        for column in feature_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_source_manifest(
    root: Path,
    paths: dict[str, Path],
    awc_source_metadata: dict[str, Any],
    inputs: dict[str, Path],
    outdir: Path,
) -> tuple[pd.DataFrame, str]:
    fetch_manifest_path = require_file(
        root
        / "metadata/reports/dlr_openlandmap_hydraulic_v6_1_2_fetch_20260717/"
        "source_file_manifest.csv"
    )
    fetch = pd.read_csv(fetch_manifest_path, low_memory=False)
    rows: list[dict[str, Any]] = []
    for key, path in paths.items():
        matched = fetch[fetch["local_path"].astype(str) == str(path)]
        if len(matched) != 1:
            raise RuntimeError(f"No unique verified source manifest row for {path}")
        source = matched.iloc[0]
        rows.append(
            {
                "source_family": "OpenLandMap_AWC",
                "role": "primary_hydraulic_context" if key != AWC_TOTAL[0] else "profile_diagnostic",
                "asset": key,
                "native_scale": "approximately_250m",
                "local_path": str(path),
                "bytes": int(source["bytes"]),
                "sha256": source["local_sha256"],
                "verification": "verified_download_manifest",
                "license": "CC-BY-NC-SA-4.0",
                "metadata": json.dumps(awc_source_metadata[key], sort_keys=True),
            }
        )

    soil_manifest_path = require_file(
        root / "metadata/pild_xdomain_v1/sen12_soilgrids_support_fetch_v1/download_results.csv"
    )
    soil = pd.read_csv(soil_manifest_path, low_memory=False)
    rows.append(
        {
            "source_family": "SoilGrids_2.0",
            "role": "soil_property_context",
            "asset": f"{len(soil)}_verified_tiles",
            "native_scale": "250m",
            "local_path": str(root / "raw_fullcopy/static/soilgrids"),
            "bytes": int(soil["bytes"].sum()),
            "sha256": sha256(soil_manifest_path),
            "verification": "sha256_of_tile_manifest_with_per_tile_hashes",
            "license": "CC-BY-4.0",
            "metadata": str(soil_manifest_path),
        }
    )
    lithology_path = require_file(root / "raw_fullcopy/static/lithology/limw.gpkg")
    rows.append(
        {
            "source_family": "LiMW_GLiM",
            "role": "polygon_lithology_context",
            "asset": "limw.gpkg",
            "native_scale": "polygon_map_approximately_1_to_1M",
            "local_path": str(lithology_path),
            "bytes": lithology_path.stat().st_size,
            "sha256": sha256(lithology_path),
            "verification": "local_full_file_sha256",
            "license": "source_terms_apply",
            "metadata": "GLiM/LiMW categorical polygon geology; never treated as 10 m geology",
        }
    )
    for role, path in inputs.items():
        rows.append(
            {
                "source_family": "Sen12_frozen_support",
                "role": role,
                "asset": path.name,
                "native_scale": "registry_or_cache",
                "local_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "verification": "local_full_file_sha256",
                "license": "project_internal_provenance",
                "metadata": "No segmentation labels or predictions opened by this builder",
            }
        )
    manifest = pd.DataFrame(rows)
    path = outdir / "material_source_manifest.csv"
    manifest.to_csv(path, index=False)
    return manifest, sha256(path)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    outdir = resolve(root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame, inputs = load_frozen_samples(root)
    frame, prior_material_path = load_existing_material(root, frame)
    inputs["prior_label_free_material_audit"] = prior_material_path

    awc_root = require_file(
        root
        / "raw_external/openlandmap_material_v612_250m_zenodo2629148_2784001/"
        "sol_available.water.capacity_usda.mm_m_250m_0..10cm_1950..2017_v0.1.tif"
    ).parent
    paths = awc_paths(awc_root)
    awc, awc_metadata = sample_awc(frame, paths)
    frame = pd.concat([frame.reset_index(drop=True), awc.reset_index(drop=True)], axis=1)

    soil_reference = require_file(
        root / "metadata/pild_xdomain_v1/tmr_support_audit_v1/vrts/soilgrids_clay_0-5cm_mean.vrt"
    )
    soil_geometry = raster_cell_geometry(frame, soil_reference, "soilgrids")
    frame = pd.concat([frame, soil_geometry], axis=1)

    terrain, terrain_path = load_terrain_controls(root, frame["sample_id"])
    inputs["terrain_support_only"] = terrain_path
    frame = pd.concat([frame, terrain], axis=1)
    apply_terrain_matched_awc_control(frame)

    frame["q_M_soilgrids"] = frame["q_M_soil"].fillna(0).clip(0, 1)
    frame["q_M_geology"] = frame["q_M_lithology"].fillna(0).clip(0, 1)
    frame["q_M_hydraulic"] = np.minimum(frame["q_M_awc"], frame["q_M_soilgrids"])
    frame["q_M"] = np.maximum(frame["q_M_hydraulic"], frame["q_M_geology"])
    frame["q_M_full"] = np.minimum(frame["q_M_hydraulic"], frame["q_M_geology"])
    frame["material_multiplier_neutral"] = 1.0
    frame["material_multiplier_min_allowed"] = 0.75
    frame["material_multiplier_max_allowed"] = 1.25

    primary_features = [f"awc_{key}_aligned_mm" for key, _, _ in AWC_LAYERS]
    soil_features = sorted(
        column for column in frame.columns if column.startswith("soil_") and column.endswith("_mean_raw")
    )
    numeric_features = primary_features + soil_features
    variation, variation_summary = variation_audit(frame, numeric_features)
    shortcut_accuracy = nearest_centroid_region_accuracy(frame, numeric_features)

    shuffle = event_shuffle_controls(frame)
    valid_shuffle = shuffle["event_shuffle_status"].eq("VALID_WITHIN_REGION_CROSS_EVENT")
    shuffle_summary = {
        "valid_samples": int(valid_shuffle.sum()),
        "coverage_fraction": float(valid_shuffle.mean()),
        "status_counts": shuffle["event_shuffle_status"].value_counts().to_dict(),
    }

    native_shift_columns = [f"awc_{key}_aligned_minus_shift_mm" for key, _, _ in AWC_LAYERS]
    native_shift_nonzero = np.any(
        np.abs(frame[native_shift_columns].to_numpy(dtype=float)) > 1e-8, axis=1
    )
    native_shift_summary = {
        "terrain_matched_shift_samples": len(frame),
        "finite_all_primary_fraction": float(frame[native_shift_columns].notna().all(axis=1).mean()),
        "any_primary_change_fraction": float(np.mean(native_shift_nonzero)),
        "median_terrain_match_distance": float(frame["terrain_matched_shift_distance"].median()),
        "interpretation": (
            "Support-mismatch control only. It tests native-cell sensitivity without labels; "
            "it is not a landslide case/background effect estimate."
        ),
    }

    event_registry = build_event_registry(frame, primary_features)
    sample_path = outdir / "material_sample_registry.csv"
    event_path = outdir / "material_event_registry.csv"
    variation_path = outdir / "material_variation_audit.csv"
    shuffle_path = outdir / "material_event_shuffle_controls.csv"
    frame.to_csv(sample_path, index=False)
    event_registry.to_csv(event_path, index=False)
    variation.to_csv(variation_path, index=False)
    shuffle.to_csv(shuffle_path, index=False)

    input_paths = {**inputs, "soilgrids_reference_vrt": soil_reference}
    manifest, manifest_hash = build_source_manifest(
        root, paths, awc_metadata, input_paths, outdir
    )

    coverage_pass = bool((frame["q_M"] > 0).mean() >= 0.9)
    variation_pass = bool(
        variation_summary["features_with_nonzero_variation"] == len(numeric_features)
        and variation_summary["median_variable_event_fraction"] >= 0.8
    )
    shortcut_risk = (
        "HIGH"
        if shortcut_accuracy >= 0.8
        or variation_summary["median_between_region_variance_fraction"] >= 0.5
        else "MODERATE"
    )
    case_background = {
        "decision": "ABSTAIN",
        "reason": (
            "The frozen Sen12 unit is an image patch, not an independently sampled landslide "
            "case/control location. A defensible Material outcome test would require either "
            "segmentation labels or an external independent inventory; neither is opened here."
        ),
        "allowed_controls": (
            "Terrain-matched one-native-cell shifts and within-region event shuffles are "
            "registered only as future support-mismatch controls."
        ),
    }
    deployment_decision = "PASS_CONTEXT_ONLY" if coverage_pass and variation_pass else "ABSTAIN"
    summary = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen 4,979-sample Sen12 cache; label-free Material registration",
        "n_samples": len(frame),
        "n_regions": int(frame["region"].nunique()),
        "n_event_clusters": int(frame["physical_event_cluster_id"].nunique()),
        "label_data_opened": False,
        "prediction_or_metric_data_opened": False,
        "native_scale_contract": {
            "openlandmap_awc": "approximately 250 m",
            "soilgrids": "250 m",
            "limw_glim": "categorical polygons at approximately 1:1,000,000 target scale",
            "prohibition": "No Material source is upsampled or represented as a 10 m boundary map",
        },
        "coverage": {
            "q_M_positive_fraction": float((frame["q_M"] > 0).mean()),
            "q_M_full_fraction": float((frame["q_M_full"] > 0).mean()),
            "awc_valid_fraction": float((frame["q_M_awc"] > 0).mean()),
            "soilgrids_valid_fraction": float((frame["q_M_soilgrids"] > 0).mean()),
            "lithology_valid_fraction": float((frame["q_M_geology"] > 0).mean()),
            "awc_native_cells_per_patch": frame["awc_footprint_native_cell_count"].describe().to_dict(),
            "soilgrids_native_cells_per_patch": frame[
                "soilgrids_footprint_native_cell_count"
            ].describe().to_dict(),
        },
        "variation": variation_summary,
        "source_shortcut": {
            "risk": shortcut_risk,
            "fivefold_nearest_centroid_region_accuracy": shortcut_accuracy,
            "interpretation": (
                "Material strongly identifies geography; all downstream models must fit "
                "normalization/encoders on outer-training regions and retain mismatch controls."
            ),
        },
        "native_cell_shift_control": native_shift_summary,
        "event_shuffle_control": shuffle_summary,
        "case_background_gate": case_background,
        "gates": {
            "coverage": "PASS" if coverage_pass else "FAIL",
            "within_event_and_region_variation": "PASS" if variation_pass else "FAIL",
            "source_shortcut": f"WARN_{shortcut_risk}",
            "outcome_association": "ABSTAIN",
            "deployment": deployment_decision,
        },
        "material_contract": {
            "scientific_role": (
                "Sample/event-scale susceptibility moderator of Terrain correction; never an "
                "independent dense boundary expert"
            ),
            "quality": "q_M=max(min(q_M_awc,q_M_soilgrids),q_M_geology)",
            "full_support": "q_M_full=min(q_M_awc,q_M_soilgrids,q_M_geology)",
            "neutral_multiplier": "m_M=1 when q_M=0 or Material branch abstains",
            "future_learned_multiplier": "m_M=1+q_M*clip(delta_M,-0.25,0.25)",
            "allowed_range": [0.75, 1.25],
            "normalization": "Fit on outer-training samples only",
        },
        "source_manifest": str(outdir / "material_source_manifest.csv"),
        "source_manifest_sha256": manifest_hash,
        "source_manifest_rows": len(manifest),
        "artifacts": {
            "sample_registry": str(sample_path),
            "sample_registry_sha256": sha256(sample_path),
            "event_registry": str(event_path),
            "event_registry_sha256": sha256(event_path),
            "variation_audit": str(variation_path),
            "variation_audit_sha256": sha256(variation_path),
            "event_shuffle_controls": str(shuffle_path),
            "event_shuffle_controls_sha256": sha256(shuffle_path),
        },
    }
    summary_path = outdir / "material_summary.json"
    write_json(summary_path, summary)
    done = {
        "status": "complete",
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "samples": len(frame),
        "deployment_gate": deployment_decision,
    }
    write_json(outdir / "material_DONE.json", done)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
