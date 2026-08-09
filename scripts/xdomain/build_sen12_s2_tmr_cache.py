#!/usr/bin/env python3
"""Build the audited Sen12Landslides S2 observation/terrain cache.

Only the 4,988 labelled S12LS-LD harmonized Sentinel-2 patches listed in the
upstream ``patch_locations.geojson`` are admitted.  Unlisted patches are never
interpreted as segmentation negatives.  The official table, the region-LOGO5
protocol, the generated NetCDF registry, and every source NetCDF are checked
before an output is committed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import netCDF4
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy import ndimage


PATCH_SIZE = 128
EXPECTED_OFFICIAL_SAMPLES = 4_988
EXPECTED_ELIGIBLE_SAMPLES = 4_979
EXPECTED_OUTER_FOLDS = frozenset(range(5))
OBS_NAMES = (
    "pre_R",
    "pre_G",
    "pre_B",
    "post_R",
    "post_G",
    "post_B",
)
OBS_VARIABLES = ("B04", "B03", "B02")
TERRAIN_NAMES = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "curvature_laplacian",
    "tpi_90m",
    "tpi_300m",
    "roughness_30m",
    "local_relief_300m",
)
TEXT_FIELDS = (
    "sample_id",
    "patch_id",
    "region_group",
    "physical_event_id",
    "event_date",
    "event_dates",
    "date_quality",
    "time_selection_contract",
)
SOURCE_REVISION = "40af2dd6b4e568edb6640d6e14dc67ebd01038a4"
TERRAIN_SOURCE = "Copernicus DEM (approximately 30 m native, resampled to 10 m)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Workspace root containing data_raw/, metadata/, processed/, and experiments/.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override sen12_s2_sample_registry_v1.csv.",
    )
    parser.add_argument(
        "--logo5",
        type=Path,
        default=None,
        help="Override sen12_s2_logo5_v1.csv.",
    )
    parser.add_argument(
        "--patch-locations",
        type=Path,
        default=None,
        help="Override the official harmonized S12LS-LD S2 patch_locations.geojson.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory. Full builds default to processed/hybrid_pinn/"
            "sen12_s2_xdomain_v1; smoke builds use a separate *_smoke_<N> directory."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Deterministic prefix smoke test. Omit for the required full 4,988-sample cache.",
    )
    parser.add_argument(
        "--terrain-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Storage dtype for terrain derivatives (default: float32).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="HDF5 gzip level in [1, 9] (default: 4).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Flush the incremental temporary HDF5 every N samples.",
    )
    return parser.parse_args(argv)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_rows(rows: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative_path, source_hash in rows:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_hash.encode("ascii"))
        digest.update(b"\n")
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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def load_official_patch_table(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise RuntimeError("Official patch table has no GeoJSON feature list")
    rows = [feature.get("properties", {}) for feature in features]
    frame = pd.DataFrame(rows)
    required = {"patch_id", "inventory", "annotated_pixels"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Official patch table misses columns: {sorted(missing)}")
    if len(frame) != EXPECTED_OFFICIAL_SAMPLES:
        raise RuntimeError(
            "Official S12LS-LD contract changed: expected "
            f"{EXPECTED_OFFICIAL_SAMPLES} rows, found {len(frame)}"
        )
    frame["patch_id"] = frame["patch_id"].astype(str)
    frame["sample_id"] = "SEN12_S2_" + frame["patch_id"]
    frame["region_group"] = frame["inventory"].astype(str).str.lower()
    frame["annotated_pixels"] = pd.to_numeric(frame["annotated_pixels"], errors="raise").astype(int)
    if frame["patch_id"].duplicated().any() or frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["patch_id"].duplicated(False), "patch_id"].tolist()
        raise RuntimeError(f"Duplicate official patch_id values: {duplicates[:20]}")
    if (frame["annotated_pixels"] < 50).any():
        bad = frame.loc[frame["annotated_pixels"] < 50, ["patch_id", "annotated_pixels"]]
        raise RuntimeError(f"Official table contains unqualified patches: {bad.head(20).to_dict('records')}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def collapse_logo5(path: Path, official: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str, "patch_id": str})
    required = {
        "sample_id",
        "patch_id",
        "region_group",
        "annotated_pixels",
        "outer_fold",
        "role",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"LOGO5 table misses columns: {sorted(missing)}")
    counts = frame.groupby("sample_id", sort=False).size()
    if len(counts) != EXPECTED_OFFICIAL_SAMPLES or not (counts == len(EXPECTED_OUTER_FOLDS)).all():
        bad = counts[counts != len(EXPECTED_OUTER_FOLDS)]
        raise RuntimeError(
            f"LOGO5 must contain five rows for each of {EXPECTED_OFFICIAL_SAMPLES} samples; "
            f"unique={len(counts)}, bad={bad.head(20).to_dict()}"
        )
    invariant = ("patch_id", "region_group", "annotated_pixels")
    varying = {
        column: frame.groupby("sample_id")[column].nunique(dropna=False).max()
        for column in invariant
    }
    if any(value != 1 for value in varying.values()):
        raise RuntimeError(f"LOGO5 sample-invariant fields vary across folds: {varying}")
    fold_sets = frame.groupby("sample_id")["outer_fold"].apply(
        lambda values: frozenset(pd.to_numeric(values, errors="raise").astype(int))
    )
    if not fold_sets.map(lambda value: value == EXPECTED_OUTER_FOLDS).all():
        raise RuntimeError("LOGO5 outer_fold coverage is incomplete for at least one sample")
    roles = set(frame["role"].astype(str))
    if roles != {"train", "val", "test"}:
        raise RuntimeError(f"Unexpected LOGO5 roles: {sorted(roles)}")
    collapsed = frame.drop_duplicates("sample_id")[list(required - {"outer_fold", "role"})].copy()
    collapsed["annotated_pixels"] = pd.to_numeric(
        collapsed["annotated_pixels"], errors="raise"
    ).astype(int)
    official_key = official.set_index("sample_id")[["patch_id", "region_group", "annotated_pixels"]]
    logo_key = collapsed.set_index("sample_id")[["patch_id", "region_group", "annotated_pixels"]]
    if set(official_key.index) != set(logo_key.index):
        raise RuntimeError("Official patch table and LOGO5 sample_id sets differ")
    logo_key = logo_key.loc[official_key.index]
    unequal = (official_key.astype(str) != logo_key.astype(str)).any(axis=1)
    if unequal.any():
        sample_ids = unequal[unequal].index.tolist()
        raise RuntimeError(f"Official patch table and LOGO5 identities differ: {sample_ids[:20]}")
    return collapsed.sort_values("sample_id").reset_index(drop=True)


def load_registry(path: Path, official: pd.DataFrame, logo: pd.DataFrame, root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str, "patch_id": str}, keep_default_na=False)
    required = {
        "sample_id",
        "patch_id",
        "relative_path",
        "physical_event_cluster_id",
        "event_date",
        "pre_index",
        "post_index",
        "annotated",
        "positive_pixels",
        "event_dates",
        "date_quality",
        "time_selection_contract",
        "change_view_eligible",
        "change_view_exclusion_reason",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Sample registry misses columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(False), "sample_id"].tolist()
        raise RuntimeError(f"Duplicate sample_id values in sample registry: {duplicates[:20]}")
    selected = frame.loc[frame["sample_id"].isin(official["sample_id"])].copy()
    if len(selected) != EXPECTED_OFFICIAL_SAMPLES or set(selected["sample_id"]) != set(official["sample_id"]):
        missing_ids = sorted(set(official["sample_id"]) - set(selected["sample_id"]))
        raise RuntimeError(
            f"Registry does not map all official samples one-to-one: selected={len(selected)}, "
            f"missing={missing_ids[:20]}"
        )
    selected = selected.sort_values("sample_id").reset_index(drop=True)
    official_sorted = official.sort_values("sample_id").reset_index(drop=True)
    logo_sorted = logo.sort_values("sample_id").reset_index(drop=True)
    for column in ("sample_id", "patch_id"):
        if not selected[column].equals(official_sorted[column]):
            raise RuntimeError(f"Registry {column} does not match the official table")
        if not selected[column].equals(logo_sorted[column]):
            raise RuntimeError(f"Registry {column} does not match LOGO5")
    annotated = pd.to_numeric(selected["annotated"], errors="raise").astype(int)
    if not (annotated == 1).all():
        bad = selected.loc[annotated != 1, "sample_id"].tolist()
        raise RuntimeError(f"Official supervised samples marked unannotated in registry: {bad[:20]}")
    positive = pd.to_numeric(selected["positive_pixels"], errors="raise").astype(int)
    expected_positive = official_sorted["annotated_pixels"].astype(int)
    if not positive.equals(expected_positive):
        bad = selected.loc[positive != expected_positive, "sample_id"].tolist()
        raise RuntimeError(f"Registry MASK counts differ from official table: {bad[:20]}")
    if selected["physical_event_cluster_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Registry contains missing physical_event_cluster_id values")
    selected["pre_index"] = pd.to_numeric(selected["pre_index"], errors="raise").astype(int)
    selected["post_index"] = pd.to_numeric(selected["post_index"], errors="raise").astype(int)
    eligible = pd.to_numeric(selected["change_view_eligible"], errors="raise").astype(int)
    exclusion = selected.loc[eligible != 1, [
        "sample_id", "region", "date_quality", "change_view_exclusion_reason"
    ]].copy()
    if len(exclusion) != EXPECTED_OFFICIAL_SAMPLES - EXPECTED_ELIGIBLE_SAMPLES:
        raise RuntimeError(
            "Change-view eligibility contract changed: expected "
            f"{EXPECTED_OFFICIAL_SAMPLES - EXPECTED_ELIGIBLE_SAMPLES} exclusions, "
            f"found {len(exclusion)}"
        )
    selected = selected.loc[eligible == 1].copy().reset_index(drop=True)
    if len(selected) != EXPECTED_ELIGIBLE_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_SAMPLES} eligible samples, found {len(selected)}"
        )
    if (selected["event_dates"].astype(str).str.strip() == "").any():
        raise RuntimeError("Eligible registry rows contain missing event date lists")
    if (selected[["pre_index", "post_index"]] < 0).any(axis=None):
        raise RuntimeError("Eligible registry rows contain invalid pre/post indices")
    if (selected["pre_index"] >= selected["post_index"]).any():
        raise RuntimeError("Eligible registry rows do not have a distinct pre/post pair")
    selected["source_path"] = selected["relative_path"].map(lambda value: (root / value).resolve())
    missing_files = [str(path) for path in selected["source_path"] if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Registry source NetCDF files are incomplete ({len(missing_files)} missing): "
            f"{missing_files[:20]}"
        )
    logo_by_id = logo_sorted.set_index("sample_id")
    positive_by_id = official_sorted.set_index("sample_id")["annotated_pixels"]
    selected["region_group"] = selected["sample_id"].map(
        logo_by_id["region_group"].astype(str).str.lower()
    )
    selected["annotated_pixels"] = selected["sample_id"].map(positive_by_id).astype(int)
    selected["physical_event_id"] = selected["physical_event_cluster_id"].astype(str)
    selected.attrs["exclusions"] = exclusion.to_dict(orient="records")
    return selected


def parse_pre_post(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed_list = ast.literal_eval(f"[{str(value)}]")
        except (SyntaxError, ValueError) as error:
            raise RuntimeError(f"Cannot parse pre_post_dates={value!r}") from error
        if not isinstance(parsed_list, list) or not parsed_list:
            raise RuntimeError(f"Invalid pre_post_dates={value!r}")
        if any(not isinstance(item, dict) for item in parsed_list):
            raise RuntimeError(f"Invalid pre_post_dates={value!r}")
        parsed = {
            "pre": min(int(item["pre"]) for item in parsed_list),
            "post": max(int(item["post"]) for item in parsed_list),
        }
    if not isinstance(parsed, dict) or not {"pre", "post"} <= set(parsed):
        raise RuntimeError(f"Invalid pre_post_dates={value!r}")
    return {"pre": int(parsed["pre"]), "post": int(parsed["post"])}


def masked_to_array(value: Any, *, fill: float) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        if math.isnan(fill):
            return np.asarray(value.astype(np.float32).filled(fill))
        return np.asarray(value.filled(fill))
    return np.asarray(value)


def strict_boolean(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise RuntimeError(f"Cannot interpret {label}={value!r} as a boolean")


def read_spatial(variable: netCDF4.Variable, time_index: int | None, *, fill: float) -> np.ndarray:
    dimensions = list(variable.dimensions)
    selection: list[Any] = [slice(None)] * variable.ndim
    if "time" in dimensions:
        if time_index is None:
            raise RuntimeError(f"A time index is required for variable {variable.name}")
        time_axis = dimensions.index("time")
        if not 0 <= time_index < variable.shape[time_axis]:
            raise RuntimeError(
                f"Time index {time_index} is outside {variable.name} shape {variable.shape}"
            )
        selection[time_axis] = time_index
        dimensions.pop(time_axis)
    elif time_index not in (None, 0):
        raise RuntimeError(f"Variable {variable.name} has no time dimension")
    array = masked_to_array(variable[tuple(selection)], fill=fill)
    if set(dimensions) != {"x", "y"} or len(dimensions) != 2:
        raise RuntimeError(
            f"Variable {variable.name} must reduce to x/y dimensions, got {dimensions}"
        )
    array = np.transpose(array, (dimensions.index("y"), dimensions.index("x")))
    if array.shape != (PATCH_SIZE, PATCH_SIZE):
        raise RuntimeError(f"Variable {variable.name} has unexpected spatial shape {array.shape}")
    return array


def coordinate_resolution(dataset: netCDF4.Dataset) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    if "x" not in dataset.variables or "y" not in dataset.variables:
        raise RuntimeError("NetCDF is missing x/y coordinates")
    x = np.asarray(dataset.variables["x"][:], dtype=np.float64)
    y = np.asarray(dataset.variables["y"][:], dtype=np.float64)
    if x.shape != (PATCH_SIZE,) or y.shape != (PATCH_SIZE,):
        raise RuntimeError(f"Unexpected coordinate shapes x={x.shape}, y={y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError("Coordinates contain non-finite values")
    dx = np.diff(x)
    dy = np.diff(y)
    if (dx == 0).any() or (dy == 0).any() or not (
        (np.all(dx > 0) or np.all(dx < 0)) and (np.all(dy > 0) or np.all(dy < 0))
    ):
        raise RuntimeError("Coordinates must be strictly monotonic")
    xres = float(np.median(np.abs(dx)))
    yres = float(np.median(np.abs(dy)))
    if np.max(np.abs(np.abs(dx) - xres)) > max(1e-6, xres * 1e-3):
        raise RuntimeError("x coordinates are not uniformly spaced")
    if np.max(np.abs(np.abs(dy) - yres)) > max(1e-6, yres * 1e-3):
        raise RuntimeError("y coordinates are not uniformly spaced")
    crs_text = str(dataset.getncattr("crs")) if "crs" in dataset.ncattrs() else ""
    if not crs_text and "spatial_ref" in dataset.variables:
        spatial_ref = dataset.variables["spatial_ref"]
        for attribute in ("crs_wkt", "spatial_ref"):
            if attribute in spatial_ref.ncattrs():
                crs_text = str(spatial_ref.getncattr(attribute))
                break
    if not crs_text:
        raise RuntimeError("NetCDF has no CRS metadata")
    crs = CRS.from_user_input(crs_text)
    if not crs.is_projected:
        raise RuntimeError(f"Terrain derivatives require a projected CRS, got {crs.to_string()}")
    unit_factors = [axis.unit_conversion_factor for axis in crs.axis_info[:2]]
    if len(unit_factors) < 2 or any(abs(float(factor) - 1.0) > 1e-6 for factor in unit_factors):
        raise RuntimeError(f"Terrain derivatives require metre coordinates, got {crs.axis_info}")
    if not (8.0 <= xres <= 12.0 and 8.0 <= yres <= 12.0):
        raise RuntimeError(f"Expected approximately 10 m grid, got x={xres}, y={yres}")
    return x, y, xres, yres, crs.to_string()


def fill_invalid_nearest(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        raise RuntimeError("DEM has no valid pixels")
    if valid.all():
        return np.asarray(array, dtype=np.float32)
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.asarray(array, dtype=np.float32)[tuple(indices)]


def odd_window(distance_m: float, resolution_m: float) -> int:
    pixels = max(1, int(round(distance_m / resolution_m)))
    if pixels % 2 == 0:
        pixels += 1
    return pixels


def terrain_stack(
    dem: np.ndarray,
    dem_valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xres: float,
    yres: float,
) -> np.ndarray:
    dem0 = fill_invalid_nearest(dem, dem_valid).astype(np.float32)
    dz_dy, dz_dx = np.gradient(dem0, y, x)
    gradient = np.hypot(dz_dx, dz_dy)
    slope = np.degrees(np.arctan(gradient)).astype(np.float32)
    aspect = np.arctan2(-dz_dx, dz_dy)
    aspect_sin = np.sin(aspect).astype(np.float32)
    aspect_cos = np.cos(aspect).astype(np.float32)
    flat = gradient < 1e-7
    aspect_sin[flat] = 0.0
    aspect_cos[flat] = 0.0
    curvature = (
        np.gradient(dz_dx, x, axis=1) + np.gradient(dz_dy, y, axis=0)
    ).astype(np.float32)
    nominal_resolution = float((xres + yres) / 2.0)
    window_90 = odd_window(90.0, nominal_resolution)
    window_300 = odd_window(300.0, nominal_resolution)
    window_30 = odd_window(30.0, nominal_resolution)
    mean_90 = ndimage.uniform_filter(dem0, size=window_90, mode="reflect")
    mean_300 = ndimage.uniform_filter(dem0, size=window_300, mode="reflect")
    maximum_30 = ndimage.maximum_filter(dem0, size=window_30, mode="reflect")
    minimum_30 = ndimage.minimum_filter(dem0, size=window_30, mode="reflect")
    maximum_300 = ndimage.maximum_filter(dem0, size=window_300, mode="reflect")
    minimum_300 = ndimage.minimum_filter(dem0, size=window_300, mode="reflect")
    output = np.stack(
        [
            dem0,
            slope,
            aspect_sin,
            aspect_cos,
            curvature,
            dem0 - mean_90,
            dem0 - mean_300,
            maximum_30 - minimum_30,
            maximum_300 - minimum_300,
        ],
        axis=0,
    ).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("Derived terrain contains non-finite values")
    output[:, ~dem_valid] = 0.0
    return output


def create_hdf5(
    path: Path,
    n_samples: int,
    terrain_dtype: str,
    compression_level: int,
) -> h5py.File:
    handle = h5py.File(path, "w", libver="latest")
    compression = {
        "compression": "gzip",
        "compression_opts": compression_level,
        "shuffle": True,
        "fletcher32": True,
    }
    handle.create_dataset(
        "obs",
        shape=(n_samples, len(OBS_NAMES), PATCH_SIZE, PATCH_SIZE),
        dtype="float16",
        chunks=(1, len(OBS_NAMES), PATCH_SIZE, PATCH_SIZE),
        **compression,
    )
    handle.create_dataset(
        "terrain",
        shape=(n_samples, len(TERRAIN_NAMES), PATCH_SIZE, PATCH_SIZE),
        dtype=terrain_dtype,
        chunks=(1, len(TERRAIN_NAMES), PATCH_SIZE, PATCH_SIZE),
        **compression,
    )
    for name in ("mask", "valid_mask"):
        handle.create_dataset(
            name,
            shape=(n_samples, 1, PATCH_SIZE, PATCH_SIZE),
            dtype="uint8",
            chunks=(1, 1, PATCH_SIZE, PATCH_SIZE),
            **compression,
        )
    handle.create_dataset(
        "scl",
        shape=(n_samples, 2, PATCH_SIZE, PATCH_SIZE),
        dtype="uint8",
        chunks=(1, 2, PATCH_SIZE, PATCH_SIZE),
        **compression,
    )
    handle.create_dataset("q_T", shape=(n_samples,), dtype="float32")
    text_type = h5py.string_dtype(encoding="utf-8")
    for name in TEXT_FIELDS + ("source_relative_path", "source_sha256"):
        handle.create_dataset(name, shape=(n_samples,), dtype=text_type)
    handle.create_dataset("pre_index", shape=(n_samples,), dtype="int16")
    handle.create_dataset("post_index", shape=(n_samples,), dtype="int16")
    handle.create_dataset("annotated_pixels", shape=(n_samples,), dtype="int32")
    handle.create_dataset("x_resolution_m", shape=(n_samples,), dtype="float32")
    handle.create_dataset("y_resolution_m", shape=(n_samples,), dtype="float32")
    handle.create_dataset("obs_names", data=np.asarray(OBS_NAMES, dtype=object), dtype=text_type)
    handle.create_dataset(
        "terrain_names", data=np.asarray(TERRAIN_NAMES, dtype=object), dtype=text_type
    )
    return handle


def process_sample(row: Any) -> dict[str, Any]:
    source_path = Path(row.source_path)
    before = source_path.stat()
    with netCDF4.Dataset(source_path, "r") as dataset:
        required_variables = {*OBS_VARIABLES, "SCL", "DEM", "MASK", "x", "y"}
        missing = required_variables - set(dataset.variables)
        if missing:
            raise RuntimeError(f"NetCDF misses variables: {sorted(missing)}")
        if str(dataset.getncattr("satellite")).lower() != "s2":
            raise RuntimeError(f"Expected satellite=s2, got {dataset.getncattr('satellite')!r}")
        mapping = parse_pre_post(dataset.getncattr("pre_post_dates"))
        if mapping != {"pre": int(row.pre_index), "post": int(row.post_index)}:
            raise RuntimeError(
                f"NetCDF pre/post {mapping} differs from registry "
                f"{{'pre': {row.pre_index}, 'post': {row.post_index}}}"
            )
        event_dates = sorted(
            value.strip()
            for value in str(dataset.getncattr("event_date")).split(",")
            if value.strip() and value.strip().lower() not in {"none", "nan", "nat"}
        )
        registry_dates = sorted(value for value in str(row.event_dates).split(";") if value)
        if event_dates != registry_dates:
            raise RuntimeError(
                f"NetCDF event_dates={event_dates!r} differs from registry={registry_dates!r}"
            )
        annotated = dataset.getncattr("annotated")
        if not strict_boolean(annotated, "annotated"):
            raise RuntimeError("Official supervised NetCDF is marked annotated=False")
        x, y, xres, yres, crs = coordinate_resolution(dataset)
        obs_raw = []
        for time_index in (int(row.pre_index), int(row.post_index)):
            for variable_name in OBS_VARIABLES:
                obs_raw.append(
                    read_spatial(dataset.variables[variable_name], time_index, fill=np.nan).astype(
                        np.float32
                    )
                )
        obs_raw_array = np.stack(obs_raw, axis=0)
        obs_valid = (
            np.isfinite(obs_raw_array).all(axis=0)
            & (obs_raw_array >= 0.0).all(axis=0)
            & (obs_raw_array <= 10_000.0).all(axis=0)
        )
        obs = np.clip(np.nan_to_num(obs_raw_array, nan=0.0), 0.0, 10_000.0) / 10_000.0
        scl = np.stack(
            [
                read_spatial(dataset.variables["SCL"], int(row.pre_index), fill=0.0),
                read_spatial(dataset.variables["SCL"], int(row.post_index), fill=0.0),
            ],
            axis=0,
        ).astype(np.uint8)
        dem = read_spatial(dataset.variables["DEM"], 0, fill=np.nan).astype(np.float32)
        dem_valid = np.isfinite(dem) & (dem >= 0.0) & (dem <= 8_800.0)
        valid_mask = obs_valid & dem_valid
        terrain = terrain_stack(dem, dem_valid, x, y, xres, yres)
        mask_raw = read_spatial(dataset.variables["MASK"], 0, fill=0.0)
        mask = (mask_raw > 0).astype(np.uint8)
        observed_positive = int(mask.sum())
        if observed_positive != int(row.annotated_pixels):
            raise RuntimeError(
                f"MASK(time=0) has {observed_positive} positives; official table has "
                f"{row.annotated_pixels}"
            )
    after = source_path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError("Source NetCDF changed while being read")
    source_hash = sha256_file(source_path)
    final_stat = source_path.stat()
    if after.st_size != final_stat.st_size or after.st_mtime_ns != final_stat.st_mtime_ns:
        raise RuntimeError("Source NetCDF changed while being hashed")
    return {
        "obs": obs.astype(np.float16),
        "terrain": terrain,
        "mask": mask[None],
        "valid_mask": valid_mask.astype(np.uint8)[None],
        "scl": scl,
        "q_T": float(dem_valid.mean()),
        "source_sha256": source_hash,
        "source_size_bytes": int(after.st_size),
        "x_resolution_m": xres,
        "y_resolution_m": yres,
        "crs": crs,
        "valid_fraction": float(valid_mask.mean()),
        "terrain_valid_fraction": float(dem_valid.mean()),
        "obs_valid_fraction": float(obs_valid.mean()),
        "pre_cloud_fraction": float(np.isin(scl[0], [3, 8, 9, 10, 11]).mean()),
        "post_cloud_fraction": float(np.isin(scl[1], [3, 8, 9, 10, 11]).mean()),
    }


def run(args: argparse.Namespace) -> int:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not 1 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be in [1, 9]")
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive")

    root = args.root.resolve()
    metadata = root / "metadata/pild_xdomain_v1"
    registry_path = require_file(
        args.registry or metadata / "sen12_s2_sample_registry_v1.csv", "sample registry"
    )
    logo_path = require_file(args.logo5 or metadata / "sen12_s2_logo5_v1.csv", "LOGO5 protocol")
    patch_path = require_file(
        args.patch_locations
        or root
        / "data_raw/08_Sen12Landslides/upstream_code/tasks/S12LS-LD/harmonized/s2/patch_locations.geojson",
        "official patch table",
    )
    default_base = (
        root
        / "processed/hybrid_pinn/sen12_s2_xdomain_v1"
    )
    if args.outdir is not None:
        outdir = args.outdir.resolve()
    elif args.max_samples is not None:
        outdir = default_base.with_name(f"{default_base.name}_smoke_{args.max_samples}")
    else:
        outdir = default_base
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "sen12_s2_tmr_p128.h5"
    index_path = outdir / "cache_index_v1.csv"
    exclusions_path = outdir / "cache_exclusions_v1.csv"
    summary_path = outdir / "cache_summary_v1.json"
    existing = [path for path in (cache_path, index_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite completed cache artifacts: {existing}")

    started_at = now_iso()
    summary: dict[str, Any] = {
        "status": "building",
        "started_at": started_at,
        "completed_at": None,
        "mode": "smoke" if args.max_samples is not None else "full",
        "requested_max_samples": args.max_samples,
        "expected_official_samples": EXPECTED_OFFICIAL_SAMPLES,
        "expected_eligible_samples": EXPECTED_ELIGIBLE_SAMPLES,
        "failures": [],
    }
    temporary_h5 = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
    current_sample = "preflight"
    cache_committed = False
    index_committed = False
    try:
        official = load_official_patch_table(patch_path)
        logo = collapse_logo5(logo_path, official)
        registry = load_registry(registry_path, official, logo, root)
        exclusions = pd.DataFrame(registry.attrs.get("exclusions", []))
        atomic_write_csv(exclusions_path, exclusions)
        if args.max_samples is not None:
            registry = registry.iloc[: min(args.max_samples, len(registry))].copy()
        if args.max_samples is None and len(registry) != EXPECTED_ELIGIBLE_SAMPLES:
            raise RuntimeError(
                f"Full build requires {EXPECTED_ELIGIBLE_SAMPLES} eligible samples, "
                f"selected {len(registry)}"
            )
        registry = registry.reset_index(drop=True)
        manifest_hashes = {
            "official_patch_locations_sha256": sha256_file(patch_path),
            "logo5_sha256": sha256_file(logo_path),
            "sample_registry_sha256": sha256_file(registry_path),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
        }
        handle = create_hdf5(
            temporary_h5,
            len(registry),
            args.terrain_dtype,
            args.compression_level,
        )
        index_rows: list[dict[str, Any]] = []
        try:
            handle.attrs["schema_version"] = 1
            handle.attrs["source_dataset"] = "Sen12Landslides S12LS-LD harmonized S2"
            handle.attrs["source_revision"] = SOURCE_REVISION
            handle.attrs["official_supervised_samples"] = EXPECTED_OFFICIAL_SAMPLES
            handle.attrs["eligible_change_view_samples"] = EXPECTED_ELIGIBLE_SAMPLES
            handle.attrs["excluded_change_view_samples"] = (
                EXPECTED_OFFICIAL_SAMPLES - EXPECTED_ELIGIBLE_SAMPLES
            )
            handle.attrs["selection_contract"] = (
                "official harmonized S2 patch_locations.geojson; annotated_pixels>=50; "
                "strict event-span pre/post bracketing; unlisted patches are not negatives"
            )
            handle.attrs["observation_scaling"] = (
                "B04/B03/B02 clipped to [0,10000] then divided by 10000"
            )
            handle.attrs["terrain_source"] = TERRAIN_SOURCE
            handle.attrs["terrain_native_resolution_m"] = 30
            handle.attrs["terrain_grid_resolution_m"] = 10
            handle.attrs["terrain_derivative_policy"] = (
                "local coordinate-aware derivatives only; no TWI or flow accumulation"
            )
            handle.attrs["valid_mask_policy"] = (
                "finite in-range pre/post RGB intersect finite in-range DEM; SCL/cloud is ignored"
            )
            handle.attrs["q_T_definition"] = "fraction of finite in-range embedded DEM pixels"
            handle.attrs["terrain_dtype"] = args.terrain_dtype
            handle.attrs["created_at"] = started_at
            for key, value in manifest_hashes.items():
                handle.attrs[key] = value

            for index, row in enumerate(registry.itertuples(index=False), start=0):
                current_sample = str(row.sample_id)
                values = process_sample(row)
                handle["obs"][index] = values["obs"]
                handle["terrain"][index] = values["terrain"].astype(args.terrain_dtype)
                handle["mask"][index] = values["mask"]
                handle["valid_mask"][index] = values["valid_mask"]
                handle["scl"][index] = values["scl"]
                handle["q_T"][index] = values["q_T"]
                for field in TEXT_FIELDS:
                    handle[field][index] = str(getattr(row, field))
                relative_path = str(Path(row.source_path).relative_to(root))
                handle["source_relative_path"][index] = relative_path
                handle["source_sha256"][index] = values["source_sha256"]
                handle["pre_index"][index] = int(row.pre_index)
                handle["post_index"][index] = int(row.post_index)
                handle["annotated_pixels"][index] = int(row.annotated_pixels)
                handle["x_resolution_m"][index] = values["x_resolution_m"]
                handle["y_resolution_m"][index] = values["y_resolution_m"]
                index_rows.append(
                    {
                        "cache_index": index,
                        "sample_id": str(row.sample_id),
                        "patch_id": str(row.patch_id),
                        "region_group": str(row.region_group),
                        "physical_event_id": str(row.physical_event_id),
                        "event_date": str(row.event_date),
                        "event_dates": str(row.event_dates),
                        "date_quality": str(row.date_quality),
                        "time_selection_contract": str(row.time_selection_contract),
                        "relative_path": relative_path,
                        "source_sha256": values["source_sha256"],
                        "source_size_bytes": values["source_size_bytes"],
                        "pre_index": int(row.pre_index),
                        "post_index": int(row.post_index),
                        "annotated_pixels": int(row.annotated_pixels),
                        "q_T": values["q_T"],
                        "obs_valid_fraction": values["obs_valid_fraction"],
                        "terrain_valid_fraction": values["terrain_valid_fraction"],
                        "valid_fraction": values["valid_fraction"],
                        "pre_cloud_fraction": values["pre_cloud_fraction"],
                        "post_cloud_fraction": values["post_cloud_fraction"],
                        "x_resolution_m": values["x_resolution_m"],
                        "y_resolution_m": values["y_resolution_m"],
                        "crs": values["crs"],
                    }
                )
                handle.attrs["completed_samples"] = index + 1
                if (index + 1) % args.flush_every == 0 or index + 1 == len(registry):
                    handle.flush()
                    print(
                        f"[CACHE] {index + 1}/{len(registry)} sample={row.sample_id}",
                        flush=True,
                    )
            handle.attrs["completed_at"] = now_iso()
            handle.attrs["complete"] = 1
            handle.flush()
        finally:
            handle.close()

        with temporary_h5.open("rb") as raw_handle:
            os.fsync(raw_handle.fileno())
        os.replace(temporary_h5, cache_path)
        cache_committed = True
        index_frame = pd.DataFrame(index_rows)
        if len(index_frame) != len(registry) or index_frame["sample_id"].duplicated().any():
            raise RuntimeError("Generated cache index violates the one-row-per-sample contract")
        atomic_write_csv(index_path, index_frame)
        index_committed = True
        input_digest = sha256_rows(
            zip(index_frame["relative_path"], index_frame["source_sha256"], strict=True)
        )
        summary.update(
            {
                "status": "complete",
                "completed_at": now_iso(),
                "samples": int(len(index_frame)),
                "regions": int(index_frame["region_group"].nunique()),
                "physical_events": int(index_frame["physical_event_id"].nunique()),
                "positive_pixels": int(index_frame["annotated_pixels"].sum()),
                "obs_shape": [len(index_frame), len(OBS_NAMES), PATCH_SIZE, PATCH_SIZE],
                "obs_dtype": "float16",
                "obs_channels": list(OBS_NAMES),
                "obs_scaling": "clip raw reflectance to [0,10000], then divide by 10000",
                "terrain_shape": [
                    len(index_frame),
                    len(TERRAIN_NAMES),
                    PATCH_SIZE,
                    PATCH_SIZE,
                ],
                "terrain_dtype": args.terrain_dtype,
                "terrain_channels": list(TERRAIN_NAMES),
                "terrain_source": TERRAIN_SOURCE,
                "terrain_native_resolution_m": 30,
                "terrain_grid_resolution_m": 10,
                "forbidden_derivatives": ["TWI", "flow_accumulation"],
                "mask_shape": [len(index_frame), 1, PATCH_SIZE, PATCH_SIZE],
                "mask_dtype": "uint8",
                "mask_policy": "MASK time=0 > 0",
                "valid_mask_shape": [len(index_frame), 1, PATCH_SIZE, PATCH_SIZE],
                "valid_mask_dtype": "uint8",
                "valid_mask_policy": (
                    "finite in-range pre/post RGB and embedded DEM; cloud/SCL does not remove pixels"
                ),
                "q_T_definition": "fraction of finite in-range embedded DEM pixels",
                "scl_shape": [len(index_frame), 2, PATCH_SIZE, PATCH_SIZE],
                "scl_dtype": "uint8",
                "scl_policy": "pre/post SCL retained for label-free degradation stratification; not a model input and not removed from valid_mask",
                "unannotated_patch_policy": "excluded; never interpreted as a negative sample",
                "source_revision": SOURCE_REVISION,
                "input_hashes": {
                    **manifest_hashes,
                    "selected_netcdf_manifest_sha256": input_digest,
                    "per_netcdf_sha256_location": str(index_path),
                },
                "output_hashes": {
                    "cache_sha256": sha256_file(cache_path),
                    "cache_index_sha256": sha256_file(index_path),
                },
                "compression": {
                    "algorithm": "gzip",
                    "level": args.compression_level,
                    "shuffle": True,
                    "fletcher32": True,
                    "chunk_unit": "one sample",
                },
                "quality": {
                    "q_T_min": float(index_frame["q_T"].min()),
                    "q_T_median": float(index_frame["q_T"].median()),
                    "q_T_max": float(index_frame["q_T"].max()),
                    "valid_fraction_min": float(index_frame["valid_fraction"].min()),
                    "valid_fraction_median": float(index_frame["valid_fraction"].median()),
                    "valid_fraction_max": float(index_frame["valid_fraction"].max()),
                    "x_resolution_m_min": float(index_frame["x_resolution_m"].min()),
                    "x_resolution_m_max": float(index_frame["x_resolution_m"].max()),
                    "y_resolution_m_min": float(index_frame["y_resolution_m"].min()),
                    "y_resolution_m_max": float(index_frame["y_resolution_m"].max()),
                },
                "failures": [],
            }
        )
        atomic_write_json(summary_path, summary)
        print(f"[DONE] cache={cache_path} samples={len(index_frame)}", flush=True)
        return 0
    except BaseException as error:
        temporary_h5.unlink(missing_ok=True)
        if index_committed:
            index_path.unlink(missing_ok=True)
        if cache_committed:
            cache_path.unlink(missing_ok=True)
        summary.update(
            {
                "status": "failed",
                "completed_at": now_iso(),
                "failed_sample": current_sample,
                "failures": [
                    {
                        "sample_id": current_sample,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                ],
            }
        )
        atomic_write_json(summary_path, summary)
        raise


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; no temporary cache was committed.", file=sys.stderr)
        raise
