#!/usr/bin/env python3
"""Audit role-aware Material and Trigger support for the frozen Sen12 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
SOIL_PROPERTIES = ("clay", "sand", "silt", "cec", "soc", "bdod", "cfvo", "phh2o")
SOIL_DEPTHS = ("0-5cm", "5-15cm")
TRIGGER_WINDOWS = (3, 7, 14, 30)
CONTROL_SHIFTS = (0, -56, -28, 28, 56)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_vrt(tifs: list[Path], vrt: Path) -> None:
    if not tifs:
        raise FileNotFoundError(f"No SoilGrids tiles for {vrt.stem}")
    vrt.parent.mkdir(parents=True, exist_ok=True)
    list_path = vrt.with_suffix(".files.txt")
    list_path.write_text("\n".join(str(path.resolve()) for path in tifs) + "\n")
    subprocess.run(
        ["gdalbuildvrt", "-overwrite", "-input_file_list", str(list_path), str(vrt)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def sample_soil_vrt(
    frame: pd.DataFrame, vrt: Path, offsets_m: tuple[float, ...] = (-500.0, 0.0, 500.0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(vrt) as source:
        transformer = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
        center_x, center_y = transformer.transform(
            frame["center_lon"].to_numpy(dtype=float),
            frame["center_lat"].to_numpy(dtype=float),
        )
        coordinates = [
            (float(x + dx), float(y + dy))
            for x, y in zip(center_x, center_y)
            for dy in offsets_m
            for dx in offsets_m
        ]
        values = np.asarray([sample[0] for sample in source.sample(coordinates)], dtype=float)
        values = values.reshape(len(frame), len(offsets_m) ** 2)
        valid = np.isfinite(values)
        if source.nodata is not None:
            valid &= values != float(source.nodata)
        safe = np.where(valid, values, np.nan)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(safe, axis=1)
            std = np.nanstd(safe, axis=1)
        fraction = valid.mean(axis=1)
        return mean, std, fraction


def audit_lithology(frame: pd.DataFrame, lithology_path: Path) -> tuple[pd.Series, pd.Series]:
    layers = pyogrio.list_layers(lithology_path)
    if len(layers) != 1:
        raise ValueError(f"Expected one lithology layer, found {layers.tolist()}")
    layer = str(layers[0, 0])
    target_crs = pyogrio.read_info(lithology_path, layer=layer)["crs"]
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    n_candidates = pd.Series(0, index=frame.index, dtype="int64")
    for region, indices in frame.groupby("region", sort=True).groups.items():
        points = gpd.GeoDataFrame(
            {"row_index": list(indices)},
            geometry=gpd.points_from_xy(
                frame.loc[indices, "center_lon"], frame.loc[indices, "center_lat"]
            ),
            crs="EPSG:4326",
        ).to_crs(target_crs)
        min_x, min_y, max_x, max_y = points.total_bounds
        polygons = pyogrio.read_dataframe(
            lithology_path,
            layer=layer,
            columns=["Litho"],
            bbox=(min_x - 1000.0, min_y - 1000.0, max_x + 1000.0, max_y + 1000.0),
        )
        if polygons.empty:
            continue
        joined = gpd.sjoin(points, polygons[["Litho", "geometry"]], how="left", predicate="within")
        counts = joined.groupby("row_index")["Litho"].count()
        first = joined.dropna(subset=["Litho"]).drop_duplicates("row_index").set_index("row_index")["Litho"]
        result.loc[first.index] = first.astype(str)
        n_candidates.loc[counts.index] = counts.astype(int)
        print(f"[lithology] region={region} samples={len(indices)} polygons={len(polygons)}", flush=True)
    return result, n_candidates


def chirps_path(root: Path, day: pd.Timestamp) -> Path:
    return root / f"{day.year:04d}" / f"chirps-v2.0.{day.year:04d}.{day.month:02d}.{day.day:02d}.tif.gz"


def trigger_family_map(candidate_registry: Path) -> dict[str, str]:
    frame = pd.read_csv(candidate_registry, low_memory=False)
    mapping: dict[str, str] = {}
    for cluster, group in frame.groupby("physical_event_cluster_id"):
        families = [str(value) for value in group["trigger_family"].dropna() if str(value) not in {"", "unknown"}]
        mapping[str(cluster)] = Counter(families).most_common(1)[0][0] if families else "unknown"
    return mapping


def audit_trigger(events: pd.DataFrame, chirps_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in events.itertuples(index=False):
        event_day = pd.Timestamp(row.event_date_start)
        required: set[pd.Timestamp] = set()
        case_required: set[pd.Timestamp] = set()
        for shift in CONTROL_SHIFTS:
            anchor = event_day + timedelta(days=shift)
            for window in TRIGGER_WINDOWS:
                days = {anchor - timedelta(days=offset) for offset in range(1, window + 1)}
                required.update(days)
                if shift == 0:
                    case_required.update(days)
        available = {day for day in required if chirps_path(chirps_root, day).exists()}
        case_available = {day for day in case_required if chirps_path(chirps_root, day).exists()}
        rows.append(
            {
                "physical_event_cluster_id": row.physical_event_cluster_id,
                "region": row.region,
                "event_date": event_day.date().isoformat(),
                "trigger_family": row.trigger_family,
                "n_samples": int(row.n_samples),
                "required_chirps_days": len(required),
                "available_chirps_days": len(available),
                "chirps_complete_fraction": len(available) / max(len(required), 1),
                "case_required_days": len(case_required),
                "case_available_days": len(case_available),
                "case_complete_fraction": len(case_available) / max(len(case_required), 1),
                "trigger_source_required": (
                    "CHIRPS/IMERG/ERA5-Land"
                    if row.trigger_family == "rainfall"
                    else "USGS earthquake/PGA"
                    if row.trigger_family == "earthquake"
                    else "unresolved"
                ),
                "q_R_rainfall_ready": int(
                    row.trigger_family == "rainfall"
                    and len(case_available) == len(case_required)
                ),
                "q_R_earthquake_ready": 0,
                "q_R_primary_ready": int(
                    row.trigger_family == "rainfall"
                    and len(case_available) == len(case_required)
                ),
            }
        )
    return pd.DataFrame(rows)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--outdir", type=Path, default=Path("metadata/pild_xdomain_v1/tmr_support_audit_v1"))
    parser.add_argument("--rebuild-vrts", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_index_path = root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv"
    registry_path = root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
    candidate_path = root / "metadata/pild_xdomain_v1/candidate_event_registry_v1.csv"
    soil_root = root / "raw_fullcopy/static/soilgrids"
    lithology_path = root / "raw_fullcopy/static/lithology/limw.gpkg"
    chirps_root = root / "raw_fullcopy/weather/chirps_daily_global"

    cache = pd.read_csv(cache_index_path, low_memory=False)
    registry = pd.read_csv(registry_path, low_memory=False)
    support = cache.merge(
        registry[[
            "sample_id", "region", "center_lon", "center_lat", "event_date_start",
            "date_quality", "physical_event_cluster_id",
        ]],
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_registry"),
    )
    quality_column = "date_quality_registry" if "date_quality_registry" in support else "date_quality"
    if len(support) != 4979 or not support[["center_lon", "center_lat", "event_date_start"]].notna().all().all():
        raise RuntimeError("Frozen 4,979-sample cache lacks complete geolocation/date support")

    material_columns: list[str] = []
    material_summary: dict[str, object] = {}
    vrt_dir = outdir / "vrts"
    for prop in SOIL_PROPERTIES:
        for depth in SOIL_DEPTHS:
            source_dir = soil_root / prop / f"{prop}_{depth}_mean"
            # Official SoilGrids VRT sources preserve their parent-tile
            # subdirectories; legacy DLR-only files lived directly here.
            tifs = sorted(source_dir.rglob("*.tif"))
            vrt = vrt_dir / f"soilgrids_{prop}_{depth}_mean.vrt"
            if args.rebuild_vrts or not vrt.exists():
                build_vrt(tifs, vrt)
            mean, std, fraction = sample_soil_vrt(support, vrt)
            prefix = f"soil_{prop}_{depth.replace('-', '_').replace('cm', 'cm')}"
            support[f"{prefix}_mean_raw"] = mean
            support[f"{prefix}_local_std_raw"] = std
            support[f"{prefix}_valid_fraction"] = fraction
            material_columns.extend(
                [f"{prefix}_mean_raw", f"{prefix}_local_std_raw", f"{prefix}_valid_fraction"]
            )
            material_summary[prefix] = {
                "n_tiles": len(tifs),
                "center_neighborhood_coverage": float(np.mean(fraction > 0)),
                "complete_3x3_coverage": float(np.mean(fraction == 1)),
                "nonzero_local_variation": float(np.mean(np.nan_to_num(std) > 0)),
                "vrt": str(vrt),
            }
            print(f"[soil] {prefix} coverage={np.mean(fraction > 0):.4f}", flush=True)

    lithology, lithology_candidates = audit_lithology(support, lithology_path)
    support["lithology_class"] = lithology
    support["lithology_candidate_count"] = lithology_candidates
    support["q_M_soil"] = np.min(
        np.column_stack([support[column].to_numpy() for column in material_columns if column.endswith("valid_fraction")]),
        axis=1,
    )
    support["q_M_lithology"] = support["lithology_class"].notna().astype(float)
    support["q_M_any"] = np.maximum(support["q_M_soil"], support["q_M_lithology"])
    support["q_M_full"] = np.minimum(support["q_M_soil"], support["q_M_lithology"])
    material_path = outdir / "sen12_material_support_audit_v1.csv"
    support.to_csv(material_path, index=False)

    family_by_cluster = trigger_family_map(candidate_path)
    high = support[support[quality_column].eq("high_single_event")].copy()
    high["trigger_family"] = high["physical_event_cluster_id"].map(family_by_cluster).fillna("unknown")
    events = (
        high.groupby(["physical_event_cluster_id", "region", "event_date_start", "trigger_family"], as_index=False)
        .agg(n_samples=("sample_id", "size"), center_lon=("center_lon", "median"), center_lat=("center_lat", "median"))
    )
    trigger = audit_trigger(events, chirps_root)
    trigger_path = outdir / "sen12_trigger_support_audit_v1.csv"
    trigger.to_csv(trigger_path, index=False)

    summary = {
        "n_samples": len(support),
        "n_regions": int(support["region"].nunique()),
        "n_event_clusters": int(support["physical_event_cluster_id"].nunique()),
        "date_quality_counts": support[quality_column].value_counts().to_dict(),
        "material": {
            "soilgrids_native_resolution_m": 250,
            "lithology_source": "LiMW/GLiM",
            "variable_summary": material_summary,
            "soil_complete_fraction": float(np.mean(support["q_M_soil"] == 1)),
            "lithology_coverage": float(np.mean(support["q_M_lithology"] == 1)),
            "any_q_M_positive": float(np.mean(support["q_M_any"] > 0)),
            "full_soil_and_lithology_q_M_positive": float(np.mean(support["q_M_full"] > 0)),
        },
        "trigger": {
            "n_high_single_events": len(events),
            "trigger_family_counts": events["trigger_family"].value_counts().to_dict(),
            "chirps_case_complete_events": int((trigger["case_complete_fraction"] == 1).sum()),
            "chirps_all_controls_complete_events": int((trigger["chirps_complete_fraction"] == 1).sum()),
            "primary_ready_events": int(trigger["q_R_primary_ready"].sum()),
            "rainfall_ready_events": int(trigger["q_R_rainfall_ready"].sum()),
            "earthquake_events_pending_usgs": int(
                ((trigger["trigger_family"] == "earthquake") & (trigger["q_R_earthquake_ready"] == 0)).sum()
            ),
        },
        "inputs": {
            "cache_index": str(cache_index_path),
            "cache_index_sha256": sha256(cache_index_path),
            "sample_registry": str(registry_path),
            "sample_registry_sha256": sha256(registry_path),
            "candidate_registry": str(candidate_path),
            "candidate_registry_sha256": sha256(candidate_path),
            "soilgrids_root": str(soil_root),
            "lithology_path": str(lithology_path),
            "chirps_root": str(chirps_root),
        },
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    report = [
        "# Sen12 TMR support audit v1",
        "",
        f"- Samples: `{summary['n_samples']}` across `{summary['n_regions']}` regions.",
        f"- Any valid Material branch: `{summary['material']['any_q_M_positive']:.2%}`.",
        f"- Full soil plus lithology support: `{summary['material']['full_soil_and_lithology_q_M_positive']:.2%}`.",
        f"- High-single-event Trigger units: `{summary['trigger']['n_high_single_events']}`.",
        f"- CHIRPS case-window complete events: `{summary['trigger']['chirps_case_complete_events']}`.",
        f"- CHIRPS case plus wrong-time controls complete events: `{summary['trigger']['chirps_all_controls_complete_events']}`.",
        "",
        "This audit establishes support availability only. It does not claim model benefit or causal triggering.",
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n")
    print(f"[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
