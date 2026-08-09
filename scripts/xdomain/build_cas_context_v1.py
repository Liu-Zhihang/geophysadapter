#!/usr/bin/env python3
"""Build label-free Material and earthquake Trigger context for CAS samples.

CAS image patches do not retain an audited patch-to-map transform. Therefore:

* Material is summarized over the published study-area polygon and broadcast as
  source-event context.
* Trigger is summarized from the corresponding USGS ShakeMap over that polygon.
* Terrain remains unavailable at patch level (q_T=0); this script never creates
  a dense Terrain tensor from a study-area average.

The script only reads sample identity, event metadata, study-area polygons, and
external physical sources. Segmentation labels and model outputs are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely import contains_xy

import build_pild_material_registry_v1 as material_lib


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CAS_SAMPLES = 11_091
EXPECTED_SOURCE_EVENTS = 6
EXPECTED_PHYSICAL_EVENTS = 5
SCHEMA_VERSION = "1.0"

CAS_EVENTS: dict[str, dict[str, str]] = {
    "CAS_Jiuzhai_valley": {
        "shape": "Jiuzhai_valley.shp",
        "usgs_id": "us2000a5x1",
    },
    "CAS_Lombokt": {
        "shape": "Lombokt.shp",
        "usgs_id": "us1000g3ub",
    },
    "CAS_Hokkaido": {
        "shape": "Hokkaido.shp",
        "usgs_id": "us2000h8ty",
    },
    "CAS_Palu": {
        "shape": "Palu.shp",
        "usgs_id": "us1000h3p4",
    },
    "CAS_Tiburon Peninsula (Planet)": {
        "shape": "Tiburon Peninsula (Planet).shp",
        "usgs_id": "us6000f65h",
    },
    "CAS_Tiburon_Peninsula_(Sentinel)t": {
        "shape": "Tiburon_Peninsula_(Sentinel)t.shp",
        "usgs_id": "us6000f65h",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=Path(
            "processed/hybrid_pinn/"
            "strict_t2_supervised_ready_v4_roleaware_posttrain_qc/"
            "sample_manifest_post_rgb_v4_roleaware_posttrain_qc.csv"
        ),
    )
    parser.add_argument(
        "--event-registry",
        type=Path,
        default=Path("metadata/pild_core_v2/event_registry_v2.csv"),
    )
    parser.add_argument(
        "--shape-root",
        type=Path,
        default=Path(
            "raw_fullcopy/datasets/02_CAS_Landslide/extracted/"
            "study areas shp/study areas shp"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("processed/hybrid_pinn/cas_context_v1"),
    )
    parser.add_argument(
        "--refresh-usgs",
        action="store_true",
        help="Refetch USGS detail and ShakeMap files even when cached.",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def fetch(url: str, path: Path, refresh: bool) -> Path:
    if path.is_file() and not refresh:
        return path
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "GeoPhysAdapter-CAS-context/1.0 (research artifact)"},
    )
    error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            return path
        except Exception as exc:  # pragma: no cover - network-dependent branch
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {error}")


def load_cas_samples(path: Path) -> pd.DataFrame:
    columns = ("event_uid", "dataset_id", "event_date", "sample_id", "asset_status")
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(columns) - set(header))
    if missing:
        raise RuntimeError(f"CAS sample manifest missing columns: {missing}")
    frame = pd.read_csv(path, usecols=list(columns), keep_default_na=False)
    frame = frame.loc[frame["dataset_id"].eq("CAS_Landslide")].copy()
    if len(frame) != EXPECTED_CAS_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_CAS_SAMPLES:,} CAS samples, found {len(frame):,}")
    if frame["sample_id"].duplicated().any() or not frame["asset_status"].eq("supervised_ready").all():
        raise RuntimeError("CAS samples must be unique and supervised_ready")
    if set(frame["event_uid"]) != set(CAS_EVENTS):
        raise RuntimeError(
            f"CAS event set changed: observed={sorted(frame['event_uid'].unique())}"
        )
    return frame


def physical_event_mapping(path: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"physical_event_id", "canonical_date", "physical_trigger_family", "event_uids"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Physical event registry missing columns: {missing}")
    output: dict[str, dict[str, str]] = {}
    for row in frame.itertuples(index=False):
        for event_uid in str(row.event_uids).split(";"):
            if event_uid in CAS_EVENTS:
                output[event_uid] = {
                    "physical_event_id": str(row.physical_event_id),
                    "canonical_date": str(row.canonical_date),
                    "physical_trigger_family": str(row.physical_trigger_family),
                }
    if set(output) != set(CAS_EVENTS):
        raise RuntimeError(f"Physical event registry does not cover all CAS events: {set(CAS_EVENTS) - set(output)}")
    if len({item["physical_event_id"] for item in output.values()}) != EXPECTED_PHYSICAL_EVENTS:
        raise RuntimeError("CAS source events must collapse to exactly five physical events")
    if any(item["physical_trigger_family"] != "earthquake" for item in output.values()):
        raise RuntimeError("All supervised CAS source events are expected to be earthquake-triggered")
    return output


def load_event_regions(
    samples: pd.DataFrame,
    shape_root: Path,
    event_mapping: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    geometries: dict[str, Any] = {}
    for event_uid, config in CAS_EVENTS.items():
        shape_path = shape_root / config["shape"]
        if not shape_path.is_file():
            raise FileNotFoundError(shape_path)
        region = gpd.read_file(shape_path)
        if region.empty or region.crs is None:
            raise RuntimeError(f"Invalid CAS study-area shapefile: {shape_path}")
        region = region.loc[region.geometry.notna() & ~region.geometry.is_empty].to_crs("EPSG:4326")
        geometry = region.geometry.union_all()
        if geometry.is_empty or not geometry.is_valid:
            geometry = geometry.buffer(0)
        left, bottom, right, top = geometry.bounds
        center = geometry.centroid
        dates = samples.loc[samples["event_uid"].eq(event_uid), "event_date"].unique()
        if len(dates) != 1 or str(dates[0]) != event_mapping[event_uid]["canonical_date"]:
            raise RuntimeError(f"CAS event date mismatch for {event_uid}: {dates}")
        geometries[event_uid] = geometry
        rows.append(
            {
                "event_uid": event_uid,
                "physical_event_id": event_mapping[event_uid]["physical_event_id"],
                "dataset_id": "CAS_Landslide",
                "source_scene_id": event_uid,
                "event_date": str(dates[0]),
                "target_crs": "EPSG:4326",
                "bbox_left": float(left),
                "bbox_bottom": float(bottom),
                "bbox_right": float(right),
                "bbox_top": float(top),
                "center_lon": float(center.x),
                "center_lat": float(center.y),
                "region_area_km2": float(
                    region.to_crs(region.estimate_utm_crs()).geometry.area.sum() / 1e6
                ),
                "shape_path": str(shape_path.resolve()),
                "shape_sha256": sha256(shape_path),
                "source_n_samples": int(samples["event_uid"].eq(event_uid).sum()),
                "usgs_event_id": config["usgs_id"],
                "terrain_patch_georef_available": 0,
                "q_T": 0.0,
                "q_T_reason": "cas_patch_tiffs_lack_audited_patch_to_map_transform",
            }
        )
    return pd.DataFrame(rows), geometries


def build_material(event_frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    spatial = event_frame.loc[
        :,
        [
            "event_uid",
            "physical_event_id",
            "dataset_id",
            "source_scene_id",
            "target_crs",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "bbox_top",
            "center_lon",
            "center_lat",
        ],
    ].copy()
    awc, _ = material_lib.sample_awc(spatial, material_lib.awc_paths(root))
    soil, _ = material_lib.sample_soilgrids(spatial, material_lib.soil_vrt_paths(root))
    lithology, _ = material_lib.sample_lithology(
        spatial, root / "raw_fullcopy/static/lithology/limw.gpkg"
    )
    output = pd.concat(
        [
            event_frame.reset_index(drop=True),
            awc.reset_index(drop=True),
            soil.reset_index(drop=True),
            lithology.reset_index(drop=True),
        ],
        axis=1,
    )
    output["q_M_hydraulic"] = np.minimum(output["q_M_awc"], output["q_M_soilgrids"])
    output["q_M"] = np.maximum(output["q_M_hydraulic"], output["q_M_geology"])
    output["q_M_full"] = np.minimum(output["q_M_hydraulic"], output["q_M_geology"])
    output["material_multiplier_neutral"] = 1.0
    output["material_multiplier_min_allowed"] = 0.75
    output["material_multiplier_max_allowed"] = 1.25
    output["material_scientific_role"] = "event_region_context_moderator_only"
    output["material_native_scale_contract"] = (
        "approximately_250m_soil_and_AWC_plus_polygon_scale_lithology"
    )
    return output


def choose_shakemap(detail: dict[str, Any]) -> dict[str, Any]:
    products = detail.get("properties", {}).get("products", {}).get("shakemap", [])
    if not products:
        raise RuntimeError(f"No USGS ShakeMap product for {detail.get('id')}")
    products = sorted(
        products,
        key=lambda item: (
            int(item.get("preferredWeight", 0)),
            int(item.get("updateTime", 0)),
        ),
        reverse=True,
    )
    product = products[0]
    if "download/grid.xml" not in product.get("contents", {}):
        raise RuntimeError(f"Preferred ShakeMap has no grid.xml for {detail.get('id')}")
    return product


def parse_shakemap_grid(path: Path) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    fields: dict[int, str] = {}
    grid_text: str | None = None
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "grid_field":
            fields[int(element.attrib["index"]) - 1] = str(element.attrib["name"]).upper()
        elif tag == "grid_data":
            grid_text = element.text
    if not fields or not grid_text:
        raise RuntimeError(f"Malformed ShakeMap grid: {path}")
    width = max(fields) + 1
    values = np.fromstring(grid_text, sep=" ", dtype=float)
    if values.size % width:
        raise RuntimeError(f"ShakeMap grid width mismatch: {path}")
    array = values.reshape(-1, width)
    frame = pd.DataFrame({fields[index]: array[:, index] for index in sorted(fields)})
    required = {"LON", "LAT", "MMI", "PGA", "PGV"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"ShakeMap grid missing fields {missing}: {path}")
    return frame


def summarize_shakemap(frame: pd.DataFrame, geometry: Any) -> dict[str, Any]:
    inside = contains_xy(
        geometry,
        frame["LON"].to_numpy(dtype=float),
        frame["LAT"].to_numpy(dtype=float),
    )
    selected = frame.loc[inside].copy()
    method = "study_area_polygon"
    if selected.empty:
        center = geometry.centroid
        distance = np.hypot(
            frame["LON"].to_numpy(dtype=float) - center.x,
            frame["LAT"].to_numpy(dtype=float) - center.y,
        )
        selected = frame.iloc[np.argsort(distance)[:9]].copy()
        method = "nearest_nine_grid_points_fallback"
    output: dict[str, Any] = {
        "shakemap_sampling_method": method,
        "shakemap_grid_points": int(len(selected)),
        "shakemap_polygon_intersection": int(method == "study_area_polygon"),
    }
    for field in ("MMI", "PGA", "PGV"):
        values = selected[field].to_numpy(dtype=float)
        output[f"shakemap_{field.lower()}_mean"] = float(np.mean(values))
        output[f"shakemap_{field.lower()}_median"] = float(np.median(values))
        output[f"shakemap_{field.lower()}_p90"] = float(np.quantile(values, 0.90))
        output[f"shakemap_{field.lower()}_max"] = float(np.max(values))
    return output


def build_trigger(
    event_frame: pd.DataFrame,
    geometries: dict[str, Any],
    raw_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    geod = Geod(ellps="WGS84")
    detail_cache: dict[str, dict[str, Any]] = {}
    grid_cache: dict[str, tuple[Path, pd.DataFrame, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for event in event_frame.itertuples(index=False):
        usgs_id = str(event.usgs_event_id)
        detail_path = raw_dir / f"{usgs_id}_detail.geojson"
        detail_url = (
            "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/"
            f"{usgs_id}.geojson"
        )
        fetch(detail_url, detail_path, refresh)
        if usgs_id not in detail_cache:
            detail_cache[usgs_id] = json.loads(detail_path.read_text(encoding="utf-8"))
        detail = detail_cache[usgs_id]
        if detail.get("id") != usgs_id:
            raise RuntimeError(f"USGS event identity mismatch for {usgs_id}")
        product = choose_shakemap(detail)
        grid_url = str(product["contents"]["download/grid.xml"]["url"])
        grid_path = raw_dir / f"{usgs_id}_shakemap_grid.xml"
        fetch(grid_url, grid_path, refresh)
        if usgs_id not in grid_cache:
            grid_cache[usgs_id] = (grid_path, parse_shakemap_grid(grid_path), product)
        _, grid, product = grid_cache[usgs_id]

        properties = detail["properties"]
        longitude, latitude, depth_km = map(float, detail["geometry"]["coordinates"][:3])
        utc_date = datetime.fromtimestamp(
            float(properties["time"]) / 1000.0, tz=timezone.utc
        ).date()
        event_date = datetime.fromisoformat(str(event.event_date)).date()
        date_delta_days = abs((event_date - utc_date).days)
        _, _, center_distance_m = geod.inv(
            longitude,
            latitude,
            float(event.center_lon),
            float(event.center_lat),
        )
        shaking = summarize_shakemap(grid, geometries[event.event_uid])
        q_r = int(
            date_delta_days <= 1
            and float(properties["mag"]) > 0
            and shaking["shakemap_polygon_intersection"] == 1
            and shaking["shakemap_grid_points"] > 0
        )
        rows.append(
            {
                "event_uid": event.event_uid,
                "physical_event_id": event.physical_event_id,
                "dataset_id": "CAS_Landslide",
                "event_date": event.event_date,
                "physical_trigger_family": "earthquake",
                "usgs_event_id": usgs_id,
                "usgs_detail_url": detail_url,
                "usgs_detail_path": str(detail_path.resolve()),
                "usgs_detail_sha256": sha256(detail_path),
                "usgs_shakemap_grid_url": grid_url,
                "usgs_shakemap_grid_path": str(grid_path.resolve()),
                "usgs_shakemap_grid_sha256": sha256(grid_path),
                "usgs_magnitude": float(properties["mag"]),
                "usgs_depth_km": depth_km,
                "usgs_epicenter_lon": longitude,
                "usgs_epicenter_lat": latitude,
                "usgs_origin_utc": datetime.fromtimestamp(
                    float(properties["time"]) / 1000.0, tz=timezone.utc
                ).isoformat(),
                "event_date_delta_utc_days": int(date_delta_days),
                "epicenter_to_study_area_centroid_km": float(center_distance_m / 1000.0),
                "shakemap_source": str(product.get("source", "")),
                "shakemap_version": str(product.get("properties", {}).get("version", "")),
                **shaking,
                "q_R": q_r,
                "q_R_reason": (
                    "usgs_event_and_polygon_shakemap_verified"
                    if q_r
                    else "incomplete_or_nonintersecting_usgs_shakemap_support"
                ),
                "trigger_scientific_role": "event_forcing_context_only",
            }
        )
    return pd.DataFrame(rows)


def broadcast(
    samples: pd.DataFrame,
    event_frame: pd.DataFrame,
    event_registry: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    context = event_registry.copy()
    duplicate = [column for column in ("dataset_id", "event_date") if column in context]
    context = context.drop(columns=duplicate)
    output = samples.loc[:, ["sample_id", "event_uid", "dataset_id", "event_date"]].merge(
        context,
        on="event_uid",
        how="left",
        validate="many_to_one",
    )
    if output.isna().all(axis=1).any() or len(output) != len(samples):
        raise RuntimeError(f"Failed to broadcast {prefix} context to all CAS samples")
    output.insert(1, "registry_schema_version", SCHEMA_VERSION)
    output.insert(2, "broadcast_from_source_event_registry", 1)
    return output


def validate_outputs(
    samples: pd.DataFrame,
    event_frame: pd.DataFrame,
    material_event: pd.DataFrame,
    material_sample: pd.DataFrame,
    trigger_event: pd.DataFrame,
    trigger_sample: pd.DataFrame,
) -> None:
    for label, frame in (
        ("event frame", event_frame),
        ("Material event registry", material_event),
        ("Trigger event registry", trigger_event),
    ):
        if len(frame) != EXPECTED_SOURCE_EVENTS or frame["event_uid"].duplicated().any():
            raise RuntimeError(f"{label} must contain six unique source-event rows")
    for label, frame in (
        ("Material sample registry", material_sample),
        ("Trigger sample registry", trigger_sample),
    ):
        if len(frame) != EXPECTED_CAS_SAMPLES or frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{label} must cover all CAS samples one-to-one")
        if set(frame["sample_id"]) != set(samples["sample_id"]):
            raise RuntimeError(f"{label} sample identity mismatch")
    if not material_event["material_scientific_role"].eq(
        "event_region_context_moderator_only"
    ).all():
        raise RuntimeError("CAS Material role drifted from event-region context")
    if not ((material_event["q_M"] >= 0) & (material_event["q_M"] <= 1)).all():
        raise RuntimeError("CAS q_M outside [0, 1]")
    if not trigger_event["q_R"].eq(1).all():
        bad = trigger_event.loc[trigger_event["q_R"].ne(1), ["event_uid", "q_R_reason"]]
        raise RuntimeError(f"CAS Trigger support is incomplete:\n{bad.to_string(index=False)}")
    if not event_frame["q_T"].eq(0).all():
        raise RuntimeError("CAS Terrain must remain abstained without patch georeferencing")
    if material_event["physical_event_id"].nunique() != EXPECTED_PHYSICAL_EVENTS:
        raise RuntimeError("CAS Material registry physical-event count mismatch")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    sample_manifest = resolve(root, args.sample_manifest)
    event_registry_path = resolve(root, args.event_registry)
    shape_root = resolve(root, args.shape_root)
    outdir = resolve(root, args.outdir)
    raw_dir = outdir / "usgs_raw"
    outdir.mkdir(parents=True, exist_ok=True)

    samples = load_cas_samples(sample_manifest)
    event_mapping = physical_event_mapping(event_registry_path)
    event_frame, geometries = load_event_regions(samples, shape_root, event_mapping)
    material_event = build_material(event_frame, root)
    trigger_event = build_trigger(event_frame, geometries, raw_dir, args.refresh_usgs)
    material_sample = broadcast(samples, event_frame, material_event, "Material")
    trigger_sample = broadcast(samples, event_frame, trigger_event, "Trigger")
    validate_outputs(
        samples,
        event_frame,
        material_event,
        material_sample,
        trigger_event,
        trigger_sample,
    )

    paths = {
        "source_event_registry": outdir / "cas_source_event_registry_v1.csv",
        "material_event_registry": outdir / "cas_material_event_registry_v1.csv",
        "material_sample_registry": outdir / "cas_material_sample_registry_v1.csv",
        "trigger_event_registry": outdir / "cas_trigger_event_registry_v1.csv",
        "trigger_sample_registry": outdir / "cas_trigger_sample_registry_v1.csv",
    }
    atomic_csv(event_frame, paths["source_event_registry"])
    atomic_csv(material_event, paths["material_event_registry"])
    atomic_csv(material_sample, paths["material_sample_registry"])
    atomic_csv(trigger_event, paths["trigger_event_registry"])
    atomic_csv(trigger_sample, paths["trigger_sample_registry"])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "n_samples": len(samples),
        "n_source_events": event_frame["event_uid"].nunique(),
        "n_physical_events": event_frame["physical_event_id"].nunique(),
        "material_q_M_min": float(material_event["q_M"].min()),
        "material_q_M_full_min": float(material_event["q_M_full"].min()),
        "trigger_q_R_count": int(trigger_event["q_R"].sum()),
        "terrain_q_T_count": int(event_frame["q_T"].sum()),
        "terrain_contract": (
            "q_T=0 for all CAS samples until an exact patch-to-map transform is recovered"
        ),
        "material_contract": "study-area context moderator; never a dense boundary map",
        "trigger_contract": "USGS ShakeMap study-area forcing context",
        "inputs": {
            "sample_manifest": str(sample_manifest),
            "event_registry": str(event_registry_path),
            "shape_root": str(shape_root),
        },
        "artifacts": {
            key: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for key, path in paths.items()
        },
    }
    summary_path = outdir / "summary.json"
    atomic_json(summary, summary_path)
    atomic_json(
        {
            "status": "complete",
            "created_utc": utc_now(),
            "summary_path": str(summary_path),
            "summary_sha256": sha256(summary_path),
        },
        outdir / "DONE.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
