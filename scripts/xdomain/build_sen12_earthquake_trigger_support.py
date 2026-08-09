#!/usr/bin/env python3
"""Sample official USGS ShakeMap dose for frozen Sen12 earthquake events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_grid(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = ET.parse(path).getroot()
    fields: dict[int, str] = {}
    specification: dict[str, Any] = {}
    grid_text: str | None = None
    for node in root:
        name = local_name(node.tag)
        if name == "grid_field":
            fields[int(node.attrib["index"])] = node.attrib["name"].lower()
        elif name == "grid_specification":
            specification = dict(node.attrib)
        elif name == "grid_data":
            grid_text = node.text
    if not fields or not specification or not grid_text:
        raise RuntimeError(f"Invalid USGS ShakeMap grid: {path}")
    ordered = [fields[index] for index in range(1, len(fields) + 1)]
    values = np.fromstring(grid_text, sep=" ", dtype=np.float64)
    if values.size % len(ordered):
        raise RuntimeError(f"ShakeMap grid data width mismatch: {path}")
    matrix = values.reshape(-1, len(ordered))
    expected_rows = int(specification["nlon"]) * int(specification["nlat"])
    if matrix.shape[0] != expected_rows:
        raise RuntimeError(
            f"Incomplete ShakeMap grid {path}: rows={matrix.shape[0]} expected={expected_rows}"
        )
    arrays = {name: matrix[:, index] for index, name in enumerate(ordered)}
    required = {"lon", "lat", "mmi", "pga", "pgv"}
    if not required.issubset(arrays):
        raise RuntimeError(f"ShakeMap lacks required fields: {sorted(required - set(arrays))}")
    return arrays, specification


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
    radius = 6371.0088
    lon1_r, lat1_r = np.radians(lon1), np.radians(lat1)
    lon2_r, lat2_r = math.radians(lon2), math.radians(lat2)
    dlon, dlat = lon1_r - lon2_r, lat1_r - lat2_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * math.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/sen12_earthquake_event_registry_v1.csv"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/usgs_earthquake_support_v1/raw"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/usgs_earthquake_support_v1"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    registry_path, raw_dir, outdir = map(resolve, (args.registry, args.raw_dir, args.outdir))
    sample_path = root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
    candidate_path = root / "metadata/pild_xdomain_v1/candidate_event_registry_v1.csv"
    cache_path = root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv"
    registry = pd.read_csv(registry_path, low_memory=False)
    samples = pd.read_csv(sample_path, low_memory=False)
    candidates = pd.read_csv(candidate_path, low_memory=False)
    cache = pd.read_csv(cache_path, low_memory=False)
    if cache["sample_id"].duplicated().any():
        raise RuntimeError("Frozen cache index contains duplicate sample_id values")
    frozen_ids = set(cache["sample_id"].astype(str))
    missing_frozen = frozen_ids - set(samples["sample_id"].astype(str))
    if missing_frozen:
        raise RuntimeError(f"Frozen cache samples missing from registry: {sorted(missing_frozen)[:8]}")
    samples = samples[samples["sample_id"].astype(str).isin(frozen_ids)].copy()
    if len(samples) != len(cache):
        raise RuntimeError(f"Frozen sample identity mismatch: registry={len(samples)} cache={len(cache)}")
    earthquake_clusters = set(
        candidates.loc[
            (candidates["date_quality"] == "inventory_event_confidence_ge_0.95")
            & (candidates["trigger_family"] == "earthquake"),
            "physical_event_cluster_id",
        ].astype(str)
    )
    if set(registry["physical_event_cluster_id"].astype(str)) != earthquake_clusters:
        raise RuntimeError("USGS event registry does not match high-confidence earthquake clusters")

    sample_rows: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        detail_path = raw_dir / f"{row.usgs_event_id}_detail.json"
        grid_path = raw_dir / f"{row.usgs_event_id}_grid.xml"
        if not detail_path.is_file() or not grid_path.is_file():
            raise FileNotFoundError(f"Missing cached USGS artifacts for {row.usgs_event_id}")
        detail = json.loads(detail_path.read_text())
        if detail.get("id") != row.usgs_event_id:
            raise RuntimeError(f"USGS detail identity mismatch for {row.usgs_event_id}")
        event_lon, event_lat, depth_km = map(float, detail["geometry"]["coordinates"])
        magnitude = float(detail["properties"]["mag"])
        timestamp = datetime.fromtimestamp(float(detail["properties"]["time"]) / 1000.0, tz=timezone.utc)
        date_delta = abs((pd.Timestamp(row.dataset_event_date).date() - timestamp.date()).days)
        if date_delta > 1:
            raise RuntimeError(f"USGS date mismatch for {row.usgs_event_id}: {timestamp.date()}")
        arrays, specification = parse_grid(grid_path)
        group = samples[
            (samples["physical_event_cluster_id"] == row.physical_event_cluster_id)
            & (samples["date_quality"] == "high_single_event")
        ].copy()
        if group.empty:
            raise RuntimeError(f"No high-confidence samples for {row.physical_event_cluster_id}")
        tree = cKDTree(np.column_stack([arrays["lon"], arrays["lat"]]))
        query = np.column_stack([group["center_lon"].to_numpy(float), group["center_lat"].to_numpy(float)])
        degree_distance, grid_index = tree.query(query, k=1)
        max_spacing = 1.5 * max(
            float(specification["nominal_lon_spacing"]),
            float(specification["nominal_lat_spacing"]),
        )
        if np.any(degree_distance > max_spacing):
            raise RuntimeError(f"Samples fall outside ShakeMap grid for {row.usgs_event_id}")
        lon = group["center_lon"].to_numpy(float)
        lat = group["center_lat"].to_numpy(float)
        epicentral = haversine_km(lon, lat, event_lon, event_lat)
        output = group[[
            "sample_id", "region", "physical_event_cluster_id", "event_date_start",
        ]].copy()
        output["trigger_family"] = "earthquake"
        output["usgs_event_id"] = row.usgs_event_id
        output["earthquake_magnitude"] = magnitude
        output["earthquake_depth_km"] = depth_km
        output["earthquake_epicentral_distance_km"] = epicentral
        output["earthquake_hypocentral_distance_km"] = np.sqrt(np.square(epicentral) + depth_km**2)
        for name in ("mmi", "pga", "pgv", "psa03", "psa10", "psa30", "svel"):
            if name in arrays:
                output[f"earthquake_{name}"] = arrays[name][grid_index]
        finite = np.all(
            np.isfinite(output[["earthquake_mmi", "earthquake_pga", "earthquake_pgv"]].to_numpy(float)),
            axis=1,
        )
        output["q_R"] = finite.astype(float)
        output["q_R_reason"] = np.where(finite, "usgs_shakemap_complete", "usgs_shakemap_nonfinite")
        sample_rows.append(output)
        event_rows.append({
            "physical_event_cluster_id": row.physical_event_cluster_id,
            "region": row.region,
            "dataset_event_date": row.dataset_event_date,
            "usgs_event_id": row.usgs_event_id,
            "usgs_event_time_utc": timestamp.isoformat(),
            "magnitude": magnitude,
            "depth_km": depth_km,
            "n_samples": len(output),
            "median_epicentral_distance_km": float(np.median(epicentral)),
            "median_pga_percent_g": float(output["earthquake_pga"].median()),
            "median_pgv_cm_s": float(output["earthquake_pgv"].median()),
            "median_mmi": float(output["earthquake_mmi"].median()),
            "q_R_fraction": float(output["q_R"].mean()),
        })
        manifest_rows.extend([
            {
                "usgs_event_id": row.usgs_event_id,
                "artifact_role": "comcat_detail",
                "path": str(detail_path),
                "bytes": detail_path.stat().st_size,
                "sha256": sha256(detail_path),
                "source_url": f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={row.usgs_event_id}&format=geojson",
            },
            {
                "usgs_event_id": row.usgs_event_id,
                "artifact_role": "shakemap_grid",
                "path": str(grid_path),
                "bytes": grid_path.stat().st_size,
                "sha256": sha256(grid_path),
                "source_url": row.shakemap_grid_url,
            },
        ])
        print(f"[event] {row.usgs_event_id} samples={len(output)} M={magnitude:.1f}", flush=True)

    sample_frame = pd.concat(sample_rows, ignore_index=True).sort_values("sample_id")
    event_frame = pd.DataFrame(event_rows).sort_values("physical_event_cluster_id")
    manifest = pd.DataFrame(manifest_rows)
    outdir.mkdir(parents=True, exist_ok=True)
    sample_output = outdir / "sen12_earthquake_sample_features_v1.csv"
    event_output = outdir / "sen12_earthquake_event_features_v1.csv"
    manifest_output = outdir / "usgs_source_manifest_v1.csv"
    sample_frame.to_csv(sample_output, index=False)
    event_frame.to_csv(event_output, index=False)
    manifest.to_csv(manifest_output, index=False)
    summary = {
        "n_samples": len(sample_frame),
        "n_events": len(event_frame),
        "q_R_positive_fraction": float((sample_frame["q_R"] > 0).mean()),
        "event_ids": event_frame["usgs_event_id"].tolist(),
        "label_free_contract": (
            "Inputs are the frozen H5-cache sample identities and coordinates, a source-registered USGS event identity, "
            "and official USGS ShakeMap grids; segmentation labels and predictions are not read."
        ),
        "frozen_cache_index": str(cache_path),
        "frozen_cache_index_sha256": sha256(cache_path),
        "artifacts": {
            "sample_features": str(sample_output),
            "event_features": str(event_output),
            "source_manifest": str(manifest_output),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
