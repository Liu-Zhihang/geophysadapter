#!/usr/bin/env python3
"""Audit and freeze the PILD-to-Prithvi temporal integration contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import pandas as pd


DATE_RE = re.compile(r"PEV1_(\d{4}-\d{2}-\d{2})_")
TARGET_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pild-registry",
        type=Path,
        default=root / "processed/hybrid_pinn/pild_core_geo_v2_1_native30_raw/window_registry_v2.csv",
    )
    parser.add_argument(
        "--sen12-index",
        type=Path,
        default=root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv",
    )
    parser.add_argument(
        "--sen12-prithvi-h5",
        type=Path,
        default=root / "processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_prithvi_4t6b_p128.h5",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "processed/hybrid_pinn/pild_prithvi_integration_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    pild = pd.read_csv(args.pild_registry)
    required = {
        "sample_id", "physical_event_id", "dataset_id", "source_scene_id",
        "source_image_path", "bbox_left", "bbox_bottom", "bbox_right", "bbox_top",
        "target_crs", "terrain_valid_fraction", "observation_profile",
    }
    missing = required - set(pild.columns)
    if missing:
        raise RuntimeError(f"PILD registry missing fields: {sorted(missing)}")
    dates = pild["physical_event_id"].astype(str).str.extract(DATE_RE, expand=False)
    pild["event_date"] = dates.fillna("")
    pild["event_date_valid"] = dates.notna().astype(int)
    pild["source_asset_exists"] = pild["source_image_path"].map(lambda value: int(Path(value).is_file()))
    pild["current_temporal_observations"] = 1
    pild["current_visual_bands"] = "B04,B03,B02_or_native_RGB"
    pild["prithvi_4t6b_ready"] = 0
    pild["external_s2_acquisition_required"] = 1
    pild["acquisition_unit_id"] = pild["dataset_id"].astype(str) + "::" + pild["source_scene_id"].astype(str)
    pild["integration_role"] = "PILD_Core_GEO_external_S2_rebuild"
    pild.to_csv(args.outdir / "pild_window_readiness.csv", index=False)

    source_rows = []
    for dataset, group in pild.groupby("dataset_id", sort=True):
        source_rows.append({
            "dataset_id": dataset,
            "n_windows": len(group),
            "n_physical_events": group["physical_event_id"].nunique(),
            "n_source_scenes": group["source_scene_id"].nunique(),
            "event_date_valid_fraction": group["event_date_valid"].mean(),
            "source_asset_exists_fraction": group["source_asset_exists"].mean(),
            "terrain_valid_fraction_mean": group["terrain_valid_fraction"].mean(),
            "current_observation_profiles": "|".join(sorted(group["observation_profile"].astype(str).unique())),
            "current_prithvi_ready": 0,
            "required_action": "acquire four-date Sentinel-2 L2A six-band stack per scene/event AOI",
        })
    sen12 = pd.read_csv(args.sen12_index)
    with h5py.File(args.sen12_prithvi_h5, "r") as handle:
        optical_shape = tuple(handle["optical"].shape)
        complete = int(handle.attrs.get("complete", 0))
    source_rows.append({
        "dataset_id": "Sen12Landslides_harmonized",
        "n_windows": len(sen12),
        "n_physical_events": sen12["physical_event_id"].nunique(),
        "n_source_scenes": sen12["region_group"].nunique(),
        "event_date_valid_fraction": sen12["event_date"].fillna("").ne("").mean(),
        "source_asset_exists_fraction": 1.0,
        "terrain_valid_fraction_mean": sen12["terrain_valid_fraction"].mean(),
        "current_observation_profiles": "four-date Sentinel-2 L2A six-band harmonized",
        "current_prithvi_ready": int(complete == 1 and optical_shape == (len(sen12), 6, 4, 128, 128)),
        "required_action": "none; retain as existing Prithvi-ready source",
    })
    source = pd.DataFrame(source_rows)
    source.to_csv(args.outdir / "source_readiness.csv", index=False)

    protocol = {
        "status": "audit_complete_acquisition_required",
        "target_tensor": {"shape": "[6,4,128,128]", "bands": TARGET_BANDS, "alignment_gsd_m": 10},
        "temporal_contract_provisional": {
            "pre_window_days": [-180, -7],
            "post_window_days": [7, 180],
            "observations_per_side": 2,
            "minimum_same_side_separation_days": 10,
            "selection": "lowest cloud score subject to valid coverage; labels forbidden",
            "coordinates": "actual year/day-of-year for every selected acquisition",
            "failure": "q_visual_temporal=0 and exact fallback; no synthetic date or image",
        },
        "acquisition_plan": {
            "unit": "dataset + source_scene_id + physical_event_id AOI union",
            "pild_windows": len(pild),
            "pild_events": int(pild["physical_event_id"].nunique()),
            "pild_source_scenes": int(pild["acquisition_unit_id"].nunique()),
            "backends": ["Microsoft Planetary Computer", "CDSE", "GEE"],
            "deduplication_before_download": True,
        },
        "training_contract": {
            "sampler": "source -> physical_event -> patch",
            "primary_split": "leave-one-dataset-out with event/spatial isolation",
            "source_id_as_model_input": False,
            "missing_support": "hard-zero validity and exact visual fallback",
        },
        "assets": {
            "pild_registry": {"path": str(args.pild_registry.resolve()), "sha256": sha256(args.pild_registry)},
            "sen12_index": {"path": str(args.sen12_index.resolve()), "sha256": sha256(args.sen12_index)},
            "sen12_prithvi_h5": {"path": str(args.sen12_prithvi_h5.resolve()), "sha256": sha256(args.sen12_prithvi_h5)},
        },
        "warning": "The temporal windows are an acquisition contract, not evidence of model benefit. PILD is not Prithvi-ready until the new sidecars pass identity, cloud, date and coverage audits.",
    }
    (args.outdir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = [
        "# PILD to Prithvi readiness audit", "",
        f"- Existing PILD-Core GEO: {len(pild)} windows / {pild['physical_event_id'].nunique()} events / {pild['acquisition_unit_id'].nunique()} source-scene units.",
        "- Current PILD visual contract: one post-event RGB observation; 0 windows are directly compatible with Prithvi 4x6.",
        f"- Sen12: {len(sen12)} windows / {sen12['physical_event_id'].nunique()} events; Prithvi sidecar ready={bool(source_rows[-1]['current_prithvi_ready'])}.",
        "- Required next action: acquire/cache four Sentinel-2 dates and six bands per deduplicated scene/event AOI, then crop the frozen windows.",
        "- Sampling must be source -> event -> patch; raw patch concatenation is forbidden.", "",
        "## Source table", "", source.to_markdown(index=False), "",
    ]
    (args.outdir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": protocol["status"], "pild_windows": len(pild), "pild_events": int(pild['physical_event_id'].nunique()), "pild_acquisition_units": int(pild['acquisition_unit_id'].nunique()), "sen12_ready": bool(source_rows[-1]["current_prithvi_ready"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
