#!/usr/bin/env python3
"""Join frozen Sen12 Terrain, Material, and Trigger support into one sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IDENTITY_COLUMNS = (
    "sample_id",
    "region",
    "physical_event_cluster_id",
    "event_date_start",
    "date_quality_registry",
)


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


def require_unique(frame: pd.DataFrame, name: str) -> None:
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise RuntimeError(f"{name} must contain one non-null row per sample_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--material",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/tmr_support_audit_v1/sen12_material_support_audit_v1.csv"),
    )
    parser.add_argument(
        "--trigger",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/sen12_trigger_support_v2/sen12_trigger_sample_support_v1.csv"),
    )
    parser.add_argument(
        "--earthquake",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/usgs_earthquake_support_v1/sen12_earthquake_sample_features_v1.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/sen12_tmr_sample_support_v2"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    material_path, trigger_path, earthquake_path, outdir = map(
        resolve, (args.material, args.trigger, args.earthquake, args.outdir)
    )
    material = pd.read_csv(material_path, low_memory=False)
    trigger = pd.read_csv(trigger_path, low_memory=False)
    earthquake = pd.read_csv(earthquake_path, low_memory=False)
    require_unique(material, "material support")
    require_unique(trigger, "trigger support")
    require_unique(earthquake, "earthquake support")
    if set(material["sample_id"]) != set(trigger["sample_id"]):
        raise RuntimeError("Material and Trigger sample sets differ")
    if not set(earthquake["sample_id"]).issubset(set(trigger["sample_id"])):
        raise RuntimeError("Earthquake support contains samples outside the frozen Trigger set")
    earthquake_features = sorted(column for column in earthquake.columns if column.startswith("earthquake_"))
    required_earthquake = {
        "earthquake_magnitude",
        "earthquake_depth_km",
        "earthquake_epicentral_distance_km",
        "earthquake_hypocentral_distance_km",
        "earthquake_mmi",
        "earthquake_pga",
        "earthquake_pgv",
    }
    if not required_earthquake.issubset(earthquake_features):
        raise RuntimeError(f"Earthquake support lacks {sorted(required_earthquake - set(earthquake_features))}")
    earthquake_join = earthquake[["sample_id", "usgs_event_id", "q_R", "q_R_reason", *earthquake_features]].rename(
        columns={"q_R": "q_R_earthquake", "q_R_reason": "q_R_reason_earthquake"}
    )
    trigger = trigger.merge(earthquake_join, on="sample_id", how="left", validate="one_to_one")
    earthquake_rows = trigger["q_R_earthquake"].notna()
    if not (trigger.loc[earthquake_rows, "trigger_family"] == "earthquake").all():
        raise RuntimeError("USGS-supported samples do not match the earthquake trigger family")
    trigger.loc[earthquake_rows, "q_R"] = trigger.loc[earthquake_rows, "q_R_earthquake"]
    trigger.loc[earthquake_rows, "q_R_reason"] = trigger.loc[earthquake_rows, "q_R_reason_earthquake"]

    material_features = sorted(
        column
        for column in material.columns
        if column.startswith("soil_")
        and (column.endswith("_mean_raw") or column.endswith("_local_std_raw"))
    )
    if len(material_features) != 32:
        raise RuntimeError(f"Expected 32 continuous Material features, found {len(material_features)}")
    rainfall_features = sorted(
        column
        for column in trigger.columns
        if column.startswith("rain_") and column.endswith("_mm")
    )
    expected_trigger = {
        *(f"rain_d{window}_{timing}_{role}_mm" for window in (3, 7, 14, 30)
          for timing in ("inclusive", "antecedent")
          for role in ("case", "wrongtime_median", "delta")),
        "rain_max1d_d30_inclusive_mm",
        "rain_api09_d30_mm",
    }
    missing_trigger = expected_trigger - set(rainfall_features)
    if missing_trigger:
        raise RuntimeError(f"Trigger support lacks required features: {sorted(missing_trigger)}")
    trigger_features = sorted(expected_trigger) + earthquake_features

    material_columns = [
        *IDENTITY_COLUMNS,
        *material_features,
        "lithology_class",
        "q_M_soil",
        "q_M_lithology",
        "q_M_any",
        "q_M_full",
    ]
    trigger_columns = [
        "sample_id",
        "trigger_family",
        "trigger_anchor_date",
        "trigger_anchor_role",
        "trigger_anchor_confidence",
        "usgs_event_id",
        "q_R",
        "q_R_reason",
        *trigger_features,
    ]
    output = material[material_columns].merge(
        trigger[trigger_columns], on="sample_id", how="left", validate="one_to_one"
    )
    output["q_T"] = material["q_T"].to_numpy(dtype=float)
    if output[["q_T", "q_M_any", "q_R"]].isna().any().any():
        raise RuntimeError("T/M/R quality flags must never be missing")
    for column in trigger_features:
        output.loc[output["q_R"] <= 0, column] = np.nan
    output = output.sort_values("sample_id").reset_index(drop=True)

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "sen12_tmr_sample_support_v1.csv"
    schema_path = outdir / "schema.json"
    summary_path = outdir / "summary.json"
    output.to_csv(csv_path, index=False)
    schema = {
        "sample_id": "immutable join key",
        "terrain": {
            "q_column": "q_T",
            "role": "only dense spatial correction direction; dense arrays remain in the H5 cache",
        },
        "material": {
            "continuous_columns": material_features,
            "categorical_columns": ["lithology_class"],
            "q_columns": ["q_M_soil", "q_M_lithology", "q_M_any", "q_M_full"],
            "role": "sample-scale susceptibility modulator of the Terrain correction",
            "normalization": "fit on outer-training samples only",
            "missing_contract": "q_M=0 gives exact multiplier one",
        },
        "trigger": {
            "continuous_columns": trigger_features,
            "q_column": "q_R",
            "role": "event/context-scale dose modulator; never a dense boundary expert",
            "window_contract": "both source-fixed D0-inclusive and strict antecedent windows retained",
            "normalization": "fit on outer-training samples with q_R>0 only",
            "missing_contract": "q_R=0 gives exact multiplier one",
        },
    }
    schema_path.write_text(json.dumps(json_safe(schema), indent=2, allow_nan=False) + "\n")
    summary = {
        "n_samples": len(output),
        "n_regions": int(output["region"].nunique()),
        "n_event_clusters": int(output["physical_event_cluster_id"].nunique()),
        "material_feature_count": len(material_features),
        "trigger_feature_count": len(trigger_features),
        "lithology_classes": sorted(str(value) for value in output["lithology_class"].dropna().unique()),
        "support_fraction": {
            "terrain": float((output["q_T"] > 0).mean()),
            "material_any": float((output["q_M_any"] > 0).mean()),
            "material_full": float((output["q_M_full"] > 0).mean()),
            "trigger": float((output["q_R"] > 0).mean()),
        },
        "inputs": {
            "material": str(material_path),
            "material_sha256": sha256(material_path),
            "trigger": str(trigger_path),
            "trigger_sha256": sha256(trigger_path),
            "earthquake": str(earthquake_path),
            "earthquake_sha256": sha256(earthquake_path),
        },
        "artifacts": {"csv": str(csv_path), "schema": str(schema_path)},
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
