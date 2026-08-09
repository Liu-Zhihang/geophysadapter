#!/usr/bin/env python3
"""Build the four-source PILD-GEO manifest with outcome-blind DLR QC.

The source data are never deleted. CAS is excluded from this research manifest
because its patches lack an audited patch-to-map transform. DLR hard exclusion
uses only temporal/optical completeness and geospatial support quality. Missing
Material or Trigger support produces role-specific abstention, not whole-event
deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "metadata/pild_sen12_training_v2"
DEFAULT_MANIFEST = DEFAULT_METADATA / "unified_sample_manifest_v2.csv"
DEFAULT_EVENT_SPLIT = DEFAULT_METADATA / "event_isolated_split_v2.csv"
DEFAULT_LODO_SPLIT = DEFAULT_METADATA / "leave_one_dataset_out_split_v2.csv"
DEFAULT_INTEGRATION = (
    PROJECT_ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
)
DEFAULT_OUTDIR = PROJECT_ROOT / "metadata/pild_geo4_qc_v1"

CAS_DATASET = "CAS_Landslide"
DLR_DATASET = "DLR_Landslide_Ref_2025"

MIN_EVENT_Q_VISUAL_TEMPORAL_FRACTION = 0.80
MIN_EVENT_OPTICAL_VALID_FRACTION = 0.99
MIN_EVENT_TERRAIN_VALID_FRACTION = 0.99
MIN_EVENT_ELIGIBLE_SAMPLES = 5
MIN_EVENT_Q_M_FULL = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--event-split", type=Path, default=DEFAULT_EVENT_SPLIT)
    parser.add_argument("--lodo-split", type=Path, default=DEFAULT_LODO_SPLIT)
    parser.add_argument("--integration-dir", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


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


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(value), indent=2, allow_nan=False, sort_keys=True)
        + "\n",
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def dlr_event_audit(
    manifest: pd.DataFrame,
    integration_dir: Path,
) -> pd.DataFrame:
    dlr = manifest.loc[manifest["dataset_id"].eq(DLR_DATASET)].copy()
    readiness = pd.read_csv(
        integration_dir / "pild_window_readiness.csv",
        keep_default_na=False,
        low_memory=False,
    )
    readiness = readiness.loc[readiness["dataset_id"].eq(DLR_DATASET)].copy()
    sample_metadata = readiness[
        [
            "sample_id",
            "physical_event_id",
            "event_uid",
            "source_scene_id",
            "terrain_valid_fraction",
        ]
    ].copy()
    dlr = dlr.merge(
        sample_metadata,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if dlr["physical_event_id"].eq("").any() or dlr["physical_event_id"].isna().any():
        raise RuntimeError("DLR physical-event mapping is incomplete")

    cache_path = integration_dir / "pild_prithvi_4t6b_p128.h5"
    with h5py.File(cache_path, "r") as handle:
        cache_ids = decode(handle["sample_id"][:])
        cache_index = {sample_id: index for index, sample_id in enumerate(cache_ids)}
        q_visual = np.asarray(handle["q_visual_temporal"][:], dtype=float)
        optical_valid = np.asarray(handle["optical_valid"][:], dtype=float)
        cloud_fraction = np.asarray(handle["cloud_fraction"][:], dtype=float)
        indices = dlr["sample_id"].map(cache_index)
        if indices.isna().any():
            raise RuntimeError("DLR sample missing from the four-date optical cache")
        integer_indices = indices.astype(int).to_numpy()
        dlr["q_visual_temporal"] = q_visual[integer_indices]
        dlr["optical_valid_fraction"] = optical_valid[integer_indices].mean(
            axis=(1, 2, 3)
        )
        dlr["cloud_fraction_mean"] = np.nanmean(
            cloud_fraction[integer_indices], axis=1
        )

    event = (
        dlr.groupby("physical_event_id", as_index=False)
        .agg(
            event_uids=("event_uid", lambda x: ";".join(sorted(set(map(str, x))))),
            source_scenes=(
                "source_scene_id",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
            n_samples=("sample_id", "size"),
            n_q_visual_temporal=(
                "q_visual_temporal",
                lambda x: int(np.asarray(x).sum()),
            ),
            q_visual_temporal_fraction=("q_visual_temporal", "mean"),
            optical_valid_fraction_mean=("optical_valid_fraction", "mean"),
            optical_valid_fraction_min=("optical_valid_fraction", "min"),
            cloud_fraction_mean=("cloud_fraction_mean", "mean"),
            terrain_valid_fraction_mean=("terrain_valid_fraction", "mean"),
            terrain_valid_fraction_min=("terrain_valid_fraction", "min"),
        )
    )

    material = pd.read_csv(
        integration_dir / "material_event_registry_v1.csv",
        keep_default_na=False,
    )
    material = material.loc[material["dataset_id"].eq(DLR_DATASET)][
        [
            "physical_event_id",
            "q_M_mean",
            "q_M_full_mean",
            "lithology_classes",
            "lithology_boundary_crossing_fraction",
        ]
    ]
    trigger = pd.read_csv(
        integration_dir / "pild_trigger_event_registry_v1.csv",
        keep_default_na=False,
    )
    trigger = trigger.loc[
        trigger["dataset_ids"].str.contains(DLR_DATASET, na=False),
        [
            "physical_event_id",
            "q_R",
            "q_R_reason",
            "physical_trigger_family",
            "rain_d7_case_minus_wrongtime_mm",
        ],
    ]
    event = event.merge(
        material,
        on="physical_event_id",
        how="left",
        validate="one_to_one",
    ).merge(
        trigger,
        on="physical_event_id",
        how="left",
        validate="one_to_one",
    )

    event["hard_geospatial_pass"] = (
        event["terrain_valid_fraction_mean"]
        >= MIN_EVENT_TERRAIN_VALID_FRACTION
    )
    event["hard_temporal_pass"] = (
        (
            event["q_visual_temporal_fraction"]
            >= MIN_EVENT_Q_VISUAL_TEMPORAL_FRACTION
        )
        & (
            event["optical_valid_fraction_mean"]
            >= MIN_EVENT_OPTICAL_VALID_FRACTION
        )
        & (event["n_q_visual_temporal"] >= MIN_EVENT_ELIGIBLE_SAMPLES)
    )
    event["primary_event_included"] = (
        event["hard_geospatial_pass"] & event["hard_temporal_pass"]
    )
    event["terrain_eligible"] = event["primary_event_included"]
    event["material_eligible"] = (
        event["primary_event_included"]
        & (pd.to_numeric(event["q_M_full_mean"], errors="coerce") >= MIN_EVENT_Q_M_FULL)
    )
    event["trigger_eligible"] = (
        event["primary_event_included"]
        & pd.to_numeric(event["q_R"], errors="coerce").eq(1)
    )
    event["full_tmr_eligible"] = (
        event["terrain_eligible"]
        & event["material_eligible"]
        & event["trigger_eligible"]
    )

    def reason(row: pd.Series) -> str:
        reasons = []
        if not row["hard_geospatial_pass"]:
            reasons.append("terrain_or_georef_quality_below_threshold")
        if row["q_visual_temporal_fraction"] < MIN_EVENT_Q_VISUAL_TEMPORAL_FRACTION:
            reasons.append("event_four_date_quality_fraction_below_0.80")
        if row["optical_valid_fraction_mean"] < MIN_EVENT_OPTICAL_VALID_FRACTION:
            reasons.append("event_optical_valid_fraction_below_0.99")
        if row["n_q_visual_temporal"] < MIN_EVENT_ELIGIBLE_SAMPLES:
            reasons.append("fewer_than_5_temporally_valid_samples")
        return ";".join(reasons) if reasons else "included"

    event["primary_decision_reason"] = event.apply(reason, axis=1)
    event["material_decision_reason"] = np.where(
        event["material_eligible"],
        "eligible",
        "q_M_full_mean_below_0.80_or_primary_event_excluded",
    )
    event["trigger_decision_reason"] = np.where(
        event["trigger_eligible"],
        "eligible",
        event["q_R_reason"].replace("", "q_R_zero_or_primary_event_excluded"),
    )
    return event.sort_values(
        [
            "primary_event_included",
            "full_tmr_eligible",
            "q_visual_temporal_fraction",
            "q_M_full_mean",
        ],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, keep_default_na=False, low_memory=False)
    if set(manifest["dataset_id"]) != {
        DLR_DATASET,
        "GDCLD",
        "GLaD4CD_v1",
        "SEN12LS_HARMONIZED",
    }:
        raise RuntimeError("Input must be the frozen four-source v2 manifest")
    if len(manifest) != 7_916:
        raise RuntimeError(f"Expected 7,916 input rows, found {len(manifest):,}")

    audit = dlr_event_audit(manifest, args.integration_dir)
    excluded_events = set(
        audit.loc[~audit["primary_event_included"], "physical_event_id"]
    )
    if excluded_events != {"PEV1_2019-05-16_3ce3e7c0cc5e"}:
        raise RuntimeError(
            "Outcome-blind hard gate changed unexpectedly: "
            f"{sorted(excluded_events)}"
        )

    primary = manifest.loc[
        ~(
            manifest["dataset_id"].eq(DLR_DATASET)
            & manifest["source_event_id"].isin(excluded_events)
        )
    ].copy()
    primary["manifest_index"] = np.arange(len(primary), dtype=int)
    if len(primary) != 7_890:
        raise RuntimeError(f"Expected 7,890 retained rows, found {len(primary):,}")

    audit_index = audit.set_index("physical_event_id")
    primary["primary_qc_included"] = 1
    primary["dlr_terrain_role_eligible"] = pd.Series(
        pd.NA, index=primary.index, dtype="Int64"
    )
    primary["dlr_material_role_eligible"] = pd.Series(
        pd.NA, index=primary.index, dtype="Int64"
    )
    primary["dlr_trigger_role_eligible"] = pd.Series(
        pd.NA, index=primary.index, dtype="Int64"
    )
    primary["dlr_full_tmr_role_eligible"] = pd.Series(
        pd.NA, index=primary.index, dtype="Int64"
    )
    dlr_mask = primary["dataset_id"].eq(DLR_DATASET)
    for column, audit_column in (
        ("dlr_terrain_role_eligible", "terrain_eligible"),
        ("dlr_material_role_eligible", "material_eligible"),
        ("dlr_trigger_role_eligible", "trigger_eligible"),
        ("dlr_full_tmr_role_eligible", "full_tmr_eligible"),
    ):
        primary.loc[dlr_mask, column] = (
            primary.loc[dlr_mask, "source_event_id"]
            .map(audit_index[audit_column])
            .astype(int)
        )

    event_split = pd.read_csv(args.event_split, keep_default_na=False)
    lodo_split = pd.read_csv(args.lodo_split, keep_default_na=False)
    retained_ids = set(primary["sample_id"])
    event_split = event_split.loc[event_split["sample_id"].isin(retained_ids)].copy()
    lodo_split = lodo_split.loc[lodo_split["sample_id"].isin(retained_ids)].copy()
    event_split["protocol_id"] = "pild_geo4_event_isolated_qc_v1"
    lodo_split["protocol_id"] = "pild_geo4_lodo_qc_v1"
    if set(event_split["sample_id"]) != retained_ids:
        raise RuntimeError("Filtered event split does not cover the retained manifest")
    if set(lodo_split["sample_id"]) != retained_ids:
        raise RuntimeError("Filtered LODO split does not cover the retained manifest")

    exclusion_ledger = pd.DataFrame(
        [
            {
                "scope": "source",
                "dataset_id": CAS_DATASET,
                "physical_event_id": "*",
                "n_samples": 11_091,
                "decision": "exclude_from_research",
                "reason": (
                    "CAS patches lack audited patch-level CRS/affine transform; "
                    "event-region coordinates cannot support dense Terrain alignment"
                ),
                "outcome_used": 0,
                "raw_data_deleted": 0,
            },
            {
                "scope": "physical_event",
                "dataset_id": DLR_DATASET,
                "physical_event_id": "PEV1_2019-05-16_3ce3e7c0cc5e",
                "n_samples": 26,
                "decision": "exclude_from_primary_manifest",
                "reason": (
                    "CA0002: four-date quality pass 3/26 and mean optical-valid "
                    "fraction 0.284, below frozen event-level thresholds"
                ),
                "outcome_used": 0,
                "raw_data_deleted": 0,
            },
        ]
    )

    manifest_path = args.outdir / "unified_sample_manifest_geo4_qc_v1.csv"
    event_split_path = args.outdir / "event_isolated_split_geo4_qc_v1.csv"
    lodo_split_path = args.outdir / "leave_one_dataset_out_split_geo4_qc_v1.csv"
    audit_path = args.outdir / "dlr_event_support_audit_v1.csv"
    ledger_path = args.outdir / "exclusion_ledger_v1.csv"
    atomic_csv(primary, manifest_path)
    atomic_csv(event_split, event_split_path)
    atomic_csv(lodo_split, lodo_split_path)
    atomic_csv(audit, audit_path)
    atomic_csv(exclusion_ledger, ledger_path)

    dataset_counts = {
        str(key): int(value)
        for key, value in primary["dataset_id"].value_counts().sort_index().items()
    }
    summary = {
        "status": "complete",
        "protocol": "PILD-GEO4-QC-v1",
        "selection_is_outcome_blind": True,
        "input_samples": int(len(manifest)),
        "retained_samples": int(len(primary)),
        "retained_dataset_counts": dataset_counts,
        "dlr_input_events": int(len(audit)),
        "dlr_primary_events": int(audit["primary_event_included"].sum()),
        "dlr_terrain_events": int(audit["terrain_eligible"].sum()),
        "dlr_material_events": int(audit["material_eligible"].sum()),
        "dlr_trigger_events": int(audit["trigger_eligible"].sum()),
        "dlr_full_tmr_events": int(audit["full_tmr_eligible"].sum()),
        "excluded_dlr_events": sorted(excluded_events),
        "thresholds": {
            "min_event_q_visual_temporal_fraction": (
                MIN_EVENT_Q_VISUAL_TEMPORAL_FRACTION
            ),
            "min_event_optical_valid_fraction": MIN_EVENT_OPTICAL_VALID_FRACTION,
            "min_event_terrain_valid_fraction": MIN_EVENT_TERRAIN_VALID_FRACTION,
            "min_event_eligible_samples": MIN_EVENT_ELIGIBLE_SAMPLES,
            "min_event_q_M_full_for_material_role": MIN_EVENT_Q_M_FULL,
        },
        "artifacts": {
            path.name: {"path": str(path), "sha256": sha256(path)}
            for path in (
                manifest_path,
                event_split_path,
                lodo_split_path,
                audit_path,
                ledger_path,
            )
        },
    }
    atomic_json(args.outdir / "summary.json", summary)

    material_abstain = audit.loc[
        audit["primary_event_included"] & ~audit["material_eligible"],
        ["source_scenes", "physical_event_id", "q_M_full_mean"],
    ]
    trigger_abstain = audit.loc[
        audit["primary_event_included"] & ~audit["trigger_eligible"],
        ["source_scenes", "physical_event_id", "q_R_reason"],
    ]
    lines = [
        "# PILD-GEO4 outcome-blind QC",
        "",
        f"- Retained samples: `{len(primary):,}/{len(manifest):,}`.",
        f"- DLR retained events: `{int(audit['primary_event_included'].sum())}/{len(audit)}`.",
        "- CAS: excluded from this research manifest; source files remain untouched.",
        "- DLR hard exclusion: `CA0002` only.",
        f"- DLR role eligibility: Terrain `{int(audit['terrain_eligible'].sum())}`, "
        f"Material `{int(audit['material_eligible'].sum())}`, "
        f"Trigger `{int(audit['trigger_eligible'].sum())}`, "
        f"Full-TMR `{int(audit['full_tmr_eligible'].sum())}` events.",
        "",
        "## Hard exclusion",
        "",
        "`CA0002` has only `3/26` four-date-quality-pass samples and an event "
        "mean optical-valid fraction of approximately `0.284`. No labels, "
        "predictions, IoU, or model errors were used.",
        "",
        "## Role-specific abstention",
        "",
        "Material abstention events:",
        "",
    ]
    for row in material_abstain.itertuples(index=False):
        lines.append(
            f"- `{row.source_scenes}`: q_M_full_mean=`{row.q_M_full_mean:.4f}`."
        )
    lines.extend(["", "Trigger abstention events:", ""])
    for row in trigger_abstain.itertuples(index=False):
        lines.append(
            f"- `{row.source_scenes}`: `{row.q_R_reason}`."
        )
    lines.extend(
        [
            "",
            "Role abstention does not remove an event from Terrain experiments. "
            "Unsupported branches must return the exact parent prediction.",
            "",
        ]
    )
    atomic_text(args.outdir / "report.md", "\n".join(lines))


if __name__ == "__main__":
    main()
