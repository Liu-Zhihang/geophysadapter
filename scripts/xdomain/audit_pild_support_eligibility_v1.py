#!/usr/bin/env python3
"""Build label-independent PILD sample/event eligibility and support strata.

This audit never reads labels, predictions, checkpoints, or evaluation metrics.
It does not delete source data. It separates:

1. hard input-quality exclusions;
2. role-specific Terrain/Material/Trigger support;
3. label-free visual-degradation strata.

The resulting flags may be joined to frozen OOF predictions only after this
audit has been generated and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

POLICY: dict[str, Any] = {
    "schema_version": 1,
    "selection_contract": "outcome_blind_data_quality_and_physical_support",
    "forbidden_inputs": [
        "segmentation labels",
        "model predictions",
        "IoU/AP/RER",
        "corrected/harmed status",
        "checkpoint selection",
    ],
    "hard_quality": {
        "minimum_optical_valid_fraction": 0.99,
        "minimum_optical_finite_fraction": 1.0,
        "maximum_optical_zero_fraction": 0.99,
        "required_core_assets_ready": 1,
    },
    "terrain_support": {
        "minimum_terrain_valid_fraction": 0.99,
        "minimum_terrain_finite_fraction": 1.0,
        "note": "Flat terrain remains informative as a physical veto and is not excluded.",
    },
    "material_support": {
        "minimum_q_M_full": 0.95,
        "minimum_soil_valid_property_fraction": 0.75,
        "note": "Material is contextual modulation, never an independent dense boundary expert.",
    },
    "trigger_support": {
        "minimum_q_R": 0.999,
        "require_unique_event_date": True,
        "require_complete_case_and_control_windows": True,
        "note": "Unsupported mechanisms force exact R-branch abstention; they do not remove the sample.",
    },
    "visual_degradation": {
        "cloud_mean_fraction": 0.10,
        "cloud_max_fraction": 0.20,
        "minimum_temporal_unique_fraction": 1.0,
        "low_contrast_rule": "within-dataset bottom decile, frozen without labels or model outputs",
    },
    "reporting_contract": {
        "primary_result": "all hard-quality-eligible samples",
        "mechanism_strata": [
            "T-qualified",
            "M-qualified",
            "R-qualified",
            "Full-TMR-qualified",
            "visual-degraded and role-qualified",
        ],
        "required_sensitivity": "report full eligible population beside every support-qualified result",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        path,
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def decode_strings(values: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def grouped_h5_indices(frame: pd.DataFrame, path_column: str, index_column: str):
    for path_text, group in frame.groupby(path_column, sort=True):
        positions = group.index.to_numpy(dtype=int)
        indices = group[index_column].to_numpy(dtype=int)
        order = np.argsort(indices)
        yield Path(path_text), positions[order], indices[order]


def read_rows(dataset: h5py.Dataset, indices: np.ndarray, batch_size: int = 32):
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        yield start, selected, dataset[selected]


def optical_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=manifest.index)
    columns = (
        "optical_valid_fraction",
        "optical_finite_fraction",
        "optical_zero_fraction",
        "optical_saturation_fraction",
        "optical_spatial_contrast",
        "cloud_mean_fraction",
        "cloud_max_fraction",
        "temporal_unique_fraction",
    )
    for column in columns:
        result[column] = np.nan

    for path, positions, indices in grouped_h5_indices(
        manifest, "optical_h5_path", "optical_h5_index"
    ):
        with h5py.File(path, "r") as handle:
            if "optical" not in handle or "optical_valid" not in handle:
                raise ValueError(f"{path} lacks optical/optical_valid")
            optical = handle["optical"]
            optical_valid = handle["optical_valid"]

            for start, selected, values in read_rows(optical, indices):
                target = positions[start : start + len(selected)]
                array = values.astype(np.float32)
                scale = 10000.0 if np.nanmax(array) > 2.0 else 1.0
                array /= scale
                result.loc[target, "optical_finite_fraction"] = np.isfinite(array).mean(
                    axis=tuple(range(1, array.ndim))
                )
                result.loc[target, "optical_zero_fraction"] = (array == 0).mean(
                    axis=tuple(range(1, array.ndim))
                )
                result.loc[target, "optical_saturation_fraction"] = (array >= 0.999).mean(
                    axis=tuple(range(1, array.ndim))
                )

                if array.ndim == 5:
                    rgb = array[:, :3]
                    gray = rgb.mean(axis=1)
                    spatial_std = gray.std(axis=(-2, -1))
                    contrast = np.median(spatial_std, axis=1)
                elif array.ndim == 4:
                    rgb = array[:, :3]
                    contrast = rgb.mean(axis=1).std(axis=(-2, -1))
                else:
                    raise ValueError(f"unexpected optical shape in {path}: {array.shape}")
                result.loc[target, "optical_spatial_contrast"] = contrast

            for start, selected, values in read_rows(optical_valid, indices, batch_size=128):
                target = positions[start : start + len(selected)]
                result.loc[target, "optical_valid_fraction"] = values.astype(bool).mean(
                    axis=tuple(range(1, values.ndim))
                )

            if "cloud_fraction" in handle:
                cloud = np.asarray(handle["cloud_fraction"][indices], dtype=float)
                result.loc[positions, "cloud_mean_fraction"] = np.nanmean(cloud, axis=1)
                result.loc[positions, "cloud_max_fraction"] = np.nanmax(cloud, axis=1)
            if "q_visual_temporal" in handle:
                result.loc[positions, "temporal_unique_fraction"] = np.asarray(
                    handle["q_visual_temporal"][indices], dtype=float
                )
            elif "duplicated_observation" in handle:
                duplicated = np.asarray(handle["duplicated_observation"][indices], dtype=float)
                result.loc[positions, "temporal_unique_fraction"] = 1.0 - duplicated.mean(axis=1)
            else:
                result.loc[positions, "temporal_unique_fraction"] = 1.0
    return result


def terrain_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=manifest.index)
    columns = (
        "terrain_valid_fraction",
        "terrain_finite_fraction",
        "terrain_slope_mean_deg",
        "terrain_slope_std_deg",
        "terrain_relief_mean_m",
        "terrain_relief_std_m",
    )
    for column in columns:
        result[column] = np.nan

    for path, positions, indices in grouped_h5_indices(
        manifest, "terrain_h5_path", "terrain_h5_index"
    ):
        group = manifest.loc[positions]
        channel_specs = group["terrain_channel_indices"].astype(str).unique()
        if len(channel_specs) != 1:
            raise ValueError(f"{path} has multiple terrain channel contracts")
        selected_channels = np.asarray(
            [int(item) for item in channel_specs[0].split(";") if item != ""], dtype=int
        )
        with h5py.File(path, "r") as handle:
            if "terrain" not in handle:
                raise ValueError(f"{path} lacks terrain")
            terrain = handle["terrain"]
            names = decode_strings(handle["terrain_names"][:])
            selected_names = [names[index] for index in selected_channels]
            slope_index = next(
                (i for i, name in enumerate(selected_names) if "slope" in name.lower()), None
            )
            relief_index = next(
                (i for i, name in enumerate(selected_names) if "relief_300" in name.lower()),
                None,
            )
            if slope_index is None or relief_index is None:
                raise ValueError(f"{path} lacks slope/local-relief in selected terrain channels")

            valid_name = "terrain_valid" if "terrain_valid" in handle else "valid_mask"
            valid = handle[valid_name]
            for start, selected, values in read_rows(valid, indices, batch_size=128):
                target = positions[start : start + len(selected)]
                result.loc[target, "terrain_valid_fraction"] = values.astype(bool).mean(
                    axis=tuple(range(1, values.ndim))
                )
            for start, selected, values in read_rows(terrain, indices):
                target = positions[start : start + len(selected)]
                values = values[:, selected_channels].astype(np.float32)
                result.loc[target, "terrain_finite_fraction"] = np.isfinite(values).mean(
                    axis=tuple(range(1, values.ndim))
                )
                slope = values[:, slope_index]
                relief = values[:, relief_index]
                result.loc[target, "terrain_slope_mean_deg"] = np.nanmean(slope, axis=(-2, -1))
                result.loc[target, "terrain_slope_std_deg"] = np.nanstd(slope, axis=(-2, -1))
                result.loc[target, "terrain_relief_mean_m"] = np.nanmean(relief, axis=(-2, -1))
                result.loc[target, "terrain_relief_std_m"] = np.nanstd(relief, axis=(-2, -1))
    return result


def registry_rows(manifest: pd.DataFrame, path_column: str, index_column: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path_text, group in manifest.groupby(path_column, sort=True):
        registry = pd.read_csv(path_text, low_memory=False)
        indices = group[index_column].to_numpy(dtype=int)
        if indices.min() < 0 or indices.max() >= len(registry):
            raise IndexError(f"{path_text} registry index out of bounds")
        selected = registry.iloc[indices].copy()
        selected.index = group.index
        if "sample_id" in selected and not selected["sample_id"].astype(str).equals(
            group["sample_id"].astype(str)
        ):
            raise ValueError(f"{path_text} sample identity differs from unified manifest")
        pieces.append(selected)
    return pd.concat(pieces).sort_index()


def material_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    registry = registry_rows(manifest, "material_registry_path", "material_registry_index")
    result = pd.DataFrame(index=manifest.index)
    for column in ("q_M", "q_M_full", "q_M_hydraulic", "q_M_geology", "q_M_soil"):
        result[column] = pd.to_numeric(registry.get(column, 0.0), errors="coerce").fillna(0.0)

    valid_columns = [
        column
        for column in registry.columns
        if column.startswith("soil_") and column.endswith("valid_fraction")
    ]
    std_columns = [
        column
        for column in registry.columns
        if column.startswith("soil_")
        and ("local_std_raw" in column or "native_cell_std_raw" in column)
    ]
    if valid_columns:
        valid = registry[valid_columns].apply(pd.to_numeric, errors="coerce")
        result["material_valid_property_fraction"] = (valid >= 0.95).mean(axis=1)
    else:
        result["material_valid_property_fraction"] = 0.0
    if std_columns:
        local_std = registry[std_columns].apply(pd.to_numeric, errors="coerce")
        result["material_varying_property_fraction"] = (local_std > 0).mean(axis=1)
    else:
        result["material_varying_property_fraction"] = 0.0
    result["material_lithology_varies"] = pd.to_numeric(
        registry.get("lithology_native_cell_variation", 0), errors="coerce"
    ).fillna(0).clip(0, 1)
    return result


def trigger_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    registry = registry_rows(manifest, "trigger_registry_path", "trigger_registry_index")
    result = pd.DataFrame(index=manifest.index)

    def coalesce(*columns: str, default: Any = np.nan) -> pd.Series:
        value = pd.Series(default, index=registry.index)
        for column in reversed(columns):
            if column in registry:
                value = registry[column].combine_first(value)
        return value

    result["q_R"] = pd.to_numeric(registry.get("q_R", 0.0), errors="coerce").fillna(0.0)
    result["trigger_family"] = coalesce(
        "physical_trigger_family", "mechanism_family", default="unknown"
    ).fillna("unknown").astype(str)
    result["trigger_date_quality"] = coalesce(
        "date_quality", default="canonical"
    ).fillna("unknown").astype(str)
    unique = coalesce("date_unique_and_canonical", "unique_event_date", default=0)
    result["trigger_date_unique"] = pd.to_numeric(unique, errors="coerce").fillna(0).astype(int)
    case_coverage = coalesce(
        "q_R_case_coverage", "chirps_coverage_complete", default=0
    )
    control_coverage = coalesce(
        "q_R_control_coverage", "chirps_coverage_complete", default=0
    )
    result["trigger_case_coverage"] = pd.to_numeric(
        case_coverage, errors="coerce"
    ).fillna(0.0)
    result["trigger_control_coverage"] = pd.to_numeric(
        control_coverage, errors="coerce"
    ).fillna(0.0)
    dose = coalesce("rain_d7_case_minus_wrongtime_mm", default=np.nan)
    if isinstance(dose, pd.Series):
        result["trigger_case_minus_wrongtime_mm"] = pd.to_numeric(dose, errors="coerce")
    else:
        result["trigger_case_minus_wrongtime_mm"] = np.nan
    return result


def reason_join(frame: pd.DataFrame, rules: list[tuple[str, pd.Series]]) -> pd.Series:
    reasons = np.full(len(frame), "", dtype=object)
    for name, failed in rules:
        failed_values = failed.fillna(True).to_numpy(dtype=bool)
        for index in np.flatnonzero(failed_values):
            reasons[index] = f"{reasons[index]};{name}".strip(";")
    return pd.Series(reasons, index=frame.index, dtype="string")


def summarize_flags(frame: pd.DataFrame) -> dict[str, Any]:
    flags = [
        "hard_quality_eligible",
        "qT_eligible",
        "qM_eligible",
        "qR_eligible",
        "full_tmr_eligible",
        "visual_degraded",
        "terrain_mechanism_opportunity",
        "full_tmr_mechanism_opportunity",
    ]
    summary: dict[str, Any] = {
        "n_samples": len(frame),
        "n_canonical_events": int(frame["canonical_event_id"].nunique()),
        "overall": {},
        "by_dataset": {},
    }
    for flag in flags:
        count = int(frame[flag].sum())
        summary["overall"][flag] = {"n": count, "fraction": count / max(len(frame), 1)}
    for dataset_id, group in frame.groupby("dataset_id", sort=True):
        values: dict[str, Any] = {
            "n_samples": len(group),
            "n_canonical_events": int(group["canonical_event_id"].nunique()),
        }
        for flag in flags:
            count = int(group[flag].sum())
            values[flag] = {"n": count, "fraction": count / max(len(group), 1)}
        summary["by_dataset"][str(dataset_id)] = values
    return summary


def build_report(summary: dict[str, Any], frame: pd.DataFrame) -> str:
    lines = [
        "# PILD label-independent support eligibility audit v1",
        "",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Samples: **{summary['n_samples']}**",
        f"- Canonical events: **{summary['n_canonical_events']}**",
        "- Labels, predictions, IoU/AP/RER, and corrected/harmed status were not read.",
        "- No source sample was deleted; this audit writes immutable eligibility flags.",
        "",
        "## Overall strata",
        "",
        "| stratum | n | fraction |",
        "|---|---:|---:|",
    ]
    for name, values in summary["overall"].items():
        lines.append(f"| {name} | {values['n']} | {values['fraction']:.2%} |")
    lines.extend(
        [
            "",
            "## By dataset",
            "",
            "| dataset | samples | events | hard | qT | qM | qR | full TMR | visual degraded |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_id, values in summary["by_dataset"].items():
        lines.append(
            f"| {dataset_id} | {values['n_samples']} | {values['n_canonical_events']} | "
            f"{values['hard_quality_eligible']['n']} | {values['qT_eligible']['n']} | "
            f"{values['qM_eligible']['n']} | {values['qR_eligible']['n']} | "
            f"{values['full_tmr_eligible']['n']} | {values['visual_degraded']['n']} |"
        )
    reason_counts = Counter(
        reason
        for value in frame["hard_exclusion_reasons"].fillna("")
        for reason in str(value).split(";")
        if reason
    )
    lines.extend(["", "## Hard-exclusion reasons", ""])
    if reason_counts:
        for reason, count in reason_counts.most_common():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `hard_quality_eligible=0` is the only status that may justify excluding a sample from the primary analysis.",
            "- `qT/qM/qR=0` means the corresponding physical branch must abstain exactly; the sample remains in the visual or other supported-role analysis.",
            "- `visual_degraded=1` defines a prespecified mechanism stratum, not a replacement test set.",
            "- Any effect computed after joining predictions must be reported beside the full hard-quality-eligible result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/pild_sen12_training_v2/unified_sample_manifest_v2.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/pild_sen12_training_v2/support_eligibility_v1"),
    )
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, low_memory=False)
    require_columns(
        manifest,
        {
            "dataset_id",
            "canonical_event_id",
            "sample_id",
            "optical_h5_path",
            "optical_h5_index",
            "terrain_h5_path",
            "terrain_h5_index",
            "terrain_channel_indices",
            "material_registry_path",
            "material_registry_index",
            "trigger_registry_path",
            "trigger_registry_index",
            "core_assets_ready",
        },
        "unified manifest",
    )
    if manifest["sample_id"].duplicated().any():
        raise ValueError("unified manifest sample_id is not unique")

    audit = manifest[
        [
            "manifest_index",
            "dataset_id",
            "source_id",
            "source_event_id",
            "canonical_event_id",
            "sample_id",
        ]
    ].copy()
    audit = pd.concat(
        [
            audit,
            optical_audit(manifest),
            terrain_audit(manifest),
            material_audit(manifest),
            trigger_audit(manifest),
        ],
        axis=1,
    )

    hard_rules = [
        ("core_assets_not_ready", manifest["core_assets_ready"].ne(1)),
        (
            "optical_valid_below_0.99",
            audit["optical_valid_fraction"].lt(
                POLICY["hard_quality"]["minimum_optical_valid_fraction"]
            ),
        ),
        (
            "optical_nonfinite",
            audit["optical_finite_fraction"].lt(
                POLICY["hard_quality"]["minimum_optical_finite_fraction"]
            ),
        ),
        (
            "optical_nearly_all_zero",
            audit["optical_zero_fraction"].gt(
                POLICY["hard_quality"]["maximum_optical_zero_fraction"]
            ),
        ),
    ]
    audit["hard_exclusion_reasons"] = reason_join(audit, hard_rules)
    audit["hard_quality_eligible"] = audit["hard_exclusion_reasons"].eq("").astype(int)

    audit["qT_eligible"] = (
        audit["hard_quality_eligible"].eq(1)
        & audit["terrain_valid_fraction"].ge(
            POLICY["terrain_support"]["minimum_terrain_valid_fraction"]
        )
        & audit["terrain_finite_fraction"].ge(
            POLICY["terrain_support"]["minimum_terrain_finite_fraction"]
        )
    ).astype(int)
    audit["terrain_state"] = np.select(
        [
            audit["qT_eligible"].eq(0),
            audit["terrain_slope_mean_deg"].lt(2.0)
            & audit["terrain_relief_mean_m"].lt(5.0),
        ],
        ["unsupported", "flat_veto"],
        default="varied",
    )
    audit["qM_eligible"] = (
        audit["hard_quality_eligible"].eq(1)
        & audit["q_M_full"].ge(POLICY["material_support"]["minimum_q_M_full"])
        & audit["material_valid_property_fraction"].ge(
            POLICY["material_support"]["minimum_soil_valid_property_fraction"]
        )
    ).astype(int)
    audit["qR_eligible"] = (
        audit["hard_quality_eligible"].eq(1)
        & audit["q_R"].ge(POLICY["trigger_support"]["minimum_q_R"])
        & audit["trigger_date_unique"].eq(1)
        & audit["trigger_case_coverage"].ge(0.999)
        & audit["trigger_control_coverage"].ge(0.999)
    ).astype(int)
    audit["full_tmr_eligible"] = (
        audit[["qT_eligible", "qM_eligible", "qR_eligible"]].eq(1).all(axis=1)
    ).astype(int)

    contrast_thresholds = (
        audit.loc[audit["hard_quality_eligible"].eq(1)]
        .groupby("dataset_id")["optical_spatial_contrast"]
        .quantile(0.10)
        .to_dict()
    )
    audit["low_contrast_threshold"] = audit["dataset_id"].map(contrast_thresholds)
    audit["visual_degraded_cloud"] = (
        audit["cloud_mean_fraction"].ge(
            POLICY["visual_degradation"]["cloud_mean_fraction"]
        )
        | audit["cloud_max_fraction"].ge(
            POLICY["visual_degradation"]["cloud_max_fraction"]
        )
    ).astype(int)
    audit["visual_degraded_temporal"] = audit["temporal_unique_fraction"].lt(
        POLICY["visual_degradation"]["minimum_temporal_unique_fraction"]
    ).astype(int)
    audit["visual_degraded_low_contrast"] = audit["optical_spatial_contrast"].le(
        audit["low_contrast_threshold"]
    ).astype(int)
    audit["visual_degraded"] = (
        audit["hard_quality_eligible"].eq(1)
        & audit[
            [
                "visual_degraded_cloud",
                "visual_degraded_temporal",
                "visual_degraded_low_contrast",
            ]
        ]
        .eq(1)
        .any(axis=1)
    ).astype(int)
    audit["terrain_mechanism_opportunity"] = (
        audit["qT_eligible"].eq(1) & audit["visual_degraded"].eq(1)
    ).astype(int)
    audit["full_tmr_mechanism_opportunity"] = (
        audit["full_tmr_eligible"].eq(1) & audit["visual_degraded"].eq(1)
    ).astype(int)

    event = (
        audit.groupby(["dataset_id", "canonical_event_id"], sort=True)
        .agg(
            n_samples=("sample_id", "size"),
            hard_eligible_fraction=("hard_quality_eligible", "mean"),
            qT_fraction=("qT_eligible", "mean"),
            qM_fraction=("qM_eligible", "mean"),
            qR_fraction=("qR_eligible", "mean"),
            full_tmr_fraction=("full_tmr_eligible", "mean"),
            visual_degraded_fraction=("visual_degraded", "mean"),
            terrain_opportunity_fraction=("terrain_mechanism_opportunity", "mean"),
            full_tmr_opportunity_fraction=("full_tmr_mechanism_opportunity", "mean"),
            trigger_family=("trigger_family", lambda values: Counter(values).most_common(1)[0][0]),
        )
        .reset_index()
    )
    event["event_hard_eligible"] = event["hard_eligible_fraction"].gt(0).astype(int)
    event["event_qT_eligible"] = event["qT_fraction"].eq(1).astype(int)
    event["event_qM_eligible"] = event["qM_fraction"].eq(1).astype(int)
    event["event_qR_eligible"] = event["qR_fraction"].eq(1).astype(int)
    event["event_full_tmr_eligible"] = event["full_tmr_fraction"].eq(1).astype(int)

    summary = summarize_flags(audit)
    summary.update(
        {
            "status": "complete",
            "generated_at_utc": utc_now(),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "policy": POLICY,
            "contrast_thresholds_by_dataset": contrast_thresholds,
            "n_events_with_any_hard_eligible_sample": int(event["event_hard_eligible"].sum()),
            "n_events_full_tmr_eligible": int(event["event_full_tmr_eligible"].sum()),
        }
    )

    policy_path = outdir / "filter_policy_v1.json"
    sample_path = outdir / "sample_eligibility_v1.csv"
    event_path = outdir / "event_eligibility_v1.csv"
    summary_path = outdir / "summary.json"
    report_path = outdir / "report.md"
    atomic_json(POLICY, policy_path)
    atomic_csv(audit, sample_path)
    atomic_csv(event, event_path)
    atomic_json(summary, summary_path)
    atomic_text(build_report(summary, audit), report_path)
    atomic_json(
        {
            "status": "complete",
            "generated_at_utc": utc_now(),
            "inputs": {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
            },
            "outputs": {
                path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in (policy_path, sample_path, event_path, summary_path, report_path)
            },
            "forbidden_inputs_confirmed": True,
        },
        outdir / "FREEZE.json",
    )
    print(json.dumps(json_safe(summary["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
