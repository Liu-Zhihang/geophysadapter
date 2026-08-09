#!/usr/bin/env python3
"""Audit whether CAS can support event-level Material/Trigger inference.

CAS patches do not have an audited patch-to-map transform, so this audit does
not test dense Terrain correction. It aggregates the segmentation labels only
to describe event burden, then checks a small, physically fixed set of
event-context variables without selecting variables from the outcomes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "metadata/pild_five_source_rgb_v3/unified_rgb_manifest_v3.csv"
)
DEFAULT_CONTEXT = PROJECT_ROOT / "processed/hybrid_pinn/cas_context_v1"
DEFAULT_OUTDIR = (
    PROJECT_ROOT
    / "experiments/revision2026/cas_event_context_feasibility_v1_20260724"
)

FIXED_CONTEXT_COLUMNS = {
    "mmi": "shakemap_mmi_median",
    "pga": "shakemap_pga_median",
    "magnitude": "usgs_magnitude",
    "awc_0_200": "awc_0_200_footprint_mean_mm",
    "clay_0_5": "soil_clay_0_5cm_mean_raw",
    "sand_0_5": "soil_sand_0_5cm_mean_raw",
    "lithology_dominance": "lithology_dominant_fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--context-dir", type=Path, default=DEFAULT_CONTEXT)
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


def aggregate_label_burden(cas: pd.DataFrame) -> pd.DataFrame:
    accumulators: dict[str, dict[str, Any]] = {}
    for h5_path, group in cas.groupby("rgb_h5_path", sort=False):
        with h5py.File(h5_path, "r") as handle:
            for row in group.itertuples(index=False):
                index = int(row.rgb_h5_index)
                label = np.asarray(handle[row.mask_dataset_key][index]) > 0.5
                valid = np.asarray(handle[row.valid_dataset_key][index]) > 0.5
                event = str(row.physical_event_id)
                item = accumulators.setdefault(
                    event,
                    {
                        "positive_pixels": 0,
                        "valid_pixels": 0,
                        "n_samples": 0,
                        "source_events": set(),
                    },
                )
                item["positive_pixels"] += int((label & valid).sum())
                item["valid_pixels"] += int(valid.sum())
                item["n_samples"] += 1
                item["source_events"].add(str(row.source_event_id))

    records = []
    for event, item in sorted(accumulators.items()):
        records.append(
            {
                "physical_event_id": event,
                "n_samples": item["n_samples"],
                "positive_pixels": item["positive_pixels"],
                "valid_pixels": item["valid_pixels"],
                "positive_fraction": (
                    item["positive_pixels"] / item["valid_pixels"]
                    if item["valid_pixels"]
                    else float("nan")
                ),
                "source_events": ";".join(sorted(item["source_events"])),
            }
        )
    return pd.DataFrame.from_records(records)


def exact_spearman_permutation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]
    if len(x_valid) < 4 or np.std(x_valid) == 0:
        return {"n": int(len(x_valid)), "rho": None, "exact_permutation_p": None}
    rho = float(spearmanr(x_valid, y_valid).statistic)
    permutation_statistics = [
        abs(float(spearmanr(x_valid, permutation).statistic))
        for permutation in itertools.permutations(y_valid.tolist())
    ]
    p_value = sum(
        statistic >= abs(rho) - 1e-12 for statistic in permutation_statistics
    ) / len(permutation_statistics)
    return {
        "n": int(len(x_valid)),
        "rho": rho,
        "exact_permutation_p": float(p_value),
        "n_exact_permutations": len(permutation_statistics),
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, keep_default_na=False, low_memory=False)
    cas = manifest.loc[manifest["dataset_id"].eq("CAS_Landslide")].copy()
    if len(cas) != 11_091:
        raise RuntimeError(f"Expected 11,091 CAS samples, found {len(cas):,}")
    if not pd.to_numeric(cas["q_T_asset"], errors="raise").eq(0).all():
        raise RuntimeError("CAS must remain q_T=0 without audited patch georeferencing")

    burden = aggregate_label_burden(cas)
    trigger = pd.read_csv(
        args.context_dir / "cas_trigger_event_registry_v1.csv",
        keep_default_na=False,
    )
    material = pd.read_csv(
        args.context_dir / "cas_material_event_registry_v1.csv",
        keep_default_na=False,
    )

    trigger_columns = [
        "physical_event_id",
        "shakemap_mmi_median",
        "shakemap_pga_median",
        "usgs_magnitude",
        "q_R",
    ]
    trigger_event = (
        trigger[trigger_columns]
        .groupby("physical_event_id", as_index=False)
        .mean(numeric_only=True)
    )
    material_columns = [
        "physical_event_id",
        "awc_0_200_footprint_mean_mm",
        "soil_clay_0_5cm_mean_raw",
        "soil_sand_0_5cm_mean_raw",
        "lithology_dominant_fraction",
        "q_M_full",
    ]
    for column in material_columns[1:]:
        material[column] = pd.to_numeric(material[column], errors="coerce")
    material_event = (
        material[material_columns]
        .groupby("physical_event_id", as_index=False)
        .mean(numeric_only=True)
    )
    event_table = burden.merge(
        trigger_event,
        on="physical_event_id",
        how="left",
        validate="one_to_one",
    ).merge(
        material_event,
        on="physical_event_id",
        how="left",
        validate="one_to_one",
    )

    associations = {}
    outcome = event_table["positive_fraction"].to_numpy(dtype=float)
    for label, column in FIXED_CONTEXT_COLUMNS.items():
        associations[label] = exact_spearman_permutation(
            pd.to_numeric(event_table[column], errors="coerce").to_numpy(dtype=float),
            outcome,
        )

    q_m_full_events = int(
        (pd.to_numeric(event_table["q_M_full"], errors="coerce") > 0).sum()
    )
    largest_event_fraction = float(
        event_table["n_samples"].max() / event_table["n_samples"].sum()
    )
    promotion_pass = (
        len(event_table) >= 10
        and q_m_full_events >= 8
        and largest_event_fraction <= 0.5
        and any(
            result["exact_permutation_p"] is not None
            and result["exact_permutation_p"] <= 0.05
            for result in associations.values()
        )
    )
    decision = (
        "PROMOTE_TO_EVENT_HELD_OUT_MR_SEGMENTATION"
        if promotion_pass
        else "DO_NOT_PROMOTE_CAS_AS_PRIMARY_PHYSICAL_EFFECT_DATASET"
    )

    event_table.to_csv(args.outdir / "event_context_and_label_burden.csv", index=False)
    summary = {
        "status": "complete",
        "scientific_status": "exploratory_event_level_feasibility_audit",
        "n_samples": int(len(cas)),
        "n_physical_events": int(len(event_table)),
        "largest_event_sample_fraction": largest_event_fraction,
        "q_T_events": 0,
        "q_M_full_events": q_m_full_events,
        "q_R_events": int(
            (pd.to_numeric(event_table["q_R"], errors="coerce") > 0).sum()
        ),
        "associations": associations,
        "promotion_criteria": {
            "at_least_10_physical_events": len(event_table) >= 10,
            "at_least_8_full_material_events": q_m_full_events >= 8,
            "largest_event_at_most_50_percent": largest_event_fraction <= 0.5,
            "fixed_context_association_exact_p_at_most_0_05": any(
                result["exact_permutation_p"] is not None
                and result["exact_permutation_p"] <= 0.05
                for result in associations.values()
            ),
        },
        "decision": decision,
        "interpretation_boundary": (
            "Label positive fraction reflects dataset sampling and annotation as well "
            "as event severity. Associations are feasibility diagnostics only and "
            "must not be used to select datasets or claim causal physical effects."
        ),
    }
    atomic_json(args.outdir / "summary.json", summary)

    lines = [
        "# CAS event-context feasibility audit",
        "",
        f"- Samples: `{len(cas):,}`.",
        f"- Independent physical events: `{len(event_table)}`.",
        f"- Largest event share: `{largest_event_fraction:.2%}`.",
        "- Dense Terrain eligibility: `0 events` (`q_T=0`).",
        f"- Full Material eligibility: `{q_m_full_events}/{len(event_table)} events`.",
        f"- Trigger eligibility: `{summary['q_R_events']}/{len(event_table)} events`.",
        f"- Decision: **{decision}**.",
        "",
        "## Fixed event-context associations",
        "",
        "| variable | n events | Spearman rho | exact permutation p |",
        "|---|---:|---:|---:|",
    ]
    for label, result in associations.items():
        rho = "NA" if result["rho"] is None else f"{result['rho']:+.3f}"
        p_value = (
            "NA"
            if result["exact_permutation_p"] is None
            else f"{result['exact_permutation_p']:.3f}"
        )
        lines.append(f"| {label} | {result['n']} | {rho} | {p_value} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "CAS patch counts must not be treated as independent physical-context "
            "replicates. Label positive fraction also reflects source-specific "
            "sampling and annotation. This audit therefore gates additional CAS "
            "training; it does not estimate a segmentation benefit and cannot "
            "justify outcome-based dataset removal.",
            "",
        ]
    )
    atomic_text(args.outdir / "report.md", "\n".join(lines))


if __name__ == "__main__":
    main()
