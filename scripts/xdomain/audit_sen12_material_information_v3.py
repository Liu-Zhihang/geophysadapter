#!/usr/bin/env python3
"""Audit real Material variability and scale matching in Sen12.

Variation and source-quality analyses are label-free. Target associations are
reported in a separate exploratory table and must not be used to select a
manuscript cohort or to claim confirmatory predictive value.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from material_factors_v3 import FACTOR_GROUPS, build_material_factors


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v2/material_sample_registry_v2.csv"
DEFAULT_H5 = PROJECT_ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5"
DEFAULT_OUT = PROJECT_ROOT / "metadata/reports/sen12_material_information_audit_v3_20260722"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


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


def variation_row(name: str, values: pd.Series, events: pd.Series, group: str) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.notna()
    x = numeric[finite]
    event = events[finite].astype(str)
    grand = float(x.mean()) if len(x) else math.nan
    total_ss = float(np.square(x - grand).sum()) if len(x) else 0.0
    group_mean = x.groupby(event).transform("mean")
    within_ss = float(np.square(x - group_mean).sum())
    between_ss = max(0.0, total_ss - within_ss)
    per_event_unique = x.groupby(event).nunique(dropna=True)
    event_medians = x.groupby(event).median()
    return {
        "factor": name,
        "factor_group": group,
        "n_finite": int(finite.sum()),
        "finite_fraction": float(finite.mean()),
        "global_mean": grand,
        "global_std": float(x.std(ddof=0)) if len(x) else math.nan,
        "global_unique": int(x.nunique(dropna=True)),
        "within_event_variance_fraction": within_ss / total_ss if total_ss > 0 else 0.0,
        "between_event_variance_fraction": between_ss / total_ss if total_ss > 0 else 0.0,
        "events_with_within_variation_fraction": float((per_event_unique > 1).mean()),
        "event_median_min": float(event_medians.min()) if len(event_medians) else math.nan,
        "event_median_max": float(event_medians.max()) if len(event_medians) else math.nan,
    }


def target_and_terrain(h5_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle:
        sample_ids = decode(handle["sample_id"][:])
        events = decode(handle["physical_event_id"][:])
        terrain_names = decode(handle["terrain_names"][:])
        slope_index = terrain_names.index("slope")
        relief_index = terrain_names.index("local_relief_300m")
        roughness_index = terrain_names.index("roughness_30m")
        for index, sample_id in enumerate(sample_ids):
            valid = handle["valid_mask"][index, 0].astype(bool)
            target = handle["mask"][index, 0].astype(bool)
            terrain = handle["terrain"][index]
            if valid.any():
                slope = terrain[slope_index][valid]
                relief = terrain[relief_index][valid]
                roughness = terrain[roughness_index][valid]
                rows.append({
                    "sample_id": sample_id,
                    "physical_event_id_h5": events[index],
                    "target_positive_fraction": float(target[valid].mean()),
                    "slope_mean_deg": float(np.nanmean(slope)),
                    "slope_p90_deg": float(np.nanpercentile(slope, 90)),
                    "relief_300m_mean_m": float(np.nanmean(relief)),
                    "roughness_30m_mean_m": float(np.nanmean(roughness)),
                })
            else:
                rows.append({
                    "sample_id": sample_id,
                    "physical_event_id_h5": events[index],
                    "target_positive_fraction": math.nan,
                    "slope_mean_deg": math.nan,
                    "slope_p90_deg": math.nan,
                    "relief_300m_mean_m": math.nan,
                    "roughness_30m_mean_m": math.nan,
                })
    return pd.DataFrame(rows)


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return math.nan, math.nan, len(pair)
    result = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(result.statistic), float(result.pvalue), len(pair)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry, low_memory=False)
    factors = build_material_factors(registry)
    events = registry["physical_event_id"].astype(str)

    group_by_factor = {
        name: group for group, names in FACTOR_GROUPS.items() for name in names
    }
    variation = pd.DataFrame([
        variation_row(name, factors[name], events, group_by_factor[name])
        for name in factors.columns
    ]).sort_values(["factor_group", "within_event_variance_fraction", "factor"], ascending=[True, False, True])
    variation.to_csv(args.outdir / "factor_variation.csv", index=False)

    audit = registry[["sample_id", "physical_event_id", "region_group", "q_M", "q_M_status"]].copy()
    audit = pd.concat([audit, factors], axis=1)
    diagnostics = target_and_terrain(args.h5)
    audit = audit.merge(diagnostics, on="sample_id", validate="one_to_one")
    mismatch = audit["physical_event_id"].astype(str) != audit["physical_event_id_h5"].astype(str)
    if mismatch.any():
        raise RuntimeError(f"registry/H5 event mismatch for {int(mismatch.sum())} samples")
    audit.to_csv(args.outdir / "sample_factors_and_exploratory_targets.csv", index=False)

    target = audit["target_positive_fraction"]
    target_within = target - target.groupby(audit["physical_event_id"]).transform("mean")
    association_rows: list[dict[str, Any]] = []
    for name in factors.columns:
        x = audit[name]
        x_within = x - x.groupby(audit["physical_event_id"]).transform("mean")
        rho_global, p_global, n_global = safe_spearman(x, target)
        rho_within, p_within, n_within = safe_spearman(x_within, target_within)
        association_rows.append({
            "factor": name,
            "factor_group": group_by_factor[name],
            "global_spearman_rho": rho_global,
            "global_spearman_p_exploratory": p_global,
            "within_event_spearman_rho": rho_within,
            "within_event_spearman_p_exploratory": p_within,
            "n_global": n_global,
            "n_within": n_within,
        })
    association = pd.DataFrame(association_rows)
    association.to_csv(args.outdir / "exploratory_target_association.csv", index=False)

    interaction_rows: list[dict[str, Any]] = []
    for material_name in (
        "awc_total_mean_mm",
        "awc_heterogeneity_rss_mm",
        "soil_fine_fraction_percent",
        "soil_coarse_fragment_mean_percent",
        "soil_bulk_density_mean_kg_dm3",
        "lithology_unconsolidated_fraction",
        "lithology_sedimentary_fraction",
    ):
        for terrain_name in ("slope_mean_deg", "slope_p90_deg", "relief_300m_mean_m"):
            x = audit[material_name] * audit[terrain_name]
            x_within = x - x.groupby(audit["physical_event_id"]).transform("mean")
            rho, p_value, n = safe_spearman(x_within, target_within)
            interaction_rows.append({
                "interaction": f"{terrain_name}_x_{material_name}",
                "terrain_factor": terrain_name,
                "material_factor": material_name,
                "within_event_spearman_rho": rho,
                "within_event_spearman_p_exploratory": p_value,
                "n": n,
            })
    interactions = pd.DataFrame(interaction_rows).sort_values(
        "within_event_spearman_rho", key=lambda value: value.abs(), ascending=False
    )
    interactions.to_csv(args.outdir / "exploratory_terrain_material_interactions.csv", index=False)

    variation_lookup = variation.set_index("factor")
    summary = {
        "scope": "Sen12 Material information and scale audit v3",
        "n_samples": int(len(audit)),
        "n_events": int(audit["physical_event_id"].nunique()),
        "n_regions": int(audit["region_group"].nunique()),
        "q_M_positive_fraction": float((pd.to_numeric(audit["q_M"], errors="coerce") > 0).mean()),
        "native_support": {
            "openlandmap_awc_resolution_m": 250,
            "soilgrids_resolution_m": 250,
            "glim_nominal_map_scale": "1:1,000,000",
            "sen12_patch_width_m": 1280,
            "median_awc_native_cells_per_patch": float(pd.to_numeric(registry["awc_min_native_cell_count"], errors="coerce").median()),
            "median_soilgrids_native_cells_per_patch": float(pd.to_numeric(registry["soil_min_native_cell_count"], errors="coerce").median()),
        },
        "factor_groups": {key: list(value) for key, value in FACTOR_GROUPS.items()},
        "median_within_event_variance_fraction": {
            group: float(variation[variation["factor_group"] == group]["within_event_variance_fraction"].median())
            for group in FACTOR_GROUPS
        },
        "awc_total_within_event_variance_fraction": float(variation_lookup.loc["awc_total_mean_mm", "within_event_variance_fraction"]),
        "lithology_unconsolidated_events_with_variation_fraction": float(variation_lookup.loc["lithology_unconsolidated_fraction", "events_with_within_variation_fraction"]),
        "interpretation_contract": {
            "dense_role": "none; Material remains a footprint-scale moderator of Terrain support",
            "permitted": "physically compressed Material factors may modulate an aligned Terrain susceptibility direction",
            "forbidden": "upsampling, synthetic jitter, label-based sample selection, or presenting exploratory correlations as confirmatory evidence",
        },
        "exploratory_only": "Target and Terrain interaction associations were inspected after factor definitions were frozen in code; they require new held-out intervention tests before promotion.",
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    strongest = association.assign(abs_within=association["within_event_spearman_rho"].abs()).sort_values("abs_within", ascending=False).head(8)
    lines = [
        "# Sen12 Material information audit v3",
        "",
        "## Data shape",
        "",
        f"- Samples/events/regions: {len(audit)}/{audit['physical_event_id'].nunique()}/{audit['region_group'].nunique()}.",
        f"- Valid complete Material support: {summary['q_M_positive_fraction']:.2%}.",
        "- OpenLandMap AWC and SoilGrids are approximately 250 m products summarized over each 1.28 km patch footprint; median native-cell counts are "
        f"{summary['native_support']['median_awc_native_cells_per_patch']:.0f} and {summary['native_support']['median_soilgrids_native_cells_per_patch']:.0f}.",
        "- GLiM is polygon geology at nominal 1:1,000,000 scale. It is context, not a dense boundary map.",
        "",
        "## Real variability",
        "",
        "| Factor group | Median within-event variance fraction |",
        "|---|---:|",
    ]
    for group, value in summary["median_within_event_variance_fraction"].items():
        lines.append(f"| {group} | {value:.3f} |")
    lines += [
        "",
        "## Exploratory within-event associations",
        "",
        "These correlations diagnose information content only. They are not model-selection or manuscript evidence.",
        "",
        "| Factor | Group | Spearman rho |",
        "|---|---|---:|",
    ]
    for row in strongest.itertuples():
        lines.append(f"| {row.factor} | {row.factor_group} | {row.within_event_spearman_rho:+.3f} |")
    lines += [
        "",
        "## Decision",
        "",
        "Material has real footprint-scale variation, but raw 55-dimensional conditioning is not justified by only 15 event contexts. The next admissible test is a low-dimensional, outer-training-only AWC/soil moderator of an aligned Terrain susceptibility map, with exact Terrain fallback and aligned-versus-event-shuffle controls. Synthetic variability and dense upsampling are prohibited.",
        "",
    ]
    (args.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
