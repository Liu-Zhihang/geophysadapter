#!/usr/bin/env python3
"""Deterministic, physically interpretable factors from Material registry v2.

The factors are footprint-scale moderators. They are not 10 m boundary maps,
and no fitted transform or segmentation label is used to construct them.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


AWC_DEPTHS = ("0_10", "10_30", "30_60", "60_100", "100_200")
SOIL_PROPERTIES = ("bdod", "cec", "cfvo", "clay", "phh2o", "sand", "silt", "soc")
SOIL_DEPTHS = ("0_5cm", "5_15cm")


FACTOR_GROUPS: Mapping[str, tuple[str, ...]] = {
    "awc_core": (
        "awc_total_mean_mm",
        "awc_shallow_mean_mm",
        "awc_deep_fraction",
        "awc_heterogeneity_rss_mm",
        "awc_vertical_contrast",
    ),
    "soil_hydraulic": (
        "soil_bulk_density_mean_kg_dm3",
        "soil_cec_mean_cmolc_kg",
        "soil_coarse_fragment_mean_percent",
        "soil_clay_mean_percent",
        "soil_silt_mean_percent",
        "soil_sand_mean_percent",
        "soil_fine_fraction_percent",
        "soil_soc_mean_g_kg",
        "soil_ph_mean",
        "soil_vertical_texture_contrast",
        "soil_heterogeneity_relative",
    ),
    "lithology_composition": (
        "lithology_unconsolidated_fraction",
        "lithology_sedimentary_fraction",
        "lithology_volcanic_fraction",
        "lithology_plutonic_fraction",
        "lithology_metamorphic_fraction",
        "lithology_normalized_entropy",
        "lithology_contact_indicator",
    ),
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise KeyError(f"Material registry is missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def _mean_depth(frame: pd.DataFrame, prop: str) -> pd.Series:
    values = [
        _numeric(frame, f"soil_{prop}_{depth}_mean_{_soil_unit(prop)}")
        for depth in SOIL_DEPTHS
    ]
    return pd.concat(values, axis=1).mean(axis=1)


def _soil_unit(prop: str) -> str:
    return {
        "bdod": "kg_dm3",
        "cec": "cmolc_kg",
        "cfvo": "percent_volume",
        "clay": "g_kg",
        "phh2o": "pH",
        "sand": "g_kg",
        "silt": "g_kg",
        "soc": "g_kg",
    }[prop]


def build_material_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Return label-free physical factors with the input index preserved."""

    output = pd.DataFrame(index=frame.index)

    awc_means = pd.concat(
        [_numeric(frame, f"awc_{depth}_mean_mm") for depth in AWC_DEPTHS], axis=1
    )
    awc_stds = pd.concat(
        [_numeric(frame, f"awc_{depth}_std_mm") for depth in AWC_DEPTHS], axis=1
    )
    output["awc_total_mean_mm"] = awc_means.sum(axis=1, min_count=len(AWC_DEPTHS))
    output["awc_shallow_mean_mm"] = awc_means.iloc[:, :2].sum(axis=1, min_count=2)
    deep = awc_means.iloc[:, 2:].sum(axis=1, min_count=3)
    output["awc_deep_fraction"] = deep / output["awc_total_mean_mm"].replace(0.0, np.nan)
    output["awc_heterogeneity_rss_mm"] = np.sqrt(np.square(awc_stds).sum(axis=1))
    layer_thickness_cm = np.asarray([10.0, 20.0, 30.0, 40.0, 100.0])
    density = awc_means.to_numpy(dtype=float) / layer_thickness_cm[None, :]
    output["awc_vertical_contrast"] = np.nanstd(density, axis=1)

    output["soil_bulk_density_mean_kg_dm3"] = _mean_depth(frame, "bdod")
    output["soil_cec_mean_cmolc_kg"] = _mean_depth(frame, "cec")
    output["soil_coarse_fragment_mean_percent"] = _mean_depth(frame, "cfvo")
    output["soil_clay_mean_percent"] = _mean_depth(frame, "clay")
    output["soil_silt_mean_percent"] = _mean_depth(frame, "silt")
    output["soil_sand_mean_percent"] = _mean_depth(frame, "sand")
    output["soil_fine_fraction_percent"] = (
        output["soil_clay_mean_percent"] + output["soil_silt_mean_percent"]
    )
    output["soil_soc_mean_g_kg"] = _mean_depth(frame, "soc")
    output["soil_ph_mean"] = _mean_depth(frame, "phh2o")

    texture_contrasts = []
    for prop in ("clay", "silt", "sand"):
        unit = _soil_unit(prop)
        shallow = _numeric(frame, f"soil_{prop}_0_5cm_mean_{unit}")
        deep_value = _numeric(frame, f"soil_{prop}_5_15cm_mean_{unit}")
        texture_contrasts.append(np.square(shallow - deep_value))
    output["soil_vertical_texture_contrast"] = np.sqrt(
        pd.concat(texture_contrasts, axis=1).sum(axis=1)
    )

    relative_heterogeneity = []
    for prop in SOIL_PROPERTIES:
        unit = _soil_unit(prop)
        for depth in SOIL_DEPTHS:
            mean = _numeric(frame, f"soil_{prop}_{depth}_mean_{unit}").abs()
            std = _numeric(frame, f"soil_{prop}_{depth}_std_{unit}")
            relative_heterogeneity.append(std / mean.clip(lower=1e-6))
    output["soil_heterogeneity_relative"] = pd.concat(
        relative_heterogeneity, axis=1
    ).median(axis=1)

    frac = lambda code: _numeric(frame, f"limw_frac_{code}")
    output["lithology_unconsolidated_fraction"] = frac("su")
    output["lithology_sedimentary_fraction"] = (
        frac("sc") + frac("sm") + frac("ss") + frac("su")
    )
    output["lithology_volcanic_fraction"] = (
        frac("va") + frac("vb") + frac("vi") + frac("py")
    )
    output["lithology_plutonic_fraction"] = frac("pa") + frac("pb") + frac("pi")
    output["lithology_metamorphic_fraction"] = frac("mt")
    output["lithology_normalized_entropy"] = _numeric(frame, "limw_normalized_entropy")
    output["lithology_contact_indicator"] = (
        _numeric(frame, "limw_broad_class_count") > 1
    ).astype(float)

    return output.replace([np.inf, -np.inf], np.nan)


def factor_names(group: str = "all") -> tuple[str, ...]:
    if group == "all":
        return tuple(name for names in FACTOR_GROUPS.values() for name in names)
    if group == "awc_soil":
        return FACTOR_GROUPS["awc_core"] + FACTOR_GROUPS["soil_hydraulic"]
    if group not in FACTOR_GROUPS:
        raise KeyError(f"unknown Material factor group: {group}")
    return FACTOR_GROUPS[group]


__all__ = ["FACTOR_GROUPS", "build_material_factors", "factor_names"]
