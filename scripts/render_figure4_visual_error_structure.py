#!/usr/bin/env python3
"""Render Figure 4: the object-level structure of cross-domain visual errors.

Four panels, diagnostic rather than performance-oriented:
  (a) three baseline-only OOF examples chosen for visual readability of large,
      near-pure false bodies under a frozen score (distinct confusion contexts);
  (b) false-positive mass over the candidate-body area x purity plane;
  (c) pixel-mass retention as the candidate-body area floor increases;
  (d) the scale bridge: mass-weighted candidate-body size against the native
      support scale of every geophysical prior.

The pooled error budget and per-source near-pure shares are reported in the
manuscript text rather than as figure panels.

Layout targets the ISPRS JPRS double-column width (176 mm). Only a 600 dpi PNG
is written, following the revision package image policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from PIL import Image
from scipy import ndimage

from figure_chrome import export_axes


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPERIMENTS = PROJECT_ROOT / "experiments/revision2026"
DEFAULT_OUTDIR = (
    PROJECT_ROOT.parent
    / "docs/assets"
)

DECISIONS_PATH = (
    EXPERIMENTS / "pild_object_veto_final_v1/component_decisions.parquet"
)
SUMMARY_PATH = EXPERIMENTS / "pild_object_veto_final_v1/summary.json"
OOF_CACHE = EXPERIMENTS / "pild_object_physical_diagnostic_v1/oof_cache"
FOLD_IDS = [f"source_stratified_{index}" for index in range(4)]

ANALYSIS_GSD_M = 10.0
CANVAS_WIDTH_MM = 176.0
CANVAS_HEIGHT_MM = 139.0
PRIOR_SCALES_M = {
    "10 m pixel": 10.0,
    "30 m terrain": 30.0,
    "250 m material": 250.0,
    "~5 km trigger": 5000.0,
}

# Sentinel-2 band order frozen with the optical cache.
BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = range(6)

# --- Palette -----------------------------------------------------------------
INK = "#1E242B"
MUTED = "#6B7580"
GRID = "#DFE4E9"
TP_COLOR = "#3E7F76"
FP_COLOR = "#C0503C"
FN_COLOR = "#D9B570"
GRAY = "#AAB3BC"
THRESH_COLOR = "#49545E"
THRESH_STYLE = (0, (3.0, 2.2))
THRESH_LW = 1.0
FP_CMAP = LinearSegmentedColormap.from_list(
    "fp_mass", ["#F6EFEC", "#E8B7A6", "#D07C63", "#C0503C", "#7E2E20"]
)

SOURCE_LABELS = {
    "DLR_Landslide_Ref_2025": "DLR",
    "GDCLD": "GDCLD",
    "SEN12LS_HARMONIZED": "Sen12",
    "GLaD4CD_v1": "GLaD4CD",
}
SOURCE_LABELS_LONG = {
    "DLR_Landslide_Ref_2025": "DLR reference",
    "GDCLD": "GDCLD",
    "SEN12LS_HARMONIZED": "Sen12Landslides",
    "GLaD4CD_v1": "GLaD4CD",
}
ALL_SOURCES = [
    "SEN12LS_HARMONIZED",
    "GDCLD",
    "DLR_Landslide_Ref_2025",
    "GLaD4CD_v1",
]

# Contexts shown in panel (a), in draw order.
CONTEXT_ORDER = [
    "bare or cultivated soil",
    "linear corridor",
    "bright bare surface",
]

SOURCE_ORDER = [
    "DLR_Landslide_Ref_2025",
    "GDCLD",
    "SEN12LS_HARMONIZED",
]

# Original pre-registered panel-(a) tiles (median-near-pure rule, first freeze).
FIXED_PANEL_A = [
    {
        "dataset_id": "DLR_Landslide_Ref_2025",
        "sample_id": "EID_KG0002__SID_00320",
        "context": "bare or cultivated soil",
    },
    {
        "dataset_id": "GDCLD",
        "sample_id": "GDCLD_Mesetas::Mesetas/Mesetas.tif::g10_r0c2176",
        "context": "linear corridor",
    },
    {
        "dataset_id": "SEN12LS_HARMONIZED",
        "sample_id": "SEN12_S2_italy_5219",
        "context": "bright bare surface",
    },
]

# --- Frozen example-selection rule -------------------------------------------
# Restored first-freeze rule: one tile per source, nearest the source median
# near-pure FP mass, distinct spectral contexts, body area inside the IQR.
RULE = {
    "candidate_area_px_min": 200,
    "candidate_purity_max": 0.10,
    "near_pure_fp_px_min": 200,
    "sample_predicted_tp_px_min": 500,
    "largest_near_pure_body_area_within": "interquartile range of all eligible bodies",
    "separation_dilation_px": 5,
    "context_ndvi_vegetated_min": 0.35,
    "context_ndvi_bare_max": 0.25,
    "context_brightness_percentile": 80,
    "context_elongation_min": 3.0,
    "contexts_shown": list(CONTEXT_ORDER),
    "within_source_choice": "nearest to eligible source median near-pure FP mass",
    "fixed_panel_a_samples": [row["sample_id"] for row in FIXED_PANEL_A],
    "adapter_outcome_used": False,
}


def configure_style() -> None:
    """Type sizes are final printed sizes on the 176 mm manuscript canvas."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.7,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.linewidth": 0.9,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(ax, text: str, *, x: float = -0.20, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        f"({text})",
        transform=ax.transAxes,
        fontsize=10.8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )


def panel_header_row(
    fig,
    axes_titles: list[tuple[object, str, str]],
    *,
    dy: float = 0.010,
    title_dx: float = 0.054,
) -> None:
    """Place (b)/(c)/(d) letters and titles on one shared baseline.

    `title_dx` is deliberately wider than the letter width so "(b)" and the
    title do not crowd each other.
    """

    y = max(ax.get_position().y1 for ax, _, _ in axes_titles) + dy
    for ax, letter, title in axes_titles:
        bbox = ax.get_position()
        fig.text(
            bbox.x0,
            y,
            f"({letter})",
            fontsize=10.8,
            fontweight="bold",
            va="bottom",
            ha="left",
            color=INK,
        )
        fig.text(
            bbox.x0 + title_dx,
            y,
            title,
            fontsize=9.5,
            fontweight="semibold",
            va="bottom",
            ha="left",
            color=INK,
            linespacing=1.05,
        )


def tidy(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, alpha=0.85)
    ax.set_axisbelow(True)


def equivalent_diameter_m(area_px: np.ndarray) -> np.ndarray:
    return 2.0 * ANALYSIS_GSD_M * np.sqrt(np.asarray(area_px, dtype=float) / np.pi)


def mass_weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: list[float]
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    cumulative = cumulative / cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def event_prefix(sample_id: str) -> str:
    """Stable event key for soft diversity (source-agnostic prefix of sample id)."""

    text = str(sample_id)
    if text.startswith("SEN12_"):
        parts = text.split("_")
        return "_".join(parts[:4]) if len(parts) >= 4 else text
    if text.startswith("EID_"):
        return text.split("__")[0]
    if "::" in text:
        return text.split("::")[0]
    return text


# --- Example selection -------------------------------------------------------


def build_fold_index() -> dict[str, tuple[str, int]]:
    index: dict[str, tuple[str, int]] = {}
    for fold_id in FOLD_IDS:
        with np.load(OOF_CACHE / f"{fold_id}_oof_cache.npz", allow_pickle=False) as h:
            for position, sample_id in enumerate(h["sample_id"]):
                index[str(sample_id)] = (fold_id, position)
    return index


def eligible_tiles(decisions: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Tiles that satisfy the first-freeze eligibility thresholds, per source."""

    sample_tp = (
        decisions.groupby(["dataset_id", "sample_id"], as_index=False)
        .intersection_px.sum()
        .rename(columns={"intersection_px": "sample_tp"})
    )
    bodies = decisions[
        (decisions.area_px >= RULE["candidate_area_px_min"])
        & (decisions.purity <= RULE["candidate_purity_max"])
    ]
    q1, q3 = np.percentile(bodies.area_px, [25, 75])
    tiles = (
        bodies.groupby(["dataset_id", "sample_id"], as_index=False)
        .agg(
            near_pure_fp=("false_px", "sum"),
            near_pure_bodies=("component_id", "size"),
            largest_near_pure_area=("area_px", "max"),
        )
        .merge(sample_tp, on=["dataset_id", "sample_id"], how="inner")
    )
    tiles = tiles[
        (tiles.near_pure_fp >= RULE["near_pure_fp_px_min"])
        & (tiles.sample_tp >= RULE["sample_predicted_tp_px_min"])
        & (tiles.largest_near_pure_area >= q1)
        & (tiles.largest_near_pure_area <= q3)
    ].copy()
    # Always keep the frozen panel-(a) tiles in the load pool.
    fixed_ids = {row["sample_id"] for row in FIXED_PANEL_A}
    fixed_rows = (
        bodies.groupby(["dataset_id", "sample_id"], as_index=False)
        .agg(
            near_pure_fp=("false_px", "sum"),
            near_pure_bodies=("component_id", "size"),
            largest_near_pure_area=("area_px", "max"),
        )
        .merge(sample_tp, on=["dataset_id", "sample_id"], how="inner")
    )
    fixed_rows = fixed_rows[fixed_rows.sample_id.isin(fixed_ids)]
    tiles = (
        pd.concat([tiles, fixed_rows], ignore_index=True)
        .drop_duplicates(subset=["dataset_id", "sample_id"])
        .reset_index(drop=True)
    )
    return tiles, float(q1), float(q3)


def stretch_rgb(post: np.ndarray) -> np.ndarray:
    rgb = np.stack([post[RED], post[GREEN], post[BLUE]], axis=-1).astype(np.float32)
    for channel in range(3):
        band = rgb[..., channel]
        finite = band[np.isfinite(band)]
        low, high = np.percentile(finite, [2, 98])
        rgb[..., channel] = np.clip((band - low) / max(high - low, 1e-6), 0.0, 1.0)
    return np.power(rgb, 0.85)


def near_pure_mask(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(
        predicted, structure=ndimage.generate_binary_structure(2, 2)
    )
    union = np.zeros_like(predicted, dtype=bool)
    for component_id in range(1, count + 1):
        mask = labels == component_id
        area = int(mask.sum())
        if area < RULE["candidate_area_px_min"]:
            continue
        purity = float((mask & target).sum() / max(area, 1))
        if purity <= RULE["candidate_purity_max"]:
            union |= mask
    return union


def largest_elongation(mask: np.ndarray) -> float:
    labels, count = ndimage.label(
        mask, structure=ndimage.generate_binary_structure(2, 2)
    )
    if count == 0:
        return 1.0
    sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
    biggest = int(np.argmax(sizes)) + 1
    rows, cols = np.nonzero(labels == biggest)
    if rows.size < 3:
        return 1.0
    covariance = np.cov(np.vstack([rows.astype(float), cols.astype(float)]))
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 1e-9, None)
    return float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))


def load_candidates(
    tiles: pd.DataFrame, fold_index: dict[str, tuple[str, int]]
) -> dict[str, dict]:
    wanted: dict[str, list[tuple[str, int]]] = {}
    for sample_id in tiles.sample_id:
        fold_id, position = fold_index[str(sample_id)]
        wanted.setdefault(fold_id, []).append((str(sample_id), position))

    loaded: dict[str, dict] = {}
    dilate = int(RULE["separation_dilation_px"])
    for fold_id, entries in wanted.items():
        receipt = json.loads(
            (OOF_CACHE / f"{fold_id}_oof_cache_receipt.json").read_text("utf-8")
        )
        threshold = float(receipt["threshold"])
        rows = [position for _, position in entries]
        with np.load(OOF_CACHE / f"{fold_id}_oof_cache.npz", allow_pickle=False) as h:
            probability = h["visual_probability"][rows].astype(np.float32)
            target = h["target"][rows].astype(bool)
            valid = h["valid"][rows].astype(bool)
        with np.load(
            OOF_CACHE / f"{fold_id}_optical_cache.npz", allow_pickle=False
        ) as h:
            post = h["optical_post"][rows].astype(np.float32)

        for slot, (sample_id, _) in enumerate(entries):
            tile_target = target[slot] & valid[slot]
            predicted = (probability[slot] >= threshold) & valid[slot]
            false_bodies = near_pure_mask(predicted, tile_target)
            cube = post[slot]
            probe = false_bodies if false_bodies.any() else (predicted & ~tile_target)
            nir, red, green = cube[NIR][probe], cube[RED][probe], cube[GREEN][probe]

            tp_px = int((predicted & tile_target).sum())
            fp_px = int((predicted & ~tile_target).sum())
            fn_px = int((~predicted & tile_target).sum())
            near_pure_fp = int((false_bodies & ~tile_target).sum())
            if false_bodies.any():
                target_halo = ndimage.binary_dilation(tile_target, iterations=dilate)
                touch = float((false_bodies & target_halo).sum() / false_bodies.sum())
            else:
                touch = 1.0
            separation = 1.0 - touch
            labeled = max(tp_px + fp_px + fn_px, 1)
            clutter = (tp_px + fn_px) / labeled
            largest = 0
            n_near_pure = 0
            compactness = 0.0
            if false_bodies.any():
                labels, count = ndimage.label(
                    false_bodies, structure=ndimage.generate_binary_structure(2, 2)
                )
                sizes = ndimage.sum(false_bodies, labels, index=range(1, count + 1))
                n_near_pure = int(count)
                biggest = int(np.argmax(sizes)) + 1
                largest = int(sizes[biggest - 1])
                rows_idx, cols_idx = np.nonzero(labels == biggest)
                bbox = max(
                    (rows_idx.max() - rows_idx.min() + 1)
                    * (cols_idx.max() - cols_idx.min() + 1),
                    1,
                )
                compactness = largest / bbox

            tile_area = float(predicted.shape[0] * predicted.shape[1])
            cover = largest / max(tile_area, 1.0)
            # Prefer one compact body; penalise fragmented whole-tile floods.
            cover_penalty = 1.0 - 0.85 * max(cover - 0.10, 0.0) / 0.90
            body_penalty = 1.0 - 0.18 * max(n_near_pure - 1, 0)
            readability = (
                np.log1p(largest)
                * (0.25 + separation)
                * (0.35 + near_pure_fp / max(fp_px, 1))
                * (1.0 - 0.55 * clutter)
                * (0.35 + 0.65 * compactness)
                * cover_penalty
                * body_penalty
            )

            loaded[sample_id] = {
                "rgb": stretch_rgb(cube),
                "target": tile_target,
                "predicted": predicted,
                "near_pure": false_bodies,
                "threshold": threshold,
                "fold_id": fold_id,
                "ndvi": float(np.median((nir - red) / (nir + red + 1e-6))),
                "ndwi": float(np.median((green - nir) / (green + nir + 1e-6))),
                "brightness": float(np.median(cube.mean(axis=0)[probe])),
                "elongation": largest_elongation(false_bodies),
                "separation": float(separation),
                "clutter": float(clutter),
                "largest_near_pure_area": largest,
                "near_pure_fp_px": near_pure_fp,
                "near_pure_tile_cover": float(cover),
                "near_pure_bodies": n_near_pure,
                "near_pure_compactness": float(compactness),
                "readability_score": float(readability),
            }
        del probability, target, valid, post
    return loaded


def assign_context(candidates: dict[str, dict]) -> dict[str, str]:
    brightness = np.asarray([item["brightness"] for item in candidates.values()])
    bright_high = float(np.percentile(brightness, RULE["context_brightness_percentile"]))
    contexts: dict[str, str] = {}
    for sample_id, item in candidates.items():
        if item["elongation"] >= RULE["context_elongation_min"]:
            contexts[sample_id] = "linear corridor"
        elif (
            item["brightness"] >= bright_high
            and item["ndvi"] < RULE["context_ndvi_bare_max"]
        ):
            contexts[sample_id] = "bright bare surface"
        elif item["ndvi"] >= RULE["context_ndvi_vegetated_min"]:
            contexts[sample_id] = "vegetated slope"
        else:
            contexts[sample_id] = "bare or cultivated soil"
    contexts["__brightness_threshold__"] = bright_high  # type: ignore[assignment]
    return contexts


def choose_examples(
    tiles: pd.DataFrame, candidates: dict[str, dict], contexts: dict[str, str]
) -> pd.DataFrame:
    """Return the first-freeze panel-(a) tiles in source draw order."""

    del contexts  # contexts are frozen with FIXED_PANEL_A
    rows = []
    for record in FIXED_PANEL_A:
        sample_id = record["sample_id"]
        if sample_id not in candidates:
            raise RuntimeError(f"Frozen panel-(a) tile missing from cache: {sample_id}")
        match = tiles[tiles.sample_id == sample_id]
        if match.empty:
            raise RuntimeError(f"Frozen panel-(a) tile missing from decisions: {sample_id}")
        row = match.iloc[0].to_dict()
        row["context"] = record["context"]
        row["separation"] = candidates[sample_id]["separation"]
        row["clutter"] = candidates[sample_id]["clutter"]
        row["readability_score"] = candidates[sample_id]["readability_score"]
        row["near_pure_tile_cover"] = candidates[sample_id]["near_pure_tile_cover"]
        row["event_key"] = event_prefix(sample_id)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["ndvi"] = out.sample_id.map(lambda s: candidates[str(s)]["ndvi"])
    out["brightness"] = out.sample_id.map(lambda s: candidates[str(s)]["brightness"])
    out["elongation"] = out.sample_id.map(lambda s: candidates[str(s)]["elongation"])
    out["fold_id"] = out.sample_id.map(lambda s: candidates[str(s)]["fold_id"])
    out["draw_rank"] = out.dataset_id.map(SOURCE_ORDER.index)
    return out.sort_values("draw_rank").reset_index(drop=True)


# --- Panel drawing -----------------------------------------------------------


def draw_case(ax, tile: dict, source_label: str, context: str) -> None:
    target = np.asarray(tile["target"], dtype=bool)
    predicted = np.asarray(tile["predicted"], dtype=bool)
    near_pure = np.asarray(tile["near_pure"], dtype=bool)

    overlay = np.zeros((*target.shape, 4), dtype=float)
    overlay[predicted & target] = mpl.colors.to_rgba(TP_COLOR, 0.46)
    overlay[predicted & ~target] = mpl.colors.to_rgba(FP_COLOR, 0.46)
    overlay[~predicted & target] = mpl.colors.to_rgba(FN_COLOR, 0.58)

    ax.imshow(np.asarray(tile["rgb"]))
    ax.imshow(overlay)

    if target.any():
        ax.contour(target.astype(float), levels=[0.5], colors="white", linewidths=1.2)
        dashed = ax.contour(
            target.astype(float),
            levels=[0.5],
            colors=INK,
            linewidths=0.50,
            alpha=0.76,
        )
        dashed.set_dashes([(0.0, (2.6, 1.8))])
    if near_pure.any():
        ax.contour(
            near_pure.astype(float), levels=[0.5], colors="white", linewidths=2.4
        )
        ax.contour(
            near_pure.astype(float), levels=[0.5], colors=FP_COLOR, linewidths=1.5
        )

    ax.text(
        0.035,
        0.965,
        source_label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": INK,
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )
    ax.text(
        0.5,
        -0.018,
        context,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )

    height, width = target.shape
    bar_px = 500.0 / ANALYSIS_GSD_M
    x0, y0 = 0.055 * width, 0.945 * height
    plate_x = x0 - 2.5
    plate_y = y0 - 11.5
    ax.add_patch(
        Rectangle(
            (plate_x, plate_y),
            bar_px + 5.0,
            15.0,
            facecolor=INK,
            edgecolor="none",
            alpha=0.62,
            zorder=8,
        )
    )
    ax.plot(
        [x0, x0 + bar_px],
        [y0, y0],
        color="white",
        linewidth=2.1,
        solid_capstyle="butt",
        zorder=9,
    )
    ax.text(
        x0 + bar_px / 2.0,
        y0 - 0.030 * height,
        "500 m",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="white",
        fontweight="bold",
        zorder=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.8)


def draw_mass_plane(ax, fig, decisions: pd.DataFrame) -> dict[str, float]:
    total_fp = float(decisions.false_px.sum())
    contributing = decisions[decisions.false_px > 0]
    percent = contributing.false_px.to_numpy(dtype=float) / total_fp * 100.0
    hexes = ax.hexbin(
        contributing.area_px.to_numpy(dtype=float),
        contributing.purity.to_numpy(dtype=float),
        C=percent,
        reduce_C_function=np.sum,
        gridsize=(26, 15),
        xscale="log",
        cmap=FP_CMAP,
        norm=LogNorm(vmin=0.01, vmax=float(percent.sum() * 0.25)),
        mincnt=1,
        linewidths=0.0,
    )

    near_pure_share = float(
        decisions.loc[decisions.purity <= RULE["candidate_purity_max"], "false_px"].sum()
        / total_fp
    )
    corner = decisions[
        (decisions.area_px >= RULE["candidate_area_px_min"])
        & (decisions.purity <= RULE["candidate_purity_max"])
    ]
    corner_share = float(corner.false_px.sum() / total_fp)
    max_area = float(decisions.area_px.max())
    floor = float(RULE["candidate_area_px_min"])
    purity_cut = float(RULE["candidate_purity_max"])
    ax.axvspan(
        floor,
        max_area * 1.25,
        ymin=0.0,
        ymax=purity_cut,
        color=FP_COLOR,
        alpha=0.07,
        linewidth=0,
        zorder=1,
    )
    ax.axvline(
        floor,
        color=THRESH_COLOR,
        linewidth=THRESH_LW,
        linestyle=THRESH_STYLE,
        zorder=5,
    )
    ax.axhline(
        purity_cut,
        color=THRESH_COLOR,
        linewidth=THRESH_LW,
        linestyle=THRESH_STYLE,
        zorder=5,
    )
    ax.text(
        0.035,
        0.975,
        "FP mass share\n"
        f"Purity $\\leq$ 0.10: {near_pure_share:.1%}\n"
        f"Also area $\\geq$ 200 px: {corner_share:.1%}",
        transform=ax.transAxes,
        fontsize=7.6,
        color=INK,
        ha="left",
        va="top",
        linespacing=1.28,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": GRID,
            "linewidth": 0.5,
            "alpha": 0.94,
        },
        zorder=6,
    )
    ax.set_xscale("log")
    ax.set_xlim(1.0, max_area * 1.25)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("candidate body area (pixels)")
    ax.set_ylabel("false-positive purity", labelpad=2)
    ax.tick_params(axis="both", pad=2)
    # A very short bar at mid-purity leaves the decisive low-purity region clear.
    cax = ax.inset_axes([0.925, 0.27, 0.026, 0.25])
    bar = fig.colorbar(hexes, cax=cax, orientation="vertical")
    bar.ax.set_title("FP mass\nper hexbin (%)", fontsize=6.3, pad=2.5)
    bar.set_ticks([0.01, 0.1, 1.0, 10.0])
    bar.ax.tick_params(labelsize=6.1, pad=1.0, length=2.2)
    bar.outline.set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    return {"corner_share": corner_share, "near_pure_share": near_pure_share}


def mass_survival(
    decisions: pd.DataFrame, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    areas = decisions.area_px.to_numpy(dtype=float)
    tp = decisions.intersection_px.to_numpy(dtype=float)
    fp = decisions.false_px.to_numpy(dtype=float)
    tp_share = np.asarray([tp[areas >= cut].sum() / tp.sum() for cut in thresholds])
    fp_share = np.asarray([fp[areas >= cut].sum() / fp.sum() for cut in thresholds])
    return tp_share, fp_share


def draw_survival(ax, decisions: pd.DataFrame) -> dict[str, float]:
    max_area = float(decisions.area_px.max())
    thresholds = np.unique(np.rint(np.geomspace(1, max_area, 180))).astype(float)
    tp_share, fp_share = mass_survival(decisions, thresholds)
    ax.plot(thresholds, tp_share, color=TP_COLOR, linewidth=2.0, label="TP mass")
    ax.plot(thresholds, fp_share, color=FP_COLOR, linewidth=2.0, label="FP mass")

    floor = float(RULE["candidate_area_px_min"])
    ax.axvline(
        floor,
        color=THRESH_COLOR,
        linewidth=THRESH_LW,
        linestyle=THRESH_STYLE,
    )
    tp200, fp200 = mass_survival(decisions, np.asarray([floor]))
    ax.scatter(
        [floor, floor],
        [tp200[0], fp200[0]],
        color=[TP_COLOR, FP_COLOR],
        s=34,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.annotate(
        f"TP: {tp200[0]:.1%}",
        xy=(floor, tp200[0]),
        xytext=(380.0, 0.86),
        fontsize=7.8,
        color=TP_COLOR,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": TP_COLOR, "lw": 0.8},
    )
    ax.annotate(
        f"FP: {fp200[0]:.1%}",
        xy=(floor, fp200[0]),
        xytext=(380.0, 0.53),
        fontsize=7.8,
        color=FP_COLOR,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": FP_COLOR, "lw": 0.8},
    )
    blend = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        floor * 1.15,
        0.985,
        "200 px threshold",
        transform=blend,
        fontsize=7.2,
        color=MUTED,
        ha="left",
        va="top",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("minimum body area (pixels)")
    ax.set_ylabel("retained mass fraction", labelpad=2)
    ax.tick_params(axis="both", pad=2)
    ax.legend(frameon=False, loc="lower left", handlelength=1.4, borderaxespad=0.2)
    tidy(ax, grid_axis="both")
    return {"tp_at_floor": float(tp200[0]), "fp_at_floor": float(fp200[0])}


def draw_scale_bridge(ax, decisions: pd.DataFrame) -> dict[str, float]:
    diameter = equivalent_diameter_m(decisions.area_px.to_numpy(dtype=float))
    order = np.argsort(diameter)
    sorted_diameter = diameter[order]

    stats: dict[str, float] = {}
    for name, column, color in (
        ("TP mass", "intersection_px", TP_COLOR),
        ("FP mass", "false_px", FP_COLOR),
    ):
        weights = decisions[column].to_numpy(dtype=float)[order]
        cumulative = np.cumsum(weights) / weights.sum()
        ax.plot(sorted_diameter, cumulative, color=color, linewidth=2.0, label=name)
        median = float(np.interp(0.5, cumulative, sorted_diameter))
        ax.scatter(
            [median],
            [0.5],
            color=color,
            s=34,
            zorder=6,
            edgecolor="white",
            linewidth=0.8,
        )
        stats[f"{column}_median_diameter_m"] = median

    fp_quartiles = mass_weighted_quantile(
        diameter, decisions.false_px.to_numpy(dtype=float), [0.25, 0.75]
    )
    ax.axvspan(
        fp_quartiles[0],
        fp_quartiles[1],
        color=FP_COLOR,
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )
    stats["false_px_q1_diameter_m"] = float(fp_quartiles[0])
    stats["false_px_q3_diameter_m"] = float(fp_quartiles[1])

    # Keep prior-scale labels inside the axes so the panel header baseline stays clean.
    blend = blended_transform_factory(ax.transData, ax.transAxes)
    label_positions = {
        "10 m pixel": (0.975, "left"),
        "30 m terrain": (0.895, "left"),
        "~5 km trigger": (0.815, "right"),
        "250 m material": (0.735, "center"),
    }
    for label, scale in PRIOR_SCALES_M.items():
        ax.axvline(scale, color=MUTED, linewidth=0.9, linestyle=(0, (1.5, 1.8)))
        y, alignment = label_positions[label]
        ax.text(
            scale,
            y,
            label,
            transform=blend,
            fontsize=7.2,
            color=MUTED,
            ha=alignment,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5},
        )

    fp_median = stats["false_px_median_diameter_m"]
    tp_median = stats["intersection_px_median_diameter_m"]
    ax.text(
        float(np.sqrt(fp_quartiles[0] * fp_quartiles[1])),
        0.625,
        f"FP range: {fp_quartiles[0]:.0f}\u2013{fp_quartiles[1]:.0f} m",
        fontsize=7.5,
        color=FP_COLOR,
        ha="center",
        va="bottom",
    )
    ax.annotate(
        f"median: {fp_median:.0f} m",
        xy=(fp_median, 0.5),
        xytext=(72.0, 0.42),
        fontsize=7.5,
        color=FP_COLOR,
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "-", "color": FP_COLOR, "lw": 0.8},
    )
    ax.set_xscale("log")
    ax.set_xlim(5.0, 12000.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("equivalent diameter (m)")
    ax.set_ylabel("cumulative mass fraction")
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.31),
        handlelength=1.4,
        borderaxespad=0.0,
    )
    tidy(ax, grid_axis="both")
    stats["tp_median_over_fp_median"] = tp_median / fp_median
    return stats


def per_source_near_pure(decisions: pd.DataFrame) -> pd.DataFrame:
    """Kept for the report / manuscript numbers; not drawn as a panel."""

    rows = []
    for source in ALL_SOURCES:
        block = decisions[decisions.dataset_id == source]
        fp_mass = float(block.false_px.sum())
        near_pure = float(
            block.loc[block.purity <= RULE["candidate_purity_max"], "false_px"].sum()
        )
        rows.append(
            {
                "dataset_id": source,
                "label": SOURCE_LABELS_LONG[source],
                "fp_mass": fp_mass,
                "near_pure_share": near_pure / fp_mass if fp_mass else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    pooled = float(
        decisions.loc[decisions.purity <= RULE["candidate_purity_max"], "false_px"].sum()
        / decisions.false_px.sum()
    )
    frame["pooled_share"] = pooled
    return frame


def pixel_budget(decisions: pd.DataFrame, summary: dict) -> dict[str, float]:
    tp = float(decisions.intersection_px.sum())
    fp = float(decisions.false_px.sum())
    baseline_iou = float(summary["baseline_iou"])
    fn = tp / baseline_iou - tp - fp
    return {"tp": tp, "fp": fp, "fn": fn, "union": tp + fp + fn}


# --- Figure assembly ---------------------------------------------------------


def render_figure(
    decisions: pd.DataFrame,
    examples: pd.DataFrame,
    candidates: dict[str, dict],
    outdir: Path,
    dpi: int,
) -> tuple[Path, dict]:
    # Shared 2 x 3 grid: identical column geometry for (a) and (b)–(d).
    # The inter-row band holds two legend rows plus independent panel headers.
    fig = plt.figure(
        figsize=(CANVAS_WIDTH_MM / 25.4, CANVAS_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        3,
        left=0.078,
        right=0.982,
        top=0.940,
        bottom=0.105,
        wspace=0.30,
        hspace=0.36,
        height_ratios=[1.00, 1.10],
    )

    image_axes = []
    for index, row in examples.iterrows():
        ax = fig.add_subplot(grid[0, index])
        ax.set_anchor("N")
        draw_case(
            ax,
            candidates[str(row.sample_id)],
            SOURCE_LABELS[str(row.dataset_id)],
            str(row.context),
        )
        image_axes.append(ax)
    panel_label(image_axes[0], "a", x=-0.040, y=1.040)
    image_axes[0].text(
        0.090,
        1.040,
        "Cross-domain errors form coherent candidate bodies",
        transform=image_axes[0].transAxes,
        fontsize=10.0,
        fontweight="semibold",
        va="bottom",
        ha="left",
        color=INK,
    )

    ax_b = fig.add_subplot(grid[1, 0])
    plane = draw_mass_plane(ax_b, fig, decisions)

    ax_c = fig.add_subplot(grid[1, 1])
    survival = draw_survival(ax_c, decisions)

    ax_d = fig.add_subplot(grid[1, 2])
    bridge = draw_scale_bridge(ax_d, decisions)

    # Lock bottom-panel frames to the exact x0/width of the images above.
    fig.canvas.draw()
    for image_ax, plot_ax in zip(image_axes, (ax_b, ax_c, ax_d), strict=True):
        image_box = image_ax.get_position()
        plot_box = plot_ax.get_position()
        plot_ax.set_position(
            [image_box.x0, plot_box.y0, image_box.width, plot_box.height]
        )

    # Two semantic legend rows keep categorical fills and boundary encodings apart.
    legend_y = min(ax.get_position().y0 for ax in image_axes)
    fig.legend(
        handles=[
            Patch(facecolor=TP_COLOR, alpha=0.46, label="true positive"),
            Patch(facecolor=FP_COLOR, alpha=0.46, label="false positive"),
            Patch(facecolor=FN_COLOR, alpha=0.58, label="false negative"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, legend_y - 0.024),
        ncol=3,
        frameon=False,
        columnspacing=1.45,
        handlelength=1.35,
        borderaxespad=0.0,
        fontsize=8.1,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=INK,
                lw=0.8,
                linestyle=(0, (2.6, 1.8)),
                label="reference inventory",
            ),
            Line2D(
                [0],
                [0],
                color=FP_COLOR,
                lw=1.6,
                label="near-pure false positive body",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, legend_y - 0.051),
        ncol=2,
        frameon=False,
        columnspacing=1.65,
        handlelength=1.45,
        borderaxespad=0.0,
        fontsize=8.1,
    )

    panel_header_row(
        fig,
        [
            (ax_b, "b", "False-positive area\nand purity"),
            (ax_c, "c", "Retention by size\nthreshold"),
            (ax_d, "d", "Body size relative\nto prior scales"),
        ],
        title_dx=0.050,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "figure4_revision_visual_error_structure.png"
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    tolerance_in = 0.02
    if (
        tight.x0 < -tolerance_in
        or tight.y0 < -tolerance_in
        or tight.x1 > fig.get_figwidth() + tolerance_in
        or tight.y1 > fig.get_figheight() + tolerance_in
    ):
        raise RuntimeError(f"Figure 4 artist extends beyond canvas: {tight}")
    fig.savefig(output, dpi=dpi, facecolor="white", format="png")
    expected_size = (
        round(CANVAS_WIDTH_MM / 25.4 * dpi),
        round(CANVAS_HEIGHT_MM / 25.4 * dpi),
    )
    with Image.open(output) as rendered:
        if rendered.size != expected_size:
            raise RuntimeError(
                f"Unexpected Figure 4 raster size: {rendered.size} != {expected_size}"
            )

    # Standalone panels for PPT assembly (no composite required).
    panel_dir = outdir / "panels" / "figure4"
    source_tags = [
        SOURCE_LABELS[str(row.dataset_id)].lower() for _, row in examples.iterrows()
    ]
    for index, (ax, tag) in enumerate(zip(image_axes, source_tags, strict=True), start=1):
        export_axes(fig, ax, panel_dir / f"a{index}_{tag}.png", dpi=dpi, pad_inches=0.04)
    export_axes(fig, image_axes, panel_dir / "a_row.png", dpi=dpi, pad_inches=0.05)
    export_axes(fig, ax_b, panel_dir / "b_false_mass.png", dpi=dpi, pad_inches=0.08)
    export_axes(fig, ax_c, panel_dir / "c_mass_retained.png", dpi=dpi, pad_inches=0.08)
    export_axes(fig, ax_d, panel_dir / "d_scale_bridge.png", dpi=dpi, pad_inches=0.08)

    plt.close(fig)

    statistics = {
        "mass_plane": plane,
        "survival": survival,
        "scale_bridge": bridge,
    }
    return output, statistics


def write_source_data(
    path: Path,
    decisions: pd.DataFrame,
    statistics: dict,
) -> None:
    rows: list[dict[str, object]] = []
    total_fp = float(decisions.false_px.sum())
    area_edges = np.geomspace(1.0, float(decisions.area_px.max()) * 1.001, 25)
    purity_edges = np.linspace(0.0, 1.0, 11)
    for a0, a1 in zip(area_edges[:-1], area_edges[1:], strict=True):
        for p0, p1 in zip(purity_edges[:-1], purity_edges[1:], strict=True):
            mask = (
                (decisions.area_px >= a0)
                & (decisions.area_px < a1)
                & (decisions.purity >= p0)
                & (decisions.purity < p1)
            )
            mass = float(decisions.loc[mask, "false_px"].sum())
            if mass <= 0:
                continue
            rows.append(
                {
                    "panel": "b",
                    "series": "false_positive_mass_by_area_purity",
                    "x": f"area[{a0:.1f},{a1:.1f})_purity[{p0:.1f},{p1:.1f})",
                    "value": mass / total_fp,
                }
            )

    thresholds = np.unique(
        np.rint(np.geomspace(1, float(decisions.area_px.max()), 180))
    ).astype(float)
    tp_survival, fp_survival = mass_survival(decisions, thresholds)
    for threshold, tp_value, fp_value in zip(
        thresholds, tp_survival, fp_survival, strict=True
    ):
        rows.append(
            {
                "panel": "c",
                "series": "true_positive_mass_survival",
                "x": float(threshold),
                "value": float(tp_value),
                "secondary_value": float(fp_value),
            }
        )

    diameter = equivalent_diameter_m(decisions.area_px.to_numpy(dtype=float))
    order = np.argsort(diameter)
    fp_cdf = np.cumsum(decisions.false_px.to_numpy(dtype=float)[order])
    tp_cdf = np.cumsum(decisions.intersection_px.to_numpy(dtype=float)[order])
    fp_cdf = fp_cdf / fp_cdf[-1]
    tp_cdf = tp_cdf / tp_cdf[-1]
    step = max(len(order) // 400, 1)
    for position in range(0, len(order), step):
        rows.append(
            {
                "panel": "d",
                "series": "mass_weighted_diameter_cdf",
                "x": float(diameter[order][position]),
                "value": float(fp_cdf[position]),
                "secondary_value": float(tp_cdf[position]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    started = time.time()
    configure_style()

    decisions = pd.read_parquet(DECISIONS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text("utf-8"))

    tiles, area_q1, area_q3 = eligible_tiles(decisions)
    fold_index = build_fold_index()
    candidates = load_candidates(tiles, fold_index)
    contexts = assign_context(candidates)
    brightness_threshold = float(contexts.pop("__brightness_threshold__"))  # type: ignore[arg-type]
    examples = choose_examples(tiles, candidates, contexts)

    output, statistics = render_figure(
        decisions, examples, candidates, args.outdir, args.dpi
    )
    budget = pixel_budget(decisions, summary)
    per_source = per_source_near_pure(decisions)

    source_data_path = args.outdir / "figure4_visual_error_structure_source_data.csv"
    write_source_data(source_data_path, decisions, statistics)

    report = {
        "schema_version": "figure4_visual_error_structure.v5",
        "backend": "Python/Matplotlib",
        "output_policy": "PNG only",
        "canvas_mm": [CANVAS_WIDTH_MM, CANVAS_HEIGHT_MM],
        "typography": "sizes are final printed sizes; no journal downscaling assumed",
        "dpi_requested": args.dpi,
        "render_script": str(SCRIPT_PATH),
        "render_script_sha256": sha256(SCRIPT_PATH),
        "source_files": {
            str(DECISIONS_PATH.relative_to(PROJECT_ROOT)): sha256(DECISIONS_PATH),
            str(SUMMARY_PATH.relative_to(PROJECT_ROOT)): sha256(SUMMARY_PATH),
        },
        "selection_rule": RULE
        | {
            "eligible_body_area_iqr_px": [area_q1, area_q3],
            "context_brightness_threshold_reflectance": brightness_threshold,
            "n_eligible_tiles_by_source": tiles.dataset_id.value_counts().to_dict(),
            "n_eligible_tiles_total": int(len(tiles)),
        },
        "selected_examples": examples.drop(columns=["draw_rank"]).to_dict(
            orient="records"
        ),
        "image_integrity": {
            "crop": "none; complete 128 x 128 OOF tiles are shown",
            "rgb_stretch": "per-channel 2nd-98th percentile within each tile",
            "gamma": 0.85,
            "pseudocolor": (
                "TP/FN categorical overlay; near-pure FP emphasised over other FP"
            ),
            "scale_bar_m": 500,
            "example_selection_uses_adapter_output": False,
        },
        "prior_support_scales_m": {
            key.replace("\n", " "): value for key, value in PRIOR_SCALES_M.items()
        },
        "statistics": {
            "n_candidate_bodies": int(len(decisions)),
            "n_samples_with_prediction": int(decisions.sample_id.nunique()),
            "n_events_with_candidate_bodies": int(decisions.canonical_event_id.nunique()),
            "n_events_with_true_positive_pixels": int(
                (decisions.groupby("canonical_event_id").intersection_px.sum() > 0).sum()
            ),
            "true_positive_px": budget["tp"],
            "false_positive_px": budget["fp"],
            "false_negative_px": budget["fn"],
            "near_pure_fp_share": float(
                decisions.loc[decisions.purity <= 0.10, "false_px"].sum()
                / decisions.false_px.sum()
            ),
            "large_and_near_pure_fp_share": statistics["mass_plane"]["corner_share"],
            "tp_share_in_area_ge_200": statistics["survival"]["tp_at_floor"],
            "fp_share_in_area_ge_200": statistics["survival"]["fp_at_floor"],
            "scale_bridge": statistics["scale_bridge"],
            "per_source_near_pure_share": per_source.to_dict(orient="records"),
        },
        "source_data": str(source_data_path),
        "output": str(output),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = args.outdir / "figure4_visual_error_structure_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "report": str(report_path),
                "examples": examples[
                    [
                        "dataset_id",
                        "sample_id",
                        "context",
                        "readability_score",
                        "separation",
                        "largest_near_pure_area",
                    ]
                ].to_dict(orient="records"),
                "scale_bridge": statistics["scale_bridge"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
