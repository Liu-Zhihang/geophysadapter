#!/usr/bin/env python3
"""Render Figure 7 as a case-only object-scale correction plate.

The main plate shows five objectively selected high-effect examples: the
largest sample-level delta IoU from each source, plus the largest remaining
example from a distinct event. This is explicitly an illustrative upper-tail
gallery and is not used to estimate prevalence or effect size. A separate
supplementary plate preserves the five pre-registered behavioural strata -
clean clearance, collateral loss, dense-inventory clearance, net harm, and
abstention - with median-within-stratum selection.

Post-review state is reconstructed exactly: `component_id` in the frozen
decision table is the SciPy label of the baseline prediction, verified here by
re-deriving area, true-positive and false-positive counts for every body.

The analytic criterion and gain-concentration statistics remain in the frozen
report and source tables, but are not repeated in the main figure because the
main-text equations and tables already carry those quantitative claims.

Only a 600 dpi PNG is written, following the revision package image policy.
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
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from figure_chrome import export_axes, export_figure_box
from matplotlib.patches import Patch
from PIL import Image
from scipy import ndimage


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXP = PROJECT_ROOT / "experiments/revision2026"
DEFAULT_OUTDIR = (
    PROJECT_ROOT.parent
    / "submission_package_jprs_revision1/geophysadapter/revision1/figures_revision"
)

DECISIONS_PATH = EXP / "pild_object_veto_final_v1/component_decisions.parquet"
SUMMARY_PATH = EXP / "pild_object_veto_final_v1/summary.json"
CONCENTRATION_PATH = (
    EXP / "pild_object_gain_concentration_v1/event_structure_vs_gain.csv"
)
OOF_CACHE = EXP / "pild_object_physical_diagnostic_v1/oof_cache"
FOLD_IDS = [f"source_stratified_{index}" for index in range(4)]

ANALYSIS_GSD_M = 10.0
BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = range(6)

INK = "#1E242B"
MUTED = "#6B7580"
GRID = "#DFE4E9"
TP_COLOR = "#3E7F76"
FP_COLOR = "#C0503C"
FN_COLOR = "#D9B570"
REMOVED_COLOR = "#4E6E8E"
GRAY = "#AAB3BC"
GRAY_LIGHT = "#EDF0F3"
TP_LIGHT = "#DCEAE7"
FP_LIGHT = "#F5DED8"

SOURCE_LABELS = {
    "DLR_Landslide_Ref_2025": "DLR",
    "GDCLD": "GDCLD",
    "SEN12LS_HARMONIZED": "Sen12",
    "GLaD4CD_v1": "GLaD4CD",
}
# Reused from Figure 2 so a source keeps one colour across the whole paper.
SOURCE_COLORS = {
    "GLaD4CD_v1": "#C43C51",
    "DLR_Landslide_Ref_2025": "#2F5FAA",
    "GDCLD": "#8E44D6",
    "SEN12LS_HARMONIZED": "#E1126E",
}

# --- Case-selection contracts -----------------------------------------------
# The main figure uses an explicit upper-tail contract for illustration. The
# supplementary behaviour plate uses frozen strata and median selection.
READABILITY_TP_KEPT_MIN = 300

STRATA = [
    {
        "key": "clean_clearance_dlr",
        "label": "clean clearance",
        "source": "DLR_Landslide_Ref_2025",
        "query": "fp_cleared >= 1000 and tp_lost <= 100",
        "governing": "net",
    },
    {
        "key": "mixed_sen12",
        "label": "clearance with collateral loss",
        "source": "SEN12LS_HARMONIZED",
        "query": "fp_cleared >= 500 and tp_lost >= 200",
        "governing": "net",
    },
    {
        "key": "action_gdcld",
        "label": "clearance, dense inventory",
        "source": "GDCLD",
        "query": "removed_bodies >= 1 and fp_cleared >= 300",
        "governing": "net",
    },
    {
        "key": "net_harm",
        "label": "net harm",
        "source": None,
        "query": "net < 0 and tp_lost >= 200",
        "governing": "net",
    },
    {
        "key": "abstention_dlr",
        "label": "abstention, identity fallback",
        "source": "DLR_Landslide_Ref_2025",
        "query": "removed_bodies == 0 and fp_kept >= 300",
        "governing": "fp_kept",
    },
]

# Keep the figure-side labels compact; the full stratum definitions remain in
# the source-data CSV and render report.
DISPLAY_LABELS = {
    "clean_clearance_dlr": "clean clearance",
    "mixed_sen12": "collateral loss",
    "action_gdcld": "dense inventory",
    "net_harm": "net harm",
    "abstention_dlr": "identity fallback",
}

HIGH_EFFECT_MIN_TARGET_PX = 100


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK,
            "text.color": INK,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_outcomes(decisions: pd.DataFrame) -> pd.DataFrame:
    """Per-sample ledger of what the analytic criterion removed and kept."""

    removed = decisions.removed.astype(bool)
    frame = decisions.assign(_removed=removed)
    grouped = frame.groupby(
        ["dataset_id", "canonical_event_id", "sample_id"], as_index=False
    ).apply(
        lambda block: pd.Series(
            {
                "removed_bodies": int(block._removed.sum()),
                "kept_bodies": int((~block._removed).sum()),
                "fp_cleared": float(block.loc[block._removed, "false_px"].sum()),
                "tp_lost": float(block.loc[block._removed, "intersection_px"].sum()),
                "fp_kept": float(block.loc[~block._removed, "false_px"].sum()),
                "tp_kept": float(block.loc[~block._removed, "intersection_px"].sum()),
            }
        ),
        include_groups=False,
    )
    grouped["net"] = grouped.fp_cleared - grouped.tp_lost
    return grouped


def choose_behavior_gallery(outcomes: pd.DataFrame) -> pd.DataFrame:
    eligible = outcomes[outcomes.tp_kept >= READABILITY_TP_KEPT_MIN]
    picks = []
    for stratum in STRATA:
        block = eligible.query(stratum["query"])
        if stratum["source"] is not None:
            block = block[block.dataset_id == stratum["source"]]
        if block.empty:
            raise RuntimeError(f"Stratum {stratum['key']} is empty")
        governing = stratum["governing"]
        median = float(block[governing].median())
        ordered = block.assign(_distance=(block[governing] - median).abs()).sort_values(
            ["_distance", "sample_id"]
        )
        record = ordered.iloc[0].to_dict()
        record.update(
            {
                "stratum": stratum["key"],
                "stratum_label": stratum["label"],
                "stratum_size": int(len(block)),
                "stratum_median": median,
                "governing": governing,
            }
        )
        picks.append(record)
    return pd.DataFrame(picks)


def load_target_areas() -> pd.DataFrame:
    records: list[tuple[str, int]] = []
    for fold_id in FOLD_IDS:
        with np.load(OOF_CACHE / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
            target = handle["target"].astype(bool)
            valid = handle["valid"].astype(bool)
            areas = (target & valid).sum(axis=(1, 2))
            records.extend(
                (str(sample_id), int(area))
                for sample_id, area in zip(handle["sample_id"], areas)
            )
    return pd.DataFrame(records, columns=["sample_id", "target_area"])


def attach_exact_sample_effects(outcomes: pd.DataFrame) -> pd.DataFrame:
    frame = outcomes.merge(load_target_areas(), on="sample_id", how="left", validate="one_to_one")
    if frame.target_area.isna().any():
        missing = frame.loc[frame.target_area.isna(), "sample_id"].head().tolist()
        raise RuntimeError(f"Missing target area for samples: {missing}")
    baseline_tp = frame.tp_kept + frame.tp_lost
    baseline_fp = frame.fp_kept + frame.fp_cleared
    reviewed_tp = baseline_tp - frame.tp_lost
    reviewed_fp = baseline_fp - frame.fp_cleared
    frame["iou_base"] = baseline_tp / (frame.target_area + baseline_fp).clip(lower=1)
    frame["iou_review"] = reviewed_tp / (frame.target_area + reviewed_fp).clip(lower=1)
    frame["delta_iou"] = frame.iou_review - frame.iou_base
    return frame


def choose_high_effect_gallery(outcomes: pd.DataFrame) -> pd.DataFrame:
    eligible = outcomes.query(
        "target_area >= @HIGH_EFFECT_MIN_TARGET_PX and removed_bodies > 0 and delta_iou > 0"
    ).copy()
    if eligible.empty:
        raise RuntimeError("No high-effect examples satisfy the frozen eligibility rule")

    picks: list[dict] = []
    for source in SOURCE_LABELS:
        block = eligible[eligible.dataset_id == source].sort_values(
            ["delta_iou", "net", "sample_id"], ascending=[False, False, True]
        )
        if block.empty:
            raise RuntimeError(f"No eligible high-effect example for {source}")
        record = block.iloc[0].to_dict()
        record.update(
            {
                "stratum": "source_maximum",
                "stratum_label": "source-wise maximum sample-level delta IoU",
                "selection_group": f"source:{source}",
            }
        )
        picks.append(record)

    selected_samples = {str(record["sample_id"]) for record in picks}
    selected_events = {str(record["canonical_event_id"]) for record in picks}
    remainder = eligible[
        ~eligible.sample_id.astype(str).isin(selected_samples)
        & ~eligible.canonical_event_id.astype(str).isin(selected_events)
    ].sort_values(["delta_iou", "net", "sample_id"], ascending=[False, False, True])
    if remainder.empty:
        raise RuntimeError("No distinct-event example remains for the fifth high-effect row")
    record = remainder.iloc[0].to_dict()
    record.update(
        {
            "stratum": "next_distinct_event",
            "stratum_label": "largest remaining sample-level delta IoU from a distinct event",
            "selection_group": "next distinct event",
        }
    )
    picks.append(record)

    gallery = pd.DataFrame(picks).sort_values(
        ["delta_iou", "net", "sample_id"], ascending=[False, False, True]
    ).reset_index(drop=True)
    if gallery.canonical_event_id.nunique() != len(gallery):
        raise RuntimeError("High-effect gallery must contain distinct events")
    return gallery


def build_fold_index() -> dict[str, tuple[str, int]]:
    index: dict[str, tuple[str, int]] = {}
    for fold_id in FOLD_IDS:
        with np.load(OOF_CACHE / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
            for position, sample_id in enumerate(handle["sample_id"]):
                index[str(sample_id)] = (fold_id, position)
    return index


def stretch_rgb(cube: np.ndarray) -> np.ndarray:
    rgb = np.stack([cube[RED], cube[GREEN], cube[BLUE]], axis=-1).astype(np.float32)
    for channel in range(3):
        band = rgb[..., channel]
        finite = band[np.isfinite(band)]
        low, high = np.percentile(finite, [2, 98])
        rgb[..., channel] = np.clip((band - low) / max(high - low, 1e-6), 0.0, 1.0)
    return np.power(rgb, 0.85)


def load_tiles(
    gallery: pd.DataFrame, decisions: pd.DataFrame, fold_index: dict[str, tuple[str, int]]
) -> dict[str, dict]:
    """Load rasters and rebuild the post-review prediction for each sample."""

    wanted: dict[str, list[tuple[str, int]]] = {}
    for sample_id in gallery.sample_id:
        fold_id, position = fold_index[str(sample_id)]
        wanted.setdefault(fold_id, []).append((str(sample_id), position))

    tiles: dict[str, dict] = {}
    for fold_id, entries in wanted.items():
        receipt = json.loads(
            (OOF_CACHE / f"{fold_id}_oof_cache_receipt.json").read_text("utf-8")
        )
        threshold = float(receipt["threshold"])
        rows = [position for _, position in entries]
        with np.load(OOF_CACHE / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
            probability = handle["visual_probability"][rows].astype(np.float32)
            target = handle["target"][rows].astype(bool)
            valid = handle["valid"][rows].astype(bool)
        with np.load(
            OOF_CACHE / f"{fold_id}_optical_cache.npz", allow_pickle=False
        ) as handle:
            post = handle["optical_post"][rows].astype(np.float32)

        for slot, (sample_id, _) in enumerate(entries):
            tile_target = target[slot] & valid[slot]
            baseline = (probability[slot] >= threshold) & valid[slot]
            labels, count = ndimage.label(
                baseline, structure=ndimage.generate_binary_structure(2, 2)
            )
            table = decisions[decisions.sample_id == sample_id]
            # Verify the frozen component_id really is the SciPy label before
            # trusting the removal flags to rebuild the post-review mask.
            for record in table.itertuples(index=False):
                mask = labels == record.component_id
                area = int(mask.sum())
                intersection = int((mask & tile_target).sum())
                if area != int(record.area_px) or intersection != int(record.intersection_px):
                    raise RuntimeError(
                        f"component_id {record.component_id} of {sample_id} does not "
                        "match the SciPy labelling; post-review mask cannot be rebuilt"
                    )
            removed_ids = table.loc[table.removed.astype(bool), "component_id"].to_numpy()
            removed_mask = np.isin(labels, removed_ids)
            tiles[sample_id] = {
                "rgb": stretch_rgb(post[slot]),
                "target": tile_target,
                "baseline": baseline,
                "reviewed": baseline & ~removed_mask,
                "removed": removed_mask,
                "threshold": threshold,
                "fold_id": fold_id,
            }
        del probability, target, valid, post
    return tiles


def add_scale_bar(ax, shape: tuple[int, int]) -> None:
    height, width = shape
    bar = 500.0 / ANALYSIS_GSD_M
    x0, y0 = 0.06 * width, 0.94 * height
    ax.plot([x0, x0 + bar], [y0, y0], color="white", linewidth=2.6,
            solid_capstyle="butt")
    ax.plot([x0, x0 + bar], [y0, y0], color=INK, linewidth=1.1,
            solid_capstyle="butt")
    ax.text(
        x0 + bar / 2.0,
        y0 - 0.035 * height,
        "500 m",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="white",
        fontweight="bold",
        path_effects=[patheffects.withStroke(linewidth=1.5, foreground=INK)],
    )


def finish_image_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.7)


def draw_post_event(ax, tile: dict) -> None:
    ax.imshow(np.asarray(tile["rgb"]))
    add_scale_bar(ax, np.asarray(tile["target"]).shape)
    finish_image_axis(ax)


def draw_reference(ax, tile: dict) -> None:
    target = np.asarray(tile["target"], dtype=bool)
    overlay = np.zeros((*target.shape, 4), dtype=float)
    overlay[target] = mpl.colors.to_rgba(FP_COLOR, 0.62)
    ax.imshow(np.asarray(tile["rgb"]))
    ax.imshow(overlay)
    if target.any():
        ax.contour(target.astype(float), levels=[0.5], colors="white", linewidths=1.2)
        dashed = ax.contour(target.astype(float), levels=[0.5], colors=INK, linewidths=0.55)
        dashed.set_dashes([(0.0, (2.4, 1.6))])
    finish_image_axis(ax)


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def add_metric_chip(ax, value: float) -> None:
    ax.text(
        0.965,
        0.035,
        f"IoU {value:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="white",
        zorder=9,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": INK,
            "edgecolor": "none",
            "alpha": 0.78,
        },
    )


def draw_state(
    ax,
    tile: dict,
    prediction: np.ndarray,
    *,
    show_removed: bool,
    metric_value: float,
) -> None:
    target = np.asarray(tile["target"], dtype=bool)
    overlay = np.zeros((*target.shape, 4), dtype=float)
    overlay[prediction & target] = mpl.colors.to_rgba(TP_COLOR, 0.60)
    overlay[prediction & ~target] = mpl.colors.to_rgba(FP_COLOR, 0.62)
    overlay[~prediction & target] = mpl.colors.to_rgba(FN_COLOR, 0.62)

    ax.imshow(np.asarray(tile["rgb"]))
    ax.imshow(overlay)
    if target.any():
        ax.contour(target.astype(float), levels=[0.5], colors="white", linewidths=1.2)
        dashed = ax.contour(target.astype(float), levels=[0.5], colors=INK, linewidths=0.55)
        dashed.set_dashes([(0.0, (2.4, 1.6))])
    removed = np.asarray(tile["removed"], dtype=bool)
    if show_removed and removed.any():
        ax.contour(removed.astype(float), levels=[0.5], colors="white", linewidths=2.0)
        ax.contour(removed.astype(float), levels=[0.5], colors=REMOVED_COLOR, linewidths=1.1)

    add_metric_chip(ax, metric_value)
    finish_image_axis(ax)


def draw_analytic_cut(ax, decisions: pd.DataFrame, baseline_iou: float) -> dict[str, float]:
    """Removal is decided by an identity, not by a tuned threshold."""

    cut = baseline_iou / (1.0 + baseline_iou)
    purity = decisions.purity.to_numpy(dtype=float)
    weights = decisions.area_px.to_numpy(dtype=float)
    total = weights.sum()
    bins = np.linspace(0.0, 1.0, 41)
    below = purity < cut
    ax.hist(
        purity[below],
        bins=bins,
        weights=weights[below] / total,
        color=FP_LIGHT,
        edgecolor=FP_COLOR,
        linewidth=0.55,
    )
    ax.hist(
        purity[~below],
        bins=bins,
        weights=weights[~below] / total,
        color=TP_LIGHT,
        edgecolor=TP_COLOR,
        linewidth=0.55,
    )
    ax.axvline(cut, color=INK, linewidth=1.2)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.30)
    # Anchored to the axes corners, not to the cut: beside the cut the left label
    # ran over the y-axis label.
    ax.text(
        0.02,
        0.955,
        "\u25c0  removed",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.0,
        color=FP_COLOR,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.955,
        "kept  \u25b6",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.0,
        color=TP_COLOR,
        fontweight="bold",
    )
    ax.annotate(
        f"analytic cut  $p<\\mathrm{{IoU}}/(1+\\mathrm{{IoU}})={cut:.3f}$\n"
        "no tunable parameter",
        xy=(cut, top * 0.42),
        xytext=(cut + 0.16, top * 0.80),
        fontsize=7.0,
        color=INK,
        va="center",
        linespacing=1.4,
        arrowprops={
            "arrowstyle": "->",
            "color": INK,
            "lw": 0.8,
            "connectionstyle": "arc3,rad=-0.18",
        },
    )
    removed_share = float(weights[below].sum() / total)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("candidate-body purity")
    ax.set_ylabel("share of predicted area")
    tidy_axis(ax, grid_axis="y")
    return {"analytic_cut": cut, "removed_area_share": removed_share}


def draw_gain_concentration(ax, concentration: pd.DataFrame) -> dict[str, float]:
    """The gain follows the anchor's own error structure, so it is uneven."""

    from scipy import stats as scipy_stats

    x = concentration.addressable_share.to_numpy(dtype=float)
    y = concentration.rer.to_numpy(dtype=float)
    rho, p_value = scipy_stats.spearmanr(x, y)

    # Regime shading: the readable statement is that events with little
    # addressable false-positive mass have little to gain, whatever the physics.
    x_split, y_split = 0.5, 0.0
    ax.axhspan(y_split, 1.05, xmin=0.0, xmax=0.5, color=GRAY_LIGHT, alpha=0.55, zorder=0)
    ax.axvspan(x_split, 1.02, color=TP_LIGHT, alpha=0.45, zorder=0)

    for source, block in concentration.groupby("dataset_id"):
        ax.scatter(
            block.addressable_share,
            block.rer,
            s=24,
            color=SOURCE_COLORS.get(source, MUTED),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            label=SOURCE_LABELS.get(source, source),
            zorder=4,
        )
    fit = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 50)
    ax.plot(grid, np.polyval(fit, grid), color=INK, linewidth=0.9, linestyle=(0, (4, 2)), zorder=3)
    ax.axhline(0.0, color=MUTED, linewidth=0.7, zorder=2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(min(-0.06, float(y.min()) * 1.15), 1.30)
    # Legend parked in the headroom above the data; the regimes are carried by
    # the shading, so no extra text is needed inside the point cloud.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.9,
        borderaxespad=0.1,
        fontsize=7.0,
    )
    ax.text(
        0.025,
        0.845,
        f"Spearman $\\rho$ = {rho:+.3f}\n$p$ = {p_value:.1e},  {len(concentration)} events",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=INK,
        linespacing=1.4,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": GRID,
            "linewidth": 0.5,
            "alpha": 0.9,
        },
    )
    ax.text(
        0.985,
        0.025,
        "much addressable mass",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=TP_COLOR,
        fontweight="bold",
    )
    ax.set_xlabel("addressable false-positive share")
    ax.set_ylabel("per-event error reduction")
    tidy_axis(ax, grid_axis="both")
    return {"spearman_rho": float(rho), "spearman_p": float(p_value)}


def tidy_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.95)
    ax.set_axisbelow(True)


def assert_canvas_clear(output: Path) -> dict[str, int]:
    image = np.asarray(Image.open(output).convert("L"))
    ink = image < 160
    ys = np.where(ink.any(axis=1))[0]
    xs = np.where(ink.any(axis=0))[0]
    bounds = {
        "left": int(xs.min()),
        "right": int(image.shape[1] - 1 - xs.max()),
        "top": int(ys.min()),
        "bottom": int(image.shape[0] - 1 - ys.max()),
    }
    if min(bounds.values()) < 3:
        bands = []
        for y0 in range(0, image.shape[0], 60):
            columns = np.where(ink[y0 : y0 + 60].any(axis=0))[0]
            if len(columns) and (columns.max() >= image.shape[1] - 3 or columns.min() <= 2):
                bands.append(y0)
        raise RuntimeError(f"Figure ink touches canvas edge: {bounds}; bands at y={bands}")
    return bounds


def render(
    gallery: pd.DataFrame,
    tiles: dict[str, dict],
    outdir: Path,
    dpi: int,
    *,
    output_name: str,
    panel_stem: str,
    display_mode: str,
) -> tuple[Path, dict[str, int]]:
    fig = plt.figure(figsize=(7.48, 9.20), facecolor="white")
    grid = fig.add_gridspec(
        len(gallery),
        4,
        left=0.178,
        right=0.990,
        top=0.925,
        bottom=0.072,
        wspace=0.025,
        hspace=0.070,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
    )
    column_titles = ["Post-event", "Reference", "Frozen visual", "GeoPhysAdapter"]
    gallery_rows: list[list[object]] = []

    for index, row in gallery.iterrows():
        tile = tiles[str(row.sample_id)]
        target = np.asarray(tile["target"], dtype=bool)
        baseline = np.asarray(tile["baseline"], dtype=bool)
        reviewed = np.asarray(tile["reviewed"], dtype=bool)
        axes = [fig.add_subplot(grid[int(index), column]) for column in range(4)]
        for axis in axes:
            axis.set_anchor("W")
            axis.set_box_aspect(1)

        draw_post_event(axes[0], tile)
        draw_reference(axes[1], tile)
        draw_state(
            axes[2], tile, baseline, show_removed=False,
            metric_value=iou_score(baseline, target),
        )
        draw_state(
            axes[3], tile, reviewed, show_removed=True,
            metric_value=iou_score(reviewed, target),
        )
        gallery_rows.append(axes)

        if int(index) == 0:
            for axis, title in zip(axes, column_titles):
                axis.set_title(title, fontsize=8.8, fontweight="bold", color=INK, pad=4.0)

        position = axes[0].get_position()
        center_y = (position.y0 + position.y1) / 2.0
        source = SOURCE_LABELS[str(row.dataset_id)]
        source_label = source
        fig.text(
            0.022, center_y + 0.028, source_label,
            ha="left", va="center", fontsize=8.5, fontweight="bold", color=INK,
        )
        if display_mode == "effect":
            descriptor = f"ΔIoU {float(row.delta_iou):+.3f}"
        else:
            descriptor = DISPLAY_LABELS[str(row.stratum)]
        fig.text(
            0.022, center_y + 0.002, descriptor,
            ha="left", va="center", fontsize=7.4,
            fontweight="bold",
            color=MUTED,
        )
        if int(row.removed_bodies) == 0:
            ledger = "unchanged"
        else:
            ledger = f"net {int(row.net):+,} px"
        fig.text(
            0.022, center_y - 0.024, ledger,
            ha="left", va="center", fontsize=7.4,
            fontweight="bold", color=MUTED,
        )

        if int(index) < len(gallery) - 1:
            separator_y = position.y0 - 0.011
            fig.add_artist(
                Line2D(
                    [0.020, 0.990], [separator_y, separator_y],
                    transform=fig.transFigure, color=GRID, lw=0.55,
                )
            )

    fig.legend(
        handles=[
            Patch(facecolor=TP_COLOR, alpha=0.60, label="true positive"),
            Patch(facecolor=FP_COLOR, alpha=0.62, label="false positive"),
            Patch(facecolor=FN_COLOR, alpha=0.62, label="false negative"),
            Line2D([0], [0], color=INK, lw=0.6, linestyle=(0, (2.4, 1.6)), label="reference inventory"),
            Line2D([0], [0], color=REMOVED_COLOR, lw=1.1, label="removed candidate body"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.018),
        ncol=5,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.35,
        fontsize=7.5,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / output_name
    fig.savefig(output, dpi=dpi, facecolor="white")

    panel_dir = outdir / "panels" / "figure7"
    for index, axes in enumerate(gallery_rows):
        letter = "abcde"[index]
        export_axes(
            fig,
            axes,
            panel_dir / f"{panel_stem}_{letter}_case_row.png",
            dpi=dpi,
            pad_inches=0.08,
        )
    export_figure_box(
        fig,
        panel_dir / f"{panel_stem}_gallery_a_to_e.png",
        (0.012, 0.045, 0.995, 0.955),
        dpi=dpi,
    )

    plt.close(fig)
    return output, assert_canvas_clear(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    started = time.time()
    configure_style()

    decisions = pd.read_parquet(DECISIONS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text("utf-8"))
    outcomes = sample_outcomes(decisions)

    population = {
        "n_samples_with_prediction": int(len(outcomes)),
        "net_positive": int((outcomes.net > 0).sum()),
        "net_zero": int((outcomes.net == 0).sum()),
        "net_negative": int((outcomes.net < 0).sum()),
        "samples_with_removal": int((outcomes.removed_bodies > 0).sum()),
    }
    if population["n_samples_with_prediction"] != 6927:
        raise RuntimeError(f"Sample inventory changed: {population}")
    if (population["net_positive"], population["net_zero"], population["net_negative"]) != (
        5264,
        1198,
        465,
    ):
        raise RuntimeError(f"Population outcome split changed: {population}")

    event_net = decisions.assign(_removed=decisions.removed.astype(bool)).groupby(
        "canonical_event_id"
    ).apply(
        lambda block: float(
            block.loc[block._removed, "false_px"].sum()
            - block.loc[block._removed, "intersection_px"].sum()
        ),
        include_groups=False,
    )
    population["events_net_positive"] = int((event_net > 0).sum())
    population["n_events"] = int(len(event_net))

    concentration = pd.read_csv(CONCENTRATION_PATH)
    if len(concentration) != 54:
        raise RuntimeError(f"Expected 54 events in the concentration table, found {len(concentration)}")

    from scipy import stats as scipy_stats

    baseline_iou = float(summary["baseline_iou"])
    analytic_cut = baseline_iou / (1.0 + baseline_iou)
    weights = decisions.area_px.to_numpy(dtype=float)
    removed_area_share = float(weights[decisions.purity.to_numpy(dtype=float) < analytic_cut].sum() / weights.sum())
    spearman_rho, spearman_p = scipy_stats.spearmanr(
        concentration.addressable_share.to_numpy(dtype=float),
        concentration.rer.to_numpy(dtype=float),
    )

    outcomes = attach_exact_sample_effects(outcomes)
    gallery = choose_high_effect_gallery(outcomes)
    behavior_gallery = choose_behavior_gallery(outcomes)
    fold_index = build_fold_index()
    combined_gallery = pd.concat([gallery, behavior_gallery], ignore_index=True).drop_duplicates("sample_id")
    tiles = load_tiles(combined_gallery, decisions, fold_index)
    output, edges = render(
        gallery,
        tiles,
        args.outdir,
        args.dpi,
        output_name="figure7_revision_object_scale_correction.png",
        panel_stem="main_high_effect",
        display_mode="effect",
    )
    supplementary_output, supplementary_edges = render(
        behavior_gallery,
        tiles,
        args.outdir,
        args.dpi,
        output_name="figureS1_object_scale_behavior_spectrum.png",
        panel_stem="supplementary_behavior",
        display_mode="behavior",
    )
    if abs(analytic_cut - 0.179111) > 1e-5:
        raise RuntimeError(f"Analytic cut changed: {analytic_cut}")
    if abs(spearman_rho - 0.7701) > 5e-4:
        raise RuntimeError(f"Concentration correlation changed: {spearman_rho}")

    realised = {
        "net_positive_cells": int((gallery.net > 0).sum()),
        "net_zero_cells": int((gallery.net == 0).sum()),
        "net_negative_cells": int((gallery.net < 0).sum()),
    }
    source_data = args.outdir / "figure7_object_scale_correction_source_data.csv"
    gallery.assign(
        source=gallery.dataset_id.map(SOURCE_LABELS),
    )[
        [
            "stratum",
            "stratum_label",
            "selection_group",
            "source",
            "canonical_event_id",
            "sample_id",
            "target_area",
            "iou_base",
            "iou_review",
            "delta_iou",
            "removed_bodies",
            "kept_bodies",
            "fp_cleared",
            "tp_lost",
            "fp_kept",
            "tp_kept",
            "net",
        ]
    ].to_csv(source_data, index=False)
    supplementary_source_data = args.outdir / "figureS1_object_scale_behavior_spectrum_source_data.csv"
    behavior_gallery.assign(source=behavior_gallery.dataset_id.map(SOURCE_LABELS)).to_csv(
        supplementary_source_data, index=False
    )

    report = {
        "schema_version": "figure7_object_scale_correction.v3",
        "backend": "Python/Matplotlib",
        "output_policy": "PNG only",
        "canvas_mm": [190.0, 233.7],
        "dpi_requested": args.dpi,
        "render_script": str(SCRIPT_PATH),
        "render_script_sha256": sha256(SCRIPT_PATH),
        "supersedes": "the mixed gallery-plus-diagnostic Figure 7; the main figure now "
        "contains only map-level cases while quantitative diagnostics remain in tables",
        "source_files": {
            str(DECISIONS_PATH.relative_to(PROJECT_ROOT)): sha256(DECISIONS_PATH),
            str(SUMMARY_PATH.relative_to(PROJECT_ROOT)): sha256(SUMMARY_PATH),
            str(CONCENTRATION_PATH.relative_to(PROJECT_ROOT)): sha256(CONCENTRATION_PATH),
        },
        "main_figure_selection_rule": {
            "purpose": "illustrative upper-tail gallery; not used for prevalence or effect-size inference",
            "eligibility": f"target_area >= {HIGH_EFFECT_MIN_TARGET_PX}, at least one removed body, delta_iou > 0",
            "source_coverage": "largest exact sample-level delta IoU from each of four sources",
            "fifth_row": "largest remaining exact sample-level delta IoU from a distinct event",
            "tie_break": "net corrected pixels, then sample_id",
            "selected_by_visual_appearance": False,
            "selected_by_outcome_magnitude": True,
        },
        "supplementary_behavior_selection_rule": {
            "readability_precondition": f"tp_kept >= {READABILITY_TP_KEPT_MIN}",
            "within_stratum_choice": "sample nearest the stratum median of the governing quantity",
            "tie_break": "sample_id",
            "strata": STRATA,
            "selected_by_outcome_magnitude": False,
            "note": "no stratum is defined by largest gain or best appearance",
        },
        "population_outcome_split": population,
        "gallery_allocation": realised,
        "image_integrity": {
            "crop": "none; complete 128 x 128 OOF tiles are shown",
            "rgb_stretch": "per-channel 2nd-98th percentile within each tile",
            "gamma": 0.85,
            "post_review_mask": "rebuilt from frozen removal flags; component_id verified "
            "against SciPy labelling by re-deriving area and true-positive counts",
            "scale_bar_m": 500,
        },
        "object_scale_corpus": {
            "baseline_iou": float(summary["baseline_iou"]),
            "delta_iou": float(summary["verdict"]["delta_iou"]),
            "rer": float(summary["verdict"]["rer"]),
        },
        "analytic_criterion": {
            "cut": analytic_cut,
            "identity": "purity < IoU / (1 + IoU)",
            "tunable_parameters": 0,
            "removed_share_of_predicted_area": removed_area_share,
        },
        "gain_concentration": {
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "n_events": int(len(concentration)),
            "display": "not repeated in the main figure; retained in the report and tables",
            "caveat": "addressable share is computed from labels and is explanatory, "
            "not a deployable predictor",
        },
        "selected_samples": gallery.to_dict(orient="records"),
        "supplementary_behavior_samples": behavior_gallery.to_dict(orient="records"),
        "edge_self_check": edges,
        "supplementary_edge_self_check": supplementary_edges,
        "source_data": str(source_data),
        "supplementary_source_data": str(supplementary_source_data),
        "output": str(output),
        "supplementary_output": str(supplementary_output),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = args.outdir / "figure7_object_scale_correction_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "report": str(report_path),
                "population": population,
                "allocation": realised,
                "selected": gallery[
                    ["stratum", "sample_id", "delta_iou", "fp_cleared", "tp_lost", "net"]
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
