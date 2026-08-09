#!/usr/bin/env python3
"""Render Figure 8: correction mechanism space and event gain landscape.

Panel (a) uses an analytical field: net error reduction equals the fraction of
baseline errors cleared minus the fraction newly introduced. Five frozen visual
anchors and the development/sealed mismatch trajectories are placed in that
space. Panel (b) uses only occupied empirical bins from 54 event diagnostics;
no smoothing or interpolation is applied. Only PNG files are written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr

from figure_chrome import (
    CARD_EDGE,
    DEGRADE,
    GRAY,
    IMPROVE,
    INK,
    MUTED,
    NEUTRAL,
    SLATE,
    assert_canvas_clear,
    configure_style,
    export_figure_box,
    panel_title,
    sha256,
    tidy,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXP = PROJECT_ROOT / "experiments/revision2026"
DEFAULT_OUTDIR = (
    PROJECT_ROOT.parent
    / "docs/assets"
)
DEFAULT_PUBLISH_PATH = (
    PROJECT_ROOT.parent
    / "docs/assets/"
    "figure8_revision_attribution_and_reproduction.png"
)

MISMATCH_PATH = EXP / "pild_object_veto_mismatch_v1/conditions.csv"
SEALED_PATH = EXP / "pild_object_sealed_confirmation_v1/sealed_conditions.csv"
EVENT_PATH = EXP / "pild_object_gain_concentration_v1/event_structure_vs_gain.csv"

CONDITION_ORDER = ["aligned", "shift32", "roll64", "donor"]
ANCHOR_ROWS = [
    ("Prithvi-EO-2.0", "pild_object_veto_final_v1"),
    ("DINOv3-SAT-L", "pild_alt_anchor_dinov3_sat_l_fpn_v1/object_veto_final"),
    ("Hiera-S-MAE", "pild_alt_anchor_hiera_small_mae_fpn_v1/object_veto_final"),
    ("ConvNeXtV2-T", "pild_alt_anchor_fcmae_convnextv2_tiny_fpn_v1/object_veto_final"),
    ("DINOv2-S", "pild_alt_anchor_dinov2_s_fpn_v1/object_veto_final"),
]
ANCHOR_COLORS = {
    "Prithvi-EO-2.0": "#4E6E8E",
    "DINOv3-SAT-L": "#796AAE",
    "Hiera-S-MAE": "#4C9A8A",
    "ConvNeXtV2-T": "#C58E32",
    "DINOv2-S": "#2F5FAA",
}
SOURCE_MARKERS = {
    "DLR_Landslide_Ref_2025": "o",
    "GLaD4CD_v1": "s",
    "GDCLD": "D",
    "SEN12LS_HARMONIZED": "^",
}
SOURCE_LABELS = {
    "DLR_Landslide_Ref_2025": "DLR",
    "GLaD4CD_v1": "GLaD4CD",
    "GDCLD": "GDCLD",
    "SEN12LS_HARMONIZED": "Sen12",
}


def load_conditions() -> tuple[pd.DataFrame, pd.DataFrame]:
    mismatch = pd.read_csv(MISMATCH_PATH).set_index("condition").loc[CONDITION_ORDER]
    sealed = pd.read_csv(SEALED_PATH).set_index("condition").loc[CONDITION_ORDER]
    return mismatch, sealed


def flow_from_rer_ratio(rer: float, ratio: float) -> tuple[float, float]:
    """Recover corrected and harmed fractions from RER and their ratio."""

    if rer <= 0 or ratio <= 1:
        raise RuntimeError(f"Invalid flow pair: rer={rer}, ratio={ratio}")
    harmed = rer / (ratio - 1.0)
    corrected = ratio * harmed
    return corrected, harmed


def load_anchors() -> pd.DataFrame:
    rows: list[dict] = []
    for label, relative in ANCHOR_ROWS:
        path = EXP / relative / "summary.json"
        payload = json.loads(path.read_text("utf-8"))
        result = payload["results"]["source_conditioned"]
        rer = float(result["rer"])
        corrected = float(result["cleared_fp"])
        harmed = float(result["lost_tp"])
        baseline_error = (corrected - harmed) / rer
        corrected_fraction = corrected / baseline_error
        harmed_fraction = harmed / baseline_error
        rows.append(
            {
                "label": label,
                "baseline_iou": float(payload["baseline_iou"]),
                "delta_iou": float(result["delta_iou"]),
                "rer": rer,
                "cleared_fp": corrected,
                "lost_tp": harmed,
                "corrected_to_harmed": float(result["corrected_to_harmed"]),
                "baseline_error": baseline_error,
                "corrected_fraction": corrected_fraction,
                "harmed_fraction": harmed_fraction,
                "summary_path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def add_condition_flows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    corrected, harmed = [], []
    for row in frame.itertuples():
        c, h = flow_from_rer_ratio(float(row.rer), float(row.corrected_to_harmed))
        corrected.append(c)
        harmed.append(h)
    frame["corrected_fraction"] = corrected
    frame["harmed_fraction"] = harmed
    return frame


def load_events() -> pd.DataFrame:
    events = pd.read_csv(EVENT_PATH)
    required = {
        "canonical_event_id",
        "dataset_id",
        "n_units",
        "addressable_share",
        "fp_to_tp_ratio",
        "rer",
        "delta_iou",
    }
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Event landscape is missing columns: {sorted(missing)}")
    events = events.copy()
    events["addressable_percentile"] = (
        events.addressable_share.rank(method="average", pct=True) * 100.0
    )
    events["burden_percentile"] = (
        events.fp_to_tp_ratio.rank(method="average", pct=True) * 100.0
    )
    return events


def draw_mechanism_space(
    ax,
    anchors: pd.DataFrame,
    development: pd.DataFrame,
    sealed: pd.DataFrame,
    source_rows: list[dict],
) -> dict:
    """Analytical correction-harm field with anchors and mismatch paths."""

    x_min, x_max = 0.145, 0.385
    y_min, y_max = 0.008, 0.058
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 260),
        np.linspace(y_min, y_max, 220),
    )
    net = xx - yy
    gain_cmap = LinearSegmentedColormap.from_list(
        "analytic_gain", ["#F7F8F8", "#E0EEEB", "#A9CEC7", "#5D9A90"]
    )
    levels = np.linspace(float(net.min()), float(net.max()), 13)
    ax.contourf(xx, yy, net, levels=levels, cmap=gain_cmap, alpha=0.82, antialiased=True)
    contours = ax.contour(
        xx,
        yy,
        net,
        levels=[0.15, 0.20, 0.25, 0.30, 0.35],
        colors="#71958E",
        linewidths=0.55,
        alpha=0.72,
    )
    ax.clabel(contours, inline=True, fontsize=5.8, fmt=lambda value: f"{value:.0%}")

    for frame, color, marker, label, zorder in (
        (development, SLATE, "o", "development mismatch path", 5),
        (sealed, NEUTRAL, "s", "sealed mismatch path", 6),
    ):
        x = frame.corrected_fraction.to_numpy(float)
        y = frame.harmed_fraction.to_numpy(float)
        ax.plot(x, y, color=color, linewidth=1.45, alpha=0.90, zorder=zorder)
        ax.scatter(
            x,
            y,
            s=np.linspace(34, 22, len(x)),
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.75,
            zorder=zorder + 1,
            label=label,
        )
        ax.annotate(
            "aligned",
            xy=(x[0], y[0]),
            xytext=(-4, 8 if label.startswith("development") else 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6.0,
            color=color,
        )
        ax.annotate(
            "donor",
            xy=(x[-1], y[-1]),
            xytext=(-4, -8 if label.startswith("development") else 7),
            textcoords="offset points",
            ha="right",
            va="top" if label.startswith("development") else "bottom",
            fontsize=6.0,
            color=color,
        )
        for condition, row in frame.iterrows():
            source_rows.extend(
                [
                    {
                        "panel": "a",
                        "entity": label,
                        "series": "corrected_fraction",
                        "condition": condition,
                        "value": float(row.corrected_fraction),
                    },
                    {
                        "panel": "a",
                        "entity": label,
                        "series": "harmed_fraction",
                        "condition": condition,
                        "value": float(row.harmed_fraction),
                    },
                    {
                        "panel": "a",
                        "entity": label,
                        "series": "delta_iou",
                        "condition": condition,
                        "value": float(row.delta_iou),
                    },
                ]
            )

    label_positions = {
        "Prithvi-EO-2.0": ("Prithvi", 0.275, 0.0240, "left"),
        "DINOv3-SAT-L": ("DINOv3", 0.286, 0.0227, "left"),
        "Hiera-S-MAE": ("Hiera", 0.337, 0.0215, "center"),
        "ConvNeXtV2-T": ("ConvNeXtV2", 0.345, 0.0112, "center"),
        "DINOv2-S": ("DINOv2", 0.307, 0.0146, "left"),
    }
    for row in anchors.itertuples(index=False):
        ax.scatter(
            row.corrected_fraction,
            row.harmed_fraction,
            s=75 if row.label.startswith("Prithvi") else 62,
            marker="*" if row.label.startswith("Prithvi") else "o",
            facecolor=ANCHOR_COLORS[row.label],
            edgecolor="white",
            linewidth=0.9,
            zorder=9,
        )
        short_label, tx, ty, align = label_positions[row.label]
        ax.annotate(
            short_label,
            xy=(row.corrected_fraction, row.harmed_fraction),
            xytext=(tx, ty),
            textcoords="data",
            ha=align,
            va="center",
            fontsize=6.3,
            color=INK,
            fontweight="bold" if row.label.startswith("Prithvi") else "normal",
            arrowprops={
                "arrowstyle": "-",
                "color": ANCHOR_COLORS[row.label],
                "linewidth": 0.55,
                "alpha": 0.75,
            },
            zorder=10,
        )
        for metric in (
            "corrected_fraction",
            "harmed_fraction",
            "rer",
            "delta_iou",
            "corrected_to_harmed",
        ):
            source_rows.append(
                {
                    "panel": "a",
                    "entity": row.label,
                    "series": metric,
                    "condition": "source_conditioned",
                    "value": float(getattr(row, metric)),
                }
            )

    retention = float(sealed.delta_iou.iloc[0] / development.delta_iou.iloc[0])
    ax.text(
        0.03,
        0.965,
        f"sealed aligned effect retained: {retention:.0%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
        color=SLATE,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": CARD_EDGE,
            "linewidth": 0.45,
            "alpha": 0.90,
        },
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("cleared false positives / baseline pixel errors")
    ax.set_ylabel("added errors / baseline pixel errors")
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor=CARD_EDGE,
        fontsize=6.2,
        handletextpad=0.4,
    )
    tidy(ax, grid_axis="both")
    return {
        "sealed_aligned_retention": retention,
        "development_delta_iou": {
            key: float(value) for key, value in development.delta_iou.items()
        },
        "sealed_delta_iou": {key: float(value) for key, value in sealed.delta_iou.items()},
    }


def draw_event_landscape(
    fig,
    ax,
    events: pd.DataFrame,
    source_rows: list[dict],
) -> dict:
    """Support-aware kernel landscape with all raw events overlaid."""

    diverging = LinearSegmentedColormap.from_list(
        "event_rer", ["#B94D43", "#F5F4F0", "#71C0B3", "#17678A"]
    )
    norm = TwoSlopeNorm(
        vmin=float(events.rer.min()),
        vcenter=0.0,
        vmax=float(events.rer.max()),
    )
    x_event = events.addressable_percentile.to_numpy(float)
    y_event = events.burden_percentile.to_numpy(float)
    z_event = events.rer.to_numpy(float)
    bandwidth_candidates = np.array([8, 10, 12, 15, 18, 22, 26, 30, 36], dtype=float)
    loo_mse: dict[float, float] = {}
    for bandwidth in bandwidth_candidates:
        errors = []
        for index in range(len(events)):
            distance_sq = (x_event - x_event[index]) ** 2 + (y_event - y_event[index]) ** 2
            weights = np.exp(-distance_sq / (2.0 * bandwidth**2))
            weights[index] = 0.0
            prediction = float(np.sum(weights * z_event) / np.sum(weights))
            errors.append((prediction - z_event[index]) ** 2)
        loo_mse[float(bandwidth)] = float(np.mean(errors))
    bandwidth = min(loo_mse, key=loo_mse.get)

    x_grid = np.linspace(0.0, 100.0, 220)
    y_grid = np.linspace(0.0, 100.0, 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    distance_sq = (
        (grid_x[..., None] - x_event[None, None, :]) ** 2
        + (grid_y[..., None] - y_event[None, None, :]) ** 2
    )
    weights = np.exp(-distance_sq / (2.0 * bandwidth**2))
    weight_sum = np.sum(weights, axis=2)
    surface = np.sum(weights * z_event[None, None, :], axis=2) / weight_sum
    effective_n = weight_sum**2 / np.sum(weights**2, axis=2)
    nearest_distance = np.sqrt(np.min(distance_sq, axis=2))

    rgba = diverging(norm(surface))
    support = np.clip((effective_n - 1.0) / 4.0, 0.0, 1.0)
    proximity = np.clip((42.0 - nearest_distance) / 28.0, 0.0, 1.0)
    rgba[..., 3] = 0.18 + 0.62 * support * proximity
    ax.imshow(
        rgba,
        origin="lower",
        extent=[0, 100, 0, 100],
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )
    contour_levels = np.unique(
        np.quantile(surface, [0.10, 0.28, 0.46, 0.64, 0.82, 0.94]).round(4)
    )
    ax.contour(
        grid_x,
        grid_y,
        surface,
        levels=contour_levels,
        colors="#68767C",
        linewidths=0.55,
        alpha=0.62,
        zorder=1,
    )

    log_units = np.log1p(events.n_units.to_numpy(float))
    sizes = 24.0 + 54.0 * (log_units - log_units.min()) / (np.ptp(log_units) or 1.0)
    for dataset, marker in SOURCE_MARKERS.items():
        subset = events.dataset_id.eq(dataset)
        ax.scatter(
            events.loc[subset, "addressable_percentile"],
            events.loc[subset, "burden_percentile"],
            s=sizes[subset],
            c=events.loc[subset, "rer"],
            cmap=diverging,
            norm=norm,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            zorder=3,
        )

    rho_rer, p_rer = spearmanr(events.addressable_share, events.rer)
    rho_iou, p_iou = spearmanr(events.addressable_share, events.delta_iou)
    ax.text(
        0.03,
        0.965,
        f"addressable share vs RER   $\\rho$ = {rho_rer:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=IMPROVE,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": CARD_EDGE,
            "linewidth": 0.45,
            "alpha": 0.90,
        },
    )
    ax.text(
        0.03,
        0.885,
        f"addressable share vs $\\Delta$IoU   $\\rho$ = {rho_iou:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=SLATE,
    )
    ax.text(
        0.50,
        1.035,
        f"n = {len(events)} event-isolated diagnostics",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
        fontweight="bold",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=GRAY,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.0,
            label=SOURCE_LABELS[dataset],
        )
        for dataset, marker in SOURCE_MARKERS.items()
    ]
    ax.legend(
        handles=legend_handles,
        ncol=2,
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor=CARD_EDGE,
        fontsize=6.3,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 102)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("addressable false-positive share percentile")
    ax.set_ylabel("visual FP / TP burden percentile")
    tidy(ax, grid_axis="both")

    cax = fig.add_axes([0.946, 0.235, 0.012, 0.475])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=diverging), cax=cax)
    colorbar.ax.set_title("RER", fontsize=7.0, color=INK, pad=4, fontweight="bold")
    colorbar.ax.tick_params(labelsize=6.3, colors=MUTED, length=2)
    colorbar.outline.set_linewidth(0.55)
    colorbar.outline.set_edgecolor(CARD_EDGE)

    for row, size in zip(events.itertuples(index=False), sizes, strict=True):
        for metric in (
            "addressable_share",
            "fp_to_tp_ratio",
            "addressable_percentile",
            "burden_percentile",
            "rer",
            "delta_iou",
            "n_units",
        ):
            source_rows.append(
                {
                    "panel": "b",
                    "entity": row.canonical_event_id,
                    "series": metric,
                    "condition": row.dataset_id,
                    "value": float(getattr(row, metric)),
                }
            )
        source_rows.append(
            {
                "panel": "b",
                "entity": row.canonical_event_id,
                "series": "marker_area",
                "condition": row.dataset_id,
                "value": float(size),
            }
        )

    return {
        "n_events": int(len(events)),
        "source_counts": {key: int(value) for key, value in events.dataset_id.value_counts().items()},
        "addressable_vs_rer": {"spearman_rho": float(rho_rer), "p_value": float(p_rer)},
        "addressable_vs_delta_iou": {
            "spearman_rho": float(rho_iou),
            "p_value": float(p_iou),
        },
        "rer_range": [float(events.rer.min()), float(events.rer.max())],
        "delta_iou_range": [float(events.delta_iou.min()), float(events.delta_iou.max())],
        "kernel_surface": {
            "axes": ["addressable_percentile", "burden_percentile"],
            "estimator": "Gaussian Nadaraya-Watson regression",
            "bandwidth_candidates_percentile_units": bandwidth_candidates.tolist(),
            "leave_one_event_out_mse": {str(key): value for key, value in loo_mse.items()},
            "selected_bandwidth_percentile_units": float(bandwidth),
            "grid_shape": list(surface.shape),
            "surface_range": [float(surface.min()), float(surface.max())],
            "effective_n_range": [float(effective_n.min()), float(effective_n.max())],
            "alpha_rule": "0.18 + 0.62 * effective-support * proximity",
            "inferential_use": False,
        },
    }


def render(
    mismatch: pd.DataFrame,
    sealed: pd.DataFrame,
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    outdir: Path,
    dpi: int,
) -> tuple[Path, list[dict], dict]:
    fig = plt.figure(figsize=(7.48, 4.12), facecolor="white")  # 190 x 105 mm
    source_rows: list[dict] = []

    ax_mechanism = fig.add_axes([0.075, 0.155, 0.400, 0.675], facecolor="white")
    mechanism_statistics = draw_mechanism_space(
        ax_mechanism,
        anchors,
        add_condition_flows(mismatch),
        add_condition_flows(sealed),
        source_rows,
    )

    ax_events = fig.add_axes([0.555, 0.155, 0.365, 0.675], facecolor="white")
    event_statistics = draw_event_landscape(fig, ax_events, events, source_rows)

    panel_title(
        fig,
        0.045,
        0.925,
        "a",
        "Cross-anchor correction mechanism space",
        gap=0.036,
        title_size=9.2,
    )
    panel_title(
        fig,
        0.525,
        0.925,
        "b",
        "UGCoP context-support gain landscape",
        gap=0.036,
        title_size=9.2,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "figure8_revision_attribution_and_reproduction.png"
    fig.savefig(output, dpi=dpi, facecolor="white")

    panel_dir = outdir / "panels" / "figure8"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for stale in panel_dir.glob("*.png"):
        stale.unlink()
    export_figure_box(
        fig,
        panel_dir / "a_correction_mechanism_space.png",
        (0.025, 0.055, 0.505, 0.985),
        dpi=dpi,
    )
    export_figure_box(
        fig,
        panel_dir / "b_event_gain_landscape.png",
        (0.505, 0.055, 0.990, 0.985),
        dpi=dpi,
    )
    plt.close(fig)

    anchor_statistics = []
    for row in anchors.itertuples(index=False):
        anchor_statistics.append(
            {
                "label": row.label,
                "baseline_iou": float(row.baseline_iou),
                "delta_iou": float(row.delta_iou),
                "rer": float(row.rer),
                "corrected_to_harmed": float(row.corrected_to_harmed),
                "corrected_fraction": float(row.corrected_fraction),
                "harmed_fraction": float(row.harmed_fraction),
            }
        )
    statistics = {
        "anchors": anchor_statistics,
        "mechanism_space": mechanism_statistics,
        "event_landscape": event_statistics,
        "edge_margins_pixels": assert_canvas_clear(output),
    }
    return output, source_rows, statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--publish-path", type=Path, default=DEFAULT_PUBLISH_PATH)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    started = time.time()
    configure_style()
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

    mismatch, sealed = load_conditions()
    anchors = load_anchors()
    events = load_events()

    for name, series in (("development", mismatch.delta_iou), ("sealed", sealed.delta_iou)):
        if not (np.diff(series.to_numpy(float)) < 0).all():
            raise RuntimeError(f"{name} delta_iou is no longer monotonically decreasing")
    if len(anchors) != 5 or not (anchors.rer > 0).all():
        raise RuntimeError("Cross-anchor mechanism contract no longer holds")
    if len(events) != 54:
        raise RuntimeError(f"Expected 54 event diagnostics, found {len(events)}")

    output, source_rows, statistics = render(
        mismatch,
        sealed,
        anchors,
        events,
        args.outdir,
        args.dpi,
    )
    args.publish_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, args.publish_path)

    source_data = args.outdir / "figure8_attribution_and_reproduction_source_data.csv"
    pd.DataFrame(
        source_rows,
        columns=["panel", "entity", "series", "condition", "value"],
    ).to_csv(source_data, index=False)

    report = {
        "schema_version": "figure8_attribution_and_reproduction.v3",
        "backend": "Python/Matplotlib",
        "output_policy": "PNG only",
        "canvas_mm": [190.0, 104.7],
        "background": "white",
        "empirical_surface_policy": {
            "event_landscape": "support-aware Gaussian kernel regression over raw event diagnostics",
            "bandwidth_selection": "leave-one-event-out MSE",
            "raw_events_overlaid": True,
            "inferential_claims_from_surface": False,
        },
        "dpi_requested": args.dpi,
        "render_script": str(SCRIPT_PATH),
        "render_script_sha256": sha256(SCRIPT_PATH),
        "chrome_module": "scripts/figure_chrome.py",
        "chrome_module_sha256": sha256(SCRIPT_PATH.parent / "figure_chrome.py"),
        "source_files": {
            str(MISMATCH_PATH.relative_to(PROJECT_ROOT)): sha256(MISMATCH_PATH),
            str(SEALED_PATH.relative_to(PROJECT_ROOT)): sha256(SEALED_PATH),
            str(EVENT_PATH.relative_to(PROJECT_ROOT)): sha256(EVENT_PATH),
            **{
                row.summary_path: sha256(PROJECT_ROOT / row.summary_path)
                for row in anchors.itertuples(index=False)
            },
        },
        "statistics": statistics,
        "source_data": str(source_data),
        "output": str(output),
        "published_output": str(args.publish_path),
        "published_output_sha256": sha256(args.publish_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = args.outdir / "figure8_attribution_and_reproduction_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "published_output": str(args.publish_path),
                "report": str(report_path),
                "sealed_retention": statistics["mechanism_space"]["sealed_aligned_retention"],
                "event_rho_rer": statistics["event_landscape"]["addressable_vs_rer"][
                    "spearman_rho"
                ],
                "edges": statistics["edge_margins_pixels"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
