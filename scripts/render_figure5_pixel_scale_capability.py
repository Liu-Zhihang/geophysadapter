#!/usr/bin/env python3
"""Render Figure 5: pixel-scale capability and its decision-scale limit.

Panel weighting follows the evidence hierarchy of section 3.5. The unified PILD
corpus is the primary evidence and takes the full top row; Landslide4Sense is an
auxiliary cohort that may only support cross-architecture consistency, so it is
demoted to a compact strip and labelled as such inside the figure.

  (a) PILD, 55 event-isolated events: per-event delta IoU coloured by source,
      the 26 abstentions made explicit, and three corpus reference levels
      (object scale, pixel pooled, pixel event mean with its 95% interval);
  (b) Landslide4Sense, 8 architectures x 6 metrics ordered by decision hardness:
      the improvement count decays 8 -> 7 -> 7 -> 7 -> 5 -> 4;
  (c) Landslide4Sense terrain content control: aligned minus each spatial
      control, positive on every architecture.

The script reads only frozen experiment artifacts. It does not fit a model,
select a test configuration, or compute a cross-metric correlation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

from figure_chrome import export_figure_box


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXP = PROJECT_ROOT / "experiments/revision2026"
DEFAULT_OUTDIR = (
    PROJECT_ROOT.parent
    / "docs/assets"
)

PROBABILITY_PATH = (
    EXP
    / "l4s_terrain_same_threshold_audit_formal_v2_20260718/architecture_probability_summary.csv"
)
THRESHOLD_PATH = (
    EXP
    / "l4s_terrain_same_threshold_audit_formal_v2_20260718/architecture_threshold_summary.csv"
)
CONTROL_PATH = (
    EXP
    / "l4s_terrain_controls_same_threshold_formal_v2_20260718/architecture_control_contrasts.csv"
)
DECISION_PATH = (
    EXP
    / "l4s_terrain_controls_same_threshold_formal_v2_20260718/decision_summary.csv"
)
PIXEL_ROOT = EXP / "pild_native17_source_stratified_tempered075_v1"
EVENT_PATH = PIXEL_ROOT / "crossfold_benefit_gate_v2_full/event_metrics.csv"
DATASET_PATH = PIXEL_ROOT / "crossfold_benefit_gate_v2_full/dataset_metrics.csv"
FULL_SUMMARY_PATH = PIXEL_ROOT / "crossfold_benefit_gate_v2_full/summary.json"
BOOTSTRAP_PATH = PIXEL_ROOT / "crossfold_benefit_gate_v2_bootstrap/summary.json"
# Object-scale corpus result, drawn only as a reference level so that the
# pixel-scale limitation is legible instead of merely stated.
OBJECT_SUMMARY_PATH = EXP / "pild_object_veto_final_v1/summary.json"
# Event -> source mapping. The object decision table is the only frozen artifact
# carrying both identifiers on the same 55 events.
EVENT_SOURCE_PATH = EXP / "pild_object_veto_final_v1/component_decisions.parquet"

INK = "#1E242B"
MUTED = "#6B7580"
GRID = "#DFE4E9"
IMPROVE = "#3E7F76"
DEGRADE = "#C0503C"
NEUTRAL = "#D9B570"
SLATE = "#4E6E8E"
GRAY = "#AAB3BC"
GRAY_LIGHT = "#EDF0F3"
# Darker member of the improvement family: same semantics as IMPROVE, but
# visually separable from the event markers it is compared against.
OBJECT_REF = "#1F4F49"

SOURCE_COLORS = {
    "GLaD4CD_v1": "#C43C51",
    "DLR_Landslide_Ref_2025": "#2F5FAA",
    "GDCLD": "#8E44D6",
    "SEN12LS_HARMONIZED": "#E1126E",
}
SOURCE_LABELS = {
    "GLaD4CD_v1": "GLaD4CD",
    "DLR_Landslide_Ref_2025": "DLR",
    "GDCLD": "GDCLD",
    "SEN12LS_HARMONIZED": "Sen12",
}
SOURCE_LEGEND_ORDER = [
    "GDCLD",
    "GLaD4CD_v1",
    "DLR_Landslide_Ref_2025",
    "SEN12LS_HARMONIZED",
]
ARCH_SHORT = {
    "dinov2_s": "DINOv2-S",
    "fcmae_convnextv2_t": "FCMAE-CNv2-T",
    "hiera_s_mae": "Hiera-S-MAE",
    "satmae_vit_b": "SatMAE-B",
    "deeplabv3plus": "DeepLabV3+",
    "unetplusplus": "U-Net++",
    "fpn": "FPN",
    "deeplabv3plus_imagenet": "DeepLabV3+ IN",
}
# `zero` blanks the terrain tensor and therefore also removes capacity; the two
# spatial controls keep the tensor and only move it, so they are the ones drawn.
SPATIAL_CONTROLS = ["sample_shift", "spatial_roll"]
CONTROL_LABELS = {
    "zero": "zero",
    "sample_shift": "neighbour shift",
    "spatial_roll": "spatial roll",
}
EXPECTED_HARDNESS_COUNTS = [8, 7, 7, 7, 5, 4]


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
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ci(raw: object) -> tuple[float, float]:
    if isinstance(raw, str):
        values = ast.literal_eval(raw)
    else:
        values = raw
    if len(values) != 2:
        raise ValueError(f"Expected two CI endpoints, received {values}")
    return float(values[0]), float(values[1])


def panel_heading(fig: Figure, x_letter: float, x_title: float, y: float, letter: str, title: str,
                  *, subtitle: str | None = None) -> None:
    fig.text(x_letter, y, f"({letter})", fontsize=9.8, fontweight="bold", ha="left", va="center")
    fig.text(x_title, y, title, fontsize=8.8, fontweight="bold", ha="left", va="center")
    if subtitle is not None:
        fig.text(x_title, y - 0.026, subtitle, fontsize=7.0, color=MUTED, ha="left", va="center")


def tidy(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.95)
    ax.set_axisbelow(True)


# --- Data loading ------------------------------------------------------------


def load_architecture_table() -> pd.DataFrame:
    probability = pd.read_csv(PROBABILITY_PATH)
    threshold = pd.read_csv(THRESHOLD_PATH)
    threshold = threshold[threshold.threshold_policy == "frozen_visual"].copy()
    if len(probability) != 8 or len(threshold) != 8:
        raise RuntimeError("Figure 5 requires exactly eight frozen architecture rows")
    frame = probability.merge(
        threshold,
        on=["architecture_key", "architecture", "family"],
        validate="one_to_one",
    )
    frame = frame.sort_values("mean_delta_average_precision").reset_index(drop=True)
    frame["display_name"] = frame.architecture_key.map(ARCH_SHORT)
    if frame.display_name.isna().any():
        raise RuntimeError("Missing short architecture label")
    return frame


def load_event_sources() -> tuple[pd.Series, dict[str, str]]:
    """Assign each canonical event to the source contributing most of its tiles.

    One event is observed by two sources (56 source-event observations over 55
    canonical events, as reported for Figure 2). It is assigned by tile majority
    and the tie is recorded in the render report rather than silently dropped.
    """

    table = pd.read_parquet(
        EVENT_SOURCE_PATH, columns=["dataset_id", "canonical_event_id", "sample_id"]
    )
    tally = (
        table.groupby(["canonical_event_id", "dataset_id"])
        .sample_id.nunique()
        .rename("n_samples")
        .reset_index()
    )
    shared = tally.canonical_event_id.value_counts()
    shared_events = sorted(shared[shared > 1].index.tolist())
    winners = (
        tally.sort_values(["canonical_event_id", "n_samples", "dataset_id"],
                          ascending=[True, False, True])
        .drop_duplicates("canonical_event_id")
        .set_index("canonical_event_id")
        .dataset_id
    )
    resolved = {
        event: str(winners.loc[event]) for event in shared_events
    }
    return winners, resolved


def metric_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Six metrics ordered by distance from a hard segmentation decision.

    Signs are normalised so that positive always means improvement; Brier and
    NLL are therefore negated.
    """

    payload: list[dict[str, object]] = []
    values = frame.mean_delta_average_precision.to_numpy(float)
    payload.append(
        {
            "key": "delta_ap",
            "label": "$\\Delta$AP",
            "values": values,
            "interval": "seed min-max",
        }
    )
    for key, label, value_col in [
        ("brier_improvement", "Brier", "mean_cluster_delta_brier_adapter_minus_visual"),
        ("nll_improvement", "NLL", "mean_cluster_delta_nll_adapter_minus_visual"),
    ]:
        payload.append(
            {
                "key": key,
                "label": label,
                "values": -frame[value_col].to_numpy(float),
                "interval": "cluster bootstrap 95% CI",
            }
        )
    for key, label, value_col in [
        ("mean_patch_delta_iou", "patch-mean\n$\\Delta$IoU", "mean_cluster_delta_foreground_iou"),
        ("pooled_delta_iou", "pooled\n$\\Delta$IoU", "delta_foreground_iou"),
        ("net_error_reduction", "net error\nreduction", "net_error_reduction_fraction"),
    ]:
        payload.append(
            {
                "key": key,
                "label": label,
                "values": frame[value_col].to_numpy(float),
                "interval": "cluster bootstrap 95% CI",
            }
        )
    observed = [int(np.sum(np.asarray(item["values"]) > 0)) for item in payload]
    if observed != EXPECTED_HARDNESS_COUNTS:
        raise RuntimeError(f"Decision-hardness counts changed: {observed}")
    return payload


# --- Panels ------------------------------------------------------------------


def render_panel_a(
    fig: Figure,
    slot,
    events: pd.DataFrame,
    event_sources: pd.Series,
    corpus: dict,
    event_bootstrap: dict,
    object_delta_iou: float,
    source_rows: list[dict],
) -> dict[str, object]:
    """PILD main result: where the pixel-scale gain lives and where it abstains."""

    ax = fig.add_subplot(slot)
    ordered = events.sort_values(["delta_iou", "canonical_event_id"]).reset_index(drop=True)
    ordered["dataset_id"] = ordered.canonical_event_id.map(event_sources)
    if ordered.dataset_id.isna().any():
        raise RuntimeError("Some events could not be assigned to a source")

    values = ordered.delta_iou.to_numpy(float)
    x = np.arange(1, len(ordered) + 1)
    zero = np.isclose(values, 0.0, atol=1e-12)
    positive = values > 1e-12
    negative = values < -1e-12
    counts = {
        "positive": int(positive.sum()),
        "zero": int(zero.sum()),
        "negative": int(negative.sum()),
        "positive_rer": int((events.rer > 0).sum()),
    }
    if counts != {"positive": 19, "zero": 26, "negative": 10, "positive_rer": 25}:
        raise RuntimeError(f"Event distribution changed: {counts}")

    colors = np.asarray([SOURCE_COLORS[key] for key in ordered.dataset_id])
    ax.vlines(x[~zero], 0, values[~zero], color=colors[~zero], linewidth=1.0, alpha=0.85)
    ax.scatter(
        x[~zero], values[~zero], c=colors[~zero], s=17, edgecolor="white", linewidth=0.4, zorder=3
    )
    # Abstentions are drawn hollow: the adapter returned the visual prediction
    # unchanged, which is a committed behaviour rather than a missing value.
    ax.scatter(
        x[zero],
        np.zeros(int(zero.sum())),
        facecolor="white",
        edgecolor=colors[zero],
        s=17,
        linewidth=0.85,
        zorder=4,
    )

    macro = float(event_bootstrap["event_macro_delta_iou"])
    macro_ci = [float(value) for value in event_bootstrap["event_macro_delta_iou_ci"]]
    pooled = float(corpus["delta_iou"])
    ratio = object_delta_iou / pooled

    ax.axhspan(macro_ci[0], macro_ci[1], color=SLATE, alpha=0.13, zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.65, zorder=1)
    ax.axhline(macro, color=SLATE, linewidth=1.0, linestyle=(0, (4, 2)), zorder=2)
    ax.axhline(pooled, color=INK, linewidth=1.05, zorder=2)
    ax.axhline(object_delta_iou, color=OBJECT_REF, linewidth=1.3, linestyle=(0, (7, 2.5)), zorder=2)

    # Reference levels are labelled directly above their own lines, in the empty
    # upper-left quadrant (events 1-36 are non-positive), so no legend box is
    # needed and no line strikes through its own label.
    for level, color, weight, text in [
        (object_delta_iou, OBJECT_REF, "bold",
         f"object scale, corpus  {object_delta_iou:+.4f}   ({ratio:.1f}$\\times$ the pixel scale)"),
        (pooled, INK, "bold", f"pixel scale, corpus  {pooled:+.4f}"),
        (macro, SLATE, "normal",
         f"pixel scale, event mean  {macro:+.4f}  [{macro_ci[0]:+.4f}, {macro_ci[1]:+.4f}]"),
    ]:
        ax.text(
            1.6,
            level + 0.0013,
            text,
            ha="left",
            va="bottom",
            fontsize=6.9,
            fontweight=weight,
            color=color,
        )

    zero_positions = x[zero]
    ax.annotate(
        f"{counts['zero']} events unchanged: bounded adapter returned the visual prediction",
        xy=(float(np.median(zero_positions)), 0.0),
        xytext=(float(np.median(zero_positions)), -0.0085),
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTED,
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.7},
    )

    # Source legend goes in the empty lower-right quadrant: every event right of
    # x = 37 is positive, so nothing is occluded.
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=4.2,
                   markerfacecolor=SOURCE_COLORS[key], markeredgecolor="white",
                   markeredgewidth=0.4, label=SOURCE_LABELS[key])
            for key in SOURCE_LEGEND_ORDER
        ],
        loc="lower right",
        frameon=False,
        fontsize=7.0,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.35,
        labelspacing=0.35,
        borderaxespad=0.4,
    )

    ax.set_xlim(0, 56.5)
    ax.set_ylim(-0.0375, 0.0400)
    ax.set_xlabel("55 event-isolated events, sorted by $\\Delta$IoU")
    ax.set_ylabel("event $\\Delta$IoU")
    ax.set_xticks([1, 10, 20, 30, 40, 50, 55])
    tidy(ax, grid_axis="y")

    for row in ordered.itertuples(index=False):
        source_rows.append(
            {
                "panel": "a",
                "series": "event_delta_iou",
                "x": row.canonical_event_id,
                "value": float(row.delta_iou),
                "secondary_value": float(row.rer),
            }
        )
    source_rows.extend(
        [
            {"panel": "a", "series": "pooled_delta_iou", "x": "corpus", "value": pooled,
             "secondary_value": np.nan},
            {"panel": "a", "series": "event_macro_delta_iou", "x": "event_macro", "value": macro,
             "secondary_value": np.nan},
            {"panel": "a", "series": "event_macro_delta_iou_ci", "x": "event_macro",
             "value": macro_ci[0], "secondary_value": macro_ci[1]},
            {"panel": "a", "series": "object_scale_reference_delta_iou", "x": "corpus",
             "value": object_delta_iou, "secondary_value": ratio},
        ]
    )
    by_source = (
        ordered.groupby("dataset_id")
        .agg(n_events=("delta_iou", "size"),
             n_positive=("delta_iou", lambda s: int((s > 1e-12).sum())),
             n_zero=("delta_iou", lambda s: int(np.isclose(s, 0.0, atol=1e-12).sum())))
        .to_dict(orient="index")
    )
    return {"counts": counts, "by_source": {k: {kk: int(vv) for kk, vv in v.items()}
                                            for k, v in by_source.items()}}


def render_panel_b(
    fig: Figure,
    stair_slot,
    matrix_slot,
    frame: pd.DataFrame,
    source_rows: list[dict],
) -> tuple[list[str], dict[str, int]]:
    """Auxiliary cohort: the effect decays monotonically as the decision hardens."""

    payload = metric_payload(frame)
    matrix = np.vstack([np.asarray(item["values"], dtype=float) for item in payload]).T
    n_improved = (matrix > 0).sum(axis=1)
    # Order architectures by how far along the hardness axis they keep improving,
    # so the decay reads as a shape rather than as eight unrelated rows.
    order = np.lexsort((-frame.mean_delta_average_precision.to_numpy(float), -n_improved))
    frame = frame.iloc[order].reset_index(drop=True)
    matrix = matrix[order]
    arch_order = frame.architecture_key.tolist()

    stair_ax = fig.add_subplot(stair_slot)
    ax = fig.add_subplot(matrix_slot)

    counts = np.asarray([int((matrix[:, j] > 0).sum()) for j in range(matrix.shape[1])])
    if counts.tolist() != EXPECTED_HARDNESS_COUNTS:
        raise RuntimeError(f"Decision-hardness counts changed: {counts.tolist()}")
    columns = np.arange(matrix.shape[1])
    stair_ax.step(columns, counts, where="mid", color=SLATE, linewidth=1.25)
    stair_ax.scatter(columns, counts, s=17, color=SLATE, zorder=3)
    for column, value in zip(columns, counts, strict=True):
        stair_ax.text(column, value + 0.55, f"{value}/8", ha="center", va="bottom",
                      fontsize=7.2, fontweight="bold", color=SLATE)
    stair_ax.set_xlim(-0.6, matrix.shape[1] - 0.4)
    stair_ax.set_ylim(3.0, 10.2)
    stair_ax.set_xticks([])
    stair_ax.set_yticks([])
    stair_ax.set_frame_on(False)
    stair_ax.text(-0.012, 0.42, "architectures\nimproved", transform=stair_ax.transAxes,
                  ha="right", va="center", fontsize=7.0, color=MUTED, linespacing=1.3)

    # Colour carries the sign, area carries the effect size within its own
    # column. Magnitudes are not comparable across columns, so they are never
    # placed on a shared numeric axis.
    for column in columns:
        values = matrix[:, column]
        scale = float(np.abs(values).max())
        sizes = 15.0 + 60.0 * (np.abs(values) / scale)
        improved = values > 0
        ax.scatter(np.full(improved.sum(), column), np.where(improved)[0],
                   s=sizes[improved], color=IMPROVE, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(np.full((~improved).sum(), column), np.where(~improved)[0],
                   s=sizes[~improved], color=DEGRADE, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xlim(-0.6, matrix.shape[1] - 0.4)
    ax.set_ylim(len(frame) - 0.5, -0.7)
    ax.set_xticks(columns)
    ax.set_xticklabels([item["label"] for item in payload], fontsize=6.9, linespacing=1.3)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame.display_name, fontsize=7.2)
    ax.tick_params(axis="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    # Hardness direction sits under the tick labels, clear of the matrix.
    ax.annotate(
        "", xy=(0.99, -0.235), xytext=(0.01, -0.235), xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8},
        annotation_clip=False,
    )
    ax.text(0.5, -0.275, "threshold-free  $\\rightarrow$  hard decision",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.9, color=MUTED)

    for row_index, arch in enumerate(arch_order):
        for column, item in enumerate(payload):
            source_rows.append(
                {
                    "panel": "b",
                    "series": f"{item['key']}:estimate",
                    "x": arch,
                    "value": float(matrix[row_index, column]),
                    "secondary_value": np.nan,
                }
            )
    return arch_order, {str(item["key"]): int(value)
                        for item, value in zip(payload, counts, strict=True)}


def render_panel_c(
    fig: Figure,
    slot,
    controls: pd.DataFrame,
    arch_order: list[str],
    source_rows: list[dict],
) -> dict[str, dict[str, int]]:
    """Auxiliary cohort: the change depends on real terrain content, not capacity."""

    ax = fig.add_subplot(slot)
    pivot = (
        controls.pivot(index="architecture_key", columns="control",
                       values="aligned_minus_control_pooled_foreground_iou")
        .loc[arch_order]
    )
    pass_counts: dict[str, dict[str, int]] = {}
    for control in ["zero", *SPATIAL_CONTROLS]:
        block = controls[controls.control == control].set_index("architecture_key").loc[arch_order]
        pass_counts[control] = {
            "threshold_free": int(block.threshold_free_content_pass.sum()),
            "hard_decision": int(block.hard_decision_content_pass.sum()),
        }
    expected = {
        "zero": {"threshold_free": 6, "hard_decision": 7},
        "sample_shift": {"threshold_free": 8, "hard_decision": 8},
        "spatial_roll": {"threshold_free": 8, "hard_decision": 8},
    }
    if pass_counts != expected:
        raise RuntimeError(f"Terrain-control counts changed: {pass_counts}")

    y = np.arange(len(arch_order))
    shift = pivot["sample_shift"].to_numpy(float) * 1e3
    roll = pivot["spatial_roll"].to_numpy(float) * 1e3
    if not (shift > 0).all() or not (roll > 0).all():
        raise RuntimeError("A spatial control is no longer beaten on every architecture")

    ax.hlines(y, np.minimum(shift, roll), np.maximum(shift, roll),
              color=GRAY, linewidth=1.1, zorder=2)
    ax.scatter(shift, y, s=22, marker="o", color=IMPROVE, edgecolor="white",
               linewidth=0.5, zorder=3, label=CONTROL_LABELS["sample_shift"])
    ax.scatter(roll, y, s=24, marker="D", color=SLATE, edgecolor="white",
               linewidth=0.5, zorder=3, label=CONTROL_LABELS["spatial_roll"])
    ax.axvline(0, color=INK, linewidth=0.85, zorder=1)

    ax.set_ylim(len(arch_order) - 0.5, -0.7)
    ax.set_xlim(-1.6, float(max(shift.max(), roll.max())) * 1.20)
    ax.set_yticks(y)
    ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_xlabel("aligned $-$ control, pooled $\\Delta$IoU ($\\times10^{-3}$)", fontsize=7.4,
                  labelpad=3.0)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    # Upper right is empty: the two top architectures have the smallest margins.
    ax.legend(loc="upper right", frameon=False, fontsize=7.0, handletextpad=0.3,
              labelspacing=0.35, borderaxespad=0.35)
    tidy(ax, grid_axis="x")
    ax.text(0.0, -0.345, "all 16 contrasts positive; zero control in Table S4",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=MUTED)

    for control, values in [("sample_shift", shift), ("spatial_roll", roll)]:
        for arch, value in zip(arch_order, values, strict=True):
            source_rows.append(
                {
                    "panel": "c",
                    "series": f"aligned_minus_{control}",
                    "x": arch,
                    "value": float(value) / 1e3,
                    "secondary_value": np.nan,
                }
            )
    return pass_counts


def assert_canvas_clear(output: Path) -> dict[str, int]:
    image = np.asarray(Image.open(output).convert("L"))
    ink = image < 160
    ys = np.where(ink.any(axis=1))[0]
    xs = np.where(ink.any(axis=0))[0]
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Rendered figure is blank")
    bounds = {
        "left": int(xs.min()),
        "right": int(image.shape[1] - 1 - xs.max()),
        "top": int(ys.min()),
        "bottom": int(image.shape[0] - 1 - ys.max()),
    }
    if min(bounds.values()) < 3:
        raise RuntimeError(f"Figure ink touches canvas edge: {bounds}")
    return bounds


# --- Assembly ----------------------------------------------------------------


FIG_WIDTH_IN = 7.48
FIG_HEIGHT_IN = 6.80


def render(
    frame: pd.DataFrame,
    controls: pd.DataFrame,
    events: pd.DataFrame,
    event_sources: pd.Series,
    corpus: dict,
    event_bootstrap: dict,
    object_delta_iou: float,
    outdir: Path,
    dpi: int,
) -> tuple[Path, Path, dict]:
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), facecolor="white")
    # The primary corpus owns the full top row; the auxiliary cohort shares one
    # architecture axis across the two lower panels.
    row_a = fig.add_gridspec(1, 1, left=0.088, right=0.984, top=0.872, bottom=0.578)
    # Shared two-row grid so the architecture rows of (b) and (c) line up; the
    # top-right cell stays empty because only (b) carries the count staircase.
    row_bc = fig.add_gridspec(
        2, 2, left=0.152, right=0.984, top=0.412, bottom=0.140,
        width_ratios=[1.34, 1.0], height_ratios=[1.0, 3.25],
        wspace=0.075, hspace=0.08,
    )

    source_rows: list[dict] = []
    event_stats = render_panel_a(
        fig, row_a[0, 0], events, event_sources, corpus, event_bootstrap,
        object_delta_iou, source_rows,
    )
    arch_order, hardness_counts = render_panel_b(
        fig, row_bc[0, 0], row_bc[1, 0], frame, source_rows
    )
    pass_counts = render_panel_c(fig, row_bc[1, 1], controls, arch_order, source_rows)

    corrected = float(corpus["corrected"])
    harmed = float(corpus["harmed"])
    net = float(corpus["net_error_reduction"])
    panel_heading(
        fig, 0.010, 0.048, 0.962, "a",
        "Unified PILD corpus: where the pixel-scale gain lives",
        subtitle=(
            f"corrected {corrected / 1e3:.0f}k \u00b7 harmed {harmed / 1e3:.0f}k \u00b7 "
            f"net \u2212{net / 1e3:.0f}k pixels ({corrected / harmed:.2f}\u2009:\u20091), "
            f"pooled error reduction {float(corpus['rer']):.2%}"
        ),
    )
    panel_heading(
        fig, 0.010, 0.048, 0.496, "b",
        "Effect decays as the decision hardens",
        subtitle="Landslide4Sense auxiliary cohort \u00b7 cross-architecture consistency only",
    )
    panel_heading(
        fig, 0.583, 0.621, 0.496, "c",
        "The change needs real terrain content",
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "figure5_pixel_scale_capability.png"
    fig.savefig(output, dpi=dpi, facecolor="white")

    panel_dir = outdir / "panels" / "figure5"
    export_figure_box(fig, panel_dir / "a_event_wise_pild.png", (0.004, 0.530, 0.996, 0.996), dpi=dpi)
    export_figure_box(fig, panel_dir / "b_hardness_decay.png", (0.004, 0.010, 0.600, 0.522), dpi=dpi)
    export_figure_box(fig, panel_dir / "c_terrain_content.png", (0.575, 0.010, 0.996, 0.522), dpi=dpi)

    plt.close(fig)
    edge_margins = assert_canvas_clear(output)

    source_data = outdir / "figure5_pixel_scale_capability_source_data.csv"
    pd.DataFrame(source_rows, columns=["panel", "series", "x", "value", "secondary_value"]).to_csv(
        source_data, index=False
    )
    return output, source_data, {
        "pass_counts": pass_counts,
        "hardness_counts": hardness_counts,
        "event_counts": event_stats["counts"],
        "event_counts_by_source": event_stats["by_source"],
        "architecture_order": arch_order,
        "edge_margins_pixels": edge_margins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    started = time.time()
    configure_style()

    frame = load_architecture_table()
    controls = pd.read_csv(CONTROL_PATH)
    events = pd.read_csv(EVENT_PATH)
    datasets = pd.read_csv(DATASET_PATH)
    full_summary = json.loads(FULL_SUMMARY_PATH.read_text("utf-8"))
    bootstrap_summary = json.loads(BOOTSTRAP_PATH.read_text("utf-8"))
    if len(events) != 55 or len(datasets) != 4:
        raise RuntimeError("Unexpected event or dataset inventory")

    corpus = full_summary["corpus"]
    event_bootstrap = bootstrap_summary["event_bootstrap"]
    event_sources, shared_events = load_event_sources()

    object_summary = json.loads(OBJECT_SUMMARY_PATH.read_text("utf-8"))
    object_delta_iou = float(object_summary["verdict"]["delta_iou"])
    if abs(object_delta_iou - 0.030946) > 5e-6:
        raise RuntimeError(f"Object-scale reference changed: {object_delta_iou}")
    if abs(float(object_summary["baseline_iou"]) - float(corpus["baseline_iou"])) > 1e-3:
        raise RuntimeError(
            "Object and pixel scales no longer share the same visual baseline; "
            "the reference level in panel (a) would not be comparable"
        )

    output, source_data, derived = render(
        frame, controls, events, event_sources, corpus, event_bootstrap,
        object_delta_iou, args.outdir, args.dpi,
    )

    family_rer = (
        frame.assign(rer_positive=frame.net_error_reduction_fraction > 0)
        .groupby("family")
        .agg(n_architectures=("architecture_key", "size"), n_rer_positive=("rer_positive", "sum"))
        .to_dict(orient="index")
    )
    family_rer_clean = {
        family: {key: int(value) for key, value in record.items()}
        for family, record in family_rer.items()
    }
    expected_family = {
        "foundation_model": {"n_architectures": 4, "n_rer_positive": 2},
        "modern_bn_frozen": {"n_architectures": 4, "n_rer_positive": 2},
    }
    if family_rer_clean != expected_family:
        raise RuntimeError(f"Architecture-family counts changed: {family_rer_clean}")

    source_files = [
        PROBABILITY_PATH, THRESHOLD_PATH, CONTROL_PATH, DECISION_PATH,
        EVENT_PATH, DATASET_PATH, FULL_SUMMARY_PATH, BOOTSTRAP_PATH,
        OBJECT_SUMMARY_PATH, EVENT_SOURCE_PATH,
    ]
    report = {
        "schema_version": "figure5_pixel_scale_capability.v2",
        "backend": "Python/Matplotlib",
        "output_policy": "PNG only",
        "canvas_mm": [round(FIG_WIDTH_IN * 25.4, 1), round(FIG_HEIGHT_IN * 25.4, 1)],
        "dpi_requested": args.dpi,
        "panel_weighting": (
            "PILD corpus is primary and occupies the full top row; Landslide4Sense "
            "is labelled in-figure as an auxiliary cross-architecture cohort, per "
            "section 3.5"
        ),
        "render_script": str(SCRIPT_PATH),
        "render_script_sha256": sha256(SCRIPT_PATH),
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in source_files
        },
        "event_source_assignment": {
            "rule": "source contributing the most tiles to the canonical event",
            "shared_events_resolved": shared_events,
        },
        "statistics": {
            "decision_hardness_improved_counts": derived["hardness_counts"],
            "terrain_control_pass_counts": derived["pass_counts"],
            "event_delta_iou_counts": derived["event_counts"],
            "event_delta_iou_counts_by_source": derived["event_counts_by_source"],
            "pooled_delta_iou": float(corpus["delta_iou"]),
            "pooled_rer": float(corpus["rer"]),
            "corrected_pixels": int(corpus["corrected"]),
            "harmed_pixels": int(corpus["harmed"]),
            "net_error_reduction_pixels": int(corpus["net_error_reduction"]),
            "corrected_to_harmed": float(corpus["corrected"] / corpus["harmed"]),
            "event_macro_delta_iou": float(event_bootstrap["event_macro_delta_iou"]),
            "event_macro_delta_iou_ci95": [
                float(value) for value in event_bootstrap["event_macro_delta_iou_ci"]
            ],
            "source_delta_iou": {
                str(row.dataset_id): float(row.delta_iou)
                for row in datasets.itertuples(index=False)
            },
            "rer_positive_by_family": family_rer_clean,
            "object_scale_reference_delta_iou": object_delta_iou,
            "object_over_pixel_delta_iou_ratio": object_delta_iou / float(corpus["delta_iou"]),
        },
        "overclaim_guards": {
            "delta_ap_rer_correlation_reported": False,
            "delta_ap_rer_fit_drawn": False,
            "brier_and_nll_sign_reversed_so_positive_means_improvement": True,
            "dot_area_normalised_within_column_not_across_columns": True,
            "l4s_labelled_as_auxiliary_cohort_in_figure": True,
        },
        "edge_self_check": derived["edge_margins_pixels"],
        "source_data": str(source_data),
        "output": str(output),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = args.outdir / "figure5_pixel_scale_capability_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "source_data": str(source_data),
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
