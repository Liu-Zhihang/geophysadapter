#!/usr/bin/env python3
"""Render Figure 5 v3: pixel-scale correction, limits, and terrain attribution.

The figure is an asymmetric quantitative composite built from frozen artifacts:

  (a) PILD main evidence: corpus-pooled, event-macro, and four source-pooled
      delta-IoU estimates share one reference axis. Event outcome counts and
      error-flow statistics remain in the source data and figure caption.
  (b) Landslide4Sense auxiliary evidence: six compact bars show how many of
      eight architectures improve from ranking/probability metrics to hard
      decisions. Architecture-level estimates remain in the source-data CSV.
  (c) Landslide4Sense terrain-content controls: eight labelled dumbbells show
      aligned terrain minus neighbour-shift and spatial-roll controls.

The script does not fit a model, tune a threshold, or read the object-scale
result from Section 6.4. It changes only the visual encoding of frozen results.
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
EVENT_SOURCE_PATH = EXP / "pild_object_veto_final_v1/component_decisions.parquet"

INK = "#20272E"
MUTED = "#6C7783"
GRID = "#DDE3E8"
IMPROVE = "#2F7D73"
IMPROVE_LIGHT = "#E9F2F0"
DEGRADE = "#C95843"
DEGRADE_LIGHT = "#F8ECE8"
SLATE = "#4B6F91"
SLATE_LIGHT = "#ECF2F7"
GOLD = "#C38E35"
GOLD_LIGHT = "#F2E7CF"
WARM = "#B86F4B"
WARM_LIGHT = "#F1DDD2"
SAGE = "#6F9875"
SAGE_LIGHT = "#DDEBDD"
VIOLET = "#7567A5"
VIOLET_LIGHT = "#E4E0F0"
CONTROL_BLUE = "#35658F"
CONTROL_BLUE_LIGHT = "#BFD2E3"
CONTROL_VIOLET = "#7A63A8"
CONTROL_VIOLET_LIGHT = "#D8CFE9"
GRAY = "#A9B3BC"
GRAY_LIGHT = "#F3F5F7"

SOURCE_COLORS = {
    "DLR_Landslide_Ref_2025": "#315F9F",
    "GDCLD": "#8A49C7",
    "SEN12LS_HARMONIZED": "#D92E72",
    "GLaD4CD_v1": "#BE4653",
}
SOURCE_LABELS = {
    "DLR_Landslide_Ref_2025": "DLR",
    "GDCLD": "GDCLD",
    "SEN12LS_HARMONIZED": "Sen12",
    "GLaD4CD_v1": "GLaD4CD",
}
SOURCE_ORDER = [
    "DLR_Landslide_Ref_2025",
    "GDCLD",
    "SEN12LS_HARMONIZED",
    "GLaD4CD_v1",
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
SPATIAL_CONTROLS = ["sample_shift", "spatial_roll"]
CONTROL_LABELS = {
    "sample_shift": "neighbour shift",
    "spatial_roll": "spatial roll",
}
EXPECTED_HARDNESS_COUNTS = [8, 7, 7, 7, 5, 4]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
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
    values = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if len(values) != 2:
        raise ValueError(f"Expected two CI endpoints, received {values}")
    return float(values[0]), float(values[1])


def panel_heading(
    fig: Figure,
    letter: str,
    title: str,
    *,
    x: float,
    y: float,
    tag: str | None = None,
    tag_x: float = 0.985,
) -> None:
    fig.text(x, y, f"({letter})", fontsize=10.0, fontweight="bold", ha="left", va="center")
    fig.text(x + 0.036, y, title, fontsize=9.0, fontweight="bold", ha="left", va="center")
    if tag is not None:
        fig.text(tag_x, y, tag, fontsize=6.8, color=MUTED, ha="right", va="center")


def tidy(ax, *, grid_axis: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.95)
    ax.set_axisbelow(True)


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
    frame["display_name"] = frame.architecture_key.map(ARCH_SHORT)
    if frame.display_name.isna().any():
        raise RuntimeError("Missing short architecture label")
    return frame


def load_event_sources() -> tuple[pd.Series, dict[str, str]]:
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
        tally.sort_values(
            ["canonical_event_id", "n_samples", "dataset_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("canonical_event_id")
        .set_index("canonical_event_id")
        .dataset_id
    )
    return winners, {event: str(winners.loc[event]) for event in shared_events}


def metric_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = [
        {
            "key": "delta_ap",
            "label": "$\\Delta$AP",
            "values": frame.mean_delta_average_precision.to_numpy(float),
            "group": "Ranking / probability",
        },
        {
            "key": "brier_improvement",
            "label": "Brier",
            "values": -frame.mean_cluster_delta_brier_adapter_minus_visual.to_numpy(float),
            "group": "Ranking / probability",
        },
        {
            "key": "nll_improvement",
            "label": "NLL",
            "values": -frame.mean_cluster_delta_nll_adapter_minus_visual.to_numpy(float),
            "group": "Ranking / probability",
        },
        {
            "key": "mean_patch_delta_iou",
            "label": "Patch-mean $\\Delta$IoU",
            "values": frame.mean_cluster_delta_foreground_iou.to_numpy(float),
            "group": "Hard decision",
        },
        {
            "key": "pooled_delta_iou",
            "label": "Pooled $\\Delta$IoU",
            "values": frame.delta_foreground_iou.to_numpy(float),
            "group": "Hard decision",
        },
        {
            "key": "net_error_reduction",
            "label": "Net error reduction",
            "values": frame.net_error_reduction_fraction.to_numpy(float),
            "group": "Hard decision",
        },
    ]
    counts = [int(np.sum(np.asarray(item["values"]) > 0)) for item in payload]
    if counts != EXPECTED_HARDNESS_COUNTS:
        raise RuntimeError(f"Decision-metric counts changed: {counts}")
    return payload


def render_panel_a_dense_legacy(
    fig: Figure,
    flow_slot,
    event_slot,
    events: pd.DataFrame,
    datasets: pd.DataFrame,
    event_sources: pd.Series,
    corpus: dict,
    event_bootstrap: dict,
    source_rows: list[dict[str, object]],
) -> dict[str, object]:
    events = events.copy()
    events["dataset_id"] = events.canonical_event_id.map(event_sources)
    if events.dataset_id.isna().any():
        raise RuntimeError("Some events could not be assigned to a source")

    values = events.delta_iou.to_numpy(float)
    zero = np.isclose(values, 0.0, atol=1e-12)
    counts = {
        "positive": int((values > 1e-12).sum()),
        "zero": int(zero.sum()),
        "negative": int((values < -1e-12).sum()),
        "positive_rer": int((events.rer > 0).sum()),
    }
    expected_counts = {"positive": 19, "zero": 26, "negative": 10, "positive_rer": 25}
    if counts != expected_counts:
        raise RuntimeError(f"Event distribution changed: {counts}")

    # Hero error-flow ledger.
    flow_ax = fig.add_subplot(flow_slot)
    corrected = float(corpus["corrected"]) / 1e3
    harmed = float(corpus["harmed"]) / 1e3
    net = float(corpus["net_error_reduction"]) / 1e3
    flow_ax.barh(0, corrected, left=0, height=0.46, color=IMPROVE, edgecolor="none")
    flow_ax.barh(0, -harmed, left=0, height=0.46, color=DEGRADE, edgecolor="none")
    flow_ax.axvline(0, color=INK, linewidth=0.8)
    flow_ax.text(corrected - 8, 0, f"corrected  {corrected:.0f}k", color="white",
                 fontsize=7.5, fontweight="bold", ha="right", va="center")
    flow_ax.text(-harmed + 5, 0, f"harmed  {harmed:.0f}k", color="white",
                 fontsize=7.0, fontweight="bold", ha="left", va="center")
    flow_ax.text(
        0.5,
        1.02,
        f"net {net:.0f}k fewer errors   |   RER {float(corpus['rer']):.2%}   |   "
        f"corrected : harmed = {corrected / harmed:.2f} : 1",
        transform=flow_ax.transAxes,
        fontsize=7.2,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=INK,
    )
    flow_ax.set_xlim(-100, 620)
    flow_ax.set_ylim(-0.65, 0.65)
    flow_ax.set_yticks([])
    flow_ax.set_xticks([])
    for spine in ["top", "right", "left", "bottom"]:
        flow_ax.spines[spine].set_visible(False)

    # Source-stratified event forest. Exact-zero events are represented once per
    # source with a multiplicity label, preserving their scientific meaning as
    # abstentions without drawing 26 coincident markers.
    ax = fig.add_subplot(event_slot)
    ax.axvspan(-0.036, 0, color=DEGRADE_LIGHT, alpha=0.52, zorder=0)
    ax.axvspan(0, 0.036, color=IMPROVE_LIGHT, alpha=0.62, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.85, zorder=1)
    pooled = float(corpus["delta_iou"])
    ax.axvline(pooled, color=IMPROVE, linewidth=1.0, linestyle=(0, (4, 2)), zorder=1)
    ax.text(pooled + 0.0006, 3.38, f"pooled {pooled:+.4f}", color=IMPROVE,
            fontsize=6.9, fontweight="bold", ha="left", va="center")

    source_delta = datasets.set_index("dataset_id").delta_iou.to_dict()
    y_by_source = {source: 3 - index for index, source in enumerate(SOURCE_ORDER)}
    by_source: dict[str, dict[str, int]] = {}

    for source in SOURCE_ORDER:
        block = events[events.dataset_id == source].sort_values(
            ["delta_iou", "canonical_event_id"]
        )
        event_values = block.delta_iou.to_numpy(float)
        is_zero = np.isclose(event_values, 0.0, atol=1e-12)
        nonzero_values = event_values[~is_zero]
        y0 = y_by_source[source]
        offsets = np.linspace(-0.15, 0.15, max(len(nonzero_values), 1))[: len(nonzero_values)]
        color = SOURCE_COLORS[source]
        ax.scatter(nonzero_values, y0 + offsets, s=21, color=color, alpha=0.88,
                   edgecolor="white", linewidth=0.45, zorder=3)

        n_zero = int(is_zero.sum())
        if n_zero:
            ax.scatter([0], [y0], s=43, facecolor="white", edgecolor=color,
                       linewidth=1.0, zorder=4)
            ax.annotate(f"$\\times${n_zero}", xy=(0, y0), xytext=(-7, 0),
                        textcoords="offset points", fontsize=6.7, color=color,
                        ha="right", va="center")

        delta = float(source_delta[source])
        ax.scatter([delta], [y0], s=58, marker="D", color=color,
                   edgecolor="white", linewidth=0.65, zorder=5)
        source_rows.append({
            "panel": "a",
            "series": "source_pooled_delta_iou",
            "x": source,
            "value": delta,
            "secondary_value": np.nan,
        })

        by_source[source] = {
            "n_events": int(len(block)),
            "n_positive": int((event_values > 1e-12).sum()),
            "n_zero": n_zero,
            "n_negative": int((event_values < -1e-12).sum()),
        }
        for row in block.itertuples(index=False):
            source_rows.append({
                "panel": "a",
                "series": "event_delta_iou",
                "x": row.canonical_event_id,
                "value": float(row.delta_iou),
                "secondary_value": float(row.rer),
            })

    macro = float(event_bootstrap["event_macro_delta_iou"])
    macro_ci = [float(value) for value in event_bootstrap["event_macro_delta_iou_ci"]]
    macro_y = -1.0
    ax.hlines(macro_y, macro_ci[0], macro_ci[1], color=SLATE, linewidth=2.0, zorder=3)
    ax.scatter([macro], [macro_y], marker="D", s=49, color=SLATE,
               edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(macro_ci[1] + 0.0008, macro_y,
            f"{macro:+.4f} [{macro_ci[0]:+.4f}, {macro_ci[1]:+.4f}]",
            fontsize=6.7, color=SLATE, ha="left", va="center")

    ax.set_xlim(-0.036, 0.036)
    ax.set_ylim(-1.55, 3.55)
    ax.set_yticks([3, 2, 1, 0, -1])
    ax.set_yticklabels([
        SOURCE_LABELS[SOURCE_ORDER[0]],
        SOURCE_LABELS[SOURCE_ORDER[1]],
        SOURCE_LABELS[SOURCE_ORDER[2]],
        SOURCE_LABELS[SOURCE_ORDER[3]],
        "Event macro",
    ])
    ax.set_xlabel("event $\\Delta$IoU")
    ax.xaxis.set_major_locator(MaxNLocator(7))
    tidy(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=5)
    ax.text(0.0, 1.045, "events: filled circles   |   abstentions: open circle $\\times n$   |   source-pooled: diamond",
            transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="left", va="bottom")
    ax.text(1.0, 1.045, "events + / 0 / − = 19 / 26 / 10",
            transform=ax.transAxes, fontsize=6.7, fontweight="bold", color=INK,
            ha="right", va="bottom")

    source_rows.extend([
        {"panel": "a", "series": "corrected_pixels", "x": "corpus",
         "value": float(corpus["corrected"]), "secondary_value": np.nan},
        {"panel": "a", "series": "harmed_pixels", "x": "corpus",
         "value": float(corpus["harmed"]), "secondary_value": np.nan},
        {"panel": "a", "series": "net_error_reduction_pixels", "x": "corpus",
         "value": float(corpus["net_error_reduction"]), "secondary_value": float(corpus["rer"])},
        {"panel": "a", "series": "pooled_delta_iou", "x": "corpus",
         "value": pooled, "secondary_value": np.nan},
        {"panel": "a", "series": "event_macro_delta_iou", "x": "event_macro",
         "value": macro, "secondary_value": np.nan},
        {"panel": "a", "series": "event_macro_delta_iou_ci", "x": "event_macro",
         "value": macro_ci[0], "secondary_value": macro_ci[1]},
    ])
    return {"counts": counts, "by_source": by_source}


def render_panel_b_dense_legacy(
    fig: Figure,
    slot,
    frame: pd.DataFrame,
    source_rows: list[dict[str, object]],
) -> dict[str, int]:
    payload = metric_payload(frame)
    counts = np.asarray([int((np.asarray(item["values"]) > 0).sum()) for item in payload])
    if counts.tolist() != EXPECTED_HARDNESS_COUNTS:
        raise RuntimeError(f"Decision-metric counts changed: {counts.tolist()}")

    ax = fig.add_subplot(slot)
    y = np.arange(len(payload))
    ax.axhspan(-0.5, 2.5, color=SLATE_LIGHT, alpha=0.75, zorder=0)
    ax.axhspan(2.5, 5.5, color=IMPROVE_LIGHT, alpha=0.72, zorder=0)
    colors = [SLATE] * 3 + [IMPROVE] * 3
    ax.barh(y, counts, height=0.48, color=colors, alpha=0.92, edgecolor="none", zorder=2)
    ax.scatter(counts, y, s=24, color=colors, edgecolor="white", linewidth=0.55, zorder=3)
    for yi, value, color in zip(y, counts, colors, strict=True):
        ax.text(value + 0.18, yi, f"{value}/8", fontsize=7.2, fontweight="bold",
                color=color, ha="left", va="center")

    ax.set_xlim(0, 8.9)
    ax.set_ylim(5.55, -0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([str(item["label"]) for item in payload])
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("architectures improved (of 8)")
    ax.tick_params(axis="y", length=0, pad=4)
    tidy(ax, grid_axis="x")

    for item, count in zip(payload, counts, strict=True):
        source_rows.append({
            "panel": "b",
            "series": f"{item['key']}:n_architectures_improved",
            "x": "count",
            "value": int(count),
            "secondary_value": 8,
        })
        for arch, value in zip(frame.architecture_key, item["values"], strict=True):
            source_rows.append({
                "panel": "b",
                "series": f"{item['key']}:estimate",
                "x": arch,
                "value": float(value),
                "secondary_value": np.nan,
            })
    return {str(item["key"]): int(count) for item, count in zip(payload, counts, strict=True)}


def render_panel_c_dense_legacy(
    fig: Figure,
    slot,
    controls: pd.DataFrame,
    frame: pd.DataFrame,
    source_rows: list[dict[str, object]],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    pivot = controls.pivot(
        index="architecture_key",
        columns="control",
        values="aligned_minus_control_pooled_foreground_iou",
    )
    all_arch = frame.architecture_key.tolist()
    pivot = pivot.loc[all_arch]

    pass_counts: dict[str, dict[str, int]] = {}
    for control in ["zero", *SPATIAL_CONTROLS]:
        block = controls[controls.control == control].set_index("architecture_key").loc[all_arch]
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

    shift_all = pivot.sample_shift.to_numpy(float) * 1e3
    roll_all = pivot.spatial_roll.to_numpy(float) * 1e3
    if not (shift_all > 0).all() or not (roll_all > 0).all():
        raise RuntimeError("A spatial control is no longer beaten on every architecture")

    order = np.argsort(np.minimum(shift_all, roll_all))
    arch_order = [all_arch[index] for index in order]
    shift = shift_all[order]
    roll = roll_all[order]
    y = np.arange(len(arch_order))

    ax = fig.add_subplot(slot)
    xmax = float(max(shift.max(), roll.max())) * 1.14
    ax.axvspan(0, xmax, color=IMPROVE_LIGHT, alpha=0.72, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)
    ax.hlines(y, 0, np.maximum(shift, roll), color="#D1E3DF", linewidth=2.1, zorder=1)
    ax.hlines(y, np.minimum(shift, roll), np.maximum(shift, roll),
              color=GRAY, linewidth=1.2, zorder=2)
    ax.scatter(shift, y, s=27, marker="o", color=IMPROVE, edgecolor="white",
               linewidth=0.55, zorder=3, label=CONTROL_LABELS["sample_shift"])
    ax.scatter(roll, y, s=29, marker="D", color=SLATE, edgecolor="white",
               linewidth=0.55, zorder=3, label=CONTROL_LABELS["spatial_roll"])

    ax.set_xlim(-1.2, xmax)
    ax.set_ylim(len(arch_order) - 0.5, -0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([ARCH_SHORT[key] for key in arch_order])
    ax.set_xlabel("aligned − control, pooled $\\Delta$IoU ($\\times10^{-3}$)")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.tick_params(axis="y", length=0, pad=4)
    tidy(ax, grid_axis="x")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=4.8,
                   markerfacecolor=IMPROVE, markeredgecolor="white",
                   label=CONTROL_LABELS["sample_shift"]),
            Line2D([0], [0], marker="D", linestyle="none", markersize=4.8,
                   markerfacecolor=SLATE, markeredgecolor="white",
                   label=CONTROL_LABELS["spatial_roll"]),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        frameon=False,
        ncol=2,
        handletextpad=0.35,
        labelspacing=0.30,
        borderaxespad=0.0,
    )
    ax.text(0.985, 0.965, "16 / 16 contrasts > 0", transform=ax.transAxes,
            fontsize=7.2, fontweight="bold", color=IMPROVE, ha="right", va="top")

    for control, values in [("sample_shift", shift), ("spatial_roll", roll)]:
        for arch, value in zip(arch_order, values, strict=True):
            source_rows.append({
                "panel": "c",
                "series": f"aligned_minus_{control}",
                "x": arch,
                "value": float(value) / 1e3,
                "secondary_value": np.nan,
            })
    return pass_counts, arch_order


def render_panel_a(
    fig: Figure,
    slot,
    events: pd.DataFrame,
    datasets: pd.DataFrame,
    event_sources: pd.Series,
    corpus: dict,
    event_bootstrap: dict,
    source_rows: list[dict[str, object]],
) -> dict[str, object]:
    """PILD source-level summary in the visual language of Figure 6."""

    events = events.copy()
    events["dataset_id"] = events.canonical_event_id.map(event_sources)
    if events.dataset_id.isna().any():
        raise RuntimeError("Some events could not be assigned to a source")

    values = events.delta_iou.to_numpy(float)
    counts = {
        "positive": int((values > 1e-12).sum()),
        "zero": int(np.isclose(values, 0.0, atol=1e-12).sum()),
        "negative": int((values < -1e-12).sum()),
        "positive_rer": int((events.rer > 0).sum()),
    }
    expected_counts = {"positive": 19, "zero": 26, "negative": 10, "positive_rer": 25}
    if counts != expected_counts:
        raise RuntimeError(f"Event distribution changed: {counts}")

    source_delta = datasets.set_index("dataset_id").delta_iou.to_dict()
    macro = float(event_bootstrap["event_macro_delta_iou"])
    macro_ci = [float(value) for value in event_bootstrap["event_macro_delta_iou_ci"]]
    pooled = float(corpus["delta_iou"])

    rows = [
        ("All PILD, pooled", pooled, GOLD, "D"),
        ("Event macro", macro, MUTED, "D"),
        ("GDCLD", float(source_delta["GDCLD"]), WARM, "o"),
        ("GLaD4CD", float(source_delta["GLaD4CD_v1"]), WARM, "o"),
        ("DLR", float(source_delta["DLR_Landslide_Ref_2025"]), WARM, "o"),
        ("Sen12 within PILD", float(source_delta["SEN12LS_HARMONIZED"]), WARM, "o"),
    ]

    ax = fig.add_subplot(slot)
    # The two aggregate estimates are the hero comparison. Source-specific
    # estimates use a compressed row pitch and thinner marks below them.
    y = np.asarray([0.0, 1.0, 2.15, 2.82, 3.49, 4.16])
    ax.axhspan(-0.45, 0.45, color=GOLD_LIGHT, alpha=0.38, zorder=0)
    ax.axhspan(0.55, 1.45, color=GRAY_LIGHT, alpha=0.48, zorder=0)
    ax.axhspan(1.82, 4.49, color=WARM_LIGHT, alpha=0.24, zorder=0)
    ax.axhline(1.68, color="#E5D8D0", linewidth=0.65, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)
    for index, ((label, value, color, marker), yi) in enumerate(zip(rows, y, strict=True)):
        if label == "Event macro":
            ax.hlines(yi, macro_ci[0], macro_ci[1], color="#D8DDE2",
                      linewidth=7.5, alpha=0.70, zorder=1)
            ax.hlines(yi, macro_ci[0], macro_ci[1], color=MUTED,
                      linewidth=1.35, zorder=2)
        else:
            light = GOLD_LIGHT if index == 0 else WARM_LIGHT
            width = 8.0 if index == 0 else 5.5
            ax.plot([0, value], [yi, yi], color=light, linewidth=width,
                    alpha=0.76 if index == 0 else 0.68,
                    solid_capstyle="round", zorder=1)
        ax.scatter([value], [yi], s=52 if index < 2 else 35, marker=marker,
                   color=color, edgecolor="white", linewidth=0.65, zorder=3)
        if label == "Event macro":
            value_text = f"{value:+.4f}  [{macro_ci[0]:+.4f}, {macro_ci[1]:+.4f}]"
            text_x = macro_ci[1] + 0.00055
        else:
            value_text = f"{value:+.4f}"
            text_x = value + 0.00055
        ax.text(text_x, yi, value_text, fontsize=7.0 if index < 2 else 6.6, color=color,
                fontweight="bold" if index == 0 else "normal", ha="left", va="center")

    ax.set_xlim(-0.0038, 0.0235)
    ax.set_ylim(4.52, -0.45)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    for index, tick in enumerate(ax.get_yticklabels()):
        tick.set_fontsize(7.8 if index < 2 else 7.0)
    ax.set_xlabel("$\\Delta$IoU relative to the frozen visual prediction")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="y", length=0, pad=6)
    tidy(ax, grid_axis="x")

    by_source: dict[str, dict[str, int]] = {}
    for source in SOURCE_ORDER:
        block = events[events.dataset_id == source]
        event_values = block.delta_iou.to_numpy(float)
        by_source[source] = {
            "n_events": int(len(block)),
            "n_positive": int((event_values > 1e-12).sum()),
            "n_zero": int(np.isclose(event_values, 0.0, atol=1e-12).sum()),
            "n_negative": int((event_values < -1e-12).sum()),
        }
        for row in block.itertuples(index=False):
            source_rows.append({
                "panel": "a",
                "series": "event_delta_iou_not_drawn",
                "x": row.canonical_event_id,
                "value": float(row.delta_iou),
                "secondary_value": float(row.rer),
            })

    for label, value, _, _ in rows:
        source_rows.append({
            "panel": "a",
            "series": "displayed_delta_iou",
            "x": label,
            "value": value,
            "secondary_value": np.nan,
        })
    source_rows.append({
        "panel": "a",
        "series": "event_macro_delta_iou_ci",
        "x": "Event macro",
        "value": macro_ci[0],
        "secondary_value": macro_ci[1],
    })
    return {"counts": counts, "by_source": by_source}


def render_panel_b(
    fig: Figure,
    slot,
    frame: pd.DataFrame,
    source_rows: list[dict[str, object]],
) -> dict[str, int]:
    """Six architecture-consistency rows with eight as the reference."""

    payload = metric_payload(frame)
    counts = np.asarray([int((np.asarray(item["values"]) > 0).sum()) for item in payload])
    if counts.tolist() != EXPECTED_HARDNESS_COUNTS:
        raise RuntimeError(f"Decision-metric counts changed: {counts.tolist()}")

    ax = fig.add_subplot(slot)
    y = np.arange(len(payload))
    colors = [SAGE] * 3 + [IMPROVE] * 3
    pale = [SAGE_LIGHT] * 3 + ["#D2E5E1"] * 3
    ax.axhspan(-0.5, 2.5, color=SAGE_LIGHT, alpha=0.28, zorder=0)
    ax.axhspan(2.5, 5.5, color=IMPROVE_LIGHT, alpha=0.36, zorder=0)
    ax.axvline(8, color=GRAY, linewidth=0.9, zorder=1)
    for yi, count, color, light in zip(y, counts, colors, pale, strict=True):
        ax.plot([count, 8], [yi, yi], color=light, linewidth=8.0,
                solid_capstyle="round", zorder=1)
        ax.scatter([count], [yi], s=43, color=color, edgecolor="white",
                   linewidth=0.65, zorder=3)
        ax.text(count - 0.12, yi, f"{count}/8", fontsize=7.0, fontweight="bold",
                color=color, ha="right", va="center")

    ax.set_xlim(3.35, 8.55)
    ax.set_ylim(len(payload) - 0.5, -0.5)
    ax.set_xticks([4, 5, 6, 7, 8])
    ax.set_yticks(y)
    ax.set_yticklabels([str(item["label"]) for item in payload], fontsize=7.2)
    ax.set_xlabel("architectures improved (of 8)")
    ax.tick_params(axis="y", length=0, pad=5)
    tidy(ax, grid_axis="x")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=4.8,
                   markerfacecolor=SAGE, markeredgecolor="white",
                   label="ranking / probability"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=4.8,
                   markerfacecolor=IMPROVE, markeredgecolor="white",
                   label="hard decision"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.035),
        frameon=False,
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    for item, count in zip(payload, counts, strict=True):
        source_rows.append({
            "panel": "b",
            "series": f"{item['key']}:n_architectures_improved",
            "x": "count",
            "value": int(count),
            "secondary_value": 8,
        })
        for arch, value in zip(frame.architecture_key, item["values"], strict=True):
            source_rows.append({
                "panel": "b",
                "series": f"{item['key']}:estimate_not_drawn",
                "x": arch,
                "value": float(value),
                "secondary_value": np.nan,
            })
    return {str(item["key"]): int(count) for item, count in zip(payload, counts, strict=True)}


def render_panel_c(
    fig: Figure,
    slot,
    controls: pd.DataFrame,
    frame: pd.DataFrame,
    source_rows: list[dict[str, object]],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Two aligned-minus-control series, drawn against the zero reference."""

    all_arch = frame.architecture_key.tolist()
    pivot = controls.pivot(
        index="architecture_key",
        columns="control",
        values="aligned_minus_control_pooled_foreground_iou",
    ).loc[all_arch]

    pass_counts: dict[str, dict[str, int]] = {}
    for control in ["zero", *SPATIAL_CONTROLS]:
        block = controls[controls.control == control].set_index("architecture_key").loc[all_arch]
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

    shift_all = pivot.sample_shift.to_numpy(float) * 1e3
    roll_all = pivot.spatial_roll.to_numpy(float) * 1e3
    if not (shift_all > 0).all() or not (roll_all > 0).all():
        raise RuntimeError("A spatial control is no longer beaten on every architecture")
    order = np.argsort(np.minimum(shift_all, roll_all))
    arch_order = [all_arch[index] for index in order]
    shift = shift_all[order]
    roll = roll_all[order]

    ax = fig.add_subplot(slot)
    y = np.arange(len(arch_order))
    offset = 0.10
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)
    for yi, shift_value, roll_value in zip(y, shift, roll, strict=True):
        ax.plot([0, shift_value], [yi - offset, yi - offset], color=CONTROL_BLUE_LIGHT,
                linewidth=7.0, alpha=0.76, solid_capstyle="round", zorder=2)
        ax.plot([0, roll_value], [yi + offset, yi + offset], color=CONTROL_VIOLET_LIGHT,
                linewidth=7.0, alpha=0.48, solid_capstyle="round", zorder=1)
        # The semi-transparent diamond remains visible without masking a nearby
        # blue endpoint; the blue endpoint is deliberately drawn last.
        ax.scatter([roll_value], [yi + offset], s=40, marker="D", color=CONTROL_VIOLET,
                   edgecolor="white", linewidth=0.6, alpha=0.70, zorder=3)
        ax.scatter([shift_value], [yi - offset], s=39, marker="o", color=CONTROL_BLUE,
                   edgecolor="white", linewidth=0.6, alpha=0.96, zorder=4)

    ax.set_xlim(-1.2, 25.5)
    ax.set_ylim(len(arch_order) - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([ARCH_SHORT[key] for key in arch_order], fontsize=7.1)
    ax.set_xlabel("aligned − control, pooled $\\Delta$IoU ($\\times10^{-3}$)")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.tick_params(axis="y", length=0, pad=5)
    tidy(ax, grid_axis="x")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=4.8,
                   markerfacecolor=CONTROL_BLUE, markeredgecolor="white", alpha=0.96,
                   label=CONTROL_LABELS["sample_shift"]),
            Line2D([0], [0], marker="D", linestyle="none", markersize=4.8,
                   markerfacecolor=CONTROL_VIOLET, markeredgecolor="white", alpha=0.70,
                   label=CONTROL_LABELS["spatial_roll"]),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.035),
        frameon=False,
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax.text(1.0, 1.055, "16 / 16 > 0", transform=ax.transAxes, fontsize=7.0,
            color=CONTROL_BLUE, fontweight="bold", ha="right", va="bottom")

    for control, vals in [("sample_shift", shift), ("spatial_roll", roll)]:
        for arch, value in zip(arch_order, vals, strict=True):
            source_rows.append({
                "panel": "c",
                "series": f"aligned_minus_{control}",
                "x": arch,
                "value": float(value) / 1e3,
                "secondary_value": np.nan,
            })
    return pass_counts, arch_order


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


FIG_WIDTH_IN = 7.48
FIG_HEIGHT_IN = 5.55


def render(
    frame: pd.DataFrame,
    controls: pd.DataFrame,
    events: pd.DataFrame,
    datasets: pd.DataFrame,
    event_sources: pd.Series,
    corpus: dict,
    event_bootstrap: dict,
    outdir: Path,
    dpi: int,
) -> tuple[Path, Path, dict[str, object]]:
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), facecolor="white")
    top = fig.add_gridspec(
        1, 1,
        left=0.155,
        right=0.985,
        top=0.905,
        bottom=0.565,
    )
    bottom = fig.add_gridspec(
        1, 2,
        left=0.145,
        right=0.985,
        top=0.360,
        bottom=0.095,
        width_ratios=[0.92, 1.08],
        wspace=0.39,
    )

    source_rows: list[dict[str, object]] = []
    event_stats = render_panel_a(
        fig,
        top[0, 0],
        events,
        datasets,
        event_sources,
        corpus,
        event_bootstrap,
        source_rows,
    )
    hardness_counts = render_panel_b(fig, bottom[0, 0], frame, source_rows)
    pass_counts, control_arch_order = render_panel_c(
        fig, bottom[0, 1], controls, frame, source_rows
    )

    panel_heading(
        fig,
        "a",
        "PILD pixel-scale correction and heterogeneity",
        x=0.012,
        y=0.965,
    )
    panel_heading(
        fig,
        "b",
        "L4S architecture consistency",
        x=0.012,
        y=0.445,
    )
    panel_heading(
        fig,
        "c",
        "L4S terrain-content controls",
        x=0.555,
        y=0.445,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "figure5_revision_pixel_scale_capability.png"
    fig.savefig(output, dpi=dpi, facecolor="white")

    panel_dir = outdir / "panels" / "figure5"
    export_figure_box(fig, panel_dir / "a_pild_source_gain.png",
                      (0.004, 0.515, 0.996, 0.996), dpi=dpi)
    export_figure_box(fig, panel_dir / "b_metric_consistency.png",
                      (0.004, 0.010, 0.545, 0.495), dpi=dpi)
    export_figure_box(fig, panel_dir / "c_terrain_content_controls.png",
                      (0.535, 0.010, 0.996, 0.495), dpi=dpi)
    plt.close(fig)

    edge_margins = assert_canvas_clear(output)
    source_data = outdir / "figure5_pixel_scale_capability_source_data.csv"
    pd.DataFrame(
        source_rows,
        columns=["panel", "series", "x", "value", "secondary_value"],
    ).to_csv(source_data, index=False)
    return output, source_data, {
        "pass_counts": pass_counts,
        "hardness_counts": hardness_counts,
        "event_counts": event_stats["counts"],
        "event_counts_by_source": event_stats["by_source"],
        "control_architecture_order": control_arch_order,
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
    output, source_data, derived = render(
        frame,
        controls,
        events,
        datasets,
        event_sources,
        corpus,
        event_bootstrap,
        args.outdir,
        args.dpi,
    )

    family_rer = (
        frame.assign(rer_positive=frame.net_error_reduction_fraction > 0)
        .groupby("family")
        .agg(
            n_architectures=("architecture_key", "size"),
            n_rer_positive=("rer_positive", "sum"),
        )
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
        PROBABILITY_PATH,
        THRESHOLD_PATH,
        CONTROL_PATH,
        DECISION_PATH,
        EVENT_PATH,
        DATASET_PATH,
        FULL_SUMMARY_PATH,
        BOOTSTRAP_PATH,
        EVENT_SOURCE_PATH,
    ]
    report = {
        "schema_version": "figure5_pixel_scale_capability.v3",
        "backend": "Python/Matplotlib",
        "output_policy": "PNG only",
        "canvas_mm": [round(FIG_WIDTH_IN * 25.4, 1), round(FIG_HEIGHT_IN * 25.4, 1)],
        "dpi_requested": args.dpi,
        "panel_weighting": (
            "PILD corpus, event-macro, and source-pooled effects form the hero panel; "
            "Landslide4Sense supplies compact cross-architecture and content-control evidence."
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
            "decision_metric_improved_counts": derived["hardness_counts"],
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
        },
        "visual_encoding": {
            "event_estimates": "retained in source data and caption; not individually drawn",
            "source_pooled_estimates": "reference-line lollipops on a common delta-IoU axis",
            "event_macro": "diamond with event-bootstrap 95% interval",
            "architecture_effect_magnitudes": "retained in source data; main panel shows improvement counts only",
            "control_order": "sorted by smaller aligned-minus-control margin for display only",
        },
        "overclaim_guards": {
            "object_scale_result_drawn": False,
            "delta_ap_rer_correlation_reported": False,
            "cross_metric_effect_sizes_compared": False,
            "brier_and_nll_sign_reversed_so_positive_means_improvement": True,
            "l4s_labelled_as_auxiliary_cohort_in_figure": True,
        },
        "control_architecture_order": derived["control_architecture_order"],
        "edge_self_check": derived["edge_margins_pixels"],
        "source_data": str(source_data),
        "output": str(output),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = args.outdir / "figure5_pixel_scale_capability_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "source_data": str(source_data),
        "report": str(report_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
