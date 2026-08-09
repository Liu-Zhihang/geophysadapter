#!/usr/bin/env python3
"""Render standalone Figure 4(b): false-positive body area and purity.

The panel reuses the frozen object-decision table and the pre-registered
candidate thresholds. It is formatted for later PPT assembly with Figure 4(a)
and writes a single fixed-size 600 dpi PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from PIL import Image

import render_figure4_visual_error_structure as base
from render_figure4a_cross_domain_examples import (
    PANEL_LETTER_PT,
    PANEL_TITLE_PT,
)


CANVAS_WIDTH_MM = 64.0
CANVAS_HEIGHT_MM = 62.0
DEFAULT_OUTDIR = base.DEFAULT_OUTDIR / "panels" / "figure4"
OUTPUT_NAME = "figure4b_false_positive_area_purity.png"

AXIS_LABEL_PT = 8.3
TICK_PT = 7.5
ANNOTATION_PT = 5.7
COLORBAR_PT = 5.8


def configure_style() -> None:
    base.configure_style()
    mpl.rcParams.update(
        {
            "axes.labelsize": AXIS_LABEL_PT,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "axes.linewidth": 0.85,
        }
    )


def render(outdir: Path, dpi: int) -> Path:
    configure_style()
    decisions = pd.read_parquet(base.DECISIONS_PATH)
    contributing = decisions[decisions.false_px > 0].copy()
    total_fp = float(contributing.false_px.sum())
    if total_fp <= 0:
        raise RuntimeError("Object-decision table contains no false-positive mass")
    percent = contributing.false_px.to_numpy(dtype=float) / total_fp * 100.0

    floor = float(base.RULE["candidate_area_px_min"])
    purity_cut = float(base.RULE["candidate_purity_max"])
    max_area = float(decisions.area_px.max())
    near_pure_share = float(
        decisions.loc[decisions.purity <= purity_cut, "false_px"].sum() / total_fp
    )
    corner = decisions[
        (decisions.area_px >= floor) & (decisions.purity <= purity_cut)
    ]
    corner_share = float(corner.false_px.sum() / total_fp)

    fig = plt.figure(
        figsize=(CANVAS_WIDTH_MM / 25.4, CANVAS_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    # The plot keeps the same physical height as panel (a). The color bar is a
    # separate, shorter axis, preventing either element from squeezing the data.
    ax = fig.add_axes([0.155, 0.155, 0.670, 0.670])
    cax = fig.add_axes([0.865, 0.275, 0.025, 0.430])

    hexes = ax.hexbin(
        contributing.area_px.to_numpy(dtype=float),
        contributing.purity.to_numpy(dtype=float),
        C=percent,
        reduce_C_function=np.sum,
        gridsize=(27, 17),
        xscale="log",
        cmap=base.FP_CMAP,
        norm=LogNorm(vmin=0.01, vmax=5.0),
        mincnt=1,
        linewidths=0.0,
        rasterized=True,
        zorder=2,
    )

    # Highlight the pre-registered object-adjudication region using the same
    # false-positive hue family as the original Figure 4.
    ax.axvspan(
        floor,
        max_area * 1.25,
        ymin=0.0,
        ymax=purity_cut,
        color=base.FP_COLOR,
        alpha=0.08,
        linewidth=0,
        zorder=1,
    )
    ax.axvline(
        floor,
        color=base.THRESH_COLOR,
        linewidth=1.0,
        linestyle=(0, (3.0, 2.2)),
        zorder=5,
    )
    ax.axhline(
        purity_cut,
        color=base.THRESH_COLOR,
        linewidth=1.0,
        linestyle=(0, (3.0, 2.2)),
        zorder=5,
    )

    # A compact transparent callout retains the decisive values without
    # covering the distribution. Its width is checked against the 200 px line.
    annotation = ax.text(
        0.016,
        0.965,
        f"purity $\\leq$ .10: {near_pure_share:.1%}\n"
        f"area $\\geq$ 200: {corner_share:.1%}",
        transform=ax.transAxes,
        fontsize=ANNOTATION_PT,
        fontstretch="condensed",
        color=base.INK,
        ha="left",
        va="top",
        linespacing=1.28,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "none",
            "edgecolor": base.GRAY,
            "linewidth": 0.50,
            "alpha": 0.82,
        },
        zorder=7,
    )

    ax.set_xscale("log")
    ax.set_xlim(1.0, max_area * 1.25)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("candidate body area (pixels)", labelpad=3)
    ax.set_ylabel("false-positive purity", labelpad=3)
    ax.tick_params(axis="both", pad=2, length=3.0, width=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#E3E8EC", linewidth=0.48, alpha=0.82)
    ax.set_axisbelow(True)

    bar = fig.colorbar(hexes, cax=cax, orientation="vertical")
    bar.set_ticks([0.01, 0.1, 1.0, 5.0])
    bar.set_ticklabels(["0.01", "0.1", "1", "5"])
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=COLORBAR_PT, pad=1.2, length=2.0, width=0.60)
    bar.set_label(
        "FP mass per bin (%)",
        fontsize=COLORBAR_PT,
        rotation=90,
        labelpad=6.0,
        color=base.INK,
    )
    bar.ax.yaxis.set_label_position("left")
    bar.outline.set_linewidth(0.65)
    bar.outline.set_edgecolor("#7B8790")

    # Match Figure 4(a): the complete header begins at the left canvas edge,
    # with the title immediately following the panel letter on one baseline.
    fig.text(
        0.010,
        0.955,
        "(b)",
        fontsize=PANEL_LETTER_PT,
        fontweight="bold",
        ha="left",
        va="top",
        color=base.INK,
    )
    fig.text(
        0.112,
        0.955,
        "FP body area and purity",
        fontsize=PANEL_TITLE_PT,
        fontweight="semibold",
        ha="left",
        va="top",
        color=base.INK,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / OUTPUT_NAME
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    annotation_bbox = annotation.get_window_extent(renderer)
    threshold_x_px = ax.transData.transform((floor, purity_cut))[0]
    if annotation_bbox.x1 >= threshold_x_px - 2.0:
        raise RuntimeError(
            "Figure 4(b) annotation crosses the 200 px threshold: "
            f"{annotation_bbox.x1:.2f} >= {threshold_x_px - 2.0:.2f}"
        )
    colorbar_bbox = cax.get_window_extent(renderer)
    colorbar_label_bbox = bar.ax.yaxis.label.get_window_extent(renderer)
    if colorbar_bbox.height < colorbar_label_bbox.height:
        raise RuntimeError(
            "Figure 4(b) color bar is shorter than its vertical label: "
            f"{colorbar_bbox.height:.2f} < {colorbar_label_bbox.height:.2f}"
        )
    tight = fig.get_tightbbox(renderer)
    tolerance_in = 0.025
    if (
        tight.x0 < -tolerance_in
        or tight.y0 < -tolerance_in
        or tight.x1 > fig.get_figwidth() + tolerance_in
        or tight.y1 > fig.get_figheight() + tolerance_in
    ):
        raise RuntimeError(f"Figure 4(b) artist extends beyond canvas: {tight}")
    fig.savefig(output, dpi=dpi, facecolor="white", format="png")
    plt.close(fig)

    expected = (
        round(CANVAS_WIDTH_MM / 25.4 * dpi),
        round(CANVAS_HEIGHT_MM / 25.4 * dpi),
    )
    with Image.open(output) as rendered:
        if any(abs(actual - target) > 1 for actual, target in zip(rendered.size, expected)):
            raise RuntimeError(
                f"Unexpected Figure 4(b) size: {rendered.size} != {expected}"
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    print(render(args.outdir, args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
