#!/usr/bin/env python3
"""Render standalone Figure 4(d): body size relative to prior scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from PIL import Image

import render_figure4_visual_error_structure as base
from render_figure4a_cross_domain_examples import PANEL_LETTER_PT, PANEL_TITLE_PT
from render_figure4b_false_positive_area_purity import (
    AXIS_LABEL_PT,
    CANVAS_HEIGHT_MM,
    CANVAS_WIDTH_MM,
    TICK_PT,
)


DEFAULT_OUTDIR = base.DEFAULT_OUTDIR / "panels" / "figure4"
OUTPUT_NAME = "figure4d_body_size_prior_scales.png"
SCALE_LABEL_PT = 5.8
LEGEND_PT = 6.3


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
    diameter = base.equivalent_diameter_m(decisions.area_px.to_numpy(dtype=float))
    order = np.argsort(diameter)
    sorted_diameter = diameter[order]

    fig = plt.figure(
        figsize=(CANVAS_WIDTH_MM / 25.4, CANVAS_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    ax = fig.add_axes([0.155, 0.155, 0.805, 0.670])

    medians: dict[str, float] = {}
    for label, column, color in (
        ("TP mass", "intersection_px", base.TP_COLOR),
        ("FP mass", "false_px", base.FP_COLOR),
    ):
        weights = decisions[column].to_numpy(dtype=float)[order]
        cumulative = np.cumsum(weights) / weights.sum()
        ax.plot(
            sorted_diameter,
            cumulative,
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
            label=label,
            zorder=3,
        )
        median = float(np.interp(0.5, cumulative, sorted_diameter))
        medians[column] = median
        ax.scatter(
            [median],
            [0.5],
            color=color,
            s=30,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

    fp_quartiles = base.mass_weighted_quantile(
        diameter,
        decisions.false_px.to_numpy(dtype=float),
        [0.25, 0.75],
    )
    ax.axvspan(
        fp_quartiles[0],
        fp_quartiles[1],
        color=base.FP_COLOR,
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )

    # Only compact scale values remain in-panel; physical-role descriptions
    # belong in the caption. Every label uses the same height and sits just to
    # the right of its corresponding reference line.
    scale_specs = [
        (10.0, "10 m"),
        (30.0, "30 m"),
        (250.0, "250 m"),
        (5000.0, "5 km"),
    ]
    blend = blended_transform_factory(ax.transData, ax.transAxes)
    scale_texts = []
    for scale, label in scale_specs:
        ax.axvline(
            scale,
            color=base.MUTED,
            linewidth=0.85,
            linestyle=(0, (1.5, 1.8)),
            alpha=0.82,
            zorder=1,
        )
        scale_texts.append(
            ax.text(
                scale * 1.08,
                0.975,
                label,
                transform=blend,
                fontsize=SCALE_LABEL_PT,
                color=base.MUTED,
                ha="left",
                va="top",
            )
        )

    ax.set_xscale("log")
    ax.set_xlim(5.0, 20000.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("equivalent diameter (m)", labelpad=3)
    ax.set_ylabel("cumulative mass fraction", labelpad=3)
    ax.tick_params(axis="both", pad=2, length=3.0, width=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#E3E8EC", linewidth=0.48, alpha=0.82)
    ax.set_axisbelow(True)
    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.035),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.80,
        fontsize=LEGEND_PT,
        handlelength=1.45,
        borderaxespad=0.0,
        labelspacing=0.28,
    )

    fig.text(
        0.010,
        0.955,
        "(d)",
        fontsize=PANEL_LETTER_PT,
        fontweight="bold",
        ha="left",
        va="top",
        color=base.INK,
    )
    fig.text(
        0.112,
        0.955,
        "Body size and prior scales",
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
    scale_boxes = [text.get_window_extent(renderer) for text in scale_texts]
    for index, first in enumerate(scale_boxes):
        for second in scale_boxes[index + 1 :]:
            if first.overlaps(second):
                raise RuntimeError("Figure 4(d) prior-scale labels overlap")
    axes_box = ax.get_window_extent(renderer)
    if not axes_box.contains(*legend.get_window_extent(renderer).get_points()[0]) or not axes_box.contains(
        *legend.get_window_extent(renderer).get_points()[1]
    ):
        raise RuntimeError("Figure 4(d) legend is not fully inside the plotting area")
    tight = fig.get_tightbbox(renderer)
    tolerance_in = 0.025
    if (
        tight.x0 < -tolerance_in
        or tight.y0 < -tolerance_in
        or tight.x1 > fig.get_figwidth() + tolerance_in
        or tight.y1 > fig.get_figheight() + tolerance_in
    ):
        raise RuntimeError(f"Figure 4(d) artist extends beyond canvas: {tight}")
    fig.savefig(output, dpi=dpi, facecolor="white", format="png")
    plt.close(fig)

    expected = (
        round(CANVAS_WIDTH_MM / 25.4 * dpi),
        round(CANVAS_HEIGHT_MM / 25.4 * dpi),
    )
    with Image.open(output) as rendered:
        if any(abs(actual - target) > 1 for actual, target in zip(rendered.size, expected)):
            raise RuntimeError(f"Unexpected Figure 4(d) size: {rendered.size} != {expected}")
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
