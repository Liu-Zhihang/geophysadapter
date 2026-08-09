#!/usr/bin/env python3
"""Render standalone Figure 4(c): retained mass by body-size threshold."""

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
OUTPUT_NAME = "figure4c_retention_by_size.png"
DIRECT_LABEL_PT = 6.5
THRESHOLD_LABEL_PT = 5.9


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
    max_area = float(decisions.area_px.max())
    thresholds = np.unique(np.rint(np.geomspace(1, max_area, 180))).astype(float)
    tp_share, fp_share = base.mass_survival(decisions, thresholds)

    floor = float(base.RULE["candidate_area_px_min"])
    tp_at_floor, fp_at_floor = base.mass_survival(decisions, np.asarray([floor]))
    tp_value = float(tp_at_floor[0])
    fp_value = float(fp_at_floor[0])

    fig = plt.figure(
        figsize=(CANVAS_WIDTH_MM / 25.4, CANVAS_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    ax = fig.add_axes([0.155, 0.155, 0.805, 0.670])

    ax.plot(
        thresholds,
        tp_share,
        color=base.TP_COLOR,
        linewidth=2.0,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        thresholds,
        fp_share,
        color=base.FP_COLOR,
        linewidth=2.0,
        solid_capstyle="round",
        zorder=3,
    )
    ax.axvline(
        floor,
        color=base.THRESH_COLOR,
        linewidth=base.THRESH_LW,
        linestyle=base.THRESH_STYLE,
        zorder=2,
    )
    ax.scatter(
        [floor, floor],
        [tp_value, fp_value],
        color=[base.TP_COLOR, base.FP_COLOR],
        s=30,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

    # Direct labels replace the two arrows used in the combined legacy panel.
    # The FP value is deliberately anchored left of the threshold as requested.
    fp_label = ax.text(
        floor / 1.18,
        fp_value - 0.035,
        f"FP {fp_value:.1%}",
        fontsize=DIRECT_LABEL_PT,
        fontweight="semibold",
        color=base.FP_COLOR,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        zorder=6,
    )
    ax.text(
        floor * 1.32,
        tp_value + 0.035,
        f"TP {tp_value:.1%}",
        fontsize=DIRECT_LABEL_PT,
        fontweight="semibold",
        color=base.TP_COLOR,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        zorder=6,
    )
    blend = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        floor * 1.10,
        0.970,
        "200 px",
        transform=blend,
        fontsize=THRESHOLD_LABEL_PT,
        color=base.MUTED,
        ha="left",
        va="top",
    )

    ax.set_xscale("log")
    ax.set_xlim(1.0, max_area * 1.25)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("minimum body area (pixels)", labelpad=3)
    ax.set_ylabel("retained mass fraction", labelpad=3)
    ax.tick_params(axis="both", pad=2, length=3.0, width=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#E3E8EC", linewidth=0.48, alpha=0.82)
    ax.set_axisbelow(True)

    fig.text(
        0.010,
        0.955,
        "(c)",
        fontsize=PANEL_LETTER_PT,
        fontweight="bold",
        ha="left",
        va="top",
        color=base.INK,
    )
    fig.text(
        0.112,
        0.955,
        "Retention by size threshold",
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
    threshold_x_px = ax.transData.transform((floor, fp_value))[0]
    if fp_label.get_window_extent(renderer).x1 >= threshold_x_px - 2.0:
        raise RuntimeError("Figure 4(c) FP label is not fully left of the 200 px line")
    tight = fig.get_tightbbox(renderer)
    tolerance_in = 0.025
    if (
        tight.x0 < -tolerance_in
        or tight.y0 < -tolerance_in
        or tight.x1 > fig.get_figwidth() + tolerance_in
        or tight.y1 > fig.get_figheight() + tolerance_in
    ):
        raise RuntimeError(f"Figure 4(c) artist extends beyond canvas: {tight}")
    fig.savefig(output, dpi=dpi, facecolor="white", format="png")
    plt.close(fig)

    expected = (
        round(CANVAS_WIDTH_MM / 25.4 * dpi),
        round(CANVAS_HEIGHT_MM / 25.4 * dpi),
    )
    with Image.open(output) as rendered:
        if any(abs(actual - target) > 1 for actual, target in zip(rendered.size, expected)):
            raise RuntimeError(f"Unexpected Figure 4(c) size: {rendered.size} != {expected}")
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
