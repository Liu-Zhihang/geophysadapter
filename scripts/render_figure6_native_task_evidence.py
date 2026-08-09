#!/usr/bin/env python3
"""Render revised Figure 6 from frozen native-task evidence.

The figure is deliberately data-led: one real spatial example and one cohort
distribution for Terrain, real probability maps plus internal/external event
distributions for Material, and a 138-event rainfall-window heatmap plus dose
distribution for Trigger. Only a 600 dpi PNG is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EXP = PROJECT_ROOT / "experiments/revision2026"
PA = PROJECT_ROOT / "metadata/protocol_assets/pild_core_v2_1_phase14_physical_20260719"
DEFAULT_OUTDIR = REPO_ROOT / "submission_package_jprs_revision1/geophysadapter/revision1/figures_revision"

TERRAIN_DIR = PA / "external_terrain_ranker_spatial_exclusion_100km"
TERRAIN_SUMMARY = TERRAIN_DIR / "summary.json"
TERRAIN_EVENTS = TERRAIN_DIR / "oof_event_metrics.csv"
TERRAIN_SCORES = TERRAIN_DIR / "oof_scores.h5"
TERRAIN_MANIFEST = TERRAIN_DIR / "surviving_fold_manifest.csv"
COPDEM_ROOT = PROJECT_ROOT / "raw_fullcopy/static/copdem_glo30_2021"

MATERIAL_DIR = EXP / "pild_material_susceptibility_interaction_v1"
MATERIAL_SUMMARY = MATERIAL_DIR / "summary.json"
MATERIAL_EVENTS = MATERIAL_DIR / "event_paired_deltas.csv"
MATERIAL_MAPS = MATERIAL_DIR / "figure6_material_case_maps.npz"
MATERIAL_MAPS_META = MATERIAL_DIR / "figure6_material_case_maps.json"

EXTERNAL_DIR = EXP / "glad_material_intrinsic_association_prospective_v2_20260718"
EXTERNAL_SUMMARY = EXTERNAL_DIR / "run_summary.json"
EXTERNAL_EVENTS = EXTERNAL_DIR / "metrics/oof_event_metrics.csv"

TRIGGER_DIR = PA / "trigger_preflight"
TRIGGER_SUMMARY = TRIGGER_DIR / "summary.json"
TRIGGER_EVENTS = TRIGGER_DIR / "trigger_oof_event_scores.csv"
TRIGGER_SAME_SEASON = TRIGGER_DIR / "same_season_cross_year_oof.csv"

WHITE = "#FFFFFF"
INK = "#20262D"
MUTED = "#65717D"
FRAME = "#CDD4DA"
GRID = "#E5E9EC"
TERRAIN = "#477C68"
TERRAIN_PALE = "#C9DDD2"
MATERIAL = "#C18A32"
MATERIAL_PALE = "#EAD8B7"
TRIGGER = "#B6382D"
TRIGGER_PALE = "#F0B39C"
NEGATIVE = "#B95E4B"

FIG_WIDTH_IN = 7.48
FIG_HEIGHT_IN = 9.62


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.titlesize": 8.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.char.decode(values, "utf-8") if values.dtype.kind == "S" else values.astype(str)


def panel_title(ax, letter: str, title: str) -> None:
    ax.axis("off")
    ax.text(0.0, 0.5, f"({letter}) {title}", ha="left", va="center", fontsize=9.4,
            fontweight="bold", color=INK, transform=ax.transAxes)


def clean_image_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME)
        spine.set_linewidth(0.65)


def image_label(ax, text: str, *, corner: str = "upper left") -> None:
    positions = {
        "upper left": (0.035, 0.965, "left", "top"),
        "lower left": (0.035, 0.035, "left", "bottom"),
        "lower right": (0.965, 0.035, "right", "bottom"),
    }
    x, y, ha, va = positions[corner]
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va,
        fontsize=6.7, fontweight="bold", color="white", zorder=8,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#4B5158",
            "edgecolor": "none",
            "alpha": 0.58,
        },
    )


def add_scale_bar(ax, pixels: float, label: str) -> None:
    x0, x1 = ax.get_xlim()
    fraction = pixels / abs(x1 - x0)
    ax.plot([0.06, 0.06 + fraction], [0.07, 0.07], transform=ax.transAxes,
            color="white", lw=2.0, solid_capstyle="butt",
            path_effects=[pe.Stroke(linewidth=3.2, foreground="black", alpha=0.55), pe.Normal()])
    ax.text(0.06, 0.105, label, transform=ax.transAxes, color="white", fontsize=6.1,
            ha="left", va="bottom",
            path_effects=[pe.withStroke(linewidth=1.35, foreground="black", alpha=0.8)])


def parse_native_cell(cell_id: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"EPSG(\d+):X(-?\d+):Y(-?\d+)", str(cell_id))
    if not match:
        raise ValueError(f"Unexpected native cell id: {cell_id}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def hillshade(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    fill = float(np.nanmedian(values[finite])) if finite.any() else 0.0
    values = np.where(finite, values, fill)
    return LightSource(azdeg=315, altdeg=43).shade(
        values, cmap=mpl.colormaps["Greys"], vert_exag=1.15, blend_mode="soft"
    )


def load_terrain() -> dict[str, object]:
    event_metrics = pd.read_csv(TERRAIN_EVENTS)
    main = event_metrics[event_metrics.condition == "main_six"].copy()
    supported = main[main.pairwise_auc > 0.5].copy()
    median = float(supported.pairwise_auc.median())
    supported["distance"] = (supported.pairwise_auc - median).abs()
    selected = supported.sort_values(["distance", "physical_event_id"]).iloc[0]
    event_id = str(selected.physical_event_id)

    with h5py.File(TERRAIN_SCORES, "r") as handle:
        keep = decode(handle["physical_event_id"][...]) == event_id
        rows = np.asarray(handle["native_grid_row"][...][keep], dtype=int)
        cols = np.asarray(handle["native_grid_col"][...][keep], dtype=int)
        roles = np.asarray(handle["role"][...][keep], dtype=int)
        scores = np.asarray(handle["score_main_six"][...][keep], dtype=float)
        cell_ids = decode(handle["native_cell_id"][...][keep])

    row_min, col_min = int(rows.min()), int(cols.min())
    score_grid = np.full((int(rows.max()) - row_min + 1, int(cols.max()) - col_min + 1), np.nan)
    score_grid[rows - row_min, cols - col_min] = scores
    case = np.flatnonzero(roles == 1)
    if len(case) != 1:
        raise RuntimeError(f"Expected one terrain case cell, found {len(case)}")
    case_index = int(case[0])

    parsed = [parse_native_cell(value) for value in cell_ids]
    epsg_values = {item[0] for item in parsed}
    if len(epsg_values) != 1:
        raise RuntimeError("Terrain example spans multiple CRSs")
    xs = np.asarray([item[1] for item in parsed], dtype=int)
    ys = np.asarray([item[2] for item in parsed], dtype=int)
    x_min, x_max = xs.min() * 30.0, (xs.max() + 1) * 30.0
    y_min, y_max = ys.min() * 30.0, (ys.max() + 1) * 30.0
    manifest = pd.read_csv(TERRAIN_MANIFEST)
    manifest_row = manifest[manifest.physical_event_id == event_id].iloc[0]
    dem_path = COPDEM_ROOT / str(manifest_row.home_copdem_tile)
    with rasterio.open(dem_path) as dataset:
        bounds = transform_bounds(
            f"EPSG:{next(iter(epsg_values))}", dataset.crs,
            x_min, y_min, x_max, y_max, densify_pts=21,
        )
        dem = dataset.read(
            1,
            window=from_bounds(*bounds, transform=dataset.transform),
            out_shape=score_grid.shape,
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=np.nan,
        )
    summary = json.loads(TERRAIN_SUMMARY.read_text("utf-8"))["conditions"]["main_six"]
    return {
        "events": main.pairwise_auc.to_numpy(float),
        "estimate": float(summary["event_balanced_pairwise_auc"]),
        "ci": [float(value) for value in summary["event_bootstrap_pairwise_auc_ci95"]],
        "score_grid": score_grid,
        "dem": dem,
        "case_row": int(rows[case_index] - row_min),
        "case_col": int(cols[case_index] - col_min),
        "event_id": event_id,
        "selection_rule": "event nearest the median AUC among above-chance events",
    }


def load_material() -> dict[str, object]:
    maps = np.load(MATERIAL_MAPS)
    metadata = json.loads(MATERIAL_MAPS_META.read_text("utf-8"))
    paired = pd.read_csv(MATERIAL_EVENTS)
    internal = paired[
        (paired.contrast == "TXM_ALIGNED_MINUS_T_ONLY") & (paired.metric == "ap")
    ].delta.to_numpy(float)
    summary = json.loads(MATERIAL_SUMMARY.read_text("utf-8"))
    records = pd.DataFrame(summary["contrasts"])
    record = records[
        (records.contrast == "TXM_ALIGNED_MINUS_T_ONLY")
        & (records.metric == "ap")
        & (records.scope == "all_events")
    ].iloc[0]
    external_events = pd.read_csv(EXTERNAL_EVENTS)
    external = external_events.tm_minus_t_pairwise_auc.to_numpy(float)
    external_summary = json.loads(EXTERNAL_SUMMARY.read_text("utf-8"))["primary_estimand"]
    return {
        "rgb": maps["rgb"],
        "mask": maps["mask"].astype(bool),
        "terrain_only": maps["terrain_only"],
        "aligned": maps["aligned_t_x_m"],
        "donor": maps["donor_material"],
        "ap": metadata["ap"],
        "sample_id": metadata["sample_id"],
        "selection_rule": metadata["selection_rule"],
        "internal": internal,
        "internal_estimate": float(record.mean_delta),
        "internal_ci": [float(record.ci95_low), float(record.ci95_high)],
        "external": external,
        "external_estimate": float(external_summary["tm_minus_t_pairwise_auc"]),
        "external_ci": [
            float(external_summary["block_bootstrap"]["ci95_low"]),
            float(external_summary["block_bootstrap"]["ci95_high"]),
        ],
    }


def load_trigger() -> dict[str, object]:
    events = pd.read_csv(TRIGGER_EVENTS)
    columns = [
        "d7_control_m56_mm", "d7_control_m28_mm", "d7_case_mm",
        "d7_control_p28_mm", "d7_control_p56_mm",
    ]
    values = events[columns].to_numpy(float)
    ranks = pd.DataFrame(values).rank(axis=1, method="average", pct=True).to_numpy(float)
    order = np.argsort(-events.trigger_dose_oof.to_numpy(float), kind="stable")
    summary = json.loads(TRIGGER_SUMMARY.read_text("utf-8"))["metrics"]
    same_season = pd.read_csv(TRIGGER_SAME_SEASON)
    return {
        "rank_heatmap": ranks[order],
        "dose": events.trigger_dose_oof.to_numpy(float),
        "estimate": float(summary["oof_matched_pairwise_auc"]),
        "positive": int(summary["oof_positive_dose_events"]),
        "events": int(summary["n_primary_events"]),
        "same_season_first": int(same_season.case_ranks_first.sum()),
    }


def draw_distribution(
    ax,
    values: np.ndarray,
    *,
    estimate: float,
    ci: tuple[float, float] | None,
    reference: float,
    color: str,
    pale: str,
    xlim: tuple[float, float],
    xlabel: str,
    annotation: str,
    seed: int,
    annotation_side: str = "left",
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    parts = ax.violinplot(values, positions=[0], vert=False, widths=0.55,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(pale)
        body.set_edgecolor(color)
        body.set_linewidth(0.6)
        body.set_alpha(0.72)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.16, 0.16, len(values))
    ax.scatter(values, jitter, s=8, color=color, alpha=0.72,
               edgecolor="white", linewidth=0.25, zorder=3)
    ax.axvline(reference, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    if ci is not None:
        ax.plot(ci, [0.36, 0.36], color=color, lw=1.25, zorder=4)
        ax.scatter([estimate], [0.36], s=27, marker="D", color=color,
                   edgecolor="white", linewidth=0.55, zorder=5)
    annotation_x = 0.98 if annotation_side == "right" else 0.02
    annotation_ha = "right" if annotation_side == "right" else "left"
    ax.text(annotation_x, 0.94, annotation, transform=ax.transAxes,
            ha=annotation_ha, va="top",
            fontsize=6.8, color=color, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.34, 0.48)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, labelpad=2)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(FRAME)


def draw_terrain_row(fig, spec, terrain: dict[str, object]) -> None:
    grid = spec.subgridspec(1, 2, width_ratios=[1, 1], wspace=0.065)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_dist = fig.add_subplot(grid[0, 1])
    ax_map.imshow(hillshade(terrain["dem"]), origin="upper")
    cmap = LinearSegmentedColormap.from_list(
        "terrain_score", ["#FFF3CF", "#F4B45F", "#D85A3A", "#7F1D1D"]
    )
    finite = np.isfinite(terrain["score_grid"])
    percentile_grid = np.full_like(terrain["score_grid"], np.nan, dtype=float)
    percentile_grid[finite] = pd.Series(
        terrain["score_grid"][finite]
    ).rank(method="average", pct=True).to_numpy()
    image = ax_map.imshow(percentile_grid, origin="upper", cmap=cmap,
                          vmin=0, vmax=1, alpha=0.80, interpolation="nearest")
    ax_map.contour(percentile_grid, levels=[0.75], colors="#7F1D1D",
                   linewidths=0.60, alpha=0.72)
    ax_map.scatter([terrain["case_col"]], [terrain["case_row"]], s=47, marker="*",
                   color="#F4C658", edgecolor="white", linewidth=0.85, zorder=5)
    clean_image_axis(ax_map)
    add_scale_bar(ax_map, 500 / 30, "500 m")
    ax_map.set_box_aspect(1)
    ax_map.set_anchor("W")
    cax = inset_axes(
        ax_map, width="3.4%", height="76%", loc="lower left",
        bbox_to_anchor=(1.035, 0.12, 1, 1), bbox_transform=ax_map.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.015, "high", transform=cbar.ax.transAxes,
                 ha="center", va="bottom", fontsize=7.0, color=MUTED)
    cbar.ax.text(0.5, -0.015, "low", transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=7.0, color=MUTED)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label(
        "OOF susceptibility", fontsize=6.8, fontweight="bold",
        labelpad=3.2, color=INK,
    )
    cbar.ax.yaxis.set_label_position("right")
    draw_distribution(
        ax_dist, terrain["events"], estimate=terrain["estimate"], ci=None,
        reference=0.5, color=TERRAIN, pale=TERRAIN_PALE, xlim=(0, 1),
        xlabel="event-level pairwise AUC", annotation="0.625  [0.540, 0.708]",
        seed=20260803, annotation_side="right",
    )


def draw_probability_tile(ax, values: np.ndarray, mask: np.ndarray, *, cmap, norm) -> None:
    ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
    ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=0.65)
    clean_image_axis(ax)


def draw_material_row(fig, spec, material: dict[str, object]) -> None:
    grid = spec.subgridspec(2, 1, height_ratios=[1.04, 0.96], hspace=0.075)
    plate = grid[0].subgridspec(1, 5, wspace=0.018)
    axes = [fig.add_subplot(plate[0, index]) for index in range(5)]
    axes[0].imshow(material["rgb"])
    clean_image_axis(axes[0])
    axes[1].imshow(material["rgb"])
    axes[1].imshow(np.ma.masked_where(~material["mask"], material["mask"]),
                   cmap=LinearSegmentedColormap.from_list("mask", ["#E85D55", "#E85D55"]),
                   alpha=0.62, interpolation="nearest")
    axes[1].contour(material["mask"].astype(float), levels=[0.5], colors="white", linewidths=0.65)
    clean_image_axis(axes[1])
    score_cmap = LinearSegmentedColormap.from_list(
        "material_probability", ["#13283B", "#2B708E", "#F0C85A", "#D85B42"]
    )
    norm = Normalize(vmin=0.15, vmax=0.85)
    draw_probability_tile(axes[2], material["terrain_only"], material["mask"], cmap=score_cmap, norm=norm)
    draw_probability_tile(axes[3], material["aligned"], material["mask"], cmap=score_cmap, norm=norm)
    draw_probability_tile(axes[4], material["donor"], material["mask"], cmap=score_cmap, norm=norm)
    titles = ["post-event", "reference", "T only", "aligned T×M", "donor M"]
    for axis, title in zip(axes, titles):
        image_label(axis, title, corner="upper left")
    for axis, key in zip(axes[2:], ("T_ONLY", "TXM_ALIGNED", "TXM_TEST_SHUFFLED_SAME_MODEL")):
        image_label(axis, f"AP {material['ap'][key]:.3f}", corner="lower right")

    dist = grid[1].subgridspec(1, 2, wspace=0.06)
    ax_internal = fig.add_subplot(dist[0, 0])
    ax_external = fig.add_subplot(dist[0, 1])
    draw_distribution(
        ax_internal, material["internal"], estimate=material["internal_estimate"],
        ci=tuple(material["internal_ci"]), reference=0, color=MATERIAL,
        pale="#F3C56B", xlim=(-0.04, 0.18), xlabel="event ΔAP",
        annotation="internal  +0.0072", seed=20260804, annotation_side="right",
    )
    draw_distribution(
        ax_external, material["external"], estimate=material["external_estimate"],
        ci=tuple(material["external_ci"]), reference=0, color="#2B708E",
        pale="#A9D3E2", xlim=(-0.8, 1.05), xlabel="external event ΔAUC",
        annotation="external  −0.0041", seed=20260805, annotation_side="right",
    )


def draw_trigger_row(fig, spec, trigger: dict[str, object]) -> None:
    grid = spec.subgridspec(1, 2, width_ratios=[1, 1], wspace=0.130)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_dose = fig.add_subplot(grid[0, 1])
    cmap = LinearSegmentedColormap.from_list(
        "rain_rank", ["#FFF5D8", "#FDB863", "#EF6548", "#B30000", "#5A0018"]
    )
    image = ax_heat.imshow(trigger["rank_heatmap"], aspect="auto", interpolation="nearest",
                           cmap=cmap, vmin=0.2, vmax=1.0)
    ax_heat.add_patch(Rectangle((1.5, -0.5), 1.0, trigger["events"], fill=False,
                                edgecolor=TRIGGER, linewidth=1.0))
    ax_heat.set_xticks(range(5), ["−56 d", "−28 d", "event", "+28 d", "+56 d"])
    ax_heat.set_yticks([])
    image_label(ax_heat, "high dose", corner="upper left")
    image_label(ax_heat, "low dose", corner="lower left")
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_color(FRAME)
        spine.set_linewidth(0.6)
    cax = inset_axes(
        ax_heat, width="3.4%", height="76%", loc="lower left",
        bbox_to_anchor=(1.018, 0.12, 1, 1), bbox_transform=ax_heat.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.015, "high", transform=cbar.ax.transAxes,
                 ha="center", va="bottom", fontsize=7.0, color=MUTED)
    cbar.ax.text(0.5, -0.015, "low", transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=7.0, color=MUTED)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label(
        "rainfall rank", fontsize=7.0, fontweight="bold",
        rotation=90, labelpad=5.0, color=INK,
    )
    cbar.ax.yaxis.set_label_position("right")
    draw_distribution(
        ax_dose, trigger["dose"], estimate=float(np.mean(trigger["dose"])), ci=None,
        reference=0, color=TRIGGER, pale=TRIGGER_PALE, xlim=(-0.05, 1.45),
        xlabel="OOF trigger dose", annotation="108/138 > 0   AUC 0.722",
        seed=20260806, annotation_side="right",
    )


def write_source_data(outdir: Path, terrain, material, trigger) -> None:
    rows = [
        {"panel": "a", "prior": "Terrain", "metric": "event-balanced pairwise AUC",
         "estimate": terrain["estimate"], "ci95_low": terrain["ci"][0],
         "ci95_high": terrain["ci"][1], "n": len(terrain["events"])},
        {"panel": "b", "prior": "Material", "metric": "internal event delta AP",
         "estimate": material["internal_estimate"], "ci95_low": material["internal_ci"][0],
         "ci95_high": material["internal_ci"][1], "n": len(material["internal"])},
        {"panel": "b", "prior": "Material", "metric": "external event delta AUC",
         "estimate": material["external_estimate"], "ci95_low": material["external_ci"][0],
         "ci95_high": material["external_ci"][1], "n": len(material["external"])},
        {"panel": "c", "prior": "Trigger", "metric": "OOF matched pairwise AUC",
         "estimate": trigger["estimate"], "ci95_low": np.nan,
         "ci95_high": np.nan, "n": trigger["events"]},
    ]
    pd.DataFrame(rows).to_csv(outdir / "figure6_source_data.csv", index=False)
    selection = {
        "terrain": {"event_id": terrain["event_id"], "rule": terrain["selection_rule"]},
        "material": {"sample_id": material["sample_id"], "rule": material["selection_rule"]},
        "trigger": {"visualization": "all 138 events sorted by frozen OOF trigger dose"},
    }
    (outdir / "figure6_case_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )


def render(outdir: Path) -> Path:
    configure_style()
    outdir.mkdir(parents=True, exist_ok=True)
    terrain = load_terrain()
    material = load_material()
    trigger = load_trigger()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=600)
    outer = fig.add_gridspec(
        5, 1, height_ratios=[2.15, 0.14, 2.05, 0.12, 1.55],
        left=0.065, right=0.985, top=0.988, bottom=0.034, hspace=0.030,
    )
    sections = [
        outer[0].subgridspec(2, 1, height_ratios=[0.12, 0.88], hspace=0.08),
        outer[2].subgridspec(2, 1, height_ratios=[0.075, 0.925], hspace=0.06),
        outer[4].subgridspec(2, 1, height_ratios=[0.12, 0.88], hspace=0.08),
    ]
    title_a = fig.add_subplot(sections[0][0])
    title_b = fig.add_subplot(sections[1][0])
    title_c = fig.add_subplot(sections[2][0])
    panel_title(title_a, "a", "Terrain: spatial susceptibility")
    panel_title(title_b, "b", "Material: terrain interaction")
    panel_title(title_c, "c", "Trigger: event timing")
    draw_terrain_row(fig, sections[0][1], terrain)
    draw_material_row(fig, sections[1][1], material)
    draw_trigger_row(fig, sections[2][1], trigger)

    output = outdir / "figure6_revision_native_task_evidence.png"
    fig.savefig(output, dpi=600, facecolor=WHITE, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    write_source_data(outdir, terrain, material, trigger)
    report = {
        "status": "complete",
        "output": str(output),
        "output_sha256": sha256(output),
        "dimensions_px": list(mpl.image.imread(output).shape[:2][::-1]),
        "backend": "Python/matplotlib",
        "material_map_verification": json.loads(MATERIAL_MAPS_META.read_text("utf-8"))["ap"],
        "rendered_at_unix": time.time(),
        "source_hashes": {
            str(path): sha256(path)
            for path in (
                TERRAIN_SUMMARY, TERRAIN_EVENTS, TERRAIN_SCORES,
                MATERIAL_SUMMARY, MATERIAL_EVENTS, MATERIAL_MAPS, MATERIAL_MAPS_META,
                EXTERNAL_SUMMARY, EXTERNAL_EVENTS,
                TRIGGER_SUMMARY, TRIGGER_EVENTS, TRIGGER_SAME_SEASON,
            )
        },
    }
    (outdir / "figure6_render_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(render(parse_args().outdir.resolve()))
