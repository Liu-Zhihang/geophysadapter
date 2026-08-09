#!/usr/bin/env python3
"""Render Figure 8: before/after gallery of object-scale physical review.

The gallery exists to show what the correction looks like on the map, so it must
not be assembled from the best-looking outcomes. Six samples are drawn from six
pre-registered strata that span the full behavioural repertoire of the analytic
criterion - clean clearance, clearance with collateral loss, net harm, and
abstention - and within every stratum the sample nearest the stratum median is
taken. The realised allocation and the population proportions are both recorded
so a reader can see the gallery is close to proportional rather than curated.

Post-review state is reconstructed exactly: `component_id` in the frozen
decision table is the SciPy label of the baseline prediction, verified here by
re-deriving area, true-positive and false-positive counts for every body.

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
from matplotlib.patches import Patch
from PIL import Image
from scipy import ndimage


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXP = PROJECT_ROOT / "experiments/revision2026"
DEFAULT_OUTDIR = (
    PROJECT_ROOT.parent
    / "docs/assets"
)

DECISIONS_PATH = EXP / "pild_object_veto_final_v1/component_decisions.parquet"
SUMMARY_PATH = EXP / "pild_object_veto_final_v1/summary.json"
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

SOURCE_LABELS = {
    "DLR_Landslide_Ref_2025": "DLR",
    "GDCLD": "GDCLD",
    "SEN12LS_HARMONIZED": "Sen12",
    "GLaD4CD_v1": "GLaD4CD",
}

# --- Frozen selection rule ---------------------------------------------------
# Every stratum is defined before rendering. Within a stratum the sample nearest
# the stratum median of the governing quantity is taken, ties broken by
# sample_id. No stratum is defined by "largest gain" or "best appearance".
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
        "key": "clean_clearance_sen12",
        "label": "clean clearance",
        "source": "SEN12LS_HARMONIZED",
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
    grouped = frame.groupby(["dataset_id", "sample_id"], as_index=False).apply(
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


def choose_gallery(outcomes: pd.DataFrame) -> pd.DataFrame:
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


def draw_state(
    ax,
    tile: dict,
    prediction: np.ndarray,
    *,
    show_removed: bool,
    scale_bar: bool,
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

    if scale_bar:
        height, width = target.shape
        bar = 500.0 / ANALYSIS_GSD_M
        x0, y0 = 0.06 * width, 0.94 * height
        ax.plot([x0, x0 + bar], [y0, y0], color="white", linewidth=2.6, solid_capstyle="butt")
        ax.plot([x0, x0 + bar], [y0, y0], color=INK, linewidth=1.1, solid_capstyle="butt")
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

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.7)


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


def render(gallery: pd.DataFrame, tiles: dict[str, dict], outdir: Path, dpi: int) -> tuple[Path, dict]:
    fig = plt.figure(figsize=(7.48, 6.55), facecolor="white")
    # Three rows of before/after pairs. The wide gap separates pairs; the narrow
    # gap sits inside a pair so the two states read as one comparison.
    grid = fig.add_gridspec(
        3,
        4,
        left=0.052,
        right=0.988,
        top=0.885,
        bottom=0.088,
        wspace=0.055,
        hspace=0.30,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
    )
    pair_columns = [(0, 1), (2, 3)]

    for index, row in gallery.iterrows():
        grid_row, pair = divmod(int(index), 2)
        left_column, right_column = pair_columns[pair]
        tile = tiles[str(row.sample_id)]

        ax_before = fig.add_subplot(grid[grid_row, left_column])
        ax_before.set_anchor("N")
        draw_state(ax_before, tile, tile["baseline"], show_removed=False, scale_bar=True)

        ax_after = fig.add_subplot(grid[grid_row, right_column])
        ax_after.set_anchor("N")
        draw_state(ax_after, tile, tile["reviewed"], show_removed=True, scale_bar=False)

        ax_before.text(
            0.035,
            0.965,
            SOURCE_LABELS[str(row.dataset_id)],
            transform=ax_before.transAxes,
            va="top",
            ha="left",
            fontsize=7.0,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": INK,
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )
        for ax, state in ((ax_before, "baseline"), (ax_after, "after review")):
            ax.text(
                0.5,
                1.015,
                state,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=MUTED,
            )
        if int(row.removed_bodies) == 0:
            ledger = "no body met the removal criterion; prediction unchanged"
        else:
            ledger = (
                f"cleared {int(row.fp_cleared):,} FP  \u00b7  lost {int(row.tp_lost):,} TP"
                f"  \u00b7  net {int(row.net):+,}"
            )
        caption = f"({'abcdef'[int(index)]}) {row.stratum_label}\n{ledger}"
        ax_before.text(
            0.0,
            -0.055,
            caption,
            transform=ax_before.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=INK,
            linespacing=1.4,
        )

    fig.legend(
        handles=[
            Patch(facecolor=TP_COLOR, alpha=0.60, label="true positive"),
            Patch(facecolor=FP_COLOR, alpha=0.62, label="false positive"),
            Patch(facecolor=FN_COLOR, alpha=0.62, label="false negative"),
            Line2D([0], [0], color=INK, lw=0.6, linestyle=(0, (2.4, 1.6)), label="reference inventory"),
            Line2D([0], [0], color=REMOVED_COLOR, lw=1.1, label="body removed by the analytic criterion"),
        ],
        loc="center",
        bbox_to_anchor=(0.52, 0.938),
        ncol=5,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.5,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "figure8_revision_object_veto_gallery.png"
    fig.savefig(output, dpi=dpi, facecolor="white")
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

    gallery = choose_gallery(outcomes)
    fold_index = build_fold_index()
    tiles = load_tiles(gallery, decisions, fold_index)
    output, edges = render(gallery, tiles, args.outdir, args.dpi)

    realised = {
        "net_positive_cells": int((gallery.net > 0).sum()),
        "net_zero_cells": int((gallery.net == 0).sum()),
        "net_negative_cells": int((gallery.net < 0).sum()),
    }
    source_data = args.outdir / "figure8_object_veto_gallery_source_data.csv"
    gallery.assign(
        source=gallery.dataset_id.map(SOURCE_LABELS),
    )[
        [
            "stratum",
            "stratum_label",
            "stratum_size",
            "stratum_median",
            "governing",
            "source",
            "sample_id",
            "removed_bodies",
            "kept_bodies",
            "fp_cleared",
            "tp_lost",
            "fp_kept",
            "tp_kept",
            "net",
        ]
    ].to_csv(source_data, index=False)

    report = {
        "schema_version": "figure8_object_veto_gallery.v1",
        "backend": "Python/Matplotlib",
        "output_policy": "PNG only",
        "canvas_mm": [190.0, 166.4],
        "dpi_requested": args.dpi,
        "render_script": str(SCRIPT_PATH),
        "render_script_sha256": sha256(SCRIPT_PATH),
        "source_files": {
            str(DECISIONS_PATH.relative_to(PROJECT_ROOT)): sha256(DECISIONS_PATH),
            str(SUMMARY_PATH.relative_to(PROJECT_ROOT)): sha256(SUMMARY_PATH),
        },
        "selection_rule": {
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
        "selected_samples": gallery.to_dict(orient="records"),
        "edge_self_check": edges,
        "source_data": str(source_data),
        "output": str(output),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = args.outdir / "figure8_object_veto_gallery_report.json"
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
                    ["stratum", "sample_id", "fp_cleared", "tp_lost", "net"]
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
