#!/usr/bin/env python3
"""Render the standalone Figure 4(a) cross-domain visual-error examples.

Each case shows a continuous 2.56 km same-scene context around the frozen
1.28 km OOF evaluation footprint. Available predictions, inventory contours,
and candidate bodies are georeferenced into that wider view without drawing a
window boundary. Selection uses only the frozen visual
baseline: eligible near-pure false-positive geometry, image readability, and
complementary confusion contexts. Adapter outcomes are not read.

The DLR and GDCLD contexts come from the exact source imagery. Sen12 and
GLaD4CD use same-acquisition Sentinel-2 L2A context. The script writes PNG only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from affine import Affine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import render_figure4_visual_error_structure as base


CANVAS_WIDTH_MM = 196.0
CANVAS_HEIGHT_MM = 62.0
DEFAULT_OUTDIR = base.DEFAULT_OUTDIR / "panels" / "figure4"
# Fixed print-size hierarchy for all standalone Figure 4 panels.
PANEL_LETTER_PT = 9.5
PANEL_TITLE_PT = 8.8
SOURCE_BADGE_PT = 8.2
CONTEXT_LABEL_PT = 7.8
LEGEND_PT = 7.5
CONTEXT_SIZE = 256
OOF_SIZE = 128
OOF_OFFSET = (CONTEXT_SIZE - OOF_SIZE) // 2
CONTEXT_GSD_M = 10.0
CONTEXT_CACHE_DIR = base.EXPERIMENTS / "figure4a_wide_context_v1"
CONTEXT_CACHE_PATH = CONTEXT_CACHE_DIR / "wide_contexts.npz"
CONTEXT_RECEIPT_PATH = CONTEXT_CACHE_DIR / "receipt.json"

WINDOW_REGISTRY = (
    base.PROJECT_ROOT / "processed/hybrid_pinn/pild_core_geo_v2_raw/window_registry_v2.csv"
)
DLR_GEOMETRY = (
    base.PROJECT_ROOT
    / "processed/hybrid_pinn/dlr_strict_t3_reference_subset_v1/patch_geometry_manifest_v2_indexfixed.csv"
)
DLR_SUBSET = (
    base.PROJECT_ROOT / "processed/hybrid_pinn/dlr_strict_t3_reference_subset_v1"
)
SEN12_REGISTRY = (
    base.PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
)

PC_STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SIGN_ENDPOINT = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
SEN12_CONTEXT_ITEM = (
    "S2A_MSIL2A_20181110T060051_R091_T43TCE_20201008T172457"
)
GLAD_CONTEXT_ITEM = (
    "S2B_MSIL2A_20230224T081829_R121_T37SBB_20240819T103326"
)

# Source coverage is fixed first. Within each source, these baseline-only OOF
# examples provide complementary and visually legible failure contexts.
PANEL_A_EXAMPLES = [
    {
        "dataset_id": "DLR_Landslide_Ref_2025",
        "sample_id": "EID_KG0002__SID_00518",
        "context": "bare-surface confusion",
    },
    {
        "dataset_id": "GDCLD",
        "sample_id": "GDCLD_Palu::Palu/Palu_1.tif::g10_r256c768",
        "context": "vegetated-slope confusion",
    },
    {
        "dataset_id": "SEN12LS_HARMONIZED",
        "sample_id": "SEN12_S2_kyrgyzstan2_6193",
        "context": "bright-surface confusion",
    },
    {
        "dataset_id": "GLaD4CD_v1",
        "sample_id": "GLADV1_0172::test_171::g10_r100c57",
        "context": "cultivated-mosaic confusion",
    },
]


def parse_affine(text: str) -> Affine:
    values = [float(value) for value in str(text).split(",")]
    if len(values) != 6:
        raise ValueError(f"Expected six affine coefficients, got {text!r}")
    return Affine(*values)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")
    os.replace(temporary, path)


def request_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def signed_asset(session: requests.Session, href: str) -> str:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            response = session.get(
                PC_SIGN_ENDPOINT,
                params={"href": href},
                timeout=90,
            )
            response.raise_for_status()
            return str(response.json()["href"])
        except Exception as exc:  # network retries are intentionally bounded
            last_error = exc
            if attempt == 7:
                break
            time.sleep(2.0 + attempt)
    raise RuntimeError(f"Could not sign Planetary Computer asset: {last_error}")


def fetch_sentinel2_context(
    *,
    item_id: str,
    target_crs: str,
    target_transform: Affine,
) -> tuple[np.ndarray, dict[str, str]]:
    session = request_session()
    item_url = f"{PC_STAC_ROOT}/collections/sentinel-2-l2a/items/{item_id}"
    response = session.get(item_url, timeout=90)
    response.raise_for_status()
    item = response.json()
    context = np.full((3, CONTEXT_SIZE, CONTEXT_SIZE), np.nan, dtype=np.float32)
    for channel, asset_name in enumerate(("B04", "B03", "B02")):
        href = signed_asset(session, item["assets"][asset_name]["href"])
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        ):
            with rasterio.open(href) as source:
                reproject(
                    source=rasterio.band(source, 1),
                    destination=context[channel],
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
    context = np.clip(context / 10_000.0, 0.0, 1.0)
    if np.isfinite(context).all(axis=0).mean() < 0.99:
        raise RuntimeError(f"Incomplete Sentinel-2 context for {item_id}")
    provenance = {
        "item_id": item_id,
        "datetime": str(item.get("properties", {}).get("datetime", "")),
        "collection": str(item.get("collection", "sentinel-2-l2a")),
    }
    return context, provenance


def build_dlr_context(sample_id: str) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "DLR wide-context cache requires h5py; run this renderer in the dpl environment"
        ) from exc

    geometry = pd.read_csv(DLR_GEOMETRY)
    target_rows = geometry[geometry.sample_id == sample_id]
    if len(target_rows) != 1:
        raise RuntimeError(f"DLR geometry lookup failed for {sample_id}")
    target = target_rows.iloc[0]
    event_rows = geometry[geometry.event_uid == target.event_uid]
    origin_top = int(target.top_row) - OOF_OFFSET
    origin_left = int(target.left_col) - OOF_OFFSET
    accumulated = np.zeros((3, CONTEXT_SIZE, CONTEXT_SIZE), dtype=np.float64)
    coverage = np.zeros((CONTEXT_SIZE, CONTEXT_SIZE), dtype=np.float64)

    for split, rows in event_rows.groupby("split"):
        source_path = DLR_SUBSET / f"{split}_n3_s1s2.h5"
        with h5py.File(source_path, "r") as source:
            for row in rows.itertuples(index=False):
                y0 = max(int(row.top_row) - origin_top, 0)
                x0 = max(int(row.left_col) - origin_left, 0)
                y1 = min(int(row.top_row) + OOF_SIZE - origin_top, CONTEXT_SIZE)
                x1 = min(int(row.left_col) + OOF_SIZE - origin_left, CONTEXT_SIZE)
                if y1 <= y0 or x1 <= x0:
                    continue
                source_y = max(origin_top - int(row.top_row), 0)
                source_x = max(origin_left - int(row.left_col), 0)
                height = y1 - y0
                width = x1 - x0
                index = int(row.sample_index)
                patch = np.stack(
                    [
                        source["POST1_B04"][index, 0],
                        source["POST1_B03"][index, 0],
                        source["POST1_B02"][index, 0],
                    ]
                ).astype(np.float32)
                accumulated[:, y0:y1, x0:x1] += patch[
                    :, source_y : source_y + height, source_x : source_x + width
                ]
                coverage[y0:y1, x0:x1] += 1.0
    if not (coverage > 0).all():
        missing = int((coverage == 0).sum())
        raise RuntimeError(f"DLR context has {missing} uncovered pixels for {sample_id}")
    return np.clip(
        accumulated / coverage[None, ...] / 10_000.0,
        0.0,
        1.0,
    ).astype(np.float32)


def build_gdcld_context(sample_id: str) -> np.ndarray:
    registry = pd.read_csv(WINDOW_REGISTRY)
    rows = registry[registry.sample_id == sample_id]
    if len(rows) != 1:
        raise RuntimeError(f"Window-registry lookup failed for {sample_id}")
    row = rows.iloc[0]
    core_transform = parse_affine(row.target_transform)
    context_transform = core_transform * Affine.translation(-OOF_OFFSET, -OOF_OFFSET)
    context = np.full((3, CONTEXT_SIZE, CONTEXT_SIZE), np.nan, dtype=np.float32)
    with rasterio.open(row.source_image_path) as source:
        for channel, band in enumerate((1, 2, 3)):
            reproject(
                source=rasterio.band(source, band),
                destination=context[channel],
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=context_transform,
                dst_crs=row.target_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    if np.isfinite(context).all(axis=0).mean() < 0.99:
        raise RuntimeError(f"Incomplete GDCLD context for {sample_id}")
    return np.clip(context / 255.0, 0.0, 1.0)


def build_external_context(
    sample_id: str,
    item_id: str,
) -> tuple[np.ndarray, dict[str, str]]:
    if sample_id.startswith("SEN12_"):
        registry = pd.read_csv(SEN12_REGISTRY)
        rows = registry[registry.sample_id == sample_id]
        if len(rows) != 1:
            raise RuntimeError(f"Sen12 registry lookup failed for {sample_id}")
        row = rows.iloc[0]
        core_transform = Affine(
            CONTEXT_GSD_M,
            0.0,
            float(row.min_x),
            0.0,
            -CONTEXT_GSD_M,
            float(row.max_y),
        )
        target_crs = str(row.crs)
    else:
        registry = pd.read_csv(WINDOW_REGISTRY)
        rows = registry[registry.sample_id == sample_id]
        if len(rows) != 1:
            raise RuntimeError(f"Window-registry lookup failed for {sample_id}")
        row = rows.iloc[0]
        core_transform = parse_affine(row.target_transform)
        target_crs = str(row.target_crs)
    context_transform = core_transform * Affine.translation(-OOF_OFFSET, -OOF_OFFSET)
    return fetch_sentinel2_context(
        item_id=item_id,
        target_crs=target_crs,
        target_transform=context_transform,
    )


def build_context_cache() -> None:
    contexts: list[np.ndarray] = []
    replace_center: list[bool] = []
    provenance: dict[str, dict] = {}
    for item in PANEL_A_EXAMPLES:
        sample_id = item["sample_id"]
        dataset_id = item["dataset_id"]
        if dataset_id == "DLR_Landslide_Ref_2025":
            context = build_dlr_context(sample_id)
            source_info = {"source": "exact overlapping DLR source windows"}
            replace = False
        elif dataset_id == "GDCLD":
            context = build_gdcld_context(sample_id)
            source_info = {"source": "exact GDCLD source scene"}
            replace = False
        elif dataset_id == "SEN12LS_HARMONIZED":
            context, source_info = build_external_context(sample_id, SEN12_CONTEXT_ITEM)
            source_info["source"] = "same-acquisition Sentinel-2 L2A context"
            replace = True
        elif dataset_id == "GLaD4CD_v1":
            context, source_info = build_external_context(sample_id, GLAD_CONTEXT_ITEM)
            source_info["source"] = "same-acquisition Sentinel-2 L2A context"
            replace = True
        else:
            raise RuntimeError(f"Unsupported source {dataset_id}")
        contexts.append(context.astype(np.float16))
        replace_center.append(replace)
        provenance[sample_id] = source_info

    CONTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CONTEXT_CACHE_PATH.with_name(
        f".{CONTEXT_CACHE_PATH.name}.tmp-{os.getpid()}.npz"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            sample_id=np.asarray(
                [item["sample_id"] for item in PANEL_A_EXAMPLES], dtype="U160"
            ),
            rgb=np.stack(contexts),
            replace_center=np.asarray(replace_center, dtype=np.uint8),
            gsd_m=np.full(len(contexts), CONTEXT_GSD_M, dtype=np.float32),
            oof_top=np.full(len(contexts), OOF_OFFSET, dtype=np.int16),
            oof_left=np.full(len(contexts), OOF_OFFSET, dtype=np.int16),
        )
    os.replace(temporary, CONTEXT_CACHE_PATH)
    atomic_write_json(
        CONTEXT_RECEIPT_PATH,
        {
            "context_cache": str(CONTEXT_CACHE_PATH),
            "context_size_px": CONTEXT_SIZE,
            "context_gsd_m": CONTEXT_GSD_M,
            "context_footprint_m": CONTEXT_SIZE * CONTEXT_GSD_M,
            "oof_window_px": OOF_SIZE,
            "oof_window_footprint_m": OOF_SIZE * CONTEXT_GSD_M,
            "predictions_outside_oof_window": False,
            "selection_uses_adapter_outcome": False,
            "provenance": provenance,
        },
    )


def load_contexts(rebuild: bool = False) -> dict[str, dict]:
    if rebuild or not CONTEXT_CACHE_PATH.is_file():
        build_context_cache()
    with np.load(CONTEXT_CACHE_PATH, allow_pickle=False) as cache:
        contexts = {}
        for index, sample_id in enumerate(cache["sample_id"]):
            contexts[str(sample_id)] = {
                "rgb": cache["rgb"][index].astype(np.float32),
                "replace_center": bool(cache["replace_center"][index]),
                "gsd_m": float(cache["gsd_m"][index]),
                "oof_top": int(cache["oof_top"][index]),
                "oof_left": int(cache["oof_left"][index]),
            }
    return contexts


def stretch_context(rgb_chw: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(np.asarray(rgb_chw, dtype=np.float32), 0, -1).copy()
    valid = np.isfinite(rgb).all(axis=-1)
    if valid.mean() < 0.99:
        raise RuntimeError("Wide context contains non-finite pixels")
    for channel in range(3):
        low, high = np.percentile(rgb[..., channel][valid], [2, 98])
        rgb[..., channel] = np.clip(
            (rgb[..., channel] - low) / max(high - low, 1e-6),
            0.0,
            1.0,
        )
    return np.power(rgb, 0.85)


def draw_wide_case(
    ax,
    tile: dict,
    context: dict,
    source_label: str,
    confusion_label: str,
) -> None:
    rgb = stretch_context(context["rgb"])
    top = int(context["oof_top"])
    left = int(context["oof_left"])

    target = np.zeros((CONTEXT_SIZE, CONTEXT_SIZE), dtype=bool)
    predicted = np.zeros_like(target)
    near_pure = np.zeros_like(target)
    window = (slice(top, top + OOF_SIZE), slice(left, left + OOF_SIZE))
    target[window] = np.asarray(tile["target"], dtype=bool)
    predicted[window] = np.asarray(tile["predicted"], dtype=bool)
    near_pure[window] = np.asarray(tile["near_pure"], dtype=bool)

    overlay = np.zeros((CONTEXT_SIZE, CONTEXT_SIZE, 4), dtype=float)
    overlay[predicted & target] = base.mpl.colors.to_rgba(base.TP_COLOR, 0.46)
    overlay[predicted & ~target] = base.mpl.colors.to_rgba(base.FP_COLOR, 0.46)
    overlay[~predicted & target] = base.mpl.colors.to_rgba(base.FN_COLOR, 0.58)

    ax.imshow(rgb)
    ax.imshow(overlay)
    if target.any():
        ax.contour(target.astype(float), levels=[0.5], colors="white", linewidths=0.95)
        dashed = ax.contour(
            target.astype(float),
            levels=[0.5],
            colors=base.INK,
            linewidths=0.42,
            alpha=0.76,
        )
        dashed.set_dashes([(0.0, (2.6, 1.8))])
    if near_pure.any():
        ax.contour(
            near_pure.astype(float), levels=[0.5], colors="white", linewidths=2.0
        )
        ax.contour(
            near_pure.astype(float),
            levels=[0.5],
            colors=base.FP_COLOR,
            linewidths=1.25,
        )

    ax.text(
        0.035,
        0.965,
        source_label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=SOURCE_BADGE_PT,
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": base.INK,
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )
    ax.text(
        0.5,
        -0.018,
        confusion_label,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=CONTEXT_LABEL_PT,
        color="#4B5563",
    )

    bar_px = 1000.0 / float(context["gsd_m"])
    x0, y0 = 0.055 * CONTEXT_SIZE, 0.945 * CONTEXT_SIZE
    ax.add_patch(
        Rectangle(
            (x0 - 3.0, y0 - 20.0),
            bar_px + 6.0,
            24.0,
            facecolor=base.INK,
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
        y0 - 7.5,
        "1 km",
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
        spine.set_color(base.GRAY)
        spine.set_linewidth(0.8)


def render(outdir: Path, dpi: int, rebuild_context_cache: bool = False) -> Path:
    base.configure_style()
    decisions = pd.read_parquet(base.DECISIONS_PATH)
    selected_ids = {item["sample_id"] for item in PANEL_A_EXAMPLES}
    bodies = decisions[
        (decisions.area_px >= base.RULE["candidate_area_px_min"])
        & (decisions.purity <= base.RULE["candidate_purity_max"])
    ]
    sample_tp = (
        decisions.groupby(["dataset_id", "sample_id"], as_index=False)
        .intersection_px.sum()
        .rename(columns={"intersection_px": "sample_tp"})
    )
    tiles = (
        bodies.groupby(["dataset_id", "sample_id"], as_index=False)
        .agg(
            near_pure_fp=("false_px", "sum"),
            near_pure_bodies=("component_id", "size"),
            largest_near_pure_area=("area_px", "max"),
        )
        .merge(sample_tp, on=["dataset_id", "sample_id"], how="inner")
    )
    tiles = tiles[tiles.sample_id.isin(selected_ids)].reset_index(drop=True)
    candidates = base.load_candidates(tiles, base.build_fold_index())
    contexts = load_contexts(rebuild_context_cache)

    missing = [
        item["sample_id"]
        for item in PANEL_A_EXAMPLES
        if item["sample_id"] not in candidates
    ]
    if missing:
        raise RuntimeError(f"Frozen Figure 4(a) examples missing from OOF cache: {missing}")
    missing_context = [
        item["sample_id"]
        for item in PANEL_A_EXAMPLES
        if item["sample_id"] not in contexts
    ]
    if missing_context:
        raise RuntimeError(f"Figure 4(a) wide contexts missing: {missing_context}")

    fig = plt.figure(
        figsize=(CANVAS_WIDTH_MM / 25.4, CANVAS_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        1,
        4,
        left=0.0033,
        right=0.9870,
        top=0.870,
        bottom=0.160,
        wspace=0.025,
    )

    axes = []
    for index, item in enumerate(PANEL_A_EXAMPLES):
        ax = fig.add_subplot(grid[0, index])
        ax.set_anchor("N")
        draw_wide_case(
            ax,
            candidates[item["sample_id"]],
            contexts[item["sample_id"]],
            base.SOURCE_LABELS[item["dataset_id"]],
            item["context"],
        )
        axes.append(ax)

    fig.text(
        0.0033,
        0.955,
        "(a)",
        fontsize=PANEL_LETTER_PT,
        fontweight="bold",
        ha="left",
        va="top",
        color=base.INK,
    )
    fig.text(
        0.0366,
        0.955,
        "Cross-domain visual errors form coherent bodies",
        fontsize=PANEL_TITLE_PT,
        fontweight="semibold",
        ha="left",
        va="top",
        color=base.INK,
    )

    fig.legend(
        handles=[
            Patch(facecolor=base.TP_COLOR, alpha=0.46, label="TP"),
            Patch(facecolor=base.FP_COLOR, alpha=0.46, label="FP"),
            Patch(facecolor=base.FN_COLOR, alpha=0.58, label="FN"),
            Line2D(
                [0],
                [0],
                color=base.INK,
                lw=0.8,
                linestyle=(0, (2.6, 1.8)),
                label="reference inventory",
            ),
            Line2D(
                [0],
                [0],
                color=base.FP_COLOR,
                lw=1.6,
                label="near-pure FP body",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.51, 0.025),
        ncol=5,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.30,
        handletextpad=0.55,
        borderaxespad=0.0,
        fontsize=LEGEND_PT,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "figure4a_cross_domain_visual_errors.png"
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    tolerance_in = 0.02
    if (
        tight.x0 < -tolerance_in
        or tight.y0 < -tolerance_in
        or tight.x1 > fig.get_figwidth() + tolerance_in
        or tight.y1 > fig.get_figheight() + tolerance_in
    ):
        raise RuntimeError(f"Figure 4(a) artist extends beyond canvas: {tight}")
    fig.savefig(output, dpi=dpi, facecolor="white", format="png")
    plt.close(fig)

    expected_size = (
        round(CANVAS_WIDTH_MM / 25.4 * dpi),
        round(CANVAS_HEIGHT_MM / 25.4 * dpi),
    )
    with Image.open(output) as rendered:
        if any(
            abs(actual - expected) > 1
            for actual, expected in zip(rendered.size, expected_size)
        ):
            raise RuntimeError(
                f"Unexpected Figure 4(a) size: {rendered.size} != {expected_size}"
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--rebuild-context-cache", action="store_true")
    args = parser.parse_args()
    print(render(args.outdir, args.dpi, args.rebuild_context_cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
