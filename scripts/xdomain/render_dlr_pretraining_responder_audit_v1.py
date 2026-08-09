#!/usr/bin/env python3
"""Render raw optical/Terrain evidence for post-hoc DLR responder samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_RUNS = Path(
    "experiments/revision2026/"
    "dlr_geo4qc_sen12_exactcommon9_scratch_seed20260724_v4"
)
DEFAULT_CACHE = Path(
    "processed/hybrid_pinn/dlr_geo4qc_sen12_exactcommon9_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def stretch_pair(optical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Stored channel order is B02/B03/B04/B8A/B11/B12. Display RGB is B04/B03/B02.
    rgb = optical[[2, 1, 0]][:, [0, 3]].astype(np.float32) / 10000.0
    values = np.transpose(rgb, (1, 2, 3, 0))
    lo = np.percentile(values, 2, axis=(0, 1, 2))
    hi = np.percentile(values, 98, axis=(0, 1, 2))
    values = np.clip((values - lo) / np.maximum(hi - lo, 1e-5), 0.0, 1.0)
    return values[0] ** 0.9, values[1] ** 0.9


def boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, iterations=1)
    return mask & ~eroded


def label_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    output[mask] = 0.35 * output[mask] + 0.65 * np.array([1.0, 0.45, 0.08])
    output[boundary(mask)] = np.array([1.0, 0.95, 0.1])
    return np.clip(output, 0.0, 1.0)


def robust_gray(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    finite = value[valid & np.isfinite(value)]
    if not len(finite):
        return np.zeros_like(value, dtype=float)
    lo, hi = np.percentile(finite, (2, 98))
    return np.clip((value - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def draw_row(
    axes: np.ndarray,
    row: pd.Series,
    optical: np.ndarray,
    terrain: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> None:
    first, last = stretch_pair(optical)
    change = np.mean(np.abs(last - first), axis=2)
    slope = robust_gray(terrain[1], valid)
    relief = robust_gray(terrain[8], valid)
    panels = (
        (first, "Earliest S2 RGB", None),
        (last, "Latest S2 RGB", None),
        (change, "Absolute RGB change", "magma"),
        (slope, "Slope (display stretch)", "viridis"),
        (relief, "Local relief (display stretch)", "terrain"),
        (label_overlay(last, mask), "Reference overlay", None),
    )
    for axis, (image, title, cmap) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    axes[0].set_ylabel(
        (
            f"{row.sample_id}\n"
            f"fold {int(row.fold)} | RER {row.rer:+.1%}\n"
            f"DeltaIoU {row.delta_iou:+.3f}"
        ),
        fontsize=7,
        rotation=0,
        ha="right",
        va="center",
        labelpad=54,
    )


def main() -> int:
    args = parse_args()
    source = (
        args.runs_dir
        / "responder_profile_analysis_v1/posthoc_oracle_sample_rer_ge_10pct.csv"
    )
    rows = pd.read_csv(source).sort_values(
        ["event_id", "rer"], ascending=[True, False]
    )
    outdir = args.outdir or args.runs_dir / "responder_visual_audit_v1"
    individual_dir = outdir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(
        args.cache_dir / "dlr_base_temporalvalid_p128.h5", "r"
    ) as base, h5py.File(
        args.cache_dir / "dlr_prithvi_4t6b_p128.h5", "r"
    ) as optical_h5, h5py.File(
        args.cache_dir / "dlr_common_terrain9_p128.h5", "r"
    ) as terrain_h5:
        ids = decode(base["sample_id"][:])
        index = {sample_id: position for position, sample_id in enumerate(ids)}
        audit_rows = []
        rendered = []
        for row in rows.itertuples(index=False):
            position = index[row.sample_id]
            optical = np.asarray(optical_h5["optical"][position])
            terrain = np.asarray(terrain_h5["terrain"][position])
            mask = np.asarray(base["mask"][position, 0], dtype=bool)
            valid = np.asarray(base["valid_mask"][position, 0], dtype=bool)
            components, n_components = ndimage.label(mask & valid)
            component_areas = np.bincount(components.ravel())[1:]
            first, last = stretch_pair(optical)
            audit_rows.append(
                {
                    "sample_id": row.sample_id,
                    "event_id": row.event_id,
                    "fold": int(row.fold),
                    "rer": float(row.rer),
                    "delta_iou": float(row.delta_iou),
                    "mask_fraction": float((mask & valid).sum() / max(valid.sum(), 1)),
                    "mask_components": int(n_components),
                    "largest_component_fraction": (
                        float(component_areas.max() / max((mask & valid).sum(), 1))
                        if len(component_areas)
                        else 0.0
                    ),
                    "earliest_latest_rgb_abs_change": float(
                        np.abs(last - first)[valid].mean()
                    ),
                    "slope_mean_deg": float(terrain[1][valid].mean()),
                    "slope_std_deg": float(terrain[1][valid].std()),
                    "local_relief_mean_m": float(terrain[8][valid].mean()),
                    "local_relief_std_m": float(terrain[8][valid].std()),
                }
            )
            fig, axes = plt.subplots(1, 6, figsize=(13.5, 2.45), dpi=150)
            draw_row(axes, pd.Series(row._asdict()), optical, terrain, mask, valid)
            fig.suptitle(
                f"{row.event_id} | post-hoc diagnostic only",
                fontsize=9,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0.05, 0.0, 1.0, 0.92))
            path = individual_dir / f"{row.sample_id}.png"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            rendered.append((row, optical, terrain, mask, valid))

        for sheet_index in range(math.ceil(len(rendered) / args.rows_per_sheet)):
            start = sheet_index * args.rows_per_sheet
            subset = rendered[start : start + args.rows_per_sheet]
            fig, axes = plt.subplots(
                len(subset),
                6,
                figsize=(14.5, 2.35 * len(subset) + 0.7),
                dpi=150,
                squeeze=False,
            )
            for row_index, (row, optical, terrain, mask, valid) in enumerate(subset):
                draw_row(
                    axes[row_index],
                    pd.Series(row._asdict()),
                    optical,
                    terrain,
                    mask,
                    valid,
                )
            fig.suptitle(
                (
                    "DLR post-hoc high-response samples: raw evidence audit "
                    "(not a deployable selection rule)"
                ),
                fontsize=11,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0.05, 0.0, 1.0, 0.965))
            fig.savefig(
                outdir / f"contact_sheet_{sheet_index + 1:02d}.png",
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(outdir / "sample_visual_audit.csv", index=False)
    event_summary = (
        audit.groupby("event_id", as_index=False)
        .agg(
            n_samples=("sample_id", "size"),
            mean_rer=("rer", "mean"),
            mean_delta_iou=("delta_iou", "mean"),
            median_mask_fraction=("mask_fraction", "median"),
            median_components=("mask_components", "median"),
            median_rgb_change=("earliest_latest_rgb_abs_change", "median"),
            median_slope_std=("slope_std_deg", "median"),
        )
        .sort_values("n_samples", ascending=False)
    )
    event_summary.to_csv(outdir / "event_summary.csv", index=False)
    summary = {
        "scientific_status": (
            "post-hoc visual diagnosis only; labels and response outcomes were "
            "used to define the rendered set"
        ),
        "n_samples": int(len(audit)),
        "n_events": int(audit["event_id"].nunique()),
        "n_contact_sheets": int(
            math.ceil(len(audit) / max(args.rows_per_sheet, 1))
        ),
        "largest_event_share": float(
            event_summary.iloc[0]["n_samples"] / max(len(audit), 1)
        ),
        "n_positive_delta_iou": int((audit["delta_iou"] > 0).sum()),
        "n_nonpositive_delta_iou": int((audit["delta_iou"] <= 0).sum()),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (outdir / "report.md").write_text(
        "\n".join(
            [
                "# DLR responder visual audit v1",
                "",
                "- This is a post-hoc diagnostic, not a performance subset.",
                f"- Rendered {summary['n_samples']} samples from {summary['n_events']} events.",
                f"- Largest event share: {summary['largest_event_share']:.1%}.",
                (
                    f"- DeltaIoU positive/non-positive samples: "
                    f"{summary['n_positive_delta_iou']}/"
                    f"{summary['n_nonpositive_delta_iou']}."
                ),
                "- RGB panels use per-sample display stretches and are not model inputs.",
                "- Reference overlays are shown for diagnosis but are forbidden in any",
                "  prospective inclusion or routing rule.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
