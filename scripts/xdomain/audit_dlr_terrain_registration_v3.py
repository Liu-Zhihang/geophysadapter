#!/usr/bin/env python3
"""Audit DLR external Terrain registration against source-native CDEM without labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "processed/hybrid_pinn/pild_core_geo_v2_1_native30_raw/dlr_window_registry_v2.csv"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT / "metadata/pild_core_v2/phase2_audit/dlr_legacy_cdem_support_audit_v2.csv"
)
DEFAULT_TERRAIN = (
    PROJECT_ROOT / "processed/hybrid_pinn/dlr_terrain_v3/dlr_copdem_native17_p128.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source-audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--terrain-h5", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "metadata/reports/dlr_terrain_registration_v3",
    )
    parser.add_argument("--max-shift-pixels", type=int, default=3)
    parser.add_argument("--minimum-valid-pixels", type=int, default=2048)
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def parse_affine(text: str) -> Affine:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 6:
        raise ValueError(f"invalid target_transform: {text}")
    return Affine(*values)


def overlap(
    reference: np.ndarray, candidate: np.ndarray, dx: int, dy: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference.shape
    ref_y = slice(max(0, dy), min(height, height + dy))
    ref_x = slice(max(0, dx), min(width, width + dx))
    can_y = slice(max(0, -dy), min(height, height - dy))
    can_x = slice(max(0, -dx), min(width, width - dx))
    return reference[ref_y, ref_x], candidate[can_y, can_x]


def correlation(left: np.ndarray, right: np.ndarray, minimum_pixels: int) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < minimum_pixels:
        return float("nan")
    x = left[valid].astype(np.float64)
    y = right[valid].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")


def read_source_patch(
    dataset: rasterio.DatasetReader, transform: Affine
) -> np.ndarray:
    if dataset.crs is None:
        raise RuntimeError(f"source raster lacks CRS: {dataset.name}")
    col = int(round((transform.c - dataset.transform.c) / dataset.transform.a))
    row = int(round((transform.f - dataset.transform.f) / dataset.transform.e))
    expected = dataset.transform * (col, row)
    if abs(expected[0] - transform.c) > 0.1 or abs(expected[1] - transform.f) > 0.1:
        raise RuntimeError(
            f"DLR source and target grids are not integer aligned: {dataset.name}"
        )
    return dataset.read(
        1,
        window=Window(col, row, 128, 128),
        boundless=True,
        fill_value=np.nan,
    ).astype(np.float32)


def main() -> int:
    args = parse_args()
    if args.max_shift_pixels < 0:
        raise ValueError("max-shift-pixels must be non-negative")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with args.registry.open("r", encoding="utf-8-sig", newline="") as stream:
        registry_rows = list(csv.DictReader(stream))
    registry = {row["sample_id"]: row for row in registry_rows}
    with args.source_audit_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    source_dem_paths = {row["event_eid"]: Path(row["path"]) for row in source_rows}
    sample_rows = []
    with h5py.File(args.terrain_h5, "r") as terrain, ExitStack() as stack:
        sample_ids = decode(terrain["sample_id"][:])
        names = decode(terrain["terrain_names"][:])
        if names[:2] != ["elevation", "slope_deg"]:
            raise RuntimeError("Terrain cache is not canonical native17")
        opened: dict[str, tuple[rasterio.DatasetReader, rasterio.DatasetReader]] = {}
        for index, sample_id in enumerate(sample_ids):
            row = registry[sample_id]
            event = row["source_scene_id"]
            if event not in opened:
                dem_path = source_dem_paths[event]
                slope_path = dem_path.with_name(dem_path.name.replace("__DEM.tif", "__SLOPE.tif"))
                opened[event] = (
                    stack.enter_context(rasterio.open(dem_path)),
                    stack.enter_context(rasterio.open(slope_path)),
                )
            source_dem, source_slope = opened[event]
            transform = parse_affine(row["target_transform"])
            if str(source_dem.crs) != row["target_crs"]:
                raise RuntimeError(
                    f"CRS mismatch for {sample_id}: {source_dem.crs} vs {row['target_crs']}"
                )
            reference_elevation = read_source_patch(source_dem, transform)
            reference_slope = read_source_patch(source_slope, transform)
            candidate_elevation = np.asarray(terrain["terrain"][index, 0], dtype=np.float32)
            candidate_slope = np.asarray(terrain["terrain"][index, 1], dtype=np.float32)
            valid = np.asarray(terrain["terrain_valid"][index, 0], dtype=bool)
            reference_elevation[~valid] = np.nan
            reference_slope[~valid] = np.nan
            candidate_elevation[~valid] = np.nan
            candidate_slope[~valid] = np.nan
            reference_highpass = reference_elevation - ndimage.gaussian_filter(
                np.nan_to_num(reference_elevation, nan=np.nanmedian(reference_elevation)),
                sigma=3.0,
                mode="nearest",
            )
            candidate_highpass = candidate_elevation - ndimage.gaussian_filter(
                np.nan_to_num(candidate_elevation, nan=np.nanmedian(candidate_elevation)),
                sigma=3.0,
                mode="nearest",
            )
            scores = []
            for dy in range(-args.max_shift_pixels, args.max_shift_pixels + 1):
                for dx in range(-args.max_shift_pixels, args.max_shift_pixels + 1):
                    ref_s, can_s = overlap(reference_slope, candidate_slope, dx, dy)
                    ref_h, can_h = overlap(
                        reference_highpass, candidate_highpass, dx, dy
                    )
                    slope_correlation = correlation(
                        ref_s, can_s, args.minimum_valid_pixels
                    )
                    highpass_correlation = correlation(
                        ref_h, can_h, args.minimum_valid_pixels
                    )
                    score = float(np.nanmean([slope_correlation, highpass_correlation]))
                    scores.append((score, dx, dy, slope_correlation, highpass_correlation))
            scores.sort(reverse=True)
            best = scores[0]
            zero = next(score for score in scores if score[1] == 0 and score[2] == 0)
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "event_id": event,
                    "best_dx_pixels": best[1],
                    "best_dy_pixels": best[2],
                    "best_score": best[0],
                    "zero_score": zero[0],
                    "score_gain": best[0] - zero[0],
                    "best_slope_correlation": best[3],
                    "best_highpass_elevation_correlation": best[4],
                    "zero_slope_correlation": zero[3],
                    "zero_highpass_elevation_correlation": zero[4],
                }
            )
    with (args.outdir / "per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=sample_rows[0].keys())
        writer.writeheader()
        writer.writerows(sample_rows)
    event_samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        event_samples[str(row["event_id"])].append(row)
    event_rows = []
    for event, rows in sorted(event_samples.items()):
        offsets = Counter(
            (int(row["best_dx_pixels"]), int(row["best_dy_pixels"])) for row in rows
        )
        modal_offset, modal_count = offsets.most_common(1)[0]
        event_rows.append(
            {
                "event_id": event,
                "n_samples": len(rows),
                "modal_dx_pixels": modal_offset[0],
                "modal_dy_pixels": modal_offset[1],
                "modal_fraction": modal_count / len(rows),
                "median_score_gain": float(
                    np.median([float(row["score_gain"]) for row in rows])
                ),
                "median_zero_score": float(
                    np.median([float(row["zero_score"]) for row in rows])
                ),
            }
        )
    with (args.outdir / "per_event.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=event_rows[0].keys())
        writer.writeheader()
        writer.writerows(event_rows)
    modal_events = Counter(
        (int(row["modal_dx_pixels"]), int(row["modal_dy_pixels"])) for row in event_rows
    )
    global_modal, global_count = modal_events.most_common(1)[0]
    summary = {
        "status": "complete",
        "scientific_status": "label-free registration audit",
        "terrain_h5": str(args.terrain_h5.resolve()),
        "n_samples": len(sample_rows),
        "n_events": len(event_rows),
        "global_modal_event_offset_pixels": list(global_modal),
        "global_modal_event_fraction": global_count / len(event_rows),
        "zero_modal_events": modal_events.get((0, 0), 0),
        "median_sample_zero_score": float(
            np.median([float(row["zero_score"]) for row in sample_rows])
        ),
        "median_sample_best_score_gain": float(
            np.median([float(row["score_gain"]) for row in sample_rows])
        ),
        "decision_rule": (
            "create a corrected cache only if one non-zero event-level modal offset "
            "covers at least 60% of events and yields median score gain >=0.02"
        ),
    }
    summary["systematic_shift_detected"] = bool(
        global_modal != (0, 0)
        and summary["global_modal_event_fraction"] >= 0.60
        and summary["median_sample_best_score_gain"] >= 0.02
    )
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
