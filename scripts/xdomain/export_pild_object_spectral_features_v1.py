#!/usr/bin/env python3
"""Export object-level spectral and change descriptors for veto models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
FOLD_IDS = [f"source_stratified_{i}" for i in range(4)]

BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = range(6)
EPS = 1e-6


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized difference index; EPS avoids division by zero."""
    return (a - b) / (a + b + EPS)


def spectral_index_stack(cube: np.ndarray) -> dict[str, np.ndarray]:
    """Interpretive indices from a 6-band cube of shape (6, H, W)."""
    blue, green, red = cube[BLUE], cube[GREEN], cube[RED]
    nir, swir1, swir2 = cube[NIR], cube[SWIR1], cube[SWIR2]
    return {
        "ndvi": normalized_difference(nir, red),
        "nbr": normalized_difference(nir, swir2),
        "ndwi": normalized_difference(green, nir),
        # Bare-soil index: increases when scar material is exposed
        "bsi": normalized_difference(swir1 + red, nir + blue),
        "brightness": cube.mean(axis=0),
    }


def component_spectral_features(
    mask: np.ndarray,
    window: tuple[slice, slice],
    pre_idx: dict[str, np.ndarray],
    post_idx: dict[str, np.ndarray],
    pre_cube: np.ndarray,
    post_cube: np.ndarray,
    valid: np.ndarray,
    ring_radius: int,
) -> dict[str, float]:
    """Spectral descriptors for one component, including ring contrast."""
    row_slice, col_slice = window
    # Leave a margin for the ring
    r0 = max(row_slice.start - ring_radius, 0)
    r1 = min(row_slice.stop + ring_radius, valid.shape[0])
    c0 = max(col_slice.start - ring_radius, 0)
    c1 = min(col_slice.stop + ring_radius, valid.shape[1])

    big = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    big[row_slice.start - r0 : row_slice.stop - r0, col_slice.start - c0 : col_slice.stop - c0] = mask
    dilated = ndimage.binary_dilation(big, iterations=ring_radius)
    ring = dilated & ~big & valid[r0:r1, c0:c1]

    out: dict[str, float] = {}
    sub = (slice(r0, r1), slice(c0, c1))

    #
    for name, band in (("blue", BLUE), ("green", GREEN), ("red", RED),
                       ("nir", NIR), ("swir1", SWIR1), ("swir2", SWIR2)):
        post_band = post_cube[band][sub][big]
        pre_band = pre_cube[band][sub][big]
        out[f"spec_post_{name}"] = float(post_band.mean())
        out[f"spec_delta_{name}"] = float((post_band - pre_band).mean())

    #
    for key in ("ndvi", "nbr", "ndwi", "bsi", "brightness"):
        pre_vals = pre_idx[key][sub][big]
        post_vals = post_idx[key][sub][big]
        delta = post_vals - pre_vals
        out[f"spec_pre_{key}"] = float(pre_vals.mean())
        out[f"spec_post_{key}"] = float(post_vals.mean())
        out[f"spec_delta_{key}"] = float(delta.mean())
        if key == "ndvi":
            out["spec_delta_ndvi_std"] = float(delta.std())
            out["spec_post_ndvi_p10"] = float(np.percentile(post_vals, 10))

    #
    if ring.any():
        for key in ("ndvi", "bsi", "brightness"):
            ring_post = post_idx[key][sub][ring]
            ring_delta = (post_idx[key][sub] - pre_idx[key][sub])[ring]
            obj_post = post_idx[key][sub][big]
            obj_delta = (post_idx[key][sub] - pre_idx[key][sub])[big]
            out[f"spec_ring_{key}_post"] = float(ring_post.mean())
            out[f"spec_contrast_{key}_post"] = float(obj_post.mean() - ring_post.mean())
            out[f"spec_contrast_{key}_delta"] = float(obj_delta.mean() - ring_delta.mean())
        out["spec_ring_valid_fraction"] = float(ring.sum() / max(dilated.sum() - big.sum(), 1))
    else:
        for key in ("ndvi", "bsi", "brightness"):
            out[f"spec_ring_{key}_post"] = np.nan
            out[f"spec_contrast_{key}_post"] = np.nan
            out[f"spec_contrast_{key}_delta"] = np.nan
        out["spec_ring_valid_fraction"] = 0.0
    return out


def process_fold(
    cache_dir: Path, fold_id: str, threshold: float | None, min_area: int, ring_radius: int
) -> list[dict]:
    """。

    （0.92 / 0.95 / 0.805 / 0.92）， receipt ，
    。
    """
    if threshold is None:
        receipt = json.loads(
            (cache_dir / f"{fold_id}_oof_cache_receipt.json").read_text(encoding="utf-8")
        )
        threshold = float(receipt["threshold"])
    with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
        sample_id = [str(item) for item in handle["sample_id"]]
        probability = handle["visual_probability"]
        valid_all = handle["valid"]
    with np.load(cache_dir / f"{fold_id}_optical_cache.npz", allow_pickle=False) as handle:
        optical_ids = [str(item) for item in handle["sample_id"]]
        pre_all = handle["optical_pre"]
        post_all = handle["optical_post"]
    if optical_ids != sample_id:
        raise RuntimeError(f"text")

    structure = ndimage.generate_binary_structure(2, 2)
    rows: list[dict] = []
    for index in range(len(sample_id)):
        keep = valid_all[index].astype(bool)
        predicted = (probability[index].astype(np.float32) >= threshold) & keep
        if not predicted.any():
            continue
        labels, count = ndimage.label(predicted, structure=structure)
        if count == 0:
            continue
        pre_cube = pre_all[index].astype(np.float32)
        post_cube = post_all[index].astype(np.float32)
        pre_idx = spectral_index_stack(pre_cube)
        post_idx = spectral_index_stack(post_cube)
        windows = ndimage.find_objects(labels)
        for label_value in range(1, count + 1):
            window = windows[label_value - 1]
            local = labels[window] == label_value
            if int(np.count_nonzero(local)) < min_area:
                continue
            feats = component_spectral_features(
                local, window, pre_idx, post_idx, pre_cube, post_cube, keep, ring_radius
            )
            feats["sample_id"] = sample_id[index]
            feats["component_id"] = int(label_value)
            rows.append(feats)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="text",
    )
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--ring-radius", type=int, default=5)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_spectral_features_v1",
    )
    args = parser.parse_args()
    started = time.time()

    frames = []
    thresholds = {}
    for fold_id in FOLD_IDS:
        receipt = json.loads(
            (args.cache / f"{fold_id}_oof_cache_receipt.json").read_text(encoding="utf-8")
        )
        thresholds[fold_id] = args.threshold or float(receipt["threshold"])
        rows = process_fold(
            args.cache, fold_id, args.threshold, args.min_area, args.ring_radius
        )
        print(f"text")
        frames.append(pd.DataFrame(rows))
    table = pd.concat(frames, ignore_index=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.outdir / "object_spectral_features.parquet", index=False)
    feature_cols = [c for c in table.columns if c.startswith("spec_")]
    receipt = {
        "schema_version": "pild_object_spectral_features.v1",
        "thresholds_per_fold": thresholds,
        "min_area": args.min_area,
        "ring_radius": args.ring_radius,
        "band_order": ["B02", "B03", "B04", "B8A", "B11", "B12"],
        "n_objects": int(len(table)),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.outdir / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"text")
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
