#!/usr/bin/env python3
"""Audit landslide label geometry against optical and Terrain support scales.

This is a descriptive target-geometry audit. It is deliberately separate from
the label-independent eligibility audit because it reads segmentation labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
PIXEL_SIZE_M = 10.0
PIXEL_AREA_M2 = PIXEL_SIZE_M**2
MMU_THRESHOLDS_M2 = (100.0, 900.0, 3600.0, 8100.0)
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        path,
    )


def component_metrics(mask: np.ndarray) -> dict[str, float | int]:
    labels, n_components = ndimage.label(mask, structure=STRUCTURE_8)
    if n_components == 0:
        return {
            "positive_pixels": 0,
            "total_area_m2": 0.0,
            "n_components": 0,
            "largest_component_pixels": 0,
            "largest_component_area_m2": 0.0,
            "largest_component_fraction": 0.0,
            "sub_dem_cell_area_fraction": 0.0,
        }
    counts = np.bincount(labels.ravel())[1:]
    positive_pixels = int(counts.sum())
    largest = int(counts.max())
    sub_dem = int(counts[counts * PIXEL_AREA_M2 < 900.0].sum())
    return {
        "positive_pixels": positive_pixels,
        "total_area_m2": positive_pixels * PIXEL_AREA_M2,
        "n_components": int(n_components),
        "largest_component_pixels": largest,
        "largest_component_area_m2": largest * PIXEL_AREA_M2,
        "largest_component_fraction": largest / max(positive_pixels, 1),
        "sub_dem_cell_area_fraction": sub_dem / max(positive_pixels, 1),
    }


def describe(group: pd.DataFrame) -> dict[str, Any]:
    largest = group["largest_component_area_m2"]
    total = group["total_area_m2"]
    values: dict[str, Any] = {
        "n_samples": len(group),
        "n_events": int(group["canonical_event_id"].nunique()),
        "zero_positive_samples": int(group["positive_pixels"].eq(0).sum()),
        "total_area_m2": {
            "median": float(total.median()),
            "p10": float(total.quantile(0.10)),
            "p90": float(total.quantile(0.90)),
        },
        "largest_component_area_m2": {
            "median": float(largest.median()),
            "p10": float(largest.quantile(0.10)),
            "p90": float(largest.quantile(0.90)),
        },
        "n_components_median": float(group["n_components"].median()),
        "sub_dem_cell_area_fraction_mean": float(
            group["sub_dem_cell_area_fraction"].mean()
        ),
        "mmu_pass_fraction": {},
    }
    for threshold in MMU_THRESHOLDS_M2:
        values["mmu_pass_fraction"][str(int(threshold))] = float(
            largest.ge(threshold).mean()
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/pild_sen12_training_v2/unified_sample_manifest_v2.csv"),
    )
    parser.add_argument(
        "--eligibility",
        type=Path,
        default=Path(
            "metadata/pild_sen12_training_v2/support_eligibility_v1/"
            "sample_eligibility_v1.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/pild_sen12_training_v2/label_geometry_v1"),
    )
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    eligibility_path = (
        args.eligibility if args.eligibility.is_absolute() else ROOT / args.eligibility
    )
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, low_memory=False)
    eligibility = pd.read_csv(eligibility_path, low_memory=False)
    if manifest["sample_id"].duplicated().any() or eligibility["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")

    rows: list[dict[str, Any]] = []
    for path_text, group in manifest.groupby("base_h5_path", sort=True):
        path = Path(path_text)
        indices = group["base_h5_index"].to_numpy(dtype=int)
        order = np.argsort(indices)
        indices = indices[order]
        group = group.iloc[order]
        with h5py.File(path, "r") as handle:
            if "mask" not in handle or "valid_mask" not in handle:
                raise ValueError(f"{path} lacks mask/valid_mask")
            for start in range(0, len(indices), 128):
                selected = indices[start : start + 128]
                selected_group = group.iloc[start : start + len(selected)]
                mask = np.asarray(handle["mask"][selected]) > 0
                valid = np.asarray(handle["valid_mask"][selected]) > 0
                mask = mask & valid
                for position, (_, sample) in enumerate(selected_group.iterrows()):
                    values = component_metrics(mask[position, 0])
                    rows.append(
                        {
                            "dataset_id": sample.dataset_id,
                            "canonical_event_id": sample.canonical_event_id,
                            "sample_id": sample.sample_id,
                            **values,
                        }
                    )
    geometry = pd.DataFrame(rows)
    if geometry["sample_id"].duplicated().any() or len(geometry) != len(manifest):
        raise ValueError("label geometry identity/row-count mismatch")
    frame = geometry.merge(
        eligibility[
            [
                "sample_id",
                "hard_quality_eligible",
                "qT_eligible",
                "qM_eligible",
                "qR_eligible",
                "full_tmr_eligible",
                "visual_degraded",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if frame["hard_quality_eligible"].isna().any():
        raise ValueError("eligibility join is incomplete")
    for threshold in MMU_THRESHOLDS_M2:
        frame[f"largest_component_ge_{int(threshold)}m2"] = (
            frame["largest_component_area_m2"].ge(threshold).astype(int)
        )

    strata = {
        "all_hard_eligible": frame["hard_quality_eligible"].eq(1),
        "terrain_qualified": frame["qT_eligible"].eq(1),
        "material_qualified": frame["qM_eligible"].eq(1),
        "trigger_qualified": frame["qR_eligible"].eq(1),
        "full_tmr_qualified": frame["full_tmr_eligible"].eq(1),
        "visual_degraded_terrain_qualified": (
            frame["visual_degraded"].eq(1) & frame["qT_eligible"].eq(1)
        ),
    }
    summary: dict[str, Any] = {
        "status": "complete",
        "scientific_status": "descriptive label-geometry audit; not an label-independent filter",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pixel_size_m": PIXEL_SIZE_M,
        "pixel_area_m2": PIXEL_AREA_M2,
        "mmu_thresholds_m2": MMU_THRESHOLDS_M2,
        "manifest_sha256": sha256_file(manifest_path),
        "eligibility_sha256": sha256_file(eligibility_path),
        "overall_strata": {
            name: describe(frame.loc[mask])
            for name, mask in strata.items()
        },
        "by_dataset": {},
        "guardrail": (
            "MMU thresholds are resolution-derived sensitivity strata. They must not replace "
            "the primary test population unless the target mapping definition is prospectively "
            "changed and every method is retrained/evaluated under the same definition."
        ),
    }
    for dataset_id, dataset in frame.groupby("dataset_id", sort=True):
        summary["by_dataset"][str(dataset_id)] = {
            "all": describe(dataset),
            "terrain_qualified": describe(dataset.loc[dataset["qT_eligible"].eq(1)]),
            "full_tmr_qualified": describe(
                dataset.loc[dataset["full_tmr_eligible"].eq(1)]
            ),
        }

    sample_path = outdir / "sample_label_geometry_v1.csv"
    summary_path = outdir / "summary.json"
    report_path = outdir / "report.md"
    atomic_csv(frame, sample_path)
    atomic_json(summary, summary_path)

    lines = [
        "# PILD label geometry versus physical support scale v1",
        "",
        "- This audit reads labels and is separate from the label-independent support filter.",
        "- Pixel size: 10 m; one pixel = 100 m2.",
        "- Terrain native support: about 30 m; one Terrain cell = about 900 m2.",
        "- Resolution-derived largest-component strata: 100, 900, 3,600, and 8,100 m2.",
        "",
        "| dataset | samples | median largest m2 | >=900 m2 | >=3600 m2 | >=8100 m2 | zero-positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id, values in summary["by_dataset"].items():
        current = values["all"]
        lines.append(
            f"| {dataset_id} | {current['n_samples']} | "
            f"{current['largest_component_area_m2']['median']:.0f} | "
            f"{current['mmu_pass_fraction']['900']:.2%} | "
            f"{current['mmu_pass_fraction']['3600']:.2%} | "
            f"{current['mmu_pass_fraction']['8100']:.2%} | "
            f"{current['zero_positive_samples']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            summary["guardrail"],
            "",
        ]
    )
    atomic_text("\n".join(lines), report_path)
    atomic_json(
        {
            "status": "complete",
            "inputs": {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "eligibility": str(eligibility_path.resolve()),
                "eligibility_sha256": sha256_file(eligibility_path),
            },
            "outputs": {
                path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in (sample_path, summary_path, report_path)
            },
        },
        outdir / "FREEZE.json",
    )
    print(json.dumps(json_safe(summary["by_dataset"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
