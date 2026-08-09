#!/usr/bin/env python3
"""Evaluate frozen per-sample predictions on prespecified support strata."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SEED_PATTERN = re.compile(r"seed(\d+)")

STRATA = {
    "all_hard_eligible": "hard_quality_eligible == 1",
    "terrain_qualified": "hard_quality_eligible == 1 and qT_eligible == 1",
    "material_qualified": "hard_quality_eligible == 1 and qM_eligible == 1",
    "trigger_qualified": "hard_quality_eligible == 1 and qR_eligible == 1",
    "full_tmr_qualified": "hard_quality_eligible == 1 and full_tmr_eligible == 1",
    "visual_degraded": "hard_quality_eligible == 1 and visual_degraded == 1",
    "visual_degraded_terrain_qualified": (
        "hard_quality_eligible == 1 and visual_degraded == 1 and qT_eligible == 1"
    ),
    "visual_degraded_full_tmr_qualified": (
        "hard_quality_eligible == 1 and visual_degraded == 1 and full_tmr_eligible == 1"
    ),
    "visual_clear_terrain_qualified": (
        "hard_quality_eligible == 1 and visual_degraded == 0 and qT_eligible == 1"
    ),
}

COUNT_COLUMNS = (
    "baseline_tp",
    "baseline_fp",
    "baseline_fn",
    "baseline_tn",
    "adapted_tp",
    "adapted_fp",
    "adapted_fn",
    "adapted_tn",
    "corrected",
    "harmed",
)


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


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        path,
    )


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        paths.update(Path(match).resolve() for match in matches)
    return sorted(paths)


def seed_from_path(path: Path) -> int:
    match = SEED_PATTERN.search(str(path))
    if not match:
        raise ValueError(f"cannot infer seed from path: {path}")
    return int(match.group(1))


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    required = {"sample_id", "canonical_event_id", *COUNT_COLUMNS}
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        frame = frame.copy()
        frame["seed"] = seed_from_path(path)
        frame["prediction_path"] = str(path)
        pieces.append(frame)
    if not pieces:
        raise ValueError("no prediction files")
    predictions = pd.concat(pieces, ignore_index=True)
    duplicate = predictions.duplicated(["seed", "sample_id"], keep=False)
    if duplicate.any():
        examples = predictions.loc[duplicate, ["seed", "sample_id", "prediction_path"]].head()
        raise ValueError(f"duplicate per-seed predictions:\n{examples}")
    return predictions


def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    counts = {
        column: int(pd.to_numeric(frame[column], errors="raise").sum())
        for column in COUNT_COLUMNS
    }
    baseline_union = counts["baseline_tp"] + counts["baseline_fp"] + counts["baseline_fn"]
    adapted_union = counts["adapted_tp"] + counts["adapted_fp"] + counts["adapted_fn"]
    baseline_errors = counts["baseline_fp"] + counts["baseline_fn"]
    adapted_errors = counts["adapted_fp"] + counts["adapted_fn"]
    baseline_iou = counts["baseline_tp"] / max(baseline_union, 1)
    adapted_iou = counts["adapted_tp"] / max(adapted_union, 1)
    return {
        "n_rows": len(frame),
        "n_samples": int(frame["sample_id"].nunique()),
        "n_events": int(frame["canonical_event_id"].nunique()),
        **counts,
        "baseline_iou": baseline_iou,
        "adapted_iou": adapted_iou,
        "delta_iou": adapted_iou - baseline_iou,
        "baseline_errors": baseline_errors,
        "adapted_errors": adapted_errors,
        "rer": (baseline_errors - adapted_errors) / max(baseline_errors, 1),
        "corrected_to_harmed": counts["corrected"] / max(counts["harmed"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eligibility",
        type=Path,
        default=Path(
            "metadata/pild_sen12_training_v2/support_eligibility_v1/"
            "sample_eligibility_v1.csv"
        ),
    )
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--label-geometry", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    eligibility_path = (
        args.eligibility if args.eligibility.is_absolute() else ROOT / args.eligibility
    )
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    eligibility = pd.read_csv(eligibility_path, low_memory=False)
    if eligibility["sample_id"].duplicated().any():
        raise ValueError("eligibility sample_id is not unique")
    paths = expand_inputs(args.predictions)
    predictions = load_predictions(paths)
    if args.dataset_id:
        predictions = predictions.loc[predictions["dataset_id"].eq(args.dataset_id)].copy()
    merged = predictions.merge(
        eligibility,
        on="sample_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_eligibility"),
    )
    if merged["hard_quality_eligible"].isna().any():
        missing = merged.loc[merged["hard_quality_eligible"].isna(), "sample_id"].head().tolist()
        raise ValueError(f"predictions absent from eligibility audit: {missing}")
    if "canonical_event_id_eligibility" in merged:
        mismatch = merged["canonical_event_id"].astype(str).ne(
            merged["canonical_event_id_eligibility"].astype(str)
        )
        if mismatch.any():
            raise ValueError("prediction and eligibility canonical_event_id differ")

    strata = dict(STRATA)
    label_geometry_path: Path | None = None
    if args.label_geometry is not None:
        label_geometry_path = (
            args.label_geometry
            if args.label_geometry.is_absolute()
            else ROOT / args.label_geometry
        )
        geometry = pd.read_csv(label_geometry_path, low_memory=False)
        columns = [
            "sample_id",
            "largest_component_area_m2",
            "largest_component_ge_900m2",
            "largest_component_ge_3600m2",
            "largest_component_ge_8100m2",
        ]
        missing = sorted(set(columns) - set(geometry.columns))
        if missing:
            raise ValueError(f"label geometry missing columns: {missing}")
        merged = merged.merge(
            geometry[columns], on="sample_id", how="left", validate="many_to_one"
        )
        if merged["largest_component_area_m2"].isna().any():
            raise ValueError("label geometry join is incomplete")
        strata.update(
            {
                "terrain_qualified_mmu_ge_900m2": (
                    "hard_quality_eligible == 1 and qT_eligible == 1 and "
                    "largest_component_ge_900m2 == 1"
                ),
                "terrain_qualified_mmu_ge_3600m2": (
                    "hard_quality_eligible == 1 and qT_eligible == 1 and "
                    "largest_component_ge_3600m2 == 1"
                ),
                "terrain_qualified_mmu_ge_8100m2": (
                    "hard_quality_eligible == 1 and qT_eligible == 1 and "
                    "largest_component_ge_8100m2 == 1"
                ),
            }
        )

    per_seed: dict[str, Any] = {}
    for seed, seed_frame in merged.groupby("seed", sort=True):
        per_seed[str(seed)] = {
            name: aggregate(seed_frame.query(query))
            for name, query in strata.items()
        }
    pooled = {
        name: aggregate(merged.query(query))
        for name, query in strata.items()
    }
    summary = {
        "status": "complete",
        "scientific_status": "prespecified support-stratum sensitivity analysis",
        "eligibility": str(eligibility_path.resolve()),
        "label_geometry": (
            str(label_geometry_path.resolve()) if label_geometry_path is not None else None
        ),
        "prediction_files": [str(path) for path in paths],
        "dataset_id": args.dataset_id,
        "n_seeds": int(merged["seed"].nunique()),
        "strata_contract": strata,
        "per_seed": per_seed,
        "pooled": pooled,
        "caveat": (
            "Rows are pooled across optimization seeds; seeds repeat the same test samples. "
            "The all-hard-eligible stratum remains the primary denominator."
        ),
    }
    atomic_json(summary, outdir / "summary.json")

    lines = [
        "# PILD support-stratum sensitivity analysis",
        "",
        f"- Dataset: `{args.dataset_id or 'all'}`",
        f"- Seeds: `{summary['n_seeds']}`",
        "- Primary denominator: `all_hard_eligible`.",
        "- T/M/R support flags were frozen without labels, predictions, or outcome metrics.",
        *(
            [
                "- MMU strata read label geometry but use fixed sensor-resolution thresholds, "
                "not model outcomes."
            ]
            if label_geometry_path is not None
            else []
        ),
        "",
        "| stratum | samples | events | visual IoU | adapted IoU | DeltaIoU | RER | C/H |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in pooled.items():
        lines.append(
            f"| {name} | {values['n_samples']} | {values['n_events']} | "
            f"{values['baseline_iou']:.6f} | {values['adapted_iou']:.6f} | "
            f"{values['delta_iou']:+.6f} | {values['rer']:+.2%} | "
            f"{values['corrected_to_harmed']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Support-qualified results are mechanism analyses, not replacements for the primary result.",
            "",
        ]
    )
    atomic_text("\n".join(lines), outdir / "report.md")
    print(json.dumps(json_safe(pooled), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
