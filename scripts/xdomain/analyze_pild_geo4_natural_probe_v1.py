#!/usr/bin/env python3
"""Summarize natural-proportion PILD-GEO4 additive probes by seed and dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


COUNT_COLUMNS = (
    "baseline_tp",
    "baseline_fp",
    "baseline_fn",
    "adapted_tp",
    "adapted_fp",
    "adapted_fn",
    "corrected",
    "harmed",
)


def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    counts = {name: float(frame[name].sum()) for name in COUNT_COLUMNS}
    baseline_union = (
        counts["baseline_tp"] + counts["baseline_fp"] + counts["baseline_fn"]
    )
    adapted_union = (
        counts["adapted_tp"] + counts["adapted_fp"] + counts["adapted_fn"]
    )
    baseline_errors = counts["baseline_fp"] + counts["baseline_fn"]
    adapted_errors = counts["adapted_fp"] + counts["adapted_fn"]
    return {
        "n_samples": int(len(frame)),
        "n_events": int(frame["canonical_event_id"].nunique()),
        "baseline_iou": counts["baseline_tp"] / max(baseline_union, 1.0),
        "adapted_iou": counts["adapted_tp"] / max(adapted_union, 1.0),
        "delta_iou": (
            counts["adapted_tp"] / max(adapted_union, 1.0)
            - counts["baseline_tp"] / max(baseline_union, 1.0)
        ),
        "baseline_errors": baseline_errors,
        "adapted_errors": adapted_errors,
        "corrected": counts["corrected"],
        "harmed": counts["harmed"],
        "net_error_reduction": baseline_errors - adapted_errors,
        "rer": (baseline_errors - adapted_errors) / max(baseline_errors, 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold-id", default="event_isolated")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    fold_root = args.root.resolve() / args.fold_id
    dataset_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for seed_dir in sorted(fold_root.glob("seed*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed"))
        except ValueError:
            continue
        result_path = seed_dir / "additive" / "result.json"
        sample_path = seed_dir / "additive" / "per_sample_metrics.csv"
        if not result_path.is_file() or not sample_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(sample_path)
        if frame.empty:
            raise RuntimeError(f"empty per-sample evidence: {sample_path}")
        seed_rows.append(
            {
                "seed": seed,
                "selected_alpha": float(result["selection"]["alpha"]),
                "selected_uncertainty_power": float(
                    result["selection"]["uncertainty_power"]
                ),
                **aggregate(frame),
            }
        )
        for dataset_id, group in frame.groupby("dataset_id", sort=True):
            dataset_rows.append(
                {"seed": seed, "dataset_id": str(dataset_id), **aggregate(group)}
            )

        controls_path = seed_dir / "controls_v2" / "summary.json"
        if controls_path.is_file():
            controls = json.loads(controls_path.read_text(encoding="utf-8"))
            for row in controls["conditions"]:
                control_rows.append(
                    {
                        "seed": seed,
                        "condition": str(row["condition"]),
                        "delta_iou": float(row["delta_iou"]),
                        "rer": float(row["rer"]),
                        "corrected": float(row["corrected"]),
                        "harmed": float(row["harmed"]),
                    }
                )
    if not seed_rows:
        raise RuntimeError(f"no completed probe seeds under {fold_root}")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seed_frame = pd.DataFrame(seed_rows).sort_values("seed")
    dataset_frame = pd.DataFrame(dataset_rows).sort_values(["seed", "dataset_id"])
    seed_frame.to_csv(outdir / "seed_metrics.csv", index=False)
    dataset_frame.to_csv(outdir / "dataset_metrics.csv", index=False)
    dataset_summary = (
        dataset_frame.groupby("dataset_id", sort=True)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_baseline_iou=("baseline_iou", "mean"),
            mean_adapted_iou=("adapted_iou", "mean"),
            mean_delta_iou=("delta_iou", "mean"),
            mean_rer=("rer", "mean"),
            positive_delta_iou_seeds=("delta_iou", lambda value: int((value > 0).sum())),
            positive_rer_seeds=("rer", lambda value: int((value > 0).sum())),
        )
        .reset_index()
    )
    dataset_summary.to_csv(outdir / "dataset_summary.csv", index=False)
    if control_rows:
        pd.DataFrame(control_rows).sort_values(["seed", "condition"]).to_csv(
            outdir / "control_metrics.csv", index=False
        )

    metrics = ["baseline_iou", "adapted_iou", "delta_iou", "rer"]
    summary = {
        "status": "complete",
        "n_seeds": int(len(seed_frame)),
        "seeds": seed_frame["seed"].astype(int).tolist(),
        "mean": {name: float(seed_frame[name].mean()) for name in metrics},
        "std": {
            name: float(seed_frame[name].std(ddof=1 if len(seed_frame) > 1 else 0))
            for name in metrics
        },
        "min": {name: float(seed_frame[name].min()) for name in metrics},
        "max": {name: float(seed_frame[name].max()) for name in metrics},
        "positive_delta_iou_seeds": int((seed_frame["delta_iou"] > 0).sum()),
        "positive_rer_seeds": int((seed_frame["rer"] > 0).sum()),
        "abstained_seeds": int((seed_frame["selected_alpha"] == 0).sum()),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# PILD-GEO4 natural-proportion probe",
        "",
        f"- completed seeds: {summary['seeds']}",
        f"- mean delta IoU: {summary['mean']['delta_iou']:.6f}",
        f"- mean RER: {summary['mean']['rer']:.2%}",
        "",
        "## Seed results",
        "",
        seed_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Per-dataset results",
        "",
        dataset_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Per-dataset seed summary",
        "",
        dataset_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
