#!/usr/bin/env python3
"""Pool non-overlapping nested-fold support-only Terrain evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


COUNT_KEYS = (
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
    "valid_pixels",
)


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "errors": float(fp + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for path in sorted(args.runs_root.glob(f"*/seed{args.seed}/additive/result.json")):
        done = path.parent / "DONE.json"
        per_event = path.parent / "per_event_metrics.csv"
        if not done.is_file() or not per_event.is_file():
            raise RuntimeError(f"incomplete evaluation next to {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload["test"]
        row = {
            "fold_id": payload["fold_id"],
            "seed": payload["seed"],
            "selection": payload["selection"],
            **{key: int(test[key]) for key in COUNT_KEYS},
            "baseline_iou": float(test["baseline_iou"]),
            "adapted_iou": float(test["adapted_iou"]),
            "delta_iou": float(test["delta_iou"]),
            "rer": float(test["rer"]),
            "baseline_ap": float(test["baseline_ap"]),
            "adapted_ap": float(test["adapted_ap"]),
            "n_samples": int(test["n_samples"]),
            "n_events": int(test["n_events"]),
        }
        import csv

        with per_event.open(encoding="utf-8", newline="") as stream:
            fold_events = {item["canonical_event_id"] for item in csv.DictReader(stream)}
        overlap = event_ids & fold_events
        if overlap:
            raise RuntimeError(f"test events repeat across folds: {sorted(overlap)}")
        event_ids.update(fold_events)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no completed nested evaluations below {args.runs_root}")

    pooled_counts = {key: sum(row[key] for row in rows) for key in COUNT_KEYS}
    baseline = metrics(
        pooled_counts["baseline_tp"],
        pooled_counts["baseline_fp"],
        pooled_counts["baseline_fn"],
        pooled_counts["baseline_tn"],
    )
    adapted = metrics(
        pooled_counts["adapted_tp"],
        pooled_counts["adapted_fp"],
        pooled_counts["adapted_fn"],
        pooled_counts["adapted_tn"],
    )
    corrected = pooled_counts["corrected"]
    harmed = pooled_counts["harmed"]
    baseline_errors = int(baseline["errors"])
    payload = {
        "status": "complete",
        "scientific_status": "exploratory one-seed nested event quick gate",
        "seed": args.seed,
        "n_folds": len(rows),
        "n_unique_test_events": len(event_ids),
        "n_test_samples": sum(row["n_samples"] for row in rows),
        "folds": rows,
        "pooled": {
            **pooled_counts,
            "baseline_iou": baseline["iou"],
            "adapted_iou": adapted["iou"],
            "delta_iou": adapted["iou"] - baseline["iou"],
            "rer": (corrected - harmed) / max(baseline_errors, 1),
            "corrected_to_harmed": corrected / max(harmed, 1),
            "mean_fold_delta_iou": float(np.mean([row["delta_iou"] for row in rows])),
            "positive_delta_iou_folds": sum(row["delta_iou"] > 0 for row in rows),
            "positive_rer_folds": sum(row["rer"] > 0 for row in rows),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
