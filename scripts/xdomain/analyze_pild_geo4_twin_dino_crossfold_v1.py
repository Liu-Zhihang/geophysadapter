#!/usr/bin/env python3
"""Pool matched Twin DINOv2-S and Terrain results across held-out folds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


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
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty metrics CSV: {path}")
    return rows


def add_counts(target: dict[str, int], row: Mapping[str, Any]) -> None:
    for key in COUNT_KEYS:
        target[key] += int(float(row[key]))


def metrics(counts: Mapping[str, int]) -> dict[str, float | int]:
    btp, bfp, bfn = (
        counts["baseline_tp"],
        counts["baseline_fp"],
        counts["baseline_fn"],
    )
    atp, afp, afn = (
        counts["adapted_tp"],
        counts["adapted_fp"],
        counts["adapted_fn"],
    )
    baseline_iou = btp / max(btp + bfp + bfn, 1)
    adapted_iou = atp / max(atp + afp + afn, 1)
    baseline_errors = bfp + bfn
    net = counts["corrected"] - counts["harmed"]
    return {
        "baseline_iou": baseline_iou,
        "adapted_iou": adapted_iou,
        "delta_iou": adapted_iou - baseline_iou,
        "baseline_errors": baseline_errors,
        "adapted_errors": afp + afn,
        "corrected": counts["corrected"],
        "harmed": counts["harmed"],
        "net_error_reduction": net,
        "rer": net / max(baseline_errors, 1),
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--evaluation-name", default="terrain_decision_margin_v1")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    pooled: dict[str, int] = defaultdict(int)
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fold_rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    fold_artifacts: list[dict[str, Any]] = []
    for fold in range(4):
        fold_id = f"source_stratified_{fold}"
        run = args.root / fold_id / f"seed{args.seed}"
        evaluation = run / args.evaluation_name
        done = evaluation / "DONE.json"
        if not done.is_file():
            raise RuntimeError(f"incomplete evaluation: {evaluation}")
        rows = read_csv(evaluation / "per_sample_metrics.csv")
        fold_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in sample_ids:
                raise RuntimeError(f"sample appears in multiple test folds: {sample_id}")
            sample_ids.add(sample_id)
            add_counts(fold_counts, row)
            add_counts(pooled, row)
            add_counts(by_source[row["dataset_id"]], row)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "n_samples": len(rows),
                **metrics(fold_counts),
            }
        )
        fold_artifacts.append(
            {
                "fold_id": fold_id,
                "evaluation": str(evaluation.resolve()),
                "n_samples": len(rows),
            }
        )

    source_rows = [
        {"dataset_id": dataset_id, **metrics(counts)}
        for dataset_id, counts in sorted(by_source.items())
    ]
    summary = {
        "schema_version": "pild_geo4_twin_dino_crossfold_summary.v1",
        "seed": args.seed,
        "evaluation_name": args.evaluation_name,
        "n_unique_test_samples": len(sample_ids),
        "pooled": metrics(pooled),
        "folds": fold_rows,
        "sources": source_rows,
        "fold_artifacts": fold_artifacts,
    }
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.outdir / "per_fold_metrics.csv", fold_rows)
    write_csv(args.outdir / "per_source_metrics.csv", source_rows)
    pooled_metrics = summary["pooled"]
    print(
        f"n={len(sample_ids)} baseline_iou={pooled_metrics['baseline_iou']:.6f} "
        f"adapted_iou={pooled_metrics['adapted_iou']:.6f} "
        f"delta_iou={pooled_metrics['delta_iou']:+.6f} "
        f"rer={pooled_metrics['rer']:+.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
