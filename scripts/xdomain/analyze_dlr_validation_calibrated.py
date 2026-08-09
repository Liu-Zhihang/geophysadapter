#!/usr/bin/env python3
"""Aggregate DLR validation-calibrated Terrain intervention tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--result-glob", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


def aggregate(rows: list[dict]) -> dict:
    sums = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "visual_tp", "visual_fp", "visual_fn", "visual_errors",
            "adapted_tp", "adapted_fp", "adapted_fn", "adapted_errors",
            "corrected", "harmed",
        )
    }
    visual_iou = sums["visual_tp"] / max(sums["visual_tp"] + sums["visual_fp"] + sums["visual_fn"], 1)
    adapted_iou = sums["adapted_tp"] / max(sums["adapted_tp"] + sums["adapted_fp"] + sums["adapted_fn"], 1)
    return {
        **sums,
        "visual_iou": visual_iou,
        "adapted_iou": adapted_iou,
        "delta_iou": adapted_iou - visual_iou,
        "rer": (sums["visual_errors"] - sums["adapted_errors"]) / max(sums["visual_errors"], 1),
        "corrected_to_harmed": sums["corrected"] / max(sums["harmed"], 1),
    }


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.runs_dir.glob(args.result_glob)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "test_frozen_from_validation":
            raise RuntimeError(f"not validation-frozen: {path}")
        baseline, adapted = payload["baseline"], payload["adapted"]
        rows.append({
            "seed": int(payload["seed"]), "fold": int(payload["fold"]),
            "method": payload["method"],
            "selection": json.dumps(payload["validation_selection"], sort_keys=True),
            "visual_tp": baseline["tp"], "visual_fp": baseline["fp"],
            "visual_fn": baseline["fn"], "visual_errors": baseline["errors"],
            "adapted_tp": adapted["tp"], "adapted_fp": adapted["fp"],
            "adapted_fn": adapted["fn"], "adapted_errors": adapted["errors"],
            "corrected": adapted["corrected"], "harmed": adapted["harmed"],
            "visual_iou": baseline["iou"], "adapted_iou": adapted["iou"],
            "delta_iou": adapted["delta_iou"], "rer": adapted["rer"],
            "result_path": str(path.resolve()),
        })
    if not rows:
        raise FileNotFoundError(f"no results matching {args.result_glob}")
    keys = {(row["seed"], row["fold"]) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("duplicate seed/fold")
    with (args.outdir / "per_seed_fold.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    pooled = aggregate(rows)
    rng = np.random.default_rng(args.bootstrap_seed)
    delta = np.empty(args.bootstrap_reps)
    rer = np.empty(args.bootstrap_reps)
    for index in range(args.bootstrap_reps):
        sampled = [rows[value] for value in rng.integers(0, len(rows), len(rows))]
        current = aggregate(sampled)
        delta[index], rer[index] = current["delta_iou"], current["rer"]
    summary = {
        "status": "complete",
        "scientific_status": "exploratory validation-calibrated DLR transfer",
        "n_units": len(rows),
        "seeds": sorted({row["seed"] for row in rows}),
        "positive_delta_iou_units": sum(float(row["delta_iou"]) > 0 for row in rows),
        "positive_rer_units": sum(float(row["rer"]) > 0 for row in rows),
        "pooled": pooled,
        "fold_bootstrap": {
            "reps": args.bootstrap_reps,
            "delta_iou_ci95": np.quantile(delta, (0.025, 0.975)).tolist(),
            "rer_ci95": np.quantile(rer, (0.025, 0.975)).tolist(),
        },
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    low_d, high_d = summary["fold_bootstrap"]["delta_iou_ci95"]
    low_r, high_r = summary["fold_bootstrap"]["rer_ci95"]
    (args.outdir / "report.md").write_text(
        f"# DLR validation-calibrated Terrain transfer\n\n"
        f"- Pooled visual/adapted IoU: `{pooled['visual_iou']:.6f}` / `{pooled['adapted_iou']:.6f}`.\n"
        f"- DeltaIoU: `{pooled['delta_iou']:+.6f}`; CI95 `[{low_d:+.6f}, {high_d:+.6f}]`.\n"
        f"- RER: `{pooled['rer']:+.2%}`; CI95 `[{low_r:+.2%}, {high_r:+.2%}]`.\n"
        f"- Corrected/harmed: `{pooled['corrected_to_harmed']:.3f}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
