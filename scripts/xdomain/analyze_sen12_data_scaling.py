#!/usr/bin/env python3
"""Aggregate the frozen Sen12 25/50/75/100% data-scaling experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--full-reference-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def path_for(args, fraction, fold):
    if fraction == 1.0:
        return args.full_reference_dir / f"fold{fold}_test" / "result.json"
    tag = f"fraction{int(round(fraction * 100)):03d}"
    return args.runs_dir / tag / f"seed{args.seed}" / f"fold{fold}" / "gate_test" / "result.json"


def aggregate(rows):
    sums = {key: sum(int(row[key]) for row in rows) for key in (
        "visual_tp", "visual_fp", "visual_fn", "visual_errors",
        "adapted_tp", "adapted_fp", "adapted_fn", "adapted_errors",
        "corrected", "harmed",
    )}
    vi = sums["visual_tp"] / max(sums["visual_tp"] + sums["visual_fp"] + sums["visual_fn"], 1)
    ai = sums["adapted_tp"] / max(sums["adapted_tp"] + sums["adapted_fp"] + sums["adapted_fn"], 1)
    return {
        **sums,
        "visual_iou": vi,
        "adapted_iou": ai,
        "delta_iou": ai - vi,
        "relative_iou_gain": (ai - vi) / max(vi, 1e-12),
        "rer": (sums["visual_errors"] - sums["adapted_errors"]) / max(sums["visual_errors"], 1),
        "corrected_to_harmed": sums["corrected"] / max(sums["harmed"], 1),
    }


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fraction in FRACTIONS:
        for fold in range(5):
            path = path_for(args, fraction, fold)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "confirmatory_fixed_configuration":
                raise RuntimeError(f"invalid status: {path}")
            base, adapted = payload["baseline"], payload["grid"][0]
            rows.append({
                "fraction": fraction,
                "fold": fold,
                "regions": ",".join(payload.get("regions", [])),
                "visual_tp": base["tp"], "visual_fp": base["fp"], "visual_fn": base["fn"],
                "visual_errors": base["errors"],
                "adapted_tp": adapted["tp"], "adapted_fp": adapted["fp"], "adapted_fn": adapted["fn"],
                "adapted_errors": adapted["errors"], "corrected": adapted["corrected"], "harmed": adapted["harmed"],
                "visual_iou": base["iou"], "adapted_iou": adapted["iou"],
                "delta_iou": adapted["delta_iou"], "rer": adapted["rer"],
                "result_path": str(path.resolve()),
            })
    with (args.outdir / "per_fraction_fold.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    by_fraction = {
        f"{fraction:.2f}": aggregate([row for row in rows if row["fraction"] == fraction])
        for fraction in FRACTIONS
    }
    delta_values = [by_fraction[f"{fraction:.2f}"]["delta_iou"] for fraction in FRACTIONS]
    summary = {
        "status": "complete",
        "contract": "one-seed exploratory nested region-stratified Sen12 scaling; fixed validation/test and fixed Terrain gate",
        "seed": args.seed,
        "fractions": by_fraction,
        "delta_iou_monotonic_non_decreasing": all(a <= b for a, b in zip(delta_values, delta_values[1:])),
        "warning": "This first scaling curve uses one optimization seed; promotion requires replication if a trend is observed.",
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Sen12 nested data-scaling experiment", "", "| Train fraction | Visual IoU | Adapted IoU | DeltaIoU | Relative gain | RER |", "|---:|---:|---:|---:|---:|---:|"]
    for fraction in FRACTIONS:
        row = by_fraction[f"{fraction:.2f}"]
        lines.append(f"| {fraction:.0%} | {row['visual_iou']:.5f} | {row['adapted_iou']:.5f} | {row['delta_iou']:+.5f} | {row['relative_iou_gain']:+.2%} | {row['rer']:+.2%} |")
    lines += ["", f"Monotonic DeltaIoU: `{summary['delta_iou_monotonic_non_decreasing']}`.", "", summary["warning"]]
    (args.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
