#!/usr/bin/env python3
"""Summarize nested-OOF Terrain proposer results without averaging fold metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def iou(tp: int, fp: int, fn: int) -> float:
    return tp / max(tp + fp + fn, 1)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: sum(int(row[key]) for row in rows) for key in COUNT_KEYS}
    baseline_iou = iou(
        counts["baseline_tp"], counts["baseline_fp"], counts["baseline_fn"]
    )
    adapted_iou = iou(
        counts["adapted_tp"], counts["adapted_fp"], counts["adapted_fn"]
    )
    baseline_errors = counts["baseline_fp"] + counts["baseline_fn"]
    return {
        **counts,
        "n_inner_folds": len(rows),
        "n_samples": sum(int(row["n_samples"]) for row in rows),
        "n_events": sum(int(row["n_events"]) for row in rows),
        "baseline_iou": baseline_iou,
        "adapted_iou": adapted_iou,
        "delta_iou": adapted_iou - baseline_iou,
        "rer": (counts["corrected"] - counts["harmed"])
        / max(baseline_errors, 1),
        "corrected_to_harmed": counts["corrected"] / max(counts["harmed"], 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    result_paths = sorted(
        args.run_root.glob(
            "source_stratified_*__inner_*/seed*/dualthreshold_test/result.json"
        )
    )
    if not result_paths:
        raise FileNotFoundError(f"no nested OOF results below {args.run_root}")

    rows: list[dict[str, Any]] = []
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload["test"]
        fold_id = str(payload["fold_id"])
        outer_id = fold_id.split("__inner_", 1)[0]
        rows.append(
            {
                "fold_id": fold_id,
                "outer_fold_id": outer_id,
                "path": str(path.resolve()),
                "delta_ap": float(test["delta_ap"]),
                **{
                    key: int(test[key])
                    for key in COUNT_KEYS
                },
                **{
                    key: test[key]
                    for key in (
                        "n_samples",
                        "n_events",
                        "baseline_iou",
                        "adapted_iou",
                        "delta_iou",
                        "rer",
                    )
                },
            }
        )

    by_outer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_outer[row["outer_fold_id"]].append(row)
    outer = {key: aggregate(value) for key, value in sorted(by_outer.items())}
    overall = aggregate(rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "pild_nested_oof_proposer_summary.v1",
        "interpretation": (
            "OOF proposer performance is a routing-training diagnostic, not an "
            "outer-test manuscript result."
        ),
        "folds": rows,
        "outer_train_oof": outer,
        "all_development_oof": overall,
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PILD nested-OOF Terrain proposer audit",
        "",
        "These are inner-test predictions for routing development. They are not "
        "outer-test manuscript results.",
        "",
        "| inner fold | n | events | baseline IoU | delta IoU | RER | delta AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold_id']} | {row['n_samples']} | {row['n_events']} | "
            f"{row['baseline_iou']:.4f} | {row['delta_iou']:+.4f} | "
            f"{row['rer']:+.2%} | {row['delta_ap']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Count-pooled outer-train OOF",
            "",
            "| outer fold | n | events | baseline IoU | delta IoU | RER | corrected/harmed |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold_id, row in outer.items():
        lines.append(
            f"| {fold_id} | {row['n_samples']} | {row['n_events']} | "
            f"{row['baseline_iou']:.4f} | {row['delta_iou']:+.4f} | "
            f"{row['rer']:+.2%} | {row['corrected_to_harmed']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Gate decision",
            "",
            "The fixed Terrain proposer is not independently deployable: no inner "
            "fold jointly improved IoU, RER, and AP. It may only be used to "
            "generate OOF rescue/veto candidates for an abstaining router.",
            "",
        ]
    )
    (args.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.outdir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
