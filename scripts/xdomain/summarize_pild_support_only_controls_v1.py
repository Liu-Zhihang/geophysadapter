#!/usr/bin/env python3
"""Pool nested-fold Terrain falsification controls by condition."""

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


def summarize(counts: dict[str, int]) -> dict[str, float]:
    btp, bfp, bfn = counts["baseline_tp"], counts["baseline_fp"], counts["baseline_fn"]
    atp, afp, afn = counts["adapted_tp"], counts["adapted_fp"], counts["adapted_fn"]
    baseline_iou = btp / max(btp + bfp + bfn, 1)
    adapted_iou = atp / max(atp + afp + afn, 1)
    baseline_errors = bfp + bfn
    corrected, harmed = counts["corrected"], counts["harmed"]
    return {
        "baseline_iou": baseline_iou,
        "adapted_iou": adapted_iou,
        "delta_iou": adapted_iou - baseline_iou,
        "rer": (corrected - harmed) / max(baseline_errors, 1),
        "corrected_to_harmed": corrected / max(harmed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--controls-name", default="controls_v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pooled: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fold_rows: list[dict[str, Any]] = []
    for path in sorted(
        args.runs_root.glob(f"*/seed{args.seed}/{args.controls_name}/summary.json")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["conditions"]:
            condition = row["condition"]
            counts = {key: int(row[key]) for key in COUNT_KEYS}
            for key, value in counts.items():
                pooled[condition][key] += value
            fold_rows.append(
                {
                    "fold_id": payload["fold_id"],
                    "condition": condition,
                    "delta_iou": float(row["delta_iou"]),
                    "rer": float(row["rer"]),
                }
            )
    if not pooled:
        raise RuntimeError(f"no control summaries below {args.runs_root}")
    conditions = {
        condition: {**dict(counts), **summarize(dict(counts))}
        for condition, counts in sorted(pooled.items())
    }
    aligned = conditions["aligned"]
    contrasts = {
        condition: {
            "aligned_minus_control_delta_iou": aligned["delta_iou"] - row["delta_iou"],
            "aligned_minus_control_rer": aligned["rer"] - row["rer"],
        }
        for condition, row in conditions.items()
        if condition != "aligned"
    }
    payload = {
        "status": "complete",
        "scientific_status": "pooled nested-fold test-time falsification controls",
        "seed": args.seed,
        "n_folds": len({row["fold_id"] for row in fold_rows}),
        "conditions": conditions,
        "contrasts": contrasts,
        "fold_rows": fold_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
