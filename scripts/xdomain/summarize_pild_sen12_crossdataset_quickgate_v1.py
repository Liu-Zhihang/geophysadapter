#!/usr/bin/env python3
"""Summarize one-seed cross-dataset V versus VT quick gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def test_row(payload: dict[str, Any], condition: str) -> dict[str, Any]:
    rows = [
        row
        for row in payload.get("corpus_metrics", [])
        if row.get("split") == "test"
        and row.get("condition") == condition
        and row.get("evaluation_context") == "aligned"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one aligned test row for {condition}, found {len(rows)}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for fold_dir in sorted(path for path in args.runs_root.glob("lodo_*") if path.is_dir()):
        visual_dir = fold_dir / f"V_seed{args.seed}"
        terrain_dir = fold_dir / f"VT_seed{args.seed}"
        required = [
            visual_dir / "DONE.json",
            visual_dir / "result.json",
            visual_dir / "checkpoint.pt",
            terrain_dir / "DONE.json",
            terrain_dir / "result.json",
            terrain_dir / "checkpoint.pt",
        ]
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            continue
        visual = test_row(strict_json(visual_dir / "result.json"), "V")
        terrain = test_row(strict_json(terrain_dir / "result.json"), "VT")
        reference_errors = int(terrain["reference_errors"])
        corrected = int(terrain["corrected"])
        harmed = int(terrain["harmed"])
        rows.append(
            {
                "fold_id": fold_dir.name,
                "seed": args.seed,
                "n_samples": int(terrain["n_samples"]),
                "n_events": int(terrain["n_events"]),
                "visual_iou": float(visual["iou"]),
                "terrain_iou": float(terrain["iou"]),
                "delta_iou": float(terrain["iou"]) - float(visual["iou"]),
                "visual_ap": float(visual["average_precision"]),
                "terrain_ap": float(terrain["average_precision"]),
                "delta_ap": float(terrain["average_precision"]) - float(visual["average_precision"]),
                "reference_errors": reference_errors,
                "corrected": corrected,
                "harmed": harmed,
                "net_error_reduction": corrected - harmed,
                "rer": (corrected - harmed) / max(reference_errors, 1),
                "corrected_to_harmed": corrected / max(harmed, 1),
            }
        )

    if not rows:
        raise RuntimeError(f"no complete V/VT fold pairs found under {args.runs_root}")
    payload = {
        "status": "complete",
        "scientific_status": "exploratory one-seed cross-dataset quick gate",
        "seed": args.seed,
        "n_folds": len(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
