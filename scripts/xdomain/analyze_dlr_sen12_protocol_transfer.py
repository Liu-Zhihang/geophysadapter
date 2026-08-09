#!/usr/bin/env python3
"""Aggregate DLR tests of the frozen Sen12 Terrain-expert routing protocol."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FIXED_CONFIG = (0.3, 0.7, 4.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


def aggregate(rows: list[dict]) -> dict:
    values = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "visual_tp", "visual_fp", "visual_fn", "visual_errors",
            "adapted_tp", "adapted_fp", "adapted_fn", "adapted_errors",
            "corrected", "harmed",
        )
    }
    visual_iou = values["visual_tp"] / max(
        values["visual_tp"] + values["visual_fp"] + values["visual_fn"], 1
    )
    adapted_iou = values["adapted_tp"] / max(
        values["adapted_tp"] + values["adapted_fp"] + values["adapted_fn"], 1
    )
    return {
        **values,
        "visual_iou": visual_iou,
        "adapted_iou": adapted_iou,
        "delta_iou": adapted_iou - visual_iou,
        "relative_iou_gain": (adapted_iou - visual_iou) / max(visual_iou, 1e-12),
        "rer": (values["visual_errors"] - values["adapted_errors"])
        / max(values["visual_errors"], 1),
        "corrected_to_harmed": values["corrected"] / max(values["harmed"], 1),
    }


def load_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(runs_dir.glob("seed*/fold*/gate_test/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "confirmatory_fixed_configuration":
            raise RuntimeError(f"not a fixed one-shot test: {path}")
        gate = payload["grid"][0]
        observed = tuple(
            float(gate[key])
            for key in ("low_threshold", "high_threshold", "alpha", "visual_margin")
        )
        if observed != FIXED_CONFIG:
            raise RuntimeError(f"gate mismatch {observed}: {path}")
        baseline = payload["baseline"]
        seed = int(path.parents[2].name.removeprefix("seed"))
        fold = int(path.parents[1].name.removeprefix("fold"))
        rows.append(
            {
                "seed": seed,
                "fold": fold,
                "regions": ",".join(payload.get("regions", [])),
                "visual_tp": int(baseline["tp"]),
                "visual_fp": int(baseline["fp"]),
                "visual_fn": int(baseline["fn"]),
                "visual_errors": int(baseline["errors"]),
                "adapted_tp": int(gate["tp"]),
                "adapted_fp": int(gate["fp"]),
                "adapted_fn": int(gate["fn"]),
                "adapted_errors": int(gate["errors"]),
                "visual_iou": float(baseline["iou"]),
                "adapted_iou": float(gate["iou"]),
                "delta_iou": float(gate["delta_iou"]),
                "rer": float(gate["rer"]),
                "corrected": int(gate["corrected"]),
                "harmed": int(gate["harmed"]),
                "result_path": str(path.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no fixed-test results under {runs_dir}")
    keys = [(row["seed"], row["fold"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate seed/fold results")
    return rows


def bootstrap(rows: list[dict], reps: int, seed: int) -> dict:
    seeds = sorted({row["seed"] for row in rows})
    grouped = {value: [row for row in rows if row["seed"] == value] for value in seeds}
    rng = np.random.default_rng(seed)
    delta = np.empty(reps, dtype=np.float64)
    rer = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sampled = []
        for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
            source = grouped[int(sampled_seed)]
            sampled.extend(source[position] for position in rng.integers(0, len(source), len(source)))
        current = aggregate(sampled)
        delta[index] = current["delta_iou"]
        rer[index] = current["rer"]
    return {
        "method": (
            "hierarchical seed/fold bootstrap" if len(seeds) > 1 else "fold bootstrap"
        ),
        "reps": reps,
        "delta_iou_ci95": np.quantile(delta, (0.025, 0.975)).tolist(),
        "rer_ci95": np.quantile(rer, (0.025, 0.975)).tolist(),
    }


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.runs_dir)
    with (args.outdir / "per_seed_fold.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    seeds = sorted({row["seed"] for row in rows})
    folds_by_seed: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        folds_by_seed[row["seed"]].append(row["fold"])
    pooled = aggregate(rows)
    uncertainty = bootstrap(rows, args.bootstrap_reps, args.bootstrap_seed)
    positive_delta = sum(row["delta_iou"] > 0 for row in rows)
    positive_rer = sum(row["rer"] > 0 for row in rows)
    gate_pass = (
        pooled["delta_iou"] > 0
        and pooled["rer"] > 0
        and pooled["corrected"] > pooled["harmed"]
        and positive_delta >= (len(rows) // 2 + 1)
    )
    summary = {
        "status": "complete",
        "scientific_status": "exploratory cross-dataset protocol transfer",
        "contract": (
            "DLR event-isolated nested 5-fold; Prithvi visual + independent common9 Terrain expert; "
            "Sen12 gate frozen without DLR test tuning"
        ),
        "fixed_config": FIXED_CONFIG,
        "seeds": seeds,
        "folds_by_seed": {str(key): sorted(value) for key, value in folds_by_seed.items()},
        "n_units": len(rows),
        "positive_delta_iou_units": positive_delta,
        "positive_rer_units": positive_rer,
        "pooled": pooled,
        "bootstrap": uncertainty,
        "expansion_gate_pass": gate_pass,
        "expansion_gate": (
            "pooled DeltaIoU>0, RER>0, corrected>harmed, and a strict majority of units positive"
        ),
        "boundary": (
            "A positive result transfers the Sen12 mechanism to DLR common Terrain support; "
            "it does not establish Material/Trigger effects or a universal average-IoU gain."
        ),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    low_d, high_d = uncertainty["delta_iou_ci95"]
    low_r, high_r = uncertainty["rer_ci95"]
    report = f"""# DLR transfer of the frozen Sen12 Terrain protocol

- Scientific status: exploratory cross-dataset protocol transfer.
- Units: `{len(rows)}` seed-fold tests; seeds `{seeds}`.
- Pooled visual IoU: `{pooled['visual_iou']:.6f}`.
- Pooled adapted IoU: `{pooled['adapted_iou']:.6f}`.
- Pooled DeltaIoU: `{pooled['delta_iou']:+.6f}` ({pooled['relative_iou_gain']:+.2%}); CI95 `[{low_d:+.6f}, {high_d:+.6f}]`.
- Pooled RER: `{pooled['rer']:+.2%}`; CI95 `[{low_r:+.2%}, {high_r:+.2%}]`.
- Corrected/harmed: `{pooled['corrected']}/{pooled['harmed']} = {pooled['corrected_to_harmed']:.3f}`.
- Positive units: DeltaIoU `{positive_delta}/{len(rows)}`; RER `{positive_rer}/{len(rows)}`.
- Expansion gate: `{'PASS' if gate_pass else 'FAIL'}`.

The routing thresholds are the frozen Sen12 values `(0.3, 0.7, 4.0, 1.0)` and were not tuned on DLR test labels. DLR uses the audited global common-9 Terrain contract rather than Sen12 native-17, so this tests mechanism portability under stricter input availability, not exact data parity.
"""
    (args.outdir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
