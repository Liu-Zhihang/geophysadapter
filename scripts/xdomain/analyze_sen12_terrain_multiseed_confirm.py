#!/usr/bin/env python3
"""Aggregate frozen 5-seed x 5-fold Sen12 Terrain-gate confirmations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FOLDS = tuple(range(5))
SEEDS = (20260751, 20260752, 20260753, 20260754, 20260755)
FIXED_CONFIG = (0.3, 0.7, 4.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--seed51-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    return parser.parse_args()


def metrics(tp: int, fp: int, fn: int, errors: int) -> dict[str, float]:
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "errors": int(errors),
    }


def result_path(args: argparse.Namespace, seed: int, fold: int) -> Path:
    if seed == 20260751:
        return args.seed51_dir / f"fold{fold}_test" / "result.json"
    return args.runs_dir / f"seed{seed}" / f"fold{fold}" / "gate_test" / "result.json"


def load_rows(args: argparse.Namespace) -> list[dict]:
    rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            path = result_path(args, seed, fold)
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "confirmatory_fixed_configuration":
                raise RuntimeError(f"non-confirmatory result: {path}")
            if payload.get("split") != "test" or int(payload.get("fold", -1)) != fold:
                raise RuntimeError(f"split/fold mismatch: {path}")
            adapted = payload["grid"][0]
            observed = tuple(
                float(adapted[key])
                for key in ("low_threshold", "high_threshold", "alpha", "visual_margin")
            )
            if observed != FIXED_CONFIG:
                raise RuntimeError(f"fixed configuration mismatch {observed}: {path}")
            baseline = payload["baseline"]
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "regions": ",".join(payload.get("regions", [])),
                    "visual_tp": int(baseline["tp"]),
                    "visual_fp": int(baseline["fp"]),
                    "visual_fn": int(baseline["fn"]),
                    "visual_errors": int(baseline["errors"]),
                    "adapted_tp": int(adapted["tp"]),
                    "adapted_fp": int(adapted["fp"]),
                    "adapted_fn": int(adapted["fn"]),
                    "adapted_errors": int(adapted["errors"]),
                    "visual_iou": float(baseline["iou"]),
                    "adapted_iou": float(adapted["iou"]),
                    "delta_iou": float(adapted["delta_iou"]),
                    "rer": float(adapted["rer"]),
                    "corrected": int(adapted["corrected"]),
                    "harmed": int(adapted["harmed"]),
                    "result_path": str(path.resolve()),
                }
            )
    return rows


def aggregate(rows: list[dict]) -> dict:
    values = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "visual_tp",
            "visual_fp",
            "visual_fn",
            "visual_errors",
            "adapted_tp",
            "adapted_fp",
            "adapted_fn",
            "adapted_errors",
            "corrected",
            "harmed",
        )
    }
    visual = metrics(
        values["visual_tp"], values["visual_fp"], values["visual_fn"], values["visual_errors"]
    )
    adapted = metrics(
        values["adapted_tp"], values["adapted_fp"], values["adapted_fn"], values["adapted_errors"]
    )
    return {
        **values,
        "visual_iou": visual["iou"],
        "adapted_iou": adapted["iou"],
        "delta_iou": adapted["iou"] - visual["iou"],
        "rer": (visual["errors"] - adapted["errors"]) / max(visual["errors"], 1),
        "corrected_to_harmed": values["corrected"] / max(values["harmed"], 1),
    }


def hierarchical_bootstrap(rows: list[dict], reps: int, seed: int) -> dict:
    grouped = {current: [row for row in rows if row["seed"] == current] for current in SEEDS}
    rng = np.random.default_rng(seed)
    delta, rer = np.empty(reps), np.empty(reps)
    for index in range(reps):
        sampled = []
        for sampled_seed in rng.choice(SEEDS, size=len(SEEDS), replace=True):
            source = grouped[int(sampled_seed)]
            sampled.extend(source[position] for position in rng.integers(0, len(source), len(source)))
        current = aggregate(sampled)
        delta[index] = current["delta_iou"]
        rer[index] = current["rer"]
    return {
        "method": "hierarchical bootstrap over optimization seeds then LOGO folds",
        "reps": reps,
        "delta_iou_ci95": np.quantile(delta, (0.025, 0.975)).tolist(),
        "rer_ci95": np.quantile(rer, (0.025, 0.975)).tolist(),
    }


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)
    with (args.outdir / "per_seed_fold.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    per_seed = {str(seed): aggregate([row for row in rows if row["seed"] == seed]) for seed in SEEDS}
    pooled = aggregate(rows)
    bootstrap = hierarchical_bootstrap(rows, args.bootstrap_reps, args.bootstrap_seed)
    summary = {
        "status": "complete" if len(rows) == 25 else "incomplete",
        "contract": "Sen12 LOGO-5 x 5 seeds; frozen global Terrain gate",
        "fixed_config": FIXED_CONFIG,
        "n_seed_folds": len(rows),
        "positive_delta_iou_units": sum(row["delta_iou"] > 0 for row in rows),
        "positive_rer_units": sum(row["rer"] > 0 for row in rows),
        "per_seed": per_seed,
        "pooled": pooled,
        "bootstrap": bootstrap,
        "caveat": "Seeds repeat the same geographic test folds; the interval captures optimization and fold variation, not new independent regions.",
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    low_d, high_d = bootstrap["delta_iou_ci95"]
    low_r, high_r = bootstrap["rer_ci95"]
    report = f"""# Sen12 Prithvi + Terrain multi-seed confirmation

- Contract: 5 optimization seeds x 5 LOGO folds, one global frozen gate `{FIXED_CONFIG}`.
- Pooled visual IoU: `{pooled['visual_iou']:.6f}`.
- Pooled adapted IoU: `{pooled['adapted_iou']:.6f}`.
- Pooled DeltaIoU: `{pooled['delta_iou']:+.6f}`; hierarchical CI95 `[{low_d:+.6f}, {high_d:+.6f}]`.
- Pooled RER: `{pooled['rer']:+.2%}`; hierarchical CI95 `[{low_r:+.2%}, {high_r:+.2%}]`.
- Corrected/harmed: `{pooled['corrected_to_harmed']:.3f}`.
- Positive units: DeltaIoU `{summary['positive_delta_iou_units']}/25`; RER `{summary['positive_rer_units']}/25`.

The same geographic folds are repeated across seeds; uncertainty therefore does not represent 25 independent regions.
"""
    (args.outdir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
