#!/usr/bin/env python3
"""Summarize nested-fold support-only results across independent seeds."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


def interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    if values.size == 1:
        value = float(values[0])
        return [value, value]
    draws = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    return [float(item) for item in np.quantile(draws, [0.025, 0.975])]


def sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    permutations = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=np.float64)))
            for signs in itertools.product((-1.0, 1.0), repeat=values.size)
        ]
    )
    return float(np.mean(np.abs(permutations) >= observed - 1e-15))


def describe(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "bootstrap_ci95_mean": interval(array, rng),
        "exact_sign_flip_p": sign_flip_p(array),
        "n_positive": int(np.sum(array > 0)),
        "n_zero": int(np.sum(array == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()

    summaries: dict[int, dict[str, Any]] = {}
    controls: dict[int, dict[str, Any]] = {}
    for path in sorted(args.runs_root.glob("summary_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries[int(payload["seed"])] = payload
    for path in sorted(args.runs_root.glob("controls_v2_summary_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        controls[int(payload["seed"])] = payload
    common = sorted(set(summaries) & set(controls))
    if not common:
        raise RuntimeError("no common seed summaries and control summaries")
    if set(summaries) != set(controls):
        raise RuntimeError(
            f"seed mismatch: summaries={sorted(summaries)} controls={sorted(controls)}"
        )

    rows: list[dict[str, Any]] = []
    for seed in common:
        pooled = summaries[seed]["pooled"]
        aligned = controls[seed]["conditions"]["aligned"]
        if abs(float(pooled["delta_iou"]) - float(aligned["delta_iou"])) > 1e-12:
            raise RuntimeError(f"aligned control disagrees with primary result for seed {seed}")
        rows.append(
            {
                "seed": seed,
                "baseline_iou": float(pooled["baseline_iou"]),
                "adapted_iou": float(pooled["adapted_iou"]),
                "delta_iou": float(pooled["delta_iou"]),
                "rer": float(pooled["rer"]),
                "corrected_to_harmed": float(pooled["corrected_to_harmed"]),
                "positive_delta_iou_folds": int(pooled["positive_delta_iou_folds"]),
                "positive_rer_folds": int(pooled["positive_rer_folds"]),
            }
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    metrics = {
        key: describe([float(row[key]) for row in rows], rng)
        for key in ("baseline_iou", "adapted_iou", "delta_iou", "rer", "corrected_to_harmed")
    }
    conditions = sorted(controls[common[0]]["conditions"])
    control_contrasts: dict[str, Any] = {}
    for condition in conditions:
        if condition == "aligned":
            continue
        delta_values = [
            float(controls[seed]["conditions"]["aligned"]["delta_iou"])
            - float(controls[seed]["conditions"][condition]["delta_iou"])
            for seed in common
        ]
        rer_values = [
            float(controls[seed]["conditions"]["aligned"]["rer"])
            - float(controls[seed]["conditions"][condition]["rer"])
            for seed in common
        ]
        control_contrasts[condition] = {
            "aligned_minus_control_delta_iou": describe(delta_values, rng),
            "aligned_minus_control_rer": describe(rer_values, rng),
        }

    payload = {
        "status": "complete",
        "scientific_status": "multi-seed exploratory nested event quick gate",
        "seeds": common,
        "rows": rows,
        "metrics": metrics,
        "control_contrasts": control_contrasts,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Support-only Terrain multi-seed quick-gate summary",
        "",
        f"- Seeds: `{', '.join(map(str, common))}`",
        f"- Mean DeltaIoU: `{metrics['delta_iou']['mean']:.6f}` "
        f"(95% seed-bootstrap CI "
        f"`[{metrics['delta_iou']['bootstrap_ci95_mean'][0]:.6f}, "
        f"{metrics['delta_iou']['bootstrap_ci95_mean'][1]:.6f}]`; "
        f"exact sign-flip p=`{metrics['delta_iou']['exact_sign_flip_p']:.6f}`)",
        f"- Mean RER: `{100 * metrics['rer']['mean']:.3f}%` "
        f"(95% seed-bootstrap CI "
        f"`[{100 * metrics['rer']['bootstrap_ci95_mean'][0]:.3f}%, "
        f"{100 * metrics['rer']['bootstrap_ci95_mean'][1]:.3f}%]`)",
        "",
        "| Seed | Visual IoU | Adapted IoU | DeltaIoU | RER | Corrected/Harmed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['baseline_iou']:.6f} | "
            f"{row['adapted_iou']:.6f} | {row['delta_iou']:+.6f} | "
            f"{100 * row['rer']:+.3f}% | {row['corrected_to_harmed']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Aligned minus falsification controls",
            "",
            "| Control | DeltaIoU contrast | Positive seeds | RER contrast | Positive seeds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, item in control_contrasts.items():
        delta = item["aligned_minus_control_delta_iou"]
        rer = item["aligned_minus_control_rer"]
        lines.append(
            f"| {condition} | {delta['mean']:+.6f} | {delta['n_positive']}/{delta['n']} | "
            f"{100 * rer['mean']:+.3f}% | {rer['n_positive']}/{rer['n']} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
