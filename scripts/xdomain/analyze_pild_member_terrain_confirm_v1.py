#!/usr/bin/env python3
"""Aggregate matched Prithvi-plus-Terrain confirmations for one PILD member."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.replace(",", " ").split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--gate-subdir", default="gate_test")
    parser.add_argument("--allow-varying-config", action="store_true")
    parser.add_argument("--fixed-config", default="0.3,0.7,4.0,1.0")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def metrics(tp: int, fp: int, fn: int, errors: int) -> dict[str, float | int]:
    return {"iou": tp / max(tp + fp + fn, 1), "errors": int(errors)}


def aggregate(rows: list[dict]) -> dict:
    keys = (
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
    values = {key: sum(int(row[key]) for row in rows) for key in keys}
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


def load_results(
    runs_dir: Path,
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
    fixed: tuple[float, ...],
    gate_subdir: str,
    allow_varying_config: bool,
) -> tuple[list[dict], list[dict]]:
    fold_rows: list[dict] = []
    sample_rows: list[dict] = []
    for seed in seeds:
        seen_samples: set[str] = set()
        for fold in folds:
            gate_dir = runs_dir / f"seed{seed}" / f"fold{fold}" / gate_subdir
            result_path = gate_dir / "result.json"
            sample_path = gate_dir / "per_sample.csv"
            if not result_path.is_file() or not sample_path.is_file():
                raise FileNotFoundError(f"missing result or per-sample output in {gate_dir}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("status") != "confirmatory_fixed_configuration":
                raise RuntimeError(f"non-confirmatory result: {result_path}")
            adapted = payload["grid"][0]
            observed = tuple(
                float(adapted[key])
                for key in ("low_threshold", "high_threshold", "alpha", "visual_margin")
            )
            if not allow_varying_config and observed != fixed:
                raise RuntimeError(f"fixed configuration mismatch {observed}: {result_path}")
            baseline = payload["baseline"]
            fold_rows.append(
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
                    "corrected": int(adapted["corrected"]),
                    "harmed": int(adapted["harmed"]),
                    "visual_iou": float(baseline["iou"]),
                    "adapted_iou": float(adapted["iou"]),
                    "delta_iou": float(adapted["delta_iou"]),
                    "rer": float(adapted["rer"]),
                    "selected_config": ",".join(str(value) for value in observed),
                    "result_path": str(result_path.resolve()),
                }
            )
            with sample_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    sample_id = row["sample_id"]
                    if sample_id in seen_samples:
                        raise RuntimeError(f"seed={seed} repeats test sample {sample_id}")
                    seen_samples.add(sample_id)
                    sample_rows.append(
                        {
                            "seed": seed,
                            **row,
                            **{
                                key: int(row[key])
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
                            },
                        }
                    )
    return fold_rows, sample_rows


def group_events(sample_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in sample_rows:
        grouped[(int(row["seed"]), row["event_id"])].append(row)
    rows = []
    for (seed, event_id), values in sorted(grouped.items()):
        current = aggregate(values)
        rows.append(
            {
                "seed": seed,
                "event_id": event_id,
                "n_samples": len(values),
                **current,
            }
        )
    return rows


def hierarchical_bootstrap(
    event_rows: list[dict],
    seeds: tuple[int, ...],
    reps: int,
    random_seed: int,
) -> dict:
    grouped = {
        seed: [row for row in event_rows if int(row["seed"]) == seed] for seed in seeds
    }
    if any(not values for values in grouped.values()):
        raise RuntimeError("a seed has no event-level test units")
    rng = np.random.default_rng(random_seed)
    delta = np.empty(reps)
    rer = np.empty(reps)
    for index in range(reps):
        sampled = []
        for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
            source = grouped[int(sampled_seed)]
            sampled.extend(source[position] for position in rng.integers(0, len(source), len(source)))
        current = aggregate(sampled)
        delta[index] = current["delta_iou"]
        rer[index] = current["rer"]
    method = (
        "event-cluster bootstrap"
        if len(seeds) == 1
        else "hierarchical bootstrap over optimization seeds then physical events"
    )
    return {
        "method": method,
        "reps": reps,
        "delta_iou_ci95": np.quantile(delta, (0.025, 0.975)).tolist(),
        "rer_ci95": np.quantile(rer, (0.025, 0.975)).tolist(),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    seeds = parse_ints(args.seeds)
    folds = parse_ints(args.folds)
    fixed = tuple(float(value) for value in args.fixed_config.split(","))
    if len(fixed) != 4:
        raise ValueError("--fixed-config requires low,high,alpha,margin")
    args.outdir.mkdir(parents=True, exist_ok=True)
    fold_rows, sample_rows = load_results(
        args.runs_dir,
        seeds,
        folds,
        fixed,
        args.gate_subdir,
        args.allow_varying_config,
    )
    event_rows = group_events(sample_rows)
    pooled = aggregate(fold_rows)
    bootstrap = hierarchical_bootstrap(
        event_rows, seeds, args.bootstrap_reps, args.bootstrap_seed
    )
    write_csv(args.outdir / "per_seed_fold.csv", fold_rows)
    write_csv(args.outdir / "per_sample.csv", sample_rows)
    write_csv(args.outdir / "per_seed_event.csv", event_rows)
    summary = {
        "status": "complete",
        "dataset_id": args.dataset_id,
        "contract": "member-specific training; frozen Sen12 Terrain gate; event-isolated test folds",
        "fixed_config": None if args.allow_varying_config else fixed,
        "configuration_contract": (
            "varying configurations selected before test evaluation"
            if args.allow_varying_config
            else "one fixed configuration"
        ),
        "seeds": seeds,
        "folds": folds,
        "n_seed_folds": len(fold_rows),
        "n_test_samples_per_seed": len(sample_rows) // len(seeds),
        "n_physical_events_per_seed": len(event_rows) // len(seeds),
        "positive_delta_iou_folds": sum(float(row["delta_iou"]) > 0 for row in fold_rows),
        "positive_rer_folds": sum(float(row["rer"]) > 0 for row in fold_rows),
        "positive_delta_iou_events": sum(float(row["delta_iou"]) > 0 for row in event_rows),
        "positive_rer_events": sum(float(row["rer"]) > 0 for row in event_rows),
        "positive_delta_iou_samples": sum(float(row["delta_iou"]) > 0 for row in sample_rows),
        "positive_rer_samples": sum(float(row["rer"]) > 0 for row in sample_rows),
        "pooled": pooled,
        "bootstrap": bootstrap,
        "caveat": (
            "Sample/event responder counts are diagnostic only and were not used to select "
            "training or headline evaluation samples."
        ),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    low_d, high_d = bootstrap["delta_iou_ci95"]
    low_r, high_r = bootstrap["rer_ci95"]
    report = f"""# {args.dataset_id} matched Prithvi + Terrain confirmation

- Protocol: member-specific Prithvi visual and Terrain expert training; event-isolated test folds.
- Gate: frozen Sen12 configuration `{fixed}`; no test-time tuning.
- Test support per seed: `{summary['n_test_samples_per_seed']}` samples, `{summary['n_physical_events_per_seed']}` physical events.
- Pooled visual IoU: `{pooled['visual_iou']:.6f}`.
- Pooled adapted IoU: `{pooled['adapted_iou']:.6f}`.
- Pooled DeltaIoU: `{pooled['delta_iou']:+.6f}`; cluster CI95 `[{low_d:+.6f}, {high_d:+.6f}]`.
- Pooled RER: `{pooled['rer']:+.2%}`; cluster CI95 `[{low_r:+.2%}, {high_r:+.2%}]`.
- Corrected/harmed: `{pooled['corrected_to_harmed']:.3f}`.
- Positive folds: DeltaIoU `{summary['positive_delta_iou_folds']}/{len(fold_rows)}`; RER `{summary['positive_rer_folds']}/{len(fold_rows)}`.

Responder counts are post-hoc mechanism diagnostics, not a sample-selection rule. Average IoU is
reported as a completeness/no-negative-transfer check; the experiment primarily tests whether
the frozen physical intervention transfers beyond Sen12 and where that mechanism fails.
"""
    (args.outdir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
