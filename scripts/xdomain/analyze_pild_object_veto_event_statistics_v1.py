#!/usr/bin/env python3
"""Event- and source-level statistics for an object-level veto decision.

Pooled pixel counts are dominated by whichever event contributes the most area, so the
endpoint reviewers care about is the event macro average with an event-clustered
interval. This script recomputes every confusion count from the prediction caches, then
applies the frozen removal decisions, so false negatives are exact per event rather than
inferred from the predicted side alone.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)


def iou(tp: float, fp: float, fn: float) -> float:
    return float(tp / max(tp + fp + fn, 1.0))


def accumulate(
    decisions: pd.DataFrame, cache_dir: Path, min_area: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    removed_lookup: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in decisions[decisions.removed].itertuples():
        removed_lookup[(str(row.fold_id), str(row.sample_id))].add(int(row.component_id))
    thresholds = {
        fold_id: float(
            json.loads(
                (cache_dir / f"{fold_id}_oof_cache_receipt.json").read_text(
                    encoding="utf-8"
                )
            )["threshold"]
        )
        for fold_id in sorted(decisions.fold_id.unique())
    }
    structure = ndimage.generate_binary_structure(2, 2)
    records: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "baseline_tp": 0.0,
            "baseline_fp": 0.0,
            "baseline_fn": 0.0,
            "adapted_tp": 0.0,
            "adapted_fp": 0.0,
            "adapted_fn": 0.0,
            "n_samples": 0.0,
        }
    )
    dataset_of: dict[str, str] = {}

    for fold_id, threshold in thresholds.items():
        with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as h:
            sample_id = [str(item) for item in h["sample_id"]]
            dataset_id = [str(item) for item in h["dataset_id"]]
            event_id = [str(item) for item in h["canonical_event_id"]]
            probability = h["visual_probability"]
            target = h["target"]
            valid = h["valid"]
        for index, name in enumerate(sample_id):
            keep = valid[index].astype(bool)
            truth = target[index].astype(bool) & keep
            predicted = (probability[index].astype(np.float32) >= threshold) & keep
            adapted = predicted.copy()
            drop = removed_lookup.get((fold_id, name))
            if drop and predicted.any():
                labels, count = ndimage.label(predicted, structure=structure)
                if count:
                    remove_mask = np.isin(labels, list(drop))
                    adapted = adapted & ~remove_mask
            event = event_id[index]
            dataset_of[event] = dataset_id[index]
            record = records[event]
            record["n_samples"] += 1.0
            record["baseline_tp"] += float(np.count_nonzero(predicted & truth))
            record["baseline_fp"] += float(np.count_nonzero(predicted & ~truth))
            record["baseline_fn"] += float(np.count_nonzero(~predicted & truth))
            record["adapted_tp"] += float(np.count_nonzero(adapted & truth))
            record["adapted_fp"] += float(np.count_nonzero(adapted & ~truth))
            record["adapted_fn"] += float(np.count_nonzero(~adapted & truth))

    rows = []
    for event, record in records.items():
        baseline = iou(record["baseline_tp"], record["baseline_fp"], record["baseline_fn"])
        adapted = iou(record["adapted_tp"], record["adapted_fp"], record["adapted_fn"])
        baseline_errors = record["baseline_fp"] + record["baseline_fn"]
        adapted_errors = record["adapted_fp"] + record["adapted_fn"]
        rows.append(
            {
                "canonical_event_id": event,
                "dataset_id": dataset_of[event],
                "n_samples": int(record["n_samples"]),
                **{key: value for key, value in record.items() if key != "n_samples"},
                "baseline_iou": baseline,
                "adapted_iou": adapted,
                "delta_iou": adapted - baseline,
                "baseline_errors": baseline_errors,
                "adapted_errors": adapted_errors,
                "rer": float(
                    (baseline_errors - adapted_errors) / max(baseline_errors, 1.0)
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("canonical_event_id").reset_index(drop=True)
    pooled = {
        key: float(frame[key].sum())
        for key in (
            "baseline_tp",
            "baseline_fp",
            "baseline_fn",
            "adapted_tp",
            "adapted_fp",
            "adapted_fn",
        )
    }
    pooled_baseline = iou(pooled["baseline_tp"], pooled["baseline_fp"], pooled["baseline_fn"])
    pooled_adapted = iou(pooled["adapted_tp"], pooled["adapted_fp"], pooled["adapted_fn"])
    pooled_errors = pooled["baseline_fp"] + pooled["baseline_fn"]
    pooled_adapted_errors = pooled["adapted_fp"] + pooled["adapted_fn"]
    summary = {
        "min_area": int(min_area),
        "pooled_baseline_iou": pooled_baseline,
        "pooled_adapted_iou": pooled_adapted,
        "pooled_delta_iou": pooled_adapted - pooled_baseline,
        "pooled_rer": float((pooled_errors - pooled_adapted_errors) / max(pooled_errors, 1.0)),
        **pooled,
    }
    return frame, summary


def bootstrap(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return float("nan"), float("nan")
    draws = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_test_p(values: np.ndarray) -> float:
    """Exact two-sided sign test over events, ignoring exact zeros."""
    from math import comb

    non_zero = values[values != 0]
    n = non_zero.size
    if n == 0:
        return 1.0
    positive = int((non_zero > 0).sum())
    tail = min(positive, n - positive)
    total = 2.0 ** n
    cumulative = sum(comb(n, k) for k in range(tail + 1))
    return float(min(1.0, 2.0 * cumulative / total))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--label", default="aligned")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(args.decisions)
    frame, summary = accumulate(decisions, args.cache_dir.resolve(), args.min_area)
    frame.to_csv(outdir / f"event_metrics_{args.label}.csv", index=False)

    delta = frame.delta_iou.to_numpy(dtype=float)
    rer = frame.rer.to_numpy(dtype=float)
    low_delta, high_delta = bootstrap(delta, args.iterations, args.seed)
    low_rer, high_rer = bootstrap(rer, args.iterations, args.seed + 1)

    source_rows = []
    for dataset, part in frame.groupby("dataset_id"):
        values = part.delta_iou.to_numpy(dtype=float)
        source_rows.append(
            {
                "dataset_id": str(dataset),
                "n_events": int(len(part)),
                "event_macro_delta_iou": float(values.mean()),
                "event_macro_rer": float(part.rer.mean()),
                "positive_events": int((values > 0).sum()),
            }
        )
    source_frame = pd.DataFrame(source_rows)
    source_frame.to_csv(outdir / f"source_metrics_{args.label}.csv", index=False)

    payload = {
        "schema_version": "pild_object_veto_event_statistics.v1",
        "label": args.label,
        "decisions": str(args.decisions),
        "n_events": int(len(frame)),
        "pooled": summary,
        "event_macro_delta_iou": float(delta.mean()),
        "event_macro_delta_iou_ci95": [low_delta, high_delta],
        "event_macro_rer": float(rer.mean()),
        "event_macro_rer_ci95": [low_rer, high_rer],
        "positive_delta_events": int((delta > 0).sum()),
        "negative_delta_events": int((delta < 0).sum()),
        "zero_delta_events": int((delta == 0).sum()),
        "positive_rer_events": int((rer > 0).sum()),
        "delta_sign_test_p": sign_test_p(delta),
        "rer_sign_test_p": sign_test_p(rer),
        "source_macro_delta_iou": float(source_frame.event_macro_delta_iou.mean()),
        "source_macro_rer": float(source_frame.event_macro_rer.mean()),
        "per_source": source_rows,
    }
    (outdir / f"summary_{args.label}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )

    print(f"=== event statistics [{args.label}] ===")
    print(
        f"  pooled   : {summary['pooled_baseline_iou']:.5f} -> "
        f"{summary['pooled_adapted_iou']:.5f}  dIoU={summary['pooled_delta_iou']:+.5f} "
        f"RER={summary['pooled_rer']:+.2%}"
    )
    print(
        f"  event    : dIoU={delta.mean():+.5f} CI95=[{low_delta:+.5f},{high_delta:+.5f}] "
        f"positive={int((delta > 0).sum())}/{len(frame)} p={payload['delta_sign_test_p']:.4f}"
    )
    print(
        f"  event RER: {rer.mean():+.2%} CI95=[{low_rer:+.2%},{high_rer:+.2%}] "
        f"positive={int((rer > 0).sum())}/{len(frame)} p={payload['rer_sign_test_p']:.4f}"
    )
    print(f"  source   : dIoU={payload['source_macro_delta_iou']:+.5f} RER={payload['source_macro_rer']:+.2%}")
    print(source_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
