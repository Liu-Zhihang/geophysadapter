#!/usr/bin/env python3
"""Pool source-stratified PILD-GEO4 test folds without averaging fold metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_pild_geo4_natural_probe_v1 import aggregate


COUNT_COLUMNS = (
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
)


def metrics_from_count_array(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline_tp = values[..., 0]
    baseline_fp = values[..., 1]
    baseline_fn = values[..., 2]
    adapted_tp = values[..., 4]
    adapted_fp = values[..., 5]
    adapted_fn = values[..., 6]
    corrected = values[..., 8]
    harmed = values[..., 9]
    baseline_iou = baseline_tp / np.maximum(
        baseline_tp + baseline_fp + baseline_fn, 1
    )
    adapted_iou = adapted_tp / np.maximum(
        adapted_tp + adapted_fp + adapted_fn, 1
    )
    baseline_errors = baseline_fp + baseline_fn
    return (
        adapted_iou - baseline_iou,
        (corrected - harmed) / np.maximum(baseline_errors, 1),
    )


def event_bootstrap(
    event_frame: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, Any]:
    if replicates < 1000:
        raise ValueError("event bootstrap requires at least 1000 replicates")
    values = event_frame[list(COUNT_COLUMNS)].to_numpy(float)
    event_delta_iou, event_rer = metrics_from_count_array(values)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(event_frame), size=(replicates, len(event_frame))
    )
    macro_delta_iou = event_delta_iou[indices].mean(axis=1)
    macro_rer = event_rer[indices].mean(axis=1)
    pooled_delta_iou, pooled_rer = metrics_from_count_array(
        values[indices].sum(axis=1)
    )

    def interval(samples: np.ndarray) -> list[float]:
        return [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ]

    return {
        "replicates": replicates,
        "seed": seed,
        "n_events": len(event_frame),
        "event_positive_delta_iou": int((event_delta_iou > 0).sum()),
        "event_positive_rer": int((event_rer > 0).sum()),
        "event_positive_both": int(
            ((event_delta_iou > 0) & (event_rer > 0)).sum()
        ),
        "event_macro_delta_iou": float(event_delta_iou.mean()),
        "event_macro_delta_iou_ci": interval(macro_delta_iou),
        "event_macro_rer": float(event_rer.mean()),
        "event_macro_rer_ci": interval(macro_rer),
        "pooled_delta_iou_ci": interval(pooled_delta_iou),
        "pooled_rer_ci": interval(pooled_rer),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", default="additive")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_dir in sorted(args.root.resolve().glob("source_stratified_*")):
        sample_path = fold_dir / f"seed{args.seed}" / args.stage / "per_sample_metrics.csv"
        if not sample_path.is_file():
            continue
        frame = pd.read_csv(sample_path)
        if frame.empty:
            raise RuntimeError(f"empty evidence file: {sample_path}")
        frame.insert(0, "fold_id", fold_dir.name)
        frames.append(frame)
        fold_rows.append({"fold_id": fold_dir.name, **aggregate(frame)})
    if len(frames) != 4:
        raise RuntimeError(f"expected four completed folds, found {len(frames)}")

    pooled = pd.concat(frames, ignore_index=True)
    duplicated = pooled.loc[pooled["sample_id"].duplicated(keep=False), "sample_id"]
    if not duplicated.empty:
        raise RuntimeError(
            "test folds are not disjoint; duplicated sample IDs: "
            + ", ".join(sorted(duplicated.astype(str).unique())[:10])
        )
    dataset_rows = [
        {"dataset_id": str(dataset_id), **aggregate(group)}
        for dataset_id, group in pooled.groupby("dataset_id", sort=True)
    ]
    event_rows = []
    for event_id, group in pooled.groupby("canonical_event_id", sort=True):
        counts = {
            key: int(group[key].sum())
            for key in COUNT_COLUMNS
        }
        event_rows.append(
            {
                "canonical_event_id": str(event_id),
                **counts,
                **aggregate(group),
            }
        )
    corpus = aggregate(pooled)
    event_frame = pd.DataFrame(event_rows)
    bootstrap = event_bootstrap(
        event_frame,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    source_macro = {
        "delta_iou": float(
            np.mean([float(row["delta_iou"]) for row in dataset_rows])
        ),
        "rer": float(np.mean([float(row["rer"]) for row in dataset_rows])),
    }
    summary = {
        "schema_version": "pild_geo4_crossfold_summary.v1",
        "status": "complete",
        "stage": args.stage,
        "seed": args.seed,
        "n_folds": len(frames),
        "corpus": corpus,
        "folds": fold_rows,
        "datasets": dataset_rows,
        "source_macro": source_macro,
        "event_bootstrap": bootstrap,
    }

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(outdir / "per_sample_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_metrics.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(outdir / "dataset_metrics.csv", index=False)
    pd.DataFrame(event_rows).to_csv(outdir / "event_metrics.csv", index=False)
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# PILD-GEO4 source-stratified cross-fold result",
        "",
        f"- samples: {corpus['n_samples']}",
        f"- events: {corpus['n_events']}",
        f"- baseline IoU: {corpus['baseline_iou']:.6f}",
        f"- adapted IoU: {corpus['adapted_iou']:.6f}",
        f"- delta IoU: {corpus['delta_iou']:+.6f}",
        f"- relative error reduction: {corpus['rer']:+.2%}",
        f"- corrected / harmed: {corpus['corrected']:.0f} / {corpus['harmed']:.0f}",
        f"- source-macro delta IoU / RER: {source_macro['delta_iou']:+.6f} / {source_macro['rer']:+.2%}",
        f"- event-macro delta IoU: {bootstrap['event_macro_delta_iou']:+.6f} "
        f"[{bootstrap['event_macro_delta_iou_ci'][0]:+.6f}, "
        f"{bootstrap['event_macro_delta_iou_ci'][1]:+.6f}]",
        f"- event-macro RER: {bootstrap['event_macro_rer']:+.2%} "
        f"[{bootstrap['event_macro_rer_ci'][0]:+.2%}, "
        f"{bootstrap['event_macro_rer_ci'][1]:+.2%}]",
        f"- positive events (delta IoU / RER / both): "
        f"{bootstrap['event_positive_delta_iou']} / "
        f"{bootstrap['event_positive_rer']} / "
        f"{bootstrap['event_positive_both']} of {bootstrap['n_events']}",
        "",
        "## Folds",
        "",
        pd.DataFrame(fold_rows).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Sources",
        "",
        pd.DataFrame(dataset_rows).to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
