#!/usr/bin/env python3
"""Evaluate the frozen Sen12 dual-threshold Terrain rule on PILD-GEO4 DINO."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from audit_pild_support_only_controls_v1 import CONDITIONS, ControlledTerrainDataset
from evaluate_pild_geo4_twin_dino_terrain_v1 import (
    load_terrain,
    load_visual,
    validate_baseline_replay,
)
from evaluate_pild_support_only_additive_v1 import (
    BinaryHistogram,
    EvaluationDataset,
    aggregate_pair_counts,
    aggregate_samples_to_events,
    counts_from_predictions,
    metrics_from_pair_counts,
    visual_and_terrain_logits,
    write_csv,
)
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file
from train_pild_geo4_twin_dinov2_v1 import set_seed
from train_pild_sen12_roleaware_v1 import validate_protocol_schema
from train_pild_support_only_terrain_v1 import write_json


@torch.no_grad()
def evaluate(
    visual: torch.nn.Module,
    terrain: torch.nn.Module,
    loader: DataLoader,
    *,
    threshold: float,
    low: float,
    high: float,
    alpha: float,
    visual_margin: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_histogram = BinaryHistogram()
    adapted_histogram = BinaryHistogram()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual, terrain, batch, device=device
        )
        visual_probability = torch.sigmoid(visual_logits)
        terrain_probability = torch.sigmoid(terrain_logits)
        visual_positive = visual_probability >= threshold
        near_boundary = (visual_probability - threshold).abs() <= visual_margin
        veto = (
            ((low - terrain_probability) / max(low, 1e-6)).clamp(0.0, 1.0)
            * visual_positive
            * near_boundary
            * q_t
        )
        rescue = (
            ((terrain_probability - high) / max(1.0 - high, 1e-6)).clamp(0.0, 1.0)
            * (~visual_positive)
            * near_boundary
            * q_t
        )
        adapted_logits = visual_logits + alpha * (rescue - veto)
        adapted_probability = torch.sigmoid(adapted_logits)
        baseline_histogram.update(visual_probability, target, valid)
        adapted_histogram.update(adapted_probability, target, valid)
        for index in range(target.shape[0]):
            pair = counts_from_predictions(
                visual_probability[index : index + 1],
                adapted_probability[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                threshold=threshold,
            )
            rows.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "dataset_id": str(batch["dataset_id"][index]),
                    "source_id": str(batch["source_id"][index]),
                    "source_event_id": str(batch["source_event_id"][index]),
                    "canonical_event_id": str(batch["canonical_event_id"][index]),
                    **pair,
                    **metrics_from_pair_counts(pair),
                }
            )
    event_rows = aggregate_samples_to_events(rows)
    counts = aggregate_pair_counts(rows)
    corpus = {
        **counts,
        **metrics_from_pair_counts(counts),
        "baseline_ap": baseline_histogram.average_precision(),
        "adapted_ap": adapted_histogram.average_precision(),
        "n_samples": len(rows),
        "n_events": len(event_rows),
    }
    corpus["delta_ap"] = corpus["adapted_ap"] - corpus["baseline_ap"]
    return rows, event_rows, corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-summary", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-per-sample", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--low", type=float, default=0.3)
    parser.add_argument("--high", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--visual-margin", type=float, default=1.0)
    parser.add_argument("--max-replay-iou-drift", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.low < args.high <= 1:
        raise ValueError("expected 0 <= low < high <= 1")
    if args.alpha < 0 or args.visual_margin < 0:
        raise ValueError("alpha and visual-margin must be non-negative")

    schema = validate_protocol_schema(
        args.manifest, args.protocol_summary, args.split, args.fold_id
    )
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    stage = outdir.parent / f".{outdir.name}.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    (stage / "run.log").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    started = time.time()
    set_seed(args.seed)
    device = torch.device(args.device)
    visual, visual_payload, visual_receipt = load_visual(
        args.visual_checkpoint,
        schema=schema,
        fold_id=args.fold_id,
        seed=args.seed,
        device=device,
    )
    terrain, terrain_payload, terrain_receipt = load_terrain(
        args.terrain_checkpoint,
        schema=schema,
        protocol_summary=args.protocol_summary,
        fold_id=args.fold_id,
        seed=args.seed,
        device=device,
    )
    normalization = terrain_payload["normalization"]
    base = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.split,
        fold_id=args.fold_id,
        role="test",
        readiness="core",
    )
    dataset = EvaluationDataset(
        base, mean=normalization["mean"], scale=normalization["scale"]
    )
    condition_results: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    aligned_events: list[dict[str, Any]] = []
    replay: dict[str, Any] | None = None
    for condition in CONDITIONS:
        loader = DataLoader(
            ControlledTerrainDataset(dataset, condition),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        rows, events, corpus = evaluate(
            visual,
            terrain,
            loader,
            threshold=float(visual_payload["threshold"]),
            low=args.low,
            high=args.high,
            alpha=args.alpha,
            visual_margin=args.visual_margin,
            device=device,
        )
        if condition == "aligned":
            replay = validate_baseline_replay(
                rows,
                args.visual_per_sample,
                max_iou_drift=args.max_replay_iou_drift,
            )
            aligned_rows, aligned_events = rows, events
        condition_results.append({"condition": condition, **corpus})
    write_csv(stage / "per_sample_metrics.csv", aligned_rows)
    write_csv(stage / "per_event_metrics.csv", aligned_events)
    aligned = condition_results[0]
    result = {
        "schema_version": "pild_geo4_twin_dino_dualthreshold_result.v1",
        "status": "COMPLETE",
        "artifact_state": "external_fixed_rule_transfer",
        "fixed_config": {
            "low": args.low,
            "high": args.high,
            "alpha": args.alpha,
            "visual_margin": args.visual_margin,
            "source": "Sen12 five-seed validation-frozen confirmation",
        },
        "fold_id": args.fold_id,
        "seed": args.seed,
        "test": aligned,
        "conditions": condition_results,
        "visual_replay": replay,
        "visual_receipt": visual_receipt,
        "terrain_receipt": terrain_receipt,
        "elapsed_seconds": time.time() - started,
    }
    write_json(stage / "result.json", result)
    write_json(
        stage / "DONE.json",
        {
            "status": "COMPLETE",
            "result_sha256": sha256_file(stage / "result.json"),
            "per_sample_sha256": sha256_file(stage / "per_sample_metrics.csv"),
            "per_event_sha256": sha256_file(stage / "per_event_metrics.csv"),
        },
    )
    os.replace(stage, outdir)
    print(
        f"completed {args.fold_id}: delta_iou={aligned['delta_iou']:+.6f}, "
        f"rer={aligned['rer']:+.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
