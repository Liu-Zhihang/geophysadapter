#!/usr/bin/env python3
"""Evaluate the frozen Sen12 dual-threshold rule on unified PILD Prithvi + Terrain."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audit_pild_support_only_controls_v1 import CONDITIONS, ControlledTerrainDataset
from evaluate_pild_geo4_twin_dino_dualthreshold_v1 import evaluate
from evaluate_pild_geo4_twin_dino_terrain_v1 import validate_baseline_replay
from evaluate_pild_support_only_additive_v1 import (
    EvaluationDataset,
    build_models,
    validate_terrain_checkpoint,
    write_csv,
)
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file
from train_pild_sen12_roleaware_v1 import set_seed, validate_protocol_schema
from train_pild_support_only_terrain_v1 import (
    validate_parent_v_checkpoint,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-summary", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-id", default="event_isolated")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-per-sample", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--low", type=float, default=0.3)
    parser.add_argument("--high", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--visual-margin", type=float, default=1.0)
    parser.add_argument("--max-replay-iou-drift", type=float, default=1e-8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--prithvi-snapshot", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.low < args.high <= 1:
        raise ValueError("expected 0 <= low < high <= 1")
    if args.alpha < 0 or args.visual_margin < 0:
        raise ValueError("alpha and visual-margin must be non-negative")

    schema = validate_protocol_schema(
        args.manifest, args.protocol_summary, args.split, args.fold_id
    )
    visual_payload, visual_receipt = validate_parent_v_checkpoint(
        args.visual_checkpoint,
        manifest_sha256=schema["manifest_sha256"],
        split_sha256=schema["split_sha256"],
        fold_id=args.fold_id,
        seed=args.seed,
    )
    terrain_payload, terrain_receipt = validate_terrain_checkpoint(
        args.terrain_checkpoint,
        schema=schema,
        protocol_summary_path=args.protocol_summary,
        fold_id=args.fold_id,
        seed=args.seed,
        parent_receipt=visual_receipt,
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
    visual, terrain, provenance = build_models(
        parent=visual_payload,
        terrain_checkpoint=terrain_payload,
        prithvi_snapshot=args.prithvi_snapshot,
        decoder_width=args.decoder_width,
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
    condition_results = []
    aligned_rows = []
    aligned_events = []
    replay = None
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
        "schema_version": "pild_prithvi_native17_dualthreshold_result.v1",
        "status": "COMPLETE",
        "artifact_state": "Sen12_fixed_rule_transfer_to_unified_PILD",
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
        "prithvi_provenance": provenance,
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
        f"completed unified PILD dual-threshold: "
        f"delta_iou={aligned['delta_iou']:+.6f}, rer={aligned['rer']:+.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
