#!/usr/bin/env python3
"""Evaluate a predeclared alpha cap from a frozen validation margin grid."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from audit_pild_support_only_controls_v1 import CONDITIONS, ControlledTerrainDataset
from evaluate_pild_decision_margin_terrain_v1 import evaluate
from evaluate_pild_support_only_additive_v1 import (
    EvaluationDataset,
    build_models,
    validate_terrain_checkpoint,
    write_csv,
)
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file
from train_pild_sen12_roleaware_v1 import validate_protocol_schema
from train_pild_support_only_terrain_v1 import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_SUMMARY,
    validate_parent_v_checkpoint,
    write_json,
)


def choose_capped(
    frozen: dict[str, Any], *, max_alpha: float
) -> dict[str, dict[str, Any]]:
    selections: dict[str, dict[str, Any]] = {}
    for dataset_id, rows in frozen["grids"].items():
        feasible = [
            dict(row)
            for row in rows
            if float(row["positive_alpha"]) <= max_alpha
            and float(row["negative_alpha"]) <= max_alpha
            and float(row["delta_iou"]) >= -1e-12
            and float(row["rer"]) >= -1e-12
        ]
        if not feasible:
            raise RuntimeError(f"no capped identity candidate for {dataset_id}")
        selected = max(
            feasible,
            key=lambda row: (
                row["delta_iou"],
                row["rer"],
                row["delta_ap"],
                -(row["positive_alpha"] + row["negative_alpha"]),
            ),
        )
        if selected["temperature"] is None:
            selected["temperature"] = math.inf
        selections[str(dataset_id)] = selected
    return selections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parent-v-checkpoint", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-selection", type=Path, required=True)
    parser.add_argument("--max-alpha", type=float, default=0.5)
    parser.add_argument("--prithvi-snapshot", type=Path)
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_alpha <= 0:
        raise ValueError("--max-alpha must be positive")

    schema = validate_protocol_schema(
        args.manifest, args.protocol_summary, args.split, args.fold_id
    )
    parent, parent_receipt = validate_parent_v_checkpoint(
        args.parent_v_checkpoint,
        manifest_sha256=schema["manifest_sha256"],
        split_sha256=schema["split_sha256"],
        fold_id=args.fold_id,
        seed=args.seed,
    )
    terrain_checkpoint, terrain_receipt = validate_terrain_checkpoint(
        args.terrain_checkpoint,
        schema=schema,
        protocol_summary_path=args.protocol_summary,
        fold_id=args.fold_id,
        seed=args.seed,
        parent_receipt=parent_receipt,
    )
    frozen = json.loads(args.frozen_selection.read_text(encoding="utf-8"))
    identity = frozen.get("identity", {})
    expected = {
        "manifest_sha256": schema["manifest_sha256"],
        "split_sha256": schema["split_sha256"],
        "fold_id": args.fold_id,
        "seed": args.seed,
        "parent_v_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
        "terrain_checkpoint_sha256": terrain_receipt["checkpoint_sha256"],
    }
    mismatch = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"frozen selection identity mismatch: {mismatch}")
    selections = choose_capped(frozen, max_alpha=args.max_alpha)

    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    stage = outdir.parent / f".{outdir.name}.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    started = time.time()
    receipt = {
        "schema_version": "pild_regularized_margin_selection.v1",
        "selection_scope": "frozen validation grid only",
        "regularization": {"max_positive_alpha": args.max_alpha, "max_negative_alpha": args.max_alpha},
        "frozen_selection_path": str(args.frozen_selection.resolve()),
        "frozen_selection_sha256": sha256_file(args.frozen_selection),
        "frozen_before_test_open": True,
        "selections": selections,
        "identity": expected,
    }
    write_json(stage / "selection.json", receipt)

    device = torch.device(args.device)
    visual, terrain, provenance = build_models(
        parent=parent,
        terrain_checkpoint=terrain_checkpoint,
        prithvi_snapshot=args.prithvi_snapshot,
        decoder_width=args.decoder_width,
        device=device,
    )
    normalization = terrain_checkpoint["normalization"]
    test_base = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.split,
        fold_id=args.fold_id,
        role="test",
        readiness="core",
    )
    evaluation_base = EvaluationDataset(
        test_base, mean=normalization["mean"], scale=normalization["scale"]
    )
    conditions: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    aligned_events: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        loader = DataLoader(
            ControlledTerrainDataset(evaluation_base, condition),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        sample_rows, event_rows, corpus = evaluate(
            visual,
            terrain,
            loader,
            selections=selections,
            threshold=float(parent["threshold"]),
            device=device,
        )
        conditions.append({"condition": condition, **corpus})
        if condition == "aligned":
            aligned_rows, aligned_events = sample_rows, event_rows
    write_csv(stage / "per_sample_metrics.csv", aligned_rows)
    write_csv(stage / "per_event_metrics.csv", aligned_events)
    aligned = conditions[0]
    result = {
        "schema_version": "pild_regularized_margin_terrain_result.v1",
        "status": "COMPLETE",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "max_alpha": args.max_alpha,
        "selections": selections,
        "test": aligned,
        "conditions": conditions,
        "contrasts": {
            row["condition"]: {
                "aligned_minus_control_delta_iou": aligned["delta_iou"] - row["delta_iou"],
                "aligned_minus_control_rer": aligned["rer"] - row["rer"],
            }
            for row in conditions[1:]
        },
        "parent_v_receipt": parent_receipt,
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
            "per_sample_metrics_sha256": sha256_file(stage / "per_sample_metrics.csv"),
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
