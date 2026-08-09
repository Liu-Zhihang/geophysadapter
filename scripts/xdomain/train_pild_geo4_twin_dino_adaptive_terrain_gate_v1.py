#!/usr/bin/env python3
"""Train a bounded reliability gate over frozen Twin DINO and Terrain experts."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from audit_pild_support_only_controls_v1 import CONDITIONS, ControlledTerrainDataset
from evaluate_pild_geo4_twin_dino_terrain_v1 import (
    load_terrain,
    load_visual,
    validate_baseline_replay,
)
from evaluate_pild_support_only_additive_v1 import EvaluationDataset, write_csv
from pild_sen12_training_loader_v2 import (
    NaturalPatchSampler,
    UnifiedPILDSen12Dataset,
    sha256_file,
)
from train_pild_adaptive_terrain_gate_v1 import (
    AdaptiveTerrainGate,
    correction_loss,
    evaluate,
)
from train_pild_geo4_twin_dinov2_v1 import set_seed
from train_pild_sen12_roleaware_v1 import (
    state_to_cpu,
    tensor_sha256,
    validate_protocol_schema,
)
from train_pild_support_only_terrain_v1 import write_json


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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--epoch-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--decision-temperature", type=float, default=0.5)
    parser.add_argument("--gate-penalty", type=float, default=0.02)
    parser.add_argument("--min-validation-delta-iou", type=float, default=0.002)
    parser.add_argument("--min-validation-rer", type=float, default=0.02)
    parser.add_argument("--max-replay-iou-drift", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.epoch_samples, args.batch_size, args.alpha_max) <= 0:
        raise ValueError("epochs, epoch-samples, batch-size, and alpha-max must be positive")

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

    def log(message: str) -> None:
        print(message, flush=True)
        with (stage / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

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
    datasets: dict[str, EvaluationDataset] = {}
    for role in ("train", "val", "test"):
        base = UnifiedPILDSen12Dataset(
            args.manifest,
            args.protocol_summary,
            split_path=args.split,
            fold_id=args.fold_id,
            role=role,
            readiness="core",
        )
        datasets[role] = EvaluationDataset(
            base, mean=normalization["mean"], scale=normalization["scale"]
        )
    sampler = NaturalPatchSampler(
        datasets["train"].frame,
        num_samples=args.epoch_samples,
        seed=args.seed,
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    threshold = float(visual_payload["threshold"])
    threshold_logit = math.log(threshold / (1.0 - threshold))
    gate_model = AdaptiveTerrainGate(alpha_max=args.alpha_max).to(device)
    optimizer = torch.optim.AdamW(
        gate_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    identity_validation, _, _ = evaluate(
        visual,
        terrain,
        gate_model,
        val_loader,
        threshold=threshold,
        use_adapter=False,
        device=device,
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": -1,
            "identity": True,
            "validation": identity_validation,
            "validation_feasible": True,
        }
    ]
    best_key = (0.0, 0.0, 0.0, 0.0)
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        gate_model.train()
        losses: list[float] = []
        for batch in train_loader:
            with torch.no_grad():
                from evaluate_pild_support_only_additive_v1 import (
                    visual_and_terrain_logits,
                )

                visual_logits, terrain_logits, q_t, target, valid = (
                    visual_and_terrain_logits(visual, terrain, batch, device=device)
                )
            adapted_logits, gate = gate_model(
                visual_logits,
                terrain_logits,
                q_t,
                threshold_logit=threshold_logit,
            )
            loss = correction_loss(
                adapted_logits,
                target,
                valid,
                gate,
                threshold_logit=threshold_logit,
                decision_temperature=args.decision_temperature,
                gate_penalty=args.gate_penalty,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        gate_model.eval()
        validation, _, _ = evaluate(
            visual,
            terrain,
            gate_model,
            val_loader,
            threshold=threshold,
            use_adapter=True,
            device=device,
        )
        feasible = bool(
            validation["delta_iou"] >= args.min_validation_delta_iou
            and validation["rer"] >= args.min_validation_rer
            and validation["delta_ap"] >= 0.0
        )
        key = (
            float(validation["delta_iou"]),
            float(validation["rer"]),
            float(validation["delta_ap"]),
            -float(validation["mean_gate"]),
        )
        if feasible and key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = state_to_cpu(gate_model)
        history.append(
            {
                "epoch": epoch,
                "identity": False,
                "train_loss": float(np.mean(losses)),
                "validation": validation,
                "validation_feasible": feasible,
            }
        )
        log(
            f"epoch={epoch:02d} loss={np.mean(losses):.6f} "
            f"val_diou={validation['delta_iou']:+.6f} "
            f"val_rer={validation['rer']:+.2%} "
            f"val_dap={validation['delta_ap']:+.6f} "
            f"gate={validation['mean_gate']:.4f} feasible={feasible}"
        )

    use_adapter = best_state is not None
    if use_adapter:
        gate_model.load_state_dict(best_state, strict=True)
    else:
        log("no validation-feasible gate; exact identity selected")

    condition_results: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    aligned_events: list[dict[str, Any]] = []
    replay_report: dict[str, Any] | None = None
    for condition in CONDITIONS:
        loader = DataLoader(
            ControlledTerrainDataset(datasets["test"], condition),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        corpus, sample_rows, event_rows = evaluate(
            visual,
            terrain,
            gate_model,
            loader,
            threshold=threshold,
            use_adapter=use_adapter,
            device=device,
            export_rows=(condition == "aligned"),
        )
        if condition == "aligned":
            replay_report = validate_baseline_replay(
                sample_rows,
                args.visual_per_sample,
                max_iou_drift=args.max_replay_iou_drift,
            )
            aligned_rows, aligned_events = sample_rows, event_rows
        condition_results.append({"condition": condition, **corpus})
    write_csv(stage / "per_sample_metrics.csv", aligned_rows)
    write_csv(stage / "per_event_metrics.csv", aligned_events)

    gate_state = state_to_cpu(gate_model) if use_adapter else None
    checkpoint = {
        "schema_version": "pild_geo4_twin_dino_adaptive_terrain_gate_checkpoint.v1",
        "use_adapter": use_adapter,
        "best_epoch": best_epoch,
        "gate_state_dict": gate_state,
        "gate_state_sha256": tensor_sha256(gate_state) if gate_state else None,
        "threshold": threshold,
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "split_sha256": schema["split_sha256"],
            "fold_id": args.fold_id,
            "seed": args.seed,
            "visual_checkpoint_sha256": visual_receipt["checkpoint_sha256"],
            "terrain_checkpoint_sha256": terrain_receipt["checkpoint_sha256"],
        },
    }
    torch.save(checkpoint, stage / "gate_checkpoint.pt")
    aligned = condition_results[0]
    result = {
        "schema_version": "pild_geo4_twin_dino_adaptive_terrain_gate_result.v1",
        "status": "COMPLETE",
        "artifact_state": "discovery_single_seed",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "use_adapter": use_adapter,
        "best_epoch": best_epoch,
        "selection_gate": {
            "min_validation_delta_iou": args.min_validation_delta_iou,
            "min_validation_rer": args.min_validation_rer,
            "min_validation_delta_ap": 0.0,
        },
        "history": history,
        "test": aligned,
        "conditions": condition_results,
        "visual_replay": replay_report,
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
            "checkpoint_sha256": sha256_file(stage / "gate_checkpoint.pt"),
            "per_sample_sha256": sha256_file(stage / "per_sample_metrics.csv"),
            "per_event_sha256": sha256_file(stage / "per_event_metrics.csv"),
        },
    )
    os.replace(stage, outdir)
    print(
        f"completed {args.fold_id}: use_adapter={use_adapter}, "
        f"delta_iou={aligned['delta_iou']:+.6f}, rer={aligned['rer']:+.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
