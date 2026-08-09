#!/usr/bin/env python3
"""Train a bounded Terrain gate on frozen visual and support-only experts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from audit_pild_support_only_controls_v1 import CONDITIONS, ControlledTerrainDataset
from evaluate_pild_support_only_additive_v1 import (
    BinaryHistogram,
    EvaluationDataset,
    aggregate_pair_counts,
    aggregate_samples_to_events,
    build_models,
    counts_from_predictions,
    metrics_from_pair_counts,
    validate_terrain_checkpoint,
    visual_and_terrain_logits,
    write_csv,
)
from pild_sen12_training_loader_v2 import (
    NaturalPatchSampler,
    UnifiedPILDSen12Dataset,
    sha256_file,
)
from train_pild_sen12_roleaware_v1 import state_to_cpu, validate_protocol_schema
from train_pild_support_only_terrain_v1 import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_SUMMARY,
    set_seed,
    validate_parent_v_checkpoint,
    write_json,
)


class AdaptiveTerrainGate(nn.Module):
    """Visual reliability gate that cannot create a non-Terrain correction."""

    def __init__(self, *, alpha_max: float = 2.0) -> None:
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.head = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -3.0)

    def forward(
        self,
        visual_logits: torch.Tensor,
        terrain_logits: torch.Tensor,
        q_t: torch.Tensor,
        *,
        threshold_logit: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction = torch.tanh(terrain_logits) * q_t.clamp(0.0, 1.0)
        signed_margin = ((visual_logits - threshold_logit) / 4.0).clamp(-2, 2)
        boundary_distance = torch.abs(signed_margin)
        features = torch.cat(
            (signed_margin, boundary_distance, direction, q_t.clamp(0.0, 1.0)),
            dim=1,
        )
        gates = torch.sigmoid(self.head(features))
        gate = torch.where(direction >= 0, gates[:, :1], gates[:, 1:2])
        correction = self.alpha_max * gate * direction
        return visual_logits + correction, gate


def correction_loss(
    adapted_logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    gate: torch.Tensor,
    *,
    threshold_logit: float,
    decision_temperature: float,
    gate_penalty: float,
) -> torch.Tensor:
    decision_probability = torch.sigmoid(
        (adapted_logits - threshold_logit) / decision_temperature
    )
    probability = decision_probability * valid
    truth = target * valid
    tp = (probability * truth).sum(dim=(1, 2, 3))
    fp = (probability * (1.0 - truth) * valid).sum(dim=(1, 2, 3))
    fn = ((1.0 - probability) * truth).sum(dim=(1, 2, 3))
    soft_iou_loss = (1.0 - (tp + 1.0) / (tp + fp + fn + 1.0)).mean()
    decision_bce = F.binary_cross_entropy(
        decision_probability, target, reduction="none"
    )
    decision_bce = (decision_bce * valid).sum() / valid.sum().clamp_min(1.0)
    gate_cost = (gate * valid).sum() / valid.sum().clamp_min(1.0)
    return soft_iou_loss + 0.25 * decision_bce + gate_penalty * gate_cost


@torch.no_grad()
def evaluate(
    visual: nn.Module,
    terrain: nn.Module,
    gate_model: AdaptiveTerrainGate,
    loader: DataLoader,
    *,
    threshold: float,
    use_adapter: bool,
    device: torch.device,
    export_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    threshold_logit = math.log(threshold / (1.0 - threshold))
    baseline_histogram = BinaryHistogram()
    adapted_histogram = BinaryHistogram()
    sample_rows: list[dict[str, Any]] = []
    event_histograms: dict[str, tuple[BinaryHistogram, BinaryHistogram]] = {}
    gate_sum = 0.0
    gate_pixels = 0
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual, terrain, batch, device=device
        )
        if use_adapter:
            adapted_logits, gate = gate_model(
                visual_logits,
                terrain_logits,
                q_t,
                threshold_logit=threshold_logit,
            )
        else:
            adapted_logits = visual_logits
            gate = torch.zeros_like(visual_logits)
        baseline_probability = torch.sigmoid(visual_logits)
        adapted_probability = torch.sigmoid(adapted_logits)
        baseline_histogram.update(baseline_probability, target, valid)
        adapted_histogram.update(adapted_probability, target, valid)
        gate_sum += float((gate * valid).sum().item())
        gate_pixels += int(valid.sum().item())
        for index in range(target.shape[0]):
            pair = counts_from_predictions(
                baseline_probability[index : index + 1],
                adapted_probability[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                threshold=threshold,
            )
            event_id = str(batch["canonical_event_id"][index])
            row = {
                "sample_id": str(batch["sample_id"][index]),
                "dataset_id": str(batch["dataset_id"][index]),
                "source_id": str(batch["source_id"][index]),
                "source_event_id": str(batch["source_event_id"][index]),
                "canonical_event_id": event_id,
                **pair,
                **metrics_from_pair_counts(pair),
            }
            sample_rows.append(row)
            if export_rows:
                if event_id not in event_histograms:
                    event_histograms[event_id] = (BinaryHistogram(), BinaryHistogram())
                event_histograms[event_id][0].update(
                    baseline_probability[index : index + 1],
                    target[index : index + 1],
                    valid[index : index + 1],
                )
                event_histograms[event_id][1].update(
                    adapted_probability[index : index + 1],
                    target[index : index + 1],
                    valid[index : index + 1],
                )
    corpus_counts = aggregate_pair_counts(sample_rows)
    corpus = {
        **corpus_counts,
        **metrics_from_pair_counts(corpus_counts),
        "baseline_ap": baseline_histogram.average_precision(),
        "adapted_ap": adapted_histogram.average_precision(),
        "n_samples": len(sample_rows),
        "n_events": len({row["canonical_event_id"] for row in sample_rows}),
        "mean_gate": gate_sum / max(gate_pixels, 1),
    }
    corpus["delta_ap"] = corpus["adapted_ap"] - corpus["baseline_ap"]
    event_rows = aggregate_samples_to_events(sample_rows) if export_rows else []
    if export_rows:
        for row in event_rows:
            baseline_event, adapted_event = event_histograms[row["canonical_event_id"]]
            row["baseline_ap"] = baseline_event.average_precision()
            row["adapted_ap"] = adapted_event.average_precision()
            row["delta_ap"] = row["adapted_ap"] - row["baseline_ap"]
    return corpus, sample_rows, event_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parent-v-checkpoint", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--prithvi-snapshot", type=Path)
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--epoch-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--decision-temperature", type=float, default=0.5)
    parser.add_argument("--gate-penalty", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.epoch_samples, args.batch_size) <= 0:
        raise ValueError("epochs, epoch-samples, and batch-size must be positive")

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
        with (stage / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    started = time.time()
    set_seed(args.seed)
    device = torch.device(args.device)
    visual, terrain, provenance = build_models(
        parent=parent,
        terrain_checkpoint=terrain_checkpoint,
        prithvi_snapshot=args.prithvi_snapshot,
        decoder_width=args.decoder_width,
        device=device,
    )
    normalization = terrain_checkpoint["normalization"]
    datasets: dict[str, EvaluationDataset] = {}
    for role in ("train", "val"):
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
    train_sampler = NaturalPatchSampler(
        datasets["train"].frame,
        num_samples=args.epoch_samples,
        seed=args.seed,
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        sampler=train_sampler,
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
    gate_model = AdaptiveTerrainGate(alpha_max=args.alpha_max).to(device)
    optimizer = torch.optim.AdamW(
        gate_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    threshold = float(parent["threshold"])
    threshold_logit = math.log(threshold / (1.0 - threshold))
    history: list[dict[str, Any]] = []
    best_key = (0.0, 0.0, 0.0)
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        gate_model.train()
        losses: list[float] = []
        for batch in train_loader:
            with torch.no_grad():
                visual_logits, terrain_logits, q_t, target, valid = (
                    visual_and_terrain_logits(visual, terrain, batch, device=device)
                )
            adapted_logits, gate = gate_model(
                visual_logits,
                terrain_logits,
                q_t,
                threshold_logit=threshold_logit,
            )
            if hasattr(gate_model, "correction_objective"):
                loss = gate_model.correction_objective(
                    visual_logits,
                    adapted_logits,
                    target,
                    valid,
                    gate,
                    threshold_logit=threshold_logit,
                    decision_temperature=args.decision_temperature,
                    gate_penalty=args.gate_penalty,
                )
            else:
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
        feasible = validation["delta_iou"] >= 0 and validation["rer"] >= 0
        key = (
            float(validation["delta_iou"]),
            float(validation["rer"]),
            float(validation["delta_ap"]),
        )
        if feasible and key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = state_to_cpu(gate_model)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation": validation,
                "validation_feasible": feasible,
            }
        )
        log(
            f"epoch={epoch:02d} loss={np.mean(losses):.6f} "
            f"val_diou={validation['delta_iou']:+.6f} "
            f"val_rer={validation['rer']:+.2%} gate={validation['mean_gate']:.4f}"
        )
    use_adapter = best_state is not None
    if use_adapter:
        gate_model.load_state_dict(best_state, strict=True)
    else:
        log("no validation-feasible gate; using exact visual identity")

    test_base = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.split,
        fold_id=args.fold_id,
        role="test",
        readiness="core",
    )
    test_dataset = EvaluationDataset(
        test_base, mean=normalization["mean"], scale=normalization["scale"]
    )
    condition_results: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    aligned_events: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        loader = DataLoader(
            ControlledTerrainDataset(test_dataset, condition),
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
        condition_results.append({"condition": condition, **corpus})
        if condition == "aligned":
            aligned_rows, aligned_events = sample_rows, event_rows
    write_csv(stage / "per_sample_metrics.csv", aligned_rows)
    write_csv(stage / "per_event_metrics.csv", aligned_events)
    checkpoint = {
        "schema_version": "pild_adaptive_terrain_gate_checkpoint.v1",
        "adapter_class": gate_model.__class__.__name__,
        "use_adapter": use_adapter,
        "best_epoch": best_epoch,
        "gate_state_dict": best_state,
        "alpha_max": args.alpha_max,
        "threshold": threshold,
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "split_sha256": schema["split_sha256"],
            "fold_id": args.fold_id,
            "seed": args.seed,
            "parent_v_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
            "terrain_checkpoint_sha256": terrain_receipt["checkpoint_sha256"],
        },
    }
    torch.save(checkpoint, stage / "gate_checkpoint.pt")
    aligned = condition_results[0]
    result = {
        "schema_version": "pild_adaptive_terrain_gate_result.v1",
        "status": "COMPLETE",
        "adapter_class": gate_model.__class__.__name__,
        "fold_id": args.fold_id,
        "seed": args.seed,
        "use_adapter": use_adapter,
        "best_epoch": best_epoch,
        "history": history,
        "test": aligned,
        "conditions": condition_results,
        "contrasts": {
            row["condition"]: {
                "aligned_minus_control_delta_iou": aligned["delta_iou"] - row["delta_iou"],
                "aligned_minus_control_rer": aligned["rer"] - row["rer"],
            }
            for row in condition_results[1:]
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
            "checkpoint_sha256": sha256_file(stage / "gate_checkpoint.pt"),
            "per_sample_metrics_sha256": sha256_file(stage / "per_sample_metrics.csv"),
        },
    )
    write_json(
        stage / "run_summary.json",
        {
            "status": "EXPLORATORY_COMPLETE",
            "question": "Can a spatial visual-reliability gate improve bounded Terrain corrections?",
            "fold_id": args.fold_id,
            "seed": args.seed,
            "use_adapter": use_adapter,
            "best_epoch": best_epoch,
            "delta_iou": aligned["delta_iou"],
            "delta_ap": aligned["delta_ap"],
            "rer": aligned["rer"],
            "aligned_minus_shift32_delta_iou": result["contrasts"][
                "terrain-shift32"
            ]["aligned_minus_control_delta_iou"],
            "aligned_minus_roll64_delta_iou": result["contrasts"][
                "terrain-roll64"
            ]["aligned_minus_control_delta_iou"],
            "aligned_minus_donor_delta_iou": result["contrasts"][
                "terrain-donor"
            ]["aligned_minus_control_delta_iou"],
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
