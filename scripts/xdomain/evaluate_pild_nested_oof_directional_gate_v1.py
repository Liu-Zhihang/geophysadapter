#!/usr/bin/env python3
"""Fit a rescue/veto gate on nested-OOF proposals, then evaluate an outer fold.

Only inner-test events from the outer training partition are used to fit the
gate. The immutable gate receipt is written before the outer test dataset is
constructed. Rejected proposals exactly recover the frozen visual prediction.
"""

from __future__ import annotations

import argparse
import json
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
from evaluate_pild_benefit_gate_v1 import (
    FEATURE_NAMES,
    apply_gate,
    make_features,
)
from evaluate_pild_directional_benefit_gate_v1 import fit_directional_gate
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
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file
from train_pild_sen12_roleaware_v1 import set_seed, validate_protocol_schema
from train_pild_support_only_terrain_v1 import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_SUMMARY,
    validate_parent_v_checkpoint,
    write_json,
)


def candidate_logits(
    visual_logits: torch.Tensor,
    terrain_logits: torch.Tensor,
    q_t: torch.Tensor,
    *,
    threshold: float,
    low: float,
    high: float,
    alpha: float,
    visual_margin: float,
) -> torch.Tensor:
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
    return visual_logits + alpha * (rescue - veto)


def deterministic_choice(
    indices: np.ndarray, limit: int, token: str
) -> np.ndarray:
    if indices.size <= limit:
        return indices
    import hashlib

    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).choice(indices, size=limit, replace=False)


@torch.no_grad()
def collect_oof_rows(
    visual: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    threshold: float,
    low: float,
    high: float,
    alpha: float,
    visual_margin: float,
    pixels_per_sample: int,
    seed: int,
    fold_id: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []
    event_out: list[np.ndarray] = []
    proposal_count = beneficial_count = 0
    observed_events: set[str] = set()
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual, terrain_model, batch, device=device
        )
        candidate = candidate_logits(
            visual_logits,
            terrain_logits,
            q_t,
            threshold=threshold,
            low=low,
            high=high,
            alpha=alpha,
            visual_margin=visual_margin,
        )
        features, proposal = make_features(
            visual_logits,
            terrain_logits,
            q_t,
            batch["terrain"].to(device),
            candidate,
            batch["dataset_id"],
            threshold=threshold,
        )
        truth = target >= 0.5
        candidate_correct = (torch.sigmoid(candidate) >= threshold) == truth
        proposal &= valid.bool()
        for index, sample_id in enumerate(batch["sample_id"]):
            event_id = str(batch["canonical_event_id"][index])
            observed_events.add(event_id)
            selected = torch.nonzero(
                proposal[index, 0].reshape(-1), as_tuple=False
            ).flatten().cpu().numpy()
            proposal_count += int(selected.size)
            if selected.size == 0:
                continue
            selected = deterministic_choice(
                selected,
                pixels_per_sample,
                f"{seed}|{fold_id}|{sample_id}|nested-oof-gate",
            )
            flat_features = features[index].permute(1, 2, 0).reshape(
                -1, features.shape[1]
            )
            labels = candidate_correct[index, 0].reshape(-1)[selected]
            x_out.append(
                flat_features[selected].cpu().numpy().astype(np.float32)
            )
            y_out.append(labels.cpu().numpy().astype(np.uint8))
            event_out.append(np.repeat(event_id, len(selected)))
            beneficial_count += int(labels.sum().item())
    if not x_out:
        raise RuntimeError(f"{fold_id} produced no OOF correction proposals")
    x = np.concatenate(x_out)
    y = np.concatenate(y_out)
    events = np.concatenate(event_out)
    return x, y, events, {
        "fold_id": fold_id,
        "n_events": len(observed_events),
        "all_proposals": proposal_count,
        "sampled_proposals": len(y),
        "sampled_beneficial": beneficial_count,
        "sampled_harmful": len(y) - beneficial_count,
    }


@torch.no_grad()
def evaluate_outer(
    visual: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    gate: Any,
    visual_threshold: float,
    gate_threshold: float,
    low: float,
    high: float,
    alpha: float,
    visual_margin: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_histogram = BinaryHistogram()
    adapted_histogram = BinaryHistogram()
    rows: list[dict[str, Any]] = []
    proposed_total = accepted_total = 0
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual, terrain_model, batch, device=device
        )
        candidate = candidate_logits(
            visual_logits,
            terrain_logits,
            q_t,
            threshold=visual_threshold,
            low=low,
            high=high,
            alpha=alpha,
            visual_margin=visual_margin,
        )
        features, proposal = make_features(
            visual_logits,
            terrain_logits,
            q_t,
            batch["terrain"].to(device),
            candidate,
            batch["dataset_id"],
            threshold=visual_threshold,
        )
        final_logits, accepted = apply_gate(
            gate,
            features,
            proposal,
            candidate,
            visual_logits,
            threshold=gate_threshold,
        )
        proposal &= valid.bool()
        accepted &= valid.bool()
        proposed_total += int(proposal.sum().item())
        accepted_total += int(accepted.sum().item())
        baseline_probability = torch.sigmoid(visual_logits)
        adapted_probability = torch.sigmoid(final_logits)
        baseline_histogram.update(baseline_probability, target, valid)
        adapted_histogram.update(adapted_probability, target, valid)
        for index in range(target.shape[0]):
            pair = counts_from_predictions(
                baseline_probability[index : index + 1],
                adapted_probability[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                threshold=visual_threshold,
            )
            rows.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "dataset_id": str(batch["dataset_id"][index]),
                    "source_id": str(batch["source_id"][index]),
                    "source_event_id": str(batch["source_event_id"][index]),
                    "canonical_event_id": str(batch["canonical_event_id"][index]),
                    "proposed": int(proposal[index].sum().item()),
                    "accepted": int(accepted[index].sum().item()),
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
        "delta_ap": adapted_histogram.average_precision()
        - baseline_histogram.average_precision(),
        "n_samples": len(rows),
        "n_events": len(event_rows),
        "proposed": proposed_total,
        "accepted": accepted_total,
        "acceptance_rate": accepted_total / max(proposed_total, 1),
    }
    return rows, event_rows, corpus


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    result.add_argument("--outer-split", type=Path, default=DEFAULT_SPLIT)
    result.add_argument("--nested-split", type=Path, required=True)
    result.add_argument("--outer-fold-id", required=True)
    result.add_argument("--inner-run-root", type=Path, required=True)
    result.add_argument("--inner-seed", type=int, default=20260725)
    result.add_argument("--outer-seed", type=int, required=True)
    result.add_argument("--outer-visual-checkpoint", type=Path, required=True)
    result.add_argument("--outer-terrain-checkpoint", type=Path, required=True)
    result.add_argument("--outdir", type=Path, required=True)
    result.add_argument("--prithvi-snapshot", type=Path)
    result.add_argument("--decoder-width", type=int, default=128)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--pixels-per-sample", type=int, default=512)
    result.add_argument("--c-value", type=float, default=0.1)
    result.add_argument("--rescue-harm-weight", type=float, default=2.0)
    result.add_argument("--veto-harm-weight", type=float, default=8.0)
    result.add_argument("--gate-threshold", type=float, default=0.5)
    result.add_argument("--low", type=float, default=0.3)
    result.add_argument("--high", type=float, default=0.7)
    result.add_argument("--alpha", type=float, default=4.0)
    result.add_argument("--visual-margin", type=float, default=1.0)
    result.add_argument("--device", default="cuda")
    return result


def main() -> int:
    args = parser().parse_args()
    if not 0 <= args.low < args.high <= 1:
        raise ValueError("expected 0 <= low < high <= 1")
    outer_schema = validate_protocol_schema(
        args.manifest,
        args.protocol_summary,
        args.outer_split,
        args.outer_fold_id,
    )
    outer_parent, outer_parent_receipt = validate_parent_v_checkpoint(
        args.outer_visual_checkpoint,
        manifest_sha256=outer_schema["manifest_sha256"],
        split_sha256=outer_schema["split_sha256"],
        fold_id=args.outer_fold_id,
        seed=args.outer_seed,
    )
    outer_terrain, outer_terrain_receipt = validate_terrain_checkpoint(
        args.outer_terrain_checkpoint,
        schema=outer_schema,
        protocol_summary_path=args.protocol_summary,
        fold_id=args.outer_fold_id,
        seed=args.outer_seed,
        parent_receipt=outer_parent_receipt,
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
    set_seed(args.inner_seed)
    device = torch.device(args.device)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    event_parts: list[np.ndarray] = []
    inner_metadata: list[dict[str, Any]] = []
    all_events: set[str] = set()
    for inner_index in range(3):
        fold_id = f"{args.outer_fold_id}__inner_{inner_index}"
        schema = validate_protocol_schema(
            args.manifest,
            args.protocol_summary,
            args.nested_split,
            fold_id,
        )
        run = args.inner_run_root / fold_id / f"seed{args.inner_seed}"
        parent, parent_receipt = validate_parent_v_checkpoint(
            run / "V" / "checkpoint.pt",
            manifest_sha256=schema["manifest_sha256"],
            split_sha256=schema["split_sha256"],
            fold_id=fold_id,
            seed=args.inner_seed,
        )
        terrain_payload, _ = validate_terrain_checkpoint(
            run / "terrain_expert" / "terrain_expert.pt",
            schema=schema,
            protocol_summary_path=args.protocol_summary,
            fold_id=fold_id,
            seed=args.inner_seed,
            parent_receipt=parent_receipt,
        )
        visual, terrain_model, _ = build_models(
            parent=parent,
            terrain_checkpoint=terrain_payload,
            prithvi_snapshot=args.prithvi_snapshot,
            decoder_width=args.decoder_width,
            device=device,
        )
        normalization = terrain_payload["normalization"]
        base = UnifiedPILDSen12Dataset(
            args.manifest,
            args.protocol_summary,
            split_path=args.nested_split,
            fold_id=fold_id,
            role="test",
            readiness="core",
        )
        dataset = EvaluationDataset(
            base, mean=normalization["mean"], scale=normalization["scale"]
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        x, y, events, metadata = collect_oof_rows(
            visual,
            terrain_model,
            loader,
            threshold=float(parent["threshold"]),
            low=args.low,
            high=args.high,
            alpha=args.alpha,
            visual_margin=args.visual_margin,
            pixels_per_sample=args.pixels_per_sample,
            seed=args.inner_seed,
            fold_id=fold_id,
            device=device,
        )
        overlap = all_events.intersection(np.unique(events).tolist())
        if overlap:
            raise RuntimeError(f"inner-test event overlap detected: {sorted(overlap)}")
        all_events.update(np.unique(events).tolist())
        x_parts.append(x)
        y_parts.append(y)
        event_parts.append(events)
        inner_metadata.append(metadata)
        del visual, terrain_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    events = np.concatenate(event_parts)
    gate, fit_metadata = fit_directional_gate(
        x,
        y,
        events,
        c_value=args.c_value,
        rescue_harm_weight=args.rescue_harm_weight,
        veto_harm_weight=args.veto_harm_weight,
        seed=args.inner_seed,
    )
    receipt = {
        "schema_version": "pild_nested_oof_directional_gate_receipt.v1",
        "frozen_before_outer_test_open": True,
        "scope": "outer-train nested-OOF visual/Terrain correction utility",
        "feature_names": list(FEATURE_NAMES),
        "candidate": {
            "low": args.low,
            "high": args.high,
            "alpha": args.alpha,
            "visual_margin": args.visual_margin,
        },
        "gate_threshold": args.gate_threshold,
        "gate": gate.to_dict(),
        "training": {
            "inner_folds": inner_metadata,
            "sampled_rows": len(y),
            "events": len(np.unique(events)),
            "beneficial_fraction": float(y.mean()),
            **fit_metadata,
        },
        "identity": {
            "manifest_sha256": outer_schema["manifest_sha256"],
            "protocol_summary_sha256": sha256_file(args.protocol_summary),
            "outer_split_sha256": outer_schema["split_sha256"],
            "nested_split_sha256": sha256_file(args.nested_split),
            "outer_fold_id": args.outer_fold_id,
            "outer_seed": args.outer_seed,
            "inner_seed": args.inner_seed,
            "outer_visual_checkpoint_sha256": outer_parent_receipt[
                "checkpoint_sha256"
            ],
            "outer_terrain_checkpoint_sha256": outer_terrain_receipt[
                "checkpoint_sha256"
            ],
        },
    }
    write_json(stage / "gate_receipt.json", receipt)
    gate_receipt_sha256 = sha256_file(stage / "gate_receipt.json")

    outer_visual, outer_terrain_model, provenance = build_models(
        parent=outer_parent,
        terrain_checkpoint=outer_terrain,
        prithvi_snapshot=args.prithvi_snapshot,
        decoder_width=args.decoder_width,
        device=device,
    )
    normalization = outer_terrain["normalization"]
    outer_base = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.outer_split,
        fold_id=args.outer_fold_id,
        role="test",
        readiness="core",
    )
    outer_dataset = EvaluationDataset(
        outer_base, mean=normalization["mean"], scale=normalization["scale"]
    )
    condition_results: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    aligned_events: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        loader = DataLoader(
            ControlledTerrainDataset(outer_dataset, condition),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        sample_rows, event_rows, corpus = evaluate_outer(
            outer_visual,
            outer_terrain_model,
            loader,
            gate=gate,
            visual_threshold=float(outer_parent["threshold"]),
            gate_threshold=args.gate_threshold,
            low=args.low,
            high=args.high,
            alpha=args.alpha,
            visual_margin=args.visual_margin,
            device=device,
        )
        condition_results.append({"condition": condition, **corpus})
        if condition == "aligned":
            aligned_rows, aligned_events = sample_rows, event_rows

    write_csv(stage / "per_sample_metrics.csv", aligned_rows)
    write_csv(stage / "per_event_metrics.csv", aligned_events)
    aligned = condition_results[0]
    result = {
        "schema_version": "pild_nested_oof_directional_gate_result.v1",
        "status": "COMPLETE",
        "development_status": (
            "exploratory outer folds 0/1; confirm unchanged on outer folds 2/3"
        ),
        "outer_fold_id": args.outer_fold_id,
        "outer_seed": args.outer_seed,
        "contract": {
            "visual_anchor": "frozen Prithvi-EO-2.0-300M-TL",
            "proposal": "bounded bidirectional Terrain correction",
            "gate": "directional utility fitted only on nested-OOF outer-train events",
            "rejection": "exact visual identity",
        },
        "gate_receipt_sha256": gate_receipt_sha256,
        "test": aligned,
        "conditions": condition_results,
        "contrasts": {
            row["condition"]: {
                "aligned_minus_control_delta_iou": aligned["delta_iou"]
                - row["delta_iou"],
                "aligned_minus_control_rer": aligned["rer"] - row["rer"],
            }
            for row in condition_results[1:]
        },
        "outer_visual_receipt": outer_parent_receipt,
        "outer_terrain_receipt": outer_terrain_receipt,
        "prithvi_provenance": provenance,
        "elapsed_seconds": time.time() - started,
    }
    write_json(stage / "result.json", result)
    write_json(
        stage / "DONE.json",
        {
            "status": "COMPLETE",
            "gate_receipt_sha256": gate_receipt_sha256,
            "result_sha256": sha256_file(stage / "result.json"),
            "per_sample_sha256": sha256_file(stage / "per_sample_metrics.csv"),
            "per_event_sha256": sha256_file(stage / "per_event_metrics.csv"),
        },
    )
    os.replace(stage, outdir)
    print(
        f"{args.outer_fold_id}: nested-OOF gate "
        f"delta_iou={aligned['delta_iou']:+.6f}, "
        f"delta_ap={aligned['delta_ap']:+.6f}, rer={aligned['rer']:+.2%}, "
        f"accepted={aligned['acceptance_rate']:.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
