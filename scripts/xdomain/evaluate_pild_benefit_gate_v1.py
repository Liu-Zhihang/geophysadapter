#!/usr/bin/env python3
"""Validation-fitted utility gate for bounded PILD Terrain corrections.

The frozen visual and Terrain models first propose a hard rescue or veto.
A small linear gate learns on validation events whether such proposals are
likely to help. Test inference uses only label-free visual reliability,
Terrain support, known source metadata, and normalized Terrain variables.
The fitted gate receipt is persisted before test data are constructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from audit_pild_support_only_controls_v1 import CONDITIONS, ControlledTerrainDataset
from evaluate_pild_source_calibrated_terrain_v1 import select_by_source
from evaluate_pild_support_only_additive_v1 import (
    BinaryHistogram,
    EvaluationDataset,
    aggregate_pair_counts,
    aggregate_samples_to_events,
    build_models,
    counts_from_predictions,
    fuse_logits,
    metrics_from_pair_counts,
    validate_terrain_checkpoint,
    visual_and_terrain_logits,
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


SOURCE_ORDER = (
    "DLR_Landslide_Ref_2025",
    "GDCLD",
    "GLaD4CD_v1",
    "SEN12LS_HARMONIZED",
)
TERRAIN_FEATURE_INDICES = (1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
TERRAIN_FEATURE_NAMES = (
    "slope_deg",
    "tpi_90m",
    "tpi_300m",
    "tpi_900m",
    "local_std_90m",
    "local_std_300m",
    "local_relief_300m",
    "local_relief_900m",
    "valley_depth_900m",
    "ridge_height_900m",
    "ruggedness_90m",
)
FEATURE_NAMES = (
    "visual_logit",
    "visual_probability",
    "visual_uncertainty",
    "abs_visual_logit",
    "terrain_logit",
    "terrain_direction",
    "abs_terrain_direction",
    "terrain_validity",
    "uncertainty_x_direction",
    "visual_positive",
    "proposal_rescue",
    "proposal_veto",
    *TERRAIN_FEATURE_NAMES,
    *(f"source_{source}" for source in SOURCE_ORDER),
)


@dataclass(frozen=True)
class LinearBenefitGate:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    constant_probability: float | None = None

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full(len(features), self.constant_probability, np.float32)
        standardized = (features - self.mean) / self.scale
        score = standardized @ self.coefficient + self.intercept
        score = np.clip(score, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
            "constant_probability": self.constant_probability,
        }


def deterministic_choice(indices: np.ndarray, limit: int, token: str) -> np.ndarray:
    if indices.size <= limit:
        return indices
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).choice(indices, size=limit, replace=False)


def selected_candidate_logits(
    visual_logits: torch.Tensor,
    terrain_logits: torch.Tensor,
    q_t: torch.Tensor,
    dataset_ids: list[str] | tuple[str, ...],
    selections: Mapping[str, Mapping[str, Any]],
) -> torch.Tensor:
    output = visual_logits.clone()
    source_indices: dict[str, list[int]] = defaultdict(list)
    for index, dataset_id in enumerate(dataset_ids):
        source_indices[str(dataset_id)].append(index)
    for dataset_id, indices in source_indices.items():
        if dataset_id not in selections:
            raise RuntimeError(f"source absent from validation selection: {dataset_id}")
        selected = selections[dataset_id]
        output[indices] = fuse_logits(
            visual_logits[indices],
            terrain_logits[indices],
            q_t[indices],
            alpha=float(selected["alpha"]),
            uncertainty_power=float(selected["uncertainty_power"]),
        )
    return output


def make_features(
    visual_logits: torch.Tensor,
    terrain_logits: torch.Tensor,
    q_t: torch.Tensor,
    terrain: torch.Tensor,
    candidate_logits: torch.Tensor,
    dataset_ids: list[str] | tuple[str, ...],
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    visual_probability = torch.sigmoid(visual_logits)
    uncertainty = (
        1.0 - 2.0 * torch.abs(visual_probability - 0.5)
    ).clamp(0.0, 1.0)
    direction = torch.tanh(terrain_logits) * q_t.clamp(0.0, 1.0)
    visual_positive = visual_probability >= threshold
    candidate_positive = torch.sigmoid(candidate_logits) >= threshold
    rescue = (~visual_positive & candidate_positive).float()
    veto = (visual_positive & ~candidate_positive).float()
    source_planes = []
    for source in SOURCE_ORDER:
        values = torch.tensor(
            [float(str(dataset_id) == source) for dataset_id in dataset_ids],
            device=visual_logits.device,
            dtype=visual_logits.dtype,
        )[:, None, None, None]
        source_planes.append(values.expand_as(visual_logits))
    parts = [
        visual_logits.clamp(-12.0, 12.0),
        visual_probability,
        uncertainty,
        visual_logits.abs().clamp(0.0, 12.0),
        terrain_logits.clamp(-12.0, 12.0),
        direction,
        direction.abs(),
        q_t.clamp(0.0, 1.0),
        uncertainty * direction,
        visual_positive.float(),
        rescue,
        veto,
        terrain[:, TERRAIN_FEATURE_INDICES],
        *source_planes,
    ]
    features = torch.cat(parts, dim=1)
    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"feature width mismatch: {features.shape[1]} != {len(FEATURE_NAMES)}"
        )
    proposal = visual_positive != candidate_positive
    return features, proposal


@torch.no_grad()
def collect_gate_training_rows(
    visual_model: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    selections: Mapping[str, Mapping[str, Any]],
    threshold: float,
    pixels_per_sample: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    features_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    events_out: list[np.ndarray] = []
    sources_out: list[np.ndarray] = []
    proposal_count = beneficial_count = 0
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual_model, terrain_model, batch, device=device
        )
        candidate_logits = selected_candidate_logits(
            visual_logits,
            terrain_logits,
            q_t,
            batch["dataset_id"],
            selections,
        )
        features, proposal = make_features(
            visual_logits,
            terrain_logits,
            q_t,
            batch["terrain"].to(device),
            candidate_logits,
            batch["dataset_id"],
            threshold=threshold,
        )
        truth = target >= 0.5
        candidate_correct = (torch.sigmoid(candidate_logits) >= threshold) == truth
        proposal &= valid.bool()
        for index, sample_id in enumerate(batch["sample_id"]):
            flat_indices = (
                torch.nonzero(proposal[index, 0].reshape(-1), as_tuple=False)
                .flatten()
                .cpu()
                .numpy()
            )
            proposal_count += int(flat_indices.size)
            if flat_indices.size == 0:
                continue
            selected = deterministic_choice(
                flat_indices,
                pixels_per_sample,
                f"{seed}|{sample_id}|benefit-gate",
            )
            flat_features = (
                features[index].permute(1, 2, 0).reshape(-1, features.shape[1])
            )
            flat_target = candidate_correct[index, 0].reshape(-1)
            y = flat_target[selected].cpu().numpy().astype(np.uint8)
            beneficial_count += int(y.sum())
            features_out.append(flat_features[selected].cpu().numpy().astype(np.float32))
            targets_out.append(y)
            events_out.append(
                np.repeat(str(batch["canonical_event_id"][index]), len(selected))
            )
            sources_out.append(
                np.repeat(str(batch["dataset_id"][index]), len(selected))
            )
    if not features_out:
        raise RuntimeError("validation produced no visual/Terrain proposals")
    x = np.concatenate(features_out)
    y = np.concatenate(targets_out)
    events = np.concatenate(events_out)
    sources = np.concatenate(sources_out)
    metadata = {
        "all_validation_proposals": proposal_count,
        "sampled_proposals": int(len(y)),
        "sampled_beneficial": int(y.sum()),
        "sampled_harmful": int(len(y) - y.sum()),
        "sampled_events": int(len(np.unique(events))),
        "sampled_sources": sorted(np.unique(sources).tolist()),
    }
    return x, y, events, metadata


def fit_gate(
    x: np.ndarray,
    y: np.ndarray,
    events: np.ndarray,
    *,
    c_value: float,
    harm_weight: float,
    seed: int,
) -> tuple[LinearBenefitGate, dict[str, Any]]:
    if len(np.unique(y)) < 2:
        probability = float(y[0]) if len(y) else 0.0
        gate = LinearBenefitGate(
            np.zeros(x.shape[1], np.float32),
            np.ones(x.shape[1], np.float32),
            np.zeros(x.shape[1], np.float32),
            0.0,
            probability,
        )
        return gate, {"constant_probability": probability}
    unique_events, event_counts = np.unique(events, return_counts=True)
    inverse_event = {
        event: 1.0 / count for event, count in zip(unique_events, event_counts)
    }
    weights = np.asarray([inverse_event[event] for event in events], np.float64)
    weights *= np.where(y == 1, 1.0, harm_weight)
    weights /= max(weights.mean(), 1e-12)
    scaler = StandardScaler().fit(x)
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    standardized = (x - scaler.mean_) / scale
    classifier = LogisticRegression(
        C=c_value,
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    ).fit(standardized, y, sample_weight=weights)
    gate = LinearBenefitGate(
        scaler.mean_.astype(np.float32),
        scale.astype(np.float32),
        classifier.coef_[0].astype(np.float32),
        float(classifier.intercept_[0]),
    )
    return gate, {
        "c_value": c_value,
        "harm_weight": harm_weight,
        "n_iter": int(classifier.n_iter_[0]),
        "weighted_training_accuracy": float(
            np.average(classifier.predict(standardized) == y, weights=weights)
        ),
    }


def apply_gate(
    gate: LinearBenefitGate,
    features: torch.Tensor,
    proposal: torch.Tensor,
    candidate_logits: torch.Tensor,
    visual_logits: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    accept = torch.zeros_like(proposal)
    for index in range(features.shape[0]):
        selected = (
            torch.nonzero(proposal[index, 0].reshape(-1), as_tuple=False)
            .flatten()
            .cpu()
            .numpy()
        )
        if selected.size == 0:
            continue
        flat = features[index].permute(1, 2, 0).reshape(-1, features.shape[1])
        probability = gate.predict_probability(
            flat[selected].cpu().numpy().astype(np.float32)
        )
        accepted = selected[probability >= threshold]
        accept[index, 0].view(-1)[
            torch.from_numpy(accepted).to(accept.device)
        ] = True
    output = torch.where(accept, candidate_logits, visual_logits)
    return output, accept


@torch.no_grad()
def evaluate(
    visual_model: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    selections: Mapping[str, Mapping[str, Any]],
    gate: LinearBenefitGate,
    visual_threshold: float,
    gate_threshold: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_histogram = BinaryHistogram()
    adapted_histogram = BinaryHistogram()
    event_histograms: dict[str, tuple[BinaryHistogram, BinaryHistogram]] = {}
    rows: list[dict[str, Any]] = []
    accepted_total = proposed_total = 0
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual_model, terrain_model, batch, device=device
        )
        candidate_logits = selected_candidate_logits(
            visual_logits,
            terrain_logits,
            q_t,
            batch["dataset_id"],
            selections,
        )
        features, proposal = make_features(
            visual_logits,
            terrain_logits,
            q_t,
            batch["terrain"].to(device),
            candidate_logits,
            batch["dataset_id"],
            threshold=visual_threshold,
        )
        final_logits, accept = apply_gate(
            gate,
            features,
            proposal,
            candidate_logits,
            visual_logits,
            threshold=gate_threshold,
        )
        proposed_total += int((proposal & valid.bool()).sum().item())
        accepted_total += int((accept & valid.bool()).sum().item())
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
            event_id = str(batch["canonical_event_id"][index])
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
            rows.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "dataset_id": str(batch["dataset_id"][index]),
                    "source_id": str(batch["source_id"][index]),
                    "source_event_id": str(batch["source_event_id"][index]),
                    "canonical_event_id": event_id,
                    "proposed": int((proposal[index] & valid[index].bool()).sum()),
                    "accepted": int((accept[index] & valid[index].bool()).sum()),
                    **pair,
                    **metrics_from_pair_counts(pair),
                }
            )
    event_rows = aggregate_samples_to_events(rows)
    for row in event_rows:
        baseline_event, adapted_event = event_histograms[row["canonical_event_id"]]
        row["baseline_ap"] = baseline_event.average_precision()
        row["adapted_ap"] = adapted_event.average_precision()
        row["delta_ap"] = row["adapted_ap"] - row["baseline_ap"]
    corpus_counts = aggregate_pair_counts(rows)
    corpus = {
        **corpus_counts,
        **metrics_from_pair_counts(corpus_counts),
        "baseline_ap": baseline_histogram.average_precision(),
        "adapted_ap": adapted_histogram.average_precision(),
        "n_samples": len(rows),
        "n_events": len(event_rows),
        "proposed": proposed_total,
        "accepted": accepted_total,
        "acceptance_rate": accepted_total / max(proposed_total, 1),
    }
    corpus["delta_ap"] = corpus["adapted_ap"] - corpus["baseline_ap"]
    return rows, event_rows, corpus


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixels-per-sample", type=int, default=2048)
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--harm-weight", type=float, default=2.0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
    started = time.time()
    device = torch.device(args.device)
    visual, terrain, provenance = build_models(
        parent=parent,
        terrain_checkpoint=terrain_checkpoint,
        prithvi_snapshot=args.prithvi_snapshot,
        decoder_width=args.decoder_width,
        device=device,
    )
    normalization = terrain_checkpoint["normalization"]
    val_base = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.split,
        fold_id=args.fold_id,
        role="val",
        readiness="core",
    )
    val_dataset = EvaluationDataset(
        val_base, mean=normalization["mean"], scale=normalization["scale"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    selections, selection_grid = select_by_source(
        visual,
        terrain,
        val_loader,
        threshold=float(parent["threshold"]),
        device=device,
    )
    x, y, events, training_metadata = collect_gate_training_rows(
        visual,
        terrain,
        val_loader,
        selections=selections,
        threshold=float(parent["threshold"]),
        pixels_per_sample=args.pixels_per_sample,
        seed=args.seed,
        device=device,
    )
    gate, fit_metadata = fit_gate(
        x,
        y,
        events,
        c_value=args.c_value,
        harm_weight=args.harm_weight,
        seed=args.seed,
    )
    receipt = {
        "schema_version": "pild_benefit_gate_receipt.v1",
        "frozen_before_test_open": True,
        "scope": "validation-fitted utility of visual/Terrain disagreement proposals",
        "source_selections": selections,
        "source_selection_grid": selection_grid,
        "gate": gate.to_dict(),
        "gate_threshold": args.gate_threshold,
        "training": {**training_metadata, **fit_metadata},
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "protocol_summary_sha256": sha256_file(args.protocol_summary),
            "split_sha256": schema["split_sha256"],
            "fold_id": args.fold_id,
            "seed": args.seed,
            "parent_v_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
            "terrain_checkpoint_sha256": terrain_receipt["checkpoint_sha256"],
        },
    }
    write_json(stage / "gate_receipt.json", receipt)
    gate_receipt_sha256 = sha256_file(stage / "gate_receipt.json")

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
        sample_rows, event_rows, corpus = evaluate(
            visual,
            terrain,
            loader,
            selections=selections,
            gate=gate,
            visual_threshold=float(parent["threshold"]),
            gate_threshold=args.gate_threshold,
            device=device,
        )
        condition_results.append({"condition": condition, **corpus})
        if condition == "aligned":
            aligned_rows, aligned_events = sample_rows, event_rows
    write_csv(stage / "per_sample_metrics.csv", aligned_rows)
    write_csv(stage / "per_event_metrics.csv", aligned_events)
    aligned = condition_results[0]
    result = {
        "schema_version": "pild_benefit_gate_result.v1",
        "status": "COMPLETE",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "contract": {
            "visual_anchor": "frozen Prithvi-EO-2.0-300M-TL",
            "proposal": "validation-selected bounded Terrain residual",
            "gate": "validation-fitted linear expected-benefit classifier",
            "test_inference": "label-free features only; exact rejection fallback",
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
        "parent_v_receipt": parent_receipt,
        "terrain_receipt": terrain_receipt,
        "prithvi_provenance": provenance,
        "elapsed_seconds": time.time() - started,
    }
    write_json(stage / "result.json", result)
    write_json(
        stage / "run_summary.json",
        {
            "status": "COMPLETE",
            "gate_receipt_sha256": gate_receipt_sha256,
            "result_sha256": sha256_file(stage / "result.json"),
            "delta_iou": aligned["delta_iou"],
            "delta_ap": aligned["delta_ap"],
            "rer": aligned["rer"],
            "corrected_to_harmed": aligned["corrected"]
            / max(aligned["harmed"], 1),
        },
    )
    os.replace(stage, outdir)
    print(
        f"{args.fold_id}: benefit gate delta_iou={aligned['delta_iou']:+.6f}, "
        f"delta_ap={aligned['delta_ap']:+.6f}, rer={aligned['rer']:+.2%}, "
        f"accepted={aligned['acceptance_rate']:.1%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
