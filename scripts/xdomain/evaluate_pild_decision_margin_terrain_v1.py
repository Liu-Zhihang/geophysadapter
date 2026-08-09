#!/usr/bin/env python3
"""Decision-boundary-aware Terrain correction for heterogeneous PILD sources.

Unlike entropy around probability 0.5, this gate is centered on the frozen
visual decision threshold. Positive and negative Terrain directions receive
separate validation-selected strengths, allowing asymmetric FN recovery and
FP suppression. Selection remains source-specific and test-blind.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
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
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file
from train_pild_sen12_roleaware_v1 import validate_protocol_schema
from train_pild_support_only_terrain_v1 import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_SUMMARY,
    validate_parent_v_checkpoint,
    write_json,
)


ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)
TEMPERATURES = (0.5, 1.0, 2.0, 4.0, math.inf)
CONFIGURATIONS = tuple(
    (positive_alpha, negative_alpha, temperature)
    for positive_alpha in ALPHAS
    for negative_alpha in ALPHAS
    for temperature in TEMPERATURES
    if positive_alpha != 0.0 or negative_alpha != 0.0
) + ((0.0, 0.0, math.inf),)


def subset(value: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return value[torch.tensor(indices, device=value.device)]


def fuse_logits(
    visual_logits: torch.Tensor,
    terrain_logits: torch.Tensor,
    q_t: torch.Tensor,
    *,
    threshold: float,
    positive_alpha: float,
    negative_alpha: float,
    temperature: float,
) -> torch.Tensor:
    """Apply bounded Terrain direction near the deployed decision boundary."""

    threshold = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    threshold_logit = math.log(threshold / (1.0 - threshold))
    if math.isinf(temperature):
        gate = torch.ones_like(visual_logits)
    else:
        gate = torch.exp(
            -torch.abs(visual_logits - threshold_logit) / float(temperature)
        )
    direction = torch.tanh(terrain_logits) * q_t.clamp(0.0, 1.0)
    strength = torch.where(
        direction >= 0,
        torch.full_like(direction, float(positive_alpha)),
        torch.full_like(direction, float(negative_alpha)),
    )
    return visual_logits + gate * strength * direction


@torch.no_grad()
def select_by_source(
    visual_model: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    threshold: float,
    device: torch.device,
    max_alpha: float = math.inf,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    baseline_histograms: dict[str, BinaryHistogram] = {}
    histograms: dict[str, dict[tuple[float, float, float], BinaryHistogram]] = {}
    counts: dict[
        str, dict[tuple[float, float, float], defaultdict[str, int]]
    ] = {}
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual_model, terrain_model, batch, device=device
        )
        baseline_probability = torch.sigmoid(visual_logits)
        source_indices: dict[str, list[int]] = defaultdict(list)
        for index, dataset_id in enumerate(batch["dataset_id"]):
            source_indices[str(dataset_id)].append(index)
        for dataset_id, indices in source_indices.items():
            v_logits = subset(visual_logits, indices)
            t_logits = subset(terrain_logits, indices)
            support = subset(q_t, indices)
            labels = subset(target, indices)
            valid_pixels = subset(valid, indices)
            baseline = subset(baseline_probability, indices)
            baseline_histograms.setdefault(dataset_id, BinaryHistogram()).update(
                baseline, labels, valid_pixels
            )
            if dataset_id not in histograms:
                histograms[dataset_id] = {
                    configuration: BinaryHistogram()
                    for configuration in CONFIGURATIONS
                    if max(configuration[0], configuration[1]) <= max_alpha
                }
                counts[dataset_id] = {
                    configuration: defaultdict(int)
                    for configuration in CONFIGURATIONS
                    if max(configuration[0], configuration[1]) <= max_alpha
                }
            for configuration in histograms[dataset_id]:
                positive_alpha, negative_alpha, temperature = configuration
                if positive_alpha == 0.0 and negative_alpha == 0.0:
                    adapted = baseline
                else:
                    adapted = torch.sigmoid(
                        fuse_logits(
                            v_logits,
                            t_logits,
                            support,
                            threshold=threshold,
                            positive_alpha=positive_alpha,
                            negative_alpha=negative_alpha,
                            temperature=temperature,
                        )
                    )
                if not torch.isfinite(adapted).all():
                    continue
                histograms[dataset_id][configuration].update(
                    adapted, labels, valid_pixels
                )
                pair = counts_from_predictions(
                    baseline,
                    adapted,
                    labels,
                    valid_pixels,
                    threshold=threshold,
                )
                for key, value in pair.items():
                    counts[dataset_id][configuration][key] += int(value)

    selections: dict[str, dict[str, Any]] = {}
    grids: dict[str, list[dict[str, Any]]] = {}
    for dataset_id in sorted(baseline_histograms):
        baseline_ap = baseline_histograms[dataset_id].average_precision()
        rows: list[dict[str, Any]] = []
        for configuration in histograms[dataset_id]:
            positive_alpha, negative_alpha, temperature = configuration
            row = {
                "dataset_id": dataset_id,
                "positive_alpha": positive_alpha,
                "negative_alpha": negative_alpha,
                "temperature": temperature,
                **metrics_from_pair_counts(counts[dataset_id][configuration]),
                "baseline_ap": baseline_ap,
                "adapted_ap": histograms[dataset_id][configuration].average_precision(),
            }
            row["delta_ap"] = row["adapted_ap"] - row["baseline_ap"]
            row["validation_feasible"] = bool(
                row["delta_iou"] >= -1e-12 and row["rer"] >= -1e-12
            )
            rows.append(row)
        selected = max(
            (
                row
                for row in rows
                if row["validation_feasible"]
                and math.isfinite(float(row["delta_iou"]))
                and math.isfinite(float(row["rer"]))
            ),
            key=lambda row: (
                row["delta_iou"],
                row["rer"],
                row["delta_ap"],
                -(row["positive_alpha"] + row["negative_alpha"]),
                -(
                    1e9
                    if math.isinf(float(row["temperature"]))
                    else float(row["temperature"])
                ),
            ),
        )
        selections[dataset_id] = dict(selected)
        grids[dataset_id] = rows
    if not selections:
        raise RuntimeError("validation contains no source for calibration")
    return selections, grids


@torch.no_grad()
def evaluate(
    visual_model: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    selections: Mapping[str, Mapping[str, Any]],
    threshold: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_histogram = BinaryHistogram()
    adapted_histogram = BinaryHistogram()
    event_histograms: dict[str, tuple[BinaryHistogram, BinaryHistogram]] = {}
    rows: list[dict[str, Any]] = []
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual_model, terrain_model, batch, device=device
        )
        baseline_probability = torch.sigmoid(visual_logits)
        adapted_logits = visual_logits.clone()
        source_indices: dict[str, list[int]] = defaultdict(list)
        for index, dataset_id in enumerate(batch["dataset_id"]):
            source_indices[str(dataset_id)].append(index)
        for dataset_id, indices in source_indices.items():
            if dataset_id not in selections:
                raise RuntimeError(f"test source absent from validation: {dataset_id}")
            selected = selections[dataset_id]
            adapted_logits[indices] = fuse_logits(
                subset(visual_logits, indices),
                subset(terrain_logits, indices),
                subset(q_t, indices),
                threshold=threshold,
                positive_alpha=float(selected["positive_alpha"]),
                negative_alpha=float(selected["negative_alpha"]),
                temperature=float(selected["temperature"]),
            )
        adapted_probability = torch.sigmoid(adapted_logits)
        baseline_histogram.update(baseline_probability, target, valid)
        adapted_histogram.update(adapted_probability, target, valid)
        for index in range(target.shape[0]):
            dataset_id = str(batch["dataset_id"][index])
            pair = counts_from_predictions(
                baseline_probability[index : index + 1],
                adapted_probability[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                threshold=threshold,
            )
            baseline_sample = BinaryHistogram()
            adapted_sample = BinaryHistogram()
            baseline_sample.update(
                baseline_probability[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
            )
            adapted_sample.update(
                adapted_probability[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
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
            selected = selections[dataset_id]
            rows.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "dataset_id": dataset_id,
                    "source_id": str(batch["source_id"][index]),
                    "source_event_id": str(batch["source_event_id"][index]),
                    "canonical_event_id": event_id,
                    "positive_alpha": float(selected["positive_alpha"]),
                    "negative_alpha": float(selected["negative_alpha"]),
                    "temperature": float(selected["temperature"]),
                    **pair,
                    **metrics_from_pair_counts(pair),
                    "baseline_ap": baseline_sample.average_precision(),
                    "adapted_ap": adapted_sample.average_precision(),
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
    }
    corpus["delta_ap"] = corpus["adapted_ap"] - corpus["baseline_ap"]
    return rows, event_rows, corpus


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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

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
    selections, grids = select_by_source(
        visual,
        terrain,
        val_loader,
        threshold=float(parent["threshold"]),
        device=device,
    )
    receipt = {
        "schema_version": "pild_decision_margin_selection.v1",
        "selection_scope": "validation-only by known dataset source",
        "selection_rule": "per source asymmetric alpha and boundary temperature; identity feasible",
        "frozen_before_test_open": True,
        "decision_threshold": float(parent["threshold"]),
        "selections": selections,
        "grids": grids,
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
    write_json(stage / "selection.json", receipt)
    selection_sha256 = sha256_file(stage / "selection.json")

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
    condition_results: list[dict[str, Any]] = []
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
        condition_results.append({"condition": condition, **corpus})
        if condition == "aligned":
            aligned_rows, aligned_events = sample_rows, event_rows
    write_csv(stage / "per_sample_metrics.csv", aligned_rows)
    write_csv(stage / "per_event_metrics.csv", aligned_events)
    aligned = condition_results[0]
    result = {
        "schema_version": "pild_decision_margin_terrain_result.v1",
        "status": "COMPLETE",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "contract": {
            "fusion": "threshold-margin gated asymmetric bounded Terrain residual",
            "selection": "per-source validation-only, frozen before test",
            "abstention": "positive_alpha=negative_alpha=0 identity fallback",
        },
        "selections": selections,
        "selection_sha256": selection_sha256,
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
        stage / "DONE.json",
        {
            "status": "COMPLETE",
            "selection_sha256": selection_sha256,
            "result_sha256": sha256_file(stage / "result.json"),
            "per_sample_metrics_sha256": sha256_file(
                stage / "per_sample_metrics.csv"
            ),
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
