#!/usr/bin/env python3
"""Measure the validation-only correction capacity of a frozen Terrain expert.

This is a diagnostic upper bound, not a deployable model. For each validation
pixel, the oracle may keep the visual prediction or choose any predeclared
``alpha x uncertainty_power`` Terrain correction that makes the hard decision
correct. Test data are never constructed.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from evaluate_pild_source_calibrated_terrain_v1 import CONFIGURATIONS
from evaluate_pild_support_only_additive_v1 import (
    EvaluationDataset,
    build_models,
    fuse_logits,
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


def empty_counts() -> defaultdict[str, int]:
    return defaultdict(int)


def update_counts(
    counts: defaultdict[str, int],
    *,
    baseline: torch.Tensor,
    candidates: list[torch.Tensor],
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    keep = valid.bool()
    baseline_correct = baseline == truth
    any_candidate_correct = torch.zeros_like(baseline_correct)
    for candidate in candidates:
        any_candidate_correct |= candidate == truth
    oracle_correct = baseline_correct | any_candidate_correct
    oracle = torch.where(oracle_correct, truth, baseline)

    counts["baseline_tp"] += int((baseline & truth & keep).sum().item())
    counts["baseline_fp"] += int((baseline & ~truth & keep).sum().item())
    counts["baseline_fn"] += int((~baseline & truth & keep).sum().item())
    counts["baseline_tn"] += int((~baseline & ~truth & keep).sum().item())
    counts["oracle_tp"] += int((oracle & truth & keep).sum().item())
    counts["oracle_fp"] += int((oracle & ~truth & keep).sum().item())
    counts["oracle_fn"] += int((~oracle & truth & keep).sum().item())
    counts["oracle_tn"] += int((~oracle & ~truth & keep).sum().item())
    counts["correctable"] += int(
        ((~baseline_correct) & any_candidate_correct & keep).sum().item()
    )
    baseline_fn = (~baseline) & truth & keep
    baseline_fp = baseline & (~truth) & keep
    counts["rescue_correctable"] += int(
        (baseline_fn & any_candidate_correct).sum().item()
    )
    counts["veto_correctable"] += int(
        (baseline_fp & any_candidate_correct).sum().item()
    )
    counts["baseline_correct"] += int((baseline_correct & keep).sum().item())
    counts["valid_pixels"] += int(keep.sum().item())


def metrics(counts: Mapping[str, int]) -> dict[str, float | int]:
    baseline_tp = int(counts["baseline_tp"])
    baseline_fp = int(counts["baseline_fp"])
    baseline_fn = int(counts["baseline_fn"])
    oracle_tp = int(counts["oracle_tp"])
    oracle_fp = int(counts["oracle_fp"])
    oracle_fn = int(counts["oracle_fn"])
    baseline_errors = baseline_fp + baseline_fn
    oracle_errors = oracle_fp + oracle_fn
    baseline_iou = baseline_tp / max(baseline_tp + baseline_fp + baseline_fn, 1)
    oracle_iou = oracle_tp / max(oracle_tp + oracle_fp + oracle_fn, 1)
    correctable = int(counts["correctable"])
    rescue_correctable = int(counts["rescue_correctable"])
    veto_correctable = int(counts["veto_correctable"])
    return {
        **{key: int(value) for key, value in counts.items()},
        "baseline_errors": baseline_errors,
        "oracle_errors": oracle_errors,
        "baseline_iou": baseline_iou,
        "oracle_iou": oracle_iou,
        "oracle_delta_iou": oracle_iou - baseline_iou,
        "oracle_rer": (baseline_errors - oracle_errors) / max(baseline_errors, 1),
        "correctable_fraction_of_baseline_errors": correctable
        / max(baseline_errors, 1),
        "rescue_fraction_of_baseline_fn": rescue_correctable
        / max(baseline_fn, 1),
        "veto_fraction_of_baseline_fp": veto_correctable
        / max(baseline_fp, 1),
        "rescue_fraction_of_correctable": rescue_correctable
        / max(correctable, 1),
        "veto_fraction_of_correctable": veto_correctable
        / max(correctable, 1),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def analyze(
    visual_model: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = empty_counts()
    by_source: dict[str, defaultdict[str, int]] = defaultdict(empty_counts)
    by_event: dict[str, defaultdict[str, int]] = defaultdict(empty_counts)
    event_sources: dict[str, str] = {}

    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual_model, terrain_model, batch, device=device
        )
        truth = target >= 0.5
        baseline = torch.sigmoid(visual_logits) >= threshold
        candidates = [
            torch.sigmoid(
                fuse_logits(
                    visual_logits,
                    terrain_logits,
                    q_t,
                    alpha=alpha,
                    uncertainty_power=power,
                )
            )
            >= threshold
            for alpha, power in CONFIGURATIONS
            if alpha > 0
        ]
        update_counts(
            corpus,
            baseline=baseline,
            candidates=candidates,
            truth=truth,
            valid=valid,
        )

        source_indices: dict[str, list[int]] = defaultdict(list)
        event_indices: dict[str, list[int]] = defaultdict(list)
        for index, dataset_id in enumerate(batch["dataset_id"]):
            source_indices[str(dataset_id)].append(index)
            event_id = str(batch["canonical_event_id"][index])
            event_indices[event_id].append(index)
            event_sources[event_id] = str(dataset_id)
        for dataset_id, indices in source_indices.items():
            index_tensor = torch.tensor(indices, device=device)
            update_counts(
                by_source[dataset_id],
                baseline=baseline[index_tensor],
                candidates=[candidate[index_tensor] for candidate in candidates],
                truth=truth[index_tensor],
                valid=valid[index_tensor],
            )
        for event_id, indices in event_indices.items():
            index_tensor = torch.tensor(indices, device=device)
            update_counts(
                by_event[event_id],
                baseline=baseline[index_tensor],
                candidates=[candidate[index_tensor] for candidate in candidates],
                truth=truth[index_tensor],
                valid=valid[index_tensor],
            )

    source_rows = [
        {"dataset_id": dataset_id, **metrics(counts)}
        for dataset_id, counts in sorted(by_source.items())
    ]
    event_rows = [
        {
            "canonical_event_id": event_id,
            "dataset_id": event_sources[event_id],
            **metrics(counts),
        }
        for event_id, counts in sorted(by_event.items())
    ]
    return metrics(corpus), source_rows, event_rows


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
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
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
    condition_results: list[dict[str, Any]] = []
    all_source_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        val_loader = DataLoader(
            ControlledTerrainDataset(val_dataset, condition),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        corpus, source_rows, event_rows = analyze(
            visual,
            terrain,
            val_loader,
            threshold=float(parent["threshold"]),
            device=device,
        )
        condition_results.append({"condition": condition, **corpus})
        all_source_rows.extend(
            {"condition": condition, **row} for row in source_rows
        )
        all_event_rows.extend(
            {"condition": condition, **row} for row in event_rows
        )
    write_csv(stage / "per_source_oracle.csv", all_source_rows)
    write_csv(stage / "per_event_oracle.csv", all_event_rows)
    aligned = condition_results[0]
    result = {
        "schema_version": "pild_validation_oracle.v1",
        "status": "DIAGNOSTIC_COMPLETE",
        "scope": "validation-only label-aware upper bound; not deployable performance",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "threshold": float(parent["threshold"]),
        "configurations": [
            {"alpha": alpha, "uncertainty_power": power}
            for alpha, power in CONFIGURATIONS
            if alpha > 0
        ],
        "validation": {
            **aligned,
            "n_samples": len(val_dataset),
            "n_events": len(all_event_rows) // len(CONDITIONS),
        },
        "conditions": condition_results,
        "contrasts": {
            row["condition"]: {
                "aligned_minus_control_oracle_delta_iou": aligned[
                    "oracle_delta_iou"
                ]
                - row["oracle_delta_iou"],
                "aligned_minus_control_oracle_rer": aligned["oracle_rer"]
                - row["oracle_rer"],
            }
            for row in condition_results[1:]
        },
        "by_source": [
            row for row in all_source_rows if row["condition"] == "aligned"
        ],
        "parent_v_receipt": parent_receipt,
        "terrain_receipt": terrain_receipt,
        "prithvi_provenance": provenance,
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "protocol_summary_sha256": sha256_file(args.protocol_summary),
            "split_sha256": schema["split_sha256"],
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(stage / "result.json", result)
    write_json(
        stage / "run_summary.json",
        {
            "status": "DIAGNOSTIC_COMPLETE",
            "result_sha256": sha256_file(stage / "result.json"),
            "n_samples": len(val_dataset),
            "n_events": len(all_event_rows) // len(CONDITIONS),
            "oracle_delta_iou": aligned["oracle_delta_iou"],
            "oracle_rer": aligned["oracle_rer"],
        },
    )
    os.replace(stage, outdir)
    print(
        f"{args.fold_id}: validation oracle delta_iou="
        f"{aligned['oracle_delta_iou']:+.6f}, rer={aligned['oracle_rer']:+.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
