#!/usr/bin/env python3
"""Validation-select regional and local Terrain corrections for frozen PILD V.

The regional branch captures low-frequency susceptibility. The local branch is
admitted only when validation evidence shows both incremental value over the
matched regional-only setting and an aligned-over-roll advantage.
"""

from __future__ import annotations

import argparse
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
from torch.nn import functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pild_support_only_controls_v1 import ControlledTerrainDataset  # noqa: E402
from evaluate_pild_support_only_additive_v1 import (  # noqa: E402
    EvaluationDataset,
    aggregate_samples_to_events,
    build_models,
    counts_from_predictions,
    metrics_from_pair_counts,
    validate_terrain_checkpoint,
    write_csv,
)
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file  # noqa: E402
from train_pild_sen12_roleaware_v1 import (  # noqa: E402
    BinaryHistogram,
    validate_protocol_schema,
)
from train_pild_support_only_terrain_v1 import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_SUMMARY,
    validate_parent_v_checkpoint,
    write_json,
)


REGIONAL_ALPHAS = (0.0, 0.25, 0.5, 1.0)
LOCAL_ALPHAS = (0.0, 0.25, 0.5, 1.0)
REGIONAL_POWERS = (0.0, 1.0)
LOCAL_POWERS = (1.0, 2.0)
REGIONAL_KERNEL = 65
ROLL_PIXELS = 64
MIN_LOCAL_GAIN = 1e-4
MIN_ALIGNMENT_MARGIN = 1e-4
CONTROL_CONDITIONS = (
    "aligned",
    "terrain-zero",
    "terrain-shift32",
    "terrain-roll64",
    "terrain-donor",
)


def uncertainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    return (1.0 - 2.0 * torch.abs(probability - 0.5)).clamp(0.0, 1.0)


def terrain_levels(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    regional = F.avg_pool2d(
        logits,
        kernel_size=REGIONAL_KERNEL,
        stride=1,
        padding=REGIONAL_KERNEL // 2,
    )
    local = logits - regional
    return torch.tanh(regional), torch.tanh(local)


def fuse_twolevel(
    visual_logits: torch.Tensor,
    terrain_logits: torch.Tensor,
    q_t: torch.Tensor,
    *,
    regional_alpha: float,
    local_alpha: float,
    regional_power: float,
    local_power: float,
) -> torch.Tensor:
    uncertainty = uncertainty_from_logits(visual_logits)
    regional_direction, local_direction = terrain_levels(terrain_logits)
    regional_gate = (
        torch.ones_like(uncertainty)
        if regional_power == 0
        else uncertainty.pow(regional_power)
    )
    local_gate = uncertainty.pow(local_power)
    correction = (
        float(regional_alpha) * regional_gate * regional_direction
        + float(local_alpha) * local_gate * local_direction
    )
    return visual_logits + q_t.clamp(0.0, 1.0) * correction


def configurations() -> list[tuple[float, float, float, float]]:
    output: list[tuple[float, float, float, float]] = []
    for regional_alpha in REGIONAL_ALPHAS:
        regional_powers = REGIONAL_POWERS if regional_alpha > 0 else (0.0,)
        for local_alpha in LOCAL_ALPHAS:
            local_powers = LOCAL_POWERS if local_alpha > 0 else (1.0,)
            for regional_power in regional_powers:
                for local_power in local_powers:
                    output.append(
                        (
                            regional_alpha,
                            local_alpha,
                            regional_power,
                            local_power,
                        )
                    )
    return output


def add_counts(target: defaultdict[str, int], values: Mapping[str, int]) -> None:
    for key, value in values.items():
        target[key] += int(value)


@torch.no_grad()
def select_on_validation(
    visual_model: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = configurations()
    baseline_histogram = BinaryHistogram()
    aligned_histograms = {candidate: BinaryHistogram() for candidate in candidates}
    roll_histograms = {candidate: BinaryHistogram() for candidate in candidates}
    aligned_counts = {candidate: defaultdict(int) for candidate in candidates}
    roll_counts = {candidate: defaultdict(int) for candidate in candidates}

    for batch in loader:
        optical = batch["optical"].to(device) * 10_000.0
        visual_logits = visual_model(
            optical,
            batch["temporal_coords"].to(device),
            batch["location_coords"].to(device),
        )["logits"]
        terrain = batch["terrain"].to(device)
        q_t = batch["q_t"].to(device)
        target = batch["mask"].to(device)
        valid = batch["valid_mask"].to(device)
        terrain_logits, _ = terrain_model(terrain)
        rolled_terrain = torch.roll(
            terrain, shifts=(ROLL_PIXELS, ROLL_PIXELS), dims=(-2, -1)
        )
        rolled_q_t = torch.roll(
            q_t, shifts=(ROLL_PIXELS, ROLL_PIXELS), dims=(-2, -1)
        )
        rolled_logits, _ = terrain_model(rolled_terrain)
        baseline_probability = torch.sigmoid(visual_logits)
        baseline_histogram.update(baseline_probability, target, valid)

        for candidate in candidates:
            regional_alpha, local_alpha, regional_power, local_power = candidate
            aligned_probability = torch.sigmoid(
                fuse_twolevel(
                    visual_logits,
                    terrain_logits,
                    q_t,
                    regional_alpha=regional_alpha,
                    local_alpha=local_alpha,
                    regional_power=regional_power,
                    local_power=local_power,
                )
            )
            roll_probability = torch.sigmoid(
                fuse_twolevel(
                    visual_logits,
                    rolled_logits,
                    rolled_q_t,
                    regional_alpha=regional_alpha,
                    local_alpha=local_alpha,
                    regional_power=regional_power,
                    local_power=local_power,
                )
            )
            aligned_histograms[candidate].update(
                aligned_probability, target, valid
            )
            roll_histograms[candidate].update(roll_probability, target, valid)
            add_counts(
                aligned_counts[candidate],
                counts_from_predictions(
                    baseline_probability,
                    aligned_probability,
                    target,
                    valid,
                    threshold=threshold,
                ),
            )
            add_counts(
                roll_counts[candidate],
                counts_from_predictions(
                    baseline_probability,
                    roll_probability,
                    target,
                    valid,
                    threshold=threshold,
                ),
            )

    baseline_ap = baseline_histogram.average_precision()
    rows: list[dict[str, Any]] = []
    row_by_candidate: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    for candidate in candidates:
        regional_alpha, local_alpha, regional_power, local_power = candidate
        aligned = metrics_from_pair_counts(aligned_counts[candidate])
        rolled = metrics_from_pair_counts(roll_counts[candidate])
        row = {
            "regional_alpha": regional_alpha,
            "local_alpha": local_alpha,
            "regional_power": regional_power,
            "local_power": local_power,
            **{f"aligned_{key}": value for key, value in aligned.items()},
            **{f"roll_{key}": value for key, value in rolled.items()},
            "baseline_ap": baseline_ap,
            "aligned_ap": aligned_histograms[candidate].average_precision(),
            "roll_ap": roll_histograms[candidate].average_precision(),
        }
        row["aligned_delta_ap"] = row["aligned_ap"] - baseline_ap
        row["roll_delta_ap"] = row["roll_ap"] - baseline_ap
        row["alignment_margin_iou"] = (
            row["aligned_delta_iou"] - row["roll_delta_iou"]
        )
        rows.append(row)
        row_by_candidate[candidate] = row

    for row in rows:
        regional_only_key = (
            float(row["regional_alpha"]),
            0.0,
            float(row["regional_power"]),
            1.0,
        )
        regional_only = row_by_candidate[regional_only_key]
        row["local_increment_iou"] = (
            row["aligned_delta_iou"] - regional_only["aligned_delta_iou"]
        )
        base_feasible = bool(
            row["aligned_delta_iou"] >= -1e-12
            and row["aligned_rer"] >= -1e-12
        )
        local_feasible = bool(
            float(row["local_alpha"]) == 0.0
            or (
                row["local_increment_iou"] >= MIN_LOCAL_GAIN
                and row["alignment_margin_iou"] >= MIN_ALIGNMENT_MARGIN
            )
        )
        row["validation_feasible"] = base_feasible and local_feasible

    feasible = [row for row in rows if row["validation_feasible"]]
    if not feasible:
        raise RuntimeError("two-level validation grid has no feasible identity")
    selected = max(
        feasible,
        key=lambda row: (
            row["aligned_delta_iou"],
            row["aligned_rer"],
            row["aligned_delta_ap"],
            row["alignment_margin_iou"],
            -row["regional_alpha"] - row["local_alpha"],
        ),
    )
    return dict(selected), rows


@torch.no_grad()
def evaluate(
    visual_model: torch.nn.Module,
    terrain_model: torch.nn.Module,
    loader: DataLoader,
    *,
    selection: Mapping[str, Any],
    threshold: float,
    device: torch.device,
    export_samples: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_histogram = BinaryHistogram()
    adapted_histogram = BinaryHistogram()
    sample_rows: list[dict[str, Any]] = []
    corpus_counts: defaultdict[str, int] = defaultdict(int)
    for batch in loader:
        optical = batch["optical"].to(device) * 10_000.0
        visual_logits = visual_model(
            optical,
            batch["temporal_coords"].to(device),
            batch["location_coords"].to(device),
        )["logits"]
        terrain_logits, _ = terrain_model(batch["terrain"].to(device))
        target = batch["mask"].to(device)
        valid = batch["valid_mask"].to(device)
        baseline_probability = torch.sigmoid(visual_logits)
        adapted_probability = torch.sigmoid(
            fuse_twolevel(
                visual_logits,
                terrain_logits,
                batch["q_t"].to(device),
                regional_alpha=float(selection["regional_alpha"]),
                local_alpha=float(selection["local_alpha"]),
                regional_power=float(selection["regional_power"]),
                local_power=float(selection["local_power"]),
            )
        )
        baseline_histogram.update(baseline_probability, target, valid)
        adapted_histogram.update(adapted_probability, target, valid)
        batch_size = target.shape[0]
        for index in range(batch_size):
            counts = counts_from_predictions(
                baseline_probability[index : index + 1],
                adapted_probability[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                threshold=threshold,
            )
            add_counts(corpus_counts, counts)
            if export_samples:
                sample_rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "dataset_id": str(batch["dataset_id"][index]),
                        "source_id": str(batch["source_id"][index]),
                        "source_event_id": str(batch["source_event_id"][index]),
                        "canonical_event_id": str(
                            batch["canonical_event_id"][index]
                        ),
                        **counts,
                        **metrics_from_pair_counts(counts),
                    }
                )
    corpus = {
        **dict(corpus_counts),
        **metrics_from_pair_counts(corpus_counts),
        "baseline_ap": baseline_histogram.average_precision(),
        "adapted_ap": adapted_histogram.average_precision(),
    }
    corpus["delta_ap"] = corpus["adapted_ap"] - corpus["baseline_ap"]
    if not export_samples:
        return [], [], corpus
    event_rows = aggregate_samples_to_events(sample_rows)
    corpus["n_samples"] = len(sample_rows)
    corpus["n_events"] = len(event_rows)
    return sample_rows, event_rows, corpus


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
        raise FileExistsError(f"refusing to overwrite existing evaluation: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    stage = outdir.parent / f".{outdir.name}.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    command = shlex.join([sys.executable, *sys.argv])
    (stage / "run.log").write_text(command + "\n", encoding="utf-8")
    started = time.time()
    device = torch.device(args.device)
    visual_model, terrain_model, provenance = build_models(
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
        val_base,
        mean=normalization["mean"],
        scale=normalization["scale"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    selection, grid = select_on_validation(
        visual_model,
        terrain_model,
        val_loader,
        threshold=float(parent["threshold"]),
        device=device,
    )
    selection_receipt = {
        "schema_version": "pild_twolevel_terrain_selection.v1",
        "selection_scope": "validation-only with validation-roll local gate",
        "frozen_before_test_open": True,
        "regional_kernel": REGIONAL_KERNEL,
        "roll_pixels": ROLL_PIXELS,
        "minimum_local_gain": MIN_LOCAL_GAIN,
        "minimum_alignment_margin": MIN_ALIGNMENT_MARGIN,
        "selected": selection,
        "grid": grid,
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "split_sha256": schema["split_sha256"],
            "fold_id": args.fold_id,
            "seed": args.seed,
            "parent_v_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
            "terrain_checkpoint_sha256": terrain_receipt["checkpoint_sha256"],
        },
    }
    write_json(stage / "selection.json", selection_receipt)

    test_base = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.split,
        fold_id=args.fold_id,
        role="test",
        readiness="core",
    )
    test_dataset = EvaluationDataset(
        test_base,
        mean=normalization["mean"],
        scale=normalization["scale"],
    )
    control_results: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    aligned_corpus: dict[str, Any] | None = None
    for condition in CONTROL_CONDITIONS:
        controlled = ControlledTerrainDataset(test_dataset, condition)
        loader = DataLoader(
            controlled,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        rows, events, corpus = evaluate(
            visual_model,
            terrain_model,
            loader,
            selection=selection,
            threshold=float(parent["threshold"]),
            device=device,
            export_samples=condition == "aligned",
        )
        control_results.append({"condition": condition, **corpus})
        if condition == "aligned":
            sample_rows, event_rows, aligned_corpus = rows, events, corpus
    if aligned_corpus is None:
        raise AssertionError("aligned control was not evaluated")
    write_csv(stage / "per_sample_metrics.csv", sample_rows)
    write_csv(stage / "per_event_metrics.csv", event_rows)
    write_json(
        stage / "result.json",
        {
            "schema_version": "pild_twolevel_terrain_result.v1",
            "status": "COMPLETE",
            "fold_id": args.fold_id,
            "seed": args.seed,
            "contract": (
                "regional low-frequency susceptibility plus validation-roll-gated "
                "local Terrain residual"
            ),
            "selection": selection,
            "test": aligned_corpus,
            "controls": control_results,
            "schema_validation": schema,
            "parent_v_receipt": parent_receipt,
            "terrain_receipt": terrain_receipt,
            "prithvi_provenance": provenance,
            "elapsed_seconds": time.time() - started,
        },
    )
    write_json(
        stage / "DONE.json",
        {
            "status": "COMPLETE",
            "result_sha256": sha256_file(stage / "result.json"),
            "selection_sha256": sha256_file(stage / "selection.json"),
            "per_sample_metrics_sha256": sha256_file(
                stage / "per_sample_metrics.csv"
            ),
            "per_event_metrics_sha256": sha256_file(
                stage / "per_event_metrics.csv"
            ),
        },
    )
    os.replace(stage, outdir)
    print(
        json.dumps(
            {
                "outdir": str(outdir),
                "selection": selection,
                "test": aligned_corpus,
                "controls": [
                    {
                        "condition": row["condition"],
                        "delta_iou": row["delta_iou"],
                        "rer": row["rer"],
                    }
                    for row in control_results
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
