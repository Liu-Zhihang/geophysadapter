#!/usr/bin/env python3
"""Validation-selected additive fusion of frozen PILD V and Terrain support.

The protocol is deliberately two-stage:

1. Load only validation data and select ``alpha x uncertainty_power``.
2. Persist a frozen selection receipt, then construct the test dataset and
   evaluate exactly that configuration once.

Terrain uses the checkpoint's audited support-only schema and spatial pooling
is fixed to one (no smoothing or post-hoc morphology).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pild_sen12_training_loader_v2 import (  # noqa: E402
    UnifiedPILDSen12Dataset,
    sha256_file,
)
from sen12_prithvi_v2 import (  # noqa: E402
    PrithviEO2ChangeModel,
    load_prithvi_encoder,
)
from sen12_terrain_v2 import SupportOnlyMultiScaleTerrainPyramid  # noqa: E402
from train_pild_sen12_roleaware_v1 import (  # noqa: E402
    BinaryHistogram,
    metrics_from_counts,
    resolve_terrain_contract,
    tensor_sha256,
    validate_protocol_schema,
)
from train_pild_support_only_terrain_v1 import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_SUMMARY,
    json_safe,
    validate_parent_v_checkpoint,
    write_json,
)


ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
UNCERTAINTY_POWERS = (0.0, 1.0, 2.0)
SPATIAL_POOL = 1


class EvaluationDataset(Dataset[dict[str, Any]]):
    """Normalize an audited Terrain schema while preserving unified visual inputs."""

    def __init__(
        self,
        base: UnifiedPILDSen12Dataset,
        *,
        mean: Iterable[float],
        scale: Iterable[float],
    ) -> None:
        self.base = base
        self.frame = base.frame
        self.mean = torch.tensor(tuple(mean), dtype=torch.float32)[:, None, None]
        self.scale = torch.tensor(tuple(scale), dtype=torch.float32)[:, None, None]
        if self.mean.ndim != 3 or self.mean.shape[1:] != (1, 1):
            raise ValueError("Terrain mean must have shape [channels,1,1]")
        if self.scale.shape != self.mean.shape:
            raise ValueError("Terrain mean/scale shapes differ")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all():
            raise ValueError("Terrain normalization is non-finite")
        if not torch.all(self.scale > 0):
            raise ValueError("Terrain normalization scale must be positive")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        terrain = item["terrain"].float()
        q_t = item["terrain_valid"].float()
        if q_t.ndim == 2:
            q_t = q_t[None]
        finite = torch.isfinite(terrain).all(dim=0, keepdim=True)
        q_t = q_t * finite.to(q_t.dtype)
        normalized = (terrain - self.mean) / self.scale
        item["terrain"] = torch.where(q_t > 0, normalized, torch.zeros_like(normalized))
        item["q_t"] = q_t
        if item["mask"].ndim == 2:
            item["mask"] = item["mask"][None]
        if item["valid_mask"].ndim == 2:
            item["valid_mask"] = item["valid_mask"][None]
        return item


def validate_terrain_checkpoint(
    checkpoint_path: Path,
    *,
    schema: Mapping[str, Any],
    protocol_summary_path: Path,
    fold_id: str,
    seed: int,
    parent_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "pild_support_only_terrain_checkpoint.v1":
        raise RuntimeError("unsupported Terrain checkpoint schema")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("Terrain checkpoint lacks identity mapping")
    expected = {
        "manifest_sha256": schema["manifest_sha256"],
        "protocol_summary_sha256": sha256_file(protocol_summary_path),
        "split_sha256": schema["split_sha256"],
        "fold_id": str(fold_id),
        "seed": int(seed),
        "parent_v_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
        "parent_visual_decoder_sha256": parent_receipt["visual_decoder_sha256"],
        "parent_prithvi_checkpoint_sha256": parent_receipt["prithvi_checkpoint_sha256"],
    }
    mismatch = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Terrain checkpoint identity mismatch: {mismatch}")
    terrain_names, terrain_groups, _ = resolve_terrain_contract(
        payload.get("terrain_channel_order", ())
    )
    expected_groups = {
        "fine": list(terrain_groups.fine),
        "meso": list(terrain_groups.meso),
        "macro": list(terrain_groups.macro),
    }
    if payload.get("terrain_scale_groups") != expected_groups:
        raise RuntimeError("Terrain checkpoint scale groups differ from audited schema")
    state = payload.get("terrain_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Terrain checkpoint lacks terrain_state_dict")
    observed_state_hash = tensor_sha256(state)
    if observed_state_hash != payload.get("terrain_state_sha256"):
        raise RuntimeError("Terrain tensor hash differs from terrain_state_sha256")
    normalization = payload.get("normalization")
    if not isinstance(normalization, Mapping):
        raise RuntimeError("Terrain checkpoint lacks normalization")
    if normalization.get("fit_scope") != "train-only-valid-terrain-pixels":
        raise RuntimeError("Terrain normalization was not fitted on train-only pixels")
    if tuple(normalization.get("feature_names", ())) != terrain_names:
        raise RuntimeError("Terrain normalization feature order differs from checkpoint")
    receipt = {
        "path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "terrain_state_sha256": observed_state_hash,
        "best_epoch": int(payload.get("best_epoch", -1)),
    }
    return payload, receipt


def fuse_logits(
    visual_logits: torch.Tensor,
    terrain_logits: torch.Tensor,
    q_t: torch.Tensor,
    *,
    alpha: float,
    uncertainty_power: float,
) -> torch.Tensor:
    """Bounded support-only residual with exact q=0 abstention."""

    if SPATIAL_POOL != 1:
        raise AssertionError("support-only additive v1 fixes spatial pool to one")
    probability = torch.sigmoid(visual_logits)
    uncertainty = (1.0 - 2.0 * torch.abs(probability - 0.5)).clamp(0.0, 1.0)
    gate = torch.ones_like(uncertainty) if uncertainty_power == 0 else uncertainty.pow(uncertainty_power)
    direction = torch.tanh(terrain_logits) * q_t.clamp(0.0, 1.0)
    return visual_logits + float(alpha) * gate * direction


def counts_from_predictions(
    baseline_probability: torch.Tensor,
    adapted_probability: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, int]:
    keep = valid.bool()
    truth = target >= 0.5
    baseline = baseline_probability >= threshold
    adapted = adapted_probability >= threshold
    result = {
        "baseline_tp": int((baseline & truth & keep).sum().item()),
        "baseline_fp": int((baseline & ~truth & keep).sum().item()),
        "baseline_fn": int((~baseline & truth & keep).sum().item()),
        "baseline_tn": int((~baseline & ~truth & keep).sum().item()),
        "adapted_tp": int((adapted & truth & keep).sum().item()),
        "adapted_fp": int((adapted & ~truth & keep).sum().item()),
        "adapted_fn": int((~adapted & truth & keep).sum().item()),
        "adapted_tn": int((~adapted & ~truth & keep).sum().item()),
        "corrected": int(((baseline != truth) & (adapted == truth) & keep).sum().item()),
        "harmed": int(((baseline == truth) & (adapted != truth) & keep).sum().item()),
        "valid_pixels": int(keep.sum().item()),
    }
    return result


def metrics_from_pair_counts(counts: Mapping[str, int]) -> dict[str, float]:
    baseline_counts = {
        key: int(counts[f"baseline_{key}"]) for key in ("tp", "fp", "fn", "tn")
    }
    adapted_counts = {
        key: int(counts[f"adapted_{key}"]) for key in ("tp", "fp", "fn", "tn")
    }
    baseline = metrics_from_counts(baseline_counts)
    adapted = metrics_from_counts(adapted_counts)
    corrected = int(counts["corrected"])
    harmed = int(counts["harmed"])
    baseline_errors = int(baseline_counts["fp"] + baseline_counts["fn"])
    return {
        "baseline_iou": baseline["iou"],
        "adapted_iou": adapted["iou"],
        "delta_iou": adapted["iou"] - baseline["iou"],
        "baseline_precision": baseline["precision"],
        "adapted_precision": adapted["precision"],
        "baseline_recall": baseline["recall"],
        "adapted_recall": adapted["recall"],
        "baseline_errors": float(baseline_errors),
        "adapted_errors": adapted["errors"],
        "corrected": float(corrected),
        "harmed": float(harmed),
        "net_error_reduction": float(corrected - harmed),
        "rer": (corrected - harmed) / max(baseline_errors, 1),
    }


@torch.no_grad()
def visual_and_terrain_logits(
    visual_model: nn.Module,
    terrain_model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    optical = batch["optical"].to(device) * 10_000.0
    visual = visual_model(
        optical,
        batch["temporal_coords"].to(device),
        batch["location_coords"].to(device),
    )
    visual_logits = visual["logits"].float()
    terrain_logits, _ = terrain_model(batch["terrain"].to(device))
    terrain_logits = terrain_logits.float()
    q_t = batch["q_t"].to(device).float()
    target = batch["mask"].to(device)
    valid = batch["valid_mask"].to(device)
    return visual_logits, terrain_logits, q_t, target, valid


@torch.no_grad()
def select_on_validation(
    visual_model: nn.Module,
    terrain_model: nn.Module,
    loader: DataLoader,
    *,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select solely from validation predictions; identity is always feasible."""

    configurations = [
        (alpha, power)
        for alpha in ALPHAS
        for power in UNCERTAINTY_POWERS
        if alpha != 0.0 or power == 0.0
    ]
    baseline_histogram = BinaryHistogram()
    histograms = {configuration: BinaryHistogram() for configuration in configurations}
    pair_counts = {
        configuration: defaultdict(int) for configuration in configurations
    }
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual_model, terrain_model, batch, device=device
        )
        baseline_probability = torch.sigmoid(visual_logits)
        baseline_histogram.update(baseline_probability, target, valid)
        for configuration in configurations:
            alpha, power = configuration
            adapted_probability = torch.sigmoid(
                fuse_logits(
                    visual_logits,
                    terrain_logits,
                    q_t,
                    alpha=alpha,
                    uncertainty_power=power,
                )
            )
            histograms[configuration].update(adapted_probability, target, valid)
            counts = counts_from_predictions(
                baseline_probability,
                adapted_probability,
                target,
                valid,
                threshold=threshold,
            )
            for key, value in counts.items():
                pair_counts[configuration][key] += int(value)
    baseline_ap = baseline_histogram.average_precision()
    rows: list[dict[str, Any]] = []
    for configuration in configurations:
        alpha, power = configuration
        row = {
            "alpha": alpha,
            "uncertainty_power": power,
            "spatial_pool": SPATIAL_POOL,
            **metrics_from_pair_counts(pair_counts[configuration]),
            "baseline_ap": baseline_ap,
            "adapted_ap": histograms[configuration].average_precision(),
        }
        row["delta_ap"] = row["adapted_ap"] - row["baseline_ap"]
        row["validation_feasible"] = bool(
            row["delta_iou"] >= -1e-12 and row["rer"] >= -1e-12
        )
        rows.append(row)
    feasible = [row for row in rows if row["validation_feasible"]]
    if not feasible:
        raise RuntimeError("validation selection has no feasible identity configuration")
    selected = max(
        feasible,
        key=lambda row: (
            row["delta_iou"],
            row["rer"],
            row["delta_ap"],
            -row["alpha"],
            -row["uncertainty_power"],
        ),
    )
    return dict(selected), rows


def aggregate_pair_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "baseline_tp",
        "baseline_fp",
        "baseline_fn",
        "baseline_tn",
        "adapted_tp",
        "adapted_fp",
        "adapted_fn",
        "adapted_tn",
        "corrected",
        "harmed",
        "valid_pixels",
    )
    return {key: sum(int(row[key]) for row in rows) for key in keys}


def aggregate_samples_to_events(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["canonical_event_id"])].append(row)
    output: list[dict[str, Any]] = []
    for event_id in sorted(grouped):
        rows = grouped[event_id]
        counts = aggregate_pair_counts(rows)
        output.append(
            {
                "canonical_event_id": event_id,
                "dataset_ids": ";".join(sorted({str(row["dataset_id"]) for row in rows})),
                "source_ids": ";".join(sorted({str(row["source_id"]) for row in rows})),
                "n_samples": len(rows),
                **counts,
                **metrics_from_pair_counts(counts),
            }
        )
    return output


@torch.no_grad()
def evaluate_test_once(
    visual_model: nn.Module,
    terrain_model: nn.Module,
    loader: DataLoader,
    *,
    selection: Mapping[str, Any],
    threshold: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_histogram = BinaryHistogram()
    adapted_histogram = BinaryHistogram()
    event_histograms: dict[str, tuple[BinaryHistogram, BinaryHistogram]] = {}
    sample_rows: list[dict[str, Any]] = []
    for batch in loader:
        visual_logits, terrain_logits, q_t, target, valid = visual_and_terrain_logits(
            visual_model, terrain_model, batch, device=device
        )
        baseline_probability = torch.sigmoid(visual_logits)
        adapted_probability = torch.sigmoid(
            fuse_logits(
                visual_logits,
                terrain_logits,
                q_t,
                alpha=float(selection["alpha"]),
                uncertainty_power=float(selection["uncertainty_power"]),
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
            baseline_sample_histogram = BinaryHistogram()
            adapted_sample_histogram = BinaryHistogram()
            baseline_sample_histogram.update(
                baseline_probability[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
            )
            adapted_sample_histogram.update(
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
            sample_rows.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "dataset_id": str(batch["dataset_id"][index]),
                    "source_id": str(batch["source_id"][index]),
                    "source_event_id": str(batch["source_event_id"][index]),
                    "canonical_event_id": event_id,
                    **counts,
                    **metrics_from_pair_counts(counts),
                    "baseline_ap": baseline_sample_histogram.average_precision(),
                    "adapted_ap": adapted_sample_histogram.average_precision(),
                }
            )
    event_rows = aggregate_samples_to_events(sample_rows)
    for row in event_rows:
        baseline_event, adapted_event = event_histograms[row["canonical_event_id"]]
        row["baseline_ap"] = baseline_event.average_precision()
        row["adapted_ap"] = adapted_event.average_precision()
        row["delta_ap"] = row["adapted_ap"] - row["baseline_ap"]
    corpus_counts = aggregate_pair_counts(sample_rows)
    corpus = {
        **corpus_counts,
        **metrics_from_pair_counts(corpus_counts),
        "baseline_ap": baseline_histogram.average_precision(),
        "adapted_ap": adapted_histogram.average_precision(),
        "n_samples": len(sample_rows),
        "n_events": len(event_rows),
    }
    corpus["delta_ap"] = corpus["adapted_ap"] - corpus["baseline_ap"]
    return sample_rows, event_rows, corpus


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError(f"inconsistent CSV row schema: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_models(
    *,
    parent: Mapping[str, Any],
    terrain_checkpoint: Mapping[str, Any],
    prithvi_snapshot: Path | None,
    decoder_width: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    encoder, provenance = load_prithvi_encoder(prithvi_snapshot)
    parent_prithvi_hash = parent["identity"]["prithvi_checkpoint_sha256"]
    if provenance["checkpoint_sha256"] != parent_prithvi_hash:
        raise RuntimeError(
            "loaded Prithvi snapshot differs from parent V checkpoint identity"
        )
    visual = PrithviEO2ChangeModel(
        encoder, decoder_width=decoder_width, freeze_encoder=True
    )
    visual.decoder.load_state_dict(
        parent["components"]["visual_decoder"], strict=True
    )
    if tensor_sha256(visual.decoder.state_dict()) != parent["component_sha256"]["visual_decoder"]:
        raise RuntimeError("loaded visual decoder hash differs from authenticated parent")
    terrain_names, terrain_groups, _ = resolve_terrain_contract(
        terrain_checkpoint["terrain_channel_order"]
    )
    terrain = SupportOnlyMultiScaleTerrainPyramid(
        len(terrain_names), terrain_groups
    )
    terrain.load_state_dict(terrain_checkpoint["terrain_state_dict"], strict=True)
    if tensor_sha256(terrain.state_dict()) != terrain_checkpoint["terrain_state_sha256"]:
        raise RuntimeError("loaded Terrain model hash differs from authenticated checkpoint")
    visual.eval().to(device)
    terrain.eval().to(device)
    for model in (visual, terrain):
        for parameter in model.parameters():
            parameter.requires_grad = False
    return visual, terrain, provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-select and test frozen PILD V + support-only Terrain."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parent-v-checkpoint", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--prithvi-snapshot", type=Path)
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    command = shlex.join([sys.executable, *(sys.argv if argv is None else [__file__, *argv])])
    (stage / "run.log").write_text(command + "\n", encoding="utf-8")
    started = time.time()
    device = torch.device(args.device)
    visual_model, terrain_model, prithvi_provenance = build_models(
        parent=parent,
        terrain_checkpoint=terrain_checkpoint,
        prithvi_snapshot=args.prithvi_snapshot,
        decoder_width=args.decoder_width,
        device=device,
    )
    normalization = terrain_checkpoint["normalization"]

    # Validation is the only data opened before the frozen receipt is written.
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
    selection, validation_grid = select_on_validation(
        visual_model,
        terrain_model,
        val_loader,
        threshold=float(parent["threshold"]),
        device=device,
    )
    selection_receipt = {
        "schema_version": "pild_support_only_additive_selection.v1",
        "selection_scope": "validation-only",
        "selection_rule": "max delta_iou, rer, delta_ap among delta_iou>=0 and rer>=0",
        "frozen_before_test_open": True,
        "terrain_contract": "common9",
        "spatial_pool": SPATIAL_POOL,
        "threshold": float(parent["threshold"]),
        "threshold_source": "parent_V_visual_validation",
        "selected": selection,
        "grid": validation_grid,
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "protocol_summary_sha256": sha256_file(args.protocol_summary),
            "split_sha256": schema["split_sha256"],
            "fold_id": str(args.fold_id),
            "seed": int(args.seed),
            "parent_v_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
            "terrain_checkpoint_sha256": terrain_receipt["checkpoint_sha256"],
        },
    }
    write_json(stage / "selection.json", selection_receipt)
    selection_sha256 = sha256_file(stage / "selection.json")

    # The test role is intentionally created only after the selection receipt.
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    sample_rows, event_rows, corpus = evaluate_test_once(
        visual_model,
        terrain_model,
        test_loader,
        selection=selection,
        threshold=float(parent["threshold"]),
        device=device,
    )
    write_csv(stage / "per_sample_metrics.csv", sample_rows)
    write_csv(stage / "per_event_metrics.csv", event_rows)
    result = {
        "schema_version": "pild_support_only_additive_result.v1",
        "status": "COMPLETE",
        "fold_id": str(args.fold_id),
        "seed": int(args.seed),
        "contract": {
            "visual": "fixed PILD V checkpoint",
            "terrain": "fixed support-only common9 pyramid",
            "fusion": "visual_logits + alpha * visual_uncertainty^power * q_T * tanh(terrain_logits)",
            "spatial_pool": SPATIAL_POOL,
            "selection": "validation-only, then test once",
        },
        "selection": selection,
        "selection_sha256": selection_sha256,
        "test": corpus,
        "n_per_sample_rows": len(sample_rows),
        "n_per_event_rows": len(event_rows),
        "artifacts": {
            "per_sample_metrics_sha256": sha256_file(stage / "per_sample_metrics.csv"),
            "per_event_metrics_sha256": sha256_file(stage / "per_event_metrics.csv"),
        },
        "schema_validation": schema,
        "parent_v_receipt": parent_receipt,
        "terrain_receipt": terrain_receipt,
        "prithvi_provenance": prithvi_provenance,
        "elapsed_seconds": time.time() - started,
    }
    write_json(stage / "result.json", result)
    write_json(
        stage / "DONE.json",
        {
            "status": "COMPLETE",
            "selection_sha256": selection_sha256,
            "result_sha256": sha256_file(stage / "result.json"),
            "per_sample_metrics_sha256": result["artifacts"]["per_sample_metrics_sha256"],
            "per_event_metrics_sha256": result["artifacts"]["per_event_metrics_sha256"],
        },
    )
    os.replace(stage, outdir)
    print(
        f"completed frozen test evaluation: {outdir} "
        f"(alpha={selection['alpha']}, power={selection['uncertainty_power']}, "
        f"delta_iou={corpus['delta_iou']:.6f}, rer={corpus['rer']:.6f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
