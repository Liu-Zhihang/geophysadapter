#!/usr/bin/env python3
"""Evaluate matched Twin DINOv2-S plus independent multiscale Terrain."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from audit_pild_support_only_controls_v1 import CONDITIONS, ControlledTerrainDataset
from evaluate_pild_decision_margin_terrain_v1 import evaluate, select_by_source
from evaluate_pild_support_only_additive_v1 import (
    EvaluationDataset,
    validate_terrain_checkpoint,
    write_csv,
)
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file
from sen12_terrain_v2 import SupportOnlyMultiScaleTerrainPyramid
from train_pild_geo4_twin_dinov2_v1 import TwinDINOv2FPN, set_seed
from train_pild_sen12_roleaware_v1 import (
    COMMON_TERRAIN9_NAMES,
    COMMON_TERRAIN9_SCALE_GROUPS,
    tensor_sha256,
    validate_protocol_schema,
)
from train_pild_support_only_terrain_v1 import write_json


class DINOCompat(nn.Module):
    def __init__(self, model: TwinDINOv2FPN) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        optical: torch.Tensor,
        temporal_coords: torch.Tensor,
        location_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del temporal_coords, location_coords
        with torch.autocast(
            device_type=optical.device.type,
            dtype=torch.bfloat16,
            enabled=optical.device.type == "cuda",
        ):
            logits = self.model(optical / 10_000.0)
        return {"logits": logits}


def validate_baseline_replay(
    rows: list[dict[str, Any]],
    reference_path: Path,
    *,
    max_iou_drift: float,
) -> dict[str, Any]:
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    # The role-aware trainer exports val and test rows together. Evaluation
    # replays the test loader, so select the matching role before identity
    # checks. Older test-only artifacts remain unchanged.
    if any(row.get("split") == "test" for row in reference_rows):
        reference_rows = [
            row for row in reference_rows if row.get("split") == "test"
        ]
    reference = {row["sample_id"]: row for row in reference_rows}
    observed = {str(row["sample_id"]): row for row in rows}
    if observed.keys() != reference.keys():
        missing = sorted(reference.keys() - observed.keys())[:5]
        extra = sorted(observed.keys() - reference.keys())[:5]
        raise RuntimeError(
            f"visual baseline replay sample mismatch: missing={missing}, extra={extra}"
        )
    differences: list[dict[str, Any]] = []
    expected_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    observed_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    for sample_id, row in observed.items():
        expected = reference[sample_id]
        first_difference: dict[str, Any] | None = None
        for key in ("tp", "fp", "fn", "tn"):
            actual_value = int(row[f"baseline_{key}"])
            expected_value = int(expected[key])
            expected_counts[key] += expected_value
            observed_counts[key] += actual_value
            if actual_value != expected_value and first_difference is None:
                first_difference = {
                        "sample_id": sample_id,
                        "key": key,
                        "expected": expected_value,
                        "observed": actual_value,
                }
        if first_difference is not None:
            differences.append(first_difference)
    expected_iou = expected_counts["tp"] / max(
        expected_counts["tp"] + expected_counts["fp"] + expected_counts["fn"], 1
    )
    observed_iou = observed_counts["tp"] / max(
        observed_counts["tp"] + observed_counts["fp"] + observed_counts["fn"], 1
    )
    iou_drift = observed_iou - expected_iou
    if abs(iou_drift) > max_iou_drift:
        raise RuntimeError(
            f"visual baseline replay IoU drift {iou_drift:+.8f} exceeds "
            f"tolerance {max_iou_drift:.8f}; changed_samples={len(differences)}, "
            f"examples={differences[:5]}"
        )
    return {
        "reference_path": str(reference_path.resolve()),
        "n_samples": len(observed),
        "n_samples_with_count_drift": len(differences),
        "reference_iou": expected_iou,
        "replayed_iou": observed_iou,
        "iou_drift": iou_drift,
        "max_abs_iou_drift": max_iou_drift,
        "status": "PASS",
    }


def load_visual(
    path: Path,
    *,
    schema: Mapping[str, Any],
    fold_id: str,
    seed: int,
    device: torch.device,
) -> tuple[DINOCompat, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "pild_geo4_twin_dinov2_checkpoint.v2":
        raise RuntimeError("unsupported Twin DINO checkpoint")
    identity = payload.get("identity", {})
    expected = {
        "manifest_sha256": schema["manifest_sha256"],
        "split_sha256": schema["split_sha256"],
        "fold_id": fold_id,
        "seed": seed,
    }
    mismatch = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Twin DINO identity mismatch: {mismatch}")
    model = TwinDINOv2FPN(pretrained=False)
    model.encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    model.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    if tensor_sha256(model.encoder.state_dict()) != payload["encoder_sha256"]:
        raise RuntimeError("Twin DINO encoder hash mismatch")
    if tensor_sha256(model.decoder.state_dict()) != payload["decoder_sha256"]:
        raise RuntimeError("Twin DINO decoder hash mismatch")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    receipt = {
        "path": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "encoder_sha256": payload["encoder_sha256"],
        "decoder_sha256": payload["decoder_sha256"],
        "threshold": float(payload["threshold"]),
        "threshold_source": payload["threshold_source"],
    }
    return DINOCompat(model), payload, receipt


def load_terrain(
    path: Path,
    *,
    schema: Mapping[str, Any],
    protocol_summary: Path,
    fold_id: str,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "pild_support_only_terrain_checkpoint.v1":
        raise RuntimeError("unsupported Terrain checkpoint")
    identity = payload.get("identity", {})
    expected = {
        "manifest_sha256": schema["manifest_sha256"],
        "protocol_summary_sha256": sha256_file(protocol_summary),
        "split_sha256": schema["split_sha256"],
        "fold_id": fold_id,
        "seed": seed,
    }
    mismatch = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Terrain identity mismatch: {mismatch}")
    if tuple(payload["terrain_channel_order"]) != tuple(COMMON_TERRAIN9_NAMES):
        raise RuntimeError("Terrain common9 order mismatch")
    model = SupportOnlyMultiScaleTerrainPyramid(
        len(COMMON_TERRAIN9_NAMES), COMMON_TERRAIN9_SCALE_GROUPS
    )
    model.load_state_dict(payload["terrain_state_dict"], strict=True)
    if tensor_sha256(model.state_dict()) != payload["terrain_state_sha256"]:
        raise RuntimeError("Terrain state hash mismatch")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    receipt = {
        "path": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "terrain_state_sha256": payload["terrain_state_sha256"],
        "selection_metric": payload["selection_metric"],
        "parent_binding_ignored_for_reason": (
            "support-only Terrain contains no visual input; matched manifest/split/fold/seed "
            "and tensor hash are enforced"
        ),
    }
    return model, payload, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-summary", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-per-sample", type=Path, required=True)
    parser.add_argument("--max-replay-iou-drift", type=float, default=1e-3)
    parser.add_argument("--max-alpha", type=float, default=4.0)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_alpha < 0:
        raise ValueError("--max-alpha must be non-negative")

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
        threshold=float(visual_payload["threshold"]),
        device=device,
        max_alpha=args.max_alpha,
    )
    selection = {
        "schema_version": "pild_geo4_twin_dino_terrain_selection.v1",
        "selection_scope": "validation-only by known source",
        "frozen_before_test_open": True,
        "selections": selections,
        "max_alpha": args.max_alpha,
        "grids": grids,
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "split_sha256": schema["split_sha256"],
            "fold_id": args.fold_id,
            "seed": args.seed,
            "visual_checkpoint_sha256": visual_receipt["checkpoint_sha256"],
            "terrain_checkpoint_sha256": terrain_receipt["checkpoint_sha256"],
        },
    }
    write_json(stage / "selection.json", selection)

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
    conditions: list[dict[str, Any]] = []
    aligned_samples: list[dict[str, Any]] = []
    aligned_events: list[dict[str, Any]] = []
    replay_report: dict[str, Any] | None = None
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
            threshold=float(visual_payload["threshold"]),
            device=device,
        )
        if condition == "aligned":
            replay_report = validate_baseline_replay(
                sample_rows,
                args.visual_per_sample,
                max_iou_drift=args.max_replay_iou_drift,
            )
        conditions.append({"condition": condition, **corpus})
        if condition == "aligned":
            aligned_samples, aligned_events = sample_rows, event_rows
    write_csv(stage / "per_sample_metrics.csv", aligned_samples)
    write_csv(stage / "per_event_metrics.csv", aligned_events)
    aligned = conditions[0]
    result = {
        "schema_version": "pild_geo4_twin_dino_terrain_result.v1",
        "status": "COMPLETE",
        "artifact_state": "exploratory_single_seed_gate",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "test": aligned,
        "conditions": conditions,
        "selections": selections,
        "contrasts": {
            row["condition"]: {
                "aligned_minus_control_delta_iou": aligned["delta_iou"] - row["delta_iou"],
                "aligned_minus_control_rer": aligned["rer"] - row["rer"],
            }
            for row in conditions[1:]
        },
        "visual_receipt": visual_receipt,
        "terrain_receipt": terrain_receipt,
        "visual_replay": replay_report,
        "elapsed_seconds": time.time() - started,
    }
    write_json(stage / "result.json", result)
    write_json(
        stage / "DONE.json",
        {
            "status": "COMPLETE",
            "result_sha256": sha256_file(stage / "result.json"),
            "selection_sha256": sha256_file(stage / "selection.json"),
            "per_sample_sha256": sha256_file(stage / "per_sample_metrics.csv"),
            "per_event_sha256": sha256_file(stage / "per_event_metrics.csv"),
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
