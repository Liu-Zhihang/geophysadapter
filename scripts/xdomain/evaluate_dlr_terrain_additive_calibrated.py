#!/usr/bin/env python3
"""Evaluate a validation-selected additive Terrain dosage once on a DLR test fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import train_sen12_prithvi_terrain_v2 as trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--optical-h5", type=Path, required=True)
    parser.add_argument("--terrain-h5", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def counts(prediction, target, valid):
    return {
        "tp": int((prediction & target & valid).sum()),
        "fp": int((prediction & ~target & valid).sum()),
        "fn": int((~prediction & target & valid).sum()),
        "tn": int((~prediction & ~target & valid).sum()),
    }


def select_validation_configuration(result: dict) -> dict:
    candidates = [
        row for row in result["fusion_grid"]
        if float(row["delta_iou"]) >= 0.0 and float(row["rer"]) >= 0.0
    ]
    if not candidates:
        raise RuntimeError("validation grid lacks a non-negative identity/abstention option")
    return max(
        candidates,
        key=lambda row: (float(row["delta_iou"]), float(row["rer"]), float(row["ap"])),
    )


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    trainer.set_seed(args.seed)
    all_ids, event_ids = trainer.validate_sidecars(
        args.base_h5, args.optical_h5, args.terrain_h5
    )
    terrain_names, terrain_groups, terrain_schema = trainer.resolve_terrain_contract(
        args.terrain_h5
    )
    rows, roles, split_regions = trainer.protocol.load_logo_rows(args.split_csv, args.fold)
    allowed = set(all_ids)
    roles = {role: [sample for sample in values if sample in allowed] for role, values in roles.items()}
    terrain_payload = torch.load(args.terrain_checkpoint, map_location="cpu", weights_only=False)
    terrain_result = terrain_payload["result"]
    if int(terrain_result["fold"]) != args.fold or int(terrain_result["seed"]) != args.seed:
        raise RuntimeError("Terrain checkpoint fold/seed mismatch")
    selected = select_validation_configuration(terrain_result)
    alpha = float(selected["alpha"])
    uncertainty_power = float(selected["uncertainty_power"])
    direction_pool_factor = int(selected.get("direction_pool_factor", 1))
    mean = np.asarray(terrain_result["terrain_mean"], dtype=np.float32)
    std = np.asarray(terrain_result["terrain_std"], dtype=np.float32)
    dataset = trainer.PrithviTerrainDataset(
        args.base_h5, args.optical_h5, args.terrain_h5, all_ids, event_ids,
        rows, roles["test"], mean, std, args.seed, roles["train"], True,
    )
    loader = trainer.protocol.make_loader(
        dataset,
        SimpleNamespace(seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers),
        shuffle=False,
    )
    terrain = trainer.SupportOnlyMultiScaleTerrainPyramid(
        len(terrain_names), terrain_groups
    )
    trainer.load_trainable_state(terrain, terrain_payload["trainable_state_dict"])
    terrain = terrain.to(args.device).eval()
    encoder, provenance = trainer.load_prithvi_encoder()
    visual = trainer.PrithviVisualCompat(
        trainer.PrithviEO2ChangeModel(encoder, decoder_width=128, freeze_encoder=True)
    )
    visual_payload = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    trainer.load_trainable_state(visual, visual_payload["trainable_state_dict"])
    visual = visual.to(args.device).eval()
    threshold = float(visual_payload["threshold"])

    baseline = defaultdict(int)
    adapted = defaultdict(int)
    corrected = 0
    harmed = 0
    with torch.inference_mode():
        for batch in loader:
            optical = batch["pre"].to(args.device, non_blocking=True)
            coordinates = batch["post"].to(args.device, non_blocking=True)
            terrain_input = batch["terrain"].to(args.device, non_blocking=True)
            q_t = batch["q_t"].to(args.device, non_blocking=True)
            target = batch["mask"].to(args.device, non_blocking=True) >= 0.5
            valid = batch["valid"].to(args.device, non_blocking=True) >= 0.5
            with trainer.protocol.autocast_context(True):
                visual_logits, _ = visual(optical, coordinates)
                terrain_logits, _ = terrain(terrain_input)
            visual_logits = visual_logits.float()
            uncertainty = 1.0 - 2.0 * torch.abs(torch.sigmoid(visual_logits) - 0.5)
            gate = torch.ones_like(uncertainty) if uncertainty_power == 0 else uncertainty.pow(uncertainty_power)
            direction = torch.tanh(terrain_logits.float()) * q_t
            if direction_pool_factor > 1:
                direction = F.interpolate(
                    F.avg_pool2d(
                        direction,
                        kernel_size=direction_pool_factor,
                        stride=direction_pool_factor,
                    ),
                    size=direction.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            visual_prediction = torch.sigmoid(visual_logits) >= threshold
            adapted_prediction = torch.sigmoid(visual_logits + alpha * gate * direction) >= threshold
            for key, value in counts(visual_prediction, target, valid).items():
                baseline[key] += value
            for key, value in counts(adapted_prediction, target, valid).items():
                adapted[key] += value
            visual_correct = visual_prediction == target
            adapted_correct = adapted_prediction == target
            corrected += int(((~visual_correct) & adapted_correct & valid).sum())
            harmed += int((visual_correct & (~adapted_correct) & valid).sum())

    baseline_metrics = trainer.protocol.metrics_from_counts(baseline)
    adapted_metrics = trainer.protocol.metrics_from_counts(adapted)
    baseline_errors = baseline["fp"] + baseline["fn"]
    adapted_errors = adapted["fp"] + adapted["fn"]
    result = {
        "status": "test_frozen_from_validation",
        "method": "additive_terrain_dosage",
        "fold": args.fold,
        "seed": args.seed,
        "regions": sorted(set(split_regions["test"])),
        "terrain_schema": terrain_schema,
        "validation_selection": {
            "alpha": alpha,
            "uncertainty_power": uncertainty_power,
            "direction_pool_factor": direction_pool_factor,
            "validation_delta_iou": selected["delta_iou"],
            "validation_rer": selected["rer"],
            "source_result": str((args.terrain_checkpoint.parent / "result.json").resolve()),
            "source_result_sha256": sha256_file(args.terrain_checkpoint.parent / "result.json"),
        },
        "baseline": {**baseline, **baseline_metrics, "errors": baseline_errors},
        "adapted": {
            **adapted,
            **adapted_metrics,
            "errors": adapted_errors,
            "delta_iou": adapted_metrics["iou"] - baseline_metrics["iou"],
            "rer": (baseline_errors - adapted_errors) / max(baseline_errors, 1),
            "corrected": corrected,
            "harmed": harmed,
            "corrected_to_harmed": corrected / max(harmed, 1),
        },
        "prithvi_provenance": provenance,
    }
    (args.outdir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
