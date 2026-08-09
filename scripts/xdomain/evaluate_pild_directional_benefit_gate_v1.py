#!/usr/bin/env python3
"""Directional validation-fitted utility gates for bounded Terrain corrections.

The visual/Terrain disagreement proposals are split into two decisions:

* rescue: visual negative -> Terrain candidate positive;
* veto: visual positive -> Terrain candidate negative.

Independent linear gates avoid forcing these decisions to share one boundary.
The veto gate uses a larger harmful-proposal weight because removing a true
positive is more damaging to IoU than rejecting one false positive is helpful.
Rejected proposals exactly recover the frozen visual prediction.

This script is exploratory. It must not be described as an independently
confirmed result when its test folds have already informed method development.
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
    LinearBenefitGate,
    collect_gate_training_rows,
    evaluate,
    fit_gate,
    write_csv,
)
from evaluate_pild_source_calibrated_terrain_v1 import select_by_source
from evaluate_pild_support_only_additive_v1 import (
    EvaluationDataset,
    build_models,
    validate_terrain_checkpoint,
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


RESCUE_INDEX = FEATURE_NAMES.index("proposal_rescue")
VETO_INDEX = FEATURE_NAMES.index("proposal_veto")


class DirectionalBenefitGate:
    """Dispatch rescue and veto proposals to separately fitted gates."""

    def __init__(
        self,
        rescue: LinearBenefitGate,
        veto: LinearBenefitGate,
    ) -> None:
        self.rescue = rescue
        self.veto = veto

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        output = np.zeros(len(features), dtype=np.float32)
        rescue_mask = features[:, RESCUE_INDEX] > 0.5
        veto_mask = features[:, VETO_INDEX] > 0.5
        if np.any(rescue_mask):
            output[rescue_mask] = self.rescue.predict_probability(
                features[rescue_mask]
            )
        if np.any(veto_mask):
            output[veto_mask] = self.veto.predict_probability(features[veto_mask])
        if np.any(~(rescue_mask | veto_mask)):
            raise RuntimeError("directional gate received non-proposal rows")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch": {
                "rescue_feature": FEATURE_NAMES[RESCUE_INDEX],
                "veto_feature": FEATURE_NAMES[VETO_INDEX],
            },
            "rescue": self.rescue.to_dict(),
            "veto": self.veto.to_dict(),
        }


def fit_directional_gate(
    x: np.ndarray,
    y: np.ndarray,
    events: np.ndarray,
    *,
    c_value: float,
    rescue_harm_weight: float,
    veto_harm_weight: float,
    seed: int,
) -> tuple[DirectionalBenefitGate, dict[str, Any]]:
    rescue_mask = x[:, RESCUE_INDEX] > 0.5
    veto_mask = x[:, VETO_INDEX] > 0.5
    if np.any(rescue_mask & veto_mask) or np.any(~(rescue_mask | veto_mask)):
        raise RuntimeError("proposal directions are not a strict partition")
    if not np.any(rescue_mask) or not np.any(veto_mask):
        raise RuntimeError(
            "both rescue and veto proposals are required to fit directional gates"
        )
    rescue, rescue_fit = fit_gate(
        x[rescue_mask],
        y[rescue_mask],
        events[rescue_mask],
        c_value=c_value,
        harm_weight=rescue_harm_weight,
        seed=seed,
    )
    veto, veto_fit = fit_gate(
        x[veto_mask],
        y[veto_mask],
        events[veto_mask],
        c_value=c_value,
        harm_weight=veto_harm_weight,
        seed=seed + 1,
    )
    return DirectionalBenefitGate(rescue, veto), {
        "objective": "direction-specific expected correction benefit",
        "rescue": {
            **rescue_fit,
            "harm_weight": rescue_harm_weight,
            "n_rows": int(rescue_mask.sum()),
            "beneficial_fraction": float(y[rescue_mask].mean()),
        },
        "veto": {
            **veto_fit,
            "harm_weight": veto_harm_weight,
            "n_rows": int(veto_mask.sum()),
            "beneficial_fraction": float(y[veto_mask].mean()),
        },
    }


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
    parser.add_argument("--rescue-harm-weight", type=float, default=2.0)
    parser.add_argument("--veto-harm-weight", type=float, default=8.0)
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
    gate, fit_metadata = fit_directional_gate(
        x,
        y,
        events,
        c_value=args.c_value,
        rescue_harm_weight=args.rescue_harm_weight,
        veto_harm_weight=args.veto_harm_weight,
        seed=args.seed,
    )
    receipt = {
        "schema_version": "pild_directional_benefit_gate_receipt.v1",
        "frozen_before_test_open": True,
        "scope": "exploratory validation-fitted directional proposal utility",
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
        "schema_version": "pild_directional_benefit_gate_result.v1",
        "status": "EXPLORATORY_COMPLETE",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "contract": {
            "visual_anchor": "frozen Prithvi-EO-2.0-300M-TL",
            "proposal": "validation-selected bounded Terrain residual",
            "gate": "separate rescue and veto expected-benefit classifiers",
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
            "status": "EXPLORATORY_COMPLETE",
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
        f"{args.fold_id}: directional gate delta_iou="
        f"{aligned['delta_iou']:+.6f}, delta_ap={aligned['delta_ap']:+.6f}, "
        f"rer={aligned['rer']:+.2%}, accepted={aligned['acceptance_rate']:.1%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
