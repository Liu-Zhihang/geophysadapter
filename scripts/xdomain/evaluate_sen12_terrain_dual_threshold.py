#!/usr/bin/env python3
"""Validation-only dual-threshold routing for a frozen Terrain expert."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train_sen12_prithvi_terrain_v2 as trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, default=trainer.BASE_H5)
    parser.add_argument("--optical-h5", type=Path, default=trainer.OPTICAL_H5)
    parser.add_argument("--terrain-h5", type=Path, default=trainer.TERRAIN_H5)
    parser.add_argument("--split-csv", type=Path, default=trainer.SPLIT_CSV)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--routing-mode",
        choices=("both", "veto", "rescue"),
        default="both",
        help="Whether Terrain may veto positives, rescue negatives, or do both.",
    )
    parser.add_argument(
        "--expanded-grid",
        action="store_true",
        help="Use a wider validation-only dosage grid for one-sided routing.",
    )
    parser.add_argument(
        "--fixed-config",
        default="",
        help="Optional low,high,alpha,margin tuple for one-shot confirmatory evaluation.",
    )
    parser.add_argument(
        "--emit-per-region",
        action="store_true",
        help="Emit per-region metrics for validation-only consistency checks.",
    )
    parser.add_argument(
        "--emit-per-sample",
        action="store_true",
        help="Emit per-sample diagnostics for post-hoc mechanism discovery.",
    )
    parser.add_argument(
        "--sample-support-csv",
        type=Path,
        default=None,
        help="Optional label-independent sample support gate; ineligible rows set q_T=0.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def counts(prediction, target, valid):
    return {
        "tp": int((prediction & target & valid).sum()),
        "fp": int((prediction & ~target & valid).sum()),
        "fn": int((~prediction & target & valid).sum()),
        "tn": int((~prediction & ~target & valid).sum()),
    }


def scalar_entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    return -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    trainer.set_seed(args.seed)
    all_ids, event_ids = trainer.validate_sidecars(
        args.base_h5, args.optical_h5, args.terrain_h5
    )
    terrain_names, terrain_scale_groups, terrain_schema = trainer.resolve_terrain_contract(
        args.terrain_h5
    )
    rows, roles, split_regions = trainer.protocol.load_logo_rows(args.split_csv, args.fold)
    allowed = set(all_ids)
    roles = {role: [sample for sample in values if sample in allowed] for role, values in roles.items()}
    terrain_payload = torch.load(args.terrain_checkpoint, map_location="cpu", weights_only=False)
    terrain_result = terrain_payload.get("result", {})
    if "terrain_mean" in terrain_result and "terrain_std" in terrain_result:
        mean = np.asarray(terrain_result["terrain_mean"], dtype=np.float32)
        std = np.asarray(terrain_result["terrain_std"], dtype=np.float32)
    else:
        # Backward-compatible path for already frozen full-data checkpoints.
        mean, std = trainer.estimate_terrain_stats(
            args.terrain_h5, all_ids, roles["train"]
        )
    dataset = trainer.PrithviTerrainDataset(
        args.base_h5,
        args.optical_h5,
        args.terrain_h5,
        all_ids,
        event_ids,
        rows,
        roles[args.split],
        mean,
        std,
        args.seed,
        roles["train"],
        True,
    )
    sample_support = None
    if args.sample_support_csv is not None:
        with args.sample_support_csv.open(newline="", encoding="utf-8") as handle:
            support_rows = list(csv.DictReader(handle))
        sample_support = {}
        for row in support_rows:
            sample_id = row["sample_id"]
            if sample_id in sample_support:
                raise RuntimeError(f"duplicate support row for {sample_id}")
            sample_support[sample_id] = int(row["support_eligible"])
        missing = sorted(set(dataset.sample_ids) - set(sample_support))
        if missing:
            raise RuntimeError(
                f"sample support CSV misses {len(missing)} {args.split} samples"
            )
    loader = trainer.protocol.make_loader(
        dataset,
        SimpleNamespace(seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers),
        shuffle=False,
    )

    terrain = trainer.SupportOnlyMultiScaleTerrainPyramid(
        len(terrain_names), terrain_scale_groups
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
    visual_threshold = float(visual_payload["threshold"])

    low_thresholds = (0.20, 0.30, 0.40)
    high_thresholds = (0.60, 0.70, 0.80)
    alphas = (2.0, 4.0, 8.0)
    visual_margins = (0.10, 0.25, 1.0)
    if args.expanded_grid:
        low_thresholds = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
        high_thresholds = (0.50, 0.60, 0.70, 0.80, 0.90)
        alphas = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
        visual_margins = (0.10, 0.25, 0.50, 1.0)
    if args.routing_mode == "veto":
        high_thresholds = (1.0,)
    elif args.routing_mode == "rescue":
        low_thresholds = (0.0,)
    if args.fixed_config:
        values = tuple(float(value) for value in args.fixed_config.split(","))
        if len(values) != 4:
            raise ValueError("--fixed-config requires low,high,alpha,margin")
        configurations = [values]
    else:
        configurations = [
            (low, high, alpha, margin)
            for low in low_thresholds
            for high in high_thresholds
            for alpha in alphas
            for margin in visual_margins
        ]
    if args.emit_per_sample and len(configurations) != 1:
        raise ValueError("--emit-per-sample requires --fixed-config")
    totals = {configuration: defaultdict(int) for configuration in configurations}
    opportunity_baselines = {
        configuration: defaultdict(int) for configuration in configurations
    }
    opportunity_totals = {
        configuration: defaultdict(int) for configuration in configurations
    }
    transitions = {configuration: defaultdict(int) for configuration in configurations}
    active_baselines = {configuration: defaultdict(int) for configuration in configurations}
    active_totals = {configuration: defaultdict(int) for configuration in configurations}
    baseline = defaultdict(int)
    baseline_regions = defaultdict(lambda: defaultdict(int))
    total_regions = {
        configuration: defaultdict(lambda: defaultdict(int))
        for configuration in configurations
    }
    sample_rows = []
    with torch.inference_mode():
        for batch in loader:
            optical = batch["pre"].to(args.device, non_blocking=True)
            coordinates = batch["post"].to(args.device, non_blocking=True)
            terrain_input = batch["terrain"].to(args.device, non_blocking=True)
            q_t = batch["q_t"].to(args.device, non_blocking=True)
            if sample_support is not None:
                support = torch.as_tensor(
                    [sample_support[value] for value in batch["sample_id"]],
                    dtype=q_t.dtype,
                    device=args.device,
                ).view(-1, 1, 1, 1)
                q_t = q_t * support
            target = batch["mask"].to(args.device, non_blocking=True) >= 0.5
            valid = batch["valid"].to(args.device, non_blocking=True) >= 0.5
            with trainer.protocol.autocast_context(True):
                visual_logits, _ = visual(optical, coordinates)
                terrain_logits, _ = terrain(terrain_input)
            visual_logits = visual_logits.float()
            visual_probability = torch.sigmoid(visual_logits)
            terrain_probability = torch.sigmoid(terrain_logits.float())
            visual_positive = visual_probability >= visual_threshold
            base_counts = counts(visual_positive, target, valid)
            for key, value in base_counts.items():
                baseline[key] += value
            region_masks = {}
            if args.emit_per_region:
                for region in sorted(set(batch["region"])):
                    selector = torch.as_tensor(
                        [value == region for value in batch["region"]],
                        dtype=torch.bool,
                        device=args.device,
                    ).view(-1, 1, 1, 1)
                    region_masks[region] = valid & selector
                    for key, value in counts(
                        visual_positive, target, region_masks[region]
                    ).items():
                        baseline_regions[region][key] += value
            for configuration in configurations:
                low, high, alpha, margin = configuration
                near_boundary = (visual_probability - visual_threshold).abs() <= margin
                veto = torch.zeros_like(visual_probability)
                rescue = torch.zeros_like(visual_probability)
                if args.routing_mode in ("both", "veto"):
                    veto = (
                        ((low - terrain_probability) / max(low, 1e-6)).clamp(0.0, 1.0)
                        * visual_positive
                        * near_boundary
                        * q_t
                    )
                if args.routing_mode in ("both", "rescue"):
                    rescue = (
                        ((terrain_probability - high) / max(1.0 - high, 1e-6)).clamp(0.0, 1.0)
                        * (~visual_positive)
                        * near_boundary
                        * q_t
                    )
                prediction = visual_logits + alpha * (rescue - veto) >= torch.logit(
                    torch.tensor(visual_threshold, device=args.device)
                )
                current = counts(prediction, target, valid)
                for key, value in current.items():
                    totals[configuration][key] += value
                if args.emit_per_region:
                    for region, region_valid in region_masks.items():
                        for key, value in counts(
                            prediction, target, region_valid
                        ).items():
                            total_regions[configuration][region][key] += value
                opportunity = valid & near_boundary & (
                    (visual_positive & (terrain_probability <= low))
                    | ((~visual_positive) & (terrain_probability >= high))
                ) & (q_t >= 0.5)
                for key, value in counts(visual_positive, target, opportunity).items():
                    opportunity_baselines[configuration][key] += value
                for key, value in counts(prediction, target, opportunity).items():
                    opportunity_totals[configuration][key] += value
                active = valid & (prediction != visual_positive)
                for key, value in counts(visual_positive, target, active).items():
                    active_baselines[configuration][key] += value
                for key, value in counts(prediction, target, active).items():
                    active_totals[configuration][key] += value
                visual_correct = visual_positive == target
                adapted_correct = prediction == target
                transitions[configuration]["corrected"] += int(
                    ((~visual_correct) & adapted_correct & valid).sum()
                )
                transitions[configuration]["harmed"] += int(
                    (visual_correct & (~adapted_correct) & valid).sum()
                )
                if args.emit_per_sample:
                    entropy = scalar_entropy(visual_probability)
                    disagreement = visual_positive != (
                        terrain_probability >= 0.5
                    )
                    for position, sample_id in enumerate(batch["sample_id"]):
                        sample_valid = valid[position : position + 1]
                        visual_sample = visual_positive[position : position + 1]
                        prediction_sample = prediction[position : position + 1]
                        target_sample = target[position : position + 1]
                        visual_counts = counts(
                            visual_sample, target_sample, sample_valid
                        )
                        adapted_counts = counts(
                            prediction_sample, target_sample, sample_valid
                        )
                        visual_metrics = trainer.protocol.metrics_from_counts(
                            visual_counts
                        )
                        adapted_metrics = trainer.protocol.metrics_from_counts(
                            adapted_counts
                        )
                        visual_errors = visual_counts["fp"] + visual_counts["fn"]
                        adapted_errors = adapted_counts["fp"] + adapted_counts["fn"]
                        corrected = int(
                            (
                                (visual_sample != target_sample)
                                & (prediction_sample == target_sample)
                                & sample_valid
                            ).sum()
                        )
                        harmed = int(
                            (
                                (visual_sample == target_sample)
                                & (prediction_sample != target_sample)
                                & sample_valid
                            ).sum()
                        )
                        valid_pixels = max(int(sample_valid.sum()), 1)
                        active_sample = sample_valid & (
                            prediction_sample != visual_sample
                        )
                        sample_rows.append(
                            {
                                "sample_id": sample_id,
                                "event_id": batch["event_id"][position],
                                "region": batch["region"][position],
                                "fold": args.fold,
                                "split": args.split,
                                "low_threshold": low,
                                "high_threshold": high,
                                "alpha": alpha,
                                "visual_margin": margin,
                                "valid_pixels": valid_pixels,
                                "visual_tp": visual_counts["tp"],
                                "visual_fp": visual_counts["fp"],
                                "visual_fn": visual_counts["fn"],
                                "visual_tn": visual_counts["tn"],
                                "visual_errors": visual_errors,
                                "visual_iou": visual_metrics["iou"],
                                "adapted_tp": adapted_counts["tp"],
                                "adapted_fp": adapted_counts["fp"],
                                "adapted_fn": adapted_counts["fn"],
                                "adapted_tn": adapted_counts["tn"],
                                "adapted_errors": adapted_errors,
                                "adapted_iou": adapted_metrics["iou"],
                                "delta_iou": (
                                    adapted_metrics["iou"] - visual_metrics["iou"]
                                ),
                                "rer": (
                                    (visual_errors - adapted_errors)
                                    / max(visual_errors, 1)
                                ),
                                "corrected": corrected,
                                "harmed": harmed,
                                "net_corrected": corrected - harmed,
                                "active_pixels": int(active_sample.sum()),
                                "active_fraction": (
                                    float(active_sample.sum()) / valid_pixels
                                ),
                                "visual_entropy_mean": float(
                                    entropy[position][sample_valid[0]].mean()
                                ),
                                "visual_margin_mean": float(
                                    (
                                        visual_probability[position]
                                        - visual_threshold
                                    )
                                    .abs()[sample_valid[0]]
                                    .mean()
                                ),
                                "terrain_probability_mean": float(
                                    terrain_probability[position][
                                        sample_valid[0]
                                    ].mean()
                                ),
                                "terrain_probability_std": float(
                                    terrain_probability[position][
                                        sample_valid[0]
                                    ].std()
                                ),
                                "terrain_quality_mean": float(
                                    q_t[position][sample_valid[0]].mean()
                                ),
                                "sample_support_eligible": (
                                    1
                                    if sample_support is None
                                    else sample_support[sample_id]
                                ),
                                "visual_terrain_disagreement_fraction": float(
                                    disagreement[position][sample_valid[0]]
                                    .float()
                                    .mean()
                                ),
                            }
                        )

    baseline_metrics = trainer.protocol.metrics_from_counts(baseline)
    baseline_errors = baseline["fp"] + baseline["fn"]
    grid = []
    for configuration, current in totals.items():
        low, high, alpha, margin = configuration
        metrics = trainer.protocol.metrics_from_counts(current)
        errors = current["fp"] + current["fn"]
        opportunity_base = opportunity_baselines[configuration]
        opportunity_current = opportunity_totals[configuration]
        opportunity_base_metrics = trainer.protocol.metrics_from_counts(opportunity_base)
        opportunity_metrics = trainer.protocol.metrics_from_counts(opportunity_current)
        opportunity_base_errors = opportunity_base["fp"] + opportunity_base["fn"]
        opportunity_errors = opportunity_current["fp"] + opportunity_current["fn"]
        active_base = active_baselines[configuration]
        active_current = active_totals[configuration]
        active_base_metrics = trainer.protocol.metrics_from_counts(active_base)
        active_metrics = trainer.protocol.metrics_from_counts(active_current)
        active_base_errors = active_base["fp"] + active_base["fn"]
        active_errors = active_current["fp"] + active_current["fn"]
        corrected = transitions[configuration]["corrected"]
        harmed = transitions[configuration]["harmed"]
        row = {
                "low_threshold": low,
                "high_threshold": high,
                "alpha": alpha,
                "visual_margin": margin,
                **current,
                **metrics,
                "errors": errors,
                "delta_iou": metrics["iou"] - baseline_metrics["iou"],
                "rer": (baseline_errors - errors) / max(baseline_errors, 1),
                "corrected": corrected,
                "harmed": harmed,
                "corrected_to_harmed": corrected / max(harmed, 1),
                "opportunity_pixels": sum(opportunity_base.values()),
                "opportunity_visual_iou": opportunity_base_metrics["iou"],
                "opportunity_adapted_iou": opportunity_metrics["iou"],
                "opportunity_delta_iou": (
                    opportunity_metrics["iou"] - opportunity_base_metrics["iou"]
                ),
                "opportunity_rer": (
                    (opportunity_base_errors - opportunity_errors)
                    / max(opportunity_base_errors, 1)
                ),
                "active_pixels": sum(active_base.values()),
                "active_visual_iou": active_base_metrics["iou"],
                "active_adapted_iou": active_metrics["iou"],
                "active_delta_iou": active_metrics["iou"] - active_base_metrics["iou"],
                "active_rer": (
                    (active_base_errors - active_errors) / max(active_base_errors, 1)
                ),
            }
        if args.emit_per_region:
            per_region = {}
            for region in sorted(baseline_regions):
                region_base = baseline_regions[region]
                region_current = total_regions[configuration][region]
                region_base_metrics = trainer.protocol.metrics_from_counts(region_base)
                region_metrics = trainer.protocol.metrics_from_counts(region_current)
                region_base_errors = region_base["fp"] + region_base["fn"]
                region_errors = region_current["fp"] + region_current["fn"]
                per_region[region] = {
                    "baseline": {**region_base, **region_base_metrics},
                    "adapted": {**region_current, **region_metrics},
                    "delta_iou": (
                        region_metrics["iou"] - region_base_metrics["iou"]
                    ),
                    "rer": (
                        (region_base_errors - region_errors)
                        / max(region_base_errors, 1)
                    ),
                }
            row["per_region"] = per_region
        grid.append(row)
    grid.sort(key=lambda row: (row["delta_iou"], row["rer"]), reverse=True)
    feasible = [row for row in grid if row["rer"] >= 0.10]
    result = {
        "status": (
            "confirmatory_fixed_configuration"
            if args.split == "test" and args.fixed_config
            else "validation_development_only"
        ),
        "fold": args.fold,
        "split": args.split,
        "routing_mode": args.routing_mode,
        "terrain_schema": terrain_schema,
        "terrain_names": terrain_names,
        "regions": sorted(set(split_regions[args.split])),
        "visual_threshold": visual_threshold,
        "baseline": {**baseline, **baseline_metrics, "errors": baseline_errors},
        "best_iou": grid[0],
        "best_with_rer_ge_10pct": feasible[0] if feasible else None,
        "grid": grid,
        "prithvi_provenance": provenance,
        "sample_support_csv": (
            None
            if args.sample_support_csv is None
            else str(args.sample_support_csv.resolve())
        ),
    }
    (args.outdir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.emit_per_sample:
        trainer.protocol.write_csv(sample_rows, args.outdir / "per_sample.csv")
    print(json.dumps({k: result[k] for k in ("baseline", "best_iou", "best_with_rer_ge_10pct")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
