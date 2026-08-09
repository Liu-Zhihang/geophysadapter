#!/usr/bin/env python3
"""Audit label-free burn-like/Terrain veto rules against frozen Prithvi errors.

This is a development diagnostic, not a fire-label product.  dNBR and dNDVI
describe burn-like spectral change but cannot establish that a pixel burned.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_sen12_prithvi_terrain_v2 as trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalized_difference(value: torch.Tensor, left: int, right: int) -> torch.Tensor:
    return (value[:, left] - value[:, right]) / (
        value[:, left] + value[:, right] + 1e-4
    )


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    trainer.set_seed(args.seed)
    all_ids, event_ids = trainer.validate_sidecars(
        trainer.BASE_H5, trainer.OPTICAL_H5, trainer.TERRAIN_H5
    )
    rows, roles, split_regions = trainer.protocol.load_logo_rows(trainer.SPLIT_CSV, args.fold)
    allowed = set(all_ids)
    roles = {role: [sample for sample in values if sample in allowed] for role, values in roles.items()}
    train_ids = roles["train"]
    mean, std = trainer.estimate_terrain_stats(trainer.TERRAIN_H5, all_ids, train_ids)
    dataset = trainer.PrithviTerrainDataset(
        trainer.BASE_H5,
        trainer.OPTICAL_H5,
        trainer.TERRAIN_H5,
        all_ids,
        event_ids,
        rows,
        roles[args.split],
        mean,
        std,
        args.seed,
        train_ids,
        True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    encoder, provenance = trainer.load_prithvi_encoder()
    core = trainer.PrithviEO2ChangeModel(encoder, decoder_width=128, freeze_encoder=True)
    model = trainer.PrithviVisualCompat(core)
    checkpoint = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["identity"]["fold"] != args.fold or checkpoint["identity"]["seed"] != args.seed:
        raise RuntimeError("checkpoint fold/seed does not match requested audit")
    if checkpoint["identity"]["prithvi_checkpoint_sha256"] != provenance["checkpoint_sha256"]:
        raise RuntimeError("Prithvi checkpoint provenance mismatch")
    trainer.load_trainable_state(model, checkpoint["trainable_state_dict"])
    model = model.to(args.device).eval()
    threshold = float(checkpoint["threshold"])

    slope_thresholds = (2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)
    dnbr_thresholds = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    probability_ceilings = (0.60, 0.70, 0.80, 0.90, 1.01)
    removed: dict[tuple[str, float, float, float], list[int]] = defaultdict(lambda: [0, 0])
    base = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    with torch.inference_mode():
        for batch in loader:
            optical = batch["pre"].to(args.device, non_blocking=True)
            coordinates = batch["post"].to(args.device, non_blocking=True)
            target = batch["mask"].to(args.device, non_blocking=True) >= 0.5
            valid = batch["valid"].to(args.device, non_blocking=True) >= 0.5
            terrain = batch["terrain"].to(args.device, non_blocking=True)
            with trainer.protocol.autocast_context(True):
                logits, _ = model(optical, coordinates)
            probability = torch.sigmoid(logits.float())
            prediction = probability >= threshold
            base["tp"] += int((prediction & target & valid).sum())
            base["fp"] += int((prediction & ~target & valid).sum())
            base["fn"] += int((~prediction & target & valid).sum())
            base["tn"] += int((~prediction & ~target & valid).sum())

            reflectance = optical.float() / 10000.0
            before = reflectance[:, :, :2].mean(dim=2)
            after = reflectance[:, :, 2:].mean(dim=2)
            dnbr = normalized_difference(before, 3, 5) - normalized_difference(after, 3, 5)
            slope = terrain[:, 1] * float(std[1]) + float(mean[1])
            positive = prediction[:, 0] & valid[:, 0]
            fp = positive & ~target[:, 0]
            tp = positive & target[:, 0]
            confidence = probability[:, 0]

            for slope_max in slope_thresholds:
                flat = slope < slope_max
                for ceiling in probability_ceilings:
                    uncertain = confidence < ceiling
                    rule = positive & flat & uncertain
                    key = ("flat", slope_max, -999.0, ceiling)
                    removed[key][0] += int((rule & fp).sum())
                    removed[key][1] += int((rule & tp).sum())
                    for dnbr_min in dnbr_thresholds:
                        burn_like = dnbr > dnbr_min
                        rule = positive & burn_like & flat & uncertain
                        key = ("burn_like_and_flat", slope_max, dnbr_min, ceiling)
                        removed[key][0] += int((rule & fp).sum())
                        removed[key][1] += int((rule & tp).sum())

    baseline_iou = base["tp"] / max(base["tp"] + base["fp"] + base["fn"], 1)
    baseline_errors = base["fp"] + base["fn"]
    output_rows = []
    for (rule, slope_max, dnbr_min, ceiling), (fixed_fp, harmed_tp) in removed.items():
        tp = base["tp"] - harmed_tp
        fp = base["fp"] - fixed_fp
        fn = base["fn"] + harmed_tp
        iou = tp / max(tp + fp + fn, 1)
        net_corrected = fixed_fp - harmed_tp
        output_rows.append(
            {
                "rule": rule,
                "slope_max_deg": slope_max,
                "dnbr_min": None if dnbr_min < -100 else dnbr_min,
                "visual_probability_ceiling": ceiling,
                "corrected_fp": fixed_fp,
                "harmed_tp": harmed_tp,
                "net_corrected": net_corrected,
                "rer": net_corrected / max(baseline_errors, 1),
                "iou": iou,
                "delta_iou": iou - baseline_iou,
            }
        )
    output_rows.sort(key=lambda row: (row["delta_iou"], row["rer"]), reverse=True)
    with (args.outdir / "rule_grid.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "status": "development_diagnostic_only",
        "warning": "dNBR is a burn-like spectral proxy, not verified fire occurrence",
        "fold": args.fold,
        "split": args.split,
        "regions": sorted(set(split_regions[args.split])),
        "threshold": threshold,
        "baseline": {**base, "iou": baseline_iou, "errors": baseline_errors},
        "best_rules": output_rows[:20],
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
