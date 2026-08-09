#!/usr/bin/env python3
"""Train a role-pure Terrain expert and calibrate its frozen-visual fusion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

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
    parser.add_argument(
        "--sample-support-csv",
        type=Path,
        default=None,
        help="Optional label-independent eligibility table used to specialize train/val.",
    )
    parser.add_argument(
        "--init-terrain-checkpoint",
        type=Path,
        default=None,
        help="Optional same-schema Terrain expert used only to initialize DLR fine-tuning.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument(
        "--direction-pool-factors",
        default="1",
        help="Comma-separated average-pooling factors for validation-only residual scale selection.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Must match the nested visual-anchor training fraction.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def score_terrain(model, loader, device):
    histogram = trainer.protocol.ProbabilityHistogram()
    model.eval()
    for batch in loader:
        terrain = batch["terrain"].to(device, non_blocking=True)
        with trainer.protocol.autocast_context(True):
            logits, _ = model(terrain)
        probability = torch.sigmoid(logits).float().cpu().numpy()
        target = batch["mask"].numpy()
        valid = batch["valid"].numpy() > 0.5
        histogram.update(probability[valid], target[valid])
    return histogram


def main() -> int:
    args = parse_args()
    direction_pool_factors = tuple(
        sorted({int(value) for value in args.direction_pool_factors.split(",")})
    )
    if not direction_pool_factors or any(value < 1 for value in direction_pool_factors):
        raise ValueError("direction-pool-factors must contain positive integers")
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
    support_signature = None
    if args.sample_support_csv is not None:
        with args.sample_support_csv.open(newline="", encoding="utf-8") as handle:
            support_rows = list(csv.DictReader(handle))
        support = {}
        for row in support_rows:
            sample_id = row["sample_id"]
            if sample_id in support:
                raise RuntimeError(f"duplicate support row for {sample_id}")
            support[sample_id] = int(row["support_eligible"])
        required = set(roles["train"]) | set(roles["val"])
        missing = sorted(required - set(support))
        if missing:
            raise RuntimeError(f"sample support CSV misses {len(missing)} train/val rows")
        roles["train"] = [sample for sample in roles["train"] if support[sample] == 1]
        roles["val"] = [sample for sample in roles["val"] if support[sample] == 1]
        if not roles["train"] or not roles["val"]:
            raise RuntimeError("sample support gate removed all train or validation rows")
        support_signature = trainer.file_signature(args.sample_support_csv)
    roles["train"] = trainer.stratified_fraction_subset(
        roles["train"], rows, args.train_fraction, args.seed
    )
    mean, std = trainer.estimate_terrain_stats(args.terrain_h5, all_ids, roles["train"])
    datasets = {
        role: trainer.PrithviTerrainDataset(
            args.base_h5,
            args.optical_h5,
            args.terrain_h5,
            all_ids,
            event_ids,
            rows,
            ids,
            mean,
            std,
            args.seed,
            roles["train"],
            True,
        )
        for role, ids in (("train", roles["train"]), ("val", roles["val"]))
    }
    loader_args = SimpleNamespace(
        seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers
    )
    loaders = {
        role: trainer.protocol.make_loader(data, loader_args, shuffle=role == "train")
        for role, data in datasets.items()
    }
    model = trainer.SupportOnlyMultiScaleTerrainPyramid(
        len(terrain_names), terrain_scale_groups
    ).to(args.device)
    initialization = None
    if args.init_terrain_checkpoint is not None:
        payload = torch.load(
            args.init_terrain_checkpoint, map_location="cpu", weights_only=False
        )
        state = payload["trainable_state_dict"]
        parameters = dict(model.named_parameters())
        loaded = []
        skipped = []
        with torch.no_grad():
            for name, parameter in parameters.items():
                source = state.get(name)
                if source is not None and tuple(source.shape) == tuple(parameter.shape):
                    parameter.copy_(source)
                    loaded.append(name)
                else:
                    skipped.append(
                        {
                            "name": name,
                            "target_shape": list(parameter.shape),
                            "source_shape": (
                                list(source.shape) if source is not None else None
                            ),
                        }
                    )
        if not loaded:
            raise RuntimeError("Terrain initialization has no shape-compatible parameters")
        initialization = {
            **trainer.file_signature(args.init_terrain_checkpoint),
            "state_sha256": trainer.tensor_dict_sha256(state),
            "mode": "shape_compatible_transfer",
            "loaded_parameters": loaded,
            "skipped_parameters": skipped,
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if args.pos_weight > 0:
        pos_weight = float(args.pos_weight)
    else:
        positive = 0.0
        total = 0.0
        for index in range(len(datasets["train"])):
            target = datasets["train"][index]["mask"]
            valid = datasets["train"][index]["valid"]
            positive += float((target * valid).sum())
            total += float(valid.sum())
        pos_weight = min(40.0, max(1.0, (total - positive) / max(positive, 1.0)))
    positive_tensor = torch.tensor([pos_weight], device=args.device).view(1, 1, 1, 1)
    best_ap = -1.0
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in loaders["train"]:
            terrain = batch["terrain"].to(args.device, non_blocking=True)
            target = batch["mask"].to(args.device, non_blocking=True)
            valid = batch["valid"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with trainer.protocol.autocast_context(True):
                logits, _ = model(terrain)
                bce = F.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=positive_tensor, reduction="none"
                )
                bce = trainer.masked_mean(bce, valid)
                loss = bce + 0.5 * trainer.protocol.dice_loss_per_sample(
                    logits, target, valid
                ).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * terrain.shape[0]
            seen += terrain.shape[0]
        histogram = score_terrain(model, loaders["val"], args.device)
        threshold, metrics = trainer.protocol.choose_threshold(histogram)
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(seen, 1),
            "val_ap": histogram.average_precision,
            "val_iou": metrics["iou"],
            "val_threshold": threshold,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if histogram.average_precision > best_ap:
            best_ap = histogram.average_precision
            best_epoch = epoch
            best_state = trainer.trainable_state(model)
    if best_state is None:
        raise RuntimeError("Terrain expert did not train")
    trainer.load_trainable_state(model, best_state)

    encoder, provenance = trainer.load_prithvi_encoder()
    visual = trainer.PrithviVisualCompat(
        trainer.PrithviEO2ChangeModel(encoder, decoder_width=128, freeze_encoder=True)
    )
    checkpoint = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    trainer.load_trainable_state(visual, checkpoint["trainable_state_dict"])
    visual = visual.to(args.device).eval()
    threshold = float(checkpoint["threshold"])
    alphas = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    uncertainty_powers = (0.0, 1.0, 2.0)
    histograms = {
        (alpha, power, pool_factor): trainer.protocol.ProbabilityHistogram()
        for alpha in alphas
        for power in uncertainty_powers
        for pool_factor in direction_pool_factors
    }
    with torch.inference_mode():
        for batch in loaders["val"]:
            optical = batch["pre"].to(args.device, non_blocking=True)
            coordinates = batch["post"].to(args.device, non_blocking=True)
            terrain = batch["terrain"].to(args.device, non_blocking=True)
            q_t = batch["q_t"].to(args.device, non_blocking=True)
            with trainer.protocol.autocast_context(True):
                visual_logits, _ = visual(optical, coordinates)
                terrain_logits, _ = model(terrain)
            visual_logits = visual_logits.float()
            uncertainty = 1.0 - 2.0 * torch.abs(torch.sigmoid(visual_logits) - 0.5)
            direction = torch.tanh(terrain_logits.float()) * q_t
            directions = {1: direction}
            for pool_factor in direction_pool_factors:
                if pool_factor == 1:
                    continue
                pooled = F.avg_pool2d(
                    direction, kernel_size=pool_factor, stride=pool_factor
                )
                directions[pool_factor] = F.interpolate(
                    pooled,
                    size=direction.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            target = batch["mask"].numpy()
            valid = batch["valid"].numpy() > 0.5
            for alpha in alphas:
                for power in uncertainty_powers:
                    gate = torch.ones_like(uncertainty) if power == 0 else uncertainty.pow(power)
                    for pool_factor in direction_pool_factors:
                        probability = torch.sigmoid(
                            visual_logits + alpha * gate * directions[pool_factor]
                        )
                        histograms[(alpha, power, pool_factor)].update(
                            probability.cpu().numpy()[valid], target[valid]
                        )
    baseline_counts = histograms[
        (0.0, 0.0, direction_pool_factors[0])
    ].counts_at(threshold)
    baseline_metrics = trainer.protocol.metrics_from_counts(baseline_counts)
    baseline_errors = baseline_counts["fp"] + baseline_counts["fn"]
    grid = []
    for (alpha, power, pool_factor), histogram in histograms.items():
        counts = histogram.counts_at(threshold)
        metrics = trainer.protocol.metrics_from_counts(counts)
        errors = counts["fp"] + counts["fn"]
        grid.append(
            {
                "alpha": alpha,
                "uncertainty_power": power,
                "direction_pool_factor": pool_factor,
                "ap": histogram.average_precision,
                "iou": metrics["iou"],
                "delta_iou": metrics["iou"] - baseline_metrics["iou"],
                "errors": errors,
                "rer": (baseline_errors - errors) / max(baseline_errors, 1),
                **counts,
            }
        )
    grid.sort(key=lambda row: (row["iou"], row["rer"], row["ap"]), reverse=True)
    result = {
        "status": "validation_development_only",
        "fold": args.fold,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "n_train_samples": len(roles["train"]),
        "n_validation_samples": len(roles["val"]),
        "sample_support_csv": support_signature,
        "terrain_mean": mean.tolist(),
        "terrain_std": std.tolist(),
        "terrain_schema": terrain_schema,
        "terrain_names": terrain_names,
        "initialization": initialization,
        "regions": sorted(set(split_regions["val"])),
        "pos_weight": pos_weight,
        "terrain_best_epoch": best_epoch,
        "terrain_best_ap": best_ap,
        "direction_pool_factors": list(direction_pool_factors),
        "history": history,
        "visual_threshold": threshold,
        "baseline": {**baseline_metrics, **baseline_counts, "errors": baseline_errors},
        "fusion_grid": grid,
        "best_fusion": grid[0],
        "prithvi_provenance": provenance,
    }
    torch.save(
        {"trainable_state_dict": best_state, "result": result},
        args.outdir / "terrain_expert.pt",
    )
    (args.outdir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"baseline": result["baseline"], "best": grid[0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
