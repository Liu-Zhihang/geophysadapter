#!/usr/bin/env python3
"""Shared stronger-backbone utilities for strict_t2 visual training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet101_Weights, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50

from train_strict_t2_postrgb_baseline import compute_segmentation_loss, masked_stats


@dataclass
class EpochStat:
    epoch: int
    train_loss: float
    train_bce: float
    train_dice: float
    train_iou: float
    train_f1: float
    val_loss: float
    val_bce: float
    val_dice: float
    val_iou: float
    val_f1: float
    lr: float
    sec: float


def _expand_conv_weight(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    old_in = weight.shape[1]
    if in_channels == old_in:
        return weight.clone()
    repeat = int(math.ceil(in_channels / old_in))
    expanded = weight.repeat(1, repeat, 1, 1)[:, :in_channels].clone()
    expanded.mul_(old_in / float(in_channels))
    return expanded


def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(_expand_conv_weight(conv.weight.detach(), in_channels))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias.detach())
    return new_conv


class BinaryDeepLabV3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        backbone_name: str = "deeplabv3_resnet50",
        pretrained_backbone: bool = True,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.backbone_name = backbone_name
        self.pretrained_backbone = bool(pretrained_backbone)
        self.aux_loss = bool(aux_loss)

        if backbone_name == "deeplabv3_resnet50":
            weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
            model = deeplabv3_resnet50(weights=None, weights_backbone=weights_backbone, aux_loss=aux_loss, num_classes=21)
        elif backbone_name == "deeplabv3_resnet101":
            weights_backbone = ResNet101_Weights.IMAGENET1K_V2 if pretrained_backbone else None
            model = deeplabv3_resnet101(weights=None, weights_backbone=weights_backbone, aux_loss=aux_loss, num_classes=21)
        else:
            raise ValueError(f"unsupported backbone_name={backbone_name}")

        model.backbone.conv1 = adapt_first_conv(model.backbone.conv1, in_channels)
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[-1] = nn.Conv2d(model.aux_classifier[-1].in_channels, 1, kernel_size=1)
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.model(x)
        if isinstance(out, dict):
            return out["out"], out.get("aux")
        return out, None


def run_epoch_strong(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int = 0,
    bce_pos_weight: float = 1.0,
    dataset_loss_weights: dict[str, float] | None = None,
    threshold: float = 0.5,
    aux_loss_weight: float = 0.2,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_items = 0
    agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    by_dataset: dict[str, dict[str, float]] = {}

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader, start=1):
            x = batch["image"].to(device, non_blocking=False)
            y = batch["mask"].to(device, non_blocking=False)
            v = batch["valid"].to(device, non_blocking=False)
            logits, aux_logits = model(x)

            sample_weights = None
            if dataset_loss_weights:
                vals = [float(dataset_loss_weights.get(dataset_id, 1.0)) for dataset_id in batch["dataset_id"]]
                sample_weights = torch.tensor(vals, device=logits.device, dtype=logits.dtype)

            bce, dice, loss = compute_segmentation_loss(
                logits,
                y,
                v,
                pos_weight=bce_pos_weight,
                sample_weights=sample_weights,
            )
            if aux_logits is not None and aux_loss_weight > 0.0:
                _, _, aux_loss = compute_segmentation_loss(
                    aux_logits,
                    y,
                    v,
                    pos_weight=bce_pos_weight,
                    sample_weights=sample_weights,
                )
                loss = loss + float(aux_loss_weight) * aux_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            logits_detached = logits.detach()
            stats = masked_stats(logits_detached, y, v, threshold=threshold)
            bsz = x.size(0)
            total_loss += float(loss.item()) * bsz
            total_bce += float(bce.item()) * bsz
            total_dice += float(dice.item()) * bsz
            total_items += bsz
            agg["tp"] += stats["tp"]
            agg["fp"] += stats["fp"]
            agg["fn"] += stats["fn"]

            for i, ds_name in enumerate(batch["dataset_id"]):
                item_stats = masked_stats(logits_detached[i : i + 1], y[i : i + 1], v[i : i + 1], threshold=threshold)
                if ds_name not in by_dataset:
                    by_dataset[ds_name] = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n": 0.0}
                by_dataset[ds_name]["tp"] += item_stats["tp"]
                by_dataset[ds_name]["fp"] += item_stats["fp"]
                by_dataset[ds_name]["fn"] += item_stats["fn"]
                by_dataset[ds_name]["n"] += 1

            if max_steps > 0 and step >= max_steps:
                break

    def _finish(a: dict[str, float]) -> dict[str, float]:
        tp = a["tp"]
        fp = a["fp"]
        fn = a["fn"]
        return {
            "iou": tp / (tp + fp + fn + 1e-7),
            "f1": (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7),
            "precision": tp / (tp + fp + 1e-7),
            "recall": tp / (tp + fn + 1e-7),
        }

    overall = {
        "loss": total_loss / max(total_items, 1),
        "bce": total_bce / max(total_items, 1),
        "dice": total_dice / max(total_items, 1),
        **_finish(agg),
    }
    dataset_metrics = {name: _finish(stats) | {"samples": stats["n"]} for name, stats in by_dataset.items()}
    return overall, dataset_metrics


def find_best_threshold_strong(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: list[float],
    max_steps: int = 0,
) -> tuple[float, list[dict[str, float]]]:
    model.eval()
    aggs = {thr: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for thr in thresholds}
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            x = batch["image"].to(device, non_blocking=False)
            y = batch["mask"].to(device, non_blocking=False)
            v = batch["valid"].to(device, non_blocking=False)
            logits, _ = model(x)
            probs = torch.sigmoid(logits)
            tgt = (y >= 0.5).float() * v
            for thr in thresholds:
                pred = (probs >= thr).float() * v
                aggs[thr]["tp"] += float((pred * tgt).sum().item())
                aggs[thr]["fp"] += float((pred * (1.0 - tgt)).sum().item())
                aggs[thr]["fn"] += float((((1.0 - pred) * tgt) * v).sum().item())
            if max_steps > 0 and step >= max_steps:
                break

    rows: list[dict[str, float]] = []
    for thr in thresholds:
        tp = aggs[thr]["tp"]
        fp = aggs[thr]["fp"]
        fn = aggs[thr]["fn"]
        rows.append(
            {
                "threshold": float(thr),
                "iou": tp / (tp + fp + fn + 1e-7),
                "f1": (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7),
                "precision": tp / (tp + fp + 1e-7),
                "recall": tp / (tp + fn + 1e-7),
            }
        )
    best = max(rows, key=lambda item: (item["iou"], item["f1"], -abs(item["threshold"] - 0.5)))
    return float(best["threshold"]), rows
