#!/usr/bin/env python3
"""Train non-Prithvi visual anchors on the same PILD protocol and export OOF caches.

Uses the same manifest, splits, fold, seed, sampling temperatures, optimization
budget, and threshold rule as the Prithvi anchor. Only the visual encoder changes.
Optical tensors are (6 bands, 4 dates, 128, 128); the first two dates are averaged
as pre-event and the last two as post-event to form a 12-channel input. Channel
normalization statistics are estimated on the training fold only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for candidate in (SCRIPT_DIR, PROJECT_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pild_core_v21_phase14_visual_common import (  # noqa: E402
    DECODER_HIDDEN_CHANNELS,
    BackboneSpec,
    get_backbone_spec,
)
from pild_sen12_training_loader_v2 import (  # noqa: E402
    UnifiedPILDSen12Dataset,
    sha256_file,
)
from run_support_adapter_timmfm import TimmFeatureEncoder, TimmFPNHead  # noqa: E402
from train_pild_sen12_roleaware_v1 import (  # noqa: E402
    BinaryHistogram,
    TemperedDatasetEventPatchSampler,
    choose_threshold,
    validate_protocol_schema,
)
from train_pild_support_only_terrain_v1 import write_json  # noqa: E402

DEFAULT_META = PROJECT_ROOT / "metadata/pild_geo4_qc_native17_v1"
DEFAULT_MANIFEST = DEFAULT_META / "unified_sample_manifest_geo4_qc_native17_v1.csv"
DEFAULT_SUMMARY = DEFAULT_META / "protocol_summary_geo4_qc_native17_v1.json"
DEFAULT_SPLIT = PROJECT_ROOT / "metadata/pild_geo4_qc_v1/source_stratified_event_folds_v1.csv"
FOLDS = tuple(f"source_stratified_{index}" for index in range(4))
IN_CHANNELS = 12

# Additional remote-sensing self-supervised anchors.
EXTRA_BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "dinov3_sat_l_fpn": BackboneSpec(
        slug="dinov3_sat_l_fpn",
        model_name="vit_large_patch16_dinov3.sat493m",
        img_size=224,
        out_indices=(5, 11, 17, 23),
        batch_size=8,
        provenance="remote-sensing self-supervised foundation model pretrained on 493M satellite images",
    ),
}


def resolve_backbone_spec(slug: str) -> BackboneSpec:
    if slug in EXTRA_BACKBONE_SPECS:
        return EXTRA_BACKBONE_SPECS[slug]
    return get_backbone_spec(slug)


class AltVisualAnchor(nn.Module):
    """Frozen timm encoder with an FPN decoder for 12-channel pre/post stacks."""

    def __init__(self, slug: str, *, pretrained: bool = True) -> None:
        super().__init__()
        spec = resolve_backbone_spec(slug)
        self.spec = spec
        self.encoder = TimmFeatureEncoder(
            spec.model_name,
            in_chans=IN_CHANNELS,
            pretrained=pretrained,
            img_size=spec.img_size,
            out_indices=spec.out_indices,
            freeze_backbone=True,
        )
        self.decoder = TimmFPNHead(self.encoder.channels, hidden=DECODER_HIDDEN_CHANNELS)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # 冻结编码器不得保留训练态随机行为
        return self

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        features = self.encoder(visual)
        logits, _ = self.decoder(features, visual.shape[-2:])
        return logits


def build_input(batch: dict[str, Any]) -> torch.Tensor:
    """(B, 6, 4, H, W) -> (B, 12, H, W)：前两时相均值为震前，后两时相均值为震后。"""
    optical = batch["optical"]
    if optical.ndim != 5 or optical.shape[2] != 4:
        raise RuntimeError(f"unexpected optical shape {tuple(optical.shape)}")
    pre = optical[:, :, :2].mean(dim=2)
    post = optical[:, :, 2:].mean(dim=2)
    return torch.cat([pre, post], dim=1)


def masked_targets(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["mask"]
    if mask.ndim == 3:
        mask = mask[:, None]
    keep = batch["valid_mask"]
    if keep.ndim == 3:
        keep = keep[:, None]
    return (mask[:, :1] >= 0.5).float(), (keep[:, :1] > 0).float()


def estimate_normalization(
    dataset: UnifiedPILDSen12Dataset, *, max_samples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """在该折训练集上估计逐通道均值与标准差。不接触标签，因此不构成泄漏。"""
    rng = np.random.default_rng(seed)
    count = min(max_samples, len(dataset))
    picks = rng.choice(len(dataset), size=count, replace=False)
    total = torch.zeros(IN_CHANNELS, dtype=torch.float64)
    total_square = torch.zeros(IN_CHANNELS, dtype=torch.float64)
    pixels = 0
    for index in picks:
        item = dataset[int(index)]
        stacked = build_input({"optical": item["optical"][None]})[0].double()
        total += stacked.sum(dim=(1, 2))
        total_square += (stacked**2).sum(dim=(1, 2))
        pixels += stacked.shape[-1] * stacked.shape[-2]
    mean = total / max(pixels, 1)
    variance = torch.clamp(total_square / max(pixels, 1) - mean**2, min=1e-8)
    return mean.float(), variance.sqrt().float()


def estimate_pos_weight(dataset: UnifiedPILDSen12Dataset, *, max_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed + 1)
    count = min(max_samples, len(dataset))
    picks = rng.choice(len(dataset), size=count, replace=False)
    positive = negative = 0.0
    for index in picks:
        item = dataset[int(index)]
        target, keep = masked_targets(
            {"mask": item["mask"][None], "valid_mask": item["valid_mask"][None]}
        )
        positive += float((target * keep).sum())
        negative += float(((1.0 - target) * keep).sum())
    return float(np.clip(negative / max(positive, 1.0), 1.0, 200.0))


@torch.no_grad()
def validation_histogram(model, loader, mean, std, device) -> BinaryHistogram:
    histogram = BinaryHistogram()
    model.eval()
    for batch in loader:
        visual = ((build_input(batch) - mean) / std).to(device)
        target, keep = masked_targets(batch)
        probability = torch.sigmoid(model(visual)).cpu()
        histogram.update(probability, target, keep)
    return histogram


def train_fold(args, fold_id: str) -> dict[str, Any]:
    device = torch.device(args.device)
    schema = validate_protocol_schema(
        args.manifest, args.protocol_summary, args.split, fold_id
    )
    datasets = {
        role: UnifiedPILDSen12Dataset(
            args.manifest,
            args.protocol_summary,
            split_path=args.split,
            fold_id=fold_id,
            role=role,
            readiness="core",
        )
        for role in ("train", "val", "test")
    }
    mean, std = estimate_normalization(
        datasets["train"], max_samples=args.normalization_samples, seed=args.seed
    )
    pos_weight = estimate_pos_weight(
        datasets["train"], max_samples=args.normalization_samples, seed=args.seed
    )
    print(
        f"[{fold_id}] train={len(datasets['train'])} val={len(datasets['val'])} "
        f"test={len(datasets['test'])} pos_weight={pos_weight:.2f}",
        flush=True,
    )

    sampler = TemperedDatasetEventPatchSampler(
        datasets["train"].frame,
        num_samples=len(datasets["train"]),
        seed=args.seed,
        dataset_temperature=args.temperature,
        event_temperature=args.temperature,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        ),
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = AltVisualAnchor(args.backbone).to(device)
    mean_d, std_d = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(pos_weight, device=device)
    )

    best = {"iou": -1.0, "state": None, "epoch": -1, "threshold": None, "ap": None}
    history = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running, batches = 0.0, 0
        for batch in loaders["train"]:
            visual = ((build_input(batch) - mean_d) / std_d).to(device, non_blocking=True)
            target, keep = masked_targets(batch)
            target = target.to(device, non_blocking=True)
            keep = keep.to(device, non_blocking=True)
            logits = model(visual)
            loss = (criterion(logits, target) * keep).sum() / keep.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        histogram = validation_histogram(model, loaders["val"], mean_d, std_d, device)
        threshold, metrics = choose_threshold(histogram)
        average_precision = histogram.average_precision()
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(batches, 1),
                "val_iou": metrics["iou"],
                "val_ap": average_precision,
                "val_threshold": threshold,
            }
        )
        print(
            f"[{fold_id}] epoch {epoch:02d} loss={running / max(batches, 1):.4f} "
            f"val_iou={metrics['iou']:.5f} val_ap={average_precision:.5f} "
            f"thr={threshold:.3f}",
            flush=True,
        )
        if metrics["iou"] > best["iou"]:
            best = {
                "iou": metrics["iou"],
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "threshold": float(threshold),
                "ap": float(average_precision),
            }
    if best["state"] is None:
        raise RuntimeError("training produced no selected checkpoint")
    model.load_state_dict(best["state"])

    cache = export_test_cache(
        model, loaders["test"], mean_d, std_d, device, fold_id, args
    )
    receipt = {
        "fold_id": fold_id,
        "backbone": args.backbone,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "best_epoch": int(best["epoch"]),
        "val_iou": float(best["iou"]),
        "val_ap": float(best["ap"]),
        "threshold": float(best["threshold"]),
        "pos_weight": pos_weight,
        "normalization_mean": [float(v) for v in mean],
        "normalization_std": [float(v) for v in std],
        "manifest_sha256": schema["manifest_sha256"],
        "split_sha256": schema["split_sha256"],
        "n_samples": cache["n_samples"],
        "n_events": cache["n_events"],
        "cache_path": cache["path"],
        "cache_sha256": cache["sha256"],
        "terrain_channel_order": list(
            json.loads(args.protocol_summary.read_text(encoding="utf-8"))
            ["terrain_contract"]["names"]
        ),
        "history": history,
    }
    write_json(args.cache_outdir / f"{fold_id}_oof_cache_receipt.json", receipt)
    return receipt


@torch.no_grad()
def export_test_cache(model, loader, mean, std, device, fold_id, args) -> dict[str, Any]:
    """导出与 Prithvi 锚点 schema 逐字段一致的 OOF 缓存。"""
    model.eval()
    buffers: dict[str, list] = {
        key: [] for key in
        ("probability", "target", "valid", "terrain", "terrain_valid")
    }
    sample_id, dataset_id, event_id = [], [], []
    for batch in loader:
        visual = ((build_input(batch) - mean) / std).to(device)
        probability = torch.sigmoid(model(visual))[:, 0].cpu().numpy().astype(np.float16)
        target, keep = masked_targets(batch)
        buffers["probability"].append(probability)
        buffers["target"].append(target[:, 0].numpy().astype(np.uint8))
        buffers["valid"].append(keep[:, 0].numpy().astype(np.uint8))
        buffers["terrain"].append(batch["terrain"].numpy().astype(np.float16))
        support = batch["terrain_valid"]
        if support.ndim == 4:
            support = support[:, 0]
        buffers["terrain_valid"].append((support.numpy() > 0).astype(np.uint8))
        sample_id.extend(str(item) for item in batch["sample_id"])
        dataset_id.extend(str(item) for item in batch["dataset_id"])
        event_id.extend(str(item) for item in batch["canonical_event_id"])

    payload = {
        "sample_id": np.asarray(sample_id, dtype="U160"),
        "dataset_id": np.asarray(dataset_id, dtype="U64"),
        "canonical_event_id": np.asarray(event_id, dtype="U96"),
        "visual_probability": np.concatenate(buffers["probability"]),
        "target": np.concatenate(buffers["target"]),
        "valid": np.concatenate(buffers["valid"]),
        "terrain": np.concatenate(buffers["terrain"]),
        "terrain_valid": np.concatenate(buffers["terrain_valid"]),
    }
    if len({len(payload[key]) for key in payload}) != 1:
        raise RuntimeError("exported arrays disagree in length")
    if len(set(sample_id)) != len(sample_id):
        raise RuntimeError("duplicate sample_id inside a single fold export")

    args.cache_outdir.mkdir(parents=True, exist_ok=True)
    path = args.cache_outdir / f"{fold_id}_oof_cache.npz"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_samples": len(sample_id),
        "n_events": len(set(event_id)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone", required=True,
        choices=(
            "dinov2_s_fpn", "hiera_small_mae_fpn", "fcmae_convnextv2_tiny_fpn",
            "dinov3_sat_l_fpn",
        ),
    )
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalization-samples", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-outdir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cache_outdir is None:
        args.cache_outdir = (
            PROJECT_ROOT
            / f"experiments/revision2026/pild_alt_anchor_{args.backbone}_v1/oof_cache"
        )
    args.cache_outdir = args.cache_outdir.resolve()
    args.manifest = args.manifest.resolve()
    args.protocol_summary = args.protocol_summary.resolve()
    args.split = args.split.resolve()
    started = time.time()

    receipts = []
    for fold_id in args.folds:
        cache = args.cache_outdir / f"{fold_id}_oof_cache.npz"
        receipt_path = args.cache_outdir / f"{fold_id}_oof_cache_receipt.json"
        if cache.is_file() and receipt_path.is_file():
            print(f"[skip] {fold_id}: cache already present", flush=True)
            receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
            continue
        receipts.append(train_fold(args, fold_id))

    write_json(
        args.cache_outdir / "export_summary.json",
        {
            "schema_version": "pild_alt_visual_anchor_oof_cache.v1",
            "scope": "alternative visual anchor trained under the Prithvi anchor data contract",
            "backbone": args.backbone,
            "labels_used_for_selection": False,
            "folds": [item["fold_id"] for item in receipts],
            "total_samples": sum(int(item["n_samples"]) for item in receipts),
            "elapsed_seconds": round(time.time() - started, 2),
            "receipts": receipts,
        },
    )
    print(f"[done] {args.backbone}: 缓存写出到 {args.cache_outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
