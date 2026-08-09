#!/usr/bin/env python3
"""Train a matched Twin DINOv2-S visual baseline on unified PILD-GEO4."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SCRIPT_DIR, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_pild_support_only_additive_v1 import (  # noqa: E402
    counts_from_predictions,
)
from pild_sen12_training_loader_v2 import (  # noqa: E402
    NaturalPatchSampler,
    UnifiedPILDSen12Dataset,
    sha256_file,
)
from run_support_adapter_timmfm import TimmFPNHead, TimmFeatureEncoder  # noqa: E402
from train_pild_sen12_roleaware_v1 import (  # noqa: E402
    BinaryHistogram,
    choose_threshold,
    metrics_from_counts,
    state_to_cpu,
    tensor_sha256,
    validate_protocol_schema,
)
from train_pild_support_only_terrain_v1 import write_json  # noqa: E402


MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
OUT_INDICES = (2, 5, 8, 11)
IMAGE_SIZE = 224
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[None, :, None, None]
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[None, :, None, None]


class TwinDINOv2FPN(nn.Module):
    """Shared frozen encoder with post and absolute-change FPN decoding."""

    def __init__(self, *, pretrained: bool = True, hidden: int = 96) -> None:
        super().__init__()
        self.encoder = TimmFeatureEncoder(
            MODEL_NAME,
            in_chans=3,
            pretrained=pretrained,
            img_size=IMAGE_SIZE,
            out_indices=OUT_INDICES,
            freeze_backbone=True,
        )
        self.decoder = TimmFPNHead(
            [2 * channels for channels in self.encoder.channels], hidden=hidden
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, optical: torch.Tensor) -> torch.Tensor:
        if optical.ndim != 5 or optical.shape[1:3] != (6, 4):
            raise ValueError(f"expected optical [B,6,4,H,W], got {tuple(optical.shape)}")
        pre = optical[:, :3, :2].mean(dim=2)
        post = optical[:, :3, 2:].mean(dim=2)
        mean = IMAGE_MEAN.to(optical.device)
        std = IMAGE_STD.to(optical.device)
        pre = (pre - mean) / std
        post = (post - mean) / std
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        change_features = [
            torch.cat((post_feature, torch.abs(post_feature - pre_feature)), dim=1)
            for pre_feature, post_feature in zip(pre_features, post_features)
        ]
        logits, _ = self.decoder(change_features, optical.shape[-2:])
        return logits


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    pos_weight: float,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    bce = (raw * valid).sum() / valid.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits) * valid
    truth = target * valid
    intersection = (probability * truth).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + 0.5 * dice


def estimate_pos_weight(dataset: UnifiedPILDSen12Dataset) -> float:
    positive = 0
    negative = 0
    for index in range(len(dataset)):
        item = dataset[index]
        target = item["mask"] >= 0.5
        valid = item["valid_mask"].bool()
        positive += int((target & valid).sum())
        negative += int((~target & valid).sum())
    if positive == 0:
        raise RuntimeError("training role contains no positive pixels")
    return float(np.clip(negative / positive, 1.0, 50.0))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float | None = None,
    export_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    histogram = BinaryHistogram()
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, Any]]] = []
    for batch in loader:
        optical = batch["optical"].to(device)
        target = batch["mask"].to(device)
        valid = batch["valid_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(optical)
        probability = torch.sigmoid(logits.float())
        histogram.update(probability, target, valid)
        if export_rows:
            batches.append((probability.cpu(), target.cpu(), valid.cpu(), batch))
    if threshold is None:
        threshold, metrics = choose_threshold(histogram)
    else:
        metrics = metrics_from_counts(histogram.counts(threshold))
    result = {
        **metrics,
        "ap": histogram.average_precision(),
        "threshold": float(threshold),
    }
    rows: list[dict[str, Any]] = []
    if export_rows:
        for probability, target, valid, batch in batches:
            for index in range(target.shape[0]):
                pair = counts_from_predictions(
                    probability[index : index + 1],
                    probability[index : index + 1],
                    target[index : index + 1],
                    valid[index : index + 1],
                    threshold=float(threshold),
                )
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "dataset_id": str(batch["dataset_id"][index]),
                        "source_event_id": str(batch["source_event_id"][index]),
                        "canonical_event_id": str(batch["canonical_event_id"][index]),
                        "tp": pair["baseline_tp"],
                        "fp": pair["baseline_fp"],
                        "fn": pair["baseline_fn"],
                        "tn": pair["baseline_tn"],
                        "valid_pixels": pair["valid_pixels"],
                    }
                )
    return result, rows


def aggregate_events(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        groups[str(row["canonical_event_id"])].append(row)
    output: list[dict[str, Any]] = []
    for event_id in sorted(groups):
        rows = groups[event_id]
        counts = {
            key: sum(int(row[key]) for row in rows)
            for key in ("tp", "fp", "fn", "tn")
        }
        output.append(
            {
                "canonical_event_id": event_id,
                "dataset_ids": ";".join(sorted({str(row["dataset_id"]) for row in rows})),
                "n_samples": len(rows),
                **counts,
                **metrics_from_counts(counts),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-summary", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch-samples", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--decoder-hidden", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")

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
    (stage / "metrics").mkdir(parents=True)
    (stage / "checkpoints").mkdir()
    command = shlex.join([sys.executable, *sys.argv])
    (stage / "run.log").write_text(command + "\n", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with (stage / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    started = time.time()
    set_seed(args.seed)
    datasets = {
        role: UnifiedPILDSen12Dataset(
            args.manifest,
            args.protocol_summary,
            split_path=args.split,
            fold_id=args.fold_id,
            role=role,
            readiness="core",
        )
        for role in ("train", "val", "test")
    }
    pos_weight = estimate_pos_weight(datasets["train"])
    epoch_samples = args.epoch_samples or len(datasets["train"])
    sampler = NaturalPatchSampler(
        datasets["train"].frame, num_samples=epoch_samples, seed=args.seed
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        ),
    }
    device = torch.device(args.device)
    model = TwinDINOv2FPN(
        pretrained=not args.no_pretrained, hidden=args.decoder_hidden
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_threshold = 0.5
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(loaders["train"]):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            optical = batch["optical"].to(device)
            target = batch["mask"].to(device)
            valid = batch["valid_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(optical)
                loss = masked_loss(logits, target, valid, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation, _ = evaluate(model, loaders["val"], device=device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
        }
        history.append(row)
        key = (
            float(validation["iou"]),
            float(validation["ap"]),
            -float(validation["errors"]),
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_threshold = float(validation["threshold"])
            best_state = state_to_cpu(model.decoder)
        log(
            f"epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"val_iou={validation['iou']:.6f} val_ap={validation['ap']:.6f} "
            f"threshold={validation['threshold']:.3f}"
        )
    if best_state is None:
        raise RuntimeError("training produced no selectable decoder")
    model.decoder.load_state_dict(best_state, strict=True)
    test, sample_rows = evaluate(
        model,
        loaders["test"],
        device=device,
        threshold=best_threshold,
        export_rows=True,
    )
    event_rows = aggregate_events(sample_rows)
    write_csv(stage / "metrics" / "per_sample_metrics.csv", sample_rows)
    write_csv(stage / "metrics" / "per_event_metrics.csv", event_rows)
    encoder_state = state_to_cpu(model.encoder)
    checkpoint = {
        "schema_version": "pild_geo4_twin_dinov2_checkpoint.v2",
        "variant": "V",
        "model": {
            "name": MODEL_NAME,
            "out_indices": list(OUT_INDICES),
            "image_size": IMAGE_SIZE,
            "temporal_contract": "mean(first two RGB), mean(last two RGB)",
            "fusion": "concat(post_features, abs(post_features-pre_features))",
        },
        "encoder_state_dict": encoder_state,
        "encoder_sha256": tensor_sha256(encoder_state),
        "decoder_state_dict": best_state,
        "decoder_sha256": tensor_sha256(best_state),
        "threshold": best_threshold,
        "threshold_source": "visual_validation",
        "best_epoch": best_epoch,
        "determinism": {
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "protocol_summary_sha256": sha256_file(args.protocol_summary),
            "split_sha256": schema["split_sha256"],
            "fold_id": args.fold_id,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, stage / "checkpoints" / "best_model.pt")
    summary = {
        "schema_version": "pild_geo4_twin_dinov2_run.v1",
        "status": "COMPLETE",
        "artifact_state": "exploratory_single_seed_gate",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "pos_weight": pos_weight,
        "history": history,
        "test": test,
        "counts": {
            role: {
                "samples": len(dataset),
                "events": int(dataset.frame["canonical_event_id"].nunique()),
            }
            for role, dataset in datasets.items()
        },
        "identity": checkpoint["identity"],
        "artifacts": {
            "checkpoint_sha256": sha256_file(stage / "checkpoints" / "best_model.pt"),
            "per_sample_sha256": sha256_file(stage / "metrics" / "per_sample_metrics.csv"),
            "per_event_sha256": sha256_file(stage / "metrics" / "per_event_metrics.csv"),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(stage / "run_summary.json", summary)
    write_json(
        stage / "DONE.json",
        {
            "status": "COMPLETE",
            "run_summary_sha256": sha256_file(stage / "run_summary.json"),
            **summary["artifacts"],
        },
    )
    os.replace(stage, outdir)
    print(
        f"completed {args.fold_id}: test_iou={test['iou']:.6f}, "
        f"test_ap={test['ap']:.6f}, best_epoch={best_epoch}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
