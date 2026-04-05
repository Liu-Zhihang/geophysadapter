#!/usr/bin/env python3
"""Train a stronger-backbone strict_t2 post_rgb visual baseline."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from strict_t2_strong_backbone_common import (
    BinaryDeepLabV3,
    EpochStat,
    find_best_threshold_strong,
    run_epoch_strong,
)
from train_strict_t2_postrgb_baseline import (
    CachedPostRgbH5Dataset,
    StrictT2PostRgbDataset,
    build_dataset_weight_map,
    build_weighted_sampler,
    parse_named_value_overrides,
    parse_threshold_grid,
    read_gdcld_index,
    read_manifest,
    set_seed,
    subset_rows,
    default_eval_cache_path,
    default_train_cache_path,
)


def resolve_postrgb_train_cache_path(root: Path, patch_size: int) -> Path:
    candidates = [
        root / "processed" / "hybrid_pinn" / "strict_t2_postrgb_train_cache_v2_skiperr" / f"train_postrgb_p{patch_size}.h5",
        root / "processed" / "hybrid_pinn" / "strict_t2_postrgb_train_cache_bal2048_v1" / f"train_postrgb_p{patch_size}.h5",
        default_train_cache_path(root, patch_size),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train stronger-backbone strict_t2 post_rgb visual baseline")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--manifest", default="", help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_manifest_post_rgb.csv")
    p.add_argument("--outdir", default="", help="default: experiments/strict_t2_postrgb_strong_visual_stage0")
    p.add_argument("--backbone", default="deeplabv3_resnet50", choices=["deeplabv3_resnet50", "deeplabv3_resnet101"])
    p.add_argument("--no-pretrained-backbone", action="store_true")
    p.add_argument("--aux-loss-weight", type=float, default=0.2)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--gdcld-crop-size", type=int, default=512)
    p.add_argument("--gdcld-jitter", type=int, default=64)
    p.add_argument("--gdcld-index", default="", help="optional precomputed GDCLD scene index csv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epoch-samples", type=int, default=2048)
    p.add_argument("--sampler-power", type=float, default=0.5)
    p.add_argument("--sampler-overrides", default="")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-pos-weight", type=float, default=6.0)
    p.add_argument("--loss-balance-power", type=float, default=0.25)
    p.add_argument("--loss-balance-overrides", default="")
    p.add_argument("--seed", type=int, default=20260310)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "auto"])
    p.add_argument("--train-cache-h5", default="", help="optional cached HDF5 for train split")
    p.add_argument("--val-cache-h5", default="", help="optional cached HDF5 for val split")
    p.add_argument("--test-cache-h5", default="", help="optional cached HDF5 for test split")
    p.add_argument("--exclude-datasets", default="", help="comma-separated dataset_id values to drop")
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--tune-threshold-on-val", action="store_true")
    p.add_argument("--threshold-grid", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-eval-steps", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("DeepLabV3 training requires --batch-size >= 2 because BatchNorm is used in the ASPP head.")
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)

    root = Path(args.root)
    manifest = (
        Path(args.manifest)
        if args.manifest.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_manifest_post_rgb.csv"
    )
    default_outdir = f"strict_t2_postrgb_{args.backbone}_visual_stage0"
    outdir = Path(args.outdir) if args.outdir.strip() else root / "experiments" / default_outdir
    outdir.mkdir(parents=True, exist_ok=True)
    gdcld_index_path = (
        Path(args.gdcld_index)
        if args.gdcld_index.strip()
        else root / "metadata" / "manifests" / "gdcld_postrgb_scene_index_v1.csv"
    )
    train_cache_h5 = Path(args.train_cache_h5) if args.train_cache_h5.strip() else resolve_postrgb_train_cache_path(root, args.patch_size)
    val_cache_h5 = Path(args.val_cache_h5) if args.val_cache_h5.strip() else default_eval_cache_path(root, "val", args.patch_size)
    test_cache_h5 = Path(args.test_cache_h5) if args.test_cache_h5.strip() else default_eval_cache_path(root, "test", args.patch_size)
    sampler_overrides = parse_named_value_overrides(args.sampler_overrides)
    loss_balance_overrides = parse_named_value_overrides(args.loss_balance_overrides)
    threshold_grid = parse_threshold_grid(args.threshold_grid)

    rows = read_manifest(manifest)
    exclude_datasets = {item.strip() for item in args.exclude_datasets.split(",") if item.strip()}
    if exclude_datasets:
        rows = [row for row in rows if row["dataset_id"] not in exclude_datasets]
    train_rows = subset_rows([row for row in rows if row["role"] == "train"], args.train_limit)
    val_rows = subset_rows([row for row in rows if row["role"] == "val"], args.val_limit)
    test_rows = subset_rows([row for row in rows if row["role"] == "test"], args.test_limit)
    use_train_cache = train_cache_h5.exists() and args.train_limit <= 0
    use_val_cache = val_cache_h5.exists() and args.val_limit <= 0
    use_test_cache = test_cache_h5.exists() and args.test_limit <= 0

    device = torch.device("cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"[info] device={device}")
    print(f"[info] manifest={manifest}")
    print(f"[info] backbone={args.backbone}")
    print(f"[info] pretrained_backbone={not args.no_pretrained_backbone}")
    print(f"[info] gdcld_index={gdcld_index_path}")
    print(f"[info] train_cache_h5={train_cache_h5 if train_cache_h5.exists() else 'missing'}")
    print(f"[info] val_cache_h5={val_cache_h5 if val_cache_h5.exists() else 'missing'}")
    print(f"[info] test_cache_h5={test_cache_h5 if test_cache_h5.exists() else 'missing'}")
    print(f"[info] rows train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"[info] use_cache train/val/test={use_train_cache}/{use_val_cache}/{use_test_cache}")

    gdcld_index = read_gdcld_index(gdcld_index_path)
    if use_train_cache:
        train_ds = CachedPostRgbH5Dataset(train_cache_h5, exclude_datasets=exclude_datasets)
        train_dataset_ids = list(train_ds.dataset_ids)
        train_counts = dict(train_ds.dataset_counter)
    else:
        train_ds = StrictT2PostRgbDataset(
            train_rows,
            args.patch_size,
            args.gdcld_crop_size,
            deterministic_scene_crop=False,
            gdcld_index=gdcld_index,
            gdcld_jitter=args.gdcld_jitter,
        )
        train_dataset_ids = [row["dataset_id"] for row in train_rows]
        train_counts = dict(Counter(train_dataset_ids))
    if use_val_cache:
        val_ds = CachedPostRgbH5Dataset(val_cache_h5, exclude_datasets=exclude_datasets)
    else:
        val_ds = StrictT2PostRgbDataset(
            val_rows,
            args.patch_size,
            args.gdcld_crop_size,
            deterministic_scene_crop=True,
            gdcld_index=gdcld_index,
            gdcld_jitter=0,
        )
    if use_test_cache:
        test_ds = CachedPostRgbH5Dataset(test_cache_h5, exclude_datasets=exclude_datasets)
    else:
        test_ds = StrictT2PostRgbDataset(
            test_rows,
            args.patch_size,
            args.gdcld_crop_size,
            deterministic_scene_crop=True,
            gdcld_index=gdcld_index,
            gdcld_jitter=0,
        )

    val_counts = dict(getattr(val_ds, "dataset_counter", Counter(row["dataset_id"] for row in val_rows)))
    test_counts = dict(getattr(test_ds, "dataset_counter", Counter(row["dataset_id"] for row in test_rows)))
    sampler_weight_map = build_dataset_weight_map(train_dataset_ids, power=args.sampler_power, overrides=sampler_overrides)
    loss_weight_map = build_dataset_weight_map(
        train_dataset_ids,
        power=args.loss_balance_power,
        overrides=loss_balance_overrides,
    )
    train_loss_weights = None
    if args.loss_balance_power > 0 or loss_balance_overrides:
        train_loss_weights = loss_weight_map
    print(f"[info] effective train datasets={train_counts}")
    print(f"[info] effective val datasets={val_counts}")
    print(f"[info] effective test datasets={test_counts}")
    print(f"[info] sampler_weight_map={json.dumps(sampler_weight_map, ensure_ascii=False)}")
    if train_loss_weights:
        print(f"[info] train_loss_weight_map={json.dumps(train_loss_weights, ensure_ascii=False)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=build_weighted_sampler(
            train_dataset_ids,
            epoch_samples=args.epoch_samples,
            power=args.sampler_power,
            overrides=sampler_overrides,
        ),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = BinaryDeepLabV3(
        in_channels=3,
        backbone_name=args.backbone,
        pretrained_backbone=not args.no_pretrained_backbone,
        aux_loss=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_iou = -1.0
    history: list[EpochStat] = []
    best_model_path = outdir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics, _ = run_epoch_strong(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_steps=args.max_train_steps,
            bce_pos_weight=args.bce_pos_weight,
            dataset_loss_weights=train_loss_weights,
            aux_loss_weight=args.aux_loss_weight,
        )
        val_metrics, val_by_dataset = run_epoch_strong(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            max_steps=args.max_eval_steps,
            bce_pos_weight=args.bce_pos_weight,
            aux_loss_weight=args.aux_loss_weight,
        )
        scheduler.step()
        sec = time.time() - t0
        history.append(
            EpochStat(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_bce=train_metrics["bce"],
                train_dice=train_metrics["dice"],
                train_iou=train_metrics["iou"],
                train_f1=train_metrics["f1"],
                val_loss=val_metrics["loss"],
                val_bce=val_metrics["bce"],
                val_dice=val_metrics["dice"],
                val_iou=val_metrics["iou"],
                val_f1=val_metrics["f1"],
                lr=float(scheduler.get_last_lr()[0]),
                sec=sec,
            )
        )
        print(
            f"[epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.4f} train_iou={train_metrics['iou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['iou']:.4f} sec={sec:.1f}"
        )
        print(f"[epoch {epoch}] val_by_dataset={json.dumps(val_by_dataset, ensure_ascii=False)}")
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_model_path)

    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics_default, test_by_dataset_default = run_epoch_strong(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        max_steps=args.max_eval_steps,
        bce_pos_weight=args.bce_pos_weight,
        threshold=0.5,
        aux_loss_weight=args.aux_loss_weight,
    )
    eval_threshold = 0.5
    threshold_search: list[dict[str, float]] = []
    if args.tune_threshold_on_val:
        eval_threshold, threshold_search = find_best_threshold_strong(
            model=model,
            loader=val_loader,
            device=device,
            thresholds=threshold_grid,
            max_steps=args.max_eval_steps,
        )
        print(f"[info] tuned_eval_threshold={eval_threshold:.2f}")
    test_metrics = test_metrics_default
    test_by_dataset = test_by_dataset_default
    if abs(eval_threshold - 0.5) > 1e-8:
        test_metrics, test_by_dataset = run_epoch_strong(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=None,
            max_steps=args.max_eval_steps,
            bce_pos_weight=args.bce_pos_weight,
            threshold=eval_threshold,
            aux_loss_weight=args.aux_loss_weight,
        )

    summary = {
        "manifest": str(manifest),
        "outdir": str(outdir),
        "device": str(device),
        "model_family": "strong_visual",
        "backbone": args.backbone,
        "pretrained_backbone": not args.no_pretrained_backbone,
        "rows": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "resolved_cache_h5": {
            "train": str(train_cache_h5),
            "val": str(val_cache_h5),
            "test": str(test_cache_h5),
        },
        "used_cache": {
            "train": use_train_cache,
            "val": use_val_cache,
            "test": use_test_cache,
        },
        "dataset_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "config": vars(args),
        "best_val_iou": best_iou,
        "best_epoch": int(ckpt["epoch"]),
        "sampler_weight_map": sampler_weight_map,
        "train_loss_weight_map": train_loss_weights or {},
        "eval_threshold": eval_threshold,
        "threshold_search": threshold_search,
        "test_metrics_default": test_metrics_default,
        "test_by_dataset_default": test_by_dataset_default,
        "test_metrics": test_metrics,
        "test_by_dataset": test_by_dataset,
        "history": [asdict(item) for item in history],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": summary["best_epoch"], "best_val_iou": best_iou, "test": test_metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
