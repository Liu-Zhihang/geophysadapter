#!/usr/bin/env python3
"""Train strict_t2 change_rgb baseline with numeric physics conditioning."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

from train_strict_t2_change_rgb_baseline import (
    CachedChangeRgbH5Dataset,
    StrictT2ChangeRgbDataset,
    default_cache_path,
)
from train_strict_t2_postrgb_baseline import (
    build_weighted_sampler,
    build_dataset_weight_map,
    parse_named_value_overrides,
    parse_threshold_grid,
    read_manifest,
    set_seed,
    subset_rows,
)
from train_strict_t2_postrgb_phys_baseline import (
    EpochStat,
    PhysicsAugmentedDataset,
    PhysicsFiLMUNet,
    find_best_threshold_physics,
    gather_train_stats,
    load_physics_maps,
    resolve_physics_vector_cols,
    run_epoch,
)


def metadata_from_base_dataset(ds: Dataset) -> tuple[list[str], list[str]]:
    if isinstance(ds, CachedChangeRgbH5Dataset):
        with h5py.File(ds.h5_path, "r") as f:
            sample_ids = [f["sample_id"][idx].decode("utf-8") for idx in ds.indices]
            event_uids = [f["event_uid"][idx].decode("utf-8") for idx in ds.indices]
        return sample_ids, event_uids
    if isinstance(ds, StrictT2ChangeRgbDataset):
        return [row["sample_id"] for row in ds.rows], [row["event_uid"] for row in ds.rows]
    raise TypeError(f"unsupported dataset type: {type(ds)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strict_t2 change_rgb visual + numeric physics baseline")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--manifest", default="", help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_manifest_change_rgb.csv")
    p.add_argument("--physics-csv", default="", help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_physics_vectors_change_rgb_v1.csv")
    p.add_argument("--outdir", default="", help="default: experiments/strict_t2_change_rgb_phys_baseline_stage0")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epoch-samples", type=int, default=0)
    p.add_argument("--sampler-power", type=float, default=1.0)
    p.add_argument("--sampler-overrides", default="")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-pos-weight", type=float, default=4.0)
    p.add_argument("--loss-balance-power", type=float, default=0.0)
    p.add_argument("--loss-balance-overrides", default="")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--physics-groups", default="material", help="comma-separated: terrain,material,trigger,proxy or full")
    p.add_argument("--dynamic-gate-mode", default="none", choices=["none", "proxy"])
    p.add_argument("--dynamic-gate-source", default="proxy", choices=["proxy", "dynamic"])
    p.add_argument("--dynamic-gate-scale", type=float, default=1.0)
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
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)

    root = Path(args.root)
    manifest = (
        Path(args.manifest)
        if args.manifest.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_manifest_change_rgb.csv"
    )
    physics_csv = (
        Path(args.physics_csv)
        if args.physics_csv.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_physics_vectors_change_rgb_v1.csv"
    )
    outdir = Path(args.outdir) if args.outdir.strip() else root / "experiments" / "strict_t2_change_rgb_phys_baseline_stage0"
    outdir.mkdir(parents=True, exist_ok=True)
    train_cache_h5 = Path(args.train_cache_h5) if args.train_cache_h5.strip() else default_cache_path(root, "train", args.patch_size)
    val_cache_h5 = Path(args.val_cache_h5) if args.val_cache_h5.strip() else default_cache_path(root, "val", args.patch_size)
    test_cache_h5 = Path(args.test_cache_h5) if args.test_cache_h5.strip() else default_cache_path(root, "test", args.patch_size)
    sampler_overrides = parse_named_value_overrides(args.sampler_overrides)
    loss_balance_overrides = parse_named_value_overrides(args.loss_balance_overrides)
    threshold_grid = parse_threshold_grid(args.threshold_grid)

    vector_cols = resolve_physics_vector_cols(args.physics_groups)
    sample_map, event_map = load_physics_maps(physics_csv, vector_cols)
    rows = read_manifest(manifest)
    exclude_datasets = {item.strip() for item in args.exclude_datasets.split(",") if item.strip()}
    if exclude_datasets:
        rows = [row for row in rows if row["dataset_id"] not in exclude_datasets]
    train_rows = subset_rows([row for row in rows if row["role"] == "train"], args.train_limit)
    val_rows = subset_rows([row for row in rows if row["role"] == "val"], args.val_limit)
    test_rows = subset_rows([row for row in rows if row["role"] == "test"], args.test_limit)

    device = torch.device("cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"[info] device={device}")
    print(f"[info] manifest={manifest}")
    print(f"[info] physics_csv={physics_csv}")
    print(f"[info] physics_groups={args.physics_groups}")
    print(f"[info] train_cache_h5={train_cache_h5 if train_cache_h5.exists() else 'missing'}")
    print(f"[info] val_cache_h5={val_cache_h5 if val_cache_h5.exists() else 'missing'}")
    print(f"[info] test_cache_h5={test_cache_h5 if test_cache_h5.exists() else 'missing'}")
    print(f"[info] rows train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    if train_cache_h5.exists():
        train_base = CachedChangeRgbH5Dataset(train_cache_h5, exclude_datasets=exclude_datasets)
    else:
        train_base = StrictT2ChangeRgbDataset(train_rows, patch_size=args.patch_size)
    if val_cache_h5.exists():
        val_base = CachedChangeRgbH5Dataset(val_cache_h5, exclude_datasets=exclude_datasets)
    else:
        val_base = StrictT2ChangeRgbDataset(val_rows, patch_size=args.patch_size)
    if test_cache_h5.exists():
        test_base = CachedChangeRgbH5Dataset(test_cache_h5, exclude_datasets=exclude_datasets)
    else:
        test_base = StrictT2ChangeRgbDataset(test_rows, patch_size=args.patch_size)

    train_sample_ids, train_event_uids = metadata_from_base_dataset(train_base)
    mean, std = gather_train_stats(train_sample_ids, train_event_uids, sample_map, event_map)

    train_ds = PhysicsAugmentedDataset(train_base, sample_map, event_map, mean, std)
    val_ds = PhysicsAugmentedDataset(val_base, sample_map, event_map, mean, std)
    test_ds = PhysicsAugmentedDataset(test_base, sample_map, event_map, mean, std)

    train_counts = dict(train_ds.dataset_counter or Counter(row["dataset_id"] for row in train_rows))
    val_counts = dict(val_ds.dataset_counter or Counter(row["dataset_id"] for row in val_rows))
    test_counts = dict(test_ds.dataset_counter or Counter(row["dataset_id"] for row in test_rows))
    sampler_weight_map = build_dataset_weight_map(train_ds.dataset_ids, power=args.sampler_power, overrides=sampler_overrides)
    loss_weight_map = build_dataset_weight_map(
        train_ds.dataset_ids,
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
            train_ds.dataset_ids,
            epoch_samples=args.epoch_samples,
            power=args.sampler_power,
            overrides=sampler_overrides,
        ),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
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

    model = PhysicsFiLMUNet(
        in_ch=6,
        physics_dim=len(vector_cols),
        base=32,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        physics_cols=vector_cols,
        dynamic_gate_mode=args.dynamic_gate_mode,
        dynamic_gate_source=args.dynamic_gate_source,
        dynamic_gate_scale=args.dynamic_gate_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_iou = -1.0
    history: list[EpochStat] = []
    best_model_path = outdir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_steps=args.max_train_steps,
            bce_pos_weight=args.bce_pos_weight,
            dataset_loss_weights=train_loss_weights,
        )
        val_metrics, val_by_dataset = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            max_steps=args.max_eval_steps,
            bce_pos_weight=args.bce_pos_weight,
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
    test_metrics_default, test_by_dataset_default = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        max_steps=args.max_eval_steps,
        bce_pos_weight=args.bce_pos_weight,
    )
    eval_threshold = 0.5
    threshold_search: list[dict[str, float]] = []
    if args.tune_threshold_on_val:
        eval_threshold, threshold_search = find_best_threshold_physics(
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
        test_metrics, test_by_dataset = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=None,
            max_steps=args.max_eval_steps,
            bce_pos_weight=args.bce_pos_weight,
            threshold=eval_threshold,
        )

    summary = {
        "manifest": str(manifest),
        "physics_csv": str(physics_csv),
        "physics_groups": args.physics_groups,
        "physics_dim": len(vector_cols),
        "physics_vector_cols": vector_cols,
        "outdir": str(outdir),
        "device": str(device),
        "rows": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "dataset_counts": {"train": train_counts, "val": val_counts, "test": test_counts},
        "physics_norm": {"mean": mean.tolist(), "std": std.tolist()},
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
