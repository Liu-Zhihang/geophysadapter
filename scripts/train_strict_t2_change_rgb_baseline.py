#!/usr/bin/env python3
"""Train a strict_t2 change_rgb baseline on real pre/post pairs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from train_baseline_dlr_unet import SmallUNet
from train_strict_t2_postrgb_baseline import (
    EpochStat,
    batch_dataset_weights,
    build_weighted_sampler,
    build_dataset_weight_map,
    compute_segmentation_loss,
    find_best_threshold_visual,
    masked_stats,
    normalize_rgb,
    parse_named_value_overrides,
    parse_threshold_grid,
    read_manifest,
    resize_image_mask,
    set_seed,
    subset_rows,
)


DLR_PRE_KEYS = ["PRE1_B02", "PRE1_B03", "PRE1_B04"]
DLR_POST_KEYS = ["POST1_B02", "POST1_B03", "POST1_B04"]
DLR_MASK_KEY = "None_MASK"


def default_cache_path(root: Path, split: str, patch_size: int) -> Path:
    return root / "processed" / "hybrid_pinn" / "strict_t2_change_rgb_cache_v1" / f"{split}_changergb_p{patch_size}.h5"


class StrictT2ChangeRgbDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], patch_size: int) -> None:
        self.rows = rows
        self.patch_size = patch_size
        self._h5_cache: dict[Path, h5py.File] = {}
        self._glad_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.dataset_ids = [row["dataset_id"] for row in rows]
        self.dataset_counter = Counter(self.dataset_ids)

    def __len__(self) -> int:
        return len(self.rows)

    def _get_h5(self, path: Path) -> h5py.File:
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def _load_glad(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_id = row["sample_id"]
        if sample_id in self._glad_cache:
            image, target, valid = self._glad_cache[sample_id]
            return image.copy(), target.copy(), valid.copy()
        pre = tifffile.imread(row["pre_path"]).astype(np.float32)[..., :3]
        post = tifffile.imread(row["post_path"]).astype(np.float32)[..., :3]
        image = np.concatenate([pre, post], axis=-1)
        label = tifffile.imread(row["label_path"]).astype(np.float32)
        valid = np.ones_like(label, dtype=np.float32)
        target = (label > 0.5).astype(np.float32)
        self._glad_cache[sample_id] = (image, target, valid)
        return image.copy(), target.copy(), valid.copy()

    def _load_dlr(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h5_path = Path(row["h5_path"])
        idx = int(float(row["h5_sample_index"]))
        f = self._get_h5(h5_path)
        pre = [f[key][idx, 0].astype(np.float32) / 10000.0 for key in DLR_PRE_KEYS]
        post = [f[key][idx, 0].astype(np.float32) / 10000.0 for key in DLR_POST_KEYS]
        image = np.stack(pre + post, axis=-1)
        target = f[DLR_MASK_KEY][idx, 0].astype(np.float32)
        valid = np.ones_like(target, dtype=np.float32)
        return image, target, valid

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        if row["sample_kind"] == "glad_pre_post":
            image, target, valid = self._load_glad(row)
        elif row["sample_kind"] == "dlr_h5_patch":
            image, target, valid = self._load_dlr(row)
        else:
            raise ValueError(f"unsupported sample_kind={row['sample_kind']}")
        image = normalize_rgb(image)
        x, y, v = resize_image_mask(
            image=image,
            mask=target.astype(np.float32),
            valid=valid.astype(np.float32),
            patch_size=self.patch_size,
        )
        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(y),
            "valid": torch.from_numpy(v),
            "dataset_id": row["dataset_id"],
            "sample_kind": row["sample_kind"],
            "sample_id": row["sample_id"],
            "event_uid": row["event_uid"],
            "role": row["role"],
        }


class CachedChangeRgbH5Dataset(Dataset):
    def __init__(self, h5_path: Path, exclude_datasets: set[str] | None = None):
        self.h5_path = Path(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            dataset_ids = [v.decode("utf-8") for v in f["dataset_id"][:]]
        exclude = exclude_datasets or set()
        self.indices = [idx for idx, ds in enumerate(dataset_ids) if ds not in exclude]
        self.dataset_ids = [dataset_ids[idx] for idx in self.indices]
        self.dataset_counter = Counter(dataset_ids[idx] for idx in self.indices)
        self.length = len(self.indices)

    def __len__(self) -> int:
        return self.length

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        f = self._get_h5()
        real_idx = self.indices[idx]
        return {
            "image": torch.from_numpy(f["image"][real_idx]),
            "mask": torch.from_numpy(f["mask"][real_idx]),
            "valid": torch.from_numpy(f["valid"][real_idx]),
            "dataset_id": f["dataset_id"][real_idx].decode("utf-8"),
            "sample_kind": f["sample_kind"][real_idx].decode("utf-8"),
            "sample_id": f["sample_id"][real_idx].decode("utf-8"),
            "event_uid": f["event_uid"][real_idx].decode("utf-8"),
            "role": f["role"][real_idx].decode("utf-8"),
        }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int = 0,
    bce_pos_weight: float = 1.0,
    dataset_loss_weights: dict[str, float] | None = None,
    threshold: float = 0.5,
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
            logits = model(x)
            sample_weights = batch_dataset_weights(
                batch["dataset_id"],
                dataset_loss_weights=dataset_loss_weights if train else None,
                device=logits.device,
                dtype=logits.dtype,
            )
            bce, dice, loss = compute_segmentation_loss(
                logits,
                y,
                v,
                pos_weight=bce_pos_weight,
                sample_weights=sample_weights,
            )
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strict_t2 change_rgb baseline")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument(
        "--manifest",
        default="",
        help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_manifest_change_rgb.csv",
    )
    p.add_argument("--outdir", default="", help="default: experiments/strict_t2_change_rgb_baseline_stage0")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epoch-samples", type=int, default=0)
    p.add_argument("--sampler-power", type=float, default=1.0)
    p.add_argument("--sampler-overrides", default="")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-pos-weight", type=float, default=4.0)
    p.add_argument("--loss-balance-power", type=float, default=0.0)
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
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)

    root = Path(args.root)
    manifest = (
        Path(args.manifest)
        if args.manifest.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_manifest_change_rgb.csv"
    )
    outdir = Path(args.outdir) if args.outdir.strip() else root / "experiments" / "strict_t2_change_rgb_baseline_stage0"
    outdir.mkdir(parents=True, exist_ok=True)
    train_cache_h5 = Path(args.train_cache_h5) if args.train_cache_h5.strip() else default_cache_path(root, "train", args.patch_size)
    val_cache_h5 = Path(args.val_cache_h5) if args.val_cache_h5.strip() else default_cache_path(root, "val", args.patch_size)
    test_cache_h5 = Path(args.test_cache_h5) if args.test_cache_h5.strip() else default_cache_path(root, "test", args.patch_size)
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

    device = torch.device("cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"[info] device={device}")
    print(f"[info] manifest={manifest}")
    print(f"[info] train_cache_h5={train_cache_h5 if train_cache_h5.exists() else 'missing'}")
    print(f"[info] val_cache_h5={val_cache_h5 if val_cache_h5.exists() else 'missing'}")
    print(f"[info] test_cache_h5={test_cache_h5 if test_cache_h5.exists() else 'missing'}")
    print(f"[info] rows train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    if train_cache_h5.exists():
        train_ds = CachedChangeRgbH5Dataset(train_cache_h5, exclude_datasets=exclude_datasets)
    else:
        train_ds = StrictT2ChangeRgbDataset(train_rows, patch_size=args.patch_size)
    if val_cache_h5.exists():
        val_ds = CachedChangeRgbH5Dataset(val_cache_h5, exclude_datasets=exclude_datasets)
    else:
        val_ds = StrictT2ChangeRgbDataset(val_rows, patch_size=args.patch_size)
    if test_cache_h5.exists():
        test_ds = CachedChangeRgbH5Dataset(test_cache_h5, exclude_datasets=exclude_datasets)
    else:
        test_ds = StrictT2ChangeRgbDataset(test_rows, patch_size=args.patch_size)

    train_counts = dict(train_ds.dataset_counter)
    val_counts = dict(val_ds.dataset_counter)
    test_counts = dict(test_ds.dataset_counter)
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

    model = SmallUNet(in_ch=6, base=32).to(device)
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
        eval_threshold, threshold_search = find_best_threshold_visual(
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
        "outdir": str(outdir),
        "device": str(device),
        "rows": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
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
