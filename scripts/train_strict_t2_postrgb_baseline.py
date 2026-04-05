#!/usr/bin/env python3
"""Train a first strict_t2 multi-source post-event RGB baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_baseline_dlr_unet import SmallUNet


DLR_POST_KEYS = ["POST1_B02", "POST1_B03", "POST1_B04"]
DLR_MASK_KEY = "None_MASK"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(path).fillna("")
    return df.to_dict(orient="records")


def read_gdcld_index(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path).fillna("")
    return {row["sample_id"]: row for row in df.to_dict(orient="records")}


def resize_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    y = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    v = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
    y = F.interpolate(y, size=(patch_size, patch_size), mode="nearest")
    v = F.interpolate(v, size=(patch_size, patch_size), mode="nearest")
    return (
        x.squeeze(0).numpy(),
        y.squeeze(0).numpy(),
        v.squeeze(0).numpy(),
    )


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    maxv = float(np.nanmax(image)) if image.size else 0.0
    if maxv > 255.0:
        image = image / 10000.0
    elif maxv > 1.5:
        image = image / 255.0
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


class StrictT2PostRgbDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        patch_size: int,
        gdcld_crop_size: int,
        deterministic_scene_crop: bool,
        gdcld_index: dict[str, dict[str, str]] | None = None,
        gdcld_jitter: int = 64,
    ) -> None:
        self.rows = rows
        self.patch_size = patch_size
        self.gdcld_crop_size = gdcld_crop_size
        self.deterministic_scene_crop = deterministic_scene_crop
        self.gdcld_jitter = gdcld_jitter
        self._h5_cache: dict[Path, h5py.File] = {}
        self._gdcld_index = gdcld_index or {}
        self._gdcld_center_cache: dict[Path, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _get_h5(self, path: Path) -> h5py.File:
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def _load_cas(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = tifffile.imread(row["image_path"])
        label_raw = tifffile.imread(row["label_path"])
        aux_mask = tifffile.imread(row["valid_mask_path"]).astype(np.float32)
        # CAS labels use white background / black landslide polygons.
        # The sidecar `mask/*.tif` matches the landslide pixels, not an ignore mask.
        target = (label_raw.max(axis=-1) <= 127).astype(np.float32)
        aux_target = (aux_mask > 0).astype(np.float32)
        if not np.array_equal(target.astype(np.uint8), aux_target.astype(np.uint8)):
            raise ValueError(f"CAS label/mask mismatch: {row['sample_id']}")
        valid = np.ones_like(target, dtype=np.float32)
        return image, target, valid

    def _load_glad(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        post = tifffile.imread(row["post_path"]).astype(np.float32)
        image = post[..., :3]
        label = tifffile.imread(row["label_path"]).astype(np.float32)
        valid = np.ones_like(label, dtype=np.float32)
        return image, (label > 0.5).astype(np.float32), valid

    def _load_dlr(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h5_path = Path(row["h5_path"])
        idx = int(float(row["h5_sample_index"]))
        f = self._get_h5(h5_path)
        chans = [f[key][idx, 0].astype(np.float32) / 10000.0 for key in DLR_POST_KEYS]
        image = np.stack(chans, axis=-1)
        target = f[DLR_MASK_KEY][idx, 0].astype(np.float32)
        valid = np.ones_like(target, dtype=np.float32)
        return image, target, valid

    def _gdcld_center_from_entry(self, row: dict[str, str], scene_h: int, scene_w: int) -> tuple[int, int]:
        entry = self._gdcld_index.get(row["sample_id"])
        if not entry:
            raise KeyError(row["sample_id"])
        cy = int(entry["center_y"])
        cx = int(entry["center_x"])
        if not self.deterministic_scene_crop:
            jitter = max(0, int(self.gdcld_jitter))
            if jitter > 0:
                cy += random.randint(-jitter, jitter)
                cx += random.randint(-jitter, jitter)
        cy = int(np.clip(cy, 0, max(scene_h - 1, 0)))
        cx = int(np.clip(cx, 0, max(scene_w - 1, 0)))
        return cy, cx

    def _gdcld_center(self, row: dict[str, str], label_path: Path) -> tuple[int, int]:
        entry = self._gdcld_index.get(row["sample_id"])
        if entry:
            return int(entry["center_y"]), int(entry["center_x"])
        if label_path in self._gdcld_center_cache:
            return self._gdcld_center_cache[label_path]
        label_raw = tifffile.imread(label_path)
        pos_bbox = bbox_from_mask(label_raw == 1)
        valid_bbox = bbox_from_mask(label_raw != 3)
        if pos_bbox is not None:
            center = np.array([(pos_bbox[0] + pos_bbox[1]) // 2, (pos_bbox[2] + pos_bbox[3]) // 2])
        elif valid_bbox is not None:
            center = np.array([(valid_bbox[0] + valid_bbox[1]) // 2, (valid_bbox[2] + valid_bbox[3]) // 2])
        else:
            center = np.array(label_raw.shape[:2]) // 2
        out = (int(center[0]), int(center[1]))
        self._gdcld_center_cache[label_path] = out
        return out

    def _load_gdcld(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        with rasterio.open(image_path) as src_img:
            crop = max(64, min(int(self.gdcld_crop_size), src_img.height, src_img.width))
            try:
                cy, cx = self._gdcld_center_from_entry(row, src_img.height, src_img.width)
            except KeyError:
                cy, cx = self._gdcld_center(row, label_path)
            y0 = int(np.clip(cy - crop // 2, 0, max(src_img.height - crop, 0)))
            x0 = int(np.clip(cx - crop // 2, 0, max(src_img.width - crop, 0)))
            window = Window(x0, y0, crop, crop)
            image = src_img.read([1, 2, 3], window=window, boundless=False).transpose(1, 2, 0)
        with rasterio.open(label_path) as src_lbl:
            label_raw = src_lbl.read(1, window=window, boundless=False)
        valid = (label_raw != 3).astype(np.float32)
        target = (label_raw == 1).astype(np.float32)
        return image, target, valid

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sample_kind = row["sample_kind"]
        if sample_kind == "cas_single_rgb":
            image, target, valid = self._load_cas(row)
        elif sample_kind == "glad_pre_post":
            image, target, valid = self._load_glad(row)
        elif sample_kind == "dlr_h5_patch":
            image, target, valid = self._load_dlr(row)
        elif sample_kind == "gdcld_single_rgb":
            image, target, valid = self._load_gdcld(row)
        else:
            raise ValueError(f"unsupported sample_kind={sample_kind}")
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


class CachedPostRgbH5Dataset(Dataset):
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


def parse_named_value_overrides(raw: str) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"expected dataset override as name=value, got: {token}")
        name, value = token.split("=", 1)
        overrides[name.strip()] = float(value.strip())
    return overrides


def parse_threshold_grid(raw: str) -> list[float]:
    vals = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not vals:
        return [0.5]
    clipped = [min(0.95, max(0.05, val)) for val in vals]
    return sorted(set(clipped))


def build_dataset_weight_map(
    dataset_ids: list[str],
    power: float = 1.0,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    dataset_counts = Counter(dataset_ids)
    if not dataset_counts:
        return {}
    overrides = overrides or {}
    weights: dict[str, float] = {}
    for dataset_id, count in dataset_counts.items():
        base = float(count) ** (-float(power)) if power > 0 else 1.0
        weights[dataset_id] = base * float(overrides.get(dataset_id, 1.0))
    mean_weight = sum(weights.values()) / max(len(weights), 1)
    if mean_weight > 0:
        weights = {key: value / mean_weight for key, value in weights.items()}
    return weights


def build_weighted_sampler(
    dataset_ids: list[str],
    epoch_samples: int = 0,
    power: float = 1.0,
    overrides: dict[str, float] | None = None,
) -> WeightedRandomSampler:
    dataset_weights = build_dataset_weight_map(dataset_ids, power=power, overrides=overrides)
    weights = [dataset_weights[dataset_id] for dataset_id in dataset_ids]
    num_samples = int(epoch_samples) if epoch_samples and epoch_samples > 0 else len(dataset_ids)
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def batch_dataset_weights(
    dataset_ids: list[str],
    dataset_loss_weights: dict[str, float] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not dataset_loss_weights:
        return None
    vals = [float(dataset_loss_weights.get(dataset_id, 1.0)) for dataset_id in dataset_ids]
    return torch.tensor(vals, device=device, dtype=dtype)


def masked_bce_loss_per_item(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    pw = torch.tensor(float(max(pos_weight, 1e-6)), device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    denom = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    return (loss * valid).flatten(1).sum(dim=1) / denom


def masked_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    return masked_bce_loss_per_item(logits, target, valid, pos_weight=pos_weight).mean()


def masked_dice_loss_per_item(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits) * valid
    tgt = target * valid
    inter = (prob * tgt).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return masked_dice_loss_per_item(logits, target, valid).mean()


def compute_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample_weights is None:
        bce = masked_bce_loss(logits, target, valid, pos_weight=pos_weight)
        dice = masked_dice_loss(logits, target, valid)
        return bce, dice, bce + dice
    weight_vec = sample_weights / sample_weights.sum().clamp_min(1e-6)
    bce_items = masked_bce_loss_per_item(logits, target, valid, pos_weight=pos_weight)
    dice_items = masked_dice_loss_per_item(logits, target, valid)
    bce = (bce_items * weight_vec).sum()
    dice = (dice_items * weight_vec).sum()
    return bce, dice, bce + dice


def masked_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).float() * valid
    tgt = (target >= 0.5).float() * valid
    tp = float((pred * tgt).sum().item())
    fp = float((pred * (1.0 - tgt)).sum().item())
    fn = float((((1.0 - pred) * tgt) * valid).sum().item())
    iou = tp / (tp + fp + fn + 1e-7)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


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


def run_epoch(
    model: nn.Module,
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
    by_dataset: dict[str, dict[str, float]] = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n": 0.0})

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


def find_best_threshold_visual(
    model: nn.Module,
    loader: DataLoader,
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
            probs = torch.sigmoid(model(x))
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


def subset_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    out: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        ds = row["dataset_id"]
        if counts[ds] * len(Counter(r["dataset_id"] for r in rows)) >= limit:
            continue
        out.append(row)
        counts[ds] += 1
        if len(out) >= limit:
            break
    return out


def default_eval_cache_path(root: Path, split: str, patch_size: int) -> Path:
    return root / "processed" / "hybrid_pinn" / "strict_t2_postrgb_eval_cache_v2" / f"{split}_postrgb_p{patch_size}.h5"


def default_train_cache_path(root: Path, patch_size: int) -> Path:
    return root / "processed" / "hybrid_pinn" / "strict_t2_postrgb_eval_cache_v2" / f"train_postrgb_p{patch_size}.h5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train first strict_t2 post_rgb multi-source baseline")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument(
        "--manifest",
        default="",
        help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_manifest_post_rgb.csv",
    )
    p.add_argument("--outdir", default="", help="default: experiments/strict_t2_postrgb_baseline_stage0")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--gdcld-crop-size", type=int, default=1024)
    p.add_argument("--gdcld-jitter", type=int, default=64)
    p.add_argument("--gdcld-index", default="", help="optional precomputed GDCLD scene index csv")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epoch-samples", type=int, default=0, help="train samples drawn per epoch after dataset balancing")
    p.add_argument("--sampler-power", type=float, default=1.0, help="dataset reweight power for sampler: 0=no rebalance, 1=inverse-count")
    p.add_argument("--sampler-overrides", default="", help="optional dataset multipliers, e.g. DLR=1.5,CAS=0.7")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-pos-weight", type=float, default=1.0)
    p.add_argument("--loss-balance-power", type=float, default=0.0, help="dataset reweight power applied to train loss")
    p.add_argument("--loss-balance-overrides", default="", help="optional dataset multipliers applied to train loss")
    p.add_argument("--seed", type=int, default=20260309)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "auto"])
    p.add_argument("--train-cache-h5", default="", help="optional cached HDF5 for train split")
    p.add_argument("--val-cache-h5", default="", help="optional cached HDF5 for val split")
    p.add_argument("--test-cache-h5", default="", help="optional cached HDF5 for test split")
    p.add_argument("--exclude-datasets", default="", help="comma-separated dataset_id values to drop")
    p.add_argument("--train-limit", type=int, default=0, help="optional row limit for train split")
    p.add_argument("--val-limit", type=int, default=0, help="optional row limit for val split")
    p.add_argument("--test-limit", type=int, default=0, help="optional row limit for test split")
    p.add_argument("--tune-threshold-on-val", action="store_true", help="sweep thresholds on val after model selection")
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
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_manifest_post_rgb.csv"
    )
    outdir = Path(args.outdir) if args.outdir.strip() else root / "experiments" / "strict_t2_postrgb_baseline_stage0"
    outdir.mkdir(parents=True, exist_ok=True)
    gdcld_index_path = (
        Path(args.gdcld_index)
        if args.gdcld_index.strip()
        else root / "metadata" / "manifests" / "gdcld_postrgb_scene_index_v1.csv"
    )
    train_cache_h5 = Path(args.train_cache_h5) if args.train_cache_h5.strip() else default_train_cache_path(root, args.patch_size)
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

    device = torch.device("cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"[info] device={device}")
    print(f"[info] manifest={manifest}")
    print(f"[info] gdcld_index={gdcld_index_path}")
    print(f"[info] train_cache_h5={train_cache_h5 if train_cache_h5.exists() else 'missing'}")
    print(f"[info] val_cache_h5={val_cache_h5 if val_cache_h5.exists() else 'missing'}")
    print(f"[info] test_cache_h5={test_cache_h5 if test_cache_h5.exists() else 'missing'}")
    print(f"[info] rows train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    gdcld_index = read_gdcld_index(gdcld_index_path)

    if train_cache_h5.exists():
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
    print(f"[info] effective train datasets={train_counts}")
    sampler_weight_map = build_dataset_weight_map(train_dataset_ids, power=args.sampler_power, overrides=sampler_overrides)
    loss_weight_map = build_dataset_weight_map(
        train_dataset_ids,
        power=args.loss_balance_power,
        overrides=loss_balance_overrides,
    )
    train_loss_weights = None
    if args.loss_balance_power > 0 or loss_balance_overrides:
        train_loss_weights = loss_weight_map
    print(f"[info] sampler_weight_map={json.dumps(sampler_weight_map, ensure_ascii=False)}")
    if train_loss_weights:
        print(f"[info] train_loss_weight_map={json.dumps(train_loss_weights, ensure_ascii=False)}")
    if val_cache_h5.exists():
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
    if test_cache_h5.exists():
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
    )
    print(f"[info] effective val datasets={val_counts}")
    print(f"[info] effective test datasets={test_counts}")
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

    model = SmallUNet(in_ch=3, base=32).to(device)
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
