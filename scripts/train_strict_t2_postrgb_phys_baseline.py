#!/usr/bin/env python3
"""Train a first strict_t2 post_rgb baseline with numeric physics conditioning."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_baseline_dlr_unet import ConvBlock
from train_strict_t2_postrgb_baseline import (
    CachedPostRgbH5Dataset,
    StrictT2PostRgbDataset,
    batch_dataset_weights,
    build_weighted_sampler,
    build_dataset_weight_map,
    compute_segmentation_loss,
    default_eval_cache_path,
    default_train_cache_path,
    masked_stats,
    parse_named_value_overrides,
    parse_threshold_grid,
    find_best_threshold_visual,
    read_gdcld_index,
    read_manifest,
    set_seed,
    subset_rows,
)


PHYSICS_GROUPS = {
    "terrain": [f"terrain_{idx}" for idx in range(4)],
    "material": [f"material_{idx}" for idx in range(9)],
    "trigger": [f"trigger_{idx}" for idx in range(9)] + [f"trigger_ext_{idx}" for idx in range(2)],
    "proxy": ["hydro_proxy", "stability_proxy"],
}
PHYSICS_VECTOR_COLUMNS = PHYSICS_GROUPS["terrain"] + PHYSICS_GROUPS["material"] + PHYSICS_GROUPS["trigger"] + PHYSICS_GROUPS["proxy"]


def resolve_physics_vector_cols(groups: str) -> list[str]:
    tokens = [item.strip() for item in groups.split(",") if item.strip()]
    if not tokens or tokens == ["full"]:
        return list(PHYSICS_VECTOR_COLUMNS)
    cols: list[str] = []
    for token in tokens:
        if token not in PHYSICS_GROUPS:
            raise ValueError(f"unsupported physics group: {token}")
        cols.extend(PHYSICS_GROUPS[token])
    return cols


def load_physics_maps(csv_path: Path, vector_cols: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [col for col in vector_cols if col not in df.columns]
    if missing:
        raise KeyError(f"physics csv missing columns: {missing}")
    df = df.fillna(0.0)
    sample_map: dict[str, np.ndarray] = {}
    event_map: dict[str, np.ndarray] = {}
    for row in df.to_dict(orient="records"):
        vec = np.asarray([float(row[col]) for col in vector_cols], dtype=np.float32)
        sample_map[str(row["sample_id"])] = vec
        event_map.setdefault(str(row["event_uid"]), vec)
    return sample_map, event_map


def metadata_from_base_dataset(ds: Dataset) -> tuple[list[str], list[str]]:
    if isinstance(ds, CachedPostRgbH5Dataset):
        with h5py.File(ds.h5_path, "r") as f:
            sample_ids = [f["sample_id"][idx].decode("utf-8") for idx in ds.indices]
            event_uids = [f["event_uid"][idx].decode("utf-8") for idx in ds.indices]
        return sample_ids, event_uids
    if isinstance(ds, StrictT2PostRgbDataset):
        return [row["sample_id"] for row in ds.rows], [row["event_uid"] for row in ds.rows]
    raise TypeError(f"unsupported dataset type: {type(ds)}")


def gather_train_stats(
    sample_ids: list[str],
    event_uids: list[str],
    sample_map: dict[str, np.ndarray],
    event_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    sample_dim = len(next(iter(sample_map.values()))) if sample_map else len(next(iter(event_map.values())))
    zero = np.zeros((sample_dim,), dtype=np.float32)
    for sample_id, event_uid in zip(sample_ids, event_uids, strict=True):
        vectors.append(sample_map.get(sample_id, event_map.get(event_uid, zero)))
    mat = np.stack(vectors, axis=0) if vectors else np.zeros((1, len(PHYSICS_VECTOR_COLUMNS)), dtype=np.float32)
    mean = mat.mean(axis=0).astype(np.float32)
    std = mat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


class PhysicsAugmentedDataset(Dataset):
    def __init__(
        self,
        base_ds: Dataset,
        sample_map: dict[str, np.ndarray],
        event_map: dict[str, np.ndarray],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.base_ds = base_ds
        self.sample_map = sample_map
        self.event_map = event_map
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.zero = np.zeros_like(self.mean, dtype=np.float32)
        self.dataset_ids = list(getattr(base_ds, "dataset_ids", []))
        self.dataset_counter = getattr(base_ds, "dataset_counter", Counter())

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        item = self.base_ds[idx]
        sample_id = item["sample_id"]
        event_uid = item["event_uid"]
        vec = self.sample_map.get(sample_id, self.event_map.get(event_uid, self.zero))
        vec = np.nan_to_num((vec - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        item["physics"] = torch.from_numpy(vec)
        return item


class PhysicsFiLMUNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        physics_dim: int,
        base: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        physics_cols: list[str] | None = None,
        dynamic_gate_mode: str = "none",
        dynamic_gate_source: str = "proxy",
        dynamic_gate_scale: float = 1.0,
    ):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

        bottleneck_ch = base * 8
        self.physics_film = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_ch * 2),
        )
        self.physics_bias = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dynamic_gate_mode = dynamic_gate_mode
        self.dynamic_gate_scale = float(dynamic_gate_scale)
        cols = physics_cols or []
        static_idx = [idx for idx, col in enumerate(cols) if col.startswith("terrain_") or col.startswith("material_")]
        dynamic_idx = [
            idx
            for idx, col in enumerate(cols)
            if col.startswith("trigger_") or col.startswith("trigger_ext_") or col in {"hydro_proxy", "stability_proxy"}
        ]
        proxy_idx = [idx for idx, col in enumerate(cols) if col in {"hydro_proxy", "stability_proxy"}]
        gate_input_idx = proxy_idx if dynamic_gate_source == "proxy" and proxy_idx else dynamic_idx
        self.use_dynamic_gate = (
            dynamic_gate_mode != "none" and len(static_idx) > 0 and len(dynamic_idx) > 0 and len(gate_input_idx) > 0
        )
        if self.use_dynamic_gate:
            static_dim = len(static_idx)
            dynamic_dim = len(dynamic_idx)
            gate_dim = len(gate_input_idx)
            self.register_buffer("static_idx", torch.tensor(static_idx, dtype=torch.long), persistent=False)
            self.register_buffer("dynamic_idx", torch.tensor(dynamic_idx, dtype=torch.long), persistent=False)
            self.register_buffer("gate_idx", torch.tensor(gate_input_idx, dtype=torch.long), persistent=False)
            self.static_film = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, bottleneck_ch * 2),
            )
            self.dynamic_film = nn.Sequential(
                nn.Linear(dynamic_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, bottleneck_ch * 2),
            )
            self.static_bias = nn.Sequential(
                nn.Linear(static_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.dynamic_bias = nn.Sequential(
                nn.Linear(dynamic_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            )
            gate_hidden = max(8, hidden_dim // 4)
            self.dynamic_gate = nn.Sequential(
                nn.Linear(gate_dim, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.register_buffer("static_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("dynamic_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("gate_idx", torch.empty(0, dtype=torch.long), persistent=False)

    def forward(self, image: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        if self.use_dynamic_gate:
            static_vec = physics.index_select(1, self.static_idx)
            dynamic_vec = physics.index_select(1, self.dynamic_idx)
            gate_input = physics.index_select(1, self.gate_idx)
            gate = self.dynamic_gate(gate_input)
            gamma_beta = self.static_film(static_vec) + self.dynamic_gate_scale * gate * self.dynamic_film(dynamic_vec)
            bias = self.static_bias(static_vec) + self.dynamic_gate_scale * gate * self.dynamic_bias(dynamic_vec)
        else:
            gamma_beta = self.physics_film(physics)
            bias = self.physics_bias(physics)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        b = b * (1.0 + 0.1 * gamma) + beta

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        logits = logits + bias.unsqueeze(-1).unsqueeze(-1)
        return logits


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
            image = batch["image"].to(device, non_blocking=False)
            physics = batch["physics"].to(device, non_blocking=False)
            mask = batch["mask"].to(device, non_blocking=False)
            valid = batch["valid"].to(device, non_blocking=False)
            logits = model(image, physics)
            sample_weights = batch_dataset_weights(
                batch["dataset_id"],
                dataset_loss_weights=dataset_loss_weights if train else None,
                device=logits.device,
                dtype=logits.dtype,
            )
            bce, dice, loss = compute_segmentation_loss(
                logits,
                mask,
                valid,
                pos_weight=bce_pos_weight,
                sample_weights=sample_weights,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            logits_detached = logits.detach()
            stats = masked_stats(logits_detached, mask, valid, threshold=threshold)
            bsz = image.size(0)
            total_loss += float(loss.item()) * bsz
            total_bce += float(bce.item()) * bsz
            total_dice += float(dice.item()) * bsz
            total_items += bsz
            agg["tp"] += stats["tp"]
            agg["fp"] += stats["fp"]
            agg["fn"] += stats["fn"]
            for i, ds_name in enumerate(batch["dataset_id"]):
                item_stats = masked_stats(
                    logits_detached[i : i + 1],
                    mask[i : i + 1],
                    valid[i : i + 1],
                    threshold=threshold,
                )
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


def find_best_threshold_physics(
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
            image = batch["image"].to(device, non_blocking=False)
            physics = batch["physics"].to(device, non_blocking=False)
            mask = batch["mask"].to(device, non_blocking=False)
            valid = batch["valid"].to(device, non_blocking=False)
            probs = torch.sigmoid(model(image, physics))
            tgt = (mask >= 0.5).float() * valid
            for thr in thresholds:
                pred = (probs >= thr).float() * valid
                aggs[thr]["tp"] += float((pred * tgt).sum().item())
                aggs[thr]["fp"] += float((pred * (1.0 - tgt)).sum().item())
                aggs[thr]["fn"] += float((((1.0 - pred) * tgt) * valid).sum().item())
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strict_t2 post_rgb visual + numeric physics baseline")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--manifest", default="", help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_manifest_post_rgb.csv")
    p.add_argument("--physics-csv", default="", help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_physics_vectors_post_rgb_v1.csv")
    p.add_argument("--outdir", default="", help="default: experiments/strict_t2_postrgb_phys_baseline_stage0")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--gdcld-crop-size", type=int, default=512)
    p.add_argument("--gdcld-jitter", type=int, default=64)
    p.add_argument("--gdcld-index", default="", help="optional precomputed GDCLD scene index csv")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epoch-samples", type=int, default=0)
    p.add_argument("--sampler-power", type=float, default=1.0)
    p.add_argument("--sampler-overrides", default="")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-pos-weight", type=float, default=6.0)
    p.add_argument("--loss-balance-power", type=float, default=0.0)
    p.add_argument("--loss-balance-overrides", default="")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--physics-groups", default="full", help="comma-separated: terrain,material,trigger,proxy or full")
    p.add_argument("--dynamic-gate-mode", default="none", choices=["none", "proxy"])
    p.add_argument("--dynamic-gate-source", default="proxy", choices=["proxy", "dynamic"])
    p.add_argument("--dynamic-gate-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260309)
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
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_manifest_post_rgb.csv"
    )
    physics_csv = (
        Path(args.physics_csv)
        if args.physics_csv.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_physics_vectors_post_rgb_v1.csv"
    )
    outdir = Path(args.outdir) if args.outdir.strip() else root / "experiments" / "strict_t2_postrgb_phys_baseline_stage0"
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
    vector_cols = resolve_physics_vector_cols(args.physics_groups)
    print(f"[info] physics_csv={physics_csv}")
    print(f"[info] physics_groups={args.physics_groups}")
    print(f"[info] gdcld_index={gdcld_index_path}")
    print(f"[info] train_cache_h5={train_cache_h5 if train_cache_h5.exists() else 'missing'}")
    print(f"[info] val_cache_h5={val_cache_h5 if val_cache_h5.exists() else 'missing'}")
    print(f"[info] test_cache_h5={test_cache_h5 if test_cache_h5.exists() else 'missing'}")
    print(f"[info] rows train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    gdcld_index = read_gdcld_index(gdcld_index_path)
    if train_cache_h5.exists():
        train_base = CachedPostRgbH5Dataset(train_cache_h5, exclude_datasets=exclude_datasets)
        train_dataset_ids = list(train_base.dataset_ids)
        train_counts = dict(train_base.dataset_counter)
    else:
        train_base = StrictT2PostRgbDataset(
            train_rows,
            args.patch_size,
            args.gdcld_crop_size,
            deterministic_scene_crop=False,
            gdcld_index=gdcld_index,
            gdcld_jitter=args.gdcld_jitter,
        )
        train_dataset_ids = [row["dataset_id"] for row in train_rows]
        train_counts = dict(Counter(train_dataset_ids))

    if val_cache_h5.exists():
        val_base = CachedPostRgbH5Dataset(val_cache_h5, exclude_datasets=exclude_datasets)
        val_counts = dict(val_base.dataset_counter)
    else:
        val_base = StrictT2PostRgbDataset(
            val_rows,
            args.patch_size,
            args.gdcld_crop_size,
            deterministic_scene_crop=True,
            gdcld_index=gdcld_index,
            gdcld_jitter=0,
        )
        val_counts = dict(Counter(row["dataset_id"] for row in val_rows))

    if test_cache_h5.exists():
        test_base = CachedPostRgbH5Dataset(test_cache_h5, exclude_datasets=exclude_datasets)
        test_counts = dict(test_base.dataset_counter)
    else:
        test_base = StrictT2PostRgbDataset(
            test_rows,
            args.patch_size,
            args.gdcld_crop_size,
            deterministic_scene_crop=True,
            gdcld_index=gdcld_index,
            gdcld_jitter=0,
        )
        test_counts = dict(Counter(row["dataset_id"] for row in test_rows))

    sample_map, event_map = load_physics_maps(physics_csv, vector_cols)
    train_sample_ids, train_event_uids = metadata_from_base_dataset(train_base)
    mean, std = gather_train_stats(train_sample_ids, train_event_uids, sample_map, event_map)

    train_ds = PhysicsAugmentedDataset(train_base, sample_map, event_map, mean, std)
    val_ds = PhysicsAugmentedDataset(val_base, sample_map, event_map, mean, std)
    test_ds = PhysicsAugmentedDataset(test_base, sample_map, event_map, mean, std)

    print(f"[info] effective train datasets={train_counts}")
    print(f"[info] effective val datasets={val_counts}")
    print(f"[info] effective test datasets={test_counts}")
    print(f"[info] physics_dim={len(vector_cols)}")
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
        in_ch=3,
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
        "physics_norm": {
            "mean": mean.tolist(),
            "std": std.tolist(),
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
