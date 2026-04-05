#!/usr/bin/env python3
"""Stage-0 baseline training on DLR reference_data (U-Net, binary segmentation)."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


FEATURE_KEYS = [
    "PRE1_B02",
    "PRE1_B03",
    "PRE1_B04",
    "PRE1_B08",
    "POST1_B02",
    "POST1_B03",
    "POST1_B04",
    "POST1_B08",
    "None_DEM",
    "None_SLOPE",
]
MASK_KEY = "None_MASK"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EpochStat:
    epoch: int
    train_loss: float
    train_bce_loss: float
    train_topo_loss: float
    train_phys_loss: float
    train_scale_loss: float
    val_loss: float
    val_bce_loss: float
    val_topo_loss: float
    val_phys_loss: float
    val_scale_loss: float
    val_iou: float
    val_f1: float
    lr: float
    sec: float


def _scale_by_key(x: np.ndarray, key: str) -> np.ndarray:
    if key.startswith("PRE1_") or key.startswith("POST1_"):
        if any(b in key for b in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]):
            return x / 10000.0
        return x
    if key == "None_DEM":
        return x / 3000.0
    if key == "None_SLOPE":
        return x / 90.0
    return x


class DLRH5Dataset(Dataset):
    def __init__(self, h5_path: Path, feature_keys: list[str], mask_key: str = MASK_KEY):
        self.h5_path = Path(h5_path)
        self.feature_keys = feature_keys
        self.mask_key = mask_key
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            self.length = int(f[self.mask_key].shape[0])

    def __len__(self) -> int:
        return self.length

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        f = self._get_h5()
        xs = []
        for k in self.feature_keys:
            a = f[k][idx, 0].astype(np.float32)
            a = _scale_by_key(a, k)
            xs.append(a)
        x = np.stack(xs, axis=0)
        y = f[self.mask_key][idx, 0].astype(np.float32)
        y = np.expand_dims(y, axis=0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x), torch.from_numpy(y)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
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

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5):
    pred = (torch.sigmoid(logits) >= thresh).float()
    tgt = (target >= 0.5).float()
    tp = (pred * tgt).sum().item()
    fp = (pred * (1.0 - tgt)).sum().item()
    fn = ((1.0 - pred) * tgt).sum().item()
    iou = tp / (tp + fp + fn + 1e-7)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return tp, fp, fn, iou, f1


def compute_topo_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    slope_idx: int,
    slope_low_thresh_norm: float,
    slope_temp_norm: float,
) -> torch.Tensor:
    """
    Penalize high landslide probability in low-slope areas.

    x is normalized feature tensor where slope is expected in [0, 1].
    """
    prob = torch.sigmoid(logits)
    slope = x[:, slope_idx : slope_idx + 1]
    # Weight is close to 1 when slope < threshold, close to 0 when slope is high.
    low_slope_w = torch.sigmoid((slope_low_thresh_norm - slope) / max(slope_temp_norm, 1e-6))
    return (prob * low_slope_w).mean()


def compute_phys_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    pre_red_idx: int,
    pre_nir_idx: int,
    post_red_idx: int,
    post_nir_idx: int,
    ndvi_drop_thresh: float = 0.05,
    ndvi_temp: float = 0.03,
) -> torch.Tensor:
    """
    Penalize high landslide probability where NDVI drop is not evident.

    For landslide pixels, a vegetation decrease (post-pre NDVI < 0) is often expected.
    """
    eps = 1e-6
    pre_red = x[:, pre_red_idx : pre_red_idx + 1]
    pre_nir = x[:, pre_nir_idx : pre_nir_idx + 1]
    post_red = x[:, post_red_idx : post_red_idx + 1]
    post_nir = x[:, post_nir_idx : post_nir_idx + 1]

    pre_ndvi = (pre_nir - pre_red) / (pre_nir + pre_red + eps)
    post_ndvi = (post_nir - post_red) / (post_nir + post_red + eps)
    ndvi_drop = pre_ndvi - post_ndvi

    # Large when drop is insufficient; small when drop is large enough.
    inconsistent_w = torch.sigmoid((ndvi_drop_thresh - ndvi_drop) / max(ndvi_temp, 1e-6))
    prob = torch.sigmoid(logits)
    return (prob * inconsistent_w).mean()


def compute_scale_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    huber_delta: float = 0.02,
) -> torch.Tensor:
    """
    Penalize mismatch between predicted and target landslide area ratio.

    This is a scale-aware regularization on per-sample positive-area statistics.
    """
    prob = torch.sigmoid(logits)
    pred_ratio = prob.mean(dim=(2, 3))  # (B,1)
    tgt_ratio = target.mean(dim=(2, 3))  # (B,1)
    diff = pred_ratio - tgt_ratio
    abs_diff = diff.abs()
    delta = max(float(huber_delta), 1e-6)
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
    lin = abs_diff - quad
    # Smooth-L1 style: 0.5*x^2/delta + (|x|-delta)
    loss = 0.5 * (quad**2) / delta + lin
    return loss.mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    lambda_topo: float = 0.0,
    lambda_phys: float = 0.0,
    lambda_scale: float = 0.0,
    slope_idx: int | None = None,
    slope_low_thresh_norm: float = 10.0 / 90.0,
    slope_temp_norm: float = 2.0 / 90.0,
    pre_red_idx: int | None = None,
    pre_nir_idx: int | None = None,
    post_red_idx: int | None = None,
    post_nir_idx: int | None = None,
    ndvi_drop_thresh: float = 0.05,
    ndvi_temp: float = 0.03,
    scale_huber_delta: float = 0.02,
):
    train = optimizer is not None
    model.train(train)

    loss_sum = 0.0
    bce_sum = 0.0
    topo_sum = 0.0
    phys_sum = 0.0
    scale_sum = 0.0
    n = 0
    t_tp = t_fp = t_fn = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        bce_loss = criterion(logits, y)
        topo_loss = torch.tensor(0.0, device=device)
        phys_loss = torch.tensor(0.0, device=device)
        scale_loss = torch.tensor(0.0, device=device)
        if lambda_topo > 0.0:
            if slope_idx is None:
                raise ValueError("slope_idx is required when lambda_topo > 0")
            topo_loss = compute_topo_loss(
                logits=logits,
                x=x,
                slope_idx=slope_idx,
                slope_low_thresh_norm=slope_low_thresh_norm,
                slope_temp_norm=slope_temp_norm,
            )
        if lambda_phys > 0.0:
            if None in (pre_red_idx, pre_nir_idx, post_red_idx, post_nir_idx):
                raise ValueError("pre/post red/nir indices are required when lambda_phys > 0")
            phys_loss = compute_phys_loss(
                logits=logits,
                x=x,
                pre_red_idx=pre_red_idx,
                pre_nir_idx=pre_nir_idx,
                post_red_idx=post_red_idx,
                post_nir_idx=post_nir_idx,
                ndvi_drop_thresh=ndvi_drop_thresh,
                ndvi_temp=ndvi_temp,
            )
        if lambda_scale > 0.0:
            scale_loss = compute_scale_loss(
                logits=logits,
                target=y,
                huber_delta=scale_huber_delta,
            )
        loss = bce_loss + lambda_topo * topo_loss + lambda_phys * phys_loss + lambda_scale * scale_loss
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bsz = x.size(0)
        loss_sum += float(loss.item()) * bsz
        bce_sum += float(bce_loss.item()) * bsz
        topo_sum += float(topo_loss.item()) * bsz
        phys_sum += float(phys_loss.item()) * bsz
        scale_sum += float(scale_loss.item()) * bsz
        n += bsz
        tp, fp, fn, _, _ = compute_metrics(logits, y)
        t_tp += tp
        t_fp += fp
        t_fn += fn

    avg_loss = loss_sum / max(n, 1)
    avg_bce = bce_sum / max(n, 1)
    avg_topo = topo_sum / max(n, 1)
    avg_phys = phys_sum / max(n, 1)
    avg_scale = scale_sum / max(n, 1)
    iou = t_tp / (t_tp + t_fp + t_fn + 1e-7)
    f1 = (2.0 * t_tp) / (2.0 * t_tp + t_fp + t_fn + 1e-7)
    return avg_loss, iou, f1, avg_bce, avg_topo, avg_phys, avg_scale


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lambda_topo: float = 0.0,
    lambda_phys: float = 0.0,
    lambda_scale: float = 0.0,
    slope_idx: int | None = None,
    slope_low_thresh_norm: float = 10.0 / 90.0,
    slope_temp_norm: float = 2.0 / 90.0,
    pre_red_idx: int | None = None,
    pre_nir_idx: int | None = None,
    post_red_idx: int | None = None,
    post_nir_idx: int | None = None,
    ndvi_drop_thresh: float = 0.05,
    ndvi_temp: float = 0.03,
    scale_huber_delta: float = 0.02,
):
    model.eval()
    with torch.no_grad():
        return run_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            lambda_topo=lambda_topo,
            lambda_phys=lambda_phys,
            lambda_scale=lambda_scale,
            slope_idx=slope_idx,
            slope_low_thresh_norm=slope_low_thresh_norm,
            slope_temp_norm=slope_temp_norm,
            pre_red_idx=pre_red_idx,
            pre_nir_idx=pre_nir_idx,
            post_red_idx=post_red_idx,
            post_nir_idx=post_nir_idx,
            ndvi_drop_thresh=ndvi_drop_thresh,
            ndvi_temp=ndvi_temp,
            scale_huber_delta=scale_huber_delta,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="PILD root",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260305)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--lambda-topo", type=float, default=0.0, help="weight for topographic regularization")
    p.add_argument("--lambda-phys", type=float, default=0.0, help="weight for NDVI-based physical regularization")
    p.add_argument("--lambda-scale", type=float, default=0.0, help="weight for area-scale regularization")
    p.add_argument("--slope-low-thresh-deg", type=float, default=10.0, help="low-slope threshold in degrees")
    p.add_argument("--slope-temp-deg", type=float, default=2.0, help="transition temperature in degrees")
    p.add_argument("--ndvi-drop-thresh", type=float, default=0.05, help="expected minimum NDVI drop for landslide")
    p.add_argument("--ndvi-temp", type=float, default=0.03, help="smoothness for NDVI consistency weighting")
    p.add_argument("--scale-huber-delta", type=float, default=0.02, help="delta for area-ratio huber regularization")
    p.add_argument("--outdir", default="", help="output dir; default is experiments/dlr_baseline_stage0")
    p.add_argument(
        "--dlr-ref-dir",
        default="",
        help="optional override path containing train_n3_s1s2.h5/val_n3_s1s2.h5/testind_n3_s1s2.h5/testspt_n3_s1s2.h5",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)

    root = Path(args.root)
    if args.dlr_ref_dir.strip():
        ref = Path(args.dlr_ref_dir)
    else:
        ref = (
            root
            / "raw"
            / "datasets"
            / "05_DLR_Landslide_Ref_2025"
            / "extracted"
            / "s1s2_landslide_reference_data"
            / "s1s2_landslide_reference_data"
            / "reference_data"
        )
    train_h5 = ref / "train_n3_s1s2.h5"
    val_h5 = ref / "val_n3_s1s2.h5"
    testind_h5 = ref / "testind_n3_s1s2.h5"
    testspt_h5 = ref / "testspt_n3_s1s2.h5"
    for p in [train_h5, val_h5, testind_h5, testspt_h5]:
        if not p.exists():
            raise FileNotFoundError(p)

    if args.outdir.strip():
        out_dir = Path(args.outdir)
    else:
        out_dir = root / "experiments" / "dlr_baseline_stage0"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        print(f"[info] gpu={torch.cuda.get_device_name(0)}")

    train_ds = DLRH5Dataset(train_h5, FEATURE_KEYS)
    val_ds = DLRH5Dataset(val_h5, FEATURE_KEYS)
    testind_ds = DLRH5Dataset(testind_h5, FEATURE_KEYS)
    testspt_ds = DLRH5Dataset(testspt_h5, FEATURE_KEYS)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
    testind_loader = DataLoader(
        testind_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    testspt_loader = DataLoader(
        testspt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"[info] samples train/val/testind/testspt="
        f"{len(train_ds)}/{len(val_ds)}/{len(testind_ds)}/{len(testspt_ds)}"
    )

    model = SmallUNet(in_ch=len(FEATURE_KEYS), base=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    slope_idx = FEATURE_KEYS.index("None_SLOPE")
    slope_low_thresh_norm = args.slope_low_thresh_deg / 90.0
    slope_temp_norm = args.slope_temp_deg / 90.0
    pre_red_idx = FEATURE_KEYS.index("PRE1_B04")
    pre_nir_idx = FEATURE_KEYS.index("PRE1_B08")
    post_red_idx = FEATURE_KEYS.index("POST1_B04")
    post_nir_idx = FEATURE_KEYS.index("POST1_B08")

    best_iou = -1.0
    best_epoch = -1
    history: list[EpochStat] = []
    ckpt_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_iou, tr_f1, tr_bce, tr_topo, tr_phys, tr_scale = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            lambda_topo=args.lambda_topo,
            lambda_phys=args.lambda_phys,
            lambda_scale=args.lambda_scale,
            slope_idx=slope_idx,
            slope_low_thresh_norm=slope_low_thresh_norm,
            slope_temp_norm=slope_temp_norm,
            pre_red_idx=pre_red_idx,
            pre_nir_idx=pre_nir_idx,
            post_red_idx=post_red_idx,
            post_nir_idx=post_nir_idx,
            ndvi_drop_thresh=args.ndvi_drop_thresh,
            ndvi_temp=args.ndvi_temp,
            scale_huber_delta=args.scale_huber_delta,
        )
        val_loss, val_iou, val_f1, val_bce, val_topo, val_phys, val_scale = evaluate_split(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            lambda_topo=args.lambda_topo,
            lambda_phys=args.lambda_phys,
            lambda_scale=args.lambda_scale,
            slope_idx=slope_idx,
            slope_low_thresh_norm=slope_low_thresh_norm,
            slope_temp_norm=slope_temp_norm,
            pre_red_idx=pre_red_idx,
            pre_nir_idx=pre_nir_idx,
            post_red_idx=post_red_idx,
            post_nir_idx=post_nir_idx,
            ndvi_drop_thresh=args.ndvi_drop_thresh,
            ndvi_temp=args.ndvi_temp,
            scale_huber_delta=args.scale_huber_delta,
        )
        scheduler.step()
        sec = time.time() - t0

        cur = EpochStat(
            epoch=epoch,
            train_loss=tr_loss,
            train_bce_loss=tr_bce,
            train_topo_loss=tr_topo,
            train_phys_loss=tr_phys,
            train_scale_loss=tr_scale,
            val_loss=val_loss,
            val_bce_loss=val_bce,
            val_topo_loss=val_topo,
            val_phys_loss=val_phys,
            val_scale_loss=val_scale,
            val_iou=val_iou,
            val_f1=val_f1,
            lr=float(optimizer.param_groups[0]["lr"]),
            sec=sec,
        )
        history.append(cur)
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={tr_loss:.4f} (bce={tr_bce:.4f},topo={tr_topo:.4f},phys={tr_phys:.4f},scale={tr_scale:.4f}) "
            f"train_iou={tr_iou:.4f} train_f1={tr_f1:.4f} "
            f"val_loss={val_loss:.4f} (bce={val_bce:.4f},topo={val_topo:.4f},phys={val_phys:.4f},scale={val_scale:.4f}) "
            f"val_iou={val_iou:.4f} val_f1={val_f1:.4f} sec={sec:.1f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "feature_keys": FEATURE_KEYS, "epoch": epoch}, ckpt_path)

    # load best and evaluate on tests
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    _, val_iou, val_f1, val_bce, val_topo, val_phys, val_scale = evaluate_split(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        lambda_topo=args.lambda_topo,
        lambda_phys=args.lambda_phys,
        lambda_scale=args.lambda_scale,
        slope_idx=slope_idx,
        slope_low_thresh_norm=slope_low_thresh_norm,
        slope_temp_norm=slope_temp_norm,
        pre_red_idx=pre_red_idx,
        pre_nir_idx=pre_nir_idx,
        post_red_idx=post_red_idx,
        post_nir_idx=post_nir_idx,
        ndvi_drop_thresh=args.ndvi_drop_thresh,
        ndvi_temp=args.ndvi_temp,
        scale_huber_delta=args.scale_huber_delta,
    )
    _, tind_iou, tind_f1, tind_bce, tind_topo, tind_phys, tind_scale = evaluate_split(
        model=model,
        loader=testind_loader,
        criterion=criterion,
        device=device,
        lambda_topo=args.lambda_topo,
        lambda_phys=args.lambda_phys,
        lambda_scale=args.lambda_scale,
        slope_idx=slope_idx,
        slope_low_thresh_norm=slope_low_thresh_norm,
        slope_temp_norm=slope_temp_norm,
        pre_red_idx=pre_red_idx,
        pre_nir_idx=pre_nir_idx,
        post_red_idx=post_red_idx,
        post_nir_idx=post_nir_idx,
        ndvi_drop_thresh=args.ndvi_drop_thresh,
        ndvi_temp=args.ndvi_temp,
        scale_huber_delta=args.scale_huber_delta,
    )
    _, tspt_iou, tspt_f1, tspt_bce, tspt_topo, tspt_phys, tspt_scale = evaluate_split(
        model=model,
        loader=testspt_loader,
        criterion=criterion,
        device=device,
        lambda_topo=args.lambda_topo,
        lambda_phys=args.lambda_phys,
        lambda_scale=args.lambda_scale,
        slope_idx=slope_idx,
        slope_low_thresh_norm=slope_low_thresh_norm,
        slope_temp_norm=slope_temp_norm,
        pre_red_idx=pre_red_idx,
        pre_nir_idx=pre_nir_idx,
        post_red_idx=post_red_idx,
        post_nir_idx=post_nir_idx,
        ndvi_drop_thresh=args.ndvi_drop_thresh,
        ndvi_temp=args.ndvi_temp,
        scale_huber_delta=args.scale_huber_delta,
    )

    result = {
        "timestamp": int(time.time()),
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_topo": args.lambda_topo,
        "lambda_phys": args.lambda_phys,
        "lambda_scale": args.lambda_scale,
        "slope_low_thresh_deg": args.slope_low_thresh_deg,
        "slope_temp_deg": args.slope_temp_deg,
        "ndvi_drop_thresh": args.ndvi_drop_thresh,
        "ndvi_temp": args.ndvi_temp,
        "scale_huber_delta": args.scale_huber_delta,
        "feature_keys": FEATURE_KEYS,
        "samples": {
            "train": len(train_ds),
            "val": len(val_ds),
            "testind": len(testind_ds),
            "testspt": len(testspt_ds),
        },
        "best": {"epoch": best_epoch, "val_iou": best_iou},
        "metrics": {
            "val_iou": val_iou,
            "val_f1": val_f1,
            "val_bce_loss": val_bce,
            "val_topo_loss": val_topo,
            "val_phys_loss": val_phys,
            "val_scale_loss": val_scale,
            "testind_iou": tind_iou,
            "testind_f1": tind_f1,
            "testind_bce_loss": tind_bce,
            "testind_topo_loss": tind_topo,
            "testind_phys_loss": tind_phys,
            "testind_scale_loss": tind_scale,
            "testspt_iou": tspt_iou,
            "testspt_f1": tspt_f1,
            "testspt_bce_loss": tspt_bce,
            "testspt_topo_loss": tspt_topo,
            "testspt_phys_loss": tspt_phys,
            "testspt_scale_loss": tspt_scale,
        },
        "history": [asdict(x) for x in history],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[done] best_epoch=", best_epoch, "best_val_iou=", round(best_iou, 6))
    print("[done] testind_iou=", round(tind_iou, 6), "testspt_iou=", round(tspt_iou, 6))
    print("[done] wrote:", out_dir / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
