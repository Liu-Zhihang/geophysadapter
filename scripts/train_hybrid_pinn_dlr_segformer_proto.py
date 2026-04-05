#!/usr/bin/env python3
"""Hybrid PINN prototype on DLR subset.

This is the first prototype for the new manuscript direction:
- visual backbone: true SegFormer encoder via `transformers`, or a timm encoder fallback
- branch structure: visual / terrain / material / trigger + physics state heads

The script is intentionally conservative:
- it can run with the current DLR processed subset even before external physics vectors are fully attached
- material/trigger vectors default to zeros when not yet available
- hydro/stability residual losses become active once physics vectors are attached
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

try:
    import timm  # type: ignore
except Exception:
    timm = None

try:
    from transformers import SegformerConfig, SegformerModel  # type: ignore
except Exception:
    SegformerConfig = None
    SegformerModel = None

VISUAL_KEYS = [
    "PRE1_B02",
    "PRE1_B03",
    "PRE1_B04",
    "PRE1_B08",
    "POST1_B02",
    "POST1_B03",
    "POST1_B04",
    "POST1_B08",
]
TERRAIN_KEYS = ["None_DEM", "None_SLOPE"]
MASK_KEY = "None_MASK"
TRIGGER_NAMES = ["rainfall", "earthquake", "storm", "snowmelt", "complex", "unknown"]
MATERIAL_DIM = 8
TRIGGER_DIM = len(TRIGGER_NAMES) + 3  # trigger one-hot + era5_tp + smap_sm + era5_hydro

SEGFORMER_VARIANTS = {
    "segformer_b0": {
        "hidden_sizes": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decoder_hidden_size": 256,
    },
    "segformer_b1": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "decoder_hidden_size": 256,
    },
    "segformer_b2": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decoder_hidden_size": 256,
    },
    "segformer_b3": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "decoder_hidden_size": 256,
    },
}

SEGFORMER_HF_IDS = {
    "segformer_b0": "nvidia/mit-b0",
    "segformer_b1": "nvidia/mit-b1",
    "segformer_b2": "nvidia/mit-b2",
    "segformer_b3": "nvidia/mit-b3",
}


@dataclass
class EpochStat:
    epoch: int
    train_loss: float
    val_loss: float
    train_iou: float
    val_iou: float
    train_f1: float
    val_f1: float
    train_seg_loss: float
    val_seg_loss: float
    train_topo_loss: float
    val_topo_loss: float
    train_hydro_loss: float
    val_hydro_loss: float
    train_stability_loss: float
    val_stability_loss: float
    train_obs_loss: float
    val_obs_loss: float
    sec: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _scale_by_key(x: np.ndarray, key: str) -> np.ndarray:
    if key.startswith("PRE") or key.startswith("POST"):
        if any(b in key for b in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]):
            return x / 10000.0
        return x
    if key == "None_DEM":
        return x / 3000.0
    if key == "None_SLOPE":
        return x / 90.0
    return x


class DLRHybridSubset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        sample_manifest_path: Path,
        split: str,
        physics_csv_path: Path | None = None,
        physics_resid_csv_path: Path | None = None,
    ):
        self.h5_path = Path(h5_path)
        self.sample_manifest_path = Path(sample_manifest_path)
        self.split = split
        self.physics_csv_path = Path(physics_csv_path) if physics_csv_path else None
        self.physics_resid_csv_path = Path(physics_resid_csv_path) if physics_resid_csv_path else None
        self._h5: h5py.File | None = None
        self.rows = [r for r in self._load_manifest() if r["split"] == split]
        self.zero_material = torch.zeros(MATERIAL_DIM, dtype=torch.float32)
        self.zero_trigger = torch.zeros(TRIGGER_DIM, dtype=torch.float32)
        self.sample_physics_map, self.event_physics_map = self._load_physics_map()
        self.sample_resid_map, self.event_resid_map = self._load_resid_map()
        with h5py.File(self.h5_path, "r") as f:
            self.ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f.attrs["IDs_order"]]
            self.id_to_index = {sid: idx for idx, sid in enumerate(self.ids)}

    def _load_manifest(self) -> list[dict[str, str]]:
        with self.sample_manifest_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _parse_vector(self, row: dict[str, str], prefix: str, dim: int) -> torch.Tensor:
        vals = []
        for i in range(dim):
            key = f"{prefix}_{i}"
            raw = row.get(key, "")
            try:
                vals.append(float(raw))
            except Exception:
                vals.append(0.0)
        return torch.tensor(vals, dtype=torch.float32)

    def _load_vector_map(self, csv_path: Path | None) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        sample_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        event_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if csv_path is None or not csv_path.exists():
            return sample_map, event_map
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            material = self._parse_vector(row, "material", MATERIAL_DIM)
            trigger = self._parse_vector(row, "trigger", TRIGGER_DIM)
            sample_id = row.get("sample_id", "").strip()
            event_uid = row.get("event_uid", "").strip()
            if sample_id:
                sample_map[sample_id] = (material, trigger)
            if event_uid and event_uid not in event_map:
                event_map[event_uid] = (material, trigger)
        return sample_map, event_map

    def _load_physics_map(self) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        return self._load_vector_map(self.physics_csv_path)

    def _load_resid_map(self) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        return self._load_vector_map(self.physics_resid_csv_path)

    def __len__(self) -> int:
        return len(self.rows)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sample_id = row["sample_id"]
        h5_idx = self.id_to_index[sample_id]
        f = self._get_h5()

        vis = []
        for key in VISUAL_KEYS:
            arr = f[key][h5_idx, 0].astype(np.float32)
            vis.append(_scale_by_key(arr, key))
        terrain = []
        for key in TERRAIN_KEYS:
            arr = f[key][h5_idx, 0].astype(np.float32)
            terrain.append(_scale_by_key(arr, key))
        mask = f[MASK_KEY][h5_idx, 0].astype(np.float32)

        vis = np.stack(vis, axis=0)
        terrain = np.stack(terrain, axis=0)
        mask = np.expand_dims(mask, axis=0)
        vis = np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
        terrain = np.nan_to_num(terrain, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        material, trigger = self.sample_physics_map.get(
            sample_id,
            self.event_physics_map.get(row["event_uid"], (self.zero_material, self.zero_trigger)),
        )
        material_resid, trigger_resid = self.sample_resid_map.get(
            sample_id,
            self.event_resid_map.get(row["event_uid"], (material, trigger)),
        )
        return {
            "visual": torch.from_numpy(vis),
            "terrain": torch.from_numpy(terrain),
            "material": material.clone(),
            "trigger": trigger.clone(),
            "material_resid": material_resid.clone(),
            "trigger_resid": trigger_resid.clone(),
            "mask": torch.from_numpy(mask),
            "sample_id": sample_id,
            "event_uid": row["event_uid"],
        }


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyVisualEncoder(nn.Module):
    def __init__(self, in_ch: int = 8, base: int = 32):
        super().__init__()
        self.stage1 = nn.Sequential(ConvBNReLU(in_ch, base), ConvBNReLU(base, base))
        self.stage2 = nn.Sequential(ConvBNReLU(base, base * 2, stride=2), ConvBNReLU(base * 2, base * 2))
        self.stage3 = nn.Sequential(ConvBNReLU(base * 2, base * 4, stride=2), ConvBNReLU(base * 4, base * 4))
        self.stage4 = nn.Sequential(ConvBNReLU(base * 4, base * 8, stride=2), ConvBNReLU(base * 8, base * 8))
        self.channels = [base, base * 2, base * 4, base * 8]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class TransformersSegformerBackbone(nn.Module):
    def __init__(self, name: str, in_chans: int = 8, pretrained: bool = False):
        super().__init__()
        if SegformerConfig is None or SegformerModel is None:
            raise RuntimeError("transformers_segformer_unavailable")
        variant = SEGFORMER_VARIANTS.get(name.lower())
        if variant is None:
            raise RuntimeError(f"unsupported_segformer_variant:{name}")
        self.pretrained = pretrained
        if pretrained:
            hf_id = SEGFORMER_HF_IDS.get(name.lower())
            if hf_id is None:
                raise RuntimeError(f"unsupported_segformer_hf_id:{name}")
            self.model = SegformerModel.from_pretrained(hf_id)
            self.model.config.output_hidden_states = True
            self._adapt_input_conv(in_chans)
        else:
            config = SegformerConfig(
                num_channels=in_chans,
                depths=variant["depths"],
                hidden_sizes=variant["hidden_sizes"],
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                mlp_ratios=[4, 4, 4, 4],
                hidden_act="gelu",
                classifier_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                hidden_dropout_prob=0.0,
                reshape_last_stage=True,
                decoder_hidden_size=variant["decoder_hidden_size"],
                output_hidden_states=True,
            )
            self.model = SegformerModel(config)
        self.channels = list(variant["hidden_sizes"])

    def _adapt_input_conv(self, in_chans: int) -> None:
        conv = self.model.encoder.patch_embeddings[0].proj
        if conv.in_channels == in_chans:
            return
        new_conv = nn.Conv2d(
            in_chans,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.zero_()
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
            rgb_w = conv.weight
            mean_w = rgb_w.mean(dim=1)
            if in_chans == 8:
                # PRE: B02,B03,B04,B08 / POST: B02,B03,B04,B08
                new_conv.weight[:, 0].copy_(rgb_w[:, 2] * 0.5)  # blue
                new_conv.weight[:, 1].copy_(rgb_w[:, 1] * 0.5)  # green
                new_conv.weight[:, 2].copy_(rgb_w[:, 0] * 0.5)  # red
                new_conv.weight[:, 3].copy_(mean_w * 0.25)      # nir proxy
                new_conv.weight[:, 4].copy_(rgb_w[:, 2] * 0.5)
                new_conv.weight[:, 5].copy_(rgb_w[:, 1] * 0.5)
                new_conv.weight[:, 6].copy_(rgb_w[:, 0] * 0.5)
                new_conv.weight[:, 7].copy_(mean_w * 0.25)
            else:
                scale = conv.in_channels / float(max(in_chans, 1))
                for ch in range(in_chans):
                    new_conv.weight[:, ch].copy_(rgb_w[:, ch % conv.in_channels] * scale)
        self.model.encoder.patch_embeddings[0].proj = new_conv
        self.model.config.num_channels = in_chans

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        return list(outputs.hidden_states)


class VisualBackbone(nn.Module):
    def __init__(self, name: str = "segformer_b0", in_chans: int = 8, pretrained: bool = False):
        super().__init__()
        self.name = name
        self.fallback = False
        self.fallback_reason = ""
        self.backend = "tiny"
        if name.lower().startswith("segformer_"):
            try:
                self.encoder = TransformersSegformerBackbone(name=name, in_chans=in_chans, pretrained=pretrained)
                self.channels = self.encoder.channels
                self.backend = "transformers_segformer"
                return
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
        if timm is not None and name.lower() != "tiny_cnn":
            try:
                self.encoder = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=in_chans)
                self.channels = self.encoder.feature_info.channels()
                self.backend = "timm"
                return
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
        if timm is None:
            self.fallback_reason = "timm_unavailable"
        if not self.fallback_reason:
            self.fallback_reason = f"unsupported_backbone:{name}"
        self.encoder = TinyVisualEncoder(in_ch=in_chans, base=32)
        self.channels = self.encoder.channels
        self.fallback = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.encoder(x)
        if isinstance(feats, tuple):
            feats = list(feats)
        return feats


class TerrainEncoder(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 32),
            ConvBNReLU(32, out_ch, stride=2),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionDecoder(nn.Module):
    def __init__(self, vis_channels: list[int], terrain_ch: int = 64, vec_ch: int = 64, fuse_ch: int = 128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(ch, fuse_ch, 1) for ch in vis_channels])
        self.terrain_proj = nn.Conv2d(terrain_ch, fuse_ch, 1)
        self.seg_head = nn.Sequential(
            ConvBNReLU(fuse_ch * 5, 256),
            ConvBNReLU(256, 128),
            nn.Conv2d(128, 1, 1),
        )
        self.state_head = nn.Sequential(
            ConvBNReLU(fuse_ch * 5, 128),
            nn.Conv2d(128, 2, 1),
        )
        self.vec_to_bias = nn.Linear(vec_ch * 2, fuse_ch)

    def forward(
        self,
        vis_feats: list[torch.Tensor],
        terrain_feat: torch.Tensor,
        material_vec: torch.Tensor,
        trigger_vec: torch.Tensor,
        out_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ups = []
        for feat, proj in zip(vis_feats, self.proj):
            x = proj(feat)
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
            ups.append(x)
        terr = self.terrain_proj(terrain_feat)
        terr = F.interpolate(terr, size=out_size, mode="bilinear", align_corners=False)
        vec_bias = self.vec_to_bias(torch.cat([material_vec, trigger_vec], dim=1))
        vec_bias = vec_bias.unsqueeze(-1).unsqueeze(-1)
        terr = terr + vec_bias
        fused = torch.cat(ups + [terr], dim=1)
        logits = self.seg_head(fused)
        states = self.state_head(fused)
        u_hat = states[:, :1]
        fos_hat = states[:, 1:2]
        return logits, u_hat, fos_hat


class HybridPINNProto(nn.Module):
    def __init__(self, visual_backbone: str = "segformer_b0", pretrained_backbone: bool = False):
        super().__init__()
        self.visual = VisualBackbone(name=visual_backbone, in_chans=len(VISUAL_KEYS), pretrained=pretrained_backbone)
        self.terrain = TerrainEncoder(in_ch=len(TERRAIN_KEYS), out_ch=64)
        self.material = VectorEncoder(MATERIAL_DIM, 64)
        self.trigger = VectorEncoder(TRIGGER_DIM, 64)
        self.decoder = FusionDecoder(self.visual.channels, terrain_ch=64, vec_ch=64, fuse_ch=128)

    def forward(self, visual: torch.Tensor, terrain: torch.Tensor, material: torch.Tensor, trigger: torch.Tensor):
        vis_feats = self.visual(visual)
        terr_feat = self.terrain(terrain)
        mat_vec = self.material(material)
        trg_vec = self.trigger(trigger)
        logits, u_hat, fos_hat = self.decoder(vis_feats, terr_feat, mat_vec, trg_vec, out_size=visual.shape[-2:])
        return logits, u_hat, fos_hat


def compute_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    tgt = (target >= 0.5).float()
    tp = (pred * tgt).sum().item()
    fp = (pred * (1.0 - tgt)).sum().item()
    fn = ((1.0 - pred) * tgt).sum().item()
    iou = tp / (tp + fp + fn + 1e-7)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return iou, f1


def compute_topo_loss(logits: torch.Tensor, terrain: torch.Tensor, slope_low_thresh_norm: float, slope_temp_norm: float) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    slope = terrain[:, 1:2]
    low_slope_w = torch.sigmoid((slope_low_thresh_norm - slope) / max(slope_temp_norm, 1e-6))
    return (prob * low_slope_w).mean()


def compute_change_loss(logits: torch.Tensor, visual: torch.Tensor, ndvi_drop_thresh: float, ndvi_temp: float) -> torch.Tensor:
    eps = 1e-6
    pre_red = visual[:, 2:3]
    pre_nir = visual[:, 3:4]
    post_red = visual[:, 6:7]
    post_nir = visual[:, 7:8]
    pre_ndvi = (pre_nir - pre_red) / (pre_nir + pre_red + eps)
    post_ndvi = (post_nir - post_red) / (post_nir + post_red + eps)
    ndvi_drop = pre_ndvi - post_ndvi
    inconsistent_w = torch.sigmoid((ndvi_drop_thresh - ndvi_drop) / max(ndvi_temp, 1e-6))
    prob = torch.sigmoid(logits)
    return (prob * inconsistent_w).mean()


def _trigger_hydro_proxy(trigger_vec: torch.Tensor) -> torch.Tensor:
    rainfall = trigger_vec[:, len(TRIGGER_NAMES) + 0:len(TRIGGER_NAMES) + 1]
    smap = trigger_vec[:, len(TRIGGER_NAMES) + 1:len(TRIGGER_NAMES) + 2]
    hydro = trigger_vec[:, len(TRIGGER_NAMES) + 2:len(TRIGGER_NAMES) + 3]
    # Keep SMAP as the anchor signal and let ERA5 act as a bounded adjustment.
    return (smap + 0.20 * hydro + 0.10 * rainfall).clamp(0.0, 1.0)


def compute_hydro_loss(u_hat: torch.Tensor, trigger_vec: torch.Tensor) -> torch.Tensor:
    hydro_proxy = _trigger_hydro_proxy(trigger_vec)
    proxy = hydro_proxy.unsqueeze(-1).unsqueeze(-1)
    proxy = proxy.expand_as(u_hat)
    return F.smooth_l1_loss(torch.sigmoid(u_hat), proxy)


def compute_stability_loss(fos_hat: torch.Tensor, terrain: torch.Tensor, material_vec: torch.Tensor, trigger_vec: torch.Tensor) -> torch.Tensor:
    slope = terrain[:, 1:2]
    wc_bare = material_vec[:, 2:3].unsqueeze(-1).unsqueeze(-1)
    clay = material_vec[:, 3:4].unsqueeze(-1).unsqueeze(-1)
    sand = material_vec[:, 4:5].unsqueeze(-1).unsqueeze(-1)
    lith = material_vec[:, 7:8].unsqueeze(-1).unsqueeze(-1)
    rainfall = trigger_vec[:, len(TRIGGER_NAMES) + 0:len(TRIGGER_NAMES) + 1].unsqueeze(-1).unsqueeze(-1)
    hydro = _trigger_hydro_proxy(trigger_vec).unsqueeze(-1).unsqueeze(-1)
    unstable_proxy = 0.35 * slope + 0.25 * hydro + 0.10 * rainfall + 0.10 * clay + 0.08 * wc_bare + 0.05 * lith - 0.03 * sand
    return F.smooth_l1_loss(torch.sigmoid(-fos_hat), unstable_proxy.clamp(0.0, 1.0))


def compute_obs_consistency(logits: torch.Tensor, u_hat: torch.Tensor, fos_hat: torch.Tensor) -> torch.Tensor:
    physics_prob = 0.5 * torch.sigmoid(u_hat) + 0.5 * torch.sigmoid(-fos_hat)
    return F.smooth_l1_loss(torch.sigmoid(logits), physics_prob)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambda_change: float,
    lambda_topo: float,
    lambda_hydro: float,
    lambda_stability: float,
    lambda_obs: float,
    slope_low_thresh_norm: float,
    slope_temp_norm: float,
    ndvi_drop_thresh: float,
    ndvi_temp: float,
):
    train = optimizer is not None
    model.train(train)
    criterion = nn.BCEWithLogitsLoss()

    loss_sum = seg_sum = topo_sum = hydro_sum = stab_sum = obs_sum = 0.0
    iou_sum = f1_sum = 0.0
    n = 0

    for batch in loader:
        visual = batch["visual"].to(device)
        terrain = batch["terrain"].to(device)
        material = batch["material"].to(device)
        trigger = batch["trigger"].to(device)
        material_resid = batch.get("material_resid", batch["material"]).to(device)
        trigger_resid = batch.get("trigger_resid", batch["trigger"]).to(device)
        mask = batch["mask"].to(device)

        logits, u_hat, fos_hat = model(visual, terrain, material, trigger)
        seg_loss = criterion(logits, mask)
        change_loss = compute_change_loss(logits, visual, ndvi_drop_thresh, ndvi_temp)
        topo_loss = compute_topo_loss(logits, terrain, slope_low_thresh_norm, slope_temp_norm)
        hydro_loss = compute_hydro_loss(u_hat, trigger_resid)
        stab_loss = compute_stability_loss(fos_hat, terrain, material_resid, trigger_resid)
        obs_loss = compute_obs_consistency(logits, u_hat, fos_hat)
        loss = seg_loss + lambda_change * change_loss + lambda_topo * topo_loss + lambda_hydro * hydro_loss + lambda_stability * stab_loss + lambda_obs * obs_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bsz = visual.size(0)
        iou, f1 = compute_metrics(logits.detach(), mask)
        loss_sum += float(loss.item()) * bsz
        seg_sum += float(seg_loss.item()) * bsz
        topo_sum += float(topo_loss.item()) * bsz
        hydro_sum += float(hydro_loss.item()) * bsz
        stab_sum += float(stab_loss.item()) * bsz
        obs_sum += float(obs_loss.item()) * bsz
        iou_sum += iou * bsz
        f1_sum += f1 * bsz
        n += bsz

    return {
        "loss": loss_sum / max(n, 1),
        "seg": seg_sum / max(n, 1),
        "topo": topo_sum / max(n, 1),
        "hydro": hydro_sum / max(n, 1),
        "stability": stab_sum / max(n, 1),
        "obs": obs_sum / max(n, 1),
        "iou": iou_sum / max(n, 1),
        "f1": f1_sum / max(n, 1),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--subset-dir", default="", help="default: processed/hybrid_pinn/dlr_strict_t3_reference_subset_v1")
    p.add_argument("--physics-csv", default="", help="default: <subset-dir>/sample_physics_vectors_v1.csv")
    p.add_argument("--physics-resid-csv", default="", help="default: physics-csv")
    p.add_argument("--visual-backbone", default="segformer_b0", help="preferred backbone; uses transformers SegFormer when name starts with segformer_")
    p.add_argument("--pretrained-backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260307)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--lambda-change", type=float, default=0.05)
    p.add_argument("--lambda-topo", type=float, default=0.2)
    p.add_argument("--lambda-hydro", type=float, default=0.0)
    p.add_argument("--lambda-stability", type=float, default=0.0)
    p.add_argument("--lambda-obs", type=float, default=0.0)
    p.add_argument("--slope-low-thresh-deg", type=float, default=10.0)
    p.add_argument("--slope-temp-deg", type=float, default=2.0)
    p.add_argument("--ndvi-drop-thresh", type=float, default=0.05)
    p.add_argument("--ndvi-temp", type=float, default=0.03)
    p.add_argument("--outdir", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)

    root = Path(args.root)
    subset_dir = Path(args.subset_dir) if args.subset_dir.strip() else root / "processed" / "hybrid_pinn" / "dlr_strict_t3_reference_subset_v1"
    physics_csv = Path(args.physics_csv) if args.physics_csv.strip() else subset_dir / "sample_physics_vectors_v1.csv"
    physics_resid_csv = Path(args.physics_resid_csv) if args.physics_resid_csv.strip() else physics_csv
    outdir = Path(args.outdir) if args.outdir.strip() else root / "experiments" / "hybrid_pinn_dlr_segformer_proto"
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        print(f"[info] gpu={torch.cuda.get_device_name(0)}")
    print(f"[info] timm_available={timm is not None}")
    print(f"[info] segformer_available={SegformerModel is not None}")
    print(f"[info] physics_csv={physics_csv} exists={physics_csv.exists()}")
    print(f"[info] physics_resid_csv={physics_resid_csv} exists={physics_resid_csv.exists()}")

    sample_manifest = subset_dir / "sample_manifest.csv"
    train_ds = DLRHybridSubset(subset_dir / "train_n3_s1s2.h5", sample_manifest, split="train", physics_csv_path=physics_csv, physics_resid_csv_path=physics_resid_csv)
    val_ds = DLRHybridSubset(subset_dir / "val_n3_s1s2.h5", sample_manifest, split="val", physics_csv_path=physics_csv, physics_resid_csv_path=physics_resid_csv)
    testind_ds = DLRHybridSubset(subset_dir / "testind_n3_s1s2.h5", sample_manifest, split="testind", physics_csv_path=physics_csv, physics_resid_csv_path=physics_resid_csv)
    testspt_ds = DLRHybridSubset(subset_dir / "testspt_n3_s1s2.h5", sample_manifest, split="testspt", physics_csv_path=physics_csv, physics_resid_csv_path=physics_resid_csv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    testind_loader = DataLoader(testind_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    testspt_loader = DataLoader(testspt_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    print(f"[info] samples train/val/testind/testspt={len(train_ds)}/{len(val_ds)}/{len(testind_ds)}/{len(testspt_ds)}")

    model = HybridPINNProto(visual_backbone=args.visual_backbone, pretrained_backbone=args.pretrained_backbone).to(device)
    if getattr(model.visual, "fallback", False):
        print(f"[warn] visual_backbone_fallback requested={args.visual_backbone} reason={getattr(model.visual, 'fallback_reason', 'unknown')}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_iou = -1.0
    best_epoch = -1
    history: list[EpochStat] = []
    ckpt_path = outdir / "best_model.pt"

    slope_low_thresh_norm = args.slope_low_thresh_deg / 90.0
    slope_temp_norm = args.slope_temp_deg / 90.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(
            model, train_loader, device, optimizer,
            args.lambda_change, args.lambda_topo, args.lambda_hydro, args.lambda_stability, args.lambda_obs,
            slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp,
        )
        va = run_epoch(
            model, val_loader, device, None,
            args.lambda_change, args.lambda_topo, args.lambda_hydro, args.lambda_stability, args.lambda_obs,
            slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp,
        )
        scheduler.step()
        sec = time.time() - t0
        history.append(EpochStat(
            epoch=epoch,
            train_loss=tr["loss"], val_loss=va["loss"], train_iou=tr["iou"], val_iou=va["iou"], train_f1=tr["f1"], val_f1=va["f1"],
            train_seg_loss=tr["seg"], val_seg_loss=va["seg"], train_topo_loss=tr["topo"], val_topo_loss=va["topo"],
            train_hydro_loss=tr["hydro"], val_hydro_loss=va["hydro"], train_stability_loss=tr["stability"], val_stability_loss=va["stability"],
            train_obs_loss=tr["obs"], val_obs_loss=va["obs"], sec=sec,
        ))
        print(
            f"[epoch {epoch:02d}] train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"train_iou={tr['iou']:.4f} val_iou={va['iou']:.4f} "
            f"seg={tr['seg']:.4f}/{va['seg']:.4f} topo={tr['topo']:.4f}/{va['topo']:.4f} "
            f"hydro={tr['hydro']:.4f}/{va['hydro']:.4f} stab={tr['stability']:.4f}/{va['stability']:.4f} obs={tr['obs']:.4f}/{va['obs']:.4f} sec={sec:.1f}"
        )
        if va["iou"] > best_iou:
            best_iou = va["iou"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "visual_backbone": args.visual_backbone}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = {
        "val": run_epoch(model, val_loader, device, None, args.lambda_change, args.lambda_topo, args.lambda_hydro, args.lambda_stability, args.lambda_obs, slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp),
        "testind": run_epoch(model, testind_loader, device, None, args.lambda_change, args.lambda_topo, args.lambda_hydro, args.lambda_stability, args.lambda_obs, slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp),
        "testspt": run_epoch(model, testspt_loader, device, None, args.lambda_change, args.lambda_topo, args.lambda_hydro, args.lambda_stability, args.lambda_obs, slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp),
    }
    result = {
        "timestamp": int(time.time()),
        "seed": args.seed,
        "device": str(device),
        "visual_backbone": args.visual_backbone,
        "pretrained_backbone": bool(args.pretrained_backbone),
        "timm_available": timm is not None,
        "segformer_available": SegformerModel is not None,
        "backbone_fallback": bool(getattr(model.visual, 'fallback', False)),
        "backbone_backend": getattr(model.visual, "backend", "unknown"),
        "subset_dir": str(subset_dir),
        "physics_csv": str(physics_csv),
        "physics_csv_exists": physics_csv.exists(),
        "physics_resid_csv": str(physics_resid_csv),
        "physics_resid_csv_exists": physics_resid_csv.exists(),
        "samples": {"train": len(train_ds), "val": len(val_ds), "testind": len(testind_ds), "testspt": len(testspt_ds)},
        "lambdas": {
            "change": args.lambda_change,
            "topo": args.lambda_topo,
            "hydro": args.lambda_hydro,
            "stability": args.lambda_stability,
            "obs": args.lambda_obs,
        },
        "best": {"epoch": best_epoch, "val_iou": best_iou},
        "metrics": metrics,
        "history": [asdict(x) for x in history],
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] wrote {outdir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
