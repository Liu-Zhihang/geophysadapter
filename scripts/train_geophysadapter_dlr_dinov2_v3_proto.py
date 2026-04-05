#!/usr/bin/env python3
"""GeoPhysAdapter v3 on DLR strict_t3.

Changes over v2:
- dual-expert decoder with visual / physics experts
- domain-aware gate to blend expert outputs
- optional teacher distillation from Stage-1 topophys U-Net
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

from train_baseline_dlr_unet import SmallUNet

try:
    import timm  # type: ignore
except Exception:
    timm = None

PRE_KEYS = ["PRE1_B02", "PRE1_B03", "PRE1_B04", "PRE1_B08"]
POST_KEYS = ["POST1_B02", "POST1_B03", "POST1_B04", "POST1_B08"]
TERRAIN_KEYS = ["None_DEM", "None_SLOPE"]
MASK_KEY = "None_MASK"
TRIGGER_NAMES = ["rainfall", "earthquake", "storm", "snowmelt", "complex", "unknown"]
MATERIAL_DIM = 8
TRIGGER_DIM = len(TRIGGER_NAMES) + 3


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
    train_distill_loss: float
    val_distill_loss: float
    train_gate_mean: float
    val_gate_mean: float
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
        self.sample_physics_map, self.event_physics_map = self._load_vector_map(self.physics_csv_path)
        self.sample_resid_map, self.event_resid_map = self._load_vector_map(self.physics_resid_csv_path)
        with h5py.File(self.h5_path, "r") as f:
            self.ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f.attrs["IDs_order"]]
            self.id_to_index = {sid: idx for idx, sid in enumerate(self.ids)}

    def _load_manifest(self) -> list[dict[str, str]]:
        with self.sample_manifest_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _parse_vector(self, row: dict[str, str], prefix: str, dim: int) -> torch.Tensor:
        vals = []
        for i in range(dim):
            raw = row.get(f"{prefix}_{i}", "")
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

        pre = np.stack([_scale_by_key(f[k][h5_idx, 0].astype(np.float32), k) for k in PRE_KEYS], axis=0)
        post = np.stack([_scale_by_key(f[k][h5_idx, 0].astype(np.float32), k) for k in POST_KEYS], axis=0)
        terrain = np.stack([_scale_by_key(f[k][h5_idx, 0].astype(np.float32), k) for k in TERRAIN_KEYS], axis=0)
        mask = np.expand_dims(f[MASK_KEY][h5_idx, 0].astype(np.float32), axis=0)

        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0)
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
            "pre": torch.from_numpy(pre),
            "post": torch.from_numpy(post),
            "terrain": torch.from_numpy(terrain),
            "material": material.clone(),
            "trigger": trigger.clone(),
            "material_resid": material_resid.clone(),
            "trigger_resid": trigger_resid.clone(),
            "mask": torch.from_numpy(mask),
        }


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return self.act(x + y)


class Dinov2TwinBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, dino_input_size: int, in_chans: int = 4, freeze_backbone: bool = True, unfreeze_last_blocks: int = 0):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm_unavailable")
        self.model = timm.create_model(model_name, pretrained=pretrained, img_size=dino_input_size, in_chans=in_chans, num_classes=0)
        self.embed_dim = int(getattr(self.model, "embed_dim", 384))
        self.dino_input_size = dino_input_size
        self.indices = self._resolve_indices()
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_blocks = max(0, int(unfreeze_last_blocks))
        self._configure_trainable()

    def _resolve_indices(self) -> tuple[int, int, int, int]:
        n_blocks = len(self.model.blocks)
        candidates = [max(0, math.ceil(n_blocks * frac) - 1) for frac in (0.25, 0.5, 0.75, 1.0)]
        uniq = []
        for idx in candidates:
            if idx not in uniq:
                uniq.append(idx)
        while len(uniq) < 4:
            uniq.insert(0, max(0, uniq[0] - 1 if uniq else 0))
            uniq = sorted(set(uniq))
        return tuple(uniq[-4:])

    def _configure_trainable(self) -> None:
        if not self.freeze_backbone:
            return
        for p in self.model.parameters():
            p.requires_grad = False
        if self.unfreeze_last_blocks > 0:
            for blk in self.model.blocks[-self.unfreeze_last_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            if hasattr(self.model, "norm"):
                for p in self.model.norm.parameters():
                    p.requires_grad = True

    @property
    def any_trainable(self) -> bool:
        return any(p.requires_grad for p in self.model.parameters())

    def _forward_single(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[-1] != self.dino_input_size or x.shape[-2] != self.dino_input_size:
            x = F.interpolate(x, size=(self.dino_input_size, self.dino_input_size), mode="bilinear", align_corners=False)
        kwargs = {"indices": self.indices, "norm": False, "output_fmt": "NCHW", "intermediates_only": False}
        if self.freeze_backbone and not self.any_trainable:
            with torch.no_grad():
                _, feats = self.model.forward_intermediates(x, **kwargs)
        else:
            _, feats = self.model.forward_intermediates(x, **kwargs)
        return list(feats)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self._forward_single(pre), self._forward_single(post)


class DetailStem(nn.Module):
    def __init__(self, in_ch: int = 12, out_ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 32),
            ConvBNAct(32, out_ch),
            ResidualConvBlock(out_ch),
            ResidualConvBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class TerrainEncoder(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 64, token_dim: int = 128):
        super().__init__()
        self.map_net = nn.Sequential(
            ConvBNAct(in_ch, 32),
            ConvBNAct(32, out_ch),
            ResidualConvBlock(out_ch),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(out_ch, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.map_net(x)
        token = self.token_proj(F.adaptive_avg_pool2d(feat, 1).flatten(1))
        return feat, token


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, token_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsFiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Linear(cond_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + 0.1 * gamma) + 0.1 * beta


class CrossAttentionRefiner(nn.Module):
    def __init__(self, feat_dim: int, token_dim: int, num_heads: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(feat_dim)
        self.kv_norm = nn.LayerNorm(feat_dim)
        self.token_proj = nn.Linear(token_dim, feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, feat: torch.Tensor, phys_tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        kv = self.token_proj(phys_tokens)
        attn_out, _ = self.attn(self.query_norm(x), self.kv_norm(kv), self.kv_norm(kv), need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x.transpose(1, 2).reshape(b, c, h, w)


class DomainGate(nn.Module):
    def __init__(self, sem_dim: int, cond_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sem_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sem_feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        sem_pool = F.adaptive_avg_pool2d(sem_feat, 1).flatten(1)
        gate = torch.sigmoid(self.net(torch.cat([sem_pool, cond], dim=1)))
        return gate.unsqueeze(-1).unsqueeze(-1)


class GeoPhysAdapterDecoderV3(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        token_dim: int = 128,
        sem_ch: int = 96,
        detail_ch: int = 64,
        terrain_ch: int = 64,
        visual_only: bool = False,
        disable_routing: bool = False,
        disable_state_heads: bool = False,
    ):
        super().__init__()
        self.visual_only = bool(visual_only)
        self.disable_routing = bool(disable_routing)
        self.disable_state_heads = bool(disable_state_heads)
        self.stage_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim * 3, sem_ch, 1, bias=False),
                nn.BatchNorm2d(sem_ch),
                nn.GELU(),
                ResidualConvBlock(sem_ch),
            )
            for _ in range(4)
        ])
        self.stage_film = nn.ModuleList([PhysicsFiLM(token_dim * 3, sem_ch) for _ in range(4)])
        self.detail_film = PhysicsFiLM(token_dim * 3, detail_ch)
        self.semantic_fuse = nn.Sequential(
            ConvBNAct(sem_ch * 4, 192),
            ResidualConvBlock(192),
        )
        self.semantic_refiner = CrossAttentionRefiner(192, token_dim, num_heads=4)
        self.terrain_proj = nn.Conv2d(terrain_ch, 64, 1)
        self.visual_fuse = nn.Sequential(
            ConvBNAct(192 + detail_ch, 192),
            ResidualConvBlock(192),
            ConvBNAct(192, 128),
        )
        self.physics_fuse = nn.Sequential(
            ConvBNAct(192 + detail_ch + 64, 256),
            ResidualConvBlock(256),
            ConvBNAct(256, 128),
        )
        self.gate = DomainGate(sem_dim=192, cond_dim=token_dim * 3, hidden_dim=128)
        self.visual_head = nn.Conv2d(128, 1, 1)
        self.physics_head = nn.Conv2d(128, 1, 1)
        self.state_head = nn.Conv2d(128, 2, 1)

    def forward(
        self,
        pre_feats: list[torch.Tensor],
        post_feats: list[torch.Tensor],
        detail_feat: torch.Tensor,
        terrain_feat: torch.Tensor,
        terrain_token: torch.Tensor,
        material_token: torch.Tensor,
        trigger_token: torch.Tensor,
        out_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = torch.cat([terrain_token, material_token, trigger_token], dim=1)
        phys_tokens = torch.stack([terrain_token, material_token, trigger_token], dim=1)
        if self.visual_only:
            cond = torch.zeros_like(cond)
            phys_tokens = torch.zeros_like(phys_tokens)
            terrain_feat = torch.zeros_like(terrain_feat)
        stage_maps = []
        for proj, film, pre, post in zip(self.stage_proj, self.stage_film, pre_feats, post_feats):
            change = torch.abs(post - pre)
            x = proj(torch.cat([pre, post, change], dim=1))
            x = film(x, cond)
            stage_maps.append(x)
        sem = self.semantic_fuse(torch.cat(stage_maps, dim=1))
        sem = self.semantic_refiner(sem, phys_tokens)
        sem = F.interpolate(sem, size=out_size, mode="bilinear", align_corners=False)
        detail = self.detail_film(detail_feat, cond)
        terr = self.terrain_proj(terrain_feat)
        terr = F.interpolate(terr, size=out_size, mode="bilinear", align_corners=False)
        visual_feat = self.visual_fuse(torch.cat([sem, detail], dim=1))
        physics_feat = self.physics_fuse(torch.cat([sem, detail, terr], dim=1))
        visual_logits = self.visual_head(visual_feat)
        if self.visual_only:
            gate = torch.zeros((visual_feat.shape[0], 1, 1, 1), device=visual_feat.device, dtype=visual_feat.dtype)
            physics_logits = torch.zeros_like(visual_logits)
            logits = visual_logits
            fused = visual_feat
        else:
            physics_logits = self.physics_head(physics_feat)
            if self.disable_routing:
                gate = torch.full((visual_feat.shape[0], 1, 1, 1), 0.5, device=visual_feat.device, dtype=visual_feat.dtype)
            else:
                gate = self.gate(sem, cond)
            logits = (1.0 - gate) * visual_logits + gate * physics_logits
            fused = (1.0 - gate) * visual_feat + gate * physics_feat
        if self.disable_state_heads:
            states = torch.zeros((fused.shape[0], 2, fused.shape[2], fused.shape[3]), device=fused.device, dtype=fused.dtype)
        else:
            states = self.state_head(fused)
        return logits, states[:, :1], states[:, 1:2], gate, visual_logits, physics_logits


class GeoPhysAdapterHybridPINNV3(nn.Module):
    def __init__(
        self,
        dino_model: str,
        pretrained_backbone: bool,
        freeze_backbone: bool,
        unfreeze_last_blocks: int,
        dino_input_size: int,
        visual_only: bool = False,
        disable_routing: bool = False,
        disable_state_heads: bool = False,
    ):
        super().__init__()
        self.visual = Dinov2TwinBackbone(
            model_name=dino_model,
            pretrained=pretrained_backbone,
            dino_input_size=dino_input_size,
            in_chans=len(PRE_KEYS),
            freeze_backbone=freeze_backbone,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )
        self.detail = DetailStem(in_ch=12, out_ch=64)
        self.terrain = TerrainEncoder(in_ch=len(TERRAIN_KEYS), out_ch=64, token_dim=128)
        self.material = VectorEncoder(MATERIAL_DIM, token_dim=128)
        self.trigger = VectorEncoder(TRIGGER_DIM, token_dim=128)
        self.decoder = GeoPhysAdapterDecoderV3(
            embed_dim=self.visual.embed_dim,
            token_dim=128,
            sem_ch=96,
            detail_ch=64,
            terrain_ch=64,
            visual_only=visual_only,
            disable_routing=disable_routing,
            disable_state_heads=disable_state_heads,
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor, terrain: torch.Tensor, material: torch.Tensor, trigger: torch.Tensor):
        pre_feats, post_feats = self.visual(pre, post)
        detail_input = torch.cat([pre, post, torch.abs(post - pre)], dim=1)
        detail_feat = self.detail(detail_input)
        terrain_feat, terrain_token = self.terrain(terrain)
        material_token = self.material(material)
        trigger_token = self.trigger(trigger)
        return self.decoder(pre_feats, post_feats, detail_feat, terrain_feat, terrain_token, material_token, trigger_token, out_size=pre.shape[-2:])


def resolve_teacher_ckpt(root: Path, teacher_ckpt_arg: str, seed: int) -> Path | None:
    if teacher_ckpt_arg.strip():
        return Path(teacher_ckpt_arg)
    candidates = [
        root / "experiments" / f"dlr_topophys_best_seed{seed}_run1" / "best_model.pt",
        root / "experiments" / "dlr_topophys_best_seed20260307_run1" / "best_model.pt",
        root / "experiments" / "dlr_topophys_best_seed20260306_run1" / "best_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_teacher_model(teacher_ckpt: Path, device: torch.device) -> nn.Module:
    model = SmallUNet(in_ch=len(PRE_KEYS) + len(POST_KEYS) + len(TERRAIN_KEYS), base=32).to(device)
    ckpt = torch.load(teacher_ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits))


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    tgt = (target >= 0.5).float()
    inter = (prob * tgt).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def compute_seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    return 0.7 * bce + 0.3 * dice


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


def compute_change_loss(logits: torch.Tensor, pre: torch.Tensor, post: torch.Tensor, ndvi_drop_thresh: float, ndvi_temp: float) -> torch.Tensor:
    eps = 1e-6
    pre_red = pre[:, 2:3]
    pre_nir = pre[:, 3:4]
    post_red = post[:, 2:3]
    post_nir = post[:, 3:4]
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
    return (smap + 0.20 * hydro + 0.10 * rainfall).clamp(0.0, 1.0)


def compute_hydro_loss(u_hat: torch.Tensor, trigger_vec: torch.Tensor) -> torch.Tensor:
    proxy = _trigger_hydro_proxy(trigger_vec).unsqueeze(-1).unsqueeze(-1).expand_as(u_hat)
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
    teacher_model: nn.Module | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
    lambda_change: float,
    lambda_topo: float,
    lambda_hydro: float,
    lambda_stability: float,
    lambda_obs: float,
    lambda_distill: float,
    slope_low_thresh_norm: float,
    slope_temp_norm: float,
    ndvi_drop_thresh: float,
    ndvi_temp: float,
):
    train = optimizer is not None
    model.train(train)

    loss_sum = seg_sum = topo_sum = hydro_sum = stab_sum = obs_sum = distill_sum = gate_sum = 0.0
    iou_sum = f1_sum = 0.0
    n = 0

    autocast_enabled = bool(amp_enabled and device.type == "cuda")
    autocast_dtype = torch.float16

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        terrain = batch["terrain"].to(device, non_blocking=True)
        material = batch["material"].to(device, non_blocking=True)
        trigger = batch["trigger"].to(device, non_blocking=True)
        material_resid = batch.get("material_resid", batch["material"]).to(device, non_blocking=True)
        trigger_resid = batch.get("trigger_resid", batch["trigger"]).to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            logits, u_hat, fos_hat, gate, _, _ = model(pre, post, terrain, material, trigger)
            seg_loss = compute_seg_loss(logits, mask)
            change_loss = compute_change_loss(logits, pre, post, ndvi_drop_thresh, ndvi_temp)
            topo_loss = compute_topo_loss(logits, terrain, slope_low_thresh_norm, slope_temp_norm)
            hydro_loss = compute_hydro_loss(u_hat, trigger_resid)
            stab_loss = compute_stability_loss(fos_hat, terrain, material_resid, trigger_resid)
            obs_loss = compute_obs_consistency(logits, u_hat, fos_hat)
            distill_loss = torch.zeros((), device=device, dtype=logits.dtype)
            if teacher_model is not None and lambda_distill > 0.0:
                with torch.no_grad():
                    teacher_inp = torch.cat([pre, post, terrain], dim=1)
                    teacher_logits = teacher_model(teacher_inp)
                distill_loss = compute_distill_loss(logits, teacher_logits)
            loss = seg_loss + lambda_change * change_loss + lambda_topo * topo_loss + lambda_hydro * hydro_loss + lambda_stability * stab_loss + lambda_obs * obs_loss + lambda_distill * distill_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bsz = pre.size(0)
        iou, f1 = compute_metrics(logits.detach().float(), mask.float())
        loss_sum += float(loss.detach().item()) * bsz
        seg_sum += float(seg_loss.detach().item()) * bsz
        topo_sum += float(topo_loss.detach().item()) * bsz
        hydro_sum += float(hydro_loss.detach().item()) * bsz
        stab_sum += float(stab_loss.detach().item()) * bsz
        obs_sum += float(obs_loss.detach().item()) * bsz
        distill_sum += float(distill_loss.detach().item()) * bsz
        gate_sum += float(gate.detach().mean().item()) * bsz
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
        "distill": distill_sum / max(n, 1),
        "gate_mean": gate_sum / max(n, 1),
        "iou": iou_sum / max(n, 1),
        "f1": f1_sum / max(n, 1),
    }


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--subset-dir", default="")
    p.add_argument("--physics-csv", default="")
    p.add_argument("--physics-resid-csv", default="")
    p.add_argument("--dino-model", default="vit_small_patch14_dinov2")
    p.add_argument("--dino-input-size", type=int, default=196)
    p.add_argument("--pretrained-backbone", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--unfreeze-last-blocks", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260307)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--lambda-change", type=float, default=0.05)
    p.add_argument("--lambda-topo", type=float, default=0.2)
    p.add_argument("--lambda-hydro", type=float, default=0.02)
    p.add_argument("--lambda-stability", type=float, default=0.02)
    p.add_argument("--lambda-obs", type=float, default=0.01)
    p.add_argument("--lambda-distill", type=float, default=0.05)
    p.add_argument("--teacher-ckpt", default="")
    p.add_argument("--visual-only", action="store_true", help="disable physics conditioning and use only the visual expert inside the same twin-backbone scaffold")
    p.add_argument("--disable-routing", action="store_true", help="replace sample-adaptive routing with a fixed 0.5 blend between visual and physics experts")
    p.add_argument("--disable-state-heads", action="store_true", help="disable state-head predictions; hydro/stability/obs losses will also be zeroed")
    p.add_argument("--slope-low-thresh-deg", type=float, default=10.0)
    p.add_argument("--slope-temp-deg", type=float, default=2.0)
    p.add_argument("--ndvi-drop-thresh", type=float, default=0.05)
    p.add_argument("--ndvi-temp", type=float, default=0.03)
    p.add_argument("--outdir", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.visual_only and args.disable_routing:
        print("[warn] --visual-only supersedes --disable-routing; ignoring fixed-routing request.")
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    root = Path(args.root)
    subset_dir = Path(args.subset_dir) if args.subset_dir.strip() else root / "processed" / "hybrid_pinn" / "dlr_strict_t3_reference_subset_v1"
    physics_csv = Path(args.physics_csv) if args.physics_csv.strip() else subset_dir / "sample_physics_vectors_v1.csv"
    physics_resid_csv = Path(args.physics_resid_csv) if args.physics_resid_csv.strip() else physics_csv
    outdir = Path(args.outdir) if args.outdir.strip() else root / "experiments" / "geophysadapter_dinov2_v2_proto"
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        print(f"[info] gpu={torch.cuda.get_device_name(0)}")
    print(f"[info] timm_available={timm is not None}")
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

    lambda_hydro = 0.0 if args.disable_state_heads else args.lambda_hydro
    lambda_stability = 0.0 if args.disable_state_heads else args.lambda_stability
    lambda_obs = 0.0 if args.disable_state_heads else args.lambda_obs

    model = GeoPhysAdapterHybridPINNV3(
        dino_model=args.dino_model,
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        dino_input_size=args.dino_input_size,
        visual_only=args.visual_only,
        disable_routing=args.disable_routing and not args.visual_only,
        disable_state_heads=args.disable_state_heads,
    ).to(device)
    teacher_ckpt = resolve_teacher_ckpt(root, args.teacher_ckpt, args.seed)
    if args.lambda_distill > 0.0 and teacher_ckpt is None:
        raise FileNotFoundError("teacher_ckpt_not_found_for_distillation")
    teacher_model = load_teacher_model(teacher_ckpt, device) if teacher_ckpt is not None else None
    trainable_params = count_trainable_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[info] trainable_params={trainable_params} total_params={total_params}")
    print(f"[info] teacher_ckpt={teacher_ckpt} exists={teacher_ckpt.exists() if teacher_ckpt else False}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))
    else:
        scaler = None

    best_iou = -1.0
    best_epoch = -1
    history: list[EpochStat] = []
    ckpt_path = outdir / "best_model.pt"

    slope_low_thresh_norm = args.slope_low_thresh_deg / 90.0
    slope_temp_norm = args.slope_temp_deg / 90.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, teacher_model, scaler, args.amp, args.lambda_change, args.lambda_topo, lambda_hydro, lambda_stability, lambda_obs, args.lambda_distill, slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp)
        va = run_epoch(model, val_loader, device, None, teacher_model, None, args.amp, args.lambda_change, args.lambda_topo, lambda_hydro, lambda_stability, lambda_obs, args.lambda_distill, slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp)
        scheduler.step()
        sec = time.time() - t0
        history.append(EpochStat(
            epoch=epoch,
            train_loss=tr["loss"], val_loss=va["loss"], train_iou=tr["iou"], val_iou=va["iou"], train_f1=tr["f1"], val_f1=va["f1"],
            train_seg_loss=tr["seg"], val_seg_loss=va["seg"], train_topo_loss=tr["topo"], val_topo_loss=va["topo"],
            train_hydro_loss=tr["hydro"], val_hydro_loss=va["hydro"], train_stability_loss=tr["stability"], val_stability_loss=va["stability"],
            train_obs_loss=tr["obs"], val_obs_loss=va["obs"], train_distill_loss=tr["distill"], val_distill_loss=va["distill"],
            train_gate_mean=tr["gate_mean"], val_gate_mean=va["gate_mean"], sec=sec,
        ))
        print(
            f"[epoch {epoch:02d}] train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"train_iou={tr['iou']:.4f} val_iou={va['iou']:.4f} "
            f"seg={tr['seg']:.4f}/{va['seg']:.4f} topo={tr['topo']:.4f}/{va['topo']:.4f} "
            f"hydro={tr['hydro']:.4f}/{va['hydro']:.4f} stab={tr['stability']:.4f}/{va['stability']:.4f} "
            f"obs={tr['obs']:.4f}/{va['obs']:.4f} distill={tr['distill']:.4f}/{va['distill']:.4f} "
            f"gate={tr['gate_mean']:.3f}/{va['gate_mean']:.3f} sec={sec:.1f}"
        )
        if va["iou"] > best_iou:
            best_iou = va["iou"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "dino_model": args.dino_model, "dino_input_size": args.dino_input_size, "unfreeze_last_blocks": args.unfreeze_last_blocks}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = {
        "val": run_epoch(model, val_loader, device, None, teacher_model, None, args.amp, args.lambda_change, args.lambda_topo, lambda_hydro, lambda_stability, lambda_obs, args.lambda_distill, slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp),
        "testind": run_epoch(model, testind_loader, device, None, teacher_model, None, args.amp, args.lambda_change, args.lambda_topo, lambda_hydro, lambda_stability, lambda_obs, args.lambda_distill, slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp),
        "testspt": run_epoch(model, testspt_loader, device, None, teacher_model, None, args.amp, args.lambda_change, args.lambda_topo, lambda_hydro, lambda_stability, lambda_obs, args.lambda_distill, slope_low_thresh_norm, slope_temp_norm, args.ndvi_drop_thresh, args.ndvi_temp),
    }
    result = {
        "timestamp": int(time.time()),
        "seed": args.seed,
        "device": str(device),
        "architecture": "GeoPhysAdapterHybridPINNV3",
        "dino_model": args.dino_model,
        "dino_input_size": args.dino_input_size,
        "pretrained_backbone": bool(args.pretrained_backbone),
        "freeze_backbone": bool(args.freeze_backbone),
        "unfreeze_last_blocks": int(args.unfreeze_last_blocks),
        "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt is not None else "",
        "teacher_ckpt_exists": bool(teacher_ckpt.exists()) if teacher_ckpt is not None else False,
        "amp": bool(args.amp),
        "timm_available": timm is not None,
        "subset_dir": str(subset_dir),
        "physics_csv": str(physics_csv),
        "physics_csv_exists": physics_csv.exists(),
        "physics_resid_csv": str(physics_resid_csv),
        "physics_resid_csv_exists": physics_resid_csv.exists(),
        "samples": {"train": len(train_ds), "val": len(val_ds), "testind": len(testind_ds), "testspt": len(testspt_ds)},
        "params": {"trainable": trainable_params, "total": total_params},
        "lambdas": {"change": args.lambda_change, "topo": args.lambda_topo, "hydro": lambda_hydro, "stability": lambda_stability, "obs": lambda_obs, "distill": args.lambda_distill},
        "ablations": {
            "visual_only": bool(args.visual_only),
            "disable_routing": bool(args.disable_routing and not args.visual_only),
            "disable_state_heads": bool(args.disable_state_heads),
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
