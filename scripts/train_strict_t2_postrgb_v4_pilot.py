#!/usr/bin/env python3
"""Train a strict_t2 post_rgb v4/v5 pilot with structured priors and failure-aware variants."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from strict_t2_strong_backbone_common import BinaryDeepLabV3
from train_strict_t2_postrgb_baseline import (
    CachedPostRgbH5Dataset,
    StrictT2PostRgbDataset,
    batch_dataset_weights,
    build_dataset_weight_map,
    build_weighted_sampler,
    compute_segmentation_loss,
    default_eval_cache_path,
    default_train_cache_path,
    masked_stats,
    parse_named_value_overrides,
    parse_threshold_grid,
    read_gdcld_index,
    read_manifest,
    set_seed,
    subset_rows,
)
from train_strict_t2_postrgb_phys_baseline import (
    gather_train_stats,
    load_physics_maps,
    metadata_from_base_dataset,
)


PHYSICS_SPLITS = {
    "terrain": [f"terrain_{idx}" for idx in range(4)],
    "material": [f"material_{idx}" for idx in range(9)],
    "trigger": [f"trigger_{idx}" for idx in range(9)] + [f"trigger_ext_{idx}" for idx in range(2)],
    "proxy": ["hydro_proxy", "stability_proxy"],
}
PHYSICS_VECTOR_COLUMNS = (
    PHYSICS_SPLITS["terrain"]
    + PHYSICS_SPLITS["material"]
    + PHYSICS_SPLITS["trigger"]
    + PHYSICS_SPLITS["proxy"]
)
META_DIM = 18
DATASET_META_ORDER = ["DLR_Landslide_Ref_2025", "CAS_Landslide", "GDCLD", "GLaD4CD_v1"]


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


def build_meta_vector(row: dict[str, object], eps: float = 1e-6, drop_source_identity: bool = False) -> np.ndarray:
    dataset_id = str(row.get("dataset_id", ""))
    terrain_available = 1.0 if float(row.get("terrain_available", 0.0) or 0.0) > 0.5 else 0.0
    support_worldcover = 1.0 if any(abs(float(row.get(col, 0.0) or 0.0)) > eps for col in PHYSICS_SPLITS["material"][:3]) else 0.0
    support_soil = 1.0 if any(abs(float(row.get(col, 0.0) or 0.0)) > eps for col in PHYSICS_SPLITS["material"][3:8]) else 0.0
    support_lithology = 1.0 if abs(float(row.get("material_8", 0.0) or 0.0)) > eps or abs(float(row.get("lithology_raw", 0.0) or 0.0)) > eps else 0.0
    support_trigger_dynamic = 1.0 if (
        any(abs(float(row.get(col, 0.0) or 0.0)) > eps for col in ("trigger_ext_0", "trigger_ext_1"))
        or any(abs(float(row.get(col, 0.0) or 0.0)) > eps for col in ("era5_tp_raw", "era5_hydro_raw"))
    ) else 0.0
    support_wetness = 1.0 if abs(float(row.get("trigger_8", 0.0) or 0.0)) > eps or abs(float(row.get("smap_sm_raw", 0.0) or 0.0)) > eps else 0.0

    quality_terrain = 0.85 if terrain_available > 0.5 else 0.35
    quality_material = min(1.0, 0.25 + 0.30 * support_worldcover + 0.20 * support_soil + 0.15 * support_lithology)
    quality_trigger = min(1.0, 0.25 + 0.25 * support_wetness + 0.50 * support_trigger_dynamic)
    quality_physics = 0.35 * quality_terrain + 0.30 * quality_material + 0.35 * quality_trigger

    granularity_terrain = 0.60 if terrain_available > 0.5 else 0.25
    granularity_worldcover = 0.60 if support_worldcover > 0.5 else 0.25
    granularity_soil = 0.45 if support_soil > 0.5 else 0.20
    granularity_lithology = 0.45 if support_lithology > 0.5 else 0.20
    granularity_trigger = 0.55 if support_trigger_dynamic > 0.5 else (0.40 if support_wetness > 0.5 else 0.20)

    dataset_one_hot = [1.0 if dataset_id == name else 0.0 for name in DATASET_META_ORDER]
    if drop_source_identity:
        dataset_one_hot = [0.0 for _ in DATASET_META_ORDER]
    geom_uncertainty = 1.0 - quality_terrain
    return np.asarray(
        [
            quality_terrain,
            quality_material,
            quality_trigger,
            quality_physics,
            support_worldcover,
            support_soil,
            support_lithology,
            support_trigger_dynamic,
            granularity_terrain,
            granularity_worldcover,
            granularity_soil,
            granularity_lithology,
            granularity_trigger,
            *dataset_one_hot,
            geom_uncertainty,
        ],
        dtype=np.float32,
    )


def load_meta_maps(
    csv_path: Path,
    drop_source_identity: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    df = pd.read_csv(csv_path, low_memory=False).fillna(0.0)
    sample_map: dict[str, np.ndarray] = {}
    event_map: dict[str, np.ndarray] = {}
    quality_values = []
    for row in df.to_dict(orient="records"):
        meta = build_meta_vector(row, drop_source_identity=drop_source_identity)
        sample_id = str(row["sample_id"])
        event_uid = str(row["event_uid"])
        sample_map[sample_id] = meta
        event_map.setdefault(event_uid, meta)
        quality_values.append(float(meta[3]))
    stats = {
        "meta_dim": META_DIM,
        "quality_physics_mean": float(np.mean(quality_values)) if quality_values else 0.0,
        "drop_source_identity": bool(drop_source_identity),
    }
    return sample_map, event_map, stats


class PhysicsPriorDataset(Dataset):
    def __init__(
        self,
        base_ds: Dataset,
        sample_physics: dict[str, np.ndarray],
        event_physics: dict[str, np.ndarray],
        sample_meta: dict[str, np.ndarray],
        event_meta: dict[str, np.ndarray],
        physics_mean: np.ndarray,
        physics_std: np.ndarray,
    ) -> None:
        self.base_ds = base_ds
        self.sample_physics = sample_physics
        self.event_physics = event_physics
        self.sample_meta = sample_meta
        self.event_meta = event_meta
        self.physics_mean = physics_mean.astype(np.float32)
        self.physics_std = physics_std.astype(np.float32)
        self.zero_physics = np.zeros_like(self.physics_mean, dtype=np.float32)
        self.zero_meta = np.zeros((META_DIM,), dtype=np.float32)
        self.dataset_ids = list(getattr(base_ds, "dataset_ids", []))
        self.dataset_counter = getattr(base_ds, "dataset_counter", Counter())

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        item = self.base_ds[idx]
        sample_id = item["sample_id"]
        event_uid = item["event_uid"]
        physics_raw = self.sample_physics.get(sample_id, self.event_physics.get(event_uid, self.zero_physics))
        physics_norm = np.nan_to_num((physics_raw - self.physics_mean) / self.physics_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        meta = self.sample_meta.get(sample_id, self.event_meta.get(event_uid, self.zero_meta))
        item["physics"] = torch.from_numpy(physics_norm)
        item["physics_raw"] = torch.from_numpy(physics_raw.astype(np.float32, copy=False))
        item["meta"] = torch.from_numpy(meta.astype(np.float32, copy=False))
        return item


class QualityMaskSidecar:
    def __init__(self, h5_path: Path, mask_key: str) -> None:
        self.h5_path = Path(h5_path)
        self.mask_key = mask_key
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            sample_ids = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in f["sample_id"][:]]
        self.sample_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def get_mask(self, sample_id: str) -> np.ndarray | None:
        idx = self.sample_to_index.get(sample_id)
        if idx is None:
            return None
        return np.asarray(self._get_h5()[self.mask_key][idx], dtype=np.float32)


class QualityMaskedDataset(Dataset):
    def __init__(
        self,
        base_ds: Dataset,
        sidecar: QualityMaskSidecar,
        dataset_filters: set[str] | None = None,
    ) -> None:
        self.base_ds = base_ds
        self.sidecar = sidecar
        self.dataset_filters = dataset_filters or set()
        self.dataset_ids = list(getattr(base_ds, "dataset_ids", []))
        self.dataset_counter = getattr(base_ds, "dataset_counter", Counter())

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        item = self.base_ds[idx]
        dataset_id = str(item["dataset_id"])
        if self.dataset_filters and dataset_id not in self.dataset_filters:
            return item
        sample_id = str(item["sample_id"])
        qmask = self.sidecar.get_mask(sample_id)
        if qmask is None:
            return item
        if isinstance(item["valid"], torch.Tensor):
            qmask_t = torch.from_numpy(qmask.astype(np.float32, copy=False))
            item["valid"] = item["valid"] * (1.0 - qmask_t)
        else:
            item["valid"] = item["valid"] * (1.0 - qmask.astype(np.float32, copy=False))
        return item


def unwrap_base_dataset(ds: Dataset) -> Dataset:
    cur = ds
    while hasattr(cur, "base_ds"):
        cur = getattr(cur, "base_ds")
    return cur


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, token_dim: int) -> None:
        super().__init__()
        hidden = max(token_dim, in_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return F.gelu(out + x)


class PhysicsRegionalizer(nn.Module):
    def __init__(self, token_dim: int, latent_dim: int) -> None:
        super().__init__()
        cond_dim = token_dim * 5
        hidden = token_dim * 4
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim + 2),
        )

    def forward(
        self,
        terrain_token: torch.Tensor,
        material_token: torch.Tensor,
        trigger_token: torch.Tensor,
        proxy_token: torch.Tensor,
        meta_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = torch.cat([terrain_token, material_token, trigger_token, proxy_token, meta_token], dim=1)
        out = self.net(cond)
        latent, support, uncertainty = torch.split(out, [out.shape[1] - 2, 1, 1], dim=1)
        return latent, torch.sigmoid(support), torch.sigmoid(uncertainty)


class PhysicsPriorDeepLabV3(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained_backbone: bool,
        token_dim: int = 96,
        latent_dim: int = 128,
        prior_fusion_scale: float = 1.0,
        interaction_scale: float = 0.25,
        artifact_scale: float = 0.0,
        body_anchor_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.prior_fusion_scale = float(prior_fusion_scale)
        self.interaction_scale = float(interaction_scale)
        self.artifact_scale = float(artifact_scale)
        self.body_anchor_scale = float(body_anchor_scale)
        self.visual = BinaryDeepLabV3(
            in_channels=3,
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            aux_loss=True,
        )
        feat_ch = 2048
        self.visual_proj = nn.Sequential(
            nn.Conv2d(feat_ch, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResidualConvBlock(256),
        )
        self.terrain_encoder = VectorEncoder(len(PHYSICS_SPLITS["terrain"]), token_dim)
        self.material_encoder = VectorEncoder(len(PHYSICS_SPLITS["material"]), token_dim)
        self.trigger_encoder = VectorEncoder(len(PHYSICS_SPLITS["trigger"]), token_dim)
        self.proxy_encoder = VectorEncoder(len(PHYSICS_SPLITS["proxy"]), token_dim)
        self.meta_encoder = VectorEncoder(META_DIM, token_dim)
        self.regionalizer = PhysicsRegionalizer(token_dim=token_dim, latent_dim=latent_dim)
        self.prior_gamma = nn.Linear(latent_dim, 256)
        self.prior_beta = nn.Linear(latent_dim, 256)
        self.prior_head = nn.Sequential(
            ConvBNAct(256, 128),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.interaction_head = nn.Sequential(
            ConvBNAct(512, 256),
            ResidualConvBlock(256),
            nn.Conv2d(256, 1, kernel_size=1),
        )
        self.body_anchor_head = nn.Sequential(
            ConvBNAct(256 + 256 + 2, 128),
            ResidualConvBlock(128),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.artifact_head = nn.Sequential(
            ConvBNAct(256 + 3 + 2, 128),
            ResidualConvBlock(128),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        gate_in_dim = 256 + token_dim * 5 + 2
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.prior_head[-1].weight)
        nn.init.zeros_(self.prior_head[-1].bias)
        nn.init.zeros_(self.interaction_head[-1].weight)
        nn.init.zeros_(self.interaction_head[-1].bias)
        nn.init.zeros_(self.body_anchor_head[-1].weight)
        nn.init.zeros_(self.body_anchor_head[-1].bias)
        nn.init.zeros_(self.artifact_head[-1].weight)
        nn.init.zeros_(self.artifact_head[-1].bias)

    def split_physics(self, physics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t0 = len(PHYSICS_SPLITS["terrain"])
        t1 = t0 + len(PHYSICS_SPLITS["material"])
        t2 = t1 + len(PHYSICS_SPLITS["trigger"])
        return physics[:, :t0], physics[:, t0:t1], physics[:, t1:t2], physics[:, t2:]

    def forward(
        self,
        image: torch.Tensor,
        physics: torch.Tensor,
        meta: torch.Tensor,
        body_anchor_scale: float | None = None,
    ) -> dict[str, torch.Tensor]:
        terrain, material, trigger, proxy = self.split_physics(physics)
        features = self.visual.model.backbone(image)
        feat = features["out"]
        visual_logits = self.visual.model.classifier(feat)
        aux_logits = None
        if self.visual.model.aux_classifier is not None and "aux" in features:
            aux_logits = self.visual.model.aux_classifier(features["aux"])

        visual_feat = self.visual_proj(feat)
        terrain_token = self.terrain_encoder(terrain)
        material_token = self.material_encoder(material)
        trigger_token = self.trigger_encoder(trigger)
        proxy_token = self.proxy_encoder(proxy)
        meta_token = self.meta_encoder(meta)
        latent, support, uncertainty = self.regionalizer(
            terrain_token=terrain_token,
            material_token=material_token,
            trigger_token=trigger_token,
            proxy_token=proxy_token,
            meta_token=meta_token,
        )
        gamma = torch.tanh(self.prior_gamma(latent)).unsqueeze(-1).unsqueeze(-1)
        beta = self.prior_beta(latent).unsqueeze(-1).unsqueeze(-1)
        prior_feat = visual_feat * (1.0 + 0.1 * gamma) + beta
        prior_logits = self.prior_head(prior_feat)
        interaction_logits = self.interaction_head(torch.cat([visual_feat, prior_feat], dim=1))
        support_map = support.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, visual_feat.shape[-2], visual_feat.shape[-1])
        uncertainty_map = uncertainty.unsqueeze(-1).unsqueeze(-1).expand_as(support_map)
        body_anchor_logits = self.body_anchor_head(torch.cat([visual_feat, prior_feat, support_map, uncertainty_map], dim=1))
        body_anchor_logits_lowres = body_anchor_logits
        image_lr = F.interpolate(image, size=visual_feat.shape[-2:], mode="bilinear", align_corners=False)
        artifact_logits = self.artifact_head(torch.cat([visual_feat, image_lr, support_map, uncertainty_map], dim=1))

        pooled = F.adaptive_avg_pool2d(visual_feat, output_size=1).flatten(1)
        gate = self.gate(torch.cat([pooled, terrain_token, material_token, trigger_token, proxy_token, meta_token, support, uncertainty], dim=1))
        fusion = self.prior_fusion_scale * gate.unsqueeze(-1).unsqueeze(-1) * (1.0 - uncertainty.unsqueeze(-1).unsqueeze(-1)) * support.unsqueeze(-1).unsqueeze(-1) * prior_logits
        anchor_scale = self.body_anchor_scale if body_anchor_scale is None else float(body_anchor_scale)
        body_anchor_context = support_map * (1.0 - 0.5 * uncertainty_map)
        body_anchor_gain = anchor_scale * body_anchor_context * F.relu(body_anchor_logits)
        artifact_context = 0.5 * (1.0 - support_map) + 0.5 * uncertainty_map
        artifact_penalty = self.artifact_scale * artifact_context * torch.sigmoid(artifact_logits)
        logits = visual_logits + fusion + self.interaction_scale * interaction_logits + body_anchor_gain - artifact_penalty
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        visual_logits = F.interpolate(visual_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        prior_logits = F.interpolate(prior_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        body_anchor_logits = F.interpolate(body_anchor_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        body_anchor_gain = F.interpolate(body_anchor_gain, size=image.shape[-2:], mode="bilinear", align_corners=False)
        artifact_logits = F.interpolate(artifact_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if aux_logits is not None:
            aux_logits = F.interpolate(aux_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return {
            "logits": logits,
            "visual_logits": visual_logits,
            "prior_logits": prior_logits,
            "body_anchor_logits_lowres": body_anchor_logits_lowres,
            "body_anchor_logits": body_anchor_logits,
            "body_anchor_gain": body_anchor_gain,
            "artifact_logits": artifact_logits,
            "aux_logits": aux_logits,
            "support": support,
            "uncertainty": uncertainty,
            "gate": gate,
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
    train_prior: float
    val_prior: float
    train_uncert: float
    val_uncert: float
    train_distill: float
    val_distill: float
    train_completion: float
    val_completion: float
    train_artifact: float
    val_artifact: float
    train_gate_mean: float
    val_gate_mean: float
    lr: float
    sec: float


def compute_prior_target(physics_raw: torch.Tensor) -> torch.Tensor:
    terrain_dim = len(PHYSICS_SPLITS["terrain"])
    material_dim = len(PHYSICS_SPLITS["material"])
    trigger_dim = len(PHYSICS_SPLITS["trigger"])
    terrain = physics_raw[:, :terrain_dim]
    material = physics_raw[:, terrain_dim : terrain_dim + material_dim]
    trigger = physics_raw[:, terrain_dim + material_dim : terrain_dim + material_dim + trigger_dim]
    proxy = physics_raw[:, -len(PHYSICS_SPLITS["proxy"]) :]

    slope = terrain[:, 1:2]
    wc_bare = material[:, 2:3]
    clay = material[:, 3:4]
    sand = material[:, 4:5]
    lith = material[:, 8:9]
    smap = trigger[:, 8:9]
    trigger_dyn = trigger[:, 9:10] + trigger[:, 10:11]
    hydro = proxy[:, 0:1]
    target = 0.30 * slope + 0.22 * hydro + 0.10 * smap + 0.10 * clay + 0.08 * wc_bare + 0.10 * lith + 0.10 * trigger_dyn - 0.03 * sand
    return target.clamp(0.0, 1.0)


def compute_prior_loss(prior_logits: torch.Tensor, physics_raw: torch.Tensor) -> torch.Tensor:
    target = compute_prior_target(physics_raw)
    pred = torch.sigmoid(prior_logits).flatten(1).mean(dim=1, keepdim=True)
    return F.smooth_l1_loss(pred, target)


def compute_uncert_loss(uncertainty: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    target = (1.0 - meta[:, 3:4]).clamp(0.0, 1.0)
    return F.smooth_l1_loss(uncertainty, target)


def compute_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits))


def compute_masked_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits), reduction="none")
    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom


def compute_artifact_proxy_mask(
    image: torch.Tensor,
    valid: torch.Tensor,
    *,
    bright_threshold: float = 0.72,
    bright_chroma_threshold: float = 0.08,
    white_threshold: float = 0.90,
    white_chroma_threshold: float = 0.04,
) -> torch.Tensor:
    rgb = image[:, :3]
    luminance = rgb.mean(dim=1, keepdim=True)
    chroma = rgb.std(dim=1, keepdim=True, unbiased=False)
    bright_low_chroma = ((luminance > bright_threshold) & (chroma < bright_chroma_threshold)).float()
    white_low_chroma = ((luminance > white_threshold) & (chroma < white_chroma_threshold)).float()
    proxy = torch.clamp(bright_low_chroma + white_low_chroma, 0.0, 1.0)
    return proxy * valid


def compute_white_blank_invalid_mask(
    image: torch.Tensor,
    valid: torch.Tensor,
    *,
    white_threshold: float = 0.90,
    white_chroma_threshold: float = 0.04,
) -> torch.Tensor:
    rgb = image[:, :3]
    luminance = rgb.mean(dim=1, keepdim=True)
    chroma = rgb.std(dim=1, keepdim=True, unbiased=False)
    white_low_chroma = ((luminance > white_threshold) & (chroma < white_chroma_threshold)).float()
    return white_low_chroma * valid


def compute_artifact_proxy_target(
    image: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    bright_threshold: float = 0.72,
    bright_chroma_threshold: float = 0.08,
    white_threshold: float = 0.90,
    white_chroma_threshold: float = 0.04,
) -> torch.Tensor:
    target = (mask >= 0.5).float() * valid
    proxy = compute_artifact_proxy_mask(
        image=image,
        valid=valid,
        bright_threshold=bright_threshold,
        bright_chroma_threshold=bright_chroma_threshold,
        white_threshold=white_threshold,
        white_chroma_threshold=white_chroma_threshold,
    )
    # Penalize only negative pixels; keep uncertain positive GT interiors out of the proxy target.
    return proxy * (1.0 - target) * valid


def compute_artifact_loss(
    artifact_logits: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    target = compute_artifact_proxy_target(image=image, mask=mask, valid=valid)
    loss = F.binary_cross_entropy_with_logits(artifact_logits, target, reduction="none")
    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom


def build_body_anchor_target(
    mask: torch.Tensor,
    valid: torch.Tensor,
    output_size: tuple[int, int],
    *,
    pos_threshold: float = 0.30,
    neg_threshold: float = 0.05,
    valid_threshold: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = (mask >= 0.5).float() * valid
    target_low = F.adaptive_avg_pool2d(target, output_size)
    valid_low = F.adaptive_avg_pool2d(valid, output_size)
    supervised = valid_low >= float(valid_threshold)
    pos_cells = (target_low >= float(pos_threshold)) & supervised
    neg_cells = (target_low <= float(neg_threshold)) & supervised
    target_bin = pos_cells.float()
    weight = (pos_cells | neg_cells).float()
    return target_low, target_bin, weight, valid_low


def compute_body_anchor_loss(
    body_anchor_logits: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    pos_threshold: float = 0.30,
    neg_threshold: float = 0.05,
    valid_threshold: float = 0.75,
    body_ratio_floor: float = 0.02,
    area_margin: float = 0.80,
    area_weight: float = 1.0,
) -> torch.Tensor:
    target_low, target_bin, weight, valid_low = build_body_anchor_target(
        mask=mask,
        valid=valid,
        output_size=body_anchor_logits.shape[-2:],
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
        valid_threshold=valid_threshold,
    )

    bce_map = F.binary_cross_entropy_with_logits(body_anchor_logits, target_bin, reduction="none")
    if weight.sum() > 0:
        coarse_bce = (bce_map * weight).sum() / weight.sum().clamp_min(1.0)
    else:
        coarse_bce = body_anchor_logits.new_zeros(())

    probs = torch.sigmoid(body_anchor_logits)
    supervised = (weight > 0).float()
    supervised_counts = supervised.flatten(1).sum(dim=1)
    gt_ratio_low = torch.where(
        supervised_counts > 0,
        (target_low * supervised).flatten(1).sum(dim=1) / supervised_counts.clamp_min(1.0),
        body_anchor_logits.new_zeros((body_anchor_logits.shape[0],)),
    )
    pred_ratio_low = torch.where(
        supervised_counts > 0,
        (probs * supervised).flatten(1).sum(dim=1) / supervised_counts.clamp_min(1.0),
        body_anchor_logits.new_zeros((body_anchor_logits.shape[0],)),
    )

    valid_pixels = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    gt_ratio_full = ((mask >= 0.5).float() * valid).flatten(1).sum(dim=1) / valid_pixels
    large_body = (gt_ratio_full >= float(body_ratio_floor)) & (valid_low.flatten(1).mean(dim=1) >= float(valid_threshold))
    if large_body.any():
        area_floor = torch.clamp(gt_ratio_low[large_body] * float(area_margin), min=0.0, max=1.0)
        area_loss = F.relu(area_floor - pred_ratio_low[large_body]).mean()
    else:
        area_loss = body_anchor_logits.new_zeros(())
    return coarse_bce + float(area_weight) * area_loss


def compute_completion_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    mode: str = "legacy",
    pool_kernel: int = 8,
    area_weight: float = 1.0,
    body_ratio_floor: float = 0.02,
    area_margin: float = 0.80,
    cell_target_floor: float = 0.25,
) -> torch.Tensor:
    target = (mask >= 0.5).float() * valid
    probs = torch.sigmoid(logits) * valid
    k = max(1, int(pool_kernel))

    if mode == "body_recall":
        pred_low = F.avg_pool2d(probs, kernel_size=k, stride=k)
        target_low = F.avg_pool2d(target, kernel_size=k, stride=k)
        pos_cells = target_low >= float(cell_target_floor)
        if pos_cells.any():
            cell_loss = F.relu(target_low[pos_cells] - pred_low[pos_cells]).mean()
        else:
            cell_loss = logits.new_zeros(())

        valid_pixels = valid.flatten(1).sum(dim=1).clamp_min(1.0)
        gt_ratio = target.flatten(1).sum(dim=1) / valid_pixels
        pred_ratio = probs.flatten(1).sum(dim=1) / valid_pixels
        positive = gt_ratio >= float(body_ratio_floor)
        if positive.any():
            target_ratio = gt_ratio[positive] * float(area_margin)
            area_loss = F.relu(target_ratio - pred_ratio[positive]).mean()
        else:
            area_loss = logits.new_zeros(())
        return cell_loss + float(area_weight) * area_loss

    pred_low = F.avg_pool2d(probs, kernel_size=k, stride=k)
    target_low = F.max_pool2d(target, kernel_size=k, stride=k)
    occ_loss = F.binary_cross_entropy(pred_low.clamp(1e-4, 1.0 - 1e-4), target_low, reduction="mean")

    valid_pixels = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    gt_ratio = target.flatten(1).sum(dim=1) / valid_pixels
    pred_ratio = probs.flatten(1).sum(dim=1) / valid_pixels
    positive = gt_ratio >= float(body_ratio_floor)
    if positive.any():
        area_loss = F.relu(gt_ratio[positive] - pred_ratio[positive]).mean()
    else:
        area_loss = logits.new_zeros(())
    return occ_loss + float(area_weight) * area_loss


def load_teacher_state(path: Path, backbone_name: str, device: torch.device) -> BinaryDeepLabV3:
    model = BinaryDeepLabV3(
        in_channels=3,
        backbone_name=backbone_name,
        pretrained_backbone=False,
        aux_loss=True,
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def maybe_init_student_from_teacher(model: PhysicsPriorDeepLabV3, teacher_ckpt: Path | None, device: torch.device) -> bool:
    if teacher_ckpt is None or not teacher_ckpt.exists():
        return False
    ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.visual.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected teacher keys during student init: {unexpected}")
    if missing:
        missing = [key for key in missing if not key.startswith("model.aux_classifier")]
        if missing:
            raise RuntimeError(f"missing teacher keys during student init: {missing}")
    return True


def resolve_teacher_ckpt(root: Path, teacher_ckpt_arg: str, backbone_name: str) -> Path | None:
    if teacher_ckpt_arg.strip():
        return Path(teacher_ckpt_arg)
    if backbone_name == "deeplabv3_resnet50":
        candidates = [
            root / "experiments" / "strict_t2_postrgb_deeplabv3_resnet50_visual_e3_sp05_lb025_thr_v2_localenv" / "best_model.pt",
            root / "experiments" / "strict_t2_postrgb_deeplabv3_resnet50_visual_e3_sp05_lb025_thr_v1_localenv" / "best_model.pt",
        ]
    else:
        candidates = []
    for path in candidates:
        if path.exists():
            return path
    return None


def run_epoch_v4(
    model: PhysicsPriorDeepLabV3,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    teacher_model: BinaryDeepLabV3 | None = None,
    max_steps: int = 0,
    bce_pos_weight: float = 1.0,
    dataset_loss_weights: dict[str, float] | None = None,
    threshold: float = 0.5,
    aux_loss_weight: float = 0.2,
    lambda_prior: float = 0.0,
    lambda_uncert: float = 0.0,
    lambda_distill: float = 0.0,
    lambda_completion: float = 0.0,
    lambda_artifact: float = 0.0,
    completion_mode: str = "legacy",
    completion_pool_kernel: int = 8,
    body_anchor_scale: float = 0.0,
    body_anchor_pos_threshold: float = 0.30,
    body_anchor_neg_threshold: float = 0.05,
    body_anchor_valid_threshold: float = 0.75,
    body_anchor_ratio_floor: float = 0.02,
    body_anchor_area_margin: float = 0.80,
    artifact_invalid_policy: str = "none",
    artifact_invalid_source: str = "white_blank",
    zero_physics_input: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_prior = 0.0
    total_uncert = 0.0
    total_distill = 0.0
    total_completion = 0.0
    total_artifact = 0.0
    total_gate = 0.0
    total_items = 0
    agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    by_dataset: dict[str, dict[str, float]] = {}

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=False)
            physics = batch["physics"].to(device, non_blocking=False)
            physics_raw = batch["physics_raw"].to(device, non_blocking=False)
            meta = batch["meta"].to(device, non_blocking=False)
            if zero_physics_input:
                physics = torch.zeros_like(physics)
                physics_raw = torch.zeros_like(physics_raw)
            mask = batch["mask"].to(device, non_blocking=False)
            valid = batch["valid"].to(device, non_blocking=False)
            target = (mask >= 0.5).float()
            if artifact_invalid_source == "artifact_proxy":
                artifact_invalid = compute_artifact_proxy_mask(image=image, valid=valid) * (1.0 - target)
            else:
                artifact_invalid = compute_white_blank_invalid_mask(image=image, valid=valid) * (1.0 - target)
            use_artifact_invalid = artifact_invalid_policy == "all" or (artifact_invalid_policy == "train_only" and train)
            effective_valid = valid * (1.0 - artifact_invalid) if use_artifact_invalid else valid

            out = model(image=image, physics=physics, meta=meta, body_anchor_scale=body_anchor_scale)
            logits = out["logits"]
            sample_weights = batch_dataset_weights(
                batch["dataset_id"],
                dataset_loss_weights=dataset_loss_weights if train else None,
                device=logits.device,
                dtype=logits.dtype,
            )
            bce, dice, seg_loss = compute_segmentation_loss(
                logits,
                mask,
                effective_valid,
                pos_weight=bce_pos_weight,
                sample_weights=sample_weights,
            )
            loss = seg_loss
            if out["aux_logits"] is not None and aux_loss_weight > 0.0:
                _, _, aux_loss = compute_segmentation_loss(
                    out["aux_logits"],
                    mask,
                    effective_valid,
                    pos_weight=bce_pos_weight,
                    sample_weights=sample_weights,
                )
                loss = loss + float(aux_loss_weight) * aux_loss

            prior_loss = compute_prior_loss(out["prior_logits"], physics_raw)
            uncert_loss = compute_uncert_loss(out["uncertainty"], meta)
            if completion_mode == "body_anchor":
                completion_loss = compute_body_anchor_loss(
                    out["body_anchor_logits_lowres"],
                    mask,
                    effective_valid,
                    pos_threshold=body_anchor_pos_threshold,
                    neg_threshold=body_anchor_neg_threshold,
                    valid_threshold=body_anchor_valid_threshold,
                    body_ratio_floor=body_anchor_ratio_floor,
                    area_margin=body_anchor_area_margin,
                )
            else:
                completion_loss = compute_completion_loss(
                    logits,
                    mask,
                    effective_valid,
                    mode=completion_mode,
                    pool_kernel=completion_pool_kernel,
                )
            artifact_loss = compute_artifact_loss(
                out["artifact_logits"],
                image=image,
                mask=mask,
                valid=valid,
            )
            loss = (
                loss
                + float(lambda_prior) * prior_loss
                + float(lambda_uncert) * uncert_loss
                + float(lambda_completion) * completion_loss
                + float(lambda_artifact) * artifact_loss
            )

            distill_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
            if teacher_model is not None and lambda_distill > 0.0:
                teacher_logits, _ = teacher_model(image)
                distill_loss = compute_masked_distill_loss(logits, teacher_logits.detach(), effective_valid)
                loss = loss + float(lambda_distill) * distill_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            logits_detached = logits.detach()
            metric_valid = effective_valid if artifact_invalid_policy == "all" else valid
            stats = masked_stats(logits_detached, mask, metric_valid, threshold=threshold)
            bsz = image.size(0)
            total_loss += float(loss.item()) * bsz
            total_bce += float(bce.item()) * bsz
            total_dice += float(dice.item()) * bsz
            total_prior += float(prior_loss.item()) * bsz
            total_uncert += float(uncert_loss.item()) * bsz
            total_distill += float(distill_loss.item()) * bsz
            total_completion += float(completion_loss.item()) * bsz
            total_artifact += float(artifact_loss.item()) * bsz
            total_gate += float(out["gate"].mean().item()) * bsz
            total_items += bsz
            agg["tp"] += stats["tp"]
            agg["fp"] += stats["fp"]
            agg["fn"] += stats["fn"]
            for i, ds_name in enumerate(batch["dataset_id"]):
                item_stats = masked_stats(
                    logits_detached[i : i + 1],
                    mask[i : i + 1],
                    metric_valid[i : i + 1],
                    threshold=threshold,
                )
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
        "prior": total_prior / max(total_items, 1),
        "uncert": total_uncert / max(total_items, 1),
        "distill": total_distill / max(total_items, 1),
        "completion": total_completion / max(total_items, 1),
        "artifact": total_artifact / max(total_items, 1),
        "gate_mean": total_gate / max(total_items, 1),
        **_finish(agg),
    }
    dataset_metrics = {name: _finish(stats) | {"samples": stats["n"]} for name, stats in by_dataset.items()}
    return overall, dataset_metrics


def find_best_threshold_v4(
    model: PhysicsPriorDeepLabV3,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    max_steps: int = 0,
    body_anchor_scale: float = 0.0,
    zero_physics_input: bool = False,
) -> tuple[float, list[dict[str, float]]]:
    model.eval()
    aggs = {thr: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for thr in thresholds}
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=False)
            physics = batch["physics"].to(device, non_blocking=False)
            meta = batch["meta"].to(device, non_blocking=False)
            if zero_physics_input:
                physics = torch.zeros_like(physics)
            mask = batch["mask"].to(device, non_blocking=False)
            valid = batch["valid"].to(device, non_blocking=False)
            probs = torch.sigmoid(model(image=image, physics=physics, meta=meta, body_anchor_scale=body_anchor_scale)["logits"])
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
    p = argparse.ArgumentParser(description="Train strict_t2 post_rgb v4/v5 pilot")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--manifest", default="", help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_manifest_post_rgb.csv")
    p.add_argument("--physics-csv", default="", help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_physics_vectors_post_rgb_v1.csv")
    p.add_argument("--outdir", default="", help="default: experiments/strict_t2_postrgb_v4_pilot_stage0")
    p.add_argument("--teacher-ckpt", default="", help="default: strongest available strict_t2 post_rgb visual checkpoint")
    p.add_argument("--eval-only-ckpt", default="", help="optional checkpoint to evaluate without further training")
    p.add_argument("--backbone", default="deeplabv3_resnet50", choices=["deeplabv3_resnet50", "deeplabv3_resnet101"])
    p.add_argument("--no-pretrained-backbone", action="store_true")
    p.add_argument("--init-from-teacher-visual", action="store_true")
    p.add_argument("--token-dim", type=int, default=96)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--prior-fusion-scale", type=float, default=1.0)
    p.add_argument("--interaction-scale", type=float, default=0.25)
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
    p.add_argument("--lambda-prior", type=float, default=0.05)
    p.add_argument("--lambda-uncert", type=float, default=0.02)
    p.add_argument("--lambda-distill", type=float, default=0.05)
    p.add_argument("--lambda-completion", type=float, default=0.0)
    p.add_argument("--lambda-artifact", type=float, default=0.0)
    p.add_argument("--completion-mode", default="legacy", choices=["legacy", "body_recall", "body_anchor"])
    p.add_argument("--completion-pool-kernel", type=int, default=8)
    p.add_argument("--artifact-scale", type=float, default=0.0)
    p.add_argument("--body-anchor-scale", type=float, default=0.0)
    p.add_argument("--body-anchor-warmup-epochs", type=int, default=0)
    p.add_argument("--body-anchor-pos-threshold", type=float, default=0.30)
    p.add_argument("--body-anchor-neg-threshold", type=float, default=0.05)
    p.add_argument("--body-anchor-valid-threshold", type=float, default=0.75)
    p.add_argument("--body-anchor-ratio-floor", type=float, default=0.02)
    p.add_argument("--body-anchor-area-margin", type=float, default=0.80)
    p.add_argument("--artifact-invalid-policy", default="none", choices=["none", "train_only", "all"])
    p.add_argument("--artifact-invalid-source", default="white_blank", choices=["white_blank", "artifact_proxy"])
    p.add_argument("--quality-mask-sidecar-dir", default="", help="optional directory containing split-aligned *_qualitymask.h5 files")
    p.add_argument("--quality-mask-key", default="", help="dataset key inside the quality-mask sidecar h5; empty disables sidecar masking")
    p.add_argument("--quality-mask-datasets", default="", help="comma-separated dataset_id values to apply sidecar masks to")
    p.add_argument("--quality-mask-apply-to", default="train,val,test", help="comma-separated split names from train,val,test")
    p.add_argument("--drop-source-identity", action="store_true", help="zero the dataset one-hot component in the heterogeneity metadata vector")
    p.add_argument("--zero-physics-input", action="store_true", help="zero the normalized and raw physics vectors while keeping heterogeneity metadata active")
    p.add_argument("--seed", type=int, default=20260311)
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
    physics_csv = (
        Path(args.physics_csv)
        if args.physics_csv.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_physics_vectors_post_rgb_v1.csv"
    )
    if args.completion_mode == "body_anchor":
        default_outdir = f"strict_t2_postrgb_{args.backbone}_v5_bodyanchor_stage0"
    else:
        default_outdir = f"strict_t2_postrgb_{args.backbone}_v4_pilot_stage0"
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
    teacher_ckpt = resolve_teacher_ckpt(root, args.teacher_ckpt, args.backbone)
    eval_only_ckpt = Path(args.eval_only_ckpt) if args.eval_only_ckpt.strip() else None
    sampler_overrides = parse_named_value_overrides(args.sampler_overrides)
    loss_balance_overrides = parse_named_value_overrides(args.loss_balance_overrides)
    threshold_grid = parse_threshold_grid(args.threshold_grid)
    quality_mask_dir = (
        Path(args.quality_mask_sidecar_dir)
        if args.quality_mask_sidecar_dir.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_postrgb_qualitymask_sidecar_v1"
    )
    quality_mask_datasets = {item.strip() for item in args.quality_mask_datasets.split(",") if item.strip()}
    quality_mask_apply_to = {item.strip() for item in args.quality_mask_apply_to.split(",") if item.strip()}

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
    print(f"[info] physics_csv={physics_csv}")
    print(f"[info] teacher_ckpt={teacher_ckpt if teacher_ckpt is not None else 'missing'}")
    print(f"[info] eval_only_ckpt={eval_only_ckpt if eval_only_ckpt is not None else 'disabled'}")
    print(f"[info] backbone={args.backbone}")
    print(f"[info] pretrained_backbone={not args.no_pretrained_backbone}")
    print(f"[info] init_from_teacher_visual={args.init_from_teacher_visual}")
    print(f"[info] drop_source_identity={args.drop_source_identity}")
    print(f"[info] zero_physics_input={args.zero_physics_input}")
    print(
        "[info] failure_mode_losses="
        f"completion:{args.lambda_completion} artifact:{args.lambda_artifact} "
        f"artifact_scale:{args.artifact_scale} completion_mode:{args.completion_mode} "
        f"completion_pool:{args.completion_pool_kernel} "
        f"body_anchor_scale:{args.body_anchor_scale} body_anchor_warmup:{args.body_anchor_warmup_epochs} "
        f"artifact_invalid_policy:{args.artifact_invalid_policy} artifact_invalid_source:{args.artifact_invalid_source}"
    )
    print(
        "[info] quality_mask="
        f"key:{args.quality_mask_key or 'disabled'} dir:{quality_mask_dir if args.quality_mask_key else 'n/a'} "
        f"datasets:{sorted(quality_mask_datasets) if quality_mask_datasets else 'all'} "
        f"apply_to:{sorted(quality_mask_apply_to) if args.quality_mask_key else '[]'}"
    )
    print(f"[info] gdcld_index={gdcld_index_path}")
    print(f"[info] train_cache_h5={train_cache_h5 if train_cache_h5.exists() else 'missing'}")
    print(f"[info] val_cache_h5={val_cache_h5 if val_cache_h5.exists() else 'missing'}")
    print(f"[info] test_cache_h5={test_cache_h5 if test_cache_h5.exists() else 'missing'}")
    print(f"[info] rows train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"[info] use_cache train/val/test={use_train_cache}/{use_val_cache}/{use_test_cache}")

    gdcld_index = read_gdcld_index(gdcld_index_path)
    if use_train_cache:
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
    if use_val_cache:
        val_base = CachedPostRgbH5Dataset(val_cache_h5, exclude_datasets=exclude_datasets)
    else:
        val_base = StrictT2PostRgbDataset(
            val_rows,
            args.patch_size,
            args.gdcld_crop_size,
            deterministic_scene_crop=True,
            gdcld_index=gdcld_index,
            gdcld_jitter=0,
        )
    if use_test_cache:
        test_base = CachedPostRgbH5Dataset(test_cache_h5, exclude_datasets=exclude_datasets)
    else:
        test_base = StrictT2PostRgbDataset(
            test_rows,
            args.patch_size,
            args.gdcld_crop_size,
            deterministic_scene_crop=True,
            gdcld_index=gdcld_index,
            gdcld_jitter=0,
        )

    if args.quality_mask_key:
        if not quality_mask_dir.exists():
            raise FileNotFoundError(f"quality mask sidecar dir not found: {quality_mask_dir}")
        split_to_base = {"train": train_base, "val": val_base, "test": test_base}
        for split_name, base_ds in list(split_to_base.items()):
            if split_name not in quality_mask_apply_to:
                continue
            sidecar_path = quality_mask_dir / f"{split_name}_postrgb_p{args.patch_size}_qualitymask.h5"
            if not sidecar_path.exists():
                raise FileNotFoundError(f"quality mask sidecar missing for split={split_name}: {sidecar_path}")
            wrapped = QualityMaskedDataset(
                base_ds=base_ds,
                sidecar=QualityMaskSidecar(sidecar_path, args.quality_mask_key),
                dataset_filters=quality_mask_datasets,
            )
            split_to_base[split_name] = wrapped
        train_base = split_to_base["train"]
        val_base = split_to_base["val"]
        test_base = split_to_base["test"]

    sample_physics, event_physics = load_physics_maps(physics_csv, PHYSICS_VECTOR_COLUMNS)
    sample_meta, event_meta, meta_stats = load_meta_maps(physics_csv, drop_source_identity=args.drop_source_identity)
    train_sample_ids, train_event_uids = metadata_from_base_dataset(unwrap_base_dataset(train_base))
    physics_mean, physics_std = gather_train_stats(train_sample_ids, train_event_uids, sample_physics, event_physics)

    train_ds = PhysicsPriorDataset(
        base_ds=train_base,
        sample_physics=sample_physics,
        event_physics=event_physics,
        sample_meta=sample_meta,
        event_meta=event_meta,
        physics_mean=physics_mean,
        physics_std=physics_std,
    )
    val_ds = PhysicsPriorDataset(
        base_ds=val_base,
        sample_physics=sample_physics,
        event_physics=event_physics,
        sample_meta=sample_meta,
        event_meta=event_meta,
        physics_mean=physics_mean,
        physics_std=physics_std,
    )
    test_ds = PhysicsPriorDataset(
        base_ds=test_base,
        sample_physics=sample_physics,
        event_physics=event_physics,
        sample_meta=sample_meta,
        event_meta=event_meta,
        physics_mean=physics_mean,
        physics_std=physics_std,
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
    print(f"[info] meta_stats={json.dumps(meta_stats, ensure_ascii=False)}")

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

    model = PhysicsPriorDeepLabV3(
        backbone_name=args.backbone,
        pretrained_backbone=not args.no_pretrained_backbone,
        token_dim=args.token_dim,
        latent_dim=args.latent_dim,
        prior_fusion_scale=args.prior_fusion_scale,
        interaction_scale=args.interaction_scale,
        artifact_scale=args.artifact_scale,
        body_anchor_scale=args.body_anchor_scale if args.completion_mode == "body_anchor" else 0.0,
    ).to(device)
    teacher_model = None
    if args.lambda_distill > 0.0:
        if teacher_ckpt is None or not teacher_ckpt.exists():
            raise FileNotFoundError("strict_t2 teacher checkpoint not found")
        teacher_model = load_teacher_state(teacher_ckpt, args.backbone, device)
    init_from_teacher = False
    if args.init_from_teacher_visual:
        if teacher_ckpt is None or not teacher_ckpt.exists():
            raise FileNotFoundError("strict_t2 teacher checkpoint not found for visual init")
        init_from_teacher = maybe_init_student_from_teacher(model, teacher_ckpt, device)

    history: list[EpochStat] = []
    best_epoch = 0
    best_iou: float | None = None
    if eval_only_ckpt is not None:
        if not eval_only_ckpt.exists():
            raise FileNotFoundError(f"eval-only checkpoint not found: {eval_only_ckpt}")
        ckpt = torch.load(eval_only_ckpt, map_location=device, weights_only=False)
        load_result = model.load_state_dict(ckpt["model"], strict=False)
        best_epoch = int(ckpt.get("epoch", 0))
        print(f"[info] loaded eval-only checkpoint from {eval_only_ckpt}")
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "[info] eval-only checkpoint load mismatch "
                f"missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
            )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        best_iou = -1.0
        best_model_path = outdir / "best_model.pt"

        for epoch in range(1, args.epochs + 1):
            current_body_anchor_scale = 0.0
            if args.completion_mode == "body_anchor" and args.body_anchor_scale > 0.0:
                if args.body_anchor_warmup_epochs <= 0:
                    current_body_anchor_scale = float(args.body_anchor_scale)
                else:
                    warmup_epochs = max(1, int(args.body_anchor_warmup_epochs))
                    current_body_anchor_scale = float(args.body_anchor_scale) * min(float(epoch) / float(warmup_epochs), 1.0)
            t0 = time.time()
            train_metrics, _ = run_epoch_v4(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                teacher_model=teacher_model,
                max_steps=args.max_train_steps,
                bce_pos_weight=args.bce_pos_weight,
                dataset_loss_weights=train_loss_weights,
                aux_loss_weight=args.aux_loss_weight,
                lambda_prior=args.lambda_prior,
                lambda_uncert=args.lambda_uncert,
                lambda_distill=args.lambda_distill,
                lambda_completion=args.lambda_completion,
                lambda_artifact=args.lambda_artifact,
                completion_mode=args.completion_mode,
                completion_pool_kernel=args.completion_pool_kernel,
                body_anchor_scale=current_body_anchor_scale,
                body_anchor_pos_threshold=args.body_anchor_pos_threshold,
                body_anchor_neg_threshold=args.body_anchor_neg_threshold,
                body_anchor_valid_threshold=args.body_anchor_valid_threshold,
                body_anchor_ratio_floor=args.body_anchor_ratio_floor,
                body_anchor_area_margin=args.body_anchor_area_margin,
                artifact_invalid_policy=args.artifact_invalid_policy,
                artifact_invalid_source=args.artifact_invalid_source,
                zero_physics_input=args.zero_physics_input,
            )
            val_metrics, val_by_dataset = run_epoch_v4(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                teacher_model=teacher_model,
                max_steps=args.max_eval_steps,
                bce_pos_weight=args.bce_pos_weight,
                threshold=0.5,
                aux_loss_weight=args.aux_loss_weight,
                lambda_prior=args.lambda_prior,
                lambda_uncert=args.lambda_uncert,
                lambda_distill=args.lambda_distill,
                lambda_completion=args.lambda_completion,
                lambda_artifact=args.lambda_artifact,
                completion_mode=args.completion_mode,
                completion_pool_kernel=args.completion_pool_kernel,
                body_anchor_scale=current_body_anchor_scale,
                body_anchor_pos_threshold=args.body_anchor_pos_threshold,
                body_anchor_neg_threshold=args.body_anchor_neg_threshold,
                body_anchor_valid_threshold=args.body_anchor_valid_threshold,
                body_anchor_ratio_floor=args.body_anchor_ratio_floor,
                body_anchor_area_margin=args.body_anchor_area_margin,
                artifact_invalid_policy=args.artifact_invalid_policy,
                artifact_invalid_source=args.artifact_invalid_source,
                zero_physics_input=args.zero_physics_input,
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
                    train_prior=train_metrics["prior"],
                    val_prior=val_metrics["prior"],
                    train_uncert=train_metrics["uncert"],
                    val_uncert=val_metrics["uncert"],
                    train_distill=train_metrics["distill"],
                    val_distill=val_metrics["distill"],
                    train_completion=train_metrics["completion"],
                    val_completion=val_metrics["completion"],
                    train_artifact=train_metrics["artifact"],
                    val_artifact=val_metrics["artifact"],
                    train_gate_mean=train_metrics["gate_mean"],
                    val_gate_mean=val_metrics["gate_mean"],
                    lr=float(scheduler.get_last_lr()[0]),
                    sec=sec,
                )
            )
            print(
                f"[epoch {epoch}] "
                f"train_loss={train_metrics['loss']:.4f} train_iou={train_metrics['iou']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['iou']:.4f} "
                f"prior={val_metrics['prior']:.4f} uncert={val_metrics['uncert']:.4f} "
                f"completion={val_metrics['completion']:.4f} artifact={val_metrics['artifact']:.4f} "
                f"distill={val_metrics['distill']:.4f} body_anchor_scale={current_body_anchor_scale:.3f} sec={sec:.1f}"
            )
            print(f"[epoch {epoch}] val_by_dataset={json.dumps(val_by_dataset, ensure_ascii=False)}")
            if val_metrics["iou"] > float(best_iou):
                best_iou = val_metrics["iou"]
                torch.save({"model": model.state_dict(), "epoch": epoch}, best_model_path)

        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_epoch = int(ckpt["epoch"])
    test_metrics_default, test_by_dataset_default = run_epoch_v4(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        teacher_model=teacher_model,
        max_steps=args.max_eval_steps,
        bce_pos_weight=args.bce_pos_weight,
        threshold=0.5,
        aux_loss_weight=args.aux_loss_weight,
        lambda_prior=args.lambda_prior,
        lambda_uncert=args.lambda_uncert,
        lambda_distill=args.lambda_distill,
        lambda_completion=args.lambda_completion,
        lambda_artifact=args.lambda_artifact,
        completion_mode=args.completion_mode,
        completion_pool_kernel=args.completion_pool_kernel,
        body_anchor_scale=args.body_anchor_scale if args.completion_mode == "body_anchor" else 0.0,
        body_anchor_pos_threshold=args.body_anchor_pos_threshold,
        body_anchor_neg_threshold=args.body_anchor_neg_threshold,
        body_anchor_valid_threshold=args.body_anchor_valid_threshold,
        body_anchor_ratio_floor=args.body_anchor_ratio_floor,
        body_anchor_area_margin=args.body_anchor_area_margin,
        artifact_invalid_policy=args.artifact_invalid_policy,
        artifact_invalid_source=args.artifact_invalid_source,
        zero_physics_input=args.zero_physics_input,
    )
    eval_threshold = 0.5
    threshold_search: list[dict[str, float]] = []
    if args.tune_threshold_on_val:
        eval_threshold, threshold_search = find_best_threshold_v4(
            model=model,
            loader=val_loader,
            device=device,
            thresholds=threshold_grid,
            max_steps=args.max_eval_steps,
            body_anchor_scale=args.body_anchor_scale if args.completion_mode == "body_anchor" else 0.0,
            zero_physics_input=args.zero_physics_input,
        )
        print(f"[info] tuned_eval_threshold={eval_threshold:.2f}")
    test_metrics = test_metrics_default
    test_by_dataset = test_by_dataset_default
    if abs(eval_threshold - 0.5) > 1e-8:
        test_metrics, test_by_dataset = run_epoch_v4(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=None,
            teacher_model=teacher_model,
            max_steps=args.max_eval_steps,
            bce_pos_weight=args.bce_pos_weight,
            threshold=eval_threshold,
            aux_loss_weight=args.aux_loss_weight,
            lambda_prior=args.lambda_prior,
            lambda_uncert=args.lambda_uncert,
            lambda_distill=args.lambda_distill,
            lambda_completion=args.lambda_completion,
            lambda_artifact=args.lambda_artifact,
            completion_mode=args.completion_mode,
            completion_pool_kernel=args.completion_pool_kernel,
            body_anchor_scale=args.body_anchor_scale if args.completion_mode == "body_anchor" else 0.0,
            body_anchor_pos_threshold=args.body_anchor_pos_threshold,
            body_anchor_neg_threshold=args.body_anchor_neg_threshold,
            body_anchor_valid_threshold=args.body_anchor_valid_threshold,
            body_anchor_ratio_floor=args.body_anchor_ratio_floor,
            body_anchor_area_margin=args.body_anchor_area_margin,
            artifact_invalid_policy=args.artifact_invalid_policy,
            artifact_invalid_source=args.artifact_invalid_source,
            zero_physics_input=args.zero_physics_input,
        )

    summary = {
        "manifest": str(manifest),
        "physics_csv": str(physics_csv),
        "outdir": str(outdir),
        "device": str(device),
        "model_family": "strict_t2_postrgb_v5_bodyanchor" if args.completion_mode == "body_anchor" else "strict_t2_postrgb_v4_pilot",
        "backbone": args.backbone,
        "pretrained_backbone": not args.no_pretrained_backbone,
        "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt is not None else "",
        "teacher_ckpt_exists": bool(teacher_ckpt.exists()) if teacher_ckpt is not None else False,
        "eval_only_ckpt": str(eval_only_ckpt) if eval_only_ckpt is not None else "",
        "init_from_teacher_visual": init_from_teacher,
        "drop_source_identity": bool(args.drop_source_identity),
        "zero_physics_input": bool(args.zero_physics_input),
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
        "meta_stats": meta_stats,
        "config": vars(args),
        "best_val_iou": best_iou,
        "best_epoch": best_epoch,
        "sampler_weight_map": sampler_weight_map,
        "train_loss_weight_map": train_loss_weights or {},
        "physics_mean": physics_mean.tolist(),
        "physics_std": physics_std.tolist(),
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
