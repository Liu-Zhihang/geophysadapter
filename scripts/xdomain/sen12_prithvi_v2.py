#!/usr/bin/env python3
"""Prithvi-EO-2.0 visual anchor with a role-pure Terrain-v2 adapter.

The visual path uses four S2 observations and the official temporal/location
encodings. Terrain never enters the visual encoder or decoder. It may only add
a bounded logit correction through ``BoundedTerrainAdapterV2`` so that aligned
and mismatched support remain meaningful attribution tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from sen12_terrain_v2 import (
    BoundedTerrainAdapterV2,
    NATIVE_TERRAIN_V2_SCALE_GROUPS,
)


PRITHVI_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
PRITHVI_MEAN = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
PRITHVI_STD = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)
DEFAULT_DEPTHS = (5, 11, 17, 23)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def default_prithvi_snapshot() -> Path:
    root = (
        Path.home()
        / ".cache/huggingface/hub/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M-TL/snapshots"
    )
    snapshots = sorted(path for path in root.glob("*") if path.is_dir())
    if len(snapshots) != 1:
        raise FileNotFoundError(f"expected exactly one Prithvi snapshot under {root}, found {snapshots}")
    return snapshots[0].resolve()


def load_source_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("prithvi_eo2_official", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official Prithvi source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prithvi_encoder(snapshot: Path | None = None) -> tuple[nn.Module, dict[str, str]]:
    snapshot = (snapshot or default_prithvi_snapshot()).resolve()
    source = snapshot / "prithvi_mae.py"
    config_path = snapshot / "config.json"
    checkpoint = snapshot / "Prithvi_EO_V2_300M_TL.pt"
    for path in (source, config_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))["pretrained_cfg"]
    module = load_source_module(source)
    encoder = module.PrithviViT(
        img_size=int(config["img_size"]),
        patch_size=tuple(config["patch_size"]),
        num_frames=int(config["num_frames"]),
        in_chans=int(config["in_chans"]),
        embed_dim=int(config["embed_dim"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        coords_encoding=list(config["coords_encoding"]),
        coords_scale_learn=bool(config["coords_scale_learn"]),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    encoder_state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Prithvi encoder load mismatch: missing={missing}, unexpected={unexpected}")
    provenance = {
        "snapshot": str(snapshot),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_sha256": sha256_file(source),
        "config_sha256": sha256_file(config_path),
        "bands": ";".join(PRITHVI_BANDS),
    }
    return encoder, provenance


class ConvNeXtDecodeBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(8, channels)
        self.expand = nn.Conv2d(channels, channels * 4, 1)
        self.contract = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(value)
        residual = self.norm(residual)
        residual = self.expand(residual)
        residual = F.gelu(residual)
        return value + self.contract(residual)


class MultiDepthTemporalChangeDecoder(nn.Module):
    """Fuse transformer depths, temporal states, and full-resolution spectra."""

    def __init__(self, embed_dim: int = 1024, width: int = 128, depths: int = 4) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(embed_dim * 3, width, 1),
                nn.GroupNorm(8, width),
                nn.GELU(),
            )
            for _ in range(depths)
        )
        self.depth_fusion = nn.Sequential(
            nn.Conv2d(width * depths, width * 2, 1),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
            ConvNeXtDecodeBlock(width * 2),
            nn.Conv2d(width * 2, width, 1),
        )
        self.spectral_stem = nn.Sequential(
            nn.Conv2d(18, width // 2, 3, padding=1),
            nn.GroupNorm(8, width // 2),
            nn.GELU(),
            ConvNeXtDecodeBlock(width // 2),
        )
        self.full_resolution = nn.Sequential(
            nn.Conv2d(width + width // 2, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            ConvNeXtDecodeBlock(width),
            ConvNeXtDecodeBlock(width),
        )
        self.logits = nn.Conv2d(width, 1, 1)

    def forward(
        self,
        temporal_maps: Sequence[torch.Tensor],
        normalized_optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(temporal_maps) != len(self.projections):
            raise ValueError("number of transformer maps does not match decoder projections")
        fused_depths = []
        for temporal, projection in zip(temporal_maps, self.projections, strict=True):
            if temporal.ndim != 5 or temporal.shape[1] != 4:
                raise ValueError(f"expected [B,4,E,H,W] map, got {tuple(temporal.shape)}")
            pre = temporal[:, :2].mean(dim=1)
            post = temporal[:, 2:].mean(dim=1)
            fused_depths.append(projection(torch.cat((pre, post, torch.abs(post - pre)), dim=1)))
        deep = self.depth_fusion(torch.cat(fused_depths, dim=1))
        deep = F.interpolate(deep, size=normalized_optical.shape[-2:], mode="bilinear", align_corners=False)
        pre_spectral = normalized_optical[:, :, :2].mean(dim=2)
        post_spectral = normalized_optical[:, :, 2:].mean(dim=2)
        spectral = self.spectral_stem(
            torch.cat((pre_spectral, post_spectral, torch.abs(post_spectral - pre_spectral)), dim=1)
        )
        feature = self.full_resolution(torch.cat((deep, spectral), dim=1))
        return self.logits(feature), feature


class PrithviEO2ChangeModel(nn.Module):
    """Four-date Prithvi visual model, optionally corrected by Terrain-v2."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        selected_depths: Sequence[int] = DEFAULT_DEPTHS,
        decoder_width: int = 128,
        freeze_encoder: bool = True,
        terrain_channels: int | None = None,
        alpha_max: float = 2.0,
    ) -> None:
        super().__init__()
        if tuple(selected_depths) != tuple(sorted(set(selected_depths))):
            raise ValueError("selected depths must be sorted and unique")
        if max(selected_depths) >= len(encoder.blocks):
            raise ValueError("selected depth outside Prithvi encoder")
        self.encoder = encoder
        self.selected_depths = tuple(int(depth) for depth in selected_depths)
        self.freeze_encoder = bool(freeze_encoder)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not self.freeze_encoder
        self.decoder = MultiDepthTemporalChangeDecoder(
            embed_dim=int(encoder.embed_dim), width=decoder_width, depths=len(self.selected_depths)
        )
        self.terrain_adapter = (
            BoundedTerrainAdapterV2(
                terrain_channels,
                decoder_width,
                NATIVE_TERRAIN_V2_SCALE_GROUPS,
                alpha_max=alpha_max,
            )
            if terrain_channels is not None
            else None
        )
        self.register_buffer("optical_mean", torch.tensor(PRITHVI_MEAN).view(1, 6, 1, 1, 1))
        self.register_buffer("optical_std", torch.tensor(PRITHVI_STD).view(1, 6, 1, 1, 1))

    def normalize(self, optical: torch.Tensor) -> torch.Tensor:
        if optical.ndim != 5 or optical.shape[1:3] != (6, 4):
            raise ValueError(f"expected optical [B,6,4,H,W], got {tuple(optical.shape)}")
        return (torch.clamp(optical.float(), 0.0, 10_000.0) - self.optical_mean) / self.optical_std

    def encode(
        self,
        optical: torch.Tensor,
        temporal_coords: torch.Tensor,
        location_coords: torch.Tensor,
    ) -> list[torch.Tensor]:
        if temporal_coords.shape != (optical.shape[0], 4, 2):
            raise ValueError(
                f"expected temporal coordinates [B,4,2], got {tuple(temporal_coords.shape)}"
            )
        if location_coords.shape != (optical.shape[0], 2):
            raise ValueError(
                f"expected location coordinates [B,2], got {tuple(location_coords.shape)}"
            )
        temporal_coords = temporal_coords.to(device=optical.device, dtype=torch.float32)
        location_coords = location_coords.to(device=optical.device, dtype=torch.float32)
        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with context:
            features = self.encoder.forward_features(optical, temporal_coords, location_coords)
        height = optical.shape[-2] // int(self.encoder.patch_embed.patch_size[-2])
        width = optical.shape[-1] // int(self.encoder.patch_embed.patch_size[-1])
        maps = []
        for depth in self.selected_depths:
            tokens = features[depth][:, 1:]
            batch, n_tokens, channels = tokens.shape
            if n_tokens != 4 * height * width:
                raise RuntimeError(f"unexpected Prithvi token count {n_tokens}; expected {4 * height * width}")
            maps.append(tokens.reshape(batch, 4, height, width, channels).permute(0, 1, 4, 2, 3).contiguous())
        return maps

    def forward(
        self,
        optical: torch.Tensor,
        temporal_coords: torch.Tensor,
        location_coords: torch.Tensor,
        *,
        terrain: torch.Tensor | None = None,
        q_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        normalized = self.normalize(optical)
        temporal_maps = self.encode(normalized, temporal_coords, location_coords)
        visual_logits, visual_feature = self.decoder(temporal_maps, normalized)
        output = {
            "logits": visual_logits,
            "visual_logits": visual_logits,
            "visual_feature": visual_feature,
        }
        if self.terrain_adapter is None:
            return output
        if terrain is None or q_t is None:
            raise ValueError("Terrain-enabled model requires terrain and q_t")
        probability = torch.sigmoid(visual_logits.detach())
        uncertainty = 1.0 - 2.0 * torch.abs(probability - 0.5)
        logits, diagnostics = self.terrain_adapter(
            visual_logits, visual_feature, uncertainty, terrain, q_t
        )
        output["logits"] = logits
        output.update(diagnostics)
        return output
