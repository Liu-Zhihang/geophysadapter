#!/usr/bin/env python3
"""Terrain-v2 modules for role-pure, multiscale geophysical correction.

The Terrain direction is intentionally support-only. Visual features may decide
whether a correction is permitted, but they cannot generate or change the raw
Terrain direction. This separation is required for aligned-versus-mismatched
Terrain attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


CURRENT_TERRAIN_NAMES = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "curvature_laplacian",
    "tpi_90m",
    "tpi_300m",
    "roughness_30m",
    "local_relief_300m",
)


@dataclass(frozen=True)
class TerrainScaleGroups:
    """Channel indices grouped by their physical support scale."""

    fine: tuple[int, ...]
    meso: tuple[int, ...]
    macro: tuple[int, ...]

    def validate(self, channels: int) -> None:
        groups = self.fine + self.meso + self.macro
        if not self.fine or not self.meso or not self.macro:
            raise ValueError("fine, meso, and macro Terrain groups must all be non-empty")
        if len(groups) != len(set(groups)):
            raise ValueError("Terrain scale groups must not overlap")
        if min(groups) < 0 or max(groups) >= channels:
            raise ValueError(f"Terrain scale group index outside [0,{channels})")


CURRENT_SCALE_GROUPS = TerrainScaleGroups(
    fine=(1, 2, 3, 4, 7),
    meso=(5, 6),
    macro=(0, 8),
)

NATIVE_TERRAIN_V2_NAMES = (
    "elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "profile_curvature",
    "plan_curvature",
    "laplacian_curvature",
    "tpi_90m",
    "tpi_300m",
    "tpi_900m",
    "local_std_90m",
    "local_std_300m",
    "local_relief_300m",
    "local_relief_900m",
    "valley_depth_900m",
    "ridge_height_900m",
    "ruggedness_90m",
)

NATIVE_TERRAIN_V2_SCALE_GROUPS = TerrainScaleGroups(
    fine=(1, 2, 3, 4, 5, 6, 7, 10, 16),
    meso=(8, 11, 12),
    macro=(0, 9, 13, 14, 15),
)


def group_count(channels: Sequence[int]) -> int:
    for groups in (8, 4, 2, 1):
        if len(channels) % groups == 0:
            return groups
    return 1


class ZeroPreservingConv(nn.Module):
    """Bias-free convolution, non-affine normalization, and zero-safe GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(group_count(range(out_channels)), out_channels, affine=False),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ZeroPreservingConvNeXtBlock(nn.Module):
    """A compact ConvNeXt-style residual block that maps exact zero to zero."""

    def __init__(self, channels: int, *, dilation: int = 1, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            7,
            padding=3 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(group_count(range(channels)), channels, affine=False)
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.contract = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(value)
        residual = self.norm(residual)
        residual = self.expand(residual)
        residual = F.gelu(residual)
        residual = self.contract(residual)
        return value + residual


class SupportOnlyMultiScaleTerrainPyramid(nn.Module):
    """Encode fine, meso, and macro Terrain support without visual shortcuts."""

    def __init__(
        self,
        terrain_channels: int,
        groups: TerrainScaleGroups = CURRENT_SCALE_GROUPS,
        *,
        fine_width: int = 32,
        meso_width: int = 40,
        macro_width: int = 48,
        output_width: int = 48,
    ) -> None:
        super().__init__()
        groups.validate(terrain_channels)
        self.terrain_channels = int(terrain_channels)
        self.groups = groups
        self.register_buffer("fine_indices", torch.tensor(groups.fine, dtype=torch.long), persistent=True)
        self.register_buffer("meso_indices", torch.tensor(groups.meso, dtype=torch.long), persistent=True)
        self.register_buffer("macro_indices", torch.tensor(groups.macro, dtype=torch.long), persistent=True)

        self.fine_stem = ZeroPreservingConv(len(groups.fine), fine_width)
        self.fine_blocks = nn.Sequential(
            ZeroPreservingConvNeXtBlock(fine_width),
            ZeroPreservingConvNeXtBlock(fine_width, dilation=2),
        )
        self.meso_stem = ZeroPreservingConv(len(groups.meso), meso_width)
        self.meso_blocks = nn.Sequential(
            ZeroPreservingConvNeXtBlock(meso_width),
            ZeroPreservingConvNeXtBlock(meso_width, dilation=2),
        )
        self.macro_stem = ZeroPreservingConv(len(groups.macro), macro_width)
        self.macro_blocks = nn.Sequential(
            ZeroPreservingConvNeXtBlock(macro_width, dilation=2),
            ZeroPreservingConvNeXtBlock(macro_width, dilation=4),
        )

        self.fuse_32 = nn.Sequential(
            ZeroPreservingConv(fine_width + meso_width + macro_width, 96, kernel_size=1),
            ZeroPreservingConvNeXtBlock(96, dilation=2),
            ZeroPreservingConv(96, 64),
        )
        self.decode_64 = nn.Sequential(
            ZeroPreservingConv(64 + meso_width, 64),
            ZeroPreservingConvNeXtBlock(64),
        )
        self.decode_128 = nn.Sequential(
            ZeroPreservingConv(64 + fine_width, output_width),
            ZeroPreservingConvNeXtBlock(output_width),
        )
        self.output = nn.Conv2d(output_width, 1, 1, bias=False)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)

    @staticmethod
    def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def forward(self, terrain: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if terrain.ndim != 4 or terrain.shape[1] != self.terrain_channels:
            raise ValueError(
                f"expected Terrain [B,{self.terrain_channels},H,W], got {tuple(terrain.shape)}"
            )
        height, width = terrain.shape[-2:]
        size_64 = (max(1, height // 2), max(1, width // 2))
        size_32 = (max(1, height // 4), max(1, width // 4))

        fine_input = terrain.index_select(1, self.fine_indices)
        meso_input = self._resize(terrain.index_select(1, self.meso_indices), size_64)
        macro_input = self._resize(terrain.index_select(1, self.macro_indices), size_32)

        fine = self.fine_blocks(self.fine_stem(fine_input))
        meso = self.meso_blocks(self.meso_stem(meso_input))
        macro = self.macro_blocks(self.macro_stem(macro_input))
        context = self.fuse_32(torch.cat((self._resize(fine, size_32), self._resize(meso, size_32), macro), dim=1))
        decoded_64 = self.decode_64(torch.cat((self._resize(context, size_64), meso), dim=1))
        decoded = self.decode_128(torch.cat((self._resize(decoded_64, (height, width)), fine), dim=1))
        direction = self.output(decoded)
        return direction, {
            "terrain_fine_feature": fine,
            "terrain_meso_feature": meso,
            "terrain_macro_feature": macro,
            "terrain_pyramid_feature": decoded,
        }


class VisualReliabilityGateV2(nn.Module):
    """Label-free gate from frozen visual context and reliability diagnostics."""

    def __init__(
        self,
        visual_channels: int,
        hidden: int = 64,
        *,
        uncertainty_cutoff: float = 0.5,
        uncertainty_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 <= uncertainty_cutoff <= 1.0:
            raise ValueError("uncertainty_cutoff must be in [0, 1]")
        if uncertainty_temperature <= 0.0:
            raise ValueError("uncertainty_temperature must be positive")
        self.uncertainty_cutoff = float(uncertainty_cutoff)
        self.uncertainty_temperature = float(uncertainty_temperature)
        self.head = nn.Sequential(
            nn.Conv2d(visual_channels + 1, hidden, 3, padding=1, bias=True),
            nn.GroupNorm(group_count(range(hidden)), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )

    def forward(self, visual_feature: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        if visual_feature.shape[-2:] != uncertainty.shape[-2:]:
            visual_feature = F.interpolate(
                visual_feature, size=uncertainty.shape[-2:], mode="bilinear", align_corners=False
            )
        reliability_permission = torch.sigmoid(
            self.head(torch.cat((visual_feature.detach(), uncertainty.detach()), dim=1))
        )
        uncertainty = torch.clamp(uncertainty.detach(), 0.0, 1.0)
        uncertain_region = torch.sigmoid(
            (uncertainty - self.uncertainty_cutoff) / self.uncertainty_temperature
        )
        return reliability_permission * uncertainty * uncertain_region


class BoundedTerrainAdapterV2(nn.Module):
    """Apply a support-valid, visually gated, bounded Terrain logit residual."""

    def __init__(
        self,
        terrain_channels: int,
        visual_channels: int,
        groups: TerrainScaleGroups = CURRENT_SCALE_GROUPS,
        *,
        alpha_max: float = 2.0,
        uncertainty_cutoff: float = 0.5,
        uncertainty_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if alpha_max <= 0:
            raise ValueError("alpha_max must be positive")
        self.alpha_max = float(alpha_max)
        self.terrain = SupportOnlyMultiScaleTerrainPyramid(terrain_channels, groups)
        self.gate = VisualReliabilityGateV2(
            visual_channels,
            uncertainty_cutoff=uncertainty_cutoff,
            uncertainty_temperature=uncertainty_temperature,
        )

    def forward(
        self,
        visual_logits: torch.Tensor,
        visual_feature: torch.Tensor,
        uncertainty: torch.Tensor,
        terrain: torch.Tensor,
        q_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_direction, terrain_features = self.terrain(terrain)
        zero_direction, _ = self.terrain(torch.zeros_like(terrain))
        bounded_direction = self.alpha_max * torch.tanh(raw_direction - zero_direction)
        gate = self.gate(visual_feature, uncertainty)
        q_t = torch.clamp(q_t, 0.0, 1.0)
        if q_t.ndim == 1:
            q_t = q_t[:, None, None, None]
        correction = q_t * gate * bounded_direction
        output = visual_logits + correction
        return output, {
            "raw_terrain_direction": raw_direction,
            "bounded_terrain_direction": bounded_direction,
            "visual_reliability_gate": gate,
            "correction": correction,
            "q_t": q_t,
            **terrain_features,
        }
