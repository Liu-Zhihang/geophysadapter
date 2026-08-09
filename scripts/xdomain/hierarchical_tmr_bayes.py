#!/usr/bin/env python3
"""Role-pure Terrain x Material susceptibility and bounded Bayesian fusion.

Terrain is the only dense physical direction. Material may change how local
Terrain responses map to susceptibility, while Trigger may only scale the
event-level correction budget. Unsupported inputs reduce exactly to their
parent model.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from sen12_terrain_v2 import (
    CURRENT_SCALE_GROUPS,
    SupportOnlyMultiScaleTerrainPyramid,
    TerrainScaleGroups,
)


def _quality_vector(
    value: torch.Tensor,
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quality = value.to(device=device, dtype=dtype).reshape(-1)
    if quality.shape != (batch,):
        raise ValueError("quality must contain one scalar per sample")
    return torch.where(
        torch.isfinite(quality), quality.clamp(0.0, 1.0), torch.zeros_like(quality)
    )


def spatially_center(
    value: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Remove the per-sample spatial intercept without reading a label."""

    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("value must have shape [B,1,H,W]")
    if valid_mask is None:
        return value - value.mean(dim=(-2, -1), keepdim=True)
    valid = valid_mask.to(device=value.device, dtype=value.dtype)
    if valid.shape != value.shape:
        raise ValueError("valid_mask must have the same shape as value")
    valid = (valid > 0.5).to(value.dtype)
    denominator = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (value * valid).sum(dim=(-2, -1), keepdim=True) / denominator
    return (value - mean) * valid


class TerrainMaterialSusceptibility(nn.Module):
    """A nested T -> T x M susceptibility model with exact Material fallback."""

    def __init__(
        self,
        terrain_channels: int,
        material_features: int,
        groups: TerrainScaleGroups = CURRENT_SCALE_GROUPS,
        *,
        basis_count: int = 12,
        material_hidden: int = 48,
        material_logit_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if terrain_channels <= 0 or material_features <= 0:
            raise ValueError("terrain_channels and material_features must be positive")
        if basis_count <= 0 or material_logit_bound <= 0:
            raise ValueError("basis_count and material_logit_bound must be positive")
        self.terrain_channels = int(terrain_channels)
        self.material_features = int(material_features)
        self.basis_count = int(basis_count)
        self.material_logit_bound = float(material_logit_bound)
        self.terrain = SupportOnlyMultiScaleTerrainPyramid(
            terrain_channels, groups
        )
        self.material_basis = nn.Conv2d(48, basis_count, 1, bias=False)
        self.material_coefficients = nn.Sequential(
            nn.Linear(material_features, material_hidden),
            nn.SiLU(),
            nn.Linear(material_hidden, basis_count),
        )
        nn.init.zeros_(self.material_coefficients[-1].weight)
        nn.init.zeros_(self.material_coefficients[-1].bias)

    def freeze_terrain(self) -> None:
        for parameter in self.terrain.parameters():
            parameter.requires_grad_(False)

    def material_parameters(self):
        yield from self.material_basis.parameters()
        yield from self.material_coefficients.parameters()

    def forward(
        self,
        terrain: torch.Tensor,
        material: torch.Tensor,
        q_m: torch.Tensor,
        *,
        q_t: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if terrain.ndim != 4 or terrain.shape[1] != self.terrain_channels:
            raise ValueError(
                f"terrain must have shape [B,{self.terrain_channels},H,W]"
            )
        batch = terrain.shape[0]
        if material.shape != (batch, self.material_features):
            raise ValueError(
                f"material must have shape [B,{self.material_features}]"
            )
        terrain_quality = (
            torch.ones(batch, device=terrain.device, dtype=terrain.dtype)
            if q_t is None
            else _quality_vector(
                q_t, batch, device=terrain.device, dtype=terrain.dtype
            )
        )
        terrain_input = terrain * terrain_quality[:, None, None, None]
        terrain_logit, terrain_features = self.terrain(terrain_input)
        terrain_logit = terrain_logit * terrain_quality[:, None, None, None]
        quality = _quality_vector(
            q_m, batch, device=terrain.device, dtype=terrain.dtype
        )
        finite_material = torch.isfinite(material).all(dim=1)
        effective_q = quality * finite_material.to(quality.dtype) * terrain_quality
        clean_material = torch.where(
            torch.isfinite(material), material, torch.zeros_like(material)
        ).detach()
        basis = torch.tanh(self.material_basis(terrain_features["terrain_pyramid_feature"]))
        coefficients = self.material_coefficients(clean_material)
        raw_interaction = (
            coefficients[:, :, None, None] * basis
        ).sum(dim=1, keepdim=True) / math.sqrt(self.basis_count)
        centered_interaction = spatially_center(raw_interaction, valid_mask)
        material_delta = (
            effective_q[:, None, None, None]
            * self.material_logit_bound
            * torch.tanh(centered_interaction)
        )
        susceptibility_logit = terrain_logit + material_delta
        return susceptibility_logit, {
            "terrain_logit": terrain_logit,
            "material_basis": basis,
            "material_coefficients": coefficients,
            "material_interaction_raw": raw_interaction,
            "material_delta": material_delta,
            "q_M_effective": effective_q[:, None, None, None],
            "q_T_effective": terrain_quality[:, None, None, None],
            **terrain_features,
        }


class PositiveMaterialDose(nn.Module):
    """A sign-preserving Material multiplier with exact neutral fallback."""

    def __init__(
        self,
        material_features: int,
        *,
        hidden: int = 16,
        log_multiplier_bound: float = math.log(2.0),
    ) -> None:
        super().__init__()
        if material_features <= 0 or hidden <= 0 or log_multiplier_bound <= 0:
            raise ValueError("invalid PositiveMaterialDose dimensions or bound")
        self.material_features = int(material_features)
        self.log_multiplier_bound = float(log_multiplier_bound)
        self.network = nn.Sequential(
            nn.Linear(material_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        material: torch.Tensor,
        q_m: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if material.ndim != 2 or material.shape[1] != self.material_features:
            raise ValueError(
                f"material must have shape [B,{self.material_features}]"
            )
        batch = material.shape[0]
        quality = _quality_vector(
            q_m, batch, device=material.device, dtype=material.dtype
        )
        finite = torch.isfinite(material).all(dim=1)
        effective_q = quality * finite.to(quality.dtype)
        clean = torch.where(torch.isfinite(material), material, torch.zeros_like(material))
        raw = self.network(clean.detach()).reshape(-1)
        log_multiplier = effective_q * self.log_multiplier_bound * torch.tanh(raw)
        multiplier = torch.exp(log_multiplier)
        return multiplier, {
            "material_log_multiplier": log_multiplier,
            "material_multiplier": multiplier,
            "q_M_effective": effective_q,
        }


class BoundedBayesianTMRFusion(nn.Module):
    """Add a bounded physical log-odds update only where vision is uncertain."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        correction_bound: float = math.log(4.0),
        trigger_scale: float = 0.5,
        uncertainty_power: float = 1.0,
    ) -> None:
        super().__init__()
        if alpha < 0 or correction_bound <= 0 or trigger_scale < 0:
            raise ValueError("invalid bounded fusion hyperparameters")
        if uncertainty_power < 0:
            raise ValueError("uncertainty_power must be nonnegative")
        self.alpha = float(alpha)
        self.correction_bound = float(correction_bound)
        self.trigger_scale = float(trigger_scale)
        self.uncertainty_power = float(uncertainty_power)

    @staticmethod
    def visual_uncertainty(visual_logit: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(visual_logit).clamp(1e-6, 1.0 - 1e-6)
        entropy = -probability * torch.log(probability)
        entropy -= (1.0 - probability) * torch.log(1.0 - probability)
        return entropy / math.log(2.0)

    def forward(
        self,
        visual_logit: torch.Tensor,
        susceptibility_logit: torch.Tensor,
        q_t: torch.Tensor,
        trigger_log_bf: torch.Tensor,
        q_r: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if visual_logit.shape != susceptibility_logit.shape:
            raise ValueError("visual and susceptibility logits must have equal shape")
        if visual_logit.ndim != 4 or visual_logit.shape[1] != 1:
            raise ValueError("logits must have shape [B,1,H,W]")
        batch = visual_logit.shape[0]
        terrain_quality = _quality_vector(
            q_t, batch, device=visual_logit.device, dtype=visual_logit.dtype
        )
        trigger_quality = _quality_vector(
            q_r, batch, device=visual_logit.device, dtype=visual_logit.dtype
        )
        log_bf = trigger_log_bf.to(
            device=visual_logit.device, dtype=visual_logit.dtype
        ).reshape(-1)
        if log_bf.shape != (batch,):
            raise ValueError("trigger_log_bf must contain one scalar per sample")
        finite_trigger = torch.isfinite(log_bf)
        effective_q_r = trigger_quality * finite_trigger.to(trigger_quality.dtype)
        clean_log_bf = torch.where(finite_trigger, log_bf, torch.zeros_like(log_bf))

        centered_susceptibility = spatially_center(
            susceptibility_logit, valid_mask
        )
        trigger_multiplier = 1.0 + (
            effective_q_r * self.trigger_scale * torch.tanh(clean_log_bf)
        )
        physical_log_odds = (
            trigger_multiplier[:, None, None, None] * centered_susceptibility
        )
        bounded_physical_log_odds = physical_log_odds.clamp(
            -self.correction_bound, self.correction_bound
        )
        uncertainty = self.visual_uncertainty(visual_logit.detach()).pow(
            self.uncertainty_power
        )
        gate = terrain_quality[:, None, None, None] * uncertainty
        if valid_mask is not None:
            gate = gate * (valid_mask.to(gate.dtype) > 0.5).to(gate.dtype)
        correction = self.alpha * gate * bounded_physical_log_odds
        output = visual_logit + correction
        return output, {
            "centered_susceptibility_logit": centered_susceptibility,
            "trigger_multiplier": trigger_multiplier[:, None, None, None],
            "bounded_physical_log_odds": bounded_physical_log_odds,
            "visual_uncertainty": uncertainty,
            "fusion_gate": gate,
            "correction": correction,
            "q_T_effective": terrain_quality[:, None, None, None],
            "q_R_effective": effective_q_r[:, None, None, None],
        }


__all__ = [
    "BoundedBayesianTMRFusion",
    "PositiveMaterialDose",
    "TerrainMaterialSusceptibility",
    "spatially_center",
]
