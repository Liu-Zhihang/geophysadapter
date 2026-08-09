#!/usr/bin/env python3
"""Exploratory bounded rescue/veto corrector on frozen visual and Terrain models.

The frozen multiscale Terrain pyramid is retained, but its scalar logit is no
longer forced to provide one shared signed correction. A zero-preserving
physical evidence encoder learns separate non-negative rescue and veto
candidates. A visual reliability router may only scale those candidates:

* rescue acts only where the frozen visual decision is negative;
* veto acts only where the frozen visual decision is positive;
* q_T == 0 gives exact visual identity;
* every correction is bounded by alpha_max.

This is a development experiment. Confirmation requires an event-isolated
inner-OOF training protocol after the architecture passes development gates.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import train_pild_adaptive_terrain_gate_v1 as runner


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ZeroPreservingEvidence(nn.Module):
    """Map exact-zero Terrain logits to exact-zero directional evidence."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 5, padding=2, bias=False),
            nn.GroupNorm(group_count(width), width, affine=False),
            nn.GELU(),
            nn.Conv2d(
                width,
                width,
                5,
                padding=4,
                dilation=2,
                groups=width,
                bias=False,
            ),
            nn.GroupNorm(group_count(width), width, affine=False),
            nn.GELU(),
            nn.Conv2d(width, width, 1, bias=False),
            nn.GELU(),
        )
        self.rescue = nn.Conv2d(width, 1, 1, bias=False)
        self.veto = nn.Conv2d(width, 1, 1, bias=False)
        nn.init.normal_(self.rescue.weight, mean=0.0, std=2e-2)
        nn.init.normal_(self.veto.weight, mean=0.0, std=2e-2)

    def forward(self, terrain_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(terrain_logits)
        rescue = torch.tanh(F.relu(self.rescue(features)))
        veto = torch.tanh(F.relu(self.veto(features)))
        return rescue, veto


class BidirectionalTerrainCorrector(nn.Module):
    """Terrain-generated candidates with visually conditioned bounded routing."""

    def __init__(self, *, alpha_max: float = 4.0) -> None:
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.evidence = ZeroPreservingEvidence(32)
        self.router = nn.Sequential(
            nn.Conv2d(6, 48, 5, padding=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=2, dilation=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 2, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, -2.5)

    def forward(
        self,
        visual_logits: torch.Tensor,
        terrain_logits: torch.Tensor,
        q_t: torch.Tensor,
        *,
        threshold_logit: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_t = q_t.clamp(0.0, 1.0)
        signed_margin = ((visual_logits - threshold_logit) / 4.0).clamp(-2.0, 2.0)
        visual_probability = torch.sigmoid(visual_logits)
        uncertainty = (1.0 - 2.0 * (visual_probability - 0.5).abs()).clamp(0.0, 1.0)
        rescue_evidence, veto_evidence = self.evidence(terrain_logits * q_t)
        router_features = torch.cat(
            (
                signed_margin,
                signed_margin.abs(),
                uncertainty,
                rescue_evidence,
                veto_evidence,
                q_t,
            ),
            dim=1,
        )
        route = torch.sigmoid(self.router(router_features))
        visual_positive = visual_logits >= threshold_logit
        rescue = route[:, :1] * rescue_evidence * (~visual_positive).float()
        veto = route[:, 1:2] * veto_evidence * visual_positive.float()
        self._last_rescue = rescue
        self._last_veto = veto
        correction = self.alpha_max * q_t * (rescue - veto)
        activity = rescue + veto
        return visual_logits + correction, activity

    @staticmethod
    def _directional_bce(
        probability: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        harmful_weight: float,
    ) -> torch.Tensor:
        probability = probability.clamp(1e-5, 1.0 - 1e-5)
        positive = mask & (target >= 0.5)
        negative = mask & (target < 0.5)
        positive_loss = (
            -torch.log(probability[positive]).mean()
            if positive.any()
            else probability.sum() * 0.0
        )
        negative_loss = (
            -torch.log1p(-probability[negative]).mean()
            if negative.any()
            else probability.sum() * 0.0
        )
        return positive_loss + harmful_weight * negative_loss

    def correction_objective(
        self,
        visual_logits: torch.Tensor,
        adapted_logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        gate: torch.Tensor,
        *,
        threshold_logit: float,
        decision_temperature: float,
        gate_penalty: float,
    ) -> torch.Tensor:
        base = rescue_aware_loss(
            adapted_logits,
            target,
            valid,
            gate,
            threshold_logit=threshold_logit,
            decision_temperature=decision_temperature,
            gate_penalty=gate_penalty,
        )
        visual_positive = visual_logits >= threshold_logit
        keep = valid.bool()
        rescue_loss = self._directional_bce(
            self._last_rescue,
            target,
            keep & (~visual_positive),
            harmful_weight=2.0,
        )
        veto_loss = self._directional_bce(
            self._last_veto,
            1.0 - target,
            keep & visual_positive,
            harmful_weight=8.0,
        )
        return base + 0.25 * rescue_loss + 0.15 * veto_loss


def rescue_aware_loss(
    adapted_logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    gate: torch.Tensor,
    *,
    threshold_logit: float,
    decision_temperature: float,
    gate_penalty: float,
) -> torch.Tensor:
    """Optimize IoU while assigning a larger cost to remaining false negatives."""

    probability = torch.sigmoid(
        (adapted_logits - threshold_logit) / decision_temperature
    )
    truth = target * valid
    prediction = probability * valid
    tp = (prediction * truth).sum(dim=(1, 2, 3))
    fp = (prediction * (1.0 - truth) * valid).sum(dim=(1, 2, 3))
    fn = ((1.0 - prediction) * truth).sum(dim=(1, 2, 3))
    soft_iou = (1.0 - (tp + 1.0) / (tp + fp + fn + 1.0)).mean()
    # Tversky(alpha=0.3, beta=0.7) explicitly makes missed landslides costlier.
    tversky = (
        1.0 - (tp + 1.0) / (tp + 0.3 * fp + 0.7 * fn + 1.0)
    ).mean()
    bce = F.binary_cross_entropy(probability, target, reduction="none")
    positive_weight = torch.where(target >= 0.5, 2.0, 1.0)
    bce = (bce * positive_weight * valid).sum() / (
        positive_weight * valid
    ).sum().clamp_min(1.0)
    activity_cost = (gate * valid).sum() / valid.sum().clamp_min(1.0)
    return soft_iou + 0.5 * tversky + 0.25 * bce + gate_penalty * activity_cost


if __name__ == "__main__":
    runner.AdaptiveTerrainGate = BidirectionalTerrainCorrector
    runner.correction_loss = rescue_aware_loss
    raise SystemExit(runner.main())
