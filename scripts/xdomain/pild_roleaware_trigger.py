#!/usr/bin/env python3
"""Event-level, role-correct Trigger calibration for PILD.

Trigger inputs are restricted to event-level antecedent rainfall summaries:
the strict D-7..D-1 total, its wrong-time reference, and their contrast.  They
may change only event-scalar calibration controls.  Pixel locations are
selected exclusively from the frozen visual logits, so Trigger cannot supply
a dense 10 m boundary direction.

The public contexts are ``aligned``, ``wrong-time``, ``event-shuffle``, and
``zero-q``.  Unsupported rows (q_R=0) fall back bit-for-bit to the baseline
logits, including when their registry features are NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


CONTEXT_ALIGNED = "aligned"
CONTEXT_WRONG_TIME = "wrong-time"
CONTEXT_EVENT_SHUFFLE = "event-shuffle"
CONTEXT_ZERO_Q = "zero-q"
TRIGGER_CONTEXTS = (
    CONTEXT_ALIGNED,
    CONTEXT_WRONG_TIME,
    CONTEXT_EVENT_SHUFFLE,
    CONTEXT_ZERO_Q,
)
ALLOWED_CONTEXTS = TRIGGER_CONTEXTS
TRIGGER_FEATURE_NAMES = (
    "rain_d7_antecedent_case_mm",
    "rain_d7_wrongtime_median_mm",
    "rain_d7_case_minus_wrongtime_mm",
)

_CONTEXT_ALIASES = {
    "aligned": "aligned",
    "wrong-time": "wrong-time",
    "wrong_time": "wrong-time",
    "event-shuffle": "event-shuffle",
    "event_shuffle": "event-shuffle",
    "zero-q": "zero-q",
    "zero_q": "zero-q",
}


@dataclass(frozen=True)
class TriggerGateConfig:
    """Hard bounds for the three event-scalar Trigger controls."""

    feature_dim: int = 3
    hidden_dim: int = 8
    max_gate_budget: float = 0.5
    max_abs_logit_prior: float = 0.75
    min_uncertainty_threshold: float = 0.25
    max_uncertainty_threshold: float = 0.75

    def validate(self) -> None:
        if self.feature_dim < 3:
            raise ValueError("feature_dim must include case, wrong-time, and contrast")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        finite_positive = (
            self.max_gate_budget,
            self.max_abs_logit_prior,
        )
        if not all(math.isfinite(value) and value > 0 for value in finite_positive):
            raise ValueError("gate and prior bounds must be finite and positive")
        low = self.min_uncertainty_threshold
        high = self.max_uncertainty_threshold
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high < 1):
            raise ValueError("uncertainty thresholds must satisfy 0 <= min < max < 1")


def _normalize_context(context: str) -> str:
    try:
        return _CONTEXT_ALIASES[str(context)]
    except KeyError as exc:
        raise ValueError(f"context must be one of {TRIGGER_CONTEXTS}, got {context!r}") from exc


def _validate_feature_tensor(features: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError(f"{name} must have shape [B,F]; spatial Trigger inputs are forbidden")
    if features.shape[1] < 3:
        raise ValueError(f"{name} must include case, wrong-time, and contrast features")


def _validate_q_r(q_r: torch.Tensor, batch_size: int) -> torch.Tensor:
    if not torch.is_tensor(q_r) or q_r.shape not in ((batch_size,), (batch_size, 1)):
        raise ValueError("q_R must contain one scalar per sample")
    q = q_r.reshape(batch_size)
    if not torch.isfinite(q).all() or bool(((q != 0) & (q != 1)).any()):
        raise ValueError("q_R must be finite and binary")
    return q


def _equal_with_nan(left: torch.Tensor, right: torch.Tensor) -> bool:
    equal = (left == right) | (torch.isnan(left) & torch.isnan(right))
    return bool(equal.all())


def assert_event_level_broadcast(
    event_ids: Sequence[str],
    features: torch.Tensor,
    q_r: torch.Tensor | None = None,
) -> None:
    """Require exact feature and optional q_R equality within each event."""

    _validate_feature_tensor(features, "features")
    if len(event_ids) != features.shape[0]:
        raise ValueError("event_ids and features must have the same batch length")
    q = None if q_r is None else _validate_q_r(q_r, features.shape[0])
    first_by_event: dict[str, int] = {}
    for index, raw_event_id in enumerate(event_ids):
        event_id = str(raw_event_id)
        if not event_id:
            raise ValueError("event_ids must be non-empty")
        first = first_by_event.setdefault(event_id, index)
        if first == index:
            continue
        if not _equal_with_nan(features[first], features[index]):
            raise ValueError(f"Trigger features vary within event: {event_id}")
        if q is not None and bool(q[first] != q[index]):
            raise ValueError(f"q_R varies within event: {event_id}")


def build_wrong_time_features(aligned_features: torch.Tensor) -> torch.Tensor:
    """Replace the case total by wrong-time and set case-minus-control to zero."""

    _validate_feature_tensor(aligned_features, "aligned_features")
    wrong_time = aligned_features.clone()
    wrong_time[:, 0] = aligned_features[:, 1]
    wrong_time[:, 2] = 0.0
    return wrong_time


def build_event_shuffled_features(
    aligned_features: torch.Tensor,
    event_ids: Sequence[str],
    donor_by_event: Mapping[str, str],
    q_r: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a supported donor event vector onto each supported target event.

    q_R=1 events require a non-self donor that is present and also supported.
    q_R=0 events are left in place because ``zero-q`` is already exact neutral.
    """

    assert_event_level_broadcast(event_ids, aligned_features, q_r)
    q = _validate_q_r(q_r, aligned_features.shape[0])
    first_by_event: dict[str, int] = {}
    for index, event_id in enumerate(event_ids):
        first_by_event.setdefault(str(event_id), index)

    shuffled = aligned_features.clone()
    for index, raw_event_id in enumerate(event_ids):
        event_id = str(raw_event_id)
        if float(q[index].item()) == 0.0:
            continue
        if event_id not in donor_by_event:
            raise ValueError(f"event-shuffle donor missing for supported event: {event_id}")
        donor = str(donor_by_event[event_id])
        if donor == event_id:
            raise ValueError(f"event-shuffle donor is a fixed point: {event_id}")
        if donor not in first_by_event:
            raise ValueError(f"event-shuffle donor is absent from this materialization: {donor}")
        donor_index = first_by_event[donor]
        if float(q[donor_index].item()) == 0.0:
            raise ValueError(f"event-shuffle donor is unsupported: {donor}")
        shuffled[index] = aligned_features[donor_index]
    assert_event_level_broadcast(event_ids, shuffled, q_r)
    return shuffled


def materialize_trigger_context(
    aligned_features: torch.Tensor,
    q_r: torch.Tensor,
    event_ids: Sequence[str],
    *,
    context: str,
    wrong_time_features: torch.Tensor | None = None,
    event_shuffled_features: torch.Tensor | None = None,
    donor_by_event: Mapping[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one frozen Trigger control without changing sample/event identity."""

    context = _normalize_context(context)
    assert_event_level_broadcast(event_ids, aligned_features, q_r)
    q = _validate_q_r(q_r, aligned_features.shape[0])

    if context == "aligned" or context == "zero-q":
        selected = aligned_features
    elif context == "wrong-time":
        selected = (
            build_wrong_time_features(aligned_features)
            if wrong_time_features is None
            else wrong_time_features
        )
    else:
        if event_shuffled_features is not None and donor_by_event is not None:
            raise ValueError("provide event_shuffled_features or donor_by_event, not both")
        if event_shuffled_features is not None:
            selected = event_shuffled_features
        elif donor_by_event is not None:
            selected = build_event_shuffled_features(
                aligned_features, event_ids, donor_by_event, q
            )
        else:
            raise ValueError("event-shuffle requires materialized features or donor_by_event")

    _validate_feature_tensor(selected, f"{context}_features")
    if selected.shape != aligned_features.shape:
        raise ValueError(f"{context} features must match aligned feature shape")
    assert_event_level_broadcast(event_ids, selected, q)
    effective_q = torch.zeros_like(q) if context == "zero-q" else q
    return selected, effective_q


def _as_map(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape == reference.shape:
        return value
    if value.shape == (reference.shape[0], *reference.shape[2:]):
        return value[:, None]
    raise ValueError(f"{name} must match logits shape [B,1,H,W]")


def trigger_audit_quantities(
    logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    *,
    target: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    area_threshold: float = 0.5,
    fixed_fpr_threshold: float | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return additive per-sample receipts for calibration and area metrics.

    Sums and counts can be aggregated across patches/events before division.
    ``fixed_fpr_threshold`` must be selected outside this function from the
    designated calibration split.
    """

    if logits.shape != baseline_logits.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits and baseline_logits must share shape [B,1,H,W]")
    if not (0 < area_threshold < 1):
        raise ValueError("area_threshold must lie in (0,1)")
    probability = torch.sigmoid(logits.float())
    baseline_probability = torch.sigmoid(baseline_logits.float())
    valid = torch.ones_like(probability, dtype=torch.bool)
    if valid_mask is not None:
        valid = _as_map(valid_mask, probability, "valid_mask").bool()
    dimensions = (1, 2, 3)
    valid_float = valid.float()
    predicted = (probability >= area_threshold) & valid
    baseline_predicted = (baseline_probability >= area_threshold) & valid
    audit = {
        "valid_pixel_count": valid.sum(dim=dimensions),
        "probability_sum": (probability * valid_float).sum(dim=dimensions),
        "baseline_probability_sum": (baseline_probability * valid_float).sum(dim=dimensions),
        "predicted_positive_count": predicted.sum(dim=dimensions),
        "baseline_predicted_positive_count": baseline_predicted.sum(dim=dimensions),
    }
    if target is None:
        return audit

    target_map = _as_map(target, probability, "target")
    if not torch.isfinite(target_map[valid]).all():
        raise ValueError("target must be finite on valid pixels")
    target_float = (target_map >= 0.5).float()
    target_positive = ((target_float > 0) & valid).sum(dim=dimensions)
    brier = (probability - target_float).square() * valid_float
    baseline_brier = (baseline_probability - target_float).square() * valid_float
    nll = F.binary_cross_entropy_with_logits(
        logits.float(), target_float, reduction="none"
    ) * valid_float
    baseline_nll = F.binary_cross_entropy_with_logits(
        baseline_logits.float(), target_float, reduction="none"
    ) * valid_float
    audit.update(
        {
            "target_positive_count": target_positive,
            "brier_sum": brier.sum(dim=dimensions),
            "baseline_brier_sum": baseline_brier.sum(dim=dimensions),
            "nll_sum": nll.sum(dim=dimensions),
            "baseline_nll_sum": baseline_nll.sum(dim=dimensions),
            "soft_area_error": audit["probability_sum"] - target_positive,
            "baseline_soft_area_error": audit["baseline_probability_sum"]
            - target_positive,
            "hard_area_error": audit["predicted_positive_count"] - target_positive,
            "baseline_hard_area_error": audit["baseline_predicted_positive_count"]
            - target_positive,
        }
    )

    if fixed_fpr_threshold is not None:
        threshold = torch.as_tensor(
            fixed_fpr_threshold, device=probability.device, dtype=probability.dtype
        )
        if threshold.ndim == 0:
            threshold = threshold.expand(probability.shape[0])
        elif threshold.shape == (probability.shape[0], 1):
            threshold = threshold[:, 0]
        if threshold.shape != (probability.shape[0],):
            raise ValueError("fixed_fpr_threshold must be scalar or contain one value per sample")
        if not torch.isfinite(threshold).all() or bool(((threshold <= 0) | (threshold >= 1)).any()):
            raise ValueError("fixed_fpr_threshold must lie in (0,1)")
        fixed_prediction = probability >= threshold[:, None, None, None]
        fixed_prediction &= valid
        positive = (target_float > 0) & valid
        negative = ~positive & valid
        tp = (fixed_prediction & positive).sum(dim=dimensions)
        fn = (~fixed_prediction & positive).sum(dim=dimensions)
        fp = (fixed_prediction & negative).sum(dim=dimensions)
        tn = (~fixed_prediction & negative).sum(dim=dimensions)
        audit.update(
            {
                "fixed_fpr_threshold": threshold,
                "fixed_fpr_tp": tp,
                "fixed_fpr_fn": fn,
                "fixed_fpr_fp": fp,
                "fixed_fpr_tn": tn,
                "fixed_fpr_recall_numerator": tp,
                "fixed_fpr_recall_denominator": tp + fn,
            }
        )
    return audit


class PILDRoleAwareTrigger(nn.Module):
    """Map event features to bounded scalar controls over visual uncertainty."""

    def __init__(self, config: TriggerGateConfig | None = None) -> None:
        super().__init__()
        self.config = TriggerGateConfig() if config is None else config
        self.config.validate()
        self.calibrator = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, 3),
        )
        # Start as an exact identity intervention while retaining finite gates.
        nn.init.zeros_(self.calibrator[-1].weight)
        nn.init.zeros_(self.calibrator[-1].bias)

    def forward(
        self,
        baseline_logits: torch.Tensor,
        aligned_features: torch.Tensor,
        q_R: torch.Tensor,
        event_ids: Sequence[str],
        *,
        context: str = "aligned",
        wrong_time_features: torch.Tensor | None = None,
        event_shuffled_features: torch.Tensor | None = None,
        donor_by_event: Mapping[str, str] | None = None,
        target: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        area_threshold: float = 0.5,
        fixed_fpr_threshold: float | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if baseline_logits.ndim != 4 or baseline_logits.shape[1] != 1:
            raise ValueError("baseline_logits must have shape [B,1,H,W]")
        if not torch.isfinite(baseline_logits).all():
            raise ValueError("baseline_logits must be finite")
        if aligned_features.shape != (
            baseline_logits.shape[0],
            self.config.feature_dim,
        ):
            raise ValueError(
                f"aligned_features must have shape [B,{self.config.feature_dim}]"
            )
        context = _normalize_context(context)
        selected, effective_q = materialize_trigger_context(
            aligned_features,
            q_R,
            event_ids,
            context=context,
            wrong_time_features=wrong_time_features,
            event_shuffled_features=event_shuffled_features,
            donor_by_event=donor_by_event,
        )
        baseline = baseline_logits.detach()
        selected = selected.to(
            device=baseline.device, dtype=baseline.dtype
        ).detach()
        effective_q = effective_q.to(
            device=baseline.device, dtype=baseline.dtype
        ).detach()
        active = effective_q > 0
        active_rows = active[:, None].expand_as(selected)
        if not torch.isfinite(selected[active_rows]).all():
            raise ValueError("q_R=1 Trigger features must be finite")
        clean_features = torch.where(
            torch.isfinite(selected), selected, torch.zeros_like(selected)
        )
        raw = self.calibrator(clean_features.float()).to(dtype=baseline.dtype)
        budget = self.config.max_gate_budget * torch.sigmoid(raw[:, 0])
        prior = self.config.max_abs_logit_prior * torch.tanh(raw[:, 1])
        threshold_range = (
            self.config.max_uncertainty_threshold
            - self.config.min_uncertainty_threshold
        )
        threshold = self.config.min_uncertainty_threshold + threshold_range * torch.sigmoid(
            raw[:, 2]
        )

        zeros = torch.zeros_like(budget)
        midpoint = baseline.new_full(
            threshold.shape,
            0.5
            * (
                self.config.min_uncertainty_threshold
                + self.config.max_uncertainty_threshold
            ),
        )
        budget = torch.where(active, budget, zeros)
        prior = torch.where(active, prior, zeros)
        threshold = torch.where(active, threshold, midpoint)
        budget_map = budget[:, None, None, None]
        prior_map = prior[:, None, None, None]
        threshold_map = threshold[:, None, None, None]

        # This is the only spatial signal in the module and it is visual-only.
        baseline_probability = torch.sigmoid(baseline)
        visual_uncertainty = 1.0 - torch.abs(2.0 * baseline_probability - 1.0)
        normalized_uncertainty = torch.clamp(
            (visual_uncertainty - threshold_map) / (1.0 - threshold_map),
            min=0.0,
            max=1.0,
        )
        gate = budget_map * normalized_uncertainty
        candidate_delta = prior_map * gate
        candidate_logits = baseline + candidate_delta
        logits = torch.where(active[:, None, None, None], candidate_logits, baseline)
        logit_delta = logits - baseline

        audit = trigger_audit_quantities(
            logits,
            baseline,
            target=target,
            valid_mask=valid_mask,
            area_threshold=area_threshold,
            fixed_fpr_threshold=fixed_fpr_threshold,
        )
        audit.update(
            {
                "active_trigger": active,
                "q_R_effective": effective_q,
                "logit_delta_abs_max": logit_delta.abs().flatten(1).amax(dim=1),
                "changed_pixel_count": (logit_delta != 0).flatten(1).sum(dim=1),
                "trigger_dense_direction": False,
                "trigger_spatial_source": "detached_visual_uncertainty_only",
                "gate_budget_bounds": (0.0, self.config.max_gate_budget),
                "logit_prior_bounds": (
                    -self.config.max_abs_logit_prior,
                    self.config.max_abs_logit_prior,
                ),
                "uncertainty_threshold_bounds": (
                    self.config.min_uncertainty_threshold,
                    self.config.max_uncertainty_threshold,
                ),
            }
        )
        return {
            "logits": logits,
            "baseline_logits": baseline,
            "probability": torch.sigmoid(logits.float()),
            "baseline_probability": torch.sigmoid(baseline.float()),
            "logit_delta": logit_delta,
            "visual_uncertainty": visual_uncertainty,
            "gate": gate,
            "gate_budget": budget_map,
            "logit_prior": prior_map,
            "uncertainty_threshold": threshold_map,
            "q_R": effective_q[:, None, None, None],
            "context_features": selected,
            "context": context,
            "event_ids": tuple(str(value) for value in event_ids),
            "audit": audit,
        }


EventLevelTriggerGate = PILDRoleAwareTrigger
PILDRoleAwareTriggerGate = PILDRoleAwareTrigger
RoleAwareTriggerGate = PILDRoleAwareTrigger


__all__ = [
    "ALLOWED_CONTEXTS",
    "CONTEXT_ALIGNED",
    "CONTEXT_EVENT_SHUFFLE",
    "CONTEXT_WRONG_TIME",
    "CONTEXT_ZERO_Q",
    "EventLevelTriggerGate",
    "PILDRoleAwareTrigger",
    "PILDRoleAwareTriggerGate",
    "RoleAwareTriggerGate",
    "TRIGGER_CONTEXTS",
    "TRIGGER_FEATURE_NAMES",
    "TriggerGateConfig",
    "assert_event_level_broadcast",
    "build_event_shuffled_features",
    "build_wrong_time_features",
    "materialize_trigger_context",
    "trigger_audit_quantities",
]
