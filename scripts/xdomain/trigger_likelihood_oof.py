#!/usr/bin/env python3
"""Leakage-aware transport of the frozen 138-event rainfall ranker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np


ANCHOR_ORDER = ("case", "control_m56", "control_m28", "control_p28", "control_p56")
CHANCE_PROBABILITY = 1.0 / len(ANCHOR_ORDER)


@dataclass(frozen=True)
class TriggerScore:
    aligned_probability: float
    aligned_log_bf: float
    wrong_time_probability: float
    wrong_time_log_bf: float
    scoring_mode: str
    model_folds: tuple[int, ...]


@dataclass(frozen=True)
class EventAlias:
    physical_event_id: str
    distance_km: float
    date_delta_days: int


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def centered_log_odds(probability: float) -> float:
    epsilon = np.finfo(np.float64).eps
    value = float(np.clip(probability, epsilon, 1.0 - epsilon))
    baseline = math.log(CHANCE_PROBABILITY / (1.0 - CHANCE_PROBABILITY))
    return math.log(value / (1.0 - value)) - baseline


def haversine_km(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon_a, lat_a, lon_b, lat_b))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * math.asin(math.sqrt(value))


def match_known_event(
    event_date: str,
    center_lon: float,
    center_lat: float,
    known_events: Sequence[Mapping[str, Any]],
    *,
    max_distance_km: float = 25.0,
    max_date_delta_days: int = 14,
) -> EventAlias | None:
    """Resolve conservative cross-registry aliases without segmentation labels."""
    target_date = date.fromisoformat(str(event_date))
    candidates: list[EventAlias] = []
    for record in known_events:
        distance = haversine_km(
            float(center_lon), float(center_lat),
            float(record["center_lon"]), float(record["center_lat"]),
        )
        date_delta = abs((target_date - date.fromisoformat(str(record["canonical_date"]))).days)
        if distance <= max_distance_km and date_delta <= max_date_delta_days:
            candidates.append(EventAlias(str(record["physical_event_id"]), distance, date_delta))
    if len(candidates) > 1:
        raise RuntimeError("multiple known Trigger events satisfy the conservative alias gate")
    return candidates[0] if candidates else None


def score_anchor_set(rainfall_mm: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    rainfall = np.asarray(rainfall_mm, dtype=np.float64).reshape(-1)
    if rainfall.shape != (len(ANCHOR_ORDER),):
        raise ValueError("rainfall_mm must contain case plus four controls")
    if not np.isfinite(rainfall).all() or (rainfall < 0).any():
        raise ValueError("rainfall_mm must be finite and nonnegative")
    standardized = (
        np.log1p(rainfall) - float(model["mean"])
    ) / float(model["std"])
    return _softmax(standardized * float(model["beta"]))


def select_models_for_event(
    physical_event_id: str,
    fold_models: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], str, tuple[int, ...]]:
    heldout = [
        record
        for record in fold_models
        if physical_event_id in set(map(str, record.get("heldout_event_ids", ())))
    ]
    trained = [
        record
        for record in fold_models
        if physical_event_id in set(map(str, record.get("train_event_ids", ())))
    ]
    if heldout:
        if len(heldout) != 1:
            raise RuntimeError("event belongs to multiple held-out Trigger folds")
        records = heldout
        mode = "storm_cluster_oof"
    elif trained:
        raise RuntimeError(
            "event appears in Trigger training folds but has no held-out fold"
        )
    else:
        records = list(fold_models)
        mode = "external_five_fold_ensemble"
    folds = tuple(sorted(int(record["oof_fold"]) for record in records))
    return records, mode, folds


def score_event(
    physical_event_id: str,
    rainfall_mm: np.ndarray,
    fold_models: Sequence[Mapping[str, Any]],
    *,
    wrong_time_anchor: str = "control_m28",
) -> TriggerScore:
    if wrong_time_anchor not in ANCHOR_ORDER[1:]:
        raise ValueError("wrong_time_anchor must be one of the four controls")
    records, mode, folds = select_models_for_event(physical_event_id, fold_models)
    probabilities = np.stack(
        [score_anchor_set(rainfall_mm, record["model"]) for record in records]
    ).mean(axis=0)
    aligned = float(probabilities[0])
    wrong = float(probabilities[ANCHOR_ORDER.index(wrong_time_anchor)])
    return TriggerScore(
        aligned_probability=aligned,
        aligned_log_bf=centered_log_odds(aligned),
        wrong_time_probability=wrong,
        wrong_time_log_bf=centered_log_odds(wrong),
        scoring_mode=mode,
        model_folds=folds,
    )


__all__ = [
    "ANCHOR_ORDER",
    "CHANCE_PROBABILITY",
    "TriggerScore",
    "EventAlias",
    "centered_log_odds",
    "haversine_km",
    "match_known_event",
    "score_anchor_set",
    "score_event",
    "select_models_for_event",
]
