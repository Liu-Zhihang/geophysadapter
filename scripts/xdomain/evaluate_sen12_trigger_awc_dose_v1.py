#!/usr/bin/env python3
"""Nested-OOF evaluation of role-pure event-level Trigger doses on Terrain.

Terrain is the only dense correction direction. Trigger evidence, optionally
normalized by shallow available water capacity (AWC), may only multiply the
frozen Terrain delta by a positive event-level dose in [0.8, 1.25]. A sample
whose Trigger support quality is zero receives an exact identity multiplier.

For each target outer protocol, two formal inner-holdout bundles fit a fixed
one-dimensional ridge mapping from aligned Trigger evidence to an event-wise
NLL-optimal Terrain multiplier. The third bundle is evaluated, and the roles
rotate three times. Target outer-test artifacts are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

import train_sen12_proposal_utility_gate_v3 as gate_protocol


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "experiments/revision2026/sen12_positive_dose_v1/formal_inputs_v1"
DEFAULT_MATERIAL = (
    PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v3/material_factor_registry_v3.csv"
)
DEFAULT_TRIGGER = (
    PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v1/trigger_sample_registry_v1.csv"
)
DEFAULT_LIKELIHOOD = (
    PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v3/trigger_sample_likelihood_v1.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "experiments/revision2026/sen12_trigger_awc_dose_v1"

FAMILIES = ("RAW_D7_AWC", "FROZEN_LOGBF")
CONDITIONS = (
    "VT_identity",
    "global_dose",
    "aligned",
    "wrong_time",
    "event_shuffle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--material-registry", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--trigger-registry", type=Path, default=DEFAULT_TRIGGER)
    parser.add_argument("--trigger-likelihood", type=Path, default=DEFAULT_LIKELIHOOD)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-supported-fit-events", type=int, default=3)
    parser.add_argument("--min-multiplier", type=float, default=0.8)
    parser.add_argument("--max-multiplier", type=float, default=1.25)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--sample-batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate registries and formal manifests without loading prediction caches.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def parse_target_folds(text: str) -> tuple[int, ...]:
    folds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not folds or len(folds) != len(set(folds)) or any(item not in range(5) for item in folds):
        raise ValueError("--target-folds must contain unique values from 0,1,2,3,4")
    return folds


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("cannot scale an empty feature")
    center = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(finite))
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return center, scale


def dose_from_log(
    log_multiplier: np.ndarray | float,
    quality: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Return a bounded positive dose with exact q_R=0 identity fallback."""

    quality = np.asarray(quality, np.float64)
    log_value = np.asarray(log_multiplier, np.float64)
    clipped = np.clip(log_value, math.log(minimum), math.log(maximum))
    dose = np.exp(clipped * np.clip(quality, 0.0, 1.0))
    return np.where(quality > 0.0, dose, 1.0).astype(np.float64)


def deterministic_event_donors(
    target_events: Sequence[str],
    donor_events: Sequence[str],
    seed: int,
) -> dict[str, str]:
    candidates = tuple(sorted(set(map(str, donor_events))))
    output: dict[str, str] = {}
    for event in sorted(set(map(str, target_events))):
        valid = [candidate for candidate in candidates if candidate != event]
        if not valid:
            raise RuntimeError(f"no mismatched Trigger donor is available for {event}")
        digest = hashlib.sha256(f"{seed}|R-dose|{event}".encode()).digest()
        output[event] = valid[int.from_bytes(digest[:8], "big") % len(valid)]
    return output


@dataclass(frozen=True)
class EventEvidence:
    event_id: str
    q_r: float
    aligned: float
    wrong_time: float
    shuffle: float
    n_samples: int
    donor_event: str | None


@dataclass(frozen=True)
class DoseModel:
    center: float
    scale: float
    coefficient: float
    intercept: float
    global_log_multiplier: float
    n_fit_events: int
    status: str

    def predict(self, feature: np.ndarray) -> np.ndarray:
        value = (np.asarray(feature, np.float64) - self.center) / self.scale
        return self.intercept + self.coefficient * np.clip(value, -5.0, 5.0)


def _require_unique_samples(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if "sample_id" not in frame:
        raise RuntimeError(f"{name} lacks sample_id")
    frame = frame.assign(sample_id=frame.sample_id.astype(str))
    duplicate = frame.sample_id.duplicated(keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, "sample_id"].head(5).tolist()
        raise RuntimeError(f"{name} has duplicate sample IDs: {examples}")
    return frame.set_index("sample_id", drop=False)


def load_registries(
    material_path: Path,
    trigger_path: Path,
    likelihood_path: Path,
) -> pd.DataFrame:
    material = _require_unique_samples(pd.read_csv(material_path, low_memory=False), "Material")
    trigger = _require_unique_samples(pd.read_csv(trigger_path, low_memory=False), "Trigger")
    likelihood = _require_unique_samples(pd.read_csv(likelihood_path, low_memory=False), "likelihood")
    common = material.index.intersection(trigger.index).intersection(likelihood.index)
    if len(common) != len(material) or len(common) != len(trigger) or len(common) != len(likelihood):
        raise RuntimeError(
            "Material/Trigger/likelihood registries do not have identical sample identities"
        )
    required_material = {"physical_event_id", "awc_shallow_mean_mm", "q_M_awc"}
    required_trigger = {
        "physical_event_id", "q_R", "rain_d7_antecedent_case_mm",
        "rain_d7_wrongtime_median_mm",
    }
    required_likelihood = {
        "physical_event_id", "q_R", "trigger_aligned_log_bf",
        "trigger_wrong_time_log_bf", "trigger_event_shuffle_log_bf",
        "event_shuffle_donor",
    }
    for name, frame, required in (
        ("Material", material, required_material),
        ("Trigger", trigger, required_trigger),
        ("likelihood", likelihood, required_likelihood),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{name} registry lacks columns: {sorted(missing)}")
    event_m = material.loc[common, "physical_event_id"].astype(str)
    event_r = trigger.loc[common, "physical_event_id"].astype(str)
    event_l = likelihood.loc[common, "physical_event_id"].astype(str)
    if not event_m.equals(event_r) or not event_m.equals(event_l):
        raise RuntimeError("physical_event_id disagrees across registries")
    output = pd.DataFrame(index=common)
    output["sample_id"] = common
    output["event_id"] = event_m.to_numpy()
    output["awc_shallow"] = pd.to_numeric(
        material.loc[common, "awc_shallow_mean_mm"], errors="coerce"
    ).to_numpy()
    output["q_awc"] = pd.to_numeric(
        material.loc[common, "q_M_awc"], errors="coerce"
    ).fillna(0.0).to_numpy()
    output["q_r"] = pd.to_numeric(
        trigger.loc[common, "q_R"], errors="coerce"
    ).fillna(0.0).to_numpy()
    likelihood_q = pd.to_numeric(
        likelihood.loc[common, "q_R"], errors="coerce"
    ).fillna(0.0).to_numpy()
    if not np.allclose(output.q_r.to_numpy(), likelihood_q, atol=1e-7):
        raise RuntimeError("q_R disagrees between raw Trigger and likelihood registries")
    for source, destination in (
        ("rain_d7_antecedent_case_mm", "rain_case"),
        ("rain_d7_wrongtime_median_mm", "rain_wrong"),
    ):
        output[destination] = pd.to_numeric(
            trigger.loc[common, source], errors="coerce"
        ).to_numpy()
    for source, destination in (
        ("trigger_aligned_log_bf", "logbf_aligned"),
        ("trigger_wrong_time_log_bf", "logbf_wrong"),
        ("trigger_event_shuffle_log_bf", "logbf_shuffle"),
    ):
        output[destination] = pd.to_numeric(
            likelihood.loc[common, source], errors="coerce"
        ).to_numpy()
    output["logbf_shuffle_donor"] = likelihood.loc[common, "event_shuffle_donor"].astype(str).to_numpy()
    return output.set_index("sample_id", drop=False)


def _constant_or_median(values: pd.Series, *, label: str, tolerance: float = 1e-6) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if not len(numeric):
        return float("nan")
    median = float(np.median(numeric))
    if np.max(np.abs(numeric - median)) > tolerance:
        # AWC is intentionally aggregated separately; event Trigger quantities
        # should otherwise be event-constant in the source registries.
        raise RuntimeError(f"{label} is not constant within its physical event")
    return median


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    return float(np.median(numeric)) if len(numeric) else float("nan")


def build_event_evidence(
    sample_ids: Sequence[str],
    expected_events: Sequence[str],
    registry: pd.DataFrame,
    family: str,
    *,
    raw_shuffle_donors: Mapping[str, str] | None = None,
    raw_donor_values: Mapping[str, float] | None = None,
) -> dict[str, EventEvidence]:
    ids = list(map(str, sample_ids))
    missing = sorted(set(ids) - set(registry.index))
    if missing:
        raise RuntimeError(f"Trigger registry lacks bundle samples: {missing[:5]}")
    frame = registry.loc[ids].copy()
    expected = pd.Series(list(map(str, expected_events)), index=ids)
    observed = frame.event_id.astype(str)
    if not np.array_equal(expected.to_numpy(), observed.to_numpy()):
        raise RuntimeError("bundle event IDs disagree with Trigger registry")
    result: dict[str, EventEvidence] = {}
    for event_id, group in frame.groupby("event_id", sort=True):
        q_r = float(np.clip(pd.to_numeric(group.q_r, errors="coerce").fillna(0.0).min(), 0.0, 1.0))
        if family == "RAW_D7_AWC":
            awc_valid = group[(group.q_awc > 0) & np.isfinite(group.awc_shallow) & (group.awc_shallow > 0)]
            awc_coverage = len(awc_valid) / max(len(group), 1)
            q = q_r * awc_coverage
            awc = float(np.median(awc_valid.awc_shallow)) if len(awc_valid) else float("nan")
            # Raw gridded rainfall legitimately varies among patches. Preserve
            # the event-level role by aggregating both D7 and AWC before taking
            # their ratio; never expose a patch-wise Trigger multiplier.
            case = _finite_median(group.rain_case)
            wrong = _finite_median(group.rain_wrong)
            aligned = math.log1p(max(case, 0.0) / awc) if q > 0 and awc > 0 else float("nan")
            wrong_time = math.log1p(max(wrong, 0.0) / awc) if q > 0 and awc > 0 else float("nan")
            donor = raw_shuffle_donors.get(str(event_id)) if raw_shuffle_donors else None
            donor_rain = raw_donor_values.get(donor, float("nan")) if raw_donor_values and donor else float("nan")
            shuffle = math.log1p(max(donor_rain, 0.0) / awc) if q > 0 and awc > 0 and math.isfinite(donor_rain) else float("nan")
        elif family == "FROZEN_LOGBF":
            q = q_r
            aligned = _constant_or_median(group.logbf_aligned, label=f"{event_id} aligned logBF")
            wrong_time = _constant_or_median(group.logbf_wrong, label=f"{event_id} wrong logBF")
            shuffle = _constant_or_median(group.logbf_shuffle, label=f"{event_id} shuffled logBF")
            donor_values = sorted(set(group.logbf_shuffle_donor.astype(str)))
            if len(donor_values) != 1:
                raise RuntimeError(f"{event_id} has inconsistent frozen Trigger donors")
            donor = donor_values[0]
        else:
            raise ValueError(f"unknown Trigger family: {family}")
        if q <= 0:
            aligned = wrong_time = shuffle = 0.0
        result[str(event_id)] = EventEvidence(
            event_id=str(event_id), q_r=float(q), aligned=float(aligned),
            wrong_time=float(wrong_time), shuffle=float(shuffle),
            n_samples=int(len(group)), donor_event=donor,
        )
    return result


def event_oracle_log_multipliers(
    bundle: gate_protocol.FoldBundle,
    grid: np.ndarray,
    *,
    device: str,
    sample_batch: int,
) -> dict[str, float]:
    """Compute event-wise training targets; call only on fitting bundles."""

    result: dict[str, float] = {}
    events = np.asarray(bundle.event_ids, object)
    grid_tensor = torch.as_tensor(grid, dtype=torch.float32, device=device)[None, :, None]
    for event_id in sorted(set(map(str, events))):
        indices = np.flatnonzero(events == event_id)
        loss_sum = torch.zeros(len(grid), dtype=torch.float64, device=device)
        valid_sum = 0.0
        for start in range(0, len(indices), sample_batch):
            selected = indices[start:start + sample_batch]
            visual = torch.as_tensor(bundle.visual_logits[selected], dtype=torch.float32, device=device).flatten(1)
            delta = torch.as_tensor(bundle.terrain_delta[selected], dtype=torch.float32, device=device).flatten(1)
            target = torch.as_tensor(bundle.mask[selected], dtype=torch.float32, device=device).flatten(1)
            valid = torch.as_tensor(bundle.valid[selected], dtype=torch.float32, device=device).flatten(1)
            logits = visual[:, None, :] + delta[:, None, :] * grid_tensor
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target[:, None, :].expand_as(logits), reduction="none"
            )
            loss_sum += (losses * valid[:, None, :]).sum((0, 2)).to(torch.float64)
            valid_sum += float(valid.sum().item())
        if valid_sum <= 0:
            raise RuntimeError(f"event {event_id} has no valid pixels")
        best = int(torch.argmin(loss_sum / valid_sum).item())
        result[event_id] = float(math.log(float(grid[best])))
    return result


def fit_dose_model(
    evidence: Mapping[str, EventEvidence],
    targets: Mapping[str, float],
    *,
    ridge_alpha: float,
    min_events: int,
) -> DoseModel:
    events = sorted(set(evidence) & set(targets))
    eligible = [
        event for event in events
        if evidence[event].q_r > 0 and math.isfinite(evidence[event].aligned)
        and math.isfinite(targets[event])
    ]
    if not eligible:
        return DoseModel(0.0, 1.0, 0.0, 0.0, 0.0, 0, "no_supported_fit_events")
    features = np.asarray([evidence[event].aligned for event in eligible], np.float64)
    target = np.asarray([targets[event] for event in eligible], np.float64)
    global_log = float(np.mean(target))
    center, scale = robust_center_scale(features)
    if len(eligible) < min_events or np.std(features) <= 1e-8:
        return DoseModel(center, scale, 0.0, global_log, global_log, len(eligible), "global_only")
    standardized = np.clip((features - center) / scale, -5.0, 5.0)[:, None]
    model = Ridge(alpha=ridge_alpha).fit(standardized, target)
    return DoseModel(
        center=center, scale=scale, coefficient=float(model.coef_[0]),
        intercept=float(model.intercept_), global_log_multiplier=global_log,
        n_fit_events=len(eligible), status="ridge",
    )


def confusion(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    prediction = np.asarray(prediction, bool)
    target = np.asarray(target, bool)
    valid = np.asarray(valid, bool)
    return {
        "tp": int(np.sum(prediction & target & valid)),
        "fp": int(np.sum(prediction & ~target & valid)),
        "fn": int(np.sum(~prediction & target & valid)),
        "tn": int(np.sum(~prediction & ~target & valid)),
    }


def metric_dict(counts: Mapping[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (int(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    errors = fp + fn
    return {
        **dict(counts), "errors": errors,
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def event_multiplier_array(
    bundle: gate_protocol.FoldBundle,
    evidence: Mapping[str, EventEvidence],
    model: DoseModel,
    condition: str,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    output = np.ones(len(bundle.sample_ids), dtype=np.float64)
    events = np.asarray(bundle.event_ids, object)
    for event_id in sorted(set(map(str, events))):
        if event_id not in evidence:
            raise RuntimeError(f"missing evidence for event {event_id}")
        item = evidence[event_id]
        if condition == "VT_identity":
            log_value = 0.0
        elif condition == "global_dose":
            log_value = model.global_log_multiplier
        elif condition == "aligned":
            log_value = float(model.predict(np.asarray([item.aligned]))[0])
        elif condition == "wrong_time":
            log_value = float(model.predict(np.asarray([item.wrong_time]))[0])
        elif condition == "event_shuffle":
            log_value = float(model.predict(np.asarray([item.shuffle]))[0])
        else:
            raise ValueError(f"unknown condition: {condition}")
        dose = float(dose_from_log(log_value, np.asarray([item.q_r]), minimum, maximum)[0])
        output[events == event_id] = dose
    return output


def evaluate_bundle(
    bundle: gate_protocol.FoldBundle,
    multiplier: np.ndarray,
    condition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visual = bundle.visual_logits.astype(np.float32)
    delta = bundle.terrain_delta.astype(np.float32)
    target = bundle.mask.astype(bool)
    valid = bundle.valid.astype(bool)
    logits = visual + delta * multiplier[:, None, None, None].astype(np.float32)
    prediction = logits >= bundle.threshold_logit
    vt_prediction = (visual + delta) >= bundle.threshold_logit
    total = metric_dict(confusion(prediction, target, valid))
    parent = metric_dict(confusion(vt_prediction, target, valid))
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[valid], -30.0, 30.0)))
    labels = target[valid]
    summary = {
        **total, "condition": condition,
        "ap": float(average_precision_score(labels, probability)) if labels.any() else 0.0,
        "n_valid_pixels": int(valid.sum()), "vt_iou": parent["iou"],
        "vt_errors": parent["errors"],
        "rer_vs_vt": (parent["errors"] - total["errors"]) / max(parent["errors"], 1),
        "mean_multiplier": float(np.mean(multiplier)),
        "min_multiplier": float(np.min(multiplier)),
        "max_multiplier": float(np.max(multiplier)),
    }
    event_rows: list[dict[str, Any]] = []
    events = np.asarray(bundle.event_ids, object)
    for event_id in sorted(set(map(str, events))):
        selected = np.flatnonzero(events == event_id)
        item = metric_dict(confusion(prediction[selected], target[selected], valid[selected]))
        item_parent = metric_dict(confusion(vt_prediction[selected], target[selected], valid[selected]))
        event_probability = 1.0 / (
            1.0 + np.exp(-np.clip(logits[selected][valid[selected]], -30.0, 30.0))
        )
        event_labels = target[selected][valid[selected]]
        event_rows.append({
            "condition": condition, "event_id": event_id, **item,
            "ap": float(average_precision_score(event_labels, event_probability))
            if event_labels.any() else 0.0,
            "n_valid_pixels": int(valid[selected].sum()),
            "vt_iou": item_parent["iou"], "vt_errors": item_parent["errors"],
            "rer_vs_vt": (item_parent["errors"] - item["errors"])
            / max(item_parent["errors"], 1),
            "multiplier": float(multiplier[selected[0]]),
        })
    return summary, event_rows


def aggregate_target_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["target_outer_fold", "family", "condition"]
    for key, group in frame.groupby(keys, sort=True):
        counts = {name: int(group[name].sum()) for name in ("tp", "fp", "fn", "tn")}
        metric = metric_dict(counts)
        weight = group.n_valid_pixels.to_numpy(dtype=np.float64)
        rows.append({
            **dict(zip(keys, key)), **metric,
            "ap": float(np.average(group.ap, weights=weight)),
            "ap_aggregation": "valid-pixel-weighted mean of disjoint inner-holdout AP",
            "n_valid_pixels": int(group.n_valid_pixels.sum()),
            "vt_errors": int(group.vt_errors.sum()),
            "rer_vs_vt": (int(group.vt_errors.sum()) - metric["errors"])
            / max(int(group.vt_errors.sum()), 1),
            "n_meta_folds": int(len(group)),
        })
    return pd.DataFrame(rows)


def validate_inputs(args: argparse.Namespace, folds: Sequence[int]) -> dict[str, Any]:
    registry = load_registries(
        args.material_registry, args.trigger_registry, args.trigger_likelihood
    )
    manifests = []
    for target_fold in folds:
        manifest = args.formal_input_root / f"target_outer{target_fold}/oof_manifest.json"
        split = args.formal_input_root / f"target_outer{target_fold}/gate_split.csv"
        if not manifest.is_file() or not split.is_file():
            raise FileNotFoundError(f"target_outer{target_fold} formal manifest/split is missing")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if int(payload.get("target_outer_fold", -1)) != target_fold:
            raise RuntimeError(f"target_outer{target_fold} manifest identity mismatch")
        manifests.append({
            "target_outer_fold": target_fold,
            "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "split_csv": str(split), "split_csv_sha256": sha256_file(split),
        })
    supported = registry.groupby("event_id").q_r.min()
    return {
        "n_registry_samples": int(len(registry)),
        "n_registry_events": int(registry.event_id.nunique()),
        "n_trigger_supported_events": int((supported > 0).sum()),
        "families": list(FAMILIES), "manifests": manifests,
    }


def main() -> int:
    args = parse_args()
    folds = parse_target_folds(args.target_folds)
    if not (0 < args.min_multiplier < 1 < args.max_multiplier):
        raise ValueError("multiplier interval must strictly contain one")
    if args.grid_size < 3 or args.ridge_alpha < 0 or args.min_supported_fit_events < 2:
        raise ValueError("invalid grid/ridge/min-supported-event setting")
    validation = validate_inputs(args, folds)
    if args.validate_only:
        print(json.dumps(json_safe(validation), indent=2, sort_keys=True, allow_nan=False))
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    registry = load_registries(
        args.material_registry, args.trigger_registry, args.trigger_likelihood
    )
    grid = np.exp(np.linspace(
        math.log(args.min_multiplier), math.log(args.max_multiplier), args.grid_size
    )).astype(np.float32)
    grid = np.sort(np.unique(np.append(grid, 1.0))).astype(np.float32)
    meta_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    access_log: list[dict[str, Any]] = []

    for target_fold in folds:
        print(f"[trigger-dose] target_outer={target_fold}", flush=True)
        target_root = args.formal_input_root / f"target_outer{target_fold}"
        bundles, _receipt = gate_protocol.load_formal_nested_bundles(
            target_root / "oof_manifest.json", target_fold=target_fold,
            split_csv=target_root / "gate_split.csv", seed=args.seed,
            access_log=access_log,
        )
        oracle = {
            bundle.fold: event_oracle_log_multipliers(
                bundle, grid, device=args.device, sample_batch=args.sample_batch
            )
            for bundle in bundles
        }
        for family in FAMILIES:
            for eval_index, eval_bundle in enumerate(bundles):
                fit_bundles = [item for index, item in enumerate(bundles) if index != eval_index]
                fit_ids = [sample for bundle in fit_bundles for sample in bundle.sample_ids]
                fit_events = [event for bundle in fit_bundles for event in bundle.event_ids]
                fit_event_set = sorted(set(map(str, fit_events)))
                if family == "RAW_D7_AWC":
                    fit_frame = registry.loc[fit_ids]
                    donor_rain = {
                        str(event): _finite_median(group.rain_case)
                        for event, group in fit_frame.groupby("event_id")
                        if pd.to_numeric(group.q_r, errors="coerce").fillna(0.0).min() > 0
                    }
                    donor_map = deterministic_event_donors(
                        eval_bundle.event_ids, tuple(donor_rain),
                        args.seed + 1009 * target_fold + 37 * int(eval_bundle.fold),
                    )
                else:
                    donor_rain = None
                    donor_map = None
                fit_evidence: dict[str, EventEvidence] = {}
                for bundle in fit_bundles:
                    values = build_event_evidence(
                        bundle.sample_ids, bundle.event_ids, registry, family
                    )
                    overlap = set(fit_evidence) & set(values)
                    if overlap:
                        raise RuntimeError(f"inner holdout events overlap: {sorted(overlap)}")
                    fit_evidence.update(values)
                fit_targets = {
                    event: value for bundle in fit_bundles
                    for event, value in oracle[bundle.fold].items()
                }
                model = fit_dose_model(
                    fit_evidence, fit_targets, ridge_alpha=args.ridge_alpha,
                    min_events=args.min_supported_fit_events,
                )
                eval_evidence = build_event_evidence(
                    eval_bundle.sample_ids, eval_bundle.event_ids, registry, family,
                    raw_shuffle_donors=donor_map, raw_donor_values=donor_rain,
                )
                model_rows.append({
                    "target_outer_fold": target_fold,
                    "meta_eval_bundle": int(eval_bundle.fold), "family": family,
                    "fit_event_ids": "|".join(fit_event_set),
                    "eval_event_ids": "|".join(sorted(eval_evidence)),
                    "n_fit_events": model.n_fit_events, "model_status": model.status,
                    "center": model.center, "scale": model.scale,
                    "coefficient": model.coefficient, "intercept": model.intercept,
                    "global_log_multiplier": model.global_log_multiplier,
                })
                for condition in CONDITIONS:
                    multiplier = event_multiplier_array(
                        eval_bundle, eval_evidence, model, condition,
                        args.min_multiplier, args.max_multiplier,
                    )
                    summary, rows = evaluate_bundle(eval_bundle, multiplier, condition)
                    meta_rows.append({
                        "target_outer_fold": target_fold,
                        "meta_eval_bundle": int(eval_bundle.fold), "family": family,
                        "model_status": model.status, **summary,
                    })
                    for row in rows:
                        item = eval_evidence[str(row["event_id"])]
                        event_rows.append({
                            "target_outer_fold": target_fold,
                            "meta_eval_bundle": int(eval_bundle.fold), "family": family,
                            "model_status": model.status, "q_R_effective": item.q_r,
                            "aligned_feature": item.aligned,
                            "wrong_time_feature": item.wrong_time,
                            "event_shuffle_feature": item.shuffle,
                            "event_shuffle_donor": item.donor_event, **row,
                        })

    meta_frame = pd.DataFrame(meta_rows)
    event_frame = pd.DataFrame(event_rows)
    model_frame = pd.DataFrame(model_rows)
    target_frame = aggregate_target_metrics(meta_frame)
    meta_frame.to_csv(args.outdir / "meta_fold_metrics.csv", index=False)
    event_frame.to_csv(args.outdir / "event_metrics.csv", index=False)
    model_frame.to_csv(args.outdir / "dose_model_receipts.csv", index=False)
    target_frame.to_csv(args.outdir / "target_protocol_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_frame = target_frame[target_frame.family == family].set_index(
            ["target_outer_fold", "condition"]
        )
        for target_fold in folds:
            rows = {condition: family_frame.loc[(target_fold, condition)] for condition in CONDITIONS}
            aligned = rows["aligned"]
            comparisons = {
                name: bool(aligned.iou > rows[name].iou and aligned.errors < rows[name].errors)
                for name in ("VT_identity", "global_dose", "wrong_time", "event_shuffle")
            }
            summary_rows.append({
                "target_outer_fold": target_fold, "family": family,
                "aligned_iou": float(aligned.iou), "aligned_rer_vs_vt": float(aligned.rer_vs_vt),
                "aligned_ap": float(aligned.ap),
                **{f"aligned_beats_{name}": value for name, value in comparisons.items()},
                "aligned_beats_all_controls": bool(all(comparisons.values())),
            })
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(args.outdir / "summary_table.csv", index=False)
    family_summary = []
    for family, group in summary_frame.groupby("family", sort=True):
        family_summary.append({
            "family": family, "n_target_protocols": int(len(group)),
            "positive_rer_protocols": int((group.aligned_rer_vs_vt > 0).sum()),
            "aligned_beats_all_controls_protocols": int(group.aligned_beats_all_controls.sum()),
            "mean_aligned_rer_vs_vt": float(group.aligned_rer_vs_vt.mean()),
            "development_gate_pass": bool(
                ((group.aligned_rer_vs_vt > 0) & group.aligned_beats_all_controls).sum() >= 3
            ),
        })
    summary = {
        "status": "complete",
        "scope": "nested-OOF development evidence; target outer-test never opened",
        "role_contract": (
            "Terrain delta is the only dense direction; R or R/AWC is an event-level "
            "positive multiplier; q_R=0 gives exact identity"
        ),
        "raw_saturation_aggregation": (
            "log1p(event-median D7 rainfall / event-median shallow AWC)"
        ),
        "multiplier_bounds": [args.min_multiplier, args.max_multiplier],
        "ridge_alpha": args.ridge_alpha,
        "outer_test_labels_loaded": False,
        "validation": validation,
        "registry_hashes": {
            "material": sha256_file(args.material_registry),
            "trigger": sha256_file(args.trigger_registry),
            "likelihood": sha256_file(args.trigger_likelihood),
        },
        "access_log": access_log,
        "results": family_summary,
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Sen12 event-level Trigger/AWC positive dose v1", "",
        "Nested-OOF development only; no target outer-test artifact was opened.", "",
        "| Family | Positive RER protocols | Aligned beats all controls | Mean aligned RER | Gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in family_summary:
        lines.append(
            f"| {row['family']} | {row['positive_rer_protocols']}/{row['n_target_protocols']} | "
            f"{row['aligned_beats_all_controls_protocols']}/{row['n_target_protocols']} | "
            f"{row['mean_aligned_rer_vs_vt']:+.4%} | "
            f"{'PASS' if row['development_gate_pass'] else 'FAIL'} |"
        )
    lines += [
        "", "A family is promotable only if at least three target protocols have positive "
        "RER and aligned evidence strictly beats identity, a global dose, wrong-time "
        "evidence, and event-shuffled evidence. Failure remains the frozen VT identity.", "",
    ]
    (args.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
