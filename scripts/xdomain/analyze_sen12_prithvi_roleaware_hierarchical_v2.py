#!/usr/bin/env python3
"""Strict analysis for Sen12 Prithvi hierarchical role-aware experiments.

The analyzer treats LOGO folds as geographic coverage within an optimization
seed, never as extra optimization repeats.  Comparisons are formed within each
seed/fold from the same selected checkpoint.  Inference first averages repeated
seed predictions for each sample/event; the hierarchical interval resamples
physical events and optimization seeds as separate levels.

The v2 contract is Terrain -> Material modulation -> Trigger dose.  Local
effects are evaluated only on the label-free eligibility fields emitted by the
trainer; this analyzer never constructs a subgroup from test labels, errors, or
IoU.  Trigger-unsupported folds remain required abstention audits but are
excluded from Trigger efficacy.

The training receipt does not contain frozen-VT probabilities or probability
sums.  Consequently, aligned-versus-VT supports IoU and thresholded error-flow
metrics only.  AP/Brier/NLL/area/fixed-FPR comparisons are reported for
same-checkpoint aligned-versus-control contrasts.  ``predicted_area`` is a
thresholded area, so it is never relabelled as a soft-area metric.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - sign-flip inference remains available.
    wilcoxon = None


RUN_SCHEMA = "sen12_prithvi_roleaware_hierarchical_run.v2"
DONE_SCHEMA = "sen12_prithvi_roleaware_hierarchical_done.v2"
CONFIG_SCHEMA = "sen12_prithvi_roleaware_hierarchical_config.v2"
CHECKPOINT_SCHEMA = "sen12_prithvi_roleaware_hierarchical_checkpoint.v2"
PARENT_SCHEMA = "sen12_frozen_vt_parent_manifest.v2"
ANALYSIS_SCHEMA = "sen12_prithvi_roleaware_hierarchical_analysis.v2"

MODES = ("material", "trigger", "joint")
DEFAULT_FOLDS = tuple(range(5))
CONTROLS_BY_MODE = {
    "material": ("aligned", "material_shuffle", "material_zero_q"),
    "trigger": (
        "aligned", "trigger_wrong_time", "trigger_event_shuffle", "trigger_zero_q",
    ),
    "joint": (
        "aligned", "material_shuffle", "material_zero_q", "trigger_wrong_time",
        "trigger_event_shuffle", "trigger_zero_q", "all_zero_q",
    ),
}
CONTROL_CONTEXTS = {
    "aligned": ("aligned", "aligned"),
    "material_shuffle": ("within-source/event-shuffle", "aligned"),
    "material_zero_q": ("zero-q", "aligned"),
    "trigger_wrong_time": ("aligned", "wrong-time"),
    "trigger_event_shuffle": ("aligned", "event-shuffle"),
    "trigger_zero_q": ("aligned", "zero-q"),
    "all_zero_q": ("zero-q", "zero-q"),
}
CONTRASTS = {
    "material_aligned_vs_vt": ("material", "aligned", "frozen_vt", "material"),
    "material_aligned_vs_shuffle": (
        "material", "aligned", "material_shuffle", "material",
    ),
    "material_aligned_vs_zero_q": (
        "material", "aligned", "material_zero_q", "material",
    ),
    "trigger_aligned_vs_vt": ("trigger", "aligned", "frozen_vt", "trigger"),
    "trigger_aligned_vs_wrong_time": (
        "trigger", "aligned", "trigger_wrong_time", "trigger",
    ),
    "trigger_aligned_vs_event_shuffle": (
        "trigger", "aligned", "trigger_event_shuffle", "trigger",
    ),
    "trigger_aligned_vs_zero_q": (
        "trigger", "aligned", "trigger_zero_q", "trigger",
    ),
    "joint_aligned_vs_vt": ("joint", "aligned", "frozen_vt", "any"),
    "joint_material_aligned_vs_shuffle": (
        "joint", "aligned", "material_shuffle", "material",
    ),
    "joint_material_aligned_vs_zero_q": (
        "joint", "aligned", "material_zero_q", "material",
    ),
    "joint_trigger_aligned_vs_wrong_time": (
        "joint", "aligned", "trigger_wrong_time", "trigger",
    ),
    "joint_trigger_aligned_vs_event_shuffle": (
        "joint", "aligned", "trigger_event_shuffle", "trigger",
    ),
    "joint_trigger_aligned_vs_zero_q": (
        "joint", "aligned", "trigger_zero_q", "trigger",
    ),
}

HASHED_ARTIFACTS = (
    "config.json", "command.txt", "run.log", "checkpoint.pt", "result.json",
    "per_sample.csv", "per_event.csv", "control_rows.csv",
    "same_checkpoint_controls.csv", "paired_control_receipts.csv",
)
SAMPLE_REQUIRED = (
    "sample_id", "event_id", "source_id", "mode", "control",
    "tp", "fp", "fn", "tn", "iou", "vt_iou", "errors", "vt_errors",
    "corrected", "harmed", "baseline_condition", "baseline_correct_count",
    "preserved_correct_count", "preservation_rate", "brier", "nll",
    "predicted_area", "true_area", "fixed_fpr_tp", "fixed_fpr_fn",
    "q_M", "q_R", "effective_q_M", "effective_q_R",
    "visual_uncertainty_mean", "visual_uncertainty_q75", "visual_uncertainty_q90",
    "terrain_support_pixel_count", "terrain_support_fraction", "material_scalar",
    "material_multiplier_abs_deviation_mean", "rain_contrast", "rain_gain",
    "material_local_effect_eligible", "trigger_local_effect_eligible",
    "joint_local_effect_eligible", "local_effect_subset_uses_test_label",
    "material_donor_sample_id", "material_donor_event_id",
    "trigger_donor_sample_id", "trigger_donor_event_id", "control_applicable",
    "material_shuffle_pair_applicable", "trigger_wrongtime_pair_applicable",
    "trigger_event_shuffle_pair_applicable", "material_delta_abs_mean",
    "trigger_delta_abs_mean",
    "trigger_changed_pixel_count", "trigger_terrain_overlap_pixel_count",
    "trigger_terrain_overlap_fraction", "trigger_support_overlap_100pct",
    "trigger_signed_direction_violation_count",
)
CONTROL_REQUIRED = (
    "sample_id", "event_id", "mode", "control", "checkpoint_selection",
    "material_context", "trigger_context", "q_M", "q_R", "effective_q_M",
    "effective_q_R", "visual_uncertainty_mean", "visual_uncertainty_q75",
    "visual_uncertainty_q90", "terrain_support_pixel_count",
    "terrain_support_fraction", "material_scalar",
    "material_multiplier_abs_deviation_mean", "rain_contrast", "rain_gain",
    "trigger_changed_pixel_count", "trigger_terrain_overlap_pixel_count",
    "trigger_support_overlap_100pct", "material_local_effect_eligible",
    "trigger_local_effect_eligible", "joint_local_effect_eligible",
    "local_effect_subset_uses_test_label", "trigger_event_shuffle_donor_scope",
    "control_applicable",
)
EVENT_REQUIRED = (
    "control", "event_id", "n_samples", "tp", "fp", "fn", "tn", "iou",
    "errors", "corrected", "harmed", "baseline_condition", "brier", "nll",
)
RECEIPT_REQUIRED = (
    "sample_id", "event_id", "mode", "control", "pair_applicable",
    "checkpoint_selection", "aligned_iou", "control_iou",
    "delta_iou_aligned_minus_control", "aligned_errors", "control_errors",
    "error_reduction_aligned_minus_control",
)
INTEGER_COLUMNS = (
    "tp", "fp", "fn", "tn", "errors", "vt_errors", "corrected", "harmed",
    "baseline_correct_count", "preserved_correct_count", "predicted_area",
    "true_area", "fixed_fpr_tp", "fixed_fpr_fn",
    "terrain_support_pixel_count", "trigger_changed_pixel_count",
    "trigger_terrain_overlap_pixel_count", "trigger_signed_direction_violation_count",
)
FLOAT_COLUMNS = (
    "iou", "vt_iou", "preservation_rate", "brier", "nll", "q_M", "q_R",
    "effective_q_M", "effective_q_R", "material_delta_abs_mean",
    "trigger_delta_abs_mean",
    "visual_uncertainty_mean", "visual_uncertainty_q75", "visual_uncertainty_q90",
    "terrain_support_fraction", "material_scalar",
    "material_multiplier_abs_deviation_mean", "rain_contrast", "rain_gain",
    "trigger_terrain_overlap_fraction",
)
BOOL_COLUMNS = (
    "control_applicable", "material_shuffle_pair_applicable",
    "trigger_wrongtime_pair_applicable", "trigger_event_shuffle_pair_applicable",
    "material_local_effect_eligible", "trigger_local_effect_eligible",
    "joint_local_effect_eligible", "local_effect_subset_uses_test_label",
    "trigger_support_overlap_100pct",
)
EPS = 1.0e-12


class AnalysisContractError(RuntimeError):
    """Raised when evidence violates the frozen analysis contract."""


_FILE_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    if key in _FILE_SHA_CACHE:
        return _FILE_SHA_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _FILE_SHA_CACHE[key] = value
    return value


def strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise AnalysisContractError(f"non-finite JSON constant in {path}: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise AnalysisContractError(f"duplicate JSON key in {path}: {key}")
            payload[key] = value
        return payload

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject,
            object_pairs_hook=unique,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisContractError(f"cannot parse strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisContractError(f"JSON root must be an object: {path}")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise AnalysisContractError(f"refusing to publish empty analysis CSV: {path.name}")
    buffer = io.StringIO()
    clean = frame.replace([np.inf, -np.inf], np.nan)
    clean.to_csv(buffer, index=False, lineterminator="\n", na_rep="null")
    atomic_text(path, buffer.getvalue())


def _read_csv(path: Path, required: Sequence[str], allow_empty: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise AnalysisContractError(f"cannot read CSV {path}: {error}") from error
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AnalysisContractError(f"CSV {path} missing columns: {missing}")
    if frame.empty and not allow_empty:
        raise AnalysisContractError(f"CSV is empty: {path}")
    return frame


def _coerce_bool(series: pd.Series, label: str) -> pd.Series:
    mapping = {
        True: True, False: False, 1: True, 0: False, 1.0: True, 0.0: False,
        "True": True, "False": False, "true": True, "false": False,
        "1": True, "0": False,
    }
    mapped = series.map(mapping)
    if mapped.isna().any():
        raise AnalysisContractError(f"{label} contains non-boolean values")
    return mapped.astype(bool)


def _require_identity(payload: Mapping[str, Any], label: str, mode: str, seed: int, fold: int) -> None:
    try:
        observed_seed = int(payload.get("seed", -1))
        observed_fold = int(payload.get("fold", -1))
    except (TypeError, ValueError) as error:
        raise AnalysisContractError(f"invalid {label} seed/fold identity") from error
    if payload.get("mode") != mode or observed_seed != seed or observed_fold != fold:
        raise AnalysisContractError(
            f"{label} identity mismatch: expected {mode}/seed{seed}/fold{fold}"
        )


def _verify_artifact_hashes(run: Path, done: Mapping[str, Any]) -> None:
    hashes_path = run / "hashes.json"
    expected_hash = done.get("hashes_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise AnalysisContractError(f"DONE lacks a valid hashes_sha256: {run}")
    if sha256_file(hashes_path) != expected_hash:
        raise AnalysisContractError(f"hashes.json drift: {run}")
    hashes = strict_json(hashes_path)
    for name in HASHED_ARTIFACTS:
        path = run / name
        if not path.is_file():
            raise AnalysisContractError(f"missing hashed artifact {path}")
        expected = hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise AnalysisContractError(f"hashes.json lacks valid hash for {name}: {run}")
        if sha256_file(path) != expected:
            raise AnalysisContractError(f"artifact hash drift for {path}")


def _parent_key(parent: Mapping[str, Any], seed: int, fold: int) -> tuple[Any, ...]:
    if parent.get("schema_version") != PARENT_SCHEMA:
        raise AnalysisContractError("invalid parent_identity schema")
    try:
        parent_seed = int(parent["seed"])
        parent_fold = int(parent["fold"])
        threshold = float(parent["threshold"])
        visual = parent["visual"]
        terrain = parent["terrain"]
        visual_sha = str(visual["sha256"])
        terrain_sha = str(terrain["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisContractError("malformed parent checkpoint identity") from error
    if parent_seed != seed or parent_fold != fold:
        raise AnalysisContractError("parent checkpoint seed/fold mismatch")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise AnalysisContractError("invalid parent visual threshold")
    for label, signature, expected in (
        ("visual", visual, visual_sha), ("terrain", terrain, terrain_sha),
    ):
        if len(expected) != 64:
            raise AnalysisContractError(f"invalid parent {label} SHA")
        try:
            path = Path(signature["path"])
        except (KeyError, TypeError) as error:
            raise AnalysisContractError(f"missing parent {label} path") from error
        if not path.is_file():
            raise AnalysisContractError(f"parent {label} checkpoint is absent: {path}")
        if sha256_file(path) != expected:
            raise AnalysisContractError(f"parent {label} checkpoint SHA drift: {path}")
    return (parent_seed, parent_fold, threshold, visual_sha, terrain_sha)


def _signature_key(signature: Mapping[str, Any], label: str) -> tuple[str, str]:
    """Verify an immutable file signature and return its path/SHA identity."""

    try:
        path = Path(signature["path"])
        expected = str(signature["sha256"])
    except (KeyError, TypeError) as error:
        raise AnalysisContractError(f"malformed {label} signature") from error
    if len(expected) != 64 or not path.is_file():
        raise AnalysisContractError(f"invalid or absent {label}: {path}")
    if sha256_file(path) != expected:
        raise AnalysisContractError(f"{label} SHA drift: {path}")
    return str(path.resolve()), expected


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - binary receipt must fail closed.
        raise AnalysisContractError(f"cannot load checkpoint receipt {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise AnalysisContractError(f"checkpoint receipt is not a mapping: {path}")
    return payload


def _validate_selection_receipts(
    config: Mapping[str, Any], result: Mapping[str, Any], checkpoint: Mapping[str, Any],
    done: Mapping[str, Any], run: Path,
) -> None:
    """Validate epoch-0 identity, validation-only selection, and abstention."""

    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise AnalysisContractError(f"invalid checkpoint schema: {run}")
    history = result.get("history")
    if not isinstance(history, list) or not history:
        raise AnalysisContractError(f"missing selection history: {run}")
    epochs: list[int] = []
    for row in history:
        if not isinstance(row, Mapping):
            raise AnalysisContractError(f"malformed selection history row: {run}")
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisContractError(f"invalid selection epoch: {run}") from error
        epochs.append(epoch)
        for receipt_name in (
            "material_outer_val_receipt", "trigger_inner_supported_val_receipt",
        ):
            receipt = row.get(receipt_name)
            if not isinstance(receipt, Mapping) or receipt.get("split_usage") != (
                "validation-only; test-not-accessed"
            ):
                raise AnalysisContractError(f"test leakage in {receipt_name}: {run}")
    if epochs[0] != 0 or len(set(epochs)) != len(epochs):
        raise AnalysisContractError(f"epoch0 identity is not a unique first candidate: {run}")
    identity = history[0]
    if identity.get("identity_candidate") is not True:
        raise AnalysisContractError(f"epoch0 identity receipt absent: {run}")
    try:
        best_epoch = int(result["best_epoch"])
        checkpoint_epoch = int(checkpoint["best_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisContractError(f"invalid selected epoch: {run}") from error
    if best_epoch != checkpoint_epoch or best_epoch not in epochs:
        raise AnalysisContractError(f"selected epoch differs across receipts: {run}")
    selected_identity = best_epoch == 0
    for label, payload in (("result", result), ("checkpoint", checkpoint), ("DONE", done)):
        if payload.get("selected_identity_abstain") is not selected_identity:
            raise AnalysisContractError(f"{label} identity-abstain mismatch: {run}")
    if result.get("epoch0_identity_was_candidate") is not True or checkpoint.get(
        "epoch0_identity_was_candidate"
    ) is not True:
        raise AnalysisContractError(f"epoch0 candidate receipt missing: {run}")
    gates = [config.get("selection_gate"), result.get("selection_gate"), checkpoint.get("selection_gate")]
    config_gate = config.get("selection_gate")
    if not isinstance(config_gate, Mapping) or config_gate.get("epoch0_identity_is_candidate") is not True:
        raise AnalysisContractError(f"config omits epoch0 identity candidate: {run}")
    for gate in gates:
        if not isinstance(gate, Mapping) or gate.get("test_used_for_selection") is not False:
            raise AnalysisContractError(f"selection gate permits test access: {run}")
        try:
            if float(gate["minimum_score_gain_vs_epoch0"]) <= 0:
                raise ValueError
            if int(gate["minimum_net_error_reduction_vs_epoch0"]) < 1:
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisContractError(f"invalid selection gate: {run}") from error
    selected = next(row for row in history if int(row["epoch"]) == best_epoch)
    if checkpoint.get("selection") != selected.get("selection_contract"):
        raise AnalysisContractError(f"selected checkpoint contract mismatch: {run}")
    if best_epoch > 0:
        if selected.get("passes_minimum_gain_gate") is not True:
            raise AnalysisContractError(f"trained checkpoint bypassed gain gate: {run}")
        if selected.get("net_error_gate_passed") is not True:
            raise AnalysisContractError(f"trained checkpoint bypassed net-error gate: {run}")


def _local_role(mode: str) -> str:
    return {"material": "material", "trigger": "trigger", "joint": "joint"}[mode]


def _validate_sample_frame(frame: pd.DataFrame, run: Path, mode: str) -> pd.DataFrame:
    frame = frame.copy()
    expected_controls = set(CONTROLS_BY_MODE[mode])
    observed_controls = set(frame["control"].astype(str))
    if observed_controls != expected_controls:
        raise AnalysisContractError(
            f"control inventory mismatch in {run}: {sorted(observed_controls)}"
        )
    if set(frame["mode"].astype(str)) != {mode}:
        raise AnalysisContractError(f"per_sample mode mismatch: {run}")
    for column in INTEGER_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any() or not np.equal(numeric, np.floor(numeric)).all():
            raise AnalysisContractError(f"{column} must be nonnegative integers: {run}")
        frame[column] = numeric.astype(np.int64)
    for column in FLOAT_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise AnalysisContractError(f"{column} must be finite: {run}")
        frame[column] = numeric.astype(float)
    for column in BOOL_COLUMNS:
        frame[column] = _coerce_bool(frame[column], f"{run}:{column}")
    if (frame[["q_M", "q_R", "effective_q_M", "effective_q_R"]].to_numpy() < -EPS).any():
        raise AnalysisContractError(f"negative quality support in {run}")
    if frame["local_effect_subset_uses_test_label"].any():
        raise AnalysisContractError(f"local eligibility uses a test label/error/metric: {run}")
    if (frame["terrain_support_fraction"] < -EPS).any() or (
        frame["terrain_support_fraction"] > 1.0 + EPS
    ).any():
        raise AnalysisContractError(f"invalid Terrain support fraction: {run}")
    expected_eligibility = {
        "material_local_effect_eligible": (
            (frame["effective_q_M"] > 0) & (frame["terrain_support_pixel_count"] > 0)
        ),
        "trigger_local_effect_eligible": (
            (frame["effective_q_R"] > 0) & (frame["terrain_support_pixel_count"] > 0)
        ),
        "joint_local_effect_eligible": (
            (frame["effective_q_M"] > 0) & (frame["effective_q_R"] > 0)
            & (frame["terrain_support_pixel_count"] > 0)
        ),
    }
    for column, expected in expected_eligibility.items():
        if not np.array_equal(frame[column].to_numpy(dtype=bool), expected.to_numpy(dtype=bool)):
            raise AnalysisContractError(f"trainer eligibility identity violated for {column}: {run}")
    if set(frame["baseline_condition"].astype(str)) != {"frozen_VT"}:
        raise AnalysisContractError(f"baseline must be frozen_VT: {run}")
    if frame.duplicated(["sample_id", "control"]).any():
        raise AnalysisContractError(f"duplicate sample/control rows: {run}")
    counts = frame[["tp", "fp", "fn"]].sum(axis=1)
    expected_iou = frame["tp"] / counts.clip(lower=1)
    if not np.allclose(frame["iou"], expected_iou, atol=1e-10, rtol=0.0):
        raise AnalysisContractError(f"IoU/count identity mismatch: {run}")
    if not np.array_equal(frame["errors"], frame["fp"] + frame["fn"]):
        raise AnalysisContractError(f"error/count identity mismatch: {run}")
    if not np.array_equal(
        frame["corrected"] - frame["harmed"], frame["vt_errors"] - frame["errors"]
    ):
        raise AnalysisContractError(f"corrected-harmed identity mismatch: {run}")
    grouped = frame.groupby("sample_id", sort=False)
    for sample_id, group in grouped:
        if set(group["control"].astype(str)) != expected_controls:
            raise AnalysisContractError(f"incomplete controls for sample {sample_id}: {run}")
        for column in ("event_id", "source_id", "q_M", "q_R", "vt_iou", "vt_errors"):
            if group[column].nunique(dropna=False) != 1:
                raise AnalysisContractError(f"{column} drifts across controls for {sample_id}: {run}")
        for column in (
            "terrain_support_pixel_count", "terrain_support_fraction",
            "visual_uncertainty_mean", "visual_uncertainty_q75", "visual_uncertainty_q90",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise AnalysisContractError(f"label-free eligibility input drifts across controls: {column}: {run}")
    aligned = frame.loc[frame["control"] == "aligned"]
    if not aligned["control_applicable"].all():
        raise AnalysisContractError(f"aligned rows must be applicable: {run}")
    zero_control = {
        "material": "material_zero_q", "trigger": "trigger_zero_q", "joint": "all_zero_q",
    }[mode]
    zero = frame.loc[frame["control"] == zero_control]
    if not np.allclose(zero["iou"], zero["vt_iou"], atol=0.0, rtol=0.0):
        raise AnalysisContractError(f"q0 exact fallback IoU violated: {run}")
    if not np.array_equal(zero["errors"], zero["vt_errors"]):
        raise AnalysisContractError(f"q0 exact fallback errors violated: {run}")
    if (zero["corrected"] != 0).any() or (zero["harmed"] != 0).any():
        raise AnalysisContractError(f"q0 exact fallback error flow violated: {run}")
    if mode in ("material", "joint"):
        relevant = zero if mode == "material" else frame.loc[frame["control"] == "all_zero_q"]
        if (relevant["effective_q_M"].abs() > EPS).any():
            raise AnalysisContractError(f"q0 fallback has nonzero effective_q_M: {run}")
        m_q0 = frame["effective_q_M"].abs() <= EPS
        if (
            (frame.loc[m_q0, "material_delta_abs_mean"].abs() > EPS).any()
            or (frame.loc[m_q0, "material_multiplier_abs_deviation_mean"].abs() > EPS).any()
        ):
            raise AnalysisContractError(f"Material q0 receipt is not a no-op: {run}")
    if mode in ("trigger", "joint"):
        relevant = zero if mode == "trigger" else frame.loc[frame["control"] == "all_zero_q"]
        if (relevant["effective_q_R"].abs() > EPS).any():
            raise AnalysisContractError(f"q0 fallback has nonzero effective_q_R: {run}")
        r_q0 = frame["effective_q_R"].abs() <= EPS
        if (
            (frame.loc[r_q0, "trigger_delta_abs_mean"].abs() > EPS).any()
            or (frame.loc[r_q0, "trigger_changed_pixel_count"] != 0).any()
        ):
            raise AnalysisContractError(f"Trigger q0 receipt is not a no-op: {run}")
        trigger_rows = frame.loc[
            frame["control"].isin(
                ["aligned", "trigger_wrong_time", "trigger_event_shuffle"]
            )
        ]
        if not trigger_rows["trigger_support_overlap_100pct"].all():
            raise AnalysisContractError(f"Trigger leaves Terrain support: {run}")
        if not np.array_equal(
            trigger_rows["trigger_changed_pixel_count"],
            trigger_rows["trigger_terrain_overlap_pixel_count"],
        ):
            raise AnalysisContractError(f"Trigger/Terrain overlap count mismatch: {run}")
        if (trigger_rows["trigger_signed_direction_violation_count"] != 0).any():
            raise AnalysisContractError(f"Trigger signed-direction violation: {run}")
        no_terrain = trigger_rows["terrain_support_pixel_count"] == 0
        if (trigger_rows.loc[no_terrain, "trigger_changed_pixel_count"] != 0).any():
            raise AnalysisContractError(f"Trigger changes pixels without Terrain support: {run}")
        donor_scope = frame.loc[
            frame["control"] == "trigger_event_shuffle", "trigger_event_shuffle_donor_scope"
        ].astype(str)
        if len(donor_scope) and set(donor_scope) != {"outer-train-supported-events-only"}:
            raise AnalysisContractError(f"Trigger shuffle donor escapes outer train: {run}")
    if mode == "material":
        no_effective_support = frame["effective_q_M"].abs() <= EPS
    elif mode == "trigger":
        no_effective_support = frame["effective_q_R"].abs() <= EPS
    else:
        no_effective_support = (
            (frame["effective_q_M"].abs() <= EPS)
            & (frame["effective_q_R"].abs() <= EPS)
        )
    fallback = frame.loc[no_effective_support]
    if (
        not np.allclose(fallback["iou"], fallback["vt_iou"], atol=0.0, rtol=0.0)
        or not np.array_equal(fallback["errors"], fallback["vt_errors"])
        or (fallback["corrected"] != 0).any()
        or (fallback["harmed"] != 0).any()
    ):
        raise AnalysisContractError(f"effective-q0 exact frozen-VT fallback violated: {run}")
    return frame


def _validate_control_receipts(
    sample: pd.DataFrame, controls: pd.DataFrame, same_controls: pd.DataFrame,
    paired: pd.DataFrame, run: Path,
) -> None:
    for frame, label in ((controls, "control_rows"), (same_controls, "same_checkpoint_controls")):
        if set(frame["checkpoint_selection"].astype(str)) != {
            "same-aligned-validation-checkpoint"
        }:
            raise AnalysisContractError(f"{label} checkpoint mismatch: {run}")
        keys = frame[["sample_id", "control"]].astype(str)
        if keys.duplicated().any():
            raise AnalysisContractError(f"duplicate {label} recipients: {run}")
        for control, (material_context, trigger_context) in CONTROL_CONTEXTS.items():
            subset = frame.loc[frame["control"].astype(str) == control]
            if subset.empty:
                continue
            if set(subset["material_context"].astype(str)) != {material_context} or set(
                subset["trigger_context"].astype(str)
            ) != {trigger_context}:
                raise AnalysisContractError(f"{label} context identity mismatch for {control}: {run}")
    left = controls.sort_values(["sample_id", "control"]).reset_index(drop=True)
    right = same_controls.sort_values(["sample_id", "control"]).reset_index(drop=True)
    common = sorted(set(left.columns) & set(right.columns))
    if list(left[["sample_id", "control"]].itertuples(index=False, name=None)) != list(
        right[["sample_id", "control"]].itertuples(index=False, name=None)
    ) or not left[common].astype(str).equals(right[common].astype(str)):
        raise AnalysisContractError(f"same_checkpoint_controls differs from control_rows: {run}")
    sample_keys = set(map(tuple, sample[["sample_id", "control"]].astype(str).to_numpy()))
    control_keys = set(map(tuple, controls[["sample_id", "control"]].astype(str).to_numpy()))
    if sample_keys != control_keys:
        raise AnalysisContractError(f"control receipt/sample inventory mismatch: {run}")
    expected_pairs = sample.loc[
        (sample["control"] != "aligned") & sample["control_applicable"],
        ["sample_id", "control"],
    ].astype(str)
    expected = set(map(tuple, expected_pairs.to_numpy()))
    if paired.empty:
        observed: set[tuple[str, str]] = set()
    else:
        if set(paired["checkpoint_selection"].astype(str)) != {
            "same-aligned-validation-checkpoint"
        } or not _coerce_bool(paired["pair_applicable"], f"{run}:pair_applicable").all():
            raise AnalysisContractError(f"invalid paired checkpoint receipt: {run}")
        observed = set(map(tuple, paired[["sample_id", "control"]].astype(str).to_numpy()))
    if expected != observed:
        raise AnalysisContractError(f"paired control applicability mismatch: {run}")
    if not paired.empty:
        control_rows = sample.loc[sample["control"] != "aligned"].copy()
        aligned_rows = sample.loc[sample["control"] == "aligned", [
            "sample_id", "iou", "errors",
        ]].rename(columns={"iou": "expected_aligned_iou", "errors": "expected_aligned_errors"})
        expected_receipts = control_rows.merge(aligned_rows, on="sample_id", validate="many_to_one")
        observed_receipts = paired.merge(
            expected_receipts,
            on=["sample_id", "event_id", "mode", "control"],
            how="left", validate="one_to_one", suffixes=("_receipt", "_sample"),
        )
        if observed_receipts["expected_aligned_iou"].isna().any():
            raise AnalysisContractError(f"paired receipt has unknown recipient: {run}")
        numeric_checks = {
            "aligned_iou": "expected_aligned_iou",
            "control_iou": "iou",
            "aligned_errors": "expected_aligned_errors",
            "control_errors": "errors",
        }
        for receipt_column, expected_column in numeric_checks.items():
            if not np.allclose(
                pd.to_numeric(observed_receipts[receipt_column], errors="coerce"),
                pd.to_numeric(observed_receipts[expected_column], errors="coerce"),
                atol=1e-12, rtol=0.0,
            ):
                raise AnalysisContractError(f"paired receipt value mismatch for {receipt_column}: {run}")
        expected_delta = observed_receipts["expected_aligned_iou"] - observed_receipts["iou"]
        expected_error_reduction = (
            observed_receipts["errors"] - observed_receipts["expected_aligned_errors"]
        )
        if not np.allclose(
            observed_receipts["delta_iou_aligned_minus_control"], expected_delta,
            atol=1e-12, rtol=0.0,
        ) or not np.array_equal(
            observed_receipts["error_reduction_aligned_minus_control"], expected_error_reduction,
        ):
            raise AnalysisContractError(f"paired receipt delta mismatch: {run}")


def load_run(run: Path, mode: str, seed: int, fold: int) -> dict[str, Any]:
    required_paths = [run / name for name in (*HASHED_ARTIFACTS, "hashes.json", "DONE.json")]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise AnalysisContractError(f"incomplete run {run}: {missing}")
    done = strict_json(run / "DONE.json")
    config = strict_json(run / "config.json")
    result = strict_json(run / "result.json")
    checkpoint = _load_checkpoint(run / "checkpoint.pt")
    if done.get("schema_version") != DONE_SCHEMA or done.get("status") != "complete":
        raise AnalysisContractError(f"invalid DONE schema/status: {run}")
    if result.get("schema_version") != RUN_SCHEMA or result.get("status") != "complete":
        raise AnalysisContractError(f"invalid result schema/status: {run}")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise AnalysisContractError(f"invalid config schema: {run}")
    _require_identity(done, "DONE", mode, seed, fold)
    _require_identity(config, "config", mode, seed, fold)
    _require_identity(result, "result", mode, seed, fold)
    _require_identity(checkpoint, "checkpoint", mode, seed, fold)
    if done.get("same_checkpoint_controls") is not True:
        raise AnalysisContractError(f"same-checkpoint DONE receipt absent: {run}")
    _verify_artifact_hashes(run, done)
    _validate_selection_receipts(config, result, checkpoint, done, run)
    parent = result.get("parent_identity")
    if not isinstance(parent, dict):
        raise AnalysisContractError(f"parent_identity missing: {run}")
    parent_key = _parent_key(parent, seed, fold)
    if checkpoint.get("parent_identity") != parent:
        raise AnalysisContractError(f"checkpoint/result parent identity mismatch: {run}")
    result_material = result.get("material_schema")
    checkpoint_material = checkpoint.get("material_schema")
    if not isinstance(result_material, Mapping) or result_material != checkpoint_material:
        raise AnalysisContractError(f"checkpoint/result Material schema mismatch: {run}")
    material_schema_key = _signature_key(result_material, "Material schema")
    if result.get("material_feature_names") != checkpoint.get("material_feature_names"):
        raise AnalysisContractError(f"Material feature-name receipt mismatch: {run}")
    if not result.get("material_feature_names"):
        raise AnalysisContractError(f"empty Material feature-name receipt: {run}")
    for payload, label in ((result, "result"), (checkpoint, "checkpoint")):
        local_contract = payload.get("local_effect_audit_contract")
        if not isinstance(local_contract, Mapping) or local_contract.get(
            "test_label_used_for_subgroup_or_threshold"
        ) is not False:
            raise AnalysisContractError(f"{label} permits label-derived local subgroup: {run}")
    sample = _read_csv(run / "per_sample.csv", SAMPLE_REQUIRED)
    sample = _validate_sample_frame(sample, run, mode)
    event = _read_csv(run / "per_event.csv", EVENT_REQUIRED)
    controls = _read_csv(run / "control_rows.csv", CONTROL_REQUIRED)
    same_controls = _read_csv(run / "same_checkpoint_controls.csv", CONTROL_REQUIRED)
    # A fully unsupported Trigger test fold legitimately has no applicable
    # paired-control receipt.  The primary sample/event/control CSVs remain
    # nonempty and are still required above.
    paired = _read_csv(
        run / "paired_control_receipts.csv", RECEIPT_REQUIRED, allow_empty=True
    )
    _validate_control_receipts(sample, controls, same_controls, paired, run)
    receipt_columns = (
        "effective_q_M", "effective_q_R", "terrain_support_pixel_count",
        "terrain_support_fraction", "material_local_effect_eligible",
        "trigger_local_effect_eligible", "joint_local_effect_eligible",
        "local_effect_subset_uses_test_label", "trigger_changed_pixel_count",
        "trigger_terrain_overlap_pixel_count", "trigger_support_overlap_100pct",
    )
    sample_receipt = sample[["sample_id", "control", *receipt_columns]].copy()
    control_receipt = controls[["sample_id", "control", *receipt_columns]].copy()
    merged_receipt = sample_receipt.merge(
        control_receipt, on=["sample_id", "control"], suffixes=("_sample", "_control"),
        validate="one_to_one",
    )
    for column in receipt_columns:
        if not merged_receipt[f"{column}_sample"].astype(str).equals(
            merged_receipt[f"{column}_control"].astype(str)
        ):
            raise AnalysisContractError(f"sample/control local receipt mismatch for {column}: {run}")
    if set(event["control"].astype(str)) != set(CONTROLS_BY_MODE[mode]):
        raise AnalysisContractError(f"per_event control inventory mismatch: {run}")
    expected_event_counts = (
        sample.groupby(["control", "event_id"], sort=True).size().rename("expected").reset_index()
    )
    observed_event_counts = event[["control", "event_id", "n_samples"]].copy()
    observed_event_counts["n_samples"] = pd.to_numeric(
        observed_event_counts["n_samples"], errors="coerce"
    )
    event_check = expected_event_counts.merge(
        observed_event_counts, on=["control", "event_id"], how="outer", validate="one_to_one"
    )
    if (
        event_check[["expected", "n_samples"]].isna().any().any()
        or not np.array_equal(event_check["expected"], event_check["n_samples"])
    ):
        raise AnalysisContractError(f"per_event/sample aggregation inventory mismatch: {run}")
    expected_event = sample.groupby(["control", "event_id"], sort=True)[
        ["tp", "fp", "fn", "tn", "errors", "corrected", "harmed"]
    ].sum().reset_index()
    observed_event = event[[
        "control", "event_id", "tp", "fp", "fn", "tn", "errors", "corrected", "harmed",
    ]].copy()
    for column in ("tp", "fp", "fn", "tn", "errors", "corrected", "harmed"):
        observed_event[column] = pd.to_numeric(observed_event[column], errors="coerce")
    event_values = expected_event.merge(
        observed_event, on=["control", "event_id"], suffixes=("_expected", "_observed"),
        how="outer", validate="one_to_one",
    )
    if event_values.isna().any().any():
        raise AnalysisContractError(f"per_event contains missing/non-numeric aggregates: {run}")
    for column in ("tp", "fp", "fn", "tn", "errors", "corrected", "harmed"):
        if not np.array_equal(
            event_values[f"{column}_expected"], event_values[f"{column}_observed"]
        ):
            raise AnalysisContractError(f"per_event aggregate mismatch for {column}: {run}")
    expected_event_iou = event_values["tp_expected"] / (
        event_values["tp_expected"] + event_values["fp_expected"]
        + event_values["fn_expected"]
    ).clip(lower=1)
    observed_iou = event[["control", "event_id", "iou"]].merge(
        expected_event[["control", "event_id"]], on=["control", "event_id"],
        validate="one_to_one",
    )["iou"]
    if not np.allclose(pd.to_numeric(observed_iou), expected_event_iou, atol=1e-10, rtol=0.0):
        raise AnalysisContractError(f"per_event IoU/count identity mismatch: {run}")
    test = result.get("test")
    if not isinstance(test, dict) or not isinstance(test.get("controls"), dict):
        raise AnalysisContractError(f"result test controls missing: {run}")
    if set(test["controls"]) != set(CONTROLS_BY_MODE[mode]):
        raise AnalysisContractError(f"result control inventory mismatch: {run}")
    for control, payload in test["controls"].items():
        try:
            values = [float(payload[name]) for name in (
                "iou", "ap", "brier_mean", "nll_mean", "area_abs_error_mean",
                "fixed_fpr_recall",
            )]
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisContractError(f"fold metric schema missing for {control}: {run}") from error
        if not np.isfinite(values).all():
            raise AnalysisContractError(f"non-finite fold metrics for {control}: {run}")
        rows = sample.loc[sample["control"] == control]
        for key in ("tp", "fp", "fn", "tn", "errors", "corrected", "harmed"):
            source_key = {
                "corrected": "corrected_vs_frozen_vt",
                "harmed": "harmed_vs_frozen_vt",
            }.get(key, key)
            if int(payload[source_key]) != int(rows[key].sum()):
                raise AnalysisContractError(f"result/sample aggregate mismatch {control}/{key}: {run}")
        pooled_iou = int(payload["tp"]) / max(
            int(payload["tp"]) + int(payload["fp"]) + int(payload["fn"]), 1
        )
        if not math.isclose(float(payload["iou"]), pooled_iou, abs_tol=1e-10, rel_tol=0.0):
            raise AnalysisContractError(f"result IoU/count identity mismatch for {control}: {run}")
    vt = test.get("vt")
    if not isinstance(vt, Mapping):
        raise AnalysisContractError(f"frozen VT fold receipt missing: {run}")
    vt_errors = int(vt.get("fp", -1)) + int(vt.get("fn", -1))
    aligned = sample.loc[sample["control"] == "aligned"]
    if int(aligned["vt_errors"].sum()) != vt_errors:
        raise AnalysisContractError(f"frozen VT error receipt mismatch: {run}")
    if bool(result.get("selected_identity_abstain")):
        if (
            not np.allclose(sample["iou"], sample["vt_iou"], atol=0.0, rtol=0.0)
            or not np.array_equal(sample["errors"], sample["vt_errors"])
            or (sample["corrected"] != 0).any()
            or (sample["harmed"] != 0).any()
        ):
            raise AnalysisContractError(f"selected epoch0 identity is not exact frozen VT: {run}")
    return {
        "run": run, "mode": mode, "seed": seed, "fold": fold, "result": result,
        "sample": sample, "event": event, "parent_key": parent_key,
        "material_schema_key": material_schema_key,
        "material_feature_names": tuple(map(str, result["material_feature_names"])),
        "selected_identity_abstain": bool(result.get("selected_identity_abstain")),
        "schemas": {
            "per_sample": tuple(sample.columns), "per_event": tuple(event.columns),
            "control_rows": tuple(controls.columns),
            "same_checkpoint_controls": tuple(same_controls.columns),
            "paired_control_receipts": tuple(paired.columns),
        },
    }


def discover_seeds(runs_root: Path) -> tuple[int, ...]:
    if not runs_root.is_dir():
        raise AnalysisContractError(f"runs root is absent: {runs_root}")
    pattern = re.compile(r"^seed([0-9]+)$")
    seeds = sorted(
        int(match.group(1)) for path in runs_root.iterdir()
        if path.is_dir() and (match := pattern.fullmatch(path.name))
    )
    if not seeds:
        raise AnalysisContractError(f"no seed directories found: {runs_root}")
    return tuple(seeds)


def load_matrix(
    runs_root: Path, seeds: Sequence[int], folds: Sequence[int], modes: Sequence[str] = MODES,
) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    schemas: dict[str, tuple[str, ...]] = {}
    for seed in seeds:
        seed_sample_keys: dict[str, set[tuple[int, str]]] = {}
        seen_samples_by_mode: dict[str, set[str]] = {mode: set() for mode in modes}
        seed_material_schema = None
        for fold in folds:
            fold_parent = None
            fold_samples = None
            for mode in modes:
                run = runs_root / f"seed{seed}" / f"fold{fold}" / mode
                bundle = load_run(run, mode, int(seed), int(fold))
                for artifact, observed_schema in bundle["schemas"].items():
                    if artifact in schemas and schemas[artifact] != observed_schema:
                        raise AnalysisContractError(f"{artifact} schema drift: {run}")
                    schemas.setdefault(artifact, observed_schema)
                if fold_parent is not None and bundle["parent_key"] != fold_parent:
                    raise AnalysisContractError(
                        f"parent checkpoint SHA differs across modes: seed{seed}/fold{fold}"
                    )
                fold_parent = bundle["parent_key"]
                if seed_material_schema is not None and bundle["material_schema_key"] != seed_material_schema:
                    raise AnalysisContractError(f"Material schema SHA differs within seed {seed}")
                seed_material_schema = bundle["material_schema_key"]
                keys = set(bundle["sample"].loc[
                    bundle["sample"]["control"] == "aligned", "sample_id"
                ].astype(str))
                overlap_samples = seen_samples_by_mode[mode] & keys
                if overlap_samples:
                    raise AnalysisContractError(
                        f"sample appears in multiple folds for seed {seed}/{mode}: "
                        f"{sorted(overlap_samples)[:5]}"
                    )
                seen_samples_by_mode[mode].update(keys)
                if fold_samples is not None and keys != fold_samples:
                    raise AnalysisContractError(
                        f"aligned sample inventory differs across modes: seed{seed}/fold{fold}"
                    )
                fold_samples = keys
                event_folds = bundle["sample"].loc[
                    bundle["sample"]["control"] == "aligned", ["event_id"]
                ].assign(fold=int(fold))
                # A physical event is a single LOGO test unit within a seed.
                for prior in bundles:
                    if prior["seed"] != seed or prior["mode"] != mode or prior["fold"] == fold:
                        continue
                    prior_events = set(prior["sample"].loc[
                        prior["sample"]["control"] == "aligned", "event_id"
                    ].astype(str))
                    overlap = prior_events & set(event_folds["event_id"].astype(str))
                    if overlap:
                        raise AnalysisContractError(
                            f"physical event appears in multiple folds for seed {seed}: {sorted(overlap)}"
                        )
                seed_sample_keys[mode] = seed_sample_keys.get(mode, set()) | {
                    (int(fold), key) for key in keys
                }
                bundles.append(bundle)
        if len({frozenset(values) for values in seed_sample_keys.values()}) != 1:
            raise AnalysisContractError(f"mode sample inventory differs for seed {seed}")
    reference_inventory: dict[str, set[tuple[int, str]]] | None = None
    material_schemas = {bundle["material_schema_key"] for bundle in bundles}
    if len(material_schemas) != 1:
        raise AnalysisContractError("Material schema SHA differs across runs")
    feature_receipts = {bundle["material_feature_names"] for bundle in bundles}
    if len(feature_receipts) != 1:
        raise AnalysisContractError("Material feature-name receipt differs across runs")
    for seed in seeds:
        current: dict[str, set[tuple[int, str]]] = {}
        for bundle in bundles:
            if bundle["seed"] != seed:
                continue
            current[bundle["mode"]] = current.get(bundle["mode"], set()) | {
                (bundle["fold"], value) for value in bundle["sample"].loc[
                    bundle["sample"]["control"] == "aligned", "sample_id"
                ].astype(str)
            }
        if reference_inventory is None:
            reference_inventory = current
        elif current != reference_inventory:
            raise AnalysisContractError(f"test sample/fold inventory differs across seeds: {seed}")
    return bundles


def _support_mask(frame: pd.DataFrame, role: str) -> pd.Series:
    if role == "material":
        return frame["material_local_effect_eligible"]
    if role == "trigger":
        return frame["trigger_local_effect_eligible"]
    if role == "any":
        return frame["material_local_effect_eligible"] | frame["trigger_local_effect_eligible"]
    if role == "joint":
        return frame["joint_local_effect_eligible"]
    raise ValueError(role)


def _fixed_recall(frame: pd.DataFrame) -> pd.Series:
    denominator = frame["fixed_fpr_tp"] + frame["fixed_fpr_fn"]
    return np.where(denominator > 0, frame["fixed_fpr_tp"] / denominator, np.nan)


def build_sample_pairs(bundles: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    lookup = {(item["seed"], item["fold"], item["mode"]): item for item in bundles}
    for contrast, (mode, left_name, right_name, role) in CONTRASTS.items():
        for seed, fold, current_mode in sorted(lookup):
            if current_mode != mode:
                continue
            bundle = lookup[(seed, fold, mode)]
            frame = bundle["sample"]
            left = frame.loc[frame["control"] == left_name].copy()
            left = left.loc[_support_mask(left, role)]
            if right_name == "frozen_vt":
                for row in left.to_dict("records"):
                    output.append({
                        "contrast": contrast, "mode": mode, "seed": seed, "fold": fold,
                        "sample_id": str(row["sample_id"]), "physical_event_id": str(row["event_id"]),
                        "source_id": str(row["source_id"]), "left": left_name, "right": right_name,
                        "iou_left": float(row["iou"]), "iou_right": float(row["vt_iou"]),
                        "delta_iou": float(row["iou"] - row["vt_iou"]),
                        "tp_left": int(row["tp"]), "fp_left": int(row["fp"]),
                        "fn_left": int(row["fn"]), "tn_left": int(row["tn"]),
                        "tp_right": np.nan, "fp_right": np.nan,
                        "fn_right": np.nan, "tn_right": np.nan,
                        "errors_left": int(row["errors"]), "errors_right": int(row["vt_errors"]),
                        "corrected": int(row["corrected"]), "harmed": int(row["harmed"]),
                        "net_error_reduction": int(row["corrected"] - row["harmed"]),
                        "rer": (int(row["vt_errors"]) - int(row["errors"])) / max(int(row["vt_errors"]), 1),
                        "brier_improvement": np.nan, "nll_improvement": np.nan,
                        "area_error_improvement": np.nan,
                        "fixed_fpr_recall_improvement": np.nan,
                        "q_M": float(row["q_M"]), "q_R": float(row["q_R"]),
                        "effective_q_M": float(row["effective_q_M"]),
                        "effective_q_R": float(row["effective_q_R"]),
                    })
                continue
            right = frame.loc[
                (frame["control"] == right_name) & frame["control_applicable"]
            ].copy()
            merged = left.merge(
                right, on=["sample_id", "event_id", "source_id"], suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            for row in merged.to_dict("records"):
                left_recall_den = int(row["fixed_fpr_tp_left"]) + int(row["fixed_fpr_fn_left"])
                right_recall_den = int(row["fixed_fpr_tp_right"]) + int(row["fixed_fpr_fn_right"])
                left_recall = int(row["fixed_fpr_tp_left"]) / left_recall_den if left_recall_den else np.nan
                right_recall = int(row["fixed_fpr_tp_right"]) / right_recall_den if right_recall_den else np.nan
                left_area = abs(int(row["predicted_area_left"]) - int(row["true_area_left"]))
                right_area = abs(int(row["predicted_area_right"]) - int(row["true_area_right"]))
                output.append({
                    "contrast": contrast, "mode": mode, "seed": seed, "fold": fold,
                    "sample_id": str(row["sample_id"]), "physical_event_id": str(row["event_id"]),
                    "source_id": str(row["source_id"]), "left": left_name, "right": right_name,
                    "iou_left": float(row["iou_left"]), "iou_right": float(row["iou_right"]),
                    "delta_iou": float(row["iou_left"] - row["iou_right"]),
                    "tp_left": int(row["tp_left"]), "fp_left": int(row["fp_left"]),
                    "fn_left": int(row["fn_left"]), "tn_left": int(row["tn_left"]),
                    "tp_right": int(row["tp_right"]), "fp_right": int(row["fp_right"]),
                    "fn_right": int(row["fn_right"]), "tn_right": int(row["tn_right"]),
                    "errors_left": int(row["errors_left"]), "errors_right": int(row["errors_right"]),
                    "corrected": np.nan, "harmed": np.nan,
                    "net_error_reduction": int(row["errors_right"] - row["errors_left"]),
                    "rer": (int(row["errors_right"]) - int(row["errors_left"])) / max(int(row["errors_right"]), 1),
                    "brier_improvement": float(row["brier_right"] - row["brier_left"]),
                    "nll_improvement": float(row["nll_right"] - row["nll_left"]),
                    "area_error_improvement": float(right_area - left_area),
                    "fixed_fpr_recall_improvement": float(left_recall - right_recall),
                    "q_M": float(row["q_M_left"]), "q_R": float(row["q_R_left"]),
                    "effective_q_M": float(row["effective_q_M_left"]),
                    "effective_q_R": float(row["effective_q_R_left"]),
                })
    if not output:
        raise AnalysisContractError("no applicable sample-level contrast pairs")
    return pd.DataFrame(output)


def build_event_pairs(sample_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["contrast", "mode", "seed", "fold", "physical_event_id", "left", "right"]
    for values, group in sample_pairs.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        has_right_confusion = np.isfinite(group[["tp_right", "fp_right", "fn_right"]]).all().all()
        if has_right_confusion:
            left_tp, left_fp, left_fn = (
                float(group[column].sum()) for column in ("tp_left", "fp_left", "fn_left")
            )
            right_tp, right_fp, right_fn = (
                float(group[column].sum()) for column in ("tp_right", "fp_right", "fn_right")
            )
            event_iou_left = left_tp / max(left_tp + left_fp + left_fn, 1.0)
            event_iou_right = right_tp / max(right_tp + right_fp + right_fn, 1.0)
            event_delta_iou = event_iou_left - event_iou_right
            iou_aggregation = "pooled event confusion counts"
        else:
            event_iou_left = float(group["iou_left"].mean())
            event_iou_right = float(group["iou_right"].mean())
            event_delta_iou = float(group["delta_iou"].mean())
            iou_aggregation = "mean sample IoU; frozen-VT event confusion counts absent"
        corrected = pd.to_numeric(group["corrected"], errors="coerce").dropna()
        harmed = pd.to_numeric(group["harmed"], errors="coerce").dropna()
        row.update({
            "n_samples": len(group),
            "iou_left": event_iou_left, "iou_right": event_iou_right,
            "delta_iou": event_delta_iou, "iou_aggregation": iou_aggregation,
            "errors_left": int(group["errors_left"].sum()),
            "errors_right": int(group["errors_right"].sum()),
            "corrected": int(corrected.sum()) if len(corrected) else np.nan,
            "harmed": int(harmed.sum()) if len(harmed) else np.nan,
            "net_error_reduction": int(group["net_error_reduction"].sum()),
            "rer": (group["errors_right"].sum() - group["errors_left"].sum()) /
                   max(int(group["errors_right"].sum()), 1),
        })
        for metric in (
            "brier_improvement", "nll_improvement", "area_error_improvement",
            "fixed_fpr_recall_improvement",
        ):
            finite = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = float(finite.mean()) if len(finite) else np.nan
        rows.append(row)
    if not rows:
        raise AnalysisContractError("no physical-event pairs")
    return pd.DataFrame(rows)


def build_fold_metrics(bundles: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = {(item["seed"], item["fold"], item["mode"]): item for item in bundles}
    for contrast, (mode, left, right, role) in CONTRASTS.items():
        for seed, fold, current_mode in sorted(lookup):
            if current_mode != mode:
                continue
            bundle = lookup[(seed, fold, mode)]
            sample = bundle["sample"]
            aligned = sample.loc[sample["control"] == "aligned"]
            support = aligned.loc[_support_mask(aligned, role)]
            if support.empty:
                continue
            left_metrics = bundle["result"]["test"]["controls"][left]
            row = {
                "contrast": contrast, "mode": mode, "seed": seed, "fold": fold,
                "n_supported_samples": len(support), "left": left, "right": right,
                "ap_left": float(left_metrics["ap"]), "ap_right": np.nan,
                "delta_ap": np.nan,
            }
            if right != "frozen_vt":
                right_metrics = bundle["result"]["test"]["controls"][right]
                row["ap_right"] = float(right_metrics["ap"])
                row["delta_ap"] = float(left_metrics["ap"] - right_metrics["ap"])
            rows.append(row)
    if not rows:
        raise AnalysisContractError("no fold-level metrics")
    return pd.DataFrame(rows)


def _sign_flip_p(values: np.ndarray, iterations: int, seed: int) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    observed = abs(float(values.mean()))
    if np.all(np.abs(values) <= EPS):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(iterations, len(values)))
    draws = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(draws >= observed - EPS)) / (iterations + 1))


def _bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(iterations, len(values)))
    draws = values[index].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def paired_stats(values: Iterable[float], iterations: int, seed: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0, "mean": np.nan, "median": np.nan, "ci95_low": np.nan,
            "ci95_high": np.nan, "sign_flip_p": np.nan, "wilcoxon_p": np.nan,
            "cohens_dz": np.nan,
        }
    low, high = _bootstrap_ci(array, iterations, seed)
    std = float(array.std(ddof=1)) if len(array) > 1 else np.nan
    dz = float(array.mean() / std) if math.isfinite(std) and std > EPS else np.nan
    if wilcoxon is None or np.all(np.abs(array) <= EPS):
        wilcoxon_p = 1.0 if np.all(np.abs(array) <= EPS) else np.nan
    else:
        try:
            wilcoxon_p = float(
                wilcoxon(array, zero_method="wilcox", alternative="two-sided").pvalue
            )
        except ValueError:
            wilcoxon_p = np.nan
    return {
        "n": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "ci95_low": low, "ci95_high": high,
        "sign_flip_p": _sign_flip_p(array, iterations, seed + 1),
        "wilcoxon_p": wilcoxon_p, "cohens_dz": dz,
    }


def summarize_paired_levels(
    sample_pairs: pd.DataFrame, event_pairs: pd.DataFrame, iterations: int, seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = (
        "delta_iou", "rer", "brier_improvement", "nll_improvement",
        "area_error_improvement", "fixed_fpr_recall_improvement",
    )
    for contrast_index, contrast in enumerate(CONTRASTS):
        for level, frame, unit in (
            ("sample", sample_pairs, "sample_id"),
            ("physical_event", event_pairs, "physical_event_id"),
        ):
            subset = frame.loc[frame["contrast"] == contrast]
            if subset.empty:
                continue
            # Optimization repeats are averaged before sample/event inference.
            averaged = subset.groupby(unit, sort=True)[list(metric_columns)].mean(numeric_only=True)
            for metric_index, metric in enumerate(metric_columns):
                stats = paired_stats(
                    averaged[metric].to_numpy(), iterations,
                    seed + contrast_index * 100 + metric_index * 7 + (0 if level == "sample" else 1),
                )
                rows.append({
                    "contrast": contrast, "level": level, "metric": metric,
                    "seed_handling": "mean repeated optimization seeds before inference",
                    **stats,
                })
    if not rows:
        raise AnalysisContractError("no paired statistics produced")
    return pd.DataFrame(rows)


def hierarchical_event_bootstrap(
    event_pairs: pd.DataFrame, iterations: int, seed: int,
) -> pd.DataFrame:
    metrics = (
        "delta_iou", "rer", "brier_improvement", "nll_improvement",
        "area_error_improvement", "fixed_fpr_recall_improvement",
    )
    output: list[dict[str, Any]] = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        frame = event_pairs.loc[event_pairs["contrast"] == contrast].copy()
        if frame.empty:
            continue
        seeds = np.asarray(sorted(frame["seed"].unique()), dtype=int)
        events = np.asarray(sorted(frame["physical_event_id"].astype(str).unique()), dtype=object)
        lookup = {
            (int(row.seed), str(row.physical_event_id)): row
            for row in frame.itertuples(index=False)
        }
        for metric_index, metric in enumerate(metrics):
            available = pd.to_numeric(frame[metric], errors="coerce")
            if not np.isfinite(available).any():
                output.append({
                    "contrast": contrast, "metric": metric, "n_seeds": len(seeds),
                    "n_physical_events": len(events), "point": np.nan,
                    "ci95_low": np.nan, "ci95_high": np.nan, "bootstrap_reps": iterations,
                    "method": "metric unavailable in source schema",
                })
                continue
            per_seed = frame.groupby("seed", sort=True)[metric].mean()
            point = float(per_seed.mean())
            rng = np.random.default_rng(seed + contrast_index * 1009 + metric_index * 31)
            draws: list[float] = []
            for _ in range(iterations):
                sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                sampled_events = rng.choice(events, size=len(events), replace=True)
                seed_means = []
                for sampled_seed in sampled_seeds:
                    values = []
                    for sampled_event in sampled_events:
                        row = lookup.get((int(sampled_seed), str(sampled_event)))
                        if row is None:
                            continue
                        value = float(getattr(row, metric))
                        if math.isfinite(value):
                            values.append(value)
                    if values:
                        seed_means.append(float(np.mean(values)))
                if seed_means:
                    draws.append(float(np.mean(seed_means)))
            if not draws:
                low = high = np.nan
            else:
                low, high = map(float, np.quantile(draws, (0.025, 0.975)))
            output.append({
                "contrast": contrast, "metric": metric, "n_seeds": len(seeds),
                "n_physical_events": len(events), "point": point,
                "ci95_low": low, "ci95_high": high, "bootstrap_reps": iterations,
                "method": "crossed hierarchical bootstrap: physical events and optimization seeds",
            })
    if not output:
        raise AnalysisContractError("hierarchical bootstrap produced no rows")
    return pd.DataFrame(output)


def summarize_ap(fold_metrics: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    output = []
    for index, contrast in enumerate(CONTRASTS):
        frame = fold_metrics.loc[fold_metrics["contrast"] == contrast]
        finite = frame.loc[np.isfinite(frame["delta_ap"])]
        if finite.empty:
            output.append({
                "contrast": contrast, "metric": "delta_ap", "n_seeds": 0,
                "mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                "note": "frozen VT probabilities are absent; AP contrast unavailable",
            })
            continue
        # Folds are pooled descriptively within seed; seeds are the repeat unit.
        seed_values = finite.groupby("seed", sort=True)["delta_ap"].mean().to_numpy()
        stats = paired_stats(seed_values, iterations, seed + index * 17)
        output.append({
            "contrast": contrast, "metric": "delta_ap", "n_seeds": len(seed_values),
            "mean": stats["mean"], "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
            "note": "fold AP averaged within each seed; folds are not optimization repeats",
        })
    return pd.DataFrame(output)


def metric_availability() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "metric": "IoU", "aligned_vs_vt": True, "aligned_vs_control": True,
            "sample_level": True, "physical_event_level": True,
            "note": "control events use pooled confusion; VT events use mean sample IoU because VT counts are absent",
        },
        {
            "metric": "AP", "aligned_vs_vt": False, "aligned_vs_control": True,
            "sample_level": False, "physical_event_level": False,
            "note": "AP is a population ranking metric and is available at fold level only",
        },
        {
            "metric": "Brier", "aligned_vs_vt": False, "aligned_vs_control": True,
            "sample_level": True, "physical_event_level": True,
            "note": "sample means; frozen VT probabilities were not exported",
        },
        {
            "metric": "NLL", "aligned_vs_vt": False, "aligned_vs_control": True,
            "sample_level": True, "physical_event_level": True,
            "note": "sample means; frozen VT probabilities were not exported",
        },
        {
            "metric": "thresholded_area_error", "aligned_vs_vt": False,
            "aligned_vs_control": True, "sample_level": True,
            "physical_event_level": True,
            "note": "predicted_area is thresholded, not a soft probability area",
        },
        {
            "metric": "soft_area", "aligned_vs_vt": False, "aligned_vs_control": False,
            "sample_level": False, "physical_event_level": False,
            "note": "probability sums are absent; analyzer refuses a hard-area relabel",
        },
        {
            "metric": "recall_at_validation_fixed_FPR", "aligned_vs_vt": False,
            "aligned_vs_control": True, "sample_level": True,
            "physical_event_level": True,
            "note": "threshold selected at validation FPR=0.05; test TP/FN exported",
        },
        {
            "metric": "corrected_harmed", "aligned_vs_vt": True,
            "aligned_vs_control": False, "sample_level": True,
            "physical_event_level": True,
            "note": "exact pixel overlap exists only against frozen VT; controls retain exact net error difference",
        },
    ])


def build_pooled_full_test_metrics(bundles: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Pool mutually exclusive LOGO test folds without treating folds as repeats."""

    rows: list[dict[str, Any]] = []
    for (seed, mode), group in _bundle_groups(bundles).items():
        aligned = pd.concat([
            item["sample"].loc[item["sample"]["control"] == "aligned"] for item in group
        ], ignore_index=True)
        if aligned["sample_id"].astype(str).duplicated().any():
            raise AnalysisContractError(f"sample appears in multiple LOGO folds: seed{seed}/{mode}")
        aligned_counts = {key: int(aligned[key].sum()) for key in ("tp", "fp", "fn", "tn")}
        vt_counts = {
            key: sum(int(item["result"]["test"]["vt"][key]) for item in group)
            for key in ("tp", "fp", "fn", "tn")
        }
        aligned_iou = aligned_counts["tp"] / max(
            aligned_counts["tp"] + aligned_counts["fp"] + aligned_counts["fn"], 1
        )
        vt_iou = vt_counts["tp"] / max(vt_counts["tp"] + vt_counts["fp"] + vt_counts["fn"], 1)
        aligned_errors = aligned_counts["fp"] + aligned_counts["fn"]
        vt_errors = vt_counts["fp"] + vt_counts["fn"]
        corrected, harmed = int(aligned["corrected"].sum()), int(aligned["harmed"].sum())
        if corrected - harmed != vt_errors - aligned_errors:
            raise AnalysisContractError(f"pooled corrected-harmed identity failed: seed{seed}/{mode}")
        rows.append({
            "seed": seed, "mode": mode, "n_folds": len(group),
            "n_samples": aligned["sample_id"].nunique(),
            "n_physical_events": aligned["event_id"].nunique(),
            "aligned_iou": aligned_iou, "frozen_vt_iou": vt_iou,
            "delta_iou_aligned_minus_vt": aligned_iou - vt_iou,
            "aligned_errors": aligned_errors, "frozen_vt_errors": vt_errors,
            "corrected": corrected, "harmed": harmed,
            "net_error_reduction": corrected - harmed,
            "rer_vs_frozen_vt": (vt_errors - aligned_errors) / max(vt_errors, 1),
            "mutually_exclusive_logo_test": True,
        })
    return pd.DataFrame(rows)


def _bundle_groups(
    bundles: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for bundle in bundles:
        groups.setdefault((int(bundle["seed"]), str(bundle["mode"])), []).append(bundle)
    for group in groups.values():
        group.sort(key=lambda item: int(item["fold"]))
    return groups


def build_conditional_effect_metrics(bundles: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Summarize only trainer-predeclared, label-free local eligibility."""

    rows: list[dict[str, Any]] = []
    for (seed, mode), group in _bundle_groups(bundles).items():
        frame = pd.concat([
            item["sample"].loc[item["sample"]["control"] == "aligned"] for item in group
        ], ignore_index=True)
        eligibility_column = f"{_local_role(mode)}_local_effect_eligible"
        eligible = frame.loc[frame[eligibility_column]].copy()
        q_m_positive = int((frame["effective_q_M"] > 0).sum())
        q_r_positive = int((frame["effective_q_R"] > 0).sum())
        terrain_positive = int((frame["terrain_support_pixel_count"] > 0).sum())
        if eligible.empty:
            corrected = harmed = net = 0
            rer = delta_iou = None
        else:
            corrected = int(eligible["corrected"].sum())
            harmed = int(eligible["harmed"].sum())
            net = corrected - harmed
            rer = net / max(int(eligible["vt_errors"].sum()), 1)
            delta_iou = float((eligible["iou"] - eligible["vt_iou"]).mean())
        rows.append({
            "seed": seed, "mode": mode, "eligibility_field": eligibility_column,
            "eligibility_contract": "trainer-emitted q>0 and nonzero Terrain support; label-free",
            "test_label_or_error_used_for_selection": False,
            "n_total_samples": len(frame), "n_eligible_samples": len(eligible),
            "eligible_coverage_fraction": len(eligible) / max(len(frame), 1),
            "q_M_positive_samples": q_m_positive,
            "q_R_positive_samples": q_r_positive,
            "terrain_support_positive_samples": terrain_positive,
            "mean_sample_delta_iou_on_eligible": delta_iou,
            "eligible_corrected": corrected, "eligible_harmed": harmed,
            "eligible_net_error_reduction": net, "eligible_rer": rer,
        })
    return pd.DataFrame(rows)


def build_inventory(bundles: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        aligned = bundle["sample"].loc[bundle["sample"]["control"] == "aligned"]
        rows.append({
            "seed": bundle["seed"], "fold": bundle["fold"], "mode": bundle["mode"],
            "n_samples": len(aligned), "n_physical_events": aligned["event_id"].nunique(),
            "q_M_positive_samples": int((aligned["effective_q_M"] > 0).sum()),
            "q_R_positive_samples": int((aligned["effective_q_R"] > 0).sum()),
            "terrain_support_positive_samples": int((aligned["terrain_support_pixel_count"] > 0).sum()),
            "selected_identity_abstain": bundle["selected_identity_abstain"],
            "trigger_q0_fold_abstention": (
                bundle["mode"] in ("trigger", "joint")
                and int((aligned["effective_q_R"] > 0).sum()) == 0
            ),
            "visual_parent_sha256": bundle["parent_key"][3],
            "terrain_parent_sha256": bundle["parent_key"][4],
            "material_schema_sha256": bundle["material_schema_key"][1],
            "run_dir": str(bundle["run"].resolve()),
        })
    return pd.DataFrame(rows)


def build_report(
    summary: Mapping[str, Any], bootstrap: pd.DataFrame,
    pooled: pd.DataFrame, conditional: pd.DataFrame,
) -> str:
    lines = [
        "# Sen12 hierarchical role-aware Material/Trigger analysis", "",
        f"- Seeds: `{summary['seeds']}`.",
        f"- LOGO folds per seed: `{summary['folds']}`; folds are geographic coverage, not repeat seeds.",
        f"- Complete runs: `{summary['n_runs']}`.",
        f"- Unique test samples per seed: `{summary['n_unique_samples_per_seed']}`.",
        f"- Unique physical events per seed: `{summary['n_unique_physical_events_per_seed']}`.",
        "- All controls are inference-time evaluations of the same selected checkpoint.",
        "- Epoch-0 identity is an explicit candidate; an identity selection is an abstention.",
        "- Trigger q_R=0 folds remain fallback audits and are excluded from Trigger efficacy.",
        "- Conditional rows use trainer-emitted q>0 plus Terrain-support eligibility only.",
        "",
        "## Evidence contract", "",
        "Positive values denote improvement by aligned support. Material is compared with same-source "
        "cross-event shuffle and zero-q. Trigger is compared with wrong-time, event-shuffle, and zero-q. "
        "The joint mode uses the same role-specific controls while holding the other role aligned.",
        "",
        "AP is available only for aligned-versus-control at fold level. Frozen VT probabilities and "
        "probability sums were not exported, so aligned-versus-VT AP/Brier/NLL and soft area are not "
        "recoverable. Thresholded area is reported under its true name.",
        "", "## Complete five-fold test", "",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(
            f"- seed `{row.seed}` `{row.mode}`: IoU `{row.frozen_vt_iou:.6f}` -> "
            f"`{row.aligned_iou:.6f}` (delta `{row.delta_iou_aligned_minus_vt:+.6f}`), "
            f"RER `{row.rer_vs_frozen_vt:+.4%}`, corrected/harmed "
            f"`{row.corrected}/{row.harmed}`."
        )
    lines.extend(["", "## Label-free conditional support", ""])
    for row in conditional.itertuples(index=False):
        value = row.eligible_rer
        rer = "NA" if value is None or not math.isfinite(float(value)) else f"{value:+.4%}"
        lines.append(
            f"- seed `{row.seed}` `{row.mode}`: eligible `{row.n_eligible_samples}/"
            f"{row.n_total_samples}` (`{row.eligible_coverage_fraction:.2%}`), "
            f"conditional RER `{rer}`."
        )
    lines.extend(["", "## Hierarchical event bootstrap", ""])
    for row in bootstrap.loc[bootstrap["metric"] == "delta_iou"].itertuples(index=False):
        lines.append(
            f"- `{row.contrast}`: mean delta IoU `{row.point:+.6f}`, "
            f"95% CI `[{row.ci95_low:+.6f}, {row.ci95_high:+.6f}]`, "
            f"events `{row.n_physical_events}`, seeds `{row.n_seeds}`."
        )
    lines.extend([
        "", "The interval resamples physical events and optimization seeds as separate levels; "
        "the five LOGO folds are never counted as five independent optimization repeats.", "",
    ])
    return "\n".join(lines)


def analyze(
    runs_root: Path, outdir: Path, seeds: Sequence[int] | None = None,
    folds: Sequence[int] = DEFAULT_FOLDS, min_seeds: int = 1,
    bootstrap_reps: int = 20000, bootstrap_seed: int = 20260722,
) -> dict[str, Any]:
    if bootstrap_reps < 100:
        raise AnalysisContractError("bootstrap_reps must be >= 100")
    discovered = discover_seeds(runs_root)
    selected = tuple(sorted(discovered if seeds is None else map(int, seeds)))
    if len(selected) != len(set(selected)):
        raise AnalysisContractError("duplicate seed identities")
    if any(seed not in discovered for seed in selected):
        raise AnalysisContractError(f"requested seeds absent from runs root: {selected}")
    if len(selected) < min_seeds:
        raise AnalysisContractError(f"need at least {min_seeds} optimization seeds")
    folds = tuple(map(int, folds))
    if not folds or len(folds) != len(set(folds)):
        raise AnalysisContractError("fold list must be nonempty and unique")
    if tuple(sorted(folds)) != DEFAULT_FOLDS:
        raise AnalysisContractError("strict v2 analysis requires exactly LOGO folds 0,1,2,3,4")
    bundles = load_matrix(runs_root, selected, folds)
    inventory = build_inventory(bundles)
    pooled = build_pooled_full_test_metrics(bundles)
    conditional = build_conditional_effect_metrics(bundles)
    sample_pairs = build_sample_pairs(bundles)
    event_pairs = build_event_pairs(sample_pairs)
    fold_metrics = build_fold_metrics(bundles)
    paired_summary = summarize_paired_levels(
        sample_pairs, event_pairs, bootstrap_reps, bootstrap_seed
    )
    hierarchical = hierarchical_event_bootstrap(
        event_pairs, bootstrap_reps, bootstrap_seed + 101
    )
    ap_summary = summarize_ap(fold_metrics, bootstrap_reps, bootstrap_seed + 202)
    availability = metric_availability()
    first_seed = selected[0]
    aligned = pd.concat([
        item["sample"].loc[item["sample"]["control"] == "aligned"]
        for item in bundles if item["seed"] == first_seed and item["mode"] == "joint"
    ], ignore_index=True)
    summary = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "runs_root": str(runs_root.resolve()),
        "seeds": list(selected), "folds": list(folds), "modes": list(MODES),
        "n_runs": len(bundles),
        "n_optimization_seeds": len(selected),
        "n_logo_folds_per_seed": len(folds),
        "fold_independence_contract": (
            "LOGO folds provide geographic coverage within seed; they are not optimization repeats"
        ),
        "n_unique_samples_per_seed": int(aligned["sample_id"].nunique()),
        "n_unique_physical_events_per_seed": int(aligned["event_id"].nunique()),
        "n_sample_pairs": len(sample_pairs), "n_event_pairs": len(event_pairs),
        "same_checkpoint_controls_verified": True,
        "parent_checkpoint_sha_verified": True,
        "material_schema_sha_verified": True,
        "epoch0_identity_selection_verified": True,
        "q0_exact_fallback_verified": True,
        "trigger_terrain_support_and_sign_verified": True,
        "conditional_subset_label_free_verified": True,
        "n_identity_abstain_runs": int(inventory["selected_identity_abstain"].sum()),
        "n_trigger_q0_fold_abstentions": int(inventory["trigger_q0_fold_abstention"].sum()),
        "trigger_filter_contract": (
            "effective_q_R>0 and control_applicable; q_R=0 folds are fallback audits only"
        ),
        "metric_limitations": {
            "ap": "fold-level only; frozen-VT AP absent",
            "soft_area": "unavailable; source exports thresholded predicted_area only",
            "frozen_vt_probability_metrics": "unavailable",
        },
        "outputs": [
            "run_inventory.csv", "paired_sample_metrics.csv", "paired_event_metrics.csv",
            "pooled_full_test_metrics.csv", "conditional_effect_metrics.csv",
            "paired_fold_metrics.csv", "paired_statistics.csv",
            "hierarchical_event_bootstrap.csv", "ap_summary.csv",
            "metric_availability.csv", "summary.json", "report.md",
        ],
    }
    for name, frame in (
        ("run_inventory.csv", inventory),
        ("pooled_full_test_metrics.csv", pooled),
        ("conditional_effect_metrics.csv", conditional),
        ("paired_sample_metrics.csv", sample_pairs),
        ("paired_event_metrics.csv", event_pairs),
        ("paired_fold_metrics.csv", fold_metrics),
        ("paired_statistics.csv", paired_summary),
        ("hierarchical_event_bootstrap.csv", hierarchical),
        ("ap_summary.csv", ap_summary),
        ("metric_availability.csv", availability),
    ):
        atomic_csv(outdir / name, frame)
    atomic_json(outdir / "summary.json", summary)
    atomic_text(outdir / "report.md", build_report(summary, hierarchical, pooled, conditional))
    return summary


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_ints)
    parser.add_argument("--folds", type=_parse_ints, default=DEFAULT_FOLDS)
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(
        args.runs_root, args.outdir, args.seeds, args.folds, args.min_seeds,
        args.bootstrap_reps, args.bootstrap_seed,
    )
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
