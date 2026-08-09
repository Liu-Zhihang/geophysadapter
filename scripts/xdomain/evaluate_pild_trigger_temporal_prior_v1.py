#!/usr/bin/env python3
"""Evaluate a bounded event-level Trigger posterior on real parent OOF logits.

Trigger is never allowed to draw a spatial boundary in this protocol.  For each
held-out event fold, a monotone mapping from antecedent-rainfall evidence to a
non-negative, bounded *global* logit offset is fitted using only the other OOF
folds.  The held-out fold is evaluated with its parent model's training-only
threshold.  Aligned evidence is compared with identity, a matched-capacity
constant offset, same-location shifted-time evidence, and event-shuffled
evidence.

The script deliberately has no fallback to aggregate per-sample metrics.  It
requires pixel logits, labels, valid masks, and strict producer receipts.  If
the future PILD/Sen12 parent OOF artifacts do not exist, it writes a coverage
audit and a BLOCKED receipt, then exits non-zero without fabricating results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_sen12_roleaware_lodo_v1/parent_oof_manifest.json"
)
DEFAULT_PILD_TRIGGER = (
    PROJECT_ROOT
    / "processed/hybrid_pinn/pild_prithvi_integration_v1/"
    "pild_trigger_sample_registry_v1.csv"
)
DEFAULT_SEN12_TRIGGER = (
    PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v1/trigger_sample_registry_v1.csv"
)
DEFAULT_EXTERNAL_EVIDENCE = (
    PROJECT_ROOT
    / "metadata/reports/global_trigger_case_crossover_external_v1_20260717/summary.json"
)
DEFAULT_SEN12_PRIOR_EVALUATION = (
    PROJECT_ROOT / "experiments/revision2026/sen12_trigger_awc_dose_v1/summary.json"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "experiments/revision2026/pild_trigger_temporal_prior_v1"
)

SCHEMA_VERSION = "pild_trigger_temporal_prior_evaluation.v1"
MANIFEST_SCHEMA = "pild_trigger_parent_oof_manifest.v1"
PREDICTION_RECEIPT_SCHEMA = "pild_trigger_parent_oof_prediction.v1"
CONDITIONS = ("identity", "constant", "aligned", "wrong_time", "event_shuffled")
CONTROL_CONDITIONS = ("identity", "constant", "wrong_time", "event_shuffled")
WRONG_COLUMNS = (
    "rain_d7_wrong_m56_mm",
    "rain_d7_wrong_m28_mm",
    "rain_d7_wrong_p28_mm",
    "rain_d7_wrong_p56_mm",
)


class ContractError(RuntimeError):
    """Raised when an input could invalidate the OOF interpretation."""


@dataclass(frozen=True)
class TriggerEvidence:
    physical_event_id: str
    dataset: str
    q_r: float
    aligned: float
    wrong_time: float
    n_samples: int


@dataclass(frozen=True)
class OOFFold:
    fold_id: str
    sample_ids: np.ndarray
    split_event_ids: np.ndarray
    physical_event_ids: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    threshold_probability: float
    checkpoint_sha256: str
    prediction_path: Path
    prediction_sha256: str
    receipt_path: Path
    receipt_sha256: str


@dataclass(frozen=True)
class TemporalPriorModel:
    center: float
    scale: float
    intercept: float
    coefficient: float
    constant_offset: float
    n_fit_events: int
    status: str

    def predict(self, feature: np.ndarray, maximum: float) -> np.ndarray:
        feature = np.asarray(feature, np.float64)
        standardized = np.clip((feature - self.center) / self.scale, -5.0, 5.0)
        value = self.intercept + self.coefficient * standardized
        return np.clip(value, 0.0, maximum)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pild-trigger-registry", type=Path, default=DEFAULT_PILD_TRIGGER)
    parser.add_argument("--sen12-trigger-registry", type=Path, default=DEFAULT_SEN12_TRIGGER)
    parser.add_argument("--external-evidence", type=Path, default=DEFAULT_EXTERNAL_EVIDENCE)
    parser.add_argument(
        "--sen12-prior-evaluation", type=Path, default=DEFAULT_SEN12_PRIOR_EVALUATION
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-logit-offset", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--min-fit-supported-events", type=int, default=3)
    parser.add_argument("--min-oof-folds", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write Trigger coverage/provenance audit without requiring OOF predictions.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce")


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.median(numeric)) if len(numeric) else float("nan")


def _event_feature(case: float, controls: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in controls if math.isfinite(value)], np.float64)
    if not math.isfinite(case) or len(finite) != len(controls) or not len(finite):
        return float("nan"), float("nan")
    transformed_case = math.log1p(max(case, 0.0))
    transformed_controls = np.log1p(np.maximum(finite, 0.0))
    aligned = transformed_case - float(np.median(transformed_controls))
    # Each shifted date is treated once as a pseudo-case against the other
    # shifted dates.  Their median is the frozen same-location negative control.
    shifted_scores = []
    for index, value in enumerate(transformed_controls):
        others = np.delete(transformed_controls, index)
        shifted_scores.append(float(value - np.median(others)))
    return aligned, float(np.median(shifted_scores))


def _load_one_trigger_registry(path: Path, dataset: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{dataset} Trigger registry is missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"sample_id", "physical_event_id", "q_R", *WRONG_COLUMNS}
    case_column = (
        "rain_d7_case_mm" if "rain_d7_case_mm" in frame
        else "rain_d7_antecedent_case_mm"
    )
    required.add(case_column)
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"{dataset} Trigger registry lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sample_id"] = frame.sample_id.astype(str)
    if frame.sample_id.duplicated().any():
        examples = frame.loc[frame.sample_id.duplicated(False), "sample_id"].head(5).tolist()
        raise ContractError(f"{dataset} Trigger registry has duplicate samples: {examples}")
    frame["physical_event_id"] = frame.physical_event_id.astype(str)
    frame["q_R"] = _numeric(frame, "q_R").fillna(0.0).clip(0.0, 1.0)
    frame["rain_case_mm"] = _numeric(frame, case_column)
    for column in WRONG_COLUMNS:
        frame[column] = _numeric(frame, column)
    frame["trigger_dataset"] = dataset
    return frame[
        ["sample_id", "physical_event_id", "trigger_dataset", "q_R", "rain_case_mm", *WRONG_COLUMNS]
    ]


def load_trigger_registries(pild_path: Path, sen12_path: Path) -> pd.DataFrame:
    pild = _load_one_trigger_registry(pild_path, "PILD")
    sen12 = _load_one_trigger_registry(sen12_path, "Sen12")
    collision = sorted(set(pild.sample_id) & set(sen12.sample_id))
    if collision:
        raise ContractError(f"PILD/Sen12 Trigger sample IDs collide: {collision[:5]}")
    return pd.concat([pild, sen12], ignore_index=True).set_index("sample_id", drop=False)


def build_event_evidence(registry: pd.DataFrame) -> dict[str, TriggerEvidence]:
    output: dict[str, TriggerEvidence] = {}
    for event_id, group in registry.groupby("physical_event_id", sort=True):
        datasets = sorted(set(group.trigger_dataset.astype(str)))
        if len(datasets) != 1:
            raise ContractError(f"physical event {event_id} crosses Trigger datasets: {datasets}")
        q_r = float(np.clip(_numeric(group, "q_R").fillna(0.0).min(), 0.0, 1.0))
        case = _finite_median(group["rain_case_mm"])
        controls = [_finite_median(group[column]) for column in WRONG_COLUMNS]
        aligned, wrong = _event_feature(case, controls)
        if q_r <= 0 or not math.isfinite(aligned) or not math.isfinite(wrong):
            q_r, aligned, wrong = 0.0, 0.0, 0.0
        output[str(event_id)] = TriggerEvidence(
            physical_event_id=str(event_id), dataset=datasets[0], q_r=q_r,
            aligned=aligned, wrong_time=wrong, n_samples=int(len(group)),
        )
    return output


def coverage_audit(
    registry: pd.DataFrame,
    external_evidence_path: Path,
    sen12_prior_evaluation_path: Path,
    pild_registry_path: Path,
    sen12_registry_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evidence = build_event_evidence(registry)
    event_rows = [vars(item) for item in evidence.values()]
    event_frame = pd.DataFrame(event_rows)
    rows: list[dict[str, Any]] = []
    for dataset, group in registry.groupby("trigger_dataset", sort=True):
        events = event_frame[event_frame.dataset == dataset]
        rows.append({
            "dataset": dataset,
            "n_samples": int(len(group)),
            "n_events": int(len(events)),
            "n_supported_samples": int((group.q_R > 0).sum()),
            "n_supported_events": int((events.q_r > 0).sum()),
            "supported_sample_fraction": float((group.q_R > 0).mean()),
            "supported_event_fraction": float((events.q_r > 0).mean()),
        })
    external: dict[str, Any] = {"path": str(external_evidence_path), "exists": False}
    if external_evidence_path.is_file():
        payload = json.loads(external_evidence_path.read_text(encoding="utf-8"))
        cohort = payload.get("cohorts", {}).get("independent_glad4cd_unknown_trigger", {})
        d7 = (
            payload.get("inference", {})
            .get("independent_glad4cd_unknown_trigger", {})
            .get("median_3x3", {})
            .get("7", {})
        )
        external = {
            "path": str(external_evidence_path.resolve()),
            "exists": True,
            "sha256": sha256_file(external_evidence_path),
            "status": payload.get("status"),
            "promotion_decision": payload.get("promotion_gate", {}).get("decision"),
            "n_events": cohort.get("eligible_physical_events"),
            "d7_median_difference_mm": d7.get("median_difference_mm"),
            "d7_bootstrap_ci95_mm": d7.get("bootstrap_median_ci95_mm"),
            "d7_positive_events": d7.get("positive_events"),
            "d7_wilcoxon_p_greater": d7.get("wilcoxon_p_greater"),
            "allowed_interpretation": payload.get("allowed_interpretation"),
        }
    prior_evaluation: dict[str, Any] = {
        "path": str(sen12_prior_evaluation_path), "exists": False
    }
    if sen12_prior_evaluation_path.is_file():
        payload = json.loads(sen12_prior_evaluation_path.read_text(encoding="utf-8"))
        prior_evaluation = {
            "path": str(sen12_prior_evaluation_path.resolve()),
            "exists": True,
            "sha256": sha256_file(sen12_prior_evaluation_path),
            "status": payload.get("status"),
            "scope": payload.get("scope"),
            "outer_test_labels_loaded": payload.get("outer_test_labels_loaded"),
            "results": payload.get("results", []),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "registry_samples": int(len(registry)),
        "registry_events": int(len(event_frame)),
        "supported_samples": int((registry.q_R > 0).sum()),
        "supported_events": int((event_frame.q_r > 0).sum()),
        "datasets": rows,
        "external_138_event_evidence": external,
        "prior_sen12_trigger_awc_evaluation": prior_evaluation,
        "audited_inputs": {
            "pild_trigger_registry": {
                "path": str(pild_registry_path.resolve()),
                "sha256": sha256_file(pild_registry_path),
            },
            "sen12_trigger_registry": {
                "path": str(sen12_registry_path.resolve()),
                "sha256": sha256_file(sen12_registry_path),
            },
            "existing_sen12_evaluator": {
                "path": str((SCRIPT_DIR / "evaluate_sen12_trigger_awc_dose_v1.py").resolve()),
                "sha256": sha256_file(SCRIPT_DIR / "evaluate_sen12_trigger_awc_dose_v1.py"),
            },
        },
        "role_contract": (
            "Trigger is event-level temporal support only; it cannot localize pixel boundaries."
        ),
    }
    return event_frame, summary


def write_coverage_outputs(
    outdir: Path,
    event_frame: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_csv(outdir / "coverage_event_audit.csv", event_frame)
    atomic_json(outdir / "coverage_audit.json", summary)
    lines = [
        "# Trigger temporal-prior coverage audit", "",
        "Trigger is audited as event-level temporal support, not a pixel boundary expert.", "",
        "| Dataset | Samples | Events | Supported samples | Supported events |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["datasets"]:
        lines.append(
            f"| {row['dataset']} | {row['n_samples']} | {row['n_events']} | "
            f"{row['n_supported_samples']} | {row['n_supported_events']} |"
        )
    external = summary["external_138_event_evidence"]
    lines.extend([
        "", "## External event-time evidence", "",
        f"- Receipt available: `{external.get('exists', False)}`",
        f"- Eligible physical events: `{external.get('n_events', 'NA')}`",
        f"- Frozen decision: `{external.get('promotion_decision', 'NA')}`",
        f"- D7 median paired difference: `{external.get('d7_median_difference_mm', 'NA')}` mm",
        f"- D7 bootstrap CI95: `{external.get('d7_bootstrap_ci95_mm', 'NA')}`",
        f"- D7 positive events: `{external.get('d7_positive_events', 'NA')}`",
        "- This evidence establishes event-time association only and is not a segmentation result.",
        "", "## Prior Sen12 dose evaluation", "",
        f"- Receipt available: `{summary['prior_sen12_trigger_awc_evaluation'].get('exists', False)}`",
        f"- Frozen results: `{summary['prior_sen12_trigger_awc_evaluation'].get('results', [])}`",
        "- Neither prior dose family passed its matched-control development gate.",
    ])
    (outdir / "coverage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_required_oof_contract(outdir: Path) -> None:
    atomic_json(outdir / "required_parent_oof_contract.json", {
        "manifest_schema_version": MANIFEST_SCHEMA,
        "manifest_required_fields": {
            "selection_uses_labels": False,
            "all_available_parent_oof_folds_included": True,
            "entries": [
                "fold_id", "prediction_path", "prediction_sha256",
                "producer_receipt_path", "producer_receipt_sha256",
            ],
        },
        "prediction_npz_required_arrays": {
            "sample_ids": "[N] unique strings",
            "event_ids": "[N] held-out split event IDs",
            "logits": "[N,1,H,W] raw parent logits",
            "labels": "[N,1,H,W] labels used only for OOF evaluation",
            "valid": "[N,1,H,W] valid-pixel mask",
        },
        "producer_receipt_schema_version": PREDICTION_RECEIPT_SCHEMA,
        "producer_receipt_required_fields": {
            "prediction_role": "parent_oof",
            "prediction_value_type": "raw_logits",
            "selection_uses_holdout_labels": False,
            "threshold_uses_holdout_labels": False,
            "threshold_probability": "strictly between 0 and 1; fitted on training data",
            "checkpoint_sha256": "64-character SHA-256",
            "training_event_ids": "non-empty and disjoint from held_out_event_ids",
            "held_out_event_ids": "exactly equal to prediction event_ids",
        },
    })


def _as_strings(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array).astype(str)
    if result.ndim != 1 or not len(result) or any(not item for item in result):
        raise ContractError(f"{name} must be a non-empty one-dimensional string array")
    return result


def _as_maps(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 3:
        result = result[:, None]
    if result.ndim != 4 or result.shape[1] != 1:
        raise ContractError(f"{name} must have shape [N,1,H,W] or [N,H,W]")
    return result


def load_oof_folds(
    manifest_path: Path,
    registry: pd.DataFrame,
    *,
    min_folds: int,
) -> tuple[list[OOFFold], dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"real parent OOF manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ContractError(f"OOF manifest schema must be {MANIFEST_SCHEMA}")
    if manifest.get("selection_uses_labels") is not False:
        raise ContractError("OOF manifest must declare selection_uses_labels=false")
    if manifest.get("all_available_parent_oof_folds_included") is not True:
        raise ContractError(
            "OOF manifest must declare all_available_parent_oof_folds_included=true"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) < min_folds:
        raise ContractError(f"OOF manifest must contain at least {min_folds} fold entries")
    base = manifest_path.parent
    folds: list[OOFFold] = []
    seen_fold_ids: set[str] = set()
    seen_samples: set[str] = set()
    seen_split_events: set[str] = set()
    seen_physical_events: set[str] = set()
    access_log: list[dict[str, Any]] = []
    for entry in entries:
        fold_id = str(entry.get("fold_id", ""))
        if not fold_id or fold_id in seen_fold_ids:
            raise ContractError(f"invalid or duplicate fold_id: {fold_id!r}")
        seen_fold_ids.add(fold_id)
        prediction_path = _resolve(base, entry["prediction_path"])
        receipt_path = _resolve(base, entry["producer_receipt_path"])
        for path, expected_key in (
            (prediction_path, "prediction_sha256"),
            (receipt_path, "producer_receipt_sha256"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"OOF artifact is missing: {path}")
            observed = sha256_file(path)
            if observed != str(entry.get(expected_key, "")):
                raise ContractError(f"OOF artifact hash mismatch: {path}")
            access_log.append({"fold_id": fold_id, "path": str(path), "sha256": observed})
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != PREDICTION_RECEIPT_SCHEMA:
            raise ContractError(f"prediction receipt schema mismatch: {receipt_path}")
        required_truth = {
            "prediction_role": "parent_oof",
            "prediction_value_type": "raw_logits",
            "selection_uses_holdout_labels": False,
            "threshold_uses_holdout_labels": False,
        }
        for key, expected in required_truth.items():
            if receipt.get(key) != expected:
                raise ContractError(f"unsafe producer receipt field {key}: {receipt_path}")
        if str(receipt.get("fold_id")) != fold_id:
            raise ContractError(f"fold identity mismatch in receipt: {receipt_path}")
        threshold = float(receipt.get("threshold_probability", float("nan")))
        if not 0.0 < threshold < 1.0:
            raise ContractError(f"invalid training-only threshold: {receipt_path}")
        checkpoint_hash = str(receipt.get("checkpoint_sha256", ""))
        if len(checkpoint_hash) != 64:
            raise ContractError(f"receipt lacks a checkpoint SHA-256: {receipt_path}")
        training_events = set(map(str, receipt.get("training_event_ids", [])))
        held_events = set(map(str, receipt.get("held_out_event_ids", [])))
        if not training_events or not held_events or training_events & held_events:
            raise ContractError(f"training/held-out event contract is invalid: {receipt_path}")
        with np.load(prediction_path, allow_pickle=False) as payload:
            required_arrays = {"sample_ids", "event_ids", "logits", "labels", "valid"}
            missing = required_arrays - set(payload.files)
            if missing:
                raise ContractError(f"OOF prediction lacks arrays {sorted(missing)}: {prediction_path}")
            sample_ids = _as_strings(payload["sample_ids"], "sample_ids")
            split_events = _as_strings(payload["event_ids"], "event_ids")
            logits = _as_maps(payload["logits"], "logits").astype(np.float32)
            labels = _as_maps(payload["labels"], "labels").astype(np.float32)
            valid = _as_maps(payload["valid"], "valid").astype(bool)
        n = len(sample_ids)
        if any(len(array) != n for array in (split_events, logits, labels, valid)):
            raise ContractError(f"OOF array sample dimensions disagree: {prediction_path}")
        if logits.shape != labels.shape or logits.shape != valid.shape:
            raise ContractError(f"OOF map shapes disagree: {prediction_path}")
        if not np.isfinite(logits).all() or not np.isfinite(labels).all():
            raise ContractError(f"OOF logits/labels contain non-finite values: {prediction_path}")
        if not np.all(np.isclose(labels, 0.0) | np.isclose(labels, 1.0)):
            raise ContractError(f"OOF labels must be binary: {prediction_path}")
        if np.any(valid.reshape(n, -1).sum(axis=1) <= 0):
            raise ContractError(f"every OOF sample must contain valid pixels: {prediction_path}")
        if len(set(sample_ids)) != n or seen_samples & set(sample_ids):
            raise ContractError("OOF sample identities duplicate within or across folds")
        seen_samples.update(sample_ids)
        observed_events = set(split_events)
        if observed_events != held_events:
            raise ContractError(f"prediction events differ from receipt held-out events: {fold_id}")
        if seen_split_events & observed_events:
            raise ContractError("held-out event identities overlap across OOF folds")
        seen_split_events.update(observed_events)
        missing_registry = sorted(set(sample_ids) - set(registry.index))
        if missing_registry:
            raise ContractError(f"Trigger registry lacks OOF samples: {missing_registry[:5]}")
        physical_events = registry.loc[sample_ids, "physical_event_id"].astype(str).to_numpy()
        observed_physical_events = set(physical_events)
        if seen_physical_events & observed_physical_events:
            raise ContractError("physical event identities overlap across OOF folds")
        seen_physical_events.update(observed_physical_events)
        folds.append(OOFFold(
            fold_id=fold_id, sample_ids=sample_ids, split_event_ids=split_events,
            physical_event_ids=physical_events, logits=logits, labels=labels,
            valid=valid, threshold_probability=threshold,
            checkpoint_sha256=checkpoint_hash, prediction_path=prediction_path,
            prediction_sha256=sha256_file(prediction_path), receipt_path=receipt_path,
            receipt_sha256=sha256_file(receipt_path),
        ))
    return folds, {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "n_folds": len(folds),
        "n_samples": sum(len(item.sample_ids) for item in folds),
        "n_split_events": len(seen_split_events),
        "n_physical_events": len(seen_physical_events),
        "access_log": access_log,
    }


def optimize_event_offsets(
    folds: Iterable[OOFFold],
    maximum: float,
    grid_size: int,
) -> dict[str, float]:
    grid = np.linspace(0.0, maximum, grid_size, dtype=np.float64)
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for fold in folds:
        for event in sorted(set(fold.physical_event_ids)):
            selected = fold.physical_event_ids == event
            grouped.setdefault(str(event), []).append(
                (fold.logits[selected], fold.labels[selected], fold.valid[selected])
            )
    output: dict[str, float] = {}
    for event, chunks in grouped.items():
        loss = np.zeros(len(grid), np.float64)
        n_valid = 0
        for logits, labels, valid in chunks:
            x = logits[valid].astype(np.float64)
            y = labels[valid].astype(np.float64)
            if not len(x):
                continue
            expanded = x[:, None] + grid[None, :]
            # Stable BCE(logits, target).
            loss += np.sum(np.maximum(expanded, 0) - expanded * y[:, None] + np.log1p(np.exp(-np.abs(expanded))), axis=0)
            n_valid += len(x)
        if n_valid <= 0:
            raise ContractError(f"physical event {event} has no valid OOF pixels")
        output[event] = float(grid[int(np.argmin(loss))])
    return output


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    center = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(values))
    return center, scale if math.isfinite(scale) and scale > 1e-8 else 1.0


def fit_temporal_prior(
    event_targets: Mapping[str, float],
    evidence: Mapping[str, TriggerEvidence],
    *,
    maximum: float,
    ridge_alpha: float,
    min_events: int,
) -> TemporalPriorModel:
    eligible = sorted(
        event for event in set(event_targets) & set(evidence)
        if evidence[event].q_r > 0 and math.isfinite(evidence[event].aligned)
    )
    if not eligible:
        return TemporalPriorModel(0.0, 1.0, 0.0, 0.0, 0.0, 0, "no_supported_fit_events")
    feature = np.asarray([evidence[event].aligned for event in eligible], np.float64)
    target = np.asarray([event_targets[event] for event in eligible], np.float64)
    constant = float(np.clip(np.mean(target), 0.0, maximum))
    center, scale = _robust_center_scale(feature)
    if len(eligible) < min_events or np.std(feature) <= 1e-8:
        return TemporalPriorModel(center, scale, constant, 0.0, constant, len(eligible), "constant_only")
    z = np.clip((feature - center) / scale, -5.0, 5.0)
    design = np.column_stack([np.ones(len(z)), z])
    response = target
    if ridge_alpha > 0:
        design = np.vstack([design, [0.0, math.sqrt(ridge_alpha)]])
        response = np.append(response, 0.0)
    fit = lsq_linear(
        design, response,
        bounds=(np.asarray([0.0, 0.0]), np.asarray([maximum, maximum * 5.0])),
        method="trf",
    )
    if not fit.success:
        raise ContractError(f"monotone Trigger calibration failed: {fit.message}")
    return TemporalPriorModel(
        center=center, scale=scale, intercept=float(fit.x[0]),
        coefficient=float(fit.x[1]), constant_offset=constant,
        n_fit_events=len(eligible), status="monotone_ridge",
    )


def deterministic_donor(
    event_id: str,
    candidates: Sequence[str],
    seed: int,
) -> str:
    valid = sorted(set(map(str, candidates)) - {str(event_id)})
    if not valid:
        raise ContractError(f"no event-shuffle Trigger donor exists for {event_id}")
    digest = hashlib.sha256(f"{seed}|trigger-event-shuffle|{event_id}".encode()).digest()
    return valid[int.from_bytes(digest[:8], "big") % len(valid)]


def offsets_for_fold(
    fold: OOFFold,
    registry: pd.DataFrame,
    evidence: Mapping[str, TriggerEvidence],
    model: TemporalPriorModel,
    condition: str,
    *,
    maximum: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, str]]:
    offset = np.zeros(len(fold.sample_ids), np.float64)
    donors: dict[str, str] = {}
    all_supported = [event for event, item in evidence.items() if item.q_r > 0]
    for event_id in sorted(set(fold.physical_event_ids)):
        if event_id not in evidence:
            raise ContractError(f"Trigger evidence is absent for physical event {event_id}")
        item = evidence[event_id]
        if item.q_r <= 0 or condition == "identity":
            value = 0.0
        elif condition == "constant":
            value = model.constant_offset
        elif condition == "aligned":
            value = float(model.predict(np.asarray([item.aligned]), maximum)[0])
        elif condition == "wrong_time":
            value = float(model.predict(np.asarray([item.wrong_time]), maximum)[0])
        elif condition == "event_shuffled":
            donor = deterministic_donor(event_id, all_supported, seed)
            donors[event_id] = donor
            value = float(model.predict(np.asarray([evidence[donor].aligned]), maximum)[0])
        else:
            raise ValueError(f"unknown condition: {condition}")
        value = float(np.clip(value * item.q_r, 0.0, maximum))
        offset[fold.physical_event_ids == event_id] = value
    # Exact identity is tested from the actual sample registry, not only the
    # event aggregate, so partial or malformed quality broadcasts fail closed.
    sample_q = registry.loc[fold.sample_ids, "q_R"].to_numpy(dtype=np.float64)
    offset[sample_q <= 0] = 0.0
    return offset, donors


def _counts(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    prediction = np.asarray(prediction, bool)
    target = np.asarray(target, bool)
    valid = np.asarray(valid, bool)
    return {
        "tp": int(np.sum(prediction & target & valid)),
        "fp": int(np.sum(prediction & ~target & valid)),
        "fn": int(np.sum(~prediction & target & valid)),
        "tn": int(np.sum(~prediction & ~target & valid)),
    }


def _metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (int(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    return {
        **dict(counts), "errors": fp + fn,
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def _ap(logits: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> float:
    y = labels[valid].astype(np.uint8)
    if not len(y) or not np.any(y):
        return 0.0
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[valid], -30.0, 30.0)))
    return float(average_precision_score(y, probability))


def evaluate_condition(
    fold: OOFFold,
    condition: str,
    offsets: np.ndarray,
    registry: pd.DataFrame,
    donors: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    logits = fold.logits + offsets[:, None, None, None].astype(np.float32)
    threshold_logit = math.log(fold.threshold_probability / (1.0 - fold.threshold_probability))
    prediction = logits >= threshold_logit
    identity_prediction = fold.logits >= threshold_logit
    truth = fold.labels >= 0.5
    sample_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(fold.sample_ids):
        valid = fold.valid[index]
        counts = _counts(prediction[index], truth[index], valid)
        parent_counts = _counts(identity_prediction[index], truth[index], valid)
        parent_metrics = _metrics(parent_counts)
        parent_wrong = np.logical_xor(identity_prediction[index], truth[index]) & valid
        final_wrong = np.logical_xor(prediction[index], truth[index]) & valid
        corrected = int(np.sum(parent_wrong & ~final_wrong))
        harmed = int(np.sum(~parent_wrong & final_wrong & valid))
        event = str(fold.physical_event_ids[index])
        sample_rows.append({
            "fold_id": fold.fold_id, "condition": condition,
            "sample_id": str(sample_id), "split_event_id": str(fold.split_event_ids[index]),
            "physical_event_id": event,
            "trigger_dataset": str(registry.loc[sample_id, "trigger_dataset"]),
            "q_R": float(registry.loc[sample_id, "q_R"]),
            "offset": float(offsets[index]), "event_shuffle_donor": donors.get(event, ""),
            **_metrics(counts), "ap": _ap(logits[index], fold.labels[index], valid),
            "n_valid_pixels": int(valid.sum()),
            "parent_errors": parent_metrics["errors"], "parent_iou": parent_metrics["iou"],
            "corrected": corrected, "harmed": harmed,
            "net_error_reduction": corrected - harmed,
            "rer": (corrected - harmed) / max(parent_metrics["errors"], 1),
            "checkpoint_sha256": fold.checkpoint_sha256,
            "prediction_sha256": fold.prediction_sha256,
        })
    samples = pd.DataFrame(sample_rows)
    event_rows: list[dict[str, Any]] = []
    for event_id, group in samples.groupby("physical_event_id", sort=True):
        counts = {key: int(group[key].sum()) for key in ("tp", "fp", "fn", "tn")}
        parent_errors = int(group.parent_errors.sum())
        corrected = int(group.corrected.sum())
        harmed = int(group.harmed.sum())
        selected = fold.physical_event_ids == event_id
        event_rows.append({
            "fold_id": fold.fold_id, "condition": condition,
            "physical_event_id": event_id,
            "trigger_dataset": group.trigger_dataset.iloc[0],
            "q_R": float(group.q_R.min()), "offset": float(group.offset.iloc[0]),
            "event_shuffle_donor": group.event_shuffle_donor.iloc[0],
            **_metrics(counts), "ap": _ap(logits[selected], fold.labels[selected], fold.valid[selected]),
            "n_samples": int(len(group)), "n_valid_pixels": int(group.n_valid_pixels.sum()),
            "parent_errors": parent_errors, "corrected": corrected, "harmed": harmed,
            "net_error_reduction": corrected - harmed,
            "rer": (corrected - harmed) / max(parent_errors, 1),
        })
    total_counts = {key: int(samples[key].sum()) for key in ("tp", "fp", "fn", "tn")}
    parent_errors = int(samples.parent_errors.sum())
    corrected, harmed = int(samples.corrected.sum()), int(samples.harmed.sum())
    fold_row = {
        "fold_id": fold.fold_id, "condition": condition, **_metrics(total_counts),
        "ap": _ap(logits, fold.labels, fold.valid),
        "n_samples": int(len(samples)), "n_events": int(samples.physical_event_id.nunique()),
        "n_valid_pixels": int(samples.n_valid_pixels.sum()), "parent_errors": parent_errors,
        "corrected": corrected, "harmed": harmed,
        "net_error_reduction": corrected - harmed,
        "rer": (corrected - harmed) / max(parent_errors, 1),
        "mean_offset": float(np.mean(offsets)), "max_offset": float(np.max(offsets)),
    }
    return sample_rows, event_rows, fold_row


def paired_statistics(
    frame: pd.DataFrame,
    unit_column: str,
    condition: str,
    *,
    seed: int,
    bootstrap_replicates: int,
    permutation_replicates: int,
) -> dict[str, Any]:
    metric = frame.pivot(index=unit_column, columns="condition", values=["iou", "rer"])
    if condition not in metric["iou"] or "identity" not in metric["iou"]:
        raise ContractError(f"paired statistics lack {condition}/identity rows")
    delta_iou = (metric["iou"][condition] - metric["iou"]["identity"]).dropna().to_numpy()
    rer = metric["rer"][condition].dropna().to_numpy()
    if not len(delta_iou):
        raise ContractError(f"no paired {unit_column} units for {condition}")
    rng = np.random.default_rng(seed)
    boot_index = rng.integers(0, len(delta_iou), size=(bootstrap_replicates, len(delta_iou)))
    boot_iou = delta_iou[boot_index].mean(axis=1)
    boot_rer = rer[rng.integers(0, len(rer), size=(bootstrap_replicates, len(rer)))].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutation_replicates, len(delta_iou)))
    permutation = np.abs((signs * delta_iou).mean(axis=1))
    observed = abs(float(np.mean(delta_iou)))
    try:
        wilcoxon_p = float(wilcoxon(delta_iou, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "n": int(len(delta_iou)),
        "mean_delta_iou": float(np.mean(delta_iou)),
        "median_delta_iou": float(np.median(delta_iou)),
        "delta_iou_bootstrap_ci95": [float(np.quantile(boot_iou, 0.025)), float(np.quantile(boot_iou, 0.975))],
        "mean_rer": float(np.mean(rer)),
        "median_rer": float(np.median(rer)),
        "rer_bootstrap_ci95": [float(np.quantile(boot_rer, 0.025)), float(np.quantile(boot_rer, 0.975))],
        "sign_permutation_p_two_sided": float((np.sum(permutation >= observed) + 1) / (permutation_replicates + 1)),
        "wilcoxon_p_two_sided": wilcoxon_p,
    }


def run_evaluation(args: argparse.Namespace, registry: pd.DataFrame) -> dict[str, Any]:
    folds, provenance = load_oof_folds(args.oof_manifest, registry, min_folds=args.min_oof_folds)
    evidence = build_event_evidence(registry)
    sample_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for fold_index, target in enumerate(folds):
        fitting = [fold for fold in folds if fold.fold_id != target.fold_id]
        fit_split_events = {event for fold in fitting for event in fold.split_event_ids}
        if fit_split_events & set(target.split_event_ids):
            raise ContractError(f"target/fitting event leakage for fold {target.fold_id}")
        targets = optimize_event_offsets(fitting, args.max_logit_offset, args.grid_size)
        model = fit_temporal_prior(
            targets, evidence, maximum=args.max_logit_offset,
            ridge_alpha=args.ridge_alpha, min_events=args.min_fit_supported_events,
        )
        model_rows.append({
            "target_fold_id": target.fold_id, **vars(model),
            "fit_fold_ids": "|".join(sorted(item.fold_id for item in fitting)),
            "fit_physical_event_ids": "|".join(sorted(targets)),
            "target_split_event_ids": "|".join(sorted(set(target.split_event_ids))),
        })
        for condition in CONDITIONS:
            offsets, donors = offsets_for_fold(
                target, registry, evidence, model, condition,
                maximum=args.max_logit_offset,
                seed=args.seed + fold_index * 1009,
            )
            rows_s, rows_e, row_f = evaluate_condition(
                target, condition, offsets, registry, donors
            )
            sample_rows.extend(rows_s)
            event_rows.extend(rows_e)
            fold_rows.append(row_f)
    samples = pd.DataFrame(sample_rows)
    events = pd.DataFrame(event_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    models = pd.DataFrame(model_rows)
    expected_rows = sum(len(fold.sample_ids) for fold in folds) * len(CONDITIONS)
    if len(samples) != expected_rows:
        raise ContractError("condition/sample coverage is incomplete")
    identity = samples[samples.condition == "identity"].set_index("sample_id")
    unsupported = samples[samples.q_R <= 0]
    if not np.array_equal(unsupported.offset.to_numpy(), np.zeros(len(unsupported))):
        raise ContractError("q_R=0 did not produce exact identity offset")
    for condition in CONDITIONS:
        current = samples[samples.condition == condition].set_index("sample_id")
        if set(current.index) != set(identity.index):
            raise ContractError(f"sample identities differ for {condition}")
    def statistics_for(frame: pd.DataFrame, unit: str, seed_offset: int) -> dict[str, Any]:
        return {
            condition: paired_statistics(
                frame, unit, condition, seed=args.seed + seed_offset + 17 * index,
                bootstrap_replicates=args.bootstrap_replicates,
                permutation_replicates=args.permutation_replicates,
            )
            for index, condition in enumerate(CONDITIONS[1:], 1)
        }

    sample_stats = statistics_for(samples, "sample_id", 0)
    event_stats = statistics_for(events, "physical_event_id", 101)
    supported_samples = samples[samples.q_R > 0].copy()
    supported_events = events[events.q_R > 0].copy()
    if supported_samples.empty or supported_events.empty:
        raise ContractError("OOF cohort contains no source-valid Trigger-supported units")
    supported_sample_stats = statistics_for(supported_samples, "sample_id", 1001)
    supported_event_stats = statistics_for(supported_events, "physical_event_id", 2001)
    aligned = supported_event_stats["aligned"]
    aligned_mean = aligned["mean_rer"]
    control_means = {
        condition: supported_event_stats[condition]["mean_rer"]
        for condition in CONTROL_CONDITIONS[1:]
    }
    full_aligned = fold_metrics[fold_metrics.condition == "aligned"]
    full_net_error_reduction = int(full_aligned.net_error_reduction.sum())
    gate = {
        "full_cohort_net_error_reduction_gt_zero": full_net_error_reduction > 0,
        "supported_aligned_mean_event_rer_gt_zero": aligned_mean > 0,
        "supported_aligned_event_rer_ci_lower_gt_zero": aligned["rer_bootstrap_ci95"][0] > 0,
        "supported_aligned_beats_constant_mean_event_rer": aligned_mean > control_means["constant"],
        "supported_aligned_beats_wrong_time_mean_event_rer": aligned_mean > control_means["wrong_time"],
        "supported_aligned_beats_event_shuffled_mean_event_rer": aligned_mean > control_means["event_shuffled"],
        "full_cohort_net_error_reduction": full_net_error_reduction,
    }
    gate["all_pass"] = all(
        value for key, value in gate.items() if key != "full_cohort_net_error_reduction"
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.outdir / "sample_metrics.csv", samples)
    atomic_csv(args.outdir / "event_metrics.csv", events)
    atomic_csv(args.outdir / "fold_metrics.csv", fold_metrics)
    atomic_csv(args.outdir / "calibration_receipts.csv", models)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "cross-fitted evaluation on real parent OOF pixel logits and labels",
        "intervention": "bounded non-negative event-level posterior logit offset",
        "conditions": list(CONDITIONS),
        "sample_statistics_all": sample_stats,
        "event_statistics_all": event_stats,
        "sample_statistics_trigger_supported": supported_sample_stats,
        "event_statistics_trigger_supported": supported_event_stats,
        "promotion_gate": gate,
        "provenance": provenance,
        "guardrails": {
            "trigger_draws_pixel_boundaries": False,
            "target_fold_labels_used_for_calibration": False,
            "target_fold_labels_used_for_threshold": False,
            "label_driven_subset_selection": False,
            "support_subset_definition": "q_R > 0 from source-valid registry before OOF metrics",
            "q_R_zero_exact_identity": True,
            "maximum_logit_offset": args.max_logit_offset,
        },
        "artifact_hashes": {
            "sample_metrics.csv": sha256_file(args.outdir / "sample_metrics.csv"),
            "event_metrics.csv": sha256_file(args.outdir / "event_metrics.csv"),
            "fold_metrics.csv": sha256_file(args.outdir / "fold_metrics.csv"),
            "calibration_receipts.csv": sha256_file(args.outdir / "calibration_receipts.csv"),
        },
    }
    atomic_json(args.outdir / "summary.json", summary)
    lines = [
        "# PILD/Sen12 Trigger temporal-prior evaluation v1", "",
        "Trigger is evaluated as a bounded event-level posterior offset on real parent OOF logits. It is not a pixel boundary expert.", "",
        "| Unit | Condition | N | Mean DeltaIoU | 95% CI | Mean RER | 95% CI |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for scope, unit, stats in (
        ("all", "sample", sample_stats), ("all", "event", event_stats),
        ("q_R>0", "sample", supported_sample_stats),
        ("q_R>0", "event", supported_event_stats),
    ):
        for condition, item in stats.items():
            lines.append(
                f"| {scope} {unit} | {condition} | {item['n']} | {item['mean_delta_iou']:.6f} | "
                f"[{item['delta_iou_bootstrap_ci95'][0]:.6f}, {item['delta_iou_bootstrap_ci95'][1]:.6f}] | "
                f"{item['mean_rer']:.4%} | [{item['rer_bootstrap_ci95'][0]:.4%}, {item['rer_bootstrap_ci95'][1]:.4%}] |"
            )
    lines.extend([
        "", "## Promotion gate", "",
        f"Decision: `{'PASS' if gate['all_pass'] else 'NO-GO'}`", "",
        "Aligned Trigger must reduce errors on the full cohort; within the label-independent q_R>0 support cohort, event-level RER must have a positive bootstrap lower bound and beat constant, wrong-time, and event-shuffled matched-capacity controls.", "",
        "## Limits", "",
        "- Only events with source-valid Trigger support can receive a non-zero offset.",
        "- The intervention is spatially constant within an event and cannot establish boundary localization.",
        "- A NO-GO result retains Trigger as independent event-time evidence rather than a segmentation component.",
    ])
    (args.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(args.outdir / "DONE.json", {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "summary_sha256": sha256_file(args.outdir / "summary.json"),
        "report_sha256": sha256_file(args.outdir / "report.md"),
        "promotion_gate_pass": gate["all_pass"],
    })
    return summary


def validate_args(args: argparse.Namespace) -> None:
    if args.max_logit_offset <= 0:
        raise ValueError("--max-logit-offset must be positive")
    if args.ridge_alpha < 0 or args.grid_size < 3:
        raise ValueError("invalid ridge/grid setting")
    if args.min_fit_supported_events < 2 or args.min_oof_folds < 2:
        raise ValueError("at least two fit events/folds are required")
    if args.bootstrap_replicates < 100 or args.permutation_replicates < 100:
        raise ValueError("statistical replicate counts must be at least 100")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    registry = load_trigger_registries(args.pild_trigger_registry, args.sen12_trigger_registry)
    event_frame, audit = coverage_audit(
        registry, args.external_evidence, args.sen12_prior_evaluation,
        args.pild_trigger_registry, args.sen12_trigger_registry,
    )
    write_coverage_outputs(args.outdir, event_frame, audit)
    write_required_oof_contract(args.outdir)
    if args.audit_only:
        print(json.dumps(json_safe(audit), indent=2, sort_keys=True, allow_nan=False))
        return 0
    try:
        summary = run_evaluation(args, registry)
    except (FileNotFoundError, ContractError) as error:
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_missing_or_invalid_real_parent_oof",
            "reason": str(error),
            "oof_manifest": str(args.oof_manifest),
            "coverage_audit_sha256": sha256_file(args.outdir / "coverage_audit.json"),
            "fabricated_results_created": False,
        }
        atomic_json(args.outdir / "BLOCKED.json", blocked)
        print(f"FATAL: {error}", file=os.sys.stderr)
        return 2
    blocked = args.outdir / "BLOCKED.json"
    if blocked.exists():
        blocked.unlink()
    print(json.dumps(json_safe(summary["promotion_gate"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
