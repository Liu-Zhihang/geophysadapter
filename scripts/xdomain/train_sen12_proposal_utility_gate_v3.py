#!/usr/bin/env python3
"""Strict nested-OOF binary utility gates for frozen Sen12 Terrain proposals.

Formal evidence consumes only proposer predictions produced by nested
event/region holdouts inside the target outer fold.  A manifest, each cache, and
its producer receipt must jointly prove sample, region, event, and checkpoint
identity.  The target outer-test cache is loaded only after every gate,
regularization value, and rescue/veto threshold has been frozen.

Legacy cross-outer-cache training is retained only as an explicitly exploratory
mode.  It can never emit ``manuscript_pass`` because a non-target outer proposer
may have seen the target geography during its own training.

Terrain is the only dense direction.  A gate may either retain DeltaT or replace
it by exact zero; Material and Trigger are context features, never dense heads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT
    / "experiments/revision2026/sen12_prithvi_roleaware_hierarchical_v2/cache/seed20260751"
)
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT
    / "experiments/revision2026/sen12_prithvi_roleaware_hierarchical_v2/seed20260751"
)
DEFAULT_SPLIT_CSV = PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv"
DEFAULT_OUTROOT = PROJECT_ROOT / "experiments/revision2026/sen12_proposal_utility_gate_v3"

FOLDS = tuple(range(5))
CONTEXTS = ("proposal_only", "TM", "TR", "TMR")
PROPOSAL_TYPES = ("rescue", "veto")
BASE_FEATURE_NAMES = (
    "visual_margin",
    "proposed_margin",
    "terrain_delta",
    "abs_terrain_delta",
    "visual_uncertainty",
)
FORBIDDEN_INFERENCE_FEATURES = {
    "source_id", "region", "region_id", "event_id", "physical_event_id",
    "lon", "lat", "longitude", "latitude", "mask", "label", "target",
    "error", "errors", "corrected", "harmed", "iou", "utility_target",
}
CONTROL_NAMES = {
    "proposal_only": ("aligned",),
    "TM": ("aligned", "material_shuffle", "material_zero_q"),
    "TR": (
        "aligned", "trigger_wrong_time", "trigger_event_shuffle", "trigger_zero_q"
    ),
    "TMR": (
        "aligned", "material_shuffle", "material_zero_q", "trigger_wrong_time",
        "trigger_event_shuffle", "trigger_zero_q", "all_zero_q",
    ),
}
SCHEMA_VERSION = "sen12_proposal_utility_gate_run.v3"
FORMAL_MANIFEST_SCHEMA = "sen12_nested_proposer_oof_manifest.v1"
FORMAL_CACHE_SCHEMA = "sen12_nested_proposer_oof_cache.v1"
FORMAL_RECEIPT_SCHEMA = "sen12_nested_proposer_training_receipt.v1"
PROTOCOL_MODES = ("formal_nested_oof", "cross_outer_exploratory")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fields})
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sample_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


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
        **dict(counts),
        "errors": errors,
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


@dataclass
class FoldBundle:
    fold: int
    seed: int
    sample_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    threshold: float
    visual_logits: np.ndarray
    terrain_delta: np.ndarray
    valid: np.ndarray
    mask: np.ndarray
    material: np.ndarray
    q_material: np.ndarray
    material_shuffle: np.ndarray
    q_material_shuffle: np.ndarray
    trigger: np.ndarray
    trigger_wrong: np.ndarray
    q_trigger: np.ndarray
    trigger_shuffle: np.ndarray
    q_trigger_shuffle: np.ndarray
    identity: dict[str, Any]
    cache_path: str
    cache_sha256: str
    result_path: str
    result_sha256: str

    @property
    def threshold_logit(self) -> float:
        return float(math.log(self.threshold / (1.0 - self.threshold)))


@dataclass
class ProposalTable:
    fold: int
    sample_index: np.ndarray
    proposal_type: np.ndarray
    base_features: np.ndarray
    utility_target: np.ndarray
    sample_weights: np.ndarray
    visual_prediction: np.ndarray
    terrain_prediction: np.ndarray
    pixel_target: np.ndarray
    always_terrain_counts: dict[str, int]
    visual_counts: dict[str, int]
    n_valid: int

    def mask_for_type(self, proposal_type: str) -> np.ndarray:
        code = 1 if proposal_type == "rescue" else 0
        return self.proposal_type == code


@dataclass
class LinearHead:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    constant_probability: float | None
    alpha: float
    proposal_type: str

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full(len(features), self.constant_probability, np.float32)
        standardized = (features - self.mean) / self.scale
        score = standardized @ self.coefficient + self.intercept
        score = np.clip(score, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)

    def receipt(self) -> dict[str, Any]:
        value = {
            "feature_names": self.feature_names,
            "mean": self.mean,
            "scale": self.scale,
            "coefficient": self.coefficient,
            "intercept": self.intercept,
            "constant_probability": self.constant_probability,
            "alpha": self.alpha,
            "proposal_type": self.proposal_type,
        }
        return {**json_safe(value), "sha256": sha256_json(value)}


@dataclass
class UtilityGate:
    context: str
    rescue: LinearHead
    veto: LinearHead
    alpha: float

    def receipt(self) -> dict[str, Any]:
        value = {
            "context": self.context,
            "alpha": self.alpha,
            "rescue": self.rescue.receipt(),
            "veto": self.veto.receipt(),
        }
        return {**value, "checkpoint_sha256": sha256_json(value)}


@dataclass
class Selection:
    context: str
    alpha: float
    rescue_threshold: float
    veto_threshold: float
    meta_metrics: dict[str, Any]
    controls: dict[str, Any]
    claim_pass: bool
    fallback: str
    label_shuffle_claim_pass: bool = False


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _split_rows(split_csv: Path, fold: int) -> dict[str, dict[str, str]]:
    with split_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "outer_fold", "role", "region_group"}
        if not required <= set(reader.fieldnames or ()):
            raise RuntimeError(f"split CSV missing {sorted(required - set(reader.fieldnames or ()))}")
        rows = {
            str(row["sample_id"]): dict(row)
            for row in reader
            if int(row["outer_fold"]) == fold
        }
    if not rows:
        raise RuntimeError(f"split CSV has no rows for outer fold {fold}")
    return rows


def _split_ids(split_csv: Path, fold: int, role: str) -> tuple[str, ...]:
    rows = _split_rows(split_csv, fold)
    return tuple(sample_id for sample_id, row in rows.items() if row["role"] == role)


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _required_string_set(value: Any, name: str, *, allow_empty: bool = False) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RuntimeError(f"{name} must be a list of non-empty strings")
    output = set(value)
    if len(output) != len(value):
        raise RuntimeError(f"{name} contains duplicates")
    if not allow_empty and not output:
        raise RuntimeError(f"{name} must not be empty")
    return output


def _resolve_manifest_path(manifest_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _payload_arrays(payload: Mapping[str, Any], n: int, name: str) -> dict[str, np.ndarray]:
    required = {
        "visual_logits", "frozen_vt_correction", "valid", "mask", "material",
        "q_material", "material_shuffle", "q_material_shuffle", "trigger",
        "trigger_wrong", "q_trigger", "trigger_shuffle", "q_trigger_shuffle",
        "event_ids", "source_ids",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"{name} missing fields: {missing}")
    arrays = {
        "visual_logits": _as_numpy(payload["visual_logits"], np.float32),
        "terrain_delta": _as_numpy(payload["frozen_vt_correction"], np.float32),
        "valid": _as_numpy(payload["valid"], bool),
        "mask": _as_numpy(payload["mask"], bool),
        "material": _as_numpy(payload["material"], np.float32),
        "q_material": _as_numpy(payload["q_material"], np.float32),
        "material_shuffle": _as_numpy(payload["material_shuffle"], np.float32),
        "q_material_shuffle": _as_numpy(payload["q_material_shuffle"], np.float32),
        "trigger": _as_numpy(payload["trigger"], np.float32),
        "trigger_wrong": _as_numpy(payload["trigger_wrong"], np.float32),
        "q_trigger": _as_numpy(payload["q_trigger"], np.float32),
        "trigger_shuffle": _as_numpy(payload["trigger_shuffle"], np.float32),
        "q_trigger_shuffle": _as_numpy(payload["q_trigger_shuffle"], np.float32),
    }
    if any(value.shape[0] != n for value in arrays.values()):
        raise RuntimeError(f"{name} tensor/sample length mismatch")
    if len(payload["event_ids"]) != n or len(payload["source_ids"]) != n:
        raise RuntimeError(f"{name} identity/sample length mismatch")
    return arrays


def _bundle_from_payload(
    *,
    bundle_id: int,
    seed: int,
    sample_ids: tuple[str, ...],
    threshold: float,
    payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    cache_path: Path,
    result_path: Path,
) -> FoldBundle:
    if not 0.0 < threshold < 1.0:
        raise RuntimeError(f"bundle {bundle_id} invalid frozen visual threshold")
    arrays = _payload_arrays(payload, len(sample_ids), f"bundle {bundle_id}")
    return FoldBundle(
        fold=bundle_id,
        seed=seed,
        sample_ids=sample_ids,
        event_ids=tuple(map(str, payload["event_ids"])),
        source_ids=tuple(map(str, payload["source_ids"])),
        threshold=threshold,
        identity=dict(identity),
        cache_path=str(cache_path.resolve()),
        cache_sha256=sha256_file(cache_path),
        result_path=str(result_path.resolve()),
        result_sha256=sha256_file(result_path),
        **arrays,
    )


def load_fold_bundle(
    fold: int,
    *,
    cache_root: Path,
    runs_root: Path,
    split_csv: Path,
    seed: int,
    access_log: list[dict[str, Any]],
    purpose: str,
) -> FoldBundle:
    """Load one outer-test cache for evaluation or exploratory analysis only."""

    cache_path = cache_root / f"fold{fold}" / "frozen_test_cache.pt"
    result_path = runs_root / f"fold{fold}" / "joint" / "result.json"
    if not cache_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"missing fold{fold} cache/result")
    access_log.append({"stage": purpose, "fold": fold, "labels_loaded": True})
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    identity = dict(payload.get("identity", {}))
    if identity.get("split") != "test" or int(identity.get("schema", -1)) != 2:
        raise RuntimeError(f"fold{fold} cache is not an OOF outer-test cache")
    if int(identity.get("seed", -1)) != seed:
        raise RuntimeError(f"fold{fold} cache seed mismatch")
    if (int(result.get("fold", -1)), int(result.get("seed", -1))) != (fold, seed):
        raise RuntimeError(f"fold{fold} result identity mismatch")
    sample_ids = tuple(map(str, payload["sample_ids"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"fold{fold} cache has duplicate sample IDs")
    if identity.get("sample_sha256") != sample_hash(sample_ids):
        raise RuntimeError(f"fold{fold} cache sample hash mismatch")
    expected_ids = _split_ids(split_csv, fold, "test")
    if set(sample_ids) != set(expected_ids) or len(sample_ids) != len(expected_ids):
        raise RuntimeError(f"fold{fold} cache is not bound to that fold's test identities")
    split_rows = _split_rows(split_csv, fold)
    expected_sources = tuple(split_rows[sample_id]["region_group"] for sample_id in sample_ids)
    if tuple(map(str, payload.get("source_ids", ()))) != expected_sources:
        raise RuntimeError(f"fold{fold} cache region IDs do not match split registry")
    parent = result.get("parent_identity", {})
    for role in ("visual", "terrain"):
        expected = parent.get(role, {}).get("sha256")
        observed = identity.get(f"{role}_checkpoint", {}).get("sha256")
        if not expected or expected != observed:
            raise RuntimeError(f"fold{fold} {role} proposer checkpoint mismatch")
    return _bundle_from_payload(
        bundle_id=fold,
        seed=seed,
        sample_ids=sample_ids,
        threshold=float(result["visual_threshold"]),
        payload=payload,
        identity=identity,
        cache_path=cache_path,
        result_path=result_path,
    )


def load_formal_nested_bundles(
    manifest_path: Path,
    *,
    target_fold: int,
    split_csv: Path,
    seed: int,
    access_log: list[dict[str, Any]],
) -> tuple[list[FoldBundle], dict[str, Any]]:
    """Load and prove nested inner-holdout proposer predictions for one outer fold."""

    manifest_path = manifest_path.resolve()
    manifest = _required_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), "nested OOF manifest"
    )
    if manifest.get("schema_version") != FORMAL_MANIFEST_SCHEMA:
        raise RuntimeError("manifest is not a formal nested proposer OOF manifest")
    if int(manifest.get("target_outer_fold", -1)) != target_fold:
        raise RuntimeError("manifest target outer fold mismatch")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("manifest seed mismatch")
    if manifest.get("split_csv_sha256") != sha256_file(split_csv):
        raise RuntimeError("manifest split CSV hash mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) < 3:
        raise RuntimeError("formal nested OOF requires at least three inner holdouts")

    split_rows = _split_rows(split_csv, target_fold)
    outer_train = {sample_id for sample_id, row in split_rows.items() if row["role"] == "train"}
    outer_test = {sample_id for sample_id, row in split_rows.items() if row["role"] == "test"}
    outer_val = {sample_id for sample_id, row in split_rows.items() if row["role"] == "val"}
    target_test_regions = {split_rows[sample_id]["region_group"] for sample_id in outer_test}
    outer_train_regions = {split_rows[sample_id]["region_group"] for sample_id in outer_train}
    if not outer_train or not outer_test or not target_test_regions:
        raise RuntimeError("target outer fold lacks train/test identities or test regions")
    if outer_train_regions & target_test_regions:
        raise RuntimeError("outer split leaks target-test geography into outer-train")

    bundles: list[FoldBundle] = []
    holdout_union: set[str] = set()
    inner_ids: set[int] = set()
    entry_receipts: list[dict[str, Any]] = []
    for raw_entry in entries:
        entry = _required_mapping(raw_entry, "manifest entry")
        inner_fold = int(entry.get("inner_fold", -1))
        if inner_fold < 0 or inner_fold in inner_ids:
            raise RuntimeError("inner_fold values must be unique non-negative integers")
        inner_ids.add(inner_fold)
        cache_path = _resolve_manifest_path(manifest_path, entry.get("cache_path"), "cache_path")
        receipt_path = _resolve_manifest_path(
            manifest_path, entry.get("producer_receipt_path"), "producer_receipt_path"
        )
        if sha256_file(cache_path) != entry.get("cache_sha256"):
            raise RuntimeError(f"inner{inner_fold} cache hash mismatch")
        if sha256_file(receipt_path) != entry.get("producer_receipt_sha256"):
            raise RuntimeError(f"inner{inner_fold} producer receipt hash mismatch")

        receipt = _required_mapping(
            json.loads(receipt_path.read_text(encoding="utf-8")), "producer receipt"
        )
        if receipt.get("schema_version") != FORMAL_RECEIPT_SCHEMA:
            raise RuntimeError(f"inner{inner_fold} receipt schema is not formal nested OOF")
        receipt_identity = (
            int(receipt.get("target_outer_fold", -1)),
            int(receipt.get("inner_fold", -1)),
            int(receipt.get("seed", -1)),
        )
        if receipt_identity != (target_fold, inner_fold, seed):
            raise RuntimeError(f"inner{inner_fold} receipt identity mismatch")
        if receipt.get("split_csv_sha256") != sha256_file(split_csv):
            raise RuntimeError(f"inner{inner_fold} receipt split CSV hash mismatch")

        train_ids = _required_string_set(
            receipt.get("proposer_train_sample_ids"), "proposer_train_sample_ids"
        )
        holdout_ids = _required_string_set(
            receipt.get("inner_holdout_sample_ids"), "inner_holdout_sample_ids"
        )
        if receipt.get("proposer_train_sample_sha256") != sample_hash(sorted(train_ids)):
            raise RuntimeError(f"inner{inner_fold} proposer train sample hash mismatch")
        if receipt.get("inner_holdout_sample_sha256") != sample_hash(sorted(holdout_ids)):
            raise RuntimeError(f"inner{inner_fold} holdout sample hash mismatch")
        if entry.get("holdout_sample_sha256") != sample_hash(sorted(holdout_ids)):
            raise RuntimeError(f"inner{inner_fold} manifest holdout hash mismatch")
        if not train_ids <= outer_train or not holdout_ids <= outer_train:
            raise RuntimeError(f"inner{inner_fold} uses outer-val/test or unknown samples")
        if train_ids & holdout_ids or train_ids & outer_test or train_ids & outer_val:
            raise RuntimeError(f"inner{inner_fold} proposer sample leakage")
        if holdout_union & holdout_ids:
            raise RuntimeError("nested inner holdout sample identities overlap")
        holdout_union |= holdout_ids

        train_regions = _required_string_set(receipt.get("proposer_train_regions"), "train regions")
        holdout_regions = _required_string_set(receipt.get("inner_holdout_regions"), "holdout regions")
        derived_train_regions = {split_rows[item]["region_group"] for item in train_ids}
        derived_holdout_regions = {split_rows[item]["region_group"] for item in holdout_ids}
        if train_regions != derived_train_regions or holdout_regions != derived_holdout_regions:
            raise RuntimeError(f"inner{inner_fold} region receipt does not match split registry")
        if set(entry.get("holdout_regions", [])) != holdout_regions:
            raise RuntimeError(f"inner{inner_fold} manifest holdout regions mismatch")
        if train_regions & holdout_regions:
            raise RuntimeError(f"inner{inner_fold} proposer train/holdout region leakage")
        if train_regions & target_test_regions:
            raise RuntimeError(f"inner{inner_fold} proposer saw target outer-test geography")

        train_event_map = _required_mapping(
            receipt.get("proposer_train_sample_event_ids"), "proposer train sample-event map"
        )
        holdout_event_map = _required_mapping(
            receipt.get("inner_holdout_sample_event_ids"), "inner holdout sample-event map"
        )
        if set(train_event_map) != train_ids or set(holdout_event_map) != holdout_ids:
            raise RuntimeError(f"inner{inner_fold} sample-event map identities mismatch")
        if any(not isinstance(value, str) or not value for value in (*train_event_map.values(), *holdout_event_map.values())):
            raise RuntimeError(f"inner{inner_fold} sample-event map has invalid event IDs")
        train_events = set(train_event_map.values())
        holdout_events = set(holdout_event_map.values())
        if train_events != _required_string_set(receipt.get("proposer_train_events"), "train events"):
            raise RuntimeError(f"inner{inner_fold} train event receipt mismatch")
        if holdout_events != _required_string_set(receipt.get("inner_holdout_events"), "holdout events"):
            raise RuntimeError(f"inner{inner_fold} holdout event receipt mismatch")
        if set(entry.get("holdout_events", [])) != holdout_events:
            raise RuntimeError(f"inner{inner_fold} manifest holdout events mismatch")
        if train_events & holdout_events:
            raise RuntimeError(f"inner{inner_fold} proposer train/holdout event leakage")

        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        identity = _required_mapping(payload.get("identity"), "nested cache identity")
        if identity.get("schema_version") != FORMAL_CACHE_SCHEMA:
            raise RuntimeError(f"inner{inner_fold} cache is not nested inner-holdout OOF")
        cache_identity = (
            int(identity.get("target_outer_fold", -1)),
            int(identity.get("inner_fold", -1)),
            int(identity.get("seed", -1)),
            identity.get("split"),
        )
        if cache_identity != (target_fold, inner_fold, seed, "nested_inner_holdout"):
            raise RuntimeError(f"inner{inner_fold} cache identity mismatch")
        sample_ids = tuple(map(str, payload.get("sample_ids", ())))
        if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != holdout_ids:
            raise RuntimeError(f"inner{inner_fold} cache samples do not equal holdout receipt")
        if identity.get("sample_sha256") != sample_hash(sample_ids):
            raise RuntimeError(f"inner{inner_fold} cache sample hash mismatch")
        if tuple(map(str, payload.get("event_ids", ()))) != tuple(holdout_event_map[item] for item in sample_ids):
            raise RuntimeError(f"inner{inner_fold} cache event IDs mismatch producer receipt")
        expected_sources = tuple(split_rows[item]["region_group"] for item in sample_ids)
        if tuple(map(str, payload.get("source_ids", ()))) != expected_sources:
            raise RuntimeError(f"inner{inner_fold} cache region IDs mismatch split registry")
        if set(identity.get("holdout_regions", [])) != holdout_regions:
            raise RuntimeError(f"inner{inner_fold} cache holdout regions mismatch")
        if set(identity.get("holdout_events", [])) != holdout_events:
            raise RuntimeError(f"inner{inner_fold} cache holdout events mismatch")
        if identity.get("producer_receipt_sha256") != sha256_file(receipt_path):
            raise RuntimeError(f"inner{inner_fold} cache is not bound to producer receipt")
        for key in ("visual_checkpoint_sha256", "terrain_checkpoint_sha256"):
            expected = receipt.get(key)
            if not expected or entry.get(key) != expected or identity.get(key) != expected:
                raise RuntimeError(f"inner{inner_fold} {key} mismatch")
        threshold = float(receipt.get("visual_threshold", float("nan")))
        if float(entry.get("visual_threshold", float("nan"))) != threshold:
            raise RuntimeError(f"inner{inner_fold} manifest threshold mismatch")
        if float(identity.get("visual_threshold", float("nan"))) != threshold:
            raise RuntimeError(f"inner{inner_fold} cache threshold mismatch")

        access_log.append({
            "stage": "nested_meta_cv_train",
            "fold": inner_fold,
            "labels_loaded": True,
            "target_outer_fold": target_fold,
            "identity_role": "nested_inner_holdout_oof",
        })
        bundles.append(_bundle_from_payload(
            bundle_id=inner_fold,
            seed=seed,
            sample_ids=sample_ids,
            threshold=threshold,
            payload=payload,
            identity=identity,
            cache_path=cache_path,
            result_path=receipt_path,
        ))
        entry_receipts.append({
            "inner_fold": inner_fold,
            "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "producer_receipt_path": str(receipt_path),
            "producer_receipt_sha256": sha256_file(receipt_path),
            "holdout_sample_sha256": sample_hash(sorted(holdout_ids)),
            "holdout_regions": sorted(holdout_regions),
            "holdout_events": sorted(holdout_events),
            "proposer_train_regions": sorted(train_regions),
            "proposer_train_events": sorted(train_events),
        })

    if holdout_union != outer_train:
        missing = sorted(outer_train - holdout_union)[:20]
        extra = sorted(holdout_union - outer_train)[:20]
        raise RuntimeError(
            f"nested holdouts must partition target outer-train exactly; missing={missing}, extra={extra}"
        )
    audit = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": FORMAL_MANIFEST_SCHEMA,
        "target_outer_fold": target_fold,
        "outer_train_sample_sha256": sample_hash(sorted(outer_train)),
        "outer_test_sample_sha256": sample_hash(sorted(outer_test)),
        "target_outer_test_regions": sorted(target_test_regions),
        "inner_holdouts_partition_outer_train": True,
        "proposer_train_regions_disjoint_target_test_and_inner_holdout": True,
        "proposer_train_events_disjoint_inner_holdout": True,
        "entries": sorted(entry_receipts, key=lambda item: item["inner_fold"]),
    }
    return sorted(bundles, key=lambda bundle: bundle.fold), audit


def build_proposal_table(bundle: FoldBundle) -> ProposalTable:
    z = bundle.visual_logits.reshape(-1)
    delta = bundle.terrain_delta.reshape(-1)
    valid = bundle.valid.reshape(-1)
    target = bundle.mask.reshape(-1)
    threshold = bundle.threshold_logit
    visual = z >= threshold
    terrain = (z + delta) >= threshold
    actionable = valid & (visual != terrain)
    flat_index = np.flatnonzero(actionable)
    pixels_per_sample = int(np.prod(bundle.visual_logits.shape[1:]))
    sample_index = (flat_index // pixels_per_sample).astype(np.int32)
    probability = 1.0 / (1.0 + np.exp(-np.clip(z[flat_index], -30.0, 30.0)))
    base = np.stack(
        (
            z[flat_index] - threshold,
            z[flat_index] + delta[flat_index] - threshold,
            delta[flat_index],
            np.abs(delta[flat_index]),
            1.0 - np.abs(2.0 * probability - 1.0),
        ),
        axis=1,
    ).astype(np.float32)
    proposal_type = ((~visual[flat_index]) & terrain[flat_index]).astype(np.uint8)
    utility = (terrain[flat_index] == target[flat_index]).astype(np.uint8)
    per_sample = np.bincount(sample_index, minlength=len(bundle.sample_ids)).clip(min=1)
    weights = (1.0 / per_sample[sample_index]).astype(np.float64)
    weights /= max(float(weights.sum()), 1.0)
    return ProposalTable(
        fold=bundle.fold, sample_index=sample_index, proposal_type=proposal_type,
        base_features=base, utility_target=utility, sample_weights=weights,
        visual_prediction=visual[flat_index], terrain_prediction=terrain[flat_index],
        pixel_target=target[flat_index],
        always_terrain_counts=confusion(terrain, target, valid),
        visual_counts=confusion(visual, target, valid), n_valid=int(valid.sum()),
    )


def feature_names(context: str, material_dim: int, trigger_dim: int) -> tuple[str, ...]:
    if context not in CONTEXTS:
        raise ValueError(context)
    names = list(BASE_FEATURE_NAMES)
    if context in ("TM", "TMR"):
        names += [f"material_{index}" for index in range(material_dim)] + ["q_M"]
    if context in ("TR", "TMR"):
        names += [f"trigger_{index}" for index in range(trigger_dim)] + ["q_R"]
    if FORBIDDEN_INFERENCE_FEATURES & set(names):
        raise RuntimeError("forbidden field entered inference feature schema")
    return tuple(names)


def context_values(
    bundle: FoldBundle, context: str, control: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    material, q_m = bundle.material, bundle.q_material
    trigger, q_r = bundle.trigger, bundle.q_trigger
    if control == "material_shuffle":
        material, q_m = bundle.material_shuffle, bundle.q_material_shuffle
    elif control in ("material_zero_q", "all_zero_q"):
        q_m = np.zeros_like(q_m)
    if control == "trigger_wrong_time":
        trigger = bundle.trigger_wrong
    elif control == "trigger_event_shuffle":
        trigger, q_r = bundle.trigger_shuffle, bundle.q_trigger_shuffle
    elif control in ("trigger_zero_q", "all_zero_q"):
        q_r = np.zeros_like(q_r)
    if context == "proposal_only":
        active = np.ones(len(bundle.sample_ids), bool)
    elif context == "TM":
        active = np.isfinite(q_m) & (q_m > 0) & np.isfinite(material).all(axis=1)
    elif context == "TR":
        active = np.isfinite(q_r) & (q_r > 0) & np.isfinite(trigger).all(axis=1)
    else:
        active = (
            np.isfinite(q_m) & (q_m > 0) & np.isfinite(material).all(axis=1)
            & np.isfinite(q_r) & (q_r > 0) & np.isfinite(trigger).all(axis=1)
        )
    return material, q_m, trigger, q_r, active


def make_features(
    bundle: FoldBundle,
    table: ProposalTable,
    context: str,
    control: str = "aligned",
) -> tuple[np.ndarray, np.ndarray]:
    material, q_m, trigger, q_r, active_sample = context_values(bundle, context, control)
    sample = table.sample_index
    parts = [table.base_features]
    if context != "proposal_only":
        if context in ("TM", "TMR"):
            clean_m = np.nan_to_num(material, nan=0.0, posinf=0.0, neginf=0.0)
            parts.extend((clean_m[sample], q_m[sample, None]))
        if context in ("TR", "TMR"):
            clean_r = np.nan_to_num(trigger, nan=0.0, posinf=0.0, neginf=0.0)
            parts.extend((clean_r[sample], q_r[sample, None]))
    features = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return features, active_sample[sample]


def _fit_head(
    context: str,
    proposal_type: str,
    bundles: Sequence[FoldBundle],
    tables: Mapping[int, ProposalTable],
    alpha: float,
    seed: int,
    *,
    label_shuffle: bool = False,
) -> LinearHead:
    xs, ys, ws = [], [], []
    code = 1 if proposal_type == "rescue" else 0
    for bundle in bundles:
        table = tables[bundle.fold]
        features, active = make_features(bundle, table, context)
        selected = (table.proposal_type == code) & active
        if not np.any(selected):
            continue
        target = table.utility_target[selected].copy()
        if label_shuffle:
            rng = np.random.default_rng(seed + 1009 * bundle.fold + code)
            rng.shuffle(target)
        xs.append(features[selected])
        ys.append(target)
        ws.append(table.sample_weights[selected])
    names = feature_names(context, bundles[0].material.shape[1], bundles[0].trigger.shape[1])
    if not xs:
        return LinearHead(names, np.zeros(len(names), np.float32), np.ones(len(names), np.float32),
                          np.zeros(len(names), np.float32), 0.0, 1.0, alpha, proposal_type)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    weight = np.concatenate(ws)
    weight *= len(weight) / max(float(weight.sum()), 1e-12)
    if len(np.unique(y)) < 2:
        probability = float(np.mean(y)) if len(y) else 1.0
        return LinearHead(names, np.zeros(x.shape[1], np.float32), np.ones(x.shape[1], np.float32),
                          np.zeros(x.shape[1], np.float32), 0.0, probability, alpha, proposal_type)
    scaler = StandardScaler().fit(x, sample_weight=weight)
    scale = np.where(scaler.scale_ > 1e-8, scaler.scale_, 1.0)
    standardized = (x - scaler.mean_) / scale
    classifier = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=alpha, max_iter=30, tol=1e-5,
        random_state=seed + code, average=True, fit_intercept=True,
    )
    classifier.fit(standardized, y, sample_weight=weight)
    return LinearHead(
        names, scaler.mean_.astype(np.float32), scale.astype(np.float32),
        classifier.coef_[0].astype(np.float32), float(classifier.intercept_[0]),
        None, alpha, proposal_type,
    )


def fit_gate(
    context: str,
    bundles: Sequence[FoldBundle],
    tables: Mapping[int, ProposalTable],
    alpha: float,
    seed: int,
    *,
    label_shuffle: bool = False,
) -> UtilityGate:
    return UtilityGate(
        context=context, alpha=alpha,
        rescue=_fit_head(context, "rescue", bundles, tables, alpha, seed, label_shuffle=label_shuffle),
        veto=_fit_head(context, "veto", bundles, tables, alpha, seed, label_shuffle=label_shuffle),
    )


def gate_probabilities(
    gate: UtilityGate,
    bundle: FoldBundle,
    table: ProposalTable,
    control: str,
) -> tuple[np.ndarray, np.ndarray]:
    features, active = make_features(bundle, table, gate.context, control)
    probability = np.ones(len(table.utility_target), np.float32)
    for proposal_type, head in (("rescue", gate.rescue), ("veto", gate.veto)):
        selected = table.mask_for_type(proposal_type)
        if np.any(selected):
            probability[selected] = head.predict_probability(features[selected])
    return probability, active


def threshold_decisions(
    probability: np.ndarray,
    proposal_type: np.ndarray,
    rescue_threshold: float,
    veto_threshold: float,
) -> np.ndarray:
    thresholds = np.where(proposal_type == 1, rescue_threshold, veto_threshold)
    return probability >= thresholds


def context_decisions(
    probability: np.ndarray,
    active: np.ndarray,
    table: ProposalTable,
    rescue_threshold: float,
    veto_threshold: float,
    proposal_decisions: np.ndarray,
) -> np.ndarray:
    candidate = threshold_decisions(
        probability, table.proposal_type, rescue_threshold, veto_threshold
    )
    return np.where(active, candidate, proposal_decisions)


def apply_binary_gate(
    terrain_delta: np.ndarray, actionable: np.ndarray, accept: np.ndarray
) -> np.ndarray:
    terrain_delta = np.asarray(terrain_delta)
    actionable = np.asarray(actionable, bool)
    accept = np.asarray(accept, bool)
    if terrain_delta.shape != actionable.shape or terrain_delta.shape != accept.shape:
        raise ValueError("residual/actionable/accept shapes must match")
    residual = terrain_delta.copy()
    residual[actionable & ~accept] = 0
    if not np.all((residual == 0) | (residual == terrain_delta)):
        raise RuntimeError("binary residual contract violated")
    return residual


def score_decisions(table: ProposalTable, accept: np.ndarray) -> dict[str, Any]:
    counts = dict(table.always_terrain_counts)
    reject = ~np.asarray(accept, bool)
    rescue = table.proposal_type == 1
    beneficial = table.utility_target == 1
    # Reject rescue: Terrain 1 -> visual 0.
    value = reject & rescue & beneficial
    counts["tp"] -= int(value.sum()); counts["fn"] += int(value.sum())
    value = reject & rescue & ~beneficial
    counts["fp"] -= int(value.sum()); counts["tn"] += int(value.sum())
    # Reject veto: Terrain 0 -> visual 1.
    value = reject & ~rescue & beneficial
    counts["tn"] -= int(value.sum()); counts["fp"] += int(value.sum())
    value = reject & ~rescue & ~beneficial
    counts["fn"] -= int(value.sum()); counts["tp"] += int(value.sum())
    output = metric_dict(counts)
    output.update({
        "proposal_count": int(len(accept)),
        "proposal_rescue_count": int(rescue.sum()),
        "proposal_veto_count": int((~rescue).sum()),
        "proposal_accepted": int(np.sum(accept)),
        "proposal_rejected": int(np.sum(reject)),
        "terrain_benefits_retained": int(np.sum(accept & beneficial)),
        "terrain_benefits_rejected": int(np.sum(reject & beneficial)),
        "terrain_harms_retained": int(np.sum(accept & ~beneficial)),
        "terrain_harms_prevented": int(np.sum(reject & ~beneficial)),
    })
    return output


def aggregate_scores(tables: Sequence[ProposalTable], decisions: Sequence[np.ndarray]) -> dict[str, Any]:
    total = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    proposal = {key: 0 for key in (
        "proposal_count", "proposal_rescue_count", "proposal_veto_count",
        "proposal_accepted", "proposal_rejected", "terrain_benefits_retained",
        "terrain_benefits_rejected", "terrain_harms_retained", "terrain_harms_prevented",
    )}
    for table, accept in zip(tables, decisions):
        score = score_decisions(table, accept)
        for key in total: total[key] += int(score[key])
        for key in proposal: proposal[key] += int(score[key])
    return {**metric_dict(total), **proposal}


def claim_passes(
    aligned: Mapping[str, Any],
    proposal_only: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
) -> bool:
    if aligned["iou"] <= proposal_only["iou"] + 1e-12:
        return False
    if aligned["errors"] > proposal_only["errors"]:
        return False
    for name, control in controls.items():
        if name == "aligned":
            continue
        if aligned["iou"] <= control["iou"] + 1e-12 or aligned["errors"] > control["errors"]:
            return False
    return True


def _candidate_pairs(grid: Sequence[float]) -> list[tuple[float, float]]:
    return [(float(rescue), float(veto)) for rescue in grid for veto in grid]


def crossfit_probabilities(
    context: str,
    bundles: Sequence[FoldBundle],
    tables: Mapping[int, ProposalTable],
    alpha: float,
    seed: int,
    *,
    label_shuffle: bool = False,
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    output: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for held in bundles:
        train = [bundle for bundle in bundles if bundle.fold != held.fold]
        gate = fit_gate(context, train, tables, alpha, seed + held.fold, label_shuffle=label_shuffle)
        output[held.fold] = {
            control: gate_probabilities(gate, held, tables[held.fold], control)
            for control in CONTROL_NAMES[context]
        }
    return output


def select_proposal_only(
    bundles: Sequence[FoldBundle],
    tables: Mapping[int, ProposalTable],
    alphas: Sequence[float],
    threshold_grid: Sequence[float],
    seed: int,
) -> tuple[Selection, dict[int, np.ndarray]]:
    best: tuple[tuple[float, int, int], Selection, dict[int, np.ndarray]] | None = None
    identity = aggregate_scores([tables[b.fold] for b in bundles], [np.ones(len(tables[b.fold].utility_target), bool) for b in bundles])
    for alpha in alphas:
        oof = crossfit_probabilities("proposal_only", bundles, tables, alpha, seed)
        for rescue_threshold, veto_threshold in [(0.0, 0.0), *_candidate_pairs(threshold_grid)]:
            decisions = {
                bundle.fold: threshold_decisions(
                    oof[bundle.fold]["aligned"][0], tables[bundle.fold].proposal_type,
                    rescue_threshold, veto_threshold,
                )
                for bundle in bundles
            }
            score = aggregate_scores(
                [tables[b.fold] for b in bundles], [decisions[b.fold] for b in bundles]
            )
            if score["errors"] > identity["errors"]:
                continue
            key = (float(score["iou"]), -int(score["errors"]), int(score["proposal_accepted"]))
            selection = Selection(
                "proposal_only", float(alpha), rescue_threshold, veto_threshold,
                score, {"always_on_terrain": identity}, True, "always_on_terrain",
            )
            if best is None or key > best[0]:
                best = (key, selection, decisions)
    if best is None:
        decisions = {b.fold: np.ones(len(tables[b.fold].utility_target), bool) for b in bundles}
        return Selection("proposal_only", float(alphas[0]), 0.0, 0.0, identity,
                         {"always_on_terrain": identity}, True, "always_on_terrain"), decisions
    return best[1], best[2]


def select_context(
    context: str,
    bundles: Sequence[FoldBundle],
    tables: Mapping[int, ProposalTable],
    alphas: Sequence[float],
    threshold_grid: Sequence[float],
    seed: int,
    proposal_selection: Selection,
    proposal_decisions: Mapping[int, np.ndarray],
    *,
    label_shuffle: bool = False,
    fixed_threshold_pair: tuple[float, float] | None = None,
) -> Selection:
    proposal_score = aggregate_scores(
        [tables[b.fold] for b in bundles], [proposal_decisions[b.fold] for b in bundles]
    )
    best: tuple[tuple[float, int], Selection] | None = None
    for alpha in alphas:
        oof = crossfit_probabilities(
            context, bundles, tables, alpha, seed, label_shuffle=label_shuffle
        )
        threshold_pairs = (
            [fixed_threshold_pair]
            if fixed_threshold_pair is not None
            else _candidate_pairs(threshold_grid)
        )
        for rescue_threshold, veto_threshold in threshold_pairs:
            control_scores: dict[str, Any] = {}
            control_decisions: dict[str, list[np.ndarray]] = {name: [] for name in CONTROL_NAMES[context]}
            for bundle in bundles:
                table = tables[bundle.fold]
                for control in CONTROL_NAMES[context]:
                    probability, active = oof[bundle.fold][control]
                    control_decisions[control].append(context_decisions(
                        probability, active, table, rescue_threshold, veto_threshold,
                        proposal_decisions[bundle.fold],
                    ))
            ordered_tables = [tables[b.fold] for b in bundles]
            for control, decisions in control_decisions.items():
                control_scores[control] = aggregate_scores(ordered_tables, decisions)
            aligned = control_scores["aligned"]
            passed = claim_passes(aligned, proposal_score, control_scores)
            key = (float(aligned["iou"]), -int(aligned["errors"]))
            selection = Selection(
                context, float(alpha), rescue_threshold, veto_threshold, aligned,
                {**control_scores, "proposal_only": proposal_score}, passed,
                "proposal_only" if not passed else "none",
            )
            if best is None or (passed, key) > (best[1].claim_pass, best[0]):
                best = (key, selection)
    if best is None:
        raise RuntimeError(f"no candidate evaluated for {context}")
    return best[1]


def label_shuffle_sanity(
    context: str,
    selection: Selection,
    bundles: Sequence[FoldBundle],
    tables: Mapping[int, ProposalTable],
    seed: int,
    proposal_selection: Selection,
    proposal_decisions: Mapping[int, np.ndarray],
) -> bool:
    shuffled = select_context(
        context, bundles, tables, (selection.alpha,),
        (), seed + 900_001,
        proposal_selection, proposal_decisions, label_shuffle=True,
        fixed_threshold_pair=(selection.rescue_threshold, selection.veto_threshold),
    )
    return bool(shuffled.claim_pass)


def _fold_receipt(bundle: FoldBundle) -> dict[str, Any]:
    visual_sha = bundle.identity.get("visual_checkpoint_sha256")
    terrain_sha = bundle.identity.get("terrain_checkpoint_sha256")
    if visual_sha is None:
        visual_sha = bundle.identity.get("visual_checkpoint", {}).get("sha256")
    if terrain_sha is None:
        terrain_sha = bundle.identity.get("terrain_checkpoint", {}).get("sha256")
    return {
        "fold": bundle.fold,
        "n_samples": len(bundle.sample_ids),
        "sample_sha256": sample_hash(bundle.sample_ids),
        "cache_path": bundle.cache_path,
        "cache_sha256": bundle.cache_sha256,
        "result_path": bundle.result_path,
        "result_sha256": bundle.result_sha256,
        "producer_visual_checkpoint_sha256": visual_sha,
        "producer_terrain_checkpoint_sha256": terrain_sha,
        "producer_threshold": bundle.threshold,
        "identity_split": bundle.identity["split"],
    }


def _target_predictions(
    bundle: FoldBundle,
    table: ProposalTable,
    context: str,
    gate: UtilityGate,
    selection: Selection,
    proposal_gate: UtilityGate,
    proposal_selection: Selection,
    control: str,
    *,
    deployed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    proposal_probability, _ = gate_probabilities(
        proposal_gate, bundle, table, "aligned"
    )
    proposal_accept = threshold_decisions(
        proposal_probability, table.proposal_type,
        proposal_selection.rescue_threshold, proposal_selection.veto_threshold,
    )
    if context == "proposal_only" or (deployed and not selection.claim_pass):
        return proposal_accept, proposal_probability, np.ones(len(proposal_accept), bool), True
    probability, active = gate_probabilities(gate, bundle, table, control)
    accept = context_decisions(
        probability, active, table, selection.rescue_threshold,
        selection.veto_threshold, proposal_accept,
    )
    return accept, probability, active, False


class ProbabilityHistogram:
    def __init__(self, bins: int = 4096) -> None:
        self.bins = bins
        self.positive = np.zeros(bins, np.int64)
        self.negative = np.zeros(bins, np.int64)

    def update(self, probability: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        p = np.asarray(probability, np.float64)[valid]
        y = np.asarray(target, bool)[valid]
        index = np.clip((p * self.bins).astype(np.int64), 0, self.bins - 1)
        self.positive += np.bincount(index[y], minlength=self.bins)
        self.negative += np.bincount(index[~y], minlength=self.bins)

    def average_precision(self) -> float:
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        if not len(tp) or tp[-1] == 0:
            return 0.0
        precision = tp / np.maximum(tp + fp, 1)
        return float(np.sum(precision * (self.positive[::-1] / tp[-1])))


def evaluate_target_variant(
    bundle: FoldBundle,
    table: ProposalTable,
    accept: np.ndarray,
    context: str,
    control: str,
    checkpoint_sha256: str,
    selected_by_meta_cv: bool,
    fallback_used: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    z = bundle.visual_logits.astype(np.float32, copy=False)
    delta = bundle.terrain_delta.astype(np.float32, copy=False)
    valid = bundle.valid.astype(bool, copy=False)
    target = bundle.mask.astype(bool, copy=False)
    threshold = bundle.threshold_logit
    visual_prediction = z >= threshold
    terrain_prediction = (z + delta) >= threshold
    actionable = valid & (visual_prediction != terrain_prediction)
    flat_accept = np.ones(actionable.size, bool)
    flat_accept[np.flatnonzero(actionable.reshape(-1))] = accept
    accept_map = flat_accept.reshape(actionable.shape)
    residual = apply_binary_gate(delta, actionable, accept_map)
    logits = z + residual
    prediction = logits >= threshold
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    if not np.all((residual == 0) | (residual == delta)):
        raise RuntimeError("target residual escaped {0, DeltaT}")
    total = metric_dict(confusion(prediction, target, valid))
    visual_counts = metric_dict(confusion(visual_prediction, target, valid))
    terrain_counts = metric_dict(confusion(terrain_prediction, target, valid))
    corrected_visual = (~(visual_prediction == target)) & (prediction == target) & valid
    harmed_visual = (visual_prediction == target) & ~(prediction == target) & valid
    corrected_terrain = (~(terrain_prediction == target)) & (prediction == target) & valid
    harmed_terrain = (terrain_prediction == target) & ~(prediction == target) & valid
    histogram = ProbabilityHistogram()
    histogram.update(probability, target, valid)
    score = score_decisions(table, accept)
    summary = {
        **total,
        "ap": histogram.average_precision(),
        "visual": visual_counts,
        "always_on_terrain": terrain_counts,
        "corrected_vs_visual": int(corrected_visual.sum()),
        "harmed_vs_visual": int(harmed_visual.sum()),
        "rer_vs_visual": (visual_counts["errors"] - total["errors"]) / max(visual_counts["errors"], 1),
        "corrected_vs_always_on_terrain": int(corrected_terrain.sum()),
        "harmed_vs_always_on_terrain": int(harmed_terrain.sum()),
        "rer_vs_always_on_terrain": (terrain_counts["errors"] - total["errors"]) / max(terrain_counts["errors"], 1),
        **{key: score[key] for key in score if key.startswith("proposal_") or key.startswith("terrain_")},
        "residual_contract": "bit-exact element of {0, DeltaT}",
        "checkpoint_sha256": checkpoint_sha256,
        "selected_by_meta_cv": selected_by_meta_cv,
        "fallback_used": fallback_used,
    }
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(bundle.sample_ids):
        item_valid = valid[index]
        item_target = target[index]
        item_prediction = prediction[index]
        item_visual = visual_prediction[index]
        item_terrain = terrain_prediction[index]
        item_actionable = actionable[index]
        item_accept = accept_map[index]
        counts = metric_dict(confusion(item_prediction, item_target, item_valid))
        visual_item = metric_dict(confusion(item_visual, item_target, item_valid))
        terrain_item = metric_dict(confusion(item_terrain, item_target, item_valid))
        corrected_v = int(((item_visual != item_target) & (item_prediction == item_target) & item_valid).sum())
        harmed_v = int(((item_visual == item_target) & (item_prediction != item_target) & item_valid).sum())
        corrected_t = int(((item_terrain != item_target) & (item_prediction == item_target) & item_valid).sum())
        harmed_t = int(((item_terrain == item_target) & (item_prediction != item_target) & item_valid).sum())
        valid_probability = probability[index][item_valid]
        valid_target = item_target[item_valid]
        ap = float(average_precision_score(valid_target, valid_probability)) if np.any(valid_target) else 0.0
        rows.append({
            "fold": bundle.fold, "sample_id": sample_id, "event_id": bundle.event_ids[index],
            "source_id": bundle.source_ids[index], "context": context, "control": control,
            "selected_by_meta_cv": selected_by_meta_cv, "fallback_used": fallback_used,
            "checkpoint_sha256": checkpoint_sha256, **counts, "ap": ap,
            "visual_iou": visual_item["iou"], "always_on_terrain_iou": terrain_item["iou"],
            "visual_errors": visual_item["errors"], "always_on_terrain_errors": terrain_item["errors"],
            "corrected_vs_visual": corrected_v, "harmed_vs_visual": harmed_v,
            "rer_vs_visual": (visual_item["errors"] - counts["errors"]) / max(visual_item["errors"], 1),
            "corrected_vs_always_on_terrain": corrected_t,
            "harmed_vs_always_on_terrain": harmed_t,
            "rer_vs_always_on_terrain": (
                terrain_item["errors"] - counts["errors"]
            ) / max(terrain_item["errors"], 1),
            "proposal_count": int(item_actionable.sum()),
            "proposal_rescue_count": int((item_actionable & ~item_visual & item_terrain).sum()),
            "proposal_veto_count": int((item_actionable & item_visual & ~item_terrain).sum()),
            "proposal_accepted": int((item_actionable & item_accept).sum()),
            "proposal_rejected": int((item_actionable & ~item_accept).sum()),
            "q_M": float(bundle.q_material[index]), "q_R": float(bundle.q_trigger[index]),
            "test_label_used_for_gate": False,
            "inference_feature_forbidden_identity_fields": True,
        })
    return summary, rows


def run_protocol(
    args: argparse.Namespace,
    *,
    target_loader_fn: Callable[..., FoldBundle] = load_fold_bundle,
    nested_loader_fn: Callable[..., tuple[list[FoldBundle], dict[str, Any]]] = load_formal_nested_bundles,
) -> dict[str, Any]:
    access_log: list[dict[str, Any]] = []
    if args.protocol_mode == "formal_nested_oof":
        if args.oof_manifest is None:
            raise RuntimeError("formal_nested_oof requires --oof-manifest")
        train_bundles, provenance_audit = nested_loader_fn(
            args.oof_manifest,
            target_fold=args.target_fold,
            split_csv=args.split_csv,
            seed=args.seed,
            access_log=access_log,
        )
        train_folds = tuple(bundle.fold for bundle in train_bundles)
        evidence_status = "formal_nested_oof"
        exploratory_only = False
        manuscript_pass_prohibited_reason = None
    elif args.protocol_mode == "cross_outer_exploratory":
        train_folds = tuple(fold for fold in FOLDS if fold != args.target_fold)
        train_bundles = [
            target_loader_fn(
                fold, cache_root=args.cache_root, runs_root=args.runs_root,
                split_csv=args.split_csv, seed=args.seed, access_log=access_log,
                purpose="exploratory_cross_outer_train",
            )
            for fold in train_folds
        ]
        provenance_audit = {
            "schema_version": "cross_outer_exploratory_receipt.v1",
            "warning": (
                "Non-target outer proposer training geography may overlap the target outer-test "
                "geography; this evidence is not manuscript-eligible."
            ),
        }
        evidence_status = "exploratory_only"
        exploratory_only = True
        manuscript_pass_prohibited_reason = "cross_outer_proposer_geography_not_proven_independent"
    else:
        raise RuntimeError(f"unsupported protocol mode: {args.protocol_mode}")

    if any(
        entry["fold"] == args.target_fold
        and entry.get("identity_role") != "nested_inner_holdout_oof"
        for entry in access_log
    ):
        raise RuntimeError("target fold was accessed before selection")
    if len(train_bundles) < 3:
        raise RuntimeError("leave-one-holdout meta-CV requires at least three OOF bundles")
    sets = [set(bundle.sample_ids) for bundle in train_bundles]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise RuntimeError("gate-training OOF caches overlap")
    tables = {bundle.fold: build_proposal_table(bundle) for bundle in train_bundles}
    proposal_selection, proposal_oof = select_proposal_only(
        train_bundles, tables, args.alphas, args.threshold_grid, args.seed
    )
    selections: dict[str, Selection] = {"proposal_only": proposal_selection}
    for context in CONTEXTS[1:]:
        selection = select_context(
            context, train_bundles, tables, args.alphas, args.threshold_grid,
            args.seed, proposal_selection, proposal_oof,
        )
        selection.label_shuffle_claim_pass = label_shuffle_sanity(
            context, selection, train_bundles, tables, args.seed,
            proposal_selection, proposal_oof,
        )
        if selection.label_shuffle_claim_pass:
            selection.claim_pass = False
            selection.fallback = "proposal_only; label-shuffle sanity failed"
        selections[context] = selection

    gates = {
        context: fit_gate(
            context, train_bundles, tables, selections[context].alpha,
            args.seed + 70_000,
        )
        for context in CONTEXTS
    }
    frozen_receipt = {
        "stage": "selection_and_fit_frozen",
        "protocol_mode": args.protocol_mode,
        "target_fold": args.target_fold,
        "meta_cv_holdout_ids": train_folds,
        "target_outer_test_labels_read": False,
        "training_provenance_sha256": sha256_json(provenance_audit),
        "selections": {context: asdict(value) for context, value in selections.items()},
        "gate_receipts": {context: gate.receipt() for context, gate in gates.items()},
    }
    frozen_receipt["sha256"] = sha256_json(frozen_receipt)
    access_log.append({
        "stage": "models_frozen", "fold": None, "labels_loaded": False,
        "frozen_receipt_sha256": frozen_receipt["sha256"],
    })

    target = target_loader_fn(
        args.target_fold, cache_root=args.cache_root, runs_root=args.runs_root,
        split_csv=args.split_csv, seed=args.seed, access_log=access_log,
        purpose="target_evaluation_after_freeze",
    )
    if any(set(target.sample_ids) & values for values in sets):
        raise RuntimeError("target fold identities overlap gate-training folds")
    if args.protocol_mode == "formal_nested_oof":
        gate_training_events = set().union(*(set(bundle.event_ids) for bundle in train_bundles))
        declared_proposer_events = {
            event
            for entry in provenance_audit["entries"]
            for key in ("proposer_train_events", "holdout_events")
            for event in entry[key]
        }
        target_events = set(target.event_ids)
        overlap = sorted((gate_training_events | declared_proposer_events) & target_events)
        if overlap:
            raise RuntimeError(f"target outer-test event leakage: {overlap[:20]}")
        provenance_audit["target_outer_test_events"] = sorted(target_events)
        provenance_audit["gate_and_proposer_events_disjoint_target_outer_test"] = True
    target_table = build_proposal_table(target)
    result: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    proposal_gate = gates["proposal_only"]
    for context in CONTEXTS:
        gate = gates[context]
        selection = selections[context]
        candidate_controls: dict[str, Any] = {}
        checkpoint = gate.receipt()["checkpoint_sha256"]
        for control in CONTROL_NAMES[context]:
            accept, _, _, fallback = _target_predictions(
                target, target_table, context, gate, selection, proposal_gate,
                proposal_selection, control, deployed=False,
            )
            summary, rows = evaluate_target_variant(
                target, target_table, accept, context, control, checkpoint,
                selection.claim_pass, fallback,
            )
            candidate_controls[control] = summary
            sample_rows.extend(rows)
        deployed_accept, _, _, deployed_fallback = _target_predictions(
            target, target_table, context, gate, selection, proposal_gate,
            proposal_selection, "aligned", deployed=True,
        )
        deployed, rows = evaluate_target_variant(
            target, target_table, deployed_accept, context, "deployed", checkpoint,
            selection.claim_pass, deployed_fallback,
        )
        sample_rows.extend(rows)
        result[context] = {
            "selection": asdict(selection),
            "checkpoint": gate.receipt(),
            "candidate_controls": candidate_controls,
            "deployed": deployed,
            "negative_controls_reuse_same_checkpoint": all(
                item["checkpoint_sha256"] == checkpoint for item in candidate_controls.values()
            ),
        }
    proposal_target = result["proposal_only"]["candidate_controls"]["aligned"]
    target_claim_passes: dict[str, bool] = {"proposal_only": True}
    for context in CONTEXTS[1:]:
        candidate_controls = result[context]["candidate_controls"]
        passed = bool(
            selections[context].claim_pass
            and claim_passes(
                candidate_controls["aligned"], proposal_target, candidate_controls
            )
        )
        target_claim_passes[context] = passed
        result[context]["target_claim_pass"] = passed
    result["proposal_only"]["target_claim_pass"] = True
    label_shuffle_sanity_pass = all(
        not selection.label_shuffle_claim_pass
        for context, selection in selections.items()
        if context != "proposal_only"
    )
    manuscript_pass = bool(
        args.protocol_mode == "formal_nested_oof"
        and not exploratory_only
        and label_shuffle_sanity_pass
        and any(target_claim_passes[context] for context in CONTEXTS[1:])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "seed": args.seed,
        "target_fold": args.target_fold,
        "meta_cv_holdout_ids": train_folds,
        "evidence_status": evidence_status,
        "exploratory_only": exploratory_only,
        "manuscript_pass": manuscript_pass,
        "manuscript_pass_prohibited_reason": manuscript_pass_prohibited_reason,
        "protocol": {
            "mode": args.protocol_mode,
            "proposer_cache_role": (
                "nested inner event/region holdout OOF prediction"
                if args.protocol_mode == "formal_nested_oof"
                else "cross-outer exploratory cache; indirect geography leakage not excluded"
            ),
            "meta_cv": (
                "leave-one-inner-holdout-out within target outer-train"
                if args.protocol_mode == "formal_nested_oof"
                else "leave-one-producer-outer-cache-out; exploratory only"
            ),
            "target_outer_test_used_for_training_or_selection": False,
            "target_outer_test_loaded_only_after_models_frozen": True,
            "selection_primary": "pooled IoU",
            "selection_constraint": "total errors no worse",
            "residual_contract": "bit-exact {0, DeltaT}",
            "default_action": "accept Terrain proposal",
            "rescue_veto_heads": "separate",
            "inference_feature_forbidden_fields": sorted(FORBIDDEN_INFERENCE_FEATURES),
            "label_shuffle_sanity_pass": label_shuffle_sanity_pass,
            "target_claim_passes": target_claim_passes,
            "manuscript_pass_requires": (
                "formal provenance, label-shuffle sanity, and at least one M/R context "
                "passing meta-CV plus untouched target controls"
            ),
        },
        "provenance_audit": provenance_audit,
        "frozen_receipt": frozen_receipt,
        "access_log": access_log,
        "producer_receipts": [_fold_receipt(bundle) for bundle in train_bundles] + [_fold_receipt(target)],
        "target_evaluation": result,
        "per_sample_rows": sample_rows,
    }


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(not math.isfinite(item) or item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated finite values")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-fold", type=int, choices=FOLDS, required=True)
    parser.add_argument(
        "--protocol-mode", choices=PROTOCOL_MODES, default="formal_nested_oof",
        help="formal nested OOF (default) or explicitly non-manuscript exploratory mode",
    )
    parser.add_argument(
        "--oof-manifest", type=Path,
        help="required in formal mode; manifest for target-fold nested inner-holdout caches",
    )
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--alphas", type=parse_float_list, default=(1e-5, 1e-4, 1e-3))
    parser.add_argument("--threshold-grid", type=parse_float_list, default=(0.35, 0.50, 0.65))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.outdir is None:
        args.outdir = DEFAULT_OUTROOT / args.protocol_mode / f"fold{args.target_fold}_seed{args.seed}"
    if any(not 0.0 <= value <= 1.0 for value in args.threshold_grid):
        parser.error("threshold grid values must be in [0,1]")
    if args.protocol_mode == "formal_nested_oof" and args.oof_manifest is None and not args.dry_run:
        parser.error("formal_nested_oof requires --oof-manifest")
    if args.protocol_mode != "formal_nested_oof" and args.oof_manifest is not None:
        parser.error("--oof-manifest is only valid in formal_nested_oof mode")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(json_safe({
            "target_fold": args.target_fold,
            "protocol_mode": args.protocol_mode,
            "oof_manifest": args.oof_manifest,
            "required_inputs_complete": (
                args.protocol_mode != "formal_nested_oof" or args.oof_manifest is not None
            ),
            "manuscript_pass_possible": (
                args.protocol_mode == "formal_nested_oof" and args.oof_manifest is not None
            ),
            "exploratory_only": args.protocol_mode != "formal_nested_oof",
            "cache_root": args.cache_root,
            "runs_root": args.runs_root,
            "split_csv": args.split_csv,
            "outdir": args.outdir,
            "formal_training_started": False,
        }), indent=2, allow_nan=False))
        return 0
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    stage = args.outdir.with_name(f".{args.outdir.name}.stage-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    started = time.time()
    try:
        result = run_protocol(args)
        rows = result.pop("per_sample_rows")
        atomic_json(stage / "config.json", vars(args))
        atomic_text(stage / "command.txt", " ".join(shlex.quote(value) for value in sys.argv) + "\n")
        atomic_json(stage / "result.json", result)
        atomic_json(stage / "checkpoint.json", result["frozen_receipt"])
        atomic_csv(stage / "per_sample.csv", rows)
        hashes = {
            name: sha256_file(stage / name)
            for name in ("config.json", "command.txt", "result.json", "checkpoint.json", "per_sample.csv")
        }
        atomic_json(stage / "hashes.json", hashes)
        atomic_json(stage / "DONE.json", {
            "schema_version": "sen12_proposal_utility_gate_done.v3",
            "status": "complete", "target_fold": args.target_fold, "seed": args.seed,
            "protocol_mode": args.protocol_mode,
            "evidence_status": result["evidence_status"],
            "manuscript_pass": result["manuscript_pass"],
            "elapsed_seconds": time.time() - started,
            "hashes_sha256": sha256_file(stage / "hashes.json"),
        })
        os.replace(stage, args.outdir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"status": "complete", "outdir": str(args.outdir)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
