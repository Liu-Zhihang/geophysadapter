#!/usr/bin/env python3
"""Aggregate strict nested proposer runs into formal utility-gate manifests.

This adapter does not train or evaluate a gate.  It verifies all 5 x 3 nested
proposer tasks first, enriches their inner-test predictions with the existing
role-aware Material/Trigger contexts, and emits the exact receipt/cache schema
consumed by ``train_sen12_proposal_utility_gate_v3.py``.

No non-target outer-test cache is ever referenced as gate-training evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import h5py


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_sen12_prithvi_roleaware_hierarchical_v2 as roleaware  # noqa: E402
import train_sen12_proposal_utility_gate_v3 as gate  # noqa: E402


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "experiments/revision2026/sen12_nested_oof_proposer_cache_v1"
)
DEFAULT_PROTOCOL_ROOT = (
    PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_nested_oof_protocol_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments/revision2026/sen12_proposal_utility_gate_v3/formal_inputs_v1"
)
DEFAULT_MATERIAL = roleaware.DEFAULT_MATERIAL
DEFAULT_MATERIAL_SCHEMA = roleaware.DEFAULT_MATERIAL_SCHEMA
DEFAULT_TRIGGER = roleaware.DEFAULT_TRIGGER
PROTOCOL_MANIFEST_NAME = "sen12_nested_oof_protocol_v1_manifest.json"
TARGETS = tuple(range(5))
INNER_FOLDS = tuple(range(3))
EXPECTED_TASKS = len(TARGETS) * len(INNER_FOLDS)
PRODUCER_RUN_SCHEMA = "sen12-nested-oof-proposer-run-v1"
PRODUCER_PROTOCOL_SCHEMA = "sen12-nested-oof-protocol-v1"
PRODUCER_CACHE_SCHEMA = 1
AGGREGATE_SCHEMA = "sen12-proposal-utility-gate-formal-inputs-v1"
DONE_SCHEMA = "sen12-proposal-utility-gate-formal-inputs-done-v1"
PRODUCER_TENSOR_KEYS = (
    "visual_logits",
    "terrain_logits",
    "terrain_direction",
    "frozen_vt_correction",
    "q_t",
    "mask",
    "valid",
)
PRODUCER_METADATA_KEYS = (
    "sample_ids",
    "physical_event_ids",
    "spatial_supergroups",
    "region_groups",
    "component_ids",
    "event_ids",
    "source_ids",
    "dataset_source_ids",
)
DONE_REQUIRED_ARTIFACTS = {
    "config.json",
    "split_audit.json",
    "checkpoints/visual_proposer.pt",
    "checkpoints/terrain_proposer.pt",
    "cache/inner_test_proposer_cache.pt",
    "run_manifest.json",
}


class AggregateError(RuntimeError):
    """Fail-closed input, identity, freshness, or schema error."""


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_canonical_hash(value: Any) -> str:
    """Match build_sen12_nested_oof_protocol_v1 canonical UTF-8 hashing."""
    encoded = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_sample_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def producer_value_hash(values: Iterable[str]) -> str:
    return canonical_hash(sorted({str(value) for value in values}))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AggregateError(f"refusing to write empty CSV: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [{str(key): str(value) for key, value in row.items()} for row in reader]


def read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise AggregateError(f"missing {name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AggregateError(f"{name} must be a JSON object: {path}")
    return value


def validate_signature(
    signature: Mapping[str, Any], expected_path: Path, name: str
) -> None:
    if not expected_path.is_file():
        raise AggregateError(f"missing {name}: {expected_path}")
    observed = expected_path.stat()
    expected_resolved = Path(str(signature.get("path", ""))).resolve()
    if expected_resolved != expected_path.resolve():
        raise AggregateError(f"{name} path identity mismatch")
    if int(signature.get("size", -1)) != observed.st_size:
        raise AggregateError(f"{name} size mismatch")
    if int(signature.get("mtime_ns", -1)) != observed.st_mtime_ns:
        raise AggregateError(f"{name} mtime stale or mismatched")
    if signature.get("sha256") != sha256_file(expected_path):
        raise AggregateError(f"{name} hash mismatch")


def validate_payload_hash(
    value: Mapping[str, Any],
    field: str,
    name: str,
    *,
    hash_fn: Any = canonical_hash,
) -> None:
    expected = value.get(field)
    payload = {key: item for key, item in value.items() if key != field}
    if not isinstance(expected, str) or hash_fn(payload) != expected:
        raise AggregateError(f"{name} canonical payload hash mismatch")


def task_dir(input_root: Path, target: int, inner: int, seed: int) -> Path:
    return input_root / f"target_outer{target}" / f"inner_fold{inner}" / f"seed{seed}"


def preflight_all_tasks(input_root: Path, seed: int) -> list[Path]:
    directories: list[Path] = []
    missing: list[str] = []
    for target in TARGETS:
        for inner in INNER_FOLDS:
            directory = task_dir(input_root, target, inner, seed)
            directories.append(directory)
            for relative in (
                "run_manifest.json",
                "cache/inner_test_proposer_cache.pt",
                "DONE.json",
            ):
                if not (directory / relative).is_file():
                    missing.append(f"target={target} inner={inner}: {relative}")
    if missing:
        raise AggregateError(
            f"15/15 nested proposer tasks required; missing {len(missing)} artifacts: "
            + "; ".join(missing[:15])
        )
    expected = {path.resolve() for path in directories}
    discovered = {
        path.parent.resolve()
        for path in input_root.glob(f"target_outer*/inner_fold*/seed{seed}/DONE.json")
    }
    unexpected = sorted(map(str, discovered - expected))
    if unexpected:
        raise AggregateError(f"unexpected target/inner tasks for seed {seed}: {unexpected[:10]}")
    return directories


def _decode_strings(values: Iterable[Any]) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def protocol_inputs(
    protocol_root: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], Path, set[str]]:
    manifest_path = protocol_root / PROTOCOL_MANIFEST_NAME
    manifest = read_json(manifest_path, "protocol manifest")
    if manifest.get("schema_version") != PRODUCER_PROTOCOL_SCHEMA:
        raise AggregateError("unexpected nested protocol manifest schema")
    validate_payload_hash(
        manifest,
        "manifest_payload_sha256",
        "protocol manifest",
        hash_fn=protocol_canonical_hash,
    )
    if not manifest.get("all_targets_all_audits_pass"):
        raise AggregateError("nested protocol manifest is not globally PASS")
    targets = {int(item["target_outer_fold"]): item for item in manifest.get("targets", [])}
    if set(targets) != set(TARGETS):
        raise AggregateError("protocol manifest must contain exactly targets 0..4")
    source_split = Path(str(manifest.get("inputs", {}).get("split_csv", ""))).resolve()
    if not source_split.is_file():
        raise AggregateError("protocol source outer split is missing")
    if sha256_file(source_split) != manifest["inputs"].get("split_csv_sha256"):
        raise AggregateError("protocol source outer split hash mismatch")
    h5_path = Path(str(manifest.get("inputs", {}).get("h5_path", ""))).resolve()
    if not h5_path.is_file():
        raise AggregateError("protocol frozen H5 is missing")
    if sha256_file(h5_path) != manifest["inputs"].get("h5_sha256"):
        raise AggregateError("protocol frozen H5 hash mismatch")
    with h5py.File(h5_path, "r") as handle:
        if "sample_id" not in handle:
            raise AggregateError("protocol frozen H5 lacks sample_id")
        h5_sample_ids = _decode_strings(handle["sample_id"][:])
    if len(h5_sample_ids) != len(set(h5_sample_ids)):
        raise AggregateError("protocol frozen H5 sample IDs are duplicated")
    for target, item in targets.items():
        split_path = protocol_root / f"sen12_nested_oof_target_outer{target}_v1.csv"
        if Path(str(item.get("output_csv", ""))).resolve() != split_path.resolve():
            raise AggregateError(f"target {target} protocol split path mismatch")
        if sha256_file(split_path) != item.get("output_csv_sha256"):
            raise AggregateError(f"target {target} protocol split hash mismatch")
        inner = {int(value["inner_fold"]): value for value in item.get("inner_folds", [])}
        if set(inner) != set(INNER_FOLDS):
            raise AggregateError(f"target {target} must contain inner folds 0..2")
    return manifest, targets, source_split, set(h5_sample_ids)


def validate_done_and_freshness(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    done_path = directory / "DONE.json"
    manifest_path = directory / "run_manifest.json"
    done = read_json(done_path, "producer DONE")
    manifest = read_json(manifest_path, "producer run manifest")
    if done.get("status") != "complete" or manifest.get("status") != "complete":
        raise AggregateError(f"producer task is not complete: {directory}")
    artifacts = done.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not DONE_REQUIRED_ARTIFACTS <= set(artifacts):
        raise AggregateError(f"producer DONE artifact set is incomplete: {directory}")
    latest_artifact_mtime = 0
    for relative, expected in artifacts.items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AggregateError(f"producer DONE hash mismatch: {directory}/{relative}")
        latest_artifact_mtime = max(latest_artifact_mtime, path.stat().st_mtime_ns)
    if done_path.stat().st_mtime_ns < latest_artifact_mtime:
        raise AggregateError(f"producer DONE is stale: {directory}")
    validate_payload_hash(manifest, "manifest_payload_sha256", "producer run manifest")
    return done, manifest


def role_detail_from_rows(rows: Sequence[Mapping[str, str]], role: str) -> dict[str, Any]:
    selected = [row for row in rows if row["role"] == role]
    sample_ids = sorted(row["sample_id"] for row in selected)
    spatial = sorted({row["spatial_supergroup"] for row in selected})
    regions = sorted({row["region_group"] for row in selected})
    events = sorted({row["physical_event_id"] for row in selected})
    components = sorted({row["nested_component_id"] for row in selected})
    return {
        "n_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "spatial_supergroups": spatial,
        "region_groups": regions,
        "physical_event_ids": events,
        "component_ids": components,
        "sample_sha256": producer_value_hash(sample_ids),
        "spatial_supergroup_sha256": producer_value_hash(spatial),
        "physical_event_sha256": producer_value_hash(events),
        "component_sha256": producer_value_hash(components),
    }


def validate_task(
    directory: Path,
    *,
    target: int,
    inner: int,
    seed: int,
    protocol_manifest_path: Path,
    protocol_target: Mapping[str, Any],
    protocol_split: Path,
) -> dict[str, Any]:
    _done, manifest = validate_done_and_freshness(directory)
    identity = (
        manifest.get("schema_version"),
        int(manifest.get("target_outer_fold", -1)),
        int(manifest.get("inner_fold", -1)),
        int(manifest.get("seed", -1)),
    )
    if identity != (PRODUCER_RUN_SCHEMA, target, inner, seed):
        raise AggregateError(f"producer run identity mismatch: target={target} inner={inner}")
    split_audit = manifest.get("split_audit")
    if not isinstance(split_audit, dict):
        raise AggregateError("producer split audit is missing")
    if (
        split_audit.get("status") != "PASS"
        or not split_audit.get("zero_target_outer_leakage")
        or not split_audit.get("zero_inner_role_sample_region_event_component_leakage")
        or any(split_audit.get("leakage", {}).values())
    ):
        raise AggregateError(f"producer split audit is not leak-free: target={target} inner={inner}")
    if (int(split_audit.get("target_outer_fold", -1)), int(split_audit.get("inner_fold", -1))) != (target, inner):
        raise AggregateError("producer split audit target/inner mismatch")
    validate_signature(split_audit["protocol_manifest"], protocol_manifest_path, "protocol manifest")
    validate_signature(split_audit["split_csv"], protocol_split, "nested split CSV")
    split_audit_path = directory / "split_audit.json"
    stored_audit = read_json(split_audit_path, "split audit artifact")
    if stored_audit != split_audit:
        raise AggregateError("run manifest split audit differs from split_audit.json")
    validate_payload_hash(split_audit, "audit_sha256", "producer split audit")

    rows = [row for row in read_csv(protocol_split) if int(row["outer_fold"]) == inner]
    expected_inner = {
        int(item["inner_fold"]): item for item in protocol_target["inner_folds"]
    }[inner]
    role_details: dict[str, dict[str, Any]] = {}
    for role in ("train", "val", "test"):
        detail = role_detail_from_rows(rows, role)
        role_details[role] = detail
        for key in (
            "n_samples",
            "sample_sha256",
            "spatial_supergroup_sha256",
            "physical_event_sha256",
            "component_sha256",
        ):
            if detail[key] != expected_inner["roles"][role][key]:
                raise AggregateError(f"protocol role mismatch: target={target} inner={inner} {role}.{key}")
            if detail[key] != split_audit["roles"][role][key]:
                raise AggregateError(f"run role mismatch: target={target} inner={inner} {role}.{key}")

    cache_path = directory / "cache/inner_test_proposer_cache.pt"
    visual_path = directory / "checkpoints/visual_proposer.pt"
    terrain_path = directory / "checkpoints/terrain_proposer.pt"
    validate_signature(manifest["cache"], cache_path, "producer cache")
    validate_signature(manifest["checkpoints"]["visual"], visual_path, "visual proposer")
    validate_signature(manifest["checkpoints"]["terrain"], terrain_path, "terrain proposer")
    manifest_mtime = (directory / "run_manifest.json").stat().st_mtime_ns
    dependencies = (
        cache_path,
        visual_path,
        terrain_path,
        directory / "config.json",
        split_audit_path,
    )
    if manifest_mtime < max(path.stat().st_mtime_ns for path in dependencies):
        raise AggregateError(f"producer run manifest is stale: target={target} inner={inner}")

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_identity = payload.get("identity", {})
    observed_cache_identity = (
        int(cache_identity.get("cache_schema_version", -1)),
        int(cache_identity.get("target_outer_fold", -1)),
        int(cache_identity.get("inner_fold", -1)),
        int(cache_identity.get("seed", -1)),
        cache_identity.get("export_role"),
    )
    if observed_cache_identity != (
        PRODUCER_CACHE_SCHEMA,
        target,
        inner,
        seed,
        "inner_test_post_selection_only",
    ):
        raise AggregateError(f"producer cache identity mismatch: target={target} inner={inner}")
    if cache_identity.get("split_audit_sha256") != split_audit["audit_sha256"]:
        raise AggregateError("producer cache split audit binding mismatch")
    if cache_identity.get("split_csv_sha256") != sha256_file(protocol_split):
        raise AggregateError("producer cache split CSV binding mismatch")
    if cache_identity.get("visual_checkpoint_sha256") != sha256_file(visual_path):
        raise AggregateError("producer cache visual checkpoint binding mismatch")
    if cache_identity.get("terrain_checkpoint_sha256") != sha256_file(terrain_path):
        raise AggregateError("producer cache terrain checkpoint binding mismatch")
    if manifest.get("selection", {}).get("target_outer_test_used_anywhere") is not False:
        raise AggregateError("producer selection does not prove untouched target outer-test")
    if manifest.get("selection", {}).get("inner_test_used_for_selection") is not False:
        raise AggregateError("producer selection used inner-test")
    if manifest.get("cache_schema", {}).get("paired_same_checkpoint") is not True:
        raise AggregateError("producer cache is not paired from fixed checkpoints")

    required = {"identity", *PRODUCER_TENSOR_KEYS, *PRODUCER_METADATA_KEYS}
    missing = sorted(required - set(payload))
    if missing:
        raise AggregateError(f"producer cache missing fields: {missing}")
    sample_ids = tuple(map(str, payload["sample_ids"]))
    if sample_ids != tuple(role_details["test"]["sample_ids"]):
        raise AggregateError("producer cache sample order/identity differs from inner-test")
    if cache_identity.get("sample_sha256") != role_details["test"]["sample_sha256"]:
        raise AggregateError("producer cache sample hash mismatch")
    if tuple(map(str, payload["event_ids"])) != tuple(map(str, payload["physical_event_ids"])):
        raise AggregateError("producer cache event alias mismatch")
    if tuple(map(str, payload["source_ids"])) != tuple(map(str, payload["spatial_supergroups"])):
        raise AggregateError("producer cache spatial alias mismatch")
    expected_shape = tuple(payload["visual_logits"].shape)
    if len(expected_shape) != 4 or expected_shape[0] != len(sample_ids) or expected_shape[1] != 1:
        raise AggregateError("producer cache has invalid visual tensor shape")
    if any(tuple(payload[key].shape) != expected_shape for key in PRODUCER_TENSOR_KEYS[1:]):
        raise AggregateError("producer cache tensors do not share shape")
    if payload["mask"].dtype != torch.uint8 or payload["valid"].dtype != torch.uint8:
        raise AggregateError("producer cache mask/valid dtype mismatch")
    if not torch.allclose(
        payload["frozen_vt_correction"].float(),
        payload["terrain_direction"].float() * float(cache_identity["routing"]["alpha"]),
        atol=5e-3,
        rtol=0.0,
    ):
        raise AggregateError("producer frozen Terrain correction contract mismatch")
    return {
        "target": target,
        "inner": inner,
        "directory": directory,
        "manifest": manifest,
        "payload": payload,
        "rows": rows,
        "roles": role_details,
        "cache_path": cache_path,
        "visual_path": visual_path,
        "terrain_path": terrain_path,
    }


def build_context_source(
    material_registry: Path,
    material_schema: Path,
    trigger_registry: Path,
) -> dict[str, Any]:
    material = pd.read_csv(material_registry)
    trigger = pd.read_csv(trigger_registry)
    material_feature_names = roleaware.load_material_feature_names(material_schema)
    if "sample_id" not in material or "sample_id" not in trigger:
        raise AggregateError("M/R registries lack sample_id")
    material_ids = tuple(material["sample_id"].astype(str))
    trigger_ids = tuple(trigger["sample_id"].astype(str))
    if len(material_ids) != len(set(material_ids)) or len(trigger_ids) != len(set(trigger_ids)):
        raise AggregateError("M/R registries contain duplicate sample identities")
    if set(material_ids) != set(trigger_ids):
        raise AggregateError("M/R registry sample identities differ")
    trigger = trigger.assign(sample_id=trigger["sample_id"].astype(str)).set_index("sample_id")
    material = material.assign(sample_id=material["sample_id"].astype(str)).set_index("sample_id")
    all_ids = trigger_ids
    events = tuple(trigger.loc[list(all_ids), "physical_event_id"].astype(str))
    source_by_id = {
        sample_id: str(material.loc[sample_id, "region_group"])
        for sample_id in all_ids
    }
    return {
        "all_ids": all_ids,
        "events": events,
        "source_by_id": source_by_id,
        "material_sha256": sha256_file(material_registry),
        "material_schema_sha256": sha256_file(material_schema),
        "material_feature_names": material_feature_names,
        "trigger_sha256": sha256_file(trigger_registry),
    }


def create_gate_split(
    target: int,
    protocol_split: Path,
    source_split: Path,
    h5_sample_ids: set[str],
    protocol_target: Mapping[str, Any],
    output_path: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    nested_rows = read_csv(protocol_split)
    development: dict[str, dict[str, str]] = {}
    for row in nested_rows:
        sample_id = row["sample_id"]
        current = development.setdefault(sample_id, row)
        for key in ("spatial_supergroup", "physical_event_id"):
            if current[key] != row[key]:
                raise AggregateError(f"nested protocol identity varies for {sample_id}")
    source_rows = [
        row for row in read_csv(source_split)
        if int(row["outer_fold"]) == target
        and row["role"] == "test"
        and row["sample_id"] in h5_sample_ids
    ]
    if not source_rows:
        raise AggregateError(f"target {target} source split has no outer-test")
    if set(development) & {row["sample_id"] for row in source_rows}:
        raise AggregateError(f"target {target} development overlaps outer-test")
    target_ids = sorted(row["sample_id"] for row in source_rows)
    target_receipt = protocol_target["target_outer_test"]
    if (
        len(target_ids) != int(target_receipt["n_h5_samples"])
        or producer_value_hash(target_ids) != target_receipt["h5_sample_sha256"]
    ):
        raise AggregateError(f"target {target} H5-filtered outer-test identity mismatch")
    output: list[dict[str, str]] = []
    identity: dict[str, dict[str, str]] = {}
    for sample_id, row in sorted(development.items()):
        item = {
            "sample_id": sample_id,
            "outer_fold": str(target),
            "role": "train",
            "region_group": row["spatial_supergroup"],
            "physical_event_id": row["physical_event_id"],
            "source_role": row["source_outer_role"],
        }
        output.append(item)
        identity[sample_id] = item
    for row in sorted(source_rows, key=lambda item: item["sample_id"]):
        item = {
            "sample_id": row["sample_id"],
            "outer_fold": str(target),
            "role": "test",
            "region_group": row["spatial_supergroup"],
            "physical_event_id": "TARGET_OUTER_TEST_EVENT_WITHHELD_UNTIL_EVALUATION",
            "source_role": "test",
        }
        output.append(item)
        identity[item["sample_id"]] = item
    atomic_csv(output_path, output)
    return output, identity


def context_for_task(
    context_source: Mapping[str, Any],
    material_registry: Path,
    trigger_registry: Path,
    train_ids: Sequence[str],
    holdout_ids: Sequence[str],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = sorted((set(train_ids) | set(holdout_ids)) - set(context_source["all_ids"]))
    if missing:
        raise AggregateError(f"nested proposer samples absent from M/R registries: {missing[:10]}")
    context = roleaware.RoleContext(
        material_registry,
        trigger_registry,
        context_source["all_ids"],
        context_source["events"],
        context_source["source_by_id"],
        train_ids,
        train_ids,
        train_ids,
        seed,
        material_feature_names=context_source["material_feature_names"],
    )
    arrays = context.arrays(holdout_ids)
    audit = context.audit(train_ids, train_ids)
    audit.update({
        "normalization_scope": "current inner proposer-train only",
        "holdout_labels_used_for_context": False,
        "material_registry_sha256": context_source["material_sha256"],
        "material_schema_sha256": context_source["material_schema_sha256"],
        "trigger_registry_sha256": context_source["trigger_sha256"],
    })
    return arrays, audit


def validate_context_arrays(arrays: Mapping[str, Any], n_samples: int) -> None:
    required = {
        "material", "q_material", "material_shuffle", "q_material_shuffle",
        "trigger", "trigger_wrong", "q_trigger", "trigger_shuffle",
        "q_trigger_shuffle",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise AggregateError(f"role-aware context misses gate fields: {missing}")
    for key in required:
        value = arrays[key]
        if not torch.is_tensor(value) or value.shape[0] != n_samples:
            raise AggregateError(f"role-aware context shape mismatch: {key}")
    if arrays["material"].ndim != 2 or arrays["material_shuffle"].shape != arrays["material"].shape:
        raise AggregateError("Material aligned/shuffle shape mismatch")
    if arrays["trigger"].ndim != 2 or arrays["trigger"].shape[1] != 3:
        raise AggregateError("Trigger context must be [N,3]")
    if arrays["trigger_wrong"].shape != arrays["trigger"].shape:
        raise AggregateError("Trigger wrong-time shape mismatch")
    if arrays["trigger_shuffle"].shape != arrays["trigger"].shape:
        raise AggregateError("Trigger event-shuffle shape mismatch")
    for key in ("q_material", "q_material_shuffle", "q_trigger", "q_trigger_shuffle"):
        value = arrays[key].float()
        if value.ndim != 1 or not torch.isfinite(value).all():
            raise AggregateError(f"invalid quality vector: {key}")


def build_formal_target(
    stage: Path,
    final_root: Path,
    target: int,
    seed: int,
    tasks: Sequence[Mapping[str, Any]],
    protocol_split: Path,
    source_split: Path,
    h5_sample_ids: set[str],
    protocol_target: Mapping[str, Any],
    material_registry: Path,
    trigger_registry: Path,
    context_source: Mapping[str, Any],
) -> dict[str, Any]:
    target_stage = stage / f"target_outer{target}"
    gate_split = target_stage / "gate_split.csv"
    _rows, gate_identity = create_gate_split(
        target,
        protocol_split,
        source_split,
        h5_sample_ids,
        protocol_target,
        gate_split,
    )
    gate_split_sha = sha256_file(gate_split)
    outer_train = {sample_id for sample_id, row in gate_identity.items() if row["role"] == "train"}
    holdout_union: set[str] = set()
    entries: list[dict[str, Any]] = []
    task_receipts: list[dict[str, Any]] = []

    for task in sorted(tasks, key=lambda item: int(item["inner"])):
        inner = int(task["inner"])
        payload = task["payload"]
        manifest = task["manifest"]
        train_ids = tuple(task["roles"]["train"]["sample_ids"])
        holdout_ids = tuple(map(str, payload["sample_ids"]))
        if holdout_union & set(holdout_ids):
            raise AggregateError(f"target {target} inner holdout overlap")
        holdout_union |= set(holdout_ids)
        train_events = {
            row["sample_id"]: row["physical_event_id"]
            for row in task["rows"] if row["role"] == "train"
        }
        holdout_events = dict(zip(holdout_ids, map(str, payload["event_ids"])))
        train_regions = sorted({gate_identity[sample_id]["region_group"] for sample_id in train_ids})
        holdout_regions = sorted({gate_identity[sample_id]["region_group"] for sample_id in holdout_ids})
        if set(train_regions) & set(holdout_regions):
            raise AggregateError(f"target {target} inner {inner} region leakage")
        if set(train_events.values()) & set(holdout_events.values()):
            raise AggregateError(f"target {target} inner {inner} event leakage")

        context_arrays, context_audit = context_for_task(
            context_source,
            material_registry,
            trigger_registry,
            train_ids,
            holdout_ids,
            seed + 1009 * target + inner,
        )
        validate_context_arrays(context_arrays, len(holdout_ids))
        receipt = {
            "schema_version": gate.FORMAL_RECEIPT_SCHEMA,
            "target_outer_fold": target,
            "inner_fold": inner,
            "seed": seed,
            "split_csv_sha256": gate_split_sha,
            "proposer_train_sample_ids": list(train_ids),
            "proposer_train_sample_sha256": gate_sample_hash(sorted(train_ids)),
            "inner_holdout_sample_ids": list(holdout_ids),
            "inner_holdout_sample_sha256": gate_sample_hash(sorted(holdout_ids)),
            "proposer_train_regions": train_regions,
            "inner_holdout_regions": holdout_regions,
            "proposer_train_sample_event_ids": train_events,
            "inner_holdout_sample_event_ids": holdout_events,
            "proposer_train_events": sorted(set(train_events.values())),
            "inner_holdout_events": sorted(set(holdout_events.values())),
            "visual_checkpoint_sha256": manifest["checkpoints"]["visual"]["sha256"],
            "terrain_checkpoint_sha256": manifest["checkpoints"]["terrain"]["sha256"],
            "visual_threshold": float(manifest["selection"]["visual"]["threshold"]),
            "producer_run_manifest_sha256": sha256_file(task["directory"] / "run_manifest.json"),
            "producer_cache_sha256": sha256_file(task["cache_path"]),
            "producer_cache_tensor_contract": (
                "visual_logits/frozen_vt_correction/mask/valid copied without numerical transform"
            ),
            "context_audit": context_audit,
        }
        receipt_path = target_stage / "receipts" / f"inner_fold{inner}_producer_receipt.json"
        atomic_json(receipt_path, receipt)
        receipt_sha = sha256_file(receipt_path)
        formal_identity = {
            "schema_version": gate.FORMAL_CACHE_SCHEMA,
            "split": "nested_inner_holdout",
            "target_outer_fold": target,
            "inner_fold": inner,
            "seed": seed,
            "sample_sha256": gate_sample_hash(holdout_ids),
            "holdout_regions": holdout_regions,
            "holdout_events": sorted(set(holdout_events.values())),
            "producer_receipt_sha256": receipt_sha,
            "visual_checkpoint_sha256": receipt["visual_checkpoint_sha256"],
            "terrain_checkpoint_sha256": receipt["terrain_checkpoint_sha256"],
            "visual_threshold": receipt["visual_threshold"],
            "producer_cache_identity": deepcopy(payload["identity"]),
        }
        formal_payload = {
            "identity": formal_identity,
            "sample_ids": list(holdout_ids),
            "event_ids": list(map(str, payload["event_ids"])),
            "source_ids": list(map(str, payload["source_ids"])),
            "visual_logits": payload["visual_logits"],
            "frozen_vt_correction": payload["frozen_vt_correction"],
            "valid": payload["valid"],
            "mask": payload["mask"],
            **context_arrays,
        }
        cache_path = target_stage / "cache" / f"inner_fold{inner}_formal_gate_cache.pt"
        atomic_torch_save(cache_path, formal_payload)
        if not torch.equal(formal_payload["visual_logits"], payload["visual_logits"]):
            raise AggregateError("visual logits changed during formal cache adaptation")
        if not torch.equal(formal_payload["frozen_vt_correction"], payload["frozen_vt_correction"]):
            raise AggregateError("Terrain correction changed during formal cache adaptation")
        entry = {
            "inner_fold": inner,
            "cache_path": str(cache_path.relative_to(target_stage)),
            "cache_sha256": sha256_file(cache_path),
            "producer_receipt_path": str(receipt_path.relative_to(target_stage)),
            "producer_receipt_sha256": receipt_sha,
            "holdout_sample_sha256": gate_sample_hash(sorted(holdout_ids)),
            "holdout_regions": holdout_regions,
            "holdout_events": sorted(set(holdout_events.values())),
            "visual_checkpoint_sha256": receipt["visual_checkpoint_sha256"],
            "terrain_checkpoint_sha256": receipt["terrain_checkpoint_sha256"],
            "visual_threshold": receipt["visual_threshold"],
        }
        entries.append(entry)
        task_receipts.append({
            "target_outer_fold": target,
            "inner_fold": inner,
            "producer_run": str(task["directory"].resolve()),
            "producer_run_manifest_sha256": receipt["producer_run_manifest_sha256"],
            "formal_cache_sha256": entry["cache_sha256"],
            "formal_receipt_sha256": receipt_sha,
            "n_holdout_samples": len(holdout_ids),
        })

    if holdout_union != outer_train:
        raise AggregateError(
            f"target {target} inner holdouts do not exactly cover outer-development: "
            f"missing={len(outer_train-holdout_union)} extra={len(holdout_union-outer_train)}"
        )
    oof_manifest = {
        "schema_version": gate.FORMAL_MANIFEST_SCHEMA,
        "target_outer_fold": target,
        "seed": seed,
        "split_csv_sha256": gate_split_sha,
        "entries": entries,
    }
    oof_path = target_stage / "oof_manifest.json"
    atomic_json(oof_path, oof_manifest)
    access_log: list[dict[str, Any]] = []
    bundles, audit = gate.load_formal_nested_bundles(
        oof_path,
        target_fold=target,
        split_csv=gate_split,
        seed=seed,
        access_log=access_log,
    )
    if len(bundles) != len(INNER_FOLDS) or len(access_log) != len(INNER_FOLDS):
        raise AggregateError(f"target {target} gate validator did not consume exactly 3 inner OOF bundles")
    return {
        "target_outer_fold": target,
        "gate_split": str((final_root / f"target_outer{target}/gate_split.csv").resolve()),
        "gate_split_sha256": gate_split_sha,
        "oof_manifest": str((final_root / f"target_outer{target}/oof_manifest.json").resolve()),
        "oof_manifest_sha256": sha256_file(oof_path),
        "n_inner_folds": len(entries),
        "n_outer_development_samples": len(outer_train),
        "gate_validator_audit_sha256": canonical_hash(audit),
        "tasks": task_receipts,
    }


def collect_output_hashes(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "DONE.json":
            continue
        output[relative] = sha256_file(path)
    return output


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    preflight_all_tasks(args.input_root, args.seed)
    protocol_manifest, protocol_targets, source_split, h5_sample_ids = protocol_inputs(
        args.protocol_root
    )
    protocol_manifest_path = args.protocol_root / PROTOCOL_MANIFEST_NAME
    validated: dict[int, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    for target in TARGETS:
        protocol_split = args.protocol_root / f"sen12_nested_oof_target_outer{target}_v1.csv"
        for inner in INNER_FOLDS:
            validated[target].append(validate_task(
                task_dir(args.input_root, target, inner, args.seed),
                target=target,
                inner=inner,
                seed=args.seed,
                protocol_manifest_path=protocol_manifest_path,
                protocol_target=protocol_targets[target],
                protocol_split=protocol_split,
            ))
    if sum(map(len, validated.values())) != EXPECTED_TASKS:
        raise AggregateError("validated task count is not 15/15")
    context_source = build_context_source(
        args.material_registry,
        args.material_schema,
        args.trigger_registry,
    )
    # Full prepublication preflight: context normalization and all negative
    # controls must materialize for all 15 tasks before any output directory is created.
    for target in TARGETS:
        for task in validated[target]:
            train_ids = tuple(task["roles"]["train"]["sample_ids"])
            holdout_ids = tuple(map(str, task["payload"]["sample_ids"]))
            arrays, _audit = context_for_task(
                context_source,
                args.material_registry,
                args.trigger_registry,
                train_ids,
                holdout_ids,
                args.seed + 1009 * target + int(task["inner"]),
            )
            validate_context_arrays(arrays, len(holdout_ids))
    plan = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "validated",
        "seed": args.seed,
        "expected_tasks": EXPECTED_TASKS,
        "validated_tasks": EXPECTED_TASKS,
        "protocol_manifest_sha256": sha256_file(protocol_manifest_path),
        "protocol_payload_sha256": protocol_manifest["manifest_payload_sha256"],
        "material_registry_sha256": context_source["material_sha256"],
        "material_schema_sha256": context_source["material_schema_sha256"],
        "material_feature_names": list(context_source["material_feature_names"]),
        "trigger_registry_sha256": context_source["trigger_sha256"],
        "gate_validator_schemas": {
            "manifest": gate.FORMAL_MANIFEST_SCHEMA,
            "cache": gate.FORMAL_CACHE_SCHEMA,
            "receipt": gate.FORMAL_RECEIPT_SCHEMA,
        },
        "training_cache_contract": "target outer-fold nested inner OOF only",
        "other_outer_test_cache_referenced_for_gate_training": False,
    }
    if args.dry_run:
        return {**plan, "status": "dry_run_validated", "output_written": False}
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    stage = args.output_root.with_name(f".{args.output_root.name}.stage-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        targets = []
        for target in TARGETS:
            targets.append(build_formal_target(
                stage,
                args.output_root,
                target,
                args.seed,
                validated[target],
                args.protocol_root / f"sen12_nested_oof_target_outer{target}_v1.csv",
                source_split,
                h5_sample_ids,
                protocol_targets[target],
                args.material_registry,
                args.trigger_registry,
                context_source,
            ))
        summary = {**plan, "status": "complete", "targets": targets}
        summary["summary_payload_sha256"] = canonical_hash(summary)
        atomic_json(stage / "aggregate_summary.json", summary)
        hashes = collect_output_hashes(stage)
        atomic_json(stage / "hashes.json", hashes)
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "complete",
            "expected_tasks": EXPECTED_TASKS,
            "validated_tasks": EXPECTED_TASKS,
            "hashes_sha256": sha256_file(stage / "hashes.json"),
            "aggregate_summary_sha256": sha256_file(stage / "aggregate_summary.json"),
        }
        atomic_json(stage / "DONE.json", done)
        os.replace(stage, args.output_root)
        return {**summary, "output_written": True, "output_root": str(args.output_root)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--material-registry", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--material-schema", type=Path, default=DEFAULT_MATERIAL_SCHEMA)
    parser.add_argument("--trigger-registry", type=Path, default=DEFAULT_TRIGGER)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = aggregate(args)
    except Exception as error:
        print(json.dumps({
            "status": "failed_closed",
            "error_type": type(error).__name__,
            "error": str(error),
        }, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
