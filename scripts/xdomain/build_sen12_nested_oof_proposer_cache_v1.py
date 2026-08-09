#!/usr/bin/env python3
"""Train strict nested-OOF visual/Terrain proposers and export inner-test cache.

For one target-outer and inner fold, optimization reads labels from inner train,
checkpoint and threshold selection read labels from inner validation, and the
inner-test labels are materialized only after both checkpoints are frozen.
The target outer-test identities are exclusion-only and are never loaded into a
training, selection, or export dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_sen12_prithvi_terrain_v2 as terrain_trainer  # noqa: E402
import train_sen12_prithvi_tmr_modulator as frozen_protocol  # noqa: E402


PROTOCOL_ROOT = PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_nested_oof_protocol_v1"
PROTOCOL_MANIFEST = PROTOCOL_ROOT / "sen12_nested_oof_protocol_v1_manifest.json"
SOURCE_LOGO_CSV = PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv"
OUT_ROOT = PROJECT_ROOT / "experiments/revision2026/sen12_nested_oof_proposer_cache_v1"
CACHE_SCHEMA_VERSION = 1
ROUTING_CONFIG = {
    "low": float(frozen_protocol.ROUTING_LOW),
    "high": float(frozen_protocol.ROUTING_HIGH),
    "alpha": float(frozen_protocol.ROUTING_ALPHA),
    "visual_margin": float(frozen_protocol.ROUTING_MARGIN),
    "selection": "fixed_preregistered_not_searched_on_inner_test",
}


class ProposerProtocolError(RuntimeError):
    """Raised when a split or artifact violates the nested OOF contract."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_values(values: Iterable[str]) -> str:
    return canonical_hash(sorted({str(value) for value in values}))


def file_signature(path: Path, *, content_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        result["sha256"] = sha256_file(path)
    return result


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [{key: str(value) for key, value in row.items()} for row in reader]


def decode_h5(values: Iterable[Any]) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def h5_event_map(path: Path) -> dict[str, str]:
    with h5py.File(path, "r") as handle:
        sample_ids = decode_h5(handle["sample_id"][:])
        event_ids = decode_h5(handle["physical_event_id"][:])
    if len(sample_ids) != len(event_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ProposerProtocolError("H5 sample/event identity is not one-to-one")
    return dict(zip(sample_ids, event_ids))


def identity_details(
    sample_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, str]],
    event_by_sample: Mapping[str, str],
) -> dict[str, Any]:
    samples = sorted(map(str, sample_ids))
    regions = sorted({str(rows[sample]["spatial_supergroup"]) for sample in samples})
    region_groups = sorted({str(rows[sample]["region_group"]) for sample in samples})
    events = sorted({str(event_by_sample[sample]) for sample in samples})
    components = sorted({str(rows[sample]["nested_component_id"]) for sample in samples})
    return {
        "n_samples": len(samples),
        "n_spatial_supergroups": len(regions),
        "n_region_groups": len(region_groups),
        "n_physical_events": len(events),
        "n_components": len(components),
        "sample_ids": samples,
        "spatial_supergroups": regions,
        "region_groups": region_groups,
        "physical_event_ids": events,
        "component_ids": components,
        "sample_sha256": hash_values(samples),
        "spatial_supergroup_sha256": hash_values(regions),
        "region_group_sha256": hash_values(region_groups),
        "physical_event_sha256": hash_values(events),
        "component_sha256": hash_values(components),
    }


def assert_pairwise_disjoint(
    role_details: Mapping[str, Mapping[str, Any]], field: str
) -> None:
    values = {role: set(role_details[role][field]) for role in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = values[left] & values[right]
        if overlap:
            raise ProposerProtocolError(
                f"Nested {field} leakage between {left}/{right}: {sorted(overlap)[:10]}"
            )


def audit_nested_split(
    split_csv: Path,
    source_logo_csv: Path,
    protocol_manifest_path: Path,
    target_outer_fold: int,
    inner_fold: int,
    base_h5: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]], dict[str, Any]]:
    manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sen12-nested-oof-protocol-v1":
        raise ProposerProtocolError("Unexpected nested protocol manifest schema")
    targets = {int(item["target_outer_fold"]): item for item in manifest["targets"]}
    if target_outer_fold not in targets:
        raise ProposerProtocolError(f"Target outer fold {target_outer_fold} absent from manifest")
    target_manifest = targets[target_outer_fold]
    split_sha = sha256_file(split_csv)
    if split_sha != target_manifest["output_csv_sha256"]:
        raise ProposerProtocolError("Nested split CSV hash differs from protocol manifest")

    rows, roles, _ = terrain_trainer.protocol.load_logo_rows(split_csv, inner_fold)
    required = {"target_outer_fold", "nested_component_id", "physical_event_id", "source_id"}
    missing = required - set(next(iter(rows.values())))
    if missing:
        raise ProposerProtocolError(f"Nested split misses columns: {sorted(missing)}")
    if {row["target_outer_fold"] for row in rows.values()} != {str(target_outer_fold)}:
        raise ProposerProtocolError("Nested rows do not bind to requested target outer fold")

    event_by_sample = h5_event_map(base_h5)
    unknown = sorted(set(rows) - set(event_by_sample))
    if unknown:
        raise ProposerProtocolError(f"Nested split samples absent from H5: {unknown[:10]}")
    for sample_id, row in rows.items():
        if row["physical_event_id"] != event_by_sample[sample_id]:
            raise ProposerProtocolError(f"Nested event identity mismatch for {sample_id}")

    role_details = {
        role: identity_details(sample_ids, rows, event_by_sample)
        for role, sample_ids in roles.items()
    }
    for field in (
        "sample_ids",
        "spatial_supergroups",
        "region_groups",
        "physical_event_ids",
        "component_ids",
    ):
        assert_pairwise_disjoint(role_details, field)

    source_rows = [
        row
        for row in read_csv(source_logo_csv)
        if row["outer_fold"] == str(target_outer_fold) and row["role"] == "test"
    ]
    if not source_rows:
        raise ProposerProtocolError("Source outer target has no test identities")
    target_samples = {row["sample_id"] for row in source_rows}
    target_regions = {row["spatial_supergroup"] for row in source_rows}
    target_region_groups = {row["region_group"] for row in source_rows}
    target_events = {
        event_by_sample[sample_id]
        for sample_id in target_samples
        if sample_id in event_by_sample
    }
    nested_samples = set(rows)
    nested_regions = {row["spatial_supergroup"] for row in rows.values()}
    nested_region_groups = {row["region_group"] for row in rows.values()}
    nested_events = {event_by_sample[sample] for sample in rows}
    leakage = {
        "sample_ids": sorted(nested_samples & target_samples),
        "spatial_supergroups": sorted(nested_regions & target_regions),
        "region_groups": sorted(nested_region_groups & target_region_groups),
        "physical_event_ids": sorted(nested_events & target_events),
    }
    if any(leakage.values()):
        raise ProposerProtocolError(f"Target outer-test leakage into nested OOF: {leakage}")

    expected_inner = {
        int(item["inner_fold"]): item for item in target_manifest["inner_folds"]
    }[inner_fold]
    for role in ("train", "val", "test"):
        expected = expected_inner["roles"][role]
        for key in (
            "n_samples",
            "sample_sha256",
            "spatial_supergroup_sha256",
            "physical_event_sha256",
            "component_sha256",
        ):
            if role_details[role][key] != expected[key]:
                raise ProposerProtocolError(
                    f"Role receipt mismatch for {role}.{key}: "
                    f"{role_details[role][key]} != {expected[key]}"
                )

    audit = {
        "status": "PASS",
        "target_outer_fold": target_outer_fold,
        "inner_fold": inner_fold,
        "split_csv": file_signature(split_csv),
        "source_logo_csv": file_signature(source_logo_csv),
        "protocol_manifest": file_signature(protocol_manifest_path),
        "roles": role_details,
        "target_outer_exclusions": {
            "n_source_test_samples": len(target_samples),
            "n_h5_events": len(target_events),
            "sample_sha256": hash_values(target_samples),
            "spatial_supergroup_sha256": hash_values(target_regions),
            "region_group_sha256": hash_values(target_region_groups),
            "physical_event_sha256": hash_values(target_events),
        },
        "leakage": leakage,
        "zero_target_outer_leakage": True,
        "zero_inner_role_sample_region_event_component_leakage": True,
        "label_access_contract": {
            "inner_train": "optimization_and_train_only_normalization",
            "inner_val": "checkpoint_selection_and_visual_threshold_only",
            "inner_test": "post_selection_paired_cache_export_only",
            "target_outer_test": "never_materialized",
        },
    }
    audit["audit_sha256"] = canonical_hash(audit)
    return rows, roles, audit


def make_dataset(
    sample_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, str]],
    all_ids: Sequence[str],
    event_ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
    donor_pool: Sequence[str],
    include_terrain: bool,
) -> terrain_trainer.PrithviTerrainDataset:
    return terrain_trainer.PrithviTerrainDataset(
        terrain_trainer.BASE_H5,
        terrain_trainer.OPTICAL_H5,
        terrain_trainer.TERRAIN_H5,
        all_ids,
        event_ids,
        dict(rows),
        sample_ids,
        mean,
        std,
        seed,
        donor_pool,
        include_terrain,
    )


def make_loader(dataset, seed: int, batch_size: int, num_workers: int, shuffle: bool):
    return terrain_trainer.protocol.make_loader(
        dataset,
        SimpleNamespace(seed=seed, batch_size=batch_size, num_workers=num_workers),
        shuffle=shuffle,
    )


def train_visual(
    model: nn.Module,
    train_loader,
    val_loader,
    pos_weight: float,
    args: argparse.Namespace,
    log,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float, float, Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.visual_lr, weight_decay=args.weight_decay
    )
    amp = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    positive = torch.tensor([pos_weight], device=args.device).view(1, 1, 1, 1)
    best_ap = -1.0
    best_epoch = 0
    best_state = None
    history = []
    steps = 0
    for epoch in range(1, args.visual_epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            optical = batch["pre"].to(args.device, non_blocking=True)
            coordinates = batch["post"].to(args.device, non_blocking=True)
            target = batch["mask"].to(args.device, non_blocking=True)
            valid = batch["valid"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with terrain_trainer.protocol.autocast_context(amp):
                logits, _ = model(optical, coordinates)
                bce = F.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=positive, reduction="none"
                )
                bce = terrain_trainer.masked_mean(bce, valid)
                loss = bce + args.dice_weight * terrain_trainer.protocol.dice_loss_per_sample(
                    logits, target, valid
                ).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * optical.shape[0]
            seen += optical.shape[0]
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        val_ap, _ = terrain_trainer.protocol.score_validation_ap(
            model, val_loader, "visual", args.device, amp
        )
        row = {
            "epoch": epoch,
            "global_step": steps,
            "train_loss": loss_sum / max(seen, 1),
            "val_average_precision": float(val_ap),
        }
        history.append(row)
        log(f"[visual] {json.dumps(row, sort_keys=True)}")
        if val_ap > best_ap:
            best_ap = float(val_ap)
            best_epoch = epoch
            best_state = terrain_trainer.trainable_state(model)
        if args.max_steps and steps >= args.max_steps:
            break
    if best_state is None:
        raise RuntimeError("Visual proposer produced no checkpoint")
    terrain_trainer.load_trainable_state(model, best_state)
    selected_ap, histogram = terrain_trainer.protocol.score_validation_ap(
        model, val_loader, "visual", args.device, amp
    )
    threshold, threshold_metrics = terrain_trainer.protocol.choose_threshold(histogram)
    return best_state, history, best_epoch, float(selected_ap), float(threshold), threshold_metrics


@torch.no_grad()
def score_terrain(model: nn.Module, loader, device: str, amp: bool):
    histogram = terrain_trainer.protocol.ProbabilityHistogram()
    model.eval()
    for batch in loader:
        terrain = batch["terrain"].to(device, non_blocking=True)
        with terrain_trainer.protocol.autocast_context(amp):
            logits, _ = model(terrain)
        probability = torch.sigmoid(logits).float().cpu().numpy()
        target = batch["mask"].numpy()
        valid = batch["valid"].numpy() > 0.5
        histogram.update(probability[valid], target[valid])
    return histogram.average_precision, histogram


def train_terrain(
    model: nn.Module,
    train_loader,
    val_loader,
    pos_weight: float,
    args: argparse.Namespace,
    log,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.terrain_lr, weight_decay=args.weight_decay
    )
    amp = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    positive = torch.tensor([pos_weight], device=args.device).view(1, 1, 1, 1)
    best_ap = -1.0
    best_epoch = 0
    best_state = None
    history = []
    steps = 0
    for epoch in range(1, args.terrain_epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            terrain = batch["terrain"].to(args.device, non_blocking=True)
            target = batch["mask"].to(args.device, non_blocking=True)
            valid = batch["valid"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with terrain_trainer.protocol.autocast_context(amp):
                logits, _ = model(terrain)
                bce = F.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=positive, reduction="none"
                )
                bce = terrain_trainer.masked_mean(bce, valid)
                loss = bce + args.dice_weight * terrain_trainer.protocol.dice_loss_per_sample(
                    logits, target, valid
                ).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * terrain.shape[0]
            seen += terrain.shape[0]
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        val_ap, _ = score_terrain(model, val_loader, args.device, amp)
        row = {
            "epoch": epoch,
            "global_step": steps,
            "train_loss": loss_sum / max(seen, 1),
            "val_average_precision": float(val_ap),
        }
        history.append(row)
        log(f"[terrain] {json.dumps(row, sort_keys=True)}")
        if val_ap > best_ap:
            best_ap = float(val_ap)
            best_epoch = epoch
            best_state = terrain_trainer.trainable_state(model)
        if args.max_steps and steps >= args.max_steps:
            break
    if best_state is None:
        raise RuntimeError("Terrain proposer produced no checkpoint")
    terrain_trainer.load_trainable_state(model, best_state)
    selected_ap, _ = score_terrain(model, val_loader, args.device, amp)
    return best_state, history, best_epoch, float(selected_ap)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return terrain_trainer.tensor_dict_sha256(dict(state))


@torch.inference_mode()
def export_inner_test_cache(
    visual: nn.Module,
    terrain: nn.Module,
    dataset,
    loader,
    rows: Mapping[str, Mapping[str, str]],
    event_by_sample: Mapping[str, str],
    visual_threshold: float,
    identity: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    amp = bool(args.amp and str(args.device).startswith("cuda"))
    visual.eval()
    terrain.eval()
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    sample_ids: list[str] = []
    for batch in loader:
        optical = batch["pre"].to(args.device, non_blocking=True)
        coordinates = batch["post"].to(args.device, non_blocking=True)
        terrain_input = batch["terrain"].to(args.device, non_blocking=True)
        q_t = batch["q_t"].to(args.device, non_blocking=True)
        with terrain_trainer.protocol.autocast_context(amp):
            visual_logits, _ = visual(optical, coordinates)
            terrain_logits, _ = terrain(terrain_input)
        visual_logits = visual_logits.float()
        terrain_logits = terrain_logits.float()
        direction = frozen_protocol.frozen_terrain_direction(
            visual_logits, terrain_logits, q_t.float(), visual_threshold
        )
        correction = ROUTING_CONFIG["alpha"] * direction
        tensors["visual_logits"].append(visual_logits.cpu().half())
        tensors["terrain_logits"].append(terrain_logits.cpu().half())
        tensors["terrain_direction"].append(direction.cpu().half())
        tensors["frozen_vt_correction"].append(correction.cpu().half())
        tensors["q_t"].append(q_t.float().cpu().half())
        tensors["mask"].append((batch["mask"] >= 0.5).cpu().to(torch.uint8))
        tensors["valid"].append((batch["valid"] >= 0.5).cpu().to(torch.uint8))
        sample_ids.extend(map(str, batch["sample_id"]))
    if sample_ids != list(dataset.sample_ids):
        raise RuntimeError("Inner-test cache sample order changed")
    physical_event_ids = [event_by_sample[sample] for sample in sample_ids]
    spatial_supergroups = [rows[sample]["spatial_supergroup"] for sample in sample_ids]
    payload = {
        "identity": dict(identity),
        "sample_ids": sample_ids,
        "physical_event_ids": physical_event_ids,
        "spatial_supergroups": spatial_supergroups,
        "region_groups": [rows[sample]["region_group"] for sample in sample_ids],
        "component_ids": [rows[sample]["nested_component_id"] for sample in sample_ids],
        # Backward-compatible aliases used by the existing frozen outer-cache
        # consumers. Historically source_ids means spatial group, not dataset.
        "event_ids": physical_event_ids,
        "source_ids": spatial_supergroups,
        "dataset_source_ids": [rows[sample]["source_id"] for sample in sample_ids],
        **{key: torch.cat(values, dim=0) for key, values in tensors.items()},
    }
    validate_cache_payload(payload)
    return payload


def validate_cache_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "identity",
        "sample_ids",
        "physical_event_ids",
        "spatial_supergroups",
        "region_groups",
        "component_ids",
        "event_ids",
        "source_ids",
        "dataset_source_ids",
        "visual_logits",
        "terrain_logits",
        "terrain_direction",
        "frozen_vt_correction",
        "q_t",
        "mask",
        "valid",
    }
    missing = required - set(payload)
    if missing:
        raise ProposerProtocolError(f"Cache payload misses keys: {sorted(missing)}")
    count = len(payload["sample_ids"])
    if count == 0 or len(set(payload["sample_ids"])) != count:
        raise ProposerProtocolError("Cache sample identity is empty or duplicated")
    for key in (
        "physical_event_ids",
        "spatial_supergroups",
        "region_groups",
        "component_ids",
        "event_ids",
        "source_ids",
        "dataset_source_ids",
    ):
        if len(payload[key]) != count:
            raise ProposerProtocolError(f"Cache metadata length mismatch for {key}")
    expected_shape = tuple(payload["visual_logits"].shape)
    if len(expected_shape) != 4 or expected_shape[0] != count or expected_shape[1] != 1:
        raise ProposerProtocolError(f"Unexpected proposer shape: {expected_shape}")
    for key in (
        "terrain_logits",
        "terrain_direction",
        "frozen_vt_correction",
        "q_t",
        "mask",
        "valid",
    ):
        if tuple(payload[key].shape) != expected_shape:
            raise ProposerProtocolError(f"Cache tensor shape mismatch for {key}")
    expected_correction = payload["terrain_direction"].float() * ROUTING_CONFIG["alpha"]
    if not torch.allclose(
        payload["frozen_vt_correction"].float(), expected_correction, atol=5e-3, rtol=0.0
    ):
        raise ProposerProtocolError("Frozen VT correction differs from fixed Terrain direction")
    if payload["mask"].dtype != torch.uint8 or payload["valid"].dtype != torch.uint8:
        raise ProposerProtocolError("Cache labels and validity must use uint8")
    if list(payload["event_ids"]) != list(payload["physical_event_ids"]):
        raise ProposerProtocolError("Legacy event_ids alias differs from physical_event_ids")
    if list(payload["source_ids"]) != list(payload["spatial_supergroups"]):
        raise ProposerProtocolError("Legacy source_ids alias differs from spatial_supergroups")
    identity = payload["identity"]
    if identity.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ProposerProtocolError("Unexpected cache schema version")
    if identity.get("export_role") != "inner_test_post_selection_only":
        raise ProposerProtocolError("Cache does not declare post-selection inner-test export")


def validate_done(outdir: Path) -> bool:
    done_path = outdir / "DONE.json"
    if not done_path.is_file():
        return False
    done = json.loads(done_path.read_text(encoding="utf-8"))
    for relative, expected in done.get("artifact_sha256", {}).items():
        path = outdir / relative
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return done.get("status") == "complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--inner-fold", type=int, choices=range(3), required=True)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, default=PROTOCOL_MANIFEST)
    parser.add_argument("--source-logo-csv", type=Path, default=SOURCE_LOGO_CSV)
    parser.add_argument("--base-h5", type=Path, default=terrain_trainer.BASE_H5)
    parser.add_argument("--optical-h5", type=Path, default=terrain_trainer.OPTICAL_H5)
    parser.add_argument("--terrain-h5", type=Path, default=terrain_trainer.TERRAIN_H5)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--visual-epochs", type=int, default=30)
    parser.add_argument("--terrain-epochs", type=int, default=20)
    parser.add_argument("--visual-batch-size", type=int, default=2)
    parser.add_argument("--terrain-batch-size", type=int, default=64)
    parser.add_argument("--export-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--visual-lr", type=float, default=3e-4)
    parser.add_argument("--terrain-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.split_csv = args.split_csv.resolve()
    args.protocol_manifest = args.protocol_manifest.resolve()
    args.source_logo_csv = args.source_logo_csv.resolve()
    args.base_h5 = args.base_h5.resolve()
    args.optical_h5 = args.optical_h5.resolve()
    args.terrain_h5 = args.terrain_h5.resolve()
    args.outdir = args.outdir.resolve()
    if validate_done(args.outdir) and not args.force:
        print(json.dumps({"status": "SKIP_COMPLETE", "outdir": str(args.outdir)}))
        return 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    atomic_text(args.outdir / "command.txt", command + "\n")
    log_path = args.outdir / "run.log"
    atomic_text(log_path, command + "\n")

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    started = time.time()
    set_seed(args.seed)
    rows, roles, split_audit = audit_nested_split(
        args.split_csv,
        args.source_logo_csv,
        args.protocol_manifest,
        args.target_outer_fold,
        args.inner_fold,
        args.base_h5,
    )
    # The legacy module constants are replaced only inside this process so all
    # dataset construction below uses the explicitly registered assets.
    terrain_trainer.BASE_H5 = args.base_h5
    terrain_trainer.OPTICAL_H5 = args.optical_h5
    terrain_trainer.TERRAIN_H5 = args.terrain_h5
    all_ids, event_ids = terrain_trainer.validate_sidecars(
        args.base_h5, args.optical_h5, args.terrain_h5
    )
    event_by_sample = dict(zip(all_ids, event_ids))

    config = {
        "schema_version": "sen12-nested-oof-proposer-config-v1",
        "target_outer_fold": args.target_outer_fold,
        "inner_fold": args.inner_fold,
        "seed": args.seed,
        "split_audit_sha256": split_audit["audit_sha256"],
        "assets": {
            "base_h5": file_signature(args.base_h5, content_hash=False),
            "base_h5_sha256_from_frozen_protocol": json.loads(
                args.protocol_manifest.read_text(encoding="utf-8")
            )["inputs"]["h5_sha256"],
            "optical_h5": file_signature(args.optical_h5, content_hash=False),
            "terrain_h5": file_signature(args.terrain_h5, content_hash=False),
            "split_csv": split_audit["split_csv"],
            "protocol_manifest": split_audit["protocol_manifest"],
            "source_logo_csv": split_audit["source_logo_csv"],
        },
        "training": {
            "visual_epochs": args.visual_epochs,
            "terrain_epochs": args.terrain_epochs,
            "visual_batch_size": args.visual_batch_size,
            "terrain_batch_size": args.terrain_batch_size,
            "visual_lr": args.visual_lr,
            "terrain_lr": args.terrain_lr,
            "weight_decay": args.weight_decay,
            "dice_weight": args.dice_weight,
            "grad_clip": args.grad_clip,
            "max_steps": args.max_steps,
            "amp": args.amp,
            "device": args.device,
            "checkpoint_selection": "inner_validation_average_precision",
            "visual_threshold_selection": "inner_validation_IoU_grid",
            "terrain_routing": ROUTING_CONFIG,
        },
        "label_access_contract": split_audit["label_access_contract"],
        "command": command,
    }
    config["config_sha256"] = canonical_hash(config)
    atomic_json(args.outdir / "config.json", config)
    atomic_json(args.outdir / "split_audit.json", split_audit)
    log(f"[audit] PASS sha={split_audit['audit_sha256']}")

    zero_mean = np.zeros(17, dtype=np.float32)
    unit_std = np.ones(17, dtype=np.float32)
    visual_datasets = {
        role: make_dataset(
            roles[role], rows, all_ids, event_ids, zero_mean, unit_std,
            args.seed, roles["train"], False,
        )
        for role in ("train", "val")
    }
    visual_loaders = {
        role: make_loader(
            visual_datasets[role], args.seed, args.visual_batch_size,
            args.num_workers, role == "train",
        )
        for role in ("train", "val")
    }
    visual_pos_weight = terrain_trainer.protocol.estimate_pos_weight(
        visual_datasets["train"]
    )
    encoder, provenance = terrain_trainer.load_prithvi_encoder()
    visual = terrain_trainer.PrithviVisualCompat(
        terrain_trainer.PrithviEO2ChangeModel(
            encoder, decoder_width=128, freeze_encoder=True
        )
    ).to(args.device)
    (
        visual_state,
        visual_history,
        visual_epoch,
        visual_val_ap,
        visual_threshold,
        visual_threshold_metrics,
    ) = train_visual(
        visual, visual_loaders["train"], visual_loaders["val"],
        visual_pos_weight, args, log,
    )

    terrain_mean, terrain_std = terrain_trainer.estimate_terrain_stats(
        args.terrain_h5, all_ids, roles["train"]
    )
    terrain_datasets = {
        role: make_dataset(
            roles[role], rows, all_ids, event_ids, terrain_mean, terrain_std,
            args.seed, roles["train"], True,
        )
        for role in ("train", "val")
    }
    terrain_loaders = {
        role: make_loader(
            terrain_datasets[role], args.seed, args.terrain_batch_size,
            args.num_workers, role == "train",
        )
        for role in ("train", "val")
    }
    terrain_pos_weight = terrain_trainer.protocol.estimate_pos_weight(
        terrain_datasets["train"]
    )
    terrain = terrain_trainer.SupportOnlyMultiScaleTerrainPyramid(
        17, terrain_trainer.NATIVE_TERRAIN_V2_SCALE_GROUPS
    ).to(args.device)
    terrain_state, terrain_history, terrain_epoch, terrain_val_ap = train_terrain(
        terrain, terrain_loaders["train"], terrain_loaders["val"],
        terrain_pos_weight, args, log,
    )

    checkpoints = args.outdir / "checkpoints"
    visual_checkpoint = checkpoints / "visual_proposer.pt"
    terrain_checkpoint = checkpoints / "terrain_proposer.pt"
    shared_identity = {
        "target_outer_fold": args.target_outer_fold,
        "inner_fold": args.inner_fold,
        "seed": args.seed,
        "config_sha256": config["config_sha256"],
        "split_audit_sha256": split_audit["audit_sha256"],
        "train_sample_sha256": split_audit["roles"]["train"]["sample_sha256"],
        "val_sample_sha256": split_audit["roles"]["val"]["sample_sha256"],
        "test_sample_sha256": split_audit["roles"]["test"]["sample_sha256"],
    }
    visual_state_hash = state_sha256(visual_state)
    terrain_state_hash = state_sha256(terrain_state)
    atomic_torch_save(
        visual_checkpoint,
        {
            "identity": {**shared_identity, "proposer": "visual"},
            "trainable_state_dict": visual_state,
            "trainable_sha256": visual_state_hash,
            "best_epoch": visual_epoch,
            "validation_average_precision": visual_val_ap,
            "threshold": visual_threshold,
            "threshold_metrics": visual_threshold_metrics,
            "history": visual_history,
            "pos_weight": visual_pos_weight,
            "prithvi_provenance": provenance,
        },
    )
    atomic_torch_save(
        terrain_checkpoint,
        {
            "identity": {**shared_identity, "proposer": "terrain"},
            "trainable_state_dict": terrain_state,
            "trainable_sha256": terrain_state_hash,
            "best_epoch": terrain_epoch,
            "validation_average_precision": terrain_val_ap,
            "history": terrain_history,
            "pos_weight": terrain_pos_weight,
            "terrain_mean": terrain_mean,
            "terrain_std": terrain_std,
        },
    )
    visual_signature = file_signature(visual_checkpoint)
    terrain_signature = file_signature(terrain_checkpoint)

    # Reload the exact selected state before the first inner-test dataset exists.
    visual_payload = torch.load(visual_checkpoint, map_location="cpu", weights_only=False)
    terrain_payload = torch.load(terrain_checkpoint, map_location="cpu", weights_only=False)
    terrain_trainer.load_trainable_state(visual, visual_payload["trainable_state_dict"])
    terrain_trainer.load_trainable_state(terrain, terrain_payload["trainable_state_dict"])
    if state_sha256(terrain_trainer.trainable_state(visual)) != visual_state_hash:
        raise RuntimeError("Reloaded visual proposer differs from selected checkpoint")
    if state_sha256(terrain_trainer.trainable_state(terrain)) != terrain_state_hash:
        raise RuntimeError("Reloaded Terrain proposer differs from selected checkpoint")

    inner_test_dataset = make_dataset(
        roles["test"], rows, all_ids, event_ids, terrain_mean, terrain_std,
        args.seed, roles["train"], True,
    )
    inner_test_loader = make_loader(
        inner_test_dataset, args.seed, args.export_batch_size, args.num_workers, False
    )
    cache_identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "target_outer_fold": args.target_outer_fold,
        "inner_fold": args.inner_fold,
        "seed": args.seed,
        "export_role": "inner_test_post_selection_only",
        "config_sha256": config["config_sha256"],
        "split_audit_sha256": split_audit["audit_sha256"],
        "split_csv_sha256": split_audit["split_csv"]["sha256"],
        "visual_checkpoint_sha256": visual_signature["sha256"],
        "terrain_checkpoint_sha256": terrain_signature["sha256"],
        "visual_trainable_sha256": visual_state_hash,
        "terrain_trainable_sha256": terrain_state_hash,
        "visual_threshold": visual_threshold,
        "routing": ROUTING_CONFIG,
        "sample_sha256": split_audit["roles"]["test"]["sample_sha256"],
        "physical_event_sha256": split_audit["roles"]["test"]["physical_event_sha256"],
        "component_sha256": split_audit["roles"]["test"]["component_sha256"],
    }
    cache_payload = export_inner_test_cache(
        visual,
        terrain,
        inner_test_dataset,
        inner_test_loader,
        rows,
        event_by_sample,
        visual_threshold,
        cache_identity,
        args,
    )
    cache_path = args.outdir / "cache/inner_test_proposer_cache.pt"
    atomic_torch_save(cache_path, cache_payload)
    cache_signature = file_signature(cache_path)

    run_manifest = {
        "schema_version": "sen12-nested-oof-proposer-run-v1",
        "status": "complete",
        "target_outer_fold": args.target_outer_fold,
        "inner_fold": args.inner_fold,
        "seed": args.seed,
        "split_audit": split_audit,
        "config": file_signature(args.outdir / "config.json"),
        "checkpoints": {
            "visual": visual_signature,
            "terrain": terrain_signature,
        },
        "selection": {
            "visual": {
                "split": "inner_val",
                "objective": "average_precision",
                "best_epoch": visual_epoch,
                "validation_average_precision": visual_val_ap,
                "threshold": visual_threshold,
                "threshold_objective": "inner_validation_IoU_grid",
                "threshold_metrics": visual_threshold_metrics,
            },
            "terrain": {
                "split": "inner_val",
                "objective": "average_precision",
                "best_epoch": terrain_epoch,
                "validation_average_precision": terrain_val_ap,
                "routing": ROUTING_CONFIG,
            },
            "inner_test_used_for_selection": False,
            "target_outer_test_used_anywhere": False,
        },
        "cache": cache_signature,
        "cache_schema": {
            "paired_same_checkpoint": True,
            "tensor_keys": [
                "visual_logits",
                "terrain_logits",
                "terrain_direction",
                "frozen_vt_correction",
                "q_t",
                "mask",
                "valid",
            ],
            "metadata_keys": [
                "sample_ids",
                "physical_event_ids",
                "spatial_supergroups",
                "region_groups",
                "component_ids",
                "event_ids",
                "source_ids",
                "dataset_source_ids",
            ],
        },
        "elapsed_seconds": time.time() - started,
    }
    run_manifest["manifest_payload_sha256"] = canonical_hash(run_manifest)
    manifest_path = args.outdir / "run_manifest.json"
    atomic_json(manifest_path, run_manifest)
    done = {
        "status": "complete",
        "artifact_sha256": {
            "config.json": sha256_file(args.outdir / "config.json"),
            "split_audit.json": sha256_file(args.outdir / "split_audit.json"),
            "checkpoints/visual_proposer.pt": visual_signature["sha256"],
            "checkpoints/terrain_proposer.pt": terrain_signature["sha256"],
            "cache/inner_test_proposer_cache.pt": cache_signature["sha256"],
            "run_manifest.json": sha256_file(manifest_path),
        },
    }
    atomic_json(args.outdir / "DONE.json", done)
    log(
        f"[done] target={args.target_outer_fold} inner={args.inner_fold} "
        f"samples={len(cache_payload['sample_ids'])} elapsed={run_manifest['elapsed_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
