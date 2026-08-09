#!/usr/bin/env python3
"""Provenance-hardened same-threshold L4S Terrain-content audit.

The audit is read-only with respect to all trained checkpoints.  It replays the
same 8 architectures x 5 seeds frozen by the formal-v2 L4S audit and evaluates
five inference variants from each unchanged adapter checkpoint:

* visual: the frozen visual branch;
* aligned: the registered Terrain channels;
* zero: an all-zero Terrain tensor;
* sample_shift: Terrain from the next test sample;
* spatial_roll: the registered Terrain tensor circularly rolled by 32 x 32 px.

Every hard prediction uses the matched visual checkpoint's validation-selected
threshold.  No adapter threshold or test outcome is used to select a threshold,
control, checkpoint, or architecture.  The threshold-free AP/Brier/NLL controls
test Terrain-content dependence; same-threshold IoU and error flow are reported
as a separate hard-decision boundary.  Formal v2 additionally binds the actual
test H5, bootstrap count, freeze receipt, and runtime arguments into the final
evidence receipt; it does not change any estimand, control, or decision rule.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import audit_l4s_terrain_same_threshold_20260718 as audit


ROOT = audit.ROOT
CACHE = audit.CACHE
BASE_PROTOCOL_DIR = audit.PROTOCOL_DIR
AMENDMENT_DIR = audit.AMENDMENT_DIR
ANALYSIS_ID = "l4s_terrain_controls_same_threshold_formal_v2_20260718"
PROTOCOL_DIR = ROOT / "metadata/protocol_assets" / ANALYSIS_ID
OUTDIR = ROOT / "experiments/revision2026" / ANALYSIS_ID
EXPECTED_SEEDS = audit.EXPECTED_SEEDS
EXPECTED_PATCHES = audit.EXPECTED_PATCHES
THRESHOLD_POLICY = "formal_v2_frozen_visual"
SENSITIVITY_THRESHOLD_POLICY = "formal_v2_frozen_adapter"
HARD_POLICIES = (THRESHOLD_POLICY, SENSITIVITY_THRESHOLD_POLICY)
VARIANTS = ("visual", "aligned", "zero", "sample_shift", "spatial_roll")
CONTROLS = ("zero", "sample_shift", "spatial_roll")
ROLL = (32, 32)
EXPECTED_BOOTSTRAP = 10000
FLOW_FIELDS = audit.FLOW_COUNT_FIELDS
REQUIRED_OUTPUTS = (
    "run.log",
    "per_seed_sample_controls.csv",
    "per_seed_probability_metrics.csv",
    "architecture_control_contrasts.csv",
    "decision_summary.csv",
    "execution_receipt.json",
    "summary.json",
    "report.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise audit.AuditError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "validate", "run"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--base-protocol-dir", type=Path, default=BASE_PROTOCOL_DIR)
    parser.add_argument("--amendment-dir", type=Path, default=AMENDMENT_DIR)
    parser.add_argument("--protocol-dir", type=Path, default=PROTOCOL_DIR)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=EXPECTED_BOOTSTRAP)
    return parser.parse_args()


class CounterfactualTerrainDataset(Dataset):
    """Add a deterministic next-sample Terrain donor without changing the H5."""

    def __init__(self, base: audit.H5SupportDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        current = self.base[index]
        donor_index = (index + 1) % len(self.base)
        donor = self.base[donor_index]
        require(
            str(current["sample_id"]) != str(donor["sample_id"]),
            f"sample-shift donor equals recipient at index={index}",
        )
        return {
            **current,
            "terrain_shift": donor["terrain"],
            "terrain_shift_sample_id": str(donor["sample_id"]),
        }


def implementation_record(path: Path, root: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing frozen asset: {path}")
    stat = path.stat()
    return {
        "path": audit.relative_to_root(path, root),
        "role": role,
        "sha256": audit.sha256_file(path),
        "size_bytes": int(stat.st_size),
    }


def checkpoint_hash_lookup(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in manifest.get("computational_inputs", []):
        role = str(record.get("role"))
        if role not in {"adapter_checkpoint", "visual_checkpoint"}:
            continue
        key = (str(record["path"]), role)
        require(key not in lookup, f"duplicate frozen checkpoint record: {key}")
        lookup[key] = record
    require(len(lookup) == 80, f"expected 80 frozen checkpoint records, found {len(lookup)}")
    return lookup


def build_protocol(
    root: Path,
    cache_dir: Path,
    base_protocol_dir: Path,
    amendment_dir: Path,
    bootstrap: int,
) -> dict[str, Any]:
    _, manifest, _ = audit.verify_freeze(root, base_protocol_dir, amendment_dir)
    require(bootstrap == EXPECTED_BOOTSTRAP, f"formal bootstrap must equal {EXPECTED_BOOTSTRAP}")
    test_records = [
        record for record in manifest.get("computational_inputs", [])
        if str(record.get("role")) == "test_h5"
    ]
    require(len(test_records) == 1, f"expected one frozen test H5, found {len(test_records)}")
    frozen_test = test_records[0]
    actual_test_path = audit.split_path(cache_dir.resolve(), "test").resolve()
    frozen_test_path = (root / str(frozen_test["path"])).resolve()
    require(actual_test_path == frozen_test_path, "requested test H5 differs from formal-v2 manifest")
    require(actual_test_path.is_file(), f"missing frozen test H5: {actual_test_path}")
    require(actual_test_path.stat().st_size == int(frozen_test["size_bytes"]), "test H5 size changed")
    require(audit.sha256_file(actual_test_path) == str(frozen_test["sha256"]), "test H5 hash changed")
    lookup = checkpoint_hash_lookup(manifest)
    pair_bindings: list[dict[str, Any]] = []
    for pair in manifest.get("checkpoint_pairs", []):
        visual_key = (str(pair["visual_checkpoint"]), "visual_checkpoint")
        adapter_key = (str(pair["adapter_checkpoint"]), "adapter_checkpoint")
        require(visual_key in lookup and adapter_key in lookup, f"missing pair hash binding: {pair}")
        pair_bindings.append(
            {
                "architecture_key": str(pair["architecture_key"]),
                "architecture": str(pair["architecture"]),
                "family": str(pair["family"]),
                "seed": int(pair["seed"]),
                "frozen_visual_threshold": float(pair["visual_threshold"]),
                "visual_checkpoint": {
                    key: lookup[visual_key][key]
                    for key in ("path", "role", "sha256", "size_bytes")
                },
                "adapter_checkpoint": {
                    key: lookup[adapter_key][key]
                    for key in ("path", "role", "sha256", "size_bytes")
                },
            }
        )
    expected = {
        (spec.key, seed) for spec in audit.ARCHITECTURES for seed in EXPECTED_SEEDS
    }
    found = {(row["architecture_key"], row["seed"]) for row in pair_bindings}
    require(found == expected, "checkpoint-pair inventory differs from formal-v2")
    bound_inputs = []
    for path, role in (
        (base_protocol_dir / "freeze.json", "base_freeze"),
        (base_protocol_dir / "input_manifest.json", "base_input_manifest"),
        (base_protocol_dir / "protocol.json", "base_protocol"),
        (base_protocol_dir / "threshold_sources.csv", "base_threshold_sources"),
        (amendment_dir / "amendment.json", "formal_v2_amendment"),
        (amendment_dir / "DONE.json", "formal_v2_amendment_done"),
    ):
        bound_inputs.append(implementation_record(path, root, role))
    return {
        "schema_version": "1.1",
        "analysis_id": ANALYSIS_ID,
        "created_at_utc": utc_now(),
        "status": "frozen",
        "root_binding": str(root.resolve()),
        "read_only_checkpoint_replay": True,
        "training_performed": False,
        "claim_scope": "terrain_content_attribution_with_hard_decision_boundary",
        "analysis_implementation": implementation_record(Path(__file__), root, "analysis_implementation"),
        "test_h5_binding": {
            key: frozen_test[key] for key in ("path", "role", "sha256", "size_bytes")
        },
        "execution_contract": {
            "bootstrap_iterations": EXPECTED_BOOTSTRAP,
            "bootstrap_count_is_decision_frozen": True,
            "actual_test_h5_must_match_binding": True,
            "freeze_done_receipt_must_validate": True,
            "frozen_assets_read_only": True,
        },
        "base_formal_v2_bindings": bound_inputs,
        "checkpoint_pair_bindings": pair_bindings,
        "control_contract": {
            "variants": list(VARIANTS),
            "aligned": "registered Terrain tensor from the same test sample",
            "zero": "all Terrain channels set exactly to zero at inference",
            "sample_shift": "Terrain tensor from the next test sample in frozen H5 order, with wrap-around",
            "spatial_roll": "registered Terrain circularly rolled by +32 rows and +32 columns",
            "roll_pixels": list(ROLL),
            "same_adapter_parameters_for_all_variants": True,
            "retraining_per_control": False,
            "control_selection_uses_test_outcomes": False,
        },
        "threshold_contract": {
            "primary_policy": THRESHOLD_POLICY,
            "sensitivity_policy": SENSITIVITY_THRESHOLD_POLICY,
            "primary_source": "matched visual checkpoint validation-selected threshold frozen by formal v2",
            "sensitivity_source": "adapter checkpoint validation-selected threshold frozen by formal v2",
            "same_threshold_for_visual_and_all_controls": True,
            "adapter_threshold_used_for_primary": False,
            "adapter_threshold_used_only_for_prespecified_sensitivity": True,
            "test_threshold_selection": False,
        },
        "estimand_contract": {
            "threshold_free": [
                "aligned_minus_control_average_precision",
                "control_minus_aligned_brier",
                "control_minus_aligned_nll",
            ],
            "hard_decision": [
                "aligned_minus_control_pooled_foreground_iou",
                "aligned_minus_control_mean_patch_foreground_iou",
            ],
            "cluster_unit": "800 unique L4S test patches after averaging five seeds within patch",
            "event_level_inference": False,
        },
        "decision_contract": {
            "per_control_threshold_free_pass": (
                "AP mean improvement > 0 with at least 4/5 positive seeds, and at least one of "
                "Brier or NLL mean improvement > 0 with at least 4/5 positive seeds"
            ),
            "architecture_terrain_content_pass": "all three controls pass the threshold-free rule",
            "per_control_hard_decision_pass": (
                "pooled foreground-IoU improvement > 0, patch-bootstrap 95% CI lower bound > 0, "
                "and at least 4/5 seed directions positive"
            ),
            "architecture_hard_decision_pass": "all three controls pass the hard-decision rule",
            "abstention_rule": (
                "A failed gate removes the corresponding Terrain-content or hard-decision claim; "
                "results are not rescued by changing thresholds, controls, or architecture subsets"
            ),
        },
    }


def freeze_protocol(args: argparse.Namespace) -> None:
    protocol_dir = args.protocol_dir.resolve()
    require(not protocol_dir.exists(), f"protocol directory already exists: {protocol_dir}")
    require(args.bootstrap == EXPECTED_BOOTSTRAP, f"formal bootstrap must equal {EXPECTED_BOOTSTRAP}")
    protocol = build_protocol(
        args.root.resolve(), args.cache_dir, args.base_protocol_dir, args.amendment_dir, args.bootstrap
    )
    protocol_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = protocol_dir / "protocol.json"
    audit.write_strict_json(protocol_path, protocol)
    freeze = {
        "schema_version": "1.1",
        "analysis_id": ANALYSIS_ID,
        "created_at_utc": utc_now(),
        "status": "frozen",
        "protocol_sha256": audit.sha256_file(protocol_path),
        "analysis_implementation_sha256": protocol["analysis_implementation"]["sha256"],
        "n_checkpoint_pairs": len(protocol["checkpoint_pair_bindings"]),
        "n_architectures": len(audit.ARCHITECTURES),
        "n_seeds_per_architecture": len(EXPECTED_SEEDS),
        "n_unique_patches": EXPECTED_PATCHES,
    }
    audit.write_strict_json(protocol_dir / "freeze.json", freeze)
    audit.write_strict_json(
        protocol_dir / "DONE.json",
        {
            "analysis_id": ANALYSIS_ID,
            "phase": "protocol_freeze",
            "status": "complete",
            "created_at_utc": utc_now(),
            "freeze_sha256": audit.sha256_file(protocol_dir / "freeze.json"),
        },
    )
    for frozen_path in (protocol_path, protocol_dir / "freeze.json", protocol_dir / "DONE.json"):
        frozen_path.chmod(0o444)
    Path(__file__).chmod(0o555)
    protocol_dir.chmod(0o555)
    print(f"[frozen] {protocol_dir}")


def verify_small_record(root: Path, record: Mapping[str, Any]) -> None:
    path = root / str(record["path"])
    require(path.is_file(), f"missing bound asset: {path}")
    require(path.stat().st_size == int(record["size_bytes"]), f"bound asset size changed: {path}")
    require(audit.sha256_file(path) == str(record["sha256"]), f"bound asset hash changed: {path}")


def verify_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    root = args.root.resolve()
    protocol_dir = args.protocol_dir.resolve()
    protocol_path = protocol_dir / "protocol.json"
    freeze_path = protocol_dir / "freeze.json"
    freeze_done_path = protocol_dir / "DONE.json"
    require(
        protocol_path.is_file() and freeze_path.is_file() and freeze_done_path.is_file(),
        "formal control protocol is not frozen",
    )
    protocol = audit.load_strict_json(protocol_path)
    freeze = audit.load_strict_json(freeze_path)
    freeze_done = audit.load_strict_json(freeze_done_path)
    require(protocol.get("analysis_id") == ANALYSIS_ID, "analysis id changed")
    require(protocol.get("status") == "frozen", "protocol is not frozen")
    require(str(protocol.get("root_binding")) == str(root), "root binding changed")
    require(audit.sha256_file(protocol_path) == freeze.get("protocol_sha256"), "protocol hash changed")
    require(
        audit.sha256_file(freeze_path) == freeze_done.get("freeze_sha256"),
        "freeze receipt hash changed",
    )
    require(freeze_done.get("analysis_id") == ANALYSIS_ID, "freeze receipt analysis id changed")
    require(freeze_done.get("phase") == "protocol_freeze", "freeze receipt phase changed")
    require(freeze_done.get("status") == "complete", "freeze receipt is incomplete")
    require(args.bootstrap == EXPECTED_BOOTSTRAP, f"formal bootstrap must equal {EXPECTED_BOOTSTRAP}")
    require(
        int(protocol["execution_contract"]["bootstrap_iterations"]) == args.bootstrap,
        "runtime bootstrap differs from frozen protocol",
    )
    implementation = protocol["analysis_implementation"]
    require(
        str(implementation["path"]) == audit.relative_to_root(Path(__file__).resolve(), root),
        "implementation path binding changed",
    )
    verify_small_record(root, implementation)
    require(
        freeze.get("analysis_implementation_sha256") == implementation["sha256"],
        "freeze implementation hash changed",
    )
    test_path = audit.split_path(args.cache_dir.resolve(), "test").resolve()
    test_binding = protocol["test_h5_binding"]
    require(test_path == (root / str(test_binding["path"])).resolve(), "runtime test H5 path differs from binding")
    verify_small_record(root, test_binding)
    for record in protocol.get("base_formal_v2_bindings", []):
        verify_small_record(root, record)
    _, manifest, _ = audit.verify_freeze(root, args.base_protocol_dir, args.amendment_dir)
    lookup = checkpoint_hash_lookup(manifest)
    pairs = protocol.get("checkpoint_pair_bindings", [])
    require(len(pairs) == 40, f"expected 40 pair bindings, found {len(pairs)}")
    expected_pair_keys = {
        (spec.key, seed) for spec in audit.ARCHITECTURES for seed in EXPECTED_SEEDS
    }
    bound_pair_keys = {
        (str(pair["architecture_key"]), int(pair["seed"])) for pair in pairs
    }
    require(bound_pair_keys == expected_pair_keys, "protocol architecture/seed inventory changed")
    manifest_pair_lookup = {
        (str(pair["architecture_key"]), int(pair["seed"])): pair
        for pair in manifest.get("checkpoint_pairs", [])
    }
    require(set(manifest_pair_lookup) == expected_pair_keys, "formal-v2 pair inventory changed")
    for pair in pairs:
        pair_key = (str(pair["architecture_key"]), int(pair["seed"]))
        frozen_pair = manifest_pair_lookup[pair_key]
        require(
            float(pair["frozen_visual_threshold"]) == float(frozen_pair["visual_threshold"]),
            f"frozen visual threshold changed: {pair_key}",
        )
        for field, role in (("visual_checkpoint", "visual_checkpoint"), ("adapter_checkpoint", "adapter_checkpoint")):
            record = pair[field]
            require(str(record["role"]) == role, f"checkpoint role changed: {record['path']}")
            frozen = lookup.get((str(record["path"]), role))
            require(frozen is not None, f"checkpoint absent from formal-v2 manifest: {record['path']}")
            for key in ("sha256", "size_bytes"):
                require(record[key] == frozen[key], f"checkpoint binding changed: {record['path']} / {key}")
    require(protocol["control_contract"]["roll_pixels"] == list(ROLL), "roll transform changed")
    require(protocol["threshold_contract"]["primary_policy"] == THRESHOLD_POLICY, "primary threshold policy changed")
    require(
        protocol["threshold_contract"]["sensitivity_policy"] == SENSITIVITY_THRESHOLD_POLICY,
        "sensitivity threshold policy changed",
    )
    require(protocol["threshold_contract"]["adapter_threshold_used_for_primary"] is False, "adapter threshold enabled for primary")
    for frozen_path in (protocol_path, freeze_path, freeze_done_path, Path(__file__)):
        require((frozen_path.stat().st_mode & 0o222) == 0, f"frozen asset is writable: {frozen_path}")
    return protocol, manifest


def custom_variant_logits(
    spec: audit.ArchitectureSpec,
    model: torch.nn.Module,
    observation: torch.Tensor,
    terrain_variants: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if spec.family == "modern_bn_frozen":
        visual_logits = model.visual(observation)
        if visual_logits.shape[-2:] != observation.shape[-2:]:
            visual_logits = F.interpolate(
                visual_logits, size=observation.shape[-2:], mode="bilinear", align_corners=False
            )
        visual_features = None
    elif spec.family == "foundation_model":
        visual_logits, visual_features = model.visual.segment(observation)
    else:
        raise audit.AuditError(f"unsupported architecture family: {spec.family}")
    probability = torch.sigmoid(visual_logits.detach())
    uncertainty = 1.0 - torch.abs(2.0 * probability - 1.0)
    outputs: dict[str, torch.Tensor] = {"visual": visual_logits}
    for variant, terrain in terrain_variants.items():
        terrain_features = model.terrain_encoder(terrain)
        if spec.family == "modern_bn_frozen":
            fused = torch.cat((terrain_features, uncertainty), dim=1)
        else:
            require(visual_features is not None, "foundation-model visual features are missing")
            fused = torch.cat((visual_features, terrain_features, uncertainty), dim=1)
        residual = model.alpha_max * torch.tanh(model.residual_head(fused))
        gate = torch.sigmoid(model.gate_head(fused)) * uncertainty
        outputs[variant] = visual_logits + gate * residual
    return visual_logits, outputs


def per_variant_flow(
    target: torch.Tensor,
    visual_probability: torch.Tensor,
    candidate_probability: torch.Tensor,
    threshold: float,
) -> list[dict[str, Any]]:
    return audit.torch_flow_batch(target, visual_probability, candidate_probability, threshold, threshold)


def replay_pair(
    spec: audit.ArchitectureSpec,
    pair: Mapping[str, Any],
    root: Path,
    dataset: CounterfactualTerrainDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    adapter_path = root / str(pair["adapter_checkpoint"])
    visual_path = root / str(pair["visual_checkpoint"])
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
    visual_checkpoint = torch.load(visual_path, map_location="cpu", weights_only=False)
    seed = int(pair["seed"])
    audit.assert_visual_state_identical(
        adapter_checkpoint, visual_checkpoint, f"formal-controls/{spec.key}/seed{seed}"
    )
    threshold = float(pair["visual_threshold"])
    sensitivity_threshold = float(pair["adapter_threshold"])
    require(float(visual_checkpoint["threshold"]) == threshold, "frozen visual threshold changed")
    require(float(adapter_checkpoint["threshold"]) == sensitivity_threshold, "frozen adapter threshold changed")
    model = audit.build_model(spec, adapter_checkpoint).to(device)
    del adapter_checkpoint, visual_checkpoint
    gc.collect()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    sample_rows: list[dict[str, Any]] = []
    probabilities: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    targets: list[np.ndarray] = []
    brier_sum = defaultdict(float)
    nll_sum = defaultdict(float)
    pixel_total = 0
    aligned_forward_max_abs_diff = 0.0
    first_batch = True
    with torch.inference_mode():
        for batch in loader:
            observation = batch["obs"].to(device, non_blocking=True)
            terrain = batch["terrain"].to(device, non_blocking=True)
            terrain_variants = {
                "aligned": terrain,
                "zero": torch.zeros_like(terrain),
                "sample_shift": batch["terrain_shift"].to(device, non_blocking=True),
                "spatial_roll": torch.roll(terrain, shifts=ROLL, dims=(-2, -1)),
            }
            target = batch["mask"].to(device, non_blocking=True)
            visual_logits, logits_by_variant = custom_variant_logits(
                spec, model, observation, terrain_variants
            )
            if first_batch:
                direct_aligned, _ = model(observation, terrain)
                aligned_forward_max_abs_diff = float(
                    (direct_aligned - logits_by_variant["aligned"]).abs().max().item()
                )
                require(
                    torch.allclose(direct_aligned, logits_by_variant["aligned"], rtol=0.0, atol=1e-6),
                    f"custom aligned forward differs from model.forward: {spec.key}/seed{seed}",
                )
                first_batch = False
            require(
                all(logits.shape == target.shape for logits in logits_by_variant.values()),
                f"logit/target shape mismatch: {spec.key}/seed{seed}",
            )
            visual_probability = torch.sigmoid(visual_logits)
            truth = (target >= 0.5).to(torch.float32)
            for variant in VARIANTS:
                logits = logits_by_variant[variant]
                require(bool(torch.isfinite(logits).all().item()), f"non-finite logits: {spec.key}/{seed}/{variant}")
                probability = torch.sigmoid(logits)
                sample_brier = ((probability - truth) ** 2).flatten(1).mean(dim=1)
                sample_nll = F.binary_cross_entropy_with_logits(
                    logits, truth, reduction="none"
                ).flatten(1).mean(dim=1)
                for policy, used_threshold in (
                    (THRESHOLD_POLICY, threshold),
                    (SENSITIVITY_THRESHOLD_POLICY, sensitivity_threshold),
                ):
                    flow_rows = per_variant_flow(
                        target, visual_probability, probability, used_threshold
                    )
                    for index, flow in enumerate(flow_rows):
                        sample_rows.append(
                            {
                                "architecture_key": spec.key,
                                "architecture": spec.label,
                                "family": spec.family,
                                "seed": seed,
                                "split": "test",
                                "sample_id": str(batch["sample_id"][index]),
                                "variant": variant,
                                "terrain_shift_sample_id": (
                                    str(batch["terrain_shift_sample_id"][index])
                                    if variant == "sample_shift" else ""
                                ),
                                "roll_y": ROLL[0] if variant == "spatial_roll" else 0,
                                "roll_x": ROLL[1] if variant == "spatial_roll" else 0,
                                "threshold_policy": policy,
                                "same_threshold_for_visual_and_candidate": True,
                                "threshold_used": used_threshold,
                                "adapter_threshold_used_for_primary": False,
                                "brier": float(sample_brier[index].item()),
                                "nll": float(sample_nll[index].item()),
                                **flow,
                            }
                        )
                flattened = probability.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                probabilities[variant].append(flattened)
                brier_sum[variant] += float(((probability - truth) ** 2).sum().item())
                nll_sum[variant] += float(
                    F.binary_cross_entropy_with_logits(logits, truth, reduction="sum").item()
                )
            targets.append(truth.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1))
            pixel_total += int(truth.numel())

    from sklearn.metrics import average_precision_score

    target_flat = np.concatenate(targets)
    require(target_flat.size == pixel_total, "target pixel inventory mismatch")
    probability_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        probability_flat = np.concatenate(probabilities[variant])
        require(probability_flat.size == target_flat.size, f"probability inventory mismatch: {variant}")
        probability_rows.append(
            {
                "architecture_key": spec.key,
                "architecture": spec.label,
                "family": spec.family,
                "seed": seed,
                "split": "test",
                "variant": variant,
                "n_patches": len(dataset),
                "n_pixels": int(target_flat.size),
                "positive_pixels": int(target_flat.sum()),
                "average_precision": float(average_precision_score(target_flat, probability_flat)),
                "brier": float(brier_sum[variant] / pixel_total),
                "nll": float(nll_sum[variant] / pixel_total),
            }
        )
    provenance = {
        "architecture_key": spec.key,
        "seed": seed,
        "visual_checkpoint": str(pair["visual_checkpoint"]),
        "adapter_checkpoint": str(pair["adapter_checkpoint"]),
        "frozen_visual_threshold": threshold,
        "frozen_adapter_threshold_sensitivity": sensitivity_threshold,
        "aligned_forward_max_abs_diff": aligned_forward_max_abs_diff,
        "visual_state_identical": True,
    }
    del model, target_flat, targets, probabilities
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    require(
        len(sample_rows) == len(dataset) * len(VARIANTS) * len(HARD_POLICIES),
        "sample-row inventory mismatch",
    )
    return sample_rows, probability_rows, provenance


def build_seedmean_samples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    audit.ensure_unique_rows(
        rows,
        ("architecture_key", "seed", "sample_id", "variant", "threshold_policy"),
        "control replay",
    )
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["architecture_key"]),
                str(row["sample_id"]),
                str(row["variant"]),
                str(row["threshold_policy"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    expected = set(EXPECTED_SEEDS)
    count_fields = (
        "pixel_count", "positive_pixels", "adapter_tp", "adapter_fp", "adapter_fn",
        "adapter_tn", "adapter_error_pixels", "adapter_correct_pixels",
    )
    for key in sorted(groups):
        members = groups[key]
        require({int(row["seed"]) for row in members} == expected, f"seed inventory mismatch: {key}")
        first = members[0]
        counts = {
            field: float(np.mean([float(row[field]) for row in members]))
            for field in count_fields
        }
        output.append(
            {
                "architecture_key": first["architecture_key"],
                "architecture": first["architecture"],
                "family": first["family"],
                "sample_id": first["sample_id"],
                "variant": first["variant"],
                "threshold_policy": first["threshold_policy"],
                "n_seeds_averaged": len(members),
                **counts,
                "foreground_iou": audit.foreground_iou(
                    counts["adapter_tp"], counts["adapter_fp"], counts["adapter_fn"]
                ),
            }
        )
    expected_rows = (
        len(audit.ARCHITECTURES)
        * EXPECTED_PATCHES
        * len(VARIANTS)
        * len(HARD_POLICIES)
    )
    require(len(output) == expected_rows, f"seed-mean sample inventory mismatch: {len(output)}")
    return output


def pooled_iou(members: Sequence[Mapping[str, Any]]) -> float:
    tp = sum(float(row["adapter_tp"]) for row in members)
    fp = sum(float(row["adapter_fp"]) for row in members)
    fn = sum(float(row["adapter_fn"]) for row in members)
    return audit.foreground_iou(tp, fp, fn)


def bootstrap_iou_contrast(
    aligned: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    iterations: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    require(len(aligned) == len(control) and len(aligned) > 0, "unpaired bootstrap inputs")
    fields = ("adapter_tp", "adapter_fp", "adapter_fn")
    a = {field: np.asarray([float(row[field]) for row in aligned]) for field in fields}
    c = {field: np.asarray([float(row[field]) for row in control]) for field in fields}
    patch_delta = np.asarray(
        [float(left["foreground_iou"]) - float(right["foreground_iou"]) for left, right in zip(aligned, control)]
    )
    rng = np.random.default_rng(seed)
    pooled_values = np.empty(iterations, dtype=np.float64)
    mean_patch_values = np.empty(iterations, dtype=np.float64)
    n = len(aligned)
    for start in range(0, iterations, 250):
        count = min(250, iterations - start)
        indices = rng.integers(0, n, size=(count, n))
        a_tp, a_fp, a_fn = (a[field][indices].sum(axis=1) for field in fields)
        c_tp, c_fp, c_fn = (c[field][indices].sum(axis=1) for field in fields)
        a_iou = a_tp / np.maximum(a_tp + a_fp + a_fn, 1e-300)
        c_iou = c_tp / np.maximum(c_tp + c_fp + c_fn, 1e-300)
        pooled_values[start:start + count] = a_iou - c_iou
        mean_patch_values[start:start + count] = patch_delta[indices].mean(axis=1)
    return (
        [float(np.quantile(pooled_values, 0.025)), float(np.quantile(pooled_values, 0.975))],
        [float(np.quantile(mean_patch_values, 0.025)), float(np.quantile(mean_patch_values, 0.975))],
    )


def threshold_free_control_decision(
    ap: Sequence[float], brier: Sequence[float], nll: Sequence[float]
) -> tuple[bool, bool, bool, bool]:
    """Apply the frozen 4/5 directional rule to one control contrast."""
    require(len(ap) == len(brier) == len(nll) == len(EXPECTED_SEEDS), "metric seed inventory changed")
    ap_values = np.asarray(ap, dtype=np.float64)
    brier_values = np.asarray(brier, dtype=np.float64)
    nll_values = np.asarray(nll, dtype=np.float64)
    require(
        np.isfinite(ap_values).all() and np.isfinite(brier_values).all() and np.isfinite(nll_values).all(),
        "non-finite threshold-free control contrast",
    )
    ap_pass = float(ap_values.mean()) > 0 and int(np.sum(ap_values > 0)) >= 4
    brier_pass = float(brier_values.mean()) > 0 and int(np.sum(brier_values > 0)) >= 4
    nll_pass = float(nll_values.mean()) > 0 and int(np.sum(nll_values > 0)) >= 4
    return bool(ap_pass and (brier_pass or nll_pass)), bool(ap_pass), bool(brier_pass), bool(nll_pass)


def hard_decision_control_decision(
    pooled_delta: float, pooled_ci: Sequence[float], seed_deltas: Sequence[float]
) -> bool:
    require(len(seed_deltas) == len(EXPECTED_SEEDS), "hard-decision seed inventory changed")
    require(len(pooled_ci) == 2, "hard-decision CI must contain two endpoints")
    values = np.asarray(seed_deltas, dtype=np.float64)
    require(np.isfinite(values).all(), "non-finite hard-decision seed contrast")
    return bool(
        float(pooled_delta) > 0
        and float(pooled_ci[0]) > 0
        and int(np.sum(values > 0)) >= 4
    )


def build_contrasts(
    sample_rows: Sequence[Mapping[str, Any]],
    probability_rows: Sequence[Mapping[str, Any]],
    raw_sample_rows: Sequence[Mapping[str, Any]],
    bootstrap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        sample_groups[
            (
                str(row["architecture_key"]),
                str(row["variant"]),
                str(row["threshold_policy"]),
            )
        ].append(row)
    probability_lookup = {
        (str(row["architecture_key"]), int(row["seed"]), str(row["variant"])): row
        for row in probability_rows
    }
    raw_lookup: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_sample_rows:
        raw_lookup[
            (
                str(row["architecture_key"]),
                int(row["seed"]),
                str(row["variant"]),
                str(row["threshold_policy"]),
            )
        ].append(row)
    contrasts: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for architecture_index, spec in enumerate(audit.ARCHITECTURES):
        control_passes = []
        primary_hard_passes = []
        sensitivity_hard_passes = []
        for control_index, control in enumerate(CONTROLS):
            seed_metrics: dict[str, list[float]] = defaultdict(list)
            for seed in EXPECTED_SEEDS:
                a = probability_lookup[(spec.key, seed, "aligned")]
                c = probability_lookup[(spec.key, seed, control)]
                seed_metrics["ap"].append(float(a["average_precision"]) - float(c["average_precision"]))
                seed_metrics["brier"].append(float(c["brier"]) - float(a["brier"]))
                seed_metrics["nll"].append(float(c["nll"]) - float(a["nll"]))
            threshold_free_pass, ap_pass, brier_pass, nll_pass = threshold_free_control_decision(
                seed_metrics["ap"], seed_metrics["brier"], seed_metrics["nll"]
            )
            hard_results: dict[str, dict[str, Any]] = {}
            for policy_index, policy in enumerate(HARD_POLICIES):
                aligned = sorted(
                    sample_groups[(spec.key, "aligned", policy)],
                    key=lambda row: str(row["sample_id"]),
                )
                compared = sorted(
                    sample_groups[(spec.key, control, policy)],
                    key=lambda row: str(row["sample_id"]),
                )
                require(
                    [row["sample_id"] for row in aligned]
                    == [row["sample_id"] for row in compared],
                    f"sample mismatch: {spec.key}/{control}/{policy}",
                )
                pooled_delta = pooled_iou(aligned) - pooled_iou(compared)
                patch_deltas = np.asarray(
                    [
                        float(a["foreground_iou"]) - float(c["foreground_iou"])
                        for a, c in zip(aligned, compared)
                    ]
                )
                pooled_ci, patch_ci = bootstrap_iou_contrast(
                    aligned,
                    compared,
                    bootstrap,
                    20260718
                    + architecture_index * 100
                    + control_index * 10
                    + policy_index,
                )
                seed_iou_deltas = []
                for seed in EXPECTED_SEEDS:
                    aligned_seed = raw_lookup[(spec.key, seed, "aligned", policy)]
                    control_seed = raw_lookup[(spec.key, seed, control, policy)]
                    seed_iou_deltas.append(
                        pooled_iou(aligned_seed) - pooled_iou(control_seed)
                    )
                hard_results[policy] = {
                    "pooled_delta": pooled_delta,
                    "pooled_ci": pooled_ci,
                    "mean_patch_delta": float(patch_deltas.mean()),
                    "mean_patch_ci": patch_ci,
                    "positive_seeds": int(np.sum(np.asarray(seed_iou_deltas) > 0)),
                    "pass": hard_decision_control_decision(
                        pooled_delta, pooled_ci, seed_iou_deltas
                    ),
                }
            primary = hard_results[THRESHOLD_POLICY]
            sensitivity = hard_results[SENSITIVITY_THRESHOLD_POLICY]
            control_passes.append(threshold_free_pass)
            primary_hard_passes.append(bool(primary["pass"]))
            sensitivity_hard_passes.append(bool(sensitivity["pass"]))
            contrasts.append(
                {
                    "architecture_key": spec.key,
                    "architecture": spec.label,
                    "family": spec.family,
                    "control": control,
                    "n_unique_patch_clusters": EXPECTED_PATCHES,
                    "n_seeds": len(EXPECTED_SEEDS),
                    "aligned_minus_control_average_precision_mean": float(np.mean(seed_metrics["ap"])),
                    "ap_positive_seeds": int(np.sum(np.asarray(seed_metrics["ap"]) > 0)),
                    "ap_directional_pass": ap_pass,
                    "control_minus_aligned_brier_mean": float(np.mean(seed_metrics["brier"])),
                    "brier_positive_seeds": int(np.sum(np.asarray(seed_metrics["brier"]) > 0)),
                    "brier_directional_pass": brier_pass,
                    "control_minus_aligned_nll_mean": float(np.mean(seed_metrics["nll"])),
                    "nll_positive_seeds": int(np.sum(np.asarray(seed_metrics["nll"]) > 0)),
                    "nll_directional_pass": nll_pass,
                    "primary_threshold_policy": THRESHOLD_POLICY,
                    "aligned_minus_control_pooled_foreground_iou": primary["pooled_delta"],
                    "pooled_foreground_iou_delta_ci95": primary["pooled_ci"],
                    "aligned_minus_control_mean_patch_foreground_iou": primary["mean_patch_delta"],
                    "mean_patch_foreground_iou_delta_ci95": primary["mean_patch_ci"],
                    "hard_iou_positive_seeds": primary["positive_seeds"],
                    "sensitivity_threshold_policy": SENSITIVITY_THRESHOLD_POLICY,
                    "sensitivity_aligned_minus_control_pooled_foreground_iou": sensitivity["pooled_delta"],
                    "sensitivity_pooled_foreground_iou_delta_ci95": sensitivity["pooled_ci"],
                    "sensitivity_aligned_minus_control_mean_patch_foreground_iou": sensitivity["mean_patch_delta"],
                    "sensitivity_mean_patch_foreground_iou_delta_ci95": sensitivity["mean_patch_ci"],
                    "sensitivity_hard_iou_positive_seeds": sensitivity["positive_seeds"],
                    "threshold_free_content_pass": threshold_free_pass,
                    "hard_decision_content_pass": primary["pass"],
                    "sensitivity_hard_decision_content_pass": sensitivity["pass"],
                }
            )
        decisions.append(
            {
                "architecture_key": spec.key,
                "architecture": spec.label,
                "family": spec.family,
                "n_controls": len(CONTROLS),
                "threshold_free_controls_passed": int(sum(control_passes)),
                "hard_decision_controls_passed": int(sum(primary_hard_passes)),
                "sensitivity_hard_decision_controls_passed": int(
                    sum(sensitivity_hard_passes)
                ),
                "terrain_content_attribution_pass": bool(all(control_passes)),
                "hard_decision_content_pass": bool(all(primary_hard_passes)),
                "sensitivity_hard_decision_content_pass": bool(
                    all(sensitivity_hard_passes)
                ),
            }
        )
    return contrasts, decisions


def write_report(
    path: Path,
    contrasts: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Formal same-threshold L4S Terrain-content controls",
        "",
        "- Read-only replay: 8 architectures x 5 seeds x 800 official test patches.",
        "- Primary hard decisions use the matched visual validation threshold; the adapter validation threshold is a same-threshold sensitivity only.",
        "- Aligned, zero, next-sample shift, and 32x32 spatial roll reuse identical adapter parameters.",
        "- Zero is zero in train-standardized feature space (the train mean), not physical zero elevation or slope.",
        "- AP/Brier/NLL establish threshold-free content dependence; IoU is a separate hard-decision boundary.",
        "",
        "## Architecture decisions",
        "",
        "| architecture | threshold-free controls passed | primary hard controls passed | sensitivity hard controls passed | Terrain-content pass | primary hard pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['architecture']} | {row['threshold_free_controls_passed']}/3 | "
            f"{row['hard_decision_controls_passed']}/3 | "
            f"{row['sensitivity_hard_decision_controls_passed']}/3 | "
            f"{row['terrain_content_attribution_pass']} | "
            f"{row['hard_decision_content_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Control contrasts",
            "",
            "| architecture | control | dAP | dBrier benefit | dNLL benefit | primary dIoU [95% CI] | sensitivity dIoU | TF pass | primary hard pass |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contrasts:
        ci = row["pooled_foreground_iou_delta_ci95"]
        lines.append(
            f"| {row['architecture']} | {row['control']} | "
            f"{row['aligned_minus_control_average_precision_mean']:+.6f} | "
            f"{row['control_minus_aligned_brier_mean']:+.6f} | "
            f"{row['control_minus_aligned_nll_mean']:+.6f} | "
            f"{row['aligned_minus_control_pooled_foreground_iou']:+.6f} "
            f"[{ci[0]:+.6f},{ci[1]:+.6f}] | "
            f"{row['sensitivity_aligned_minus_control_pooled_foreground_iou']:+.6f} | "
            f"{row['threshold_free_content_pass']} | {row['hard_decision_content_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Release rule",
            "",
            "A Terrain-content claim is released per architecture only when aligned Terrain passes all "
            "three threshold-free controls under the frozen rule. A hard-decision Terrain-content claim "
            "requires the separate three-control IoU gate. Failed gates are retained as boundaries and "
            "must not be repaired by changing thresholds or selecting architectures after outcome review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_manifest(root: Path, outdir: Path) -> dict[str, Any]:
    outputs = []
    for name in REQUIRED_OUTPUTS:
        path = outdir / name
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty output: {path}")
        outputs.append(implementation_record(path, root, "analysis_output"))
    return {
        "schema_version": "1.0",
        "analysis_id": ANALYSIS_ID,
        "created_at_utc": utc_now(),
        "status": "complete",
        "outputs": outputs,
    }


def run(args: argparse.Namespace) -> None:
    require(not args.outdir.exists(), f"output directory already exists: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=False)
    log_path = args.outdir / "run.log"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    started = time.time()
    try:
        log(f"argv={json.dumps(sys.argv, ensure_ascii=True)}")
        log("verifying frozen control protocol and formal-v2 checkpoint/H5 hashes")
        protocol, manifest = verify_protocol(args)
        cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if args.device.startswith("cuda"):
            require(cublas_workspace in {":4096:8", ":16:8"}, "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG")
            require(torch.cuda.is_available(), "CUDA requested but unavailable")
        audit.configure_determinism()
        device = torch.device(args.device)
        test_path = audit.split_path(args.cache_dir, "test")
        obs_idx, obs_names = audit.channel_indices_from_test_h5(test_path, {"observation"})
        terrain_idx, terrain_names = audit.channel_indices_from_test_h5(test_path, {"terrain"})
        first_pair = manifest["checkpoint_pairs"][0]
        require(obs_names == list(first_pair["obs_channel_names"]), "observation channels changed")
        require(terrain_names == list(first_pair["terrain_channel_names"]), "Terrain channels changed")
        base_dataset = audit.H5SupportDataset(test_path, obs_idx, terrain_idx)
        require(len(base_dataset) == EXPECTED_PATCHES, "test patch inventory changed")
        require(len(set(base_dataset.sample_ids)) == EXPECTED_PATCHES, "test sample IDs are not unique")
        dataset = CounterfactualTerrainDataset(base_dataset)
        pair_lookup = {
            (str(pair["architecture_key"]), int(pair["seed"])): pair
            for pair in manifest["checkpoint_pairs"]
        }
        all_samples: list[dict[str, Any]] = []
        all_probability: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        for spec in audit.ARCHITECTURES:
            for seed in EXPECTED_SEEDS:
                pair = pair_lookup[(spec.key, seed)]
                log(f"replay architecture={spec.key} seed={seed}")
                samples, probability, receipt = replay_pair(
                    spec, pair, args.root.resolve(), dataset, device, args.batch_size, args.workers
                )
                all_samples.extend(samples)
                all_probability.extend(probability)
                provenance.append(receipt)
        expected_sample_rows = (
            len(audit.ARCHITECTURES)
            * len(EXPECTED_SEEDS)
            * EXPECTED_PATCHES
            * len(VARIANTS)
            * len(HARD_POLICIES)
        )
        expected_probability_rows = len(audit.ARCHITECTURES) * len(EXPECTED_SEEDS) * len(VARIANTS)
        require(len(all_samples) == expected_sample_rows, f"global sample inventory mismatch: {len(all_samples)}")
        require(len(all_probability) == expected_probability_rows, "global probability inventory mismatch")
        seedmean_samples = build_seedmean_samples(all_samples)
        contrasts, decisions = build_contrasts(
            seedmean_samples, all_probability, all_samples, args.bootstrap
        )
        audit.write_csv(args.outdir / "per_seed_sample_controls.csv", all_samples)
        audit.write_csv(args.outdir / "per_seed_probability_metrics.csv", all_probability)
        audit.write_csv(args.outdir / "architecture_control_contrasts.csv", contrasts)
        audit.write_csv(args.outdir / "decision_summary.csv", decisions)
        execution_receipt = {
            "schema_version": "1.0",
            "analysis_id": ANALYSIS_ID,
            "created_at_utc": utc_now(),
            "status": "complete",
            "argv": list(sys.argv),
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "bootstrap_iterations": int(args.bootstrap),
            "cublas_workspace_config": cublas_workspace,
            "test_h5": {
                "resolved_path": str(test_path.resolve()),
                "relative_path": audit.relative_to_root(test_path.resolve(), args.root.resolve()),
                "sha256": audit.sha256_file(test_path),
                "size_bytes": int(test_path.stat().st_size),
            },
            "analysis_implementation_sha256": audit.sha256_file(Path(__file__)),
            "protocol_sha256": audit.sha256_file(args.protocol_dir / "protocol.json"),
            "freeze_sha256": audit.sha256_file(args.protocol_dir / "freeze.json"),
            "freeze_done_sha256": audit.sha256_file(args.protocol_dir / "DONE.json"),
        }
        audit.write_strict_json(args.outdir / "execution_receipt.json", execution_receipt)
        summary = {
            "schema_version": "1.1",
            "analysis_id": ANALYSIS_ID,
            "created_at_utc": utc_now(),
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "protocol_sha256": audit.sha256_file(args.protocol_dir / "protocol.json"),
            "execution_receipt_sha256": audit.sha256_file(args.outdir / "execution_receipt.json"),
            "bootstrap_iterations": int(args.bootstrap),
            "test_h5_sha256": execution_receipt["test_h5"]["sha256"],
            "primary_threshold_policy": THRESHOLD_POLICY,
            "sensitivity_threshold_policy": SENSITIVITY_THRESHOLD_POLICY,
            "variants": list(VARIANTS),
            "controls": list(CONTROLS),
            "roll_pixels": list(ROLL),
            "n_architectures": len(audit.ARCHITECTURES),
            "n_seeds": len(EXPECTED_SEEDS),
            "n_unique_patches": EXPECTED_PATCHES,
            "provenance": provenance,
            "contrasts": contrasts,
            "decisions": decisions,
            "decision_counts": {
                "terrain_content_attribution_pass": int(sum(bool(row["terrain_content_attribution_pass"]) for row in decisions)),
                "hard_decision_content_pass": int(sum(bool(row["hard_decision_content_pass"]) for row in decisions)),
                "sensitivity_hard_decision_content_pass": int(
                    sum(bool(row["sensitivity_hard_decision_content_pass"]) for row in decisions)
                ),
            },
            "decision_contract": protocol["decision_contract"],
        }
        audit.write_strict_json(args.outdir / "summary.json", summary)
        write_report(args.outdir / "report.md", contrasts, decisions)
        log("all outputs written; generating immutable output manifest")
        manifest_out = output_manifest(args.root.resolve(), args.outdir)
        audit.write_strict_json(args.outdir / "output_manifest.json", manifest_out)
        audit.write_strict_json(
            args.outdir / "DONE.json",
            {
                "analysis_id": ANALYSIS_ID,
                "status": "complete",
                "created_at_utc": utc_now(),
                "summary_sha256": audit.sha256_file(args.outdir / "summary.json"),
                "report_sha256": audit.sha256_file(args.outdir / "report.md"),
                "output_manifest_sha256": audit.sha256_file(args.outdir / "output_manifest.json"),
                "execution_receipt_sha256": audit.sha256_file(args.outdir / "execution_receipt.json"),
            },
        )
        print(f"[done] {args.outdir}")
    except Exception as exc:
        failure = {
            "analysis_id": ANALYSIS_ID,
            "status": "failed",
            "created_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        audit.write_strict_json(args.outdir / "FAILURE.json", failure)
        raise
def main() -> int:
    args = parse_args()
    if args.phase == "freeze":
        freeze_protocol(args)
    elif args.phase == "validate":
        protocol, manifest = verify_protocol(args)
        print(
            f"[valid] {protocol['analysis_id']} pairs={len(protocol['checkpoint_pair_bindings'])} "
            f"manifest_pairs={len(manifest['checkpoint_pairs'])}"
        )
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
