#!/usr/bin/env python3
"""Independent same-threshold error-flow audit for eight L4S architectures.

This script never trains or modifies a checkpoint.  The immutable base audit
used two explicit phases:

1. ``freeze`` discovers the five frozen adapter/visual checkpoint pairs for
   each architecture, validates their identity, and freezes SHA-256 hashes and
   metric definitions.
2. ``run`` verifies every frozen hash before replaying test probabilities and
   computing system-own, frozen-visual, and frozen-adapter threshold policies.

After the base run correctly stopped on an epistemically invalid hint gate,
formal v2 adds an explicit ``amend`` receipt.  It preserves the base freeze and
failure artifacts byte-for-byte, authorizes only the amended audit-script hash,
and makes collaborator rounded hints diagnostic rather than blocking.

The inference unit is a seed/patch pair.  Architecture summaries first average
the five seeds within each of the 800 unique patches; those patches are the
only available clustering units.  The constant ``event_uid`` in the L4S cache
is a split label, not an event identifier, so this audit makes no event-level
inference claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_support_adapter_smp import SmpSupportResidualAdapter
from run_support_adapter_timmfm import TimmSupportResidualAdapter
from run_support_adapter_unet import H5SupportDataset, split_path


ROOT = Path("/mnt/data_hdd/滑坡检测/physics_informed_landslide_dataset")
CACHE = ROOT / "processed/hybrid_pinn/landslide4sense_clean_multispectral_v1"
PROTOCOL_DIR = ROOT / "metadata/protocol_assets/l4s_terrain_same_threshold_audit_20260718"
OUTDIR = ROOT / "experiments/revision2026/l4s_terrain_same_threshold_audit_20260718"
AUDIT_ID = "l4s_terrain_same_threshold_audit_20260718"
FORMAL_V2_AUDIT_ID = "l4s_terrain_same_threshold_audit_formal_v2_20260718"
AMENDMENT_DIR = (
    ROOT
    / "metadata/protocol_assets/"
    "l4s_terrain_same_threshold_audit_20260718_post_failure_amendment_v2"
)
FORMAL_V2_OUTDIR = (
    ROOT / "experiments/revision2026/l4s_terrain_same_threshold_audit_formal_v2_20260718"
)
EXPECTED_SEEDS = (20260622, 20260623, 20260624, 20260625, 20260626)
EXPECTED_PATCHES = 800
POLICIES = ("system_own", "frozen_visual", "frozen_adapter")

IMMUTABLE_BASE_HASHES = {
    "metadata/protocol_assets/l4s_terrain_same_threshold_audit_20260718/DONE.json":
        "4e99c16c205e43ffa55df1e3b79f02cad9f80dfc4060819912eb0f25dd480f6c",
    "metadata/protocol_assets/l4s_terrain_same_threshold_audit_20260718/freeze.json":
        "052c72975f397cc86ea2e306c07be0213cb9b8b80b4b897a9d7c535ef03828bb",
    "metadata/protocol_assets/l4s_terrain_same_threshold_audit_20260718/input_manifest.json":
        "f24ac2d7e41c8804094d29a74c830f86a351ea3cf990a6c63e73e079fdc914be",
    "metadata/protocol_assets/l4s_terrain_same_threshold_audit_20260718/protocol.json":
        "f3ae16da5ce180c3367ca1d1522ed7b4aba19bef5fa729a9e3a3c318749cd236",
    "metadata/protocol_assets/l4s_terrain_same_threshold_audit_20260718/threshold_sources.csv":
        "38c5aca7a8508cad92b6b8b2b5b023e0c341840a0408596679dd2d2d276ad7c2",
    "experiments/revision2026/l4s_terrain_same_threshold_audit_20260718/DONE.json":
        "73fd9e172ee8ac4e5ee0c2c7cfbe35e9bdf115804f3d647542145b79f20f7149",
    "experiments/revision2026/l4s_terrain_same_threshold_audit_20260718/FAILURE.json":
        "372c0b42cedb4e574b5308d3e8246eed4189ed6e359218bb1487a652a06de4ce",
    "experiments/revision2026/l4s_terrain_same_threshold_audit_20260718/run.log":
        "c6011ea2506356d62d246623cfad762bccb933196699dc355f59fbb55901c4e5",
}

PRELIMINARY_DETERMINISTIC_DISCREPANCIES = {
    "hiera_s_mae": {
        "formal_replay_percent": -5.374401703589944,
        "legacy_count_table_percent": -5.376125830318516,
        "collaborator_rounded_hint_percent": -5.38,
        "formal_minus_hint_percentage_points": 0.005598296410055825,
        "formal_visual_error_pixels": 1507405,
        "formal_net_corrected_pixels": -81014,
        "legacy_visual_error_pixels": 1507554,
        "legacy_net_corrected_pixels": -81048,
    },
    "deeplabv3plus": {
        "formal_replay_percent": -0.05313573695410529,
        "legacy_count_table_percent": -0.05879287766282028,
        "collaborator_rounded_hint_percent": -0.06,
        "formal_minus_hint_percentage_points": 0.006864263045894707,
        "formal_visual_error_pixels": 1166823,
        "formal_net_corrected_pixels": -620,
        "legacy_visual_error_pixels": 1166808,
        "legacy_net_corrected_pixels": -686,
    },
}


class AuditError(RuntimeError):
    """Raised when an audit invariant fails closed."""


@dataclass(frozen=True)
class ArchitectureSpec:
    key: str
    label: str
    family: str
    run_dir: str
    expected_identity: Mapping[str, Any]
    collaborator_hint_same_adapter_percent: float


ARCHITECTURES = (
    ArchitectureSpec(
        "dinov2_s",
        "DINOv2-S",
        "foundation_model",
        "experiments/revision2026/r3_11_backbone_sensitivity/"
        "l4s_r3_11_dinov2_small_adapter_alpha3_frozen_e20",
        {"backend": "timm", "backbone": "vit_small_patch14_dinov2.lvd142m"},
        6.33,
    ),
    ArchitectureSpec(
        "fcmae_convnextv2_t",
        "FCMAE-ConvNeXtV2-T",
        "foundation_model",
        "experiments/revision2026/r3_11_backbone_sensitivity/"
        "l4s_r3_11_fcmae_convnextv2_tiny_adapter_alpha3_frozen_e20",
        {"backend": "timm", "backbone": "convnextv2_tiny.fcmae"},
        5.95,
    ),
    ArchitectureSpec(
        "hiera_s_mae",
        "Hiera-S MAE",
        "foundation_model",
        "experiments/revision2026/r3_11_backbone_sensitivity/"
        "l4s_r3_11_hiera_small_mae_adapter_alpha3_frozen_e20",
        {"backend": "timm", "backbone": "hiera_small_224.mae"},
        -5.38,
    ),
    ArchitectureSpec(
        "satmae_vit_b",
        "SatMAE-ViT-B",
        "foundation_model",
        "experiments/revision2026/r3_11_backbone_sensitivity/"
        "l4s_r3_11_satmae_vitbase_multispec_adapter_alpha3_frozen_e20",
        {"backend": "satmae", "backbone": "MVRL/satmae-vitbase-multispec-pretrain"},
        -5.52,
    ),
    ArchitectureSpec(
        "deeplabv3plus",
        "DeepLabV3+",
        "modern_bn_frozen",
        "experiments/revision2026/l4s_modern_bn_frozen_e20_20260715/"
        "l4s_support_adapter_deeplabv3plus_bnfrozen_e20",
        {"architecture": "deeplabv3plus", "encoder": "resnet50", "encoder_weights": None},
        -0.06,
    ),
    ArchitectureSpec(
        "unetplusplus",
        "U-Net++",
        "modern_bn_frozen",
        "experiments/revision2026/l4s_modern_bn_frozen_e20_20260715/"
        "l4s_support_adapter_unetplusplus_bnfrozen_e20",
        {"architecture": "unetplusplus", "encoder": "resnet50", "encoder_weights": None},
        2.64,
    ),
    ArchitectureSpec(
        "fpn",
        "FPN",
        "modern_bn_frozen",
        "experiments/revision2026/l4s_modern_bn_frozen_e20_20260715/"
        "l4s_support_adapter_fpn_bnfrozen_e20",
        {"architecture": "fpn", "encoder": "resnet50", "encoder_weights": None},
        6.03,
    ),
    ArchitectureSpec(
        "deeplabv3plus_imagenet",
        "DeepLabV3+ ImageNet",
        "modern_bn_frozen",
        "experiments/revision2026/l4s_modern_bn_frozen_e20_20260715/"
        "l4s_support_adapter_deeplabv3plus_imagenet_bnfrozen_e20",
        {"architecture": "deeplabv3plus", "encoder": "resnet50", "encoder_weights": "imagenet"},
        1.78,
    ),
)


METRIC_DEFINITIONS: dict[str, Any] = {
    "binary_rule": "prediction = (probability >= threshold); target = (mask >= 0.5)",
    "threshold_policies": {
        "system_own": {
            "visual": "matched observation checkpoint validation-selected threshold",
            "adapter": "adapter checkpoint validation-selected threshold",
            "same_threshold": False,
            "interpretation": "system comparison only; not pure Terrain attribution",
        },
        "frozen_visual": {
            "visual": "matched observation checkpoint validation-selected threshold",
            "adapter": "same matched observation threshold",
            "same_threshold": True,
            "interpretation": "Terrain intervention at frozen visual threshold",
        },
        "frozen_adapter": {
            "visual": "adapter checkpoint validation-selected threshold",
            "adapter": "same adapter threshold",
            "same_threshold": True,
            "interpretation": "Terrain intervention at frozen adapter threshold",
        },
    },
    "threshold_origin": {
        "selection_split": "validation",
        "selection_objective": "pooled foreground IoU",
        "grid": "0.05 through 0.95 inclusive in steps of 0.01",
        "audit_behavior": "read stored threshold; never tune on test",
    },
    "flow": {
        "E2C": "visual prediction wrong and adapter prediction correct",
        "C2E": "visual prediction correct and adapter prediction wrong",
        "net_corrected_pixels": "E2C - C2E = visual error pixels - adapter error pixels",
        "E2C_fraction_visual_errors": "E2C / visual error pixels",
        "C2E_fraction_visual_correct": "C2E / visual correct pixels",
        "net_error_reduction_fraction": "(E2C - C2E) / visual error pixels",
    },
    "foreground_iou": "TP / (TP + FP + FN); zero when the denominator is zero",
    "average_precision": (
        "exact non-interpolated pixel-pooled AP per seed over all 800 patches; architecture value is "
        "the descriptive mean of five per-seed AP values"
    ),
    "brier": "mean((probability - target)^2); lower is better",
    "nll": "mean binary cross-entropy from logits; lower is better",
    "aggregation": (
        "For threshold flows, Brier, and NLL, average five seeds within each sample first, then use "
        "800 unique patches as clusters. Ratio point estimates use sums of seed-mean counts across "
        "patches. Bootstrap resamples patches. AP is non-additive and is reported per seed, not as "
        "patch-cluster inference."
    ),
    "inference_scope": (
        "800 unique patches are the current clustering units. event_uid is constant "
        "L4S_test_official and is not an event identifier; no event-level inference is claimed."
    ),
}


FLOW_COUNT_FIELDS = (
    "pixel_count",
    "positive_pixels",
    "visual_tp",
    "visual_fp",
    "visual_fn",
    "visual_tn",
    "adapter_tp",
    "adapter_fp",
    "adapter_fn",
    "adapter_tn",
    "visual_error_pixels",
    "visual_correct_pixels",
    "adapter_error_pixels",
    "adapter_correct_pixels",
    "e2c_pixels",
    "c2e_pixels",
    "net_corrected_pixels",
    "disagreement_pixels",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_jsonable(value: Any) -> Any:
    """Convert values to RFC-compliant JSON data, mapping non-finite numbers to null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return strict_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [strict_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return strict_jsonable(asdict(value))
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def strict_json_dumps(value: Any, *, indent: int = 2) -> str:
    return json.dumps(
        strict_jsonable(value),
        ensure_ascii=True,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def load_strict_json(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def write_strict_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strict_json_dumps(value) + "\n", encoding="utf-8")
    load_strict_json(path)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AuditError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise AuditError(f"CSV schema mismatch at row {index} for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError as exc:
        raise AuditError(f"asset is outside repository root: {resolved}") from exc


def resolve_recorded_path(raw_path: str | Path, root: Path, checkpoint_path: Path) -> Path:
    recorded = Path(raw_path).expanduser()
    candidates = [recorded] if recorded.is_absolute() else [
        root / recorded,
        root.parent / recorded,
        checkpoint_path.parent / recorded,
        Path.cwd() / recorded,
    ]
    existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_file():
            existing.append(resolved)
            seen.add(resolved)
    if not existing:
        raise AuditError(
            f"missing baseline checkpoint recorded as {recorded}; attempted "
            + ", ".join(str(path.resolve()) for path in candidates)
        )
    if len(existing) != 1:
        raise AuditError(f"ambiguous baseline path {recorded}: {[str(path) for path in existing]}")
    return existing[0]


def assert_visual_state_identical(
    adapter_checkpoint: Mapping[str, Any], baseline_checkpoint: Mapping[str, Any], context: str
) -> int:
    adapter_state = adapter_checkpoint.get("visual_state_dict")
    baseline_state = baseline_checkpoint.get("visual_state_dict")
    if not isinstance(adapter_state, Mapping) or not isinstance(baseline_state, Mapping):
        raise AuditError(f"missing visual_state_dict: {context}")
    if set(adapter_state) != set(baseline_state):
        missing = sorted(set(baseline_state) - set(adapter_state))[:10]
        extra = sorted(set(adapter_state) - set(baseline_state))[:10]
        raise AuditError(f"visual-state key mismatch {context}: missing={missing}, extra={extra}")
    for key in adapter_state:
        left = adapter_state[key]
        right = baseline_state[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AuditError(f"non-tensor visual state {context}: {key}")
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            delta = None
            if left.shape == right.shape:
                delta = float((left.float() - right.float()).abs().max().item())
            raise AuditError(f"visual state changed {context}: key={key}, max_abs_diff={delta}")
    return len(adapter_state)


def validate_checkpoint_pair(
    spec: ArchitectureSpec,
    adapter_path: Path,
    root: Path,
) -> dict[str, Any]:
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if adapter.get("mode") != "adapter":
        raise AuditError(f"not an adapter checkpoint: {adapter_path}")
    seed = int(adapter.get("seed", -1))
    if seed not in EXPECTED_SEEDS:
        raise AuditError(f"unexpected seed {seed}: {adapter_path}")
    if adapter.get("visual_train_scope") != "frozen":
        raise AuditError(f"visual branch is not frozen: {adapter_path}")
    for key, expected in spec.expected_identity.items():
        if adapter.get(key) != expected:
            raise AuditError(
                f"checkpoint identity mismatch {adapter_path}: {key}={adapter.get(key)!r}, expected={expected!r}"
            )
    if spec.family == "modern_bn_frozen":
        if adapter.get("freeze_visual_state") is not True:
            raise AuditError(f"modern checkpoint lacks freeze_visual_state=true: {adapter_path}")
        if adapter.get("visual_state_sha256_initial") != adapter.get("visual_state_sha256_final"):
            raise AuditError(f"modern visual-state training hashes differ: {adapter_path}")
    baseline_path = resolve_recorded_path(adapter.get("baseline_ckpt", ""), root, adapter_path)
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    if baseline.get("mode") != "observation" or int(baseline.get("seed", -1)) != seed:
        raise AuditError(f"baseline mode/seed mismatch for {adapter_path}: {baseline_path}")
    for key, expected in spec.expected_identity.items():
        if baseline.get(key) != expected:
            raise AuditError(
                f"baseline identity mismatch {baseline_path}: {key}={baseline.get(key)!r}, expected={expected!r}"
            )
    visual_tensor_count = assert_visual_state_identical(adapter, baseline, f"{spec.key}/seed{seed}")
    adapter_threshold = float(adapter.get("threshold", math.nan))
    visual_threshold = float(baseline.get("threshold", math.nan))
    if not (0.0 < adapter_threshold < 1.0 and 0.0 < visual_threshold < 1.0):
        raise AuditError(f"invalid thresholds for {spec.key}/seed{seed}")
    obs_names = [str(item) for item in adapter.get("obs_channel_names", [])]
    terrain_names = [str(item) for item in adapter.get("terrain_channel_names", [])]
    if not obs_names or not terrain_names:
        raise AuditError(f"missing channel names in {adapter_path}")
    if obs_names != [str(item) for item in baseline.get("obs_channel_names", [])]:
        raise AuditError(f"observation-channel mismatch: {adapter_path}")
    pair = {
        "architecture_key": spec.key,
        "architecture": spec.label,
        "family": spec.family,
        "seed": seed,
        "adapter_checkpoint": relative_to_root(adapter_path, root),
        "visual_checkpoint": relative_to_root(baseline_path, root),
        "adapter_threshold": adapter_threshold,
        "visual_threshold": visual_threshold,
        "thresholds_equal": math.isclose(adapter_threshold, visual_threshold, rel_tol=0.0, abs_tol=1e-12),
        "adapter_threshold_source": "adapter checkpoint validation-selected threshold",
        "visual_threshold_source": "matched observation checkpoint validation-selected threshold",
        "visual_state_identical": True,
        "visual_state_tensor_count": visual_tensor_count,
        "obs_channel_names": obs_names,
        "terrain_channel_names": terrain_names,
        "checkpoint_identity": {key: adapter.get(key) for key in spec.expected_identity},
    }
    del adapter, baseline
    gc.collect()
    return pair


def inspect_test_cache(cache_dir: Path) -> dict[str, Any]:
    test_path = split_path(cache_dir, "test")
    if not test_path.is_file():
        raise AuditError(f"missing test cache: {test_path}")
    with h5py.File(test_path, "r") as handle:
        required = {"x", "mask", "sample_id", "event_uid", "channel_names", "channel_groups"}
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise AuditError(f"test H5 missing datasets: {missing}")
        sample_ids = [item.decode() if isinstance(item, bytes) else str(item) for item in handle["sample_id"][:]]
        event_uids = [item.decode() if isinstance(item, bytes) else str(item) for item in handle["event_uid"][:]]
        x_shape = list(handle["x"].shape)
        mask_shape = list(handle["mask"].shape)
    if len(sample_ids) != EXPECTED_PATCHES or len(set(sample_ids)) != EXPECTED_PATCHES:
        raise AuditError(
            f"expected {EXPECTED_PATCHES} unique test patches, found rows={len(sample_ids)} unique={len(set(sample_ids))}"
        )
    return {
        "test_h5": relative_to_root(test_path, ROOT),
        "x_shape": x_shape,
        "mask_shape": mask_shape,
        "n_rows": len(sample_ids),
        "n_unique_sample_ids": len(set(sample_ids)),
        "n_unique_event_uid_values": len(set(event_uids)),
        "event_uid_values": sorted(set(event_uids)),
        "event_uid_usable_for_event_inference": False,
        "cluster_unit": "unique patch/sample_id",
    }


def channel_indices_from_test_h5(
    test_path: Path, groups_wanted: set[str]
) -> tuple[list[int], list[str]]:
    """Resolve channel roles from the frozen test H5 without opening train data."""
    with h5py.File(test_path, "r") as handle:
        names = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle["channel_names"][:]
        ]
        groups = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle["channel_groups"][:]
        ]
    indices = [index for index, group in enumerate(groups) if group in groups_wanted]
    if not indices:
        raise AuditError(f"no channels found for groups={sorted(groups_wanted)} in {test_path}")
    return indices, [names[index] for index in indices]


def discover_pairs(root: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for spec in ARCHITECTURES:
        run_dir = root / spec.run_dir
        checkpoints = sorted((run_dir / "checkpoints").glob("*_adapter_seed*.pt"))
        if len(checkpoints) != len(EXPECTED_SEEDS):
            raise AuditError(
                f"expected five adapter checkpoints for {spec.key}, found {len(checkpoints)} in {run_dir}"
            )
        spec_pairs = [validate_checkpoint_pair(spec, path, root) for path in checkpoints]
        seeds = tuple(sorted(int(pair["seed"]) for pair in spec_pairs))
        if seeds != EXPECTED_SEEDS:
            raise AuditError(f"seed inventory mismatch for {spec.key}: {seeds}")
        pairs.extend(spec_pairs)
    obs_inventories = {tuple(pair["obs_channel_names"]) for pair in pairs}
    terrain_inventories = {tuple(pair["terrain_channel_names"]) for pair in pairs}
    if len(obs_inventories) != 1 or len(terrain_inventories) != 1:
        raise AuditError("checkpoint channel inventories differ across architecture/seed pairs")
    return pairs


def asset_record(path: Path, root: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing asset: {path}")
    stat = path.stat()
    print(f"[hash] {role}: {relative_to_root(path, root)}", flush=True)
    return {
        "path": relative_to_root(path, root),
        "role": role,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "numpy",
        "h5py",
        "scikit-learn",
        "timm",
        "segmentation-models-pytorch",
        "transformers",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def freeze_inputs(root: Path, cache_dir: Path, protocol_dir: Path) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve():
        raise AuditError(f"formal freeze requires root {ROOT}, received {root}")
    cache_info = inspect_test_cache(cache_dir)
    pairs = discover_pairs(root)
    computational_assets: dict[str, tuple[Path, str]] = {}
    data_assets = (
        (split_path(cache_dir, "test"), "test_h5"),
        (cache_dir / "channel_role_registry.csv", "channel_role_registry"),
        (root / "scripts/run_support_adapter_unet.py", "dataset_and_threshold_implementation"),
        (root / "scripts/run_support_adapter_smp.py", "modern_model_implementation"),
        (root / "scripts/run_support_adapter_timmfm.py", "foundation_model_implementation"),
        (Path(__file__).resolve(), "audit_implementation"),
    )
    for path, role in data_assets:
        computational_assets[str(path.resolve())] = (path.resolve(), role)
    for pair in pairs:
        adapter_path = root / str(pair["adapter_checkpoint"])
        visual_path = root / str(pair["visual_checkpoint"])
        computational_assets[str(adapter_path.resolve())] = (adapter_path.resolve(), "adapter_checkpoint")
        computational_assets[str(visual_path.resolve())] = (visual_path.resolve(), "visual_checkpoint")
    assets = [
        asset_record(path, root, role)
        for _, (path, role) in sorted(computational_assets.items(), key=lambda item: item[0])
    ]

    legacy_paths = (
        root
        / "experiments/revision2026/l4s_fm_terrain_attribution_20260715/"
        "per_seed_sample_counterfactuals.csv",
        root
        / "experiments/revision2026/l4s_modern_bn_frozen_attribution_20260715/"
        "per_seed_sample_counterfactuals.csv",
    )
    legacy_records = []
    probability_candidates: list[str] = []
    for path in legacy_paths:
        if not path.is_file():
            raise AuditError(f"missing legacy discovery artifact: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        legacy_records.append(
            {
                **asset_record(path, root, "legacy_count_counterfactual_not_computational_input"),
                "columns": header,
                "contains_raw_probabilities": False,
                "audit_use": "discovery and legacy-bug diagnosis only; no metric is read from this file",
            }
        )
        parent = path.parent
        for suffix in ("*.npy", "*.npz", "*.h5", "*.hdf5", "*prob*.pt"):
            probability_candidates.extend(relative_to_root(item, root) for item in parent.glob(suffix))

    protocol = {
        "schema_version": "1.0",
        "audit_id": AUDIT_ID,
        "created_at_utc": utc_now(),
        "status": "frozen",
        "read_only_inference": True,
        "training_performed": False,
        "architectures": [asdict(spec) for spec in ARCHITECTURES],
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_unique_patches": EXPECTED_PATCHES,
        "split": "test",
        "metric_definitions": METRIC_DEFINITIONS,
        "fail_closed_rule": (
            "Any missing probability-replay asset, hash mismatch, checkpoint identity mismatch, duplicate "
            "seed/sample row, non-finite JSON token, or incomplete 5x800 inventory aborts completion. "
            "Summary values are never reverse-engineered from legacy aggregates."
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "audit_id": AUDIT_ID,
        "created_at_utc": utc_now(),
        "root": str(root.resolve()),
        "cache": cache_info,
        "checkpoint_pairs": pairs,
        "computational_inputs": assets,
        "legacy_discovery_artifacts_not_used_for_metrics": legacy_records,
        "legacy_probability_assets_found": sorted(set(probability_candidates)),
        "probability_replay_source": (
            "No reusable per-pixel probability asset exists in the two legacy attribution directories. "
            "Probabilities are replayed from frozen checkpoints and the frozen test H5."
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
    }
    threshold_rows = [
        {
            "architecture_key": pair["architecture_key"],
            "architecture": pair["architecture"],
            "family": pair["family"],
            "seed": pair["seed"],
            "visual_threshold": pair["visual_threshold"],
            "adapter_threshold": pair["adapter_threshold"],
            "thresholds_equal": pair["thresholds_equal"],
            "visual_threshold_source": pair["visual_threshold_source"],
            "adapter_threshold_source": pair["adapter_threshold_source"],
            "visual_checkpoint": pair["visual_checkpoint"],
            "adapter_checkpoint": pair["adapter_checkpoint"],
        }
        for pair in pairs
    ]
    protocol_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = protocol_dir / "protocol.json"
    manifest_path = protocol_dir / "input_manifest.json"
    thresholds_path = protocol_dir / "threshold_sources.csv"
    write_strict_json(protocol_path, protocol)
    write_strict_json(manifest_path, manifest)
    write_csv(thresholds_path, threshold_rows)
    freeze = {
        "schema_version": "1.0",
        "audit_id": AUDIT_ID,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "threshold_sources_sha256": sha256_file(thresholds_path),
        "n_computational_inputs": len(assets),
        "n_checkpoint_pairs": len(pairs),
        "n_architectures": len(ARCHITECTURES),
        "n_seeds_per_architecture": len(EXPECTED_SEEDS),
        "n_unique_patches": EXPECTED_PATCHES,
    }
    write_strict_json(protocol_dir / "freeze.json", freeze)
    write_strict_json(
        protocol_dir / "DONE.json",
        {
            "audit_id": AUDIT_ID,
            "phase": "input_freeze",
            "status": "complete",
            "created_at_utc": utc_now(),
            "freeze_sha256": sha256_file(protocol_dir / "freeze.json"),
        },
    )
    return freeze


def verify_immutable_base_hashes(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative_path, expected in IMMUTABLE_BASE_HASHES.items():
        path = root / relative_path
        if not path.is_file():
            raise AuditError(f"missing immutable base artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise AuditError(
                f"immutable base artifact changed: {relative_path}; expected={expected}, actual={actual}"
            )
        records.append({"path": relative_path, "sha256": actual})
    return records


def load_verified_amendment(
    root: Path, amendment_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "amendment": amendment_dir / "amendment.json",
        "diagnostic": amendment_dir / "diagnostic_discrepancy.json",
        "receipt": amendment_dir / "receipt.json",
        "done": amendment_dir / "DONE.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise AuditError(f"missing post-failure amendment assets: {missing}")
    receipt = load_strict_json(required["receipt"])
    expected = {
        "amendment": receipt.get("amendment_sha256"),
        "diagnostic": receipt.get("diagnostic_discrepancy_sha256"),
    }
    for key in ("amendment", "diagnostic"):
        actual = sha256_file(required[key])
        if actual != expected[key]:
            raise AuditError(f"post-failure {key} hash mismatch")
    done = load_strict_json(required["done"])
    if sha256_file(required["receipt"]) != done.get("receipt_sha256"):
        raise AuditError("post-failure receipt hash mismatch")
    amendment = load_strict_json(required["amendment"])
    if amendment.get("base_freeze_sha256") != IMMUTABLE_BASE_HASHES[
        "metadata/protocol_assets/l4s_terrain_same_threshold_audit_20260718/freeze.json"
    ]:
        raise AuditError("amendment does not point to the immutable base freeze")
    current_script_hash = sha256_file(Path(__file__).resolve())
    if current_script_hash != amendment.get("amended_audit_script_sha256"):
        raise AuditError("current audit implementation is not the amendment-authorized script")
    verify_immutable_base_hashes(root)
    return amendment, receipt


def create_post_failure_amendment(
    root: Path,
    protocol_dir: Path,
    amendment_dir: Path,
    formal_outdir: Path,
) -> dict[str, Any]:
    base_records = verify_immutable_base_hashes(root)
    freeze_path = protocol_dir / "freeze.json"
    manifest_path = protocol_dir / "input_manifest.json"
    freeze = load_strict_json(freeze_path)
    manifest = load_strict_json(manifest_path)
    if freeze.get("input_manifest_sha256") != sha256_file(manifest_path):
        raise AuditError("immutable base manifest no longer matches base freeze")
    audit_assets = [
        asset
        for asset in manifest.get("computational_inputs", [])
        if asset.get("role") == "audit_implementation"
    ]
    if len(audit_assets) != 1:
        raise AuditError(f"expected one frozen audit implementation, found {len(audit_assets)}")
    frozen_audit_asset = audit_assets[0]
    current_script_path = Path(__file__).resolve()
    current_script_hash = sha256_file(current_script_path)
    if current_script_hash == frozen_audit_asset["sha256"]:
        raise AuditError("post-failure amendment requires a distinct amended audit implementation")

    verified_non_audit_inputs = 0
    for asset in manifest.get("computational_inputs", []):
        if asset.get("role") == "audit_implementation":
            continue
        path = root / str(asset["path"])
        if not path.is_file():
            raise AuditError(f"missing frozen computational input during amendment: {path}")
        if path.stat().st_size != int(asset["size_bytes"]):
            raise AuditError(f"frozen computational input size changed during amendment: {path}")
        print(f"[amend-verify-hash] {asset['path']}", flush=True)
        if sha256_file(path) != asset["sha256"]:
            raise AuditError(f"frozen computational input changed during amendment: {path}")
        verified_non_audit_inputs += 1

    amendment = {
        "schema_version": "1.0",
        "amendment_id": "l4s_terrain_same_threshold_audit_post_failure_v2_20260718",
        "created_at_utc": utc_now(),
        "status": "authorized",
        "base_audit_id": AUDIT_ID,
        "formal_v2_audit_id": FORMAL_V2_AUDIT_ID,
        "base_protocol_directory": relative_to_root(protocol_dir, root),
        "base_freeze_sha256": sha256_file(freeze_path),
        "base_input_manifest_sha256": sha256_file(manifest_path),
        "base_freeze_mutated": False,
        "base_failure_directory_mutated": False,
        "immutable_base_artifacts": base_records,
        "frozen_audit_script_path": frozen_audit_asset["path"],
        "frozen_audit_script_sha256": frozen_audit_asset["sha256"],
        "amended_audit_script_path": relative_to_root(current_script_path, root),
        "amended_audit_script_sha256": current_script_hash,
        "verified_unchanged_non_audit_computational_inputs": verified_non_audit_inputs,
        "formal_v2_output_directory": relative_to_root(formal_outdir, root),
        "authorized_protocol_changes": [
            {
                "change": "collaborator rounded hints become diagnostic discrepancies only",
                "reason": (
                    "Hints are legacy rounded observations, not frozen scientific inputs. A mismatch must "
                    "be recorded but cannot override complete checkpoint/H5 inference."
                ),
                "blocking": False,
            },
            {
                "change": "CUDA inference requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8",
                "reason": "Make Hiera scaled-dot-product attention reproducible under strict deterministic mode.",
                "blocking": True,
            },
            {
                "change": "publish complete formal inference in a new v2 output directory",
                "reason": "Preserve the original failed audit and its FAILURE.json as an immutable receipt.",
                "blocking": True,
            },
        ],
        "unchanged_scientific_contract": {
            "checkpoint_pairs": 40,
            "architectures": 8,
            "seeds_per_architecture": 5,
            "unique_patch_clusters": 800,
            "threshold_policies": list(POLICIES),
            "threshold_free_metrics": ["average_precision", "brier", "nll"],
            "input_checkpoint_and_h5_hashes": "must match immutable base manifest",
            "event_inference_claimed": False,
        },
    }
    diagnostic = {
        "schema_version": "1.0",
        "amendment_id": amendment["amendment_id"],
        "created_at_utc": utc_now(),
        "status": "diagnostic_only",
        "scientific_input": False,
        "blocking": False,
        "metric": "frozen_adapter net error reduction percent",
        "preliminary_deterministic_replays": PRELIMINARY_DETERMINISTIC_DISCREPANCIES,
        "diagnosis": (
            "The collaborator hints exactly reflect rounding of legacy count-level counterfactual tables. "
            "Deterministic checkpoint replay changes only a tiny number of threshold-boundary pixels; both "
            "affected architecture directions remain negative."
        ),
        "formal_v2_rule": (
            "Publish the complete frozen-checkpoint result. Record formal-minus-hint discrepancy without "
            "using the hint as a pass/fail criterion."
        ),
    }
    amendment_dir.mkdir(parents=True, exist_ok=True)
    amendment_path = amendment_dir / "amendment.json"
    diagnostic_path = amendment_dir / "diagnostic_discrepancy.json"
    write_strict_json(amendment_path, amendment)
    write_strict_json(diagnostic_path, diagnostic)
    receipt = {
        "schema_version": "1.0",
        "amendment_id": amendment["amendment_id"],
        "created_at_utc": utc_now(),
        "status": "complete",
        "amendment_sha256": sha256_file(amendment_path),
        "diagnostic_discrepancy_sha256": sha256_file(diagnostic_path),
        "base_freeze_sha256": sha256_file(freeze_path),
        "base_failure_json_sha256": IMMUTABLE_BASE_HASHES[
            "experiments/revision2026/l4s_terrain_same_threshold_audit_20260718/FAILURE.json"
        ],
        "amended_audit_script_sha256": current_script_hash,
        "all_non_audit_frozen_inputs_verified": True,
        "hints_are_nonblocking_diagnostics": True,
    }
    receipt_path = amendment_dir / "receipt.json"
    write_strict_json(receipt_path, receipt)
    write_strict_json(
        amendment_dir / "DONE.json",
        {
            "schema_version": "1.0",
            "amendment_id": amendment["amendment_id"],
            "created_at_utc": utc_now(),
            "status": "complete",
            "receipt_sha256": sha256_file(receipt_path),
            "base_freeze_mutated": False,
            "base_failure_directory_mutated": False,
        },
    )
    return receipt


def verify_freeze(
    root: Path,
    protocol_dir: Path,
    amendment_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    required = {
        "protocol": protocol_dir / "protocol.json",
        "manifest": protocol_dir / "input_manifest.json",
        "thresholds": protocol_dir / "threshold_sources.csv",
        "freeze": protocol_dir / "freeze.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise AuditError(f"missing frozen protocol assets: {missing}")
    freeze = load_strict_json(required["freeze"])
    expected_hashes = {
        "protocol": freeze.get("protocol_sha256"),
        "manifest": freeze.get("input_manifest_sha256"),
        "thresholds": freeze.get("threshold_sources_sha256"),
    }
    for key in ("protocol", "manifest", "thresholds"):
        actual = sha256_file(required[key])
        if actual != expected_hashes[key]:
            raise AuditError(f"frozen {key} hash mismatch: expected={expected_hashes[key]}, actual={actual}")
    manifest = load_strict_json(required["manifest"])
    if manifest.get("root") != str(root.resolve()):
        raise AuditError(f"frozen root mismatch: {manifest.get('root')} vs {root.resolve()}")
    amendment: dict[str, Any] | None = None
    if amendment_dir is not None:
        amendment, _ = load_verified_amendment(root, amendment_dir)
    for asset in manifest.get("computational_inputs", []):
        path = root / str(asset["path"])
        if not path.is_file():
            raise AuditError(f"missing frozen computational input: {path}")
        print(f"[verify-hash] {asset['path']}", flush=True)
        actual = sha256_file(path)
        if asset.get("role") == "audit_implementation" and amendment is not None:
            if asset["sha256"] != amendment.get("frozen_audit_script_sha256"):
                raise AuditError("amendment frozen-script hash does not match base manifest")
            if actual != amendment.get("amended_audit_script_sha256"):
                raise AuditError("audit implementation does not match authorized amendment hash")
            continue
        stat = path.stat()
        if int(stat.st_size) != int(asset["size_bytes"]):
            raise AuditError(f"size changed for frozen input: {path}")
        if actual != asset["sha256"]:
            raise AuditError(f"SHA-256 changed for frozen input: {path}")
    return freeze, manifest, amendment


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def foreground_iou(tp: float, fp: float, fn: float) -> float:
    denominator = tp + fp + fn
    return float(tp / denominator) if denominator > 0 else 0.0


def compute_flow_counts(
    target: np.ndarray,
    visual_probability: np.ndarray,
    adapter_probability: np.ndarray,
    visual_threshold: float,
    adapter_threshold: float,
) -> dict[str, Any]:
    """Pure NumPy reference implementation used by the runtime and regression tests."""
    target_bool = np.asarray(target) >= 0.5
    visual_pred = np.asarray(visual_probability) >= float(visual_threshold)
    adapter_pred = np.asarray(adapter_probability) >= float(adapter_threshold)
    if target_bool.shape != visual_pred.shape or target_bool.shape != adapter_pred.shape:
        raise AuditError("flow arrays must have identical shapes")
    visual_correct = visual_pred == target_bool
    adapter_correct = adapter_pred == target_bool
    e2c = (~visual_correct) & adapter_correct
    c2e = visual_correct & (~adapter_correct)

    def count(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    row: dict[str, Any] = {
        "pixel_count": int(target_bool.size),
        "positive_pixels": count(target_bool),
        "visual_tp": count(visual_pred & target_bool),
        "visual_fp": count(visual_pred & (~target_bool)),
        "visual_fn": count((~visual_pred) & target_bool),
        "visual_tn": count((~visual_pred) & (~target_bool)),
        "adapter_tp": count(adapter_pred & target_bool),
        "adapter_fp": count(adapter_pred & (~target_bool)),
        "adapter_fn": count((~adapter_pred) & target_bool),
        "adapter_tn": count((~adapter_pred) & (~target_bool)),
        "visual_error_pixels": count(~visual_correct),
        "visual_correct_pixels": count(visual_correct),
        "adapter_error_pixels": count(~adapter_correct),
        "adapter_correct_pixels": count(adapter_correct),
        "e2c_pixels": count(e2c),
        "c2e_pixels": count(c2e),
        "net_corrected_pixels": count(e2c) - count(c2e),
        "disagreement_pixels": count(visual_pred != adapter_pred),
    }
    validate_flow_invariants(row)
    row.update(flow_rates(row))
    return row


def validate_flow_invariants(row: Mapping[str, Any]) -> None:
    visual_error = int(row["visual_error_pixels"])
    adapter_error = int(row["adapter_error_pixels"])
    e2c = int(row["e2c_pixels"])
    c2e = int(row["c2e_pixels"])
    net = int(row["net_corrected_pixels"])
    disagreement = int(row["disagreement_pixels"])
    if net != e2c - c2e:
        raise AuditError("net flow invariant failed")
    if adapter_error != visual_error - e2c + c2e:
        raise AuditError("error-balance invariant failed")
    if disagreement != e2c + c2e:
        raise AuditError("disagreement invariant failed")


def flow_rates(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "e2c_fraction_visual_errors": safe_ratio(
            float(row["e2c_pixels"]), float(row["visual_error_pixels"])
        ),
        "c2e_fraction_visual_correct": safe_ratio(
            float(row["c2e_pixels"]), float(row["visual_correct_pixels"])
        ),
        "net_error_reduction_fraction": safe_ratio(
            float(row["net_corrected_pixels"]), float(row["visual_error_pixels"])
        ),
        "visual_foreground_iou": foreground_iou(
            float(row["visual_tp"]), float(row["visual_fp"]), float(row["visual_fn"])
        ),
        "adapter_foreground_iou": foreground_iou(
            float(row["adapter_tp"]), float(row["adapter_fp"]), float(row["adapter_fn"])
        ),
        "delta_foreground_iou": foreground_iou(
            float(row["adapter_tp"]), float(row["adapter_fp"]), float(row["adapter_fn"])
        )
        - foreground_iou(
            float(row["visual_tp"]), float(row["visual_fp"]), float(row["visual_fn"])
        ),
    }


def thresholds_for_policy(
    policy: str, visual_threshold: float, adapter_threshold: float
) -> tuple[float, float]:
    if policy == "system_own":
        return visual_threshold, adapter_threshold
    if policy == "frozen_visual":
        return visual_threshold, visual_threshold
    if policy == "frozen_adapter":
        return adapter_threshold, adapter_threshold
    raise AuditError(f"unknown threshold policy: {policy}")


def build_model(spec: ArchitectureSpec, checkpoint: Mapping[str, Any]) -> torch.nn.Module:
    obs_names = [str(item) for item in checkpoint["obs_channel_names"]]
    terrain_names = [str(item) for item in checkpoint["terrain_channel_names"]]
    if spec.family == "modern_bn_frozen":
        model = SmpSupportResidualAdapter(
            str(checkpoint["architecture"]),
            str(checkpoint["encoder"]),
            None,
            len(obs_names),
            len(terrain_names),
            alpha_max=float(checkpoint["alpha_max"]),
        )
    elif spec.family == "foundation_model":
        state = checkpoint["state_dict"]
        hidden = int(state["visual.segment.decoder.out.weight"].shape[1])
        terrain_base = int(state["terrain_encoder.net.0.weight"].shape[0])
        model = TimmSupportResidualAdapter(
            str(checkpoint["backend"]),
            str(checkpoint["backbone"]),
            len(obs_names),
            len(terrain_names),
            pretrained_backbone=False,
            img_size=int(checkpoint["img_size"]),
            out_indices=tuple(int(item) for item in checkpoint["out_indices"]),
            hidden=hidden,
            terrain_base=terrain_base,
            alpha_max=float(checkpoint["alpha_max"]),
            freeze_backbone=bool(checkpoint["freeze_backbone"]),
        )
    else:
        raise AuditError(f"unsupported family: {spec.family}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def torch_flow_batch(
    target: torch.Tensor,
    visual_probability: torch.Tensor,
    adapter_probability: torch.Tensor,
    visual_threshold: float,
    adapter_threshold: float,
) -> list[dict[str, Any]]:
    truth = target >= 0.5
    visual = visual_probability >= visual_threshold
    adapter = adapter_probability >= adapter_threshold
    visual_correct = visual == truth
    adapter_correct = adapter == truth
    masks = {
        "positive_pixels": truth,
        "visual_tp": visual & truth,
        "visual_fp": visual & (~truth),
        "visual_fn": (~visual) & truth,
        "visual_tn": (~visual) & (~truth),
        "adapter_tp": adapter & truth,
        "adapter_fp": adapter & (~truth),
        "adapter_fn": (~adapter) & truth,
        "adapter_tn": (~adapter) & (~truth),
        "visual_error_pixels": ~visual_correct,
        "visual_correct_pixels": visual_correct,
        "adapter_error_pixels": ~adapter_correct,
        "adapter_correct_pixels": adapter_correct,
        "e2c_pixels": (~visual_correct) & adapter_correct,
        "c2e_pixels": visual_correct & (~adapter_correct),
        "disagreement_pixels": visual != adapter,
    }
    reduced = {
        key: mask.flatten(1).sum(dim=1).detach().cpu().numpy().astype(np.int64)
        for key, mask in masks.items()
    }
    batch_size = int(target.shape[0])
    pixel_count = int(target[0].numel())
    rows: list[dict[str, Any]] = []
    for index in range(batch_size):
        row: dict[str, Any] = {"pixel_count": pixel_count}
        row.update({key: int(values[index]) for key, values in reduced.items()})
        row["net_corrected_pixels"] = int(row["e2c_pixels"] - row["c2e_pixels"])
        validate_flow_invariants(row)
        row.update(flow_rates(row))
        rows.append(row)
    return rows


def evaluate_checkpoint_pair(
    spec: ArchitectureSpec,
    pair: Mapping[str, Any],
    root: Path,
    dataset: H5SupportDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    adapter_path = root / str(pair["adapter_checkpoint"])
    baseline_path = root / str(pair["visual_checkpoint"])
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    assert_visual_state_identical(
        adapter_checkpoint,
        baseline_checkpoint,
        f"runtime/{spec.key}/seed{pair['seed']}",
    )
    if float(adapter_checkpoint["threshold"]) != float(pair["adapter_threshold"]):
        raise AuditError(f"adapter threshold changed after freeze: {adapter_path}")
    if float(baseline_checkpoint["threshold"]) != float(pair["visual_threshold"]):
        raise AuditError(f"visual threshold changed after freeze: {baseline_path}")
    model = build_model(spec, adapter_checkpoint).to(device)
    del adapter_checkpoint, baseline_checkpoint
    gc.collect()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    threshold_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    all_visual_probability: list[np.ndarray] = []
    all_adapter_probability: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    visual_threshold = float(pair["visual_threshold"])
    adapter_threshold = float(pair["adapter_threshold"])
    seed = int(pair["seed"])
    with torch.inference_mode():
        for batch in loader:
            observation = batch["obs"].to(device, non_blocking=True)
            terrain = batch["terrain"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            adapter_logits, diagnostics = model(observation, terrain)
            visual_logits = diagnostics["visual_logits"]
            if adapter_logits.shape != target.shape or visual_logits.shape != target.shape:
                raise AuditError(
                    f"logit/target shape mismatch for {spec.key}/seed{seed}: "
                    f"visual={tuple(visual_logits.shape)}, adapter={tuple(adapter_logits.shape)}, "
                    f"target={tuple(target.shape)}"
                )
            visual_probability = torch.sigmoid(visual_logits)
            adapter_probability = torch.sigmoid(adapter_logits)
            visual_brier = ((visual_probability - target) ** 2).flatten(1).mean(dim=1)
            adapter_brier = ((adapter_probability - target) ** 2).flatten(1).mean(dim=1)
            visual_nll = F.binary_cross_entropy_with_logits(
                visual_logits, target, reduction="none"
            ).flatten(1).mean(dim=1)
            adapter_nll = F.binary_cross_entropy_with_logits(
                adapter_logits, target, reduction="none"
            ).flatten(1).mean(dim=1)
            positive_pixels = (target >= 0.5).flatten(1).sum(dim=1)
            for index, sample_id in enumerate(batch["sample_id"]):
                probability_rows.append(
                    {
                        "architecture_key": spec.key,
                        "architecture": spec.label,
                        "family": spec.family,
                        "seed": seed,
                        "split": "test",
                        "sample_id": str(sample_id),
                        "pixel_count": int(target[index].numel()),
                        "positive_pixels": int(positive_pixels[index].item()),
                        "visual_brier": float(visual_brier[index].item()),
                        "adapter_brier": float(adapter_brier[index].item()),
                        "delta_brier_adapter_minus_visual": float(
                            adapter_brier[index].item() - visual_brier[index].item()
                        ),
                        "visual_nll": float(visual_nll[index].item()),
                        "adapter_nll": float(adapter_nll[index].item()),
                        "delta_nll_adapter_minus_visual": float(
                            adapter_nll[index].item() - visual_nll[index].item()
                        ),
                    }
                )
            for policy in POLICIES:
                used_visual, used_adapter = thresholds_for_policy(
                    policy, visual_threshold, adapter_threshold
                )
                batch_rows = torch_flow_batch(
                    target,
                    visual_probability,
                    adapter_probability,
                    used_visual,
                    used_adapter,
                )
                for index, metrics in enumerate(batch_rows):
                    threshold_rows.append(
                        {
                            "architecture_key": spec.key,
                            "architecture": spec.label,
                            "family": spec.family,
                            "seed": seed,
                            "split": "test",
                            "sample_id": str(batch["sample_id"][index]),
                            "threshold_policy": policy,
                            "visual_threshold_source_value": visual_threshold,
                            "adapter_threshold_source_value": adapter_threshold,
                            "visual_threshold_used": used_visual,
                            "adapter_threshold_used": used_adapter,
                            "thresholds_equal_used": math.isclose(
                                used_visual, used_adapter, rel_tol=0.0, abs_tol=1e-12
                            ),
                            **metrics,
                        }
                    )
            all_visual_probability.append(
                visual_probability.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            )
            all_adapter_probability.append(
                adapter_probability.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            )
            all_target.append(
                (target >= 0.5).detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1)
            )

    from sklearn.metrics import average_precision_score

    visual_flat = np.concatenate(all_visual_probability)
    adapter_flat = np.concatenate(all_adapter_probability)
    target_flat = np.concatenate(all_target)
    if target_flat.size != EXPECTED_PATCHES * int(dataset[0]["mask"].numel()):
        raise AuditError(f"unexpected pixel inventory for {spec.key}/seed{seed}: {target_flat.size}")
    seed_probability = {
        "architecture_key": spec.key,
        "architecture": spec.label,
        "family": spec.family,
        "seed": seed,
        "split": "test",
        "n_patches": EXPECTED_PATCHES,
        "n_pixels": int(target_flat.size),
        "positive_pixels": int(target_flat.sum()),
        "visual_average_precision": float(average_precision_score(target_flat, visual_flat)),
        "adapter_average_precision": float(average_precision_score(target_flat, adapter_flat)),
        "delta_average_precision": 0.0,
        "visual_brier": float(np.mean([row["visual_brier"] for row in probability_rows])),
        "adapter_brier": float(np.mean([row["adapter_brier"] for row in probability_rows])),
        "delta_brier_adapter_minus_visual": float(
            np.mean([row["delta_brier_adapter_minus_visual"] for row in probability_rows])
        ),
        "visual_nll": float(np.mean([row["visual_nll"] for row in probability_rows])),
        "adapter_nll": float(np.mean([row["adapter_nll"] for row in probability_rows])),
        "delta_nll_adapter_minus_visual": float(
            np.mean([row["delta_nll_adapter_minus_visual"] for row in probability_rows])
        ),
    }
    seed_probability["delta_average_precision"] = (
        seed_probability["adapter_average_precision"]
        - seed_probability["visual_average_precision"]
    )
    del visual_flat, adapter_flat, target_flat
    del all_visual_probability, all_adapter_probability, all_target
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return threshold_rows, probability_rows, seed_probability


def ensure_unique_rows(rows: Iterable[Mapping[str, Any]], keys: Sequence[str], context: str) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in seen:
            raise AuditError(f"duplicate {context} row: {key}")
        seen.add(key)


def build_seedmean_threshold_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
) -> list[dict[str, Any]]:
    ensure_unique_rows(
        rows,
        ("architecture_key", "seed", "sample_id", "threshold_policy"),
        "seed/sample/policy",
    )
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["architecture_key"]), str(row["threshold_policy"]), str(row["sample_id"]))].append(row)
    output: list[dict[str, Any]] = []
    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    for key in sorted(groups):
        members = groups[key]
        seeds = tuple(sorted(int(row["seed"]) for row in members))
        if seeds != expected:
            raise AuditError(f"seed inventory mismatch for {key}: {seeds}, expected={expected}")
        first = members[0]
        averaged = {
            field: float(np.mean([float(row[field]) for row in members]))
            for field in FLOW_COUNT_FIELDS
        }
        metrics = flow_rates(averaged)
        output.append(
            {
                "architecture_key": first["architecture_key"],
                "architecture": first["architecture"],
                "family": first["family"],
                "split": first["split"],
                "sample_id": first["sample_id"],
                "threshold_policy": first["threshold_policy"],
                "n_seeds_averaged": len(members),
                "seed_ids": ";".join(str(seed) for seed in seeds),
                "visual_threshold_mean": float(
                    np.mean([float(row["visual_threshold_used"]) for row in members])
                ),
                "adapter_threshold_mean": float(
                    np.mean([float(row["adapter_threshold_used"]) for row in members])
                ),
                "thresholds_equal_all_seeds": all(bool(row["thresholds_equal_used"]) for row in members),
                **averaged,
                **metrics,
                "mean_seed_visual_foreground_iou": float(
                    np.mean([float(row["visual_foreground_iou"]) for row in members])
                ),
                "mean_seed_adapter_foreground_iou": float(
                    np.mean([float(row["adapter_foreground_iou"]) for row in members])
                ),
                "mean_seed_delta_foreground_iou": float(
                    np.mean([float(row["delta_foreground_iou"]) for row in members])
                ),
            }
        )
    return output


def build_seed_threshold_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["architecture_key"]), int(row["seed"]), str(row["threshold_policy"]))].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) != EXPECTED_PATCHES or len({str(row["sample_id"]) for row in members}) != EXPECTED_PATCHES:
            raise AuditError(f"per-seed patch inventory mismatch for {key}: {len(members)}")
        first = members[0]
        totals = {field: float(sum(float(row[field]) for row in members)) for field in FLOW_COUNT_FIELDS}
        metrics = flow_rates(totals)
        output.append(
            {
                "architecture_key": first["architecture_key"],
                "architecture": first["architecture"],
                "family": first["family"],
                "seed": first["seed"],
                "split": first["split"],
                "threshold_policy": first["threshold_policy"],
                "n_patches": len(members),
                "visual_threshold_used": first["visual_threshold_used"],
                "adapter_threshold_used": first["adapter_threshold_used"],
                "thresholds_equal_used": first["thresholds_equal_used"],
                **totals,
                **metrics,
                "mean_patch_visual_foreground_iou": float(
                    np.mean([float(row["visual_foreground_iou"]) for row in members])
                ),
                "mean_patch_adapter_foreground_iou": float(
                    np.mean([float(row["adapter_foreground_iou"]) for row in members])
                ),
            }
        )
    return output


def bootstrap_threshold_summary(
    sample_rows: Sequence[Mapping[str, Any]], iterations: int, seed: int
) -> dict[str, list[float]]:
    arrays = {
        field: np.asarray([float(row[field]) for row in sample_rows], dtype=np.float64)
        for field in (
            "e2c_pixels",
            "c2e_pixels",
            "net_corrected_pixels",
            "visual_error_pixels",
            "visual_correct_pixels",
            "visual_tp",
            "visual_fp",
            "visual_fn",
            "adapter_tp",
            "adapter_fp",
            "adapter_fn",
            "delta_foreground_iou",
        )
    }
    n = len(sample_rows)
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[np.ndarray]] = defaultdict(list)
    chunk_size = min(250, iterations)
    for start in range(0, iterations, chunk_size):
        count = min(chunk_size, iterations - start)
        indices = rng.integers(0, n, size=(count, n))
        sums = {name: values[indices].sum(axis=1) for name, values in arrays.items() if name != "delta_foreground_iou"}
        visual_iou = sums["visual_tp"] / np.maximum(
            sums["visual_tp"] + sums["visual_fp"] + sums["visual_fn"], 1e-300
        )
        adapter_iou = sums["adapter_tp"] / np.maximum(
            sums["adapter_tp"] + sums["adapter_fp"] + sums["adapter_fn"], 1e-300
        )
        estimates["e2c_fraction_visual_errors"].append(
            sums["e2c_pixels"] / np.maximum(sums["visual_error_pixels"], 1e-300)
        )
        estimates["c2e_fraction_visual_correct"].append(
            sums["c2e_pixels"] / np.maximum(sums["visual_correct_pixels"], 1e-300)
        )
        estimates["net_error_reduction_fraction"].append(
            sums["net_corrected_pixels"] / np.maximum(sums["visual_error_pixels"], 1e-300)
        )
        estimates["delta_pooled_foreground_iou"].append(adapter_iou - visual_iou)
        estimates["mean_cluster_delta_foreground_iou"].append(
            arrays["delta_foreground_iou"][indices].mean(axis=1)
        )
    output: dict[str, list[float]] = {}
    for name, chunks in estimates.items():
        values = np.concatenate(chunks)
        output[f"{name}_ci95"] = [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
    return output


def direction(value: float | None, tolerance: float = 1e-12) -> str:
    if value is None:
        return "undefined"
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def ci_classification(ci: Sequence[float]) -> str:
    if float(ci[0]) > 0:
        return "positive"
    if float(ci[1]) < 0:
        return "negative"
    return "inconclusive"


def build_architecture_threshold_summary(
    sample_seedmean_rows: Sequence[Mapping[str, Any]], bootstrap: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_seedmean_rows:
        groups[(str(row["architecture_key"]), str(row["threshold_policy"]))].append(row)
    spec_by_key = {spec.key: spec for spec in ARCHITECTURES}
    output: list[dict[str, Any]] = []
    for architecture_index, spec in enumerate(ARCHITECTURES):
        for policy_index, policy in enumerate(POLICIES):
            members = groups[(spec.key, policy)]
            if len(members) != EXPECTED_PATCHES or len({str(row["sample_id"]) for row in members}) != EXPECTED_PATCHES:
                raise AuditError(f"cluster inventory mismatch for {spec.key}/{policy}: {len(members)}")
            totals = {field: float(sum(float(row[field]) for row in members)) for field in FLOW_COUNT_FIELDS}
            rates = flow_rates(totals)
            cis = bootstrap_threshold_summary(
                members,
                bootstrap,
                20260718 + architecture_index * 100 + policy_index,
            )
            net = rates["net_error_reduction_fraction"]
            net_ci = cis["net_error_reduction_fraction_ci95"]
            hint = spec_by_key[spec.key].collaborator_hint_same_adapter_percent
            actual_percent = float(net) * 100.0 if net is not None else None
            output.append(
                {
                    "architecture_key": spec.key,
                    "architecture": spec.label,
                    "family": spec.family,
                    "threshold_policy": policy,
                    "same_threshold_by_design": policy in {"frozen_visual", "frozen_adapter"},
                    "n_seeds_averaged_within_patch": len(EXPECTED_SEEDS),
                    "n_unique_patch_clusters": len(members),
                    "visual_threshold_mean_across_seeds": float(
                        np.mean([float(row["visual_threshold_mean"]) for row in members])
                    ),
                    "adapter_threshold_mean_across_seeds": float(
                        np.mean([float(row["adapter_threshold_mean"]) for row in members])
                    ),
                    **totals,
                    **rates,
                    "mean_cluster_visual_foreground_iou": float(
                        np.mean([float(row["visual_foreground_iou"]) for row in members])
                    ),
                    "mean_cluster_adapter_foreground_iou": float(
                        np.mean([float(row["adapter_foreground_iou"]) for row in members])
                    ),
                    "mean_cluster_delta_foreground_iou": float(
                        np.mean([float(row["delta_foreground_iou"]) for row in members])
                    ),
                    **cis,
                    "net_point_direction": direction(net),
                    "net_cluster_bootstrap_classification": ci_classification(net_ci),
                    "collaborator_hint_same_adapter_percent": hint if policy == "frozen_adapter" else None,
                    "difference_from_hint_percentage_points": (
                        actual_percent - hint if policy == "frozen_adapter" and actual_percent is not None else None
                    ),
                    "hint_verified_to_reported_two_decimals": (
                        round(actual_percent, 2) == round(hint, 2)
                        if policy == "frozen_adapter" and actual_percent is not None
                        else None
                    ),
                    "hint_discrepancy_role": (
                        "diagnostic_only_nonblocking" if policy == "frozen_adapter" else None
                    ),
                    "hint_is_scientific_input": False if policy == "frozen_adapter" else None,
                }
            )
    return output


def build_seedmean_probability_rows(
    rows: Sequence[Mapping[str, Any]], expected_seeds: Sequence[int] = EXPECTED_SEEDS
) -> list[dict[str, Any]]:
    ensure_unique_rows(rows, ("architecture_key", "seed", "sample_id"), "probability seed/sample")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["architecture_key"]), str(row["sample_id"]))].append(row)
    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    fields = (
        "visual_brier",
        "adapter_brier",
        "delta_brier_adapter_minus_visual",
        "visual_nll",
        "adapter_nll",
        "delta_nll_adapter_minus_visual",
    )
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        seeds = tuple(sorted(int(row["seed"]) for row in members))
        if seeds != expected:
            raise AuditError(f"probability seed inventory mismatch for {key}: {seeds}")
        first = members[0]
        output.append(
            {
                "architecture_key": first["architecture_key"],
                "architecture": first["architecture"],
                "family": first["family"],
                "split": first["split"],
                "sample_id": first["sample_id"],
                "n_seeds_averaged": len(members),
                "pixel_count": first["pixel_count"],
                "positive_pixels": first["positive_pixels"],
                **{
                    field: float(np.mean([float(row[field]) for row in members]))
                    for field in fields
                },
            }
        )
    return output


def mean_bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = values.size
    estimates = np.empty(iterations, dtype=np.float64)
    chunk_size = min(500, iterations)
    cursor = 0
    while cursor < iterations:
        count = min(chunk_size, iterations - cursor)
        indices = rng.integers(0, n, size=(count, n))
        estimates[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def build_architecture_probability_summary(
    sample_seedmean_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    bootstrap: int,
) -> list[dict[str, Any]]:
    sample_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seed_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_seedmean_rows:
        sample_groups[str(row["architecture_key"])].append(row)
    for row in seed_rows:
        seed_groups[str(row["architecture_key"])].append(row)
    output: list[dict[str, Any]] = []
    for index, spec in enumerate(ARCHITECTURES):
        samples = sample_groups[spec.key]
        seeds = seed_groups[spec.key]
        if len(samples) != EXPECTED_PATCHES or len(seeds) != len(EXPECTED_SEEDS):
            raise AuditError(
                f"probability summary inventory mismatch for {spec.key}: samples={len(samples)}, seeds={len(seeds)}"
            )
        ap_fields = ("visual_average_precision", "adapter_average_precision", "delta_average_precision")
        sample_fields = (
            "visual_brier",
            "adapter_brier",
            "delta_brier_adapter_minus_visual",
            "visual_nll",
            "adapter_nll",
            "delta_nll_adapter_minus_visual",
        )
        row: dict[str, Any] = {
            "architecture_key": spec.key,
            "architecture": spec.label,
            "family": spec.family,
            "n_seeds": len(seeds),
            "n_unique_patch_clusters": len(samples),
            "average_precision_aggregation": "mean of five exact pixel-pooled per-seed AP values; descriptive",
        }
        for field in ap_fields:
            values = np.asarray([float(seed_row[field]) for seed_row in seeds], dtype=np.float64)
            row[f"mean_{field}"] = float(values.mean())
            row[f"sd_{field}"] = float(values.std(ddof=1))
            row[f"min_{field}"] = float(values.min())
            row[f"max_{field}"] = float(values.max())
        for field_index, field in enumerate(sample_fields):
            values = np.asarray([float(sample[field]) for sample in samples], dtype=np.float64)
            row[f"mean_cluster_{field}"] = float(values.mean())
            row[f"{field}_cluster_bootstrap_ci95"] = mean_bootstrap_ci(
                values,
                bootstrap,
                20261718 + index * 100 + field_index,
            )
        output.append(row)
    return output


def build_policy_conclusions(
    threshold_summary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["architecture_key"]), str(row["threshold_policy"])): row
        for row in threshold_summary
    }
    output: list[dict[str, Any]] = []
    for spec in ARCHITECTURES:
        own = lookup[(spec.key, "system_own")]
        visual = lookup[(spec.key, "frozen_visual")]
        adapter = lookup[(spec.key, "frozen_adapter")]
        own_direction = str(own["net_point_direction"])
        visual_direction = str(visual["net_point_direction"])
        adapter_direction = str(adapter["net_point_direction"])
        output.append(
            {
                "architecture_key": spec.key,
                "architecture": spec.label,
                "family": spec.family,
                "system_own_net_error_reduction_fraction": own["net_error_reduction_fraction"],
                "frozen_visual_net_error_reduction_fraction": visual["net_error_reduction_fraction"],
                "frozen_adapter_net_error_reduction_fraction": adapter["net_error_reduction_fraction"],
                "system_own_point_direction": own_direction,
                "frozen_visual_point_direction": visual_direction,
                "frozen_adapter_point_direction": adapter_direction,
                "frozen_visual_ci_classification": visual["net_cluster_bootstrap_classification"],
                "frozen_adapter_ci_classification": adapter["net_cluster_bootstrap_classification"],
                "same_threshold_policy_changes_point_direction": visual_direction != adapter_direction,
                "system_own_changes_direction_vs_any_same_threshold": own_direction
                not in {visual_direction, adapter_direction},
                "same_threshold_policy_stable_point_direction": visual_direction == adapter_direction,
                "same_threshold_policy_spread_percentage_points": 100.0
                * abs(
                    float(visual["net_error_reduction_fraction"])
                    - float(adapter["net_error_reduction_fraction"])
                ),
                "collaborator_hint_same_adapter_percent": spec.collaborator_hint_same_adapter_percent,
                "formal_same_adapter_percent": 100.0
                * float(adapter["net_error_reduction_fraction"]),
                "hint_verified_to_reported_two_decimals": adapter[
                    "hint_verified_to_reported_two_decimals"
                ],
                "hint_difference_percentage_points": adapter[
                    "difference_from_hint_percentage_points"
                ],
                "hint_discrepancy_role": "diagnostic_only_nonblocking",
                "hint_is_scientific_input": False,
            }
        )
    return output


def format_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):+.{digits}f}%"


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def build_report(
    threshold_summary: Sequence[Mapping[str, Any]],
    probability_summary: Sequence[Mapping[str, Any]],
    conclusions: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    amendment: Mapping[str, Any] | None,
) -> str:
    lines = [
        "# L4S eight-architecture Terrain same-threshold audit",
        "",
        "## Audit status",
        "",
        "- Read-only inference replay; no training and no test-set threshold tuning.",
        "- Formal v2 is authorized by a post-failure amendment. The immutable base freeze and "
        "original FAILURE.json are retained unchanged.",
        f"- Inputs frozen before inference: `{freeze['n_checkpoint_pairs']}` checkpoint pairs, "
        f"`{freeze['n_architectures']}` architectures, five seeds each.",
        "- The five seeds are averaged within each patch first. The 800 unique patches are the "
        "current clustering units.",
        "- `event_uid=L4S_test_official` is constant and is not an event identifier. No event-level "
        "inference is claimed.",
        "- The legacy count tables contain no raw probabilities. This audit replays probabilities "
        "from frozen checkpoints and the frozen test H5; it does not reverse-engineer any summary.",
        "",
        "## Thresholded error flow",
        "",
        "`system_own` is a system comparison only. `frozen_visual` and `frozen_adapter` are the two "
        "same-threshold Terrain contrasts.",
        "",
        "| architecture | policy | E2C / visual errors | C2E / visual correct | net / visual errors [patch bootstrap 95% CI] | visual IoU | adapter IoU | delta IoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lookup = {
        (str(row["architecture_key"]), str(row["threshold_policy"])): row
        for row in threshold_summary
    }
    for spec in ARCHITECTURES:
        for policy in POLICIES:
            row = lookup[(spec.key, policy)]
            ci = row["net_error_reduction_fraction_ci95"]
            lines.append(
                f"| {spec.label} | {policy} | {format_percent(row['e2c_fraction_visual_errors'])} | "
                f"{format_percent(row['c2e_fraction_visual_correct'])} | "
                f"{format_percent(row['net_error_reduction_fraction'])} "
                f"[{format_percent(ci[0])}, {format_percent(ci[1])}] | "
                f"{float(row['visual_foreground_iou']):.4f} | "
                f"{float(row['adapter_foreground_iou']):.4f} | "
                f"{format_float(row['delta_foreground_iou'])} |"
            )
    lines.extend(
        [
            "",
            "## Threshold-policy sensitivity",
            "",
            "Point direction is based on net error reduction. CI classification uses the 800-patch "
            "cluster bootstrap; it is not event inference. Collaborator rounded hints are shown only "
            "as nonblocking diagnostic discrepancies and are not scientific inputs.",
            "",
            "| architecture | own direction | frozen visual | frozen adapter | same-threshold direction changes? | formal vs hint | discrepancy (pp) |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in conclusions:
        lines.append(
            f"| {row['architecture']} | {row['system_own_point_direction']} | "
            f"{row['frozen_visual_point_direction']} ({row['frozen_visual_ci_classification']}) | "
            f"{row['frozen_adapter_point_direction']} ({row['frozen_adapter_ci_classification']}) | "
            f"{row['same_threshold_policy_changes_point_direction']} | "
            f"{float(row['formal_same_adapter_percent']):+.2f}% vs "
            f"{float(row['collaborator_hint_same_adapter_percent']):+.2f}% | "
            f"{float(row['hint_difference_percentage_points']):+.6f} |"
        )
    frozen_visual_nonpositive = [
        str(row["architecture"])
        for row in conclusions
        if row["frozen_visual_point_direction"] != "positive"
    ]
    frozen_adapter_nonpositive = [
        str(row["architecture"])
        for row in conclusions
        if row["frozen_adapter_point_direction"] != "positive"
    ]
    lines.extend(
        [
            "",
            "## All-positive claim",
            "",
            "The old all-positive claim is **not supported** by formal same-threshold checkpoint "
            "inference.",
            f"- frozen visual threshold non-positive architectures: "
            f"`{', '.join(frozen_visual_nonpositive) if frozen_visual_nonpositive else 'none'}`",
            f"- frozen adapter threshold non-positive architectures: "
            f"`{', '.join(frozen_adapter_nonpositive) if frozen_adapter_nonpositive else 'none'}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Threshold-free and proper scores",
            "",
            "AP is exact and pixel-pooled within each seed, then descriptively averaged over five "
            "seeds. Brier/NLL average seeds within patch first and then average 800 patches. AP is "
            "not presented as patch- or event-cluster inference.",
            "",
            "| architecture | visual AP | adapter AP | delta AP | visual Brier | adapter Brier | delta Brier | visual NLL | adapter NLL | delta NLL |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    probability_by_key = {str(row["architecture_key"]): row for row in probability_summary}
    for spec in ARCHITECTURES:
        row = probability_by_key[spec.key]
        lines.append(
            f"| {spec.label} | {float(row['mean_visual_average_precision']):.4f} | "
            f"{float(row['mean_adapter_average_precision']):.4f} | "
            f"{format_float(row['mean_delta_average_precision'])} | "
            f"{float(row['mean_cluster_visual_brier']):.5f} | "
            f"{float(row['mean_cluster_adapter_brier']):.5f} | "
            f"{format_float(row['mean_cluster_delta_brier_adapter_minus_visual'], 5)} | "
            f"{float(row['mean_cluster_visual_nll']):.5f} | "
            f"{float(row['mean_cluster_adapter_nll']):.5f} | "
            f"{format_float(row['mean_cluster_delta_nll_adapter_minus_visual'], 5)} |"
        )
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- E2C: visual wrong, Terrain adapter correct.",
            "- C2E: visual correct, Terrain adapter wrong.",
            "- Net error reduction: `(E2C - C2E) / visual error pixels`.",
            "- Foreground IoU: `TP / (TP + FP + FN)`; an empty denominator is assigned zero only "
            "for the explicitly reported patch-macro diagnostic. Pooled IoU is primary here.",
            "- Thresholds were selected on validation during the original runs over 0.05--0.95; "
            "this audit reads them from frozen checkpoints and never selects a test threshold.",
            "",
            "## Fail-closed boundary",
            "",
            "Completion requires exact 8 x 5 x 800 inventories, identical adapter/baseline visual "
            "states, verified checkpoint/H5/model-source hashes, an amendment-authorized audit-script "
            "hash, deterministic CuBLAS, exact flow-balance identities, and strict JSON. A missing "
            "probability-replay asset stops the audit rather than triggering reconstruction from "
            "aggregate counts. Hint mismatches are recorded but never block checkpoint inference.",
            "",
            "## Amendment",
            "",
            f"- amendment id: `{amendment.get('amendment_id') if amendment else 'none'}`",
            f"- base freeze mutated: `{amendment.get('base_freeze_mutated') if amendment else 'NA'}`",
            f"- base failure directory mutated: "
            f"`{amendment.get('base_failure_directory_mutated') if amendment else 'NA'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def configure_determinism() -> None:
    torch.manual_seed(20260718)
    np.random.seed(20260718)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.use_deterministic_algorithms(True)


class RunLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def run_audit(
    root: Path,
    cache_dir: Path,
    protocol_dir: Path,
    outdir: Path,
    device_name: str,
    batch_size: int,
    workers: int,
    bootstrap: int,
    amendment_dir: Path | None,
    audit_id: str,
) -> dict[str, Any]:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLog(outdir / "run.log")
    try:
        if amendment_dir is None:
            raise AuditError("formal v2 run requires an explicit post-failure amendment directory")
        if outdir.resolve() == OUTDIR.resolve():
            raise AuditError("formal v2 must not write to the immutable original failure directory")
        if audit_id != FORMAL_V2_AUDIT_ID:
            raise AuditError(f"formal v2 audit id must be {FORMAL_V2_AUDIT_ID}")
        cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if device_name.startswith("cuda") and cublas_workspace not in {":4096:8", ":16:8"}:
            raise AuditError(
                "deterministic CUDA inference requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
        logger.log("verifying immutable base, amendment, and every computational input SHA-256")
        freeze, manifest, amendment = verify_freeze(root, protocol_dir, amendment_dir)
        if amendment is None:
            raise AuditError("verified amendment payload is missing")
        if int(freeze["n_checkpoint_pairs"]) != len(ARCHITECTURES) * len(EXPECTED_SEEDS):
            raise AuditError("frozen checkpoint-pair count is incomplete")
        configure_determinism()
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise AuditError("CUDA requested but unavailable")
        device = torch.device(device_name)
        logger.log(f"device={device}; batch_size={batch_size}; workers={workers}; bootstrap={bootstrap}")
        test_path = split_path(cache_dir, "test")
        obs_idx, obs_names = channel_indices_from_test_h5(test_path, {"observation"})
        terrain_idx, terrain_names = channel_indices_from_test_h5(test_path, {"terrain"})
        first_pair = manifest["checkpoint_pairs"][0]
        if obs_names != list(first_pair["obs_channel_names"]):
            raise AuditError("frozen test-H5 observation channels do not match checkpoint channels")
        if terrain_names != list(first_pair["terrain_channel_names"]):
            raise AuditError("frozen test-H5 terrain channels do not match checkpoint channels")
        dataset = H5SupportDataset(test_path, obs_idx, terrain_idx)
        if len(dataset) != EXPECTED_PATCHES:
            raise AuditError(f"runtime test-patch count changed: {len(dataset)}")
        if len(set(dataset.sample_ids)) != EXPECTED_PATCHES:
            raise AuditError("runtime sample IDs are not unique")

        spec_by_key = {spec.key: spec for spec in ARCHITECTURES}
        pair_lookup = {
            (str(pair["architecture_key"]), int(pair["seed"])): pair
            for pair in manifest["checkpoint_pairs"]
        }
        if len(pair_lookup) != len(ARCHITECTURES) * len(EXPECTED_SEEDS):
            raise AuditError("duplicate or missing checkpoint pairs in frozen manifest")
        all_threshold_rows: list[dict[str, Any]] = []
        all_probability_rows: list[dict[str, Any]] = []
        seed_probability_rows: list[dict[str, Any]] = []
        for spec in ARCHITECTURES:
            for seed in EXPECTED_SEEDS:
                pair = pair_lookup.get((spec.key, seed))
                if pair is None:
                    raise AuditError(f"missing frozen pair for {spec.key}/seed{seed}")
                logger.log(f"inference architecture={spec.key} seed={seed}")
                threshold_rows, probability_rows, seed_probability = evaluate_checkpoint_pair(
                    spec_by_key[spec.key],
                    pair,
                    root,
                    dataset,
                    device,
                    batch_size,
                    workers,
                )
                if len(threshold_rows) != EXPECTED_PATCHES * len(POLICIES):
                    raise AuditError(f"threshold row count mismatch for {spec.key}/seed{seed}")
                if len(probability_rows) != EXPECTED_PATCHES:
                    raise AuditError(f"probability row count mismatch for {spec.key}/seed{seed}")
                all_threshold_rows.extend(threshold_rows)
                all_probability_rows.extend(probability_rows)
                seed_probability_rows.append(seed_probability)

        expected_threshold_rows = len(ARCHITECTURES) * len(EXPECTED_SEEDS) * EXPECTED_PATCHES * len(POLICIES)
        expected_probability_rows = len(ARCHITECTURES) * len(EXPECTED_SEEDS) * EXPECTED_PATCHES
        if len(all_threshold_rows) != expected_threshold_rows:
            raise AuditError(f"global threshold row count mismatch: {len(all_threshold_rows)}")
        if len(all_probability_rows) != expected_probability_rows:
            raise AuditError(f"global probability row count mismatch: {len(all_probability_rows)}")
        logger.log("aggregating five seeds within each patch")
        sample_seedmean_threshold = build_seedmean_threshold_rows(all_threshold_rows)
        seed_threshold_summary = build_seed_threshold_summary(all_threshold_rows)
        architecture_threshold_summary = build_architecture_threshold_summary(
            sample_seedmean_threshold, bootstrap
        )
        sample_seedmean_probability = build_seedmean_probability_rows(all_probability_rows)
        architecture_probability_summary = build_architecture_probability_summary(
            sample_seedmean_probability, seed_probability_rows, bootstrap
        )
        policy_conclusions = build_policy_conclusions(architecture_threshold_summary)
        hint_mismatches = [
            row["architecture"]
            for row in policy_conclusions
            if not bool(row["hint_verified_to_reported_two_decimals"])
        ]
        logger.log(
            "collaborator hint comparison is diagnostic-only; exact-rounding mismatches="
            + (",".join(hint_mismatches) if hint_mismatches else "none")
        )
        frozen_visual_positive = [
            row["architecture"]
            for row in policy_conclusions
            if row["frozen_visual_point_direction"] == "positive"
        ]
        frozen_adapter_positive = [
            row["architecture"]
            for row in policy_conclusions
            if row["frozen_adapter_point_direction"] == "positive"
        ]
        claim_assessment = {
            "claim": "Terrain same-threshold net error reduction is positive for all eight architectures",
            "frozen_visual_positive_architectures": frozen_visual_positive,
            "frozen_visual_nonpositive_architectures": [
                row["architecture"]
                for row in policy_conclusions
                if row["frozen_visual_point_direction"] != "positive"
            ],
            "frozen_adapter_positive_architectures": frozen_adapter_positive,
            "frozen_adapter_nonpositive_architectures": [
                row["architecture"]
                for row in policy_conclusions
                if row["frozen_adapter_point_direction"] != "positive"
            ],
            "all_eight_positive_frozen_visual": len(frozen_visual_positive) == len(ARCHITECTURES),
            "all_eight_positive_frozen_adapter": len(frozen_adapter_positive) == len(ARCHITECTURES),
            "old_all_positive_claim_supported": (
                len(frozen_visual_positive) == len(ARCHITECTURES)
                and len(frozen_adapter_positive) == len(ARCHITECTURES)
            ),
        }

        logger.log("writing per-seed/per-patch and summary artifacts")
        write_csv(outdir / "per_seed_sample_threshold_metrics.csv", all_threshold_rows)
        write_csv(outdir / "per_seed_sample_probability_metrics.csv", all_probability_rows)
        write_csv(outdir / "sample_seedmean_threshold_metrics.csv", sample_seedmean_threshold)
        write_csv(outdir / "sample_seedmean_probability_metrics.csv", sample_seedmean_probability)
        write_csv(outdir / "per_seed_threshold_summary.csv", seed_threshold_summary)
        write_csv(outdir / "per_seed_probability_summary.csv", seed_probability_rows)
        write_csv(outdir / "architecture_threshold_summary.csv", architecture_threshold_summary)
        write_csv(outdir / "architecture_probability_summary.csv", architecture_probability_summary)
        write_csv(outdir / "threshold_policy_conclusions.csv", policy_conclusions)

        summary = {
            "schema_version": "1.0",
            "audit_id": audit_id,
            "status": "complete",
            "created_at_utc": utc_now(),
            "read_only_inference": True,
            "training_performed": False,
            "split": "test",
            "n_architectures": len(ARCHITECTURES),
            "seeds": list(EXPECTED_SEEDS),
            "n_unique_patch_clusters": EXPECTED_PATCHES,
            "event_inference_claimed": False,
            "event_inference_reason": (
                "event_uid is constant L4S_test_official; 800 unique sample_id patches are the available clusters"
            ),
            "aggregation_order": "five seeds averaged within patch first; then 800 patches",
            "threshold_policies": list(POLICIES),
            "metric_definitions": METRIC_DEFINITIONS,
            "post_failure_amendment": {
                "directory": relative_to_root(amendment_dir, root),
                "amendment_id": amendment["amendment_id"],
                "amendment_sha256": sha256_file(amendment_dir / "amendment.json"),
                "hints_are_scientific_inputs": False,
                "hint_mismatch_is_blocking": False,
                "immutable_base_reverified_before_inference": True,
            },
            "input_freeze": {
                "freeze_file": relative_to_root(protocol_dir / "freeze.json", root),
                "freeze_sha256": sha256_file(protocol_dir / "freeze.json"),
                "input_manifest_sha256": freeze["input_manifest_sha256"],
                "verified_before_inference": True,
            },
            "runtime": {
                "device_requested": device_name,
                "device_resolved": str(device),
                "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "batch_size": batch_size,
                "workers": workers,
                "bootstrap_iterations": bootstrap,
                "elapsed_seconds": time.time() - started,
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                "cublas_workspace_config": cublas_workspace,
                "deterministic_algorithms": True,
                "deterministic_algorithms_warn_only": False,
                "cuda_matmul_fp32_precision": (
                    torch.backends.cuda.matmul.fp32_precision if device.type == "cuda" else None
                ),
                "cudnn_conv_fp32_precision": (
                    torch.backends.cudnn.conv.fp32_precision if device.type == "cuda" else None
                ),
            },
            "row_counts": {
                "per_seed_sample_threshold_metrics": len(all_threshold_rows),
                "per_seed_sample_probability_metrics": len(all_probability_rows),
                "sample_seedmean_threshold_metrics": len(sample_seedmean_threshold),
                "sample_seedmean_probability_metrics": len(sample_seedmean_probability),
                "per_seed_threshold_summary": len(seed_threshold_summary),
                "per_seed_probability_summary": len(seed_probability_rows),
                "architecture_threshold_summary": len(architecture_threshold_summary),
                "architecture_probability_summary": len(architecture_probability_summary),
                "threshold_policy_conclusions": len(policy_conclusions),
            },
            "architecture_threshold_summary": architecture_threshold_summary,
            "architecture_probability_summary": architecture_probability_summary,
            "threshold_policy_conclusions": policy_conclusions,
            "hint_diagnostic": {
                "scientific_input": False,
                "blocking": False,
                "exact_two_decimal_rounding_mismatches": hint_mismatches,
            },
            "claim_assessment": claim_assessment,
        }
        write_strict_json(outdir / "summary.json", summary)
        report = build_report(
            architecture_threshold_summary,
            architecture_probability_summary,
            policy_conclusions,
            freeze,
            amendment,
        )
        (outdir / "report.md").write_text(report, encoding="utf-8")
        logger.log("validating strict JSON, closing run log, and hashing final outputs")
        load_strict_json(outdir / "summary.json")
        logger.close()
        required_outputs = [
            "run.log",
            "per_seed_sample_threshold_metrics.csv",
            "per_seed_sample_probability_metrics.csv",
            "sample_seedmean_threshold_metrics.csv",
            "sample_seedmean_probability_metrics.csv",
            "per_seed_threshold_summary.csv",
            "per_seed_probability_summary.csv",
            "architecture_threshold_summary.csv",
            "architecture_probability_summary.csv",
            "threshold_policy_conclusions.csv",
            "summary.json",
            "report.md",
        ]
        output_records = [
            asset_record(outdir / name, root, "audit_output") for name in required_outputs
        ]
        output_manifest = {
            "schema_version": "1.0",
            "audit_id": audit_id,
            "status": "complete",
            "created_at_utc": utc_now(),
            "outputs": output_records,
        }
        write_strict_json(outdir / "output_manifest.json", output_manifest)
        done = {
            "schema_version": "1.0",
            "audit_id": audit_id,
            "status": "complete",
            "created_at_utc": utc_now(),
            "read_only_inference": True,
            "training_performed": False,
            "strict_json_validated": True,
            "input_hashes_verified_before_inference": True,
            "visual_states_identical_for_all_40_pairs": True,
            "flow_invariants_validated": True,
            "inventory_validated": True,
            "deterministic_cublas_validated": True,
            "collaborator_hints_are_scientific_inputs": False,
            "collaborator_hint_mismatch_is_blocking": False,
            "collaborator_hint_exact_rounding_mismatches": hint_mismatches,
            "old_all_positive_claim_supported": claim_assessment[
                "old_all_positive_claim_supported"
            ],
            "event_inference_claimed": False,
            "output_manifest_sha256": sha256_file(outdir / "output_manifest.json"),
            "summary_sha256": sha256_file(outdir / "summary.json"),
            "report_sha256": sha256_file(outdir / "report.md"),
            "elapsed_seconds": time.time() - started,
        }
        verify_immutable_base_hashes(root)
        done["immutable_base_reverified_after_inference"] = True
        write_strict_json(outdir / "DONE.json", done)
        return summary
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("amend", "run"), default="run")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--protocol-dir", type=Path, default=PROTOCOL_DIR)
    parser.add_argument("--amendment-dir", type=Path, default=AMENDMENT_DIR)
    parser.add_argument("--outdir", type=Path, default=FORMAL_V2_OUTDIR)
    parser.add_argument("--audit-id", default=FORMAL_V2_AUDIT_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def failure_directory(args: argparse.Namespace) -> Path:
    return args.amendment_dir if args.mode == "amend" else args.outdir


def main() -> int:
    args = parse_args()
    try:
        if args.batch_size <= 0 or args.workers < 0 or args.bootstrap <= 0:
            raise AuditError("batch-size and bootstrap must be positive; workers must be non-negative")
        if args.mode == "amend":
            create_post_failure_amendment(
                args.root,
                args.protocol_dir,
                args.amendment_dir,
                args.outdir,
            )
        if args.mode == "run":
            run_audit(
                args.root,
                args.cache_dir,
                args.protocol_dir,
                args.outdir,
                args.device,
                args.batch_size,
                args.workers,
                args.bootstrap,
                args.amendment_dir,
                args.audit_id,
            )
        return 0
    except Exception as exc:
        target = failure_directory(args)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "audit_id": args.audit_id,
            "status": "failed_closed",
            "created_at_utc": utc_now(),
            "phase": args.mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "missing_or_invalid_assets": (
                [str(exc)] if isinstance(exc, (AuditError, FileNotFoundError)) else []
            ),
            "metrics_reconstructed_from_summaries": False,
            "traceback": traceback.format_exc(),
        }
        write_strict_json(target / "FAILURE.json", payload)
        if args.mode == "run":
            write_strict_json(
                args.outdir / "DONE.json",
                {
                    "schema_version": "1.0",
                    "audit_id": args.audit_id,
                    "status": "failed_closed",
                    "created_at_utc": utc_now(),
                    "failure_file": str(args.outdir / "FAILURE.json"),
                    "metrics_reconstructed_from_summaries": False,
                },
            )
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
