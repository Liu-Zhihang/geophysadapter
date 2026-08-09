#!/usr/bin/env python3
"""Export bounded-memory parent OOF pixel predictions for Trigger evaluation.

The default Trigger identity anchor is the visual-only ``V`` parent.  ``VT``
may be exported explicitly for a separate Terrain-conditioned sensitivity, but
Trigger quality or rainfall values never enter this exporter.  Physical-event
IDs are read only as identities from the registered sidecars.

Formal mode discovers complete LODO runs under the production run root.  It
requires at least five seeds common to every LODO fold and writes one isolated
``parent_oof_manifest.json`` per seed, so checkpoints from different seeds can
never be mixed.  Each LODO test fold is one compressed prediction shard.

Engineering smoke mode requires explicit ``--run-dir``, ``--nonformal``, and
``--max-samples``.  Its manifest is marked incomplete and is intentionally
rejected by the formal Trigger evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_pild_sen12_roleaware_v1 as trainer  # noqa: E402
from pild_sen12_training_loader_v2 import decode, sha256_file  # noqa: E402


DEFAULT_METADATA = PROJECT_ROOT / "metadata/pild_sen12_training_v2"
DEFAULT_MANIFEST = DEFAULT_METADATA / "unified_sample_manifest_v2.csv"
DEFAULT_PROTOCOL = DEFAULT_METADATA / "protocol_summary_v2.json"
DEFAULT_SPLIT = DEFAULT_METADATA / "leave_one_dataset_out_split_v2.csv"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "experiments/revision2026/pild_sen12_roleaware_lodo_v1"
DEFAULT_OUT = PROJECT_ROOT / "experiments/revision2026/pild_sen12_parent_oof_logits_v1"

EXPORT_SCHEMA = "pild_sen12_parent_oof_export.v1"
MANIFEST_SCHEMA = "pild_trigger_parent_oof_manifest.v1"
RECEIPT_SCHEMA = "pild_trigger_parent_oof_prediction.v1"
RUN_PATTERN = re.compile(r"^(V|VT)_seed(?P<seed>\d+)$")


class ExportContractError(RuntimeError):
    """Raised when a run or input cannot support a valid parent OOF export."""


@dataclass(frozen=True)
class ValidatedRun:
    path: Path
    fold_id: str
    variant: str
    seed: int
    checkpoint_sha256: str
    threshold_probability: float
    threshold_source: str
    config: dict[str, Any]
    result: dict[str, Any]
    done: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--run-dir", type=Path, action="append", default=[],
        help="Explicit complete run directory; repeat only for the same seed.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--variant", choices=("V", "VT"), default="V",
        help="V is the default Trigger identity anchor; VT is sensitivity only.",
    )
    parser.add_argument(
        "--seeds", default="",
        help="Optional comma-separated subset of common formal seeds.",
    )
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prithvi-snapshot", type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--nonformal", action="store_true",
        help="Engineering-only export; requires explicit run dirs and max-samples.",
    )
    parser.add_argument(
        "--compression", choices=("compressed", "stored"), default="compressed"
    )
    return parser.parse_args(argv)


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


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_seed_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != len(set(values)):
        raise ValueError("--seeds contains duplicates")
    return values


def _resolve_from_project(value: str | Path | None, fallback: Path) -> Path:
    if value in (None, ""):
        return fallback.resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ExportContractError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExportContractError(f"{label} must contain a JSON object: {path}")
    return value


def _verify_done_artifact(run: Path, done: Mapping[str, Any], name: str) -> str:
    artifacts = done.get("artifacts", {})
    if name not in artifacts:
        raise ExportContractError(f"DONE lacks required artifact {name}: {run}")
    path = run / name
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {path}")
    expected_hash = str(artifacts[name].get("sha256", ""))
    expected_size = int(artifacts[name].get("size", -1))
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash or path.stat().st_size != expected_size:
        raise ExportContractError(f"DONE artifact hash/size drift: {path}")
    done_mtime = (run / "DONE.json").stat().st_mtime_ns
    if path.stat().st_mtime_ns > done_mtime + 1_000_000_000:
        raise ExportContractError(f"artifact is newer than DONE and therefore stale: {path}")
    return observed_hash


def validate_complete_run(
    run_dir: Path,
    *,
    expected_variant: str,
    manifest_path: Path,
    split_path: Path,
) -> ValidatedRun:
    run = run_dir.resolve()
    done = _read_json(run / "DONE.json", "DONE receipt")
    result = _read_json(run / "result.json", "result")
    config = _read_json(run / "config.json", "config")
    if done.get("status") != "complete" or result.get("status") != "complete":
        raise ExportContractError(f"run is not complete: {run}")
    checkpoint_hash = _verify_done_artifact(run, done, "checkpoint.pt")
    result_hash = _verify_done_artifact(run, done, "result.json")
    config_hash = _verify_done_artifact(run, done, "config.json")
    if done.get("result_sha256") != result_hash or done.get("config_sha256") != config_hash:
        raise ExportContractError(f"DONE top-level result/config hash mismatch: {run}")
    if result.get("checkpoint_sha256") != checkpoint_hash:
        raise ExportContractError(f"result checkpoint hash mismatch: {run}")
    variants = {str(done.get("variant")), str(result.get("variant")), str(config.get("variant"))}
    if variants != {expected_variant}:
        raise ExportContractError(f"variant mismatch for {run}: {sorted(variants)}")
    seeds = {int(done.get("seed", -1)), int(result.get("seed", -2)), int(config.get("seed", -3))}
    if len(seeds) != 1 or next(iter(seeds)) < 0:
        raise ExportContractError(f"seed mismatch for {run}: {sorted(seeds)}")
    seed = next(iter(seeds))
    fold_values = {
        str(done.get("fold_id")), str(result.get("identity", {}).get("fold_id")),
        str(config.get("identity", {}).get("fold_id")),
    }
    if len(fold_values) != 1 or "None" in fold_values:
        raise ExportContractError(f"fold identity mismatch for {run}: {sorted(fold_values)}")
    fold_id = next(iter(fold_values))
    current_manifest_hash = sha256_file(manifest_path)
    current_split_hash = sha256_file(split_path)
    for source_name, identity in (
        ("result", result.get("identity", {})),
        ("config", config.get("identity", {})),
    ):
        if identity.get("manifest_sha256") != current_manifest_hash:
            raise ExportContractError(f"{source_name} manifest identity is stale: {run}")
        if identity.get("split_sha256") != current_split_hash:
            raise ExportContractError(f"{source_name} split identity is stale: {run}")
        if int(identity.get("seed", -1)) != seed or str(identity.get("fold_id")) != fold_id:
            raise ExportContractError(f"{source_name} seed/fold identity mismatch: {run}")
    threshold = float(result.get("threshold", float("nan")))
    if not 0.0 < threshold < 1.0:
        raise ExportContractError(f"invalid parent threshold: {run}")
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != expected_variant:
        raise ExportContractError(f"checkpoint variant mismatch: {run}")
    if float(checkpoint.get("threshold", float("nan"))) != threshold:
        raise ExportContractError(f"checkpoint/result threshold mismatch: {run}")
    checkpoint_identity = checkpoint.get("identity", {})
    if checkpoint_identity != result.get("identity"):
        raise ExportContractError(f"checkpoint/result identity mismatch: {run}")
    threshold_source = str(checkpoint.get("threshold_source", ""))
    allowed_sources = {"visual_validation", "matched_V_parent"}
    if threshold_source not in allowed_sources:
        raise ExportContractError(f"threshold is not documented as train-only: {run}")
    return ValidatedRun(
        path=run, fold_id=fold_id, variant=expected_variant, seed=seed,
        checkpoint_sha256=checkpoint_hash, threshold_probability=threshold,
        threshold_source=threshold_source, config=config, result=result, done=done,
    )


def expected_lodo_folds(split_path: Path) -> tuple[str, ...]:
    split = pd.read_csv(split_path, keep_default_na=False, usecols=["fold_id"])
    folds = tuple(sorted(value for value in split.fold_id.astype(str).unique() if value.startswith("lodo_")))
    if len(folds) < 2:
        raise ExportContractError(f"expected multiple LODO folds in {split_path}")
    return folds


def discover_formal_runs(
    run_root: Path,
    folds: Sequence[str],
    variant: str,
) -> dict[int, dict[str, Path]]:
    by_seed: dict[int, dict[str, Path]] = {}
    for fold in folds:
        fold_root = run_root / fold
        if not fold_root.is_dir():
            continue
        for candidate in fold_root.iterdir():
            if not candidate.is_dir() or candidate.name.startswith("."):
                continue
            match = RUN_PATTERN.fullmatch(candidate.name)
            if not match or match.group(1) != variant:
                continue
            required = ("DONE.json", "result.json", "config.json", "checkpoint.pt")
            if not all((candidate / name).is_file() for name in required):
                # Active staging directories and incomplete published paths do
                # not count toward the five-common-seed formal gate.
                continue
            seed = int(match.group("seed"))
            if fold in by_seed.setdefault(seed, {}):
                raise ExportContractError(f"duplicate formal run for seed={seed}, fold={fold}")
            by_seed[seed][fold] = candidate
    return by_seed


def select_formal_run_groups(
    discovered: Mapping[int, Mapping[str, Path]],
    folds: Sequence[str],
    requested_seeds: Sequence[int],
    min_seeds: int,
) -> dict[int, list[Path]]:
    complete = {
        seed: [Path(mapping[fold]) for fold in folds]
        for seed, mapping in discovered.items()
        if set(mapping) == set(folds)
    }
    if len(complete) < min_seeds:
        raise ExportContractError(
            f"formal export requires at least {min_seeds} common complete seeds across "
            f"{len(folds)} LODO folds; found {sorted(complete)}"
        )
    selected = tuple(requested_seeds) if requested_seeds else tuple(sorted(complete))
    missing = sorted(set(selected) - set(complete))
    if missing:
        raise ExportContractError(f"requested seeds lack complete LODO coverage: {missing}")
    return {seed: complete[seed] for seed in selected}


def _load_physical_identities(frame: pd.DataFrame) -> np.ndarray:
    """Read only identity columns; q_R and rainfall are deliberately unopened."""

    output = np.empty(len(frame), dtype=object)
    for path_text, group in frame.groupby("trigger_registry_path", sort=False):
        if not path_text:
            raise ExportContractError("test sample lacks a Trigger identity registry path")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Trigger identity registry is missing: {path}")
        registry = pd.read_csv(
            path, keep_default_na=False, low_memory=False,
            usecols=["sample_id", "physical_event_id"],
        )
        for row_index in group.index:
            registry_index = int(frame.loc[row_index, "trigger_registry_index"])
            if not 0 <= registry_index < len(registry):
                raise ExportContractError(f"Trigger identity registry index is invalid: {path}")
            value = registry.iloc[registry_index]
            if str(value.sample_id) != str(frame.loc[row_index, "sample_id"]):
                raise ExportContractError(f"Trigger identity sample mismatch: {path}")
            physical = str(value.physical_event_id)
            if not physical:
                raise ExportContractError(f"missing physical_event_id: {path}[{registry_index}]")
            output[row_index] = physical
    return output.astype(str)


class ParentOOFPixelDataset(Dataset[dict[str, Any]]):
    """Read test pixels without loading Material/Trigger feature values."""

    def __init__(
        self,
        manifest_path: Path,
        split_path: Path,
        fold_id: str,
        *,
        variant: str,
        normalization: Mapping[str, Any],
        max_samples: int = 0,
    ) -> None:
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        split = pd.read_csv(split_path, keep_default_na=False)
        selected = split[
            split.fold_id.astype(str).eq(str(fold_id)) & split.role.astype(str).eq("test")
        ].copy()
        if selected.empty:
            raise ExportContractError(f"fold has no test samples: {fold_id}")
        if selected.sample_id.duplicated().any():
            raise ExportContractError(f"fold repeats test sample IDs: {fold_id}")
        frame = manifest.merge(
            selected[["sample_id", "canonical_event_id", "role", "role_reason"]],
            on=["sample_id", "canonical_event_id"], how="inner", validate="one_to_one",
        )
        if len(frame) != len(selected):
            raise ExportContractError(f"manifest/split test membership mismatch: {fold_id}")
        if max_samples > 0:
            frame = frame.iloc[:max_samples].copy()
        if frame.empty or not frame.core_assets_ready.astype(bool).all():
            raise ExportContractError(f"test rows are empty or core-incomplete: {fold_id}")
        self.frame = frame.reset_index(drop=True)
        self.variant = variant
        self.physical_event_ids = _load_physical_identities(self.frame)
        self._h5: dict[str, h5py.File] = {}
        mean = np.asarray(normalization.get("terrain_mean", []), np.float32)
        std = np.asarray(normalization.get("terrain_std", []), np.float32)
        if variant == "VT" and (mean.shape != (9,) or std.shape != (9,) or np.any(std <= 0)):
            raise ExportContractError("VT checkpoint lacks valid train-only Terrain normalization")
        self.terrain_mean = mean[:, None, None] if len(mean) else None
        self.terrain_std = std[:, None, None] if len(std) else None

    def __len__(self) -> int:
        return len(self.frame)

    def _handle(self, path_text: str) -> h5py.File:
        if path_text not in self._h5:
            path = Path(path_text)
            if not path.is_file():
                raise FileNotFoundError(f"input HDF5 is missing: {path}")
            self._h5[path_text] = h5py.File(path, "r")
        return self._h5[path_text]

    @staticmethod
    def _assert_sample(handle: h5py.File, index: int, expected: str, path: str) -> None:
        observed = decode(handle["sample_id"][index])
        if observed != expected:
            raise ExportContractError(
                f"HDF5 sample mismatch at {path}[{index}]: {observed!r} != {expected!r}"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        sample_id = str(row.sample_id)
        base_path, optical_path, terrain_path = map(
            str, (row.base_h5_path, row.optical_h5_path, row.terrain_h5_path)
        )
        base_index, optical_index, terrain_index = map(
            int, (row.base_h5_index, row.optical_h5_index, row.terrain_h5_index)
        )
        base = self._handle(base_path)
        optical = self._handle(optical_path)
        terrain_handle = self._handle(terrain_path)
        self._assert_sample(base, base_index, sample_id, base_path)
        self._assert_sample(optical, optical_index, sample_id, optical_path)
        self._assert_sample(terrain_handle, terrain_index, sample_id, terrain_path)
        optical_value = np.asarray(optical["optical"][optical_index], np.float32) / 10_000.0
        mask = np.asarray(base["mask"][base_index], np.float32)
        base_valid = np.asarray(base["valid_mask"][base_index], np.uint8)
        optical_valid = np.asarray(optical["optical_valid"][optical_index], np.uint8)
        terrain_valid = np.asarray(terrain_handle["terrain_valid"][terrain_index], np.uint8)
        valid = np.logical_and.reduce(
            (base_valid > 0, optical_valid > 0, terrain_valid > 0)
        ).astype(np.uint8)
        item: dict[str, Any] = {
            "optical": torch.from_numpy(optical_value),
            "temporal_coords": torch.from_numpy(
                np.asarray(optical["temporal_coords"][optical_index], np.float32)
            ),
            "location_coords": torch.from_numpy(
                np.asarray(optical["location_coords"][optical_index], np.float32)
            ),
            "mask": torch.from_numpy(mask),
            "valid_mask": torch.from_numpy(valid),
            "sample_id": sample_id,
            "canonical_event_id": str(row.canonical_event_id),
            "physical_event_id": str(self.physical_event_ids[index]),
        }
        if self.variant == "VT":
            indices = [int(value) for value in str(row.terrain_channel_indices).split(";")]
            terrain = np.asarray(terrain_handle["terrain"][terrain_index], np.float32)[indices]
            q_t = terrain_valid.astype(np.float32)
            if q_t.ndim == 2:
                q_t = q_t[None]
            if q_t.shape[0] != 1:
                q_t = np.all(q_t > 0, axis=0, keepdims=True).astype(np.float32)
            assert self.terrain_mean is not None and self.terrain_std is not None
            normalized = ((terrain - self.terrain_mean) / self.terrain_std) * q_t
            item["terrain"] = torch.from_numpy(normalized.astype(np.float32, copy=False))
            item["q_t"] = torch.from_numpy(q_t.astype(np.float32, copy=False))
        return item

    def close(self) -> None:
        for handle in self._h5.values():
            if handle.id.valid:
                handle.close()
        self._h5.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = {}
        return state

    def __del__(self) -> None:
        if hasattr(self, "_h5"):
            self.close()


def build_parent_model(
    run: ValidatedRun,
    *,
    device: str,
    prithvi_snapshot: Path | None,
) -> tuple[trainer.RoleAwareGeoPhysAdapter, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(run.path / "checkpoint.pt", map_location="cpu", weights_only=False)
    config_args = run.config.get("args", {})
    snapshot = prithvi_snapshot or (
        Path(config_args["prithvi_snapshot"]) if config_args.get("prithvi_snapshot") else None
    )
    encoder, provenance = trainer.load_prithvi_encoder(snapshot)
    expected_prithvi = checkpoint.get("identity", {}).get("prithvi_checkpoint_sha256")
    if provenance.get("checkpoint_sha256") != expected_prithvi:
        raise ExportContractError("current Prithvi checkpoint differs from parent run identity")
    decoder_width = int(config_args.get("decoder_width", 128))
    alpha_max = float(config_args.get("alpha_max", 2.0))
    visual = trainer.PrithviEO2ChangeModel(
        encoder, decoder_width=decoder_width, freeze_encoder=True
    )
    model = trainer.RoleAwareGeoPhysAdapter(
        visual, run.variant, visual_channels=decoder_width, alpha_max=alpha_max
    )
    components = checkpoint.get("components", {})
    required = {"visual_decoder"} | ({"terrain_adapter"} if run.variant == "VT" else set())
    if required - set(components):
        raise ExportContractError(f"checkpoint lacks components: {sorted(required-set(components))}")
    getattr(model.visual, "decoder", model.visual).load_state_dict(
        components["visual_decoder"], strict=True
    )
    if run.variant == "VT":
        assert model.terrain_adapter is not None
        model.terrain_adapter.load_state_dict(components["terrain_adapter"], strict=True)
    observed_hashes = {
        name: trainer.tensor_sha256(value) for name, value in components.items()
    }
    if observed_hashes != checkpoint.get("component_sha256"):
        raise ExportContractError("checkpoint component hashes do not match serialized states")
    model = model.to(device).eval()
    return model, checkpoint, provenance


def _tensor_to_numpy(value: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def _write_prediction_archive(
    path: Path,
    *,
    sample_ids: Sequence[str],
    canonical_event_ids: Sequence[str],
    physical_event_ids: Sequence[str],
    logits: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    compression: str,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    writer = np.savez_compressed if compression == "compressed" else np.savez
    with temporary.open("wb") as stream:
        writer(
            stream,
            sample_ids=np.asarray(sample_ids),
            event_ids=np.asarray(canonical_event_ids),
            canonical_event_ids=np.asarray(canonical_event_ids),
            physical_event_ids=np.asarray(physical_event_ids),
            logits=logits,
            labels=labels,
            valid=valid,
        )
    os.replace(temporary, path)


def split_membership(split_path: Path, fold_id: str) -> dict[str, list[str]]:
    frame = pd.read_csv(split_path, keep_default_na=False)
    frame = frame[frame.fold_id.astype(str).eq(str(fold_id))]
    active = frame[frame.role.isin(("train", "val", "test"))]
    event_roles = active.groupby("canonical_event_id").role.nunique()
    if active.empty or int(event_roles.max()) != 1:
        raise ExportContractError(f"event leakage or empty split: {fold_id}")
    output = {
        role: sorted(set(active.loc[active.role.eq(role), "canonical_event_id"].astype(str)))
        for role in ("train", "val", "test")
    }
    if any(not output[role] for role in output):
        raise ExportContractError(f"split lacks train/val/test events: {fold_id}")
    return output


def export_one_run(
    run: ValidatedRun,
    stage: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_contract = split_membership(args.split, run.fold_id)
    if set(split_contract["test"]) & (
        set(split_contract["train"]) | set(split_contract["val"])
    ):
        raise ExportContractError(f"test event seen by parent training/validation: {run.fold_id}")
    model, checkpoint, provenance = build_parent_model(
        run, device=args.device, prithvi_snapshot=args.prithvi_snapshot
    )
    dataset = ParentOOFPixelDataset(
        args.manifest, args.split, run.fold_id, variant=run.variant,
        normalization=checkpoint.get("normalization", {}), max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )
    first = dataset[0]
    mask = trainer.ensure_map(first["mask"])
    height, width = int(mask.shape[-2]), int(mask.shape[-1])
    n_samples = len(dataset)
    work = stage / f".{run.fold_id}.work"
    work.mkdir()
    logits_map = np.memmap(work / "logits.f32", mode="w+", dtype=np.float32, shape=(n_samples, 1, height, width))
    labels_map = np.memmap(work / "labels.u8", mode="w+", dtype=np.uint8, shape=(n_samples, 1, height, width))
    valid_map = np.memmap(work / "valid.u8", mode="w+", dtype=np.uint8, shape=(n_samples, 1, height, width))
    sample_ids: list[str] = []
    canonical_ids: list[str] = []
    physical_ids: list[str] = []
    offset = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                tensor_batch = {
                    key: value.to(args.device, non_blocking=True)
                    if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                output = model(tensor_batch)
                logits = _tensor_to_numpy(output["logits"], np.float32)
                labels = _tensor_to_numpy(trainer.ensure_map(tensor_batch["mask"]), np.uint8)
                valid = _tensor_to_numpy(
                    trainer.ensure_map(tensor_batch["valid_mask"]).bool(), np.uint8
                )
                batch_size = len(logits)
                if logits.shape[1:] != (1, height, width):
                    raise ExportContractError(f"unexpected parent logits shape: {logits.shape}")
                logits_map[offset:offset + batch_size] = logits
                labels_map[offset:offset + batch_size] = labels
                valid_map[offset:offset + batch_size] = valid
                sample_ids.extend(map(str, batch["sample_id"]))
                canonical_ids.extend(map(str, batch["canonical_event_id"]))
                physical_ids.extend(map(str, batch["physical_event_id"]))
                offset += batch_size
        if offset != n_samples or len(set(sample_ids)) != n_samples:
            raise ExportContractError("exported sample coverage is incomplete or duplicated")
        if set(canonical_ids) - set(split_contract["test"]):
            raise ExportContractError("export contains events outside the unseen test split")
        logits_map.flush(); labels_map.flush(); valid_map.flush()
        prediction_name = f"{run.fold_id}.parent_oof.npz"
        prediction_path = stage / prediction_name
        _write_prediction_archive(
            prediction_path, sample_ids=sample_ids,
            canonical_event_ids=canonical_ids, physical_event_ids=physical_ids,
            logits=logits_map, labels=labels_map, valid=valid_map,
            compression=args.compression,
        )
    finally:
        dataset.close()
        del logits_map, labels_map, valid_map
        shutil.rmtree(work, ignore_errors=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    prediction_hash = sha256_file(prediction_path)
    sample_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    canonical_hash = hashlib.sha256("\n".join(canonical_ids).encode()).hexdigest()
    physical_hash = hashlib.sha256("\n".join(physical_ids).encode()).hexdigest()
    receipt_name = f"{run.fold_id}.producer_receipt.json"
    receipt_path = stage / receipt_name
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "fold_id": run.fold_id,
        "seed": run.seed,
        "variant": run.variant,
        "identity_anchor": (
            "visual-only V parent" if run.variant == "V"
            else "VT Terrain-conditioned parent sensitivity; not the default Trigger anchor"
        ),
        "prediction_role": "parent_oof",
        "prediction_value_type": "raw_logits",
        "selection_uses_holdout_labels": False,
        "threshold_uses_holdout_labels": False,
        "threshold_probability": run.threshold_probability,
        "threshold_source": run.threshold_source,
        "checkpoint_sha256": run.checkpoint_sha256,
        "source_run": str(run.path),
        "source_done_sha256": sha256_file(run.path / "DONE.json"),
        "source_result_sha256": sha256_file(run.path / "result.json"),
        "source_config_sha256": sha256_file(run.path / "config.json"),
        "manifest_sha256": sha256_file(args.manifest),
        "split_sha256": sha256_file(args.split),
        "protocol_summary_sha256": sha256_file(args.protocol_summary),
        "prithvi_checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "training_event_ids": split_contract["train"],
        "validation_event_ids": split_contract["val"],
        "held_out_event_ids": sorted(set(canonical_ids)),
        "held_out_physical_event_ids": sorted(set(physical_ids)),
        "sample_id_sha256": sample_hash,
        "canonical_event_id_sha256": canonical_hash,
        "physical_event_id_sha256": physical_hash,
        "n_samples": n_samples,
        "pixel_shape": [1, height, width],
        "logit_dtype": "float32",
        "label_dtype": "uint8",
        "valid_dtype": "uint8",
        "q_R_read_or_used": False,
        "formal": not args.nonformal,
        "nonformal_reason": (
            f"engineering smoke limited to first {args.max_samples} manifest-ordered test samples"
            if args.nonformal else None
        ),
        "prediction_path": prediction_name,
        "prediction_sha256": prediction_hash,
    }
    atomic_json(receipt_path, receipt)
    return {
        "fold_id": run.fold_id,
        "prediction_path": prediction_name,
        "prediction_sha256": prediction_hash,
        "producer_receipt_path": receipt_name,
        "producer_receipt_sha256": sha256_file(receipt_path),
        "held_out_event_ids": sorted(set(canonical_ids)),
        "n_samples": n_samples,
    }


def export_seed_group(
    run_dirs: Sequence[Path],
    target: Path,
    args: argparse.Namespace,
    *,
    expected_folds: Sequence[str],
) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite export directory: {target}")
    validated = [
        validate_complete_run(
            path, expected_variant=args.variant,
            manifest_path=args.manifest, split_path=args.split,
        )
        for path in run_dirs
    ]
    seeds = {item.seed for item in validated}
    if len(seeds) != 1:
        raise ExportContractError(f"cannot mix checkpoint seeds: {sorted(seeds)}")
    fold_ids = [item.fold_id for item in validated]
    if len(fold_ids) != len(set(fold_ids)):
        raise ExportContractError(f"duplicate folds in one seed export: {fold_ids}")
    if not args.nonformal and set(fold_ids) != set(expected_folds):
        raise ExportContractError("formal export must contain every current LODO fold")
    before = {
        "manifest": sha256_file(args.manifest),
        "split": sha256_file(args.split),
        "protocol_summary": sha256_file(args.protocol_summary),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(f"stale export staging directory exists: {stage}")
    stage.mkdir()
    started = time.time()
    try:
        entries = [export_one_run(run, stage, args) for run in validated]
        after = {
            "manifest": sha256_file(args.manifest),
            "split": sha256_file(args.split),
            "protocol_summary": sha256_file(args.protocol_summary),
        }
        if before != after:
            raise ExportContractError("input manifest/split/protocol changed during export")
        seed = next(iter(seeds))
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "export_schema_version": EXPORT_SCHEMA,
            "formal": not args.nonformal,
            "selection_uses_labels": False,
            "all_available_parent_oof_folds_included": (
                not args.nonformal and set(fold_ids) == set(expected_folds)
            ),
            "seed": seed,
            "variant": args.variant,
            "identity_anchor": (
                "visual-only V parent" if args.variant == "V"
                else "VT Terrain-conditioned sensitivity"
            ),
            "max_samples": args.max_samples if args.nonformal else 0,
            "input_hashes": before,
            "entries": sorted(entries, key=lambda item: item["fold_id"]),
        }
        atomic_json(stage / "parent_oof_manifest.json", manifest)
        hashes = {
            path.name: sha256_file(path)
            for path in sorted(stage.iterdir()) if path.is_file()
        }
        atomic_json(stage / "DONE.json", {
            "schema_version": EXPORT_SCHEMA,
            "status": "complete_nonformal" if args.nonformal else "complete",
            "seed": seed,
            "variant": args.variant,
            "n_folds": len(entries),
            "n_samples": int(sum(entry["n_samples"] for entry in entries)),
            "elapsed_seconds": time.time() - started,
            "artifact_sha256": hashes,
        })
        os.replace(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return json.loads((target / "DONE.json").read_text(encoding="utf-8"))


def validate_args(args: argparse.Namespace) -> None:
    args.manifest = args.manifest.resolve()
    args.protocol_summary = args.protocol_summary.resolve()
    args.split = args.split.resolve()
    args.run_root = args.run_root.resolve()
    args.outdir = args.outdir.resolve()
    if args.batch_size <= 0 or args.num_workers < 0 or args.min_seeds < 1:
        raise ValueError("batch-size/min-seeds must be positive and workers nonnegative")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")
    if args.nonformal:
        if not args.run_dir or args.max_samples <= 0:
            raise ValueError("--nonformal requires explicit --run-dir and --max-samples > 0")
    elif args.max_samples > 0:
        raise ValueError("--max-samples is forbidden for formal exports")
    elif args.run_dir:
        raise ValueError("explicit --run-dir is reserved for --nonformal engineering smoke")
    for path in (args.manifest, args.protocol_summary, args.split):
        if not path.is_file():
            raise FileNotFoundError(f"required current input is missing: {path}")
    summary = _read_json(args.protocol_summary, "protocol summary")
    if summary.get("validation_status") != "PASS":
        raise ExportContractError("current protocol summary validation is not PASS")
    expected_manifest = summary.get("outputs", {}).get("manifest", {}).get("sha256")
    if expected_manifest and expected_manifest != sha256_file(args.manifest):
        raise ExportContractError("current unified manifest differs from protocol summary")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    folds = expected_lodo_folds(args.split)
    if args.nonformal:
        done = export_seed_group(
            args.run_dir, args.outdir, args, expected_folds=folds
        )
        print(json.dumps(json_safe(done), indent=2, sort_keys=True, allow_nan=False))
        return 0
    discovered = discover_formal_runs(args.run_root, folds, args.variant)
    groups = select_formal_run_groups(
        discovered, folds, parse_seed_list(args.seeds), args.min_seeds
    )
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite formal export root: {args.outdir}")
    args.outdir.mkdir(parents=True)
    summaries = []
    try:
        for seed, runs in sorted(groups.items()):
            summaries.append(export_seed_group(
                runs, args.outdir / f"seed{seed}", args, expected_folds=folds
            ))
        atomic_json(args.outdir / "export_index.json", {
            "schema_version": EXPORT_SCHEMA,
            "status": "complete",
            "variant": args.variant,
            "minimum_common_seeds": args.min_seeds,
            "seeds": [item["seed"] for item in summaries],
            "seed_manifests": {
                str(item["seed"]): f"seed{item['seed']}/parent_oof_manifest.json"
                for item in summaries
            },
        })
    except Exception:
        # Completed seed subdirectories are retained as immutable evidence, but
        # no top-level completion index is written for a partial formal export.
        raise
    print(json.dumps(json_safe(summaries), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
