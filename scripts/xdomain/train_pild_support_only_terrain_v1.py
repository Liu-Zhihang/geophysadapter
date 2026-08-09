#!/usr/bin/env python3
"""Train a support-only Terrain expert against a frozen PILD protocol.

The trainer reuses ``UnifiedPILDSen12Dataset`` for manifest and split gating,
but deliberately bypasses its ``__getitem__`` method. Only the label/validity
cache and the audited Terrain cache are opened. No optical tensor, visual
feature, or visual prediction can enter optimization or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pild_sen12_training_loader_v2 import (  # noqa: E402
    DatasetEventPatchBalancedSampler,
    NaturalPatchSampler,
    SourceEventPatchBalancedSampler,
    TemperedDatasetEventPatchSampler,
    UnifiedPILDSen12Dataset,
    decode,
    sha256_file,
)
from sen12_terrain_v2 import SupportOnlyMultiScaleTerrainPyramid  # noqa: E402
from train_pild_sen12_roleaware_v1 import (  # noqa: E402
    BinaryHistogram,
    COMMON_TERRAIN9_NAMES,
    metrics_from_counts,
    resolve_terrain_contract,
    state_to_cpu,
    tensor_sha256,
    validate_protocol_schema,
)


DEFAULT_METADATA = PROJECT_ROOT / "metadata/pild_sen12_training_v2"
DEFAULT_MANIFEST = DEFAULT_METADATA / "unified_sample_manifest_v2.csv"
DEFAULT_SUMMARY = DEFAULT_METADATA / "protocol_summary_v2.json"
DEFAULT_SPLIT = DEFAULT_METADATA / "event_isolated_split_v2.csv"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(dict(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_parent_v_checkpoint(
    checkpoint_path: Path,
    *,
    manifest_sha256: str,
    split_sha256: str,
    fold_id: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and authenticate the exact PILD V parent used by the evaluator."""

    checkpoint_path = checkpoint_path.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("variant") != "V":
        raise RuntimeError(f"parent checkpoint must be variant V, got {payload.get('variant')!r}")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("parent checkpoint lacks identity mapping")
    expected = {
        "manifest_sha256": str(manifest_sha256),
        "split_sha256": str(split_sha256),
        "fold_id": str(fold_id),
        "seed": int(seed),
    }
    mismatch = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"parent checkpoint identity mismatch: {mismatch}")
    prithvi_hash = str(identity.get("prithvi_checkpoint_sha256", ""))
    if len(prithvi_hash) != 64:
        raise RuntimeError("parent checkpoint lacks a valid Prithvi checkpoint SHA-256")
    if payload.get("threshold_source") != "visual_validation":
        raise RuntimeError("PILD V threshold must originate from visual_validation")
    threshold = float(payload.get("threshold", float("nan")))
    if not 0.0 < threshold < 1.0:
        raise RuntimeError(f"invalid parent validation threshold: {threshold}")
    components = payload.get("components")
    component_hashes = payload.get("component_sha256")
    if not isinstance(components, Mapping) or "visual_decoder" not in components:
        raise RuntimeError("parent checkpoint lacks components.visual_decoder")
    if not isinstance(component_hashes, Mapping):
        raise RuntimeError("parent checkpoint lacks component_sha256")
    observed_component_hash = tensor_sha256(components["visual_decoder"])
    expected_component_hash = component_hashes.get("visual_decoder")
    if observed_component_hash != expected_component_hash:
        raise RuntimeError(
            "parent visual decoder tensor hash differs from component_sha256"
        )
    receipt = {
        "path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "visual_decoder_sha256": observed_component_hash,
        "prithvi_checkpoint_sha256": prithvi_hash,
        "threshold": threshold,
        "threshold_source": payload["threshold_source"],
    }
    return payload, receipt


class TerrainOnlyDataset(Dataset[dict[str, Any]]):
    """Terrain-only view over a protocol-gated unified dataset.

    ``base.__getitem__`` is never called. In particular, ``optical_h5_path`` is
    neither checked nor opened here.
    """

    def __init__(
        self,
        base: UnifiedPILDSen12Dataset,
        *,
        terrain_names: tuple[str, ...] = COMMON_TERRAIN9_NAMES,
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
    ) -> None:
        self.frame = base.frame.copy().reset_index(drop=True)
        self.terrain_names = tuple(terrain_names)
        self.channels = len(self.terrain_names)
        self.mean = np.zeros(self.channels, dtype=np.float32) if mean is None else np.asarray(mean, dtype=np.float32)
        self.scale = np.ones(self.channels, dtype=np.float32) if scale is None else np.asarray(scale, dtype=np.float32)
        if self.mean.shape != (self.channels,) or self.scale.shape != (self.channels,):
            raise ValueError("Terrain normalization must match the audited schema")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Terrain normalization contains invalid values")
        self._h5: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _handle(self, path_text: str) -> h5py.File:
        if path_text not in self._h5:
            path = Path(path_text)
            if not path.is_file():
                raise RuntimeError(f"Terrain-only cache is missing: {path}")
            self._h5[path_text] = h5py.File(path, "r")
        return self._h5[path_text]

    @staticmethod
    def _assert_sample(handle: h5py.File, index: int, expected: str, path: str) -> None:
        observed = decode(handle["sample_id"][index])
        if observed != expected:
            raise RuntimeError(
                f"HDF5 identity mismatch at {path}[{index}]: "
                f"observed={observed!r}, expected={expected!r}"
            )

    def read_raw(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        sample_id = str(row["sample_id"])
        base_path = str(row["base_h5_path"])
        terrain_path = str(row["terrain_h5_path"])
        base_index = int(row["base_h5_index"])
        terrain_index = int(row["terrain_h5_index"])
        base = self._handle(base_path)
        terrain_handle = self._handle(terrain_path)
        self._assert_sample(base, base_index, sample_id, base_path)
        self._assert_sample(terrain_handle, terrain_index, sample_id, terrain_path)
        channel_indices = np.asarray(
            [int(value) for value in str(row["terrain_channel_indices"]).split(";")],
            dtype=np.int64,
        )
        if channel_indices.shape != (self.channels,):
            raise RuntimeError(
                f"sample {sample_id} does not expose exactly {self.channels} Terrain channels"
            )
        terrain = np.asarray(
            terrain_handle["terrain"][terrain_index], dtype=np.float32
        )[channel_indices]
        terrain_valid = np.asarray(
            terrain_handle["terrain_valid"][terrain_index], dtype=np.uint8
        )
        mask = np.asarray(base["mask"][base_index], dtype=np.float32)
        base_valid = np.asarray(base["valid_mask"][base_index], dtype=np.uint8)
        if terrain_valid.ndim == 3 and terrain_valid.shape[0] == 1:
            terrain_valid = terrain_valid[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if base_valid.ndim == 3 and base_valid.shape[0] == 1:
            base_valid = base_valid[0]
        q_t = np.logical_and(
            terrain_valid > 0,
            np.isfinite(terrain).all(axis=0),
        ).astype(np.float32)
        valid = np.logical_and(base_valid > 0, q_t > 0).astype(np.float32)
        return {
            "terrain": terrain,
            "q_t": q_t[None],
            "mask": mask[None],
            "valid_mask": valid[None],
            "sample_id": sample_id,
            "dataset_id": str(row["dataset_id"]),
            "source_id": str(row["source_id"]),
            "source_event_id": str(row["source_event_id"]),
            "canonical_event_id": str(row["canonical_event_id"]),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.read_raw(index)
        terrain = item["terrain"]
        q_t = item["q_t"]
        terrain = ((terrain - self.mean[:, None, None]) / self.scale[:, None, None])
        terrain = np.where(q_t > 0, terrain, 0.0)
        item["terrain"] = torch.from_numpy(terrain.astype(np.float32))
        item["q_t"] = torch.from_numpy(q_t.astype(np.float32))
        item["mask"] = torch.from_numpy(item["mask"].astype(np.float32))
        item["valid_mask"] = torch.from_numpy(item["valid_mask"].astype(np.float32))
        return item

    def close(self) -> None:
        for handle in getattr(self, "_h5", {}).values():
            if handle.id.valid:
                handle.close()
        if hasattr(self, "_h5"):
            self._h5.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = {}
        return state

    def __del__(self) -> None:
        self.close()


def fit_terrain_normalization(dataset: TerrainOnlyDataset) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sums = np.zeros(dataset.channels, dtype=np.float64)
    squares = np.zeros(dataset.channels, dtype=np.float64)
    counts = np.zeros(dataset.channels, dtype=np.int64)
    for index in range(len(dataset)):
        item = dataset.read_raw(index)
        terrain = np.asarray(item["terrain"], dtype=np.float64)
        keep = np.asarray(item["q_t"][0] > 0)
        for channel in range(dataset.channels):
            values = terrain[channel][keep]
            values = values[np.isfinite(values)]
            sums[channel] += values.sum()
            squares[channel] += np.square(values).sum()
            counts[channel] += len(values)
    if (counts == 0).any():
        missing = [dataset.terrain_names[index] for index in np.flatnonzero(counts == 0)]
        raise RuntimeError(f"no valid train pixels for Terrain channels: {missing}")
    mean = sums / counts
    variance = np.maximum(squares / counts - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.0)
    audit = {
        "fit_scope": "train-only-valid-terrain-pixels",
        "feature_names": list(dataset.terrain_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "valid_pixel_counts": counts.tolist(),
    }
    return mean.astype(np.float32), scale.astype(np.float32), audit


def estimate_pos_weight(dataset: TerrainOnlyDataset) -> float:
    positive = 0
    negative = 0
    for index in range(len(dataset)):
        item = dataset.read_raw(index)
        target = item["mask"][0] >= 0.5
        valid = item["valid_mask"][0] > 0
        positive += int(np.logical_and(target, valid).sum())
        negative += int(np.logical_and(~target, valid).sum())
    if positive == 0:
        raise RuntimeError("training fold has no valid positive Terrain pixels")
    return float(np.clip(negative / positive, 1.0, 100.0))


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def terrain_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    pos_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    bce = (raw * valid).sum() / valid.sum().clamp_min(1.0)
    return bce + float(dice_weight) * masked_dice_loss(logits, target, valid)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    histogram = BinaryHistogram()
    losses: list[float] = []
    for batch in loader:
        terrain = batch["terrain"].to(device)
        target = batch["mask"].to(device)
        valid = batch["valid_mask"].to(device)
        logits, _ = model(terrain)
        probability = torch.sigmoid(logits)
        histogram.update(probability, target, valid)
        losses.append(
            float(
                F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                ).mul(valid).sum().div(valid.sum().clamp_min(1.0)).item()
            )
        )
    counts = histogram.counts(threshold)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "ap": histogram.average_precision(),
        **metrics_from_counts(counts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an audited support-only Terrain expert for one PILD fold and seed."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parent-v-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epoch-samples", type=int)
    parser.add_argument(
        "--sampling-mode",
        choices=("balanced", "dataset_balanced", "tempered", "natural"),
        default="balanced",
        help=(
            "natural preserves patch proportions; balanced uses source/event; "
            "dataset_balanced uses dataset/event hierarchy"
        ),
    )
    parser.add_argument("--dataset-temperature", type=float, default=0.75)
    parser.add_argument("--event-temperature", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-steps", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    schema = validate_protocol_schema(
        args.manifest, args.protocol_summary, args.split, args.fold_id
    )
    terrain_names, terrain_scale_groups, terrain_schema_id = resolve_terrain_contract(
        schema["terrain_channels"]
    )
    _, parent_receipt = validate_parent_v_checkpoint(
        args.parent_v_checkpoint,
        manifest_sha256=schema["manifest_sha256"],
        split_sha256=schema["split_sha256"],
        fold_id=args.fold_id,
        seed=args.seed,
    )
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    stage = outdir.parent / f".{outdir.name}.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    command = shlex.join([sys.executable, *(sys.argv if argv is None else [__file__, *argv])])
    (stage / "run.log").write_text(command + "\n", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with (stage / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    started = time.time()
    set_seed(args.seed)
    base_train = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.split,
        fold_id=args.fold_id,
        role="train",
        readiness="core",
    )
    base_val = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.split,
        fold_id=args.fold_id,
        role="val",
        readiness="core",
    )
    raw_train = TerrainOnlyDataset(base_train, terrain_names=terrain_names)
    mean, scale, normalization = fit_terrain_normalization(raw_train)
    train_dataset = TerrainOnlyDataset(
        base_train, terrain_names=terrain_names, mean=mean, scale=scale
    )
    val_dataset = TerrainOnlyDataset(
        base_val, terrain_names=terrain_names, mean=mean, scale=scale
    )
    pos_weight = estimate_pos_weight(raw_train)
    epoch_samples = args.epoch_samples or len(train_dataset)
    if args.sampling_mode == "tempered":
        train_sampler = TemperedDatasetEventPatchSampler(
            train_dataset.frame,
            num_samples=epoch_samples,
            seed=args.seed,
            dataset_temperature=args.dataset_temperature,
            event_temperature=args.event_temperature,
        )
    else:
        sampler_class = {
            "natural": NaturalPatchSampler,
            "balanced": SourceEventPatchBalancedSampler,
            "dataset_balanced": DatasetEventPatchBalancedSampler,
        }[args.sampling_mode]
        train_sampler = sampler_class(
            train_dataset.frame,
            num_samples=epoch_samples,
            seed=args.seed,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    device = torch.device(args.device)
    model = SupportOnlyMultiScaleTerrainPyramid(
        len(terrain_names), terrain_scale_groups
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_epoch = -1
    best_key = (-float("inf"), -float("inf"))
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader):
            if args.max_train_steps and step >= args.max_train_steps:
                break
            terrain = batch["terrain"].to(device)
            target = batch["mask"].to(device)
            valid = batch["valid_mask"].to(device)
            logits, _ = model(terrain)
            loss = terrain_loss(
                logits,
                target,
                valid,
                pos_weight=pos_weight,
                dice_weight=args.dice_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        validation = validate(model, val_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "validation": validation,
        }
        history.append(row)
        key = (float(validation["ap"]), float(validation["iou"]))
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = state_to_cpu(model)
        log(
            f"epoch={epoch:03d} train_loss={row['train_loss']} "
            f"val_ap={validation['ap']:.6f} val_iou={validation['iou']:.6f}"
        )
    if best_state is None:
        raise RuntimeError("training produced no selectable Terrain checkpoint")
    model.load_state_dict(best_state, strict=True)
    terrain_hash = tensor_sha256(best_state)
    checkpoint = {
        "schema_version": "pild_support_only_terrain_checkpoint.v1",
        "contract": (
            f"{terrain_schema_id} support-only Terrain; "
            "no optical or visual feature input"
        ),
        "identity": {
            "manifest_sha256": schema["manifest_sha256"],
            "protocol_summary_sha256": sha256_file(args.protocol_summary),
            "split_sha256": schema["split_sha256"],
            "fold_id": str(args.fold_id),
            "seed": int(args.seed),
            "parent_v_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
            "parent_visual_decoder_sha256": parent_receipt["visual_decoder_sha256"],
            "parent_prithvi_checkpoint_sha256": parent_receipt["prithvi_checkpoint_sha256"],
        },
        "parent_v_receipt": parent_receipt,
        "terrain_state_dict": best_state,
        "terrain_state_sha256": terrain_hash,
        "terrain_schema_id": terrain_schema_id,
        "terrain_channel_order": list(terrain_names),
        "terrain_scale_groups": {
            "fine": list(terrain_scale_groups.fine),
            "meso": list(terrain_scale_groups.meso),
            "macro": list(terrain_scale_groups.macro),
        },
        "normalization": normalization,
        "sampling": {
            "mode": args.sampling_mode,
            "epoch_samples": int(epoch_samples),
        },
        "best_epoch": best_epoch,
        "selection_metric": "validation_terrain_only_average_precision",
        "pos_weight": pos_weight,
        "history": history,
    }
    checkpoint_path = stage / "terrain_expert.pt"
    torch.save(checkpoint, checkpoint_path)
    result = {
        "schema_version": "pild_support_only_terrain_result.v1",
        "status": "COMPLETE",
        "fold_id": str(args.fold_id),
        "seed": int(args.seed),
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch]["validation"],
        "terrain_checkpoint": str((outdir / "terrain_expert.pt")),
        "terrain_checkpoint_sha256": sha256_file(checkpoint_path),
        "terrain_state_sha256": terrain_hash,
        "parent_v_receipt": parent_receipt,
        "schema_validation": schema,
        "normalization": normalization,
        "sampling": {
            "mode": args.sampling_mode,
            "epoch_samples": int(epoch_samples),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(stage / "result.json", result)
    write_json(
        stage / "DONE.json",
        {
            "status": "COMPLETE",
            "terrain_checkpoint_sha256": result["terrain_checkpoint_sha256"],
            "result_sha256": sha256_file(stage / "result.json"),
        },
    )
    os.replace(stage, outdir)
    log_message = (
        f"completed support-only Terrain run: {outdir} "
        f"(best_epoch={best_epoch}, val_ap={history[best_epoch]['validation']['ap']:.6f})"
    )
    print(log_message, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
