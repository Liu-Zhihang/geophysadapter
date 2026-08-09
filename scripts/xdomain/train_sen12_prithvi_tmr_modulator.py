#!/usr/bin/env python3
"""Train bounded Material/Trigger dosage modulators over a frozen Sen12 VT model.

Terrain is the only dense correction direction. Material and Trigger are
sample/event-level scalars that can only modulate the frozen Terrain update by
at most +/-25%. Test labels are not read until validation AP has selected the
modulator checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_sen12_prithvi_terrain_v2 as trainer  # noqa: E402


CONTEXT_ROOT = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v1"
MATERIAL_CSV = CONTEXT_ROOT / "material_sample_registry.csv"
TRIGGER_CSV = CONTEXT_ROOT / "trigger_sample_registry_v1.csv"
MODULATION_BOUND = 0.25
ROUTING_LOW = 0.30
ROUTING_HIGH = 0.70
ROUTING_ALPHA = 4.0
ROUTING_MARGIN = 1.0
METHOD_BY_MODE = {"material": "VTM", "trigger": "VTR", "joint": "VTMR"}

MATERIAL_NUMERIC_COLUMNS = tuple(
    f"soil_{name}_{depth}_{stat}_raw"
    for name in ("clay", "sand", "silt", "cec", "soc", "bdod", "cfvo", "phh2o")
    for depth in ("0_5cm", "5_15cm")
    for stat in ("mean", "local_std")
) + (
    "awc_0_10_aligned_mm",
    "awc_10_30_aligned_mm",
    "awc_30_60_aligned_mm",
    "awc_60_100_aligned_mm",
    "awc_100_200_aligned_mm",
    "awc_0_200_aligned_mm",
)
TRIGGER_COLUMNS = (
    "rain_d7_antecedent_case_mm",
    "rain_d7_wrongtime_median_mm",
    "rain_d7_case_minus_wrongtime_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--mode", choices=("material", "trigger", "joint"), default="joint")
    parser.add_argument("--seed", type=int, default=20260771)
    parser.add_argument("--visual-checkpoint", type=Path)
    parser.add_argument("--terrain-checkpoint", type=Path)
    parser.add_argument("--material-registry", type=Path, default=MATERIAL_CSV)
    parser.add_argument("--trigger-registry", type=Path, default=TRIGGER_CSV)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional fold-level frozen cache shared across material/trigger/joint modes.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a synthetic CPU contract test without loading data or checkpoints.",
    )
    return parser.parse_args()


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sample_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def deterministic_prefix(values: Sequence[str], limit: int, token: str) -> list[str]:
    values = list(values)
    if limit <= 0 or limit >= len(values):
        return values
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{token}|{value}".encode()).hexdigest(),
    )[:limit]


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass
class Standardizer:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(values), values, median)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(std > 1e-6, std, 1.0)
        return cls(median.astype(np.float32), mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        filled = np.where(np.isfinite(values), values, self.median)
        return ((filled - self.mean) / self.std).astype(np.float32)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class ContextTable:
    """Outer-train-fitted context vectors keyed by immutable sample identity."""

    def __init__(
        self,
        material_csv: Path,
        trigger_csv: Path,
        all_ids: Sequence[str],
        train_ids: Sequence[str],
    ) -> None:
        material = pd.read_csv(material_csv)
        trigger = pd.read_csv(trigger_csv)
        self._validate_registry(material, all_ids, "material")
        self._validate_registry(trigger, all_ids, "trigger")
        material = material.set_index("sample_id").loc[list(all_ids)]
        trigger = trigger.set_index("sample_id").loc[list(all_ids)]
        self.all_ids = list(all_ids)
        self.index = {sample_id: index for index, sample_id in enumerate(self.all_ids)}

        missing = [column for column in MATERIAL_NUMERIC_COLUMNS if column not in material]
        if missing:
            raise RuntimeError(f"material registry missing columns: {missing}")
        material_raw = material.loc[:, MATERIAL_NUMERIC_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
        lithology = material["lithology_class"].fillna("__missing__").astype(str)
        train_positions = np.asarray([self.index[value] for value in train_ids], dtype=np.int64)
        categories = sorted(set(lithology.iloc[train_positions]))
        self.lithology_categories = categories
        category_index = {value: index for index, value in enumerate(categories)}
        one_hot = np.zeros((len(material), len(categories)), dtype=np.float32)
        for row_index, value in enumerate(lithology):
            if value in category_index:
                one_hot[row_index, category_index[value]] = 1.0
        self.material_standardizer = Standardizer.fit(material_raw[train_positions])
        self.material = np.concatenate(
            (self.material_standardizer.transform(material_raw), one_hot), axis=1
        ).astype(np.float32)
        self.q_material = pd.to_numeric(material["q_M"], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        ).clip(0.0, 1.0)

        missing = [column for column in TRIGGER_COLUMNS if column not in trigger]
        if missing:
            raise RuntimeError(f"trigger registry missing columns: {missing}")
        trigger_raw_frame = trigger.loc[:, TRIGGER_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        # Trigger is an event-level forcing dose, never a dense boundary cue.
        trigger_event_id = trigger["physical_event_id"].astype(str)
        trigger_raw_frame = trigger_raw_frame.groupby(trigger_event_id).transform("median")
        trigger_raw = trigger_raw_frame.to_numpy(dtype=np.float32)
        trigger_raw[:, :2] = np.log1p(np.maximum(trigger_raw[:, :2], 0.0))
        trigger_raw[:, 2] = signed_log1p(trigger_raw[:, 2])
        wrongtime_raw = trigger_raw.copy()
        wrongtime_raw[:, 0] = wrongtime_raw[:, 1]
        wrongtime_raw[:, 2] = 0.0
        self.trigger_standardizer = Standardizer.fit(trigger_raw[train_positions])
        self.trigger = self.trigger_standardizer.transform(trigger_raw)
        self.trigger_wrongtime = self.trigger_standardizer.transform(wrongtime_raw)
        self.q_trigger = pd.to_numeric(trigger["q_R"], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        ).clip(0.0, 1.0)

    @staticmethod
    def _validate_registry(frame: pd.DataFrame, all_ids: Sequence[str], name: str) -> None:
        if "sample_id" not in frame:
            raise RuntimeError(f"{name} registry has no sample_id")
        if frame["sample_id"].duplicated().any():
            duplicate = frame.loc[frame["sample_id"].duplicated(), "sample_id"].iloc[0]
            raise RuntimeError(f"duplicate {name} sample_id: {duplicate}")
        registered = set(frame["sample_id"].astype(str))
        expected = set(all_ids)
        if registered != expected:
            raise RuntimeError(
                f"{name} identity mismatch: missing={len(expected-registered)}, extra={len(registered-expected)}"
            )

    @property
    def material_dim(self) -> int:
        return int(self.material.shape[1])

    @property
    def trigger_dim(self) -> int:
        return int(self.trigger.shape[1])

    def arrays(self, sample_ids: Sequence[str]) -> dict[str, torch.Tensor]:
        positions = np.asarray([self.index[value] for value in sample_ids], dtype=np.int64)
        return {
            "material": torch.from_numpy(self.material[positions]),
            "q_material": torch.from_numpy(self.q_material[positions, None]),
            "trigger": torch.from_numpy(self.trigger[positions]),
            "trigger_wrongtime": torch.from_numpy(self.trigger_wrongtime[positions]),
            "q_trigger": torch.from_numpy(self.q_trigger[positions, None]),
        }

    def provenance(self, material_csv: Path, trigger_csv: Path, train_ids: Sequence[str]) -> dict[str, Any]:
        return {
            "material_registry": file_signature(material_csv),
            "trigger_registry": file_signature(trigger_csv),
            "outer_train_sample_sha256": sample_hash(list(train_ids)),
            "material_numeric_columns": list(MATERIAL_NUMERIC_COLUMNS),
            "lithology_categories_from_outer_train": self.lithology_categories,
            "material_standardizer": self.material_standardizer.as_dict(),
            "trigger_columns": list(TRIGGER_COLUMNS),
            "trigger_transform": ["log1p(case)", "log1p(wrongtime_median)", "signed_log1p(case-minus-background)"],
            "trigger_standardizer": self.trigger_standardizer.as_dict(),
        }


class BoundedScalarModulator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        quality = quality.clamp(0.0, 1.0)
        return 1.0 + quality * MODULATION_BOUND * torch.tanh(self.network(features))


class TMRModulator(nn.Module):
    def __init__(self, mode: str, material_dim: int, trigger_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mode = mode
        self.material = (
            BoundedScalarModulator(material_dim, hidden_dim)
            if mode in ("material", "joint")
            else None
        )
        self.trigger = (
            BoundedScalarModulator(trigger_dim, hidden_dim)
            if mode in ("trigger", "joint")
            else None
        )

    def multipliers(
        self,
        material: torch.Tensor,
        q_material: torch.Tensor,
        trigger: torch.Tensor,
        q_trigger: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = material.shape[0]
        one = torch.ones((batch, 1), dtype=material.dtype, device=material.device)
        material_multiplier = (
            self.material(material, q_material) if self.material is not None else one
        )
        trigger_multiplier = (
            self.trigger(trigger, q_trigger) if self.trigger is not None else one
        )
        return material_multiplier, trigger_multiplier

    def forward(
        self,
        visual_logits: torch.Tensor,
        terrain_direction: torch.Tensor,
        material: torch.Tensor,
        q_material: torch.Tensor,
        trigger: torch.Tensor,
        q_trigger: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        material_multiplier, trigger_multiplier = self.multipliers(
            material, q_material, trigger, q_trigger
        )
        dosage = (material_multiplier * trigger_multiplier).clamp(
            1.0 - MODULATION_BOUND, 1.0 + MODULATION_BOUND
        )
        logits = visual_logits + ROUTING_ALPHA * terrain_direction * dosage[:, :, None, None]
        return logits, {
            "material_multiplier": material_multiplier,
            "trigger_multiplier": trigger_multiplier,
            "dosage": dosage,
        }


class FrozenCacheDataset(Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __len__(self) -> int:
        return len(self.payload["sample_ids"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {"index": index, "sample_id": self.payload["sample_ids"][index]}
        item["physical_event_id"] = self.payload["physical_event_ids"][index]
        item["spatial_supergroup"] = self.payload["spatial_supergroups"][index]
        for key in (
            "visual_logits",
            "terrain_direction",
            "mask",
            "valid",
            "material",
            "q_material",
            "trigger",
            "trigger_wrongtime",
            "q_trigger",
        ):
            item[key] = self.payload[key][index]
        return item


def load_frozen_models(args: argparse.Namespace):
    terrain_payload = torch.load(args.terrain_checkpoint, map_location="cpu", weights_only=False)
    terrain_result = terrain_payload.get("result", {})
    terrain = trainer.SupportOnlyMultiScaleTerrainPyramid(
        17, trainer.NATIVE_TERRAIN_V2_SCALE_GROUPS
    )
    trainer.load_trainable_state(terrain, terrain_payload["trainable_state_dict"])
    terrain.requires_grad_(False).eval().to(args.device)

    encoder, provenance = trainer.load_prithvi_encoder()
    visual = trainer.PrithviVisualCompat(
        trainer.PrithviEO2ChangeModel(encoder, decoder_width=128, freeze_encoder=True)
    )
    visual_payload = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    visual_identity = visual_payload.get("identity", {})
    expected_visual = {"mode": "visual", "fold": args.fold}
    mismatch = {
        key: (expected, visual_identity.get(key))
        for key, expected in expected_visual.items()
        if visual_identity.get(key) != expected
    }
    expected_terrain = {"fold": args.fold}
    mismatch.update({
        f"terrain_{key}": (expected, terrain_result.get(key))
        for key, expected in expected_terrain.items()
        if terrain_result.get(key) != expected
    })
    if mismatch:
        raise RuntimeError(f"frozen checkpoint identity mismatch: {mismatch}")
    if (
        terrain_result.get("seed") is not None
        and terrain_result.get("seed") != visual_identity.get("seed")
    ):
        raise RuntimeError(
            "Terrain and visual checkpoints come from different anchor seeds: "
            f"{terrain_result.get('seed')} != {visual_identity.get('seed')}"
        )
    if terrain_result.get("seed") is None:
        print(
            "[warning] legacy Terrain checkpoint has no embedded seed; "
            "fold and frozen visual threshold were verified",
            flush=True,
        )
    if not math.isclose(
        float(terrain_result.get("visual_threshold", visual_payload["threshold"])),
        float(visual_payload["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Terrain and visual checkpoints use different thresholds")
    trainer.load_trainable_state(visual, visual_payload["trainable_state_dict"])
    visual.requires_grad_(False).eval().to(args.device)
    return terrain, terrain_result, visual, visual_payload, provenance


def frozen_terrain_direction(
    visual_logits: torch.Tensor,
    terrain_logits: torch.Tensor,
    q_t: torch.Tensor,
    visual_threshold: float,
) -> torch.Tensor:
    visual_probability = torch.sigmoid(visual_logits)
    terrain_probability = torch.sigmoid(terrain_logits)
    visual_positive = visual_probability >= visual_threshold
    near_boundary = (visual_probability - visual_threshold).abs() <= ROUTING_MARGIN
    veto = (
        ((ROUTING_LOW - terrain_probability) / ROUTING_LOW).clamp(0.0, 1.0)
        * visual_positive
        * near_boundary
        * q_t
    )
    rescue = (
        ((terrain_probability - ROUTING_HIGH) / (1.0 - ROUTING_HIGH)).clamp(0.0, 1.0)
        * (~visual_positive)
        * near_boundary
        * q_t
    )
    return rescue - veto


def precompute_split(
    split: str,
    sample_ids: Sequence[str],
    all_ids: Sequence[str],
    event_ids: Sequence[str],
    rows: dict[str, dict[str, str]],
    train_ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    context: ContextTable,
    terrain: nn.Module,
    visual: nn.Module,
    visual_threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cache_root = args.cache_dir or args.outdir
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"frozen_{split}_cache.pt"
    identity = {
        "cache_schema_version": 2,
        "split": split,
        "sample_sha256": sample_hash(list(sample_ids)),
        "outer_train_sample_sha256": sample_hash(list(train_ids)),
        "visual_checkpoint": file_signature(args.visual_checkpoint),
        "terrain_checkpoint": file_signature(args.terrain_checkpoint),
        "material_registry": file_signature(args.material_registry),
        "trigger_registry": file_signature(args.trigger_registry),
        "terrain_mean_sha256": hashlib.sha256(
            np.asarray(mean, dtype=np.float32).tobytes()
        ).hexdigest(),
        "terrain_std_sha256": hashlib.sha256(
            np.asarray(std, dtype=np.float32).tobytes()
        ).hexdigest(),
        "context_dimensions": [context.material_dim, context.trigger_dim],
        "routing": [ROUTING_LOW, ROUTING_HIGH, ROUTING_ALPHA, ROUTING_MARGIN],
        "spatial_supergroup_sha256": sample_hash(
            [str(rows[sample_id]["spatial_supergroup"]) for sample_id in sample_ids]
        ),
    }
    if cache_path.exists() and not args.rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("identity") == identity:
            return payload
    dataset = trainer.PrithviTerrainDataset(
        trainer.BASE_H5,
        trainer.OPTICAL_H5,
        trainer.TERRAIN_H5,
        all_ids,
        event_ids,
        rows,
        sample_ids,
        mean,
        std,
        args.seed,
        train_ids,
        True,
    )
    loader = trainer.protocol.make_loader(
        dataset,
        SimpleNamespace(seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers),
        shuffle=False,
    )
    collected: dict[str, list[torch.Tensor]] = defaultdict(list)
    ordered_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            optical = batch["pre"].to(args.device, non_blocking=True)
            coordinates = batch["post"].to(args.device, non_blocking=True)
            terrain_input = batch["terrain"].to(args.device, non_blocking=True)
            q_t = batch["q_t"].to(args.device, non_blocking=True)
            with trainer.protocol.autocast_context(args.device.startswith("cuda")):
                visual_logits, _ = visual(optical, coordinates)
                terrain_logits, _ = terrain(terrain_input)
            direction = frozen_terrain_direction(
                visual_logits.float(), terrain_logits.float(), q_t, visual_threshold
            )
            collected["visual_logits"].append(visual_logits.float().cpu().to(torch.float16))
            collected["terrain_direction"].append(direction.cpu().to(torch.float16))
            collected["mask"].append((batch["mask"] >= 0.5).to(torch.uint8))
            collected["valid"].append((batch["valid"] >= 0.5).to(torch.uint8))
            ordered_ids.extend(list(batch["sample_id"]))
    if ordered_ids != list(sample_ids):
        raise RuntimeError(f"{split} cache sample order changed")
    event_by_sample = dict(zip(all_ids, event_ids))
    payload: dict[str, Any] = {
        "identity": identity,
        "sample_ids": ordered_ids,
        "spatial_supergroups": [
            str(rows[sample_id]["spatial_supergroup"]) for sample_id in ordered_ids
        ],
        "physical_event_ids": [
            str(event_by_sample[sample_id]) for sample_id in ordered_ids
        ],
        **{key: torch.cat(values, dim=0) for key, values in collected.items()},
        **context.arrays(ordered_ids),
    }
    torch.save(payload, cache_path)
    return payload


def counts(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, int]:
    return {
        "tp": int((prediction & target & valid).sum()),
        "fp": int((prediction & ~target & valid).sum()),
        "fn": int((~prediction & target & valid).sum()),
        "tn": int((~prediction & ~target & valid).sum()),
    }


def add_counts(total: dict[str, int], current: dict[str, int]) -> None:
    for key, value in current.items():
        total[key] += value


def deterministic_derangement(
    sample_ids: Sequence[str], groups: Sequence[str], seed: int, fold: int
) -> torch.Tensor:
    n = len(sample_ids)
    if n != len(groups):
        raise RuntimeError("material shuffle groups do not match sample identities")
    if n < 2:
        return torch.arange(n)
    mapping = list(range(n))
    for group in sorted(set(groups)):
        order = sorted(
            [index for index, value in enumerate(groups) if value == group],
            key=lambda index: hashlib.sha256(
                f"{seed}|fold{fold}|material-shuffle|{group}|{sample_ids[index]}".encode()
            ).hexdigest(),
        )
        if len(order) < 2:
            continue
        for position, destination in enumerate(order):
            mapping[destination] = order[(position + 1) % len(order)]
    for index, source in enumerate(mapping):
        if index != source and groups[index] != groups[source]:
            raise RuntimeError("material shuffle crossed a spatial supergroup")
    return torch.tensor(mapping, dtype=torch.long)


def variants_for_mode(mode: str) -> dict[str, dict[str, str]]:
    result = {
        "aligned": {
            "status": "applicable",
            "method": METHOD_BY_MODE[mode],
            "material": "aligned",
            "trigger": "aligned",
        },
        "zero_q": {
            "status": "applicable",
            "method": "VT",
            "material": "zero_q",
            "trigger": "zero_q",
        },
    }
    if mode in ("material", "joint"):
        result["material_shuffled"] = {
            "status": "applicable",
            "material": "shuffled",
            "trigger": "aligned",
        }
    else:
        result["material_shuffled"] = {
            "status": "not_applicable",
            "reason": "mode has no Material modulator",
        }
    if mode in ("trigger", "joint"):
        result["trigger_wrongtime"] = {
            "status": "applicable",
            "material": "aligned",
            "trigger": "wrongtime",
        }
    else:
        result["trigger_wrongtime"] = {
            "status": "not_applicable",
            "reason": "mode has no Trigger modulator",
        }
    return result


def forward_variant(
    model: TMRModulator,
    batch: dict[str, Any],
    variant: str,
    permutation: torch.Tensor | None,
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    material = batch["material"].to(device).float()
    q_material = batch["q_material"].to(device).float()
    trigger = batch["trigger"].to(device).float()
    q_trigger = batch["q_trigger"].to(device).float()
    if variant == "material_shuffled":
        if permutation is None:
            raise RuntimeError("material shuffle requested without permutation")
        indices = batch["index"].long()
        source = permutation[indices]
        payload = batch["_full_payload"]
        material = payload["material"][source].to(device).float()
        q_material = payload["q_material"][source].to(device).float()
    elif variant == "trigger_wrongtime":
        trigger = batch["trigger_wrongtime"].to(device).float()
    elif variant == "zero_q":
        q_material = torch.zeros_like(q_material)
        q_trigger = torch.zeros_like(q_trigger)
    return model(
        batch["visual_logits"].to(device).float(),
        batch["terrain_direction"].to(device).float(),
        material,
        q_material,
        trigger,
        q_trigger,
    )


def evaluate(
    model: TMRModulator,
    payload: dict[str, Any],
    args: argparse.Namespace,
    visual_threshold: float,
    controls: bool,
    collect_rows: bool = False,
) -> dict[str, Any]:
    dataset = FrozenCacheDataset(payload)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    threshold_logit = torch.logit(torch.tensor(visual_threshold, device=args.device))
    specifications = variants_for_mode(args.mode) if controls else {
        "aligned": {"status": "applicable", "material": "aligned", "trigger": "aligned"}
    }
    applicable = [name for name, value in specifications.items() if value["status"] == "applicable"]
    permutation = (
        deterministic_derangement(
            payload["sample_ids"], payload["spatial_supergroups"], args.seed, args.fold
        )
        if controls
        else None
    )
    histograms = {name: trainer.protocol.ProbabilityHistogram() for name in applicable}
    vt_histogram = trainer.protocol.ProbabilityHistogram()
    visual_histogram = trainer.protocol.ProbabilityHistogram()
    totals = {name: defaultdict(int) for name in applicable}
    transitions = {name: defaultdict(int) for name in applicable}
    vt_total: dict[str, int] = defaultdict(int)
    visual_total: dict[str, int] = defaultdict(int)
    multiplier_sums = {name: defaultdict(float) for name in applicable}
    multiplier_count = {name: 0 for name in applicable}
    sample_rows: list[dict[str, Any]] = []

    def append_sample_rows(
        batch: dict[str, Any], method: str, prediction: torch.Tensor,
        target: torch.Tensor, valid: torch.Tensor, baseline: torch.Tensor | None = None,
    ) -> None:
        if not collect_rows:
            return
        for index, sample_id in enumerate(batch["sample_id"]):
            current = counts(
                prediction[index:index + 1], target[index:index + 1], valid[index:index + 1]
            )
            metrics = trainer.protocol.metrics_from_counts(current)
            row = {
                "sample_id": sample_id,
                "physical_event_id": batch["physical_event_id"][index],
                "spatial_supergroup": batch["spatial_supergroup"][index],
                "method": method,
                **current,
                **metrics,
                "errors": current["fp"] + current["fn"],
            }
            if baseline is not None:
                base_correct = baseline[index:index + 1] == target[index:index + 1]
                adapted_correct = prediction[index:index + 1] == target[index:index + 1]
                row["corrected"] = int(((~base_correct) & adapted_correct & valid[index:index + 1]).sum())
                row["harmed"] = int((base_correct & (~adapted_correct) & valid[index:index + 1]).sum())
            else:
                row["corrected"] = 0
                row["harmed"] = 0
            sample_rows.append(row)
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = dict(raw_batch)
            batch["_full_payload"] = payload
            visual_logits = batch["visual_logits"].to(args.device).float()
            direction = batch["terrain_direction"].to(args.device).float()
            vt_logits = visual_logits + ROUTING_ALPHA * direction
            target = batch["mask"].to(args.device).bool()
            valid = batch["valid"].to(args.device).bool()
            visual_prediction = visual_logits >= threshold_logit
            vt_prediction = vt_logits >= threshold_logit
            add_counts(visual_total, counts(visual_prediction, target, valid))
            add_counts(vt_total, counts(vt_prediction, target, valid))
            append_sample_rows(batch, "visual", visual_prediction, target, valid)
            append_sample_rows(batch, "vt", vt_prediction, target, valid)
            visual_histogram.update(
                torch.sigmoid(visual_logits).cpu().numpy()[valid.cpu().numpy()],
                target.cpu().numpy()[valid.cpu().numpy()],
            )
            vt_histogram.update(
                torch.sigmoid(vt_logits).cpu().numpy()[valid.cpu().numpy()],
                target.cpu().numpy()[valid.cpu().numpy()],
            )
            for name in applicable:
                logits, diagnostics = forward_variant(
                    model, batch, name, permutation, args.device
                )
                prediction = logits >= threshold_logit
                add_counts(totals[name], counts(prediction, target, valid))
                baseline_correct = vt_prediction == target
                adapted_correct = prediction == target
                transitions[name]["corrected"] += int(
                    ((~baseline_correct) & adapted_correct & valid).sum()
                )
                transitions[name]["harmed"] += int(
                    (baseline_correct & (~adapted_correct) & valid).sum()
                )
                append_sample_rows(
                    batch, name, prediction, target, valid, baseline=vt_prediction
                )
                valid_numpy = valid.cpu().numpy()
                histograms[name].update(
                    torch.sigmoid(logits).cpu().numpy()[valid_numpy],
                    target.cpu().numpy()[valid_numpy],
                )
                for key in ("material_multiplier", "trigger_multiplier", "dosage"):
                    multiplier_sums[name][key] += float(diagnostics[key].sum())
                multiplier_count[name] += int(logits.shape[0])
    visual_metrics = trainer.protocol.metrics_from_counts(visual_total)
    vt_metrics = trainer.protocol.metrics_from_counts(vt_total)
    vt_errors = vt_total["fp"] + vt_total["fn"]
    result: dict[str, Any] = {
        "visual": {**visual_total, **visual_metrics, "ap": visual_histogram.average_precision},
        "vt_baseline": {**vt_total, **vt_metrics, "ap": vt_histogram.average_precision, "errors": vt_errors},
        "variants": {},
    }
    for name, specification in specifications.items():
        if specification["status"] != "applicable":
            result["variants"][name] = specification
            continue
        current = totals[name]
        metrics = trainer.protocol.metrics_from_counts(current)
        errors = current["fp"] + current["fn"]
        corrected = transitions[name]["corrected"]
        harmed = transitions[name]["harmed"]
        result["variants"][name] = {
            **specification,
            **current,
            **metrics,
            "ap": histograms[name].average_precision,
            "errors": errors,
            "delta_iou_vs_vt": metrics["iou"] - vt_metrics["iou"],
            "delta_ap_vs_vt": histograms[name].average_precision - vt_histogram.average_precision,
            "rer_vs_vt": (vt_errors - errors) / max(vt_errors, 1),
            "corrected": corrected,
            "harmed": harmed,
            "net_corrected": corrected - harmed,
            "corrected_to_harmed": corrected / max(harmed, 1),
            "mean_material_multiplier": multiplier_sums[name]["material_multiplier"] / max(multiplier_count[name], 1),
            "mean_trigger_multiplier": multiplier_sums[name]["trigger_multiplier"] / max(multiplier_count[name], 1),
            "mean_joint_dosage": multiplier_sums[name]["dosage"] / max(multiplier_count[name], 1),
        }
    if controls:
        zero = result["variants"]["zero_q"]
        for key in ("tp", "fp", "fn", "tn"):
            if zero[key] != result["vt_baseline"][key]:
                raise RuntimeError("zero-q control is not exactly equal to frozen VT")
    if collect_rows:
        result["_sample_rows"] = sample_rows
    return result


def train_modulator(
    model: TMRModulator,
    train_payload: dict[str, Any],
    val_payload: dict[str, Any],
    args: argparse.Namespace,
    visual_threshold: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    train_loader = DataLoader(
        FrozenCacheDataset(train_payload),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_ap = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                batch["visual_logits"].to(args.device).float(),
                batch["terrain_direction"].to(args.device).float(),
                batch["material"].to(args.device).float(),
                batch["q_material"].to(args.device).float(),
                batch["trigger"].to(args.device).float(),
                batch["q_trigger"].to(args.device).float(),
            )
            target = batch["mask"].to(args.device).float()
            valid = batch["valid"].to(args.device).float()
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = trainer.masked_mean(bce, valid) + 0.5 * trainer.protocol.dice_loss_per_sample(
                logits, target, valid
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * logits.shape[0]
            seen += logits.shape[0]
        val = evaluate(model, val_payload, args, visual_threshold, controls=False)
        aligned = val["variants"]["aligned"]
        validation_ap = float(aligned["ap"])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_ap": validation_ap,
            "val_iou": float(aligned["iou"]),
            "val_delta_iou_vs_vt": aligned["delta_iou_vs_vt"],
            "val_rer_vs_vt": aligned["rer_vs_vt"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_ap > best_ap:
            best_ap = validation_ap
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("modulator training produced no checkpoint")
    return best_state, history, best_epoch


def self_test() -> int:
    torch.manual_seed(7)
    for mode in ("material", "trigger", "joint"):
        model = TMRModulator(mode, material_dim=5, trigger_dim=3, hidden_dim=8)
        visual = torch.randn(4, 1, 8, 8)
        direction = torch.randn_like(visual).clamp(-1.0, 1.0)
        material = torch.randn(4, 5)
        trigger = torch.randn(4, 3)
        q_material = torch.tensor([[0.0], [1.0], [0.5], [1.0]])
        q_trigger = torch.tensor([[0.0], [1.0], [1.0], [0.5]])
        logits, diagnostics = model(
            visual, direction, material, q_material, trigger, q_trigger
        )
        assert logits.shape == visual.shape
        assert torch.all((diagnostics["material_multiplier"] >= 0.75) & (diagnostics["material_multiplier"] <= 1.25))
        assert torch.all((diagnostics["trigger_multiplier"] >= 0.75) & (diagnostics["trigger_multiplier"] <= 1.25))
        assert torch.all((diagnostics["dosage"] >= 0.75) & (diagnostics["dosage"] <= 1.25))
        if mode in ("material", "joint"):
            assert diagnostics["material_multiplier"][0].item() == 1.0
        if mode in ("trigger", "joint"):
            assert diagnostics["trigger_multiplier"][0].item() == 1.0
        zero_logits, _ = model(
            visual,
            direction,
            material,
            torch.zeros_like(q_material),
            trigger,
            torch.zeros_like(q_trigger),
        )
        expected = visual + ROUTING_ALPHA * direction
        assert torch.equal(zero_logits, expected)
    assert torch.equal(
        deterministic_derangement(["a", "b", "c"], ["g", "g", "g"], 1, 0),
        deterministic_derangement(["a", "b", "c"], ["g", "g", "g"], 1, 0),
    )
    print(json.dumps({"status": "ok", "device": "cpu", "modes": ["material", "trigger", "joint"]}))
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    required = {
        "fold": args.fold,
        "visual_checkpoint": args.visual_checkpoint,
        "terrain_checkpoint": args.terrain_checkpoint,
        "outdir": args.outdir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.self_test:
        return self_test()
    args.outdir.mkdir(parents=True, exist_ok=True)
    trainer.set_seed(args.seed)
    all_ids, event_ids = trainer.validate_sidecars(
        trainer.BASE_H5, trainer.OPTICAL_H5, trainer.TERRAIN_H5
    )
    rows, roles, split_regions = trainer.protocol.load_logo_rows(trainer.SPLIT_CSV, args.fold)
    allowed = set(all_ids)
    roles = {
        role: [sample_id for sample_id in values if sample_id in allowed]
        for role, values in roles.items()
    }
    outer_train_ids = list(roles["train"])
    role_limits = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }
    for role, limit in role_limits.items():
        roles[role] = deterministic_prefix(
            roles[role], limit, f"{args.seed}|fold{args.fold}|{role}|tmr-smoke"
        )
    context = ContextTable(
        args.material_registry, args.trigger_registry, all_ids, outer_train_ids
    )
    terrain, terrain_result, visual, visual_payload, provenance = load_frozen_models(args)
    if "terrain_mean" in terrain_result and "terrain_std" in terrain_result:
        mean = np.asarray(terrain_result["terrain_mean"], dtype=np.float32)
        std = np.asarray(terrain_result["terrain_std"], dtype=np.float32)
    else:
        mean, std = trainer.estimate_terrain_stats(
            trainer.TERRAIN_H5, all_ids, outer_train_ids
        )
    visual_threshold = float(visual_payload["threshold"])

    train_payload = precompute_split(
        "train", roles["train"], all_ids, event_ids, rows, outer_train_ids,
        mean, std, context, terrain, visual, visual_threshold, args,
    )
    val_payload = precompute_split(
        "val", roles["val"], all_ids, event_ids, rows, outer_train_ids,
        mean, std, context, terrain, visual, visual_threshold, args,
    )
    model = TMRModulator(
        args.mode, context.material_dim, context.trigger_dim, args.hidden_dim
    ).to(args.device)
    best_state, history, best_epoch = train_modulator(
        model, train_payload, val_payload, args, visual_threshold
    )
    model.load_state_dict(best_state)
    validation = evaluate(model, val_payload, args, visual_threshold, controls=False)

    # Test labels are first read here, after validation AP has frozen the model.
    test_payload = precompute_split(
        "test", roles["test"], all_ids, event_ids, rows, outer_train_ids,
        mean, std, context, terrain, visual, visual_threshold, args,
    )
    test = evaluate(
        model, test_payload, args, visual_threshold, controls=True, collect_rows=True
    )
    sample_rows = test.pop("_sample_rows")
    sample_frame = pd.DataFrame(sample_rows)
    vt_sample = sample_frame.loc[sample_frame["method"] == "vt"].set_index("sample_id")
    sample_frame["delta_iou_vs_vt"] = sample_frame.apply(
        lambda row: row["iou"] - vt_sample.loc[row["sample_id"], "iou"], axis=1
    )
    sample_frame["rer_vs_vt"] = sample_frame.apply(
        lambda row: (
            vt_sample.loc[row["sample_id"], "errors"] - row["errors"]
        ) / max(vt_sample.loc[row["sample_id"], "errors"], 1),
        axis=1,
    )
    event_rows = []
    for (method, event_id, group), frame in sample_frame.groupby(
        ["method", "physical_event_id", "spatial_supergroup"], sort=True
    ):
        current = {key: int(frame[key].sum()) for key in ("tp", "fp", "fn", "tn")}
        metrics = trainer.protocol.metrics_from_counts(current)
        event_rows.append({
            "method": method,
            "physical_event_id": event_id,
            "spatial_supergroup": group,
            "n_samples": len(frame),
            **current,
            **metrics,
            "errors": current["fp"] + current["fn"],
            "corrected": int(frame["corrected"].sum()),
            "harmed": int(frame["harmed"].sum()),
        })
    event_frame = pd.DataFrame(event_rows)
    vt_event = event_frame.loc[event_frame["method"] == "vt"].set_index("physical_event_id")
    event_frame["delta_iou_vs_vt"] = event_frame.apply(
        lambda row: row["iou"] - vt_event.loc[row["physical_event_id"], "iou"], axis=1
    )
    event_frame["rer_vs_vt"] = event_frame.apply(
        lambda row: (
            vt_event.loc[row["physical_event_id"], "errors"] - row["errors"]
        ) / max(vt_event.loc[row["physical_event_id"], "errors"], 1),
        axis=1,
    )
    sample_path = args.outdir / "per_sample.csv"
    event_path = args.outdir / "per_event.csv"
    sample_frame.to_csv(sample_path, index=False)
    event_frame.to_csv(event_path, index=False)
    result = {
        "status": "test_evaluated_after_validation_checkpoint_selection",
        "mode": args.mode,
        "method": METHOD_BY_MODE[args.mode],
        "fold": args.fold,
        "seed": args.seed,
        "role_contract": {
            "terrain": "only dense correction direction; frozen",
            "material": "bounded sample-level dosage multiplier",
            "trigger": "bounded event-level dosage multiplier",
            "formula": "L_out=L_vis+4*d_T*m_M*tau_R",
            "m_M": "1+q_M*0.25*tanh(MLP(z_M))",
            "tau_R": "1+q_R*0.25*tanh(MLP(z_R))",
        },
        "routing_config": {
            "low": ROUTING_LOW,
            "high": ROUTING_HIGH,
            "alpha": ROUTING_ALPHA,
            "margin": ROUTING_MARGIN,
        },
        "n_samples": {role: len(values) for role, values in roles.items()},
        "regions": {role: sorted(set(split_regions[role])) for role in ("train", "val", "test")},
        "visual_threshold": visual_threshold,
        "selection_metric": "validation_average_precision; earliest epoch retained on exact tie",
        "best_epoch": best_epoch,
        "history": history,
        "validation": validation,
        "test": test,
        "context_provenance": context.provenance(
            args.material_registry, args.trigger_registry, outer_train_ids
        ),
        "frozen_provenance": {
            "visual_checkpoint": file_signature(args.visual_checkpoint),
            "terrain_checkpoint": file_signature(args.terrain_checkpoint),
            "prithvi": provenance,
        },
        "paired_artifacts": {
            "per_sample": str(sample_path.resolve()),
            "per_event": str(event_path.resolve()),
        },
    }
    torch.save(
        {"state_dict": best_state, "result": result}, args.outdir / "modulator.pt"
    )
    result_path = args.outdir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    done = {
        "status": "complete",
        "mode": args.mode,
        "fold": args.fold,
        "seed": args.seed,
        "result_sha256": trainer.protocol.sha256_file(result_path),
        "checkpoint_sha256": trainer.protocol.sha256_file(args.outdir / "modulator.pt"),
        "per_sample_sha256": trainer.protocol.sha256_file(sample_path),
        "per_event_sha256": trainer.protocol.sha256_file(event_path),
    }
    (args.outdir / "DONE.json").write_text(
        json.dumps(done, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"validation": validation, "test": test}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
