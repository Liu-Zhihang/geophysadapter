#!/usr/bin/env python3
"""Train role-correct Material/Trigger heads over a frozen Sen12 Prithvi VT anchor.

Only aligned context is used for optimization.  Every negative control is
materialized at inference from the same selected checkpoint.  Terrain remains
the sole dense physical direction; Material can only modulate that frozen
direction and Trigger can only calibrate pixels selected by visual uncertainty.
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
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

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

import train_sen12_prithvi_terrain_v2 as terrain_trainer  # noqa: E402
import train_sen12_prithvi_tmr_modulator as frozen_protocol  # noqa: E402
from pild_roleaware_material import (  # noqa: E402
    CONTEXT_ALIGNED as M_ALIGNED,
    CONTEXT_SHUFFLED as M_SHUFFLED,
    CONTEXT_ZERO_Q as M_ZERO_Q,
    MATERIAL_FEATURE_COUNT,
    MATERIAL_FEATURE_NAMES,
    OuterTrainMaterialNormalizer,
    RoleAwareMaterialInteraction,
)
from pild_roleaware_trigger import (  # noqa: E402
    CONTEXT_ALIGNED as R_ALIGNED,
    CONTEXT_EVENT_SHUFFLE as R_EVENT_SHUFFLE,
    CONTEXT_WRONG_TIME as R_WRONG_TIME,
    CONTEXT_ZERO_Q as R_ZERO_Q,
    PILDRoleAwareTrigger,
    TriggerGateConfig,
    assert_event_level_broadcast,
    build_wrong_time_features,
)


CONTEXT_ROOT = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v1"
DEFAULT_MATERIAL = CONTEXT_ROOT / "material_sample_registry.csv"
DEFAULT_TRIGGER = CONTEXT_ROOT / "trigger_sample_registry_v1.csv"
DEFAULT_OUTROOT = PROJECT_ROOT / "experiments/revision2026/sen12_prithvi_roleaware_mr_v1"
MODES = ("material", "trigger", "joint")
ROUTING_ALPHA = frozen_protocol.ROUTING_ALPHA
RESPONSE_GROUPS = {
    "slope": (1,),
    "curvature": (4,),
    "relief": (5, 6, 7, 8),
}
NATIVE_TERRAIN17_NAMES = (
    "elevation", "slope_deg", "aspect_sin", "aspect_cos",
    "profile_curvature", "plan_curvature", "laplacian_curvature",
    "tpi_90m", "tpi_300m", "tpi_900m", "local_std_90m",
    "local_std_300m", "local_relief_300m", "local_relief_900m",
    "valley_depth_900m", "ridge_height_900m", "ruggedness_90m",
)
COMMON_TERRAIN9_NAMES = (
    "elevation", "slope_deg", "aspect_sin", "aspect_cos",
    "laplacian_curvature", "tpi_90m", "tpi_300m",
    "ruggedness_90m", "local_relief_300m",
)
COMMON_TERRAIN9_INDICES = (0, 1, 2, 3, 6, 7, 8, 16, 12)
AWC_SOURCE_COLUMNS = {
    f"awc_{depth}_footprint_mean_mm": f"awc_{depth}_aligned_mm"
    for depth in ("0_10", "10_30", "30_60", "60_100", "100_200")
}
TRIGGER_FEATURE_NAMES = (
    "rain_d7_antecedent_case_mm",
    "rain_d7_wrongtime_median_mm",
    "rain_d7_case_minus_wrongtime_mm",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def sample_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] | None = None,
) -> None:
    if not rows and not empty_fields:
        raise RuntimeError(f"refusing to publish empty evidence file: {path.name}")
    fields = (
        sorted({str(key) for row in rows for key in row})
        if rows
        else sorted(map(str, empty_fields or ()))
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_safe(row.get(field)) for field in fields})
    os.replace(temporary, path)


@dataclass(frozen=True)
class TriggerNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    event_ids: tuple[str, ...]

    @classmethod
    def fit(
        cls, values: np.ndarray, quality: np.ndarray, event_ids: Sequence[str], train_ids: Sequence[int]
    ) -> "TriggerNormalizer":
        rows_by_event: dict[str, list[int]] = defaultdict(list)
        for index in train_ids:
            if quality[index] > 0 and np.isfinite(values[index]).all():
                rows_by_event[str(event_ids[index])].append(index)
        if not rows_by_event:
            return cls(np.zeros(3, np.float32), np.ones(3, np.float32), tuple())
        events = tuple(sorted(rows_by_event))
        # One explicit row per event.  This remains correct even if a caller
        # provides unbroadcast registry rows; first-row selection is forbidden.
        event_values = np.stack([
            np.median(values[rows_by_event[event]], axis=0) for event in events
        ]).astype(np.float64)
        mean = event_values.mean(axis=0)
        scale = event_values.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32), events)

    def transform(self, values: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(values), values, self.mean)
        return np.clip((clean - self.mean) / self.scale, -5.0, 5.0).astype(np.float32)

    def audit(self) -> dict[str, Any]:
        return {
            "fit_scope": "outer-train-supported-events-only",
            "event_ids": list(self.event_ids),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


class RoleContext:
    """Outer-train-normalized 21-D Material and event-level Trigger contexts."""

    def __init__(
        self,
        material_csv: Path,
        trigger_csv: Path,
        all_ids: Sequence[str],
        event_ids: Sequence[str],
        source_by_id: Mapping[str, str],
        material_train_ids: Sequence[str],
        trigger_train_ids: Sequence[str],
        seed: int,
    ) -> None:
        material = pd.read_csv(material_csv)
        trigger = pd.read_csv(trigger_csv)
        self._validate_identity(material, all_ids, "Material")
        self._validate_identity(trigger, all_ids, "Trigger")
        material = material.assign(sample_id=material["sample_id"].astype(str)).set_index("sample_id").loc[list(all_ids)]
        trigger = trigger.assign(sample_id=trigger["sample_id"].astype(str)).set_index("sample_id").loc[list(all_ids)]
        self.sample_ids = tuple(map(str, all_ids))
        self.event_ids = tuple(map(str, event_ids))
        self.source_ids = tuple(str(source_by_id[sample_id]) for sample_id in all_ids)
        self.index = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        material_train_positions = [self.index[sample_id] for sample_id in material_train_ids]
        trigger_train_positions = [self.index[sample_id] for sample_id in trigger_train_ids]

        columns = [AWC_SOURCE_COLUMNS.get(name, name) for name in MATERIAL_FEATURE_NAMES]
        missing = [column for column in columns if column not in material]
        if missing:
            raise RuntimeError(f"Material registry lacks role columns: {missing}")
        material_raw = material.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        q_material = pd.to_numeric(material.get("q_M_full", material["q_M"]), errors="coerce").fillna(0.0).to_numpy(np.float32)
        normalizer = OuterTrainMaterialNormalizer.fit(
            material_raw,
            self.sample_ids,
            self.source_ids,
            self.event_ids,
            material_train_ids,
            feature_names=MATERIAL_FEATURE_NAMES,
        )
        self.material_normalizer = normalizer
        self.material = normalizer.transform(material_raw)
        finite_material = np.isfinite(material_raw).all(axis=1)
        self.q_material = np.where(finite_material, np.clip(q_material, 0.0, 1.0), 0.0).astype(np.float32)

        missing = [column for column in TRIGGER_FEATURE_NAMES if column not in trigger]
        if missing:
            raise RuntimeError(f"Trigger registry lacks role columns: {missing}")
        trigger_raw = trigger.loc[:, TRIGGER_FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        q_trigger = pd.to_numeric(trigger["q_R"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        if not np.isin(q_trigger, (0.0, 1.0)).all():
            raise RuntimeError("Sen12 q_R must be binary")
        # Registry rows retain patch-coordinate rainfall sampling.  Trigger is
        # deliberately event-level here: aggregate first, then broadcast.  The
        # event minimum q_R is a conservative all-patches support contract.
        registry_event_ids = trigger["physical_event_id"].astype(str).to_numpy()
        if tuple(registry_event_ids) != self.event_ids:
            raise RuntimeError("Trigger physical_event_id differs from frozen H5 canonical event identity")
        event_series = pd.Series(registry_event_ids, index=np.arange(len(registry_event_ids)))
        trigger_raw = (
            pd.DataFrame(trigger_raw)
            .groupby(event_series, sort=False)
            .transform("median")
            .to_numpy(np.float32)
        )
        q_trigger = (
            pd.Series(q_trigger)
            .groupby(event_series, sort=False)
            .transform("min")
            .to_numpy(np.float32)
        )
        self._assert_event_broadcast(trigger_raw, q_trigger)
        trigger_fit = trigger_raw.copy()
        trigger_fit[:, :2] = np.log1p(np.maximum(trigger_fit[:, :2], 0.0))
        trigger_fit[:, 2] = np.sign(trigger_fit[:, 2]) * np.log1p(np.abs(trigger_fit[:, 2]))
        self.trigger_normalizer = TriggerNormalizer.fit(
            trigger_fit, q_trigger, self.event_ids, trigger_train_positions
        )
        self.trigger = self.trigger_normalizer.transform(trigger_fit)
        wrong_raw = trigger_fit.copy()
        wrong_raw[:, 0] = wrong_raw[:, 1]
        wrong_raw[:, 2] = 0.0
        self.trigger_wrong = self.trigger_normalizer.transform(wrong_raw)
        self.q_trigger = q_trigger.astype(np.float32)
        assert_event_level_broadcast(
            self.event_ids, torch.from_numpy(self.trigger), torch.from_numpy(self.q_trigger)
        )
        self.seed = int(seed)

    @staticmethod
    def _validate_identity(frame: pd.DataFrame, all_ids: Sequence[str], name: str) -> None:
        if "sample_id" not in frame or frame["sample_id"].astype(str).duplicated().any():
            raise RuntimeError(f"{name} registry has missing/duplicate sample identity")
        expected, observed = set(map(str, all_ids)), set(frame["sample_id"].astype(str))
        if expected != observed:
            raise RuntimeError(
                f"{name} identity mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}"
            )

    def _assert_event_broadcast(self, values: np.ndarray, quality: np.ndarray) -> None:
        first: dict[str, int] = {}
        for index, event in enumerate(self.event_ids):
            if event in first:
                reference = first[event]
                if not np.array_equal(values[index], values[reference], equal_nan=True):
                    raise RuntimeError(f"Trigger varies within event: {event}")
                if quality[index] != quality[reference]:
                    raise RuntimeError(f"q_R varies within event: {event}")
            else:
                first[event] = index

    def _material_donors(self, positions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(positions, np.int64)
        donors = positions.copy()
        supported = self.q_material > 0
        abstain = np.ones(len(positions), dtype=bool)
        for output_index, target in enumerate(positions):
            candidates = [
                index for index in positions
                if self.source_ids[index] == self.source_ids[target]
                and self.event_ids[index] != self.event_ids[target]
                and supported[index]
            ]
            if not supported[target] or not candidates:
                continue
            token = hashlib.sha256(
                f"{self.seed}|M|{self.sample_ids[target]}".encode()
            ).digest()
            donors[output_index] = candidates[int.from_bytes(token[:8], "big") % len(candidates)]
            abstain[output_index] = False
        return donors, abstain

    def _trigger_donors(self, positions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(positions, np.int64)
        donors = positions.copy()
        abstain = np.ones(len(positions), dtype=bool)
        first_by_event: dict[str, int] = {}
        for index in positions:
            if self.q_trigger[index] > 0:
                first_by_event.setdefault(self.event_ids[index], index)
        events = sorted(first_by_event)
        for output_index, target in enumerate(positions):
            if self.q_trigger[target] <= 0:
                continue
            candidates = [event for event in events if event != self.event_ids[target]]
            if not candidates:
                continue
            token = hashlib.sha256(
                f"{self.seed}|R|{self.event_ids[target]}".encode()
            ).digest()
            donor_event = candidates[int.from_bytes(token[:8], "big") % len(candidates)]
            donors[output_index] = first_by_event[donor_event]
            abstain[output_index] = False
        return donors, abstain

    def arrays(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        positions = np.asarray([self.index[sample_id] for sample_id in sample_ids], np.int64)
        m_donors, m_abstain = self._material_donors(positions)
        r_donors, r_abstain = self._trigger_donors(positions)
        return {
            "material": torch.from_numpy(self.material[positions]),
            "q_material": torch.from_numpy(self.q_material[positions]),
            "material_shuffle": torch.from_numpy(self.material[m_donors]),
            "q_material_shuffle": torch.from_numpy(
                np.where(m_abstain, 0.0, self.q_material[m_donors]).astype(np.float32)
            ),
            "material_shuffle_abstain": torch.from_numpy(m_abstain),
            "material_donor_sample_ids": [self.sample_ids[index] for index in m_donors],
            "material_donor_event_ids": [self.event_ids[index] for index in m_donors],
            "material_control_applicable": torch.from_numpy(~m_abstain),
            "trigger": torch.from_numpy(self.trigger[positions]),
            "trigger_wrong": torch.from_numpy(self.trigger_wrong[positions]),
            "q_trigger": torch.from_numpy(self.q_trigger[positions]),
            "trigger_shuffle": torch.from_numpy(self.trigger[r_donors]),
            "q_trigger_shuffle": torch.from_numpy(
                np.where(r_abstain, 0.0, self.q_trigger[r_donors]).astype(np.float32)
            ),
            "trigger_shuffle_abstain": torch.from_numpy(r_abstain),
            "trigger_donor_sample_ids": [self.sample_ids[index] for index in r_donors],
            "trigger_donor_event_ids": [self.event_ids[index] for index in r_donors],
            "trigger_control_applicable": torch.from_numpy(~r_abstain),
        }

    def audit(
        self, material_train_ids: Sequence[str], trigger_train_ids: Sequence[str]
    ) -> dict[str, Any]:
        return {
            "material_features": list(MATERIAL_FEATURE_NAMES),
            "material_source_columns": [AWC_SOURCE_COLUMNS.get(name, name) for name in MATERIAL_FEATURE_NAMES],
            "material_normalizer": self.material_normalizer.audit(),
            "trigger_features": list(TRIGGER_FEATURE_NAMES),
            "trigger_normalizer": self.trigger_normalizer.audit(),
            "material_outer_train_sample_sha256": sample_hash(material_train_ids),
            "trigger_train_excluding_inner_event_sha256": sample_hash(trigger_train_ids),
        }


def trigger_support_plan(
    trigger_csv: Path,
    all_ids: Sequence[str],
    event_ids: Sequence[str],
    outer_train_ids: Sequence[str],
    outer_val_ids: Sequence[str],
    outer_test_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    """Create a label-free inner Trigger split from supported outer-train events."""

    frame = pd.read_csv(trigger_csv)
    RoleContext._validate_identity(frame, all_ids, "Trigger")
    frame = frame.assign(sample_id=frame["sample_id"].astype(str)).set_index("sample_id").loc[list(all_ids)]
    quality = pd.to_numeric(frame["q_R"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    index = {sample_id: position for position, sample_id in enumerate(all_ids)}
    event_by_id = dict(zip(all_ids, map(str, event_ids)))
    registry_event_ids = frame["physical_event_id"].astype(str).to_numpy()
    if tuple(registry_event_ids) != tuple(map(str, event_ids)):
        raise RuntimeError("Trigger registry physical_event_id differs from frozen H5 identity")
    event_series = pd.Series(registry_event_ids, index=np.arange(len(event_ids)))
    quality = (
        pd.Series(quality)
        .groupby(event_series, sort=False)
        .transform("min")
        .to_numpy(np.float32)
    )

    def supported_events(sample_ids: Sequence[str]) -> list[str]:
        return sorted({
            event_by_id[sample_id]
            for sample_id in sample_ids
            if quality[index[sample_id]] > 0
        })

    train_events = supported_events(outer_train_ids)
    if len(train_events) < 2:
        raise RuntimeError(
            "Trigger inner validation requires at least two supported outer-train events"
        )
    hashes = {
        event: hashlib.sha256(
            f"{seed}|trigger-inner-val-v1|{event}".encode()
        ).hexdigest()
        for event in train_events
    }
    inner_event = min(train_events, key=lambda event: (hashes[event], event))
    inner_ids = [sample_id for sample_id in outer_train_ids if event_by_id[sample_id] == inner_event]
    role_train_ids = [sample_id for sample_id in outer_train_ids if event_by_id[sample_id] != inner_event]
    if not inner_ids or not role_train_ids:
        raise RuntimeError("Trigger inner split produced an empty train or validation partition")
    if any(quality[index[sample_id]] <= 0 for sample_id in inner_ids):
        raise RuntimeError("selected Trigger inner event contains q_R=0 samples")
    selection_receipt = {
        "contract": "label-free sha256(seed|trigger-inner-val-v1|canonical_event_id); minimum hash",
        "candidate_events": train_events,
        "candidate_hashes": hashes,
        "selected_event": inner_event,
        "selected_hash": hashes[inner_event],
        "inner_sample_sha256": sample_hash(inner_ids),
        "role_train_sample_sha256": sample_hash(role_train_ids),
    }
    return {
        "inner_event": inner_event,
        "inner_ids": inner_ids,
        "role_train_ids": role_train_ids,
        "outer_train_supported_events": train_events,
        "outer_val_supported_events": supported_events(outer_val_ids),
        "outer_test_supported_events": supported_events(outer_test_ids),
        "outer_val_q_R_positive": int(sum(quality[index[value]] > 0 for value in outer_val_ids)),
        "outer_test_q_R_positive": int(sum(quality[index[value]] > 0 for value in outer_test_ids)),
        "selection_receipt": selection_receipt,
    }


def trigger_support_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, artifact-safe receipt without thousands of sample identities."""

    return {
        "inner_event": plan["inner_event"],
        "n_inner_samples": len(plan["inner_ids"]),
        "n_role_train_samples": len(plan["role_train_ids"]),
        "outer_train_supported_events": plan["outer_train_supported_events"],
        "outer_val_supported_events": plan["outer_val_supported_events"],
        "outer_test_supported_events": plan["outer_test_supported_events"],
        "outer_val_q_R_positive": plan["outer_val_q_R_positive"],
        "outer_test_q_R_positive": plan["outer_test_q_R_positive"],
        "selection_receipt": plan["selection_receipt"],
    }


class FrozenDataset(Dataset):
    TENSOR_KEYS = (
        "visual_logits", "frozen_vt_correction", "terrain_common9", "mask", "valid",
        "material", "q_material", "material_shuffle", "q_material_shuffle",
        "material_shuffle_abstain", "trigger", "trigger_wrong", "q_trigger",
        "trigger_shuffle", "q_trigger_shuffle", "trigger_shuffle_abstain",
        "material_control_applicable", "trigger_control_applicable",
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload

    def __len__(self) -> int:
        return len(self.payload["sample_ids"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "index": index,
            "sample_id": self.payload["sample_ids"][index],
            "event_id": self.payload["event_ids"][index],
            "source_id": self.payload["source_ids"][index],
            "material_donor_sample_id": self.payload["material_donor_sample_ids"][index],
            "material_donor_event_id": self.payload["material_donor_event_ids"][index],
            "trigger_donor_sample_id": self.payload["trigger_donor_sample_ids"][index],
            "trigger_donor_event_id": self.payload["trigger_donor_event_ids"][index],
        }
        for key in self.TENSOR_KEYS:
            item[key] = self.payload[key][index]
        return item


def precompute_split(
    split: str,
    sample_ids: Sequence[str],
    all_ids: Sequence[str],
    event_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, str]],
    train_ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    context: RoleContext,
    terrain: nn.Module,
    visual: nn.Module,
    visual_threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"frozen_{split}_cache.pt"
    identity = {
        "schema": 1,
        "split": split,
        "sample_sha256": sample_hash(sample_ids),
        "outer_train_sha256": sample_hash(train_ids),
        "visual_checkpoint": file_signature(args.visual_checkpoint),
        "terrain_checkpoint": file_signature(args.terrain_checkpoint),
        "material_registry": file_signature(args.material_registry),
        "trigger_registry": file_signature(args.trigger_registry),
        "seed": args.seed,
        "terrain_response_groups": {key: list(value) for key, value in RESPONSE_GROUPS.items()},
        "frozen_direction_schema": {
            "name": "frozen VT correction direction",
            "source": "legacy frozen 17-channel Terrain expert",
            "native17_names": list(NATIVE_TERRAIN17_NAMES),
        },
        "material_response_schema": {
            "name": "common9 Terrain content",
            "common9_names": list(COMMON_TERRAIN9_NAMES),
            "native17_indices": list(COMMON_TERRAIN9_INDICES),
            "groups": {key: list(value) for key, value in RESPONSE_GROUPS.items()},
        },
    }
    if path.exists() and not args.rebuild_cache:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("identity") == identity:
            return payload

    dataset = terrain_trainer.PrithviTerrainDataset(
        terrain_trainer.BASE_H5,
        terrain_trainer.OPTICAL_H5,
        terrain_trainer.TERRAIN_H5,
        all_ids,
        event_ids,
        dict(rows),
        sample_ids,
        mean,
        std,
        args.seed,
        train_ids,
        True,
    )
    loader = terrain_trainer.protocol.make_loader(
        dataset,
        SimpleNamespace(seed=args.seed, batch_size=args.cache_batch_size, num_workers=args.num_workers),
        shuffle=False,
    )
    collected: dict[str, list[torch.Tensor]] = defaultdict(list)
    ordered: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            optical = batch["pre"].to(args.device, non_blocking=True)
            coordinates = batch["post"].to(args.device, non_blocking=True)
            terrain_input = batch["terrain"].to(args.device, non_blocking=True)
            q_t = batch["q_t"].to(args.device, non_blocking=True)
            with terrain_trainer.protocol.autocast_context(args.device.startswith("cuda")):
                visual_logits, _ = visual(optical, coordinates)
                terrain_logits, _ = terrain(terrain_input)
            direction = frozen_protocol.frozen_terrain_direction(
                visual_logits.float(), terrain_logits.float(), q_t, visual_threshold
            )
            collected["visual_logits"].append(visual_logits.float().cpu().half())
            collected["frozen_vt_correction"].append((ROUTING_ALPHA * direction).cpu().half())
            collected["terrain_common9"].append(
                terrain_input[:, COMMON_TERRAIN9_INDICES].float().cpu().half()
            )
            collected["mask"].append((batch["mask"] >= 0.5).cpu().to(torch.uint8))
            collected["valid"].append((batch["valid"] >= 0.5).cpu().to(torch.uint8))
            ordered.extend(map(str, batch["sample_id"]))
    if ordered != list(sample_ids):
        raise RuntimeError(f"{split} cache sample order changed")
    event_by_id = dict(zip(all_ids, event_ids))
    payload: dict[str, Any] = {
        "identity": identity,
        "sample_ids": ordered,
        "event_ids": [str(event_by_id[sample_id]) for sample_id in ordered],
        "source_ids": [str(rows[sample_id]["spatial_supergroup"]) for sample_id in ordered],
        **{key: torch.cat(value, dim=0) for key, value in collected.items()},
        **context.arrays(ordered),
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload


class RoleAwareMR(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.material = (
            RoleAwareMaterialInteraction(RESPONSE_GROUPS) if mode in ("material", "joint") else None
        )
        self.trigger = (
            PILDRoleAwareTrigger(TriggerGateConfig(feature_dim=3))
            if mode in ("trigger", "joint") else None
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        material_context: str = M_ALIGNED,
        trigger_context: str = R_ALIGNED,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        visual = batch["visual_logits"].float()
        terrain_residual = batch["frozen_vt_correction"].float()
        conditioned = terrain_residual
        m_audit = None
        if self.material is not None:
            if material_context == M_ALIGNED:
                material, q_m = batch["material"], batch["q_material"]
            elif material_context == M_SHUFFLED:
                material, q_m = batch["material_shuffle"], batch["q_material_shuffle"]
            elif material_context == M_ZERO_Q:
                material, q_m = batch["material"], torch.zeros_like(batch["q_material"])
            else:
                raise ValueError(f"unknown Material context {material_context!r}")
            conditioned, m_audit = self.material(
                batch["terrain_common9"].float(),
                terrain_residual,
                material.float(),
                q_m.float(),
                context=material_context,
            )

        trigger_delta = torch.zeros_like(visual)
        r_audit = None
        if self.trigger is not None:
            q_r = batch["q_trigger"].float()
            kwargs: dict[str, Any] = {}
            if trigger_context == R_WRONG_TIME:
                kwargs["wrong_time_features"] = batch["trigger_wrong"].float()
            elif trigger_context == R_EVENT_SHUFFLE:
                kwargs["event_shuffled_features"] = batch["trigger_shuffle"].float()
                q_r = batch["q_trigger_shuffle"].float()
            elif trigger_context == R_ZERO_Q:
                q_r = torch.zeros_like(q_r)
            r_audit = self.trigger(
                visual,
                batch["trigger"].float(),
                q_r,
                batch["event_id"],
                context=trigger_context,
                **kwargs,
            )
            trigger_delta = r_audit["logit_delta"]
        logits = visual + conditioned + trigger_delta
        return logits, {
            "material": m_audit,
            "trigger": r_audit,
            "material_delta": conditioned - terrain_residual,
            "trigger_delta": trigger_delta,
        }


class ProbabilityHistogram:
    def __init__(self, bins: int = 4096) -> None:
        self.bins = bins
        self.positive = np.zeros(bins, np.int64)
        self.negative = np.zeros(bins, np.int64)

    def update(self, probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        p = probability.detach().float().cpu().numpy()[valid.cpu().numpy()]
        y = target.detach().bool().cpu().numpy()[valid.cpu().numpy()]
        index = np.clip((p * self.bins).astype(np.int64), 0, self.bins - 1)
        self.positive += np.bincount(index[y], minlength=self.bins)
        self.negative += np.bincount(index[~y], minlength=self.bins)

    def average_precision(self) -> float:
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        if tp[-1] == 0:
            return 0.0
        precision = tp / np.maximum(tp + fp, 1)
        recall_step = self.positive[::-1] / tp[-1]
        return float(np.sum(precision * recall_step))

    def fixed_fpr_threshold(self, target_fpr: float = 0.05) -> float:
        fp = np.cumsum(self.negative[::-1])
        total = max(int(self.negative.sum()), 1)
        ok = np.flatnonzero(fp / total <= target_fpr)
        reverse_index = int(ok[-1]) if len(ok) else 0
        return float((self.bins - 1 - reverse_index) / self.bins)


def counts(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, int]:
    return {
        "tp": int((prediction & target & valid).sum()),
        "fp": int((prediction & ~target & valid).sum()),
        "fn": int((~prediction & target & valid).sum()),
        "tn": int((~prediction & ~target & valid).sum()),
    }


def metrics(count: Mapping[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (int(count[key]) for key in ("tp", "fp", "fn", "tn"))
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def move_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    target = target.float()
    valid = valid.float()
    positive = (target * valid).sum()
    negative = ((1.0 - target) * valid).sum()
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits) * valid
    intersection = (probability * target).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + (target * valid).sum() + 1.0)
    return bce + dice


def controls_for_mode(mode: str) -> dict[str, tuple[str, str]]:
    controls = {"aligned": (M_ALIGNED, R_ALIGNED)}
    if mode in ("material", "joint"):
        controls.update({
            "material_shuffle": (M_SHUFFLED, R_ALIGNED),
            "material_zero_q": (M_ZERO_Q, R_ALIGNED),
        })
    if mode in ("trigger", "joint"):
        controls.update({
            "trigger_wrong_time": (M_ALIGNED, R_WRONG_TIME),
            "trigger_event_shuffle": (M_ALIGNED, R_EVENT_SHUFFLE),
            "trigger_zero_q": (M_ALIGNED, R_ZERO_Q),
        })
    if mode == "joint":
        controls["all_zero_q"] = (M_ZERO_Q, R_ZERO_Q)
    return controls


def loader(payload: Mapping[str, Any], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FrozenDataset(payload), batch_size=batch_size, shuffle=shuffle,
        num_workers=0, generator=generator, pin_memory=torch.cuda.is_available(),
    )


def validation_ap(model: RoleAwareMR, payload: Mapping[str, Any], args: argparse.Namespace) -> float:
    histogram = ProbabilityHistogram()
    model.eval()
    with torch.inference_mode():
        for batch in loader(payload, args.batch_size, False, args.seed):
            batch = move_batch(batch, args.device)
            logits, _ = model(batch)
            histogram.update(torch.sigmoid(logits), batch["mask"].bool(), batch["valid"].bool())
    return histogram.average_precision()


def train_aligned_only(
    model: RoleAwareMR,
    train_payload: Mapping[str, Any],
    outer_val_payload: Mapping[str, Any],
    trigger_inner_val_payload: Mapping[str, Any],
    args: argparse.Namespace,
    log,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("role-aware model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for batch in loader(train_payload, args.batch_size, True, args.seed + epoch):
            batch = move_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch, material_context=M_ALIGNED, trigger_context=R_ALIGNED)
            loss = segmentation_loss(logits, batch["mask"], batch["valid"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        outer_ap = validation_ap(model, outer_val_payload, args)
        inner_ap = validation_ap(model, trigger_inner_val_payload, args)
        if args.mode == "material":
            selection_score = outer_ap
            selection_contract = "Material outer-val AP"
        elif args.mode == "trigger":
            selection_score = inner_ap
            selection_contract = "Trigger held-out supported-event inner-val AP"
        else:
            selection_score = 0.5 * (outer_ap + inner_ap)
            selection_contract = "frozen equal-weight mean(Material outer-val AP, Trigger inner-val AP)"
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "material_outer_val_ap": outer_ap,
            "trigger_inner_supported_val_ap": inner_ap,
            "selection_score": selection_score,
            "selection_contract": selection_contract,
        }
        history.append(row)
        log(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"outer_ap={outer_ap:.6f} inner_R_ap={inner_ap:.6f} "
            f"selection={selection_score:.6f}"
        )
        if selection_score > best_score + 1e-12:
            best_score, best_epoch = selection_score, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    return best_state, history, best_epoch


def evaluate(
    model: RoleAwareMR,
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    threshold: float,
    fixed_fpr_threshold: float,
    with_controls: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    specifications = controls_for_mode(args.mode) if with_controls else {"aligned": (M_ALIGNED, R_ALIGNED)}
    totals = {name: defaultdict(int) for name in specifications}
    histograms = {name: ProbabilityHistogram() for name in specifications}
    sums = {name: defaultdict(float) for name in specifications}
    sample_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    threshold_logit = float(torch.logit(torch.tensor(threshold)).item())
    vt_total: dict[str, int] = defaultdict(int)
    model.eval()
    with torch.inference_mode():
        for raw in loader(payload, args.batch_size, False, args.seed):
            batch = move_batch(raw, args.device)
            target, valid = batch["mask"].bool(), batch["valid"].bool()
            vt_logits = batch["visual_logits"].float() + batch["frozen_vt_correction"].float()
            vt_prediction = vt_logits >= threshold_logit
            for key, value in counts(vt_prediction, target, valid).items():
                vt_total[key] += value
            for control, (m_context, r_context) in specifications.items():
                logits, audit = model(batch, material_context=m_context, trigger_context=r_context)
                probability = torch.sigmoid(logits)
                prediction = logits >= threshold_logit
                current = counts(prediction, target, valid)
                for key, value in current.items():
                    totals[control][key] += value
                histograms[control].update(probability, target, valid)
                brier = torch.square(probability - target.float())
                nll = F.binary_cross_entropy(probability.clamp(1e-7, 1 - 1e-7), target.float(), reduction="none")
                corrected = (~(vt_prediction == target) & (prediction == target) & valid)
                harmed = ((vt_prediction == target) & ~(prediction == target) & valid)
                baseline_correct = (vt_prediction == target) & valid
                preserved_correct = baseline_correct & (prediction == target)
                fixed_prediction = probability >= fixed_fpr_threshold
                for index, sample_id in enumerate(batch["sample_id"]):
                    item_valid = valid[index]
                    item_target = target[index]
                    item_prediction = prediction[index]
                    item_vt = vt_prediction[index]
                    item_counts = counts(item_prediction[None], item_target[None], item_valid[None])
                    vt_counts = counts(item_vt[None], item_target[None], item_valid[None])
                    fixed_counts = counts(fixed_prediction[index:index+1], item_target[None], item_valid[None])
                    if m_context == M_SHUFFLED:
                        effective_q_m = float(batch["q_material_shuffle"][index])
                        material_donor_sample = batch["material_donor_sample_id"][index]
                        material_donor_event = batch["material_donor_event_id"][index]
                    elif m_context == M_ZERO_Q:
                        effective_q_m = 0.0
                        material_donor_sample = sample_id
                        material_donor_event = batch["event_id"][index]
                    else:
                        effective_q_m = float(batch["q_material"][index])
                        material_donor_sample = sample_id
                        material_donor_event = batch["event_id"][index]
                    if r_context == R_EVENT_SHUFFLE:
                        effective_q_r = float(batch["q_trigger_shuffle"][index])
                        trigger_donor_sample = batch["trigger_donor_sample_id"][index]
                        trigger_donor_event = batch["trigger_donor_event_id"][index]
                    elif r_context == R_ZERO_Q:
                        effective_q_r = 0.0
                        trigger_donor_sample = sample_id
                        trigger_donor_event = batch["event_id"][index]
                    else:
                        effective_q_r = float(batch["q_trigger"][index])
                        trigger_donor_sample = sample_id
                        trigger_donor_event = batch["event_id"][index]
                    material_pair_applicable = bool(batch["material_control_applicable"][index])
                    trigger_shuffle_pair_applicable = bool(batch["trigger_control_applicable"][index])
                    trigger_supported = float(batch["q_trigger"][index]) > 0
                    if control == "material_shuffle":
                        control_applicable = material_pair_applicable
                    elif control in ("trigger_wrong_time", "trigger_zero_q"):
                        control_applicable = trigger_supported
                    elif control == "trigger_event_shuffle":
                        control_applicable = trigger_shuffle_pair_applicable
                    elif control == "material_zero_q":
                        control_applicable = float(batch["q_material"][index]) > 0
                    elif control == "all_zero_q":
                        control_applicable = (
                            float(batch["q_material"][index]) > 0 and trigger_supported
                        )
                    else:
                        control_applicable = True
                    baseline_correct_count = int(baseline_correct[index].sum())
                    preserved_correct_count = int(preserved_correct[index].sum())
                    row = {
                        "sample_id": sample_id,
                        "event_id": batch["event_id"][index],
                        "source_id": batch["source_id"][index],
                        "mode": args.mode,
                        "control": control,
                        **item_counts,
                        **metrics(item_counts),
                        "vt_iou": metrics(vt_counts)["iou"],
                        "errors": item_counts["fp"] + item_counts["fn"],
                        "vt_errors": vt_counts["fp"] + vt_counts["fn"],
                        "corrected": int(corrected[index].sum()),
                        "harmed": int(harmed[index].sum()),
                        "baseline_condition": "frozen_VT",
                        "baseline_correct_count": baseline_correct_count,
                        "preserved_correct_count": preserved_correct_count,
                        "preservation_rate": preserved_correct_count / max(baseline_correct_count, 1),
                        "brier": float(brier[index][item_valid].mean()) if item_valid.any() else 0.0,
                        "nll": float(nll[index][item_valid].mean()) if item_valid.any() else 0.0,
                        "predicted_area": int((item_prediction & item_valid).sum()),
                        "true_area": int((item_target & item_valid).sum()),
                        "fixed_fpr_tp": fixed_counts["tp"],
                        "fixed_fpr_fn": fixed_counts["fn"],
                        "q_M": float(batch["q_material"][index]),
                        "q_R": float(batch["q_trigger"][index]),
                        "effective_q_M": effective_q_m,
                        "effective_q_R": effective_q_r,
                        "material_donor_sample_id": material_donor_sample,
                        "material_donor_event_id": material_donor_event,
                        "trigger_donor_sample_id": trigger_donor_sample,
                        "trigger_donor_event_id": trigger_donor_event,
                        "control_applicable": control_applicable,
                        "material_shuffle_pair_applicable": material_pair_applicable,
                        "trigger_wrongtime_pair_applicable": trigger_supported,
                        "trigger_event_shuffle_pair_applicable": trigger_shuffle_pair_applicable,
                        "material_delta_abs_mean": float(audit["material_delta"][index].abs().mean()),
                        "trigger_delta_abs_mean": float(audit["trigger_delta"][index].abs().mean()),
                    }
                    sample_rows.append(row)
                    control_rows.append({
                        "sample_id": sample_id,
                        "event_id": batch["event_id"][index],
                        "mode": args.mode,
                        "control": control,
                        "checkpoint_selection": "same-aligned-validation-checkpoint",
                        "material_context": m_context,
                        "trigger_context": r_context,
                        "effective_q_M": effective_q_m,
                        "effective_q_R": effective_q_r,
                        "material_donor_sample_id": material_donor_sample,
                        "material_donor_event_id": material_donor_event,
                        "trigger_donor_sample_id": trigger_donor_sample,
                        "trigger_donor_event_id": trigger_donor_event,
                        "control_applicable": control_applicable,
                        "pairing_rule": "aligned-versus-control only where both recipient rows are applicable",
                        "material_shuffle_abstain": bool(batch["material_shuffle_abstain"][index]),
                        "trigger_shuffle_abstain": bool(batch["trigger_shuffle_abstain"][index]),
                    })
                    sums[control]["brier_sum"] += row["brier"]
                    sums[control]["nll_sum"] += row["nll"]
                    sums[control]["area_abs_error_sum"] += abs(row["predicted_area"] - row["true_area"])
                    sums[control]["fixed_tp"] += fixed_counts["tp"]
                    sums[control]["fixed_fn"] += fixed_counts["fn"]
                    sums[control]["corrected"] += row["corrected"]
                    sums[control]["harmed"] += row["harmed"]
                    sums[control]["baseline_correct"] += baseline_correct_count
                    sums[control]["preserved_correct"] += preserved_correct_count
                    sums[control]["samples"] += 1
    vt_metrics = {**dict(vt_total), **metrics(vt_total)}
    result: dict[str, Any] = {"vt": vt_metrics, "controls": {}}
    vt_errors = vt_total["fp"] + vt_total["fn"]
    for name in specifications:
        current = totals[name]
        current_metrics = metrics(current)
        errors = current["fp"] + current["fn"]
        current_metrics.update({
            **dict(current),
            "ap": histograms[name].average_precision(),
            "errors": errors,
            "rer_vs_vt": (vt_errors - errors) / max(vt_errors, 1),
            "brier_mean": sums[name]["brier_sum"] / max(sums[name]["samples"], 1),
            "nll_mean": sums[name]["nll_sum"] / max(sums[name]["samples"], 1),
            "area_abs_error_mean": sums[name]["area_abs_error_sum"] / max(sums[name]["samples"], 1),
            "fixed_fpr_recall": sums[name]["fixed_tp"] / max(sums[name]["fixed_tp"] + sums[name]["fixed_fn"], 1),
            "corrected_vs_frozen_vt": int(sums[name]["corrected"]),
            "harmed_vs_frozen_vt": int(sums[name]["harmed"]),
            "preservation_rate_vs_frozen_vt": sums[name]["preserved_correct"] / max(sums[name]["baseline_correct"], 1),
        })
        result["controls"][name] = current_metrics
    return result, sample_rows, control_rows


def per_event_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(sample_rows)
    output = []
    for (control, event), group in frame.groupby(["control", "event_id"], sort=True):
        current = {key: int(group[key].sum()) for key in ("tp", "fp", "fn", "tn")}
        output.append({
            "control": control,
            "event_id": event,
            "n_samples": len(group),
            **current,
            **metrics(current),
            "errors": current["fp"] + current["fn"],
            "corrected": int(group["corrected"].sum()),
            "harmed": int(group["harmed"].sum()),
            "baseline_condition": "frozen_VT",
            "baseline_correct_count": int(group["baseline_correct_count"].sum()),
            "preserved_correct_count": int(group["preserved_correct_count"].sum()),
            "brier": float(group["brier"].mean()),
            "nll": float(group["nll"].mean()),
        })
    return output


def paired_control_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return aligned-control pairs only for bilaterally applicable recipients."""

    frame = pd.DataFrame(sample_rows)
    aligned = frame.loc[frame["control"] == "aligned"].set_index("sample_id")
    output: list[dict[str, Any]] = []
    for row in frame.loc[frame["control"] != "aligned"].to_dict("records"):
        if not bool(row["control_applicable"]):
            continue
        sample_id = row["sample_id"]
        if sample_id not in aligned.index:
            raise RuntimeError(f"control recipient lacks aligned same-checkpoint row: {sample_id}")
        reference = aligned.loc[sample_id]
        output.append({
            "sample_id": sample_id,
            "event_id": row["event_id"],
            "source_id": row["source_id"],
            "mode": row["mode"],
            "control": row["control"],
            "pair_applicable": True,
            "checkpoint_selection": "same-aligned-validation-checkpoint",
            "aligned_iou": float(reference["iou"]),
            "control_iou": float(row["iou"]),
            "delta_iou_aligned_minus_control": float(reference["iou"] - row["iou"]),
            "aligned_errors": int(reference["errors"]),
            "control_errors": int(row["errors"]),
            "error_reduction_aligned_minus_control": int(row["errors"] - reference["errors"]),
            "effective_q_M_control": float(row["effective_q_M"]),
            "effective_q_R_control": float(row["effective_q_R"]),
            "material_donor_sample_id": row["material_donor_sample_id"],
            "material_donor_event_id": row["material_donor_event_id"],
            "trigger_donor_sample_id": row["trigger_donor_sample_id"],
            "trigger_donor_event_id": row["trigger_donor_event_id"],
        })
    return output


PAIRED_CONTROL_FIELDS = (
    "sample_id",
    "event_id",
    "source_id",
    "mode",
    "control",
    "pair_applicable",
    "checkpoint_selection",
    "aligned_iou",
    "control_iou",
    "delta_iou_aligned_minus_control",
    "aligned_errors",
    "control_errors",
    "error_reduction_aligned_minus_control",
    "effective_q_M_control",
    "effective_q_R_control",
    "material_donor_sample_id",
    "material_donor_event_id",
    "trigger_donor_sample_id",
    "trigger_donor_event_id",
)


def validate_real_inputs(args: argparse.Namespace) -> dict[str, Any]:
    required = (args.visual_checkpoint, args.terrain_checkpoint, args.material_registry, args.trigger_registry)
    missing = [str(path) for path in required if path is None or not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")
    all_ids, event_ids = terrain_trainer.validate_sidecars(
        terrain_trainer.BASE_H5, terrain_trainer.OPTICAL_H5, terrain_trainer.TERRAIN_H5
    )
    rows, roles, _ = terrain_trainer.protocol.load_logo_rows(terrain_trainer.SPLIT_CSV, args.fold)
    allowed = set(all_ids)
    roles = {
        role: [sample_id for sample_id in values if sample_id in allowed]
        for role, values in roles.items()
    }
    support = trigger_support_plan(
        args.trigger_registry, all_ids, event_ids,
        roles["train"], roles["val"], roles["test"], args.seed,
    )
    visual = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    terrain = torch.load(args.terrain_checkpoint, map_location="cpu", weights_only=False)
    visual_identity = visual.get("identity", {})
    terrain_result = terrain.get("result", {})
    if visual_identity.get("mode") != "visual" or visual_identity.get("fold") != args.fold:
        raise RuntimeError("visual checkpoint is not the requested frozen fold anchor")
    if terrain_result.get("fold") != args.fold:
        raise RuntimeError("Terrain checkpoint is not the requested frozen fold anchor")
    if terrain_result.get("seed") is not None and terrain_result.get("seed") != visual_identity.get("seed"):
        raise RuntimeError("visual/Terrain checkpoint seed identity mismatch")
    if not math.isclose(
        float(terrain_result.get("visual_threshold", visual["threshold"])),
        float(visual["threshold"]), abs_tol=1e-12, rel_tol=0.0,
    ):
        raise RuntimeError("visual/Terrain threshold identity mismatch")
    visual_seed = visual_identity.get("seed")
    terrain_seed = terrain_result.get("seed")
    legacy_resolution = None
    if terrain_seed is None:
        resolved = str(args.terrain_checkpoint.resolve())
        accepted_tokens = (
            f"fold{args.fold}_seed{visual_seed}",
            f"seed{visual_seed}/fold{args.fold}",
        )
        if visual_seed is None or not any(token in resolved for token in accepted_tokens):
            raise RuntimeError(
                "legacy Terrain checkpoint lacks embedded seed and its absolute path "
                "cannot bind it to the visual fold/seed identity"
            )
        legacy_resolution = (
            "embedded Terrain seed absent; bound by immutable checkpoint SHA, "
            "absolute seed/fold path, visual identity, and shared threshold"
        )
    context = RoleContext(
        args.material_registry, args.trigger_registry, all_ids, event_ids,
        {sample_id: rows[sample_id]["spatial_supergroup"] for sample_id in all_ids},
        roles["train"], support["role_train_ids"], args.seed,
    )
    return {
        "status": "PASS",
        "fold": args.fold,
        "mode": args.mode,
        "n_samples": len(all_ids),
        "n_events": len(set(event_ids)),
        "material_shape": [len(all_ids), MATERIAL_FEATURE_COUNT],
        "trigger_shape": [len(all_ids), 3],
        "q_M_positive": int((context.q_material > 0).sum()),
        "q_R_positive": int((context.q_trigger > 0).sum()),
        "trigger_support_plan": trigger_support_receipt(support),
        "checkpoint_seed": visual_identity.get("seed"),
        "parent_identity": {
            "schema_version": "sen12_frozen_vt_parent_manifest.v1",
            "fold": args.fold,
            "seed": visual_seed,
            "threshold": float(visual["threshold"]),
            "visual": file_signature(args.visual_checkpoint),
            "terrain": file_signature(args.terrain_checkpoint),
            "terrain_embedded_seed": terrain_seed,
            "legacy_seed_resolution": legacy_resolution,
            "direction_semantics": "frozen VT correction direction from legacy 17-channel Terrain expert",
        },
    }


def self_test() -> int:
    set_seed(1)
    batch = 4
    payload = {
        "visual_logits": torch.randn(batch, 1, 8, 8),
        "frozen_vt_correction": torch.randn(batch, 1, 8, 8) * 0.1,
        "terrain_common9": torch.randn(batch, 9, 8, 8),
        "material": torch.randn(batch, 21),
        "q_material": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "material_shuffle": torch.randn(batch, 21),
        "q_material_shuffle": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "trigger": torch.tensor([[1.0, 0.0, 1.0]] * 2 + [[0.0, 1.0, -1.0]] * 2),
        "trigger_wrong": torch.zeros(batch, 3),
        "q_trigger": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "trigger_shuffle": torch.tensor([[0.0, 1.0, -1.0]] * 2 + [[1.0, 0.0, 1.0]] * 2),
        "q_trigger_shuffle": torch.ones(batch),
        "event_id": ["a", "a", "b", "b"],
    }
    for mode in MODES:
        model = RoleAwareMR(mode)
        aligned, audit = model(payload)
        assert aligned.shape == payload["visual_logits"].shape
        if mode in ("material", "joint"):
            zero, _ = model(payload, material_context=M_ZERO_Q)
            if mode == "material":
                assert torch.equal(zero, payload["visual_logits"] + payload["frozen_vt_correction"])
            assert audit["material"]["material_dense_direction"] is False
        if mode in ("trigger", "joint"):
            zero, _ = model(payload, trigger_context=R_ZERO_Q, material_context=M_ZERO_Q)
            assert torch.equal(zero, payload["visual_logits"] + payload["frozen_vt_correction"])
            assert audit["trigger"]["audit"]["trigger_dense_direction"] is False
    print(json.dumps({"status": "PASS", "modes": list(MODES)}))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--mode", choices=MODES, default="joint")
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--visual-checkpoint", type=Path)
    parser.add_argument("--terrain-checkpoint", type=Path)
    parser.add_argument("--material-registry", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--trigger-registry", type=Path, default=DEFAULT_TRIGGER)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [name for name in ("fold", "visual_checkpoint", "terrain_checkpoint") if getattr(args, name) is None]
        if missing:
            parser.error(f"missing required arguments: {missing}")
        if not (args.validate_only or args.dry_run) and (args.cache_dir is None or args.outdir is None):
            parser.error("training requires --cache-dir and --outdir")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    validation = validate_real_inputs(args)
    if args.validate_only or args.dry_run:
        validation["dry_run"] = bool(args.dry_run)
        print(json.dumps(validation, indent=2, allow_nan=False))
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.outdir.exists():
        if (args.outdir / "DONE.json").is_file():
            print(json.dumps({"status": "already_complete", "outdir": str(args.outdir)}))
            return 0
        raise FileExistsError(f"refusing to overwrite incomplete output: {args.outdir}")
    stage = args.outdir.with_name(f".{args.outdir.name}.staging-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    run_log = stage / "run.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
        print(line, flush=True)
        with run_log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    try:
        set_seed(args.seed)
        command = " ".join(shlex.quote(value) for value in sys.argv)
        atomic_text(stage / "command.txt", command + "\n")
        atomic_json(stage / "config.json", {**vars(args), "contract": {
            "training_context": "aligned-only",
            "controls": "same-selected-checkpoint-inference-only",
            "terrain": "frozen sole dense direction",
            "material": "21-D outer-train normalized bounded Terrain interaction",
            "trigger": "event scalar over visual uncertainty only",
            "trigger_broadcast": "canonical-event median forcing; event-min q_R",
        }})
        all_ids, event_ids = terrain_trainer.validate_sidecars(
            terrain_trainer.BASE_H5, terrain_trainer.OPTICAL_H5, terrain_trainer.TERRAIN_H5
        )
        rows, roles, split_regions = terrain_trainer.protocol.load_logo_rows(terrain_trainer.SPLIT_CSV, args.fold)
        allowed = set(all_ids)
        roles = {
            role: [sample_id for sample_id in values if sample_id in allowed]
            for role, values in roles.items()
        }
        support = trigger_support_plan(
            args.trigger_registry, all_ids, event_ids,
            roles["train"], roles["val"], roles["test"], args.seed,
        )
        role_train_ids = roles["train"] if args.mode == "material" else support["role_train_ids"]
        atomic_json(stage / "config.json", {
            "schema_version": "sen12_prithvi_roleaware_mr_config.v1",
            **vars(args),
            "contract": {
                "training_context": "aligned-only",
                "controls": "same-selected-checkpoint-inference-only",
                "terrain": "frozen sole dense direction",
                "material": "21-D outer-train normalized bounded Terrain interaction",
                "trigger": "event scalar over visual uncertainty only",
                "trigger_broadcast": "canonical-event median forcing; event-min q_R",
            },
            "trigger_support_plan": trigger_support_receipt(support),
            "selection_contract": {
                "material": "outer-val AP",
                "trigger": "held-out supported outer-train event AP",
                "joint": "frozen equal-weight mean of Material outer-val AP and Trigger inner-val AP",
            }[args.mode],
            "terrain_schema": {
                "frozen_vt_direction_native17": list(NATIVE_TERRAIN17_NAMES),
                "material_response_common9": list(COMMON_TERRAIN9_NAMES),
                "common9_from_native17_indices": list(COMMON_TERRAIN9_INDICES),
                "material_response_groups": {key: list(value) for key, value in RESPONSE_GROUPS.items()},
            },
            "output_schema": "sen12_prithvi_roleaware_mr_same_checkpoint_controls.v1",
        })
        context = RoleContext(
            args.material_registry, args.trigger_registry, all_ids, event_ids,
            {sample_id: rows[sample_id]["spatial_supergroup"] for sample_id in all_ids},
            roles["train"], support["role_train_ids"], args.seed,
        )
        terrain, terrain_result, visual, visual_payload, prithvi = frozen_protocol.load_frozen_models(args)
        if "terrain_mean" in terrain_result and "terrain_std" in terrain_result:
            mean = np.asarray(terrain_result["terrain_mean"], np.float32)
            std = np.asarray(terrain_result["terrain_std"], np.float32)
        else:
            mean, std = terrain_trainer.estimate_terrain_stats(
                terrain_trainer.TERRAIN_H5, all_ids, roles["train"]
            )
        threshold = float(visual_payload["threshold"])
        log("building/reusing frozen train and validation caches")
        train_payload = precompute_split(
            f"train_{args.mode}", role_train_ids, all_ids, event_ids, rows, roles["train"], mean, std,
            context, terrain, visual, threshold, args,
        )
        outer_val_payload = precompute_split(
            "outer_val", roles["val"], all_ids, event_ids, rows, roles["train"], mean, std,
            context, terrain, visual, threshold, args,
        )
        trigger_inner_val_payload = precompute_split(
            "trigger_inner_supported_val", support["inner_ids"], all_ids, event_ids,
            rows, roles["train"], mean, std,
            context, terrain, visual, threshold, args,
        )
        model = RoleAwareMR(args.mode).to(args.device)
        best_state, history, best_epoch = train_aligned_only(
            model, train_payload, outer_val_payload, trigger_inner_val_payload, args, log
        )
        model.load_state_dict(best_state, strict=True)
        val_hist = ProbabilityHistogram()
        with torch.inference_mode():
            for raw in loader(outer_val_payload, args.batch_size, False, args.seed):
                batch = move_batch(raw, args.device)
                logits, _ = model(batch)
                val_hist.update(torch.sigmoid(logits), batch["mask"].bool(), batch["valid"].bool())
        fixed_fpr = val_hist.fixed_fpr_threshold(0.05)
        validation_result, _, _ = evaluate(
            model, outer_val_payload, args, threshold, fixed_fpr, with_controls=False
        )

        log("validation selected checkpoint; test cache and all controls now materialized")
        test_payload = precompute_split(
            "test", roles["test"], all_ids, event_ids, rows, roles["train"], mean, std,
            context, terrain, visual, threshold, args,
        )
        test_result, sample_rows, control_rows = evaluate(
            model, test_payload, args, threshold, fixed_fpr, with_controls=True
        )
        event_rows = per_event_rows(sample_rows)
        paired_rows = paired_control_rows(sample_rows)
        atomic_csv(stage / "per_sample.csv", sample_rows)
        atomic_csv(stage / "per_event.csv", event_rows)
        atomic_csv(stage / "control_rows.csv", control_rows)
        # A Trigger-unsupported outer test has no scientifically applicable
        # pairs by construction.  Publish an explicit header-only receipt so
        # downstream analysis can distinguish abstention from missing output.
        atomic_csv(
            stage / "paired_control_receipts.csv",
            paired_rows,
            empty_fields=PAIRED_CONTROL_FIELDS,
        )
        checkpoint = {
            "schema_version": "sen12_prithvi_roleaware_mr_checkpoint.v1",
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "selection": history[best_epoch - 1]["selection_contract"],
            "parent_identity": validation["parent_identity"],
            "trigger_support_plan": trigger_support_receipt(support),
            "context_audit": context.audit(roles["train"], support["role_train_ids"]),
        }
        checkpoint_tmp = stage / f".checkpoint.pt.tmp-{os.getpid()}"
        torch.save(checkpoint, checkpoint_tmp)
        os.replace(checkpoint_tmp, stage / "checkpoint.pt")
        result = {
            "schema_version": "sen12_prithvi_roleaware_mr_run.v1",
            "status": "complete",
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "history": history,
            "visual_threshold": threshold,
            "fixed_fpr_threshold_from_validation": fixed_fpr,
            "validation": validation_result,
            "test": test_result,
            "n_samples": {key: len(value) for key, value in roles.items()},
            "split_regions": split_regions,
            "trigger_support_plan": trigger_support_receipt(support),
            "fold_interpretation": (
                "q_R=0 test: exact VT fallback audit only; excluded from Trigger-effect aggregation"
                if support["outer_test_q_R_positive"] == 0 else
                "q_R-supported test contributes to Trigger-effect aggregation"
            ),
            "parent_identity": validation["parent_identity"],
            "prithvi": prithvi,
            "context_audit": context.audit(roles["train"], support["role_train_ids"]),
        }
        atomic_json(stage / "result.json", result)
        atomic_csv(stage / "same_checkpoint_controls.csv", control_rows)
        required = (
            "config.json", "command.txt", "run.log", "checkpoint.pt", "result.json",
            "per_sample.csv", "per_event.csv", "control_rows.csv",
            "same_checkpoint_controls.csv",
            "paired_control_receipts.csv",
        )
        hashes = {name: sha256_file(stage / name) for name in required}
        atomic_json(stage / "hashes.json", hashes)
        atomic_json(stage / "DONE.json", {
            "schema_version": "sen12_prithvi_roleaware_mr_done.v1",
            "status": "complete",
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "same_checkpoint_controls": True,
            "hashes_sha256": sha256_file(stage / "hashes.json"),
        })
        args.outdir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, args.outdir)
        print(json.dumps({"status": "complete", "outdir": str(args.outdir)}, indent=2))
        return 0
    except Exception:
        log("FAILED; staging directory retained for audit")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
