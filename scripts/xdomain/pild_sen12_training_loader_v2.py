#!/usr/bin/env python3
"""Gated unified loader and source/event/patch balanced sampler for PILD + Sen12."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator, Literal

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


REQUIRED_MANIFEST_COLUMNS = {
    "dataset_id",
    "source_id",
    "source_event_id",
    "canonical_event_id",
    "sample_id",
    "base_h5_path",
    "base_h5_index",
    "optical_h5_path",
    "optical_h5_index",
    "terrain_h5_path",
    "terrain_h5_index",
    "terrain_channel_indices",
    "material_registry_path",
    "material_registry_index",
    "trigger_registry_path",
    "trigger_registry_index",
    "core_assets_ready",
    "full_tmr_assets_ready",
}

MATERIAL_BASES = tuple(
    (name, depth)
    for name in ("clay", "sand", "silt", "cec", "soc", "bdod", "cfvo", "phh2o")
    for depth in ("0_5cm", "5_15cm")
)
MATERIAL_AWC_COLUMNS = (
    "awc_0_10_aligned_mm",
    "awc_10_30_aligned_mm",
    "awc_30_60_aligned_mm",
    "awc_60_100_aligned_mm",
    "awc_100_200_aligned_mm",
    "awc_0_200_aligned_mm",
)
MATERIAL_FEATURE_NAMES = tuple(
    value
    for name, depth in MATERIAL_BASES
    for value in (f"soil_{name}_{depth}_mean", f"soil_{name}_{depth}_local_std")
) + MATERIAL_AWC_COLUMNS
ROLE_MATERIAL_FEATURE_NAMES = (
    "awc_0_10_footprint_mean_mm",
    "awc_10_30_footprint_mean_mm",
    "awc_30_60_footprint_mean_mm",
    "awc_60_100_footprint_mean_mm",
    "awc_100_200_footprint_mean_mm",
) + tuple(
    f"soil_{name}_{depth}_mean_raw" for name, depth in MATERIAL_BASES
)
TRIGGER_FEATURE_NAMES = (
    "rain_d7_case_mm",
    "rain_d7_wrongtime_median_mm",
    "rain_d7_case_minus_wrongtime_mm",
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def finite_float(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if name not in row or row[name] == "":
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return 0.0


def harmonize_material(row: dict[str, Any] | None) -> tuple[np.ndarray, float]:
    if row is None:
        return np.zeros(len(MATERIAL_FEATURE_NAMES), dtype=np.float32), 0.0
    values = []
    for name, depth in MATERIAL_BASES:
        prefix = f"soil_{name}_{depth}"
        values.append(finite_float(row, f"{prefix}_mean_raw"))
        values.append(
            finite_float(
                row,
                f"{prefix}_local_std_raw",
                f"{prefix}_native_cell_std_raw",
            )
        )
    values.extend(finite_float(row, column) for column in MATERIAL_AWC_COLUMNS)
    quality = finite_float(row, "q_M_full")
    return np.asarray(values, dtype=np.float32), float(np.clip(quality, 0.0, 1.0))


def harmonize_role_material(row: dict[str, Any] | None) -> tuple[np.ndarray, float]:
    """Build the fixed 21-D Material vector used by the role-aware interaction.

    PILD provides footprint AWC means, whereas Sen12 stores the aligned native
    value. The latter is the explicit fallback; this alias is recorded in the
    schema instead of changing feature order between sources.
    """

    if row is None:
        return np.zeros(len(ROLE_MATERIAL_FEATURE_NAMES), dtype=np.float32), 0.0
    values = [
        finite_float(
            row,
            f"awc_{depth}_footprint_mean_mm",
            f"awc_{depth}_aligned_mm",
        )
        for depth in ("0_10", "10_30", "30_60", "60_100", "100_200")
    ]
    values.extend(
        finite_float(row, f"soil_{name}_{depth}_mean_raw")
        for name, depth in MATERIAL_BASES
    )
    quality = finite_float(row, "q_M_full")
    return np.asarray(values, dtype=np.float32), float(np.clip(quality, 0.0, 1.0))


def harmonize_trigger(row: dict[str, Any] | None) -> tuple[np.ndarray, float]:
    if row is None:
        return np.zeros(len(TRIGGER_FEATURE_NAMES), dtype=np.float32), 0.0
    values = np.asarray(
        [
            finite_float(row, "rain_d7_antecedent_case_mm", "rain_d7_case_mm"),
            finite_float(row, "rain_d7_wrongtime_median_mm"),
            finite_float(row, "rain_d7_case_minus_wrongtime_mm"),
        ],
        dtype=np.float32,
    )
    quality = finite_float(row, "q_R")
    return values, float(np.clip(quality, 0.0, 1.0))


def load_protocol_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("validation_status") != "PASS":
        raise RuntimeError(f"protocol validation is not PASS: {path}")
    return summary


def inspect_readiness(summary_path: Path) -> dict[str, Any]:
    """Return readiness without opening any training tensor cache."""
    summary = load_protocol_summary(summary_path)
    return dict(summary.get("readiness", {}))


class SourceEventPatchBalancedSampler(Sampler[int]):
    """Balance source, then canonical event, then patch with replacement.

    Each hierarchy level is emitted in shuffled cycles. Thus counts differ by
    at most one at every level and a large event cannot dominate an epoch just
    because it contains more patches.
    """

    def __init__(self, frame: pd.DataFrame, num_samples: int | None = None, seed: int = 0) -> None:
        required = {"source_id", "canonical_event_id", "sample_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"sampler frame missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("sampler frame is empty")
        self.num_samples = int(num_samples if num_samples is not None else len(frame))
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.epoch = 0
        self.sources = tuple(sorted(frame["source_id"].astype(str).unique()))
        pools: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
        for (source, event), group in frame.groupby(["source_id", "canonical_event_id"], sort=True):
            pools[str(source)][str(event)] = tuple(int(index) for index in group.index)
        self.pools = {source: dict(events) for source, events in pools.items()}
        if set(self.pools) != set(self.sources):
            raise AssertionError("sampler source hierarchy is inconsistent")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _refill(queue: deque[Any], values: tuple[Any, ...], rng: random.Random) -> None:
        cycle = list(values)
        rng.shuffle(cycle)
        queue.extend(cycle)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        source_queue: deque[str] = deque()
        event_queues: dict[str, deque[str]] = {source: deque() for source in self.sources}
        patch_queues: dict[tuple[str, str], deque[int]] = {
            (source, event): deque()
            for source, events in self.pools.items()
            for event in events
        }
        for _ in range(self.num_samples):
            if not source_queue:
                self._refill(source_queue, self.sources, rng)
            source = source_queue.popleft()
            events = tuple(sorted(self.pools[source]))
            if not event_queues[source]:
                self._refill(event_queues[source], events, rng)
            event = event_queues[source].popleft()
            key = (source, event)
            if not patch_queues[key]:
                self._refill(patch_queues[key], self.pools[source][event], rng)
            yield patch_queues[key].popleft()

    def __len__(self) -> int:
        return self.num_samples


class DatasetEventPatchBalancedSampler(SourceEventPatchBalancedSampler):
    """Balance dataset, then canonical event, then patch with replacement.

    ``source_id`` groups the legacy PILD members together, so source balancing
    alone cannot prevent DLR, GDCLD, or GLaD from being dominated by a larger
    member. This sampler treats ``dataset_id`` as the top-level training unit.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if "dataset_id" not in frame:
            raise ValueError("dataset-balanced sampler frame missing dataset_id")
        dataset_frame = frame.copy()
        dataset_frame["source_id"] = dataset_frame["dataset_id"].astype(str)
        super().__init__(dataset_frame, num_samples=num_samples, seed=seed)


class TemperedDatasetEventPatchSampler(Sampler[int]):
    """Sample datasets and events with size-tempered probabilities."""

    def __init__(
        self,
        frame: pd.DataFrame,
        num_samples: int | None = None,
        seed: int = 0,
        dataset_temperature: float = 0.75,
        event_temperature: float = 0.75,
    ) -> None:
        required = {"dataset_id", "canonical_event_id", "sample_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"tempered sampler frame missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("tempered sampler frame is empty")
        if not 0 < dataset_temperature <= 1 or not 0 < event_temperature <= 1:
            raise ValueError("sampling temperatures must be in (0, 1]")
        self.num_samples = int(num_samples if num_samples is not None else len(frame))
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.epoch = 0
        pools: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
        for (dataset, event), group in frame.groupby(
            ["dataset_id", "canonical_event_id"], sort=True
        ):
            pools[str(dataset)][str(event)] = tuple(int(index) for index in group.index)
        self.pools = {dataset: dict(events) for dataset, events in pools.items()}
        self.datasets = tuple(sorted(self.pools))
        self.dataset_weights = tuple(
            sum(len(indices) for indices in self.pools[dataset].values())
            ** float(dataset_temperature)
            for dataset in self.datasets
        )
        self.events = {
            dataset: tuple(sorted(self.pools[dataset])) for dataset in self.datasets
        }
        self.event_weights = {
            dataset: tuple(
                len(self.pools[dataset][event]) ** float(event_temperature)
                for event in self.events[dataset]
            )
            for dataset in self.datasets
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        patch_queues: dict[tuple[str, str], deque[int]] = {
            (dataset, event): deque()
            for dataset, events in self.pools.items()
            for event in events
        }
        for _ in range(self.num_samples):
            dataset = rng.choices(self.datasets, weights=self.dataset_weights, k=1)[0]
            event = rng.choices(
                self.events[dataset],
                weights=self.event_weights[dataset],
                k=1,
            )[0]
            key = (dataset, event)
            if not patch_queues[key]:
                cycle = list(self.pools[dataset][event])
                rng.shuffle(cycle)
                patch_queues[key].extend(cycle)
            yield patch_queues[key].popleft()

    def __len__(self) -> int:
        return self.num_samples


class NaturalPatchSampler(Sampler[int]):
    """Sample patches in observed proportions without source reweighting."""

    def __init__(
        self,
        frame: pd.DataFrame,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if frame.empty:
            raise ValueError("sampler frame is empty")
        self.indices = tuple(int(index) for index in frame.index)
        self.num_samples = int(num_samples if num_samples is not None else len(frame))
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        emitted = 0
        while emitted < self.num_samples:
            cycle = list(self.indices)
            rng.shuffle(cycle)
            selected = cycle[: self.num_samples - emitted]
            yield from selected
            emitted += len(selected)

    def __len__(self) -> int:
        return self.num_samples


class UnifiedPILDSen12Dataset(Dataset[dict[str, Any]]):
    """Read a frozen unified manifest after its readiness gate passes."""

    def __init__(
        self,
        manifest_path: Path,
        protocol_summary_path: Path,
        *,
        split_path: Path | None = None,
        fold_id: str | None = None,
        role: Literal["train", "val", "test"] | None = None,
        readiness: Literal["core", "full_tmr"] = "core",
        allow_incomplete: bool = False,
        verify_manifest_hash: bool = True,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.summary_path = protocol_summary_path.resolve()
        self.summary = load_protocol_summary(self.summary_path)
        readiness_key = f"{readiness}_training_ready"
        ready = bool(self.summary.get("readiness", {}).get(readiness_key, False))
        if not ready and not allow_incomplete:
            blockers = self.summary.get("readiness", {}).get("blockers", [])
            raise RuntimeError(
                f"{readiness} training assets are not ready; loader refused to open. "
                f"Blockers: {blockers}"
            )
        outputs = self.summary.get("outputs", {})
        expected_hash = outputs.get("manifest", {}).get("sha256")
        if verify_manifest_hash and expected_hash and sha256_file(self.manifest_path) != expected_hash:
            raise RuntimeError("unified manifest SHA-256 differs from protocol summary")

        frame = pd.read_csv(self.manifest_path, keep_default_na=False)
        missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"unified manifest missing columns: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError("unified manifest sample_id is not unique")
        if split_path is not None:
            split = pd.read_csv(split_path, keep_default_na=False)
            required_split = {"fold_id", "sample_id", "canonical_event_id", "role"}
            if required_split - set(split.columns):
                raise ValueError(f"split missing columns: {sorted(required_split - set(split.columns))}")
            available_folds = sorted(split["fold_id"].astype(str).unique())
            if fold_id is None and len(available_folds) != 1:
                raise ValueError(f"fold_id is required; available folds: {available_folds}")
            selected_fold = fold_id or available_folds[0]
            split = split[split["fold_id"].astype(str).eq(selected_fold)].copy()
            if split.empty:
                raise ValueError(f"unknown or empty fold_id={selected_fold!r}")
            if split["sample_id"].duplicated().any():
                raise ValueError(f"split fold {selected_fold!r} repeats sample_id")
            if role is not None:
                split = split[split["role"].eq(role)].copy()
            frame = frame.merge(
                split[["sample_id", "canonical_event_id", "role", "role_reason"]],
                on=["sample_id", "canonical_event_id"],
                how="inner",
                validate="one_to_one",
            )
        elif role is not None:
            raise ValueError("role filtering requires split_path")
        if frame.empty:
            raise ValueError("dataset selection is empty")
        readiness_column = "core_assets_ready" if readiness == "core" else "full_tmr_assets_ready"
        if not allow_incomplete and not frame[readiness_column].astype(bool).all():
            raise RuntimeError(f"selected rows contain {readiness_column}=0")
        self.frame = frame.reset_index(drop=True)
        self._h5: dict[str, h5py.File] = {}
        self._support: dict[str, pd.DataFrame] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _handle(self, path_text: str) -> h5py.File:
        if path_text not in self._h5:
            path = Path(path_text)
            if not path.is_file():
                raise RuntimeError(f"training cache is missing: {path}")
            self._h5[path_text] = h5py.File(path, "r")
        return self._h5[path_text]

    @staticmethod
    def _assert_sample(handle: h5py.File, index: int, expected: str, path: str) -> None:
        observed = decode(handle["sample_id"][index])
        if observed != expected:
            raise RuntimeError(
                f"HDF5 identity mismatch at {path}[{index}]: observed={observed!r}, expected={expected!r}"
            )

    def support_row(self, index: int, kind: Literal["material", "trigger"]) -> dict[str, Any] | None:
        row = self.frame.iloc[index]
        path_text = str(row[f"{kind}_registry_path"])
        registry_index = int(row[f"{kind}_registry_index"])
        if not path_text or registry_index < 0:
            return None
        if path_text not in self._support:
            self._support[path_text] = pd.read_csv(path_text, keep_default_na=False, low_memory=False)
        support = self._support[path_text]
        if registry_index >= len(support):
            raise RuntimeError(f"{kind} registry index is out of range")
        value = support.iloc[registry_index].to_dict()
        if str(value.get("sample_id")) != str(row["sample_id"]):
            raise RuntimeError(f"{kind} registry identity differs from unified manifest")
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        if not bool(row["core_assets_ready"]):
            raise RuntimeError(
                f"sample {row['sample_id']} is audit-only because core_assets_ready=0"
            )
        sample_id = str(row["sample_id"])
        base_index = int(row["base_h5_index"])
        optical_index = int(row["optical_h5_index"])
        terrain_index = int(row["terrain_h5_index"])
        base_path = str(row["base_h5_path"])
        optical_path = str(row["optical_h5_path"])
        terrain_path = str(row["terrain_h5_path"])
        base = self._handle(base_path)
        optical = self._handle(optical_path)
        terrain_handle = self._handle(terrain_path)
        self._assert_sample(base, base_index, sample_id, base_path)
        self._assert_sample(optical, optical_index, sample_id, optical_path)
        self._assert_sample(terrain_handle, terrain_index, sample_id, terrain_path)

        channel_indices = np.asarray(
            [int(value) for value in str(row["terrain_channel_indices"]).split(";")],
            dtype=np.int64,
        )
        optical_value = np.asarray(optical["optical"][optical_index], dtype=np.float32) / 10_000.0
        terrain_value = np.asarray(terrain_handle["terrain"][terrain_index], dtype=np.float32)[channel_indices]
        mask = np.asarray(base["mask"][base_index], dtype=np.float32)
        base_valid = np.asarray(base["valid_mask"][base_index], dtype=np.uint8)
        optical_valid = np.asarray(optical["optical_valid"][optical_index], dtype=np.uint8)
        terrain_valid = np.asarray(terrain_handle["terrain_valid"][terrain_index], dtype=np.uint8)
        valid = np.logical_and.reduce((base_valid > 0, optical_valid > 0, terrain_valid > 0)).astype(np.float32)
        q_visual = (
            float(optical["q_visual_temporal"][optical_index])
            if "q_visual_temporal" in optical
            else 1.0
        )
        material_row = self.support_row(index, "material")
        material_features, q_material = harmonize_material(material_row)
        role_material_features, role_q_material = harmonize_role_material(material_row)
        if role_q_material != q_material:
            raise RuntimeError(f"Material quality mismatch for sample {sample_id}")
        trigger_features, q_trigger = harmonize_trigger(
            self.support_row(index, "trigger")
        )
        return {
            "optical": torch.from_numpy(optical_value),
            "temporal_coords": torch.from_numpy(
                np.asarray(optical["temporal_coords"][optical_index], dtype=np.float32)
            ),
            "location_coords": torch.from_numpy(
                np.asarray(optical["location_coords"][optical_index], dtype=np.float32)
            ),
            "terrain": torch.from_numpy(terrain_value),
            "terrain_valid": torch.from_numpy(terrain_valid.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "valid_mask": torch.from_numpy(valid),
            "q_visual_temporal": torch.tensor(q_visual, dtype=torch.float32),
            "material_features": torch.from_numpy(material_features),
            "role_material_features": torch.from_numpy(role_material_features),
            "q_material": torch.tensor(q_material, dtype=torch.float32),
            "trigger_features": torch.from_numpy(trigger_features),
            "q_trigger": torch.tensor(q_trigger, dtype=torch.float32),
            "sample_id": sample_id,
            "dataset_id": str(row["dataset_id"]),
            "source_id": str(row["source_id"]),
            "source_event_id": str(row["source_event_id"]),
            "canonical_event_id": str(row["canonical_event_id"]),
            "material_available": bool(row["material_ready"]),
            "trigger_available": bool(row["trigger_ready"]),
        }

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


def create_balanced_dataloader(
    dataset: UnifiedPILDSen12Dataset,
    *,
    batch_size: int,
    epoch_samples: int | None = None,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, SourceEventPatchBalancedSampler]:
    sampler = SourceEventPatchBalancedSampler(dataset.frame, num_samples=epoch_samples, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, sampler
