#!/usr/bin/env python3
"""Train nested Terrain then Terrain x Material susceptibility on Sen12.

This is a development-only gate for the hierarchical Full-TMR redesign. The
Terrain parent is selected first and frozen. Material is then allowed to add
only a bounded, per-patch zero-mean interaction with Terrain features. Epoch 0
is an exact Terrain-only candidate and remains selected unless aligned
Material beats both Terrain-only and an event-mismatched Material control on
held-out validation regions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

import train_sen12_xdomain_geophysadapter as protocol  # noqa: E402
from hierarchical_tmr_bayes import TerrainMaterialSusceptibility  # noqa: E402
from sen12_terrain_v2 import CURRENT_SCALE_GROUPS  # noqa: E402


BASE_H5 = PROJECT_ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5"
SPLIT_CSV = PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv"
MATERIAL_CSV = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v2/material_sample_registry_v2.csv"
MATERIAL_SCHEMA = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v2/material_feature_schema_v2.json"
DEFAULT_OUT = PROJECT_ROOT / "experiments/revision2026/sen12_tm_susceptibility_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=20260761)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--terrain-epochs", type=int, default=12)
    parser.add_argument("--material-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--terrain-lr", type=float, default=1e-3)
    parser.add_argument("--material-lr", type=float, default=5e-4)
    parser.add_argument("--material-logit-bound", type=float, default=1.0)
    parser.add_argument("--min-ap-gain", type=float, default=5e-4)
    parser.add_argument("--min-control-gap", type=float, default=5e-4)
    parser.add_argument("--max-brier-regression", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json_safe(row.get(key)) for key in fields} for row in rows)
    os.replace(temporary, path)


def state_to_cpu(parameters) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in parameters
        if value.requires_grad
    }


def load_named_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"checkpoint contains unknown parameters: {missing}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(values), values, self.mean)
        return np.clip((clean - self.mean) / self.scale, -5.0, 5.0).astype(np.float32)


def fit_material_normalizer(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    train_ids: Sequence[str],
) -> Normalizer:
    selected = frame.set_index("sample_id").loc[list(train_ids)].reset_index()
    event_medians = selected.groupby("physical_event_id")[list(feature_names)].median()
    values = event_medians.to_numpy(dtype=np.float64)
    mean = np.nanmedian(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = (q75 - q25) / 1.349
    standard = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, standard)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return Normalizer(mean.astype(np.float32), scale.astype(np.float32))


def fit_terrain_normalizer(
    h5_path: Path,
    index_by_id: Mapping[str, int],
    train_ids: Sequence[str],
) -> Normalizer:
    total = np.zeros(9, dtype=np.float64)
    squared = np.zeros(9, dtype=np.float64)
    counts = np.zeros(9, dtype=np.int64)
    with h5py.File(h5_path, "r") as handle:
        for sample_id in train_ids:
            index = index_by_id[sample_id]
            terrain = handle["terrain"][index].astype(np.float64)
            valid = handle["valid_mask"][index, 0].astype(bool)
            for channel in range(9):
                mask = valid & np.isfinite(terrain[channel])
                values = terrain[channel][mask]
                total[channel] += values.sum()
                squared[channel] += np.square(values).sum()
                counts[channel] += len(values)
    mean = total / np.maximum(counts, 1)
    variance = squared / np.maximum(counts, 1) - np.square(mean)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    return Normalizer(mean.astype(np.float32), scale.astype(np.float32))


def deterministic_material_donors(
    sample_ids: Sequence[str],
    event_ids: Sequence[str],
    region_ids: Sequence[str],
    seed: int,
) -> np.ndarray:
    donors = np.full(len(sample_ids), -1, dtype=np.int64)
    for index, sample_id in enumerate(sample_ids):
        candidates = [
            candidate
            for candidate in range(len(sample_ids))
            if event_ids[candidate] != event_ids[index]
            and region_ids[candidate] != region_ids[index]
        ]
        if not candidates:
            candidates = [
                candidate
                for candidate in range(len(sample_ids))
                if event_ids[candidate] != event_ids[index]
            ]
        if candidates:
            digest = hashlib.sha256(f"{seed}|M-shuffle|{sample_id}".encode()).digest()
            donors[index] = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
    return donors


class SusceptibilityDataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        sample_ids: Sequence[str],
        index_by_id: Mapping[str, int],
        split_rows: Mapping[str, Mapping[str, str]],
        material_frame: pd.DataFrame,
        feature_names: Sequence[str],
        terrain_normalizer: Normalizer,
        material_normalizer: Normalizer,
        *,
        control: str = "aligned",
        seed: int = 0,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.sample_ids = tuple(sample_ids)
        self.indices = np.asarray([index_by_id[value] for value in self.sample_ids], dtype=np.int64)
        lookup = material_frame.set_index("sample_id")
        ordered = lookup.loc[list(self.sample_ids)]
        self.material = material_normalizer.transform(
            ordered[list(feature_names)].to_numpy(dtype=np.float32)
        )
        self.q_m = ordered["q_M"].to_numpy(dtype=np.float32)
        self.event_ids = tuple(ordered["physical_event_id"].astype(str))
        self.region_ids = tuple(str(split_rows[value]["region_group"]) for value in self.sample_ids)
        self.terrain_normalizer = terrain_normalizer
        self.control = control
        self._h5: h5py.File | None = None
        self.donors = deterministic_material_donors(
            self.sample_ids, self.event_ids, self.region_ids, seed
        )
        if control not in {"aligned", "shuffle", "zero"}:
            raise ValueError(f"unknown Material control: {control}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _handle(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, Any]:
        handle = self._handle()
        source_index = int(self.indices[index])
        terrain = handle["terrain"][source_index].astype(np.float32)
        terrain = np.clip(
            (
                terrain
                - self.terrain_normalizer.mean[:, None, None]
            )
            / self.terrain_normalizer.scale[:, None, None],
            -5.0,
            5.0,
        ).astype(np.float32)
        mask = handle["mask"][source_index].astype(np.float32)
        valid = handle["valid_mask"][source_index].astype(np.float32)
        q_t = float(handle["q_T"][source_index])
        material_index = index
        q_m = float(self.q_m[index])
        if self.control == "shuffle":
            material_index = int(self.donors[index])
            if material_index < 0:
                q_m = 0.0
            else:
                q_m = min(q_m, float(self.q_m[material_index]))
        elif self.control == "zero":
            q_m = 0.0
        material = (
            np.zeros(self.material.shape[1], dtype=np.float32)
            if material_index < 0
            else self.material[material_index]
        )
        return {
            "sample_id": self.sample_ids[index],
            "event_id": self.event_ids[index],
            "region_id": self.region_ids[index],
            "terrain": torch.from_numpy(terrain),
            "material": torch.from_numpy(material),
            "q_m": torch.tensor(q_m, dtype=torch.float32),
            "q_t": torch.tensor(q_t, dtype=torch.float32),
            "mask": torch.from_numpy(mask),
            "valid": torch.from_numpy(valid),
        }


def make_loader(
    dataset: SusceptibilityDataset,
    args: argparse.Namespace,
    *,
    training: bool,
) -> DataLoader:
    sampler = None
    if training:
        event_counts = pd.Series(dataset.event_ids).value_counts().to_dict()
        weights = torch.tensor(
            [1.0 / event_counts[event] for event in dataset.event_ids], dtype=torch.double
        )
        generator = torch.Generator().manual_seed(args.seed + args.fold * 101)
        sampler = WeightedRandomSampler(
            weights, len(dataset), replacement=True, generator=generator
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    return bce + 0.5 * protocol.dice_loss_per_sample(logits, target, valid).mean()


@torch.inference_mode()
def score_model(
    model: TerrainMaterialSusceptibility,
    loader: DataLoader,
    device: str,
    *,
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    histogram = protocol.ProbabilityHistogram()
    brier_sum = 0.0
    nll_sum = 0.0
    count = 0
    model.eval()
    for batch in loader:
        terrain = batch["terrain"].to(device, non_blocking=True)
        material = batch["material"].to(device, non_blocking=True)
        q_m = batch["q_m"].to(device, non_blocking=True)
        q_t = batch["q_t"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        logits, _ = model(
            terrain, material, q_m, q_t=q_t, valid_mask=valid
        )
        probability = torch.sigmoid(logits.float())
        target = batch["mask"].to(device, non_blocking=True)
        selected = valid > 0.5
        p = probability[selected]
        y = target[selected]
        histogram.update(p.cpu().numpy(), y.cpu().numpy())
        brier_sum += float(torch.square(p - y).sum())
        nll_sum += float(F.binary_cross_entropy(p, y, reduction="sum"))
        count += int(selected.sum())
    if fixed_threshold is None:
        threshold, metrics = protocol.choose_threshold(histogram)
        threshold_source = "validation_grid"
    else:
        threshold = float(fixed_threshold)
        metrics = protocol.metrics_from_counts(histogram.counts_at(threshold))
        threshold_source = "frozen_validation"
    counts = histogram.counts_at(threshold)
    return {
        "ap": float(histogram.average_precision),
        "brier": brier_sum / max(count, 1),
        "nll": nll_sum / max(count, 1),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "counts": counts,
        "metrics": metrics,
    }


@torch.inference_mode()
def per_sample_metrics(
    model: TerrainMaterialSusceptibility,
    loader: DataLoader,
    device: str,
    threshold: float,
    condition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch in loader:
        terrain = batch["terrain"].to(device, non_blocking=True)
        material = batch["material"].to(device, non_blocking=True)
        q_m = batch["q_m"].to(device, non_blocking=True)
        q_t = batch["q_t"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        logits, audit = model(
            terrain, material, q_m, q_t=q_t, valid_mask=valid
        )
        prediction = torch.sigmoid(logits.float()) >= threshold
        target = batch["mask"].to(device, non_blocking=True) > 0.5
        selected = valid > 0.5
        for index, sample_id in enumerate(batch["sample_id"]):
            use = selected[index]
            pred = prediction[index] & use
            truth = target[index] & use
            tp = int((pred & truth).sum())
            fp = int((pred & ~truth & use).sum())
            fn = int((~pred & truth & use).sum())
            union = tp + fp + fn
            rows.append(
                {
                    "condition": condition,
                    "sample_id": sample_id,
                    "event_id": batch["event_id"][index],
                    "region_id": batch["region_id"][index],
                    "q_M": float(q_m[index].cpu()),
                    "q_T": float(q_t[index].cpu()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "iou": tp / union if union else 1.0,
                    "material_delta_abs_mean": float(
                        audit["material_delta"][index].detach().abs().mean().cpu()
                    ),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    args.outdir = args.outdir or DEFAULT_OUT / f"fold{args.fold}_seed{args.seed}"
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + args.fold)
    device = args.device

    schema = json.loads(MATERIAL_SCHEMA.read_text(encoding="utf-8"))
    feature_names = tuple(schema["model_eligible_features"])
    material_frame = pd.read_csv(
        MATERIAL_CSV, keep_default_na=False, low_memory=False
    )
    material_frame[list(feature_names) + ["q_M"]] = material_frame[
        list(feature_names) + ["q_M"]
    ].apply(pd.to_numeric, errors="coerce")
    with h5py.File(BASE_H5, "r") as handle:
        all_ids = decode(handle["sample_id"][:])
    index_by_id = {sample_id: index for index, sample_id in enumerate(all_ids)}
    if len(index_by_id) != len(all_ids):
        raise RuntimeError("base H5 repeats sample_id")
    if set(all_ids) != set(material_frame["sample_id"].astype(str)):
        raise RuntimeError("Material registry and base H5 sample identities differ")

    split_rows, roles, split_regions = protocol.load_logo_rows(SPLIT_CSV, args.fold)
    roles = {
        role: [sample_id for sample_id in values if sample_id in index_by_id]
        for role, values in roles.items()
    }
    if args.max_train:
        roles["train"] = roles["train"][: args.max_train]
    if args.max_eval:
        roles["val"] = roles["val"][: args.max_eval]
        roles["test"] = roles["test"][: args.max_eval]
    for role in ("train", "val", "test"):
        if not roles[role]:
            raise RuntimeError(f"empty {role} role")

    terrain_normalizer = fit_terrain_normalizer(BASE_H5, index_by_id, roles["train"])
    material_normalizer = fit_material_normalizer(
        material_frame, feature_names, roles["train"]
    )
    datasets: dict[str, SusceptibilityDataset] = {}
    for role in ("train", "val", "test"):
        for control in ("aligned", "shuffle", "zero"):
            datasets[f"{role}_{control}"] = SusceptibilityDataset(
                BASE_H5,
                roles[role],
                index_by_id,
                split_rows,
                material_frame,
                feature_names,
                terrain_normalizer,
                material_normalizer,
                control=control,
                seed=args.seed + args.fold * 1009,
            )
    loaders = {
        key: make_loader(data, args, training=key == "train_aligned")
        for key, data in datasets.items()
    }

    model = TerrainMaterialSusceptibility(
        9,
        len(feature_names),
        CURRENT_SCALE_GROUPS,
        material_logit_bound=args.material_logit_bound,
    ).to(device)
    for parameter in model.material_parameters():
        parameter.requires_grad_(False)

    positive = 0.0
    valid_count = 0.0
    with h5py.File(BASE_H5, "r") as handle:
        for sample_id in roles["train"]:
            index = index_by_id[sample_id]
            mask = handle["mask"][index].astype(np.float32)
            valid = handle["valid_mask"][index].astype(np.float32)
            positive += float((mask * valid).sum())
            valid_count += float(valid.sum())
    pos_weight_value = min(40.0, max(1.0, (valid_count - positive) / max(positive, 1.0)))
    pos_weight = torch.tensor([pos_weight_value], device=device).view(1, 1, 1, 1)
    amp_enabled = device.startswith("cuda")

    terrain_optimizer = torch.optim.AdamW(
        model.terrain.parameters(), lr=args.terrain_lr, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    terrain_history: list[dict[str, Any]] = []
    best_terrain_ap = -1.0
    best_terrain_epoch = 0
    best_terrain_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.terrain_epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in loaders["train_aligned"]:
            terrain = batch["terrain"].to(device, non_blocking=True)
            material = batch["material"].to(device, non_blocking=True)
            q_t = batch["q_t"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            terrain_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits, _ = model(
                    terrain,
                    material,
                    torch.zeros_like(batch["q_m"]).to(device),
                    q_t=q_t,
                    valid_mask=valid,
                )
                loss = masked_loss(logits, target, valid, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(terrain_optimizer)
            torch.nn.utils.clip_grad_norm_(model.terrain.parameters(), 1.0)
            scaler.step(terrain_optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * terrain.shape[0]
            seen += terrain.shape[0]
        score = score_model(model, loaders["val_zero"], device)
        row = {
            "stage": "terrain",
            "epoch": epoch,
            "loss": loss_sum / max(seen, 1),
            "val_ap": score["ap"],
            "val_brier": score["brier"],
            "val_iou": score["metrics"]["iou"],
        }
        terrain_history.append(row)
        print(json.dumps(row), flush=True)
        if score["ap"] > best_terrain_ap:
            best_terrain_ap = score["ap"]
            best_terrain_epoch = epoch
            best_terrain_state = {
                f"terrain.{name}": value.detach().cpu().clone()
                for name, value in model.terrain.named_parameters()
            }
    if best_terrain_state is None:
        raise RuntimeError("Terrain parent failed to produce a checkpoint")
    load_named_state(model, best_terrain_state)

    model.freeze_terrain()
    for parameter in model.material_parameters():
        parameter.requires_grad_(True)
    base_score = score_model(model, loaders["val_zero"], device)
    best_material_state = state_to_cpu(
        (name, value)
        for name, value in model.named_parameters()
        if name.startswith("material_")
    )
    selected_epoch = 0
    selected_reason = "identity_parent"
    selected_score = base_score
    selected_control_score = base_score
    material_history: list[dict[str, Any]] = [
        {
            "stage": "material",
            "epoch": 0,
            "qualified": True,
            "selection": "exact_terrain_identity",
            "val_ap": base_score["ap"],
            "val_brier": base_score["brier"],
            "shuffle_ap": base_score["ap"],
        }
    ]
    material_optimizer = torch.optim.AdamW(
        list(model.material_parameters()), lr=args.material_lr, weight_decay=1e-4
    )
    for epoch in range(1, args.material_epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in loaders["train_aligned"]:
            terrain = batch["terrain"].to(device, non_blocking=True)
            material = batch["material"].to(device, non_blocking=True)
            q_m = batch["q_m"].to(device, non_blocking=True)
            q_t = batch["q_t"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            material_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits, audit = model(
                    terrain, material, q_m, q_t=q_t, valid_mask=valid
                )
                loss = masked_loss(logits, target, valid, pos_weight)
                loss = loss + 1e-3 * torch.square(audit["material_delta"]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(material_optimizer)
            torch.nn.utils.clip_grad_norm_(model.material_parameters(), 1.0)
            scaler.step(material_optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * terrain.shape[0]
            seen += terrain.shape[0]
        aligned = score_model(model, loaders["val_aligned"], device)
        shuffled = score_model(model, loaders["val_shuffle"], device)
        qualified = (
            aligned["ap"] >= base_score["ap"] + args.min_ap_gain
            and aligned["ap"] >= shuffled["ap"] + args.min_control_gap
            and aligned["brier"] <= base_score["brier"] + args.max_brier_regression
        )
        row = {
            "stage": "material",
            "epoch": epoch,
            "loss": loss_sum / max(seen, 1),
            "qualified": qualified,
            "val_ap": aligned["ap"],
            "val_brier": aligned["brier"],
            "val_iou": aligned["metrics"]["iou"],
            "shuffle_ap": shuffled["ap"],
            "ap_gain_vs_T": aligned["ap"] - base_score["ap"],
            "ap_gap_vs_shuffle": aligned["ap"] - shuffled["ap"],
        }
        material_history.append(row)
        print(json.dumps(row), flush=True)
        if qualified and (
            selected_epoch == 0
            or (aligned["ap"], -aligned["brier"])
            > (selected_score["ap"], -selected_score["brier"])
        ):
            best_material_state = state_to_cpu(
                (name, value)
                for name, value in model.named_parameters()
                if name.startswith("material_")
            )
            selected_epoch = epoch
            selected_reason = "aligned_beats_parent_and_material_shuffle"
            selected_score = aligned
            selected_control_score = shuffled
    load_named_state(model, best_material_state)

    test_scores: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    if not args.skip_test:
        for control in ("zero", "aligned", "shuffle"):
            score = score_model(
                model,
                loaders[f"test_{control}"],
                device,
                fixed_threshold=selected_score["threshold"],
            )
            test_scores[control] = score
            sample_rows.extend(
                per_sample_metrics(
                    model,
                    loaders[f"test_{control}"],
                    device,
                    selected_score["threshold"],
                    control,
                )
            )

    config = {
        "status": "sen12_development_only",
        "fold": args.fold,
        "seed": args.seed,
        "roles": {key: len(value) for key, value in roles.items()},
        "regions": {key: sorted(value) for key, value in split_regions.items()},
        "feature_names": list(feature_names),
        "terrain_channels": 9,
        "pos_weight": pos_weight_value,
        "selection_contract": {
            "epoch0_exact_terrain_identity": True,
            "min_ap_gain": args.min_ap_gain,
            "min_control_gap": args.min_control_gap,
            "max_brier_regression": args.max_brier_regression,
        },
        "inputs": {
            "base_h5": {"path": str(BASE_H5), "sha256": sha256_file(BASE_H5)},
            "split": {"path": str(SPLIT_CSV), "sha256": sha256_file(SPLIT_CSV)},
            "material": {"path": str(MATERIAL_CSV), "sha256": sha256_file(MATERIAL_CSV)},
            "material_schema": {"path": str(MATERIAL_SCHEMA), "sha256": sha256_file(MATERIAL_SCHEMA)},
        },
    }
    result = {
        **config,
        "terrain_best_epoch": best_terrain_epoch,
        "terrain_best_val_ap": best_terrain_ap,
        "material_selected_epoch": selected_epoch,
        "material_selection_reason": selected_reason,
        "material_gate_pass": selected_epoch > 0,
        "val_T": base_score,
        "val_TM": selected_score,
        "val_TM_shuffle": selected_control_score,
        "test": test_scores,
        "evidence_role": "development; not independent Full-TMR confirmation",
    }
    checkpoint = {
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "terrain_normalizer": {"mean": terrain_normalizer.mean, "scale": terrain_normalizer.scale},
        "material_normalizer": {"mean": material_normalizer.mean, "scale": material_normalizer.scale},
        "feature_names": feature_names,
        "result": result,
    }
    checkpoint_path = args.outdir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    write_json(args.outdir / "config.json", config)
    write_json(args.outdir / "result.json", result)
    write_csv(args.outdir / "history.csv", terrain_history + material_history)
    if sample_rows:
        write_csv(args.outdir / "per_sample_test.csv", sample_rows)
    done = {
        "status": "complete",
        "material_gate_pass": selected_epoch > 0,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "result_sha256": sha256_file(args.outdir / "result.json"),
        "history_sha256": sha256_file(args.outdir / "history.csv"),
    }
    if sample_rows:
        done["per_sample_test_sha256"] = sha256_file(args.outdir / "per_sample_test.csv")
    write_json(args.outdir / "DONE.json", done)
    print(json.dumps({"material_gate_pass": selected_epoch > 0, "selected_epoch": selected_epoch, "test": test_scores}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
