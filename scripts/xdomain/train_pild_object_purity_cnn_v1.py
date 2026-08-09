#!/usr/bin/env python3
"""G2: predict candidate-body purity from object crops, then veto by expected gain.

G1 showed the object-level decision is the right unit but that hand-crafted scalar
summaries cannot rank bodies well enough: purity correlation was 0.333 and the achieved
delta IoU was +0.019 against a perfect-ranking ceiling of +0.111. Aggregate statistics
discard the spatial arrangement that actually distinguishes a failure scar from a river
bar or a terrace, so this stage looks at the crop itself.

Each candidate body becomes one training example built from a fixed window centred on
its centroid, with channel groups that can be ablated independently:

    terrain   17 native Terrain channels (the physical evidence under test)
    optical   pre-event, post-event and change composites (appearance evidence)
    visual     frozen visual probability map (what the anchor already believes)
    mask       the candidate footprint, so the network knows which body it is judging

Decisions reuse the analytic rule from G1: remove a body when the expected pooled-IoU
gain implied by predicted purity clears a threshold chosen on held-in events only. The
same Terrain mismatch controls apply, so attribution to aligned physics survives.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_CACHE = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
DEFAULT_COMPONENTS = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_physical_diagnostic_v1/separability_v1"
    / "component_features.csv"
)
FOLDS = (
    "source_stratified_0",
    "source_stratified_1",
    "source_stratified_2",
    "source_stratified_3",
)
CHANNEL_GROUPS = ("terrain", "optical", "visual", "mask")

# Whole-body summaries. A fixed crop cannot cover a large body, so these keep the
# aggregate view that the G1 scalar model ranked with, and the crop adds the spatial
# arrangement that those aggregates discard.
SCALAR_FEATURES = (
    "area_px",
    "log_area",
    "mean_slope",
    "p10_slope",
    "p90_slope",
    "flat_fraction",
    "steep_fraction",
    "elev_range",
    "relative_relief",
    "aspect_coherence",
    "elongation",
    "downslope_alignment",
    "descent_consistency",
    "slope_decline",
    "divide_straddle",
    "tpi900_range",
    "mean_tpi_90m",
    "mean_tpi_300m",
    "mean_tpi_900m",
    "valley_bottom_fraction",
    "mean_valley_depth",
    "mean_ridge_height",
    "mean_ruggedness",
    "mean_local_relief_300m",
    "mean_plan_curvature",
    "mean_profile_curvature",
    "compactness",
    "terrain_support_fraction",
    "mean_probability",
    "max_probability",
    "p90_probability",
)


def components_path_for(condition: str, root: Path) -> Path:
    """Scalar summaries must come from the same Terrain condition as the crops."""
    directory = "separability_v1" if condition == "aligned" else f"separability_{condition}"
    return root / directory / "component_features.csv"


def terrain_condition(
    terrain: np.ndarray, condition: str, donor_index: np.ndarray | None
) -> np.ndarray:
    """Same interventions as the G0/G1 controls, applied to the Terrain stack only."""
    if condition == "aligned":
        return terrain
    if condition == "zero":
        return np.zeros_like(terrain)
    if condition in {"shift32", "roll64"}:
        shift = 32 if condition == "shift32" else 64
        return np.roll(terrain, shift=(shift, shift), axis=(-2, -1))
    if condition == "donor":
        if donor_index is None:
            raise ValueError("donor condition needs a donor index")
        return terrain[donor_index]
    raise ValueError(f"unsupported condition: {condition}")


def build_donor_index(
    dataset_id: np.ndarray, event_id: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    donor = np.arange(len(dataset_id))
    for name in np.unique(dataset_id):
        members = np.nonzero(dataset_id == name)[0]
        events = event_id[members]
        for position, index in enumerate(members):
            candidates = members[events != events[position]]
            if candidates.size == 0:
                candidates = members[members != index]
            if candidates.size:
                donor[index] = int(rng.choice(candidates))
    return donor


class FoldArrays:
    """One fold of cached predictions, Terrain, optical composites and context."""

    def __init__(self, cache_dir: Path, fold_id: str, condition: str, seed: int) -> None:
        with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as h:
            self.sample_id = np.asarray([str(item) for item in h["sample_id"]])
            self.dataset_id = np.asarray([str(item) for item in h["dataset_id"]])
            self.event_id = np.asarray([str(item) for item in h["canonical_event_id"]])
            self.probability = h["visual_probability"].astype(np.float16)
            self.target = h["target"].astype(np.uint8)
            self.valid = h["valid"].astype(np.uint8)
            terrain = h["terrain"].astype(np.float16)
        with np.load(cache_dir / f"{fold_id}_optical_cache.npz", allow_pickle=False) as h:
            optical_ids = np.asarray([str(item) for item in h["sample_id"]])
            if not np.array_equal(optical_ids, self.sample_id):
                raise RuntimeError(f"{fold_id}: optical cache order mismatch")
            self.optical_pre = h["optical_pre"].astype(np.float16)
            self.optical_post = h["optical_post"].astype(np.float16)
        donor = (
            build_donor_index(self.dataset_id, self.event_id, seed)
            if condition == "donor"
            else None
        )
        self.terrain = terrain_condition(terrain, condition, donor)
        self.index = {value: position for position, value in enumerate(self.sample_id)}


def precompute_footprints(
    components: pd.DataFrame,
    folds: dict[str, FoldArrays],
    thresholds: dict[str, float],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Label each patch once and store the pixel coordinates of every component.

    Labelling inside ``__getitem__`` would repeat the same connected-component pass for
    every body in a patch; the coordinate lists cost only a few tens of megabytes.
    """
    structure = ndimage.generate_binary_structure(2, 2)
    footprints: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for (fold_id, sample_id), group in components.groupby(
        ["fold_id", "sample_id"], sort=False
    ):
        fold = folds[str(fold_id)]
        position = fold.index[str(sample_id)]
        predicted = (
            fold.probability[position].astype(np.float32) >= thresholds[str(fold_id)]
        ) & (fold.valid[position] > 0)
        labels, _ = ndimage.label(predicted, structure=structure)
        for row_index, component_id in zip(
            group.index.to_numpy(), group.component_id.to_numpy(), strict=True
        ):
            rows, cols = np.nonzero(labels == int(component_id))
            if rows.size == 0:
                rows, cols = np.nonzero(predicted)
            footprints[int(row_index)] = (
                rows.astype(np.int16),
                cols.astype(np.int16),
            )
    return footprints


class ObjectCropDataset(Dataset[dict[str, torch.Tensor]]):
    """Crops centred on each candidate body, with per-fold normalization statistics."""

    def __init__(
        self,
        components: pd.DataFrame,
        folds: dict[str, FoldArrays],
        footprints: dict[int, tuple[np.ndarray, np.ndarray]],
        *,
        crop: int,
        groups: Sequence[str],
        terrain_mean: np.ndarray,
        terrain_scale: np.ndarray,
        optical_scale: float,
        weight_reference: float,
        weight_cap: float,
        scalars: np.ndarray | None = None,
    ) -> None:
        self.components = components
        self.rows = components.index.to_numpy()
        self.folds = folds
        self.footprints = footprints
        self.crop = int(crop)
        self.groups = tuple(groups)
        self.terrain_mean = terrain_mean.astype(np.float32)[:, None, None]
        self.terrain_scale = terrain_scale.astype(np.float32)[:, None, None]
        self.optical_scale = float(optical_scale)
        self.weight_reference = float(weight_reference)
        self.weight_cap = float(weight_cap)
        self.scalars = None if scalars is None else scalars.astype(np.float32)

    def __len__(self) -> int:
        return len(self.components)

    def _window(self, row: int, col: int, size: int) -> tuple[slice, slice]:
        half = self.crop // 2
        top = int(np.clip(row - half, 0, size - self.crop))
        left = int(np.clip(col - half, 0, size - self.crop))
        return slice(top, top + self.crop), slice(left, left + self.crop)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        global_row = int(self.rows[index])
        record = self.components.loc[global_row]
        fold = self.folds[str(record.fold_id)]
        position = fold.index[str(record.sample_id)]
        probability = fold.probability[position].astype(np.float32)
        size = probability.shape[-1]

        rows, cols = self.footprints[global_row]
        window = self._window(int(rows.mean()), int(cols.mean()), size)
        footprint = np.zeros((size, size), dtype=bool)
        footprint[rows.astype(np.int64), cols.astype(np.int64)] = True

        planes: list[np.ndarray] = []
        if "terrain" in self.groups:
            terrain = fold.terrain[position][:, window[0], window[1]].astype(np.float32)
            planes.append((terrain - self.terrain_mean) / self.terrain_scale)
        if "optical" in self.groups:
            pre = fold.optical_pre[position][:, window[0], window[1]].astype(np.float32)
            post = fold.optical_post[position][:, window[0], window[1]].astype(np.float32)
            planes.extend(
                [
                    pre / self.optical_scale,
                    post / self.optical_scale,
                    (post - pre) / self.optical_scale,
                ]
            )
        if "visual" in self.groups:
            planes.append(probability[None, window[0], window[1]])
        if "mask" in self.groups:
            planes.append(footprint[None, window[0], window[1]].astype(np.float32))

        crop = np.concatenate(planes, axis=0)
        crop = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)
        # Mass-aware but capped: a single very large body must not dominate a batch.
        weight = min(float(record.area_px) / self.weight_reference, self.weight_cap)
        item = {
            "crop": torch.from_numpy(crop),
            "purity": torch.tensor(float(record.purity), dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "row": torch.tensor(int(index), dtype=torch.long),
        }
        if self.scalars is not None:
            item["scalars"] = torch.from_numpy(self.scalars[index])
        return item


class PurityNet(nn.Module):
    """Crop encoder fused with whole-body summaries, predicting a single purity value."""

    def __init__(self, in_channels: int, n_scalars: int = 0, width: int = 48) -> None:
        super().__init__()

        def block(inputs: int, outputs: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.GELU(),
                nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.GELU(),
            )

        self.stem = block(in_channels, width, 1)
        self.stage1 = block(width, width * 2, 2)
        self.stage2 = block(width * 2, width * 4, 2)
        self.stage3 = block(width * 4, width * 4, 2)
        embedding = width * 8
        self.scalar_branch: nn.Module | None = None
        if n_scalars > 0:
            self.scalar_branch = nn.Sequential(
                nn.Linear(n_scalars, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
            )
            embedding += 128
        self.head = nn.Sequential(
            nn.Linear(embedding, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(
        self, crop: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.stage3(self.stage2(self.stage1(self.stem(crop))))
        pooled = torch.cat(
            [features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1
        )
        if self.scalar_branch is not None:
            if scalars is None:
                raise ValueError("model expects scalar summaries")
            pooled = torch.cat([pooled, self.scalar_branch(scalars)], dim=1)
        return self.head(pooled).squeeze(-1)


def expected_gain(
    purity_hat: np.ndarray, area: np.ndarray, *, tp: float, fp: float, fn: float
) -> np.ndarray:
    denominator = tp + fp + fn
    baseline = tp / denominator
    purity_hat = np.clip(purity_hat, 0.0, 1.0)
    true_hat = purity_hat * area
    false_hat = (1.0 - purity_hat) * area
    return (tp - true_hat) / np.clip(denominator - false_hat, 1.0, None) - baseline


def best_cut(
    score: np.ndarray,
    false_px: np.ndarray,
    intersection_px: np.ndarray,
    *,
    tp: float,
    fp: float,
    fn: float,
) -> tuple[float, float]:
    denominator = tp + fp + fn
    baseline = tp / denominator
    order = np.argsort(-score, kind="stable")
    curve = (tp - np.cumsum(intersection_px[order])) / np.clip(
        denominator - np.cumsum(false_px[order]), 1.0, None
    ) - baseline
    best = int(np.argmax(curve))
    if float(curve[best]) <= 0.0:
        return float(np.nextafter(float(score.max()), np.inf)), 0.0
    return float(score[order][best]), float(curve[best])


def evaluate_removals(
    frame: pd.DataFrame, remove: np.ndarray, *, tp: float, fp: float, fn: float
) -> dict[str, float]:
    removed = frame[remove]
    lost = float(removed.intersection_px.sum())
    cleared = float(removed.false_px.sum())
    baseline = tp / (tp + fp + fn)
    adapted = (tp - lost) / max(tp + fp + fn - cleared, 1.0)
    correct = int((removed.purity <= baseline / (1.0 + baseline)).sum())
    return {
        "n_removed": int(remove.sum()),
        "baseline_iou": float(baseline),
        "adapted_iou": float(adapted),
        "delta_iou": float(adapted - baseline),
        "cleared_fp": cleared,
        "lost_tp": lost,
        "corrected_to_harmed": float(cleared / max(lost, 1.0)),
        "fp_mass_captured": float(cleared / max(fp, 1.0)),
        "tp_mass_lost": float(lost / max(tp, 1.0)),
        "rer": float(((fp + fn) - (fp - cleared + fn + lost)) / max(fp + fn, 1.0)),
        "removal_precision": float(correct / max(int(remove.sum()), 1)),
    }


def train_one(
    train_frame: pd.DataFrame,
    folds: dict[str, FoldArrays],
    footprints: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    groups: Sequence[str],
    crop: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    terrain_stats: tuple[np.ndarray, np.ndarray],
    weight_reference: float,
    weight_cap: float,
    scalars: np.ndarray | None,
    seed: int,
    workers: int,
) -> PurityNet:
    torch.manual_seed(seed)
    dataset = ObjectCropDataset(
        train_frame,
        folds,
        footprints,
        crop=crop,
        groups=groups,
        terrain_mean=terrain_stats[0],
        terrain_scale=terrain_stats[1],
        optical_scale=1.0,
        weight_reference=weight_reference,
        weight_cap=weight_cap,
        scalars=scalars,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    sample = dataset[0]
    model = PurityNet(
        int(sample["crop"].shape[0]),
        n_scalars=0 if scalars is None else int(scalars.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=max(epochs * len(loader), 1)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    model.train()
    for epoch in range(epochs):
        running = 0.0
        seen = 0
        for batch in loader:
            crop_tensor = batch["crop"].to(device, non_blocking=True)
            purity = batch["purity"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            scalar_tensor = (
                batch["scalars"].to(device, non_blocking=True)
                if "scalars" in batch
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                prediction = model(crop_tensor, scalar_tensor)
                loss = (weight * (prediction - purity) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            running += float(loss.item()) * purity.numel()
            seen += purity.numel()
        print(
            f"    epoch {epoch + 1:02d}/{epochs} weighted_mse={running / max(seen, 1):.5f}",
            flush=True,
        )
    return model


@torch.no_grad()
def predict(
    model: PurityNet,
    frame: pd.DataFrame,
    folds: dict[str, FoldArrays],
    footprints: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    groups: Sequence[str],
    crop: int,
    batch_size: int,
    device: torch.device,
    terrain_stats: tuple[np.ndarray, np.ndarray],
    weight_reference: float,
    weight_cap: float,
    scalars: np.ndarray | None,
    workers: int,
) -> np.ndarray:
    dataset = ObjectCropDataset(
        frame,
        folds,
        footprints,
        crop=crop,
        groups=groups,
        terrain_mean=terrain_stats[0],
        terrain_scale=terrain_stats[1],
        optical_scale=1.0,
        weight_reference=weight_reference,
        weight_cap=weight_cap,
        scalars=scalars,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    model.eval()
    output = np.full(len(frame), np.nan, dtype=float)
    for batch in loader:
        crop_tensor = batch["crop"].to(device, non_blocking=True)
        scalar_tensor = (
            batch["scalars"].to(device, non_blocking=True)
            if "scalars" in batch
            else None
        )
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            prediction = model(crop_tensor, scalar_tensor)
        output[batch["row"].numpy()] = prediction.float().cpu().numpy()
    return output


def terrain_statistics(folds: dict[str, FoldArrays], limit: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Channel statistics from a bounded sample of training-side patches."""
    stack = []
    for fold in folds.values():
        take = min(limit, len(fold.terrain))
        stack.append(fold.terrain[:take].astype(np.float32))
    values = np.concatenate(stack)
    mean = values.mean(axis=(0, 2, 3))
    scale = values.std(axis=(0, 2, 3))
    scale[scale < 1e-3] = 1.0
    return mean, scale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--components",
        type=Path,
        help="component table; defaults to the table matching --condition",
    )
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1",
    )
    parser.add_argument(
        "--no-scalars",
        action="store_true",
        help="drop the whole-body summary branch and use crops alone",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_purity_cnn_v1",
    )
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--condition", default="aligned", choices=("aligned", "zero", "shift32", "roll64", "donor"))
    parser.add_argument("--groups", nargs="+", default=list(CHANNEL_GROUPS))
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--threshold-event-fraction", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    components_path = args.components or components_path_for(
        args.condition, args.diagnostic_root.resolve()
    )
    components = pd.read_csv(components_path)
    thresholds = {}
    for fold_id in args.folds:
        receipt = json.loads(
            (args.cache_dir / f"{fold_id}_oof_cache_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        thresholds[fold_id] = float(receipt["threshold"])
    components = components[components.fold_id.isin(args.folds)].copy()
    components["threshold"] = components.fold_id.map(thresholds)
    components = components.reset_index(drop=True)

    print(f"[load] {len(components)} components, condition={args.condition}", flush=True)
    folds = {
        fold_id: FoldArrays(args.cache_dir, fold_id, args.condition, args.seed)
        for fold_id in args.folds
    }
    stats = terrain_statistics(folds)
    footprints = precompute_footprints(components, folds, thresholds)
    scalar_matrix = (
        None
        if args.no_scalars
        else np.nan_to_num(
            components[list(SCALAR_FEATURES)].to_numpy(dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )
    weight_reference = float(components.area_px.median())
    weight_cap = float(
        np.quantile(components.area_px.to_numpy(dtype=float), 0.99) / weight_reference
    )
    print(
        f"[load] footprints={len(footprints)} weight_reference={weight_reference:.1f} "
        f"weight_cap={weight_cap:.1f}",
        flush=True,
    )

    # Pixel totals come from the aligned run: Terrain interventions never change the
    # frozen visual prediction, so the corpus denominators are shared across conditions.
    totals_path = components_path_for("aligned", args.diagnostic_root.resolve()).parent / "summary.json"
    totals = json.loads(totals_path.read_text(encoding="utf-8"))["pixel_totals"]
    tp, fp, fn = (
        float(totals["tp_pixels"]),
        float(totals["fp_pixels"]),
        float(totals["fn_pixels"]),
    )

    event_ids = components.canonical_event_id.to_numpy()
    unique_events = np.unique(event_ids)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique_events)
    event_folds = np.array_split(shuffled, int(min(args.outer_splits, len(shuffled))))

    purity_hat = np.full(len(components), np.nan, dtype=float)
    applied = np.full(len(components), np.nan, dtype=float)
    receipts: list[dict[str, Any]] = []

    for fold_index, held_out in enumerate(event_folds):
        test_mask = np.isin(event_ids, held_out)
        train_events = np.setdiff1d(unique_events, held_out)
        n_threshold = max(1, int(round(len(train_events) * args.threshold_event_fraction)))
        threshold_events = rng.choice(train_events, size=n_threshold, replace=False)
        fit_events = np.setdiff1d(train_events, threshold_events)
        fit_mask = np.isin(event_ids, fit_events)
        threshold_mask = np.isin(event_ids, threshold_events)
        print(
            f"[outer {fold_index}] fit_events={len(fit_events)} "
            f"threshold_events={len(threshold_events)} test_events={len(held_out)} "
            f"fit_rows={int(fit_mask.sum())} test_rows={int(test_mask.sum())}",
            flush=True,
        )
        # Standardize the summaries with fit-event statistics only.
        scalar_sets: dict[str, np.ndarray | None] = {
            "fit": None,
            "threshold": None,
            "test": None,
        }
        if scalar_matrix is not None:
            centre = scalar_matrix[fit_mask].mean(axis=0)
            spread = scalar_matrix[fit_mask].std(axis=0)
            spread[spread < 1e-6] = 1.0
            for name, mask in (
                ("fit", fit_mask),
                ("threshold", threshold_mask),
                ("test", test_mask),
            ):
                scalar_sets[name] = np.clip(
                    (scalar_matrix[mask] - centre) / spread, -8.0, 8.0
                )
        model = train_one(
            components[fit_mask],
            folds,
            footprints,
            groups=args.groups,
            crop=args.crop,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            terrain_stats=stats,
            weight_reference=weight_reference,
            weight_cap=weight_cap,
            scalars=scalar_sets["fit"],
            seed=args.seed + fold_index,
            workers=args.workers,
        )
        threshold_frame = components[threshold_mask]
        threshold_prediction = predict(
            model,
            threshold_frame,
            folds,
            footprints,
            groups=args.groups,
            crop=args.crop,
            batch_size=args.batch_size,
            device=device,
            terrain_stats=stats,
            weight_reference=weight_reference,
            weight_cap=weight_cap,
            scalars=scalar_sets["threshold"],
            workers=args.workers,
        )
        threshold_gain = expected_gain(
            threshold_prediction,
            threshold_frame.area_px.to_numpy(dtype=float),
            tp=tp,
            fp=fp,
            fn=fn,
        )
        cut, inner_delta = best_cut(
            threshold_gain,
            threshold_frame.false_px.to_numpy(dtype=float),
            threshold_frame.intersection_px.to_numpy(dtype=float),
            tp=tp,
            fp=fp,
            fn=fn,
        )
        test_frame = components[test_mask]
        test_prediction = predict(
            model,
            test_frame,
            folds,
            footprints,
            groups=args.groups,
            crop=args.crop,
            batch_size=args.batch_size,
            device=device,
            terrain_stats=stats,
            weight_reference=weight_reference,
            weight_cap=weight_cap,
            scalars=scalar_sets["test"],
            workers=args.workers,
        )
        purity_hat[test_mask] = test_prediction
        applied[test_mask] = cut
        receipts.append(
            {
                "outer_fold": fold_index,
                "n_fit_events": int(len(fit_events)),
                "n_threshold_events": int(len(threshold_events)),
                "n_test_events": int(len(held_out)),
                "selected_threshold": cut,
                "inner_delta_iou": inner_delta,
            }
        )

    gain = expected_gain(
        purity_hat, components.area_px.to_numpy(dtype=float), tp=tp, fp=fp, fn=fn
    )
    decision = np.isfinite(gain) & np.isfinite(applied) & (gain >= applied)
    outcome = evaluate_removals(components, decision, tp=tp, fp=fp, fn=fn)
    finite = np.isfinite(purity_hat)
    outcome["purity_correlation"] = float(
        np.corrcoef(purity_hat[finite], components.purity.to_numpy()[finite])[0, 1]
    )
    outcome["purity_mae"] = float(
        np.mean(np.abs(purity_hat[finite] - components.purity.to_numpy()[finite]))
    )

    tag = args.tag or f"{args.condition}_{'+'.join(args.groups)}"
    decided = components.assign(
        purity_hat=purity_hat, expected_gain=gain, gate_threshold=applied, removed=decision
    )
    decided.to_csv(outdir / f"decisions_{tag}.csv", index=False)
    summary = {
        "schema_version": "pild_object_purity_cnn.v1",
        "evidence_status": "development: event-grouped cross-validation on already-opened folds",
        "condition": args.condition,
        "channel_groups": list(args.groups),
        "crop": int(args.crop),
        "epochs": int(args.epochs),
        "outcome": outcome,
        "fold_receipts": receipts,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (outdir / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )
    print(f"\n=== G2 {tag} ===")
    print(
        f"  purity corr={outcome['purity_correlation']:.4f} mae={outcome['purity_mae']:.4f}"
    )
    print(
        f"  removed={outcome['n_removed']} dIoU={outcome['delta_iou']:+.5f} "
        f"RER={outcome['rer']:+.2%} c/h={outcome['corrected_to_harmed']:.1f} "
        f"FPmass={outcome['fp_mass_captured']:.1%} TPloss={outcome['tp_mass_lost']:.2%} "
        f"precision={outcome['removal_precision']:.3f}"
    )
    print(f"  artifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
