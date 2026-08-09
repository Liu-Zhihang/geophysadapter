#!/usr/bin/env python3
"""Audit whether L4S modern-backbone adapter gains depend on aligned terrain.

The script is read-only with respect to trained checkpoints. For each frozen
adapter it evaluates the original terrain, zero terrain, a sample-shifted
terrain control, and a spatially rolled terrain control. It also reports gate
activation on visual errors versus correct pixels and net rescued pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analyze_revision_dlr_paired import json_safe, paired_stats
from run_support_adapter_smp import SmpSupportResidualAdapter
from run_support_adapter_unet import H5SupportDataset, channel_indices_by_group, split_path


ROOT = Path("${PILD_ROOT}")
CACHE = ROOT / "processed/hybrid_pinn/landslide4sense_clean_multispectral_v1"
DEFAULT_OUTDIR = ROOT / "experiments/revision2026/l4s_modern_terrain_attribution_20260715"
DEFAULT_SPECS = (
    "DeepLabV3+|experiments/revision2026/support_adapter_modern_e40/l4s_support_adapter_deeplabv3plus_alpha3_frozen_confirm_e40",
    "U-Net++|experiments/revision2026/support_adapter_modern_unetplusplus/l4s_support_adapter_unetplusplus_alpha3_frozen_e20",
    "FPN|experiments/revision2026/support_adapter_modern_fpn/l4s_support_adapter_fpn_alpha3_frozen_e20",
    "DeepLabV3+ ImageNet|experiments/revision2026/support_adapter_modern_imagenet/l4s_support_adapter_deeplabv3plus_alpha3_frozen_e20",
)
VARIANTS = (
    "visual_anchor",
    "visual_adapter_threshold",
    "terrain_true",
    "terrain_zero",
    "terrain_shift",
    "terrain_roll",
)
COMPARISONS = (
    ("terrain_true", "visual_anchor"),
    ("terrain_true", "visual_adapter_threshold"),
    ("terrain_true", "terrain_zero"),
    ("terrain_true", "terrain_shift"),
    ("terrain_true", "terrain_roll"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--spec", action="append", default=[], metavar="LABEL|RUN_DIR")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roll-y", type=int, default=32)
    parser.add_argument("--roll-x", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutation-iters", type=int, default=50000)
    parser.add_argument("--max-samples", type=int, default=0, help="Smoke-test limit; 0 uses the full split.")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="Smoke-test limit; 0 requires five checkpoints.")
    parser.add_argument(
        "--require-frozen-visual-state",
        action="store_true",
        help="Require the adapter visual state to be tensor-identical to its matched observation checkpoint.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


class CounterfactualTerrainDataset(Dataset):
    def __init__(self, base: H5SupportDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        current = self.base[index]
        donor = self.base[(index + 1) % len(self.base)]
        return {
            **current,
            "terrain_shift": donor["terrain"],
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts(prediction: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = prediction.float()
    truth = (target >= 0.5).float()
    tp = (pred * truth).sum(dim=(1, 2, 3)).cpu().numpy()
    fp = (pred * (1.0 - truth)).sum(dim=(1, 2, 3)).cpu().numpy()
    fn = ((1.0 - pred) * truth).sum(dim=(1, 2, 3)).cpu().numpy()
    return tp, fp, fn


def adapter_forward(
    model: SmpSupportResidualAdapter,
    visual_logits: torch.Tensor,
    uncertainty: torch.Tensor,
    terrain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    terrain_features = model.terrain_encoder(terrain)
    fused = torch.cat((terrain_features, uncertainty), dim=1)
    residual = model.alpha_max * torch.tanh(model.residual_head(fused))
    gate = torch.sigmoid(model.gate_head(fused)) * uncertainty
    return visual_logits + gate * residual, gate, residual


def resolve_recorded_path(raw_path: str | Path, root: Path, checkpoint_path: Path) -> Path:
    """Resolve absolute and repository-relative paths recorded in checkpoints."""
    recorded = Path(raw_path).expanduser()
    candidates = [recorded] if recorded.is_absolute() else [
        Path.cwd() / recorded,
        root / recorded,
        root.parent / recorded,
        checkpoint_path.parent / recorded,
    ]
    existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            existing.append(resolved)
    if not existing:
        attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(f"cannot resolve recorded path {recorded!s}; attempted: {attempted}")
    if len(existing) > 1:
        raise RuntimeError(f"ambiguous recorded path {recorded!s}: {[str(path) for path in existing]}")
    return existing[0]


def evaluate_checkpoint(
    label: str,
    checkpoint_path: Path,
    dataset: CounterfactualTerrainDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    roll: tuple[int, int],
    require_frozen_visual_state: bool,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(checkpoint["seed"])
    architecture = str(checkpoint["architecture"])
    encoder = str(checkpoint["encoder"])
    obs_names = [str(item) for item in checkpoint["obs_channel_names"]]
    terrain_names = [str(item) for item in checkpoint["terrain_channel_names"]]
    model = SmpSupportResidualAdapter(
        architecture,
        encoder,
        None,
        len(obs_names),
        len(terrain_names),
        alpha_max=float(checkpoint["alpha_max"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    adapter_threshold = float(checkpoint["threshold"])
    baseline_path = resolve_recorded_path(checkpoint["baseline_ckpt"], root, checkpoint_path)
    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    if int(baseline_checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(f"baseline seed mismatch: adapter={seed} baseline={baseline_checkpoint.get('seed')}")
    if str(baseline_checkpoint.get("architecture")) != architecture:
        raise RuntimeError(
            f"baseline architecture mismatch: adapter={architecture} "
            f"baseline={baseline_checkpoint.get('architecture')}"
        )
    if str(baseline_checkpoint.get("encoder")) != encoder:
        raise RuntimeError(
            f"baseline encoder mismatch: adapter={encoder} baseline={baseline_checkpoint.get('encoder')}"
        )
    if [str(item) for item in baseline_checkpoint.get("obs_channel_names", [])] != obs_names:
        raise RuntimeError(f"baseline observation-channel mismatch: {checkpoint_path}")
    visual_threshold = float(baseline_checkpoint["threshold"])
    adapter_visual_state = checkpoint.get("visual_state_dict")
    baseline_visual_state = baseline_checkpoint.get("visual_state_dict")
    visual_keys_identical = (
        isinstance(adapter_visual_state, dict)
        and isinstance(baseline_visual_state, dict)
        and set(adapter_visual_state) == set(baseline_visual_state)
    )
    visual_state_identical = bool(visual_keys_identical)
    visual_state_max_abs_diff = 0.0
    visual_state_worst_key = ""
    if visual_keys_identical:
        for key in adapter_visual_state:
            left = adapter_visual_state[key]
            right = baseline_visual_state[key]
            if left.shape != right.shape or left.dtype != right.dtype:
                visual_state_identical = False
                visual_state_max_abs_diff = math.inf
                visual_state_worst_key = key
                break
            if not torch.equal(left, right):
                visual_state_identical = False
                delta = float((left.float() - right.float()).abs().max().item())
                if delta >= visual_state_max_abs_diff:
                    visual_state_max_abs_diff = delta
                    visual_state_worst_key = key
    if require_frozen_visual_state:
        if checkpoint.get("freeze_visual_state") is not True:
            raise RuntimeError(f"checkpoint lacks freeze_visual_state=true: {checkpoint_path}")
        if checkpoint.get("visual_state_sha256_initial") != checkpoint.get("visual_state_sha256_final"):
            raise RuntimeError(f"checkpoint initial/final visual hash mismatch: {checkpoint_path}")
        if not visual_state_identical:
            raise RuntimeError(
                f"adapter visual state differs from matched baseline: {checkpoint_path}; "
                f"worst={visual_state_worst_key} max_abs_diff={visual_state_max_abs_diff}"
            )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            observation = batch["obs"].to(device, non_blocking=True)
            terrain_true = batch["terrain"].to(device, non_blocking=True)
            terrain_shift = batch["terrain_shift"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            visual_logits = model.visual(observation)
            if visual_logits.shape[-2:] != observation.shape[-2:]:
                visual_logits = torch.nn.functional.interpolate(
                    visual_logits, size=observation.shape[-2:], mode="bilinear", align_corners=False
                )
            visual_probability = torch.sigmoid(visual_logits)
            uncertainty = 1.0 - torch.abs(2.0 * visual_probability - 1.0)
            terrain_inputs = {
                "terrain_true": terrain_true,
                "terrain_zero": torch.zeros_like(terrain_true),
                "terrain_shift": terrain_shift,
                "terrain_roll": torch.roll(terrain_true, shifts=roll, dims=(-2, -1)),
            }
            logits_by_variant: dict[str, torch.Tensor] = {
                "visual_anchor": visual_logits,
                "visual_adapter_threshold": visual_logits,
            }
            gates: dict[str, torch.Tensor] = {}
            residuals: dict[str, torch.Tensor] = {}
            for variant, terrain in terrain_inputs.items():
                logits, gate, residual = adapter_forward(model, visual_logits, uncertainty, terrain)
                logits_by_variant[variant] = logits
                gates[variant] = gate
                residuals[variant] = residual
            predictions = {
                variant: (torch.sigmoid(logits) >= (visual_threshold if variant == "visual_anchor" else adapter_threshold))
                for variant, logits in logits_by_variant.items()
            }
            visual_prediction = predictions["visual_anchor"]
            visual_error = visual_prediction != (target >= 0.5)
            true_prediction = predictions["terrain_true"]
            rescued = visual_error & (true_prediction == (target >= 0.5))
            harmed = (~visual_error) & (true_prediction != (target >= 0.5))
            for variant in VARIANTS:
                tp, fp, fn = counts(predictions[variant], target)
                denominator = tp + fp + fn + 1e-7
                for index, sample_id in enumerate(batch["sample_id"]):
                    row = {
                        "backbone": label,
                        "architecture": architecture,
                        "encoder": encoder,
                        "seed": seed,
                        "split": "test",
                        "sample_id": str(sample_id),
                        "variant": variant,
                        "tp": float(tp[index]),
                        "fp": float(fp[index]),
                        "fn": float(fn[index]),
                        "iou": float(tp[index] / denominator[index]),
                        "threshold": visual_threshold if variant == "visual_anchor" else adapter_threshold,
                        "gate_mean": math.nan,
                        "gate_on_visual_error": math.nan,
                        "gate_on_visual_correct": math.nan,
                        "residual_abs_mean": math.nan,
                        "rescued_pixels": 0,
                        "harmed_pixels": 0,
                        "net_rescued_pixels": 0,
                    }
                    if variant in gates:
                        gate = gates[variant][index]
                        error_mask = visual_error[index]
                        correct_mask = ~error_mask
                        row["gate_mean"] = float(gate.mean().item())
                        row["gate_on_visual_error"] = float(gate[error_mask].mean().item()) if error_mask.any() else math.nan
                        row["gate_on_visual_correct"] = float(gate[correct_mask].mean().item()) if correct_mask.any() else math.nan
                        row["residual_abs_mean"] = float(residuals[variant][index].abs().mean().item())
                    if variant == "terrain_true":
                        rescued_count = int(rescued[index].sum().item())
                        harmed_count = int(harmed[index].sum().item())
                        row["rescued_pixels"] = rescued_count
                        row["harmed_pixels"] = harmed_count
                        row["net_rescued_pixels"] = rescued_count - harmed_count
                    rows.append(row)
    provenance = {
        "checkpoint": str(checkpoint_path),
        "baseline_checkpoint_recorded": str(checkpoint["baseline_ckpt"]),
        "baseline_checkpoint_resolved": str(baseline_path),
        "seed": seed,
        "architecture": architecture,
        "encoder": encoder,
        "encoder_weights_recorded": checkpoint.get("encoder_weights"),
        "obs_channel_names": obs_names,
        "terrain_channel_names": terrain_names,
        "adapter_threshold": adapter_threshold,
        "visual_threshold": visual_threshold,
        "best_epoch": checkpoint.get("best_epoch"),
        "visual_train_scope": checkpoint.get("visual_train_scope"),
        "freeze_visual_state": checkpoint.get("freeze_visual_state"),
        "visual_state_sha256_initial": checkpoint.get("visual_state_sha256_initial"),
        "visual_state_sha256_final": checkpoint.get("visual_state_sha256_final"),
        "visual_state_identical_to_baseline": visual_state_identical,
        "visual_state_max_abs_diff": visual_state_max_abs_diff,
        "visual_state_worst_key": visual_state_worst_key,
    }
    return rows, provenance


def aggregate(rows: list[dict[str, Any]], unit: str) -> dict[tuple[str, str, str], dict[str, float]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if unit == "sample":
            key = (row["backbone"], row["variant"], row["sample_id"])
        elif unit == "seed":
            key = (row["backbone"], row["variant"], str(row["seed"]))
        else:
            raise ValueError(unit)
        buckets[key].append(row)
    output: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, members in buckets.items():
        if unit == "sample":
            output[key] = {
                metric: float(np.mean([float(row[metric]) for row in members]))
                for metric in ("iou", "tp", "fp", "fn")
            }
        else:
            tp = sum(float(row["tp"]) for row in members)
            fp = sum(float(row["fp"]) for row in members)
            fn = sum(float(row["fn"]) for row in members)
            output[key] = {"tp": tp, "fp": fp, "fn": fn, "iou": tp / (tp + fp + fn + 1e-7)}
    return output


def compare(
    values: dict[tuple[str, str, str], dict[str, float]],
    backbone: str,
    variant_a: str,
    variant_b: str,
    bootstrap: int,
    permutation_iters: int,
) -> dict[str, Any]:
    a = {key[2]: value for key, value in values.items() if key[0] == backbone and key[1] == variant_a}
    b = {key[2]: value for key, value in values.items() if key[0] == backbone and key[1] == variant_b}
    if set(a) != set(b) or not a:
        raise ValueError(f"unpaired comparison {backbone}: {variant_a} vs {variant_b}")
    shared = sorted(a)
    return paired_stats(
        [a[key]["iou"] - b[key]["iou"] for key in shared],
        [a[key]["tp"] for key in shared],
        [a[key]["fp"] for key in shared],
        [a[key]["fn"] for key in shared],
        [b[key]["tp"] for key in shared],
        [b[key]["fp"] for key in shared],
        [b[key]["fn"] for key in shared],
        bootstrap,
        permutation_iters,
    )


def fmt(value: Any, spec: str = "+.4f") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return format(number, spec) if math.isfinite(number) else "NA"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in input order."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def apply_holm_and_decisions(
    comparisons: dict[str, Any], mechanism: list[dict[str, Any]], alpha: float = 0.05
) -> tuple[list[dict[str, Any]], int]:
    family: list[tuple[str, str, dict[str, Any]]] = []
    for backbone, contrasts in comparisons.items():
        for contrast, units in contrasts.items():
            family.append((backbone, contrast, units["sample"]))
    adjusted = holm_adjust([float(stats["permutation_p"]) for _, _, stats in family])
    for (_, _, stats), value in zip(family, adjusted):
        stats["holm_permutation_p_family"] = value

    mechanism_by_backbone = {str(row["backbone"]): row for row in mechanism}
    decisions = []
    for backbone, contrasts in comparisons.items():
        control_passes = []
        seed_directions = []
        for units in contrasts.values():
            sample = units["sample"]
            ci = sample["mean_delta_ci95"]
            control_passes.append(
                float(sample["mean_delta"]) > 0
                and float(ci[0]) > 0
                and float(sample["holm_permutation_p_family"]) <= alpha
            )
            seed_directions.append(float(units["seed"]["mean_delta"]) > 0)
        diag = mechanism_by_backbone[backbone]
        gate_ci = diag["gate_error_minus_correct_ci95"]
        rescue_ci = diag["net_error_reduction_ci95"]
        gate_pass = (
            float(diag["gate_error_minus_correct"]) > 0
            and float(gate_ci[0]) > 0
        )
        rescue_pass = (
            float(diag["net_error_reduction_fraction"]) > 0
            and float(rescue_ci[0]) > 0
        )
        controls_pass = all(control_passes)
        decisions.append(
            {
                "backbone": backbone,
                "n_sample_contrasts": len(control_passes),
                "holm_family_size": len(family),
                "all_sample_controls_pass": controls_pass,
                "all_seed_mean_effects_positive": all(seed_directions),
                "minimum_seed_positive_rate": min(
                    float(units["seed"]["positive_rate"])
                    for units in contrasts.values()
                ),
                "gate_error_gt_correct_ci_positive": gate_pass,
                "net_error_reduction_ci_positive": rescue_pass,
                "terrain_attribution_pass": controls_pass and gate_pass and rescue_pass,
                "seed_inference_note": "n=5; two-sided exact sign-flip p-value floor is 0.0625",
            }
        )
    return decisions, len(family)


def ratio_bootstrap_ci(
    numerators: np.ndarray,
    denominators: np.ndarray,
    iterations: int,
    seed: int,
) -> list[float]:
    if numerators.shape != denominators.shape or numerators.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    size = numerators.size
    for iteration in range(iterations):
        indices = rng.integers(0, size, size=size)
        denominator = float(denominators[indices].sum())
        values[iteration] = float(numerators[indices].sum()) / denominator if denominator > 0 else math.nan
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [math.nan, math.nan]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def mean_bootstrap_ci(
    values: np.ndarray,
    iterations: int,
    seed: int,
) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        indices = rng.integers(0, finite.size, size=finite.size)
        estimates[iteration] = float(finite[indices].mean())
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def build_mechanism_summary(
    rows: list[dict[str, Any]],
    labels: list[str],
    pixels_per_sample: int,
    bootstrap: int,
) -> list[dict[str, Any]]:
    mechanism = []
    for label_index, label in enumerate(labels):
        members = [row for row in rows if row["backbone"] == label and row["variant"] == "terrain_true"]
        visual = {
            (str(row["seed"]), str(row["sample_id"])): row
            for row in rows
            if row["backbone"] == label and row["variant"] == "visual_anchor"
        }
        net_rescue = np.asarray([float(row["net_rescued_pixels"]) for row in members], dtype=float)
        per_sample: dict[str, dict[str, float]] = defaultdict(
            lambda: {"rescued": 0.0, "harmed": 0.0, "visual_errors": 0.0, "visual_correct": 0.0}
        )
        gate_differences_by_sample: dict[str, list[float]] = defaultdict(list)
        for row in members:
            key = (str(row["seed"]), str(row["sample_id"]))
            visual_row = visual.get(key)
            if visual_row is None:
                raise ValueError(f"missing visual anchor for {label} seed/sample={key}")
            visual_errors = float(visual_row["fp"]) + float(visual_row["fn"])
            bucket = per_sample[str(row["sample_id"])]
            bucket["rescued"] += float(row["rescued_pixels"])
            bucket["harmed"] += float(row["harmed_pixels"])
            bucket["visual_errors"] += visual_errors
            bucket["visual_correct"] += float(pixels_per_sample) - visual_errors
            gate_difference = (
                float(row["gate_on_visual_error"])
                - float(row["gate_on_visual_correct"])
            )
            if math.isfinite(gate_difference):
                gate_differences_by_sample[str(row["sample_id"])].append(gate_difference)
        rescued = np.asarray([value["rescued"] for value in per_sample.values()], dtype=float)
        harmed = np.asarray([value["harmed"] for value in per_sample.values()], dtype=float)
        visual_errors = np.asarray([value["visual_errors"] for value in per_sample.values()], dtype=float)
        visual_correct = np.asarray([value["visual_correct"] for value in per_sample.values()], dtype=float)
        rescued_total = float(rescued.sum())
        harmed_total = float(harmed.sum())
        visual_error_total = float(visual_errors.sum())
        visual_correct_total = float(visual_correct.sum())
        net_numerator = rescued - harmed
        gate_differences = np.asarray(
            [
                float(np.mean(gate_differences_by_sample[sample_id]))
                for sample_id in sorted(gate_differences_by_sample)
                if gate_differences_by_sample[sample_id]
            ],
            dtype=float,
        )
        mechanism.append(
            {
                "backbone": label,
                "n_seed_samples": len(members),
                "n_unique_samples": len(per_sample),
                "mean_gate": float(np.nanmean([float(row["gate_mean"]) for row in members])),
                "mean_gate_on_visual_error": float(
                    np.nanmean([float(row["gate_on_visual_error"]) for row in members])
                ),
                "mean_gate_on_visual_correct": float(
                    np.nanmean([float(row["gate_on_visual_correct"]) for row in members])
                ),
                "gate_error_minus_correct": float(np.mean(gate_differences)),
                "gate_error_minus_correct_ci95": mean_bootstrap_ci(
                    gate_differences,
                    bootstrap,
                    20260725 + label_index,
                ),
                "visual_error_pixels": int(visual_error_total),
                "rescued_pixels": int(rescued_total),
                "harmed_pixels": int(harmed_total),
                "net_rescued_pixels": int(rescued_total - harmed_total),
                "rescued_fraction_of_visual_errors": rescued_total / visual_error_total,
                "harmed_fraction_of_visual_correct": harmed_total / visual_correct_total,
                "net_error_reduction_fraction": (rescued_total - harmed_total) / visual_error_total,
                "net_error_reduction_ci95": ratio_bootstrap_ci(
                    net_numerator,
                    visual_errors,
                    bootstrap,
                    20260715 + label_index,
                ),
                "positive_net_rescue_fraction": float(np.mean(net_rescue > 0)),
            }
        )
    return mechanism


def main() -> int:
    args = parse_args()
    specs = []
    for raw in args.spec or DEFAULT_SPECS:
        label, run_dir = raw.split("|", 1)
        path = Path(run_dir)
        if not path.is_absolute():
            path = args.root / path
        specs.append((label, path))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    obs_idx, _, _ = channel_indices_by_group(args.cache_dir, {"observation"})
    terrain_idx, _, _ = channel_indices_by_group(args.cache_dir, {"terrain"})
    base_dataset = H5SupportDataset(
        split_path(args.cache_dir, args.split),
        obs_idx,
        terrain_idx,
        max_samples=args.max_samples,
    )
    dataset = CounterfactualTerrainDataset(base_dataset)
    rows: list[dict[str, Any]] = []
    provenance = []
    started = time.time()
    for label, run_dir in specs:
        checkpoints = sorted((run_dir / "checkpoints").glob("*_adapter_seed*.pt"))
        if args.max_checkpoints > 0:
            checkpoints = checkpoints[: args.max_checkpoints]
        elif len(checkpoints) != 5:
            raise SystemExit(f"[FATAL] expected five adapter checkpoints in {run_dir}, found {len(checkpoints)}")
        if not checkpoints:
            raise SystemExit(f"[FATAL] no adapter checkpoints in {run_dir}")
        for checkpoint in checkpoints:
            print(f"[eval] {label} {checkpoint.name}", flush=True)
            checkpoint_rows, checkpoint_provenance = evaluate_checkpoint(
                label,
                checkpoint,
                dataset,
                args.batch_size,
                args.workers,
                device,
                (args.roll_y, args.roll_x),
                args.require_frozen_visual_state,
                args.root,
            )
            for row in checkpoint_rows:
                row["split"] = args.split
            rows.extend(checkpoint_rows)
            provenance.append(checkpoint_provenance)
    sample_values = aggregate(rows, "sample")
    seed_values = aggregate(rows, "seed")
    comparisons: dict[str, Any] = {}
    for label, _ in specs:
        comparisons[label] = {}
        for variant_a, variant_b in COMPARISONS:
            name = f"{variant_a}_vs_{variant_b}"
            comparisons[label][name] = {
                "sample": compare(sample_values, label, variant_a, variant_b, args.bootstrap, args.permutation_iters),
                "seed": compare(seed_values, label, variant_a, variant_b, args.bootstrap, args.permutation_iters),
            }
    pixels_per_sample = int(base_dataset[0]["mask"].numel())
    mechanism = build_mechanism_summary(
        rows,
        [label for label, _ in specs],
        pixels_per_sample,
        args.bootstrap,
    )
    decisions, holm_family_size = apply_holm_and_decisions(comparisons, mechanism)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "per_seed_sample_counterfactuals.csv", rows)
    write_csv(args.outdir / "mechanism_summary.csv", mechanism)
    write_csv(args.outdir / "decision_summary.csv", decisions)
    summary = json_safe(
        {
            "timestamp": int(time.time()),
            "elapsed_seconds": time.time() - started,
            "split": args.split,
            "n_unique_samples": len(base_dataset),
            "specs": [{"label": label, "run_dir": str(path)} for label, path in specs],
            "provenance": provenance,
            "require_frozen_visual_state": args.require_frozen_visual_state,
            "comparisons": comparisons,
            "mechanism": mechanism,
            "decisions": decisions,
            "holm_family_size": holm_family_size,
            "claim_rule": (
                "A terrain-specific claim requires terrain_true to beat both visual thresholds, terrain_zero, "
                "terrain_shift, and terrain_roll. An adaptive-takeover claim additionally requires "
                "the independent-sample bootstrap lower bound for gate(error-correct) and net error "
                "reduction to exceed zero."
            ),
        }
    )
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# L4S modern-backbone terrain attribution audit",
        "",
        f"- split: `{args.split}`; unique samples: `{len(base_dataset)}`; five checkpoints per backbone",
        "- counterfactuals reuse each trained checkpoint; no test-set tuning or retraining is performed",
        "",
        "## Paired terrain attribution",
        "",
        "| backbone | unit | contrast | n | mean delta | 95% CI | permutation p | Holm p (sample family) | Wilcoxon p |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for label, _ in specs:
        for name, units in comparisons[label].items():
            for unit in ("sample", "seed"):
                stats = units[unit]
                ci = stats["mean_delta_ci95"]
                lines.append(
                    f"| {label} | {unit} | {name} | {stats['n']} | {fmt(stats['mean_delta'])} | "
                    f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(stats['permutation_p'], '.4g')} | "
                    f"{fmt(stats.get('holm_permutation_p_family'), '.4g') if unit == 'sample' else 'NA'} | "
                    f"{fmt(stats['wilcoxon_p'], '.4g')} |"
                )
    lines.extend(
        [
            "",
            "## Gate and rescue diagnostics",
            "",
            "| backbone | gate(error-correct) [95% CI] | rescued / visual errors | harmed / visual correct | net error reduction [95% CI] | net rescued pixels |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mechanism:
        lines.append(
            f"| {row['backbone']} | {fmt(row['gate_error_minus_correct'])} "
            f"[{fmt(row['gate_error_minus_correct_ci95'][0])}, "
            f"{fmt(row['gate_error_minus_correct_ci95'][1])}] | "
            f"{fmt(row['rescued_fraction_of_visual_errors'], '.2%')} | "
            f"{fmt(row['harmed_fraction_of_visual_correct'], '.2%')} | "
            f"{fmt(row['net_error_reduction_fraction'], '.2%')} "
            f"[{fmt(row['net_error_reduction_ci95'][0], '.2%')}, {fmt(row['net_error_reduction_ci95'][1], '.2%')}] | "
            f"{row['net_rescued_pixels']} |"
        )
    lines.extend(
        [
            "",
            "## Attribution gate",
            "",
            f"Holm correction is applied across all `{holm_family_size}` unique-sample contrasts in this report.",
            "With five seeds, the minimum non-zero two-sided exact sign-flip p-value is 0.0625; seed inference is therefore reported as directional reproducibility rather than a p<0.05 claim.",
            "",
            "| backbone | all controls pass | all five-seed mean contrasts positive | minimum individual-seed positive rate | gate CI > 0 | net-reduction CI > 0 | final terrain-attribution gate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in decisions:
        lines.append(
            f"| {row['backbone']} | {row['all_sample_controls_pass']} | "
            f"{row['all_seed_mean_effects_positive']} | "
            f"{fmt(row['minimum_seed_positive_rate'], '.0%')} | "
            f"{row['gate_error_gt_correct_ci_positive']} | "
            f"{row['net_error_reduction_ci_positive']} | {row['terrain_attribution_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "A terrain-specific claim requires `terrain_true` to beat both visual-threshold anchors and every zero/shift/roll control. "
            "A positive adapter-versus-visual delta alone is insufficient. Automatic takeover additionally requires "
            "the independent-sample 95% bootstrap lower bounds for both higher gate values on visual errors "
            "and net error reduction to exceed zero.",
        ]
    )
    (args.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.outdir / "DONE.json").write_text(
        json.dumps({"status": "complete", "timestamp": int(time.time())}, indent=2), encoding="utf-8"
    )
    print(f"[done] {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
