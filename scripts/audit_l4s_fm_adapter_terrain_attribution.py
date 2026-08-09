#!/usr/bin/env python3
"""Audit terrain attribution for L4S foundation-model support adapters.

Each trained adapter is evaluated with aligned terrain, zero terrain, terrain
from the next sample, and spatially rolled terrain. The script also compares
the adapter against its unchanged internal visual branch at both the matched
observation threshold and the adapter threshold. It refuses to run if the
adapter visual state differs from the matched observation checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_revision_dlr_paired import json_safe
from audit_l4s_modern_adapter_terrain_attribution import (
    COMPARISONS,
    VARIANTS,
    CounterfactualTerrainDataset,
    aggregate,
    apply_holm_and_decisions,
    build_mechanism_summary,
    compare,
    counts,
    fmt,
    write_csv,
)
from run_support_adapter_timmfm import TimmSupportResidualAdapter
from run_support_adapter_unet import H5SupportDataset, channel_indices_by_group, split_path


ROOT = Path("${PILD_ROOT}")
CACHE = ROOT / "processed/hybrid_pinn/landslide4sense_clean_multispectral_v1"
RUN_ROOT = ROOT / "experiments/revision2026/r3_11_backbone_sensitivity"
DEFAULT_OUTDIR = ROOT / "experiments/revision2026/l4s_fm_terrain_attribution_20260715"
DEFAULT_SPECS = (
    "DINOv2-S|l4s_r3_11_dinov2_small_adapter_alpha3_frozen_e20",
    "FCMAE-ConvNeXtV2-T|l4s_r3_11_fcmae_convnextv2_tiny_adapter_alpha3_frozen_e20",
    "Hiera-S MAE|l4s_r3_11_hiera_small_mae_adapter_alpha3_frozen_e20",
    "SatMAE-ViT-B|l4s_r3_11_satmae_vitbase_multispec_adapter_alpha3_frozen_e20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
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
        "--reuse-counterfactuals",
        type=Path,
        default=None,
        help="Recompute statistics from an existing per-seed counterfactual CSV without model inference.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def assert_identical_visual_state(
    adapter_state: dict[str, torch.Tensor], baseline_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if set(adapter_state) != set(baseline_state):
        missing = sorted(set(baseline_state) - set(adapter_state))[:10]
        extra = sorted(set(adapter_state) - set(baseline_state))[:10]
        raise RuntimeError(f"visual-state key mismatch: missing={missing}, extra={extra}")
    changed: list[tuple[str, float]] = []
    for key in adapter_state:
        left = adapter_state[key]
        right = baseline_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"visual-state tensor metadata mismatch: {key}")
        if not torch.equal(left, right):
            delta = float((left.float() - right.float()).abs().max().item())
            changed.append((key, delta))
    if changed:
        changed.sort(key=lambda item: item[1], reverse=True)
        raise RuntimeError(f"visual state changed; largest differences={changed[:5]}")
    adapter_hash = state_sha256(adapter_state)
    baseline_hash = state_sha256(baseline_state)
    if adapter_hash != baseline_hash:
        raise RuntimeError("visual-state SHA256 mismatch despite equal tensors")
    return {
        "n_visual_state_tensors": len(adapter_state),
        "visual_state_identical": True,
        "visual_state_sha256": adapter_hash,
    }


def adapter_forward(
    model: TimmSupportResidualAdapter,
    visual_logits: torch.Tensor,
    visual_features: torch.Tensor,
    uncertainty: torch.Tensor,
    terrain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correction, gate, residual = model.support_branch(
        visual_features,
        uncertainty,
        terrain,
    )
    if getattr(model, "center_physical_correction", False):
        reference, _, _ = model.support_branch(
            visual_features,
            uncertainty,
            torch.zeros_like(terrain),
        )
        correction = correction - reference
    return visual_logits + correction, gate, residual


def evaluate_checkpoint(
    label: str,
    checkpoint_path: Path,
    dataset: CounterfactualTerrainDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    roll: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(checkpoint["seed"])
    if checkpoint.get("mode") != "adapter" or checkpoint.get("visual_train_scope") != "frozen":
        raise RuntimeError(f"checkpoint is not a frozen-visual adapter: {checkpoint_path}")
    baseline_path = Path(checkpoint["baseline_ckpt"])
    if not baseline_path.is_file():
        raise FileNotFoundError(f"missing matched observation checkpoint: {baseline_path}")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    if int(baseline["seed"]) != seed or baseline.get("mode") != "observation":
        raise RuntimeError(f"baseline identity mismatch for seed={seed}: {baseline_path}")
    identity = assert_identical_visual_state(
        checkpoint["visual_state_dict"], baseline["visual_state_dict"]
    )

    state = checkpoint["state_dict"]
    hidden = int(state["visual.segment.decoder.out.weight"].shape[1])
    terrain_base = int(state["terrain_encoder.net.0.weight"].shape[0])
    obs_names = [str(item) for item in checkpoint["obs_channel_names"]]
    terrain_names = [str(item) for item in checkpoint["terrain_channel_names"]]
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
        center_physical_correction=bool(checkpoint.get("center_physical_correction", False)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    adapter_threshold = float(checkpoint["threshold"])
    visual_threshold = float(baseline["threshold"])
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
            visual_logits, visual_features = model.visual.segment(observation)
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
                logits, gate, residual = adapter_forward(
                    model, visual_logits, visual_features, uncertainty, terrain
                )
                logits_by_variant[variant] = logits
                gates[variant] = gate
                residuals[variant] = residual
            predictions = {
                variant: (
                    torch.sigmoid(logits)
                    >= (visual_threshold if variant == "visual_anchor" else adapter_threshold)
                )
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
                        "backend": str(checkpoint["backend"]),
                        "model_name": str(checkpoint["backbone"]),
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
                        row["gate_on_visual_error"] = (
                            float(gate[error_mask].mean().item()) if error_mask.any() else math.nan
                        )
                        row["gate_on_visual_correct"] = (
                            float(gate[correct_mask].mean().item()) if correct_mask.any() else math.nan
                        )
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
        "baseline_checkpoint": str(baseline_path),
        "seed": seed,
        "backend": checkpoint["backend"],
        "backbone": checkpoint["backbone"],
        "adapter_threshold": adapter_threshold,
        "visual_threshold": visual_threshold,
        "best_epoch": checkpoint.get("best_epoch"),
        "visual_train_scope": checkpoint.get("visual_train_scope"),
        **identity,
    }
    return rows, provenance


def main() -> int:
    args = parse_args()
    specs: list[tuple[str, Path]] = []
    for raw in args.spec or DEFAULT_SPECS:
        label, run_dir = raw.split("|", 1)
        path = Path(run_dir)
        if not path.is_absolute():
            path = args.run_root / path
        specs.append((label, path))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[FATAL] CUDA requested but unavailable")
    device = torch.device(args.device)
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
    provenance: list[dict[str, Any]] = []
    started = time.time()
    if args.reuse_counterfactuals is not None:
        with args.reuse_counterfactuals.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise SystemExit(f"[FATAL] empty reuse CSV: {args.reuse_counterfactuals}")
        prior_summary = args.outdir / "summary.json"
        if prior_summary.is_file():
            provenance = json.loads(prior_summary.read_text(encoding="utf-8")).get("provenance", [])
    else:
        for label, run_dir in specs:
            checkpoints = sorted((run_dir / "checkpoints").glob("*_adapter_seed*.pt"))
            if args.max_checkpoints > 0:
                checkpoints = checkpoints[: args.max_checkpoints]
            elif len(checkpoints) != 5:
                raise SystemExit(f"[FATAL] expected five adapter checkpoints in {run_dir}, found {len(checkpoints)}")
            if not checkpoints:
                raise SystemExit(f"[FATAL] no adapter checkpoints in {run_dir}")
            for checkpoint_path in checkpoints:
                print(f"[eval] {label} {checkpoint_path.name}", flush=True)
                checkpoint_rows, checkpoint_provenance = evaluate_checkpoint(
                    label,
                    checkpoint_path,
                    dataset,
                    args.batch_size,
                    args.workers,
                    device,
                    (args.roll_y, args.roll_x),
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
                "sample": compare(
                    sample_values, label, variant_a, variant_b, args.bootstrap, args.permutation_iters
                ),
                "seed": compare(
                    seed_values, label, variant_a, variant_b, args.bootstrap, args.permutation_iters
                ),
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
            "comparisons": comparisons,
            "mechanism": mechanism,
            "decisions": decisions,
            "holm_family_size": holm_family_size,
            "claim_rule": (
                "A terrain-specific claim requires aligned terrain to beat both visual-threshold anchors "
                "and all zero/shift/roll controls. Takeover additionally requires the independent-sample "
                "bootstrap lower bound for gate(error-correct) and net error reduction to exceed zero. "
                "Every visual state must exactly equal its observation checkpoint."
            ),
        }
    )
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# L4S foundation-model terrain attribution audit",
        "",
        f"- split: `{args.split}`; unique samples: `{len(base_dataset)}`",
        "- every adapter visual state is tensor-identical to its matched observation checkpoint",
        "- counterfactuals reuse frozen checkpoints; no test-set tuning or retraining is performed",
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
            "With five seeds, the minimum non-zero two-sided exact sign-flip p-value is 0.0625; seed inference is reported as directional reproducibility rather than a p<0.05 claim.",
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
            "Aligned terrain must beat both visual-threshold anchors and every zero/shift/roll control. "
            "Automatic takeover additionally requires the independent-sample 95% bootstrap lower bounds "
            "for higher gate values on visual errors and net error reduction to exceed zero. "
            "A backbone is not counted as supporting terrain attribution if any identity or control gate fails.",
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
