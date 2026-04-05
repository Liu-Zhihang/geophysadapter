#!/usr/bin/env python3
"""Evaluate physical-consistency proxies for a strict_t2 post_rgb experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from strict_t2_strong_backbone_common import BinaryDeepLabV3
from train_baseline_dlr_unet import SmallUNet
from train_strict_t2_postrgb_baseline import CachedPostRgbH5Dataset
from train_strict_t2_postrgb_phys_baseline import PhysicsAugmentedDataset, PhysicsFiLMUNet, load_physics_maps
from train_strict_t2_postrgb_v4_pilot import (
    PHYSICS_VECTOR_COLUMNS,
    PhysicsPriorDataset,
    PhysicsPriorDeepLabV3,
    load_meta_maps,
)


def resolve_path(root: Path, raw: str, fallback: Path | None = None) -> Path:
    if raw:
        path = Path(raw)
        if path.is_absolute():
            return path
        return root / path
    if fallback is not None:
        return fallback
    raise ValueError("path is empty and no fallback was provided")


def count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
    return count


def dataset_summary(records: list[dict[str, float]]) -> dict[str, float]:
    invalid_pixels = sum(item["invalid_pixels"] for item in records)
    invalid_fp = sum(item["invalid_fp"] for item in records)
    gt_pos_records = [item for item in records if item["gt_cc"] > 0]
    area_abs = float(np.mean([item["area_abs_bias"] for item in records])) if records else 0.0
    area_signed = float(np.mean([item["area_signed_bias"] for item in records])) if records else 0.0
    frag_abs = float(np.mean([item["fragment_abs_error"] for item in gt_pos_records])) if gt_pos_records else 0.0
    frag_ratio = float(np.mean([item["fragment_ratio"] for item in gt_pos_records])) if gt_pos_records else 0.0
    return {
        "invalid_fp_rate": float(invalid_fp / max(invalid_pixels, 1.0)),
        "area_abs_bias": area_abs,
        "area_signed_bias": area_signed,
        "fragment_abs_error": frag_abs,
        "fragment_ratio": frag_ratio,
        "num_samples": len(records),
        "num_positive_samples": len(gt_pos_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate physical-consistency proxies for strict_t2 post_rgb models")
    parser.add_argument("--summary", required=True, help="experiment summary.json path")
    parser.add_argument("--outdir", default="", help="default: <experiment>/physical_consistency_v1")
    parser.add_argument("--device", default="auto", choices=["cpu", "auto"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--threshold-mode",
        default="summary_eval",
        choices=["summary_eval", "default_050"],
        help="use the manuscript-facing eval_threshold from summary.json or force 0.50",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    root = Path(summary["config"]["root"])
    exp_dir = Path(summary["outdir"])
    outdir = Path(args.outdir) if args.outdir.strip() else exp_dir / "physical_consistency_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    patch_size = int(summary["config"]["patch_size"])
    test_cache_h5 = resolve_path(
        root,
        summary["config"].get("test_cache_h5", ""),
        fallback=root / "processed" / "hybrid_pinn" / "strict_t2_postrgb_eval_cache_v2" / f"test_postrgb_p{patch_size}.h5",
    )
    physics_csv = resolve_path(
        root,
        summary.get("physics_csv", ""),
        fallback=root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_physics_vectors_post_rgb_v1.csv",
    )

    physics_df = pd.read_csv(physics_csv, low_memory=False).fillna(0.0)
    physics_cols = summary.get("physics_vector_cols", [])
    sample_map = {}
    event_map = {}
    if physics_cols:
        sample_map, event_map = load_physics_maps(physics_csv, physics_cols)

    proxy_cols = [
        "sample_id",
        "event_uid",
        "dataset_id",
        "terrain_available",
        "terrain_1",
        "hydro_proxy",
        "stability_proxy",
    ]
    proxy_df = physics_df[[col for col in proxy_cols if col in physics_df.columns]].copy()
    proxy_lookup = {str(row["sample_id"]): row for row in proxy_df.to_dict(orient="records")}

    device = torch.device("cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    base_ds = CachedPostRgbH5Dataset(test_cache_h5)
    model_family = str(summary.get("model_family", ""))
    eval_threshold = float(summary.get("eval_threshold", 0.5)) if args.threshold_mode == "summary_eval" else 0.5
    dataset = base_ds
    if model_family == "strong_visual":
        model = BinaryDeepLabV3(
            in_channels=3,
            backbone_name=str(summary.get("backbone", "deeplabv3_resnet50")),
            pretrained_backbone=False,
            aux_loss=True,
        )
    elif model_family == "strict_t2_postrgb_v4_pilot":
        physics_mean = np.asarray(summary.get("physics_mean", []), dtype=np.float32)
        physics_std = np.asarray(summary.get("physics_std", []), dtype=np.float32)
        sample_meta, event_meta, _ = load_meta_maps(physics_csv)
        sample_map, event_map = load_physics_maps(physics_csv, PHYSICS_VECTOR_COLUMNS)
        dataset = PhysicsPriorDataset(
            base_ds=base_ds,
            sample_physics=sample_map,
            event_physics=event_map,
            sample_meta=sample_meta,
            event_meta=event_meta,
            physics_mean=physics_mean,
            physics_std=physics_std,
        )
        cfg = summary.get("config", {})
        model = PhysicsPriorDeepLabV3(
            backbone_name=str(summary.get("backbone", "deeplabv3_resnet50")),
            pretrained_backbone=False,
            token_dim=int(cfg.get("token_dim", 96)),
            latent_dim=int(cfg.get("latent_dim", 128)),
            prior_fusion_scale=float(cfg.get("prior_fusion_scale", 1.0)),
            interaction_scale=float(cfg.get("interaction_scale", 0.25)),
        )
    elif "physics_csv" in summary:
        mean = np.asarray(summary["physics_norm"]["mean"], dtype=np.float32)
        std = np.asarray(summary["physics_norm"]["std"], dtype=np.float32)
        dataset = PhysicsAugmentedDataset(base_ds, sample_map, event_map, mean, std)
        model = PhysicsFiLMUNet(
            in_ch=3,
            physics_dim=int(summary["physics_dim"]),
            base=32,
            hidden_dim=int(summary["config"]["hidden_dim"]),
            dropout=float(summary["config"]["dropout"]),
        )
    else:
        model = SmallUNet(in_ch=3, base=32)
    model = model.to(device)
    ckpt = torch.load(exp_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    records: list[dict[str, float | str]] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=False)
            valid = batch["valid"].to(device, non_blocking=False)
            target = batch["mask"].to(device, non_blocking=False)
            if model_family == "strong_visual":
                logits, _ = model(image)
            elif model_family == "strict_t2_postrgb_v4_pilot":
                physics = batch["physics"].to(device, non_blocking=False)
                meta = batch["meta"].to(device, non_blocking=False)
                logits = model(image=image, physics=physics, meta=meta)["logits"]
            elif "physics_csv" in summary:
                physics = batch["physics"].to(device, non_blocking=False)
                logits = model(image, physics)
            else:
                logits = model(image)
            pred = (torch.sigmoid(logits) >= eval_threshold).float()

            for i in range(image.size(0)):
                sample_id = batch["sample_id"][i]
                dataset_id = batch["dataset_id"][i]
                event_uid = batch["event_uid"][i]
                pred_np = pred[i, 0].detach().cpu().numpy().astype(bool, copy=False)
                gt_np = (target[i, 0].detach().cpu().numpy() >= 0.5)
                valid_np = (valid[i, 0].detach().cpu().numpy() >= 0.5)
                invalid_np = ~valid_np
                pred_valid = pred_np & valid_np
                gt_valid = gt_np & valid_np
                tp = float(np.logical_and(pred_valid, gt_valid).sum())
                fp = float(np.logical_and(pred_valid, ~gt_valid).sum())
                fn = float(np.logical_and(~pred_valid & valid_np, gt_valid).sum())
                valid_pixels = float(valid_np.sum())
                invalid_pixels = float(invalid_np.sum())
                invalid_fp = float((pred_np & invalid_np).sum())
                pred_pos_ratio = float(pred_valid.sum() / max(valid_pixels, 1.0))
                gt_pos_ratio = float(gt_valid.sum() / max(valid_pixels, 1.0))
                gt_pos_pixels = float(gt_valid.sum())
                gt_cc = int(count_components(gt_valid))
                pred_cc = int(count_components(pred_valid))
                meta = proxy_lookup.get(str(sample_id), {})
                records.append(
                    {
                        "sample_id": sample_id,
                        "event_uid": event_uid,
                        "dataset_id": dataset_id,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "valid_pixels": valid_pixels,
                        "gt_pos_pixels": gt_pos_pixels,
                        "invalid_pixels": invalid_pixels,
                        "invalid_fp": invalid_fp,
                        "pred_pos_ratio": pred_pos_ratio,
                        "gt_pos_ratio": gt_pos_ratio,
                        "area_signed_bias": pred_pos_ratio - gt_pos_ratio,
                        "area_abs_bias": abs(pred_pos_ratio - gt_pos_ratio),
                        "pred_cc": pred_cc,
                        "gt_cc": gt_cc,
                        "fragment_ratio": float(pred_cc / max(gt_cc, 1)),
                        "fragment_abs_error": float(abs(pred_cc - gt_cc) / (gt_cc + 1.0)),
                        "terrain_available": float(meta.get("terrain_available", 0.0)),
                        "terrain_slope_proxy": float(meta.get("terrain_1", 0.0)),
                        "hydro_proxy": float(meta.get("hydro_proxy", 0.0)),
                        "stability_proxy": float(meta.get("stability_proxy", 0.0)),
                    }
                )

    sample_csv = outdir / "per_sample_physical_consistency.csv"
    with sample_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    overall = dataset_summary(records)
    by_dataset_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in records:
        by_dataset_records[str(row["dataset_id"])].append(row)  # type: ignore[arg-type]
    by_dataset = {name: dataset_summary(rows) for name, rows in by_dataset_records.items()}

    stabilities = np.asarray([float(row["stability_proxy"]) for row in records], dtype=np.float32)
    low_thr = float(np.quantile(stabilities, 0.25))
    high_thr = float(np.quantile(stabilities, 0.75))
    low_rows = [row for row in records if float(row["stability_proxy"]) <= low_thr]
    high_rows = [row for row in records if float(row["stability_proxy"]) >= high_thr]
    low_fp = sum(float(row["fp"]) for row in low_rows)
    low_neg = sum(float(row["valid_pixels"]) - float(row["gt_pos_pixels"]) for row in low_rows)
    high_tp = sum(float(row["tp"]) for row in high_rows)
    high_fn = sum(float(row["fn"]) for row in high_rows)
    proxy_metrics = {
        "low_instability_false_alarm_rate_proxy": float(low_fp / max(low_neg, 1.0)),
        "high_instability_hit_rate_proxy": float(high_tp / max(high_tp + high_fn, 1.0)),
        "stability_proxy_q25": low_thr,
        "stability_proxy_q75": high_thr,
        "num_low_instability_samples": len(low_rows),
        "num_high_instability_samples": len(high_rows),
    }

    summary_out = {
        "summary_json": str(summary_path),
        "model_family": model_family,
        "best_epoch": int(summary["best_epoch"]),
        "best_val_iou": float(summary["best_val_iou"]),
        "threshold_mode": args.threshold_mode,
        "threshold_used": eval_threshold,
        "test_metrics": summary["test_metrics"],
        "physical_consistency_overall": overall,
        "physical_consistency_proxy": proxy_metrics,
        "physical_consistency_by_dataset": by_dataset,
        "per_sample_csv": str(sample_csv),
    }
    (outdir / "summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# Physical Consistency Report: {exp_dir.name}",
        "",
        "## Segmentation",
        "",
        f"- `best_val_iou = {float(summary['best_val_iou']):.6f}`",
        f"- `test_iou = {float(summary['test_metrics']['iou']):.6f}`",
        f"- `test_f1 = {float(summary['test_metrics']['f1']):.6f}`",
        f"- `threshold_mode = {args.threshold_mode}`",
        f"- `threshold_used = {eval_threshold:.2f}`",
        "",
        "## Physical Consistency",
        "",
        f"- `invalid_fp_rate = {overall['invalid_fp_rate']:.6f}`",
        f"- `high_instability_hit_rate_proxy = {proxy_metrics['high_instability_hit_rate_proxy']:.6f}`",
        f"- `low_instability_false_alarm_rate_proxy = {proxy_metrics['low_instability_false_alarm_rate_proxy']:.6f}`",
        f"- `fragment_abs_error = {overall['fragment_abs_error']:.6f}`",
        f"- `fragment_ratio = {overall['fragment_ratio']:.6f}`",
        f"- `area_abs_bias = {overall['area_abs_bias']:.6f}`",
        f"- `area_signed_bias = {overall['area_signed_bias']:.6f}`",
        "",
        "## By Dataset",
        "",
        "| dataset | invalid_fp_rate | fragment_abs_error | fragment_ratio | area_abs_bias | area_signed_bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_id, metrics in by_dataset.items():
        md_lines.append(
            f"| {dataset_id} | {metrics['invalid_fp_rate']:.6f} | {metrics['fragment_abs_error']:.6f} | {metrics['fragment_ratio']:.6f} | {metrics['area_abs_bias']:.6f} | {metrics['area_signed_bias']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `high_instability_hit_rate_proxy` and `low_instability_false_alarm_rate_proxy` are inferred from sample-level `stability_proxy` quantiles in the strict_t2 physics CSV, not from pixel-level slope maps.",
            "- `invalid_fp_rate` is expected to be most informative for samples with non-trivial invalid masks (for example GDCLD).",
            f"- Per-sample details: `{sample_csv}`",
        ]
    )
    (outdir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(outdir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
