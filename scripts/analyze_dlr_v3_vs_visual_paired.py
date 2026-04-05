from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_geophysadapter_dlr_dinov2_v3_proto import (
    DLRHybridSubset,
    GeoPhysAdapterHybridPINNV3,
)


ROOT = Path(__file__).resolve().parents[1]
SUBSET_DIR = ROOT / "processed" / "hybrid_pinn" / "dlr_strict_t3_reference_subset_v1"
PHYSICS_CSV = SUBSET_DIR / "sample_physics_vectors_v1.csv"
SPLITS = ("testind", "testspt")
SEEDS = (20260305, 20260306, 20260307)

V3_CKPTS = {
    20260305: Path("/home/zhihang/local_cache/pild_hybrid/experiments/geophysadapter_dinov2s_v3_pretrained_freeze_nodistill_e3_seed20260305/best_model.pt"),
    20260306: Path("/home/zhihang/local_cache/pild_hybrid/experiments/geophysadapter_dinov2s_v3_pretrained_freeze_nodistill_e3_seed20260306/best_model.pt"),
    20260307: Path("/home/zhihang/local_cache/pild_hybrid/experiments/geophysadapter_dinov2s_v3_pretrained_freeze_nodistill_e3_seed20260307/best_model.pt"),
}
VIS_CKPTS = {
    20260305: ROOT / "experiments/geophysadapter_dinov2s_v3_ablation_visualonly_e3_seed20260305/best_model.pt",
    20260306: ROOT / "experiments/geophysadapter_dinov2s_v3_ablation_visualonly_e3_seed20260306/best_model.pt",
    20260307: ROOT / "experiments/geophysadapter_dinov2s_v3_ablation_visualonly_e3_seed20260307/best_model.pt",
}


@dataclass
class RunSpec:
    model_key: str
    seed: int
    ckpt_path: Path
    visual_only: bool


class DLRHybridSubsetWithIDs(DLRHybridSubset):
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        row = self.rows[idx]
        item["sample_id"] = row["sample_id"]
        item["event_uid"] = row["event_uid"]
        return item


def bootstrap_ci(values: np.ndarray, num_bootstrap: int = 5000, seed: int = 20260403) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    n = len(values)
    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot.append(float(values[idx].mean()))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def iou_from_counts(tp: float, fp: float, fn: float) -> float:
    return float(tp / (tp + fp + fn + 1e-7))


def bootstrap_delta_from_counts(
    tp_a: np.ndarray,
    fp_a: np.ndarray,
    fn_a: np.ndarray,
    tp_b: np.ndarray,
    fp_b: np.ndarray,
    fn_b: np.ndarray,
    num_bootstrap: int = 5000,
    seed: int = 20260403,
) -> tuple[float, float, float]:
    if len(tp_a) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(tp_a)
    boot = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        iou_a = iou_from_counts(tp_a[idx].sum(), fp_a[idx].sum(), fn_a[idx].sum())
        iou_b = iou_from_counts(tp_b[idx].sum(), fp_b[idx].sum(), fn_b[idx].sum())
        boot.append(iou_a - iou_b)
    return (
        float(np.mean(boot)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def build_model(ckpt_path: Path, visual_only: bool, device: torch.device) -> GeoPhysAdapterHybridPINNV3:
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"unexpected_ckpt_format: {ckpt_path}")
    model = GeoPhysAdapterHybridPINNV3(
        dino_model=str(ckpt.get("dino_model", "vit_small_patch14_reg4_dinov2.lvd142m")),
        pretrained_backbone=False,
        freeze_backbone=True,
        unfreeze_last_blocks=int(ckpt.get("unfreeze_last_blocks", 0)),
        dino_input_size=int(ckpt.get("dino_input_size", 518)),
        visual_only=visual_only,
        disable_routing=False,
        disable_state_heads=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def make_loader(split: str, batch_size: int, workers: int) -> DataLoader:
    ds = DLRHybridSubsetWithIDs(
        SUBSET_DIR / f"{split}_n3_s1s2.h5",
        SUBSET_DIR / "sample_manifest.csv",
        split=split,
        physics_csv_path=PHYSICS_CSV,
        physics_resid_csv_path=PHYSICS_CSV,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


def eval_run(spec: RunSpec, split: str, batch_size: int, workers: int, device: torch.device) -> list[dict[str, object]]:
    model = build_model(spec.ckpt_path, spec.visual_only, device)
    loader = make_loader(split, batch_size=batch_size, workers=workers)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            logits, _, _, _, _, _ = model(
                batch["pre"].to(device, non_blocking=True),
                batch["post"].to(device, non_blocking=True),
                batch["terrain"].to(device, non_blocking=True),
                batch["material"].to(device, non_blocking=True),
                batch["trigger"].to(device, non_blocking=True),
            )
            pred = (torch.sigmoid(logits) >= 0.5).float().cpu()
            tgt = (batch["mask"] >= 0.5).float()
            tp = (pred * tgt).sum(dim=(1, 2, 3)).numpy()
            fp = (pred * (1.0 - tgt)).sum(dim=(1, 2, 3)).numpy()
            fn = ((1.0 - pred) * tgt).sum(dim=(1, 2, 3)).numpy()
            iou = tp / (tp + fp + fn + 1e-7)
            for i, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "model": spec.model_key,
                        "seed": spec.seed,
                        "split": split,
                        "sample_id": sample_id,
                        "event_uid": batch["event_uid"][i],
                        "tp": float(tp[i]),
                        "fp": float(fp[i]),
                        "fn": float(fn[i]),
                        "iou": float(iou[i]),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.4f} +- {arr.std(ddof=1):.4f}" if len(arr) > 1 else f"{arr.mean():.4f}"


def summarize(
    rows: list[dict[str, object]],
    overview: dict[str, dict[str, list[float]]],
) -> tuple[dict[str, object], str]:

    # averaged per-sample over seeds
    sample_aggs: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["split"]), str(row["sample_id"]), str(row["event_uid"]))
        sample_aggs.setdefault(key, []).append(float(row["iou"]))

    sample_mean_rows = []
    for (model, split, sample_id, event_uid), vals in sample_aggs.items():
        tp_vals = [
            float(r["tp"])
            for r in rows
            if r["model"] == model and r["split"] == split and r["sample_id"] == sample_id and r["event_uid"] == event_uid
        ]
        fp_vals = [
            float(r["fp"])
            for r in rows
            if r["model"] == model and r["split"] == split and r["sample_id"] == sample_id and r["event_uid"] == event_uid
        ]
        fn_vals = [
            float(r["fn"])
            for r in rows
            if r["model"] == model and r["split"] == split and r["sample_id"] == sample_id and r["event_uid"] == event_uid
        ]
        sample_mean_rows.append(
            {
                "model": model,
                "split": split,
                "sample_id": sample_id,
                "event_uid": event_uid,
                "mean_iou": float(np.mean(vals)),
                "mean_tp": float(np.mean(tp_vals)),
                "mean_fp": float(np.mean(fp_vals)),
                "mean_fn": float(np.mean(fn_vals)),
                "std_iou": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
            }
        )

    paired_summary: dict[str, object] = {"overview": {}, "sample_level": {}, "event_level": {}}
    for model, split_dict in overview.items():
        paired_summary["overview"][model] = {split: mean_sd(vals) for split, vals in split_dict.items()}

    markdown = ["# DLR v3 vs strict same-backbone visual-only paired analysis", ""]
    markdown.append("## Multiseed split means")
    markdown.append("")
    for model in ("v3", "visual_only"):
        markdown.append(f"### {model}")
        for split in SPLITS:
            if split in paired_summary["overview"].get(model, {}):
                markdown.append(f"- `{split}`: `{paired_summary['overview'][model][split]}`")
        markdown.append("")

    # paired sample/event analyses
    for split in SPLITS:
        vis = {r["sample_id"]: r for r in sample_mean_rows if r["model"] == "visual_only" and r["split"] == split}
        v3 = {r["sample_id"]: r for r in sample_mean_rows if r["model"] == "v3" and r["split"] == split}
        shared_ids = sorted(set(vis) & set(v3))
        deltas = np.array([v3[sid]["mean_iou"] - vis[sid]["mean_iou"] for sid in shared_ids], dtype=float)
        tp_v3 = np.asarray([float(v3[sid]["mean_tp"]) for sid in shared_ids], dtype=float)
        fp_v3 = np.asarray([float(v3[sid]["mean_fp"]) for sid in shared_ids], dtype=float)
        fn_v3 = np.asarray([float(v3[sid]["mean_fn"]) for sid in shared_ids], dtype=float)
        tp_vis = np.asarray([float(vis[sid]["mean_tp"]) for sid in shared_ids], dtype=float)
        fp_vis = np.asarray([float(vis[sid]["mean_fp"]) for sid in shared_ids], dtype=float)
        fn_vis = np.asarray([float(vis[sid]["mean_fn"]) for sid in shared_ids], dtype=float)
        boot_mean, boot_lo, boot_hi = bootstrap_delta_from_counts(tp_v3, fp_v3, fn_v3, tp_vis, fp_vis, fn_vis)
        sample_level = {
            "n_shared_samples": len(shared_ids),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "positive_rate": float((deltas > 0).mean()),
            "negative_rate": float((deltas < 0).mean()),
            "dataset_delta_bootstrap_mean": boot_mean,
            "ci95": [boot_lo, boot_hi],
        }
        paired_summary["sample_level"][split] = sample_level

        # event-level aggregation from sample means
        by_event: dict[str, dict[str, list[float]]] = {}
        for sid in shared_ids:
            evt = str(v3[sid]["event_uid"])
            event_bucket = by_event.setdefault(
                evt,
                {"tp_v3": [], "fp_v3": [], "fn_v3": [], "tp_vis": [], "fp_vis": [], "fn_vis": [], "delta_iou": []},
            )
            event_bucket["tp_v3"].append(float(v3[sid]["mean_tp"]))
            event_bucket["fp_v3"].append(float(v3[sid]["mean_fp"]))
            event_bucket["fn_v3"].append(float(v3[sid]["mean_fn"]))
            event_bucket["tp_vis"].append(float(vis[sid]["mean_tp"]))
            event_bucket["fp_vis"].append(float(vis[sid]["mean_fp"]))
            event_bucket["fn_vis"].append(float(vis[sid]["mean_fn"]))
            event_bucket["delta_iou"].append(float(v3[sid]["mean_iou"] - vis[sid]["mean_iou"]))
        event_deltas = np.array([float(np.mean(vals["delta_iou"])) for vals in by_event.values()], dtype=float)
        evt_tp_v3 = np.asarray([sum(vals["tp_v3"]) for vals in by_event.values()], dtype=float)
        evt_fp_v3 = np.asarray([sum(vals["fp_v3"]) for vals in by_event.values()], dtype=float)
        evt_fn_v3 = np.asarray([sum(vals["fn_v3"]) for vals in by_event.values()], dtype=float)
        evt_tp_vis = np.asarray([sum(vals["tp_vis"]) for vals in by_event.values()], dtype=float)
        evt_fp_vis = np.asarray([sum(vals["fp_vis"]) for vals in by_event.values()], dtype=float)
        evt_fn_vis = np.asarray([sum(vals["fn_vis"]) for vals in by_event.values()], dtype=float)
        evt_boot_mean, evt_boot_lo, evt_boot_hi = bootstrap_delta_from_counts(
            evt_tp_v3, evt_fp_v3, evt_fn_v3, evt_tp_vis, evt_fp_vis, evt_fn_vis
        )
        event_level = {
            "n_events": len(event_deltas),
            "mean_delta": float(event_deltas.mean()),
            "median_delta": float(np.median(event_deltas)),
            "positive_rate": float((event_deltas > 0).mean()),
            "negative_rate": float((event_deltas < 0).mean()),
            "dataset_delta_bootstrap_mean": evt_boot_mean,
            "ci95": [evt_boot_lo, evt_boot_hi],
        }
        paired_summary["event_level"][split] = event_level

        markdown.append(f"## {split}")
        markdown.append("")
        markdown.append(f"- shared samples: `{sample_level['n_shared_samples']}`")
        markdown.append(f"- sample-mean ΔIoU: `{sample_level['mean_delta']:+.4f}`")
        markdown.append(f"- sample-level aggregated-IoU bootstrap Δ: `{sample_level['dataset_delta_bootstrap_mean']:+.4f}` (95% bootstrap CI `{sample_level['ci95'][0]:+.4f}` to `{sample_level['ci95'][1]:+.4f}`)")
        markdown.append(f"- sample-level median ΔIoU: `{sample_level['median_delta']:+.4f}`")
        markdown.append(f"- sample-level positive rate: `{sample_level['positive_rate']*100:.1f}%`")
        markdown.append(f"- events: `{event_level['n_events']}`")
        markdown.append(f"- event-mean ΔIoU: `{event_level['mean_delta']:+.4f}`")
        markdown.append(f"- event-level aggregated-IoU bootstrap Δ: `{event_level['dataset_delta_bootstrap_mean']:+.4f}` (95% bootstrap CI `{event_level['ci95'][0]:+.4f}` to `{event_level['ci95'][1]:+.4f}`)")
        markdown.append(f"- event-level median ΔIoU: `{event_level['median_delta']:+.4f}`")
        markdown.append(f"- event-level positive rate: `{event_level['positive_rate']*100:.1f}%`")
        markdown.append("")

    return {"summary": paired_summary, "sample_mean_rows": sample_mean_rows}, "\n".join(markdown) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate DLR v3 vs strict same-backbone visual-only controls and summarize paired deltas.")
    p.add_argument("--outdir", default=str(ROOT / "experiments" / "geophysadapter_dinov2s_v3_visualonly_paired_v1"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = []
    for seed in SEEDS:
        specs.append(RunSpec("v3", seed, V3_CKPTS[seed], visual_only=False))
        specs.append(RunSpec("visual_only", seed, VIS_CKPTS[seed], visual_only=True))
    for spec in specs:
        if not spec.ckpt_path.exists():
            raise FileNotFoundError(f"missing_ckpt: {spec.ckpt_path}")

    rows: list[dict[str, object]] = []
    overview: dict[str, dict[str, list[float]]] = {}
    for spec in specs:
        result_path = spec.ckpt_path.parent / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for split in SPLITS:
            overview.setdefault(spec.model_key, {}).setdefault(split, []).append(float(result["metrics"][split]["iou"]))
        for split in SPLITS:
            print(f"[eval] model={spec.model_key} seed={spec.seed} split={split}")
            rows.extend(eval_run(spec, split=split, batch_size=args.batch_size, workers=args.workers, device=device))

    write_csv(outdir / "per_sample_seed_metrics.csv", rows)
    result, report = summarize(rows, overview)
    write_csv(outdir / "per_sample_mean_metrics.csv", result["sample_mean_rows"])
    (outdir / "summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(f"[done] wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
