#!/usr/bin/env python3
"""Evaluate reliability metrics (ECE/Brier/NLL) for DLR segmentation runs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_baseline_dlr_unet import DLRH5Dataset, FEATURE_KEYS, SmallUNet


@dataclass
class SplitReliability:
    n_pixels: int
    pos_rate: float
    brier: float
    nll: float
    ece_prob: float
    mce_prob: float
    ece_conf: float
    mce_conf: float


def _resolve_ref_dir(root: Path, dlr_ref_dir: str) -> Path:
    if dlr_ref_dir.strip():
        return Path(dlr_ref_dir)
    return (
        root
        / "raw"
        / "datasets"
        / "05_DLR_Landslide_Ref_2025"
        / "extracted"
        / "s1s2_landslide_reference_data"
        / "s1s2_landslide_reference_data"
        / "reference_data"
    )


def _build_loaders(ref_dir: Path, batch_size: int, workers: int, device: torch.device):
    val_ds = DLRH5Dataset(ref_dir / "val_n3_s1s2.h5", FEATURE_KEYS)
    testind_ds = DLRH5Dataset(ref_dir / "testind_n3_s1s2.h5", FEATURE_KEYS)
    testspt_ds = DLRH5Dataset(ref_dir / "testspt_n3_s1s2.h5", FEATURE_KEYS)
    loaders = {
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        ),
        "testind": DataLoader(
            testind_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        ),
        "testspt": DataLoader(
            testspt_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        ),
    }
    return loaders


def _ece_from_bins(counts: np.ndarray, a_sum: np.ndarray, b_sum: np.ndarray, total: int):
    nonzero = counts > 0
    if total <= 0 or not np.any(nonzero):
        return 0.0, 0.0
    a_mean = np.zeros_like(a_sum, dtype=np.float64)
    b_mean = np.zeros_like(b_sum, dtype=np.float64)
    a_mean[nonzero] = a_sum[nonzero] / counts[nonzero]
    b_mean[nonzero] = b_sum[nonzero] / counts[nonzero]
    gaps = np.abs(a_mean - b_mean)
    weights = counts / float(total)
    ece = float(np.sum(weights * gaps))
    mce = float(np.max(gaps[nonzero]))
    return ece, mce


@torch.no_grad()
def evaluate_split_reliability(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_bins: int,
) -> SplitReliability:
    model.eval()
    n_total = 0
    pos_sum = 0.0
    brier_sum = 0.0
    nll_sum = 0.0

    # Prob-based calibration bins: compare mean(prob) vs mean(label)
    prob_counts = np.zeros(num_bins, dtype=np.float64)
    prob_p_sum = np.zeros(num_bins, dtype=np.float64)
    prob_y_sum = np.zeros(num_bins, dtype=np.float64)

    # Confidence-based calibration bins: compare mean(conf) vs mean(acc)
    conf_counts = np.zeros(num_bins, dtype=np.float64)
    conf_c_sum = np.zeros(num_bins, dtype=np.float64)
    conf_a_sum = np.zeros(num_bins, dtype=np.float64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)

        p = torch.sigmoid(logits).reshape(-1).float().cpu()
        t = (y >= 0.5).reshape(-1).float().cpu()
        n = int(p.numel())
        if n == 0:
            continue

        p_clamped = p.clamp(1e-7, 1.0 - 1e-7)
        brier_sum += torch.square(p - t).sum().item()
        nll_sum += F.binary_cross_entropy(p_clamped, t, reduction="sum").item()
        pos_sum += t.sum().item()
        n_total += n

        prob_idx = torch.clamp((p * num_bins).long(), min=0, max=num_bins - 1)
        prob_counts += torch.bincount(prob_idx, minlength=num_bins).numpy().astype(np.float64)
        prob_p_sum += torch.bincount(prob_idx, weights=p, minlength=num_bins).numpy().astype(np.float64)
        prob_y_sum += torch.bincount(prob_idx, weights=t, minlength=num_bins).numpy().astype(np.float64)

        pred = (p >= 0.5).float()
        acc = (pred == t).float()
        conf = torch.maximum(p, 1.0 - p)
        conf_idx = torch.clamp((conf * num_bins).long(), min=0, max=num_bins - 1)
        conf_counts += torch.bincount(conf_idx, minlength=num_bins).numpy().astype(np.float64)
        conf_c_sum += torch.bincount(conf_idx, weights=conf, minlength=num_bins).numpy().astype(np.float64)
        conf_a_sum += torch.bincount(conf_idx, weights=acc, minlength=num_bins).numpy().astype(np.float64)

    if n_total == 0:
        return SplitReliability(
            n_pixels=0,
            pos_rate=0.0,
            brier=0.0,
            nll=0.0,
            ece_prob=0.0,
            mce_prob=0.0,
            ece_conf=0.0,
            mce_conf=0.0,
        )

    ece_prob, mce_prob = _ece_from_bins(prob_counts, prob_p_sum, prob_y_sum, n_total)
    ece_conf, mce_conf = _ece_from_bins(conf_counts, conf_c_sum, conf_a_sum, n_total)
    return SplitReliability(
        n_pixels=n_total,
        pos_rate=float(pos_sum / n_total),
        brier=float(brier_sum / n_total),
        nll=float(nll_sum / n_total),
        ece_prob=ece_prob,
        mce_prob=mce_prob,
        ece_conf=ece_conf,
        mce_conf=mce_conf,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="PILD root",
    )
    p.add_argument(
        "--dlr-ref-dir",
        default="",
        help="optional override path containing DLR h5 files",
    )
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="one or more run directories (each containing best_model.pt and result.json)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--num-bins", type=int, default=15)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument(
        "--outdir",
        default="",
        help="where combined reports are written; default root/experiments",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)

    root = Path(args.root)
    outdir = Path(args.outdir) if args.outdir.strip() else (root / "experiments")
    outdir.mkdir(parents=True, exist_ok=True)

    ref_dir = _resolve_ref_dir(root=root, dlr_ref_dir=args.dlr_ref_dir)
    for name in ["val_n3_s1s2.h5", "testind_n3_s1s2.h5", "testspt_n3_s1s2.h5"]:
        if not (ref_dir / name).exists():
            raise FileNotFoundError(ref_dir / name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        print(f"[info] gpu={torch.cuda.get_device_name(0)}")

    loaders = _build_loaders(ref_dir, batch_size=args.batch_size, workers=args.workers, device=device)
    print(
        "[info] split_sizes val/testind/testspt="
        f"{len(loaders['val'].dataset)}/{len(loaders['testind'].dataset)}/{len(loaders['testspt'].dataset)}"
    )

    rows = []
    combined = {
        "timestamp": int(time.time()),
        "device": str(device),
        "num_bins": args.num_bins,
        "ref_dir": str(ref_dir),
        "runs": [],
    }

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        ckpt_path = run_dir / "best_model.pt"
        result_path = run_dir / "result.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        if not result_path.exists():
            raise FileNotFoundError(result_path)

        run_meta = json.loads(result_path.read_text(encoding="utf-8"))
        seed = run_meta.get("seed", -1)
        lambda_topo = run_meta.get("lambda_topo", 0.0)
        lambda_phys = run_meta.get("lambda_phys", 0.0)
        ndvi_drop_thresh = run_meta.get("ndvi_drop_thresh", 0.0)

        print(
            f"[run] {run_dir.name} seed={seed} "
            f"lambda_topo={lambda_topo} lambda_phys={lambda_phys} ndvi_drop_thresh={ndvi_drop_thresh}"
        )

        model = SmallUNet(in_ch=len(FEATURE_KEYS), base=32).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        run_payload = {
            "run_dir": str(run_dir),
            "seed": seed,
            "lambda_topo": lambda_topo,
            "lambda_phys": lambda_phys,
            "ndvi_drop_thresh": ndvi_drop_thresh,
            "splits": {},
        }
        for split_name, loader in loaders.items():
            met = evaluate_split_reliability(model=model, loader=loader, device=device, num_bins=args.num_bins)
            run_payload["splits"][split_name] = met.__dict__
            row = {
                "run": run_dir.name,
                "split": split_name,
                "seed": seed,
                "lambda_topo": lambda_topo,
                "lambda_phys": lambda_phys,
                "ndvi_drop_thresh": ndvi_drop_thresh,
                **met.__dict__,
            }
            rows.append(row)
            print(
                f"[metric] {run_dir.name} {split_name} "
                f"ece_prob={met.ece_prob:.6f} ece_conf={met.ece_conf:.6f} "
                f"brier={met.brier:.6f} nll={met.nll:.6f}"
            )

        per_run_path = run_dir / "reliability.json"
        per_run_path.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        combined["runs"].append(run_payload)
        print(f"[done] wrote {per_run_path}")

    combined_json = outdir / "reliability_dlr_runs.json"
    combined_json.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = outdir / "reliability_dlr_runs.csv"
    headers = [
        "run",
        "split",
        "seed",
        "lambda_topo",
        "lambda_phys",
        "ndvi_drop_thresh",
        "n_pixels",
        "pos_rate",
        "brier",
        "nll",
        "ece_prob",
        "mce_prob",
        "ece_conf",
        "mce_conf",
    ]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in headers))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_path = outdir / "reliability_dlr_runs.md"
    md_lines = [
        "# DLR Reliability Metrics",
        "",
        f"- num_bins: `{args.num_bins}`",
        f"- ref_dir: `{ref_dir}`",
        "",
        "| run | split | ece_prob | ece_conf | brier | nll |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['run']} | {r['split']} | {r['ece_prob']:.6f} | {r['ece_conf']:.6f} | {r['brier']:.6f} | {r['nll']:.6f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {combined_json}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
