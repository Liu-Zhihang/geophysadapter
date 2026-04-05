#!/usr/bin/env python3
"""Zero-shot cross-domain evaluation on GLaD4CD v1 validation set."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from train_baseline_dlr_unet import FEATURE_KEYS, SmallUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="PILD root",
    )
    p.add_argument(
        "--glad-v1-dir",
        default="",
        help="optional GLaD4CD v1 LANDSLIDE_DATASET dir override",
    )
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--outdir", default="", help="default: root/experiments")
    return p.parse_args()


def resolve_glad_v1(root: Path, glad_v1_dir: str) -> Path:
    if glad_v1_dir.strip():
        return Path(glad_v1_dir)
    return (
        root
        / "raw"
        / "datasets"
        / "06_GLaD4CD"
        / "GLaD4CD_v1_unpacked"
        / "LANDSLIDE_DATASET"
    )


def build_pairs(glad_v1: Path):
    a_dir = glad_v1 / "VALIDATION" / "A"
    b_dir = glad_v1 / "VALIDATION" / "B"
    y_dir = glad_v1 / "VALIDATION" / "LABEL"
    if not (a_dir.exists() and b_dir.exists() and y_dir.exists()):
        raise FileNotFoundError("GLaD4CD v1 VALIDATION A/B/LABEL dirs are required")

    pairs = []
    for y in sorted(y_dir.glob("*.tif")):
        sid = y.stem
        a = a_dir / f"{sid}.tif"
        b = b_dir / f"{sid}.tif"
        if a.exists() and b.exists():
            pairs.append((sid, a, b, y))
    return pairs


def infer_tiled_logits(model: torch.nn.Module, x: np.ndarray, device: torch.device, tile: int):
    # x: (C,H,W), float32
    c, h, w = x.shape
    logits_acc = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            r1 = min(r0 + tile, h)
            c1 = min(c0 + tile, w)
            patch = x[:, r0:r1, c0:c1]

            ph, pw = patch.shape[1], patch.shape[2]
            if ph != tile or pw != tile:
                pad = np.zeros((c, tile, tile), dtype=np.float32)
                pad[:, :ph, :pw] = patch
                patch = pad

            xt = torch.from_numpy(patch[None, ...]).to(device, non_blocking=True)
            with torch.no_grad():
                out = model(xt)
                logit = out[0, 0].detach().float().cpu().numpy()
            logit = logit[:ph, :pw]

            logits_acc[r0:r1, c0:c1] += logit
            counts[r0:r1, c0:c1] += 1.0

    counts = np.maximum(counts, 1.0)
    return logits_acc / counts


def evaluate_run(model: torch.nn.Module, pairs, device: torch.device, tile: int):
    tp = fp = fn = 0.0
    sample_rows = []
    for sid, a_path, b_path, y_path in pairs:
        with rasterio.open(a_path) as a_ds, rasterio.open(b_path) as b_ds, rasterio.open(y_path) as y_ds:
            # Use B02/B03/B04/B08 from A/B; fill DEM/SLOPE with zeros for DLR-compat 10-ch input.
            a = a_ds.read([1, 2, 3, 4]).astype(np.float32)
            b = b_ds.read([1, 2, 3, 4]).astype(np.float32)
            h, w = a.shape[1], a.shape[2]
            z = np.zeros((2, h, w), dtype=np.float32)
            x = np.concatenate([a, b, z], axis=0)
            y = y_ds.read(1).astype(np.float32)
            y = (y >= 0.5).astype(np.uint8)

        logits = infer_tiled_logits(model=model, x=x, device=device, tile=tile)
        pred = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(np.uint8)
        tp_i = float(np.sum((pred == 1) & (y == 1)))
        fp_i = float(np.sum((pred == 1) & (y == 0)))
        fn_i = float(np.sum((pred == 0) & (y == 1)))
        iou_i = tp_i / (tp_i + fp_i + fn_i + 1e-7)
        f1_i = (2.0 * tp_i) / (2.0 * tp_i + fp_i + fn_i + 1e-7)
        sample_rows.append(
            {
                "sample_id": sid,
                "tp": tp_i,
                "fp": fp_i,
                "fn": fn_i,
                "iou": iou_i,
                "f1": f1_i,
            }
        )
        tp += tp_i
        fp += fp_i
        fn += fn_i

    iou = tp / (tp + fp + fn + 1e-7)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "f1": f1,
        "num_samples": len(pairs),
        "sample_metrics": sample_rows,
    }


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)

    root = Path(args.root)
    outdir = Path(args.outdir) if args.outdir.strip() else (root / "experiments")
    outdir.mkdir(parents=True, exist_ok=True)

    glad_v1 = resolve_glad_v1(root=root, glad_v1_dir=args.glad_v1_dir)
    pairs = build_pairs(glad_v1)
    if not pairs:
        raise RuntimeError("No valid A/B/LABEL pairs found in GLaD4CD v1 VALIDATION")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        print(f"[info] gpu={torch.cuda.get_device_name(0)}")
    print(f"[info] glad_v1_pairs={len(pairs)} tile_size={args.tile_size}")

    rows = []
    payload = {
        "timestamp": int(time.time()),
        "dataset": "GLaD4CD_v1_VALIDATION_zero_shot",
        "glad_v1_dir": str(glad_v1),
        "tile_size": args.tile_size,
        "num_pairs": len(pairs),
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

        meta = json.loads(result_path.read_text(encoding="utf-8"))
        seed = meta.get("seed", -1)
        lambda_topo = meta.get("lambda_topo", 0.0)
        lambda_phys = meta.get("lambda_phys", 0.0)
        ndvi_drop_thresh = meta.get("ndvi_drop_thresh", 0.0)
        print(
            f"[run] {run_dir.name} seed={seed} lambda_topo={lambda_topo} "
            f"lambda_phys={lambda_phys} ndvi_drop_thresh={ndvi_drop_thresh}"
        )

        model = SmallUNet(in_ch=len(FEATURE_KEYS), base=32).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        metrics = evaluate_run(model=model, pairs=pairs, device=device, tile=args.tile_size)
        run_payload = {
            "run_dir": str(run_dir),
            "seed": seed,
            "lambda_topo": lambda_topo,
            "lambda_phys": lambda_phys,
            "ndvi_drop_thresh": ndvi_drop_thresh,
            **metrics,
        }
        payload["runs"].append(run_payload)

        out_json = run_dir / "zeroshot_glad4cd_v1_val.json"
        out_json.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[metric] {run_dir.name} iou={metrics['iou']:.6f} f1={metrics['f1']:.6f} "
            f"tp={metrics['tp']:.0f} fp={metrics['fp']:.0f} fn={metrics['fn']:.0f}"
        )
        print(f"[done] wrote {out_json}")

        rows.append(
            {
                "run": run_dir.name,
                "seed": seed,
                "lambda_topo": lambda_topo,
                "lambda_phys": lambda_phys,
                "ndvi_drop_thresh": ndvi_drop_thresh,
                "num_samples": metrics["num_samples"],
                "iou": metrics["iou"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )

    combined_json = outdir / "zeroshot_glad4cd_v1_runs.json"
    combined_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = outdir / "zeroshot_glad4cd_v1_runs.csv"
    headers = [
        "run",
        "seed",
        "lambda_topo",
        "lambda_phys",
        "ndvi_drop_thresh",
        "num_samples",
        "iou",
        "f1",
        "tp",
        "fp",
        "fn",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    md_path = outdir / "zeroshot_glad4cd_v1_runs.md"
    md_lines = [
        "# Zero-Shot Cross-Domain on GLaD4CD v1 Validation",
        "",
        f"- pairs: `{len(pairs)}`",
        f"- tile_size: `{args.tile_size}`",
        "",
        "| run | seed | lambda_phys | iou | f1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['run']} | {r['seed']} | {r['lambda_phys']} | {r['iou']:.6f} | {r['f1']:.6f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {combined_json}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
