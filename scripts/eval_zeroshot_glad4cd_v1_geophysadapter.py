#!/usr/bin/env python3
"""Zero-shot cross-domain evaluation on GLaD4CD v1 for GeoPhysAdapter runs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import tifffile

from train_geophysadapter_dlr_dinov2_v2_proto import GeoPhysAdapterHybridPINNV2
from train_geophysadapter_dlr_dinov2_v3_proto import GeoPhysAdapterHybridPINNV3, MATERIAL_DIM, TRIGGER_DIM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--glad-v1-dir", default="")
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def resolve_glad_v1(root: Path, glad_v1_dir: str) -> Path:
    if glad_v1_dir.strip():
        return Path(glad_v1_dir)
    return root / "raw" / "datasets" / "06_GLaD4CD" / "GLaD4CD_v1_unpacked" / "LANDSLIDE_DATASET"


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


def infer_tiled_logits(model: torch.nn.Module, pre: np.ndarray, post: np.ndarray, device: torch.device, tile: int, amp: bool):
    c, h, w = pre.shape
    logits_acc = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    zero_material = torch.zeros((1, MATERIAL_DIM), dtype=torch.float32, device=device)
    zero_trigger = torch.zeros((1, TRIGGER_DIM), dtype=torch.float32, device=device)
    zero_terrain_cache = {}
    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            r1 = min(r0 + tile, h)
            c1 = min(c0 + tile, w)
            pre_patch = pre[:, r0:r1, c0:c1]
            post_patch = post[:, r0:r1, c0:c1]
            ph, pw = pre_patch.shape[1], pre_patch.shape[2]
            if ph != tile or pw != tile:
                pre_pad = np.zeros((c, tile, tile), dtype=np.float32)
                post_pad = np.zeros((c, tile, tile), dtype=np.float32)
                pre_pad[:, :ph, :pw] = pre_patch
                post_pad[:, :ph, :pw] = post_patch
                pre_patch = pre_pad
                post_patch = post_pad
            xt_pre = torch.from_numpy(pre_patch[None, ...]).to(device, non_blocking=True)
            xt_post = torch.from_numpy(post_patch[None, ...]).to(device, non_blocking=True)
            key = (tile, tile)
            if key not in zero_terrain_cache:
                zero_terrain_cache[key] = torch.zeros((1, 2, tile, tile), dtype=torch.float32, device=device)
            terrain = zero_terrain_cache[key]
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == 'cuda')):
                out = model(xt_pre, xt_post, terrain, zero_material, zero_trigger)
                logits = out[0] if isinstance(out, tuple) else out
                logit = logits[0, 0].detach().float().cpu().numpy()
            logit = logit[:ph, :pw]
            logits_acc[r0:r1, c0:c1] += logit
            counts[r0:r1, c0:c1] += 1.0
    counts = np.maximum(counts, 1.0)
    return logits_acc / counts


def evaluate_run(model: torch.nn.Module, pairs, device: torch.device, tile: int, amp: bool):
    tp = fp = fn = 0.0
    sample_rows = []
    for sid, a_path, b_path, y_path in pairs:
        a = tifffile.imread(a_path).astype(np.float32)
        b = tifffile.imread(b_path).astype(np.float32)
        y = tifffile.imread(y_path).astype(np.float32)
        if a.ndim == 3 and a.shape[-1] >= 4:
            a = np.moveaxis(a[..., :4], -1, 0)
        if b.ndim == 3 and b.shape[-1] >= 4:
            b = np.moveaxis(b[..., :4], -1, 0)
        if y.ndim == 3:
            y = y[..., 0]
        y = (y >= 0.5).astype(np.uint8)
        logits = infer_tiled_logits(model, a, b, device, tile, amp)
        pred = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(np.uint8)
        tp_i = float(np.sum((pred == 1) & (y == 1)))
        fp_i = float(np.sum((pred == 1) & (y == 0)))
        fn_i = float(np.sum((pred == 0) & (y == 1)))
        iou_i = tp_i / (tp_i + fp_i + fn_i + 1e-7)
        f1_i = (2.0 * tp_i) / (2.0 * tp_i + fp_i + fn_i + 1e-7)
        sample_rows.append({"sample_id": sid, "tp": tp_i, "fp": fp_i, "fn": fn_i, "iou": iou_i, "f1": f1_i})
        tp += tp_i; fp += fp_i; fn += fn_i
    iou = tp / (tp + fp + fn + 1e-7)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return {"tp": tp, "fp": fp, "fn": fn, "iou": iou, "f1": f1, "num_samples": len(pairs), "sample_metrics": sample_rows}


def build_model_from_meta(meta: dict) -> torch.nn.Module:
    arch = meta.get("architecture", "")
    common_kwargs = dict(
        dino_model=meta.get("dino_model", "vit_small_patch14_dinov2"),
        pretrained_backbone=False,
        freeze_backbone=bool(meta.get("freeze_backbone", True)),
        unfreeze_last_blocks=int(meta.get("unfreeze_last_blocks", 0)),
        dino_input_size=int(meta.get("dino_input_size", 196)),
    )
    if arch == "GeoPhysAdapterHybridPINNV2":
        return GeoPhysAdapterHybridPINNV2(**common_kwargs)
    if arch == "GeoPhysAdapterHybridPINNV3":
        return GeoPhysAdapterHybridPINNV3(**common_kwargs)
    raise RuntimeError(f"unsupported_architecture:{arch}")


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(1)
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    glad_v1 = resolve_glad_v1(root, args.glad_v1_dir)
    pairs = build_pairs(glad_v1)
    if not pairs:
        raise RuntimeError("No valid GLaD4CD v1 VALIDATION pairs found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        print(f"[info] gpu={torch.cuda.get_device_name(0)}")
    print(f"[info] glad_v1_pairs={len(pairs)} tile_size={args.tile_size}")

    rows = []
    payload = {"timestamp": int(time.time()), "dataset": "GLaD4CD_v1_VALIDATION_zero_shot", "glad_v1_dir": str(glad_v1), "tile_size": args.tile_size, "num_pairs": len(pairs), "runs": []}
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        ckpt_path = run_dir / "best_model.pt"
        result_path = run_dir / "result.json"
        if not ckpt_path.exists() or not result_path.exists():
            raise FileNotFoundError(f"missing run artifacts in {run_dir}")
        meta = json.loads(result_path.read_text(encoding='utf-8'))
        seed = meta.get('seed', -1)
        print(f"[run] {run_dir.name} seed={seed}")
        model = build_model_from_meta(meta).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        metrics = evaluate_run(model, pairs, device, args.tile_size, args.amp)
        run_payload = {"run_dir": str(run_dir), "seed": seed, **metrics}
        payload['runs'].append(run_payload)
        (run_dir / 'zeroshot_glad4cd_v1_val_geophysadapter.json').write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[metric] {run_dir.name} iou={metrics['iou']:.6f} f1={metrics['f1']:.6f} tp={metrics['tp']:.0f} fp={metrics['fp']:.0f} fn={metrics['fn']:.0f}")
        rows.append({"run": run_dir.name, "seed": seed, "num_samples": metrics['num_samples'], "iou": metrics['iou'], "f1": metrics['f1'], "tp": metrics['tp'], "fp": metrics['fp'], "fn": metrics['fn']})

    combined_json = outdir / 'zeroshot_glad4cd_v1_geophysadapter_runs.json'
    combined_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_path = outdir / 'zeroshot_glad4cd_v1_geophysadapter_runs.csv'
    headers = ['run','seed','num_samples','iou','f1','tp','fp','fn']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader(); w.writerows(rows)
    md_path = outdir / 'zeroshot_glad4cd_v1_geophysadapter_runs.md'
    lines = ['# Zero-Shot Cross-Domain on GLaD4CD v1 Validation (GeoPhysAdapter)','', f'- pairs: `{len(pairs)}`', f'- tile_size: `{args.tile_size}`', '', '| run | seed | iou | f1 |', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['run']} | {r['seed']} | {r['iou']:.6f} | {r['f1']:.6f} |")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"[done] wrote {combined_json}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
