#!/usr/bin/env python3
"""Render qualitative panel sheets for strict_t2 post_rgb case studies."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from strict_t2_strong_backbone_common import BinaryDeepLabV3
from train_baseline_dlr_unet import SmallUNet
from train_strict_t2_postrgb_phys_baseline import PhysicsFiLMUNet, load_physics_maps
from train_strict_t2_postrgb_v4_pilot import (
    PHYSICS_VECTOR_COLUMNS,
    PhysicsPriorDeepLabV3,
    load_meta_maps,
)


import platform
import re

def _convert_wsl_path(path_str: str) -> str:
    """Convert WSL /mnt/X/... path to Windows X:\\... format if on Windows."""
    if platform.system() == "Windows" and path_str.startswith("/mnt/"):
        match = re.match(r"/mnt/([a-z])/(.+)", path_str)
        if match:
            drive = match.group(1).upper()
            rest = match.group(2).replace("/", "\\")
            return f"{drive}:\\{rest}"
    return path_str


def resolve_path(root: Path, raw: str, fallback: Path | None = None) -> Path:
    if raw:
        raw = _convert_wsl_path(raw)
        path = Path(raw)
        if path.is_absolute():
            return path
        return root / path
    if fallback is not None:
        return fallback
    raise ValueError("path is empty and no fallback provided")


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for raw in candidates:
        path = Path(raw)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def overlay_rgb(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray | None = None, valid: np.ndarray | None = None) -> np.ndarray:
    base = np.clip(rgb.transpose(1, 2, 0), 0.0, 1.0).copy()
    if valid is not None:
        invalid = ~valid
        base[invalid] = 0.35 * base[invalid] + 0.45
    if gt is not None:
        base[gt, 0] = np.clip(base[gt, 0] * 0.4 + 0.6, 0.0, 1.0)
        base[gt, 1] = np.clip(base[gt, 1] * 0.5, 0.0, 1.0)
        base[gt, 2] = np.clip(base[gt, 2] * 0.5, 0.0, 1.0)
    if pred is not None:
        base[pred, 0] = np.clip(base[pred, 0] * 0.35 + 0.65, 0.0, 1.0)  
        base[pred, 1] = np.clip(base[pred, 1] * 0.45 + 0.40, 0.0, 1.0)  
        base[pred, 2] = np.clip(base[pred, 2] * 0.3, 0.0, 1.0)          
    return base


def panel_to_image(panel: np.ndarray, size: int) -> Image.Image:
    arr = np.clip(panel * 255.0, 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.Resampling.NEAREST)
    return img


@dataclass
class LoadedModel:
    name: str
    summary: dict
    model: torch.nn.Module
    model_family: str
    sample_map: dict[str, np.ndarray]
    event_map: dict[str, np.ndarray]
    sample_meta: dict[str, np.ndarray]
    event_meta: dict[str, np.ndarray]
    mean: np.ndarray | None
    std: np.ndarray | None
    threshold: float

    def predict(self, image: np.ndarray, sample_id: str, event_uid: str, device: torch.device) -> np.ndarray:
        image_t = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            if self.model_family == "strong_visual":
                logits, _ = self.model(image_t)
            elif self.model_family == "strict_t2_postrgb_v4_pilot":
                assert self.mean is not None and self.std is not None
                zero = np.zeros_like(self.mean, dtype=np.float32)
                vec = self.sample_map.get(sample_id, self.event_map.get(event_uid, zero))
                vec = np.nan_to_num((vec - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                physics_t = torch.from_numpy(vec).unsqueeze(0).to(device)
                zero_meta = np.zeros((18,), dtype=np.float32)
                meta = self.sample_meta.get(sample_id, self.event_meta.get(event_uid, zero_meta))
                meta_t = torch.from_numpy(meta.astype(np.float32, copy=False)).unsqueeze(0).to(device)
                logits = self.model(image=image_t, physics=physics_t, meta=meta_t)["logits"]
            elif self.model_family == "phys_film_unet":
                assert self.mean is not None and self.std is not None
                zero = np.zeros_like(self.mean, dtype=np.float32)
                vec = self.sample_map.get(sample_id, self.event_map.get(event_uid, zero))
                vec = np.nan_to_num((vec - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                physics_t = torch.from_numpy(vec).unsqueeze(0).to(device)
                logits = self.model(image_t, physics_t)
            else:
                logits = self.model(image_t)
            pred = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= self.threshold)
        return pred


def load_model(summary_path: Path, device: torch.device) -> LoadedModel:
    summary = json.loads(summary_path.read_text())
    root = Path(_convert_wsl_path(summary["config"]["root"]))
    exp_dir_raw = Path(_convert_wsl_path(summary["outdir"]))
    if exp_dir_raw.is_absolute():
        exp_dir = exp_dir_raw
    else:
        candidates = [
            (Path.cwd() / exp_dir_raw).resolve(),
            (root.parent / exp_dir_raw).resolve(),
            summary_path.parent.resolve(),
        ]
        exp_dir = next((path for path in candidates if path.exists()), candidates[1])
    ckpt = torch.load(exp_dir / "best_model.pt", map_location=device, weights_only=False)

    model_family = str(summary.get("model_family", "small_unet"))
    sample_map: dict[str, np.ndarray] = {}
    event_map: dict[str, np.ndarray] = {}
    sample_meta: dict[str, np.ndarray] = {}
    event_meta: dict[str, np.ndarray] = {}
    mean = std = None
    threshold = float(summary.get("eval_threshold", 0.5))

    if model_family == "strong_visual":
        model = BinaryDeepLabV3(
            in_channels=3,
            backbone_name=str(summary.get("backbone", "deeplabv3_resnet50")),
            pretrained_backbone=False,
            aux_loss=True,
        )
    elif model_family == "strict_t2_postrgb_v4_pilot":
        cfg = summary.get("config", {})
        model = PhysicsPriorDeepLabV3(
            backbone_name=str(summary.get("backbone", "deeplabv3_resnet50")),
            pretrained_backbone=False,
            token_dim=int(cfg.get("token_dim", 96)),
            latent_dim=int(cfg.get("latent_dim", 128)),
            prior_fusion_scale=float(cfg.get("prior_fusion_scale", 1.0)),
            interaction_scale=float(cfg.get("interaction_scale", 0.25)),
        )
        physics_csv = resolve_path(root, summary.get("physics_csv", ""))
        sample_map, event_map = load_physics_maps(physics_csv, PHYSICS_VECTOR_COLUMNS)
        sample_meta, event_meta, _ = load_meta_maps(physics_csv)
        mean = np.asarray(summary["physics_mean"], dtype=np.float32)
        std = np.asarray(summary["physics_std"], dtype=np.float32)
    elif "physics_csv" in summary:
        model = PhysicsFiLMUNet(
            in_ch=3,
            physics_dim=int(summary["physics_dim"]),
            base=32,
            hidden_dim=int(summary["config"]["hidden_dim"]),
            dropout=float(summary["config"]["dropout"]),
        )
        physics_csv = resolve_path(root, summary.get("physics_csv", ""))
        physics_cols = list(summary.get("physics_vector_cols", []))
        sample_map, event_map = load_physics_maps(physics_csv, physics_cols)
        mean = np.asarray(summary["physics_norm"]["mean"], dtype=np.float32)
        std = np.asarray(summary["physics_norm"]["std"], dtype=np.float32)
        model_family = "phys_film_unet"
    else:
        model = SmallUNet(in_ch=3, base=32)

    if model_family == "strict_t2_postrgb_v4_pilot":
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    return LoadedModel(
        name=exp_dir.name,
        summary=summary,
        model=model,
        model_family=model_family,
        sample_map=sample_map,
        event_map=event_map,
        sample_meta=sample_meta,
        event_meta=event_meta,
        mean=mean,
        std=std,
        threshold=threshold,
    )


def choose_test_cache(summary: dict) -> Path:
    root_str = _convert_wsl_path(summary["config"]["root"])
    root = Path(root_str)
    patch_size = int(summary["config"]["patch_size"])
    return resolve_path(
        root,
        summary["config"].get("test_cache_h5", ""),
        fallback=root / "processed" / "hybrid_pinn" / "strict_t2_postrgb_eval_cache_v2" / f"test_postrgb_p{patch_size}.h5",
    )


def build_index(h5_path: Path) -> dict[str, int]:
    with h5py.File(h5_path, "r") as f:
        sample_ids = [v.decode("utf-8") for v in f["sample_id"][:]]
    return {sample_id: idx for idx, sample_id in enumerate(sample_ids)}


def load_case_arrays(h5_path: Path, index: int) -> dict[str, np.ndarray | str]:
    with h5py.File(h5_path, "r") as f:
        return {
            "image": f["image"][index],
            "mask": f["mask"][index, 0] >= 0.5,
            "valid": f["valid"][index, 0] >= 0.5,
            "sample_id": f["sample_id"][index].decode("utf-8"),
            "event_uid": f["event_uid"][index].decode("utf-8"),
            "dataset_id": f["dataset_id"][index].decode("utf-8"),
        }


def render_category_panel(
    df: pd.DataFrame,
    category_name: str,
    output_path: Path,
    cache_h5: Path,
    sample_index: dict[str, int],
    visual_model: LoadedModel,
    material_model: LoadedModel,
    full_model: LoadedModel,
    device: torch.device,
    tile_size: int = 176,
    label_w: int = 300,
    title: str | None = None,
) -> None:
    n = len(df)
    tile = tile_size
    header_h = max(42, tile // 4)
    row_h = tile + 24
    footer_h = 12
    headers = ["RGB", "GT", "Visual", "Material", "Full Physics"]
    canvas = Image.new("RGB", (label_w + 5 * tile, header_h + n * row_h + footer_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(16 if tile < 220 else 20)
    small_font = load_font(12 if tile < 220 else 15)
    score_font = load_font(11 if tile < 220 else 14)

    draw.text((12, 10), title or category_name, fill=(0, 0, 0), font=font)
    for col, title in enumerate(headers):
        x = label_w + col * tile + 8
        draw.text((x, 10), title, fill=(0, 0, 0), font=font)

    for row_idx, row in enumerate(df.itertuples(index=False)):
        sample_id = str(row.sample_id)
        dataset_name = str(getattr(row, "dataset", getattr(row, "dataset_id", "")))
        dataset_name = dataset_name.replace("DLR_Landslide_Ref_2025", "DLR").replace("CAS_Landslide", "CAS")
        idx = sample_index[sample_id]
        item = load_case_arrays(cache_h5, idx)
        image = item["image"]
        gt = item["mask"]
        valid = item["valid"]
        event_uid = str(item["event_uid"])

        pred_visual = visual_model.predict(image, sample_id, event_uid, device) & valid
        pred_material = material_model.predict(image, sample_id, event_uid, device) & valid
        pred_full = full_model.predict(image, sample_id, event_uid, device) & valid

        panels = [
            np.clip(image.transpose(1, 2, 0), 0.0, 1.0),
            overlay_rgb(image, gt, None, valid),
            overlay_rgb(image, gt, pred_visual, valid),
            overlay_rgb(image, gt, pred_material, valid),
            overlay_rgb(image, gt, pred_full, valid),
        ]
        y = header_h + row_idx * row_h
        panel_tag = str(getattr(row, "panel_tag", "")).strip()
        tag_prefix = f"{panel_tag} | " if panel_tag else ""
        draw.text(
            (10, y + 12),
            f"{tag_prefix}{dataset_name}\n{sample_id}\nstab={float(row.stability_proxy):.3f}",
            fill=(0, 0, 0),
            font=small_font,
        )
        for col, panel in enumerate(panels):
            x = label_w + col * tile
            img = panel_to_image(panel, tile - 8)
            canvas.paste(img, (x + 4, y + 4))
            draw.rectangle((x + 4, y + 4, x + tile - 4, y + tile - 4), outline=(180, 180, 180), width=1)
            if col == 2:
                label = f"IoU {float(row.visual_iou):.3f}"
            elif col == 3:
                label = f"IoU {float(row.material_iou):.3f}"
            elif col == 4:
                label = f"IoU {float(row.full_iou):.3f}"
            else:
                label = ""
            if label:
                text_bbox = draw.textbbox((0, 0), label, font=score_font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                box = (x + 8, y + 8, x + 18 + text_w, y + 14 + text_h)
                draw.rectangle(box, fill=(0, 0, 0))
                draw.text((x + 12, y + 10), label, fill=(255, 255, 255), font=score_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render strict_t2 post_rgb case panels")
    p.add_argument("--visual-summary", required=True)
    p.add_argument("--material-summary", required=True)
    p.add_argument("--full-summary", required=True)
    p.add_argument("--case-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--device", default="auto", choices=["cpu", "auto"])
    p.add_argument("--tile-size", type=int, default=176)
    p.add_argument("--label-width", type=int, default=300)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = Path(args.case_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    visual_model = load_model(Path(args.visual_summary), device)
    material_model = load_model(Path(args.material_summary), device)
    full_model = load_model(Path(args.full_summary), device)

    cache_h5 = choose_test_cache(visual_model.summary)
    sample_index = build_index(cache_h5)

    category_map = {
        "material_gain_over_visual_top.csv": "Material-only Gains Over Visual",
        "visual_gain_over_material_top.csv": "Visual Wins Over Material-only",
        "full_gain_high_instability_top.csv": "High-Instability Cases Where Full Physics Helps",
        "consensus_failures_top.csv": "Consensus Failures",
    }

    rendered = []
    for filename, title in category_map.items():
        df = pd.read_csv(case_dir / filename).head(args.top_k)
        png_path = out_dir / f"{Path(filename).stem}_top{args.top_k}.png"
        render_category_panel(
            df=df,
            category_name=title,
            output_path=png_path,
            cache_h5=cache_h5,
            sample_index=sample_index,
            visual_model=visual_model,
            material_model=material_model,
            full_model=full_model,
            device=device,
            tile_size=args.tile_size,
            label_w=args.label_width,
        )
        rendered.append((title, png_path))

    lines = [
        "# strict_t2 post_rgb Rendered Case Panels v1",
        "",
        f"- visual summary: `{args.visual_summary}`",
        f"- material summary: `{args.material_summary}`",
        f"- full summary: `{args.full_summary}`",
        f"- case source: `{case_dir}`",
        f"- top_k per category: `{args.top_k}`",
        "",
        "Rendered panels:",
        "",
    ]
    for title, png_path in rendered:
        lines.append(f"- {title}: `{png_path}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
