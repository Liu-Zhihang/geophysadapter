#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont


import platform
import re

# 自动适配Windows和Linux路径
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    ROOT = Path(r"D:\Project\滑坡检测")
else:
    ROOT = Path("/mnt/d/project/滑坡检测")
META_CSV = ROOT / "physics_informed_landslide_dataset/metadata/manifests/strict_t2_supervised_ready_test_index_v1.csv"
PAIRED_CSV = ROOT / "physics_informed_landslide_dataset/experiments/strict_t2_postrgb_v4_vs_visual_paired_v1/paired_sample_mean_diff.csv"
OUT_PNG = ROOT / "submission_package_jprs_v0_1/figures/figure5_case_candidate_preview.png"


def convert_path(path_str: str) -> Path:
    """将Linux路径转换为当前系统的路径格式"""
    if IS_WINDOWS and path_str.startswith("/mnt/"):
        # 转换 /mnt/d/... 为 D:\...
        match = re.match(r"/mnt/([a-z])/(.+)", path_str)
        if match:
            drive = match.group(1).upper()
            rest = match.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
    return Path(path_str)

CANDIDATES = [
    "EID_BR0001__SID_00380",
    "EID_BR0001__SID_00408",
    "EID_CN0001__SID_00101",
    "CAS_Palu::Palu0923",
    "CAS_Palu::Palu0930",
    "CAS_Palu::Palu0939",
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for raw in candidates:
        path = Path(raw)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def stretch_rgb(rgb: np.ndarray, qlo: float = 2.0, qhi: float = 98.0) -> np.ndarray:
    arr = rgb.astype(np.float32)
    out = np.zeros_like(arr)
    for c in range(arr.shape[2]):
        band = arr[:, :, c]
        lo, hi = np.percentile(band[np.isfinite(band)], [qlo, qhi])
        if hi <= lo:
            out[:, :, c] = np.clip(band, 0, 1)
        else:
            out[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def mask_from_label(label: np.ndarray) -> np.ndarray:
    arr = np.asarray(label)
    if arr.ndim == 3:
        return np.any(arr > 0, axis=0) if arr.shape[0] <= 4 else np.any(arr > 0, axis=2)
    return arr > 0


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return m
    core = m.copy()
    core[1:, :] &= m[:-1, :]
    core[:-1, :] &= m[1:, :]
    core[:, 1:] &= m[:, :-1]
    core[:, :-1] &= m[:, 1:]
    return m & (~core)


def overlay_boundary(rgb: np.ndarray, mask: np.ndarray, color=(235, 132, 48), alpha=0.95) -> np.ndarray:
    out = rgb.copy()
    bnd = boundary_mask(mask)
    color_arr = np.array(color, dtype=np.float32) / 255.0
    out[bnd] = (1 - alpha) * out[bnd] + alpha * color_arr
    return out


def crop_focus(rgb: np.ndarray, mask: np.ndarray, out_size: int = 256) -> np.ndarray:
    h, w = mask.shape
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        bh, bw = y1 - y0, x1 - x0
        side = int(max(bh, bw) * 2.2)
        side = max(side, 72)
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
    else:
        side = min(h, w)
        cy, cx = h // 2, w // 2
    half = side // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(h, y0 + side)
    x1 = min(w, x0 + side)
    y0 = max(0, y1 - side)
    x0 = max(0, x1 - side)
    crop = rgb[y0:y1, x0:x1]
    im = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((out_size, out_size), Image.Resampling.LANCZOS))


def read_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    return arr


def load_case_assets(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    kind = row["sample_kind"]
    if kind == "dlr_h5_patch":
        h5_path = convert_path(row["h5_path"])
        idx = int(row["h5_sample_index"])
        with h5py.File(h5_path, "r") as f:
            r = f["POST1_B04"][idx, 0]
            g = f["POST1_B03"][idx, 0]
            b = f["POST1_B02"][idx, 0]
            rgb = np.stack([r, g, b], axis=-1)
            mask = f["None_MASK"][idx, 0] > 0.5
        return stretch_rgb(rgb), mask
    if kind == "cas_single_rgb":
        rgb_arr = read_tif(str(convert_path(row["image_path"])))
        if rgb_arr.shape[0] >= 3:
            rgb = np.moveaxis(rgb_arr[:3], 0, -1)
        else:
            rgb = np.repeat(rgb_arr[:1], 3, axis=0)
            rgb = np.moveaxis(rgb, 0, -1)
        label = read_tif(str(convert_path(row["label_path"])))
        # CAS labels are white-background / black-landslide TIFFs.
        return stretch_rgb(rgb), (~mask_from_label(label))
    if kind == "glad_pre_post":
        rgb_arr = read_tif(str(convert_path(row["post_path"])))
        rgb = np.moveaxis(rgb_arr[:3], 0, -1)
        label = read_tif(str(convert_path(row["label_path"])))
        return stretch_rgb(rgb), mask_from_label(label)
    raise ValueError(f"Unsupported sample kind: {kind}")


def short_label(sample_id: str) -> str:
    if sample_id.startswith("EID_"):
        return sample_id.replace("__SID_", " / ")
    if sample_id.startswith("CAS_Palu::"):
        return sample_id.replace("CAS_Palu::", "CAS / ")
    return sample_id.replace("::", " / ")


def main() -> int:
    meta = pd.read_csv(META_CSV)
    paired = pd.read_csv(PAIRED_CSV).set_index("sample_id")
    rows = []
    for sid in CANDIDATES:
        row = meta.loc[meta["sample_id"] == sid]
        if row.empty:
            raise SystemExit(f"Missing manifest row for {sid}")
        rows.append(row.iloc[0])

    margin = 28
    gutter = 22
    title_h = 54
    info_h = 58
    tile = 256
    cols = 3
    rows_n = math.ceil(len(rows) / cols)
    canvas_w = margin * 2 + cols * tile + (cols - 1) * gutter
    canvas_h = margin * 2 + title_h + rows_n * (tile + info_h) + (rows_n - 1) * 30
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    label_font = load_font(19, bold=True)
    meta_font = load_font(15)
    chip_font = load_font(15, bold=True)

    draw.text((margin, margin), "Figure 5 candidate cases: post-event visibility preview", fill=(20, 28, 36), font=title_font)

    amber = (234, 141, 53)
    slate = (63, 74, 88)
    chip_fill = (255, 247, 238)
    chip_outline = (244, 192, 132)

    for i, row in enumerate(rows):
        rgb, mask = load_case_assets(row)
        over = overlay_boundary(rgb, mask, color=amber)
        crop = crop_focus(over, mask, out_size=tile)
        r = i // cols
        c = i % cols
        x = margin + c * (tile + gutter)
        y = margin + title_h + r * (tile + info_h + 30)
        card = Image.new("RGB", (tile, tile), (255, 255, 255))
        card.paste(Image.fromarray(crop), (0, 0))
        canvas.paste(card, (x, y))
        draw.rounded_rectangle((x, y, x + tile, y + tile), radius=10, outline=(220, 226, 234), width=2)

        sid = row["sample_id"]
        delta = float(paired.loc[sid, "mean_delta"]) if sid in paired.index else float("nan")
        v_iou = float(paired.loc[sid, "visual_iou_mean"]) if sid in paired.index else float("nan")
        p_iou = float(paired.loc[sid, "v4_iou_mean"]) if sid in paired.index else float("nan")
        label_y = y + tile + 8
        draw.text((x, label_y), short_label(sid), fill=slate, font=label_font)

        chip_text = f"visual {v_iou:.3f}  →  prior {p_iou:.3f}   Δ+{delta:.3f}"
        tw = draw.textbbox((0, 0), chip_text, font=chip_font)[2]
        chip_x0 = x
        chip_y0 = label_y + 26
        chip_x1 = x + min(tile, tw + 22)
        chip_y1 = chip_y0 + 26
        draw.rounded_rectangle((chip_x0, chip_y0, chip_x1, chip_y1), radius=13, fill=chip_fill, outline=chip_outline, width=2)
        draw.text((chip_x0 + 10, chip_y0 + 4), chip_text, fill=amber, font=chip_font)

        note = "DLR" if "EID_" in sid else ("CAS" if "CAS_" in sid else "GLaD4CD")
        draw.text((x + tile - 54, y + 10), note, fill=(255, 255, 255), font=meta_font)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG)
    print(OUT_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
