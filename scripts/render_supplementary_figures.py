#!/usr/bin/env python3
"""Render supplementary qualitative figures in the main-text visual style."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import platform

from render_strict_t2_postrgb_case_panels import (
    build_index,
    choose_test_cache,
    load_case_arrays,
    load_model as load_post_model,
    overlay_rgb,
    resolve_path,
)
from train_strict_t2_postrgb_phys_baseline import PhysicsFiLMUNet, load_physics_maps

ROOT = Path(__file__).resolve().parents[1]
PKG_FIG_DIR = ROOT / "docs" / "assets"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a cross-platform font with bold and regular fallbacks."""
    win_fonts = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    linux_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    candidates = win_fonts + linux_fonts if platform.system() == "Windows" else linux_fonts
    for raw in candidates:
        p = Path(raw)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


WHITE  = (255, 255, 255)
INK    = (32, 41, 50)
MUTED  = (98, 108, 121)
LINE   = (198, 206, 216)
GRID   = (236, 240, 244)
AMBER  = (232, 170, 76)
SLATE  = (123, 131, 145)


def save_outputs(img: Image.Image, stem: str, report_lines: list[str]) -> None:
    PKG_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PKG_FIG_DIR / f"{stem}.png"
    img.convert("RGB").save(out_png, dpi=(300, 300))
    out_report = PKG_FIG_DIR / f"{stem}_report.md"
    out_report.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")


def stretch_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    lo = np.percentile(arr, 2, axis=(0, 1))
    hi = np.percentile(arr, 98, axis=(0, 1))
    arr = (arr - lo) / (hi - lo + 1e-6)
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr ** 0.92
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    bd = np.zeros_like(mask, dtype=bool)
    bd[:-1, :] |= mask[:-1, :] != mask[1:, :]
    bd[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    bd |= np.roll(bd, 1, 0) | np.roll(bd, -1, 0) | np.roll(bd, 1, 1) | np.roll(bd, -1, 1)
    return bd


def focus_box(mask: np.ndarray, *, scale: float = 2.2) -> tuple[int, int, int, int]:
    h, w = mask.shape
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        side = int(max(y1 - y0, x1 - x0) * scale)
        side = max(side, 72)
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
    else:
        side = min(h, w)
        cy, cx = h / 2.0, w / 2.0
    half = side / 2.0
    y0 = max(0, int(round(cy - half)))
    x0 = max(0, int(round(cx - half)))
    y1 = min(h, y0 + side)
    x1 = min(w, x0 + side)
    y0 = max(0, y1 - side)
    x0 = max(0, x1 - side)
    return x0, y0, x1, y1


def crop_to_tile(arr: np.ndarray, box: tuple[int, int, int, int], size: int) -> Image.Image:
    x0, y0, x1, y1 = box
    crop = np.clip(arr[y0:y1, x0:x1], 0.0, 1.0)
    return Image.fromarray((crop * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)


def outlined_tile(img: Image.Image, size: tuple[int, int], *, outline: tuple[int, int, int] = LINE, width: int = 2) -> Image.Image:
    tile = img.resize(size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, WHITE)
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((1, 1, size[0] - 2, size[1] - 2), radius=10, outline=outline, width=width)
    return canvas


def draw_iou_chip(panel: Image.Image, value: float, font: ImageFont.ImageFont) -> Image.Image:
    panel = panel.convert("RGBA")
    draw = ImageDraw.Draw(panel)
    text = f"{value:.3f}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    x = panel.size[0] - tw - 18
    y = panel.size[1] - th - 16
    rect = (x - pad, y - pad, x + tw + pad, y + th + pad)
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(rect, radius=8, fill=(70, 70, 70, 200))
    panel = Image.alpha_composite(panel, overlay)
    draw = ImageDraw.Draw(panel)
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return panel.convert("RGB")


def load_phys_baseline_bundle(summary_path: Path, device: torch.device) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    root = Path(summary["config"]["root"])
    exp_dir_raw = Path(summary["outdir"])
    if exp_dir_raw.is_absolute():
        exp_dir = exp_dir_raw
    else:
        candidates = [
            (Path.cwd() / exp_dir_raw).resolve(),
            (root.parent / exp_dir_raw).resolve(),
            summary_path.parent.resolve(),
        ]
        exp_dir = next((path for path in candidates if path.exists()), candidates[1])
    model = PhysicsFiLMUNet(
        in_ch=3,
        physics_dim=int(summary["physics_dim"]),
        base=32,
        hidden_dim=int(summary["config"]["hidden_dim"]),
        dropout=float(summary["config"]["dropout"]),
        physics_cols=list(summary.get("physics_vector_cols", [])),
        dynamic_gate_mode=str(summary["config"].get("dynamic_gate_mode", "none")),
        dynamic_gate_source=str(summary["config"].get("dynamic_gate_source", "proxy")),
        dynamic_gate_scale=float(summary["config"].get("dynamic_gate_scale", 1.0)),
    )
    ckpt = torch.load(exp_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    physics_csv = resolve_path(root, summary.get("physics_csv", ""))
    sample_map, event_map = load_physics_maps(physics_csv, list(summary.get("physics_vector_cols", [])))
    mean = np.asarray(summary["physics_norm"]["mean"], dtype=np.float32)
    std = np.asarray(summary["physics_norm"]["std"], dtype=np.float32)
    return {
        "summary": summary,
        "model": model,
        "sample_map": sample_map,
        "event_map": event_map,
        "mean": mean,
        "std": std,
        "threshold": float(summary.get("eval_threshold", 0.5)),
    }


def predict_phys_baseline(bundle: dict[str, object], image: np.ndarray, sample_id: str, event_uid: str, device: torch.device) -> np.ndarray:
    x_t = torch.from_numpy(image).unsqueeze(0).to(device)
    zero = np.zeros_like(bundle["mean"], dtype=np.float32)
    vec = bundle["sample_map"].get(sample_id, bundle["event_map"].get(event_uid, zero))
    vec = np.nan_to_num((vec - bundle["mean"]) / bundle["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    physics_t = torch.from_numpy(vec).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = bundle["model"](x_t, physics_t)
    return torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= float(bundle["threshold"])


def load_v4_pair_bundle(summary_path: Path, device: torch.device):
    return load_post_model(summary_path, device)


def dataset_short_name(dataset_id: str) -> str:
    return (
        dataset_id.replace("DLR_Landslide_Ref_2025", "DLR")
        .replace("CAS_Landslide", "CAS")
        .replace("GLaD4CD_v1", "GLaD4CD")
    )


def render_s1() -> None:
    import pandas as pd

    device = torch.device("cpu")
    visual_summary = ROOT / "experiments" / "strict_t2_postrgb_visual_nearfull_bal2048_e3_v1" / "summary.json"
    material_summary = ROOT / "experiments" / "strict_t2_postrgb_physcache_material_nearfull_bal2048_e3_v1" / "summary.json"
    full_summary = ROOT / "experiments" / "strict_t2_postrgb_physcache_full_nearfull_bal2048_e3_v1" / "summary.json"

    visual_model = load_post_model(visual_summary, device)
    material_model = load_phys_baseline_bundle(material_summary, device)
    full_model = load_phys_baseline_bundle(full_summary, device)
    cache_h5 = choose_test_cache(visual_model.summary)
    sample_index = build_index(cache_h5)

    base = ROOT / "experiments" / "strict_t2_postrgb_case_panels_v1"
    def get_row(csv_name: str, sample_id: str) -> dict:
        df = pd.read_csv(base / csv_name)
        row = df[df["sample_id"] == sample_id].iloc[0].to_dict()
        return row

    groups = [
        ("(a) Visual-led cases", [
            ("DLR", "visual-led", get_row("visual_gain_over_material_top.csv", "EID_CN0001__SID_00042")),
            ("CAS", "visual-led", get_row("visual_gain_over_material_top.csv", "CAS_Palu::Palu0940")),
        ]),
        ("(b) Additional CAS full-physics gains", [
            ("CAS", "instability\ngain", get_row("full_gain_high_instability_top.csv", "CAS_Palu::Palu0898")),
            ("CAS", "staged full\ngain", get_row("full_gain_high_instability_top.csv", "CAS_Palu::Palu0504")),
        ]),
        ("(c) Consensus failures", [
            ("CAS", "hard failure", get_row("consensus_failures_top.csv", "CAS_Palu::Palu0584")),
            ("CAS", "hard failure", get_row("consensus_failures_top.csv", "CAS_Palu::Palu0483")),
        ]),
    ]
    tile         = 280
    gap          = 12
    stub_w       = 210   
    outer_margin = 36
    label_offset = 18    
    group_h      = 50    
    header_h     = 52    
    row_gap      = 14    
    group_gap    = 26    
    top_pad      = 36    

    cols       = ["Post-event", "Ground truth", "Visual", "Material", "Full physics"]
    total_rows = sum(len(rows) for _, rows in groups)
    canvas_w   = (2 * outer_margin + stub_w
                  + len(cols) * tile + (len(cols) - 1) * gap)
    canvas_h   = (top_pad + header_h
                  + len(groups) * group_h
                  + total_rows * tile
                  + (total_rows - len(groups)) * row_gap
                  + (len(groups) - 1) * group_gap
                  + 50)
    canvas = Image.new("RGB", (canvas_w, canvas_h), WHITE)
    draw   = ImageDraw.Draw(canvas)
    header_font   = load_font(28, bold=True)
    group_font    = load_font(28, bold=True)   
    row_tag_font  = load_font(28, bold=True)
    row_note_font = load_font(26, bold=False)  
    iou_font      = load_font(32, bold=True)
    start_x  = outer_margin + stub_w
    col_xs   = [start_x + i * (tile + gap) for i in range(len(cols))]
    header_y = top_pad
    for lab, x in zip(cols, col_xs):
        bbox = draw.textbbox((0, 0), lab, font=header_font)
        draw.text((x + (tile - (bbox[2] - bbox[0])) // 2, header_y),
                  lab, font=header_font, fill=INK)

    y = header_y + header_h
    report_lines = [
        "# figure_s1_additional_cases",
        "",
        "- status: regenerated in main-text Figure 7 style",
        "- purpose: additional visual-led cases, extra CAS full-physics gains, and consensus failures without repeating main-text cases",
        f"- visual summary: {visual_summary}",
        f"- material summary: {material_summary}",
        f"- full summary: {full_summary}",
        "",
        "| group | source | role | sample_id | visual_iou | material_iou | full_iou |",
        "|---|---|---|---|---:|---:|---:|",
    ]

    for group_title, rows in groups:
        draw.line((outer_margin, y + 18, canvas_w - outer_margin, y + 18), fill=GRID, width=2)
        draw.text((outer_margin, y), group_title, font=group_font, fill=INK)
        y += group_h

        for source, note, row in rows:
            sample_id = str(row["sample_id"])
            item = load_case_arrays(cache_h5, sample_index[sample_id])
            image    = np.asarray(item["image"], dtype=np.float32)
            gt       = np.asarray(item["mask"], dtype=bool)
            valid    = np.asarray(item["valid"], dtype=bool)
            event_uid = str(item["event_uid"])

            pred_visual   = visual_model.predict(image, sample_id, event_uid, device) & valid
            pred_material = predict_phys_baseline(material_model, image, sample_id, event_uid, device) & valid
            pred_full     = predict_phys_baseline(full_model, image, sample_id, event_uid, device) & valid
            rgb_hw = np.asarray(stretch_rgb(image[:3]), dtype=np.float32) / 255.0
            rgb_ch = np.transpose(rgb_hw, (2, 0, 1))
            box = focus_box(gt, scale=2.1)

            panels = [
                crop_to_tile(rgb_hw, box, tile),
                crop_to_tile(overlay_rgb(rgb_ch, gt, None, valid), box, tile),
                crop_to_tile(overlay_rgb(rgb_ch, gt, pred_visual, valid), box, tile),
                crop_to_tile(overlay_rgb(rgb_ch, gt, pred_material, valid), box, tile),
                crop_to_tile(overlay_rgb(rgb_ch, gt, pred_full, valid), box, tile),
            ]
            vals = [None, None, float(row["visual_iou"]), float(row["material_iou"]), float(row["full_iou"])]
            lx          = outer_margin + label_offset
            tag_h       = draw.textbbox((0, 0), source, font=row_tag_font)[3]
            note_lines  = note.split('\n')
            single_nh   = draw.textbbox((0, 0), note_lines[0], font=row_note_font)[3]
            note_total_h = single_nh * len(note_lines) + 4 * (len(note_lines) - 1)
            block_h     = tag_h + 8 + note_total_h
            tag_y       = y + (tile - block_h) // 2
            draw.text((lx, tag_y), source, font=row_tag_font, fill=INK)
            draw.multiline_text((lx, tag_y + tag_h + 8), note,
                                font=row_note_font, fill=MUTED, spacing=4)

            for x, panel, val in zip(col_xs, panels, vals):
                if val is not None:
                    panel = draw_iou_chip(panel, val, iou_font)
                canvas.paste(outlined_tile(panel, (tile, tile)), (x, y))

            report_lines.append(
                f"| {group_title} | {source} | {note} | {sample_id} | {float(row['visual_iou']):.6f} | "
                f"{float(row['material_iou']):.6f} | {float(row['full_iou']):.6f} |"
            )
            y += tile + row_gap
        y += group_gap - row_gap

    save_outputs(canvas, "figure_s1_additional_cases", report_lines)


def render_s2() -> None:
    import pandas as pd

    device = torch.device("cpu")
    visual_summary = ROOT / "experiments" / "strict_t2_postrgb_deeplabv3_resnet50_visual_e3_sp05_lb025_thr_seed20260311_localenv" / "summary.json"
    v4_summary = ROOT / "experiments" / "strict_t2_postrgb_deeplabv3_resnet50_v4_pilot_e3_sp05_lb025_distill_v1" / "summary.json"
    visual_model = load_v4_pair_bundle(visual_summary, device)
    v4_model = load_v4_pair_bundle(v4_summary, device)
    cache_h5 = choose_test_cache(visual_model.summary)
    sample_index = build_index(cache_h5)

    paired = pd.read_csv(ROOT / "experiments" / "strict_t2_postrgb_v4_vs_visual_paired_v1" / "paired_sample_mean_diff.csv")
    wanted = [
        ("DLR", "v4 gain", "EID_CN0001__SID_00100"),
        ("CAS", "v4 gain", "CAS_Palu::Palu0253"),
        ("GLaD4CD", "v4 gain", "GLADV1_0037::36"),
        ("DLR", "visual\nwins", "EID_CN0001__SID_00089"),
    ]
    metrics = {sid: paired[paired["sample_id"] == sid].iloc[0].to_dict() for _, _, sid in wanted}
    tile         = 360          
    gap          = 14
    stub_w       = 200          
    outer_margin = 36
    label_offset = 14           
    header_h     = 52           
    metric_h     = 64           
    row_gap      = 22           
    top_pad      = 110          

    cols = ["Post-event", "Ground truth", "DeepLabV3 visual", "GeoPhysAdapter-v4"]
    canvas_w = 2 * outer_margin + stub_w + len(cols) * tile + (len(cols) - 1) * gap + 200
    canvas_h = (top_pad + header_h
                + len(wanted) * (tile + metric_h)
                + (len(wanted) - 1) * row_gap
                + 50)
    canvas = Image.new("RGB", (canvas_w, canvas_h), WHITE)
    draw   = ImageDraw.Draw(canvas)
    title_font        = load_font(40, bold=True)
    header_font       = load_font(28, bold=True)
    row_tag_font      = load_font(28, bold=True)
    row_note_font     = load_font(28, bold=False)
    metric_label_font = load_font(26, bold=False)   
    metric_value_font = load_font(26, bold=True)    
    title_text = "Paired same-backbone residual-prior cases"
    tb = draw.textbbox((0, 0), title_text, font=title_font)
    draw.text(((canvas_w - (tb[2] - tb[0])) // 2, outer_margin),
              title_text, font=title_font, fill=INK)
    start_x  = outer_margin + stub_w
    col_xs   = [start_x + i * (tile + gap) for i in range(len(cols))]
    header_y = top_pad
    for lab, x in zip(cols, col_xs):
        bbox = draw.textbbox((0, 0), lab, font=header_font)
        draw.text((x + (tile - (bbox[2] - bbox[0])) // 2, header_y),
                  lab, font=header_font, fill=INK)

    y = header_y + header_h
    report_lines = [
        "# figure_s2_v4_vs_visual_cases",
        "",
        "- status: regenerated in main-text qualitative style",
        "- purpose: paired same-backbone visual vs v4 qualitative support without repeating main-text cases",
        f"- visual summary: {visual_summary}",
        f"- v4 summary: {v4_summary}",
        "",
        "| source | role | sample_id | visual_iou_mean | v4_iou_mean | mean_delta |",
        "|---|---|---|---:|---:|---:|",
    ]

    for source, note, sample_id in wanted:
        row       = metrics[sample_id]
        item      = load_case_arrays(cache_h5, sample_index[sample_id])
        image     = np.asarray(item["image"], dtype=np.float32)
        gt        = np.asarray(item["mask"], dtype=bool)
        valid     = np.asarray(item["valid"], dtype=bool)
        event_uid = str(item["event_uid"])

        pred_visual = visual_model.predict(image, sample_id, event_uid, device) & valid
        pred_v4     = v4_model.predict(image, sample_id, event_uid, device) & valid
        rgb_hw = np.asarray(stretch_rgb(image[:3]), dtype=np.float32) / 255.0
        rgb_ch = np.transpose(rgb_hw, (2, 0, 1))
        box = focus_box(gt, scale=2.1)
        panels = [
            crop_to_tile(rgb_hw, box, tile),
            crop_to_tile(overlay_rgb(rgb_ch, gt, None, valid), box, tile),
            crop_to_tile(overlay_rgb(rgb_ch, gt, pred_visual, valid), box, tile),
            crop_to_tile(overlay_rgb(rgb_ch, gt, pred_v4, valid), box, tile),
        ]
        lx          = outer_margin + label_offset
        tag_h       = draw.textbbox((0, 0), source, font=row_tag_font)[3]
        note_lines  = note.split('\n')
        single_nh   = draw.textbbox((0, 0), note_lines[0], font=row_note_font)[3]
        note_total_h = single_nh * len(note_lines) + 4 * (len(note_lines) - 1)
        block_h     = tag_h + 8 + note_total_h
        tag_y       = y + (tile - block_h) // 2
        draw.text((lx, tag_y), source, font=row_tag_font, fill=INK)
        draw.multiline_text((lx, tag_y + tag_h + 8), note,
                            font=row_note_font, fill=MUTED, spacing=4)

        for x, panel in zip(col_xs, panels):
            canvas.paste(outlined_tile(panel, (tile, tile)), (x, y))
        metric_y = y + tile + 12
        r        = 8   
        vx = col_xs[0] + 8   
        draw.ellipse((vx, metric_y + 4, vx + 2 * r, metric_y + 4 + 2 * r),
                     fill=SLATE, outline=SLATE)
        draw.text((vx + 2 * r + 8, metric_y),
                  "DeepLabV3 visual", font=metric_label_font, fill=MUTED)
        lb = draw.textbbox((0, 0), "DeepLabV3 visual", font=metric_label_font)
        draw.text((vx + 2 * r + 8 + (lb[2] - lb[0]) + 10, metric_y),
                  f"{float(row['visual_iou_mean']):.3f}",
                  font=metric_value_font, fill=SLATE)
        v4x = col_xs[2] + 8
        draw.ellipse((v4x, metric_y + 4, v4x + 2 * r, metric_y + 4 + 2 * r),
                     fill=AMBER, outline=AMBER)
        draw.text((v4x + 2 * r + 8, metric_y),
                  "GeoPhysAdapter-v4", font=metric_label_font, fill=MUTED)
        lb2   = draw.textbbox((0, 0), "GeoPhysAdapter-v4", font=metric_label_font)
        delta = float(row["mean_delta"])
        draw.text((v4x + 2 * r + 8 + (lb2[2] - lb2[0]) + 10, metric_y),
                  f"{float(row['v4_iou_mean']):.3f} ({delta:+.3f})",
                  font=metric_value_font, fill=AMBER)

        report_lines.append(
            f"| {source} | {note} | {sample_id} | {float(row['visual_iou_mean']):.6f} | "
            f"{float(row['v4_iou_mean']):.6f} | {float(row['mean_delta']):.6f} |"
        )
        y += tile + metric_h + row_gap

    save_outputs(canvas, "figure_s2_v4_vs_visual_cases", report_lines)


def main() -> int:
    render_s1()
    render_s2()
    print(PKG_FIG_DIR / "figure_s1_additional_cases.png")
    print(PKG_FIG_DIR / "figure_s2_v4_vs_visual_cases.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
