#!/usr/bin/env python3
"""Render a sample-level UGCoP mechanism landscape preview."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PKG_FIG_DIR = ROOT / "docs" / "assets"

WHITE = (251, 251, 249)
INK = (25, 30, 36)
MUTED = (88, 96, 106)
LINE = (194, 199, 205)
GRID = (228, 232, 236)
BLUE = (78, 132, 214)
TEAL = (52, 148, 134)
AMBER = (224, 167, 76)
ROSE = (185, 91, 91)
SLATE = (129, 136, 149)
BLUE_LIGHT = (230, 239, 252)
TEAL_LIGHT = (225, 244, 239)
AMBER_LIGHT = (249, 239, 218)
ROSE_LIGHT = (247, 232, 232)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    win_fonts = [
        "C:/Windows/Fonts/arial.ttf" if not bold else "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibri.ttf" if not bold else "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/segoeui.ttf" if not bold else "C:/Windows/Fonts/segoeuib.ttf",
    ]
    linux_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    candidates = win_fonts + linux_fonts if platform.system() == "Windows" else linux_fonts
    for raw in candidates:
        path = Path(raw)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def blend(base: tuple[int, int, int], target: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(base[i] + (target[i] - base[i]) * t) for i in range(3))


def source_family(sample_id: str) -> str:
    if sample_id.startswith("EID_"):
        return "DLR"
    if sample_id.startswith("CAS_Palu::"):
        return "CAS_Palu"
    if sample_id.startswith("GLADV1_"):
        return "GLaD4CD"
    return "Other"


def smooth2d(arr: np.ndarray, passes: int = 3) -> np.ndarray:
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    kernel /= kernel.sum()
    out = arr.astype(float)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        nxt = np.zeros_like(out)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                nxt[i, j] = float((padded[i : i + 3, j : j + 3] * kernel).sum())
        out = nxt
    return out


def smooth1d(arr: np.ndarray, passes: int = 3) -> np.ndarray:
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    out = arr.astype(float)
    for _ in range(passes):
        padded = np.pad(out, 2, mode="edge")
        nxt = np.zeros_like(out)
        for i in range(out.shape[0]):
            nxt[i] = float((padded[i : i + 5] * kernel).sum())
        out = nxt
    return out


def expanded_percentiles(values: np.ndarray) -> np.ndarray:
    """Spread tied values across their ECDF interval for visualization."""
    vals = np.asarray(values, dtype=float)
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    n = len(vals)
    out = np.zeros(n, dtype=float)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        lo = 100.0 * start / n
        hi = 100.0 * end / n
        count = end - start
        if count == 1:
            spread = np.array([(lo + hi) * 0.5], dtype=float)
        else:
            spread = np.linspace(lo + (hi - lo) / (count + 1), hi - (hi - lo) / (count + 1), count)
        out[order[start:end]] = spread
        start = end
    return out


def gain_color(v: float, lo: float, hi: float) -> tuple[int, int, int]:
    anchors = [
        (lo, (168, 66, 66)),
        (-0.01, (228, 170, 163)),
        (0.00, (248, 248, 246)),
        (0.04, (118, 205, 188)),
        (hi, (18, 88, 130)),
    ]
    if v <= anchors[0][0]:
        return anchors[0][1]
    for (v0, c0), (v1, c1) in zip(anchors[:-1], anchors[1:]):
        if v <= v1:
            frac = 0.0 if v1 == v0 else (v - v0) / (v1 - v0)
            return blend(c0, c1, frac)
    return anchors[-1][1]


def density_color(v: float) -> tuple[int, int, int]:
    anchors = [
        (0.00, WHITE),
        (0.18, (229, 244, 240)),
        (0.45, (172, 222, 208)),
        (0.72, (79, 161, 148)),
        (1.00, (33, 96, 128)),
    ]
    for (t0, c0), (t1, c1) in zip(anchors[:-1], anchors[1:]):
        if v <= t1:
            frac = 0 if t1 == t0 else (v - t0) / (t1 - t0)
            return blend(c0, c1, frac)
    return anchors[-1][1]


def draw_rotated_text(canvas: Image.Image, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], angle: int = 90) -> None:
    dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 8
    h = bbox[3] - bbox[1] + 8
    txt = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    td = ImageDraw.Draw(txt)
    td.text((4, 4), text, font=font, fill=fill)
    rot = txt.rotate(angle, expand=1)
    canvas.alpha_composite(rot, dest=xy)


def project_iso(x: float, y: float, z: float, origin: tuple[float, float], sx: float, sy: float, sz: float) -> tuple[float, float]:
    ox, oy = origin
    return ox + (x - y) * sx, oy + (x + y) * sy - z * sz


def trace_mask_loops(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Trace closed boundary loops of a binary mask on a square lattice."""
    h, w = mask.shape
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add_seg(p0: tuple[int, int], p1: tuple[int, int]) -> None:
        adj.setdefault(p0, []).append(p1)
        adj.setdefault(p1, []).append(p0)

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            if y == 0 or not mask[y - 1, x]:
                add_seg((x, y), (x + 1, y))
            if y == h - 1 or not mask[y + 1, x]:
                add_seg((x, y + 1), (x + 1, y + 1))
            if x == 0 or not mask[y, x - 1]:
                add_seg((x, y), (x, y + 1))
            if x == w - 1 or not mask[y, x + 1]:
                add_seg((x + 1, y), (x + 1, y + 1))

    loops: list[list[tuple[int, int]]] = []
    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    for start, nbrs in adj.items():
        for nxt in nbrs:
            key = edge_key(start, nxt)
            if key in used:
                continue
            loop = [start, nxt]
            used.add(key)
            prev, cur = start, nxt
            closed = False
            while True:
                options = [p for p in adj.get(cur, []) if p != prev]
                if not options:
                    break
                if len(options) == 1:
                    nxt2 = options[0]
                else:
                    # Prefer keeping direction smooth.
                    vx, vy = cur[0] - prev[0], cur[1] - prev[1]
                    best = None
                    best_score = None
                    for cand in options:
                        wx, wy = cand[0] - cur[0], cand[1] - cur[1]
                        score = vx * wx + vy * wy
                        if best is None or score > best_score:
                            best = cand
                            best_score = score
                    nxt2 = best
                key2 = edge_key(cur, nxt2)
                if key2 in used:
                    if nxt2 == start:
                        loop.append(start)
                        closed = True
                    break
                loop.append(nxt2)
                used.add(key2)
                prev, cur = cur, nxt2
                if cur == start:
                    closed = True
                    break
            if closed and len(loop) >= 8:
                loops.append(loop)
    return loops


def save(canvas: Image.Image, stem: str, report_lines: list[str]) -> None:
    PKG_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PKG_FIG_DIR / f"{stem}.png"
    md_path = PKG_FIG_DIR / f"{stem}_report.md"
    canvas.convert("RGB").save(png_path, dpi=(300, 300))
    md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def render() -> None:
    case_csv = ROOT / "experiments" / "strict_t2_postrgb_case_panels_v1" / "merged_case_metrics.csv"
    mech_csv = ROOT / "experiments" / "strict_t2_postrgb_v4_support_uncertainty_summary_v1" / "per_sample_support_uncertainty.csv"
    uq_csv = ROOT / "experiments" / "strict_t2_postrgb_v4_support_uncertainty_summary_v1" / "uncertainty_quartiles.csv"

    case_df = pd.read_csv(case_csv)
    mech_df = pd.read_csv(mech_csv)
    uq_df = pd.read_csv(uq_csv).sort_values("quartile")

    case_df["source_family"] = case_df["sample_id"].map(source_family)
    case_df["best_iou"] = case_df[["visual_iou", "material_iou", "full_iou"]].max(axis=1)

    x = case_df["material_minus_visual_iou"].to_numpy()
    y = case_df["full_minus_material_iou"].to_numpy()
    s = case_df["best_iou"].to_numpy()
    family = case_df["source_family"].to_numpy()

    x_lo = float(np.quantile(x, 0.01) - 0.01)
    x_hi = float(np.quantile(x, 0.99) + 0.01)
    y_lo = float(np.quantile(y, 0.05) - 0.02)
    y_hi = float(np.quantile(y, 0.99) + 0.02)

    q_tr = float(((x > 0) & (y > 0)).mean() * 100.0)
    q_tl = float(((x < 0) & (y > 0)).mean() * 100.0)
    q_bl = float(((x < 0) & (y < 0)).mean() * 100.0)
    q_br = float(((x > 0) & (y < 0)).mean() * 100.0)

    ux_raw = mech_df["uncertainty_mean"].to_numpy()
    ux_rank = expanded_percentiles(ux_raw)
    vy_rank = expanded_percentiles(mech_df["visual_iou_mean"].to_numpy())
    dy = mech_df["mean_delta"].to_numpy()
    d_lo = float(np.quantile(dy, 0.01) - 0.01)
    d_hi = float(np.quantile(dy, 0.99) + 0.01)
    canvas = Image.new("RGBA", (2300, 1240), WHITE + (255,))
    draw = ImageDraw.Draw(canvas)

    panel_font  = load_font(38, bold=True)
    axis_font   = load_font(28, bold=True)   
    tick_font   = load_font(24)              
    tick_bold   = load_font(24, bold=True)   
    body_font   = load_font(22)
    small_font  = load_font(22)
    value_font  = load_font(26, bold=True)
    quad_body_font = load_font(24)           
    cb_font        = load_font(22, bold=True)  

    margin = 56          
    gutter = 40
    panel_top = 120
    panel_h   = 1050
    panel_w   = (2300 - 2 * margin - gutter) // 2
    panel_a = (margin, panel_top, margin + panel_w, panel_top + panel_h)
    panel_b = (panel_a[2] + gutter, panel_top, 2300 - margin, panel_top + panel_h)

    for rect in [panel_a, panel_b]:
        draw.rounded_rectangle(rect, radius=20, fill=WHITE, outline=LINE, width=2)

    # Panel A
    title_a = "(a) Component-correction mechanism space"
    ta_box = draw.textbbox((0, 0), title_a, font=panel_font)
    title_a_h = ta_box[3] - ta_box[1]
    draw.text(
        (panel_a[0], panel_a[1] - title_a_h - 24),
        title_a,
        font=panel_font,
        fill=INK,
    )
    a_plot = (panel_a[0] + 110, panel_a[1] + 48, panel_a[2] - 20, panel_a[3] - 180)

    # quadrant background
    x0 = a_plot[0] + int((0 - x_lo) / (x_hi - x_lo) * (a_plot[2] - a_plot[0]))
    y0 = a_plot[3] - int((0 - y_lo) / (y_hi - y_lo) * (a_plot[3] - a_plot[1]))
    draw.rectangle((a_plot[0], a_plot[1], x0, y0), fill=ROSE_LIGHT)   # TL
    draw.rectangle((x0, a_plot[1], a_plot[2], y0), fill=TEAL_LIGHT)   # TR
    draw.rectangle((a_plot[0], y0, x0, a_plot[3]), fill=SLATE)  # placeholder overwritten below
    draw.rectangle((a_plot[0], y0, x0, a_plot[3]), fill=blend(WHITE, SLATE, 0.10))
    draw.rectangle((x0, y0, a_plot[2], a_plot[3]), fill=AMBER_LIGHT)  # BR

    # grid and zero lines
    xticks = [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    yticks = [-0.20, -0.10, 0.0, 0.10, 0.20]

    def ax(v: float) -> int:
        v = max(x_lo, min(x_hi, v))
        frac = (v - x_lo) / (x_hi - x_lo)
        return int(round(a_plot[0] + frac * (a_plot[2] - a_plot[0])))

    def ay(v: float) -> int:
        v = max(y_lo, min(y_hi, v))
        frac = (v - y_lo) / (y_hi - y_lo)
        return int(round(a_plot[3] - frac * (a_plot[3] - a_plot[1])))

    for t in xticks:
        xx = ax(t)
        draw.line((xx, a_plot[1], xx, a_plot[3]), fill=LINE if abs(t) < 1e-9 else GRID, width=2 if abs(t) < 1e-9 else 1)
        label = "0" if abs(t) < 1e-9 else f"{t:+.2f}".replace("+", "")
        lb = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((xx - (lb[2] - lb[0]) // 2, a_plot[3] + 16), label, font=tick_font, fill=MUTED)
    for t in yticks:
        yy = ay(t)
        draw.line((a_plot[0], yy, a_plot[2], yy), fill=LINE if abs(t) < 1e-9 else GRID, width=2 if abs(t) < 1e-9 else 1)
        label = "0" if abs(t) < 1e-9 else f"{t:+.2f}".replace("+", "")
        lb = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((a_plot[0] - (lb[2] - lb[0]) - 10, yy - (lb[3] - lb[1]) // 2), label, font=tick_font, fill=MUTED)

    # scatter layer
    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    od = ImageDraw.Draw(overlay)
    colors = {"DLR": BLUE, "CAS_Palu": TEAL, "GLaD4CD": AMBER, "Other": SLATE}
    s_min, s_max = float(s.min()), float(s.max())
    for xv, yv, sv, fam in zip(x, y, s, family):
        px, py = ax(float(xv)), ay(float(yv))
        r = 5 + 18 * math.sqrt((float(sv) - s_min) / max(1e-6, (s_max - s_min)))
        max_rx = max(1.0, min(px - a_plot[0], a_plot[2] - px))
        max_ry = max(1.0, min(py - a_plot[1], a_plot[3] - py))
        r = min(r, max_rx, max_ry)
        c = colors.get(str(fam), SLATE)
        od.ellipse((px - r, py - r, px + r, py + r), fill=c + (145,), outline=(255, 255, 255, 210), width=2)
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)
    quad_font = load_font(26, bold=True)
    QUAD_BG_RGBA = (210, 212, 216, 150)

    quad_labels = [
        ((a_plot[0] + 12,        a_plot[1] + 14), f"{q_tl:.1f}%", "full rescues weak material", ROSE),
        ((a_plot[2] - 280,       a_plot[1] + 14), f"{q_tr:.1f}%", "staged physics gain",         TEAL),
        ((a_plot[0] + 12,        a_plot[3] - 82), f"{q_bl:.1f}%", "visual-led regime",            SLATE),
        ((a_plot[2] - 280,       a_plot[3] - 82), f"{q_br:.1f}%", "material enough",              AMBER),
    ]
    quad_bg_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    qbd = ImageDraw.Draw(quad_bg_layer)
    for (xy, pct, label, color) in quad_labels:
        pct_box  = draw.textbbox((0, 0), pct,   font=quad_font)
        lab_box  = draw.textbbox((0, 0), label, font=quad_body_font)
        width  = max(pct_box[2] - pct_box[0], lab_box[2] - lab_box[0]) + 22
        height = (pct_box[3] - pct_box[1]) + (lab_box[3] - lab_box[1]) + 20
        qbd.rounded_rectangle((xy[0], xy[1], xy[0] + width, xy[1] + height),
                               radius=12, fill=QUAD_BG_RGBA, outline=None)
    canvas = Image.alpha_composite(canvas, quad_bg_layer)
    draw = ImageDraw.Draw(canvas)
    for (xy, pct, label, color) in quad_labels:
        draw.text((xy[0] + 10, xy[1] + 4),  pct,   font=quad_font,      fill=color)
        draw.text((xy[0] + 10, xy[1] + 34), label, font=quad_body_font, fill=MUTED)
    legend_font = load_font(24, bold=True)
    legend_items = [("DLR", BLUE), ("CAS Palu", TEAL), ("GLaD4CD", AMBER)]
    dot_r = 13
    item_gaps = 48
    item_widths: list[int] = []
    for name, _col in legend_items:
        box = draw.textbbox((0, 0), name, font=legend_font)
        item_widths.append((box[2] - box[0]) + dot_r * 2 + 12)
    total_width = sum(item_widths) + item_gaps * (len(legend_items) - 1)
    legend_y = a_plot[3] + 56
    legend_x_start = a_plot[0] + (a_plot[2] - a_plot[0] - total_width) / 2

    x_cursor = legend_x_start
    for (name, col), item_w in zip(legend_items, item_widths):
        cy = int(legend_y)
        draw.ellipse((x_cursor, cy, x_cursor + dot_r * 2, cy + dot_r * 2), fill=col, outline=WHITE, width=2)
        draw.text((x_cursor + dot_r * 2 + 8, cy - 2), name, font=legend_font, fill=INK)
        x_cursor += item_w + item_gaps
    xtitle = "ΔIoU(material − visual)"
    xtb = draw.textbbox((0, 0), xtitle, font=axis_font)
    xtitle_x = a_plot[0] + (a_plot[2] - a_plot[0] - (xtb[2] - xtb[0])) // 2
    draw.text((xtitle_x, a_plot[3] + 114), xtitle, font=axis_font, fill=INK)
    draw_rotated_text(canvas, (a_plot[0] - 104, a_plot[1] + 160), "ΔIoU(full physics − material)", axis_font, INK, angle=90)

    # Panel B
    title_b = "(b) Residual gain landscape under uncertainty"
    tb_box = draw.textbbox((0, 0), title_b, font=panel_font)
    title_b_h = tb_box[3] - tb_box[1]
    draw.text(
        (panel_b[0], panel_b[1] - title_b_h - 24),
        title_b,
        font=panel_font,
        fill=INK,
    )
    b_plot = (panel_b[0] + 90, panel_b[1] + 55, panel_b[2] - 100, panel_b[3] - 130)
    cb_x1 = panel_b[2] - 82
    cb_x2 = panel_b[2] - 46

    def bx(v: float) -> int:
        frac = float(v) / 100.0
        return int(round(b_plot[0] + frac * (b_plot[2] - b_plot[0])))

    def by(v: float) -> int:
        v = max(0.0, min(100.0, v))
        frac = v / 100.0
        return int(round(b_plot[3] - frac * (b_plot[3] - b_plot[1])))

    x_ticks = [0, 25, 50, 75, 100]
    y_ticks = [0, 25, 50, 75, 100]
    for t in x_ticks:
        xx = bx(t)
        if t not in (0, 100):
            draw.line((xx, b_plot[1], xx, b_plot[3]), fill=GRID, width=1)
        label = f"{t}"
        lb = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((xx - (lb[2] - lb[0]) // 2, b_plot[3] + 14), label, font=tick_font, fill=MUTED)
    for t in y_ticks:
        yy = by(t)
        if t not in (0, 100):
            draw.line((b_plot[0], yy, b_plot[2], yy), fill=GRID, width=1)
        label = f"{t}"
        lb = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((b_plot[0] - (lb[2] - lb[0]) - 12, yy - (lb[3] - lb[1]) // 2), label, font=tick_font, fill=MUTED)
    bxtitle = "Uncertainty percentile (low → high)"
    bxtb = draw.textbbox((0, 0), bxtitle, font=axis_font)
    bxtitle_x = b_plot[0] + (b_plot[2] - b_plot[0] - (bxtb[2] - bxtb[0])) // 2
    draw.text((bxtitle_x, b_plot[3] + 54), bxtitle, font=axis_font, fill=INK)
    draw_rotated_text(canvas, (b_plot[0] - 78, b_plot[1] + 130), "Visual-baseline percentile (weak → strong)", axis_font, INK, angle=90)
    nx, ny = 56, 56
    x_edges = np.linspace(0, 100, nx + 1)
    y_edges = np.linspace(0, 100, ny + 1)
    sum_grid = np.zeros((ny, nx), dtype=float)
    count_grid = np.zeros((ny, nx), dtype=float)
    den_grid = np.zeros((ny, nx), dtype=float)
    xi = np.clip(np.digitize(ux_rank, x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(vy_rank, y_edges) - 1, 0, ny - 1)
    for xx_i, yy_i, delta in zip(xi, yi, dy):
        sum_grid[yy_i, xx_i] += float(delta)
        count_grid[yy_i, xx_i] += 1.0
        den_grid[yy_i, xx_i] += 1.0

    sum_s = smooth2d(sum_grid, passes=4)
    count_s = smooth2d(count_grid, passes=4)
    den_s = smooth2d(den_grid, passes=3)
    field = np.divide(sum_s, np.maximum(count_s, 1e-6))
    den_norm = den_s / max(1e-6, float(den_s.max()))
    color_lo = float(np.quantile(dy, 0.05))
    color_hi = float(np.quantile(dy, 0.95))
    plot_w = b_plot[2] - b_plot[0]
    plot_h = b_plot[3] - b_plot[1]
    fine_w = max(plot_w, 460)
    fine_h = max(plot_h, 460)
    field_norm = np.clip((field - color_lo) / max(1e-6, (color_hi - color_lo)), 0.0, 1.0)
    dens_norm = np.clip(den_norm, 0.0, 1.0)
    field_big = np.asarray(
        Image.fromarray((field_norm * 255).astype(np.uint8)).convert("L").resize((fine_w, fine_h), resample=Image.Resampling.BICUBIC),
        dtype=float,
    ) / 255.0
    dens_big = np.asarray(
        Image.fromarray((dens_norm * 255).astype(np.uint8)).convert("L").resize((fine_w, fine_h), resample=Image.Resampling.BICUBIC),
        dtype=float,
    ) / 255.0
    field_vals = color_lo + field_big * (color_hi - color_lo)
    elev = np.clip(dens_big, 0.0, 1.0) ** 0.78
    gy, gx = np.gradient(elev)
    relief = 0.82 + 0.52 * np.clip((-0.88 * gx + 0.56 * gy), -0.55, 0.55)
    relief = np.clip(relief, 0.62, 1.28)

    # continuous pale terrain floor so low-density areas are not left as blank white paper
    terrain_floor = Image.new("RGBA", (plot_w, plot_h), (255, 255, 255, 0))
    floor_px = terrain_floor.load()
    for py in range(plot_h):
        for px_i in range(plot_w):
            base = gain_color(float(field_vals[py, px_i]), color_lo, color_hi)
            toned = blend((226, 236, 239), base, 0.38)
            floor_px[px_i, py] = toned + (150,)
    canvas.alpha_composite(terrain_floor, dest=(b_plot[0], b_plot[1]))
    draw = ImageDraw.Draw(canvas)

    terrain = Image.new("RGBA", (plot_w, plot_h), (255, 255, 255, 0))
    px = terrain.load()
    for py in range(plot_h):
        for px_i in range(plot_w):
            dens = elev[py, px_i]
            base = gain_color(float(field_vals[py, px_i]), color_lo, color_hi)
            lit = tuple(max(0, min(255, int(c * relief[py, px_i]))) for c in base)
            # Keep a continuous colored floor, but let higher density terrain saturate more strongly.
            blend_t = 0.34 + 0.70 * (dens ** 0.82)
            toned = blend((239, 244, 243), lit, blend_t)
            alpha = int(108 + 182 * (dens ** 0.88))
            px[px_i, py] = toned + (alpha,)
    canvas.alpha_composite(terrain, dest=(b_plot[0], b_plot[1]))
    draw = ImageDraw.Draw(canvas)

    # contour rings based on density-elevation masks for a terrain-map feel
    contour_overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    cod = ImageDraw.Draw(contour_overlay)
    elev_nonzero = elev[elev > 0.02]
    contour_levels = np.quantile(
        elev_nonzero,
        [0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82, 0.89, 0.94, 0.975],
    )
    contour_levels = np.unique(np.clip(contour_levels, 0.0, 1.0))
    sx = plot_w / fine_w
    sy = plot_h / fine_h
    for idx, level in enumerate(contour_levels):
        mask = elev >= float(level)
        loops = trace_mask_loops(mask)
        is_major = (idx % 3 == 0)
        for loop in loops:
            # Skip open-border shapes and tiny specks so the visible rings read as closed terrain contours.
            xs = [p[0] for p in loop]
            ys = [p[1] for p in loop]
            if min(xs) <= 1 or min(ys) <= 1 or max(xs) >= fine_w - 1 or max(ys) >= fine_h - 1:
                continue
            if len(loop) < (22 if is_major else 14):
                continue
            pts: list[tuple[int, int]] = []
            step = 1 if is_major else 1
            for x0, y0 in loop[::step]:
                pts.append((b_plot[0] + int(round(x0 * sx)), b_plot[1] + int(round(y0 * sy))))
            if len(pts) < 3:
                continue
            alpha = int((162 if is_major else 118) + (70 if is_major else 42) * float(level))
            fill = (36, 40, 46, alpha)
            width = 3 if is_major else 1
            cod.line(pts, fill=fill, width=width, joint="curve")
    canvas = Image.alpha_composite(canvas, contour_overlay)
    draw = ImageDraw.Draw(canvas)

    # Subtle cool atmospheric wash to unify the terrain field.
    floor_tint = Image.new("RGBA", (plot_w, plot_h), blend((222, 234, 238), (194, 216, 224), 0.78) + (52,))
    canvas.alpha_composite(floor_tint, dest=(b_plot[0], b_plot[1]))
    draw = ImageDraw.Draw(canvas)

    # sample cloud
    sample_overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    sod = ImageDraw.Draw(sample_overlay)
    for xp, yp in zip(ux_rank, vy_rank):
        px, py = bx(float(xp)), by(float(yp))
        sod.ellipse((px - 1.2, py - 1.2, px + 1.2, py + 1.2), fill=INK + (8,))
    canvas = Image.alpha_composite(canvas, sample_overlay)
    draw = ImageDraw.Draw(canvas)

    # quartile trajectory
    q_colors = [TEAL, (112, 196, 180), AMBER, ROSE]
    q_text_colors = [INK, INK, INK, INK]
    pts = []
    quartile_bins = pd.cut(ux_rank, bins=[0.0, 25.0, 50.0, 75.0, 100.0], labels=[1, 2, 3, 4], include_lowest=True)
    for i, row in uq_df.iterrows():
        px = bx((i + 0.5) * 25)
        mask = np.asarray(quartile_bins == (i + 1))
        y_mean = float(np.mean(vy_rank[mask])) if mask.any() else 50.0
        py = by(y_mean)
        pts.append((px, py))
        pr_text = f"{float(row['positive_rate']):.1f}%"
        tb = draw.textbbox((0, 0), pr_text, font=tick_bold)
        tx = int(px - (tb[2] - tb[0]) / 2)
        ty = b_plot[1] + 10
        draw.rounded_rectangle((tx - 9, ty - 5, tx + (tb[2] - tb[0]) + 9, ty + (tb[3] - tb[1]) + 5),
                                radius=7, fill=(255, 255, 255, 220))
        draw.text((tx, ty), pr_text, font=tick_bold, fill=q_colors[i])
    for p0, p1, col in zip(pts[:-1], pts[1:], q_colors[:-1]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=blend(col, INK, 0.15), width=5)
    for pt, col in zip(pts, q_colors):
        draw.ellipse((pt[0] - 9, pt[1] - 9, pt[0] + 9, pt[1] + 9), fill=col, outline=WHITE, width=2)
    cb_full_h = b_plot[3] - b_plot[1]
    cb_short_h = int(cb_full_h * 0.65)
    cb_y1 = b_plot[1] + (cb_full_h - cb_short_h) // 2
    cb_y2 = cb_y1 + cb_short_h
    for j in range(cb_y1, cb_y2):
        frac = 1.0 - (j - cb_y1) / max(1, (cb_y2 - cb_y1))
        val = color_lo + frac * (color_hi - color_lo)
        draw.line((cb_x1, j, cb_x2, j), fill=gain_color(val, color_lo, color_hi), width=1)
    draw.rounded_rectangle((cb_x1, cb_y1, cb_x2, cb_y2), radius=6, outline=LINE, width=1)
    hi_label = f"{color_hi:+.02f}"
    lo_label = f"{color_lo:+.02f}"
    hi_b = draw.textbbox((0, 0), hi_label, font=cb_font)
    lo_b = draw.textbbox((0, 0), lo_label, font=cb_font)
    draw.text((cb_x1, cb_y1 - (hi_b[3] - hi_b[1]) - 8), hi_label, font=cb_font, fill=INK)
    draw.text((cb_x1, cb_y2 + 8), lo_label, font=cb_font, fill=INK)
    cb_mid_y = (cb_y1 + cb_y2) // 2
    draw_rotated_text(canvas, (cb_x2 + 10, cb_mid_y), "local mean ΔIoU", cb_font, MUTED, angle=90)
    note = "n = 909 shared post-event samples"
    note_font = load_font(28, bold=True)
    nb = draw.textbbox((0, 0), note, font=note_font)
    note_w = nb[2] - nb[0]
    note_x = panel_b[0] + (panel_b[2] - panel_b[0] - note_w) // 2
    draw.text((note_x, panel_b[1] + 15), note, font=note_font, fill=MUTED)

    save(
        canvas,
        "figure8_ugcop_gain_landscape_preview",
        [
            "# figure8_ugcop_gain_landscape_preview",
            "",
            "- status: exploratory mechanism-landscape preview",
            "- purpose: sample-level explanation candidate for Figure 8 redesign",
            "",
            "Panel (a): component-correction mechanism landscape",
            "- x = material_minus_visual_iou",
            "- y = full_minus_material_iou",
            f"- quadrants: TR {q_tr:.1f}%, TL {q_tl:.1f}%, BL {q_bl:.1f}%, BR {q_br:.1f}%",
            "",
            "Panel (b): residual gain landscape",
            "- x = uncertainty percentile (tie-spread ECDF)",
            "- y = visual-baseline percentile",
            "- color = local mean delta (v4 - matched visual)",
            f"- low-omega quartile: ΔIoU +{uq_df.iloc[0]['delta_mean']:.3f}, win {uq_df.iloc[0]['positive_rate']:.1f}%",
            f"- high-omega quartile: ΔIoU {uq_df.iloc[-1]['delta_mean']:.3f}, win {uq_df.iloc[-1]['positive_rate']:.1f}%",
        ],
    )

    # 3D-style preview for panel (b) only.
    render_3d_preview(ux_rank, vy_rank, dy)


def render_3d_preview(ux_rank: np.ndarray, vy_rank: np.ndarray, dy: np.ndarray) -> None:
    canvas = Image.new("RGBA", (1600, 980), WHITE + (255,))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(40, bold=True)
    panel_font = load_font(29, bold=True)
    axis_font = load_font(24, bold=True)
    body_font = load_font(20)
    small_font = load_font(17)

    draw.text((44, 26), "Residual gain landscape: pseudo-3D preview", font=title_font, fill=INK)

    panel = (40, 90, 1560, 940)
    draw.rounded_rectangle(panel, radius=22, fill=WHITE, outline=LINE, width=2)
    draw.text((58, 104), "(b) Residual gain landscape under uncertainty", font=panel_font, fill=INK)

    # Build smoothed field on percentiles.
    nx, ny = 34, 34
    x_edges = np.linspace(0, 100, nx + 1)
    y_edges = np.linspace(0, 100, ny + 1)
    sum_grid = np.zeros((ny, nx), dtype=float)
    count_grid = np.zeros((ny, nx), dtype=float)
    den_grid = np.zeros((ny, nx), dtype=float)
    xi = np.clip(np.digitize(ux_rank, x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(vy_rank, y_edges) - 1, 0, ny - 1)
    for xx_i, yy_i, delta in zip(xi, yi, dy):
        sum_grid[yy_i, xx_i] += float(delta)
        count_grid[yy_i, xx_i] += 1.0
        den_grid[yy_i, xx_i] += 1.0

    field = np.divide(smooth2d(sum_grid, passes=5), np.maximum(smooth2d(count_grid, passes=5), 1e-6))
    dens = smooth2d(den_grid, passes=3)
    dens = dens / max(1e-6, float(dens.max()))

    color_lo = float(np.quantile(dy, 0.05))
    color_hi = float(np.quantile(dy, 0.95))
    z_min = color_lo
    z_max = color_hi

    plot = (110, 170, 1480, 860)
    floor_origin = (plot[0] + 610, plot[1] + 80)
    sx = 5.2
    sy = 2.65
    sz = 930

    # floor guide
    floor = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    fd = ImageDraw.Draw(floor)
    for t in [0, 25, 50, 75, 100]:
        p0 = project_iso(t, 0, 0, floor_origin, sx, sy, sz)
        p1 = project_iso(t, 100, 0, floor_origin, sx, sy, sz)
        fd.line((p0[0], p0[1], p1[0], p1[1]), fill=GRID + (170,), width=1)
        p2 = project_iso(0, t, 0, floor_origin, sx, sy, sz)
        p3 = project_iso(100, t, 0, floor_origin, sx, sy, sz)
        fd.line((p2[0], p2[1], p3[0], p3[1]), fill=GRID + (170,), width=1)
    canvas = Image.alpha_composite(canvas, floor)
    draw = ImageDraw.Draw(canvas)

    # surface polygons, drawn back-to-front.
    surface = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    sd = ImageDraw.Draw(surface)
    for iy in range(ny - 1, -1, -1):
        for ix in range(nx - 1, -1, -1):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            z00 = max(z_min, min(z_max, field[iy, ix]))
            z10 = max(z_min, min(z_max, field[iy, min(ix + 1, nx - 1)]))
            z01 = max(z_min, min(z_max, field[min(iy + 1, ny - 1), ix]))
            z11 = max(z_min, min(z_max, field[min(iy + 1, ny - 1), min(ix + 1, nx - 1)]))
            d = max(dens[iy, ix], dens[min(iy + 1, ny - 1), min(ix + 1, nx - 1)])
            if d < 0.03:
                continue

            def zn(v: float) -> float:
                return (v - z_min) / max(1e-6, (z_max - z_min))

            p00 = project_iso(x0, y0, zn(z00), floor_origin, sx, sy, sz)
            p10 = project_iso(x1, y0, zn(z10), floor_origin, sx, sy, sz)
            p11 = project_iso(x1, y1, zn(z11), floor_origin, sx, sy, sz)
            p01 = project_iso(x0, y1, zn(z01), floor_origin, sx, sy, sz)
            z_mean = (z00 + z10 + z11 + z01) / 4.0
            base = gain_color(float(z_mean), color_lo, color_hi)
            alpha = int(40 + 185 * (d ** 0.72))
            sd.polygon([p00, p10, p11, p01], fill=base + (alpha,), outline=blend(base, INK, 0.15) + (110,))

    canvas = Image.alpha_composite(canvas, surface)
    draw = ImageDraw.Draw(canvas)

    # contour highlights
    contour = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    cd = ImageDraw.Draw(contour)
    thresholds = [0.00, 0.04, 0.08]
    for thr in thresholds:
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                vals = [field[iy, ix], field[iy, ix + 1], field[iy + 1, ix + 1], field[iy + 1, ix]]
                if min(vals) <= thr <= max(vals):
                    x0, x1 = x_edges[ix], x_edges[ix + 1]
                    y0, y1 = y_edges[iy], y_edges[iy + 1]
                    p0 = project_iso((x0 + x1) * 0.5, y0, (thr - z_min) / max(1e-6, z_max - z_min), floor_origin, sx, sy, sz)
                    p1 = project_iso((x0 + x1) * 0.5, y1, (thr - z_min) / max(1e-6, z_max - z_min), floor_origin, sx, sy, sz)
                    cd.line((p0[0], p0[1], p1[0], p1[1]), fill=(255, 255, 255, 60), width=1)
    canvas = Image.alpha_composite(canvas, contour)
    draw = ImageDraw.Draw(canvas)

    # sample cloud projected slightly above the floor.
    cloud = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    cld = ImageDraw.Draw(cloud)
    for xp, yp, delta in zip(ux_rank, vy_rank, dy):
        z = (float(delta) - z_min) / max(1e-6, (z_max - z_min))
        px, py = project_iso(float(xp), float(yp), max(0.02, z * 0.8), floor_origin, sx, sy, sz)
        cld.ellipse((px - 1.6, py - 1.6, px + 1.6, py + 1.6), fill=INK + (35,))
    canvas = Image.alpha_composite(canvas, cloud)
    draw = ImageDraw.Draw(canvas)

    # Quartile trajectory.
    q_colors = [TEAL, (112, 196, 180), AMBER, ROSE]
    quartile_bins = pd.cut(ux_rank, bins=[0.0, 25.0, 50.0, 75.0, 100.0], labels=[1, 2, 3, 4], include_lowest=True)
    q_pts = []
    stats = []
    for q_idx in range(1, 5):
        mask = np.asarray(quartile_bins == q_idx)
        xm = float(np.mean(ux_rank[mask]))
        ym = float(np.mean(vy_rank[mask]))
        dm = float(np.mean(dy[mask]))
        pm = float(100.0 * np.mean(dy[mask] > 0))
        q_pts.append(project_iso(xm, ym, (dm - z_min) / max(1e-6, z_max - z_min), floor_origin, sx, sy, sz))
        stats.append((dm, pm))
    for p0, p1, col in zip(q_pts[:-1], q_pts[1:], q_colors[:-1]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=blend(col, INK, 0.15), width=4)
    for idx, (pt, col) in enumerate(zip(q_pts, q_colors), start=1):
        draw.ellipse((pt[0] - 8, pt[1] - 8, pt[0] + 8, pt[1] + 8), fill=col, outline=WHITE, width=2)
        draw.text((pt[0] + 10, pt[1] - 20), f"Q{idx}", font=small_font, fill=col)

    # Axes labels and callouts.
    draw.text((plot[0] + 420, plot[3] + 10), "Uncertainty percentile (low → high)", font=axis_font, fill=INK)
    draw_rotated_text(canvas, (plot[0] - 58, plot[1] + 200), "Visual-baseline percentile", axis_font, INK, angle=90)
    draw.text((plot[2] - 200, plot[1] + 22), "surface color = mean ΔIoU", font=body_font, fill=MUTED)
    note = "Q1 low ω: +0.084 / 87.2% wins    Q4 high ω: -0.011 / 0.4% wins"
    draw.rounded_rectangle((plot[0] + 40, plot[1] + 34, plot[0] + 540, plot[1] + 86), radius=16, fill=(255, 255, 255, 228), outline=LINE, width=2)
    draw.text((plot[0] + 58, plot[1] + 48), note, font=body_font, fill=INK)

    # Color bar
    cb_x1 = plot[0] + 70
    cb_x2 = plot[0] + 360
    cb_y1 = plot[3] + 32
    cb_y2 = cb_y1 + 18
    for i in range(cb_x1, cb_x2):
        frac = (i - cb_x1) / max(1, (cb_x2 - cb_x1))
        val = color_lo + frac * (color_hi - color_lo)
        draw.line((i, cb_y1, i, cb_y2), fill=gain_color(val, color_lo, color_hi), width=1)
    draw.rounded_rectangle((cb_x1, cb_y1, cb_x2, cb_y2), radius=5, outline=LINE, width=1)
    draw.text((cb_x1, cb_y1 - 24), "mean ΔIoU", font=small_font, fill=MUTED)
    draw.text((cb_x1 - 6, cb_y2 + 4), f"{color_lo:+.02f}", font=small_font, fill=MUTED)
    draw.text((cb_x2 - 34, cb_y2 + 4), f"{color_hi:+.02f}", font=small_font, fill=MUTED)

    save(
        canvas,
        "figure8_ugcop_gain_landscape_3d_preview",
        [
            "# figure8_ugcop_gain_landscape_3d_preview",
            "",
            "- status: exploratory pseudo-3D mechanism preview",
            "- purpose: test a more striking but still quantitative mechanism landscape",
            "",
            "Panel (b) surface:",
            "- x = uncertainty percentile",
            "- y = visual-baseline percentile",
            "- z/color = local mean ΔIoU (v4 - matched visual)",
        ],
    )


if __name__ == "__main__":
    render()
