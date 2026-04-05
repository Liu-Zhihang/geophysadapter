#!/usr/bin/env python3
"""Render the main quantitative figures used in the paper."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from render_figure5_case_candidate_preview import (
    PAIRED_CSV as FIG5_PAIRED_CSV,
    crop_focus as fig5_crop_focus,
    load_case_assets as fig5_load_case_assets,
    overlay_boundary as fig5_overlay_boundary,
    short_label as fig5_short_label,
)


WHITE = (251, 251, 249)
INK = (25, 30, 36)
MUTED = (88, 96, 106)
LINE = (194, 199, 205)
GRID = (226, 229, 233)
SLATE = (133, 137, 149)
SLATE_LIGHT = (226, 232, 240)
SAND = (186, 143, 67)
SAND_LIGHT = (247, 239, 219)
BLUE = (82, 134, 212)
BLUE_LIGHT = (225, 238, 252)
AMBER = (224, 168, 82)
AMBER_LIGHT = (250, 238, 214)
TEAL = (56, 135, 123)
TEAL_LIGHT = (222, 244, 240)
ROSE = (166, 78, 78)
ROSE_LIGHT = (248, 231, 231)
GREEN = (76, 135, 82)
GREEN_LIGHT = (229, 243, 230)
ORANGE = (194, 115, 41)
ORANGE_LIGHT = (251, 237, 223)
SKY = (122, 183, 162)
SKY_LIGHT = (228, 245, 238)


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

ROOT = Path(__file__).resolve().parents[1]
ROOT_FIG_DIR = ROOT / "docs" / "assets"
PKG_FIG_DIR = ROOT_FIG_DIR


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


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        box = draw.textbbox((0, 0), trial, font=font)
        if box[2] - box[0] <= max_w:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_w: int,
    *,
    fill: tuple[int, int, int] = MUTED,
    line_gap: int = 8,
) -> int:
    x, y = xy
    start_y = y
    for line in wrap_text(draw, text, font, max_w):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y - start_y


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str | None,
    *,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = WHITE,
    outline: tuple[int, int, int] = LINE,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=26, fill=fill, outline=outline, width=3)
    draw.text((x0 + 24, y0 + 18), title, font=title_font, fill=INK)
    cursor_y = y0 + 54
    if subtitle:
        cursor_y += draw_wrapped_text(draw, (x0 + 24, cursor_y), subtitle, subtitle_font, x1 - x0 - 48)
    return (x0 + 24, cursor_y + 16, x1 - 24, y1 - 24)


def draw_minimal_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    *,
    title_font: ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    draw.text((x0, y0), title, font=title_font, fill=INK)
    bbox = draw.multiline_textbbox((x0, y0), title, font=title_font, spacing=4)
    title_h = bbox[3] - bbox[1]
    return (x0 + 12, y0 + title_h + 18, x1 - 8, y1 - 8)


def draw_note_box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    bullets: list[str],
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = SLATE_LIGHT,
    outline: tuple[int, int, int] = SLATE,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=24, fill=fill, outline=outline, width=3)
    draw.text((x0 + 22, y0 + 16), title, font=title_font, fill=INK)
    cursor_y = y0 + 58
    for bullet in bullets:
        draw.text((x0 + 24, cursor_y), u"\u2022", font=body_font, fill=outline)
        cursor_y += draw_wrapped_text(draw, (x0 + 46, cursor_y - 2), bullet, body_font, x1 - x0 - 70, fill=MUTED)
        cursor_y += 6


def draw_legend(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    items: list[tuple[str, tuple[int, int, int]]],
    *,
    font: ImageFont.ImageFont,
    chip_w: int = 26,
    gap: int = 18,
) -> None:
    x, y = xy
    for label, color in items:
        draw.rounded_rectangle((x, y + 4, x + chip_w, y + 22), radius=6, fill=color, outline=color)
        draw.text((x + chip_w + 8, y), label, font=font, fill=MUTED)
        bbox = draw.textbbox((0, 0), label, font=font)
        x += chip_w + 8 + (bbox[2] - bbox[0]) + gap


def draw_method_bars(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    methods: list[str],
    values: list[float],
    errors: list[float | None],
    colors: list[tuple[int, int, int]],
    *,
    y_max: float,
    tick_step: float,
    legend_title: str | None = None,
    value_fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 80
    plot_y0 = y0 + 10
    plot_x1 = x1 - 24
    plot_y1 = y1 - 88
    font_axis = load_font(20)
    font_value = load_font(18, bold=True)
    font_method = load_font(19)
    usable_h = plot_y1 - plot_y0

    for tick_idx in range(int(round(y_max / tick_step)) + 1):
        tick_val = tick_idx * tick_step
        y = plot_y1 - usable_h * (tick_val / y_max)
        draw.line((plot_x0, y, plot_x1, y), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((plot_x0 - 16 - (bbox[2] - bbox[0]), y - 10), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=LINE, width=3)
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)
    bar_gap = 28
    total_w = plot_x1 - plot_x0
    bar_w = max(50, int((total_w - bar_gap * (len(methods) + 1)) / len(methods)))

    for idx, (method, val, err, color) in enumerate(zip(methods, values, errors, colors)):
        x_left = plot_x0 + bar_gap + idx * (bar_w + bar_gap)
        x_right = x_left + bar_w
        bar_h = usable_h * (val / y_max)
        y_top = plot_y1 - bar_h
        draw.rounded_rectangle((x_left, y_top, x_right, plot_y1), radius=10, fill=color, outline=color)
        if err:
            err_h = usable_h * (err / y_max)
            x_mid = (x_left + x_right) // 2
            draw.line((x_mid, y_top - err_h, x_mid, y_top + err_h), fill=INK, width=3)
            draw.line((x_mid - 10, y_top - err_h, x_mid + 10, y_top - err_h), fill=INK, width=3)
            draw.line((x_mid - 10, y_top + err_h, x_mid + 10, y_top + err_h), fill=INK, width=3)
        value_text = value_fmt.format(val)
        bbox = draw.textbbox((0, 0), value_text, font=font_value)
        draw.text((x_left + (bar_w - (bbox[2] - bbox[0])) / 2, y_top - 28), value_text, font=font_value, fill=INK)
        label_lines = method.split("\n")
        label_y = plot_y1 + 14
        for line in label_lines:
            bbox = draw.textbbox((0, 0), line, font=font_method)
            draw.text((x_left + (bar_w - (bbox[2] - bbox[0])) / 2, label_y), line, font=font_method, fill=MUTED)
            label_y += 24

    if legend_title:
        draw.text((x0, y1 - 38), legend_title, font=font_axis, fill=MUTED)


def draw_grouped_bars(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    categories: list[str],
    series: list[dict[str, object]],
    colors: list[tuple[int, int, int]],
    *,
    y_max: float,
    tick_step: float,
    lower_better: bool = False,
    value_fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 82
    plot_y0 = y0 + 14
    plot_x1 = x1 - 24
    plot_y1 = y1 - 104
    font_axis = load_font(20)
    font_value = load_font(17, bold=True)
    font_label = load_font(18)
    font_legend = load_font(18)
    usable_h = plot_y1 - plot_y0

    for tick_idx in range(int(round(y_max / tick_step)) + 1):
        tick_val = tick_idx * tick_step
        y = plot_y1 - usable_h * (tick_val / y_max)
        draw.line((plot_x0, y, plot_x1, y), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((plot_x0 - 16 - (bbox[2] - bbox[0]), y - 10), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=LINE, width=3)
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)

    group_w = (plot_x1 - plot_x0) / len(categories)
    inner_gap = 10
    bar_w = min(66, int((group_w - 24 - inner_gap * (len(series) - 1)) / len(series)))

    for cat_idx, category in enumerate(categories):
        group_x0 = plot_x0 + cat_idx * group_w
        bars_total_w = len(series) * bar_w + inner_gap * (len(series) - 1)
        bar_x = group_x0 + (group_w - bars_total_w) / 2
        for series_idx, one in enumerate(series):
            values = one["values"]
            value = float(values[cat_idx])
            y_top = plot_y1 - usable_h * (value / y_max)
            x_left = int(bar_x + series_idx * (bar_w + inner_gap))
            x_right = x_left + bar_w
            color = colors[series_idx]
            draw.rounded_rectangle((x_left, y_top, x_right, plot_y1), radius=10, fill=color, outline=color)
            label = value_fmt.format(value)
            bbox = draw.textbbox((0, 0), label, font=font_value)
            draw.text((x_left + (bar_w - (bbox[2] - bbox[0])) / 2, y_top - 24), label, font=font_value, fill=INK)
        label_y = plot_y1 + 14
        label_w = int(group_w - 20)
        for line in wrap_text(draw, category, font_label, label_w):
            bbox = draw.textbbox((0, 0), line, font=font_label)
            draw.text((group_x0 + (group_w - (bbox[2] - bbox[0])) / 2, label_y), line, font=font_label, fill=MUTED)
            label_y += 22

    legend_items = [(str(one["name"]), colors[idx]) for idx, one in enumerate(series)]
    draw_legend(draw, (x0, y1 - 40), legend_items, font=font_legend)
    if lower_better:
        lower_text = "Lower is better"
        bbox = draw.textbbox((0, 0), lower_text, font=font_legend)
        draw.text((x1 - (bbox[2] - bbox[0]), y1 - 40), lower_text, font=font_legend, fill=ROSE)


def draw_method_dotplot(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    methods: list[str],
    values: list[float],
    errors: list[float | None],
    colors: list[tuple[int, int, int]],
    *,
    x_min: float,
    x_max: float,
    tick_step: float,
    value_fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 180
    plot_y0 = y0 + 18
    plot_x1 = x1 - 28
    plot_y1 = y1 - 32
    font_axis = load_font(20)
    font_label = load_font(20)
    font_value = load_font(18, bold=True)
    usable_w = plot_x1 - plot_x0
    usable_h = plot_y1 - plot_y0

    n_ticks = int(round((x_max - x_min) / tick_step)) + 1
    for tick_idx in range(n_ticks):
        tick_val = x_min + tick_idx * tick_step
        x = plot_x0 + usable_w * ((tick_val - x_min) / (x_max - x_min))
        draw.line((x, plot_y0, x, plot_y1), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_y1 + 8), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)

    row_h = usable_h / max(1, len(methods))
    for idx, (method, val, err, color) in enumerate(zip(methods, values, errors, colors)):
        y = int(plot_y0 + row_h * (idx + 0.5))
        draw.line((plot_x0, y, plot_x1, y), fill=(241, 243, 246), width=1)
        for line_idx, line in enumerate(method.split("\n")):
            bbox = draw.textbbox((0, 0), line, font=font_label)
            text_y = y - 22 + line_idx * 22 - (len(method.split("\n")) - 1) * 10
            draw.text((plot_x0 - 18 - (bbox[2] - bbox[0]), text_y), line, font=font_label, fill=MUTED)

        x_val = int(plot_x0 + usable_w * ((val - x_min) / (x_max - x_min)))
        if err:
            err_px = usable_w * (err / (x_max - x_min))
            draw.line((x_val - err_px, y, x_val + err_px, y), fill=INK, width=3)
            draw.line((x_val - err_px, y - 9, x_val - err_px, y + 9), fill=INK, width=3)
            draw.line((x_val + err_px, y - 9, x_val + err_px, y + 9), fill=INK, width=3)
        r = 11 if idx < len(methods) - 1 else 13
        draw.ellipse((x_val - r, y - r, x_val + r, y + r), fill=color, outline=WHITE, width=3)
        value_text = value_fmt.format(val)
        bbox = draw.textbbox((0, 0), value_text, font=font_value)
        label_x = min(max(plot_x0 + 4, x_val + 14), plot_x1 - (bbox[2] - bbox[0]) - 4)
        draw.text((label_x, y - 14), value_text, font=font_value, fill=INK)


def draw_metric_dumbbell(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    metrics: list[str],
    left_label: str,
    left_values: list[float],
    right_label: str,
    right_values: list[float],
    left_color: tuple[int, int, int],
    right_color: tuple[int, int, int],
    *,
    x_min: float,
    x_max: float,
    tick_step: float,
    value_fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 150
    plot_y0 = y0 + 18
    plot_x1 = x1 - 28
    plot_y1 = y1 - 72
    font_axis = load_font(20)
    font_label = load_font(20)
    font_value = load_font(17, bold=True)
    font_legend = load_font(18)
    usable_w = plot_x1 - plot_x0
    usable_h = plot_y1 - plot_y0

    n_ticks = int(round((x_max - x_min) / tick_step)) + 1
    for tick_idx in range(n_ticks):
        tick_val = x_min + tick_idx * tick_step
        x = plot_x0 + usable_w * ((tick_val - x_min) / (x_max - x_min))
        draw.line((x, plot_y0, x, plot_y1), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_y1 + 8), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)

    row_h = usable_h / max(1, len(metrics))
    for idx, metric in enumerate(metrics):
        y = int(plot_y0 + row_h * (idx + 0.5))
        draw.line((plot_x0, y, plot_x1, y), fill=(241, 243, 246), width=1)
        bbox = draw.textbbox((0, 0), metric, font=font_label)
        draw.text((plot_x0 - 18 - (bbox[2] - bbox[0]), y - 12), metric, font=font_label, fill=MUTED)

        lv = left_values[idx]
        rv = right_values[idx]
        x_l = int(plot_x0 + usable_w * ((lv - x_min) / (x_max - x_min)))
        x_r = int(plot_x0 + usable_w * ((rv - x_min) / (x_max - x_min)))
        draw.line((x_l, y, x_r, y), fill=LINE, width=4)
        close_pair = abs(x_r - x_l) < 26
        label_specs = [
            (x_l, lv, left_color, -28),
            (x_r, rv, right_color, 10 if close_pair else -28),
        ]
        for x_val, val, color, dy in label_specs:
            draw.ellipse((x_val - 11, y - 11, x_val + 11, y + 11), fill=color, outline=WHITE, width=3)
            value_text = value_fmt.format(val)
            bbox = draw.textbbox((0, 0), value_text, font=font_value)
            draw.text((x_val - (bbox[2] - bbox[0]) / 2, y + dy), value_text, font=font_value, fill=INK)

    draw_legend(draw, (x0, y1 - 34), [(left_label, left_color), (right_label, right_color)], font=font_legend, chip_w=20, gap=14)


def draw_evidence_forest(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    row_labels: list[str],
    series_labels: list[str],
    values: list[list[float]],
    errors: list[list[float | None]],
    colors: list[tuple[int, int, int]],
    *,
    x_min: float,
    x_max: float,
    tick_step: float,
    value_fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 260
    plot_y0 = y0 + 22
    plot_x1 = x1 - 24
    plot_y1 = y1 - 74
    font_axis = load_font(20)
    font_row = load_font(21)
    font_value = load_font(18, bold=True)
    font_legend = load_font(18)
    usable_w = plot_x1 - plot_x0
    usable_h = plot_y1 - plot_y0

    n_ticks = int(round((x_max - x_min) / tick_step)) + 1
    for tick_idx in range(n_ticks):
        tick_val = x_min + tick_idx * tick_step
        x = plot_x0 + usable_w * ((tick_val - x_min) / (x_max - x_min))
        draw.line((x, plot_y0, x, plot_y1), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_y1 + 8), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)

    row_h = usable_h / max(1, len(row_labels))
    offsets = [-18, 0, 18]
    for row_idx, row_label in enumerate(row_labels):
        y_center = int(plot_y0 + row_h * (row_idx + 0.5))
        draw.line((plot_x0, y_center, plot_x1, y_center), fill=(241, 243, 246), width=1)
        row_lines = row_label.split("\n")
        for line_idx, line in enumerate(row_lines):
            bbox = draw.textbbox((0, 0), line, font=font_row)
            text_y = y_center - 22 + line_idx * 22 - (len(row_lines) - 1) * 10
            draw.text((plot_x0 - 18 - (bbox[2] - bbox[0]), text_y), line, font=font_row, fill=MUTED)

        for series_idx, (series_vals, series_errs, color) in enumerate(zip(values, errors, colors)):
            val = series_vals[row_idx]
            err = series_errs[row_idx]
            y = y_center + offsets[series_idx]
            x_val = int(plot_x0 + usable_w * ((val - x_min) / (x_max - x_min)))
            if err:
                err_px = usable_w * (err / (x_max - x_min))
                draw.line((x_val - err_px, y, x_val + err_px, y), fill=INK, width=3)
                draw.line((x_val - err_px, y - 8, x_val - err_px, y + 8), fill=INK, width=3)
                draw.line((x_val + err_px, y - 8, x_val + err_px, y + 8), fill=INK, width=3)
            r = 10 if series_idx < len(series_labels) - 1 else 12
            draw.ellipse((x_val - r, y - r, x_val + r, y + r), fill=color, outline=WHITE, width=3)
            value_text = value_fmt.format(val)
            bbox = draw.textbbox((0, 0), value_text, font=font_value)
            label_x = min(max(plot_x0 + 4, x_val + 12), plot_x1 - (bbox[2] - bbox[0]) - 4)
            draw.text((label_x, y - 13), value_text, font=font_value, fill=INK)

    draw_legend(draw, (x0, y1 - 30), list(zip(series_labels, colors)), font=font_legend, chip_w=20, gap=14)


def draw_connected_point_ranges(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    categories: list[str],
    series_labels: list[str],
    values: list[list[float]],
    errors: list[list[float | None]],
    colors: list[tuple[int, int, int]],
    *,
    y_min: float,
    y_max: float,
    tick_step: float,
) -> None:
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 70
    plot_y0 = y0 + 20
    plot_x1 = x1 - 24
    plot_y1 = y1 - 88
    font_axis = load_font(22)
    font_cat = load_font(21)
    font_legend = load_font(19)
    usable_w = plot_x1 - plot_x0
    usable_h = plot_y1 - plot_y0

    n_ticks = int(round((y_max - y_min) / tick_step)) + 1
    for tick_idx in range(n_ticks):
        tick_val = y_min + tick_idx * tick_step
        y = plot_y1 - usable_h * ((tick_val - y_min) / (y_max - y_min))
        draw.line((plot_x0, y, plot_x1, y), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((plot_x0 - 16 - (bbox[2] - bbox[0]), y - 10), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=LINE, width=3)
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)

    xs = []
    for idx, cat in enumerate(categories):
        x = int(plot_x0 + usable_w * (idx / max(1, len(categories) - 1)))
        xs.append(x)
        draw.line((x, plot_y0, x, plot_y1), fill=(241, 243, 246), width=1)
        lines = cat.split("\n")
        label_y = plot_y1 + 12
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_cat)
            draw.text((x - (bbox[2] - bbox[0]) / 2, label_y), line, font=font_cat, fill=MUTED)
            label_y += 24

    offsets = [-14, 0, 14]
    for series_idx, (label, vals, errs, color) in enumerate(zip(series_labels, values, errors, colors)):
        points = []
        for cat_idx, x in enumerate(xs):
            val = vals[cat_idx]
            err = errs[cat_idx]
            y = int(plot_y1 - usable_h * ((val - y_min) / (y_max - y_min))) + offsets[series_idx]
            points.append((x, y))
            if err:
                err_px = usable_h * (err / (y_max - y_min))
                draw.line((x, y - err_px, x, y + err_px), fill=color, width=3)
                draw.line((x - 7, y - err_px, x + 7, y - err_px), fill=color, width=3)
                draw.line((x - 7, y + err_px, x + 7, y + err_px), fill=color, width=3)
            r = 10
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=WHITE, width=3)
        draw.line(points, fill=color, width=4)

    draw_legend(draw, (x0, y1 - 34), list(zip(series_labels, colors)), font=font_legend, chip_w=20, gap=18)


def draw_grouped_points(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    categories: list[str],
    series_labels: list[str],
    values: list[list[float]],
    colors: list[tuple[int, int, int]],
    *,
    y_min: float,
    y_max: float,
    tick_step: float,
) -> None:
    x0, y0, x1, y1 = rect
    plot_x0 = x0 + 60
    plot_y0 = y0 + 20
    plot_x1 = x1 - 24
    plot_y1 = y1 - 88
    font_axis = load_font(22)
    font_cat = load_font(21)
    font_legend = load_font(19)
    usable_w = plot_x1 - plot_x0
    usable_h = plot_y1 - plot_y0

    n_ticks = int(round((y_max - y_min) / tick_step)) + 1
    for tick_idx in range(n_ticks):
        tick_val = y_min + tick_idx * tick_step
        y = plot_y1 - usable_h * ((tick_val - y_min) / (y_max - y_min))
        draw.line((plot_x0, y, plot_x1, y), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((plot_x0 - 16 - (bbox[2] - bbox[0]), y - 10), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=LINE, width=3)
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)

    xs = []
    for idx, cat in enumerate(categories):
        x = int(plot_x0 + usable_w * (idx / max(1, len(categories) - 1)))
        xs.append(x)
        draw.line((x, plot_y0, x, plot_y1), fill=(241, 243, 246), width=1)
        bbox = draw.textbbox((0, 0), cat, font=font_cat)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_y1 + 12), cat, font=font_cat, fill=MUTED)

    offsets = [-10, 10]
    for series_idx, (label, vals, color) in enumerate(zip(series_labels, values, colors)):
        for cat_idx, x in enumerate(xs):
            val = vals[cat_idx]
            y = int(plot_y1 - usable_h * ((val - y_min) / (y_max - y_min))) + offsets[series_idx]
            r = 10
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=WHITE, width=3)

    draw_legend(draw, (x0, y1 - 34), list(zip(series_labels, colors)), font=font_legend, chip_w=20, gap=18)


def draw_delta_reference_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    row_labels: list[str],
    series_labels: list[str],
    deltas: list[list[float]],
    colors: list[tuple[int, int, int]],
    *,
    x_min: float,
    x_max: float,
    tick_step: float,
    axis_label: str,
    reference_note: str = "0 (reference)",
) -> None:
    x0, y0, x1, y1 = rect
    font_axis = load_font(36)   
    font_row = load_font(36)    
    font_head = load_font(28)   
    font_note = load_font(36)   
    plot_x0 = x0 + 290
    plot_y0 = y0 + 72
    plot_x1 = x1 - 32
    plot_y1 = y1 - 96
    usable_w = plot_x1 - plot_x0
    usable_h = plot_y1 - plot_y0
    header_items = [(series_labels[0], colors[0], "circle"), (series_labels[1], colors[1], "circle")]
    draw_series_header(draw, (x0, y0 + 2), header_items, font=font_head, gap=30)

    n_ticks = int(round((x_max - x_min) / tick_step)) + 1
    x_zero = int(plot_x0 + usable_w * ((0.0 - x_min) / (x_max - x_min)))
    for tick_idx in range(n_ticks):
        tick_val = x_min + tick_idx * tick_step
        x = int(plot_x0 + usable_w * ((tick_val - x_min) / (x_max - x_min)))
        line_color = LINE if abs(tick_val) < 1e-9 else GRID
        line_w = 3 if abs(tick_val) < 1e-9 else 2
        draw.line((x, plot_y0, x, plot_y1), fill=line_color, width=line_w)
        label = "" if abs(tick_val) < 1e-9 else f"{tick_val:.02f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_y1 + 10), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)
    note_bbox = draw.textbbox((0, 0), reference_note, font=font_head)  
    header_width = measure_series_header_width(draw, header_items, font=font_head)
    note_x = x0 + header_width + 60  
    draw.text((note_x, y0 + 5), reference_note, font=font_head, fill=MUTED)  

    row_h = usable_h / max(1, len(row_labels))
    offsets = [-20, 20]
    light_fills = [interpolate_color(color, WHITE, 0.78) for color in colors]
    for row_idx, row_label in enumerate(row_labels):
        y_center = int(plot_y0 + row_h * (row_idx + 0.5))
        draw.line((plot_x0, y_center, plot_x1, y_center), fill=(241, 243, 246), width=1)
        lines = row_label.split("\n")
        label_y = y_center - 22
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_row)
            draw.text((plot_x0 - 18 - (bbox[2] - bbox[0]), label_y), line, font=font_row, fill=MUTED)
            label_y += 24

        for series_idx, (vals, color, fill_color) in enumerate(zip(deltas, colors, light_fills)):
            delta = vals[row_idx]
            y = y_center + offsets[series_idx]
            x_val = int(plot_x0 + usable_w * ((delta - x_min) / (x_max - x_min)))
            x_left = min(x_val, x_zero)
            x_right = max(x_val, x_zero)
            draw.rounded_rectangle((x_left, y - 18, x_right, y + 18), radius=14, fill=fill_color, outline=fill_color)
            draw.ellipse((x_val - 18, y - 18, x_val + 18, y + 18), fill=color, outline=WHITE, width=4)

    axis_bbox = draw.textbbox((0, 0), axis_label, font=font_note)
    draw.text((plot_x0 + (usable_w - (axis_bbox[2] - axis_bbox[0])) / 2, y1 - 34), axis_label, font=font_note, fill=MUTED)


def draw_series_header(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    items: list[tuple[str, tuple[int, int, int], str]],
    *,
    font: ImageFont.ImageFont,
    gap: int = 34,  
) -> None:
    x, y = xy
    for label, color, marker in items:
        if marker == "diamond":
            pts = [(x + 11, y + 14), (x + 22, y + 25), (x + 11, y + 36), (x, y + 25)]
            draw.polygon(pts, fill=color, outline=WHITE)
        else:
            draw.ellipse((x, y + 12, x + 22, y + 34), fill=color, outline=WHITE, width=2)
        draw.text((x + 34, y + 3), label, font=font, fill=MUTED)
        bbox = draw.textbbox((0, 0), label, font=font)
        x += 34 + (bbox[2] - bbox[0]) + gap  


def measure_series_header_width(
    draw: ImageDraw.ImageDraw,
    items: list[tuple[str, tuple[int, int, int], str]],
    *,
    font: ImageFont.ImageFont,
) -> int:
    width = 0
    for idx, (label, _color, _marker) in enumerate(items):
        bbox = draw.textbbox((0, 0), label, font=font)
        width += 34 + (bbox[2] - bbox[0])
        if idx < len(items) - 1:
            width += 34
    return width


def draw_frontier_estimates(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    row_labels: list[str],
    series_labels: list[str],
    values: list[list[float]],
    errors: list[list[float | None]],
    colors: list[tuple[int, int, int]],
    *,
    x_min: float,
    x_max: float,
    tick_step: float,
    value_fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = rect
    font_axis = load_font(27)
    font_row = load_font(34)
    font_head = load_font(32)
    plot_x0 = x0 + 264
    plot_y0 = y0 + 70
    plot_x1 = x1 - 26
    plot_y1 = y1 - 34
    usable_w = plot_x1 - plot_x0
    usable_h = plot_y1 - plot_y0
    header_items = []
    for idx, (label, color) in enumerate(zip(series_labels, colors)):
        marker = "diamond" if (len(series_labels) == 3 and idx == 1) else "circle"
        header_items.append((label, color, marker))
    header_w = measure_series_header_width(draw, header_items, font=font_head)
    draw_series_header(
        draw,
        (x0, y0 + 2),
        header_items,
        font=font_head,
    )

    n_ticks = int(round((x_max - x_min) / tick_step)) + 1
    for tick_idx in range(n_ticks):
        tick_val = x_min + tick_idx * tick_step
        x = plot_x0 + usable_w * ((tick_val - x_min) / (x_max - x_min))
        draw.line((x, plot_y0, x, plot_y1), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_y1 + 8), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)

    row_h = usable_h / max(1, len(row_labels))
    if len(row_labels) == 3:
        row_centers = [
            int(plot_y0 + usable_h * 0.16),
            int(plot_y0 + usable_h * 0.49),
            int(plot_y0 + usable_h * 0.88),
        ]
    else:
        row_centers = [int(plot_y0 + row_h * (idx + 0.5)) for idx in range(len(row_labels))]
    offsets = [-78, 0, 78]
    for row_idx, row_label in enumerate(row_labels):
        y_center = row_centers[row_idx]
        draw.line((plot_x0, y_center, plot_x1, y_center), fill=(241, 243, 246), width=1)
        lines = row_label.split("\n")
        label_y = y_center - 22
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_row)
            draw.text((plot_x0 - 18 - (bbox[2] - bbox[0]), label_y), line, font=font_row, fill=MUTED)
            label_y += 24

        xs_row = []
        for series_idx, (vals, errs, color) in enumerate(zip(values, errors, colors)):
            val = vals[row_idx]
            err = errs[row_idx]
            y = y_center + offsets[series_idx]
            x_val = int(plot_x0 + usable_w * ((val - x_min) / (x_max - x_min)))
            xs_row.append((x_val, y))
            if err:
                err_px = usable_w * (err / (x_max - x_min))
                err_px = max(err_px, 7)
                draw.line((x_val - err_px, y, x_val + err_px, y), fill=color, width=5)
                draw.line((x_val - err_px, y - 9, x_val - err_px, y + 9), fill=color, width=5)
                draw.line((x_val + err_px, y - 9, x_val + err_px, y + 9), fill=color, width=5)
            if len(series_labels) == 3 and series_idx == 1:
                pts = [(x_val, y - 18), (x_val + 18, y), (x_val, y + 18), (x_val - 18, y)]
                draw.polygon(pts, fill=color, outline=WHITE)
            else:
                draw.ellipse((x_val - 18, y - 18, x_val + 18, y + 18), fill=color, outline=WHITE, width=4)


def draw_paired_lane_estimates(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    row_labels: list[str],
    series_labels: list[str],
    values: list[list[float]],
    errors: list[list[float | None]],
    colors: list[tuple[int, int, int]],
    light_fills: list[tuple[int, int, int]],
    *,
    x_min: float,
    x_max: float,
    tick_values: list[float],
) -> None:
    x0, y0, x1, y1 = rect
    font_axis = load_font(26)
    font_row = load_font(34)
    font_head = load_font(28)
    plot_x0 = x0 + 312
    plot_y0 = y0 + 74
    plot_x1 = x1 - 18
    plot_y1 = y1 - 34
    usable_w = plot_x1 - plot_x0
    usable_h = plot_y1 - plot_y0

    header_items = [(series_labels[0], colors[0], "circle"), (series_labels[1], colors[1], "circle")]
    draw_series_header(draw, (x0, y0 + 2), header_items, font=font_head)

    for tick_idx, tick_val in enumerate(tick_values):
        x = plot_x0 + usable_w * ((tick_val - x_min) / (x_max - x_min))
        draw.line((x, plot_y0, x, plot_y1), fill=GRID if tick_idx else LINE, width=2)
        label = f"{tick_val:.2f}" if tick_val < 0.3 else f"{tick_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_y1 + 10), label, font=font_axis, fill=MUTED)

    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=LINE, width=3)

    row_h = usable_h / max(1, len(row_labels))
    offsets = [-28, 28]
    if len(row_labels) == 3:
        row_centers = [
            int(plot_y0 + usable_h * 0.15),
            int(plot_y0 + usable_h * 0.50),
            int(plot_y0 + usable_h * 0.87),
        ]
    else:
        row_centers = [int(plot_y0 + row_h * (idx + 0.5)) for idx in range(len(row_labels))]

    for row_idx, row_label in enumerate(row_labels):
        y_center = row_centers[row_idx]
        draw.line((plot_x0, y_center, plot_x1, y_center), fill=(241, 243, 246), width=1)
        lines = row_label.split("\n")
        label_y = y_center - 24
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_row)
            draw.text((plot_x0 - 18 - (bbox[2] - bbox[0]), label_y), line, font=font_row, fill=MUTED)
            label_y += 24

        for series_idx, (vals, errs, color, fill_color) in enumerate(zip(values, errors, colors, light_fills)):
            val = vals[row_idx]
            err = errs[row_idx]
            y = y_center + offsets[series_idx]
            x_val = int(plot_x0 + usable_w * ((val - x_min) / (x_max - x_min)))
            lane_x0 = plot_x0 + 8
            draw.rounded_rectangle((lane_x0, y - 10, x_val, y + 10), radius=8, fill=fill_color, outline=fill_color)
            if err:
                err_px = usable_w * (err / (x_max - x_min))
                err_px = max(err_px, 6)
                draw.line((x_val - err_px, y, x_val + err_px, y), fill=color, width=4)
                draw.line((x_val - err_px, y - 8, x_val - err_px, y + 8), fill=color, width=4)
                draw.line((x_val + err_px, y - 8, x_val + err_px, y + 8), fill=color, width=4)
            draw.ellipse((x_val - 14, y - 14, x_val + 14, y + 14), fill=color, outline=WHITE, width=3)


def interpolate_color(
    c0: tuple[int, int, int],
    c1: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def draw_protocol_heatmap(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float]],
    *,
    title_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
    low_color: tuple[int, int, int],
    high_color: tuple[int, int, int],
    best_row: int | None = None,
) -> None:
    x0, y0, x1, y1 = rect
    left_w = 300
    header_h = 92
    cell_gap = 16
    rows = len(row_labels)
    cols = len(col_labels)
    grid_x0 = x0 + left_w
    grid_y0 = y0 + header_h
    grid_w = x1 - grid_x0
    grid_h = y1 - grid_y0 - 20
    cell_w = int((grid_w - cell_gap * (cols - 1)) / cols)
    cell_h = int((grid_h - cell_gap * (rows - 1)) / rows)

    flat = [v for row in values for v in row]
    vmin = min(flat)
    vmax = max(flat)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    for j, col in enumerate(col_labels):
        cx = grid_x0 + j * (cell_w + cell_gap)
        bbox = draw.textbbox((0, 0), col, font=label_font)
        draw.text((cx + (cell_w - (bbox[2] - bbox[0])) / 2, y0 + 18), col, font=label_font, fill=MUTED)

    for i, row_label in enumerate(row_labels):
        cy = grid_y0 + i * (cell_h + cell_gap)
        lines = row_label.split("\n")
        total_h = len(lines) * 24
        ty = cy + (cell_h - total_h) / 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            draw.text((grid_x0 - 18 - (bbox[2] - bbox[0]), ty), line, font=title_font, fill=MUTED)
            ty += 24

        for j in range(cols):
            cx = grid_x0 + j * (cell_w + cell_gap)
            val = values[i][j]
            t = (val - vmin) / (vmax - vmin)
            fill = interpolate_color(low_color, high_color, t)
            outline = LINE
            width = 2
            if best_row is not None and i == best_row:
                outline = INK
                width = 3
            draw.rounded_rectangle((cx, cy, cx + cell_w, cy + cell_h), radius=20, fill=fill, outline=outline, width=width)
            txt = f"{val:.3f}"
            tb = draw.textbbox((0, 0), txt, font=value_font)
            draw.text((cx + (cell_w - (tb[2] - tb[0])) / 2, cy + (cell_h - (tb[3] - tb[1])) / 2 - 4), txt, font=value_font, fill=INK)


def draw_task_icon(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    mode: str,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=26, fill=(248, 249, 247), outline=LINE, width=2)
    if mode == "post":
        tile = (x0 + 24, y0 + 18, x0 + 156, y0 + 118)
        draw.rounded_rectangle(tile, radius=18, fill=(198, 212, 196), outline=(148, 162, 145), width=2)
        draw.line((tile[0] + 14, tile[1] + 14, tile[2] - 18, tile[3] - 20), fill=(171, 182, 168), width=3)
        draw.line((tile[0] + 24, tile[1] + 12, tile[2] - 16, tile[3] - 30), fill=(183, 194, 180), width=2)
        overlay = [
            (tile[0] + 70, tile[1] + 16),
            (tile[0] + 112, tile[1] + 28),
            (tile[0] + 106, tile[1] + 78),
            (tile[0] + 62, tile[1] + 70),
        ]
        draw.polygon(overlay, fill=(230, 158, 104), outline=(190, 108, 73))
        draw.line((x0 + 196, y0 + 66, x1 - 26, y0 + 66), fill=GRID, width=4)
        draw.ellipse((x1 - 52, y0 + 50, x1 - 24, y0 + 78), fill=AMBER, outline=WHITE, width=2)
    else:
        tile1 = (x0 + 20, y0 + 22, x0 + 122, y0 + 100)
        tile2 = (x0 + 52, y0 + 36, x0 + 154, y0 + 114)
        draw.rounded_rectangle(tile1, radius=16, fill=(204, 216, 202), outline=(149, 161, 146), width=2)
        draw.rounded_rectangle(tile2, radius=16, fill=(191, 205, 208), outline=(140, 153, 157), width=2)
        draw.line((tile1[0] + 16, tile1[1] + 14, tile1[2] - 16, tile1[3] - 14), fill=(175, 188, 173), width=2)
        draw.line((tile2[0] + 14, tile2[1] + 14, tile2[2] - 18, tile2[3] - 18), fill=(167, 181, 184), width=2)
        change = [
            (tile2[0] + 50, tile2[1] + 16),
            (tile2[0] + 82, tile2[1] + 26),
            (tile2[0] + 78, tile2[1] + 58),
            (tile2[0] + 46, tile2[1] + 52),
        ]
        draw.polygon(change, fill=(249, 229, 192), outline=AMBER)
        draw.line((x0 + 198, y0 + 66, x1 - 26, y0 + 66), fill=GRID, width=4)
        draw.polygon([(x1 - 38, y0 + 56), (x1 - 20, y0 + 66), (x1 - 38, y0 + 76)], fill=BLUE, outline=BLUE)


def draw_protocol_cards(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    row_labels: list[str],
    values: list[list[float]],
    *,
    mode: str,
    best_row: int,
) -> None:
    x0, y0, x1, y1 = rect
    header_font = load_font(28, bold=True)
    row_font = load_font(34, bold=True)
    val_font = load_font(34, bold=True)
    sub_font = load_font(24)

    icon_rect = (x0 + 6, y0 + 10, x0 + 420, y0 + 170)
    draw_task_icon(draw, icon_rect, mode=mode)
    if mode == "post":
        desc = "single-image evidence"
    else:
        desc = "paired-change evidence"
    bbox = draw.textbbox((0, 0), desc, font=sub_font)
    draw.text((icon_rect[0] + (icon_rect[2] - icon_rect[0] - (bbox[2] - bbox[0])) / 2, icon_rect[3] + 8), desc, font=sub_font, fill=MUTED)

    cols_x = [x0 + 540, x0 + 790]
    col_w = 190
    col_h = 70
    for cx, title in zip(cols_x, ["IoU@val-thr", "IoU@0.50"]):
        hb = draw.textbbox((0, 0), title, font=header_font)
        draw.text((cx + (col_w - (hb[2] - hb[0])) / 2, y0 + 50), title, font=header_font, fill=MUTED)

    top_y = y0 + 220
    rows = len(row_labels)
    row_gap = 34
    row_h = int((y1 - top_y - row_gap * (rows - 1) - 10) / rows)
    low_color = (247, 242, 232) if mode == "post" else (237, 242, 250)
    high_color = (233, 178, 95) if mode == "post" else (130, 157, 202)
    flat = [v for row in values for v in row]
    vmin = min(flat)
    vmax = max(flat)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    for i, row_label in enumerate(row_labels):
        cy = top_y + i * (row_h + row_gap)
        outline = INK if i == best_row else LINE
        width = 3 if i == best_row else 2
        draw.rounded_rectangle((x0 + 18, cy, x1 - 18, cy + row_h), radius=24, fill=WHITE, outline=outline, width=width)
        rank_fill = AMBER if i == best_row and mode == "post" else (146, 169, 207) if i == best_row else SLATE_LIGHT
        draw.ellipse((x0 + 24, cy + 22, x0 + 76, cy + 74), fill=rank_fill, outline=WHITE, width=2)
        rank_txt = str(i + 1)
        rb = draw.textbbox((0, 0), rank_txt, font=header_font)
        draw.text((x0 + 50 - (rb[2] - rb[0]) / 2, cy + 48 - (rb[3] - rb[1]) / 2), rank_txt, font=header_font, fill=INK)
        lines = row_label.split("\n")
        ly = cy + 18 if len(lines) == 2 else cy + 30
        for line in lines:
            lb = draw.textbbox((0, 0), line, font=row_font)
            draw.text((x0 + 108, ly), line, font=row_font, fill=MUTED)
            ly += 32

        for j, cx in enumerate(cols_x):
            val = values[i][j]
            t = (val - vmin) / (vmax - vmin)
            fill = interpolate_color(low_color, high_color, t)
            draw.rounded_rectangle((cx, cy + 18, cx + col_w, cy + 18 + col_h), radius=18, fill=fill, outline=(214, 218, 223), width=2)
            txt = f"{val:.3f}"
            tb = draw.textbbox((0, 0), txt, font=val_font)
            draw.text((cx + (col_w - (tb[2] - tb[0])) / 2, cy + 18 + (col_h - (tb[3] - tb[1])) / 2 - 3), txt, font=val_font, fill=INK)
def draw_calibration_estimates(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    row_labels: list[str],
    series_labels: list[str],
    values: list[list[float]],
    colors: list[tuple[int, int, int]],
    *,
    value_fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = rect
    font_axis = load_font(24)
    font_row = load_font(34)
    font_head = load_font(32)
    plot_x0 = x0 + 106
    plot_y0 = y0 + 76
    plot_x1 = x1 - 12
    plot_y1 = y1 - 24
    usable_h = plot_y1 - plot_y0
    data_x0 = plot_x0 + 54
    data_x1 = plot_x1 - 8
    usable_w = data_x1 - data_x0

    header_items = [(series_labels[0], colors[0], "circle"), (series_labels[1], colors[1], "circle")]
    header_w = measure_series_header_width(draw, header_items, font=font_head)
    draw_series_header(
        draw,
        (x1 - header_w - 8, y0 + 2),
        header_items,
        font=font_head,
    )

    x_min = 0.045
    x_max = 0.355
    tick_values = [0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    for tick_val in tick_values:
        x = data_x0 + usable_w * ((tick_val - x_min) / (x_max - x_min))
        draw.line((x, plot_y0, x, plot_y1), fill=GRID, width=2)
        label = f"{tick_val:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font_axis)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_y1 + 8), label, font=font_axis, fill=MUTED)

    draw.line((data_x0, plot_y0, data_x0, plot_y1), fill=LINE, width=3)
    draw.line((data_x0, plot_y1, data_x1, plot_y1), fill=LINE, width=3)

    row_h = usable_h / max(1, len(row_labels))
    offsets = [-42, 42]
    for row_idx, row_label in enumerate(row_labels):
        y_center = int(plot_y0 + row_h * (row_idx + 0.5))
        draw.line((data_x0, y_center, data_x1, y_center), fill=(241, 243, 246), width=1)
        bbox = draw.textbbox((0, 0), row_label, font=font_row)
        draw.text((data_x0 - 18 - (bbox[2] - bbox[0]), y_center - 12), row_label, font=font_row, fill=MUTED)

        for series_idx, (series_vals, color) in enumerate(zip(values, colors)):
            val = series_vals[row_idx]
            x_val = int(data_x0 + usable_w * ((val - x_min) / (x_max - x_min)))
            y = y_center + offsets[series_idx]
            draw.ellipse((x_val - 20, y - 20, x_val + 20, y + 20), fill=color, outline=WHITE, width=4)


def save_outputs(img: Image.Image, stem: str, report_lines: list[str]) -> None:
    ROOT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PKG_FIG_DIR.mkdir(parents=True, exist_ok=True)
    root_png = ROOT_FIG_DIR / f"{stem}.png"
    pkg_png = PKG_FIG_DIR / f"{stem}.png"
    img.save(root_png)
    shutil.copy2(root_png, pkg_png)
    report_path = PKG_FIG_DIR / f"{stem}_report.md"
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")


def draw_label_chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    text_fill: tuple[int, int, int] = INK,
    padding_x: int = 14,
    padding_y: int = 8,
    radius: int = 16,
) -> tuple[int, int, int, int]:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    rect = (x, y, x + w + 2 * padding_x, y + h + 2 * padding_y)
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline or fill, width=2)
    draw.text((x + padding_x, y + padding_y - 1), text, font=font, fill=text_fill)
    return rect


def outlined_tile(img: Image.Image, size: tuple[int, int], *, outline: tuple[int, int, int] = LINE, width: int = 2) -> Image.Image:
    tile = img.resize(size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, WHITE)
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((1, 1, size[0] - 2, size[1] - 2), radius=10, outline=outline, width=width)
    return canvas


def overlay_gt_on_rgb(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    base = np.clip(rgb.copy(), 0.0, 1.0)
    m = gt.astype(bool)
    if not m.any():
        return base
    base[m, 0] = np.clip(base[m, 0] * 0.4 + 0.6, 0.0, 1.0)
    base[m, 1] = np.clip(base[m, 1] * 0.5, 0.0, 1.0)
    base[m, 2] = np.clip(base[m, 2] * 0.5, 0.0, 1.0)
    return base


def reference_overlay_on_rgb(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    base = np.clip(rgb.copy(), 0.0, 1.0)
    gray = base.mean(axis=2, keepdims=True)
    base = 0.30 * base + 0.70 * gray
    base = np.clip(base * 0.92 + 0.05, 0.0, 1.0)
    m = gt.astype(bool)
    if not m.any():
        return base
    amber = np.array([233, 176, 79], dtype=np.float32) / 255.0
    outline = np.array([232, 160, 60], dtype=np.float32) / 255.0
    base[m] = 0.20 * base[m] + 0.80 * amber
    bd = boundary_mask(m)
    base[bd] = outline
    return np.clip(base, 0.0, 1.0)


def crop_s2_case_tile(sheet: Image.Image, row_idx: int, col_idx: int) -> Image.Image:
    row_boxes = [(274, 59, 486, 271), (274, 303, 486, 515)]
    col_boxes = [
        (274, 486),
        (494, 706),
        (714, 926),
        (934, 1146),
    ]
    y0, y1 = row_boxes[row_idx][1], row_boxes[row_idx][3]
    x0, x1 = col_boxes[col_idx]
    return sheet.crop((x0, y0, x1, y1))


def s2_focus_box(reference_tile: Image.Image) -> tuple[int, int, int, int]:
    arr = np.array(reference_tile.convert("RGB"))
    red = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 120) & (arr[:, :, 2] < 120)
    ys, xs = np.where(red)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    size = max(x1 - x0 + 1, y1 - y0 + 1) * 1.8 + 40
    size = min(max(size, 132), arr.shape[1])
    half = size / 2.0
    lx = max(0, int(round(cx - half)))
    rx = min(arr.shape[1], int(round(cx + half)))
    ty = max(0, int(round(cy - half)))
    by = min(arr.shape[0], int(round(cy + half)))
    side = min(max(rx - lx, by - ty), arr.shape[1], arr.shape[0])
    if (rx - lx) < side:
        extra = side - (rx - lx)
        lx = max(0, lx - extra // 2)
        rx = min(arr.shape[1], lx + side)
        lx = rx - side
    if (by - ty) < side:
        extra = side - (by - ty)
        ty = max(0, ty - extra // 2)
        by = min(arr.shape[0], ty + side)
        ty = by - side
    return lx, ty, rx, by


def overlay_reference_outline(base_tile: Image.Image, reference_tile: Image.Image) -> Image.Image:
    base = np.array(base_tile.convert("RGB"))
    ref = np.array(reference_tile.convert("RGB"))
    red = (ref[:, :, 0] > 150) & (ref[:, :, 1] < 120) & (ref[:, :, 2] < 120)
    bd = boundary_mask(red)
    amber = np.array([232, 170, 76], dtype=np.uint8)
    base[bd] = amber
    return Image.fromarray(base)


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


def render_change_triptych(sample_id: str) -> tuple[Image.Image, Image.Image, Image.Image]:
    cache_path = ROOT / "processed" / "hybrid_pinn" / "strict_t2_change_rgb_cache_v1" / "test_changergb_p128.h5"
    with h5py.File(cache_path, "r") as h5:
        sample_ids = [raw.decode("utf-8") for raw in h5["sample_id"][:]]
        idx = sample_ids.index(sample_id)
        image = h5["image"][idx].astype(np.float32)
        mask = h5["mask"][idx, 0] > 0.5

    outline = boundary_mask(mask)
    amber = np.array([229, 166, 73], dtype=np.uint8)

    pre = np.array(stretch_rgb(image[:3]))
    post = np.array(stretch_rgb(image[3:6]))
    for arr in (pre, post):
        arr[outline] = amber

    ref = (reference_overlay_on_rgb(post.astype(np.float32) / 255.0, mask) * 255.0).astype(np.uint8)

    return Image.fromarray(pre), Image.fromarray(post), Image.fromarray(ref)


def render_figure4() -> None:
    canvas = Image.new("RGB", (2400, 1300), WHITE)
    draw = ImageDraw.Draw(canvas)
    panel_title = load_font(40, bold=True)
    left_panel = draw_minimal_panel(draw, (64, 48, 1234, 1238), "(a) Frontier segmentation and transfer", title_font=panel_title)
    right_panel = draw_minimal_panel(draw, (1238, 48, 2358, 1238), "(b) Spatial-OOD calibration", title_font=panel_title)

    series_labels = ["Stage-1", "Transition", "Final"]
    colors = [SLATE, SKY, AMBER]
    draw_frontier_estimates(
        draw,
        left_panel,
        ["DLR in-domain", "DLR spatial-OOD", "GLaD4CD\nzero-shot"],
        series_labels,
        [
            [0.6452, 0.4987, 0.2460],
            [0.6757, 0.4819, 0.2278],
            [0.6924, 0.5404, 0.2778],
        ],
        [
            [0.0612, 0.0751, 0.0225],
            [0.0272, 0.0239, 0.0740],
            [0.0281, 0.0091, 0.0354],
        ],
        colors,
        x_min=0.1,
        x_max=0.8,
        tick_step=0.1,
    )
    draw_calibration_estimates(
        draw,
        right_panel,
        ["ECE", "Brier", "NLL"],
        ["Stage-1", "Final"],
        [
            [0.0747, 0.0747, 0.3479],
            [0.0591, 0.0651, 0.3320],
        ],
        [SLATE, AMBER],
    )

    save_outputs(
        canvas,
        "figure4_dlr_frontier_summary",
        [
            "# figure4_dlr_frontier_summary",
            "",
            "- status: generated for main-text Figure 4",
            "- source summaries:",
            "  - experiments/geophysadapter_v3_e3_vs_v2_stage1.md",
            "  - experiments/geophysadapter_v3_e3_reliability_zeroshot_vs_stage1.md",
            "",
            "Frozen values:",
            "- DLR testind IoU: Stage-1 0.6452 +- 0.0612, transition line (GeoPhysAdapter-v2) 0.6757 +- 0.0272, final model (GeoPhysAdapter-v3) 0.6924 +- 0.0281",
            "- DLR testspt IoU: Stage-1 0.4987 +- 0.0751, transition line (GeoPhysAdapter-v2) 0.4819 +- 0.0239, final model (GeoPhysAdapter-v3) 0.5404 +- 0.0091",
            "- GLaD4CD zero-shot IoU: Stage-1 0.2460 +- 0.0225, transition line (GeoPhysAdapter-v2) 0.2278 +- 0.0740, final model (GeoPhysAdapter-v3) 0.2778 +- 0.0354",
            "- testspt post-calibration: Stage-1 ECE/Brier/NLL 0.0747 / 0.0747 / 0.3479; final model 0.0591 / 0.0651 / 0.3320",
        ],
    )


def render_figure5() -> None:
    """Render Figure 5 with direct prediction results for both regimes."""
    import json
    import torch

    from render_strict_t2_postrgb_case_panels import (
        build_index,
        choose_test_cache,
        load_case_arrays,
        load_model as load_post_model,
        overlay_rgb,
    )
    from train_strict_t2_change_rgb_baseline import SmallUNet
    from train_strict_t2_change_rgb_phys_baseline import PhysicsFiLMUNet
    from train_strict_t2_postrgb_phys_baseline import load_physics_maps
    canvas_w = 1800
    a_tile = 390  
    a_gap = 12
    b_tile = 310  
    b_gap = 10
    margin = 30
    title_h = 50  
    header_h = 32
    a_metric_h = 40  
    a_row_gap = 16
    b_row_gap = 12
    b_metric_h = 48  
    a_content_h = header_h + 2 * (a_tile + a_metric_h) + a_row_gap
    b_content_h = header_h + 3 * b_tile + 2 * b_row_gap + b_metric_h
    
    canvas_h = margin + title_h + a_content_h + 24 + title_h + b_content_h + margin
    
    canvas = Image.new("RGB", (canvas_w, canvas_h), WHITE)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(38, bold=True)      
    header_font = load_font(26, bold=True)     
    row_tag_font = load_font(26, bold=True)    
    metric_label_font = load_font(26)          
    metric_value_font = load_font(26, bold=True)  
    a_top = margin
    a_box = (margin, a_top + title_h, canvas_w - margin, a_top + title_h + a_content_h)
    b_top = a_box[3] + 24
    b_box = (margin, b_top + title_h, canvas_w - margin, b_top + title_h + b_content_h)
    
    panel_outline = (218, 223, 230)
    draw.rounded_rectangle(a_box, radius=12, fill=WHITE, outline=panel_outline, width=2)
    draw.rounded_rectangle(b_box, radius=12, fill=WHITE, outline=panel_outline, width=2)
    draw.text((a_box[0], a_top), "(a) Post-event view", font=title_font, fill=INK)
    draw.text((b_box[0], b_top), "(b) Change-view control", font=title_font, fill=INK)

    paired = pd.read_csv(FIG5_PAIRED_CSV).set_index("sample_id")
    device = torch.device("cpu")

    post_visual_summary = (
        ROOT
        / "experiments"
        / "strict_t2_postrgb_deeplabv3_resnet50_visual_e3_sp05_lb025_thr_seed20260311_localenv"
        / "summary.json"
    )
    post_v4_summary = (
        ROOT
        / "experiments"
        / "strict_t2_postrgb_deeplabv3_resnet50_v4_pilot_e3_sp05_lb025_distill_v1"
        / "summary.json"
    )
    post_visual_model = load_post_model(post_visual_summary, device)
    post_v4_model = load_post_model(post_v4_summary, device)
    post_cache = choose_test_cache(post_visual_model.summary)
    post_index = build_index(post_cache)

    change_visual_summary_path = (
        ROOT
        / "experiments"
        / "strict_t2_change_rgb_visual_bal256_e3_sp05_lb025_thr_v1"
        / "summary.json"
    )
    change_material_summary_path = (
        ROOT
        / "experiments"
        / "strict_t2_change_rgb_material_bal256_e3_sp05_lb025_thr_v1"
        / "summary.json"
    )
    change_visual_summary = json.loads(change_visual_summary_path.read_text(encoding="utf-8"))
    change_material_summary = json.loads(change_material_summary_path.read_text(encoding="utf-8"))

    change_visual = SmallUNet(in_ch=6, base=32)
    change_visual_ckpt = torch.load(
        Path(_convert_wsl_path(change_visual_summary["outdir"])) / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    change_visual.load_state_dict(change_visual_ckpt["model"])
    change_visual = change_visual.to(device)
    change_visual.eval()

    change_material = PhysicsFiLMUNet(
        in_ch=6,
        physics_dim=int(change_material_summary["physics_dim"]),
        base=32,
        hidden_dim=int(change_material_summary["config"]["hidden_dim"]),
        dropout=float(change_material_summary["config"]["dropout"]),
        physics_cols=list(change_material_summary["physics_vector_cols"]),
        dynamic_gate_mode=str(change_material_summary["config"].get("dynamic_gate_mode", "none")),
        dynamic_gate_source=str(change_material_summary["config"].get("dynamic_gate_source", "proxy")),
        dynamic_gate_scale=float(change_material_summary["config"].get("dynamic_gate_scale", 1.0)),
    )
    change_material_ckpt = torch.load(
        Path(_convert_wsl_path(change_material_summary["outdir"])) / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    change_material.load_state_dict(change_material_ckpt["model"])
    change_material = change_material.to(device)
    change_material.eval()

    change_cache = Path(_convert_wsl_path(change_visual_summary["config"]["test_cache_h5"]))
    with h5py.File(change_cache, "r") as f:
        change_sample_ids = [raw.decode("utf-8") for raw in f["sample_id"][:]]
    change_index = {sample_id: idx for idx, sample_id in enumerate(change_sample_ids)}
    sample_map, event_map = load_physics_maps(
        ROOT
        / "processed"
        / "hybrid_pinn"
        / "strict_t2_supervised_ready_v1"
        / "sample_physics_vectors_change_rgb_v1.csv",
        list(change_material_summary["physics_vector_cols"]),
    )
    change_mean = np.asarray(change_material_summary["physics_norm"]["mean"], dtype=np.float32)
    change_std = np.asarray(change_material_summary["physics_norm"]["std"], dtype=np.float32)

    def focus_box(mask: np.ndarray, *, scale: float = 2.2, y_shift_frac: float = 0.0) -> tuple[int, int, int, int]:
        h, w = mask.shape
        if mask.any():
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            bh, bw = y1 - y0, x1 - x0
            side = int(max(bh, bw) * scale)
            side = max(side, 72)
            cy = (y0 + y1) / 2.0 + y_shift_frac * side
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

    def crop_to_tile(arr: np.ndarray, box: tuple[int, int, int, int], size: int, *, top_cut_frac: float = 0.0) -> Image.Image:
        x0, y0, x1, y1 = box
        crop = arr[y0:y1, x0:x1]
        if top_cut_frac > 1e-6:
            cut = int(crop.shape[0] * top_cut_frac)
            crop = crop[cut:, :, :]
        crop = np.clip(crop, 0.0, 1.0)
        return Image.fromarray((crop * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)

    def render_metric_pair(x: int, y: int, label: str, value_text: str, accent: tuple[int, int, int], *, delta_text: str | None = None) -> None:
        r = 8
        draw.ellipse((x, y + 5, x + 2 * r, y + 5 + 2 * r), fill=accent, outline=accent)
        draw.text((x + 2 * r + 8, y), label, font=metric_label_font, fill=MUTED)
        lb = draw.textbbox((0, 0), label, font=metric_label_font)
        vx = x + 2 * r + 8 + (lb[2] - lb[0]) + 10
        draw.text((vx, y - 1), value_text, font=metric_value_font, fill=accent)
        if delta_text:
            vb = draw.textbbox((0, 0), value_text, font=metric_value_font)
            draw.text((vx + (vb[2] - vb[0]) + 6, y - 1), delta_text, font=metric_value_font, fill=accent)
    a_stub_x = a_box[0] + 12
    a_stub_w = 38
    a_total_w = 4 * a_tile + 3 * a_gap
    a_start_x = a_stub_x + a_stub_w + (a_box[2] - a_box[0] - a_stub_w - a_total_w) // 2
    a_cols = [
        ("Post-event", a_start_x),
        ("Ground truth", a_start_x + (a_tile + a_gap)),
        ("Visual-only", a_start_x + 2 * (a_tile + a_gap)),
        ("Residual-prior", a_start_x + 3 * (a_tile + a_gap)),
    ]
    a_header_y = a_box[1] + 6
    for lab, x in a_cols:
        bbox = draw.textbbox((0, 0), lab, font=header_font)
        draw.text((x + (a_tile - (bbox[2] - bbox[0])) / 2, a_header_y), lab, font=header_font, fill=MUTED)

    a_row_y = a_box[1] + header_h + 4
    a_cases = [
        ("DLR", "EID_BR0001__SID_00380", 0.00, 0.00),
        ("CAS", "CAS_Palu::Palu0939", 0.30, 0.34),
    ]

    def render_post_row(y0: int, case_tag: str, sample_id: str, *, y_shift_frac: float, top_cut_frac: float) -> None:
        row = paired.loc[sample_id]
        item = load_case_arrays(post_cache, post_index[sample_id])
        image = np.asarray(item["image"], dtype=np.float32)
        gt = np.asarray(item["mask"], dtype=bool)
        valid = np.asarray(item["valid"], dtype=bool)
        event_uid = str(item["event_uid"])
        pred_visual = post_visual_model.predict(image, sample_id, event_uid, device) & valid
        pred_v4 = post_v4_model.predict(image, sample_id, event_uid, device) & valid

        rgb_hw = np.asarray(stretch_rgb(image[:3]), dtype=np.float32) / 255.0
        rgb_ch = np.transpose(rgb_hw, (2, 0, 1))
        box = focus_box(gt, scale=2.15, y_shift_frac=y_shift_frac)
        panels = [
            crop_to_tile(rgb_hw, box, a_tile, top_cut_frac=top_cut_frac),
            crop_to_tile(overlay_rgb(rgb_ch, gt, None, valid), box, a_tile, top_cut_frac=top_cut_frac),
            crop_to_tile(overlay_rgb(rgb_ch, gt, pred_visual, valid), box, a_tile, top_cut_frac=top_cut_frac),
            crop_to_tile(overlay_rgb(rgb_ch, gt, pred_v4, valid), box, a_tile, top_cut_frac=top_cut_frac),
        ]

        draw.text((a_stub_x, y0 + a_tile // 2 - 8), case_tag, font=row_tag_font, fill=INK)
        for (_, x), panel in zip(a_cols, panels):
            canvas.paste(outlined_tile(panel, (a_tile, a_tile), outline=LINE, width=2), (x, y0))
        metrics_y = y0 + a_tile + 6
        legend_x1 = a_box[0] + 80   
        legend_x2 = a_box[0] + (a_box[2] - a_box[0]) * 0.42  
        legend_x3 = a_box[0] + (a_box[2] - a_box[0]) * 0.72  
        render_metric_pair(int(legend_x1), metrics_y, "Visual-only", f"{float(row['visual_iou_mean']):.3f}", SLATE)
        render_metric_pair(int(legend_x2), metrics_y, "Ground truth", "", (220, 80, 80))  
        render_metric_pair(
            int(legend_x3),
            metrics_y,
            "Residual-prior",
            f"{float(row['v4_iou_mean']):.3f}",
            AMBER,
            delta_text=f"(+{float(row['mean_delta']):.3f})",
        )
    a_row_h = a_tile + a_metric_h
    render_post_row(a_row_y, *a_cases[0][:2], y_shift_frac=a_cases[0][2], top_cut_frac=a_cases[0][3])
    sep_y = a_row_y + a_row_h + a_row_gap // 2
    draw.line((a_box[0] + 14, sep_y, a_box[2] - 14, sep_y), fill=(235, 238, 242), width=2)
    render_post_row(a_row_y + a_row_h + a_row_gap, *a_cases[1][:2], y_shift_frac=a_cases[1][2], top_cut_frac=a_cases[1][3])

    # Panel (b): direct prediction comparison under change-view
    def change_predict_visual(image: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = change_visual(x)
            return torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= float(change_visual_summary["eval_threshold"])

    def change_predict_material(image: np.ndarray, sample_id: str, event_uid: str) -> np.ndarray:
        x = torch.from_numpy(image).unsqueeze(0).to(device)
        zero = np.zeros_like(change_mean, dtype=np.float32)
        vec = sample_map.get(sample_id, event_map.get(event_uid, zero))
        vec = np.nan_to_num((vec - change_mean) / change_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        p = torch.from_numpy(vec).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = change_material(x, p)
            return torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= float(change_material_summary["eval_threshold"])
    b_stub_x = b_box[0] + 12
    b_stub_w = 38
    b_total_w = 5 * b_tile + 4 * b_gap
    b_start_x = b_stub_x + b_stub_w + (b_box[2] - b_box[0] - b_stub_w - b_total_w) // 2
    b_cols = [
        ("Pre-event", b_start_x),
        ("Post-event", b_start_x + (b_tile + b_gap)),
        ("Ground truth", b_start_x + 2 * (b_tile + b_gap)),
        ("SmallUNet visual", b_start_x + 3 * (b_tile + b_gap)),
        ("Material-only", b_start_x + 4 * (b_tile + b_gap)),
    ]
    b_header_y = b_box[1] + 6
    for lab, x in b_cols:
        bbox = draw.textbbox((0, 0), lab, font=header_font)
        draw.text((x + (b_tile - (bbox[2] - bbox[0])) / 2, b_header_y), lab, font=header_font, fill=MUTED)

    b_row_y = b_box[1] + header_h + 4
    b_cases = [
        ("DLR", "EID_BR0001__SID_00408"),
        ("DLR", "EID_BR0001__SID_00352"),
        ("DLR", "EID_BR0001__SID_00409"),
    ]

    def render_change_row(y0: int, case_tag: str, sample_id: str) -> None:
        with h5py.File(change_cache, "r") as f:
            idx = change_index[sample_id]
            image = np.asarray(f["image"][idx], dtype=np.float32)
            gt = np.asarray(f["mask"][idx, 0] > 0.5)
            valid = np.asarray(f["valid"][idx, 0] > 0.5)
            event_uid = f["event_uid"][idx].decode("utf-8")

        pred_visual = change_predict_visual(image) & valid
        pred_material = change_predict_material(image, sample_id, event_uid) & valid
        pre_hw = np.asarray(stretch_rgb(image[:3]), dtype=np.float32) / 255.0
        post_hw = np.asarray(stretch_rgb(image[3:6]), dtype=np.float32) / 255.0
        post_ch = np.transpose(post_hw, (2, 0, 1))
        box = focus_box(gt, scale=2.12, y_shift_frac=0.0)
        panels = [
            crop_to_tile(pre_hw, box, b_tile),
            crop_to_tile(post_hw, box, b_tile),
            crop_to_tile(overlay_rgb(post_ch, gt, None, valid), box, b_tile),
            crop_to_tile(overlay_rgb(post_ch, gt, pred_visual, valid), box, b_tile),
            crop_to_tile(overlay_rgb(post_ch, gt, pred_material, valid), box, b_tile),
        ]
        draw.text((b_stub_x, y0 + b_tile // 2 - 12), case_tag, font=row_tag_font, fill=INK)
        for (_, x), panel in zip(b_cols, panels):
            canvas.paste(outlined_tile(panel, (b_tile, b_tile), outline=LINE, width=2), (x, y0))
    b_row_h = b_tile  
    for idx, (case_tag, sample_id) in enumerate(b_cases):
        render_change_row(b_row_y + idx * (b_row_h + b_row_gap), case_tag, sample_id)
    last_row_bottom = b_row_y + 2 * (b_row_h + b_row_gap) + b_tile
    legend_y = last_row_bottom + 10
    b_legend_x1 = b_box[0] + 80   
    b_legend_x2 = b_box[0] + (b_box[2] - b_box[0]) * 0.42  
    b_legend_x3 = b_box[0] + (b_box[2] - b_box[0]) * 0.72  
    render_metric_pair(int(b_legend_x1), legend_y, "SmallUNet visual", "0.442 / 0.465", SLATE)
    render_metric_pair(int(b_legend_x2), legend_y, "Ground truth", "", (220, 80, 80))  
    render_metric_pair(int(b_legend_x3), legend_y, "Material-only", "0.361 / 0.355", AMBER)

    save_outputs(
        canvas,
        "figure5_protocol_inversion",
        [
            "# figure5_protocol_inversion",
            "",
            "- status: generated for main-text Figure 5",
            "- qualitative-first comparison figure contrasting the post-event and change-view task regimes with direct prediction overlays",
            "- source assets:",
            "  - processed/hybrid_pinn/strict_t2_change_rgb_cache_v1/test_changergb_p128.h5",
            "  - experiments/strict_t2_postrgb_v4_vs_visual_paired_v1/paired_sample_mean_diff.csv",
            "- source summaries:",
            f"  - {post_visual_summary}",
            f"  - {post_v4_summary}",
            f"  - {change_visual_summary_path}",
            f"  - {change_material_summary_path}",
            "",
            "Frozen post-event evidence:",
            "- DLR corresponds to EID_BR0001__SID_00380: visual-only 0.636, residual-prior 0.784, delta +0.148",
            "- CAS corresponds to CAS_Palu::Palu0939: visual-only 0.259, residual-prior 0.471, delta +0.212",
            "Representative change-view task patches:",
            "- EID_BR0001__SID_00408",
            "- EID_BR0001__SID_00352",
            "- EID_BR0001__SID_00409",
            "View-level summary:",
            "- SmallUNet visual: val-thr 0.442099, IoU@0.50 0.464649",
            "- SmallUNet material-only: val-thr 0.361436, IoU@0.50 0.355444",
        ],
    )


def render_figure6() -> None:
    canvas = Image.new("RGB", (2200, 1260), WHITE)
    draw = ImageDraw.Draw(canvas)
    panel_title = load_font(38, bold=True)  
    left = draw_minimal_panel(draw, (72, 52, 1080, 1160), "(a) DLR control comparisons", title_font=panel_title)
    right = draw_minimal_panel(draw, (1140, 52, 2148, 1160), "(b) Heterogeneous transfer controls", title_font=panel_title)

    dlr_ref_testind = 0.704688
    dlr_ref_testspt = 0.481827
    dlr_testind = [0.632805, 0.638488, 0.636403]
    dlr_testspt = [0.434662, 0.437011, 0.456809]
    dlr_delta_testind = [v - dlr_ref_testind for v in dlr_testind]
    dlr_delta_testspt = [v - dlr_ref_testspt for v in dlr_testspt]

    heter_ref_thr = 0.190879
    heter_ref_fix = 0.163659
    heter_thr = [0.164487, 0.170630, 0.167100]
    heter_fix = [0.141247, 0.144759, 0.143000]
    heter_delta_thr = [v - heter_ref_thr for v in heter_thr]
    heter_delta_fix = [v - heter_ref_fix for v in heter_fix]

    draw_delta_reference_chart(
        draw,
        left,
        ["visual-only", "no-routing", "no-state-heads"],
        ["DLR in-domain", "DLR spatial-OOD"],  
        [dlr_delta_testind, dlr_delta_testspt],
        [BLUE, SLATE],
        x_min=-0.090,
        x_max=0.005,
        tick_step=0.02,
        axis_label="ΔIoU relative to the reference model",
    )
    draw_delta_reference_chart(
        draw,
        right,
        ["no-distill", "no-source-id", "metadata-only"],
        ["IoU@val-thr", "IoU@0.50"],
        [heter_delta_thr, heter_delta_fix],
        [TEAL, SAND],
        x_min=-0.035,
        x_max=0.005,
        tick_step=0.01,
        axis_label="ΔIoU relative to the reference model",
    )

    save_outputs(
        canvas,
        "figure6_targeted_controls",
        [
            "# figure6_targeted_controls",
            "",
            "- status: generated for main-text Figure 6",
            "- source summary: experiments/reviewer_targeted_ablation_controls_20260311.md",
            "",
            "Frozen values:",
            "- DLR full v3 ref: testind/testspt = 0.704688 / 0.481827",
            "- DLR visual-only: 0.632805 / 0.434662",
            "- DLR no-routing: 0.638488 / 0.437011",
            "- DLR no-state-heads: 0.636403 / 0.456809",
            "- strict_t2 v4 full ref: tuned/default = 0.190879 / 0.163659",
            "- strict_t2 v4 no-distill: tuned/default = 0.164487 / 0.141247",
            "- strict_t2 v4 no-source-id: tuned/default = 0.170630 / 0.144759",
            "- strict_t2 v4 metadata-only: tuned/default = 0.167100 / 0.143000",
        ],
    )


def render_figure7() -> None:
    import torch

    from render_strict_t2_postrgb_case_panels import (
        build_index,
        choose_test_cache,
        load_case_arrays,
        load_model as load_post_model,
        overlay_rgb,
        resolve_path,
    )
    from train_strict_t2_postrgb_phys_baseline import PhysicsFiLMUNet, load_physics_maps

    def focus_box(mask: np.ndarray, *, scale: float = 2.2, y_shift_frac: float = 0.0) -> tuple[int, int, int, int]:
        h, w = mask.shape
        if mask.any():
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            bh, bw = y1 - y0, x1 - x0
            side = int(max(bh, bw) * scale)
            side = max(side, 72)
            cy = (y0 + y1) / 2.0 + y_shift_frac * side
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

    def crop_to_tile(arr: np.ndarray, box: tuple[int, int, int, int], size: int, *, top_cut_frac: float = 0.0) -> Image.Image:
        x0, y0, x1, y1 = box
        crop = arr[y0:y1, x0:x1]
        if top_cut_frac > 1e-6:
            cut = int(crop.shape[0] * top_cut_frac)
            crop = crop[cut:, :, :]
        crop = np.clip(crop, 0.0, 1.0)
        return Image.fromarray((crop * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)

    def draw_metric_under_tile(
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        *,
        label: str,
        value: str,
        color: tuple[int, int, int],
        label_font: ImageFont.ImageFont,
        value_font: ImageFont.ImageFont,
    ) -> None:
        r = 7
        draw.ellipse((x, y + 4, x + 2 * r, y + 4 + 2 * r), fill=color, outline=color)
        draw.text((x + 2 * r + 8, y), label, font=label_font, fill=MUTED)
        lb = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x + 2 * r + 8 + (lb[2] - lb[0]) + 10, y - 1), value, font=value_font, fill=color)

    def load_phys_baseline_bundle(summary_path: Path, device: torch.device) -> dict[str, object]:
        import json

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
    canvas_w = 1900
    canvas_h = 1700
    margin = 30
    canvas = Image.new("RGB", (canvas_w, canvas_h), WHITE)
    draw = ImageDraw.Draw(canvas)

    # Typography aligned to Figure 5
    title_font = load_font(40, bold=True)
    header_font = load_font(28, bold=True)
    row_tag_font = load_font(28, bold=True)
    row_note_font = load_font(30)  # 30pt
    iou_font = load_font(32, bold=True)  

    outer = (margin, margin + 60, canvas_w - margin, canvas_h - margin)

    stub_x = outer[0] + 20
    stub_w = 160
    tile = 280  
    gap = 12  
    cols = ["Post-event", "Ground truth", "Visual", "Material", "Full physics"]
    total_w = len(cols) * tile + (len(cols) - 1) * gap
    start_x = stub_x + stub_w + (outer[2] - outer[0] - stub_w - total_w) // 2
    col_xs = [start_x + i * (tile + gap) for i in range(len(cols))]
    header_y = outer[1] - 6  
    title_text = "Representative component-level cases"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_x = (canvas_w - (title_bbox[2] - title_bbox[0])) // 2
    draw.text((title_x, margin), title_text, font=title_font, fill=INK)
    for lab, x in zip(cols, col_xs):
        bbox = draw.textbbox((0, 0), lab, font=header_font)
        draw.text((x + (tile - (bbox[2] - bbox[0])) / 2, header_y + 10), lab, font=header_font, fill=INK)

    device = torch.device("cpu")
    # Use the original case-panel candidate runs so the rendered behaviors
    # match the curated material/full-physics gain tables used to pick Figure 7 cases.
    visual_summary = (
        ROOT
        / "experiments"
        / "strict_t2_postrgb_visual_nearfull_bal2048_e3_v1"
        / "summary.json"
    )
    material_summary = (
        ROOT
        / "experiments"
        / "strict_t2_postrgb_physcache_material_nearfull_bal2048_e3_v1"
        / "summary.json"
    )
    full_summary = (
        ROOT
        / "experiments"
        / "strict_t2_postrgb_physcache_full_nearfull_bal2048_e3_v1"
        / "summary.json"
    )
    visual_model = load_post_model(visual_summary, device)
    material_model = load_phys_baseline_bundle(material_summary, device)
    full_model = load_phys_baseline_bundle(full_summary, device)
    cache_h5 = choose_test_cache(visual_model.summary)
    sample_index = build_index(cache_h5)

    cases = [
        ("DLR", "material gain", "EID_BR0001__SID_00325", 0.00, 0.00),
        ("DLR", "bridge / mixed", "EID_PH0001__SID_00024", 0.00, 0.00),
        ("DLR", "stable\nmaterial gain", "EID_PH0001__SID_00049", 0.00, 0.00),
        ("CAS", "instability gain", "CAS_Palu::Palu0373", 0.00, 0.00),
        ("CAS", "full-physics gain", "CAS_Palu::Palu0954", 0.00, 0.00),
    ]

    row_top = outer[1] + 50
    row_gap = 16  
    row_h = tile  

    for idx, (case_tag, case_note, sample_id, y_shift_frac, top_cut_frac) in enumerate(cases):
        y0 = row_top + idx * (row_h + row_gap)
        item = load_case_arrays(cache_h5, sample_index[sample_id])
        image = np.asarray(item["image"], dtype=np.float32)
        gt = np.asarray(item["mask"], dtype=bool)
        valid = np.asarray(item["valid"], dtype=bool)
        event_uid = str(item["event_uid"])

        pred_visual = visual_model.predict(image, sample_id, event_uid, device) & valid
        pred_material = predict_phys_baseline(material_model, image, sample_id, event_uid, device) & valid
        pred_full = predict_phys_baseline(full_model, image, sample_id, event_uid, device) & valid

        rgb_hw = np.asarray(stretch_rgb(image[:3]), dtype=np.float32) / 255.0
        rgb_ch = np.transpose(rgb_hw, (2, 0, 1))
        box = focus_box(gt, scale=2.15, y_shift_frac=y_shift_frac)
        panels = [
            crop_to_tile(rgb_hw, box, tile, top_cut_frac=top_cut_frac),
            crop_to_tile(overlay_rgb(rgb_ch, gt, None, valid), box, tile, top_cut_frac=top_cut_frac),
            crop_to_tile(overlay_rgb(rgb_ch, gt, pred_visual, valid), box, tile, top_cut_frac=top_cut_frac),
            crop_to_tile(overlay_rgb(rgb_ch, gt, pred_material, valid), box, tile, top_cut_frac=top_cut_frac),
            crop_to_tile(overlay_rgb(rgb_ch, gt, pred_full, valid), box, tile, top_cut_frac=top_cut_frac),
        ]

        draw.text((stub_x, y0 + tile // 2 - 30), case_tag, font=row_tag_font, fill=INK)
        draw.multiline_text((stub_x, y0 + tile // 2 + 10), case_note, font=row_note_font, fill=MUTED, spacing=4)
        iou_visual = (pred_visual & gt).sum() / ((pred_visual | gt).sum() if (pred_visual | gt).sum() else 1)
        iou_material = (pred_material & gt).sum() / ((pred_material | gt).sum() if (pred_material | gt).sum() else 1)
        iou_full = (pred_full & gt).sum() / ((pred_full | gt).sum() if (pred_full | gt).sum() else 1)
        iou_values = [None, None, f"{iou_visual:.3f}", f"{iou_material:.3f}", f"{iou_full:.3f}"]
        
        for i, (x, panel) in enumerate(zip(col_xs, panels)):
            if iou_values[i]:
                panel_draw = ImageDraw.Draw(panel)
                iou_text = iou_values[i]
                iou_bbox = panel_draw.textbbox((0, 0), iou_text, font=iou_font)
                iou_w = iou_bbox[2] - iou_bbox[0]
                iou_h = iou_bbox[3] - iou_bbox[1]
                padding = 6
                iou_x = tile - iou_w - 12
                iou_y = tile - iou_h - 12
                bg_rect = (iou_x - padding, iou_y - padding, iou_x + iou_w + padding, iou_y + iou_h + padding)
                overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rounded_rectangle(bg_rect, radius=6, fill=(60, 60, 60, 200))
                panel = Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")
                panel_draw = ImageDraw.Draw(panel)
                panel_draw.text((iou_x, iou_y), iou_text, font=iou_font, fill=(255, 255, 255))
            canvas.paste(outlined_tile(panel, (tile, tile), outline=LINE, width=2), (x, y0))

        if idx < len(cases) - 1:
            sep_y = y0 + row_h + row_gap // 2
            draw.line((outer[0] + 14, sep_y, outer[2] - 14, sep_y), fill=(235, 238, 242), width=2)

    save_outputs(
        canvas,
        "figure7_qualitative_cases",
        [
            "# figure7_qualitative_cases",
            "",
            "- status: regenerated for manuscript Figure 7",
            "- purpose: representative component-level cases under the heterogeneous-benchmark post-event view",
            f"- visual summary: {visual_summary}",
            f"- material summary: {material_summary}",
            f"- full summary: {full_summary}",
            "",
            "Rows:",
            "- DLR | material gain | EID_BR0001__SID_00325",
            "- DLR | bridge / mixed | EID_PH0001__SID_00024",
            "- DLR | stable material gain | EID_PH0001__SID_00049",
            "- CAS | instability gain | CAS_Palu::Palu0373",
            "- CAS | full-physics gain | CAS_Palu::Palu0954",
        ],
    )


def render_figure8() -> None:
    """Render Figure 8 as a quantitative UGCoP-oriented mechanism figure."""

    def blend(base: tuple[int, int, int], target: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return tuple(int(base[i] + (target[i] - base[i]) * t) for i in range(3))

    def fmt_delta(val: float) -> str:
        return f"{val:+.3f}"
    canvas_w = 1800
    canvas_h = 1300
    margin = 32
    gutter = 32
    top_h = 470
    bottom_h = 600

    canvas = Image.new("RGB", (canvas_w, canvas_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(38, bold=True)
    panel_font = load_font(34, bold=True)
    axis_font = load_font(26)
    tick_font = load_font(24)
    label_font = load_font(28, bold=True)
    body_font = load_font(24)
    small_font = load_font(22)
    value_font = load_font(28, bold=True)
    value_font_neg = load_font(25, bold=True)
    value_font_neg = load_font(25, bold=True)
    panel_a = (margin, margin + 80, canvas_w - margin, margin + 80 + top_h)
    bc_vertical_offset = 40
    panel_b = (
        margin,
        panel_a[3] + gutter + bc_vertical_offset,
        margin + (canvas_w - 2 * margin - gutter) // 2,
        panel_a[3] + gutter + bottom_h + bc_vertical_offset,
    )
    panel_c = (
        panel_b[2] + gutter,
        panel_a[3] + gutter + bc_vertical_offset,
        canvas_w - margin,
        panel_a[3] + gutter + bottom_h + bc_vertical_offset,
    )

    for rect in [panel_a, panel_b, panel_c]:
        draw.rounded_rectangle(rect, radius=18, fill=WHITE, outline=LINE, width=2)

    # ---------------- Panel (a): regime-dependent gain sign ----------------
    title_a = "(a) Regime-dependent sign of physics-guided gains"
    tb_a = draw.textbbox((0, 0), title_a, font=panel_font)
    draw.text(
        (panel_a[0], panel_a[1] - (tb_a[3] - tb_a[1]) - 10),
        title_a,
        font=panel_font,
        fill=INK,
    )
    center_x = canvas_w // 2
    a_left = panel_a[0] + 60
    a_top = panel_a[1] + 20
    span = int((center_x - a_left) * 3 / 2)  
    a_right = a_left + span
    a_bottom = panel_a[3] - 90

    zero_min, zero_max = -0.12, 0.06
    def ax(v: float) -> int:
        frac = (v - zero_min) / (zero_max - zero_min)
        return int(round(a_left + frac * (a_right - a_left)))

    x_zero = ax(0.0)
    for tick in [-0.12, -0.08, -0.04, 0.0, 0.04]:
        x = ax(tick)
        draw.line((x, a_top, x, a_bottom), fill=GRID if tick != 0 else LINE, width=2 if tick == 0 else 1)
        label = "0" if tick == 0 else f"{tick:.2f}"
        tb = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (tb[2] - tb[0]) // 2, a_bottom + 22), label, font=tick_font, fill=MUTED)

    regime_rows = [
        ("DLR in-domain", 0.047, BLUE),
        ("DLR spatial-OOD", 0.041, BLUE),
        ("Heterogeneous post-event", 0.036, AMBER),
        ("Change-view (IoU@val-thr)", -0.081, ROSE),
        ("Change-view (IoU@0.50)", -0.110, ROSE),
    ]
    row_gap = (a_bottom - a_top) / (len(regime_rows) + 0.2)
    for i, (label, delta, color) in enumerate(regime_rows):
        y = int(a_top + row_gap * (i + 1))
        lb = draw.textbbox((0, 0), label, font=axis_font)
        if i <= 2:
            label_x = x_zero - (lb[2] - lb[0]) - 16
        else:
            label_x = x_zero + 16
        draw.text((label_x, y - (lb[3] - lb[1]) // 2), label, font=axis_font, fill=INK)
        x_end = ax(delta)
        x0, x1 = sorted([x_zero, x_end])
        fill = blend(WHITE, color, 0.35)
        draw.rounded_rectangle((x0, y - 20, x1, y + 20), radius=18, fill=fill, outline=color, width=5)
        draw.ellipse((x_end - 10, y - 10, x_end + 10, y + 10), fill=color, outline=WHITE, width=3)
        if delta >= 0:
            tx = x_end + 14
        else:
            tx = x_end - 90
        draw.text((tx, y - 16), fmt_delta(delta), font=value_font, fill=color)
    footer_text = "ΔIoU relative to the matched regime-specific reference"
    fb = draw.textbbox((0, 0), footer_text, font=label_font)
    footer_x = (panel_a[0] + panel_a[2] - (fb[2] - fb[0])) // 2
    draw.text((footer_x, a_bottom + 52), footer_text, font=label_font, fill=MUTED)

    # ---------------- Panel (b): source-family behavior matrix ----------------
    title_b = "(b) Source-family correction patterns"
    tb_b = draw.textbbox((0, 0), title_b, font=panel_font)
    draw.text(
        (panel_b[0], panel_b[1] - (tb_b[3] - tb_b[1]) - 10),
        title_b,
        font=panel_font,
        fill=INK,
    )
    b_left = panel_b[0] + 40
    b_top = panel_b[1] + 80
    b_right = panel_b[2] - 32
    b_bottom = panel_b[3] - 80

    merged_case_csv = ROOT / "experiments" / "strict_t2_postrgb_case_panels_v1" / "merged_case_metrics.csv"
    case_df = pd.read_csv(merged_case_csv)

    def source_family(sample_id: str) -> str:
        if sample_id.startswith("EID_"):
            return "DLR"
        if sample_id.startswith("CAS_Palu::"):
            return "CAS_Palu"
        return "Other"

    case_df["source_family"] = case_df["sample_id"].map(source_family)
    b_data = {
        "DLR": {
            "material > visual": float((case_df.loc[case_df["source_family"] == "DLR", "material_iou"] > case_df.loc[case_df["source_family"] == "DLR", "visual_iou"]).mean() * 100.0),
            "full > visual": float((case_df.loc[case_df["source_family"] == "DLR", "full_iou"] > case_df.loc[case_df["source_family"] == "DLR", "visual_iou"]).mean() * 100.0),
            "full > material": float((case_df.loc[case_df["source_family"] == "DLR", "full_iou"] > case_df.loc[case_df["source_family"] == "DLR", "material_iou"]).mean() * 100.0),
            "n": int((case_df["source_family"] == "DLR").sum()),
        },
        "CAS_Palu": {
            "material > visual": float((case_df.loc[case_df["source_family"] == "CAS_Palu", "material_iou"] > case_df.loc[case_df["source_family"] == "CAS_Palu", "visual_iou"]).mean() * 100.0),
            "full > visual": float((case_df.loc[case_df["source_family"] == "CAS_Palu", "full_iou"] > case_df.loc[case_df["source_family"] == "CAS_Palu", "visual_iou"]).mean() * 100.0),
            "full > material": float((case_df.loc[case_df["source_family"] == "CAS_Palu", "full_iou"] > case_df.loc[case_df["source_family"] == "CAS_Palu", "material_iou"]).mean() * 100.0),
            "n": int((case_df["source_family"] == "CAS_Palu").sum()),
        },
    }

    cols = ["material > visual", "full > visual", "full > material"]
    rows = ["DLR", "CAS_Palu"]
    col_base_colors = [BLUE, TEAL, AMBER]
    table_x = b_left + 135
    header_y = b_top + 10
    for j, col in enumerate(cols):
        header = col.replace(" > ", "\n> ")
        tb = draw.textbbox((0, 0), header, font=axis_font)
        col_center_x = table_x + j * (b_right - table_x) / len(cols) + (b_right - table_x) / (2 * len(cols))
        header = col.replace(" > ", "\n> ")
        draw.multiline_text(
            (col_center_x - (tb[2] - tb[0]) / 2, header_y),
            header,
            font=axis_font,
            fill=INK,
            align="center",
            spacing=2,
        )
    heatmap_top = header_y + (tb[3] - tb[1]) + 16
    heatmap_bottom = b_bottom - 24
    cell_h = (heatmap_bottom - heatmap_top) / len(rows)
    inner_width = b_right - table_x - 8
    cell_w = inner_width / len(cols)

    for i, row in enumerate(rows):
        y0 = heatmap_top + i * cell_h
        label = f"{row}\n(n={b_data[row]['n']})"
        draw.multiline_text((b_left, y0 + cell_h / 2 - 24), label, font=axis_font, fill=INK, spacing=4)
        for j, col in enumerate(cols):
            val = b_data[row][col]
            x0 = int(table_x + j * cell_w)
            x1 = int(x0 + cell_w)
            y1 = int(y0 + cell_h)
            color = blend(WHITE, col_base_colors[j], min(1.0, val / 70.0))
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=LINE, width=1)
            val_text = f"{val:.1f}%"
            vb = draw.textbbox((0, 0), val_text, font=value_font)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            draw.text((cx - (vb[2] - vb[0]) / 2, cy - (vb[3] - vb[1]) / 2), val_text, font=value_font, fill=INK)

    # ---------------- Panel (c): uncertainty stratification ----------------
    title_c = "(c) Residual-prior gains by ω quartile"
    tb_c = draw.textbbox((0, 0), title_c, font=panel_font)
    draw.text(
        (panel_c[0], panel_c[1] - (tb_c[3] - tb_c[1]) - 10),
        title_c,
        font=panel_font,
        fill=INK,
    )
    c_left = panel_c[0] + 40
    c_top = panel_c[1] + 86
    c_right = panel_c[2] - 34
    c_bottom = panel_c[3] - 28

    uq = pd.read_csv(
        ROOT
        / "experiments"
        / "strict_t2_postrgb_v4_support_uncertainty_summary_v1"
        / "uncertainty_quartiles.csv"
    ).sort_values("quartile")
    line_chart = (c_left + 8, c_top + 40, c_right, c_top + 165)
    bar_chart = (c_left + 8, c_top + 212, c_right, c_bottom - 36)

    quartile_colors = [TEAL, SKY, SAND, ROSE]
    q_labels = list(uq["quartile"])
    delta_vals = list(uq["delta_mean"])
    rate_vals = list(uq["positive_rate"])

    # top inset: positive-delta rate line
    r_min, r_max = 0, 100
    for tick in [0, 50, 100]:
        ty = int(line_chart[3] - (tick - r_min) / (r_max - r_min) * (line_chart[3] - line_chart[1]))
        draw.line((line_chart[0] + 10, ty, line_chart[2], ty), fill=GRID, width=1)
        draw.text((line_chart[2] + 8, ty - 8), f"{tick}", font=tick_font, fill=MUTED)
    rate_title = "Positive-delta rate (%)"
    rt = draw.textbbox((0, 0), rate_title, font=axis_font)
    draw.text((line_chart[2] - (rt[2] - rt[0]), line_chart[1] - 28), rate_title, font=axis_font, fill=INK)

    bar_area_w = bar_chart[2] - bar_chart[0] - 40
    bar_gap = bar_area_w / 4
    bar_w = 64
    xs = [int(bar_chart[0] + 35 + bar_gap * i + bar_gap / 2) for i in range(4)]
    line_points = []
    for i, rate in enumerate(rate_vals):
        cx = xs[i]
        cy = int(line_chart[3] - (rate - r_min) / (r_max - r_min) * (line_chart[3] - line_chart[1]))
        line_points.append((cx, cy))
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=quartile_colors[i], outline=WHITE, width=2)
        draw.text((cx, cy - 24 if i != 3 else cy + 10), f"{rate:.1f}%", font=small_font, fill=quartile_colors[i], anchor="ma")
    for p0, p1, col in zip(line_points[:-1], line_points[1:], quartile_colors[:-1]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=col, width=3)

    # bottom main chart: mean delta bars
    d_min, d_max = -0.02, 0.09
    zero_y = int(bar_chart[1] + (d_max / (d_max - d_min)) * (bar_chart[3] - bar_chart[1]))
    for tick in [-0.02, 0.00, 0.04, 0.08]:
        ty = int(bar_chart[3] - (tick - d_min) / (d_max - d_min) * (bar_chart[3] - bar_chart[1]))
        draw.line((bar_chart[0] + 10, ty, bar_chart[2], ty), fill=GRID if tick != 0 else LINE, width=1 if tick != 0 else 2)
        draw.text((bar_chart[0] - 4, ty - 8), f"{tick:+.2f}" if tick != 0 else "0", font=tick_font, fill=MUTED, anchor="ra")
    draw.text((bar_chart[0] - 8, bar_chart[1] - 38), "Mean ΔIoU", font=axis_font, fill=INK)

    for i, val in enumerate(delta_vals):
        cx = xs[i]
        y_val = int(bar_chart[3] - (val - d_min) / (d_max - d_min) * (bar_chart[3] - bar_chart[1]))
        x0, x1 = cx - bar_w // 2, cx + bar_w // 2
        y0, y1 = sorted([zero_y, y_val])
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=blend(WHITE, quartile_colors[i], 0.78), outline=quartile_colors[i], width=2)
        draw.text((cx, y_val - 26 if val >= 0 else y_val + 8), f"{val:+.3f}", font=small_font, fill=quartile_colors[i], anchor="ma")
        q_label = q_labels[i] if i not in (0, 3) else ("Q1\nlow ω" if i == 0 else "Q4\nhigh ω")
        draw.multiline_text((cx, bar_chart[3] + 10), q_label, font=small_font, fill=MUTED, anchor="ma", align="center", spacing=2)

    save_outputs(
        canvas,
        "figure8_mechanistic_interpretation",
        [
            "# figure8_mechanistic_interpretation",
            "",
            "- status: generated for manuscript Figure 8 (quantitative mechanism synthesis)",
            "- purpose: quantify how UGCoP-style context mismatch changes the preferred form of physics coupling",
            "",
            "Panels:",
            "- (a) signed ΔIoU gains across regime types, showing that physics-guided gain is positive in DLR and heterogeneous post-event settings but reverses under the stricter change-view control",
            "- (b) source-family behavior matrix over the strict post_rgb case pool",
            "- (c) uncertainty-quartile analysis over the shared v4-vs-visual post_rgb test pool",
            "",
            "Key values:",
            "- DLR in-domain gain: +0.047",
            "- DLR spatial-OOD gain: +0.041",
            "- Heterogeneous post-event gain: +0.036",
            "- Change-view material-only vs visual: -0.081 (IoU@val-thr), -0.110 (IoU@0.50)",
            "- DLR material > visual: 60.7%",
            "- CAS_Palu full > visual: 59.6%",
            "- CAS_Palu full > material: 61.3%",
            "- Lowest-uncertainty quartile: mean ΔIoU +0.084, positive rate 87.3%",
            "- Highest-uncertainty quartile: mean ΔIoU -0.011, positive rate 0.4%",
        ],
    )


def render_figure_s3_regime_sign_summary() -> None:
    """Render Supplementary Figure S3 from the former Figure 8(a) regime-sign summary."""

    def blend(base: tuple[int, int, int], target: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return tuple(int(base[i] + (target[i] - base[i]) * t) for i in range(3))

    def fmt_delta(val: float) -> str:
        return f"{val:+.3f}"

    canvas_w = 1500
    canvas_h = 520
    margin = 32

    canvas = Image.new("RGB", (canvas_w, canvas_h), WHITE)
    draw = ImageDraw.Draw(canvas)

    panel_font = load_font(34, bold=True)
    axis_font = load_font(26)
    tick_font = load_font(24)
    label_font = load_font(28, bold=True)
    value_font = load_font(28, bold=True)
    value_font_neg = load_font(25, bold=True)

    panel = (margin, margin + 72, canvas_w - margin, canvas_h - margin)
    draw.rounded_rectangle(panel, radius=18, fill=WHITE, outline=LINE, width=2)

    title = "Regime-dependent sign of physics-guided gains"
    tb = draw.textbbox((0, 0), title, font=panel_font)
    title_x = (canvas_w - (tb[2] - tb[0])) // 2   
    draw.text((title_x, panel[1] - (tb[3] - tb[1]) - 10), title, font=panel_font, fill=INK)

    center_x = canvas_w // 2
    a_left = panel[0] + 48
    a_top = panel[1] + 24
    a_bottom = panel[3] - 88
    zero_min, zero_max = -0.12, 0.06
    span = int((center_x - a_left) * 3 / 2)
    a_right = a_left + span

    def ax(v: float) -> int:
        frac = (v - zero_min) / (zero_max - zero_min)
        return int(round(a_left + frac * (a_right - a_left)))

    x_zero = ax(0.0)
    for tick in [-0.12, -0.08, -0.04, 0.0, 0.04]:
        x = ax(tick)
        draw.line((x, a_top, x, a_bottom), fill=GRID if tick != 0 else LINE, width=2 if tick == 0 else 1)
        label = "0" if tick == 0 else f"{tick:.2f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) // 2, a_bottom + 18), label, font=tick_font, fill=MUTED)

    regime_rows = [
        ("DLR in-domain", 0.047, BLUE),
        ("DLR spatial-OOD", 0.041, BLUE),
        ("Heterogeneous post-event", 0.036, AMBER),
        ("Change-view (IoU@val-thr)", -0.081, ROSE),
        ("Change-view (IoU@0.50)", -0.110, ROSE),
    ]
    row_gap = (a_bottom - a_top) / (len(regime_rows) + 0.2)
    for i, (label, delta, color) in enumerate(regime_rows):
        y = int(a_top + row_gap * (i + 1))
        lb = draw.textbbox((0, 0), label, font=axis_font)
        if i <= 2:
            label_x = x_zero - (lb[2] - lb[0]) - 16
        else:
            label_x = x_zero + 16
        draw.text((label_x, y - (lb[3] - lb[1]) // 2), label, font=axis_font, fill=INK)
        x_end = ax(delta)
        x0, x1 = sorted([x_zero, x_end])
        fill = blend(WHITE, color, 0.35)
        draw.rounded_rectangle((x0, y - 20, x1, y + 20), radius=18, fill=fill, outline=color, width=5)
        draw.ellipse((x_end - 16, y - 16, x_end + 16, y + 16), fill=color, outline=WHITE, width=3)
        tx = x_end + 18 if delta >= 0 else x_end - 106
        draw.text((tx, y - 16), fmt_delta(delta), font=value_font if delta >= 0 else value_font_neg, fill=color)

    footer_text = "ΔIoU relative to the matched regime-specific reference"
    fb = draw.textbbox((0, 0), footer_text, font=label_font)
    footer_x = (panel[0] + panel[2] - (fb[2] - fb[0])) // 2
    draw.text((footer_x, a_bottom + 48), footer_text, font=label_font, fill=MUTED)

    save_outputs(
        canvas,
        "figure_s3_regime_sign_summary",
        [
            "# figure_s3_regime_sign_summary",
            "",
            "- status: generated as Supplementary Figure S3 from the former Figure 8(a) regime-sign panel",
            "- purpose: retain the cross-regime sign-reversal summary as supplementary support after Figure 8 was reoriented toward sample-level mechanism landscapes",
            "",
            "Summary:",
            "- positive gains in DLR in-domain (+0.047) and spatial-OOD (+0.041)",
            "- positive gain in heterogeneous post-event (+0.036)",
            "- sign reversal under change-view control (-0.081 at IoU@val-thr; -0.110 at IoU@0.50)",
        ],
    )


def main() -> None:
    render_figure4()
    render_figure5()
    render_figure6()
    render_figure7()
    render_figure8()
    render_figure_s3_regime_sign_summary()


if __name__ == "__main__":
    main()
