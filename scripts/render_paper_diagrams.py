#!/usr/bin/env python3
"""Render the protocol and architecture diagrams used in the paper."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WHITE = (252, 252, 250)
INK = (24, 29, 34)
MUTED = (84, 92, 102)
LINE = (120, 128, 136)
BLUE = (225, 238, 252)
BLUE_DARK = (58, 102, 168)
TEAL = (222, 244, 240)
TEAL_DARK = (39, 126, 114)
SAND = (247, 239, 219)
SAND_DARK = (150, 112, 34)
ROSE = (250, 228, 228)
ROSE_DARK = (161, 70, 70)
SLATE = (233, 238, 244)
SLATE_DARK = (83, 96, 122)
GREEN = (228, 243, 228)
GREEN_DARK = (67, 122, 72)
YELLOW = (251, 246, 219)
YELLOW_DARK = (149, 129, 37)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for raw in candidates:
        path = Path(raw)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), trial, font=font)
        if box[2] - box[0] <= max_w:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def draw_box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    body: list[str],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    title_fill: tuple[int, int, int] = INK,
    body_fill: tuple[int, int, int] = MUTED,
    radius: int = 24,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    padding: int = 18,
    body_indent: int = 0,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=3)
    draw.text((x0 + padding, y0 + padding - 2), title, fill=title_fill, font=title_font)
    cursor_y = y0 + padding + 38
    max_w = max(40, x1 - x0 - 2 * padding - body_indent)
    for raw_line in body:
        if raw_line == "":
            cursor_y += 8
            continue
        lines = wrap_text(draw, raw_line, body_font, max_w)
        for line in lines:
            draw.text((x0 + padding + body_indent, cursor_y), line, fill=body_fill, font=body_font)
            cursor_y += 28


def draw_chip(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    font: ImageFont.ImageFont,
    text_fill: tuple[int, int, int] = INK,
) -> None:
    draw.rounded_rectangle(rect, radius=18, fill=fill, outline=outline, width=2)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = rect[0] + ((rect[2] - rect[0]) - text_w) / 2
    y = rect[1] + ((rect[3] - rect[1]) - text_h) / 2 - 2
    draw.text((x, y), text, fill=text_fill, font=font)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int] = LINE,
    width: int = 6,
    head: int = 16,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, math.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    tip = end
    left = (end[0] - head * ux + head * 0.55 * px, end[1] - head * uy + head * 0.55 * py)
    right = (end[0] - head * ux - head * 0.55 * px, end[1] - head * uy - head * 0.55 * py)
    draw.polygon([tip, left, right], fill=fill)


def render_protocol_figure(out_png: Path, out_report: Path) -> None:
    canvas = Image.new("RGB", (2200, 1320), color=WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(40, bold=True)
    section_font = load_font(28, bold=True)
    box_title = load_font(24, bold=True)
    body_font = load_font(19)
    chip_font = load_font(18, bold=True)

    draw.text((64, 34), "Figure 1. PILD data system, protocol split, and manuscript evidence lines", fill=INK, font=title_font)
    draw.text((64, 84), "The paper does not use one monolithic benchmark. It separates the DLR-centered frontier claim from the harder multi-source stress evidence.", fill=MUTED, font=body_font)

    top_rect = (64, 138, 2136, 380)
    draw_box(
        draw,
        top_rect,
        "PILD: project-scale multi-source landslide data system",
        [
            "Event-centered samples, metadata manifests, cache-backed tensors, and protocol-specific partitions are organized inside one study system rather than as disconnected benchmark folders.",
            "The same repository supports stable manuscript reporting, transfer stress tests, reliability evaluation, and figure-generation assets.",
        ],
        fill=SLATE,
        outline=SLATE_DARK,
        title_font=section_font,
        body_font=body_font,
    )

    chip_y = 280
    chips = [
        ("DLR Landslide Reference 2025", (90, chip_y, 550, chip_y + 56), BLUE, BLUE_DARK),
        ("GLaD4CD v1", (580, chip_y, 880, chip_y + 56), TEAL, TEAL_DARK),
        ("GDCLD", (910, chip_y, 1160, chip_y + 56), SAND, SAND_DARK),
        ("CAS Landslide", (1190, chip_y, 1540, chip_y + 56), ROSE, ROSE_DARK),
        ("event manifests + cache-backed tensors + paper assets", (1570, chip_y, 2100, chip_y + 56), GREEN, GREEN_DARK),
    ]
    for label, rect, fill, outline in chips:
        draw_chip(draw, rect, label, fill=fill, outline=outline, font=chip_font)

    draw.text((64, 420), "Protocol family A", fill=BLUE_DARK, font=section_font)
    draw.text((1118, 420), "Protocol family B", fill=TEAL_DARK, font=section_font)

    strict_t3 = (64, 470, 1040, 980)
    strict_t2 = (1118, 470, 2136, 980)
    draw_box(
        draw,
        strict_t3,
        "strict_t3: DLR-centered high-fidelity frontier protocol",
        [
            "Purpose: evaluate the final physics-coupled foundation-model architecture under the cleanest physical support currently available in the project.",
            "Data character: DLR-centered subset with richer and more internally consistent terrain / trigger support.",
            "Interpretation: this is the headline performance line of the paper.",
        ],
        fill=BLUE,
        outline=BLUE_DARK,
        title_font=box_title,
        body_font=body_font,
    )
    draw_box(
        draw,
        strict_t2,
        "strict_t2: broader multi-source mechanism-oriented stress test",
        [
            "Purpose: test whether physics transfer survives stronger heterogeneity in source, annotation style, appearance regime, and physical-support quality.",
            "Current release: 196 events from four source families, with the largest share from GLaD4CD v1 and DLR Landslide Reference 2025.",
            "Interpretation: this benchmark probes failure modes and protocol dependence rather than replacing the DLR frontier line.",
        ],
        fill=TEAL,
        outline=TEAL_DARK,
        title_font=box_title,
        body_font=body_font,
    )

    t3_boxes = [
        ("DLR train / val / testind", (112, 710, 486, 820), BLUE, BLUE_DARK),
        ("DLR testspt spatial OOD", (514, 710, 888, 820), BLUE, BLUE_DARK),
        ("post-cal reliability", (112, 846, 486, 956), SLATE, SLATE_DARK),
        ("GLaD4CD zero-shot", (514, 846, 888, 956), YELLOW, YELLOW_DARK),
    ]
    for label, rect, fill, outline in t3_boxes:
        draw_chip(draw, rect, label, fill=fill, outline=outline, font=chip_font)
    draw.text((112, 670), "Frozen evidence lines", fill=INK, font=box_title)

    t2_boxes = [
        ("post_rgb near-full benchmark", (1172, 710, 1596, 820), TEAL, TEAL_DARK),
        ("change_rgb stricter control", (1628, 710, 2052, 820), ROSE, ROSE_DARK),
        ("teacher-guided v4 can recover stable post_rgb gain", (1172, 846, 1596, 956), GREEN, GREEN_DARK),
        ("visual still leads the hardest change regime", (1628, 846, 2052, 956), SAND, SAND_DARK),
    ]
    for label, rect, fill, outline in t2_boxes:
        draw_chip(draw, rect, label, fill=fill, outline=outline, font=chip_font)
    draw.text((1172, 670), "Task views and manuscript role", fill=INK, font=box_title)

    draw_arrow(draw, (1100, 332), (552, 470))
    draw_arrow(draw, (1100, 332), (1616, 470))
    draw_arrow(draw, (552, 610), (552, 680))
    draw_arrow(draw, (1616, 610), (1616, 680))

    footer = (64, 1040, 2136, 1240)
    draw_box(
        draw,
        footer,
        "Manuscript framing",
        [
            "Headline claim: GeoPhysAdapter-v3 improves the DLR frontier evidence chain across segmentation, spatial OOD, reliability, and external zero-shot transfer.",
            "Stress-test claim: strict_t2 reveals protocol-dependent physics gain. Physics helps most when used as calibrated residual guidance around a strong visual trunk, not as an unrestricted direct-fusion path under the hardest cross-source change regime.",
        ],
        fill=(246, 247, 244),
        outline=(140, 145, 135),
        title_font=section_font,
        body_font=body_font,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    out_report.write_text(
        "\n".join(
            [
                "# Figure 1 Report",
                "",
                "Title: PILD data system, protocol split, and manuscript evidence lines",
                "",
                f"- output: `{out_png}`",
                "- top panel: PILD data-system overview with source families and repository role",
                "- left protocol card: `strict_t3` as the DLR-centered frontier line",
                "- right protocol card: `strict_t2` as the multi-source stress-test line",
                "- footer: manuscript framing that separates frontier claims from protocol-dependent stress evidence",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def render_architecture_figure(out_png: Path, out_report: Path) -> None:
    canvas = Image.new("RGB", (2320, 1400), color=WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(40, bold=True)
    section_font = load_font(28, bold=True)
    box_title = load_font(26, bold=True)
    body_font = load_font(20)
    chip_font = load_font(18, bold=True)

    draw.text((64, 34), "Figure 2. GeoPhysAdapter overview and the heterogeneity-aware strict_t2 residual-prior transfer variant", fill=INK, font=title_font)
    draw.text((64, 84), "Main frontier model for the high-fidelity DLR protocol and lighter residual-prior variant for the heterogeneous strict_t2 benchmark.", fill=MUTED, font=body_font)

    input_boxes = [
        ("Pre-event optical\n(B02/B03/B04/B08)", (64, 180, 360, 304), BLUE, BLUE_DARK),
        ("Post-event optical\n(B02/B03/B04/B08)", (64, 342, 360, 466), TEAL, TEAL_DARK),
        ("Terrain tensor", (64, 534, 360, 626), SAND, SAND_DARK),
        ("Material descriptor", (64, 656, 360, 748), GREEN, GREEN_DARK),
        ("Trigger descriptor", (64, 778, 360, 870), YELLOW, YELLOW_DARK),
        ("Source / quality metadata\n(v4 transfer variant)", (64, 900, 360, 1022), ROSE, ROSE_DARK),
    ]
    for title, rect, fill, outline in input_boxes:
        draw_box(
            draw,
            rect,
            title.split("\n")[0],
            title.split("\n")[1:],
            fill=fill,
            outline=outline,
            title_font=box_title,
            body_font=body_font,
        )

    encoder_pre = (470, 188, 780, 330)
    encoder_post = (470, 352, 780, 494)
    draw_box(
        draw,
        encoder_pre,
        "Twin DINOv2 encoder",
        ["shared weights", "extracts `F_pre` from pre-event imagery"],
        fill=SLATE,
        outline=SLATE_DARK,
        title_font=box_title,
        body_font=body_font,
    )
    draw_box(
        draw,
        encoder_post,
        "Twin DINOv2 encoder",
        ["shared weights", "extracts `F_post` from post-event imagery"],
        fill=SLATE,
        outline=SLATE_DARK,
        title_font=box_title,
        body_font=body_font,
    )
    change_chip = (540, 540, 716, 606)
    draw_chip(draw, change_chip, "change feature: F_delta", fill=BLUE, outline=BLUE_DARK, font=chip_font)

    adapter = (870, 250, 1260, 632)
    draw_box(
        draw,
        adapter,
        "Geophysical adapter",
        [
            "terrain / material / trigger embeddings",
            "FiLM modulation + cross-attention",
            "context-aware reinterpretation of visual change",
        ],
        fill=BLUE,
        outline=BLUE_DARK,
        title_font=box_title,
        body_font=body_font,
    )

    routing = (1390, 184, 1740, 584)
    draw_box(
        draw,
        routing,
        "Dual-expert routing",
        [
            "visual expert: change and boundaries",
            "geophysically conditioned expert: terrain / trigger-consistent structure",
            "adaptive gate alpha",
        ],
        fill=TEAL,
        outline=TEAL_DARK,
        title_font=box_title,
        body_font=body_font,
    )
    draw_chip(draw, (1430, 454, 1698, 516), "P = (1 - alpha) P_vis + alpha P_phys", fill=WHITE, outline=TEAL_DARK, font=chip_font)

    output_box = (1848, 280, 2256, 514)
    draw_box(
        draw,
        output_box,
        "Segmentation output",
        [
            "landslide logit map",
            "IoU / F1 evaluation",
            "temperature-scaled reliability analysis",
        ],
        fill=GREEN,
        outline=GREEN_DARK,
        title_font=box_title,
        body_font=body_font,
    )

    state_heads = (470, 718, 1260, 1088)
    draw_box(
        draw,
        state_heads,
        "PINN-inspired surrogate-state heads",
        [
            "u_hat wetness surrogate + FoS_hat stability surrogate",
            "structured losses align segmentation and hydro-geomorphic cues",
            "transfer-stable behavior without full mechanistic inversion",
        ],
        fill=SAND,
        outline=SAND_DARK,
        title_font=box_title,
        body_font=body_font,
    )
    chip_specs = [
        ("L_seg", (520, 932, 644, 990), BLUE, BLUE_DARK),
        ("L_change", (666, 932, 824, 990), TEAL, TEAL_DARK),
        ("L_topo", (846, 932, 974, 990), SAND, SAND_DARK),
        ("L_hydro", (996, 932, 1138, 990), YELLOW, YELLOW_DARK),
        ("L_stability", (520, 1010, 690, 1068), GREEN, GREEN_DARK),
        ("L_obs", (712, 1010, 836, 1068), ROSE, ROSE_DARK),
        ("L_distill", (858, 1010, 1010, 1068), SLATE, SLATE_DARK),
    ]
    for label, rect, fill, outline in chip_specs:
        draw_chip(draw, rect, label, fill=fill, outline=outline, font=chip_font)

    v4_box = (1390, 720, 2256, 1210)
    draw_box(
        draw,
        v4_box,
        "Heterogeneity-aware strict_t2 transfer variant",
        [
            "matched DeepLabV3-ResNet50 visual backbone",
            "geophysical and metadata regionalizer",
            "prior logit, support s, uncertainty omega",
            "residual-prior guidance around P_vis",
        ],
        fill=ROSE,
        outline=ROSE_DARK,
        title_font=box_title,
        body_font=body_font,
    )
    draw_chip(draw, (1476, 1000, 2160, 1062), "P_v4 = P_vis + eta_prior * (1 - omega) * s * P_prior + eta_int * P_int", fill=WHITE, outline=ROSE_DARK, font=chip_font)

    draw_arrow(draw, (360, 242), (470, 242))
    draw_arrow(draw, (360, 404), (470, 404))
    draw_arrow(draw, (780, 258), (870, 340))
    draw_arrow(draw, (780, 422), (870, 420))
    draw_arrow(draw, (625, 494), (625, 540))
    draw_arrow(draw, (730, 574), (870, 574))
    draw_arrow(draw, (360, 580), (870, 574))
    draw_arrow(draw, (360, 702), (870, 574))
    draw_arrow(draw, (360, 824), (870, 574))
    draw_arrow(draw, (1260, 444), (1390, 444))
    draw_arrow(draw, (1740, 404), (1848, 398))
    draw_arrow(draw, (1060, 640), (1060, 718))
    draw_arrow(draw, (1260, 574), (1390, 836))
    draw_arrow(draw, (360, 960), (1390, 960))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    out_report.write_text(
        "\n".join(
            [
                "# Figure 2 Report",
                "",
                "Title: GeoPhysAdapter overview and the heterogeneity-aware strict_t2 residual-prior transfer variant",
                "",
                f"- output: `{out_png}`",
                "- upper path: twin DINOv2 encoder, geophysical adapter, dual-expert routing, and segmentation output",
                "- lower left: PINN-inspired surrogate-state heads and structured losses",
                "- lower right: heterogeneity-aware `v4` transfer variant with residualized structured-prior fusion",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render submission-facing protocol and architecture diagrams")
    p.add_argument("--figure1-png", required=True)
    p.add_argument("--figure1-report", required=True)
    p.add_argument("--figure2-png", required=True)
    p.add_argument("--figure2-report", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    render_protocol_figure(Path(args.figure1_png), Path(args.figure1_report))
    render_architecture_figure(Path(args.figure2_png), Path(args.figure2_report))
    print(args.figure1_png)
    print(args.figure1_report)
    print(args.figure2_png)
    print(args.figure2_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
