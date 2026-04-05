#!/usr/bin/env python3
"""Render a global event-coverage map for the PILD strict_t2 release."""

from __future__ import annotations

import csv
import json
import math
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
FIG_DIR = PACKAGE_ROOT / "figures"

WORLD_GEOJSON = FIG_DIR / "ne_110m_admin_0_countries.geojson"
WORLD_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"

WHITE = (248, 248, 246)
OCEAN = (228, 239, 246)
LAND = (241, 238, 225)
LAND_EDGE = (208, 205, 191)
GRID = (204, 216, 224)
INK = (29, 34, 40)
MUTED = (93, 100, 110)
DLR = (61, 103, 172)
GLAD = (170, 77, 77)
GDCLD = (190, 142, 58)
CAS = (49, 132, 118)
NOTE = (234, 242, 234)
NOTE_EDGE = (105, 141, 107)

COLOR_BY_SOURCE = {
    "DLR_Landslide_Ref_2025": DLR,
    "GLaD4CD_v1": GLAD,
    "GDCLD": GDCLD,
    "CAS_Landslide": CAS,
}


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
    if not text:
        return [""]
    words = text.split()
    lines = [words[0]]
    for word in words[1:]:
        trial = f"{lines[-1]} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_w:
            lines[-1] = trial
        else:
            lines.append(word)
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, max_w: int, *, fill=MUTED, line_gap: int = 6) -> int:
    x, y = xy
    start_y = y
    for line in wrap_text(draw, text, font, max_w):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y - start_y


def project(lon: float, lat: float, rect: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0
    x = x0 + (lon + 180.0) / 360.0 * width
    y = y0 + (90.0 - lat) / 180.0 * height
    return x, y


def ensure_world_geojson() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if WORLD_GEOJSON.exists():
        return
    urllib.request.urlretrieve(WORLD_URL, WORLD_GEOJSON)


def iter_rings(geometry: dict) -> list[list[tuple[float, float]]]:
    rings: list[list[tuple[float, float]]] = []
    geom_type = geometry["type"]
    coords = geometry["coordinates"]
    if geom_type == "Polygon":
        polygons = [coords]
    elif geom_type == "MultiPolygon":
        polygons = coords
    else:
        return rings
    for polygon in polygons:
        if not polygon:
            continue
        exterior = polygon[0]
        rings.append([(float(lon), float(lat)) for lon, lat in exterior])
    return rings


def load_t2_events() -> list[dict[str, object]]:
    event_index = DATA_DIR / "event_index_v1_strict_t2.csv"
    event_master = DATA_DIR / "event_master.csv"

    master_by_uid: dict[str, dict[str, str]] = {}
    with event_master.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            master_by_uid[row["event_uid"]] = row

    events: list[dict[str, object]] = []
    with event_index.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            master = master_by_uid[row["event_uid"]]
            if master["lat"] and master["lon"]:
                lat = float(master["lat"])
                lon = float(master["lon"])
                coord_source = "event_point"
            else:
                min_lon = float(master["min_lon"])
                max_lon = float(master["max_lon"])
                min_lat = float(master["min_lat"])
                max_lat = float(master["max_lat"])
                lon = 0.5 * (min_lon + max_lon)
                lat = 0.5 * (min_lat + max_lat)
                coord_source = "bbox_center"
            events.append(
                {
                    "event_uid": row["event_uid"],
                    "dataset_id": row["dataset_id"],
                    "trigger_type": row["trigger_type"],
                    "lat": lat,
                    "lon": lon,
                    "coord_source": coord_source,
                    "is_t3": row["strict_t3_static_era5_dlr"] == "1",
                    "notes": master.get("notes", ""),
                }
            )
    return events


def render() -> None:
    ensure_world_geojson()
    events = load_t2_events()
    counts: dict[str, int] = {}
    for event in events:
        counts[event["dataset_id"]] = counts.get(event["dataset_id"], 0) + 1

    canvas = Image.new("RGB", (2200, 1320), OCEAN)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(42, bold=True)
    sub_font = load_font(23)
    label_font = load_font(22, bold=True)
    body_font = load_font(20)
    small_font = load_font(18)

    draw.text((60, 36), "PILD strict_t2 global event coverage", font=title_font, fill=INK)
    draw_wrapped(
        draw,
        (60, 90),
        "Event-level coverage of the public benchmark release. Points use event lat/lon when available, otherwise the center of the recorded event bounding box.",
        sub_font,
        1500,
    )

    map_rect = (60, 160, 1650, 1160)
    draw.rounded_rectangle(map_rect, radius=28, fill=OCEAN, outline=GRID, width=3)

    # Graticules.
    for lon in range(-150, 181, 30):
        x0, y0 = project(float(lon), 85.0, map_rect)
        x1, y1 = project(float(lon), -85.0, map_rect)
        draw.line((x0, y0, x1, y1), fill=GRID, width=1)
        label = f"{abs(lon)}{'W' if lon < 0 else ('E' if lon > 0 else '')}"
        if lon == 0:
            label = "0"
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x0 - (bbox[2] - bbox[0]) / 2, map_rect[3] + 6), label, font=small_font, fill=MUTED)
    for lat in range(-60, 91, 30):
        x0, y0 = project(-180.0, float(lat), map_rect)
        x1, y1 = project(180.0, float(lat), map_rect)
        draw.line((x0, y0, x1, y1), fill=GRID, width=1)
        if lat not in (-90, 90):
            label = f"{abs(lat)}{'S' if lat < 0 else ('N' if lat > 0 else '')}"
            if lat == 0:
                label = "0"
            bbox = draw.textbbox((0, 0), label, font=small_font)
            draw.text((map_rect[0] - 12 - (bbox[2] - bbox[0]), y0 - 8), label, font=small_font, fill=MUTED)

    with WORLD_GEOJSON.open("r", encoding="utf-8") as f:
        world = json.load(f)
    for feature in world["features"]:
        for ring in iter_rings(feature["geometry"]):
            pts = [project(lon, lat, map_rect) for lon, lat in ring]
            if len(pts) >= 3:
                draw.polygon(pts, fill=LAND, outline=LAND_EDGE)

    # Event points.
    for event in events:
        color = COLOR_BY_SOURCE[event["dataset_id"]]
        x, y = project(float(event["lon"]), float(event["lat"]), map_rect)
        radius = 7
        if event["dataset_id"] == "DLR_Landslide_Ref_2025":
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=INK, width=2)
            radius = 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=2)

    side_rect = (1710, 160, 2140, 1160)
    draw.rounded_rectangle(side_rect, radius=28, fill=WHITE, outline=GRID, width=3)
    draw.text((1738, 190), "Coverage summary", font=label_font, fill=INK)
    cursor_y = 236
    cursor_y += draw_wrapped(
        draw,
        (1738, cursor_y),
        "The strict_t2 benchmark spans 196 events from four source families and preserves the 25-event DLR strict_t3 frontier subset inside the broader release.",
        body_font,
        364,
    )
    cursor_y += 18

    legend = [
        ("GLaD4CD v1", counts.get("GLaD4CD_v1", 0), GLAD),
        ("DLR Landslide Ref. 2025", counts.get("DLR_Landslide_Ref_2025", 0), DLR),
        ("GDCLD", counts.get("GDCLD", 0), GDCLD),
        ("CAS Landslide", counts.get("CAS_Landslide", 0), CAS),
    ]
    for name, count, color in legend:
        draw.rounded_rectangle((1740, cursor_y + 6, 1766, cursor_y + 32), radius=6, fill=color, outline=color)
        if name.startswith("DLR"):
            draw.rounded_rectangle((1737, cursor_y + 3, 1769, cursor_y + 35), radius=8, outline=INK, width=2)
        draw.text((1782, cursor_y), f"{name} ({count})", font=body_font, fill=INK)
        cursor_y += 46

    cursor_y += 10
    draw.rounded_rectangle((1730, cursor_y, 2120, cursor_y + 188), radius=22, fill=NOTE, outline=NOTE_EDGE, width=3)
    draw.text((1750, cursor_y + 16), "Why this figure matters", font=label_font, fill=INK)
    note_lines = [
        "It makes the paper's cross-domain claim visible before any model result is shown.",
        "It also clarifies that PILD is not a single-country benchmark centered on one source.",
        "DLR points are ringed because all 25 of them are also the high-fidelity strict_t3 frontier subset.",
    ]
    note_y = cursor_y + 58
    for line in note_lines:
        draw.text((1750, note_y), "-", font=body_font, fill=NOTE_EDGE)
        note_y += draw_wrapped(draw, (1766, note_y - 2), line, small_font, 326)
        note_y += 8

    footer = "Source files: event_master.csv + event_index_v1_strict_t2.csv | Base map: Natural Earth 110m admin 0 countries"
    draw.text((60, 1238), footer, font=small_font, fill=MUTED)

    out_png = FIG_DIR / "pild_global_event_coverage_t2.png"
    out_report = FIG_DIR / "pild_global_event_coverage_t2_report.md"
    canvas.save(out_png)
    out_report.write_text(
        "\n".join(
            [
                "# pild_global_event_coverage_t2",
                "",
                "- output: `figures/pild_global_event_coverage_t2.png`",
                "- event source: `data/event_master.csv`",
                "- protocol source: `data/event_index_v1_strict_t2.csv`",
                "- basemap source: Natural Earth 110m admin 0 countries (`ne_110m_admin_0_countries.geojson`)",
                "",
                "Counts:",
                "- strict_t2 events: 196",
                "- strict_t3 events: 25",
                "- by source: GLaD4CD_v1=157, DLR_Landslide_Ref_2025=25, GDCLD=8, CAS_Landslide=6",
                "",
                "Coordinate rule:",
                "- use `lat/lon` when present in `event_master.csv`",
                "- otherwise use the center of `min_lon/min_lat/max_lon/max_lat`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    render()
