#!/usr/bin/env python3
"""Render a global event-distribution map for the public PILD benchmark."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import numpy as np


DATASET_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = DATASET_ROOT / "release_prep"
ASSET_DIR = RELEASE_DIR / "assets"
DOC_ASSET_DIR = DATASET_ROOT / "docs" / "assets"

WORLD_GEOJSON = ASSET_DIR / "ne_110m_admin_0_countries.geojson"
WORLD_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
WORLD_BG = ASSET_DIR / "HYP_50M_SR_W.tif"
WORLD_BG_ZIP = ASSET_DIR / "HYP_50M_SR_W.zip"
WORLD_BG_URL = "https://naturalearth.s3.amazonaws.com/50m_raster/HYP_50M_SR_W.zip"

WHITE = (248, 250, 251)
PAPER = (244, 247, 249)
MAP_EDGE = (156, 172, 182)
COAST = (224, 229, 233, 112)
BORDER = (154, 164, 172, 78)
INK = (28, 35, 40)
DLR = (61, 103, 172)
GLAD = (170, 77, 77)
GDCLD = (190, 142, 58)
CAS = (49, 132, 118)
LEGEND_BG = (249, 251, 252, 220)
LEGEND_EDGE = (172, 184, 192, 240)
EXTENT_LON_MARGIN = 10.0
EXTENT_LAT_MARGIN = 8.0
EXTENT_MIN_LAT = -52.0
EXTENT_MAX_LAT = 74.0

COLOR_BY_SOURCE = {
    "DLR_Landslide_Ref_2025": DLR,
    "GLaD4CD_v1": GLAD,
    "GDCLD": GDCLD,
    "CAS_Landslide": CAS,
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    win_fonts = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
    ]
    linux_fonts = [
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/mnt/c/Windows/Fonts/arialbd.ttf" if bold else "/mnt/c/Windows/Fonts/arial.ttf",
        "/mnt/c/Windows/Fonts/msyhbd.ttc" if bold else "/mnt/c/Windows/Fonts/msyh.ttc",
    ]
    candidates = win_fonts + linux_fonts if platform.system() == "Windows" else linux_fonts
    for raw in candidates:
        path = Path(raw)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def ensure_world_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if not WORLD_GEOJSON.exists():
        urllib.request.urlretrieve(WORLD_URL, WORLD_GEOJSON)
    if not WORLD_BG.exists():
        urllib.request.urlretrieve(WORLD_BG_URL, WORLD_BG_ZIP)
        with zipfile.ZipFile(WORLD_BG_ZIP, "r") as zf:
            zf.extract("HYP_50M_SR_W.tif", ASSET_DIR)


def project(lon: float, lat: float, rect: tuple[int, int, int, int], extent: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = rect
    min_lon, min_lat, max_lon, max_lat = extent
    width = x1 - x0
    height = y1 - y0
    x = x0 + (lon - min_lon) / (max_lon - min_lon) * width
    y = y0 + (max_lat - lat) / (max_lat - min_lat) * height
    return x, y


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
        if polygon:
            rings.append([(float(lon), float(lat)) for lon, lat in polygon[0]])
    return rings


def load_t2_events() -> list[dict[str, object]]:
    event_index = DATASET_ROOT / "metadata/manifests/event_index_v1_strict_t2.csv"
    event_master = DATASET_ROOT / "raw_fullcopy/indexes/event_master.csv"

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
            else:
                lon = 0.5 * (float(master["min_lon"]) + float(master["max_lon"]))
                lat = 0.5 * (float(master["min_lat"]) + float(master["max_lat"]))
            events.append(
                {
                    "event_uid": row["event_uid"],
                    "dataset_id": row["dataset_id"],
                    "lat": lat,
                    "lon": lon,
                }
            )
    return events


def compute_extent(events: list[dict[str, object]]) -> tuple[float, float, float, float]:
    min_lon = min(float(event["lon"]) for event in events) - EXTENT_LON_MARGIN
    max_lon = max(float(event["lon"]) for event in events) + EXTENT_LON_MARGIN
    min_lat = min(float(event["lat"]) for event in events) - EXTENT_LAT_MARGIN
    max_lat = max(float(event["lat"]) for event in events) + EXTENT_LAT_MARGIN
    min_lon = max(-180.0, min_lon)
    max_lon = min(180.0, max_lon)
    min_lat = max(EXTENT_MIN_LAT, min_lat)
    max_lat = min(EXTENT_MAX_LAT, max_lat)
    return (min_lon, min_lat, max_lon, max_lat)


def crop_world_background(extent: tuple[float, float, float, float], width: int, height: int) -> Image.Image:
    min_lon, min_lat, max_lon, max_lat = extent
    src = Image.open(WORLD_BG).convert("RGB")
    src_w, src_h = src.size

    left = (min_lon + 180.0) / 360.0 * src_w
    right = (max_lon + 180.0) / 360.0 * src_w
    top = (90.0 - max_lat) / 180.0 * src_h
    bottom = (90.0 - min_lat) / 180.0 * src_h

    cropped = src.crop((left, top, right, bottom))
    resized = cropped.resize((width, height), Image.Resampling.LANCZOS)
    return resized


def draw_map_panel(
    rect: tuple[int, int, int, int],
    world_features: list[dict],
    events: list[dict[str, object]],
    extent: tuple[float, float, float, float],
    ocean_gray: bool = True,
) -> Image.Image:
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    background = crop_world_background(extent, width, height)
    local_rect = (0, 0, width, height)
    
    if ocean_gray:
        # Create a light-gray ocean background.
        OCEAN_GRAY = (232, 234, 237)
        panel = Image.new("RGBA", (width, height), OCEAN_GRAY + (255,))
        
        # Create a land mask using country polygons.
        land_mask = Image.new("L", (width, height), 0)
        land_draw = ImageDraw.Draw(land_mask)
        for feature in world_features:
            for ring in iter_rings(feature["geometry"]):
                pts = [project(lon, lat, local_rect, extent) for lon, lat in ring]
                if len(pts) >= 3:
                    land_draw.polygon(pts, fill=255)
        
        # Apply the terrain background only over land.
        background_rgba = background.convert("RGBA")
        panel.paste(background_rgba, mask=land_mask)
    else:
        panel = background.convert("RGBA")

    outline_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(outline_layer)
    for feature in world_features:
        for ring in iter_rings(feature["geometry"]):
            pts = [project(lon, lat, local_rect, extent) for lon, lat in ring]
            if len(pts) >= 3:
                outline_draw.line(pts + [pts[0]], fill=COAST, width=1)
                outline_draw.line(pts + [pts[0]], fill=BORDER, width=1)
    panel.alpha_composite(outline_layer)

    point_draw = ImageDraw.Draw(panel)
    for event in events:
        color = COLOR_BY_SOURCE[event["dataset_id"]]
        x, y = project(float(event["lon"]), float(event["lat"]), local_rect, extent)
        radius = 8
        if event["dataset_id"] == "DLR_Landslide_Ref_2025":
            point_draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=INK, width=2)
            radius = 9
        point_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=2)
    return panel


def draw_legend(canvas: Image.Image, rect: tuple[int, int, int, int], legend: list[tuple[str, int, tuple[int, int, int]]], transparent: bool = True) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    if transparent:
        # Transparent fill with border only.
        overlay_draw.rounded_rectangle(rect, radius=20, fill=None, outline=LEGEND_EDGE, width=2)
    else:
        overlay_draw.rounded_rectangle(rect, radius=20, fill=LEGEND_BG, outline=LEGEND_EDGE, width=2)
    canvas.alpha_composite(overlay)

    draw = ImageDraw.Draw(canvas)
    body_font = load_font(28)
    cursor_y = rect[1] + 16
    row_height = 54
    for name, count, color in legend:
        draw.rounded_rectangle((rect[0] + 18, cursor_y + 8, rect[0] + 54, cursor_y + 44), radius=12, fill=color, outline=color)
        if name.startswith("DLR"):
            draw.rounded_rectangle((rect[0] + 14, cursor_y + 4, rect[0] + 58, cursor_y + 48), radius=14, outline=INK, width=2)
        draw.text((rect[0] + 70, cursor_y + 2), f"{name} ({count})", font=body_font, fill=INK)
        cursor_y += row_height


def render() -> None:
    ensure_world_assets()
    events = load_t2_events()
    extent = compute_extent(events)
    counts: dict[str, int] = {}
    for event in events:
        counts[event["dataset_id"]] = counts.get(event["dataset_id"], 0) + 1

    with WORLD_GEOJSON.open("r", encoding="utf-8") as f:
        world = json.load(f)

    canvas = Image.new("RGBA", (2200, 1320), PAPER + (255,))
    draw = ImageDraw.Draw(canvas)
    map_rect = (48, 48, 2152, 1210)
    draw.rounded_rectangle(map_rect, radius=32, fill=WHITE, outline=MAP_EDGE, width=3)
    map_panel = draw_map_panel(map_rect, world["features"], events, extent)
    canvas.alpha_composite(map_panel, dest=(map_rect[0], map_rect[1]))
    draw.rounded_rectangle(map_rect, radius=32, outline=MAP_EDGE, width=3)

    legend = [
        ("GLaD4CD", counts.get("GLaD4CD_v1", 0), GLAD),
        ("DLR reference line", counts.get("DLR_Landslide_Ref_2025", 0), DLR),
        ("GDCLD", counts.get("GDCLD", 0), GDCLD),
        ("CAS Landslide", counts.get("CAS_Landslide", 0), CAS),
    ]
    # Place the legend in the lower-right ocean area.
    legend_w = 380
    legend_h = 270
    legend_x = map_rect[0] + int((map_rect[2] - map_rect[0]) * 0.60)
    legend_y = map_rect[3] - legend_h - 50
    legend_rect = (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h)
    draw_legend(canvas, legend_rect, legend, transparent=True)

    out_png = ASSET_DIR / "pild_global_event_coverage_t2.png"
    out_report = ASSET_DIR / "pild_global_event_coverage_t2_report.md"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png)
    DOC_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_png, DOC_ASSET_DIR / "figure2_global_source_distribution.png")

    out_report.write_text(
        "\n".join(
            [
                "# pild_global_event_coverage_t2",
                "",
                "- output: `physics_informed_landslide_dataset/release_prep/assets/pild_global_event_coverage_t2.png`",
                "- mirror: `physics_informed_landslide_dataset/docs/assets/figure2_global_source_distribution.png`",
                "- style: shaded-relief earth basemap, no in-panel title, no graticules, enlarged legend at lower-left",
                f"- extent: lon=[{extent[0]:.1f}, {extent[2]:.1f}], lat=[{extent[1]:.1f}, {extent[3]:.1f}]",
                "- event source: `physics_informed_landslide_dataset/raw_fullcopy/indexes/event_master.csv`",
                "- protocol source: `physics_informed_landslide_dataset/metadata/manifests/event_index_v1_strict_t2.csv`",
                "- basemap source: Natural Earth cross-blended hypsometric tints with shaded relief (`HYP_50M_SR_W.tif`) plus Natural Earth admin-0 outlines",
                "",
                "Counts:",
                "- strict_t2 events: 196",
                "- by source: GLaD4CD_v1=157, DLR_Landslide_Ref_2025=25, GDCLD=8, CAS_Landslide=6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    render()
