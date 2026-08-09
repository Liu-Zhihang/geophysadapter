#!/usr/bin/env python3
"""Audit local PILD-XDomain source assets and emit machine-readable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: Path, size: int, checksum: str, full_checksum: bool) -> dict:
    algorithm, expected = checksum.split(":", 1)
    item = {
        "path": str(path),
        "exists": path.is_file(),
        "expected_bytes": size,
        "actual_bytes": path.stat().st_size if path.is_file() else None,
        "checksum_algorithm": algorithm,
        "expected_checksum": expected,
        "actual_checksum": None,
        "status": "missing",
    }
    if not item["exists"]:
        return item
    if item["actual_bytes"] != size:
        item["status"] = "size_mismatch"
        return item
    if full_checksum:
        item["actual_checksum"] = digest(path, algorithm)
        item["status"] = (
            "verified" if item["actual_checksum"] == expected else "checksum_mismatch"
        )
    else:
        item["status"] = "size_verified"
    return item


def zenodo_files(record_path: Path, selected: set[str]) -> list[dict]:
    record = json.loads(record_path.read_text())
    return [
        {
            "name": item["key"],
            "size": int(item["size"]),
            "checksum": item["checksum"],
        }
        for item in record["files"]
        if item["key"] in selected
    ]


def audit_sen12(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    completed = []
    partial = []
    missing = []
    completed_bytes = 0
    partial_bytes = 0
    for item in manifest["files"]:
        target = root / item["path"]
        part = target.with_name(target.name + ".part")
        if target.is_file() and target.stat().st_size == item["size"]:
            completed.append(item["path"])
            completed_bytes += item["size"]
        elif part.is_file():
            partial.append(item["path"])
            partial_bytes += part.stat().st_size
        else:
            missing.append(item["path"])
    received = completed_bytes + partial_bytes
    return {
        "revision": manifest["revision"],
        "expected_files": manifest["total_files"],
        "expected_bytes": manifest["total_bytes"],
        "completed_files": len(completed),
        "partial_files": len(partial),
        "not_started_files": len(missing),
        "completed_bytes": completed_bytes,
        "partial_bytes": partial_bytes,
        "received_fraction": received / manifest["total_bytes"],
        "complete": len(completed) == manifest["total_files"],
        "completed_paths": completed,
        "partial_paths": partial,
        "missing_paths": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/acquisition"),
    )
    parser.add_argument("--full-checksum", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hr_root = root / "data_raw/03_HR_GLDD"
    hr_record = hr_root / "zenodo_record_7189381.json"
    hr_selected = {"trainX.npy", "trainY.npy", "valX.npy", "valY.npy", "testX.npy", "testY.npy"}
    hr_files = [
        check_file(hr_root / item["name"], item["size"], item["checksum"], args.full_checksum)
        for item in zenodo_files(hr_record, hr_selected)
    ]
    hr_shapes = {}
    for name in sorted(hr_selected):
        path = hr_root / name
        if path.is_file():
            hr_shapes[name] = list(np.load(path, mmap_mode="r").shape)

    uglc_root = root / "data_raw/10_UGLC"
    uglc_record = uglc_root / "zenodo_record_18643456.json"
    uglc_selected = {"UGLC_point.csv", "UGLC_poly.csv", "UGLC_tile_grid_map.jpeg"}
    uglc_files = [
        check_file(uglc_root / item["name"], item["size"], item["checksum"], args.full_checksum)
        for item in zenodo_files(uglc_record, uglc_selected)
    ]

    nasa_root = root / "data_raw/09_NASA_COOLR_Rainfall_Events"
    nasa_record = json.loads((nasa_root / "figshare_record_26972467.json").read_text())
    nasa_item = nasa_record["files"][0]
    nasa_file = check_file(
        nasa_root / nasa_item["name"],
        int(nasa_item["size"]),
        "md5:" + nasa_item["supplied_md5"],
        args.full_checksum,
    )
    if nasa_file["exists"]:
        try:
            with zipfile.ZipFile(nasa_file["path"]) as archive:
                nasa_file["zip_members"] = len(archive.infolist())
                nasa_file["zip_status"] = "readable"
        except zipfile.BadZipFile:
            nasa_file["zip_status"] = "invalid"

    usgs_root = root / "data_raw/07_USGS_Inventory_v3"
    usgs_zip = usgs_root / "US_Landslide_v3_gpkg.zip"
    usgs_gpkg = usgs_root / "extracted/US_Landslide_v3_gpkg/US_Landslide_v3.gpkg"
    usgs = {
        "archive_path": str(usgs_zip),
        "archive_bytes": usgs_zip.stat().st_size if usgs_zip.is_file() else None,
        "archive_readable": False,
        "gpkg_path": str(usgs_gpkg),
        "gpkg_bytes": usgs_gpkg.stat().st_size if usgs_gpkg.is_file() else None,
    }
    if usgs_zip.is_file():
        try:
            with zipfile.ZipFile(usgs_zip) as archive:
                usgs["archive_members"] = len(archive.infolist())
                usgs["archive_readable"] = True
        except zipfile.BadZipFile:
            pass

    sen12 = audit_sen12(
        root / "data_raw/08_Sen12Landslides",
        root / "metadata/pild_xdomain_v1/acquisition/sen12_harmonized_manifest.json",
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_checksum": args.full_checksum,
        "hr_gldd": {"files": hr_files, "array_shapes": hr_shapes},
        "uglc": {"files": uglc_files},
        "nasa_coolr": nasa_file,
        "usgs_nlsi_v3": usgs,
        "sen12landslides": sen12,
    }
    json_path = outdir / "local_asset_audit.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    checks = hr_files + uglc_files + [nasa_file]
    failed = [item for item in checks if item["status"] not in {"verified", "size_verified"}]
    md = [
        "# PILD-XDomain local asset audit",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Checksum mode: `{'full' if args.full_checksum else 'size-only'}`",
        f"- Static source failures: `{len(failed)}`",
        f"- Sen12 received: `{100 * sen12['received_fraction']:.2f}%` "
        f"({sen12['completed_files']} complete, {sen12['partial_files']} partial, "
        f"{sen12['not_started_files']} not started)",
        "",
        "| Source | Status | Detail |",
        "|---|---|---|",
        f"| HR-GLDD | {'PASS' if all(x['status'] in {'verified', 'size_verified'} for x in hr_files) else 'FAIL'} | {len(hr_files)} arrays; shapes recorded |",
        f"| UGLC | {'PASS' if all(x['status'] in {'verified', 'size_verified'} for x in uglc_files) else 'FAIL'} | {len(uglc_files)} selected assets |",
        f"| NASA COOLR | {nasa_file['status']} | ZIP {nasa_file.get('zip_status', 'missing')} |",
        f"| USGS NLSI v3 | {'PASS' if usgs['archive_readable'] and usgs['gpkg_bytes'] else 'FAIL'} | archive and extracted GeoPackage |",
        f"| Sen12Landslides | {'PASS' if sen12['complete'] else 'IN PROGRESS'} | pinned revision `{sen12['revision']}` |",
        "",
        "This audit establishes acquisition integrity only. Train/test eligibility is decided by the event, georeference, license, and support-quality gates.",
    ]
    (outdir / "local_asset_audit.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"json": str(json_path), "failed": len(failed), "sen12": sen12}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
