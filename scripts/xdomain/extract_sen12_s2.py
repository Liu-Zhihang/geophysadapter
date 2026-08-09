#!/usr/bin/env python3
"""Verify and safely extract the Sen12Landslides harmonized S2 archives."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import tarfile
from pathlib import Path


def sha256sum(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def extract_one(archive: Path, expected_size: int, expected_sha256: str, out_dir: Path, receipt: Path) -> dict:
    if receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    if not archive.is_file() or archive.stat().st_size != expected_size:
        raise RuntimeError(f"Archive is missing or incomplete: {archive}")
    actual = sha256sum(archive)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {archive}: {actual} != {expected_sha256}")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = [member.name for member in members]
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise RuntimeError(f"Unsafe member path in {archive}")
        handle.extractall(out_dir, filter="data")
    payload = {
        "archive": str(archive),
        "sha256": actual,
        "n_members": len(names),
        "n_netcdf": sum(name.endswith(".nc") for name in names),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = root / "data_raw/08_Sen12Landslides"
    manifest_path = root / "physics_informed_landslide_dataset/metadata/pild_xdomain_v1/acquisition/sen12_harmonized_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Sen12 acquisition manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = [item for item in manifest["files"] if item["path"].startswith("data_harmonized/s2/")]
    if len(specs) != 28:
        raise SystemExit(f"Expected 28 S2 archives, found {len(specs)}")
    out_dir = source_root / "extracted"
    receipt_dir = root / "physics_informed_landslide_dataset/metadata/pild_xdomain_v1/acquisition/sen12_s2_extract"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                extract_one,
                source_root / item["path"],
                int(item["size"]),
                item["sha256"],
                out_dir,
                receipt_dir / (Path(item["path"]).name + ".json"),
            ): item
            for item in specs
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            result = future.result()
            results.append(result)
            print(f"[EXTRACTED] {item['path']} netcdf={result['n_netcdf']}", flush=True)

    files = sorted(out_dir.rglob("*.nc"))
    expected = 13628
    if len(files) != expected:
        raise SystemExit(f"Expected {expected} extracted S2 NetCDF files, found {len(files)}")
    summary = {
        "n_archives": len(results),
        "n_netcdf": len(files),
        "source_revision": manifest["revision"],
        "output_root": str(out_dir),
    }
    (receipt_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[DONE] Extracted {len(files)} Sen12 S2 NetCDF files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
