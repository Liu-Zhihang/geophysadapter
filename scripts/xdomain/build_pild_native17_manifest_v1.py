#!/usr/bin/env python3
"""Replace the unified PILD common9 Terrain references with audited native17 caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py

from sen12_terrain_v2 import NATIVE_TERRAIN_V2_NAMES


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE_DIR = ROOT / "metadata/pild_sen12_training_v2"
DEFAULT_OUTDIR = ROOT / "metadata/pild_sen12_training_native17_v1"
TERRAIN_BY_DATASET = {
    "SEN12LS_HARMONIZED": (
        ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_native_terrain_v2_p128.h5"
    ),
    "DLR_Landslide_Ref_2025": (
        ROOT / "processed/hybrid_pinn/dlr_terrain_v3/dlr_copdem_native17_p128.h5"
    ),
    "GDCLD": (
        ROOT / "processed/hybrid_pinn/pild_member_native17_v1/gdcld/native17_p128.h5"
    ),
    "GLaD4CD_v1": (
        ROOT / "processed/hybrid_pinn/pild_member_native17_v1/glad/native17_p128.h5"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decode(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main() -> int:
    args = parse_args()
    source_manifest = args.source_dir / "unified_sample_manifest_v2.csv"
    source_summary = args.source_dir / "protocol_summary_v2.json"
    output_manifest = args.outdir / "unified_sample_manifest_native17_v1.csv"
    output_summary = args.outdir / "protocol_summary_native17_v1.json"
    if args.outdir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    with source_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("source unified manifest is empty")

    cache_identity: dict[str, dict[str, object]] = {}
    for dataset_id, path in TERRAIN_BY_DATASET.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            sample_ids = decode(handle["sample_id"][:])
            names = tuple(decode(handle["terrain_names"][:]))
            if names != tuple(NATIVE_TERRAIN_V2_NAMES):
                raise RuntimeError(f"{dataset_id}: unexpected Terrain names {names}")
            if handle["terrain"].shape[1:] != (17, 128, 128):
                raise RuntimeError(
                    f"{dataset_id}: unexpected Terrain shape {handle['terrain'].shape}"
                )
            if int(handle.attrs.get("complete", 0)) != 1:
                raise RuntimeError(f"{dataset_id}: Terrain cache is incomplete")
        cache_identity[dataset_id] = {
            "path": path.resolve(),
            "sha256": sha256(path),
            "index": {sample_id: index for index, sample_id in enumerate(sample_ids)},
        }

    for row in rows:
        dataset_id = row["dataset_id"]
        if dataset_id not in cache_identity:
            raise RuntimeError(f"no native17 cache registered for {dataset_id}")
        identity = cache_identity[dataset_id]
        sample_id = row["sample_id"]
        indices = identity["index"]
        if sample_id not in indices:
            raise RuntimeError(f"{dataset_id}: native17 cache misses {sample_id}")
        row["manifest_schema_version"] = "pild_sen12_unified_manifest.native17.v1"
        row["terrain_h5_path"] = str(identity["path"])
        row["terrain_h5_index"] = str(indices[sample_id])
        row["terrain_channel_indices"] = ";".join(str(index) for index in range(17))
        row["terrain_schema_id"] = "pild_native_terrain17_v1"
        row["terrain_h5_sha256"] = str(identity["sha256"])

    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    summary["schema_version"] = "pild_sen12_training_protocol.native17.v1"
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["mode"] = "native17 Terrain contract over frozen PILD+Sen12 identities"
    summary["terrain_contract"] = {
        "schema_id": "pild_native_terrain17_v1",
        "names": list(NATIVE_TERRAIN_V2_NAMES),
        "derivation": (
            "17 variables derived on native projected approximately 30 m DEM "
            "with buffered context before alignment to the 10 m prediction grid"
        ),
    }
    summary["outputs"]["manifest"] = {
        "path": str(output_manifest.resolve()),
        "sha256": sha256(output_manifest),
    }
    summary["native17_assets"] = {
        dataset_id: {
            "path": str(value["path"]),
            "sha256": str(value["sha256"]),
            "samples": len(value["index"]),
        }
        for dataset_id, value in cache_identity.items()
    }
    output_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(output_manifest.resolve()),
                "summary": str(output_summary.resolve()),
                "samples": len(rows),
                "terrain_channels": 17,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
