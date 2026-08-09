#!/usr/bin/env python3
"""Publish the frozen low-dimensional Sen12 Material factor registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from material_factors_v3 import FACTOR_GROUPS, build_material_factors, factor_names


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v2/material_sample_registry_v2.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input, low_memory=False)
    factors = build_material_factors(source)
    identity = [
        "sample_id", "source_id", "region", "region_group",
        "physical_event_id", "physical_event_cluster_id",
    ]
    quality = [
        "q_M_awc", "q_M_soilgrids", "q_M_lithology", "q_M_continuous",
        "q_M", "q_M_status",
    ]
    output = pd.concat([source[identity + quality].copy(), factors], axis=1)
    registry_path = args.outdir / "material_factor_registry_v3.csv"
    output.to_csv(registry_path, index=False)
    names = factor_names("all")
    schema = {
        "schema_version": "3.0",
        "scientific_role": "Footprint-scale positive multiplier of an existing Terrain correction; never a dense direction",
        "model_eligible_features": list(names),
        "model_eligible_dimension": len(names),
        "factor_groups": {key: list(value) for key, value in FACTOR_GROUPS.items()},
        "construction": "deterministic label-free physical compression of Material registry v2",
        "fallback": "q_M=0 implies exact multiplier one",
        "prohibitions": [
            "No dense upsampling",
            "No synthetic jitter",
            "No label- or prediction-based factor construction",
            "No Material-only spatial direction",
        ],
        "source_registry": str(args.input.resolve()),
        "source_registry_sha256": sha256(args.input),
    }
    schema_path = args.outdir / "material_factor_schema_v3.json"
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "status": "complete",
        "n_samples": int(len(output)),
        "n_events": int(output.physical_event_id.nunique()),
        "n_features": len(names),
        "registry": str(registry_path.resolve()),
        "registry_sha256": sha256(registry_path),
        "schema": str(schema_path.resolve()),
        "schema_sha256": sha256(schema_path),
    }
    (args.outdir / "material_factor_receipt_v3.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
