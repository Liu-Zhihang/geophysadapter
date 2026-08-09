#!/usr/bin/env python3
"""Build an auditable training protocol summary for frozen PILD-GEO4-QC."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pild_sen12_training_loader_v2 import REQUIRED_MANIFEST_COLUMNS, sha256_file
from train_pild_sen12_roleaware_v1 import COMMON_TERRAIN9_NAMES


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-id", default="event_isolated")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    split_path = args.split.resolve()
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    split = pd.read_csv(split_path, keep_default_na=False)

    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("manifest repeats sample_id")
    if not manifest["core_assets_ready"].astype(bool).all():
        raise RuntimeError("GEO4 manifest includes rows without core assets")

    selected = split[split["fold_id"].astype(str).eq(str(args.fold_id))].copy()
    if len(selected) != len(manifest):
        raise RuntimeError(
            f"fold must cover every manifest row exactly once: "
            f"split={len(selected)}, manifest={len(manifest)}"
        )
    if selected["sample_id"].duplicated().any():
        raise RuntimeError("selected fold repeats sample_id")
    if set(selected["sample_id"]) != set(manifest["sample_id"]):
        raise RuntimeError("selected fold and manifest sample identities differ")
    event_roles = selected.groupby("canonical_event_id")["role"].nunique()
    if int(event_roles.max()) != 1:
        raise RuntimeError("canonical event leakage across train/val/test")
    roles = selected["role"].value_counts()
    for role in ("train", "val", "test"):
        if int(roles.get(role, 0)) == 0:
            raise RuntimeError(f"selected fold has no {role} samples")

    merged = selected[["sample_id", "role"]].merge(
        manifest[["sample_id", "dataset_id", "canonical_event_id"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    by_role_dataset = {
        str(role): {
            str(dataset): int(count)
            for dataset, count in group["dataset_id"].value_counts().sort_index().items()
        }
        for role, group in merged.groupby("role", sort=True)
    }
    payload = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "PASS",
        "protocol": "PILD-GEO4-QC-v1 natural-proportion training probe",
        "identity_contract": {
            "fields": [
                "dataset_id",
                "source_id",
                "source_event_id",
                "canonical_event_id",
                "sample_id",
            ],
            "canonical_events": int(manifest["canonical_event_id"].nunique()),
            "split_key": "canonical_event_id",
        },
        "counts": {
            "samples": int(len(manifest)),
            "datasets": int(manifest["dataset_id"].nunique()),
            "canonical_events": int(manifest["canonical_event_id"].nunique()),
            "by_dataset": {
                str(key): int(value)
                for key, value in manifest["dataset_id"].value_counts().sort_index().items()
            },
        },
        "terrain_contract": {
            "schema_id": "pild_sen12_common_terrain9_v2",
            "names": list(COMMON_TERRAIN9_NAMES),
        },
        "sampling_contract": (
            "run-controlled: natural mode emits shuffled manifest rows in observed "
            "patch proportions; no source or event reweighting"
        ),
        "readiness": {
            "manifest_ready": True,
            "core_training_ready": True,
            "full_tmr_training_ready": bool(
                manifest["full_tmr_assets_ready"].astype(bool).all()
            ),
            "training_ready": True,
            "blockers": [],
        },
        "split_counts": {
            str(role): int(count) for role, count in roles.sort_index().items()
        },
        "split_by_dataset": by_role_dataset,
        "outputs": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "split": {
                "path": str(split_path),
                "sha256": sha256_file(split_path),
            },
        },
    }
    write_json(args.out.resolve(), payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
