#!/usr/bin/env python3
"""Build four outcome-blind source-covered event folds for PILD-GEO4-QC."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from pild_sen12_training_loader_v2 import sha256_file


N_FOLDS = 4
DATASET_PRIORITY = (
    "GDCLD",
    "SEN12LS_HARMONIZED",
    "DLR_Landslide_Ref_2025",
    "GLaD4CD_v1",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def assign_event_buckets(manifest: pd.DataFrame) -> dict[str, int]:
    event_dataset_sizes = (
        manifest.groupby(["canonical_event_id", "dataset_id"])
        .size()
        .rename("n_samples")
        .reset_index()
    )
    assignments: dict[str, int] = {}
    loads: dict[str, list[int]] = {
        dataset: [0] * N_FOLDS for dataset in sorted(manifest["dataset_id"].unique())
    }
    for dataset in DATASET_PRIORITY:
        subset = event_dataset_sizes[event_dataset_sizes["dataset_id"].eq(dataset)]
        if subset.empty:
            continue
        ordered = subset.sort_values(
            ["n_samples", "canonical_event_id"],
            ascending=[False, True],
        )
        for row in ordered.itertuples(index=False):
            event_id = str(row.canonical_event_id)
            size = int(row.n_samples)
            if event_id in assignments:
                bucket = assignments[event_id]
            else:
                bucket = min(
                    range(N_FOLDS),
                    key=lambda value: (loads[dataset][value], value),
                )
                assignments[event_id] = bucket
            loads[dataset][bucket] += size
    missing = set(manifest["canonical_event_id"].astype(str)) - set(assignments)
    if missing:
        raise RuntimeError(f"events were not assigned: {sorted(missing)}")
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "sample_id",
        "dataset_id",
        "source_id",
        "canonical_event_id",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("manifest repeats sample_id")
    assignments = assign_event_buckets(manifest)

    rows: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        test_bucket = fold
        val_bucket = (fold + 1) % N_FOLDS
        fold_id = f"source_stratified_{fold}"
        for row in manifest.itertuples(index=False):
            bucket = assignments[str(row.canonical_event_id)]
            role = (
                "test"
                if bucket == test_bucket
                else "val"
                if bucket == val_bucket
                else "train"
            )
            rows.append(
                {
                    "protocol_id": "pild_geo4_source_stratified_v1",
                    "fold_id": fold_id,
                    "sample_id": str(row.sample_id),
                    "dataset_id": str(row.dataset_id),
                    "source_id": str(row.source_id),
                    "canonical_event_id": str(row.canonical_event_id),
                    "heldout_dataset_id": "",
                    "role": role,
                    "role_reason": (
                        f"outcome-blind canonical-event bucket={bucket}; "
                        f"test={test_bucket}; val={val_bucket}"
                    ),
                }
            )
    split = pd.DataFrame(rows)
    for fold_id, fold in split.groupby("fold_id", sort=True):
        if fold["sample_id"].duplicated().any():
            raise RuntimeError(f"{fold_id} repeats sample_id")
        if set(fold["sample_id"]) != set(manifest["sample_id"]):
            raise RuntimeError(f"{fold_id} does not cover the frozen manifest")
        event_roles = fold.groupby("canonical_event_id")["role"].nunique()
        if int(event_roles.max()) != 1:
            raise RuntimeError(f"{fold_id} leaks canonical events")
        coverage = fold.groupby(["dataset_id", "role"]).size().unstack(fill_value=0)
        for dataset in sorted(manifest["dataset_id"].unique()):
            for role in ("train", "val", "test"):
                if int(coverage.loc[dataset, role]) == 0:
                    raise RuntimeError(f"{fold_id} lacks {dataset}/{role}")

    out_csv = args.out_csv.resolve()
    out_summary = args.out_summary.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_csv.with_name(f".{out_csv.name}.tmp-{os.getpid()}")
    split.to_csv(temporary, index=False)
    os.replace(temporary, out_csv)

    count_rows = (
        split.groupby(["fold_id", "dataset_id", "role"])
        .agg(
            n_samples=("sample_id", "size"),
            n_events=("canonical_event_id", "nunique"),
        )
        .reset_index()
    )
    bucket_events: defaultdict[int, list[str]] = defaultdict(list)
    for event_id, bucket in assignments.items():
        bucket_events[int(bucket)].append(event_id)
    summary = {
        "status": "PASS",
        "protocol": "pild_geo4_source_stratified_v1",
        "selection_is_outcome_blind": True,
        "n_folds": N_FOLDS,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split": str(out_csv),
        "split_sha256": sha256_file(out_csv),
        "n_samples": int(len(manifest)),
        "n_canonical_events": int(manifest["canonical_event_id"].nunique()),
        "shared_cross_source_events": sorted(
            event_id
            for event_id, group in manifest.groupby("canonical_event_id")
            if group["dataset_id"].nunique() > 1
        ),
        "event_buckets": {
            str(bucket): sorted(events)
            for bucket, events in sorted(bucket_events.items())
        },
        "counts": count_rows.to_dict(orient="records"),
    }
    write_json(out_summary, summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
