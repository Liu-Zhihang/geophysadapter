#!/usr/bin/env python3
"""Build label-independent nested event folds for one PILD member dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_event_groups(frame: pd.DataFrame, n_folds: int) -> dict[str, int]:
    counts = (
        frame.groupby("canonical_event_id", as_index=False)
        .size()
        .sort_values(["size", "canonical_event_id"], ascending=[False, True])
    )
    totals = [0] * n_folds
    groups: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        group = min(range(n_folds), key=lambda index: (totals[index], index))
        groups[str(row.canonical_event_id)] = group
        totals[group] += int(row.size)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if args.folds < 3:
        raise ValueError("nested train/val/test requires at least three folds")
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    required = {"sample_id", "dataset_id", "source_id", "canonical_event_id"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    frame = manifest[manifest["dataset_id"].eq(args.dataset_id)].copy()
    if frame.empty:
        raise ValueError(f"dataset is absent from manifest: {args.dataset_id}")
    n_events = frame["canonical_event_id"].nunique()
    if n_events < args.folds:
        raise ValueError(f"{args.dataset_id} has only {n_events} events for {args.folds} folds")
    groups = assign_event_groups(frame, args.folds)
    frame["event_group"] = frame["canonical_event_id"].map(groups)

    rows: list[dict[str, object]] = []
    fold_audit: dict[str, dict[str, object]] = {}
    for fold in range(args.folds):
        fold_id = f"{args.dataset_id}_nested{args.folds}_fold{fold}"
        test_group = fold
        val_group = (fold + 1) % args.folds
        counts: dict[str, int] = {}
        events: dict[str, int] = {}
        for row in frame.itertuples(index=False):
            if int(row.event_group) == test_group:
                role = "test"
            elif int(row.event_group) == val_group:
                role = "val"
            else:
                role = "train"
            counts[role] = counts.get(role, 0) + 1
            rows.append(
                {
                    "protocol_id": "pild_dataset_nested_event_v1",
                    "fold_id": fold_id,
                    "sample_id": row.sample_id,
                    "dataset_id": row.dataset_id,
                    "source_id": row.source_id,
                    "canonical_event_id": row.canonical_event_id,
                    "role": role,
                    "role_reason": f"event_group_{int(row.event_group)}",
                }
            )
        role_frame = pd.DataFrame(rows)
        role_frame = role_frame[role_frame["fold_id"].eq(fold_id)]
        events = {
            role: int(role_frame.loc[role_frame["role"].eq(role), "canonical_event_id"].nunique())
            for role in ("train", "val", "test")
        }
        if any(counts.get(role, 0) == 0 for role in ("train", "val", "test")):
            raise RuntimeError(f"empty role in {fold_id}: {counts}")
        if role_frame.groupby("canonical_event_id")["role"].nunique().max() != 1:
            raise RuntimeError(f"event leakage detected in {fold_id}")
        fold_audit[fold_id] = {
            "samples": counts,
            "events": events,
            "test_group": test_group,
            "val_group": val_group,
        }

    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    payload = {
        "status": "complete",
        "scientific_status": "label-independent nested event split",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "dataset_id": args.dataset_id,
        "n_samples": len(frame),
        "n_events": n_events,
        "n_folds": args.folds,
        "assignment_inputs": ["canonical_event_id", "per-event sample count"],
        "label_values_used": False,
        "event_groups": groups,
        "fold_audit": fold_audit,
        "split_csv": str(args.out.resolve()),
        "split_sha256": sha256_file(args.out),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

