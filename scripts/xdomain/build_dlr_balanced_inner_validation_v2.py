#!/usr/bin/env python3
"""Build balanced, label-independent inner validation events for DLR outer folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv"
)
DEFAULT_SOURCE_SPLIT = (
    PROJECT_ROOT
    / "metadata/protocol_assets/dlr_geo4qc_sen12_protocol_v1"
    / "dlr_eventisolated_nested5_v1.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "metadata/protocol_assets/dlr_geo4qc_sen12_protocol_v2"
    / "dlr_eventisolated_balanced_inner4_v2.csv"
)
DLR_DATASET_ID = "DLR_Landslide_Ref_2025"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-split", type=Path, default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-events", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def select_validation_events(
    candidates: set[str],
    event_counts: Counter[str],
    n_events: int,
    target_samples: int,
    fold: int,
) -> tuple[str, ...]:
    if len(candidates) < n_events:
        raise RuntimeError(
            f"fold {fold} has only {len(candidates)} non-test events; "
            f"cannot select {n_events} validation events"
        )
    ranked: list[tuple[tuple[float, float, str], tuple[str, ...]]] = []
    for combination in itertools.combinations(sorted(candidates), n_events):
        counts = [event_counts[event] for event in combination]
        total = sum(counts)
        sample_gap = abs(total - target_samples) / max(target_samples, 1)
        concentration = max(counts) / max(total, 1)
        digest = hashlib.sha256(
            f"dlr-balanced-inner-v2|{fold}|{'|'.join(combination)}".encode()
        ).hexdigest()
        ranked.append(((sample_gap, concentration, digest), combination))
    return min(ranked, key=lambda item: item[0])[1]


def main() -> int:
    args = parse_args()
    if args.validation_events < 3:
        raise ValueError("--validation-events must be at least 3")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5)")

    manifest = [
        row
        for row in read_csv(args.manifest_csv)
        if row["dataset_id"] == DLR_DATASET_ID
    ]
    manifest.sort(key=lambda row: int(row["base_h5_index"]))
    if not manifest:
        raise RuntimeError(f"no {DLR_DATASET_ID} rows in {args.manifest_csv}")
    sample_ids = [row["sample_id"] for row in manifest]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("duplicate DLR sample_id in manifest")
    sample_to_event = {
        row["sample_id"]: row["canonical_event_id"] for row in manifest
    }
    sample_to_source_event = {
        row["sample_id"]: row["source_event_id"] for row in manifest
    }
    event_counts = Counter(sample_to_event.values())

    source_rows = read_csv(args.source_split)
    folds = sorted({int(row["outer_fold"]) for row in source_rows})
    if not folds:
        raise RuntimeError("source split contains no outer folds")
    source_by_fold = {
        fold: [row for row in source_rows if int(row["outer_fold"]) == fold]
        for fold in folds
    }
    source_group = {}
    for row in source_rows:
        event = row["canonical_event_id"]
        group = int(row["event_group"])
        if event in source_group and source_group[event] != group:
            raise RuntimeError(f"inconsistent source event_group for {event}")
        source_group[event] = group

    expected_samples = set(sample_ids)
    all_events = set(event_counts)
    output_rows: list[dict[str, str | int]] = []
    fold_audit = {}
    for fold in folds:
        current = source_by_fold[fold]
        if {row["sample_id"] for row in current} != expected_samples:
            raise RuntimeError(f"source fold {fold} does not cover the QC DLR manifest")
        test_events = {
            row["canonical_event_id"] for row in current if row["role"] == "test"
        }
        if not test_events:
            raise RuntimeError(f"source fold {fold} has no test events")
        candidates = all_events - test_events
        remaining_samples = sum(event_counts[event] for event in candidates)
        target_samples = round(args.validation_fraction * remaining_samples)
        validation_events = set(
            select_validation_events(
                candidates,
                event_counts,
                args.validation_events,
                target_samples,
                fold,
            )
        )
        train_events = candidates - validation_events
        if test_events & validation_events or test_events & train_events or validation_events & train_events:
            raise RuntimeError(f"event leakage while constructing fold {fold}")

        role_events = {
            "train": train_events,
            "val": validation_events,
            "test": test_events,
        }
        fold_audit[str(fold)] = {
            role: {
                "events": len(events),
                "samples": sum(event_counts[event] for event in events),
                "event_ids": sorted(events),
            }
            for role, events in role_events.items()
        }
        fold_audit[str(fold)]["validation_target_samples"] = target_samples

        for row in manifest:
            sample = row["sample_id"]
            event = sample_to_event[sample]
            role = next(
                role for role, events in role_events.items() if event in events
            )
            output_rows.append(
                {
                    "sample_id": sample,
                    "outer_fold": fold,
                    "role": role,
                    "region_group": sample_to_source_event[sample],
                    "spatial_supergroup": event,
                    "source_event_id": sample_to_source_event[sample],
                    "canonical_event_id": event,
                    "event_group": source_group[event],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    test_counts = Counter(
        row["sample_id"] for row in output_rows if row["role"] == "test"
    )
    if set(test_counts) != expected_samples or set(test_counts.values()) != {1}:
        raise RuntimeError("each sample must appear in test exactly once across outer folds")
    for fold in folds:
        current = [
            row for row in output_rows if int(row["outer_fold"]) == fold
        ]
        if len(current) != len(manifest):
            raise RuntimeError(f"fold {fold} output sample count mismatch")

    protocol = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "label-independent balanced inner validation",
        "label_access_for_partitioning": False,
        "contract": (
            "outer test events unchanged from source split; inner validation uses "
            f"{args.validation_events} non-test events selected by sample-count balance only"
        ),
        "dataset_id": DLR_DATASET_ID,
        "n_samples": len(manifest),
        "n_events": len(all_events),
        "n_outer_folds": len(folds),
        "validation_fraction_target": args.validation_fraction,
        "validation_events_per_fold": args.validation_events,
        "manifest_csv": str(args.manifest_csv.resolve()),
        "manifest_sha256": file_sha256(args.manifest_csv),
        "source_split": str(args.source_split.resolve()),
        "source_split_sha256": file_sha256(args.source_split),
        "output_split": str(args.output.resolve()),
        "output_split_sha256": file_sha256(args.output),
        "fold_audit": fold_audit,
    }
    protocol_path = args.output.with_suffix(".protocol.json")
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
