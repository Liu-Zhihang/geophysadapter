#!/usr/bin/env python3
"""Build label-free DLR sidecars and nested event-isolated folds for Sen12 transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BASE = (
    PROJECT_ROOT
    / "processed/hybrid_pinn/pild_core_geo_v2_1_native30_raw/dlr_postrgb_terrain_raw_p128.h5"
)
DEFAULT_OPTICAL = (
    PROJECT_ROOT
    / "processed/hybrid_pinn/pild_prithvi_integration_v1/pild_prithvi_4t6b_p128.h5"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "metadata/pild_sen12_training_v2/unified_sample_manifest_v2.csv"
)
DEFAULT_OUTDIR = (
    PROJECT_ROOT / "processed/hybrid_pinn/dlr_sen12_protocol_transfer_v1"
)
DEFAULT_METADATA_DIR = (
    PROJECT_ROOT / "metadata/protocol_assets/dlr_sen12_protocol_transfer_v1"
)
DLR_DATASET_ID = "DLR_Landslide_Ref_2025"
FOLDS = tuple(range(5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--terrain-source-h5",
        type=Path,
        default=None,
        help="Optional Terrain cache aligned to base-h5 ordering; defaults to base-h5.",
    )
    parser.add_argument(
        "--terrain-output-name",
        default="dlr_common_terrain9_p128.h5",
        help="Terrain sidecar filename inside outdir.",
    )
    parser.add_argument("--global-optical-h5", type=Path, default=DEFAULT_OPTICAL)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["dataset_id"] == DLR_DATASET_ID]
    if not rows:
        raise RuntimeError(f"no {DLR_DATASET_ID} rows in {path}")
    rows.sort(key=lambda row: int(row["base_h5_index"]))
    return rows


def filter_temporally_valid_rows(
    rows: list[dict[str, str]], optical_path: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    indices = np.asarray([int(row["optical_h5_index"]) for row in rows], dtype=np.int64)
    with h5py.File(optical_path, "r") as handle:
        quality = np.asarray(handle["q_visual_temporal"][indices], dtype=np.uint8)
        valid_pixels = np.asarray(handle["optical_valid"][indices]).reshape(len(rows), -1).sum(axis=1)
    kept = []
    excluded = []
    for row, q_visual, n_valid in zip(rows, quality, valid_pixels, strict=True):
        if int(q_visual) == 1 and int(n_valid) > 0:
            kept.append(row)
        else:
            excluded.append(
                {
                    "sample_id": row["sample_id"],
                    "canonical_event_id": row["canonical_event_id"],
                    "q_visual_temporal": str(int(q_visual)),
                    "valid_optical_pixels": str(int(n_valid)),
                    "reason": "four-date optical support failed label-free readiness gate",
                }
            )
    return kept, excluded


def assign_event_groups(rows: list[dict[str, str]]) -> dict[str, int]:
    """Greedy sample-count balancing using event IDs only; labels are never opened."""
    counts = Counter(row["canonical_event_id"] for row in rows)
    ordered = sorted(
        counts,
        key=lambda event: (
            -counts[event],
            hashlib.sha256(f"dlr-sen12-transfer-v1|{event}".encode()).hexdigest(),
        ),
    )
    totals = [0] * len(FOLDS)
    event_totals = [0] * len(FOLDS)
    assignment: dict[str, int] = {}
    for event in ordered:
        group = min(FOLDS, key=lambda value: (totals[value], event_totals[value], value))
        assignment[event] = group
        totals[group] += counts[event]
        event_totals[group] += 1
    return assignment


def write_optical_sidecar(
    source_path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
    expected_ids: list[str],
) -> None:
    indices = np.asarray([int(row["optical_h5_index"]) for row in rows], dtype=np.int64)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(temp, "w") as output:
        observed = decode(source["sample_id"][indices])
        if observed != expected_ids:
            raise RuntimeError("manifest optical indices do not match DLR base sample order")
        for name in ("sample_id", "temporal_coords", "location_coords", "optical_valid", "optical"):
            values = source[name][indices]
            if values.ndim >= 4:
                chunks = (1, *values.shape[1:])
                output.create_dataset(name, data=values, chunks=chunks, compression="lzf")
            else:
                output.create_dataset(name, data=values)
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["complete"] = 1
        output.attrs["selected_samples"] = len(rows)
        output.attrs["source_global_optical_h5"] = str(source_path.resolve())
        output.attrs["selection_contract"] = (
            "DLR rows selected by frozen unified manifest; labels never accessed"
        )
        output.attrs["sample_identity_sha256"] = sha256_strings(expected_ids)
    os.replace(temp, output_path)


def write_base_sidecar(
    source_path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
    expected_ids: list[str],
) -> None:
    indices = np.asarray([int(row["base_h5_index"]) for row in rows], dtype=np.int64)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(temp, "w") as output:
        observed = decode(source["sample_id"][indices])
        if observed != expected_ids:
            raise RuntimeError("manifest base indices do not match filtered DLR sample order")
        for name in (
            "sample_id", "physical_event_id", "event_uid", "source_scene_id",
            "mask", "valid_mask",
        ):
            values = source[name][indices]
            if values.ndim >= 4:
                output.create_dataset(
                    name, data=values, chunks=(1, *values.shape[1:]), compression="lzf"
                )
            else:
                output.create_dataset(name, data=values)
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["complete"] = 1
        output.attrs["source_base_h5"] = str(source_path.resolve())
        output.attrs["selection_contract"] = (
            "label-free four-date optical readiness; mask copied but never read for selection"
        )
        output.attrs["sample_identity_sha256"] = sha256_strings(expected_ids)
    os.replace(temp, output_path)


def write_terrain_sidecar(
    source_path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
    expected_ids: list[str],
) -> None:
    indices = np.asarray([int(row["terrain_h5_index"]) for row in rows], dtype=np.int64)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(temp, "w") as output:
        observed = decode(source["sample_id"][indices])
        if observed != expected_ids:
            raise RuntimeError("DLR Terrain sample order does not match base")
        for name in (
            "sample_id",
            "terrain_names",
            "terrain_scale_roles",
            "terrain_valid",
            "q_T",
            "terrain",
        ):
            if name not in source:
                continue
            values = (
                source[name][:]
                if name in {"terrain_names", "terrain_scale_roles"}
                else source[name][indices]
            )
            if values.ndim >= 4:
                output.create_dataset(
                    name, data=values, chunks=(1, *values.shape[1:]), compression="lzf"
                )
            else:
                output.create_dataset(name, data=values)
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["complete"] = 1
        output.attrs["source_terrain_h5"] = str(source_path.resolve())
        output.attrs["terrain_schema"] = str(
            source.attrs.get("terrain_schema", f"terrain_{source['terrain'].shape[1]}ch")
        )
        output.attrs["sample_identity_sha256"] = sha256_strings(expected_ids)
    os.replace(temp, output_path)


def write_split(
    output_path: Path,
    rows: list[dict[str, str]],
    assignment: dict[str, int],
) -> None:
    fields = (
        "sample_id",
        "outer_fold",
        "role",
        "region_group",
        "spatial_supergroup",
        "source_event_id",
        "canonical_event_id",
        "event_group",
    )
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fold in FOLDS:
            test_group = fold
            val_group = (fold + 1) % len(FOLDS)
            for row in rows:
                group = assignment[row["canonical_event_id"]]
                role = "test" if group == test_group else "val" if group == val_group else "train"
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "outer_fold": fold,
                        "role": role,
                        "region_group": row["source_event_id"],
                        "spatial_supergroup": row["canonical_event_id"],
                        "source_event_id": row["source_event_id"],
                        "canonical_event_id": row["canonical_event_id"],
                        "event_group": group,
                    }
                )


def audit_split(path: Path, expected_ids: list[str]) -> dict:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    audit: dict[str, dict] = {}
    for fold in FOLDS:
        current = [row for row in rows if int(row["outer_fold"]) == fold]
        roles = {role: [row for row in current if row["role"] == role] for role in ("train", "val", "test")}
        if {row["sample_id"] for row in current} != set(expected_ids):
            raise RuntimeError(f"fold {fold} does not contain exactly all DLR samples")
        event_sets = {
            role: {row["spatial_supergroup"] for row in values} for role, values in roles.items()
        }
        if any(
            event_sets[left] & event_sets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise RuntimeError(f"canonical-event leakage in fold {fold}")
        audit[str(fold)] = {
            role: {"samples": len(values), "events": len(event_sets[role])}
            for role, values in roles.items()
        }
    test_counts = Counter(row["sample_id"] for row in rows if row["role"] == "test")
    if set(test_counts.values()) != {1}:
        raise RuntimeError("each sample must be test exactly once")
    return audit


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    optical_out = args.outdir / "dlr_prithvi_4t6b_p128.h5"
    terrain_source_h5 = args.terrain_source_h5 or args.base_h5
    terrain_out = args.outdir / args.terrain_output_name
    base_out = args.outdir / "dlr_base_temporalvalid_p128.h5"
    split_out = args.metadata_dir / "dlr_eventisolated_nested5_v1.csv"
    protocol_out = args.metadata_dir / "protocol.json"
    excluded_out = args.metadata_dir / "excluded_temporal_invalid.csv"
    outputs = (base_out, optical_out, terrain_out, split_out, excluded_out, protocol_out)
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError(f"output exists; use --force: {[str(path) for path in outputs if path.exists()]}")

    all_rows = load_manifest(args.manifest_csv)
    with h5py.File(args.base_h5, "r") as base:
        all_base_ids = decode(base["sample_id"][:])
    base_indices = np.asarray(
        [int(row["base_h5_index"]) for row in all_rows], dtype=np.int64
    )
    if np.any(base_indices < 0) or np.any(base_indices >= len(all_base_ids)):
        raise RuntimeError("unified manifest contains an out-of-range DLR base_h5_index")
    if [all_base_ids[index] for index in base_indices] != [
        row["sample_id"] for row in all_rows
    ]:
        raise RuntimeError("unified manifest DLR indices differ from base H5 identities")
    rows, excluded = filter_temporally_valid_rows(all_rows, args.global_optical_h5)
    expected_ids = [row["sample_id"] for row in rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("duplicate DLR sample IDs")

    assignment = assign_event_groups(rows)
    write_base_sidecar(args.base_h5, base_out, rows, expected_ids)
    write_optical_sidecar(args.global_optical_h5, optical_out, rows, expected_ids)
    write_terrain_sidecar(terrain_source_h5, terrain_out, rows, expected_ids)
    write_split(split_out, rows, assignment)
    with excluded_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_id",
                "canonical_event_id",
                "q_visual_temporal",
                "valid_optical_pixels",
                "reason",
            ),
        )
        writer.writeheader()
        writer.writerows(excluded)
    split_audit = audit_split(split_out, expected_ids)
    event_counts = Counter(row["canonical_event_id"] for row in rows)
    with h5py.File(terrain_source_h5, "r") as terrain_source:
        terrain_names = decode(terrain_source["terrain_names"][:])
        terrain_source_kind = str(terrain_source.attrs.get("source_kind", "legacy"))
        terrain_derivative_policy = str(
            terrain_source.attrs.get("derivative_policy", "legacy common9 contract")
        )
    protocol = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "exploratory cross-dataset protocol transfer",
        "label_access_for_partitioning": False,
        "contract": (
            "DLR all-sample nested event-isolated 5-fold; test=f, val=f+1, train=remaining groups"
        ),
        "n_samples": len(rows),
        "n_source_samples": len(all_rows),
        "n_temporal_invalid_excluded": len(excluded),
        "excluded_temporal_invalid_csv": str(excluded_out.resolve()),
        "n_canonical_events": len(event_counts),
        "event_group_samples": {
            str(group): sum(event_counts[event] for event, value in assignment.items() if value == group)
            for group in FOLDS
        },
        "fold_audit": split_audit,
        "base_h5": str(base_out.resolve()),
        "source_base_h5": str(args.base_h5.resolve()),
        "optical_h5": str(optical_out.resolve()),
        "terrain_h5": str(terrain_out.resolve()),
        "source_terrain_h5": str(terrain_source_h5.resolve()),
        "split_csv": str(split_out.resolve()),
        "sample_identity_sha256": sha256_strings(expected_ids),
        "split_csv_sha256": sha256_file(split_out),
        "terrain_contract": {
            "channels": len(terrain_names),
            "names": terrain_names,
            "source_kind": terrain_source_kind,
            "derivative_policy": terrain_derivative_policy,
        },
        "frozen_sen12_gate": [0.3, 0.7, 4.0, 1.0],
    }
    protocol_out.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
