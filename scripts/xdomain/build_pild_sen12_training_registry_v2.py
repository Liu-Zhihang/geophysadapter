#!/usr/bin/env python3
"""Build an audited PILD + Sen12 training registry without starting training.

The command is validate-only by default. Pass ``--write`` to atomically publish
the manifest and split tables. Missing future assets are readiness blockers,
not validation failures, so a planning manifest can be audited before caches
are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PILD_SOURCE = "PILD"
SEN12_SOURCE = "Sen12Landslides"
SEN12_DATASET = "SEN12LS_HARMONIZED"
EXPECTED_PILD_SAMPLES = 2_937
EXPECTED_SEN12_SAMPLES = 4_979
EXPECTED_RAW_EVENTS = 57
EXPECTED_CANONICAL_EVENTS = 56

COMMON_TERRAIN_NAMES = (
    "elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "laplacian_curvature",
    "tpi_90m",
    "tpi_300m",
    "ruggedness_90m",
    "local_relief_300m",
)
PILD_TERRAIN_NAMES = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "curvature_laplacian",
    "tpi_90m",
    "tpi_300m",
    "roughness_90m",
    "local_relief_300m",
)
SEN12_NATIVE_TERRAIN_NAMES = (
    "elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "profile_curvature",
    "plan_curvature",
    "laplacian_curvature",
    "tpi_90m",
    "tpi_300m",
    "tpi_900m",
    "local_std_90m",
    "local_std_300m",
    "local_relief_300m",
    "local_relief_900m",
    "valley_depth_900m",
    "ridge_height_900m",
    "ruggedness_90m",
)
SEN12_COMMON_TERRAIN_INDICES = (0, 1, 2, 3, 6, 7, 8, 16, 12)

MANIFEST_COLUMNS = (
    "manifest_schema_version",
    "manifest_index",
    "dataset_id",
    "source_id",
    "source_event_id",
    "canonical_event_id",
    "sample_id",
    "source_sample_id",
    "base_h5_path",
    "base_h5_index",
    "base_h5_sha256",
    "optical_h5_path",
    "optical_h5_index",
    "optical_h5_sha256",
    "terrain_h5_path",
    "terrain_h5_index",
    "terrain_channel_indices",
    "terrain_schema_id",
    "terrain_h5_sha256",
    "material_registry_path",
    "material_registry_index",
    "material_registry_sha256",
    "trigger_registry_path",
    "trigger_registry_index",
    "trigger_registry_sha256",
    "source_registry_path",
    "source_registry_sha256",
    "event_alias_registry_path",
    "event_alias_registry_sha256",
    "core_assets_ready",
    "material_ready",
    "trigger_ready",
    "full_tmr_assets_ready",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def require_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicate = frame.duplicated(columns, keep=False)
    if duplicate.any():
        rows = frame.loc[duplicate, columns].head(5).to_dict("records")
        raise ValueError(f"{label} has duplicate identities: {rows}")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def asset_record(path: Path, role: str, hash_mode: str, required: bool = True) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required {role} is missing: {path}")
        return {
            "role": role,
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
            "sha256": None,
        }
    stat = path.stat()
    return {
        "role": role,
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path) if hash_mode == "full" else None,
    }


def validate_aliases(path: Path, summary_path: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    aliases = pd.read_csv(path, keep_default_na=False)
    require_columns(
        aliases,
        {
            "source_collection",
            "source_event_id",
            "alias_decision",
            "canonical_physical_event_id",
            "split_group_id",
        },
        "event alias registry",
    )
    require_unique(aliases, ["source_collection", "source_event_id"], "event alias registry")
    if len(aliases) != EXPECTED_RAW_EVENTS:
        raise ValueError(f"expected {EXPECTED_RAW_EVENTS} raw event rows, found {len(aliases)}")
    if aliases["canonical_physical_event_id"].nunique() != EXPECTED_CANONICAL_EVENTS:
        raise ValueError(
            f"expected {EXPECTED_CANONICAL_EVENTS} canonical events, found "
            f"{aliases['canonical_physical_event_id'].nunique()}"
        )
    if not aliases["canonical_physical_event_id"].eq(aliases["split_group_id"]).all():
        raise ValueError("split_group_id must equal canonical_physical_event_id")
    canonical_sizes = aliases.groupby("canonical_physical_event_id").size()
    if int((canonical_sizes == 2).sum()) != 1 or int(canonical_sizes.max()) != 2:
        raise ValueError("57 -> 56 contract requires exactly one two-source canonical alias")
    merged = aliases[aliases["canonical_physical_event_id"].isin(canonical_sizes[canonical_sizes == 2].index)]
    if set(merged["source_collection"]) != {PILD_SOURCE, SEN12_SOURCE}:
        raise ValueError("the merged canonical alias must contain one PILD and one Sen12 event")
    if set(merged["alias_decision"]) != {"auto-match"}:
        raise ValueError("only auto-match rows may share a canonical identity")
    non_auto = aliases[aliases["alias_decision"] != "auto-match"]
    if non_auto["canonical_physical_event_id"].duplicated().any():
        raise ValueError("manual-review/distinct events must remain separate")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "source_event_rows": EXPECTED_RAW_EVENTS,
        "canonical_events_after_auto_deduplication": EXPECTED_CANONICAL_EVENTS,
        "automatic_alias_pairs": 1,
        "coverage_complete": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"event alias summary {key}={summary.get(key)!r}, expected {value!r}")
    mapping = {
        (str(row.source_collection), str(row.source_event_id)): str(row.canonical_physical_event_id)
        for row in aliases.itertuples(index=False)
    }
    return aliases, mapping


def validate_h5_identity(
    path: Path,
    expected_ids: list[str],
    *,
    required_datasets: set[str],
    expected_shapes: dict[str, tuple[int, ...]] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        missing = sorted(required_datasets - set(handle.keys()))
        if missing:
            raise ValueError(f"{path} missing HDF5 datasets: {missing}")
        observed_ids = decode_strings(handle["sample_id"][:])
        if observed_ids != expected_ids:
            raise ValueError(f"{path} sample identity/order differs from registry")
        if require_complete and int(handle.attrs.get("complete", 0)) != 1:
            raise ValueError(f"{path} does not carry complete=1")
        for name, shape in (expected_shapes or {}).items():
            if tuple(handle[name].shape) != shape:
                raise ValueError(f"{path}:{name} shape={handle[name].shape}, expected={shape}")
        return {
            "sample_identity_sha256": sha256_lines(observed_ids),
            "complete": bool(int(handle.attrs.get("complete", 0))),
        }


def validate_pild_base(frame: pd.DataFrame) -> None:
    for path_text, group in frame.groupby("h5_path", sort=True):
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            required = {"sample_id", "physical_event_id", "mask", "valid_mask", "terrain", "terrain_valid", "terrain_names"}
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise ValueError(f"{path} missing HDF5 datasets: {missing}")
            names = decode_strings(handle["terrain_names"][:])
            if tuple(names) != PILD_TERRAIN_NAMES:
                raise ValueError(f"{path} PILD Terrain schema changed: {names}")
            indices = group["h5_index"].astype(int).to_numpy()
            if indices.min() < 0 or indices.max() >= len(handle["sample_id"]):
                raise ValueError(f"{path} contains out-of-range h5_index")
            sample_ids = decode_strings(handle["sample_id"][indices])
            event_ids = decode_strings(handle["physical_event_id"][indices])
            if sample_ids != group["sample_id"].astype(str).tolist():
                raise ValueError(f"{path} sample_id does not match readiness h5_index")
            if event_ids != group["physical_event_id"].astype(str).tolist():
                raise ValueError(f"{path} physical_event_id does not match readiness")


def load_support(
    path: Path | None,
    expected: pd.DataFrame,
    event_columns: tuple[str, ...],
    label: str,
) -> tuple[pd.DataFrame | None, dict[str, int]]:
    if path is None or not path.is_file():
        return None, {}
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    require_columns(frame, {"sample_id"}, label)
    require_unique(frame, ["sample_id"], label)
    if set(frame["sample_id"].astype(str)) != set(expected["sample_id"].astype(str)):
        raise ValueError(f"{label} sample identity does not exactly cover its dataset")
    event_column = next((column for column in event_columns if column in frame.columns), None)
    if event_column is None:
        raise ValueError(f"{label} lacks an event identity column from {event_columns}")
    joined = expected[["sample_id", "source_event_id"]].merge(
        frame[["sample_id", event_column]], on="sample_id", validate="one_to_one"
    )
    mismatch = joined["source_event_id"].astype(str) != joined[event_column].astype(str)
    if mismatch.any():
        raise ValueError(f"{label} event identity mismatch for {int(mismatch.sum())} samples")
    return frame, {str(sample_id): index for index, sample_id in enumerate(frame["sample_id"].astype(str))}


def event_partition(events: Iterable[str], seed: int, val_fraction: float, test_fraction: float) -> dict[str, str]:
    ordered = sorted(
        set(str(event) for event in events),
        key=lambda event: hashlib.sha256(f"{seed}|{event}".encode("utf-8")).hexdigest(),
    )
    if len(ordered) < 3:
        raise ValueError("event-isolated split requires at least three canonical events")
    n_test = max(1, int(round(len(ordered) * test_fraction)))
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    if n_test + n_val >= len(ordered):
        raise ValueError("validation/test fractions leave no training events")
    roles = {event: "test" for event in ordered[:n_test]}
    roles.update({event: "val" for event in ordered[n_test : n_test + n_val]})
    roles.update({event: "train" for event in ordered[n_test + n_val :]})
    return roles


def build_event_isolated_split(
    manifest: pd.DataFrame, seed: int, val_fraction: float, test_fraction: float
) -> pd.DataFrame:
    roles = event_partition(manifest["canonical_event_id"], seed, val_fraction, test_fraction)
    out = manifest[["sample_id", "dataset_id", "source_id", "canonical_event_id"]].copy()
    out.insert(0, "protocol_id", f"pild_sen12_event_isolated_seed{seed}_v2")
    out.insert(1, "fold_id", "event_isolated")
    out["heldout_dataset_id"] = ""
    out["role"] = out["canonical_event_id"].map(roles)
    out["role_reason"] = "canonical_event_partition"
    validate_split(out, "event-isolated")
    return out


def build_lodo_split(manifest: pd.DataFrame, seed: int, val_fraction: float) -> pd.DataFrame:
    folds: list[pd.DataFrame] = []
    for fold_number, heldout in enumerate(sorted(manifest["dataset_id"].unique())):
        out = manifest[["sample_id", "dataset_id", "source_id", "canonical_event_id"]].copy()
        out.insert(0, "protocol_id", "pild_sen12_leave_one_dataset_out_v2")
        out.insert(1, "fold_id", f"lodo_{fold_number:02d}_{heldout}")
        out["heldout_dataset_id"] = heldout
        out["role"] = "train"
        out["role_reason"] = "nonheldout_dataset"
        test = out["dataset_id"].eq(heldout)
        out.loc[test, ["role", "role_reason"]] = ["test", "heldout_dataset"]
        test_events = set(out.loc[test, "canonical_event_id"])
        overlap = ~test & out["canonical_event_id"].isin(test_events)
        out.loc[overlap, ["role", "role_reason"]] = ["excluded", "canonical_event_overlap_with_test"]
        remaining_events = set(out.loc[out["role"].eq("train"), "canonical_event_id"])
        if len(remaining_events) < 2:
            raise ValueError(f"{heldout} leaves too few canonical events for train/val")
        n_val = max(1, int(round(len(remaining_events) * val_fraction)))
        ranked = sorted(
            remaining_events,
            key=lambda event: hashlib.sha256(
                f"{seed}|{heldout}|{event}".encode("utf-8")
            ).hexdigest(),
        )
        val_events = set(ranked[: min(n_val, len(ranked) - 1)])
        val = out["role"].eq("train") & out["canonical_event_id"].isin(val_events)
        out.loc[val, ["role", "role_reason"]] = ["val", "canonical_event_validation_partition"]
        validate_split(out, f"LODO {heldout}")
        folds.append(out)
    return pd.concat(folds, ignore_index=True)


def validate_split(frame: pd.DataFrame, label: str) -> None:
    for role in ("train", "val", "test"):
        if not frame["role"].eq(role).any():
            raise ValueError(f"{label} has empty role={role}")
    event_roles = frame[frame["role"].isin(["train", "val", "test"])].groupby(
        "canonical_event_id"
    )["role"].nunique()
    if (event_roles > 1).any():
        raise ValueError(f"{label} leaks canonical events across roles")


def parse_args() -> argparse.Namespace:
    base = ROOT / "processed/hybrid_pinn"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pild-readiness", type=Path, default=base / "pild_prithvi_integration_v1/pild_window_readiness.csv")
    parser.add_argument("--event-aliases", type=Path, default=base / "pild_prithvi_integration_v1/pild_sen12_event_aliases_v1.csv")
    parser.add_argument("--event-alias-summary", type=Path, default=base / "pild_prithvi_integration_v1/event_alias_summary_v1.json")
    parser.add_argument("--pild-optical-h5", type=Path, default=base / "pild_prithvi_integration_v1/pild_prithvi_4t6b_p128.h5")
    parser.add_argument("--pild-optical-marker", type=Path, default=base / "pild_prithvi_integration_v1/pild_prithvi_4t6b_p128.h5.complete.json")
    parser.add_argument("--sen12-cache-index", type=Path, default=base / "sen12_s2_xdomain_v1/cache_index_v1.csv")
    parser.add_argument("--sen12-base-h5", type=Path, default=base / "sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5")
    parser.add_argument("--sen12-optical-h5", type=Path, default=base / "sen12_s2_xdomain_v2/sen12_prithvi_4t6b_p128.h5")
    parser.add_argument("--sen12-terrain-h5", type=Path, default=base / "sen12_s2_xdomain_v2/sen12_native_terrain_v2_p128.h5")
    parser.add_argument("--sen12-material-registry", type=Path, default=base / "sen12_context_v1/material_sample_registry.csv")
    parser.add_argument("--sen12-trigger-registry", type=Path, default=base / "sen12_context_v1/trigger_sample_registry_v1.csv")
    parser.add_argument("--pild-material-registry", type=Path)
    parser.add_argument("--pild-trigger-registry", type=Path)
    parser.add_argument("--outdir", type=Path, default=ROOT / "metadata/pild_sen12_training_v2")
    parser.add_argument("--event-split-seed", type=int, default=20260722)
    parser.add_argument("--val-event-fraction", type=float, default=0.15)
    parser.add_argument("--test-event-fraction", type=float, default=0.20)
    parser.add_argument("--hash-mode", choices=("full", "stat"), default="full")
    parser.add_argument("--write", action="store_true", help="Publish outputs; default is validate-only/dry-run.")
    return parser.parse_args()


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0 < args.val_event_fraction < 1 or not 0 < args.test_event_fraction < 1:
        raise ValueError("event split fractions must lie strictly between zero and one")
    aliases, alias_map = validate_aliases(args.event_aliases, args.event_alias_summary)

    pild = pd.read_csv(args.pild_readiness, keep_default_na=False)
    require_columns(
        pild,
        {"sample_id", "physical_event_id", "dataset_id", "h5_path", "h5_index"},
        "PILD readiness",
    )
    require_unique(pild, ["sample_id"], "PILD readiness")
    if len(pild) != EXPECTED_PILD_SAMPLES:
        raise ValueError(f"expected {EXPECTED_PILD_SAMPLES} PILD samples, found {len(pild)}")
    validate_pild_base(pild)

    sen12 = pd.read_csv(args.sen12_cache_index, keep_default_na=False)
    require_columns(sen12, {"cache_index", "sample_id", "patch_id", "physical_event_id"}, "Sen12 cache index")
    require_unique(sen12, ["sample_id"], "Sen12 cache index")
    if len(sen12) != EXPECTED_SEN12_SAMPLES:
        raise ValueError(f"expected {EXPECTED_SEN12_SAMPLES} Sen12 samples, found {len(sen12)}")
    if sen12["cache_index"].astype(int).tolist() != list(range(EXPECTED_SEN12_SAMPLES)):
        raise ValueError("Sen12 cache_index must be contiguous and registry ordered")

    pild_event_keys = {(PILD_SOURCE, event) for event in pild["physical_event_id"].astype(str).unique()}
    sen12_event_keys = {(SEN12_SOURCE, event) for event in sen12["physical_event_id"].astype(str).unique()}
    if pild_event_keys | sen12_event_keys != set(alias_map):
        missing = (pild_event_keys | sen12_event_keys) - set(alias_map)
        extra = set(alias_map) - (pild_event_keys | sen12_event_keys)
        raise ValueError(f"sample/alias event coverage differs: missing={sorted(missing)}, extra={sorted(extra)}")

    sen12_ids = sen12["sample_id"].astype(str).tolist()
    validate_h5_identity(
        args.sen12_base_h5,
        sen12_ids,
        required_datasets={"sample_id", "mask", "valid_mask"},
        expected_shapes={"mask": (EXPECTED_SEN12_SAMPLES, 1, 128, 128)},
    )
    validate_h5_identity(
        args.sen12_optical_h5,
        sen12_ids,
        required_datasets={"sample_id", "optical", "optical_valid", "temporal_coords", "location_coords"},
        expected_shapes={"optical": (EXPECTED_SEN12_SAMPLES, 6, 4, 128, 128)},
    )
    validate_h5_identity(
        args.sen12_terrain_h5,
        sen12_ids,
        required_datasets={"sample_id", "terrain", "terrain_valid", "terrain_names"},
        expected_shapes={"terrain": (EXPECTED_SEN12_SAMPLES, 17, 128, 128)},
    )
    with h5py.File(args.sen12_terrain_h5, "r") as terrain_handle:
        if tuple(decode_strings(terrain_handle["terrain_names"][:])) != SEN12_NATIVE_TERRAIN_NAMES:
            raise ValueError("Sen12 native Terrain-v2 schema changed")

    pild_expected = pd.DataFrame(
        {
            "sample_id": pild["sample_id"].astype(str),
            "source_event_id": pild["physical_event_id"].astype(str),
        }
    )
    sen12_expected = pd.DataFrame(
        {
            "sample_id": sen12["sample_id"].astype(str),
            "source_event_id": sen12["physical_event_id"].astype(str),
        }
    )
    sen12_material, sen12_material_index = load_support(
        args.sen12_material_registry,
        sen12_expected,
        ("physical_event_cluster_id", "physical_event_id"),
        "Sen12 Material registry",
    )
    sen12_trigger, sen12_trigger_index = load_support(
        args.sen12_trigger_registry,
        sen12_expected,
        ("physical_event_id", "physical_event_cluster_id"),
        "Sen12 Trigger registry",
    )
    pild_material, pild_material_index = load_support(
        args.pild_material_registry,
        pild_expected,
        ("physical_event_id", "physical_event_cluster_id"),
        "PILD Material registry",
    )
    pild_trigger, pild_trigger_index = load_support(
        args.pild_trigger_registry,
        pild_expected,
        ("physical_event_id", "physical_event_cluster_id"),
        "PILD Trigger registry",
    )

    asset_specs: list[tuple[Path, str, bool]] = [
        (args.pild_readiness, "pild_readiness", True),
        (args.event_aliases, "event_alias_registry", True),
        (args.event_alias_summary, "event_alias_summary", True),
        (args.sen12_cache_index, "sen12_cache_index", True),
        (args.sen12_base_h5, "sen12_base_h5", True),
        (args.sen12_optical_h5, "sen12_optical_h5", True),
        (args.sen12_terrain_h5, "sen12_terrain_h5", True),
        (args.sen12_material_registry, "sen12_material_registry", True),
        (args.sen12_trigger_registry, "sen12_trigger_registry", True),
        (args.pild_optical_h5, "pild_optical_h5", False),
        (args.pild_optical_marker, "pild_optical_completion_marker", False),
    ]
    for path_text in sorted(pild["h5_path"].astype(str).unique()):
        asset_specs.append((Path(path_text), "pild_base_h5", True))
    if args.pild_material_registry is not None:
        asset_specs.append((args.pild_material_registry, "pild_material_registry", True))
    if args.pild_trigger_registry is not None:
        asset_specs.append((args.pild_trigger_registry, "pild_trigger_registry", True))
    assets = [asset_record(path, role, args.hash_mode, required) for path, role, required in asset_specs]
    asset_by_path = {record["path"]: record for record in assets}

    pild_optical_ready = bool(args.pild_optical_h5.is_file() and args.pild_optical_marker.is_file())
    pild_optical_hash = ""
    if pild_optical_ready:
        pild_ids = pild["sample_id"].astype(str).tolist()
        validate_h5_identity(
            args.pild_optical_h5,
            pild_ids,
            required_datasets={"sample_id", "optical", "optical_valid", "temporal_coords", "location_coords", "q_visual_temporal"},
            expected_shapes={"optical": (EXPECTED_PILD_SAMPLES, 6, 4, 128, 128)},
        )
        with h5py.File(args.pild_optical_h5, "r") as handle:
            readiness_hash = asset_by_path[str(args.pild_readiness.resolve())]["sha256"]
            if readiness_hash and str(handle.attrs.get("source_readiness_sha256", "")) != readiness_hash:
                raise ValueError("PILD optical cache source_readiness_sha256 mismatch")
        marker = json.loads(args.pild_optical_marker.read_text(encoding="utf-8"))
        if marker.get("complete") is not True or int(marker.get("n_samples", -1)) != EXPECTED_PILD_SAMPLES:
            raise ValueError("PILD optical completion marker is not a complete 2937-sample build")
        pild_optical_hash = asset_by_path[str(args.pild_optical_h5.resolve())]["sha256"] or ""

    alias_hash = asset_by_path[str(args.event_aliases.resolve())]["sha256"] or ""
    readiness_hash = asset_by_path[str(args.pild_readiness.resolve())]["sha256"] or ""
    sen12_index_hash = asset_by_path[str(args.sen12_cache_index.resolve())]["sha256"] or ""
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pild.itertuples(index=False)):
        base_path = str(Path(row.h5_path).resolve())
        material_ready = pild_material is not None
        trigger_ready = pild_trigger is not None
        sample_id = str(row.sample_id)
        rows.append(
            {
                "manifest_schema_version": 2,
                "manifest_index": index,
                "dataset_id": str(row.dataset_id),
                "source_id": PILD_SOURCE,
                "source_event_id": str(row.physical_event_id),
                "canonical_event_id": alias_map[(PILD_SOURCE, str(row.physical_event_id))],
                "sample_id": sample_id,
                "source_sample_id": sample_id,
                "base_h5_path": base_path,
                "base_h5_index": int(row.h5_index),
                "base_h5_sha256": asset_by_path[base_path]["sha256"] or "",
                "optical_h5_path": str(args.pild_optical_h5.resolve()),
                "optical_h5_index": index,
                "optical_h5_sha256": pild_optical_hash,
                "terrain_h5_path": base_path,
                "terrain_h5_index": int(row.h5_index),
                "terrain_channel_indices": ";".join(str(value) for value in range(9)),
                "terrain_schema_id": "pild_sen12_common_terrain9_v2",
                "terrain_h5_sha256": asset_by_path[base_path]["sha256"] or "",
                "material_registry_path": str(args.pild_material_registry.resolve()) if material_ready else "",
                "material_registry_index": pild_material_index.get(sample_id, -1),
                "material_registry_sha256": (
                    asset_by_path[str(args.pild_material_registry.resolve())]["sha256"] or ""
                    if material_ready else ""
                ),
                "trigger_registry_path": str(args.pild_trigger_registry.resolve()) if trigger_ready else "",
                "trigger_registry_index": pild_trigger_index.get(sample_id, -1),
                "trigger_registry_sha256": (
                    asset_by_path[str(args.pild_trigger_registry.resolve())]["sha256"] or ""
                    if trigger_ready else ""
                ),
                "source_registry_path": str(args.pild_readiness.resolve()),
                "source_registry_sha256": readiness_hash,
                "event_alias_registry_path": str(args.event_aliases.resolve()),
                "event_alias_registry_sha256": alias_hash,
                "core_assets_ready": int(pild_optical_ready),
                "material_ready": int(material_ready),
                "trigger_ready": int(trigger_ready),
                "full_tmr_assets_ready": int(pild_optical_ready and material_ready and trigger_ready),
            }
        )

    sen_base = str(args.sen12_base_h5.resolve())
    sen_optical = str(args.sen12_optical_h5.resolve())
    sen_terrain = str(args.sen12_terrain_h5.resolve())
    sen_material = str(args.sen12_material_registry.resolve())
    sen_trigger = str(args.sen12_trigger_registry.resolve())
    for offset, row in enumerate(sen12.itertuples(index=False), start=len(rows)):
        sample_id = str(row.sample_id)
        source_event_id = str(row.physical_event_id)
        rows.append(
            {
                "manifest_schema_version": 2,
                "manifest_index": offset,
                "dataset_id": SEN12_DATASET,
                "source_id": SEN12_SOURCE,
                "source_event_id": source_event_id,
                "canonical_event_id": alias_map[(SEN12_SOURCE, source_event_id)],
                "sample_id": sample_id,
                "source_sample_id": str(row.patch_id),
                "base_h5_path": sen_base,
                "base_h5_index": int(row.cache_index),
                "base_h5_sha256": asset_by_path[sen_base]["sha256"] or "",
                "optical_h5_path": sen_optical,
                "optical_h5_index": int(row.cache_index),
                "optical_h5_sha256": asset_by_path[sen_optical]["sha256"] or "",
                "terrain_h5_path": sen_terrain,
                "terrain_h5_index": int(row.cache_index),
                "terrain_channel_indices": ";".join(str(value) for value in SEN12_COMMON_TERRAIN_INDICES),
                "terrain_schema_id": "pild_sen12_common_terrain9_v2",
                "terrain_h5_sha256": asset_by_path[sen_terrain]["sha256"] or "",
                "material_registry_path": sen_material,
                "material_registry_index": sen12_material_index[sample_id],
                "material_registry_sha256": asset_by_path[sen_material]["sha256"] or "",
                "trigger_registry_path": sen_trigger,
                "trigger_registry_index": sen12_trigger_index[sample_id],
                "trigger_registry_sha256": asset_by_path[sen_trigger]["sha256"] or "",
                "source_registry_path": str(args.sen12_cache_index.resolve()),
                "source_registry_sha256": sen12_index_hash,
                "event_alias_registry_path": str(args.event_aliases.resolve()),
                "event_alias_registry_sha256": alias_hash,
                "core_assets_ready": 1,
                "material_ready": 1,
                "trigger_ready": 1,
                "full_tmr_assets_ready": 1,
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    require_unique(manifest, ["sample_id"], "unified manifest")
    if len(manifest) != EXPECTED_PILD_SAMPLES + EXPECTED_SEN12_SAMPLES:
        raise AssertionError("unified manifest row count changed")
    if manifest["canonical_event_id"].nunique() != EXPECTED_CANONICAL_EVENTS:
        raise ValueError("unified sample manifest does not retain all 56 canonical events")

    event_split = build_event_isolated_split(
        manifest, args.event_split_seed, args.val_event_fraction, args.test_event_fraction
    )
    lodo = build_lodo_split(manifest, args.event_split_seed, args.val_event_fraction)
    blockers = []
    if not pild_optical_ready:
        blockers.append(
            f"missing completed PILD temporal cache and marker: {args.pild_optical_h5.resolve()}"
        )
    if pild_material is None:
        blockers.append("missing exact-2937 PILD Material sample registry")
    if pild_trigger is None:
        blockers.append("missing exact-2937 PILD Trigger sample registry")
    core_ready = bool(manifest["core_assets_ready"].astype(bool).all())
    full_ready = bool(manifest["full_tmr_assets_ready"].astype(bool).all())
    summary: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "mode": "write" if args.write else "validate-only",
        "validation_status": "PASS",
        "hash_mode": args.hash_mode,
        "identity_contract": {
            "fields": ["dataset_id", "source_id", "source_event_id", "canonical_event_id", "sample_id"],
            "raw_source_events": EXPECTED_RAW_EVENTS,
            "canonical_events": EXPECTED_CANONICAL_EVENTS,
            "split_key": "canonical_event_id",
        },
        "counts": {
            "samples": len(manifest),
            "pild_samples": int(manifest["source_id"].eq(PILD_SOURCE).sum()),
            "sen12_samples": int(manifest["source_id"].eq(SEN12_SOURCE).sum()),
            "datasets": int(manifest["dataset_id"].nunique()),
            "canonical_events": int(manifest["canonical_event_id"].nunique()),
            "by_dataset": manifest["dataset_id"].value_counts().sort_index().to_dict(),
        },
        "terrain_contract": {
            "schema_id": "pild_sen12_common_terrain9_v2",
            "names": list(COMMON_TERRAIN_NAMES),
            "sen12_native_channel_indices": list(SEN12_COMMON_TERRAIN_INDICES),
        },
        "sampling_contract": "uniform source -> uniform canonical_event within source -> uniform patch within event",
        "readiness": {
            "manifest_ready": True,
            "core_training_ready": core_ready,
            "full_tmr_training_ready": full_ready,
            "training_ready": full_ready,
            "blockers": blockers,
        },
        "assets": assets,
        "split_counts": {
            "event_isolated": event_split["role"].value_counts().to_dict(),
            "leave_one_dataset_out": (
                lodo.groupby(["fold_id", "role"], as_index=False)
                .size()
                .rename(columns={"size": "n_samples"})
                .to_dict("records")
            ),
        },
    }
    return manifest, event_split, lodo, summary


def main() -> int:
    args = parse_args()
    manifest, event_split, lodo, summary = build(args)
    if args.write:
        args.outdir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.outdir / "unified_sample_manifest_v2.csv"
        event_path = args.outdir / "event_isolated_split_v2.csv"
        lodo_path = args.outdir / "leave_one_dataset_out_split_v2.csv"
        assets_path = args.outdir / "asset_inventory_v2.json"
        summary_path = args.outdir / "protocol_summary_v2.json"
        atomic_csv(manifest, manifest_path)
        atomic_csv(event_split, event_path)
        atomic_csv(lodo, lodo_path)
        atomic_json({"schema_version": 2, "assets": summary["assets"]}, assets_path)
        summary["outputs"] = {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
            "event_isolated_split": {"path": str(event_path.resolve()), "sha256": sha256_file(event_path)},
            "leave_one_dataset_out_split": {"path": str(lodo_path.resolve()), "sha256": sha256_file(lodo_path)},
            "asset_inventory": {"path": str(assets_path.resolve()), "sha256": sha256_file(assets_path)},
        }
        atomic_json(summary, summary_path)
    printable = {key: value for key, value in summary.items() if key not in {"assets", "split_counts"}}
    print(json.dumps(printable, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
