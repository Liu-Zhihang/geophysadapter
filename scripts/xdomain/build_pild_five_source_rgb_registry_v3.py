#!/usr/bin/env python3
"""Build the audited five-source PILD RGB training registry.

This v3 registry preserves the frozen four-source v2 registry and appends the
CAS supervised pool through a common post-event RGB contract. CAS is not forced
into the four-time, six-band Prithvi contract. Its patch-level Terrain support
is explicitly unavailable (q_T=0), while region-level Material and earthquake
Trigger context are linked from ``cas_context_v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 3
EXPECTED_EXISTING_SAMPLES = 7_916
EXPECTED_CAS_SAMPLES = 11_091
EXPECTED_TOTAL_SAMPLES = 19_007
EXPECTED_RAW_SOURCE_EVENTS = 63
EXPECTED_CANONICAL_EVENTS = 59

CAS_SPLIT_BY_PHYSICAL_EVENT = {
    "PEV1_2017-08-08_ee541513dd8d": "train",  # Jiuzhai Valley
    "PEV1_2018-08-05_973f44bd607e": "val",  # Lombok
    "PEV1_2018-09-06_d03700654134": "train",  # Hokkaido
    "PEV1_2021-08-14_b7aa7e7ce192": "test",  # Tiburon/Haiti
}

CAS_EXISTING_ALIAS_BY_EVENT_UID = {
    # Same 2018-09-06 Hokkaido earthquake and overlapping study area.
    "CAS_Hokkaido": ("Sen12Landslides", "XEV_c66e4983b74a4d"),
}

CAS_CACHE_PATHS = (
    "processed/hybrid_pinn/strict_t2_postrgb_train_cache_v2_skiperr/train_postrgb_p128.h5",
    "processed/hybrid_pinn/strict_t2_postrgb_eval_cache_v2/val_postrgb_p128.h5",
    "processed/hybrid_pinn/strict_t2_postrgb_eval_cache_v2/test_postrgb_p128.h5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--v2-manifest",
        type=Path,
        default=Path("metadata/pild_sen12_training_v2/unified_sample_manifest_v2.csv"),
    )
    parser.add_argument(
        "--v2-split",
        type=Path,
        default=Path("metadata/pild_sen12_training_v2/event_isolated_split_v2.csv"),
    )
    parser.add_argument(
        "--v2-aliases",
        type=Path,
        default=Path(
            "processed/hybrid_pinn/pild_prithvi_integration_v1/"
            "pild_sen12_event_aliases_v1.csv"
        ),
    )
    parser.add_argument(
        "--cas-sample-manifest",
        type=Path,
        default=Path(
            "processed/hybrid_pinn/"
            "strict_t2_supervised_ready_v4_roleaware_posttrain_qc/"
            "sample_manifest_post_rgb_v4_roleaware_posttrain_qc.csv"
        ),
    )
    parser.add_argument(
        "--cas-context-dir",
        type=Path,
        default=Path("processed/hybrid_pinn/cas_context_v1"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/pild_five_source_rgb_v3"),
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_existing_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if len(frame) != EXPECTED_EXISTING_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXISTING_SAMPLES:,} frozen v2 rows, found {len(frame):,}"
        )
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("Frozen v2 manifest sample_id is not unique")
    return frame


def load_cas_samples(path: Path) -> pd.DataFrame:
    columns = ("event_uid", "dataset_id", "event_date", "sample_id", "asset_status")
    frame = pd.read_csv(path, usecols=list(columns), keep_default_na=False)
    frame = frame.loc[frame["dataset_id"].eq("CAS_Landslide")].copy()
    if len(frame) != EXPECTED_CAS_SAMPLES or frame["sample_id"].duplicated().any():
        raise RuntimeError("CAS sample manifest must contain 11,091 unique samples")
    if not frame["asset_status"].eq("supervised_ready").all():
        raise RuntimeError("CAS registry contains samples outside supervised_ready")
    return frame


def h5_sample_mapping(
    root: Path,
    expected_ids: set[str],
) -> tuple[dict[str, tuple[str, int]], list[dict[str, Any]]]:
    mapping: dict[str, tuple[str, int]] = {}
    inventory: list[dict[str, Any]] = []
    for relative in CAS_CACHE_PATHS:
        path = (root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            required = {"sample_id", "dataset_id", "image", "mask", "valid"}
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise RuntimeError(f"{path} missing HDF5 datasets: {missing}")
            sample_ids = decode(handle["sample_id"][:])
            dataset_ids = decode(handle["dataset_id"][:])
            cas_indices = [
                index for index, name in enumerate(dataset_ids) if name == "CAS_Landslide"
            ]
            cas_ids = [sample_ids[index] for index in cas_indices]
            eligible = [
                (sample_id, index)
                for sample_id, index in zip(cas_ids, cas_indices)
                if sample_id in expected_ids
            ]
            for sample_id, index in eligible:
                if sample_id in mapping:
                    raise RuntimeError(f"CAS sample appears in multiple caches: {sample_id}")
                mapping[sample_id] = (str(path), int(index))
            inventory.append(
                {
                    "role": "cas_rgb_cache",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "n_rows": len(sample_ids),
                    "n_cas_rows": len(cas_ids),
                    "n_qc_eligible_cas_rows": len(eligible),
                    "cas_sample_identity_sha256": identity_sha256(cas_ids),
                    "qc_eligible_cas_sample_identity_sha256": identity_sha256(
                        [sample_id for sample_id, _ in eligible]
                    ),
                    "datasets": "sample_id;dataset_id;image;mask;valid",
                }
            )
    if len(mapping) != EXPECTED_CAS_SAMPLES:
        raise RuntimeError(f"CAS HDF5 caches cover {len(mapping):,}, expected 11,091")
    return mapping, inventory


def context_index(path: Path, expected_ids: set[str], quality_column: str) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    if frame["sample_id"].duplicated().any() or set(frame["sample_id"]) != expected_ids:
        raise RuntimeError(f"Context registry identity mismatch: {path}")
    quality = pd.to_numeric(frame[quality_column], errors="raise")
    if not ((quality >= 0) & (quality <= 1)).all():
        raise RuntimeError(f"{quality_column} outside [0, 1]: {path}")
    return frame, {sample_id: index for index, sample_id in enumerate(frame["sample_id"])}


def canonical_id(physical_event_id: str) -> str:
    digest = hashlib.sha256(physical_event_id.encode("utf-8")).hexdigest()[:16]
    return f"CEV3_CAS_{digest}"


def build_alias_registry(
    old_aliases: pd.DataFrame,
    existing: pd.DataFrame,
    cas_samples: pd.DataFrame,
    cas_events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    existing_physical_to_canonical = (
        existing.loc[:, ["source_event_id", "canonical_event_id"]]
        .drop_duplicates()
        .set_index("source_event_id")["canonical_event_id"]
        .to_dict()
    )
    old_alias_lookup = old_aliases.set_index(
        ["source_collection", "source_event_id"], drop=False
    )
    physical_by_event = cas_events.set_index("event_uid")["physical_event_id"].to_dict()
    canonical_by_physical: dict[str, str] = {}
    for physical_event_id in sorted(set(physical_by_event.values())):
        matching_event_uids = [
            event_uid
            for event_uid, candidate_physical in physical_by_event.items()
            if candidate_physical == physical_event_id
        ]
        override_targets = {
            str(old_alias_lookup.loc[key, "canonical_physical_event_id"])
            for event_uid in matching_event_uids
            if (key := CAS_EXISTING_ALIAS_BY_EVENT_UID.get(event_uid)) is not None
        }
        if len(override_targets) > 1:
            raise RuntimeError(f"Conflicting CAS alias overrides for {physical_event_id}")
        canonical_by_physical[physical_event_id] = (
            next(iter(override_targets))
            if override_targets
            else existing_physical_to_canonical.get(
                physical_event_id, canonical_id(physical_event_id)
            )
        )

    rows: list[dict[str, Any]] = []
    for event in cas_events.itertuples(index=False):
        physical_event_id = str(event.physical_event_id)
        target = canonical_by_physical[physical_event_id]
        explicit_key = CAS_EXISTING_ALIAS_BY_EVENT_UID.get(str(event.event_uid))
        existing_match = (
            physical_event_id in existing_physical_to_canonical
            or explicit_key is not None
        )
        if explicit_key is not None:
            candidate = old_alias_lookup.loc[explicit_key]
            candidate_collection = str(candidate.source_collection)
            candidate_event_id = str(candidate.source_event_id)
            candidate_event_names = str(candidate.source_event_names)
            candidate_event_date = str(candidate.source_event_date)
            candidate_date_basis = str(candidate.source_date_basis)
            decision_evidence = (
                "same dated Hokkaido earthquake; study-area centers are 4.07 km apart"
            )
        elif existing_match:
            candidate_collection = "PILD"
            candidate_event_id = physical_event_id
            candidate_event_names = physical_event_id
            candidate_event_date = str(event.event_date)
            candidate_date_basis = "physical_event_registry_v2"
            decision_evidence = "same physical_event_id as an existing PILD event"
        else:
            candidate_collection = ""
            candidate_event_id = ""
            candidate_event_names = ""
            candidate_event_date = ""
            candidate_date_basis = ""
            decision_evidence = (
                "CAS source units share the frozen physical_event_id only when "
                "they observe the same dated earthquake"
            )
        rows.append(
            {
                "source_collection": "CAS_Landslide",
                "source_event_id": str(event.event_uid),
                "source_dataset_ids": "CAS_Landslide",
                "source_event_names": str(event.event_uid),
                "source_event_date": str(event.event_date),
                "source_date_basis": "published_event_registry",
                "source_date_reliable": 1,
                "source_center_lon": float(event.center_lon),
                "source_center_lat": float(event.center_lat),
                "source_bbox_left": float(event.bbox_left),
                "source_bbox_bottom": float(event.bbox_bottom),
                "source_bbox_right": float(event.bbox_right),
                "source_bbox_top": float(event.bbox_top),
                "source_n_samples": int(
                    cas_samples["event_uid"].eq(str(event.event_uid)).sum()
                ),
                "candidate_collection": candidate_collection,
                "candidate_event_id": candidate_event_id,
                "candidate_event_names": candidate_event_names,
                "candidate_event_date": candidate_event_date,
                "candidate_date_basis": candidate_date_basis,
                "primary_date_delta_days": 0 if existing_match else "",
                "nearest_listed_date_delta_days": 0 if existing_match else "",
                "center_distance_km": "",
                "bbox_overlap": "",
                "alias_decision": (
                    "registry-match"
                    if existing_match
                    else "registry-physical-event-deduplication"
                ),
                "decision_evidence": decision_evidence,
                "canonical_physical_event_id": target,
                "split_group_id": target,
            }
        )
    additions = pd.DataFrame(rows).loc[:, old_aliases.columns]
    aliases = pd.concat([old_aliases, additions], ignore_index=True)
    if len(aliases) != EXPECTED_RAW_SOURCE_EVENTS:
        raise RuntimeError(f"Expected 63 alias rows, found {len(aliases)}")
    if aliases[["source_collection", "source_event_id"]].duplicated().any():
        raise RuntimeError("Five-source alias identity is not unique")
    if aliases["canonical_physical_event_id"].nunique() != EXPECTED_CANONICAL_EVENTS:
        raise RuntimeError(
            "Five-source alias registry must collapse 63 source events to 59 canonical events"
        )
    if not aliases["canonical_physical_event_id"].eq(aliases["split_group_id"]).all():
        raise RuntimeError("split_group_id drifted from canonical event identity")
    canonical_by_source_event = {
        event_uid: canonical_by_physical[physical_event_id]
        for event_uid, physical_event_id in physical_by_event.items()
    }
    return aliases, canonical_by_source_event


def augment_existing_manifest(
    frame: pd.DataFrame,
    aliases_path: Path,
    aliases_sha256: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["manifest_schema_version"] = SCHEMA_VERSION
    output["rgb_h5_path"] = output["base_h5_path"]
    output["rgb_h5_index"] = output["base_h5_index"]
    pild = output["source_id"].eq("PILD")
    output["rgb_dataset_key"] = np.where(pild, "visual", "obs")
    output["rgb_channel_indices"] = np.where(pild, "0;1;2", "3;4;5")
    output["mask_dataset_key"] = "mask"
    output["valid_dataset_key"] = "valid_mask"
    output["terrain_dataset_key"] = "terrain"
    output["terrain_valid_dataset_key"] = np.where(pild, "terrain_valid", "")
    output["q_T_asset"] = 1.0
    output["terrain_support_reason"] = "audited_patch_level_terrain_available"
    output["q_M_asset"] = 1.0
    output["q_R_asset"] = 1.0
    output["role_assets_ready"] = 1
    output["material_support_scale"] = "sample_footprint_context"
    output["material_independence_unit"] = "native_material_cell_or_event_cluster"
    output["trigger_support_scale"] = "canonical_event_context"
    output["trigger_independence_unit"] = "canonical_event_id"
    output["context_broadcast_to_patch"] = 1
    output["patch_level_MR_independence"] = 0
    output["event_alias_registry_path"] = str(aliases_path)
    output["event_alias_registry_sha256"] = aliases_sha256
    return output


def build_cas_manifest(
    samples: pd.DataFrame,
    canonical_by_event: dict[str, str],
    physical_by_event: dict[str, str],
    cache_mapping: dict[str, tuple[str, int]],
    material_path: Path,
    material_frame: pd.DataFrame,
    material_indices: dict[str, int],
    trigger_path: Path,
    trigger_frame: pd.DataFrame,
    trigger_indices: dict[str, int],
    source_registry_path: Path,
    aliases_path: Path,
    aliases_sha256: str,
    template_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    material_sha = sha256(material_path)
    trigger_sha = sha256(trigger_path)
    source_sha = sha256(source_registry_path)
    for sample in samples.itertuples(index=False):
        rgb_path, rgb_index = cache_mapping[str(sample.sample_id)]
        q_m = float(material_frame.iloc[material_indices[str(sample.sample_id)]]["q_M"])
        q_r = float(trigger_frame.iloc[trigger_indices[str(sample.sample_id)]]["q_R"])
        row = {column: "" for column in template_columns}
        row.update(
            {
                "manifest_schema_version": SCHEMA_VERSION,
                "dataset_id": "CAS_Landslide",
                "source_id": "CAS_Landslide",
                "source_event_id": str(sample.event_uid),
                "canonical_event_id": canonical_by_event[str(sample.event_uid)],
                "sample_id": str(sample.sample_id),
                "source_sample_id": str(sample.sample_id),
                "base_h5_path": rgb_path,
                "base_h5_index": rgb_index,
                "base_h5_sha256": "",
                "optical_h5_path": rgb_path,
                "optical_h5_index": rgb_index,
                "optical_h5_sha256": "",
                "terrain_h5_path": "",
                "terrain_h5_index": -1,
                "terrain_channel_indices": "",
                "terrain_schema_id": "terrain_abstained_no_patch_georef",
                "terrain_h5_sha256": "",
                "material_registry_path": str(material_path),
                "material_registry_index": material_indices[str(sample.sample_id)],
                "material_registry_sha256": material_sha,
                "trigger_registry_path": str(trigger_path),
                "trigger_registry_index": trigger_indices[str(sample.sample_id)],
                "trigger_registry_sha256": trigger_sha,
                "source_registry_path": str(source_registry_path),
                "source_registry_sha256": source_sha,
                "event_alias_registry_path": str(aliases_path),
                "event_alias_registry_sha256": aliases_sha256,
                "core_assets_ready": 1,
                "material_ready": int(q_m > 0),
                "trigger_ready": int(q_r > 0),
                "full_tmr_assets_ready": 0,
                "rgb_h5_path": rgb_path,
                "rgb_h5_index": rgb_index,
                "rgb_dataset_key": "image",
                "rgb_channel_indices": "0;1;2",
                "mask_dataset_key": "mask",
                "valid_dataset_key": "valid",
                "terrain_dataset_key": "",
                "terrain_valid_dataset_key": "",
                "q_T_asset": 0.0,
                "terrain_support_reason": (
                    "CAS patch TIFFs lack an audited patch-to-map transform; "
                    "Terrain expert must abstain"
                ),
                "q_M_asset": q_m,
                "q_R_asset": q_r,
                "role_assets_ready": int(q_m > 0 and q_r > 0),
                "physical_event_id": physical_by_event[str(sample.event_uid)],
                "material_support_scale": "source_event_study_area_context",
                "material_independence_unit": "source_event_id",
                "trigger_support_scale": "physical_event_shakemap_context",
                "trigger_independence_unit": "physical_event_id",
                "context_broadcast_to_patch": 1,
                "patch_level_MR_independence": 0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_event_split(
    manifest: pd.DataFrame,
    old_split: pd.DataFrame,
    physical_by_event: dict[str, str],
    canonical_by_event: dict[str, str],
) -> pd.DataFrame:
    old_roles = (
        old_split.loc[:, ["canonical_event_id", "role"]]
        .drop_duplicates()
        .set_index("canonical_event_id")["role"]
        .to_dict()
    )
    roles = dict(old_roles)
    for event_uid, canonical in canonical_by_event.items():
        physical = physical_by_event[event_uid]
        if canonical in roles:
            continue
        if physical not in CAS_SPLIT_BY_PHYSICAL_EVENT:
            raise RuntimeError(f"No prespecified CAS split for {physical}")
        roles[canonical] = CAS_SPLIT_BY_PHYSICAL_EVENT[physical]
    output = manifest.loc[
        :, ["sample_id", "dataset_id", "source_id", "canonical_event_id"]
    ].copy()
    output.insert(0, "fold_id", "event_isolated")
    output.insert(0, "protocol_id", "pild_five_source_rgb_event_isolated_v3")
    output["heldout_dataset_id"] = ""
    output["role"] = output["canonical_event_id"].map(roles)
    output["role_reason"] = "canonical_event_partition_preserving_v2_plus_prespecified_CAS"
    if output["role"].isna().any() or not set(output["role"]).issubset({"train", "val", "test"}):
        raise RuntimeError("Event-isolated split contains undefined roles")
    if output.groupby("canonical_event_id")["role"].nunique().max() != 1:
        raise RuntimeError("A canonical physical event crosses event-isolated roles")
    return output


def build_lodo_split(manifest: pd.DataFrame, event_split: pd.DataFrame) -> pd.DataFrame:
    base_role = event_split.set_index("sample_id")["role"].to_dict()
    rows: list[pd.DataFrame] = []
    for heldout in sorted(manifest["dataset_id"].unique()):
        heldout_events = set(
            manifest.loc[manifest["dataset_id"].eq(heldout), "canonical_event_id"]
        )
        fold = manifest.loc[
            :, ["sample_id", "dataset_id", "source_id", "canonical_event_id"]
        ].copy()
        fold.insert(0, "fold_id", f"lodo_{heldout}")
        fold.insert(0, "protocol_id", "pild_five_source_rgb_lodo_v3")
        fold["heldout_dataset_id"] = heldout
        fold["role"] = [
            "test"
            if canonical in heldout_events
            else ("val" if base_role[sample_id] == "val" else "train")
            for sample_id, canonical in zip(fold["sample_id"], fold["canonical_event_id"])
        ]
        fold["role_reason"] = np.where(
            fold["canonical_event_id"].isin(heldout_events),
            "heldout_dataset_or_cross_source_alias_event",
            "development_event_partition",
        )
        if fold.groupby("canonical_event_id")["role"].nunique().max() != 1:
            raise RuntimeError(f"Canonical event leakage in LODO fold {heldout}")
        rows.append(fold)
    return pd.concat(rows, ignore_index=True)


def validate_manifest(manifest: pd.DataFrame) -> None:
    if len(manifest) != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_SAMPLES:,} rows, found {len(manifest):,}")
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("Five-source manifest sample_id is not unique")
    expected_sources = {
        "DLR_Landslide_Ref_2025",
        "GDCLD",
        "GLaD4CD_v1",
        "SEN12LS_HARMONIZED",
        "CAS_Landslide",
    }
    if set(manifest["dataset_id"]) != expected_sources:
        raise RuntimeError(f"Five-source dataset set changed: {set(manifest['dataset_id'])}")
    if manifest["canonical_event_id"].nunique() != EXPECTED_CANONICAL_EVENTS:
        raise RuntimeError("Five-source manifest must contain 59 canonical physical events")
    cas = manifest["dataset_id"].eq("CAS_Landslide")
    if not manifest.loc[cas, "q_T_asset"].eq(0).all():
        raise RuntimeError("CAS rows must abstain from patch-level Terrain")
    if not manifest.loc[cas, "terrain_h5_path"].eq("").all():
        raise RuntimeError("CAS rows must not reference a fabricated Terrain cache")
    if not manifest.loc[cas, "role_assets_ready"].eq(1).all():
        raise RuntimeError("CAS Material/Trigger context is incomplete")
    if not manifest.loc[cas, "patch_level_MR_independence"].eq(0).all():
        raise RuntimeError("CAS M/R context must not be treated as patch-independent evidence")
    if not manifest.loc[cas, "material_independence_unit"].eq("source_event_id").all():
        raise RuntimeError("CAS Material inference unit must remain source_event_id")
    if not manifest.loc[cas, "trigger_independence_unit"].eq("physical_event_id").all():
        raise RuntimeError("CAS Trigger inference unit must remain physical_event_id")
    if not manifest.loc[~cas, "q_T_asset"].eq(1).all():
        raise RuntimeError("Existing four-source Terrain availability drifted")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    outdir = resolve(root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    v2_manifest_path = resolve(root, args.v2_manifest)
    v2_split_path = resolve(root, args.v2_split)
    v2_aliases_path = resolve(root, args.v2_aliases)
    cas_sample_path = resolve(root, args.cas_sample_manifest)
    cas_context_dir = resolve(root, args.cas_context_dir)

    existing = load_existing_manifest(v2_manifest_path)
    old_split = pd.read_csv(v2_split_path, keep_default_na=False)
    old_aliases = pd.read_csv(v2_aliases_path, keep_default_na=False)
    cas_samples = load_cas_samples(cas_sample_path)
    cas_events_path = cas_context_dir / "cas_source_event_registry_v1.csv"
    cas_events = pd.read_csv(cas_events_path, keep_default_na=False)
    physical_by_event = cas_events.set_index("event_uid")["physical_event_id"].to_dict()

    aliases, canonical_by_event = build_alias_registry(
        old_aliases, existing, cas_samples, cas_events
    )
    aliases_path = outdir / "event_alias_registry_v3.csv"
    atomic_csv(aliases, aliases_path)
    aliases_hash = sha256(aliases_path)

    cas_ids = set(cas_samples["sample_id"])
    cache_mapping, cache_inventory = h5_sample_mapping(root, cas_ids)
    if set(cache_mapping) != cas_ids:
        raise RuntimeError("CAS cache/sample registry identity mismatch")
    material_path = cas_context_dir / "cas_material_sample_registry_v1.csv"
    trigger_path = cas_context_dir / "cas_trigger_sample_registry_v1.csv"
    material_frame, material_indices = context_index(material_path, cas_ids, "q_M")
    trigger_frame, trigger_indices = context_index(trigger_path, cas_ids, "q_R")

    extra_columns = [
        "rgb_h5_path",
        "rgb_h5_index",
        "rgb_dataset_key",
        "rgb_channel_indices",
        "mask_dataset_key",
        "valid_dataset_key",
        "terrain_dataset_key",
        "terrain_valid_dataset_key",
        "q_T_asset",
        "terrain_support_reason",
        "q_M_asset",
        "q_R_asset",
        "role_assets_ready",
        "physical_event_id",
        "material_support_scale",
        "material_independence_unit",
        "trigger_support_scale",
        "trigger_independence_unit",
        "context_broadcast_to_patch",
        "patch_level_MR_independence",
    ]
    template_columns = list(existing.columns) + [
        column for column in extra_columns if column not in existing.columns
    ]
    existing_v3 = augment_existing_manifest(existing, aliases_path, aliases_hash)
    if "physical_event_id" not in existing_v3:
        existing_v3["physical_event_id"] = existing_v3["source_event_id"]
    cas_v3 = build_cas_manifest(
        cas_samples,
        canonical_by_event,
        physical_by_event,
        cache_mapping,
        material_path,
        material_frame,
        material_indices,
        trigger_path,
        trigger_frame,
        trigger_indices,
        cas_sample_path,
        aliases_path,
        aliases_hash,
        template_columns,
    )
    manifest = pd.concat(
        [
            existing_v3.reindex(columns=template_columns),
            cas_v3.reindex(columns=template_columns),
        ],
        ignore_index=True,
    )
    manifest["manifest_index"] = np.arange(len(manifest), dtype=np.int64)
    validate_manifest(manifest)

    event_split = build_event_split(
        manifest, old_split, physical_by_event, canonical_by_event
    )
    lodo_split = build_lodo_split(manifest, event_split)
    manifest_path = outdir / "unified_rgb_manifest_v3.csv"
    event_split_path = outdir / "event_isolated_split_v3.csv"
    lodo_split_path = outdir / "leave_one_dataset_out_split_v3.csv"
    atomic_csv(manifest, manifest_path)
    atomic_csv(event_split, event_split_path)
    atomic_csv(lodo_split, lodo_split_path)

    inventory = {
        "created_utc": utc_now(),
        "large_h5_hash_policy": (
            "Frozen v2 hashes are retained row-wise; CAS HDF5 integrity is registered "
            "by byte size, mtime, dataset schema, and ordered CAS sample identity hash."
        ),
        "cas_caches": cache_inventory,
        "registry_inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (
                v2_manifest_path,
                v2_split_path,
                v2_aliases_path,
                cas_sample_path,
                cas_events_path,
                material_path,
                trigger_path,
            )
        ],
    }
    inventory_path = outdir / "asset_inventory_v3.json"
    atomic_json(inventory, inventory_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "n_samples": len(manifest),
        "n_raw_source_events": len(aliases),
        "n_canonical_events": manifest["canonical_event_id"].nunique(),
        "dataset_sample_counts": manifest.groupby("dataset_id").size().to_dict(),
        "dataset_canonical_event_counts": (
            manifest.groupby("dataset_id")["canonical_event_id"].nunique().to_dict()
        ),
        "event_split_sample_counts": event_split.groupby("role").size().to_dict(),
        "event_split_event_counts": (
            event_split.drop_duplicates("canonical_event_id").groupby("role").size().to_dict()
        ),
        "cas_contract": {
            "rgb_samples": int(manifest["dataset_id"].eq("CAS_Landslide").sum()),
            "source_events": int(
                manifest.loc[
                    manifest["dataset_id"].eq("CAS_Landslide"), "source_event_id"
                ].nunique()
            ),
            "physical_events": int(
                manifest.loc[
                    manifest["dataset_id"].eq("CAS_Landslide"), "physical_event_id"
                ].nunique()
            ),
            "q_T_asset_count": int(
                manifest.loc[
                    manifest["dataset_id"].eq("CAS_Landslide"), "q_T_asset"
                ].sum()
            ),
            "q_M_asset_positive_count": int(
                manifest.loc[
                    manifest["dataset_id"].eq("CAS_Landslide"), "q_M_asset"
                ].gt(0).sum()
            ),
            "q_R_asset_positive_count": int(
                manifest.loc[
                    manifest["dataset_id"].eq("CAS_Landslide"), "q_R_asset"
                ].gt(0).sum()
            ),
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "event_aliases": str(aliases_path),
            "event_split": str(event_split_path),
            "lodo_split": str(lodo_split_path),
            "asset_inventory": str(inventory_path),
        },
    }
    summary_path = outdir / "protocol_summary_v3.json"
    atomic_json(summary, summary_path)
    atomic_json(
        {
            "status": "complete",
            "created_utc": utc_now(),
            "summary_path": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "manifest_sha256": sha256(manifest_path),
        },
        outdir / "DONE.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
