#!/usr/bin/env python3
"""Build the PILD-XDomain unified sample registry and source-isolated protocol.

This stage registers PILD-Core and Sen12Landslides under one identity and
physical-event contract.  It does not claim tensor compatibility: observation
and Terrain schema differences remain explicit until a common cache is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


DOMAIN_ORDER = (
    "DLR_Landslide_Ref_2025",
    "GDCLD",
    "GLaD4CD_v1",
    "SEN12LS_HARMONIZED",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} misses columns: {sorted(missing)}")


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_event_map(candidates: pd.DataFrame, source_id: str) -> pd.DataFrame:
    selected = candidates[candidates["source_id"] == source_id].copy()
    if selected["source_record_id"].duplicated().any():
        raise RuntimeError(f"Duplicate source_record_id in {source_id} event registry")
    return selected.set_index("source_record_id")


def pild_rows(root: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    registry_path = (
        root
        / "physics_informed_landslide_dataset/processed/hybrid_pinn/"
        "pild_core_geo_v2_1_native30_raw/window_registry_v2.csv"
    )
    frame = pd.read_csv(registry_path, keep_default_na=False)
    require_columns(
        frame,
        {
            "sample_id",
            "event_uid",
            "physical_event_id",
            "dataset_id",
            "h5_path",
            "h5_index",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "bbox_top",
            "valid_fraction",
            "terrain_valid_fraction",
            "positive_pixels",
            "valid_pixels",
        },
        "PILD-Core window registry",
    )
    event_map = load_event_map(candidates, "PILD_CORE_V2")
    missing = sorted(set(frame["physical_event_id"]) - set(event_map.index))
    if missing:
        raise RuntimeError(f"PILD-Core events missing from cross-source registry: {missing[:20]}")
    mapped = event_map.loc[frame["physical_event_id"]]
    out = pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "source_collection": "PILD_CORE_V2",
            "domain_id": frame["dataset_id"],
            "source_sample_id": frame["sample_id"],
            "source_event_id": frame["physical_event_id"],
            "physical_event_cluster_id": mapped["physical_event_cluster_id"].to_numpy(),
            "geographic_region_id": mapped["geographic_region_id"].to_numpy(),
            "event_date_start": mapped["event_date_start"].to_numpy(),
            "event_date_end": mapped["event_date_end"].to_numpy(),
            "date_quality": mapped["date_quality"].to_numpy(),
            "region_group": mapped["geographic_region_id"].to_numpy(),
            "center_lon": (frame["bbox_left"] + frame["bbox_right"]) / 2,
            "center_lat": (frame["bbox_bottom"] + frame["bbox_top"]) / 2,
            "h5_path": frame["h5_path"],
            "h5_index": frame["h5_index"].astype(int),
            "observation_contract": "post_rgb",
            "terrain_contract": "native30_derive_then_resample_roughness90",
            "label_contract": frame["label_profile"],
            "valid_fraction": frame["valid_fraction"],
            "terrain_valid_fraction": frame["terrain_valid_fraction"],
            "positive_pixels": frame["positive_pixels"].astype(int),
            "valid_pixels": frame["valid_pixels"].astype(int),
            "change_view_eligible": 0,
            "source_registry_path": str(registry_path),
        }
    )
    if len(out) != 2_937:
        raise RuntimeError(f"Expected 2,937 PILD-Core windows, found {len(out)}")
    return out


def sen12_rows(root: Path) -> pd.DataFrame:
    metadata = root / "physics_informed_landslide_dataset/metadata/pild_xdomain_v1"
    cache_dir = (
        root
        / "physics_informed_landslide_dataset/processed/hybrid_pinn/sen12_s2_xdomain_v1"
    )
    registry_path = metadata / "sen12_s2_sample_registry_v1.csv"
    cache_index_path = cache_dir / "cache_index_v1.csv"
    registry = pd.read_csv(registry_path, keep_default_na=False).set_index("sample_id")
    cache = pd.read_csv(cache_index_path, keep_default_na=False)
    require_columns(
        cache,
        {
            "cache_index",
            "sample_id",
            "physical_event_id",
            "event_dates",
            "date_quality",
            "region_group",
            "valid_fraction",
            "terrain_valid_fraction",
            "annotated_pixels",
        },
        "Sen12 cache index",
    )
    missing = sorted(set(cache["sample_id"]) - set(registry.index))
    if missing:
        raise RuntimeError(f"Sen12 cache rows missing from sample registry: {missing[:20]}")
    source = registry.loc[cache["sample_id"]]
    event_start = source["event_date_start"].astype(str).to_numpy()
    event_end = source["event_date_end"].astype(str).to_numpy()
    out = pd.DataFrame(
        {
            "sample_id": cache["sample_id"],
            "source_collection": "SEN12LS_HARMONIZED",
            "domain_id": "SEN12LS_HARMONIZED",
            "source_sample_id": cache["patch_id"],
            "source_event_id": source["physical_event_group"].to_numpy(),
            "physical_event_cluster_id": cache["physical_event_id"],
            "geographic_region_id": source["geographic_region_id"].to_numpy(),
            "event_date_start": event_start,
            "event_date_end": event_end,
            "date_quality": cache["date_quality"],
            "region_group": cache["region_group"],
            "center_lon": source["center_lon"].to_numpy(),
            "center_lat": source["center_lat"].to_numpy(),
            "h5_path": str(cache_dir / "sen12_s2_tmr_p128.h5"),
            "h5_index": cache["cache_index"].astype(int),
            "observation_contract": "prepost_rgb_event_span",
            "terrain_contract": "embedded_copdem30_resampled10_roughness30",
            "label_contract": "S12LS_LD_annotated_pixels_ge_50",
            "valid_fraction": cache["valid_fraction"],
            "terrain_valid_fraction": cache["terrain_valid_fraction"],
            "positive_pixels": cache["annotated_pixels"].astype(int),
            "valid_pixels": 128 * 128,
            "change_view_eligible": 1,
            "source_registry_path": str(registry_path),
        }
    )
    if len(out) != 4_979:
        raise RuntimeError(f"Expected 4,979 Sen12 windows, found {len(out)}")
    return out


def build_loso(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, test_domain in enumerate(DOMAIN_ORDER):
        val_domain = DOMAIN_ORDER[(fold + 1) % len(DOMAIN_ORDER)]
        protocol = frame.copy()
        protocol["fold"] = fold
        protocol["fold_id"] = f"pild-xdomain-loso4-holdout-{test_domain.lower()}"
        protocol["test_domain"] = test_domain
        protocol["validation_domain"] = val_domain
        protocol["role"] = "train"
        protocol["role_reason"] = "different_source_and_nonoverlapping_event"
        test_mask = protocol["domain_id"].eq(test_domain)
        protocol.loc[test_mask, ["role", "role_reason"]] = ["test", "heldout_source"]
        test_events = set(protocol.loc[test_mask, "physical_event_cluster_id"])
        test_overlap = (~test_mask) & protocol["physical_event_cluster_id"].isin(test_events)
        protocol.loc[test_overlap, ["role", "role_reason"]] = [
            "excluded",
            "physical_event_overlap_with_test",
        ]
        val_mask = protocol["domain_id"].eq(val_domain) & protocol["role"].eq("train")
        protocol.loc[val_mask, ["role", "role_reason"]] = [
            "val",
            "heldout_validation_source",
        ]
        val_events = set(protocol.loc[val_mask, "physical_event_cluster_id"])
        val_overlap = protocol["role"].eq("train") & protocol["physical_event_cluster_id"].isin(val_events)
        protocol.loc[val_overlap, ["role", "role_reason"]] = [
            "excluded",
            "physical_event_overlap_with_validation",
        ]
        for role in ("train", "val", "test"):
            if not protocol["role"].eq(role).any():
                raise RuntimeError(f"fold={fold} has empty role={role}")
        train_events = set(protocol.loc[protocol["role"] == "train", "physical_event_cluster_id"])
        val_events = set(protocol.loc[protocol["role"] == "val", "physical_event_cluster_id"])
        if train_events & test_events or train_events & val_events or val_events & test_events:
            raise RuntimeError(f"Physical-event leakage remains in fold={fold}")
        rows.append(protocol)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = root / "physics_informed_landslide_dataset/metadata/pild_xdomain_v1"
    candidate_path = outdir / "candidate_event_registry_v1.csv"
    candidates = pd.read_csv(candidate_path, keep_default_na=False)
    pild = pild_rows(root, candidates)
    sen12 = sen12_rows(root)
    unified = pd.concat([pild, sen12], ignore_index=True)
    if unified["sample_id"].duplicated().any():
        duplicates = unified.loc[unified["sample_id"].duplicated(False), "sample_id"].tolist()
        raise RuntimeError(f"Cross-source sample IDs collide: {duplicates[:20]}")
    if set(unified["domain_id"]) != set(DOMAIN_ORDER):
        raise RuntimeError(f"Domain contract changed: {sorted(unified['domain_id'].unique())}")
    protocol = build_loso(unified)
    registry_path = outdir / "pild_xdomain_unified_samples_v1.csv"
    protocol_path = outdir / "pild_xdomain_loso4_v1.csv"
    unified.to_csv(registry_path, index=False)
    protocol.to_csv(protocol_path, index=False)
    role_summary = (
        protocol.groupby(["fold", "test_domain", "validation_domain", "role"], as_index=False)
        .agg(
            n_samples=("sample_id", "size"),
            n_domains=("domain_id", "nunique"),
            n_event_clusters=("physical_event_cluster_id", "nunique"),
        )
    )
    summary = {
        "protocol_id": "pild_xdomain_loso4_v1",
        "n_samples": int(len(unified)),
        "n_pild_core_samples": int(len(pild)),
        "n_sen12_samples": int(len(sen12)),
        "n_domains": int(unified["domain_id"].nunique()),
        "n_physical_event_clusters": int(unified["physical_event_cluster_id"].nunique()),
        "domain_counts": unified["domain_id"].value_counts().sort_index().to_dict(),
        "observation_contracts": unified["observation_contract"].value_counts().to_dict(),
        "terrain_contracts": unified["terrain_contract"].value_counts().to_dict(),
        "pooled_tensor_training_ready": False,
        "pooled_tensor_blocker": (
            "PILD-Core is post-only with roughness90; Sen12 is pre/post with roughness30. "
            "Build the registered common post-RGB/native30-Terrain cache before pooled training."
        ),
        "event_overlap_policy": (
            "all non-test samples sharing a physical_event_cluster_id with test are excluded; "
            "training samples sharing an event cluster with validation are also excluded"
        ),
        "role_summary": role_summary.to_dict(orient="records"),
        "input_hashes": {
            "candidate_event_registry": sha256(candidate_path),
            "pild_window_registry": sha256(
                root
                / "physics_informed_landslide_dataset/processed/hybrid_pinn/"
                "pild_core_geo_v2_1_native30_raw/window_registry_v2.csv"
            ),
            "sen12_sample_registry": sha256(outdir / "sen12_s2_sample_registry_v1.csv"),
            "sen12_cache_index": sha256(
                root
                / "physics_informed_landslide_dataset/processed/hybrid_pinn/"
                "sen12_s2_xdomain_v1/cache_index_v1.csv"
            ),
        },
    }
    summary_path = outdir / "pild_xdomain_loso4_summary_v1.json"
    json_dump(summary_path, summary)
    print(role_summary.to_string(index=False))
    print(
        f"[DONE] samples={len(unified)} event_clusters={summary['n_physical_event_clusters']} "
        f"registry={registry_path} protocol={protocol_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
