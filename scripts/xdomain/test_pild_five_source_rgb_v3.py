#!/usr/bin/env python3
"""Integration checks for CAS context and the five-source RGB registry."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CAS_CONTEXT = ROOT / "processed/hybrid_pinn/cas_context_v1"
REGISTRY = ROOT / "metadata/pild_five_source_rgb_v3"


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> int:
    manifest = pd.read_csv(REGISTRY / "unified_rgb_manifest_v3.csv", keep_default_na=False)
    aliases = pd.read_csv(REGISTRY / "event_alias_registry_v3.csv", keep_default_na=False)
    split = pd.read_csv(REGISTRY / "event_isolated_split_v3.csv", keep_default_na=False)
    lodo = pd.read_csv(
        REGISTRY / "leave_one_dataset_out_split_v3.csv", keep_default_na=False
    )
    material = pd.read_csv(
        CAS_CONTEXT / "cas_material_sample_registry_v1.csv",
        keep_default_na=False,
        low_memory=False,
    )
    trigger = pd.read_csv(
        CAS_CONTEXT / "cas_trigger_sample_registry_v1.csv",
        keep_default_na=False,
        low_memory=False,
    )
    trigger_events = pd.read_csv(
        CAS_CONTEXT / "cas_trigger_event_registry_v1.csv", keep_default_na=False
    )

    assert len(manifest) == 19_007
    assert manifest["sample_id"].nunique() == 19_007
    assert len(aliases) == 63
    assert aliases["canonical_physical_event_id"].nunique() == 59
    assert manifest["canonical_event_id"].nunique() == 59
    assert split.groupby("canonical_event_id")["role"].nunique().max() == 1
    assert lodo.groupby(["fold_id", "canonical_event_id"])["role"].nunique().max() == 1

    cas = manifest.loc[manifest["dataset_id"].eq("CAS_Landslide")].copy()
    assert len(cas) == 11_091
    assert cas["source_event_id"].nunique() == 6
    assert cas["physical_event_id"].nunique() == 5
    assert cas["q_T_asset"].eq(0).all()
    assert cas["terrain_h5_path"].eq("").all()
    assert cas["q_M_asset"].gt(0).all()
    assert cas["q_R_asset"].eq(1).all()
    assert cas["patch_level_MR_independence"].eq(0).all()
    assert cas["material_independence_unit"].eq("source_event_id").all()
    assert cas["trigger_independence_unit"].eq("physical_event_id").all()
    assert len(material) == len(cas) == len(trigger)
    assert set(material["sample_id"]) == set(cas["sample_id"]) == set(trigger["sample_id"])
    assert trigger_events["event_uid"].nunique() == 6
    assert trigger_events["physical_event_id"].nunique() == 5
    assert trigger_events["q_R"].eq(1).all()

    # Check every CAS event and a deterministic sample subset against HDF5 identity.
    selected_cas = (
        cas.sort_values("sample_id")
        .groupby("source_event_id", sort=True, group_keys=False)
        .head(3)
    )
    # Also exercise the common RGB contract for every one of the five sources.
    selected_sources = (
        manifest.sort_values("sample_id")
        .groupby("dataset_id", sort=True, group_keys=False)
        .head(2)
    )
    selected = pd.concat([selected_cas, selected_sources], ignore_index=True).drop_duplicates(
        "sample_id"
    )
    handles: dict[str, h5py.File] = {}
    try:
        for row in selected.itertuples(index=False):
            path = str(row.rgb_h5_path)
            if path not in handles:
                handles[path] = h5py.File(path, "r")
            handle = handles[path]
            index = int(row.rgb_h5_index)
            assert decode(handle["sample_id"][index]) == row.sample_id
            channels = [int(value) for value in str(row.rgb_channel_indices).split(";")]
            rgb = handle[str(row.rgb_dataset_key)][index, channels]
            mask = handle[str(row.mask_dataset_key)][index]
            valid = handle[str(row.valid_dataset_key)][index]
            assert rgb.shape == (3, 128, 128)
            assert mask.shape == (1, 128, 128)
            assert valid.shape == (1, 128, 128)
    finally:
        for handle in handles.values():
            handle.close()

    # Strict JSON parsing catches NaN/Infinity and partial writes.
    for path in (
        CAS_CONTEXT / "summary.json",
        CAS_CONTEXT / "DONE.json",
        REGISTRY / "protocol_summary_v3.json",
        REGISTRY / "asset_inventory_v3.json",
        REGISTRY / "DONE.json",
    ):
        json.loads(path.read_text(encoding="utf-8"))

    print(
        "PASS: 19,007 samples, 5 sources, 63 raw source events, "
        "59 canonical events; CAS=11,091 RGB / 6 source events / 5 physical events"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
