#!/usr/bin/env python3
"""Build a deterministic five-fold region-isolated Sen12 S2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


REGION_FOLDS = {
    0: ("italy",),
    1: ("kyrgyzstan1", "kyrgyzstan2", "usa"),
    2: ("newzealand", "hokkaido", "itogon"),
    3: ("dominicamaria", "hiroshima", "china"),
    4: ("chimanimani", "indonesia", "thrissur"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = root / "data_raw/08_Sen12Landslides/upstream_code"
    locations_path = upstream / "tasks/S12LS-LD/harmonized/s2/patch_locations.geojson"
    location_summary_path = (
        root / "physics_informed_landslide_dataset/metadata/pild_xdomain_v1/sen12_location_summary_v1.csv"
    )
    outdir = root / "physics_informed_landslide_dataset/metadata/pild_xdomain_v1"

    patches = gpd.read_file(locations_path).drop(columns="geometry")
    patches["inventory"] = patches["inventory"].str.lower()
    patches = patches.rename(columns={"inventory": "region_group"})
    if patches["patch_id"].duplicated().any():
        raise SystemExit("Duplicate patch_id values in the official S12LS-LD table")
    expected_regions = {region for values in REGION_FOLDS.values() for region in values}
    actual_regions = set(patches["region_group"])
    if actual_regions != expected_regions:
        raise SystemExit(
            f"Region contract changed; missing={sorted(expected_regions - actual_regions)}, "
            f"unexpected={sorted(actual_regions - expected_regions)}"
        )

    region_to_fold = {
        region: fold for fold, regions in REGION_FOLDS.items() for region in regions
    }
    patches["region_fold"] = patches["region_group"].map(region_to_fold).astype(int)
    patches["spatial_supergroup"] = patches["region_group"].replace(
        {"kyrgyzstan1": "kyrgyzstan", "kyrgyzstan2": "kyrgyzstan"}
    )
    patches["source_id"] = "SEN12LS_HARMONIZED"
    patches["sample_id"] = "SEN12_S2_" + patches["patch_id"].astype(str)
    patches["official_random_split"] = patches.pop("split")
    patches["official_random_split_for_evidence"] = 0
    patches["label_contract"] = "S12LS_LD_annotated_pixels_ge_50"
    patches["terrain_source"] = "Copernicus_DEM_resampled_10m"
    patches["terrain_native_resolution_m"] = 30

    protocol_rows = []
    for outer_fold in range(5):
        val_fold = (outer_fold + 1) % 5
        fold = patches.copy()
        fold["outer_fold"] = outer_fold
        fold["role"] = "train"
        fold.loc[fold["region_fold"] == val_fold, "role"] = "val"
        fold.loc[fold["region_fold"] == outer_fold, "role"] = "test"
        fold["role_reason"] = fold["role"].map(
            {
                "train": "different_region_group",
                "val": "heldout_validation_region_group",
                "test": "heldout_test_region_group",
            }
        )
        protocol_rows.append(fold)
    protocol = pd.concat(protocol_rows, ignore_index=True)

    for outer_fold, fold in protocol.groupby("outer_fold"):
        roles_per_region = fold.groupby("spatial_supergroup")["role"].nunique()
        if int(roles_per_region.max()) != 1:
            raise SystemExit(f"Region leakage in fold {outer_fold}")
        if set(fold["role"]) != {"train", "val", "test"}:
            raise SystemExit(f"Missing role in fold {outer_fold}")
        if fold["patch_id"].duplicated().any():
            raise SystemExit(f"Patch duplication in fold {outer_fold}")

    locations = pd.read_csv(location_summary_path)
    locations["location_key"] = locations["location"].str.lower().replace(
        {"usa_puertorico": "usa"}
    )
    region_summary = (
        patches.groupby(["region_fold", "region_group"], as_index=False)
        .agg(
            n_samples=("patch_id", "size"),
            positive_pixels=("annotated_pixels", "sum"),
            center_lon=("lon", "median"),
            center_lat=("lat", "median"),
        )
        .merge(
            locations[
                [
                    "location_key",
                    "event_confidence_median",
                    "event_types",
                    "date_min",
                    "date_max",
                ]
            ],
            left_on="region_group",
            right_on="location_key",
            how="left",
        )
        .drop(columns="location_key")
    )
    region_summary["event_grouping_rule"] = region_summary["event_confidence_median"].apply(
        lambda value: "event_date_when_available" if pd.notna(value) and value >= 0.95 else "region_group"
    )

    protocol_path = outdir / "sen12_s2_logo5_v1.csv"
    region_path = outdir / "sen12_s2_region_groups_v1.csv"
    protocol.to_csv(protocol_path, index=False)
    region_summary.to_csv(region_path, index=False)
    role_summary = (
        protocol.groupby(["outer_fold", "role"], as_index=False)
        .agg(
            n_samples=("patch_id", "size"),
            n_regions=("spatial_supergroup", "nunique"),
            positive_pixels=("annotated_pixels", "sum"),
        )
    )
    summary = {
        "protocol_id": "sen12_s2_logo5_v1",
        "source_revision": "40af2dd6b4e568edb6640d6e14dc67ebd01038a4",
        "upstream_patch_table": str(locations_path),
        "upstream_patch_table_sha256": sha256(locations_path),
        "n_unique_samples": int(patches["patch_id"].nunique()),
        "n_region_groups": int(patches["region_group"].nunique()),
        "n_spatial_supergroups": int(patches["spatial_supergroup"].nunique()),
        "n_outer_folds": 5,
        "split_unit": "region_group; event dates refine attribution after NetCDF indexing",
        "official_random_split_used_for_evidence": False,
        "unannotated_patch_policy": "not treated as segmentation negatives",
        "role_summary": role_summary.to_dict(orient="records"),
    }
    (outdir / "sen12_s2_logo5_summary_v1.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(role_summary.to_string(index=False))
    print(
        f"[DONE] samples={summary['n_unique_samples']} regions={summary['n_region_groups']} "
        f"spatial_supergroups={summary['n_spatial_supergroups']} "
        f"protocol_rows={len(protocol)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
