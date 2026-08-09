#!/usr/bin/env python3
"""Build fold-specific, label-independent DLR Terrain support eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_RUNS = Path(
    "experiments/revision2026/"
    "dlr_geo4qc_sen12_exactcommon9_scratch_seed20260724_v4"
)
DEFAULT_ELIGIBILITY = Path(
    "metadata/pild_sen12_training_v2/support_eligibility_v1/"
    "sample_eligibility_v1.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--eligibility-csv", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/protocol_assets/dlr_terrain_support_gate_v1"),
    )
    parser.add_argument("--minimum-slope-std-deg", type=float, default=5.0)
    parser.add_argument("--minimum-relief-std-m", type=float, default=20.0)
    parser.add_argument(
        "--visual-margin-train-quantile",
        type=float,
        default=0.50,
        help="Fold-specific uncertainty cutoff estimated from train features only.",
    )
    parser.add_argument("--minimum-optical-valid-fraction", type=float, default=0.99)
    return parser.parse_args()


def signature(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    eligibility = pd.read_csv(args.eligibility_csv)
    eligibility = eligibility.loc[
        eligibility["dataset_id"].eq("DLR_Landslide_Ref_2025")
    ].drop_duplicates("sample_id")
    support_columns = [
        "sample_id",
        "optical_valid_fraction",
        "terrain_slope_std_deg",
        "terrain_relief_std_m",
    ]
    eligibility = eligibility[support_columns]
    outputs = []
    counts = {}
    for fold in range(5):
        role_rows = []
        for role in ("train", "val", "test"):
            path = (
                args.runs_dir
                / f"seed20260724/fold{fold}/responder_profile_{role}/per_sample.csv"
            )
            rows = pd.read_csv(path)[
                ["sample_id", "event_id", "visual_margin_mean"]
            ].copy()
            rows["role"] = role
            role_rows.append(rows)
        table = pd.concat(role_rows, ignore_index=True)
        if len(table) != 509 or table["sample_id"].nunique() != 509:
            raise RuntimeError(f"fold {fold} does not partition 509 unique samples")
        table = table.merge(
            eligibility, on="sample_id", how="left", validate="one_to_one"
        )
        if table[support_columns[1:]].isna().any().any():
            raise RuntimeError(f"fold {fold} has missing support features")
        train_margin_threshold = float(
            table.loc[table["role"].eq("train"), "visual_margin_mean"].quantile(
                args.visual_margin_train_quantile
            )
        )
        table["support_eligible"] = (
            table["terrain_slope_std_deg"].ge(args.minimum_slope_std_deg)
            & table["terrain_relief_std_m"].ge(args.minimum_relief_std_m)
            & table["visual_margin_mean"].le(train_margin_threshold)
            & table["optical_valid_fraction"].ge(
                args.minimum_optical_valid_fraction
            )
        ).astype(int)
        table["visual_margin_train_threshold"] = train_margin_threshold
        table["rule_version"] = "dlr_terrain_support_gate_v1"
        table = table.sort_values(["role", "event_id", "sample_id"])
        output = args.outdir / f"fold{fold}_sample_support.csv"
        table.to_csv(output, index=False)
        outputs.append(signature(output))
        fold_counts = {
            role: {
                "n_total": int(len(group)),
                "n_eligible": int(group["support_eligible"].sum()),
            }
            for role, group in table.groupby("role")
        }
        fold_counts["visual_margin_train_threshold"] = train_margin_threshold
        counts[str(fold)] = fold_counts
    manifest = {
        "status": "frozen_exploratory_rule",
        "scientific_status": (
            "label-independent support rule discovered after the first DLR "
            "analysis; requires independent confirmation"
        ),
        "rule": {
            "minimum_slope_std_deg": args.minimum_slope_std_deg,
            "minimum_relief_std_m": args.minimum_relief_std_m,
            "visual_margin_train_quantile": args.visual_margin_train_quantile,
            "minimum_optical_valid_fraction": args.minimum_optical_valid_fraction,
        },
        "forbidden_inputs": [
            "mask",
            "prediction_correctness",
            "delta_iou",
            "rer",
            "corrected",
            "harmed",
        ],
        "eligibility_source": signature(args.eligibility_csv),
        "outputs": outputs,
        "counts": counts,
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
