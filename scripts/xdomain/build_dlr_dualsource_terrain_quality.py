#!/usr/bin/env python3
"""Build a FABDEM native17 cache with label-free CopDEM/FABDEM agreement quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DIR = PROJECT_ROOT / "processed/hybrid_pinn/dlr_terrain_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copdem-h5", type=Path, default=DEFAULT_DIR / "dlr_copdem_native17_p128.h5"
    )
    parser.add_argument(
        "--fabdem-h5", type=Path, default=DEFAULT_DIR / "dlr_fabdem_native17_p128.h5"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DIR / "dlr_fabdem_native17_agreement_qt_p128.h5",
    )
    parser.add_argument("--elevation-scale-m", type=float, default=30.0)
    parser.add_argument("--slope-scale-deg", type=float, default=15.0)
    parser.add_argument("--quality-floor", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def main() -> int:
    args = parse_args()
    if args.elevation_scale_m <= 0 or args.slope_scale_deg <= 0:
        raise ValueError("agreement scales must be positive")
    if not 0 <= args.quality_floor < 1:
        raise ValueError("quality-floor must be in [0,1)")
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    summary_rows = []
    string_type = h5py.string_dtype("utf-8")
    with (
        h5py.File(args.copdem_h5, "r") as copdem,
        h5py.File(args.fabdem_h5, "r") as fabdem,
        h5py.File(temporary, "w") as output,
    ):
        copdem_ids = decode(copdem["sample_id"][:])
        fabdem_ids = decode(fabdem["sample_id"][:])
        copdem_names = decode(copdem["terrain_names"][:])
        fabdem_names = decode(fabdem["terrain_names"][:])
        if copdem_ids != fabdem_ids:
            raise RuntimeError("CopDEM/FABDEM sample identity or ordering differs")
        if copdem_names != fabdem_names:
            raise RuntimeError("CopDEM/FABDEM native17 feature contracts differ")
        if copdem_names[:2] != ["elevation", "slope_deg"]:
            raise RuntimeError("native17 elevation/slope indices are not canonical")
        if int(copdem.attrs.get("complete", 0)) != 1 or int(
            fabdem.attrs.get("complete", 0)
        ) != 1:
            raise RuntimeError("source Terrain cache is incomplete")
        for key, value in fabdem.attrs.items():
            output.attrs[key] = value
        output.attrs.update(
            {
                "complete": 0,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "terrain_source": (
                    "FABDEM V1-2 native17 direction with CopDEM/FABDEM label-free "
                    "agreement quality"
                ),
                "source_kind": "fabdem_copdem_agreement",
                "copdem_h5": str(args.copdem_h5.resolve()),
                "copdem_h5_sha256": sha256_file(args.copdem_h5),
                "fabdem_h5": str(args.fabdem_h5.resolve()),
                "fabdem_h5_sha256": sha256_file(args.fabdem_h5),
                "quality_contract": (
                    "q_T=floor+(1-floor)*exp(-abs(dz)/elevation_scale)"
                    "*exp(-abs(dslope)/slope_scale), multiplied by joint validity"
                ),
                "elevation_scale_m": args.elevation_scale_m,
                "slope_scale_deg": args.slope_scale_deg,
                "quality_floor": args.quality_floor,
            }
        )
        output.create_dataset(
            "sample_id", data=np.asarray(fabdem_ids, dtype=object), dtype=string_type
        )
        output.create_dataset(
            "terrain_names", data=np.asarray(fabdem_names, dtype=object), dtype=string_type
        )
        if "terrain_scale_roles" in fabdem:
            output.create_dataset(
                "terrain_scale_roles",
                data=fabdem["terrain_scale_roles"][:],
                dtype=string_type,
            )
        terrain_ds = output.create_dataset(
            "terrain",
            shape=fabdem["terrain"].shape,
            dtype="float16",
            chunks=fabdem["terrain"].chunks,
            compression="lzf",
        )
        valid_ds = output.create_dataset(
            "terrain_valid",
            shape=fabdem["terrain_valid"].shape,
            dtype="uint8",
            chunks=fabdem["terrain_valid"].chunks,
            compression="lzf",
        )
        quality_ds = output.create_dataset(
            "q_T",
            shape=fabdem["terrain_valid"].shape,
            dtype="float16",
            chunks=fabdem["terrain_valid"].chunks,
            compression="lzf",
        )
        source_tiles = output.create_dataset(
            "source_tiles", shape=(len(fabdem_ids),), dtype=string_type
        )
        for index, sample_id in enumerate(fabdem_ids):
            cop_value = np.asarray(copdem["terrain"][index, :2], dtype=np.float32)
            fab_value = np.asarray(fabdem["terrain"][index, :2], dtype=np.float32)
            joint_valid = (
                np.asarray(copdem["terrain_valid"][index], dtype=bool)
                & np.asarray(fabdem["terrain_valid"][index], dtype=bool)
            )
            elevation_difference = np.abs(cop_value[0] - fab_value[0])
            slope_difference = np.abs(cop_value[1] - fab_value[1])
            agreement = np.exp(
                -elevation_difference / args.elevation_scale_m
                -slope_difference / args.slope_scale_deg
            )
            quality = (
                args.quality_floor + (1.0 - args.quality_floor) * agreement
            )[None] * joint_valid
            terrain_ds[index] = fabdem["terrain"][index]
            valid_ds[index] = joint_valid.astype(np.uint8)
            quality_ds[index] = quality.astype(np.float16)
            source_tiles[index] = str(fabdem["source_tiles"][index])
            keep = joint_valid[0]
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "valid_fraction": float(keep.mean()),
                    "mean_abs_elevation_difference_m": float(
                        elevation_difference[keep].mean()
                    ),
                    "mean_abs_slope_difference_deg": float(
                        slope_difference[keep].mean()
                    ),
                    "mean_q_T": float(quality[0, keep].mean()),
                    "q_T_below_0p5_fraction": float(
                        (quality[0, keep] < 0.5).mean()
                    ),
                }
            )
            if (index + 1) % 50 == 0 or index + 1 == len(fabdem_ids):
                output.attrs["completed_samples"] = index + 1
                output.flush()
                print(
                    f"[dlr-dualsource-quality] {index + 1}/{len(fabdem_ids)} {sample_id}",
                    flush=True,
                )
        output.attrs["complete"] = 1
        output.attrs["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.flush()
    os.replace(temporary, args.out)
    csv_path = args.out.with_suffix(".quality.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {
        "status": "complete",
        "out": str(args.out.resolve()),
        "n_samples": len(summary_rows),
        "mean_abs_elevation_difference_m": float(
            np.mean([row["mean_abs_elevation_difference_m"] for row in summary_rows])
        ),
        "mean_abs_slope_difference_deg": float(
            np.mean([row["mean_abs_slope_difference_deg"] for row in summary_rows])
        ),
        "mean_q_T": float(np.mean([row["mean_q_T"] for row in summary_rows])),
        "mean_q_T_below_0p5_fraction": float(
            np.mean([row["q_T_below_0p5_fraction"] for row in summary_rows])
        ),
        "per_sample_csv": str(csv_path.resolve()),
    }
    args.out.with_suffix(".quality.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
