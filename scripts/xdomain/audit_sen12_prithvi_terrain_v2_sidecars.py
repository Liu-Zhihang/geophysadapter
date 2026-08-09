#!/usr/bin/env python3
"""Audit frozen base, Prithvi temporal, and native Terrain-v2 sidecars."""

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

from sen12_terrain_v2 import NATIVE_TERRAIN_V2_NAMES


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, default=ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5")
    parser.add_argument("--optical-h5", type=Path, default=ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_prithvi_4t6b_p128.h5")
    parser.add_argument("--terrain-h5", type=Path, default=ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_native_terrain_v2_p128.h5")
    parser.add_argument("--registry", type=Path, default=ROOT / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "metadata/pild_xdomain_v1/terrain_prithvi_v2_sidecar_audit.json")
    parser.add_argument("--hash", action="store_true")
    return parser.parse_args()


def decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    registry = {
        row["sample_id"]: row
        for row in csv.DictReader(args.registry.open(encoding="utf-8-sig", newline=""))
    }
    with h5py.File(args.base_h5, "r") as base, h5py.File(args.optical_h5, "r") as optical, h5py.File(args.terrain_h5, "r") as terrain:
        base_ids, optical_ids, terrain_ids = (
            decode(handle["sample_id"][:]) for handle in (base, optical, terrain)
        )
        identity_exact = base_ids == optical_ids == terrain_ids
        if not identity_exact:
            raise RuntimeError("sidecar sample identity/order mismatch")
        date_violations = []
        duplicate_flag_violations = []
        dates = decode(optical["selected_dates"][:])
        indices = optical["selected_indices"][:]
        duplicate = optical["duplicated_observation"][:]
        for position, sample_id in enumerate(base_ids):
            selected = dates[position].split(";")
            row = registry[sample_id]
            if not (
                len(selected) == 4
                and selected[0] <= selected[1] < row["event_date_start"]
                and selected[2] <= selected[3]
                and selected[2] >= row["event_date_end"]
            ):
                date_violations.append(sample_id)
            expected_duplicate = np.asarray(
                [indices[position, 0] == indices[position, 1]] * 2
                + [indices[position, 2] == indices[position, 3]] * 2,
                dtype=np.uint8,
            )
            if not np.array_equal(expected_duplicate, duplicate[position]):
                duplicate_flag_violations.append(sample_id)
        finite_terrain = True
        terrain_min, terrain_max = float("inf"), float("-inf")
        positive_pixels = 0
        positive_without_terrain = 0
        optical_valid_min = 1.0
        optical_base_alignment_max_abs = 0.0
        base_anchor_comparison_samples = 0
        base_anchor_missing_samples = 0
        for start in range(0, len(base_ids), 128):
            stop = min(start + 128, len(base_ids))
            values = np.asarray(terrain["terrain"][start:stop])
            finite_terrain &= bool(np.isfinite(values).all())
            terrain_min = min(terrain_min, float(values.min()))
            terrain_max = max(terrain_max, float(values.max()))
            label = np.asarray(base["mask"][start:stop]) > 0
            valid = np.asarray(terrain["terrain_valid"][start:stop]) > 0
            positive_pixels += int(label.sum())
            positive_without_terrain += int(np.logical_and(label, ~valid).sum())
            optical_valid_min = min(
                optical_valid_min,
                float(np.asarray(optical["optical_valid"][start:stop]).mean(axis=(1, 2, 3)).min()),
            )
            selected = np.asarray(optical["selected_indices"][start:stop])
            base_pre = np.asarray(base["pre_index"][start:stop])
            base_post = np.asarray(base["post_index"][start:stop])
            optical_rgb = np.asarray(optical["optical"][start:stop], dtype=np.float32) / 10_000.0
            base_rgb = np.asarray(base["obs"][start:stop], dtype=np.float32)
            for local in range(stop - start):
                pre_frames = np.flatnonzero(selected[local, :2] == base_pre[local])
                post_frames = np.flatnonzero(selected[local, 2:] == base_post[local]) + 2
                if pre_frames.size == 0 or post_frames.size == 0:
                    base_anchor_missing_samples += 1
                    continue
                base_anchor_comparison_samples += 1
                pre_frame, post_frame = int(pre_frames[-1]), int(post_frames[0])
                paired = (
                    (base_rgb[local, 0], optical_rgb[local, 2, pre_frame]),
                    (base_rgb[local, 1], optical_rgb[local, 1, pre_frame]),
                    (base_rgb[local, 2], optical_rgb[local, 0, pre_frame]),
                    (base_rgb[local, 3], optical_rgb[local, 2, post_frame]),
                    (base_rgb[local, 4], optical_rgb[local, 1, post_frame]),
                    (base_rgb[local, 5], optical_rgb[local, 0, post_frame]),
                )
                optical_base_alignment_max_abs = max(
                    optical_base_alignment_max_abs,
                    max(float(np.max(np.abs(left - right))) for left, right in paired),
                )
        q_t = np.asarray(terrain["q_T"][:])
        feature_names = decode(terrain["terrain_names"][:])
        checks = {
            "identity_and_order_exact": identity_exact,
            "n_samples_4979": len(base_ids) == 4979,
            "optical_shape": tuple(optical["optical"].shape) == (4979, 6, 4, 128, 128),
            "terrain_shape": tuple(terrain["terrain"].shape) == (4979, 17, 128, 128),
            "terrain_feature_names_exact": feature_names == list(NATIVE_TERRAIN_V2_NAMES),
            "date_contract": not date_violations,
            "duplicate_flags": not duplicate_flag_violations,
            "terrain_finite": finite_terrain,
            "base_anchor_comparison_coverage": base_anchor_comparison_samples >= int(0.98 * len(base_ids)),
            "optical_spatial_alignment_with_base": optical_base_alignment_max_abs < 1e-3,
            "optical_complete": int(optical.attrs.get("complete", 0)) == 1,
            "terrain_complete": int(terrain.attrs.get("complete", 0)) == 1,
        }
        result = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "n_samples": len(base_ids),
            "optical_shape": list(optical["optical"].shape),
            "terrain_shape": list(terrain["terrain"].shape),
            "terrain_features": feature_names,
            "date_violations": date_violations[:20],
            "duplicate_flag_violations": duplicate_flag_violations[:20],
            "samples_with_repeated_observations": int(duplicate.any(axis=1).sum()),
            "optical_valid_fraction_min": optical_valid_min,
            "base_anchor_comparison_samples": base_anchor_comparison_samples,
            "base_anchor_missing_samples": base_anchor_missing_samples,
            "optical_base_rgb_max_abs_error": optical_base_alignment_max_abs,
            "terrain_value_range": [terrain_min, terrain_max],
            "q_T": {
                "min": float(q_t.min()),
                "median": float(np.median(q_t)),
                "samples_below_0.5": int((q_t < 0.5).sum()),
                "samples_below_0.9": int((q_t < 0.9).sum()),
            },
            "positive_pixels": positive_pixels,
            "positive_pixels_without_terrain": positive_without_terrain,
            "positive_terrain_coverage": 1.0 - positive_without_terrain / max(positive_pixels, 1),
            "files": {},
        }
    for name, path in (("base_h5", args.base_h5), ("optical_h5", args.optical_h5), ("terrain_h5", args.terrain_h5), ("registry", args.registry)):
        stat = path.stat()
        result["files"][name] = {
            "path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path) if args.hash else None,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
