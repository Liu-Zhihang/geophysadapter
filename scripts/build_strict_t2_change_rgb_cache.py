#!/usr/bin/env python3
"""Build cached train/val/test HDF5 tensors for strict_t2 change_rgb baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from train_strict_t2_change_rgb_baseline import (
    StrictT2ChangeRgbDataset,
    default_cache_path,
)
from train_strict_t2_postrgb_baseline import read_manifest, subset_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cache_index",
        "role",
        "event_uid",
        "sample_id",
        "dataset_id",
        "sample_kind",
        "pre_path",
        "post_path",
        "label_path",
        "h5_path",
        "h5_sample_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            payload = {key: row.get(key, "") for key in fieldnames if key != "cache_index"}
            payload["cache_index"] = idx
            writer.writerow(payload)


def build_split_cache(
    rows: list[dict[str, str]],
    split: str,
    out_h5: Path,
    manifest_csv: Path,
    patch_size: int,
    compression: str | None,
) -> dict[str, object]:
    ds = StrictT2ChangeRgbDataset(rows, patch_size=patch_size)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_h5, "w") as f:
        n = len(rows)
        chunks_img = (1, 6, patch_size, patch_size)
        chunks_mask = (1, 1, patch_size, patch_size)
        image_ds = f.create_dataset(
            "image",
            shape=(n, 6, patch_size, patch_size),
            dtype="float32",
            compression=compression,
            chunks=chunks_img,
        )
        mask_ds = f.create_dataset(
            "mask",
            shape=(n, 1, patch_size, patch_size),
            dtype="float32",
            compression=compression,
            chunks=chunks_mask,
        )
        valid_ds = f.create_dataset(
            "valid",
            shape=(n, 1, patch_size, patch_size),
            dtype="float32",
            compression=compression,
            chunks=chunks_mask,
        )
        sample_id_ds = f.create_dataset("sample_id", shape=(n,), dtype=str_dtype)
        dataset_id_ds = f.create_dataset("dataset_id", shape=(n,), dtype=str_dtype)
        sample_kind_ds = f.create_dataset("sample_kind", shape=(n,), dtype=str_dtype)
        event_uid_ds = f.create_dataset("event_uid", shape=(n,), dtype=str_dtype)
        role_ds = f.create_dataset("role", shape=(n,), dtype=str_dtype)
        pos_ratio_ds = f.create_dataset("pos_ratio", shape=(n,), dtype="float32", compression=compression)
        valid_ratio_ds = f.create_dataset("valid_ratio", shape=(n,), dtype="float32", compression=compression)

        for idx, row in enumerate(rows):
            item = ds[idx]
            image = item["image"].numpy().astype(np.float32, copy=False)
            mask = item["mask"].numpy().astype(np.float32, copy=False)
            valid = item["valid"].numpy().astype(np.float32, copy=False)
            image_ds[idx] = image
            mask_ds[idx] = mask
            valid_ds[idx] = valid
            sample_id_ds[idx] = row["sample_id"]
            dataset_id_ds[idx] = row["dataset_id"]
            sample_kind_ds[idx] = row["sample_kind"]
            event_uid_ds[idx] = row["event_uid"]
            role_ds[idx] = row["role"]
            denom = float(valid.sum())
            pos_ratio_ds[idx] = float((mask * valid).sum() / max(denom, 1.0))
            valid_ratio_ds[idx] = float(valid.mean())
            if (idx + 1) % 32 == 0 or idx + 1 == n:
                print(f"[cache:{split}] {idx + 1}/{n}")

        f.attrs["subset_name"] = "strict_t2_change_rgb_cache_v1"
        f.attrs["split"] = split
        f.attrs["num_samples"] = int(n)
        f.attrs["patch_size"] = int(patch_size)

    write_manifest(manifest_csv, rows)
    return {
        "split": split,
        "num_samples": len(rows),
        "dataset_counts": dict(Counter(row["dataset_id"] for row in rows)),
        "out_h5": str(out_h5),
        "manifest_csv": str(manifest_csv),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict_t2 change_rgb HDF5 cache")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument(
        "--manifest",
        default="",
        help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_manifest_change_rgb.csv",
    )
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument(
        "--splits",
        default="train,val,test",
        help="comma-separated subset of splits to build; choose from train,val,test",
    )
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    p.add_argument("--outdir", default="", help="default: processed/hybrid_pinn/strict_t2_change_rgb_cache_v1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    manifest = (
        Path(args.manifest)
        if args.manifest.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_manifest_change_rgb.csv"
    )
    outdir = (
        Path(args.outdir)
        if args.outdir.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_change_rgb_cache_v1"
    )
    compression = None if args.compression == "none" else args.compression
    split_filter = {item.strip() for item in args.splits.split(",") if item.strip()}
    invalid = sorted(split_filter - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"unsupported splits: {invalid}")

    rows = read_manifest(manifest)
    split_rows = {
        "train": subset_rows([row for row in rows if row["role"] == "train"], args.train_limit),
        "val": subset_rows([row for row in rows if row["role"] == "val"], args.val_limit),
        "test": subset_rows([row for row in rows if row["role"] == "test"], args.test_limit),
    }

    summary = {
        "cache_name": "strict_t2_change_rgb_cache_v1",
        "manifest": str(manifest),
        "patch_size": int(args.patch_size),
        "splits": {},
    }
    outdir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        if split not in split_filter:
            continue
        out_h5 = default_cache_path(root, split, args.patch_size) if not args.outdir.strip() else outdir / f"{split}_changergb_p{args.patch_size}.h5"
        manifest_csv = out_h5.with_suffix(".csv")
        summary["splits"][split] = build_split_cache(
            rows=split_rows[split],
            split=split,
            out_h5=out_h5,
            manifest_csv=manifest_csv,
            patch_size=args.patch_size,
            compression=compression,
        )

    summary_path = outdir / f"summary_p{args.patch_size}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
