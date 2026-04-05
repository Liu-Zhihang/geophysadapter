#!/usr/bin/env python3
"""Build cached train/val/test HDF5 tensors for strict_t2 post_rgb baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from train_strict_t2_postrgb_baseline import (
    StrictT2PostRgbDataset,
    default_eval_cache_path,
    read_gdcld_index,
    read_manifest,
    subset_rows,
)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cache_index",
        "role",
        "event_uid",
        "sample_id",
        "dataset_id",
        "sample_kind",
        "image_path",
        "pre_path",
        "post_path",
        "label_path",
        "valid_mask_path",
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
    failure_csv: Path,
    patch_size: int,
    gdcld_crop_size: int,
    gdcld_index: dict[str, dict[str, str]],
    compression: str | None,
    skip_errors: bool,
) -> dict[str, object]:
    ds = StrictT2PostRgbDataset(
        rows,
        patch_size=patch_size,
        gdcld_crop_size=gdcld_crop_size,
        deterministic_scene_crop=True,
        gdcld_index=gdcld_index,
        gdcld_jitter=0,
    )
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    written_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    with h5py.File(out_h5, "w") as f:
        n = len(rows)
        chunks = (1, 3, patch_size, patch_size)
        chunks_mask = (1, 1, patch_size, patch_size)
        image_ds = f.create_dataset(
            "image",
            shape=(n, 3, patch_size, patch_size),
            maxshape=(None, 3, patch_size, patch_size),
            dtype="float32",
            compression=compression,
            chunks=chunks,
        )
        mask_ds = f.create_dataset(
            "mask",
            shape=(n, 1, patch_size, patch_size),
            maxshape=(None, 1, patch_size, patch_size),
            dtype="float32",
            compression=compression,
            chunks=chunks_mask,
        )
        valid_ds = f.create_dataset(
            "valid",
            shape=(n, 1, patch_size, patch_size),
            maxshape=(None, 1, patch_size, patch_size),
            dtype="float32",
            compression=compression,
            chunks=chunks_mask,
        )
        sample_id_ds = f.create_dataset("sample_id", shape=(n,), maxshape=(None,), dtype=str_dtype)
        dataset_id_ds = f.create_dataset("dataset_id", shape=(n,), maxshape=(None,), dtype=str_dtype)
        sample_kind_ds = f.create_dataset("sample_kind", shape=(n,), maxshape=(None,), dtype=str_dtype)
        event_uid_ds = f.create_dataset("event_uid", shape=(n,), maxshape=(None,), dtype=str_dtype)
        role_ds = f.create_dataset("role", shape=(n,), maxshape=(None,), dtype=str_dtype)
        pos_ratio_ds = f.create_dataset("pos_ratio", shape=(n,), maxshape=(None,), dtype="float32", compression=compression)
        valid_ratio_ds = f.create_dataset("valid_ratio", shape=(n,), maxshape=(None,), dtype="float32", compression=compression)

        write_idx = 0
        for idx, row in enumerate(rows):
            try:
                item = ds[idx]
            except Exception as exc:
                if not skip_errors:
                    raise
                failures.append(
                    {
                        "source_index": str(idx),
                        "split": split,
                        "sample_id": row.get("sample_id", ""),
                        "dataset_id": row.get("dataset_id", ""),
                        "label_path": row.get("label_path", ""),
                        "image_path": row.get("image_path", ""),
                        "error": repr(exc),
                    }
                )
                print(f"[cache:{split}:skip] idx={idx} sample_id={row.get('sample_id','')} error={exc!r}")
                continue
            image = item["image"].numpy().astype(np.float32, copy=False)
            mask = item["mask"].numpy().astype(np.float32, copy=False)
            valid = item["valid"].numpy().astype(np.float32, copy=False)
            image_ds[write_idx] = image
            mask_ds[write_idx] = mask
            valid_ds[write_idx] = valid
            sample_id_ds[write_idx] = row["sample_id"]
            dataset_id_ds[write_idx] = row["dataset_id"]
            sample_kind_ds[write_idx] = row["sample_kind"]
            event_uid_ds[write_idx] = row["event_uid"]
            role_ds[write_idx] = row["role"]
            denom = float(valid.sum())
            pos_ratio_ds[write_idx] = float((mask * valid).sum() / max(denom, 1.0))
            valid_ratio_ds[write_idx] = float(valid.mean())
            written_rows.append(row)
            write_idx += 1
            if (idx + 1) % 64 == 0 or idx + 1 == n:
                print(f"[cache:{split}] {idx + 1}/{n}")

        if write_idx != n:
            image_ds.resize((write_idx, 3, patch_size, patch_size))
            mask_ds.resize((write_idx, 1, patch_size, patch_size))
            valid_ds.resize((write_idx, 1, patch_size, patch_size))
            sample_id_ds.resize((write_idx,))
            dataset_id_ds.resize((write_idx,))
            sample_kind_ds.resize((write_idx,))
            event_uid_ds.resize((write_idx,))
            role_ds.resize((write_idx,))
            pos_ratio_ds.resize((write_idx,))
            valid_ratio_ds.resize((write_idx,))

        f.attrs["subset_name"] = "strict_t2_postrgb_eval_cache_v2"
        f.attrs["split"] = split
        f.attrs["num_samples"] = int(write_idx)
        f.attrs["patch_size"] = int(patch_size)
        f.attrs["gdcld_crop_size"] = int(gdcld_crop_size)

    write_manifest(manifest_csv, written_rows)
    if failures:
        with failure_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source_index", "split", "sample_id", "dataset_id", "label_path", "image_path", "error"],
            )
            writer.writeheader()
            writer.writerows(failures)
    counter = dict(Counter(row["dataset_id"] for row in written_rows))
    summary = {
        "split": split,
        "num_samples": len(written_rows),
        "dataset_counts": counter,
        "out_h5": str(out_h5),
        "manifest_csv": str(manifest_csv),
        "failure_csv": str(failure_csv),
        "skipped_errors": len(failures),
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict_t2 post_rgb train/val/test HDF5 cache")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument(
        "--manifest",
        default="",
        help="default: processed/hybrid_pinn/strict_t2_supervised_ready_v1/sample_manifest_post_rgb.csv",
    )
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--gdcld-crop-size", type=int, default=512)
    p.add_argument("--include-train", action="store_true", help="also build train_postrgb_p*.h5")
    p.add_argument(
        "--splits",
        default="",
        help="optional comma-separated subset of splits to build; choose from train,val,test",
    )
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    p.add_argument("--skip-errors", action="store_true", help="skip unreadable samples and record them to *_failures.csv")
    p.add_argument("--outdir", default="", help="default: processed/hybrid_pinn/strict_t2_postrgb_eval_cache_v2")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    manifest = (
        Path(args.manifest)
        if args.manifest.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_supervised_ready_v1" / "sample_manifest_post_rgb.csv"
    )
    outdir = (
        Path(args.outdir)
        if args.outdir.strip()
        else root / "processed" / "hybrid_pinn" / "strict_t2_postrgb_eval_cache_v2"
    )
    gdcld_index_path = root / "metadata" / "manifests" / "gdcld_postrgb_scene_index_v1.csv"
    gdcld_index = read_gdcld_index(gdcld_index_path)
    compression = None if args.compression == "none" else args.compression
    split_filter = {item.strip() for item in args.splits.split(",") if item.strip()}
    if split_filter:
        invalid = sorted(split_filter - {"train", "val", "test"})
        if invalid:
            raise ValueError(f"unsupported splits: {invalid}")
    rows = read_manifest(manifest)
    train_rows = subset_rows([row for row in rows if row["role"] == "train"], args.train_limit)
    val_rows = subset_rows([row for row in rows if row["role"] == "val"], args.val_limit)
    test_rows = subset_rows([row for row in rows if row["role"] == "test"], args.test_limit)

    train_h5 = default_eval_cache_path(root, "train", args.patch_size) if not args.outdir.strip() else outdir / f"train_postrgb_p{args.patch_size}.h5"
    val_h5 = default_eval_cache_path(root, "val", args.patch_size) if not args.outdir.strip() else outdir / f"val_postrgb_p{args.patch_size}.h5"
    test_h5 = default_eval_cache_path(root, "test", args.patch_size) if not args.outdir.strip() else outdir / f"test_postrgb_p{args.patch_size}.h5"
    train_manifest = train_h5.with_suffix(".csv")
    val_manifest = val_h5.with_suffix(".csv")
    test_manifest = test_h5.with_suffix(".csv")
    train_failures = train_h5.with_name(f"{train_h5.stem}_failures.csv")
    val_failures = val_h5.with_name(f"{val_h5.stem}_failures.csv")
    test_failures = test_h5.with_name(f"{test_h5.stem}_failures.csv")

    summary = {
        "cache_name": "strict_t2_postrgb_eval_cache_v2",
        "manifest": str(manifest),
        "patch_size": int(args.patch_size),
        "gdcld_crop_size": int(args.gdcld_crop_size),
        "splits": {},
    }
    build_train = args.include_train or "train" in split_filter
    build_val = (not split_filter) or ("val" in split_filter)
    build_test = (not split_filter) or ("test" in split_filter)

    if build_train:
        summary["splits"]["train"] = build_split_cache(
            rows=train_rows,
            split="train",
            out_h5=train_h5,
            manifest_csv=train_manifest,
            failure_csv=train_failures,
            patch_size=args.patch_size,
            gdcld_crop_size=args.gdcld_crop_size,
            gdcld_index=gdcld_index,
            compression=compression,
            skip_errors=args.skip_errors,
        )
    if build_val:
        summary["splits"]["val"] = build_split_cache(
            rows=val_rows,
            split="val",
            out_h5=val_h5,
            manifest_csv=val_manifest,
            failure_csv=val_failures,
            patch_size=args.patch_size,
            gdcld_crop_size=args.gdcld_crop_size,
            gdcld_index=gdcld_index,
            compression=compression,
            skip_errors=args.skip_errors,
        )
    if build_test:
        summary["splits"]["test"] = build_split_cache(
            rows=test_rows,
            split="test",
            out_h5=test_h5,
            manifest_csv=test_manifest,
            failure_csv=test_failures,
            patch_size=args.patch_size,
            gdcld_crop_size=args.gdcld_crop_size,
            gdcld_index=gdcld_index,
            compression=compression,
            skip_errors=args.skip_errors,
        )
    summary_path = outdir / f"summary_p{args.patch_size}.json"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
