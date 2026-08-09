#!/usr/bin/env python3
"""Export pre/post optical composites aligned with the object diagnostic cache.

The object-level purity model needs the appearance evidence that distinguishes a fresh
failure scar from a permanently bright surface such as a river bar or a bare field: the
pre-event state. Only the temporal means are stored, which keeps the cache small while
preserving the change signal the decoder itself relies on.

Sample order is identical to the prediction cache produced by
``export_pild_oof_object_diagnostic_cache_v1.py`` and is verified against it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_pild_oof_object_diagnostic_cache_v1 import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_SUMMARY,
    FOLDS,
)
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file  # noqa: E402
from train_pild_support_only_terrain_v1 import write_json  # noqa: E402


def export_fold(
    *,
    fold_id: str,
    manifest: Path,
    protocol_summary: Path,
    split: Path,
    reference_cache: Path,
    batch_size: int,
    num_workers: int,
    outdir: Path,
) -> dict[str, Any]:
    dataset = UnifiedPILDSen12Dataset(
        manifest,
        protocol_summary,
        split_path=split,
        fold_id=fold_id,
        role="test",
        readiness="core",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    material: list[np.ndarray] = []
    trigger: list[np.ndarray] = []
    q_material: list[np.ndarray] = []
    q_trigger: list[np.ndarray] = []
    sample_id: list[str] = []
    for batch in loader:
        optical = batch["optical"]
        if optical.ndim != 5 or optical.shape[2] != 4:
            raise RuntimeError(f"unexpected optical shape {tuple(optical.shape)}")
        pre.append(optical[:, :, :2].mean(dim=2).numpy().astype(np.float16))
        post.append(optical[:, :, 2:].mean(dim=2).numpy().astype(np.float16))
        material.append(batch["role_material_features"].numpy().astype(np.float32))
        trigger.append(batch["trigger_features"].numpy().astype(np.float32))
        q_material.append(batch["q_material"].numpy().astype(np.float32))
        q_trigger.append(batch["q_trigger"].numpy().astype(np.float32))
        sample_id.extend(str(item) for item in batch["sample_id"])

    with np.load(reference_cache, allow_pickle=False) as handle:
        reference_ids = [str(item) for item in handle["sample_id"]]
    if reference_ids != sample_id:
        raise RuntimeError(
            f"{fold_id}: optical export order differs from the prediction cache"
        )

    payload = {
        "sample_id": np.asarray(sample_id, dtype="U160"),
        "optical_pre": np.concatenate(pre),
        "optical_post": np.concatenate(post),
        "material_features": np.concatenate(material),
        "trigger_features": np.concatenate(trigger),
        "q_material": np.concatenate(q_material),
        "q_trigger": np.concatenate(q_trigger),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{fold_id}_optical_cache.npz"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    os.replace(temporary, path)

    receipt = {
        "fold_id": fold_id,
        "n_samples": len(sample_id),
        "optical_channels": int(payload["optical_pre"].shape[1]),
        "material_dimension": int(payload["material_features"].shape[1]),
        "trigger_dimension": int(payload["trigger_features"].shape[1]),
        "cache_path": str(path),
        "cache_sha256": sha256_file(path),
        "reference_cache": str(reference_cache),
        "order_matches_reference": True,
    }
    write_json(outdir / f"{fold_id}_optical_cache_receipt.json", receipt)
    print(f"[done] {fold_id}: {len(sample_id)} samples -> {path.name}", flush=True)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir.resolve()
    started = time.time()
    receipts = []
    for fold_id in args.folds:
        reference = cache_dir / f"{fold_id}_oof_cache.npz"
        if not reference.is_file():
            raise FileNotFoundError(reference)
        target = cache_dir / f"{fold_id}_optical_cache.npz"
        receipt_path = cache_dir / f"{fold_id}_optical_cache_receipt.json"
        if target.is_file() and receipt_path.is_file():
            print(f"[skip] {fold_id}: optical cache present", flush=True)
            receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
            continue
        receipts.append(
            export_fold(
                fold_id=fold_id,
                manifest=args.manifest.resolve(),
                protocol_summary=args.protocol_summary.resolve(),
                split=args.split.resolve(),
                reference_cache=reference,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                outdir=cache_dir,
            )
        )
    write_json(
        cache_dir / "optical_export_summary.json",
        {
            "schema_version": "pild_object_diagnostic_optical_cache.v1",
            "folds": [item["fold_id"] for item in receipts],
            "total_samples": sum(int(item["n_samples"]) for item in receipts),
            "elapsed_seconds": round(time.time() - started, 2),
            "receipts": receipts,
        },
    )
    print("[done] optical cache complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
