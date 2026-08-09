#!/usr/bin/env python3
"""Export out-of-fold visual predictions plus native Terrain for object-level diagnosis.

Each outer fold uses its own frozen V checkpoint to predict its own test split, so
every patch in the unified corpus receives exactly one prediction. The cache stores
the visual probability map, the label, the valid mask and the raw 17-channel Terrain
tensor, which lets all downstream object-level feature work run on CPU without
re-touching the GPU or the training pipeline.

No labels are used to select anything here; this is a pure export step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pild_sen12_training_loader_v2 import (  # noqa: E402
    UnifiedPILDSen12Dataset,
    sha256_file,
)
from sen12_prithvi_v2 import (  # noqa: E402
    PrithviEO2ChangeModel,
    load_prithvi_encoder,
)
from train_pild_sen12_roleaware_v1 import (  # noqa: E402
    tensor_sha256,
    validate_protocol_schema,
)
from train_pild_support_only_terrain_v1 import (  # noqa: E402
    validate_parent_v_checkpoint,
    write_json,
)


DEFAULT_META = PROJECT_ROOT / "metadata/pild_geo4_qc_native17_v1"
DEFAULT_MANIFEST = DEFAULT_META / "unified_sample_manifest_geo4_qc_native17_v1.csv"
DEFAULT_SUMMARY = DEFAULT_META / "protocol_summary_geo4_qc_native17_v1.json"
DEFAULT_SPLIT = (
    PROJECT_ROOT / "metadata/pild_geo4_qc_v1/source_stratified_event_folds_v1.csv"
)
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_native17_source_stratified_tempered075_v1"
)
FOLDS = (
    "source_stratified_0",
    "source_stratified_1",
    "source_stratified_2",
    "source_stratified_3",
)


def build_visual_model(
    parent: dict[str, Any],
    *,
    prithvi_snapshot: Path | None,
    decoder_width: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the exact frozen visual anchor and verify its component hash."""
    encoder, provenance = load_prithvi_encoder(prithvi_snapshot)
    expected_prithvi = parent["identity"]["prithvi_checkpoint_sha256"]
    if provenance["checkpoint_sha256"] != expected_prithvi:
        raise RuntimeError("loaded Prithvi snapshot differs from parent V identity")
    model = PrithviEO2ChangeModel(
        encoder, decoder_width=decoder_width, freeze_encoder=True
    )
    components = parent["components"]
    if "visual_full" in components:
        # Anchors trained with an opened encoder cannot be rebuilt from the pristine
        # snapshot plus a decoder, so the stored visual module is authoritative.
        model.load_state_dict(components["visual_full"], strict=True)
        observed = tensor_sha256(model.state_dict())
        if observed != parent["component_sha256"]["visual_full"]:
            raise RuntimeError("visual module hash differs from authenticated parent")
    else:
        model.decoder.load_state_dict(components["visual_decoder"], strict=True)
        observed = tensor_sha256(model.decoder.state_dict())
        if observed != parent["component_sha256"]["visual_decoder"]:
            raise RuntimeError("visual decoder hash differs from authenticated parent")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, provenance


@torch.no_grad()
def export_fold(
    *,
    fold_id: str,
    seed: int,
    manifest: Path,
    protocol_summary: Path,
    split: Path,
    parent_checkpoint: Path,
    prithvi_snapshot: Path | None,
    decoder_width: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    outdir: Path,
) -> dict[str, Any]:
    schema = validate_protocol_schema(manifest, protocol_summary, split, fold_id)
    parent, parent_receipt = validate_parent_v_checkpoint(
        parent_checkpoint,
        manifest_sha256=schema["manifest_sha256"],
        split_sha256=schema["split_sha256"],
        fold_id=fold_id,
        seed=seed,
    )
    model, provenance = build_visual_model(
        parent,
        prithvi_snapshot=prithvi_snapshot,
        decoder_width=decoder_width,
        device=device,
    )
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
        pin_memory=device.type == "cuda",
    )

    probability: list[np.ndarray] = []
    target: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    terrain: list[np.ndarray] = []
    terrain_valid: list[np.ndarray] = []
    sample_id: list[str] = []
    dataset_id: list[str] = []
    event_id: list[str] = []

    for batch in loader:
        optical = batch["optical"].to(device) * 10_000.0
        output = model(
            optical,
            batch["temporal_coords"].to(device),
            batch["location_coords"].to(device),
        )
        logits = output["logits"].float()
        probability.append(
            torch.sigmoid(logits)[:, 0].cpu().numpy().astype(np.float16)
        )
        mask = batch["mask"]
        if mask.ndim == 3:
            mask = mask[:, None]
        keep = batch["valid_mask"]
        if keep.ndim == 3:
            keep = keep[:, None]
        target.append((mask[:, 0] >= 0.5).numpy().astype(np.uint8))
        valid.append((keep[:, 0] > 0).numpy().astype(np.uint8))
        terrain.append(batch["terrain"].numpy().astype(np.float16))
        support = batch["terrain_valid"]
        if support.ndim == 4:
            support = support[:, 0]
        terrain_valid.append((support.numpy() > 0).astype(np.uint8))
        sample_id.extend(str(item) for item in batch["sample_id"])
        dataset_id.extend(str(item) for item in batch["dataset_id"])
        event_id.extend(str(item) for item in batch["canonical_event_id"])

    payload = {
        "sample_id": np.asarray(sample_id, dtype="U160"),
        "dataset_id": np.asarray(dataset_id, dtype="U64"),
        "canonical_event_id": np.asarray(event_id, dtype="U96"),
        "visual_probability": np.concatenate(probability),
        "target": np.concatenate(target),
        "valid": np.concatenate(valid),
        "terrain": np.concatenate(terrain),
        "terrain_valid": np.concatenate(terrain_valid),
    }
    if len({len(payload[key]) for key in payload}) != 1:
        raise RuntimeError("exported arrays disagree in length")
    if len(set(sample_id)) != len(sample_id):
        raise RuntimeError("duplicate sample_id inside a single fold export")

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{fold_id}_oof_cache.npz"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    os.replace(temporary, path)

    receipt = {
        "fold_id": fold_id,
        "seed": int(seed),
        "n_samples": int(len(sample_id)),
        "n_events": int(len(set(event_id))),
        "threshold": float(parent["threshold"]),
        "cache_path": str(path),
        "cache_sha256": sha256_file(path),
        "manifest_sha256": schema["manifest_sha256"],
        "split_sha256": schema["split_sha256"],
        "parent_v_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
        "prithvi_checkpoint_sha256": provenance["checkpoint_sha256"],
        "terrain_channel_order": list(
            json.loads(protocol_summary.read_text(encoding="utf-8"))["terrain_contract"][
                "names"
            ]
        ),
    }
    write_json(outdir / f"{fold_id}_oof_cache_receipt.json", receipt)
    print(
        f"[done] {fold_id}: {receipt['n_samples']} samples / "
        f"{receipt['n_events']} events -> {path.name}",
        flush=True,
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--prithvi-snapshot", type=Path)
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[FATAL] CUDA requested but unavailable")
    outdir = args.outdir.resolve()
    started = time.time()
    receipts = []
    for fold_id in args.folds:
        parent_checkpoint = (
            args.run_root / fold_id / f"seed{args.seed}" / "V" / "checkpoint.pt"
        )
        if not parent_checkpoint.is_file():
            raise FileNotFoundError(parent_checkpoint)
        cache = outdir / f"{fold_id}_oof_cache.npz"
        receipt_path = outdir / f"{fold_id}_oof_cache_receipt.json"
        if cache.is_file() and receipt_path.is_file():
            print(f"[skip] {fold_id}: cache already present", flush=True)
            receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
            continue
        receipts.append(
            export_fold(
                fold_id=fold_id,
                seed=args.seed,
                manifest=args.manifest.resolve(),
                protocol_summary=args.protocol_summary.resolve(),
                split=args.split.resolve(),
                parent_checkpoint=parent_checkpoint,
                prithvi_snapshot=args.prithvi_snapshot,
                decoder_width=args.decoder_width,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                outdir=outdir,
            )
        )

    total = sum(int(item["n_samples"]) for item in receipts)
    ids = [item["fold_id"] for item in receipts]
    summary = {
        "schema_version": "pild_object_diagnostic_oof_cache.v1",
        "scope": "out-of-fold visual predictions and raw Terrain for object-level diagnosis",
        "labels_used_for_selection": False,
        "folds": ids,
        "total_samples": total,
        "elapsed_seconds": round(time.time() - started, 2),
        "receipts": receipts,
    }
    write_json(outdir / "export_summary.json", summary)
    print(f"[done] exported {total} unique samples across {len(ids)} folds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
