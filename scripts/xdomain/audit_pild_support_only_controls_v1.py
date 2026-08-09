#!/usr/bin/env python3
"""Audit whether support-only Terrain gains require aligned spatial support."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pild_support_only_additive_v1 import (  # noqa: E402
    EvaluationDataset,
    build_models,
    evaluate_test_once,
    validate_terrain_checkpoint,
)
from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset, sha256_file  # noqa: E402
from train_pild_sen12_roleaware_v1 import validate_protocol_schema  # noqa: E402
from train_pild_support_only_terrain_v1 import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_SUMMARY,
    validate_parent_v_checkpoint,
    write_json,
)


CONDITIONS = ("aligned", "terrain-zero", "terrain-shift32", "terrain-roll64", "terrain-donor")


def zero_pad_shift(value: torch.Tensor, pixels: int) -> torch.Tensor:
    shifted = torch.roll(value, shifts=(pixels, pixels), dims=(-2, -1))
    shifted[..., :pixels, :] = 0
    shifted[..., :, :pixels] = 0
    return shifted


class ControlledTerrainDataset(Dataset[dict[str, Any]]):
    def __init__(self, base: EvaluationDataset, condition: str) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown control condition: {condition}")
        self.base = base
        self.condition = condition
        events = base.frame["canonical_event_id"].astype(str).tolist()
        self.donor_indices: list[int] = []
        for index, event in enumerate(events):
            candidates = [
                candidate
                for candidate in range(1, len(events) + 1)
                if events[(index + candidate) % len(events)] != event
            ]
            offset = candidates[0] if candidates else (1 if len(events) > 1 else 0)
            self.donor_indices.append((index + offset) % len(events))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        if self.condition == "aligned":
            return item
        if self.condition == "terrain-zero":
            item["terrain"] = torch.zeros_like(item["terrain"])
            item["q_t"] = torch.zeros_like(item["q_t"])
        elif self.condition == "terrain-shift32":
            item["terrain"] = zero_pad_shift(item["terrain"], 32)
            item["q_t"] = zero_pad_shift(item["q_t"], 32)
        elif self.condition == "terrain-roll64":
            item["terrain"] = torch.roll(item["terrain"], shifts=(64, 64), dims=(-2, -1))
            item["q_t"] = torch.roll(item["q_t"], shifts=(64, 64), dims=(-2, -1))
        elif self.condition == "terrain-donor":
            donor = self.base[self.donor_indices[index]]
            item["terrain"] = donor["terrain"]
            item["q_t"] = donor["q_t"]
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parent-v-checkpoint", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--selection-result", type=Path, required=True)
    parser.add_argument("--prithvi-snapshot", type=Path)
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    schema = validate_protocol_schema(
        args.manifest, args.protocol_summary, args.split, args.fold_id
    )
    parent, parent_receipt = validate_parent_v_checkpoint(
        args.parent_v_checkpoint,
        manifest_sha256=schema["manifest_sha256"],
        split_sha256=schema["split_sha256"],
        fold_id=args.fold_id,
        seed=args.seed,
    )
    terrain_checkpoint, terrain_receipt = validate_terrain_checkpoint(
        args.terrain_checkpoint,
        schema=schema,
        protocol_summary_path=args.protocol_summary,
        fold_id=args.fold_id,
        seed=args.seed,
        parent_receipt=parent_receipt,
    )
    selection_payload = json.loads(args.selection_result.read_text(encoding="utf-8"))
    if selection_payload.get("fold_id") != args.fold_id or selection_payload.get("seed") != args.seed:
        raise RuntimeError("selection result identity differs from requested fold/seed")
    if selection_payload["parent_v_receipt"]["checkpoint_sha256"] != parent_receipt["checkpoint_sha256"]:
        raise RuntimeError("selection result parent checkpoint differs")
    if selection_payload["terrain_receipt"]["checkpoint_sha256"] != terrain_receipt["checkpoint_sha256"]:
        raise RuntimeError("selection result Terrain checkpoint differs")
    selection = selection_payload["selection"]

    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite control audit: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    stage = outdir.parent / f".{outdir.name}.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()

    device = torch.device(args.device)
    visual, terrain, provenance = build_models(
        parent=parent,
        terrain_checkpoint=terrain_checkpoint,
        prithvi_snapshot=args.prithvi_snapshot,
        decoder_width=args.decoder_width,
        device=device,
    )
    normalization = terrain_checkpoint["normalization"]
    test_base = UnifiedPILDSen12Dataset(
        args.manifest,
        args.protocol_summary,
        split_path=args.split,
        fold_id=args.fold_id,
        role="test",
        readiness="core",
    )
    evaluation_base = EvaluationDataset(
        test_base,
        mean=normalization["mean"],
        scale=normalization["scale"],
    )
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        loader = DataLoader(
            ControlledTerrainDataset(evaluation_base, condition),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        _, _, corpus = evaluate_test_once(
            visual,
            terrain,
            loader,
            selection=selection,
            threshold=float(parent["threshold"]),
            device=device,
        )
        rows.append({"condition": condition, **corpus})

    aligned = next(row for row in rows if row["condition"] == "aligned")
    contrasts = {
        row["condition"]: {
            "aligned_minus_control_delta_iou": float(aligned["delta_iou"] - row["delta_iou"]),
            "aligned_minus_control_rer": float(aligned["rer"] - row["rer"]),
        }
        for row in rows
        if row["condition"] != "aligned"
    }
    payload = {
        "status": "complete",
        "scientific_status": "test-time falsification control audit",
        "fold_id": args.fold_id,
        "seed": args.seed,
        "selection": selection,
        "selection_result": str(args.selection_result.resolve()),
        "selection_result_sha256": sha256_file(args.selection_result),
        "conditions": rows,
        "contrasts": contrasts,
        "parent_v_receipt": parent_receipt,
        "terrain_receipt": terrain_receipt,
        "prithvi_provenance": provenance,
    }
    write_json(stage / "summary.json", payload)
    write_json(
        stage / "DONE.json",
        {"status": "COMPLETE", "summary_sha256": sha256_file(stage / "summary.json")},
    )
    os.replace(stage, outdir)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
