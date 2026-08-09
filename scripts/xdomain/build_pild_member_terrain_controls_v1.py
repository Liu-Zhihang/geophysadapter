#!/usr/bin/env python3
"""Build label-independent terrain mismatch controls for a PILD member cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--terrain-h5", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--roll-pixels", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def copy_attrs(source: h5py.File, target: h5py.File) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def donor_permutation(sample_ids: list[str], event_ids: list[str]) -> list[int]:
    by_event: dict[str, list[int]] = {}
    for index, event_id in enumerate(event_ids):
        by_event.setdefault(event_id, []).append(index)
    if len(by_event) < 2:
        raise RuntimeError("event-mismatch control requires at least two events")
    event_names = sorted(by_event)
    donors = []
    for sample_id, event_id in zip(sample_ids, event_ids, strict=True):
        other_events = [name for name in event_names if name != event_id]
        token = hashlib.sha256(f"terrain-event-mismatch-v1|{sample_id}".encode()).digest()
        donor_event = other_events[int.from_bytes(token[:8], "big") % len(other_events)]
        candidates = by_event[donor_event]
        donor = candidates[int.from_bytes(token[8:16], "big") % len(candidates)]
        donors.append(donor)
    if any(event_ids[index] == event_ids[donor] for index, donor in enumerate(donors)):
        raise RuntimeError("event-mismatch donor assignment contains a same-event pair")
    return donors


def create_like(
    source: h5py.File,
    output_path: Path,
    sample_ids: list[str],
    index_transform,
    spatial_transform,
    contract: str,
) -> None:
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w") as target:
        copy_attrs(source, target)
        target.create_dataset(
            "sample_id",
            data=np.asarray(sample_ids, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        target.create_dataset(
            "terrain_names",
            data=np.asarray(decode(source["terrain_names"][:]), dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        terrain_out = target.create_dataset(
            "terrain",
            shape=source["terrain"].shape,
            dtype=source["terrain"].dtype,
            chunks=(1, *source["terrain"].shape[1:]),
        )
        valid_out = target.create_dataset(
            "terrain_valid",
            shape=source["terrain_valid"].shape,
            dtype=source["terrain_valid"].dtype,
            chunks=(1, *source["terrain_valid"].shape[1:]),
        )
        for target_index in range(len(sample_ids)):
            source_index = index_transform(target_index)
            terrain_out[target_index] = spatial_transform(source["terrain"][source_index])
            valid_out[target_index] = spatial_transform(source["terrain_valid"][source_index])
        target.attrs["complete"] = 1
        target.attrs["control_contract"] = contract
        target.attrs["label_content_accessed"] = 0
        target.attrs["source_terrain_h5"] = str(source.filename)
    partial.replace(output_path)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    mismatch_path = args.outdir / "terrain_event_mismatch_p128.h5"
    roll_path = args.outdir / f"terrain_roll{args.roll_pixels}_p128.h5"
    summary_path = args.outdir / "terrain_controls_summary.json"
    if (
        not args.overwrite
        and summary_path.exists()
        and mismatch_path.exists()
        and roll_path.exists()
    ):
        print(summary_path.read_text(encoding="utf-8"), end="")
        return 0
    with h5py.File(args.base_h5, "r") as base, h5py.File(args.terrain_h5, "r") as terrain:
        sample_ids = decode(base["sample_id"][:])
        event_ids = decode(base["physical_event_id"][:])
        terrain_ids = decode(terrain["sample_id"][:])
        if sample_ids != terrain_ids:
            raise RuntimeError("base and terrain sample ordering differs")
        donors = donor_permutation(sample_ids, event_ids)
        create_like(
            terrain,
            mismatch_path,
            sample_ids,
            lambda index: donors[index],
            lambda value: value,
            "deterministic donor from a different physical event; labels forbidden",
        )
        shift = int(args.roll_pixels)
        create_like(
            terrain,
            roll_path,
            sample_ids,
            lambda index: index,
            lambda value: np.roll(np.roll(value, shift, axis=-2), shift, axis=-1),
            f"within-sample cyclic spatial roll by ({shift},{shift}) pixels; labels forbidden",
        )
    payload: dict[str, Any] = {
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "n_samples": len(sample_ids),
        "n_events": len(set(event_ids)),
        "event_mismatch_pairs": len(donors),
        "event_mismatch_same_event_pairs": sum(
            event_ids[index] == event_ids[donor] for index, donor in enumerate(donors)
        ),
        "roll_pixels": args.roll_pixels,
        "outputs": {
            "event_mismatch": str(mismatch_path.resolve()),
            "spatial_roll": str(roll_path.resolve()),
        },
        "selection_contract": "label-independent controls; no result-based sample selection",
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
