#!/usr/bin/env python3
"""Build deterministic event-isolated nested OOF splits for PILD-Geo4.

For each outer source-stratified fold, only outer-train events participate.
They are partitioned into three source-aware inner-test buckets. The smallest
non-test event set needed for validation is selected deterministically; all
remaining outer-train events train the inner proposer. Outer validation and
test samples are explicitly excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTER = (
    PROJECT_ROOT / "metadata/pild_geo4_qc_v1/source_stratified_event_folds_v1.csv"
)
DEFAULT_OUTDIR = PROJECT_ROOT / "metadata/pild_geo4_qc_v1/nested_oof_v1"


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def assign_test_buckets(events: pd.DataFrame, n_folds: int) -> dict[str, int]:
    assignment: dict[str, int] = {}
    for dataset_id, group in events.groupby("dataset_id", sort=True):
        loads = [0] * n_folds
        ordered = sorted(
            group.to_dict("records"),
            key=lambda row: (
                -int(row["n_samples"]),
                stable_int(f"{dataset_id}|{row['canonical_event_id']}"),
            ),
        )
        for row in ordered:
            minimum = min(loads)
            candidates = [index for index, value in enumerate(loads) if value == minimum]
            bucket = candidates[
                stable_int(str(row["canonical_event_id"])) % len(candidates)
            ]
            assignment[str(row["canonical_event_id"])] = bucket
            loads[bucket] += int(row["n_samples"])
    return assignment


def choose_validation_events(
    events: pd.DataFrame,
    test_events: set[str],
    *,
    token: str,
) -> set[str]:
    available = events[~events["canonical_event_id"].isin(test_events)]
    selected: set[str] = set()
    for dataset_id, group in available.groupby("dataset_id", sort=True):
        records = sorted(
            group.to_dict("records"),
            key=lambda row: (
                int(row["n_samples"]),
                stable_int(f"{token}|{dataset_id}|{row['canonical_event_id']}"),
            ),
        )
        if records:
            selected.add(str(records[0]["canonical_event_id"]))
    if not selected:
        raise RuntimeError(f"{token}: no validation events available")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_OUTER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-inner-folds", type=int, default=3)
    args = parser.parse_args()
    if args.n_inner_folds < 3:
        raise ValueError("at least three inner folds are required")
    outer = pd.read_csv(args.outer_split, keep_default_na=False)
    required = {
        "protocol_id", "fold_id", "sample_id", "dataset_id", "source_id",
        "canonical_event_id", "heldout_dataset_id", "role", "role_reason",
    }
    if required - set(outer.columns):
        raise ValueError(f"outer split lacks columns: {sorted(required-set(outer.columns))}")
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite nested protocol: {outdir}")
    outdir.mkdir(parents=True)
    output_rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for outer_fold in sorted(outer["fold_id"].astype(str).unique()):
        selected = outer[outer["fold_id"].astype(str).eq(outer_fold)].copy()
        outer_train = selected[selected["role"].eq("train")].copy()
        event_table = (
            outer_train.groupby(["canonical_event_id", "dataset_id"], as_index=False)
            .agg(n_samples=("sample_id", "size"))
        )
        if len(event_table) < args.n_inner_folds + 2:
            raise RuntimeError(f"{outer_fold}: insufficient outer-train events")
        buckets = assign_test_buckets(event_table, args.n_inner_folds)
        test_coverage: defaultdict[str, int] = defaultdict(int)
        for inner in range(args.n_inner_folds):
            inner_fold = f"{outer_fold}__inner_{inner}"
            test_events = {event for event, bucket in buckets.items() if bucket == inner}
            val_events = choose_validation_events(
                event_table, test_events, token=inner_fold
            )
            train_events = (
                set(event_table["canonical_event_id"].astype(str))
                - test_events
                - val_events
            )
            if not train_events or not test_events or not val_events:
                raise RuntimeError(f"{inner_fold}: empty active role")
            if (train_events & test_events) or (train_events & val_events) or (
                test_events & val_events
            ):
                raise RuntimeError(f"{inner_fold}: event leakage")
            frame = selected.copy()
            frame["fold_id"] = inner_fold
            frame["protocol_id"] = "pild_geo4_nested_oof_v1"
            frame["role"] = "excluded"
            frame["role_reason"] = "excluded: outside target outer-train"
            event_id = frame["canonical_event_id"].astype(str)
            frame.loc[event_id.isin(train_events), "role"] = "train"
            frame.loc[event_id.isin(val_events), "role"] = "val"
            frame.loc[event_id.isin(test_events), "role"] = "test"
            frame.loc[event_id.isin(train_events), "role_reason"] = (
                "nested proposer train; outer-train only"
            )
            frame.loc[event_id.isin(val_events), "role_reason"] = (
                "nested proposer validation; deterministic source-aware events"
            )
            frame.loc[event_id.isin(test_events), "role_reason"] = (
                "nested inner OOF holdout; each outer-train event exactly once"
            )
            active = frame[frame["role"].isin(("train", "val", "test"))]
            if int(active.groupby("canonical_event_id")["role"].nunique().max()) != 1:
                raise RuntimeError(f"{inner_fold}: sample-level event leakage")
            for event in test_events:
                test_coverage[event] += 1
            output_rows.append(frame[list(outer.columns)])
            audits.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner,
                    "fold_id": inner_fold,
                    "n_train_samples": int(frame["role"].eq("train").sum()),
                    "n_val_samples": int(frame["role"].eq("val").sum()),
                    "n_test_samples": int(frame["role"].eq("test").sum()),
                    "n_train_events": len(train_events),
                    "n_val_events": len(val_events),
                    "n_test_events": len(test_events),
                    "train_events": sorted(train_events),
                    "val_events": sorted(val_events),
                    "test_events": sorted(test_events),
                }
            )
        expected_events = set(event_table["canonical_event_id"].astype(str))
        if set(test_coverage) != expected_events or set(test_coverage.values()) != {1}:
            raise RuntimeError(f"{outer_fold}: inner-test does not partition outer-train")
    output = pd.concat(output_rows, ignore_index=True)
    split_path = outdir / "pild_geo4_nested_oof_v1.csv"
    output.to_csv(split_path, index=False)
    atomic_json(
        outdir / "protocol_manifest.json",
        {
            "schema_version": "pild_geo4_nested_oof_protocol.v1",
            "outer_split": str(args.outer_split.resolve()),
            "outer_split_sha256": sha256_file(args.outer_split),
            "split": str(split_path),
            "split_sha256": sha256_file(split_path),
            "n_inner_folds_per_outer": args.n_inner_folds,
            "n_total_folds": len(audits),
            "contracts": {
                "outer_val_test_excluded": True,
                "canonical_event_isolated": True,
                "each_outer_train_event_once_inner_test": True,
                "assignment_uses_labels": False,
            },
            "folds": audits,
        },
    )
    print(f"wrote {len(audits)} nested folds to {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
