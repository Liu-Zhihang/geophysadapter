#!/usr/bin/env python3
"""Restrict the native17 manifest to the frozen label-independent PILD-GEO4-QC cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_NATIVE17 = (
    ROOT
    / "metadata/pild_sen12_training_native17_v1/unified_sample_manifest_native17_v1.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "metadata/pild_sen12_training_native17_v1/protocol_summary_native17_v1.json"
)
DEFAULT_QC = ROOT / "metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv"
DEFAULT_OUTDIR = ROOT / "metadata/pild_geo4_qc_native17_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native17-manifest", type=Path, default=DEFAULT_NATIVE17)
    parser.add_argument("--native17-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--qc-manifest", type=Path, default=DEFAULT_QC)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    native_path = args.native17_manifest.resolve()
    summary_path = args.native17_summary.resolve()
    qc_path = args.qc_manifest.resolve()
    outdir = args.outdir.resolve()
    output_manifest = outdir / "unified_sample_manifest_geo4_qc_native17_v1.csv"
    output_summary = outdir / "protocol_summary_geo4_qc_native17_v1.json"
    if outdir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    native = pd.read_csv(native_path, keep_default_na=False)
    qc = pd.read_csv(qc_path, keep_default_na=False)
    for name, frame in (("native17", native), ("QC", qc)):
        if "sample_id" not in frame:
            raise ValueError(f"{name} manifest lacks sample_id")
        if frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{name} manifest repeats sample_id")

    native_ids = set(native["sample_id"].astype(str))
    qc_ids = set(qc["sample_id"].astype(str))
    missing = sorted(qc_ids - native_ids)
    if missing:
        raise RuntimeError(f"native17 manifest misses {len(missing)} QC identities")
    filtered = native[native["sample_id"].astype(str).isin(qc_ids)].copy()
    if set(filtered["sample_id"].astype(str)) != qc_ids:
        raise RuntimeError("filtered native17 identities differ from frozen QC cohort")
    filtered.to_csv(output_manifest, index=False)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "pild_geo4_qc_training_protocol.native17.v1"
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["mode"] = (
        "native17 Terrain contract restricted to the frozen label-independent "
        "PILD-GEO4-QC cohort"
    )
    summary["outputs"]["manifest"] = {
        "path": str(output_manifest),
        "sha256": sha256(output_manifest),
    }
    summary["qc_cohort"] = {
        "selection_is_outcome_blind": True,
        "manifest": str(qc_path),
        "manifest_sha256": sha256(qc_path),
        "input_native17_samples": int(len(native)),
        "retained_samples": int(len(filtered)),
        "excluded_samples": int(len(native) - len(filtered)),
        "excluded_by_dataset": {
            str(key): int(value)
            for key, value in native[
                ~native["sample_id"].astype(str).isin(qc_ids)
            ]
            .groupby("dataset_id")
            .size()
            .items()
        },
    }
    write_json(output_summary, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(output_manifest),
                "summary": str(output_summary),
                "retained_samples": int(len(filtered)),
                "excluded_samples": int(len(native) - len(filtered)),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
