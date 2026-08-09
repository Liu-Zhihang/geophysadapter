#!/usr/bin/env python3
"""Build label-free Sen12 Trigger likelihood and frozen negative controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from trigger_likelihood_oof import ANCHOR_ORDER, match_known_event, score_event


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "processed/hybrid_pinn/sen12_context_v1/trigger_sample_registry_v1.csv"
DEFAULT_MODELS = ROOT / "metadata/protocol_assets/pild_core_v2_1_phase14_physical_20260719/trigger_preflight/fold_models.json"
DEFAULT_OOF_EVENTS = ROOT / "metadata/protocol_assets/pild_core_v2_1_phase14_physical_20260719/trigger_preflight/trigger_oof_event_scores.csv"
DEFAULT_OUTDIR = ROOT / "processed/hybrid_pinn/sen12_context_v3"
RAIN_COLUMNS = (
    "rain_d7_antecedent_case_mm",
    "rain_d7_wrong_m56_mm",
    "rain_d7_wrong_m28_mm",
    "rain_d7_wrong_p28_mm",
    "rain_d7_wrong_p56_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fold-models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--oof-events", type=Path, default=DEFAULT_OOF_EVENTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def deterministic_donor(event_id: str, supported: Sequence[str], seed: int) -> str:
    candidates = [value for value in supported if value != event_id]
    if not candidates:
        return ""
    digest = hashlib.sha256(f"{seed}|R-event-shuffle|{event_id}".encode()).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(args.input, keep_default_na=False, low_memory=False)
    required = {
        "sample_id", "physical_event_id", "event_date", "center_lon", "center_lat",
        "q_R", *RAIN_COLUMNS,
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise RuntimeError(f"Trigger registry missing columns: {missing}")
    for column in ("q_R", *RAIN_COLUMNS):
        samples[column] = pd.to_numeric(samples[column], errors="coerce")
    fold_models = json.loads(args.fold_models.read_text(encoding="utf-8"))
    if not isinstance(fold_models, list) or len(fold_models) != 5:
        raise RuntimeError("expected five frozen Trigger fold models")
    known_events = pd.read_csv(args.oof_events).to_dict("records")

    event_rows: list[dict[str, Any]] = []
    for event_id, group in samples.groupby("physical_event_id", sort=True):
        quality = float(group["q_R"].max())
        supported = quality > 0 and group[list(RAIN_COLUMNS)].notna().all(axis=1).any()
        if supported:
            rainfall = group.loc[
                group[list(RAIN_COLUMNS)].notna().all(axis=1), list(RAIN_COLUMNS)
            ].median().to_numpy(dtype=np.float64)
            identity = group.iloc[0]
            alias = match_known_event(
                str(identity["event_date"]),
                float(identity["center_lon"]),
                float(identity["center_lat"]),
                known_events,
            )
            score = score_event(alias.physical_event_id if alias else str(event_id), rainfall, fold_models)
            row = {
                "physical_event_id": str(event_id),
                "matched_trigger_event_id": alias.physical_event_id if alias else "",
                "alias_distance_km": alias.distance_km if alias else "",
                "alias_date_delta_days": alias.date_delta_days if alias else "",
                "q_R": quality,
                "trigger_aligned_probability": score.aligned_probability,
                "trigger_aligned_log_bf": score.aligned_log_bf,
                "trigger_wrong_time_probability": score.wrong_time_probability,
                "trigger_wrong_time_log_bf": score.wrong_time_log_bf,
                "scoring_mode": score.scoring_mode,
                "model_folds": ";".join(map(str, score.model_folds)),
                **{name: float(value) for name, value in zip(RAIN_COLUMNS, rainfall, strict=True)},
            }
        else:
            row = {
                "physical_event_id": str(event_id),
                "matched_trigger_event_id": "",
                "alias_distance_km": "",
                "alias_date_delta_days": "",
                "q_R": 0.0,
                "trigger_aligned_probability": 0.2,
                "trigger_aligned_log_bf": 0.0,
                "trigger_wrong_time_probability": 0.2,
                "trigger_wrong_time_log_bf": 0.0,
                "scoring_mode": "exact_abstention",
                "model_folds": "",
                **{name: "" for name in RAIN_COLUMNS},
            }
        event_rows.append(row)

    by_event = {row["physical_event_id"]: row for row in event_rows}
    supported_ids = sorted(
        row["physical_event_id"] for row in event_rows if float(row["q_R"]) > 0
    )
    for row in event_rows:
        donor = deterministic_donor(row["physical_event_id"], supported_ids, args.seed)
        row["event_shuffle_donor"] = donor
        row["trigger_event_shuffle_log_bf"] = (
            float(by_event[donor]["trigger_aligned_log_bf"])
            if donor and float(row["q_R"]) > 0
            else 0.0
        )

    event_output = args.outdir / "trigger_event_likelihood_v1.csv"
    write_csv(event_output, event_rows)
    sample_rows = []
    for sample in samples.itertuples(index=False):
        event = by_event[str(sample.physical_event_id)]
        sample_rows.append(
            {
                "sample_id": str(sample.sample_id),
                "physical_event_id": str(sample.physical_event_id),
                "q_R": event["q_R"],
                "trigger_aligned_log_bf": event["trigger_aligned_log_bf"],
                "trigger_wrong_time_log_bf": event["trigger_wrong_time_log_bf"],
                "trigger_event_shuffle_log_bf": event["trigger_event_shuffle_log_bf"],
                "event_shuffle_donor": event["event_shuffle_donor"],
                "scoring_mode": event["scoring_mode"],
            }
        )
    sample_output = args.outdir / "trigger_sample_likelihood_v1.csv"
    write_csv(sample_output, sample_rows)
    summary = {
        "status": "complete",
        "scientific_role": "event-time likelihood; never a dense boundary direction",
        "n_samples": len(sample_rows),
        "n_events": len(event_rows),
        "n_supported_events": len(supported_ids),
        "n_known_alias_oof_events": int(sum(bool(row["matched_trigger_event_id"]) for row in event_rows)),
        "n_supported_samples": int(sum(float(row["q_R"]) > 0 for row in sample_rows)),
        "anchor_order": list(ANCHOR_ORDER),
        "wrong_time_control": "control_m28",
        "absolute_probability_claim": False,
        "inputs": {
            "registry_sha256": sha256_file(args.input),
            "fold_models_sha256": sha256_file(args.fold_models),
            "oof_events_sha256": sha256_file(args.oof_events),
        },
        "outputs": {
            "event_sha256": sha256_file(event_output),
            "sample_sha256": sha256_file(sample_output),
        },
    }
    write_json(args.outdir / "trigger_likelihood_summary_v1.json", summary)
    write_json(
        args.outdir / "trigger_likelihood_DONE_v1.json",
        {
            "status": "complete",
            "summary_sha256": sha256_file(args.outdir / "trigger_likelihood_summary_v1.json"),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
