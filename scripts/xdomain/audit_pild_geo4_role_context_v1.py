#!/usr/bin/env python3
"""Audit Material and Trigger coverage and variability in unified PILD-GEO4."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pild_sen12_training_loader_v2 import UnifiedPILDSen12Dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-summary", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-id", default="source_stratified_0")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role in ("train", "val", "test"):
        dataset = UnifiedPILDSen12Dataset(
            args.manifest,
            args.protocol_summary,
            split_path=args.split,
            fold_id=args.fold_id,
            role=role,
            readiness="core",
        )
        try:
            for index in range(len(dataset)):
                item = dataset[index]
                sample_id = str(item["sample_id"])
                if sample_id in seen:
                    raise RuntimeError(f"duplicate sample across roles: {sample_id}")
                seen.add(sample_id)
                records.append(
                    {
                        "sample_id": sample_id,
                        "dataset_id": str(item["dataset_id"]),
                        "event_id": str(item["canonical_event_id"]),
                        "material": item["role_material_features"].numpy().astype(np.float64),
                        "q_m": float(item["q_material"]),
                        "trigger": item["trigger_features"].numpy().astype(np.float64),
                        "q_r": float(item["q_trigger"]),
                    }
                )
        finally:
            dataset.close()

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["dataset_id"]].append(record)
        by_event[record["event_id"]].append(record)

    source_rows: list[dict[str, Any]] = []
    for source, rows in sorted(by_source.items()):
        q_m = np.asarray([row["q_m"] for row in rows])
        q_r = np.asarray([row["q_r"] for row in rows])
        material = np.stack([row["material"] for row in rows])
        trigger = np.stack([row["trigger"] for row in rows])
        supported_m = q_m > 0
        supported_r = q_r > 0
        source_rows.append(
            {
                "dataset_id": source,
                "n_samples": len(rows),
                "n_events": len({row["event_id"] for row in rows}),
                "material_supported_samples": int(supported_m.sum()),
                "material_supported_fraction": float(supported_m.mean()),
                "material_supported_events": len(
                    {row["event_id"] for row in rows if row["q_m"] > 0}
                ),
                "material_variable_features": int(
                    (np.nanstd(material[supported_m], axis=0) > 1e-8).sum()
                )
                if supported_m.any()
                else 0,
                "trigger_supported_samples": int(supported_r.sum()),
                "trigger_supported_fraction": float(supported_r.mean()),
                "trigger_supported_events": len(
                    {row["event_id"] for row in rows if row["q_r"] > 0}
                ),
                "trigger_variable_features": int(
                    (np.nanstd(trigger[supported_r], axis=0) > 1e-8).sum()
                )
                if supported_r.any()
                else 0,
            }
        )

    trigger_events_with_spatial_variation: list[dict[str, Any]] = []
    material_within_event_variable = 0
    material_supported_events = 0
    for event_id, rows in sorted(by_event.items()):
        m_rows = [row for row in rows if row["q_m"] > 0]
        if m_rows:
            material_supported_events += 1
            material = np.stack([row["material"] for row in m_rows])
            if np.any(np.nanstd(material, axis=0) > 1e-8):
                material_within_event_variable += 1
        r_rows = [row for row in rows if row["q_r"] > 0]
        if r_rows:
            trigger = np.stack([row["trigger"] for row in r_rows])
            if not np.allclose(trigger, trigger[:1], rtol=0.0, atol=1e-6):
                trigger_events_with_spatial_variation.append(
                    {
                        "event_id": event_id,
                        "n_supported_samples": len(r_rows),
                        "max_range": float(np.ptp(trigger, axis=0).max()),
                    }
                )

    summary = {
        "schema_version": "pild_geo4_role_context_audit.v2",
        "n_samples": len(records),
        "n_events": len(by_event),
        "fold_partition_used_for_exhaustive_scan": args.fold_id,
        "sources": source_rows,
        "material": {
            "feature_count": 21,
            "supported_events": material_supported_events,
            "events_with_within_event_spatial_variation": material_within_event_variable,
        },
        "trigger": {
            "feature_count": 3,
            "contract": "event-time by sample-location contextual dose",
            "within_event_spatial_variation_is_expected": True,
            "events_with_spatial_variation": trigger_events_with_spatial_variation,
        },
    }
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# PILD-GEO4 Material/Trigger role audit",
        "",
        f"- Samples: {len(records)}",
        f"- Events: {len(by_event)}",
        "- Trigger contract: event-time by sample-location contextual dose",
        (
            "- Trigger-supported events with spatial variation: "
            f"{len(trigger_events_with_spatial_variation)}"
        ),
        (
            "- Material supported events with within-event spatial variation: "
            f"{material_within_event_variable}/{material_supported_events}"
        ),
        "",
        "| Source | Samples | Events | M support | M events | M variable dims | R support | R events | R variable dims |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['dataset_id']} | {row['n_samples']} | {row['n_events']} | "
            f"{row['material_supported_fraction']:.1%} | "
            f"{row['material_supported_events']} | {row['material_variable_features']} | "
            f"{row['trigger_supported_fraction']:.1%} | "
            f"{row['trigger_supported_events']} | {row['trigger_variable_features']} |"
        )
    (args.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
