#!/usr/bin/env python3
"""Audit cross-source physical-event aliases without reading segmentation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILD = ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1/pild_window_readiness.csv"
DEFAULT_SEN12 = ROOT / "processed/hybrid_pinn/sen12_context_v1/trigger_sample_registry_v1.csv"
DEFAULT_SEN12_CACHE = ROOT / "processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv"
DEFAULT_TRIGGER_EVENTS = ROOT / "processed/hybrid_pinn/sen12_context_v1/trigger_event_registry_v1.csv"
DEFAULT_OUTDIR = ROOT / "processed/hybrid_pinn/pild_prithvi_integration_v1"
DEFAULT_REPORT = ROOT.parent / "submission_package_jprs_revision1/response_materials/PILD_Sen12事件别名审计_20260722.md"

PILD_METADATA_COLUMNS = [
    "sample_id", "physical_event_id", "dataset_id", "event_uid", "event_date",
    "bbox_left", "bbox_bottom", "bbox_right", "bbox_top", "source_scene_id",
]
SEN12_METADATA_COLUMNS = [
    "sample_id", "physical_event_id", "region", "event_date", "event_dates",
    "date_quality", "trigger_anchor_date", "center_lon", "center_lat",
]
SEN12_CACHE_ID_COLUMNS = ["sample_id", "physical_event_id"]
TRIGGER_EVENT_COLUMNS = [
    "physical_event_id", "region", "n_samples", "trigger_anchor_date",
]


def stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(x) for x in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def split_dates(value: object) -> set[pd.Timestamp]:
    if pd.isna(value):
        return set()
    out: set[pd.Timestamp] = set()
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        dt = pd.to_datetime(token, errors="coerce")
        if not pd.isna(dt):
            out.add(pd.Timestamp(dt).normalize())
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def bbox_overlap(a: dict, b: dict) -> bool:
    return not (
        a["bbox_right"] < b["bbox_left"]
        or b["bbox_right"] < a["bbox_left"]
        or a["bbox_top"] < b["bbox_bottom"]
        or b["bbox_top"] < a["bbox_bottom"]
    )


def require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def validate_sen12_inputs(
    samples: pd.DataFrame, cache: pd.DataFrame, trigger_events: pd.DataFrame
) -> dict:
    require_columns(cache, {"sample_id", "physical_event_id"}, "Sen12 cache index")
    require_columns(
        trigger_events,
        {"physical_event_id", "region", "n_samples", "trigger_anchor_date"},
        "Sen12 trigger event registry",
    )
    if samples["sample_id"].duplicated().any() or cache["sample_id"].duplicated().any():
        raise ValueError("Sen12 sample IDs must be unique in both registries")
    sample_ids = set(samples["sample_id"].astype(str))
    cache_ids = set(cache["sample_id"].astype(str))
    if sample_ids != cache_ids:
        raise ValueError(
            f"Sen12 sample identity mismatch: samples-only={len(sample_ids-cache_ids)}, "
            f"cache-only={len(cache_ids-sample_ids)}"
        )
    joined = samples[["sample_id", "physical_event_id"]].merge(
        cache[["sample_id", "physical_event_id"]], on="sample_id", suffixes=("_sample", "_cache")
    )
    mismatch = joined["physical_event_id_sample"].astype(str) != joined["physical_event_id_cache"].astype(str)
    if mismatch.any():
        raise ValueError(f"Sen12 physical-event mismatch for {int(mismatch.sum())} samples")
    sample_event_ids = set(samples["physical_event_id"].astype(str))
    trigger_event_ids = set(trigger_events["physical_event_id"].astype(str))
    if sample_event_ids != trigger_event_ids:
        raise ValueError(
            f"Sen12 event identity mismatch: samples-only={sorted(sample_event_ids-trigger_event_ids)}, "
            f"trigger-only={sorted(trigger_event_ids-sample_event_ids)}"
        )
    observed_counts = samples.groupby("physical_event_id").size().astype(int).to_dict()
    recorded_counts = trigger_events.set_index("physical_event_id")["n_samples"].astype(int).to_dict()
    if observed_counts != recorded_counts:
        raise ValueError("Sen12 trigger event sample counts do not match sample registry")
    for _, row in trigger_events.iterrows():
        event_id = str(row["physical_event_id"])
        event_samples = samples[samples["physical_event_id"].astype(str) == event_id]
        sample_anchors = {
            event_date_text(value)
            for value in event_samples["trigger_anchor_date"]
            if not pd.isna(value)
        }
        registry_anchor = event_date_text(row["trigger_anchor_date"])
        expected = {registry_anchor} if registry_anchor else set()
        if sample_anchors != expected:
            raise ValueError(
                f"Sen12 trigger anchor mismatch for {event_id}: sample={sample_anchors}, event={expected}"
            )
    return {
        "sen12_cache_sample_identity_exact": True,
        "sen12_event_identity_exact": True,
        "sen12_trigger_anchor_identity_exact": True,
        "sen12_samples_checked": len(samples),
        "sen12_events_checked": len(sample_event_ids),
    }


def build_pild_events(df: pd.DataFrame) -> list[dict]:
    require_columns(
        df,
        {
            "sample_id", "physical_event_id", "dataset_id", "event_uid", "event_date",
            "bbox_left", "bbox_bottom", "bbox_right", "bbox_top", "source_scene_id",
        },
        "PILD registry",
    )
    rows: list[dict] = []
    for event_id, group in df.groupby("physical_event_id", sort=True):
        dates = sorted({d for value in group["event_date"] for d in split_dates(value)})
        if len(dates) != 1:
            raise ValueError(f"PILD event {event_id} has {len(dates)} event dates")
        left = float(group["bbox_left"].min())
        bottom = float(group["bbox_bottom"].min())
        right = float(group["bbox_right"].max())
        top = float(group["bbox_top"].max())
        rows.append(
            {
                "source_collection": "PILD",
                "source_event_id": str(event_id),
                "source_dataset_ids": "|".join(sorted(group["dataset_id"].astype(str).unique())),
                "source_event_names": "|".join(sorted(group["event_uid"].astype(str).unique())),
                "source_event_date": dates[0],
                "date_basis": "registry_event_date",
                "date_reliable": True,
                "all_dates": set(dates),
                "center_lon": (left + right) / 2,
                "center_lat": (bottom + top) / 2,
                "bbox_left": left,
                "bbox_bottom": bottom,
                "bbox_right": right,
                "bbox_top": top,
                "n_samples": int(len(group)),
                "n_source_scenes": int(group["source_scene_id"].nunique()),
            }
        )
    return rows


def build_sen12_events(df: pd.DataFrame) -> list[dict]:
    require_columns(
        df,
        {
            "sample_id", "physical_event_id", "region", "event_date", "event_dates",
            "date_quality", "trigger_anchor_date", "center_lon", "center_lat",
        },
        "Sen12 registry",
    )
    rows: list[dict] = []
    for event_id, group in df.groupby("physical_event_id", sort=True):
        all_dates: set[pd.Timestamp] = set()
        for col in ("event_date", "event_dates"):
            for value in group[col]:
                all_dates.update(split_dates(value))
        anchors: set[pd.Timestamp] = set()
        for value in group["trigger_anchor_date"]:
            anchors.update(split_dates(value))
        qualities = set(group["date_quality"].dropna().astype(str))
        unique_observed_dates: set[pd.Timestamp] = set()
        for value in group["event_date"]:
            unique_observed_dates.update(split_dates(value))
        if len(anchors) == 1:
            primary_date = next(iter(anchors))
            date_basis = "trigger_anchor_date"
            date_reliable = True
        elif qualities == {"high_single_event"} and len(unique_observed_dates) == 1:
            primary_date = next(iter(unique_observed_dates))
            date_basis = "high_single_event_date"
            date_reliable = True
        else:
            primary_date = pd.NaT
            date_basis = "mixed_or_estimated_dates"
            date_reliable = False
        left = float(group["center_lon"].min())
        bottom = float(group["center_lat"].min())
        right = float(group["center_lon"].max())
        top = float(group["center_lat"].max())
        rows.append(
            {
                "source_collection": "Sen12Landslides",
                "source_event_id": str(event_id),
                "source_dataset_ids": "Sen12Landslides",
                "source_event_names": "|".join(sorted(group["region"].astype(str).unique())),
                "source_event_date": primary_date,
                "date_basis": date_basis,
                "date_reliable": date_reliable,
                "all_dates": all_dates,
                "center_lon": float(group["center_lon"].median()),
                "center_lat": float(group["center_lat"].median()),
                "bbox_left": left,
                "bbox_bottom": bottom,
                "bbox_right": right,
                "bbox_top": top,
                "n_samples": int(len(group)),
                "n_source_scenes": 1,
            }
        )
    return rows


def date_delta(a: pd.Timestamp, b: pd.Timestamp) -> int | None:
    if pd.isna(a) or pd.isna(b):
        return None
    return abs(int((a - b).days))


def min_any_date_delta(a: dict, b: dict) -> int | None:
    if not a["all_dates"] or not b["all_dates"]:
        return None
    return min(abs(int((x - y).days)) for x in a["all_dates"] for y in b["all_dates"])


def classify_pair(pild: dict, sen12: dict) -> dict:
    distance = haversine_km(
        pild["center_lat"], pild["center_lon"], sen12["center_lat"], sen12["center_lon"]
    )
    overlap = bbox_overlap(pild, sen12)
    primary_delta = date_delta(pild["source_event_date"], sen12["source_event_date"])
    any_delta = min_any_date_delta(pild, sen12)

    if (
        pild["date_reliable"]
        and sen12["date_reliable"]
        and primary_delta is not None
        and primary_delta <= 1
        and distance <= 75.0
        and overlap
    ):
        decision = "auto-match"
        reason = (
            f"reliable dates differ by {primary_delta} d; centers are {distance:.2f} km apart; "
            "source AOI envelopes overlap"
        )
    elif (
        (primary_delta is not None and primary_delta <= 14 and distance <= 300.0)
        or (any_delta is not None and any_delta <= 14 and distance <= 100.0)
        or (overlap and any_delta is not None and any_delta <= 30)
    ):
        decision = "manual-review"
        ambiguity = "reliable" if sen12["date_reliable"] else "mixed/estimated"
        reason = (
            f"plausible proximity (center {distance:.2f} km, primary-date delta "
            f"{primary_delta if primary_delta is not None else 'NA'} d, nearest listed-date delta "
            f"{any_delta if any_delta is not None else 'NA'} d), but Sen12 date basis is {ambiguity} "
            f"or AOI evidence is insufficient for automatic merging"
        )
    else:
        decision = "distinct"
        reason = (
            f"fails conservative alias gate (center {distance:.2f} km, primary-date delta "
            f"{primary_delta if primary_delta is not None else 'NA'} d, nearest listed-date delta "
            f"{any_delta if any_delta is not None else 'NA'} d, AOI overlap={overlap})"
        )
    return {
        "pild": pild,
        "sen12": sen12,
        "decision": decision,
        "reason": reason,
        "center_distance_km": distance,
        "bbox_overlap": overlap,
        "primary_date_delta_days": primary_delta,
        "nearest_listed_date_delta_days": any_delta,
    }


def choose_candidate(source: dict, pairs: list[dict]) -> dict:
    relevant = [
        pair
        for pair in pairs
        if pair["pild"]["source_event_id"] == source["source_event_id"]
        or pair["sen12"]["source_event_id"] == source["source_event_id"]
    ]
    rank = {"auto-match": 0, "manual-review": 1, "distinct": 2}
    return min(
        relevant,
        key=lambda pair: (
            rank[pair["decision"]],
            pair["primary_date_delta_days"] if pair["primary_date_delta_days"] is not None else 10**9,
            pair["nearest_listed_date_delta_days"]
            if pair["nearest_listed_date_delta_days"] is not None
            else 10**9,
            pair["center_distance_km"],
        ),
    )


def event_date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def build_output(pild_events: list[dict], sen12_events: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    pairs = [classify_pair(pild, sen12) for pild in pild_events for sen12 in sen12_events]
    auto_pairs = [pair for pair in pairs if pair["decision"] == "auto-match"]

    pild_auto_counts: dict[str, int] = {}
    sen_auto_counts: dict[str, int] = {}
    for pair in auto_pairs:
        pild_id = pair["pild"]["source_event_id"]
        sen_id = pair["sen12"]["source_event_id"]
        pild_auto_counts[pild_id] = pild_auto_counts.get(pild_id, 0) + 1
        sen_auto_counts[sen_id] = sen_auto_counts.get(sen_id, 0) + 1
    conflicts = {
        "pild": {key: value for key, value in pild_auto_counts.items() if value > 1},
        "sen12": {key: value for key, value in sen_auto_counts.items() if value > 1},
    }
    if conflicts["pild"] or conflicts["sen12"]:
        raise ValueError(f"automatic alias one-to-many conflict: {conflicts}")

    canonical: dict[tuple[str, str], str] = {}
    for pair in auto_pairs:
        pild = pair["pild"]
        sen12 = pair["sen12"]
        date = event_date_text(pild["source_event_date"])
        cid = stable_id("CEV1", date, pild["source_event_id"], sen12["source_event_id"])
        canonical[(pild["source_collection"], pild["source_event_id"])] = cid
        canonical[(sen12["source_collection"], sen12["source_event_id"])] = cid

    events = pild_events + sen12_events
    for event in events:
        key = (event["source_collection"], event["source_event_id"])
        canonical.setdefault(key, stable_id("CEV1_UNMERGED", *key))

    rows: list[dict] = []
    for source in events:
        pair = choose_candidate(source, pairs)
        candidate = pair["sen12"] if source["source_collection"] == "PILD" else pair["pild"]
        source_key = (source["source_collection"], source["source_event_id"])
        rows.append(
            {
                "source_collection": source["source_collection"],
                "source_event_id": source["source_event_id"],
                "source_dataset_ids": source["source_dataset_ids"],
                "source_event_names": source["source_event_names"],
                "source_event_date": event_date_text(source["source_event_date"]),
                "source_date_basis": source["date_basis"],
                "source_date_reliable": int(source["date_reliable"]),
                "source_center_lon": source["center_lon"],
                "source_center_lat": source["center_lat"],
                "source_bbox_left": source["bbox_left"],
                "source_bbox_bottom": source["bbox_bottom"],
                "source_bbox_right": source["bbox_right"],
                "source_bbox_top": source["bbox_top"],
                "source_n_samples": source["n_samples"],
                "candidate_collection": candidate["source_collection"],
                "candidate_event_id": candidate["source_event_id"],
                "candidate_event_names": candidate["source_event_names"],
                "candidate_event_date": event_date_text(candidate["source_event_date"]),
                "candidate_date_basis": candidate["date_basis"],
                "primary_date_delta_days": pair["primary_date_delta_days"],
                "nearest_listed_date_delta_days": pair["nearest_listed_date_delta_days"],
                "center_distance_km": pair["center_distance_km"],
                "bbox_overlap": int(pair["bbox_overlap"]),
                "alias_decision": pair["decision"],
                "decision_evidence": pair["reason"],
                "canonical_physical_event_id": canonical[source_key],
                "split_group_id": canonical[source_key],
            }
        )

    output = pd.DataFrame(rows).sort_values(["source_collection", "source_event_id"]).reset_index(drop=True)
    return output, auto_pairs


def validate_output(output: pd.DataFrame, expected_pild: int, expected_sen12: int) -> dict:
    expected_total = expected_pild + expected_sen12
    if len(output) != expected_total:
        raise ValueError(f"expected {expected_total} event rows, found {len(output)}")
    if output.duplicated(["source_collection", "source_event_id"]).any():
        raise ValueError("a source event is represented more than once")
    observed = output.groupby("source_collection")["source_event_id"].nunique().to_dict()
    if observed.get("PILD") != expected_pild or observed.get("Sen12Landslides") != expected_sen12:
        raise ValueError(f"source coverage mismatch: {observed}")
    canonical_sizes = output.groupby("canonical_physical_event_id").size()
    if (canonical_sizes > 2).any():
        raise ValueError("canonical event has more than two source members")
    for _, group in output.groupby("canonical_physical_event_id"):
        if len(group) == 2:
            if set(group["source_collection"]) != {"PILD", "Sen12Landslides"}:
                raise ValueError("merged canonical event does not contain exactly one event per source")
            if set(group["alias_decision"]) != {"auto-match"}:
                raise ValueError("non-auto event was merged")
    auto_rows = output[output["alias_decision"] == "auto-match"]
    if len(auto_rows) % 2:
        raise ValueError("automatic aliases must appear as symmetric source rows")
    return {
        "source_event_rows": int(len(output)),
        "pild_source_events": expected_pild,
        "sen12_source_events": expected_sen12,
        "automatic_alias_pairs": int(len(auto_rows) // 2),
        "manual_review_source_rows": int((output["alias_decision"] == "manual-review").sum()),
        "distinct_source_rows": int((output["alias_decision"] == "distinct").sum()),
        "canonical_events_after_auto_deduplication": int(output["canonical_physical_event_id"].nunique()),
        "one_to_many_auto_conflicts": 0,
        "coverage_complete": True,
    }


def write_report(path: Path, output: pd.DataFrame, summary: dict) -> None:
    auto = output[(output["source_collection"] == "PILD") & (output["alias_decision"] == "auto-match")]
    manual = output[output["alias_decision"] == "manual-review"].copy()
    manual["pair_key"] = manual.apply(
        lambda row: "|".join(sorted([str(row["source_event_id"]), str(row["candidate_event_id"])])), axis=1
    )
    manual = manual.drop_duplicates("pair_key")
    lines = [
        "# PILD + Sen12 physical-event alias audit (2026-07-22)",
        "",
        "## Scope and rule",
        "",
        "This audit uses only event dates, source/region names, and AOI coordinates. "
        "Segmentation labels and model results are not read. Automatic merging requires a reliable date "
        "difference of at most one day, center distance no more than 75 km, overlapping AOI envelopes, "
        "and a one-to-one match. Plausible but weaker cases remain separate pending manual review.",
        "",
        "## Result",
        "",
        f"- Source event records before deduplication: **{summary['source_event_rows']}** "
        f"(PILD {summary['pild_source_events']}; Sen12 {summary['sen12_source_events']}).",
        f"- High-confidence automatic alias pairs: **{summary['automatic_alias_pairs']}**.",
        f"- Canonical physical events after automatic deduplication: "
        f"**{summary['canonical_events_after_auto_deduplication']}**.",
        f"- One-to-many automatic conflicts: **{summary['one_to_many_auto_conflicts']}**.",
        "- Manual-review candidates are not merged and therefore cannot leak across event-level splits.",
        "",
        "## Automatic aliases",
        "",
    ]
    if auto.empty:
        lines.append("None.")
    else:
        for _, row in auto.iterrows():
            lines.append(
                f"- `{row['source_event_id']}` ({row['source_event_names']}) <-> "
                f"`{row['candidate_event_id']}` ({row['candidate_event_names']}): "
                f"{row['decision_evidence']}. Canonical ID: `{row['canonical_physical_event_id']}`."
            )
    lines.extend(["", "## Manual-review candidates", ""])
    if manual.empty:
        lines.append("None.")
    else:
        for _, row in manual.sort_values(["primary_date_delta_days", "center_distance_km"], na_position="last").iterrows():
            lines.append(
                f"- `{row['source_event_id']}` ({row['source_event_names']}) vs "
                f"`{row['candidate_event_id']}` ({row['candidate_event_names']}): "
                f"{row['decision_evidence']}. **Not merged.**"
            )
    lines.extend(
        [
            "",
            "## Split contract",
            "",
            "All rows sharing `canonical_physical_event_id` must be assigned to the same train/validation/test "
            "group. Unresolved manual-review candidates retain separate IDs until documentary evidence resolves "
            "them; future confirmation must regenerate this registry before any split is frozen.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pild-registry", type=Path, default=DEFAULT_PILD)
    parser.add_argument("--sen12-registry", type=Path, default=DEFAULT_SEN12)
    parser.add_argument("--sen12-cache-index", type=Path, default=DEFAULT_SEN12_CACHE)
    parser.add_argument("--trigger-event-registry", type=Path, default=DEFAULT_TRIGGER_EVENTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    # Restrict IO to label-independent metadata at the parser boundary.
    pild = pd.read_csv(args.pild_registry, usecols=PILD_METADATA_COLUMNS)
    sen12 = pd.read_csv(args.sen12_registry, usecols=SEN12_METADATA_COLUMNS)
    sen12_cache = pd.read_csv(args.sen12_cache_index, usecols=SEN12_CACHE_ID_COLUMNS)
    trigger_events = pd.read_csv(args.trigger_event_registry, usecols=TRIGGER_EVENT_COLUMNS)
    input_validation = validate_sen12_inputs(sen12, sen12_cache, trigger_events)
    pild_events = build_pild_events(pild)
    sen12_events = build_sen12_events(sen12)
    output, _ = build_output(pild_events, sen12_events)
    summary = validate_output(output, len(pild_events), len(sen12_events))
    summary.update(
        {
            "protocol": {
                "metadata_only": True,
                "automatic_date_delta_days_max": 1,
                "automatic_center_distance_km_max": 75.0,
                "automatic_bbox_overlap_required": True,
                "manual_review_is_not_merged": True,
                "split_key": "canonical_physical_event_id",
            },
            "inputs": {
                "pild_registry": str(args.pild_registry.resolve()),
                "sen12_registry": str(args.sen12_registry.resolve()),
                "sen12_cache_index": str(args.sen12_cache_index.resolve()),
                "trigger_event_registry": str(args.trigger_event_registry.resolve()),
            },
            "input_validation": input_validation,
        }
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "pild_sen12_event_aliases_v1.csv"
    json_path = args.outdir / "event_alias_summary_v1.json"
    output.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.report, output, summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
