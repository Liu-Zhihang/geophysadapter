#!/usr/bin/env python3
"""Register strict antecedent Trigger context for the frozen Sen12 cache.

The registry is label-free. It uses event identity, independently documented
trigger anchors, sample coordinates, and CHIRPS daily rainfall only. Rainfall
is summed over D-7..D-1; four same-location shifted anchors are falsification
controls. Samples without uniquely attributable rainfall events receive q_R=0.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import rowcol
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[2]
CONTROL_SHIFTS = {
    "wrong_m56": -56,
    "wrong_m28": -28,
    "wrong_p28": 28,
    "wrong_p56": 56,
}
ANCHOR_SHIFTS = {"case": 0, **CONTROL_SHIFTS}
STRICT_LAGS = tuple(range(1, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--cache-index",
        type=Path,
        default=Path("processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv"),
    )
    parser.add_argument(
        "--base-h5",
        type=Path,
        default=Path("processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5"),
    )
    parser.add_argument(
        "--prithvi-h5",
        type=Path,
        default=Path("processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_prithvi_4t6b_p128.h5"),
    )
    parser.add_argument(
        "--sample-registry",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"),
    )
    parser.add_argument(
        "--anchor-registry",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/sen12_trigger_event_anchor_registry_v1.csv"),
    )
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/candidate_event_registry_v1.csv"),
    )
    parser.add_argument(
        "--chirps-root",
        type=Path,
        default=Path("raw_fullcopy/weather/chirps_daily_global"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("processed/hybrid_pinn/sen12_context_v1"),
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def decode(values: Any) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def chirps_path(root: Path, day: pd.Timestamp) -> Path:
    return root / f"{day.year:04d}" / f"chirps-v2.0.{day:%Y.%m.%d}.tif.gz"


def sample_native_3x3(path: Path, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    compressed = path.read_bytes()
    digest = hashlib.sha256(compressed).hexdigest()
    with MemoryFile(gzip.decompress(compressed)) as memory:
        with memory.open() as source:
            if source.count != 1 or source.crs is None or not source.crs.is_geographic:
                raise RuntimeError(f"Invalid CHIRPS raster contract: {path}")
            resolution = [abs(float(source.transform.a)), abs(float(source.transform.e))]
            if not all(math.isclose(value, 0.05, abs_tol=1e-8) for value in resolution):
                raise RuntimeError(f"Unexpected CHIRPS resolution {resolution}: {path}")
            rows, columns = rowcol(source.transform, lon, lat)
            rows = np.asarray(rows, dtype=int)
            columns = np.asarray(columns, dtype=int)
            inside_global = (
                (rows >= 0)
                & (rows < source.height)
                & (columns >= 0)
                & (columns < source.width)
            )
            values = np.full(len(rows), np.nan, dtype=np.float64)
            if np.any(inside_global):
                valid_rows = rows[inside_global]
                valid_columns = columns[inside_global]
                row0 = max(0, int(valid_rows.min()) - 1)
                row1 = min(source.height, int(valid_rows.max()) + 2)
                col0 = max(0, int(valid_columns.min()) - 1)
                col1 = min(source.width, int(valid_columns.max()) + 2)
                block = source.read(
                    1,
                    window=Window(col0, row0, col1 - col0, row1 - row0),
                    out_dtype="float64",
                )
                local_rows = rows - row0
                local_columns = columns - col0
                neighborhood = np.full((len(rows), 9), np.nan, dtype=np.float64)
                position = 0
                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        rr = local_rows + delta_row
                        cc = local_columns + delta_column
                        valid = inside_global & (rr >= 0) & (rr < block.shape[0]) & (cc >= 0) & (cc < block.shape[1])
                        neighborhood[valid, position] = block[rr[valid], cc[valid]]
                        position += 1
                finite = np.isfinite(neighborhood) & (neighborhood >= 0.0)
                if source.nodata is not None and math.isfinite(float(source.nodata)):
                    finite &= ~np.isclose(neighborhood, float(source.nodata), atol=1e-8, rtol=0.0)
                with np.errstate(invalid="ignore"):
                    values = np.nanmedian(np.where(finite, neighborhood, np.nan), axis=1)
            metadata = {
                "sha256": digest,
                "bytes": len(compressed),
                "width": int(source.width),
                "height": int(source.height),
                "crs": str(source.crs),
                "resolution_degrees": resolution,
            }
    return values, metadata


def exact_signflip_median_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return 1.0
    if values.size > 20:
        raise RuntimeError("Exact sign-flip test is limited to 20 independent events")
    observed = abs(float(np.median(values)))
    exceed = 0
    for mask in range(1 << values.size):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(values.size)])
        exceed += abs(float(np.median(values * signs))) >= observed - 1e-12
    return float(exceed / (1 << values.size))


def bootstrap_median_ci(values: np.ndarray, seed: int, draws: int) -> list[float | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(draws, values.size), replace=True), axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def validate_h5(path: Path, cache: pd.DataFrame, check_metadata: bool) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if int(handle.attrs.get("complete", 0)) != 1:
            raise RuntimeError(f"H5 is not marked complete: {path}")
        ids = decode(handle["sample_id"][:])
        if ids != cache["sample_id"].astype(str).tolist():
            raise RuntimeError(f"H5 sample order differs from cache index: {path}")
        if check_metadata:
            if decode(handle["physical_event_id"][:]) != cache["physical_event_id"].astype(str).tolist():
                raise RuntimeError(f"H5 physical_event_id differs from cache index: {path}")
            h5_dates = decode(handle["event_date"][:])
            csv_dates = cache["event_date"].fillna("").astype(str).tolist()
            if h5_dates != csv_dates:
                raise RuntimeError(f"H5 event_date differs from cache index: {path}")
            if decode(handle["date_quality"][:]) != cache["date_quality"].astype(str).tolist():
                raise RuntimeError(f"H5 date_quality differs from cache index: {path}")
        return {
            "path": str(path),
            "sha256": sha256(path),
            "samples": len(ids),
            "complete": True,
        }


def mechanism_map(candidate: pd.DataFrame, anchor: pd.DataFrame, event_ids: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for event_id in sorted(event_ids):
        values = {
            str(value).strip().lower()
            for value in candidate.loc[
                candidate["physical_event_cluster_id"].astype(str) == event_id,
                "trigger_family",
            ].dropna()
            if str(value).strip().lower() not in {"", "unknown", "nan"}
        }
        output[event_id] = next(iter(values)) if len(values) == 1 else ("ambiguous" if values else "unknown")
    for row in anchor.itertuples(index=False):
        output[str(row.physical_event_cluster_id)] = str(row.trigger_family).strip().lower()
    return output


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = {
        key: resolve(root, value)
        for key, value in {
            "cache_index": args.cache_index,
            "base_h5": args.base_h5,
            "prithvi_h5": args.prithvi_h5,
            "sample_registry": args.sample_registry,
            "anchor_registry": args.anchor_registry,
            "candidate_registry": args.candidate_registry,
            "chirps_root": args.chirps_root,
            "outdir": args.outdir,
        }.items()
    }
    outdir = paths["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)

    cache = pd.read_csv(paths["cache_index"], low_memory=False)
    if len(cache) != 4979 or cache["sample_id"].duplicated().any():
        raise RuntimeError("Frozen Sen12 cache must contain 4979 unique samples")
    if cache["physical_event_id"].nunique() != 15:
        raise RuntimeError("Frozen Sen12 cache must contain 15 physical_event_id values")
    h5_checks = [
        validate_h5(paths["base_h5"], cache, check_metadata=True),
        validate_h5(paths["prithvi_h5"], cache, check_metadata=False),
    ]

    sample_registry = pd.read_csv(paths["sample_registry"], low_memory=False)
    required_registry = [
        "sample_id", "region", "physical_event_cluster_id", "center_lon", "center_lat", "crs"
    ]
    merged = cache.merge(
        sample_registry[required_registry],
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_registry"),
    )
    if merged[["center_lon", "center_lat", "crs_registry"]].isna().any().any():
        raise RuntimeError("Missing sample coordinates or CRS after registry join")
    if not np.all(merged["physical_event_id"].astype(str) == merged["physical_event_cluster_id"].astype(str)):
        raise RuntimeError("physical_event_id does not match sample-registry event identity")
    if not np.all(merged["crs"].astype(str) == merged["crs_registry"].astype(str)):
        raise RuntimeError("Cache and sample-registry CRS values disagree")

    anchor = pd.read_csv(paths["anchor_registry"], low_memory=False)
    if anchor["physical_event_cluster_id"].duplicated().any():
        raise RuntimeError("Anchor registry must contain one row per physical event")
    candidate = pd.read_csv(paths["candidate_registry"], low_memory=False)
    event_ids = set(merged["physical_event_id"].astype(str))
    families = mechanism_map(candidate, anchor, event_ids)
    anchor_by_event = anchor.set_index(anchor["physical_event_cluster_id"].astype(str), drop=False)

    sample = merged[[
        "cache_index", "sample_id", "patch_id", "region_group", "physical_event_id",
        "event_date", "event_dates", "date_quality", "time_selection_contract",
        "center_lon", "center_lat", "crs",
    ]].copy()
    sample = sample.rename(columns={"region_group": "region", "crs": "source_crs"})
    sample["mechanism_family"] = sample["physical_event_id"].map(families).fillna("unknown")
    for column in [
        "trigger_anchor_date", "anchor_role", "anchor_confidence", "source_organization",
        "source_url", "source_note",
    ]:
        sample[column] = ""
    for event_id, row in anchor_by_event.iterrows():
        mask = sample["physical_event_id"].astype(str) == str(event_id)
        for column in [
            "trigger_anchor_date", "anchor_role", "anchor_confidence", "source_organization",
            "source_url", "source_note",
        ]:
            sample.loc[mask, column] = str(row[column])

    numeric_columns = [
        "rain_d7_antecedent_case_mm", "rain_d7_wrong_m56_mm", "rain_d7_wrong_m28_mm",
        "rain_d7_wrong_p28_mm", "rain_d7_wrong_p56_mm",
        "rain_d7_wrongtime_median_mm", "rain_d7_case_minus_wrongtime_mm",
    ]
    for column in numeric_columns:
        sample[column] = np.nan
    sample["case_days_available"] = 0
    sample["control_days_available_min"] = 0
    sample["q_R_case_coverage"] = 0.0
    sample["q_R_control_coverage"] = 0.0

    chirps_manifest: list[dict[str, Any]] = []
    rainfall_events = anchor.loc[anchor["trigger_family"].astype(str).str.lower() == "rainfall"].copy()
    for event in rainfall_events.itertuples(index=False):
        event_id = str(event.physical_event_cluster_id)
        positions = sample.index[sample["physical_event_id"].astype(str) == event_id].to_numpy()
        if positions.size == 0:
            continue
        lon = sample.loc[positions, "center_lon"].to_numpy(dtype=float)
        lat = sample.loc[positions, "center_lat"].to_numpy(dtype=float)
        anchor_day = pd.Timestamp(event.trigger_anchor_date)
        totals: dict[str, np.ndarray] = {}
        available_days: dict[str, np.ndarray] = {}
        for role, shift in ANCHOR_SHIFTS.items():
            values_by_day: list[np.ndarray] = []
            for lag in STRICT_LAGS:
                day = anchor_day + timedelta(days=shift - lag)
                path = chirps_path(paths["chirps_root"], day)
                record: dict[str, Any] = {
                    "physical_event_id": event_id,
                    "region": str(event.region),
                    "anchor_role": role,
                    "anchor_shift_days": shift,
                    "antecedent_lag_days": lag,
                    "date": day.date().isoformat(),
                    "path": str(path),
                    "status": "missing",
                }
                if path.is_file():
                    day_values, metadata = sample_native_3x3(path, lon, lat)
                    values_by_day.append(day_values)
                    record.update(metadata)
                    record["status"] = "available"
                    record["finite_sample_fraction"] = float(np.mean(np.isfinite(day_values)))
                else:
                    values_by_day.append(np.full(positions.size, np.nan, dtype=np.float64))
                    record["finite_sample_fraction"] = 0.0
                chirps_manifest.append(record)
            matrix = np.column_stack(values_by_day)
            available_days[role] = np.sum(np.isfinite(matrix), axis=1)
            complete = np.all(np.isfinite(matrix), axis=1)
            totals[role] = np.where(complete, np.sum(matrix, axis=1), np.nan)
        sample.loc[positions, "rain_d7_antecedent_case_mm"] = totals["case"]
        for role in CONTROL_SHIFTS:
            sample.loc[positions, f"rain_d7_{role}_mm"] = totals[role]
        controls = np.column_stack([totals[role] for role in CONTROL_SHIFTS])
        controls_complete = np.all(np.isfinite(controls), axis=1)
        wrong_median = np.where(controls_complete, np.median(controls, axis=1), np.nan)
        sample.loc[positions, "rain_d7_wrongtime_median_mm"] = wrong_median
        sample.loc[positions, "rain_d7_case_minus_wrongtime_mm"] = totals["case"] - wrong_median
        sample.loc[positions, "case_days_available"] = available_days["case"]
        sample.loc[positions, "control_days_available_min"] = np.min(
            np.column_stack([available_days[role] for role in CONTROL_SHIFTS]), axis=1
        )
        sample.loc[positions, "q_R_case_coverage"] = np.isfinite(totals["case"]).astype(float)
        sample.loc[positions, "q_R_control_coverage"] = controls_complete.astype(float)
        print(f"[rainfall] {event_id} samples={positions.size}", flush=True)

    sample["unique_event_date"] = (
        sample["date_quality"].eq("high_single_event")
        & sample["event_date"].notna()
        & ~sample["event_date"].astype(str).str.contains("[;,|]", regex=True)
    )
    sample["q_R"] = (
        sample["unique_event_date"]
        & sample["mechanism_family"].eq("rainfall")
        & sample["trigger_anchor_date"].ne("")
        & sample["q_R_case_coverage"].eq(1.0)
        & sample["q_R_control_coverage"].eq(1.0)
    ).astype(float)
    sample["q_R_high_confidence"] = (
        sample["q_R"].eq(1.0) & sample["anchor_confidence"].eq("high")
    ).astype(float)
    sample["q_R_reason"] = np.select(
        [
            sample["q_R"].eq(1.0),
            sample["date_quality"].eq("multi_event_mixed"),
            sample["date_quality"].eq("estimated_single_event"),
            ~sample["unique_event_date"],
            sample["mechanism_family"].eq("earthquake"),
            ~sample["mechanism_family"].eq("rainfall"),
            sample["trigger_anchor_date"].eq(""),
            sample["q_R_case_coverage"].ne(1.0),
            sample["q_R_control_coverage"].ne(1.0),
        ],
        [
            "rainfall_strict_antecedent_complete",
            "multi_event_mixed_no_unique_date",
            "estimated_date_not_admitted",
            "date_not_uniquely_attributable",
            "earthquake_requires_non_chirps_trigger",
            "mechanism_not_rainfall",
            "missing_independent_trigger_anchor",
            "incomplete_case_chirps",
            "incomplete_wrongtime_chirps",
        ],
        default="unresolved",
    )
    if not sample.loc[sample["date_quality"].eq("multi_event_mixed"), "q_R"].eq(0.0).all():
        raise RuntimeError("multi_event_mixed samples must have q_R=0")
    if not sample.loc[~sample["unique_event_date"], "q_R"].eq(0.0).all():
        raise RuntimeError("Non-unique event dates must have q_R=0")

    event_rows: list[dict[str, Any]] = []
    for event_id, group in sample.groupby("physical_event_id", sort=True):
        supported = group[group["q_R"].eq(1.0)]
        deltas = supported["rain_d7_case_minus_wrongtime_mm"].to_numpy(dtype=float)
        event_rows.append({
            "physical_event_id": event_id,
            "region": str(group["region"].iloc[0]),
            "mechanism_family": str(group["mechanism_family"].iloc[0]),
            "n_samples": len(group),
            "date_quality_values": "|".join(sorted(set(group["date_quality"].astype(str)))),
            "n_unique_dataset_dates": int(group["event_date"].nunique(dropna=True)),
            "trigger_anchor_date": str(group["trigger_anchor_date"].iloc[0]),
            "anchor_role": str(group["anchor_role"].iloc[0]),
            "anchor_confidence": str(group["anchor_confidence"].iloc[0]),
            "source_organization": str(group["source_organization"].iloc[0]),
            "source_url": str(group["source_url"].iloc[0]),
            "n_q_R_positive": int(group["q_R"].sum()),
            "q_R_fraction": float(group["q_R"].mean()),
            "median_rain_d7_antecedent_case_mm": float(supported["rain_d7_antecedent_case_mm"].median()) if len(supported) else np.nan,
            "median_rain_d7_wrongtime_mm": float(supported["rain_d7_wrongtime_median_mm"].median()) if len(supported) else np.nan,
            "median_rain_d7_case_minus_wrongtime_mm": float(np.median(deltas)) if deltas.size else np.nan,
            "gate_status": "admitted" if len(supported) else "blocked",
            "gate_reason": str(group["q_R_reason"].mode().iloc[0]),
        })
    event_frame = pd.DataFrame(event_rows)

    output_paths = {
        "sample_registry": outdir / "trigger_sample_registry_v1.csv",
        "event_registry": outdir / "trigger_event_registry_v1.csv",
        "coverage_blockers": outdir / "trigger_coverage_blockers_v1.csv",
        "chirps_manifest": outdir / "trigger_chirps_source_manifest_v1.csv",
        "summary": outdir / "trigger_summary_v1.json",
        "source_manifest": outdir / "trigger_source_manifest_v1.json",
    }
    sample.to_csv(output_paths["sample_registry"], index=False)
    event_frame.to_csv(output_paths["event_registry"], index=False)
    sample.loc[sample["q_R"].eq(0.0), [
        "sample_id", "physical_event_id", "region", "date_quality", "mechanism_family",
        "event_date", "q_R", "q_R_reason",
    ]].to_csv(output_paths["coverage_blockers"], index=False)
    chirps_frame = pd.DataFrame(chirps_manifest)
    chirps_frame.to_csv(output_paths["chirps_manifest"], index=False)

    event_deltas = event_frame.loc[
        event_frame["gate_status"].eq("admitted"), "median_rain_d7_case_minus_wrongtime_mm"
    ].to_numpy(dtype=float)
    reason_counts = {str(k): int(v) for k, v in sample["q_R_reason"].value_counts().items()}
    summary = {
        "schema_version": 1,
        "contract": {
            "feature": "CHIRPS daily rainfall summed over strict D-7..D-1",
            "controls": "same sample coordinates at anchor shifts -56,-28,+28,+56 days; median of four D-7..D-1 sums",
            "q_R": "1 only for a uniquely attributable high_single_event rainfall mechanism with independent anchor and complete case/control coverage; otherwise 0",
            "no_imputation": True,
            "label_free": "No segmentation masks, predictions, logits, metrics, or checkpoints are read.",
        },
        "cache": {
            "samples": len(sample),
            "physical_events": int(sample["physical_event_id"].nunique()),
            "regions": int(sample["region"].nunique()),
            "date_quality_counts": {str(k): int(v) for k, v in sample["date_quality"].value_counts().items()},
            "mechanism_family_counts_samples": {str(k): int(v) for k, v in sample["mechanism_family"].value_counts().items()},
        },
        "coverage": {
            "q_R_positive_samples": int(sample["q_R"].sum()),
            "q_R_positive_fraction": float(sample["q_R"].mean()),
            "q_R_high_confidence_samples": int(sample["q_R_high_confidence"].sum()),
            "admitted_events": int(event_frame["gate_status"].eq("admitted").sum()),
            "blocked_events": int(event_frame["gate_status"].eq("blocked").sum()),
            "reason_counts": reason_counts,
            "required_chirps_usages": len(chirps_frame),
            "available_chirps_usages": int(chirps_frame["status"].eq("available").sum()),
            "missing_chirps_usages": int(chirps_frame["status"].eq("missing").sum()),
        },
        "event_level_gate": {
            "n_independent_rainfall_events": int(event_deltas.size),
            "median_case_minus_wrongtime_mm": float(np.median(event_deltas)) if event_deltas.size else np.nan,
            "positive_events": int(np.sum(event_deltas > 0)),
            "bootstrap_median_ci95_mm": bootstrap_median_ci(event_deltas, args.seed, args.bootstrap),
            "exact_signflip_median_p": exact_signflip_median_p(event_deltas),
            "interpretation": "Trigger support is admitted for use as event-level dose/context. This registry alone does not establish segmentation gain.",
        },
        "h5_checks": h5_checks,
        "artifacts": {key: str(value) for key, value in output_paths.items()},
    }
    write_json(output_paths["summary"], summary)

    input_manifest = {
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
            if key not in {"chirps_root", "outdir"}
        },
        "chirps_root": str(paths["chirps_root"]),
        "chirps_manifest_sha256": sha256(output_paths["chirps_manifest"]),
        "outputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in output_paths.items()
            if key not in {"source_manifest"}
        },
    }
    write_json(output_paths["source_manifest"], input_manifest)
    print(f"[done] samples={len(sample)} q_R={int(sample['q_R'].sum())} events={int(event_frame['gate_status'].eq('admitted').sum())}")
    print(f"[done] {output_paths['summary']}")


if __name__ == "__main__":
    main()
