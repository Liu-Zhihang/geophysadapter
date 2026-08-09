#!/usr/bin/env python3
"""Build label-free CHIRPS antecedent rainfall support for frozen Sen12 samples."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import rowcol
from rasterio.windows import Window
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (3, 7, 14, 30)
ANCHOR_SHIFTS = {"case": 0, "wrong_m56": -56, "wrong_m28": -28, "wrong_p28": 28, "wrong_p56": 56}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(type(value).__name__)


def chirps_path(root: Path, day: pd.Timestamp) -> Path:
    return root / f"{day.year:04d}" / f"chirps-v2.0.{day.year:04d}.{day.month:02d}.{day.day:02d}.tif.gz"


def trigger_family_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, low_memory=False)
    output: dict[str, str] = {}
    for cluster, group in frame.groupby("physical_event_cluster_id"):
        values = [str(value) for value in group["trigger_family"].dropna() if str(value) not in {"", "unknown"}]
        output[str(cluster)] = values[0] if values else "unknown"
    return output


def sample_native_3x3(path: Path, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    compressed = path.read_bytes()
    digest = sha256_bytes(compressed)
    uncompressed = gzip.decompress(compressed)
    with MemoryFile(uncompressed) as memory:
        with memory.open() as source:
            if source.count != 1 or source.crs is None or not source.crs.is_geographic:
                raise RuntimeError(f"Invalid CHIRPS raster contract: {path}")
            if not (
                math.isclose(abs(float(source.transform.a)), 0.05, abs_tol=1e-8)
                and math.isclose(abs(float(source.transform.e)), 0.05, abs_tol=1e-8)
            ):
                raise RuntimeError(f"Unexpected CHIRPS resolution: {path}")
            rows, columns = rowcol(source.transform, lon, lat)
            rows = np.asarray(rows, dtype=int)
            columns = np.asarray(columns, dtype=int)
            if np.any(rows < 0) or np.any(rows >= source.height) or np.any(columns < 0) or np.any(columns >= source.width):
                raise RuntimeError(f"Sen12 sample outside CHIRPS coverage: {path}")
            row0 = max(0, int(rows.min()) - 1)
            row1 = min(source.height, int(rows.max()) + 2)
            col0 = max(0, int(columns.min()) - 1)
            col1 = min(source.width, int(columns.max()) + 2)
            block = source.read(1, window=Window(col0, row0, col1 - col0, row1 - row0), out_dtype="float64")
            local_rows = rows - row0
            local_columns = columns - col0
            neighborhood = np.full((len(rows), 9), np.nan, dtype=np.float64)
            position = 0
            for delta_row in (-1, 0, 1):
                for delta_column in (-1, 0, 1):
                    rr = local_rows + delta_row
                    cc = local_columns + delta_column
                    inside = (rr >= 0) & (rr < block.shape[0]) & (cc >= 0) & (cc < block.shape[1])
                    neighborhood[inside, position] = block[rr[inside], cc[inside]]
                    position += 1
            valid = np.isfinite(neighborhood) & (neighborhood >= 0.0)
            if source.nodata is not None and math.isfinite(float(source.nodata)):
                valid &= ~np.isclose(neighborhood, float(source.nodata), atol=1e-8, rtol=0.0)
            safe = np.where(valid, neighborhood, np.nan)
            with np.errstate(invalid="ignore"):
                values = np.nanmedian(safe, axis=1)
            metadata = {
                "date_path": str(path),
                "sha256": digest,
                "bytes": len(compressed),
                "width": int(source.width),
                "height": int(source.height),
                "crs": str(source.crs),
                "resolution_degrees": [abs(float(source.transform.a)), abs(float(source.transform.e))],
            }
    return values, metadata


def exact_signflip_median_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return 1.0
    observed = abs(float(np.median(values)))
    statistics = []
    for mask in range(1 << values.size):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(values.size)])
        statistics.append(abs(float(np.median(values * signs))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def bootstrap_median_ci(values: np.ndarray, seed: int, draws: int = 20000) -> list[float | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(draws, values.size), replace=True), axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/sen12_trigger_support_v2"),
    )
    parser.add_argument(
        "--event-registry",
        type=Path,
        default=Path("metadata/pild_xdomain_v1/sen12_trigger_event_anchor_registry_v1.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    event_registry_path = args.event_registry if args.event_registry.is_absolute() else root / args.event_registry
    outdir.mkdir(parents=True, exist_ok=True)

    cache_path = root / "processed/hybrid_pinn/sen12_s2_xdomain_v1/cache_index_v1.csv"
    registry_path = root / "metadata/pild_xdomain_v1/sen12_s2_sample_registry_v1.csv"
    candidate_path = root / "metadata/pild_xdomain_v1/candidate_event_registry_v1.csv"
    chirps_root = root / "raw_fullcopy/weather/chirps_daily_global"
    cache = pd.read_csv(cache_path, low_memory=False)
    registry = pd.read_csv(registry_path, low_memory=False)
    frame = cache.merge(
        registry[[
            "sample_id", "region", "center_lon", "center_lat", "event_date_start",
            "date_quality", "physical_event_cluster_id",
        ]],
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_registry"),
    )
    quality = "date_quality_registry" if "date_quality_registry" in frame else "date_quality"
    family = trigger_family_map(candidate_path)
    frame["trigger_family"] = frame["physical_event_cluster_id"].map(family).fillna("unknown")
    rainfall = frame[(frame[quality] == "high_single_event") & (frame["trigger_family"] == "rainfall")].copy()
    if rainfall["physical_event_cluster_id"].nunique() != 6:
        raise RuntimeError("Expected six high-confidence rainfall event clusters")
    event_registry = pd.read_csv(event_registry_path, low_memory=False)
    if event_registry["physical_event_cluster_id"].duplicated().any():
        raise RuntimeError("Trigger event registry contains duplicate physical_event_cluster_id values")
    expected_clusters = set(rainfall["physical_event_cluster_id"].astype(str))
    registry_clusters = set(event_registry["physical_event_cluster_id"].astype(str))
    if expected_clusters != registry_clusters:
        raise RuntimeError(
            f"Trigger event registry mismatch: missing={sorted(expected_clusters - registry_clusters)} "
            f"extra={sorted(registry_clusters - expected_clusters)}"
        )
    rainfall = rainfall.merge(
        event_registry,
        on=["physical_event_cluster_id", "region", "trigger_family"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_anchor"),
    )
    if rainfall["trigger_anchor_date"].isna().any():
        raise RuntimeError("Missing trigger_anchor_date after event-registry merge")

    feature_rows: list[pd.DataFrame] = []
    event_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for (cluster, event_date, trigger_anchor_date), group in rainfall.groupby(
        ["physical_event_cluster_id", "event_date_start", "trigger_anchor_date"], sort=True
    ):
        group = group.sort_values("sample_id").copy()
        dataset_event_day = pd.Timestamp(event_date)
        event_day = pd.Timestamp(trigger_anchor_date)
        lon = group["center_lon"].to_numpy(dtype=float)
        lat = group["center_lat"].to_numpy(dtype=float)
        anchor_matrices: dict[str, np.ndarray] = {}
        for anchor_role, shift in ANCHOR_SHIFTS.items():
            # Offset 0 is the independently documented impact date. Keeping 31
            # days allows both D0..D-(w-1) and strict D-1..D-w windows.
            matrix = np.full((len(group), 31), np.nan, dtype=np.float64)
            anchor = event_day + timedelta(days=shift)
            for lag in range(0, 31):
                day = anchor - timedelta(days=lag)
                path = chirps_path(chirps_root, day)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing required CHIRPS input: {path}")
                values, metadata = sample_native_3x3(path, lon, lat)
                matrix[:, lag] = values
                manifest_rows.append(
                    {
                        "physical_event_cluster_id": cluster,
                        "anchor_role": anchor_role,
                        "lag_days": lag,
                        "date": day.date().isoformat(),
                        **metadata,
                    }
                )
            anchor_matrices[anchor_role] = matrix
        output = group[["sample_id", "region", "physical_event_cluster_id", "event_date_start"]].copy()
        output["trigger_anchor_date"] = event_day.date().isoformat()
        output["trigger_anchor_role"] = str(group["anchor_role"].iloc[0])
        output["trigger_anchor_confidence"] = str(group["anchor_confidence"].iloc[0])
        output["trigger_family"] = "rainfall"
        output["q_R_case"] = np.all(np.isfinite(anchor_matrices["case"]), axis=1).astype(float)
        output["q_R_controls"] = np.all(
            np.column_stack([np.all(np.isfinite(anchor_matrices[role]), axis=1) for role in ANCHOR_SHIFTS if role != "case"]),
            axis=1,
        ).astype(float)
        for window in WINDOWS:
            case_inclusive = np.sum(anchor_matrices["case"][:, :window], axis=1)
            case_antecedent = np.sum(anchor_matrices["case"][:, 1 : window + 1], axis=1)
            controls_inclusive = np.column_stack(
                [np.sum(anchor_matrices[role][:, :window], axis=1) for role in ANCHOR_SHIFTS if role != "case"]
            )
            controls_antecedent = np.column_stack(
                [np.sum(anchor_matrices[role][:, 1 : window + 1], axis=1) for role in ANCHOR_SHIFTS if role != "case"]
            )
            control_inclusive_median = np.median(controls_inclusive, axis=1)
            control_antecedent_median = np.median(controls_antecedent, axis=1)
            output[f"rain_d{window}_inclusive_case_mm"] = case_inclusive
            output[f"rain_d{window}_inclusive_wrongtime_median_mm"] = control_inclusive_median
            output[f"rain_d{window}_inclusive_delta_mm"] = case_inclusive - control_inclusive_median
            output[f"rain_d{window}_antecedent_case_mm"] = case_antecedent
            output[f"rain_d{window}_antecedent_wrongtime_median_mm"] = control_antecedent_median
            output[f"rain_d{window}_antecedent_delta_mm"] = case_antecedent - control_antecedent_median
        output["rain_max1d_d30_inclusive_mm"] = np.max(anchor_matrices["case"][:, :30], axis=1)
        output["rain_api09_d30_mm"] = np.sum(
            anchor_matrices["case"][:, :30] * np.power(0.9, np.arange(30, dtype=float))[None, :], axis=1
        )
        feature_rows.append(output)
        event_record: dict[str, object] = {
            "physical_event_cluster_id": cluster,
            "dataset_event_date": dataset_event_day.date().isoformat(),
            "trigger_anchor_date": event_day.date().isoformat(),
            "region": str(group["region"].iloc[0]),
            "n_samples": len(group),
            "q_R_case_fraction": float(output["q_R_case"].mean()),
            "q_R_controls_fraction": float(output["q_R_controls"].mean()),
        }
        for window in WINDOWS:
            for timing in ("inclusive", "antecedent"):
                event_record[f"median_rain_d{window}_{timing}_case_mm"] = float(
                    output[f"rain_d{window}_{timing}_case_mm"].median()
                )
                event_record[f"median_rain_d{window}_{timing}_wrongtime_mm"] = float(
                    output[f"rain_d{window}_{timing}_wrongtime_median_mm"].median()
                )
                event_record[f"median_rain_d{window}_{timing}_delta_mm"] = float(
                    output[f"rain_d{window}_{timing}_delta_mm"].median()
                )
        event_rows.append(event_record)
        print(f"[event] {cluster} samples={len(group)}", flush=True)

    rainfall_features = pd.concat(feature_rows, ignore_index=True)
    sample_support = frame[[
        "sample_id", "region", "physical_event_cluster_id", "event_date_start", quality,
        "trigger_family",
    ]].copy()
    sample_support = sample_support.merge(
        rainfall_features.drop(columns=["region", "physical_event_cluster_id", "event_date_start", "trigger_family"]),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    sample_support["q_R"] = sample_support["q_R_case"].fillna(0.0)
    sample_support["q_R_reason"] = np.select(
        [
            sample_support["q_R"] > 0,
            (sample_support[quality] == "high_single_event") & (sample_support["trigger_family"] == "earthquake"),
            sample_support[quality] != "high_single_event",
        ],
        ["rainfall_case_complete", "earthquake_pending_usgs_pga", "date_not_high_single_event"],
        default="trigger_family_unresolved",
    )
    event_frame = pd.DataFrame(event_rows).sort_values("physical_event_cluster_id")
    sample_support.to_csv(outdir / "sen12_trigger_sample_support_v1.csv", index=False)
    rainfall_features.to_csv(outdir / "sen12_rainfall_sample_features_v1.csv", index=False)
    event_frame.to_csv(outdir / "sen12_rainfall_event_features_v1.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(outdir / "chirps_source_manifest_v1.csv", index=False)

    d7 = event_frame["median_rain_d7_inclusive_delta_mm"].to_numpy(dtype=float)
    d7_antecedent = event_frame["median_rain_d7_antecedent_delta_mm"].to_numpy(dtype=float)
    try:
        wilcoxon_p = float(wilcoxon(d7).pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    summary = {
        "n_frozen_samples": len(frame),
        "n_rainfall_supported_samples": len(rainfall_features),
        "n_rainfall_events": len(event_frame),
        "n_earthquake_events_pending_usgs": int(
            frame.loc[
                (frame[quality] == "high_single_event") & (frame["trigger_family"] == "earthquake"),
                "physical_event_cluster_id",
            ].nunique()
        ),
        "d7_event_median_delta_mm": float(np.median(d7)),
        "d7_event_positive_count": int(np.sum(d7 > 0)),
        "d7_event_bootstrap_median_ci95_mm": bootstrap_median_ci(d7, args.seed),
        "d7_event_exact_signflip_median_p": exact_signflip_median_p(d7),
        "d7_event_wilcoxon_p": wilcoxon_p,
        "d7_antecedent_event_median_delta_mm": float(np.median(d7_antecedent)),
        "d7_antecedent_event_positive_count": int(np.sum(d7_antecedent > 0)),
        "d7_antecedent_event_bootstrap_median_ci95_mm": bootstrap_median_ci(d7_antecedent, args.seed),
        "d7_antecedent_event_exact_signflip_median_p": exact_signflip_median_p(d7_antecedent),
        "label_free_contract": (
            "Inputs are frozen sample/event coordinates, event dates, trigger-family registry, "
            "and CHIRPS rasters; segmentation labels, predictions, logits, and checkpoints are not read."
        ),
        "window_contract": {
            "inclusive": "D0 through D-(window-1), where D0 is the source-documented impact date",
            "antecedent": "D-1 through D-window relative to the same fixed impact date",
            "selection": "Both are reported; neither is selected using labels or model outcomes.",
        },
        "event_registry": str(event_registry_path),
        "artifacts": {
            "sample_support": str(outdir / "sen12_trigger_sample_support_v1.csv"),
            "rainfall_sample_features": str(outdir / "sen12_rainfall_sample_features_v1.csv"),
            "rainfall_event_features": str(outdir / "sen12_rainfall_event_features_v1.csv"),
            "source_manifest": str(outdir / "chirps_source_manifest_v1.csv"),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    (outdir / "report.md").write_text(
        "# Sen12 rainfall Trigger support v2\n\n"
        f"- Rainfall-supported samples: `{len(rainfall_features)}` across `{len(event_frame)}` events.\n"
        f"- Event-median inclusive D7 case-minus-wrong-time rainfall: `{np.median(d7):.3f} mm`.\n"
        f"- Inclusive D7 positive events: `{np.sum(d7 > 0)}/{len(d7)}`.\n"
        f"- Event-median strict-antecedent D7 delta: `{np.median(d7_antecedent):.3f} mm`.\n"
        f"- Strict-antecedent D7 positive events: `{np.sum(d7_antecedent > 0)}/{len(d7_antecedent)}`.\n"
        "- This is label-free Trigger support, not proof of segmentation gain.\n"
    )
    print(f"[done] {outdir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
