#!/usr/bin/env python3
"""Event-isolated probe for Material information beyond Terrain on Sen12.

This is a development diagnostic on patch landslide prevalence, not a
segmentation result. Fixed physical factor families are compared under the
frozen LOGO5 protocol. Test labels never select factors or ridge penalties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from material_factors_v3 import FACTOR_GROUPS, build_material_factors


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_AUDIT = PROJECT_ROOT / "metadata/reports/sen12_material_information_audit_v3_20260722/sample_factors_and_exploratory_targets.csv"
DEFAULT_SPLIT = PROJECT_ROOT / "metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv"
DEFAULT_OUT = PROJECT_ROOT / "experiments/revision2026/sen12_material_increment_probe_v3"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
TERRAIN_FEATURES = (
    "slope_mean_deg",
    "slope_p90_deg",
    "relief_300m_mean_m",
    "roughness_30m_mean_m",
)
MATERIAL_SETS = {
    "T": (),
    "T_AWC": FACTOR_GROUPS["awc_core"],
    "T_SOIL": FACTOR_GROUPS["soil_hydraulic"],
    "T_LITH": FACTOR_GROUPS["lithology_composition"],
    "T_AWC_SOIL": FACTOR_GROUPS["awc_core"] + FACTOR_GROUPS["soil_hydraulic"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def weighted_mse(target: np.ndarray, prediction: np.ndarray, events: Sequence[str]) -> float:
    frame = pd.DataFrame({"y": target, "p": prediction, "event": list(events)})
    return float(frame.assign(se=lambda value: np.square(value.y - value.p)).groupby("event").se.mean().mean())


def within_event_spearman(target: np.ndarray, prediction: np.ndarray, events: Sequence[str]) -> float:
    frame = pd.DataFrame({"y": target, "p": prediction, "event": list(events)})
    frame["yr"] = frame.y - frame.groupby("event").y.transform("mean")
    frame["pr"] = frame.p - frame.groupby("event").p.transform("mean")
    if frame.yr.nunique() < 2 or frame.pr.nunique() < 2:
        return math.nan
    return float(spearmanr(frame.yr, frame.pr).statistic)


def event_weights(events: Sequence[str]) -> np.ndarray:
    counts = pd.Series(list(events)).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[event] for event in events], dtype=np.float64)
    return weights / weights.mean()


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return center, scale


def transform(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(values), values, center)
    return np.clip((clean - center) / scale, -5.0, 5.0)


def design(frame: pd.DataFrame, material_names: Sequence[str]) -> np.ndarray:
    terrain = frame[list(TERRAIN_FEATURES)].to_numpy(dtype=np.float64)
    if not material_names:
        return terrain
    material = frame[list(material_names)].to_numpy(dtype=np.float64)
    interactions = np.concatenate(
        [terrain[:, index : index + 1] * material for index in range(3)], axis=1
    )
    return np.concatenate([terrain, material, interactions], axis=1)


def deterministic_shuffle(frame: pd.DataFrame, columns: Sequence[str], seed: int) -> pd.DataFrame:
    output = frame.copy()
    events = output["physical_event_id"].astype(str).to_numpy()
    regions = output["region_group"].astype(str).to_numpy()
    material = output[list(columns)].to_numpy(copy=True)
    shuffled = np.zeros_like(material)
    for index, sample_id in enumerate(output["sample_id"].astype(str)):
        candidates = np.flatnonzero((events != events[index]) & (regions != regions[index]))
        if not len(candidates):
            candidates = np.flatnonzero(events != events[index])
        if not len(candidates):
            shuffled[index] = material[index]
            continue
        digest = hashlib.sha256(f"{seed}|{sample_id}".encode()).digest()
        donor = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
        shuffled[index] = material[donor]
    output.loc[:, list(columns)] = shuffled
    return output


def select_alpha(
    train: pd.DataFrame,
    val: pd.DataFrame,
    material_names: Sequence[str],
) -> float:
    x_train_raw = design(train, material_names)
    x_val_raw = design(val, material_names)
    center, scale = fit_scaler(x_train_raw)
    x_train = transform(x_train_raw, center, scale)
    x_val = transform(x_val_raw, center, scale)
    y_train = train.target_positive_fraction.to_numpy(dtype=np.float64)
    y_val = val.target_positive_fraction.to_numpy(dtype=np.float64)
    weights = event_weights(train.physical_event_id.astype(str))
    scores = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train, sample_weight=weights)
        prediction = np.clip(model.predict(x_val), 0.0, 1.0)
        score = weighted_mse(y_val, prediction, val.physical_event_id.astype(str))
        scores.append((score, alpha))
    return min(scores)[1]


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.audit_csv, low_memory=False)
    split = pd.read_csv(args.split_csv, low_memory=False)
    split = split[split.sample_id.isin(data.sample_id)].copy()
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for fold in sorted(split.outer_fold.unique()):
        role_map = split[split.outer_fold == fold].set_index("sample_id").role.to_dict()
        fold_data = data[data.sample_id.isin(role_map)].copy()
        fold_data["role"] = fold_data.sample_id.map(role_map)
        train = fold_data[fold_data.role == "train"].copy()
        val = fold_data[fold_data.role == "val"].copy()
        test = fold_data[fold_data.role == "test"].copy()
        for condition, material_names in MATERIAL_SETS.items():
            alpha = select_alpha(train, val, material_names)
            fit = pd.concat([train, val], ignore_index=True)
            x_fit_raw = design(fit, material_names)
            center, scale = fit_scaler(x_fit_raw)
            x_fit = transform(x_fit_raw, center, scale)
            x_test = transform(design(test, material_names), center, scale)
            model = Ridge(alpha=alpha)
            model.fit(
                x_fit,
                fit.target_positive_fraction.to_numpy(dtype=np.float64),
                sample_weight=event_weights(fit.physical_event_id.astype(str)),
            )
            prediction = np.clip(model.predict(x_test), 0.0, 1.0)
            shuffled_prediction = prediction.copy()
            if material_names:
                shuffled = deterministic_shuffle(test, material_names, args.seed + int(fold))
                shuffled_prediction = np.clip(
                    model.predict(transform(design(shuffled, material_names), center, scale)),
                    0.0,
                    1.0,
                )
            target = test.target_positive_fraction.to_numpy(dtype=np.float64)
            events = test.physical_event_id.astype(str).to_numpy()
            rows.append({
                "fold": int(fold),
                "condition": condition,
                "selected_alpha_on_validation": alpha,
                "n_test": int(len(test)),
                "n_test_events": int(pd.Series(events).nunique()),
                "event_balanced_mse": weighted_mse(target, prediction, events),
                "within_event_spearman": within_event_spearman(target, prediction, events),
                "event_balanced_mse_material_shuffle": weighted_mse(target, shuffled_prediction, events),
            })
            for event in sorted(set(events)):
                use = events == event
                event_rows.append({
                    "fold": int(fold),
                    "condition": condition,
                    "physical_event_id": event,
                    "n_samples": int(use.sum()),
                    "mse_aligned": float(np.mean(np.square(target[use] - prediction[use]))),
                    "mse_material_shuffle": float(np.mean(np.square(target[use] - shuffled_prediction[use]))),
                })

    fold_metrics = pd.DataFrame(rows)
    per_event = pd.DataFrame(event_rows)
    baseline = per_event[per_event.condition == "T"][["physical_event_id", "mse_aligned"]].rename(columns={"mse_aligned": "mse_T"})
    per_event = per_event.merge(baseline, on="physical_event_id", validate="many_to_one")
    per_event["mse_gain_vs_T"] = per_event.mse_T - per_event.mse_aligned
    per_event["mse_gain_vs_shuffle"] = per_event.mse_material_shuffle - per_event.mse_aligned
    fold_metrics.to_csv(args.outdir / "fold_metrics.csv", index=False)
    per_event.to_csv(args.outdir / "per_event_metrics.csv", index=False)

    summary_rows = []
    for condition, group in per_event.groupby("condition", sort=False):
        summary_rows.append({
            "condition": condition,
            "n_events": int(group.physical_event_id.nunique()),
            "mean_event_mse": float(group.mse_aligned.mean()),
            "mean_event_mse_gain_vs_T": float(group.mse_gain_vs_T.mean()),
            "positive_events_vs_T": int((group.mse_gain_vs_T > 0).sum()),
            "mean_event_mse_gain_vs_shuffle": float(group.mse_gain_vs_shuffle.mean()),
            "positive_events_vs_shuffle": int((group.mse_gain_vs_shuffle > 0).sum()),
        })
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(args.outdir / "summary_table.csv", index=False)
    best = summary_frame.sort_values("mean_event_mse_gain_vs_T", ascending=False).iloc[0].to_dict()
    summary = {
        "scope": "Development-only event-isolated Material increment probe",
        "target": "patch landslide-positive fraction; not segmentation IoU",
        "n_samples": int(data.sample_id.nunique()),
        "n_events": int(data.physical_event_id.nunique()),
        "best_condition_by_mean_event_mse_gain_vs_T": best,
        "promotion_contract": "A factor family may advance to segmentation only if it improves over T and aligned beats event/region-shuffled Material; this probe alone is not manuscript evidence.",
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Sen12 Material increment probe v3",
        "",
        "This is a development diagnostic on patch landslide prevalence, not a segmentation result.",
        "",
        "| Condition | Events | Mean event MSE | Gain vs T | Positive events vs T | Aligned gain vs shuffled M |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_frame.itertuples():
        lines.append(
            f"| {row.condition} | {row.n_events} | {row.mean_event_mse:.6f} | "
            f"{row.mean_event_mse_gain_vs_T:+.6f} | {row.positive_events_vs_T}/{row.n_events} | "
            f"{row.mean_event_mse_gain_vs_shuffle:+.6f} |"
        )
    lines += [
        "",
        "A Material family is not promoted merely for a positive mean. It must also beat its shuffled control and then pass the separate segmentation gate with exact Terrain fallback.",
        "",
    ]
    (args.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
