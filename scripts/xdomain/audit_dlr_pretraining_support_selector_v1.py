#!/usr/bin/env python3
"""Audit whether DLR Terrain responders are identifiable before training.

Only label-free acquisition, support-quality, and registration variables are
allowed as selector inputs. Model outputs, labels, event identity, source
identity, and coordinates are excluded. Response outcomes are used only on
development events to fit/select a rule; each outer fold is evaluated once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline


DEFAULT_RUNS = Path(
    "experiments/revision2026/"
    "dlr_geo4qc_sen12_exactcommon9_scratch_seed20260724_v4"
)
DEFAULT_REGISTRATION = Path(
    "metadata/reports/dlr_fabdem_registration_v3/per_sample.csv"
)
DEFAULT_OPTICAL = Path(
    "processed/hybrid_pinn/dlr_geo4qc_sen12_exactcommon9_v1/"
    "dlr_prithvi_4t6b_p128.h5"
)
DEFAULT_ELIGIBILITY = Path(
    "metadata/pild_sen12_training_v2/support_eligibility_v1/"
    "sample_eligibility_v1.csv"
)

# All variables are available from imagery/support metadata before model
# training. Event/source identifiers and coordinates are intentionally absent.
PRETRAINING_FEATURES = (
    "optical_valid_fraction",
    "optical_finite_fraction",
    "optical_zero_fraction",
    "optical_saturation_fraction",
    "optical_spatial_contrast",
    "cloud_mean_fraction",
    "cloud_max_fraction",
    "temporal_unique_fraction",
    "terrain_valid_fraction",
    "terrain_finite_fraction",
    "terrain_slope_mean_deg",
    "terrain_slope_std_deg",
    "terrain_relief_mean_m",
    "terrain_relief_std_m",
    "q_M",
    "q_M_full",
    "q_M_hydraulic",
    "q_M_geology",
    "q_M_soil",
    "material_valid_property_fraction",
    "material_varying_property_fraction",
    "material_lithology_varies",
    "q_R",
    "trigger_date_unique",
    "trigger_case_coverage",
    "trigger_control_coverage",
    "trigger_case_minus_wrongtime_mm",
    "hard_quality_eligible",
    "qT_eligible",
    "qM_eligible",
    "qR_eligible",
    "full_tmr_eligible",
    "visual_degraded_cloud",
    "visual_degraded_temporal",
    "visual_degraded_low_contrast",
    "visual_degraded",
    "terrain_mechanism_opportunity",
    "full_tmr_mechanism_opportunity",
    "registration_abs_dx",
    "registration_abs_dy",
    "registration_linf_shift",
    "registration_l2_shift",
    "registration_best_score",
    "registration_zero_score",
    "registration_score_gain",
    "registration_best_slope_correlation",
    "registration_best_highpass_elevation_correlation",
    "registration_zero_slope_correlation",
    "registration_zero_highpass_elevation_correlation",
    "snow_fraction_t0",
    "snow_fraction_t3",
    "snow_fraction_mean",
    "snow_fraction_max",
    "snow_fraction_range",
    "snow_turnover_t0_t3",
    "snow_turnover_any",
    "ndsi_abs_change_t0_t3",
    "ndvi_abs_change_t0_t3",
    "nbr_abs_change_t0_t3",
    "reflectance_abs_change_t0_t3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--registration-csv", type=Path, default=DEFAULT_REGISTRATION
    )
    parser.add_argument("--optical-h5", type=Path, default=DEFAULT_OPTICAL)
    parser.add_argument(
        "--per-sample-csv",
        type=Path,
        default=None,
        help="Optional alternate evaluator per-sample table.",
    )
    parser.add_argument("--eligibility-csv", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--min-selected", type=int, default=30)
    return parser.parse_args()


def aggregate(rows: pd.DataFrame, selected: np.ndarray) -> dict[str, float | int]:
    selected_rows = rows.loc[selected]
    fallback_rows = rows.loc[~selected]
    visual_tp = int(rows["visual_tp"].sum())
    visual_fp = int(rows["visual_fp"].sum())
    visual_fn = int(rows["visual_fn"].sum())
    adapted_tp = int(
        selected_rows["adapted_tp"].sum() + fallback_rows["visual_tp"].sum()
    )
    adapted_fp = int(
        selected_rows["adapted_fp"].sum() + fallback_rows["visual_fp"].sum()
    )
    adapted_fn = int(
        selected_rows["adapted_fn"].sum() + fallback_rows["visual_fn"].sum()
    )
    visual_errors = visual_fp + visual_fn
    adapted_errors = adapted_fp + adapted_fn
    visual_iou = visual_tp / max(visual_tp + visual_fp + visual_fn, 1)
    adapted_iou = adapted_tp / max(adapted_tp + adapted_fp + adapted_fn, 1)
    selected_events = selected_rows.groupby("event_id")["net_corrected"].sum()
    return {
        "n_selected": int(selected.sum()),
        "coverage_fraction": float(selected.mean()) if len(selected) else 0.0,
        "n_selected_events": int(selected_rows["event_id"].nunique()),
        "n_positive_events": int((selected_events > 0).sum()),
        "positive_event_fraction": (
            float((selected_events > 0).mean()) if len(selected_events) else 0.0
        ),
        "delta_iou": adapted_iou - visual_iou,
        "rer": (visual_errors - adapted_errors) / max(visual_errors, 1),
        "net_corrected": visual_errors - adapted_errors,
    }


def estimator(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=10,
            max_features=0.7,
            random_state=seed,
            n_jobs=1,
        ),
    )


def event_weights(rows: pd.DataFrame) -> np.ndarray:
    counts = rows.groupby("event_id")["sample_id"].transform("count")
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights * len(weights) / weights.sum()


def fit(
    model,
    rows: pd.DataFrame,
    features: list[str],
    target: str,
) -> None:
    model.fit(
        rows[features],
        rows[target],
        randomforestregressor__sample_weight=event_weights(rows),
    )


def inner_event_predictions(
    rows: pd.DataFrame,
    features: list[str],
    target: str,
    seed: int,
) -> np.ndarray:
    events = sorted(rows["event_id"].unique())
    predictions = np.full(len(rows), np.nan, dtype=float)
    # Deterministic round-robin event folds avoid sample-count leakage.
    for inner_fold in range(min(5, len(events))):
        held_events = set(events[inner_fold::5])
        held = rows["event_id"].isin(held_events).to_numpy()
        development = ~held
        model = estimator(seed + inner_fold)
        fit(model, rows.loc[development], features, target)
        predictions[held] = model.predict(rows.loc[held, features])
    if not np.isfinite(predictions).all():
        raise RuntimeError("inner event predictions are incomplete")
    return predictions


def safe_auc(target: pd.Series, values: pd.Series) -> float | None:
    finite = target.notna() & values.notna()
    if target.loc[finite].nunique() < 2 or values.loc[finite].nunique() < 2:
        return None
    auc = float(roc_auc_score(target.loc[finite], values.loc[finite]))
    return max(auc, 1.0 - auc)


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def normalized_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left - right) / np.maximum(left + right, 1e-4)


def optical_temporal_features(path: Path) -> pd.DataFrame:
    records = []
    with h5py.File(path, "r") as handle:
        sample_ids = decode(handle["sample_id"][:])
        for index, sample_id in enumerate(sample_ids):
            optical = np.asarray(handle["optical"][index], dtype=np.float32) / 10000.0
            valid = np.asarray(handle["optical_valid"][index, 0], dtype=bool)
            green = optical[1]
            red = optical[2]
            nir = optical[3]
            swir1 = optical[4]
            swir2 = optical[5]
            ndsi = normalized_difference(green, swir1)
            ndvi = normalized_difference(nir, red)
            nbr = normalized_difference(nir, swir2)
            # The standard NDSI threshold is paired with a reflectance floor to
            # avoid classifying dark water/shadow as snow.
            snow = (ndsi > 0.40) & (green > 0.10) & valid[None]
            snow_fraction = snow.sum(axis=(1, 2)) / max(valid.sum(), 1)
            records.append(
                {
                    "sample_id": sample_id,
                    "snow_fraction_t0": float(snow_fraction[0]),
                    "snow_fraction_t3": float(snow_fraction[3]),
                    "snow_fraction_mean": float(snow_fraction.mean()),
                    "snow_fraction_max": float(snow_fraction.max()),
                    "snow_fraction_range": float(
                        snow_fraction.max() - snow_fraction.min()
                    ),
                    "snow_turnover_t0_t3": float(
                        np.logical_xor(snow[0], snow[3])[valid].mean()
                    ),
                    "snow_turnover_any": float(
                        (snow.any(axis=0) & ~snow.all(axis=0))[valid].mean()
                    ),
                    "ndsi_abs_change_t0_t3": float(
                        np.abs(ndsi[3] - ndsi[0])[valid].mean()
                    ),
                    "ndvi_abs_change_t0_t3": float(
                        np.abs(ndvi[3] - ndvi[0])[valid].mean()
                    ),
                    "nbr_abs_change_t0_t3": float(
                        np.abs(nbr[3] - nbr[0])[valid].mean()
                    ),
                    "reflectance_abs_change_t0_t3": float(
                        np.abs(optical[:, 3] - optical[:, 0])[:, valid].mean()
                    ),
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    outdir = args.outdir or args.runs_dir / "pretraining_support_selector_audit_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    source = args.per_sample_csv or (
        args.runs_dir
        / "responder_profile_analysis_v1/all_test_samples_with_support.csv"
    )
    rows = pd.read_csv(source)
    if "dataset_id" not in rows:
        eligibility = pd.read_csv(args.eligibility_csv)
        eligibility = eligibility.loc[
            eligibility["dataset_id"].eq("DLR_Landslide_Ref_2025")
        ].drop_duplicates("sample_id")
        rows = rows.merge(
            eligibility,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    registration = pd.read_csv(args.registration_csv).rename(
        columns={
            "best_score": "registration_best_score",
            "zero_score": "registration_zero_score",
            "score_gain": "registration_score_gain",
            "best_slope_correlation": "registration_best_slope_correlation",
            "best_highpass_elevation_correlation": (
                "registration_best_highpass_elevation_correlation"
            ),
            "zero_slope_correlation": "registration_zero_slope_correlation",
            "zero_highpass_elevation_correlation": (
                "registration_zero_highpass_elevation_correlation"
            ),
        }
    )
    registration["registration_abs_dx"] = registration["best_dx_pixels"].abs()
    registration["registration_abs_dy"] = registration["best_dy_pixels"].abs()
    registration["registration_linf_shift"] = registration[
        ["registration_abs_dx", "registration_abs_dy"]
    ].max(axis=1)
    registration["registration_l2_shift"] = np.hypot(
        registration["registration_abs_dx"], registration["registration_abs_dy"]
    )
    registration_columns = [
        "sample_id",
        *[name for name in PRETRAINING_FEATURES if name.startswith("registration_")],
    ]
    rows = rows.merge(
        registration[registration_columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    rows = rows.merge(
        optical_temporal_features(args.optical_h5),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    rows["response_target"] = rows["net_corrected"] / rows["visual_errors"].clip(
        lower=1
    )
    rows["high_responder"] = rows["rer"].ge(0.10).astype(int)

    features = [
        name
        for name in PRETRAINING_FEATURES
        if name in rows
        and rows[name].notna().sum() >= 50
        and rows[name].nunique(dropna=True) >= 2
    ]
    if not features:
        raise RuntimeError("no eligible pretraining features")

    association_rows = []
    event_centered_target = rows["response_target"] - rows.groupby("event_id")[
        "response_target"
    ].transform("mean")
    for feature in features:
        finite = rows[[feature, "response_target"]].dropna()
        rho, p_value = spearmanr(finite[feature], finite["response_target"])
        centered_feature = rows[feature] - rows.groupby("event_id")[feature].transform(
            "mean"
        )
        centered = pd.DataFrame(
            {"feature": centered_feature, "target": event_centered_target}
        ).dropna()
        centered_rho, centered_p = spearmanr(
            centered["feature"], centered["target"]
        )
        association_rows.append(
            {
                "feature": feature,
                "n_finite": int(len(finite)),
                "orientation_free_auc_high_responder": safe_auc(
                    rows["high_responder"], rows[feature]
                ),
                "sample_spearman_rho": (
                    None if not np.isfinite(rho) else float(rho)
                ),
                "sample_spearman_p": (
                    None if not np.isfinite(p_value) else float(p_value)
                ),
                "within_event_spearman_rho": (
                    None if not np.isfinite(centered_rho) else float(centered_rho)
                ),
                "within_event_spearman_p": (
                    None if not np.isfinite(centered_p) else float(centered_p)
                ),
                "high_responder_median": float(
                    rows.loc[rows["high_responder"].eq(1), feature].median()
                ),
                "other_median": float(
                    rows.loc[rows["high_responder"].eq(0), feature].median()
                ),
            }
        )
    associations = pd.DataFrame(association_rows).sort_values(
        ["orientation_free_auc_high_responder", "within_event_spearman_rho"],
        ascending=False,
        na_position="last",
    )
    associations.to_csv(outdir / "pretraining_feature_associations.csv", index=False)

    fixed_rule_rows = []
    fixed_rules = {
        **{
            f"snow_turnover_t0_t3_ge_{int(threshold * 100):02d}pct": rows[
                "snow_turnover_t0_t3"
            ].ge(threshold)
            for threshold in (0.01, 0.05, 0.10, 0.20)
        },
        **{
            f"snow_turnover_any_ge_{int(threshold * 100):02d}pct": rows[
                "snow_turnover_any"
            ].ge(threshold)
            for threshold in (0.01, 0.05, 0.10, 0.20)
        },
        "snow_turnover_10pct_and_terrain_valid": (
            rows["snow_turnover_t0_t3"].ge(0.10)
            & rows["terrain_valid_fraction"].ge(0.99)
        ),
    }
    for name, mask in fixed_rules.items():
        fixed_rule_rows.append(
            {
                "rule": name,
                "scientific_status": (
                    "label-free fixed diagnostic; not prospectively registered"
                ),
                **aggregate(rows, mask.to_numpy(dtype=bool)),
            }
        )
    pd.DataFrame(fixed_rule_rows).to_csv(
        outdir / "fixed_snow_support_rules.csv", index=False
    )

    diagnostics = []
    prediction_rows = []
    thresholds = np.linspace(0.50, 0.90, 9)
    for outer_fold in range(5):
        development = rows.loc[rows["fold"].ne(outer_fold)].reset_index(drop=True)
        outer = rows.loc[rows["fold"].eq(outer_fold)].copy()
        inner_prediction = inner_event_predictions(
            development, features, "response_target", args.seed + outer_fold * 10
        )
        correlation = float(
            np.corrcoef(inner_prediction, development["response_target"])[0, 1]
        )
        candidates = []
        for quantile in thresholds:
            threshold = float(np.quantile(inner_prediction, quantile))
            selected = inner_prediction >= threshold
            metrics = aggregate(development, selected)
            if (
                metrics["n_selected"] >= args.min_selected
                and metrics["n_selected_events"] >= 5
                and metrics["positive_event_fraction"] >= 0.60
                and metrics["delta_iou"] > 0
                and metrics["rer"] > 0
            ):
                candidates.append(
                    {"score_threshold": threshold, "quantile": quantile, **metrics}
                )
        outer["selector_score"] = np.nan
        outer["selector_selected"] = 0
        if not candidates:
            diagnostics.append(
                {
                    "fold": outer_fold,
                    "status": "abstain_no_inner_event_stable_rule",
                    "inner_prediction_correlation": correlation,
                }
            )
            prediction_rows.append(outer)
            continue
        rule = max(candidates, key=lambda item: (item["rer"], item["delta_iou"]))
        model = estimator(args.seed + outer_fold * 10)
        fit(model, development, features, "response_target")
        score = model.predict(outer[features])
        selected = score >= rule["score_threshold"]
        metrics = aggregate(outer.reset_index(drop=True), selected)
        outer["selector_score"] = score
        outer["selector_selected"] = selected.astype(int)
        diagnostics.append(
            {
                "fold": outer_fold,
                "status": "frozen_rule_applied",
                "inner_prediction_correlation": correlation,
                **{f"inner_{key}": value for key, value in rule.items()},
                **{f"outer_{key}": value for key, value in metrics.items()},
            }
        )
        prediction_rows.append(outer)

    diagnostics_frame = pd.DataFrame(diagnostics)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    selected = predictions["selector_selected"].to_numpy(dtype=bool)
    cross_fitted = aggregate(predictions, selected)
    diagnostics_frame.to_csv(outdir / "per_fold_diagnostics.csv", index=False)
    predictions.to_csv(outdir / "cross_fitted_predictions.csv", index=False)

    high = rows.loc[rows["high_responder"].eq(1)]
    event_concentration = (
        high.groupby(["event_id", "region"], as_index=False)
        .agg(
            n_samples=("sample_id", "size"),
            net_corrected=("net_corrected", "sum"),
            mean_delta_iou=("delta_iou", "mean"),
            mean_rer=("rer", "mean"),
        )
        .sort_values("n_samples", ascending=False)
    )
    event_concentration.to_csv(outdir / "high_responder_event_concentration.csv", index=False)

    status = (
        "no_go"
        if diagnostics_frame["status"]
        .eq("abstain_no_inner_event_stable_rule")
        .all()
        else (
            "candidate"
            if cross_fitted["delta_iou"] > 0
            and cross_fitted["rer"] > 0
            and cross_fitted["n_selected_events"] >= 5
            else "no_go"
        )
    )
    summary = {
        "status": status,
        "scientific_status": (
            "exploratory nested audit; promotion requires a prospectively frozen "
            "rule on untouched events or an external dataset"
        ),
        "selector_input_contract": (
            "pretraining label-free acquisition/support/registration variables only"
        ),
        "minimum_selected_samples": int(args.min_selected),
        "n_samples": int(len(rows)),
        "n_events": int(rows["event_id"].nunique()),
        "n_high_responders": int(rows["high_responder"].sum()),
        "n_high_responder_events": int(high["event_id"].nunique()),
        "largest_event_share_of_high_responders": float(
            event_concentration.iloc[0]["n_samples"] / max(len(high), 1)
        ),
        "features": features,
        "n_abstained_folds": int(
            diagnostics_frame["status"]
            .eq("abstain_no_inner_event_stable_rule")
            .sum()
        ),
        "inner_prediction_correlation_range": [
            float(diagnostics_frame["inner_prediction_correlation"].min()),
            float(diagnostics_frame["inner_prediction_correlation"].max()),
        ],
        "cross_fitted_selective_metrics": cross_fitted,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    top = associations.head(8)
    lines = [
        "# DLR pretraining support selector audit v1",
        "",
        f"- Status: **{status}**",
        f"- Samples/events: {len(rows)}/{rows['event_id'].nunique()}",
        (
            f"- High responders: {len(high)} samples from "
            f"{high['event_id'].nunique()} events; largest-event share "
            f"{summary['largest_event_share_of_high_responders']:.1%}."
        ),
        (
            f"- Outer folds abstained: {summary['n_abstained_folds']}/5; "
            f"cross-fitted selected coverage {cross_fitted['coverage_fraction']:.1%}, "
            f"DeltaIoU={cross_fitted['delta_iou']:+.6f}, "
            f"RER={cross_fitted['rer']:+.2%}."
        ),
        "",
        "## Input contract",
        "",
        "- Inputs are computable before model training and do not use labels, model",
        "  predictions, correctness, event/source identity, or coordinates.",
        "- Outcome-selected high responders are used only to diagnose whether a",
        "  transferable pretraining signature exists.",
        "",
        "## Strongest univariate diagnostics",
        "",
        "| Feature | AUC (orientation free) | Within-event rho |",
        "|---|---:|---:|",
    ]
    for row in top.itertuples():
        auc = "NA" if pd.isna(row.orientation_free_auc_high_responder) else f"{row.orientation_free_auc_high_responder:.3f}"
        rho = "NA" if pd.isna(row.within_event_spearman_rho) else f"{row.within_event_spearman_rho:+.3f}"
        lines.append(f"| {row.feature} | {auc} | {rho} |")
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "- `candidate` is not a manuscript result. Freeze the rule, then test it",
            "  prospectively on untouched events or an external dataset.",
            "- `no_go` means the observed responder subset cannot currently justify",
            "  deleting or excluding samples by a training-before objective rule.",
            "",
        ]
    )
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
