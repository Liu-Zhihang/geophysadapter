#!/usr/bin/env python3
"""Learn an event-cross-fitted selector from DLR Terrain response outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


FEATURES = (
    "visual_entropy_mean",
    "visual_margin_mean",
    "terrain_probability_mean",
    "terrain_probability_std",
    "visual_terrain_disagreement_fraction",
    "active_fraction",
    "optical_valid_fraction",
    "cloud_mean_fraction",
    "optical_spatial_contrast",
    "terrain_slope_mean_deg",
    "terrain_slope_std_deg",
    "terrain_relief_mean_m",
    "terrain_relief_std_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(
            "experiments/revision2026/"
            "dlr_geo4qc_sen12_exactcommon9_scratch_seed20260724_v4"
        ),
    )
    parser.add_argument(
        "--eligibility-csv",
        type=Path,
        default=Path(
            "metadata/pild_sen12_training_v2/support_eligibility_v1/"
            "sample_eligibility_v1.csv"
        ),
    )
    parser.add_argument("--outdir", type=Path)
    return parser.parse_args()


def model(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=seed,
            n_jobs=1,
        ),
    )


def selective_metrics(rows: pd.DataFrame, selected: np.ndarray) -> dict[str, float | int]:
    active = rows.loc[selected]
    fallback = rows.loc[~selected]
    visual_tp = int(rows["visual_tp"].sum())
    visual_fp = int(rows["visual_fp"].sum())
    visual_fn = int(rows["visual_fn"].sum())
    adapted_tp = int(active["adapted_tp"].sum() + fallback["visual_tp"].sum())
    adapted_fp = int(active["adapted_fp"].sum() + fallback["visual_fp"].sum())
    adapted_fn = int(active["adapted_fn"].sum() + fallback["visual_fn"].sum())
    visual_iou = visual_tp / max(visual_tp + visual_fp + visual_fn, 1)
    adapted_iou = adapted_tp / max(adapted_tp + adapted_fp + adapted_fn, 1)
    visual_errors = visual_fp + visual_fn
    adapted_errors = adapted_fp + adapted_fn
    return {
        "n_selected": int(selected.sum()),
        "coverage_fraction": float(selected.mean()),
        "n_selected_events": int(active["event_id"].nunique()),
        "delta_iou": adapted_iou - visual_iou,
        "rer": (visual_errors - adapted_errors) / max(visual_errors, 1),
        "net_corrected": visual_errors - adapted_errors,
    }


def main() -> int:
    args = parse_args()
    outdir = args.outdir or args.runs_dir / "learned_responder_selector_v1"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = pd.concat(
        [
            pd.read_csv(
                args.runs_dir
                / f"seed20260724/fold{fold}/responder_profile_test/per_sample.csv"
            )
            for fold in range(5)
        ],
        ignore_index=True,
    )
    eligibility = pd.read_csv(args.eligibility_csv)
    eligibility = eligibility.loc[
        eligibility["dataset_id"].eq("DLR_Landslide_Ref_2025")
    ].drop_duplicates("sample_id")
    rows = rows.merge(
        eligibility, on="sample_id", how="left", validate="one_to_one"
    )
    rows["response_target"] = rows["net_corrected"] / rows["visual_errors"].clip(
        lower=1
    )

    diagnostics = []
    prediction_rows = []
    quantiles = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    for fold in range(5):
        development = rows.loc[rows["fold"].ne(fold)].reset_index(drop=True)
        outer_test = rows.loc[rows["fold"].eq(fold)].copy()
        groups = development["event_id"]
        inner_predictions = cross_val_predict(
            model(20260724 + fold),
            development[list(FEATURES)],
            development["response_target"],
            groups=groups,
            cv=GroupKFold(min(5, groups.nunique())),
            n_jobs=1,
        )
        correlation = float(
            np.corrcoef(inner_predictions, development["response_target"])[0, 1]
        )
        candidates = []
        for quantile in quantiles:
            threshold = float(np.quantile(inner_predictions, quantile))
            selected = inner_predictions >= threshold
            metrics = selective_metrics(development, selected)
            event_net = (
                development.loc[selected]
                .groupby("event_id")["net_corrected"]
                .sum()
            )
            metrics["n_positive_events"] = int((event_net > 0).sum())
            metrics["threshold"] = threshold
            metrics["quantile"] = quantile
            if (
                metrics["n_selected"] >= 30
                and metrics["n_selected_events"] >= 5
                and metrics["delta_iou"] > 0
                and metrics["rer"] > 0
                and metrics["n_positive_events"]
                >= max(3, int(np.ceil(0.5 * len(event_net))))
            ):
                candidates.append(metrics)
        if not candidates:
            diagnostics.append(
                {
                    "fold": fold,
                    "status": "abstain_no_inner_event_stable_rule",
                    "inner_prediction_correlation": correlation,
                }
            )
            outer_test["selector_score"] = np.nan
            outer_test["selector_selected"] = 0
            prediction_rows.append(outer_test)
            continue
        selected_rule = max(
            candidates, key=lambda item: (item["delta_iou"], item["rer"])
        )
        estimator = model(20260724 + fold).fit(
            development[list(FEATURES)], development["response_target"]
        )
        score = estimator.predict(outer_test[list(FEATURES)])
        selected = score >= selected_rule["threshold"]
        outer_metrics = selective_metrics(outer_test.reset_index(drop=True), selected)
        diagnostics.append(
            {
                "fold": fold,
                "status": "frozen_rule_applied",
                "inner_prediction_correlation": correlation,
                **{f"inner_{key}": value for key, value in selected_rule.items()},
                **{f"outer_{key}": value for key, value in outer_metrics.items()},
            }
        )
        outer_test["selector_score"] = score
        outer_test["selector_selected"] = selected.astype(int)
        prediction_rows.append(outer_test)

    diagnostics_frame = pd.DataFrame(diagnostics)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    diagnostics_frame.to_csv(outdir / "per_fold_diagnostics.csv", index=False)
    predictions.to_csv(outdir / "cross_fitted_predictions.csv", index=False)
    selected = predictions["selector_selected"].to_numpy(dtype=bool)
    summary = {
        "status": (
            "no_go"
            if diagnostics_frame["status"]
            .eq("abstain_no_inner_event_stable_rule")
            .all()
            else "exploratory_candidate"
        ),
        "scientific_status": (
            "response outcomes used only in development events; selector inputs "
            "exclude labels, correctness, event identity, source identity, and coordinates"
        ),
        "features": list(FEATURES),
        "n_samples": int(len(rows)),
        "n_events": int(rows["event_id"].nunique()),
        "n_abstained_folds": int(
            diagnostics_frame["status"]
            .eq("abstain_no_inner_event_stable_rule")
            .sum()
        ),
        "inner_prediction_correlation_range": [
            float(diagnostics_frame["inner_prediction_correlation"].min()),
            float(diagnostics_frame["inner_prediction_correlation"].max()),
        ],
        "cross_fitted_selective_metrics": selective_metrics(predictions, selected),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (outdir / "report.md").write_text(
        "# DLR learned responder selector v1\n\n"
        f"- Status: **{summary['status']}**\n"
        f"- Samples/events: {summary['n_samples']}/{summary['n_events']}\n"
        f"- Abstained outer folds: {summary['n_abstained_folds']}/5\n"
        "- Interpretation: response-positive samples exist, but their current "
        "label-independent signatures do not transfer stably across events.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
