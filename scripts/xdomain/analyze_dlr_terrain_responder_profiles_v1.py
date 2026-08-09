#!/usr/bin/env python3
"""Profile DLR Terrain responders without promoting outcome-selected subsets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_RUNS = Path(
    "experiments/revision2026/"
    "dlr_geo4qc_sen12_exactcommon9_scratch_seed20260724_v4"
)
DEFAULT_ELIGIBILITY = Path(
    "metadata/pild_sen12_training_v2/support_eligibility_v1/"
    "sample_eligibility_v1.csv"
)

PROFILE_FEATURES = (
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
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--eligibility-csv", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--outdir", type=Path)
    return parser.parse_args()


def aggregate(rows: pd.DataFrame) -> dict[str, float | int]:
    visual_tp = int(rows["visual_tp"].sum())
    visual_fp = int(rows["visual_fp"].sum())
    visual_fn = int(rows["visual_fn"].sum())
    adapted_tp = int(rows["adapted_tp"].sum())
    adapted_fp = int(rows["adapted_fp"].sum())
    adapted_fn = int(rows["adapted_fn"].sum())
    visual_iou = visual_tp / max(visual_tp + visual_fp + visual_fn, 1)
    adapted_iou = adapted_tp / max(adapted_tp + adapted_fp + adapted_fn, 1)
    visual_errors = visual_fp + visual_fn
    adapted_errors = adapted_fp + adapted_fn
    return {
        "n_samples": int(len(rows)),
        "n_events": int(rows["event_id"].nunique()) if len(rows) else 0,
        "visual_iou": visual_iou,
        "adapted_iou": adapted_iou,
        "delta_iou": adapted_iou - visual_iou,
        "visual_errors": visual_errors,
        "adapted_errors": adapted_errors,
        "rer": (visual_errors - adapted_errors) / max(visual_errors, 1),
        "corrected": int(rows["corrected"].sum()),
        "harmed": int(rows["harmed"].sum()),
        "net_corrected": int(rows["net_corrected"].sum()),
    }


def selective_fallback(all_rows: pd.DataFrame, selected: pd.Series) -> dict[str, float | int]:
    active = all_rows.loc[selected]
    fallback = all_rows.loc[~selected]
    combined = all_rows.copy()
    for name in ("tp", "fp", "fn", "tn", "errors", "iou"):
        adapted_name = f"adapted_{name}"
        visual_name = f"visual_{name}"
        if adapted_name in combined and visual_name in combined:
            combined.loc[~selected, adapted_name] = fallback[visual_name]
    combined.loc[~selected, "corrected"] = 0
    combined.loc[~selected, "harmed"] = 0
    combined.loc[~selected, "net_corrected"] = 0
    result = aggregate(combined)
    result["n_selected"] = int(selected.sum())
    result["coverage_fraction"] = float(selected.mean())
    return result


def load_split(runs_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted(
        runs_dir.glob(f"seed*/fold*/responder_profile_{split}/per_sample.csv")
    )
    if len(paths) != 5:
        raise RuntimeError(f"expected five {split} files, found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main() -> int:
    args = parse_args()
    outdir = args.outdir or args.runs_dir / "responder_profile_analysis_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    eligibility = pd.read_csv(args.eligibility_csv)
    eligibility = eligibility.loc[
        eligibility["dataset_id"].eq("DLR_Landslide_Ref_2025")
    ].drop_duplicates("sample_id")
    validation = load_split(args.runs_dir, "val").merge(
        eligibility, on="sample_id", how="left", validate="many_to_one"
    )
    test = load_split(args.runs_dir, "test").merge(
        eligibility, on="sample_id", how="left", validate="one_to_one"
    )
    if len(test) != 509 or test["sample_id"].nunique() != 509:
        raise RuntimeError("test profile must contain 509 unique DLR samples")

    test.to_csv(outdir / "all_test_samples_with_support.csv", index=False)

    event_rows = []
    for event_id, rows in test.groupby("event_id", sort=True):
        event_rows.append({"event_id": event_id, **aggregate(rows)})
    pd.DataFrame(event_rows).to_csv(outdir / "per_event_response.csv", index=False)

    oracle_masks = {
        "net_corrected_positive": test["net_corrected"] > 0,
        "delta_iou_positive": test["delta_iou"] > 0,
        "sample_rer_ge_10pct": test["rer"] >= 0.10,
    }
    oracle_summary = {}
    for name, mask in oracle_masks.items():
        selected = test.loc[mask].copy()
        selected.to_csv(outdir / f"posthoc_oracle_{name}.csv", index=False)
        oracle_summary[name] = aggregate(selected)

    association_rows = []
    for feature in PROFILE_FEATURES:
        finite = test[[feature, "net_corrected"]].dropna()
        correlation, p_value = spearmanr(finite[feature], finite["net_corrected"])
        positive = test.loc[test["net_corrected"] > 0, feature].dropna()
        negative = test.loc[test["net_corrected"] < 0, feature].dropna()
        association_rows.append(
            {
                "feature": feature,
                "spearman_rho": None if not np.isfinite(correlation) else correlation,
                "spearman_p": None if not np.isfinite(p_value) else p_value,
                "positive_responder_median": (
                    None if positive.empty else float(positive.median())
                ),
                "negative_responder_median": (
                    None if negative.empty else float(negative.median())
                ),
            }
        )
    pd.DataFrame(association_rows).to_csv(
        outdir / "posthoc_feature_associations.csv", index=False
    )

    strata_rows = []
    for feature in PROFILE_FEATURES:
        for quantile in (0.25, 0.50, 0.75):
            threshold = float(test[feature].quantile(quantile))
            for direction in ("le", "ge"):
                mask = (
                    test[feature].le(threshold)
                    if direction == "le"
                    else test[feature].ge(threshold)
                )
                strata_rows.append(
                    {
                        "feature": feature,
                        "quantile": quantile,
                        "direction": direction,
                        "threshold": threshold,
                        "scientific_status": (
                            "label-blind feature stratum; exploratory multiplicity"
                        ),
                        **aggregate(test.loc[mask]),
                    }
                )
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(outdir / "label_blind_strata.csv", index=False)

    first_principles_masks = {
        "terrain_variation_and_visual_uncertainty": (
            test["terrain_slope_std_deg"].ge(5.0)
            & test["terrain_relief_std_m"].ge(20.0)
            & test["visual_margin_mean"].le(0.30)
            & test["optical_valid_fraction"].ge(0.99)
        ),
        "steep_and_visual_uncertainty": (
            test["terrain_slope_mean_deg"].ge(10.0)
            & test["visual_margin_mean"].le(0.30)
            & test["optical_valid_fraction"].ge(0.99)
        ),
    }
    first_principles_summary = {}
    first_principles_rows = []
    for name, mask in first_principles_masks.items():
        selected = test.loc[mask].copy()
        selected["rule_name"] = name
        first_principles_rows.append(selected)
        first_principles_summary[name] = {
            "subset": aggregate(selected),
            "full_test_with_exact_fallback": selective_fallback(test, mask),
        }
    pd.concat(first_principles_rows, ignore_index=True).to_csv(
        outdir / "first_principles_selected_samples.csv", index=False
    )

    selection_rows = []
    selected_test_rows = []
    for fold in range(5):
        fold_validation = validation.loc[validation["fold"].eq(fold)].copy()
        fold_test = test.loc[test["fold"].eq(fold)].copy()
        candidates = []
        for feature in PROFILE_FEATURES:
            for quantile in (0.20, 0.35, 0.50, 0.65, 0.80):
                threshold = float(fold_validation[feature].quantile(quantile))
                for direction in ("le", "ge"):
                    mask = (
                        fold_validation[feature].le(threshold)
                        if direction == "le"
                        else fold_validation[feature].ge(threshold)
                    )
                    selected = fold_validation.loc[mask]
                    metrics = aggregate(selected)
                    event_net = selected.groupby("event_id")["net_corrected"].sum()
                    n_positive_events = int((event_net > 0).sum())
                    minimum_samples = max(12, math.ceil(0.20 * len(fold_validation)))
                    if (
                        metrics["n_samples"] >= minimum_samples
                        and metrics["n_events"] >= 3
                        and n_positive_events
                        >= math.ceil(0.75 * metrics["n_events"])
                        and metrics["delta_iou"] > 0
                        and metrics["rer"] > 0
                    ):
                        candidates.append(
                            {
                                "feature": feature,
                                "direction": direction,
                                "threshold": threshold,
                                "n_positive_validation_events": n_positive_events,
                                **metrics,
                            }
                        )
        if not candidates:
            selection_rows.append(
                {
                    "fold": fold,
                    "status": "abstain_no_validation_rule",
                    "feature": "",
                    "direction": "",
                    "threshold": None,
                }
            )
            continue
        selected_rule = max(
            candidates,
            key=lambda row: (
                row["rer"],
                row["delta_iou"],
                row["n_samples"],
            ),
        )
        feature = selected_rule["feature"]
        threshold = selected_rule["threshold"]
        direction = selected_rule["direction"]
        test_mask = (
            fold_test[feature].le(threshold)
            if direction == "le"
            else fold_test[feature].ge(threshold)
        )
        selected_test = fold_test.loc[test_mask].copy()
        selected_test["selection_fold"] = fold
        selected_test["selection_feature"] = feature
        selected_test["selection_direction"] = direction
        selected_test["selection_threshold"] = threshold
        selected_test_rows.append(selected_test)
        test_metrics = aggregate(selected_test)
        selection_rows.append(
            {
                "fold": fold,
                "status": "validation_selected_test_applied",
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                **{f"validation_{key}": value for key, value in selected_rule.items()
                   if key not in {"feature", "direction", "threshold"}},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    selection_table = pd.DataFrame(selection_rows)
    selection_table.to_csv(outdir / "nested_selection_rules.csv", index=False)
    selected_test = (
        pd.concat(selected_test_rows, ignore_index=True)
        if selected_test_rows
        else test.iloc[0:0].copy()
    )
    selected_test.to_csv(outdir / "nested_selected_test_samples.csv", index=False)

    summary = {
        "status": "exploratory_only",
        "warning": (
            "Oracle subsets use observed outcomes and are invalid as performance "
            "claims. Nested selection uses validation labels but applies frozen "
            "label-independent rules to outer tests."
        ),
        "full_test": aggregate(test),
        "posthoc_oracles": oracle_summary,
        "first_principles_exploratory": first_principles_summary,
        "nested_validation_selected": aggregate(selected_test),
        "nested_test_coverage_fraction": len(selected_test) / len(test),
        "n_nested_abstained_folds": int(
            selection_table["status"].eq("abstain_no_validation_rule").sum()
        ),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    responder = oracle_summary["net_corrected_positive"]
    rer10 = oracle_summary["sample_rer_ge_10pct"]
    nested = summary["nested_validation_selected"]
    physical_rule = first_principles_summary[
        "terrain_variation_and_visual_uncertainty"
    ]
    report = f"""# DLR Terrain responder profile v1

## Interpretation contract

- Oracle rows are selected with observed model outcomes. They are hypothesis-generation
  material only and cannot be reported as model performance.
- Nested rows select a label-independent feature rule on each fold's validation events
  and apply it unchanged to that fold's outer test.
- No raw sample or frozen primary manifest is deleted.

## Results

- Full test: DeltaIoU={summary['full_test']['delta_iou']:+.6f},
  RER={summary['full_test']['rer']:+.2%}.
- Post-hoc positive responders: n={responder['n_samples']},
  DeltaIoU={responder['delta_iou']:+.6f}, RER={responder['rer']:+.2%}.
- Post-hoc sample RER >= 10%: n={rer10['n_samples']},
  DeltaIoU={rer10['delta_iou']:+.6f}, RER={rer10['rer']:+.2%}.
- First-principles terrain-variation/visual-uncertainty stratum:
  n={physical_rule['subset']['n_samples']},
  DeltaIoU={physical_rule['subset']['delta_iou']:+.6f},
  RER={physical_rule['subset']['rer']:+.2%}; with exact visual fallback outside
  the stratum, full-test DeltaIoU={physical_rule['full_test_with_exact_fallback']['delta_iou']:+.6f}
  and RER={physical_rule['full_test_with_exact_fallback']['rer']:+.2%}.
- Validation-selected label-independent subsets: n={nested['n_samples']}
  ({summary['nested_test_coverage_fraction']:.2%} coverage),
  DeltaIoU={nested['delta_iou']:+.6f}, RER={nested['rer']:+.2%};
  abstained folds={summary['n_nested_abstained_folds']}.

## Decision

The oracle quantifies recoverable cases but does not prove deployable selection.
Promotion requires a frozen label-independent rule that is positive on untouched
outer events. The current nested result must be judged from `summary.json`, not
from the oracle CSVs.
"""
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
