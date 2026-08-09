#!/usr/bin/env python3
"""Strict full-corpus OOF analysis for the unified PILD-XDomain V/VT matrix.

Every dataset is held out once.  The primary estimate concatenates the four
held-out test partitions, so every registered sample occurs exactly once per
seed.  Dataset-specific results are diagnostics, never a substitute for the
full-corpus estimate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import analyze_pild_sen12_roleaware_v1 as shared


SCHEMA = "pild_sen12_lodo_vt_analysis.v1"
RUN_SCHEMA = "pild_sen12_roleaware_run.v1"
DONE_SCHEMA = "pild_sen12_roleaware_done.v1"
VARIANTS = ("V", "VT")
COUNT_COLUMNS = ("tp", "fp", "fn", "tn")
SUM_COLUMNS = (
    "brier_sum",
    "nll_sum",
    "soft_area_error",
    "fixed_fpr_tp",
    "fixed_fpr_fn",
    "fixed_fpr_fp",
    "fixed_fpr_tn",
    "valid_pixel_count",
    "target_positive_count",
)
REQUIRED_SAMPLE_COLUMNS = (
    "sample_id",
    "dataset_id",
    "canonical_event_id",
    "split",
    "condition",
    "variant",
    "seed",
    "checkpoint_sha256",
    "reference_condition",
    "ap",
    "corrected",
    "harmed",
    "reference_errors",
    "final_errors",
    "visual_errors",
    *COUNT_COLUMNS,
    *SUM_COLUMNS,
)


class AnalysisContractError(RuntimeError):
    """The frozen LODO evidence contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisContractError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return shared.strict_json(path)
    except shared.AnalysisContractError as error:
        raise AnalysisContractError(str(error)) from error


def artifact_receipt(done: Mapping[str, Any], name: str, path: Path) -> None:
    receipt = done.get("artifacts", {}).get(name)
    require(isinstance(receipt, dict), f"DONE receipt missing {name}: {path.parent}")
    require(path.is_file() and path.stat().st_size > 0, f"artifact absent: {path}")
    require(int(receipt.get("size", -1)) == path.stat().st_size, f"size mismatch: {path}")
    require(receipt.get("sha256") == shared.sha256_file(path), f"hash mismatch: {path}")


def discover_matrix(runs_root: Path, min_seeds: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
    require(runs_root.is_dir(), f"runs root absent: {runs_root}")
    fold_dirs = tuple(sorted(path.name for path in runs_root.glob("lodo_*") if path.is_dir()))
    require(len(fold_dirs) >= 2, "fewer than two LODO folds discovered")
    seed_sets: list[set[int]] = []
    for fold in fold_dirs:
        fold_root = runs_root / fold
        for variant in VARIANTS:
            seeds = {
                int(path.name.split("seed", 1)[1])
                for path in fold_root.glob(f"{variant}_seed*")
                if path.is_dir() and (path / "DONE.json").is_file()
            }
            require(seeds, f"no complete {variant} runs in {fold}")
            seed_sets.append(seeds)
    require(all(values == seed_sets[0] for values in seed_sets[1:]), "fold/variant seed sets differ")
    seeds = tuple(sorted(seed_sets[0]))
    require(len(seeds) >= min_seeds, f"only {len(seeds)} common seeds; require >= {min_seeds}")
    return fold_dirs, seeds


def expected_oof_inventory(split_path: Path, folds: Sequence[str]) -> tuple[dict[str, set[str]], pd.DataFrame]:
    frame = pd.read_csv(split_path, dtype=str)
    required = {"fold_id", "sample_id", "dataset_id", "canonical_event_id", "role"}
    require(required.issubset(frame.columns), f"split columns missing: {sorted(required - set(frame.columns))}")
    all_folds = set(frame["fold_id"].dropna().unique())
    require(set(folds) == all_folds, f"discovered folds are not the complete split: {sorted(set(folds) ^ all_folds)}")
    frame = frame.loc[frame["fold_id"].isin(folds) & frame["role"].eq("test")].copy()
    require(set(frame["fold_id"]) == set(folds), "split does not cover every discovered fold")
    duplicate = frame.duplicated("sample_id", keep=False)
    require(not duplicate.any(), f"OOF test samples repeat across folds: {frame.loc[duplicate, 'sample_id'].head().tolist()}")
    inventory = {
        fold: set(frame.loc[frame["fold_id"].eq(fold), "sample_id"])
        for fold in folds
    }
    for fold in folds:
        heldout = frame.loc[frame["fold_id"].eq(fold), "dataset_id"].unique()
        require(len(heldout) == 1, f"fold does not contain exactly one held-out dataset: {fold}")
    return inventory, frame


def load_run(run_dir: Path, fold: str, seed: int, variant: str, expected_samples: set[str]) -> pd.DataFrame:
    done = read_json(run_dir / "DONE.json")
    config = read_json(run_dir / "config.json")
    result = read_json(run_dir / "result.json")
    require(done.get("schema_version") == DONE_SCHEMA and done.get("status") == "complete", f"bad DONE: {run_dir}")
    for payload, label in ((done, "DONE"), (config, "config"), (result, "result")):
        require(payload.get("variant") == variant, f"{label} variant mismatch: {run_dir}")
        require(int(payload.get("seed", -1)) == seed, f"{label} seed mismatch: {run_dir}")
    require(done.get("fold_id") == fold, f"DONE fold mismatch: {run_dir}")
    require(result.get("schema_version") == RUN_SCHEMA and result.get("status") == "complete", f"bad result: {run_dir}")
    require(result.get("identity", {}).get("fold_id") == fold, f"result fold mismatch: {run_dir}")
    require(config.get("identity", {}).get("fold_id") == fold, f"config fold mismatch: {run_dir}")
    require(result.get("evaluation_split") == "test", f"non-test result: {run_dir}")
    for name in ("checkpoint.pt", "config.json", "result.json", "per_sample_metrics.csv"):
        artifact_receipt(done, name, run_dir / name)
    checkpoint_sha = shared.sha256_file(run_dir / "checkpoint.pt")
    require(result.get("checkpoint_sha256") == checkpoint_sha, f"checkpoint SHA mismatch: {run_dir}")

    frame = pd.read_csv(run_dir / "per_sample_metrics.csv")
    missing = set(REQUIRED_SAMPLE_COLUMNS) - set(frame.columns)
    require(not missing, f"sample columns missing in {run_dir}: {sorted(missing)}")
    require(len(frame) == len(expected_samples), f"test row count mismatch: {run_dir}")
    require(not frame["sample_id"].duplicated().any(), f"duplicate test samples: {run_dir}")
    require(set(frame["sample_id"].astype(str)) == expected_samples, f"test inventory mismatch: {run_dir}")
    require(frame["split"].eq("test").all(), f"non-test rows: {run_dir}")
    require(frame["variant"].eq(variant).all() and frame["condition"].eq(variant).all(), f"condition mismatch: {run_dir}")
    require(frame["seed"].eq(seed).all(), f"CSV seed mismatch: {run_dir}")
    require(frame["checkpoint_sha256"].eq(checkpoint_sha).all(), f"CSV checkpoint mismatch: {run_dir}")
    require(np.isfinite(frame[[*COUNT_COLUMNS, *SUM_COLUMNS, "ap", "corrected", "harmed"]].to_numpy(float)).all(), f"non-finite metrics: {run_dir}")
    require((frame[[*COUNT_COLUMNS, "corrected", "harmed", "reference_errors", "final_errors", "visual_errors"]] >= 0).all().all(), f"negative counts: {run_dir}")
    require((frame["final_errors"] == frame["fp"] + frame["fn"]).all(), f"final error identity failed: {run_dir}")
    if variant == "V":
        require(frame["reference_condition"].eq("V").all(), f"V reference mismatch: {run_dir}")
        require((frame["corrected"] == 0).all() and (frame["harmed"] == 0).all(), f"V has correction flow: {run_dir}")
    else:
        require(frame["reference_condition"].eq("V").all(), f"VT reference mismatch: {run_dir}")
    frame = frame.copy()
    frame["fold_id"] = fold
    return frame


def validate_pair(v: pd.DataFrame, vt: pd.DataFrame, v_dir: Path, vt_dir: Path) -> None:
    keys = ["sample_id", "dataset_id", "canonical_event_id"]
    require(v[keys].sort_values("sample_id").reset_index(drop=True).equals(vt[keys].sort_values("sample_id").reset_index(drop=True)), f"V/VT sample identity differs: {vt_dir}")
    expected_parent = (v_dir / "checkpoint.pt").resolve()
    config = read_json(vt_dir / "config.json")
    require(Path(str(config.get("parent_checkpoint", ""))).resolve() == expected_parent, f"VT parent path mismatch: {vt_dir}")
    require(shared.sha256_file(expected_parent) == read_json(v_dir / "DONE.json")["artifacts"]["checkpoint.pt"]["sha256"], f"VT parent hash mismatch: {vt_dir}")
    v_result = read_json(v_dir / "result.json")
    vt_result = read_json(vt_dir / "result.json")
    require(v_result["identity"] == vt_result["identity"], f"V/VT identity mismatch: {vt_dir}")
    require(v_result["component_sha256"].get("visual_decoder") == vt_result["component_sha256"].get("visual_decoder"), f"visual component changed in VT: {vt_dir}")
    require(math.isclose(float(v_result["threshold"]), float(vt_result["threshold"]), abs_tol=1e-12), f"V/VT thresholds differ: {vt_dir}")
    merged = v[["sample_id", "final_errors"]].merge(
        vt[["sample_id", "reference_errors", "corrected", "harmed", "final_errors"]], on="sample_id", validate="one_to_one"
    )
    require((merged["final_errors_x"] == merged["reference_errors"]).all(), f"VT reference errors do not equal V errors: {vt_dir}")
    require((merged["final_errors_y"] == merged["reference_errors"] - merged["corrected"] + merged["harmed"]).all(), f"corrected/harmed identity failed: {vt_dir}")


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    tp, fp, fn, tn = (float(frame[name].sum()) for name in COUNT_COLUMNS)
    valid = float(frame["valid_pixel_count"].sum())
    errors = fp + fn
    return {
        "n_samples": int(len(frame)),
        "n_events": int(frame["canonical_event_id"].nunique()),
        "n_datasets": int(frame["dataset_id"].nunique()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": tp / max(tp + fp + fn, 1.0),
        "macro_sample_ap": float(frame["ap"].mean()),
        "errors": errors,
        "brier": float(frame["brier_sum"].sum()) / max(valid, 1.0),
        "nll": float(frame["nll_sum"].sum()) / max(valid, 1.0),
        "corrected": float(frame["corrected"].sum()),
        "harmed": float(frame["harmed"].sum()),
    }


def exact_sign_flip_p(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    require(len(array) <= 20, "exact sign-flip enumeration is restricted to <=20 units")
    observed = abs(float(array.mean()))
    draws = [abs(float((array * np.asarray(signs)).mean())) for signs in itertools.product((-1.0, 1.0), repeat=len(array))]
    return float(np.mean(np.asarray(draws) >= observed - 1e-12))


def bootstrap_ci(values: Sequence[float], seed: int, n: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(n, len(array)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def add_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        "delta_iou": float(left["iou"] - right["iou"]),
        "delta_macro_sample_ap": float(left["macro_sample_ap"] - right["macro_sample_ap"]),
        "rer": float((right["errors"] - left["errors"]) / max(float(right["errors"]), 1.0)),
        "delta_brier": float(left["brier"] - right["brier"]),
        "delta_nll": float(left["nll"] - right["nll"]),
    }


def analyze(runs_root: Path, split_path: Path, outdir: Path, *, min_seeds: int, bootstrap: int, bootstrap_seed: int) -> dict[str, Any]:
    folds, seeds = discover_matrix(runs_root, min_seeds)
    expected, split_test = expected_oof_inventory(split_path, folds)
    frames: list[pd.DataFrame] = []
    fold_seed_rows: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            loaded: dict[str, pd.DataFrame] = {}
            dirs = {variant: runs_root / fold / f"{variant}_seed{seed}" for variant in VARIANTS}
            for variant in VARIANTS:
                loaded[variant] = load_run(dirs[variant], fold, seed, variant, expected[fold])
                frames.append(loaded[variant])
            validate_pair(loaded["V"], loaded["VT"], dirs["V"], dirs["VT"])
            v_metrics, vt_metrics = metrics(loaded["V"]), metrics(loaded["VT"])
            fold_seed_rows.append({"fold_id": fold, "seed": seed, "heldout_dataset": next(iter(loaded["V"]["dataset_id"].unique())), **{f"v_{k}": v for k, v in v_metrics.items()}, **{f"vt_{k}": v for k, v in vt_metrics.items()}, **add_delta(vt_metrics, v_metrics)})

    raw = pd.concat(frames, ignore_index=True)
    require(raw.loc[raw["variant"].eq("V")].shape[0] == len(split_test) * len(seeds), "full OOF V inventory size mismatch")
    require(raw.loc[raw["variant"].eq("VT")].shape[0] == len(split_test) * len(seeds), "full OOF VT inventory size mismatch")

    seed_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for seed in seeds:
        current = raw.loc[raw["seed"].eq(seed)]
        v_metrics = metrics(current.loc[current["variant"].eq("V")])
        vt_metrics = metrics(current.loc[current["variant"].eq("VT")])
        seed_rows.append({"seed": seed, **{f"v_{k}": v for k, v in v_metrics.items()}, **{f"vt_{k}": v for k, v in vt_metrics.items()}, **add_delta(vt_metrics, v_metrics)})
        for dataset_id, group in current.groupby("dataset_id", sort=True):
            vm, vtm = metrics(group.loc[group["variant"].eq("V")]), metrics(group.loc[group["variant"].eq("VT")])
            dataset_rows.append({"seed": seed, "dataset_id": dataset_id, **{f"v_{k}": v for k, v in vm.items()}, **{f"vt_{k}": v for k, v in vtm.items()}, **add_delta(vtm, vm)})
        for event_id, group in current.groupby("canonical_event_id", sort=True):
            vm, vtm = metrics(group.loc[group["variant"].eq("V")]), metrics(group.loc[group["variant"].eq("VT")])
            event_rows.append({"seed": seed, "canonical_event_id": event_id, "dataset_ids": ";".join(sorted(group["dataset_id"].unique())), **{f"v_{k}": v for k, v in vm.items()}, **{f"vt_{k}": v for k, v in vtm.items()}, **add_delta(vtm, vm)})

    seed_metrics = pd.DataFrame(seed_rows)
    fold_seed = pd.DataFrame(fold_seed_rows)
    dataset_metrics = pd.DataFrame(dataset_rows)
    event_metrics = pd.DataFrame(event_rows)
    sample_mean = raw.groupby(["variant", "dataset_id", "canonical_event_id", "sample_id"], as_index=False)[[*COUNT_COLUMNS, "ap", "corrected", "harmed", "reference_errors", "final_errors", "visual_errors", *SUM_COLUMNS]].mean()
    wide = sample_mean.pivot(index=["dataset_id", "canonical_event_id", "sample_id"], columns="variant").reset_index()
    wide.columns = ["_".join(part for part in column if part).lower() if isinstance(column, tuple) else column for column in wide.columns]
    wide["delta_iou"] = wide["tp_vt"] / (wide["tp_vt"] + wide["fp_vt"] + wide["fn_vt"]).clip(lower=1) - wide["tp_v"] / (wide["tp_v"] + wide["fp_v"] + wide["fn_v"]).clip(lower=1)
    wide["delta_ap"] = wide["ap_vt"] - wide["ap_v"]

    paired_for_bootstrap = pd.DataFrame({
        "source": wide["dataset_id"],
        "canonical_event_id": wide["canonical_event_id"],
        "tp_left": wide["tp_vt"], "fp_left": wide["fp_vt"], "fn_left": wide["fn_vt"],
        "tp_right": wide["tp_v"], "fp_right": wide["fp_v"], "fn_right": wide["fn_v"],
        "delta_iou": wide["delta_iou"], "delta_ap": wide["delta_ap"],
        "brier_sum_left": wide["brier_sum_vt"], "brier_sum_right": wide["brier_sum_v"],
        "nll_sum_left": wide["nll_sum_vt"], "nll_sum_right": wide["nll_sum_v"],
        "soft_area_error_left": wide["soft_area_error_vt"], "soft_area_error_right": wide["soft_area_error_v"],
        "valid_pixel_count_left": wide["valid_pixel_count_vt"], "valid_pixel_count_right": wide["valid_pixel_count_v"],
        "fixed_fpr_tp_left": wide["fixed_fpr_tp_vt"], "fixed_fpr_fn_left": wide["fixed_fpr_fn_vt"],
        "fixed_fpr_tp_right": wide["fixed_fpr_tp_v"], "fixed_fpr_fn_right": wide["fixed_fpr_fn_v"],
    })
    hierarchical_ci = shared.hierarchical_bootstrap_metrics(
        paired_for_bootstrap, n_bootstrap=bootstrap, seed=bootstrap_seed + 1000
    )

    statistic_fields = ("delta_iou", "delta_macro_sample_ap", "rer", "delta_brier", "delta_nll")
    primary: dict[str, Any] = {}
    for index, field in enumerate(statistic_fields):
        values = seed_metrics[field].to_numpy(float)
        primary[field] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "seed_bootstrap_95ci": bootstrap_ci(values, bootstrap_seed + index, bootstrap),
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(values),
            "positive_seeds": int((values > 0).sum()) if not field.startswith("delta_brier") and not field.startswith("delta_nll") else int((values < 0).sum()),
        }
    ci_name = {
        "delta_iou": "pooled_delta_iou",
        "delta_macro_sample_ap": "delta_ap",
        "rer": "rer",
        "delta_brier": "delta_brier",
        "delta_nll": "delta_nll",
    }
    for field, bootstrap_field in ci_name.items():
        primary[field]["hierarchical_dataset_event_sample_95ci"] = hierarchical_ci[bootstrap_field]

    event_mean = event_metrics.groupby("canonical_event_id", as_index=False).agg(
        delta_iou=("delta_iou", "mean"),
        delta_macro_sample_ap=("delta_macro_sample_ap", "mean"),
        rer=("rer", "mean"),
    )
    primary["event_macro_delta_iou"] = {
        "mean": float(event_mean["delta_iou"].mean()),
        "event_bootstrap_95ci": bootstrap_ci(event_mean["delta_iou"], bootstrap_seed + 2000, bootstrap),
        "monte_carlo_two_sided_sign_flip_p": shared.sign_flip_p(
            event_mean["delta_iou"].to_numpy(), iterations=max(20000, bootstrap), seed=bootstrap_seed + 2001
        ),
        "positive_events": int((event_mean["delta_iou"] > 0).sum()),
        "n_events": int(len(event_mean)),
    }

    summary = {
        "schema_version": SCHEMA,
        "status": "complete",
        "primary_scope": "all registered PILD-XDomain samples pooled from four leave-one-dataset-out test folds",
        "n_samples": int(len(split_test)),
        "n_canonical_events": int(split_test["canonical_event_id"].nunique()),
        "n_datasets": int(split_test["dataset_id"].nunique()),
        "folds": list(folds),
        "seeds": list(seeds),
        "n_runs": len(folds) * len(seeds) * len(VARIANTS),
        "primary_vt_minus_v": primary,
        "guardrails": [
            "primary result pools every held-out sample exactly once per seed",
            "dataset-specific effects are heterogeneity diagnostics, not the headline",
            "VT inherits the exact matched V checkpoint, threshold, and visual decoder",
            "average precision is macro sample AP because pixel-level score histograms are not mergeable across checkpoints",
            "Material and Trigger are not included in this V/VT attribution matrix",
        ],
    }
    report = [
        "# Full PILD-XDomain OOF Terrain analysis", "",
        f"Primary inventory: `{summary['n_samples']}` samples, `{summary['n_canonical_events']}` canonical events, `{summary['n_datasets']}` datasets, `{len(seeds)}` optimization seeds.", "",
        "The headline estimate pools all four held-out datasets. Per-dataset rows are retained only to diagnose heterogeneity.", "",
        "## Primary VT minus V", "", pd.DataFrame({name: value for name, value in primary.items()}).T.reset_index(names="metric").to_markdown(index=False), "",
        "## Per-seed full-corpus metrics", "", seed_metrics.to_markdown(index=False), "",
        "## Guardrails", "", *[f"- {item}" for item in summary["guardrails"]], "",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    shared.atomic_write_csv(outdir / "full_corpus_seed_metrics.csv", seed_metrics)
    shared.atomic_write_csv(outdir / "fold_seed_metrics.csv", fold_seed)
    shared.atomic_write_csv(outdir / "dataset_seed_metrics.csv", dataset_metrics)
    shared.atomic_write_csv(outdir / "event_seed_metrics.csv", event_metrics)
    shared.atomic_write_csv(outdir / "paired_sample_seed_mean.csv", wide)
    shared.atomic_write_json(outdir / "summary.json", summary)
    shared.atomic_write_text(outdir / "report.md", "\n".join(report))
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=root / "experiments/revision2026/pild_sen12_roleaware_lodo_v1")
    parser.add_argument("--split", type=Path, default=root / "metadata/pild_sen12_training_v2/leave_one_dataset_out_split_v2.csv")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = analyze(args.runs_root, args.split, args.outdir or args.runs_root / "analysis_full_oof", min_seeds=args.min_seeds, bootstrap=args.bootstrap, bootstrap_seed=args.bootstrap_seed)
    except AnalysisContractError as error:
        raise SystemExit(f"[FATAL] {error}") from error
    print(json.dumps({"status": summary["status"], "n_samples": summary["n_samples"], "n_runs": summary["n_runs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
