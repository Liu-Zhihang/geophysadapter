#!/usr/bin/env python3
"""Formal strict analysis of the Sen12 LOGO-5 GeoPhysAdapter matrix.

The analyzer accepts only artifact-complete matched visual/adapter runs.  It
verifies dataset, split, checkpoint, sample-order, exact-fallback, and spatial
control identities before writing any manuscript-facing result.  Statistical
resampling uses spatial folds, physical events, or regions as independent
units; pixels are never treated as independent replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RUNS_DEFAULT = PROJECT_ROOT / "experiments/revision2026/sen12_xdomain_geophysadapter_v1"
RUN_PATTERN = re.compile(r"fold(?P<fold>\d+)_seed(?P<seed>\d+)$")
VISUAL_CONTROL = "visual"
ANCHOR_CONTROL = "visual_anchor"
ADAPTER_CONTROLS = (
    ANCHOR_CONTROL,
    "aligned",
    "zero",
    "roll32",
    "roll64",
    "other_region_donor",
)
COMPARATORS = (ANCHOR_CONTROL, "zero", "roll32", "roll64", "other_region_donor")
RUN_ARTIFACTS = (
    "DONE.json",
    "result.json",
    "checkpoint.pt",
    "config.json",
    "per_sample.csv",
    "per_event.csv",
    "per_region.csv",
)
IDENTITY_FIELDS = (
    "fold",
    "seed",
    "backbone",
    "split_csv_sha256",
    "h5_signature",
    "sample_identity_sha256",
    "reflectance_scale",
    "image_size",
    "out_indices",
    "hidden",
    "pretrained_backbone",
)
PAIR_METRICS = (
    "iou",
    "average_precision",
    "brier",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "specificity",
    "rer",
    "corrected",
    "harmed",
    "net_corrected",
    "visual_errors",
    "adapter_errors",
)
SUMMARY_METRICS = (
    "iou",
    "average_precision",
    "brier",
    "rer",
    "corrected",
    "harmed",
    "net_corrected",
)
OUTPUT_FILES = (
    "per_run_paired.csv",
    "per_sample_paired.csv",
    "per_event_paired.csv",
    "per_region_paired.csv",
    "summary.json",
    "report.md",
    "artifact_check.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--expected-folds", default="0,1,2,3,4")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Development only: relax fold and seed-count gates, never emit PASS.",
    )
    args = parser.parse_args(argv)
    if args.min_seeds < 1 or args.bootstrap < 100 or args.permutations < 100:
        parser.error("--min-seeds must be >=1 and resampling counts must be >=100")
    return args


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def require_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty artifact: {path}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")


def normalized_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def exact_identity_subset(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in IDENTITY_FIELDS}


def assert_actual_data_identity(identity: dict[str, Any], config: dict[str, Any], run_dir: Path) -> None:
    split_path = Path(str(config.get("split_csv", ""))).expanduser()
    if not split_path.is_file():
        raise RuntimeError(f"Recorded split CSV is unavailable for {run_dir}: {split_path}")
    if sha256_file(split_path) != identity.get("split_csv_sha256"):
        raise RuntimeError(f"Split CSV SHA256 mismatch for {run_dir}")
    h5_signature = identity.get("h5_signature")
    if not isinstance(h5_signature, dict):
        raise RuntimeError(f"Missing H5 identity for {run_dir}")
    h5_path = Path(str(h5_signature.get("path", ""))).expanduser()
    if not h5_path.is_file():
        raise RuntimeError(f"Recorded H5 is unavailable for {run_dir}: {h5_path}")
    actual = {"path": str(h5_path.resolve()), "size": h5_path.stat().st_size, "mtime_ns": h5_path.stat().st_mtime_ns}
    recorded = {
        "path": normalized_path(h5_signature.get("path")),
        "size": h5_signature.get("size"),
        "mtime_ns": h5_signature.get("mtime_ns"),
    }
    if actual != recorded:
        raise RuntimeError(f"H5 identity mismatch for {run_dir}: recorded={recorded}, actual={actual}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("identity"), dict):
        raise RuntimeError(f"Checkpoint identity missing: {path}")
    return payload


def assert_run_artifacts(run_dir: Path, mode: str, fold: int, seed: int) -> dict[str, Any]:
    for name in RUN_ARTIFACTS:
        require_nonempty_file(run_dir / name)
    done = load_json(run_dir / "DONE.json")
    result = load_json(run_dir / "result.json")
    config = load_json(run_dir / "config.json")
    checkpoint = load_checkpoint(run_dir / "checkpoint.pt")
    expected = {"mode": mode, "fold": fold, "seed": seed}
    for source_name, source in (("DONE", done), ("result", result), ("config", config), ("checkpoint", checkpoint["identity"])):
        mismatches = {key: {"expected": value, "actual": source.get(key)} for key, value in expected.items() if source.get(key) != value}
        if mismatches:
            raise RuntimeError(f"{source_name} mode/fold/seed mismatch in {run_dir}: {mismatches}")
    if done.get("status") != "complete":
        raise RuntimeError(f"DONE status is not complete in {run_dir}")
    if done.get("result_sha256") != sha256_file(run_dir / "result.json"):
        raise RuntimeError(f"DONE/result SHA256 mismatch in {run_dir}")
    if done.get("checkpoint_sha256") != sha256_file(run_dir / "checkpoint.pt"):
        raise RuntimeError(f"DONE/checkpoint SHA256 mismatch in {run_dir}")
    identity = result.get("identity")
    if not isinstance(identity, dict) or identity != checkpoint["identity"]:
        raise RuntimeError(f"result/checkpoint identity mismatch in {run_dir}")
    config_identity = {
        "fold": config.get("fold"),
        "seed": config.get("seed"),
        "backbone": config.get("backbone"),
        "split_csv_sha256": config.get("split_csv_sha256"),
        "h5_signature": config.get("h5_signature"),
        "sample_identity_sha256": config.get("sample_identity_sha256"),
        "reflectance_scale": config.get("reflectance_scale"),
        "image_size": config.get("args", {}).get("image_size"),
        "out_indices": config.get("args", {}).get("out_indices"),
        "hidden": config.get("args", {}).get("hidden"),
        "pretrained_backbone": config.get("pretrained_backbone"),
    }
    if exact_identity_subset(identity) != config_identity:
        raise RuntimeError(f"config/result identity mismatch in {run_dir}")
    assert_actual_data_identity(identity, config, run_dir)
    return {"done": done, "result": result, "config": config, "checkpoint": checkpoint}


def find_test_audit(result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    audits = result.get("identity_and_control_audits")
    if not isinstance(audits, list):
        raise RuntimeError(f"Missing identity audits in {run_dir}")
    matches = [item for item in audits if isinstance(item, dict) and item.get("split") == "test"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one test identity audit in {run_dir}, found {len(matches)}")
    return matches[0]


def load_nonempty_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Empty CSV: {path}")
    require_columns(frame, required, path)
    return frame


def assert_row_contract(frame: pd.DataFrame, path: Path, mode: str, fold: int, seed: int) -> pd.DataFrame:
    require_columns(frame, ("mode", "fold", "seed", "split", "control"), path)
    if set(frame["mode"].astype(str)) != {mode}:
        raise RuntimeError(f"Mode mismatch in {path}")
    if set(pd.to_numeric(frame["fold"]).astype(int)) != {fold}:
        raise RuntimeError(f"Fold mismatch in {path}")
    if set(pd.to_numeric(frame["seed"]).astype(int)) != {seed}:
        raise RuntimeError(f"Seed mismatch in {path}")
    test = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if test.empty:
        raise RuntimeError(f"No test rows in {path}")
    return test


def assert_control_coverage(frame: pd.DataFrame, unit: str, controls: Sequence[str], path: Path) -> None:
    if frame.duplicated([unit, "control"]).any():
        raise RuntimeError(f"Duplicate {unit}/control rows in {path}")
    actual_controls = set(frame["control"].astype(str))
    if actual_controls != set(controls):
        raise RuntimeError(f"Control mismatch in {path}: {sorted(actual_controls)}")
    sets = {
        control: set(frame.loc[frame["control"].eq(control), unit].astype(str))
        for control in controls
    }
    if len({frozenset(values) for values in sets.values()}) != 1 or not next(iter(sets.values())):
        raise RuntimeError(f"Incomplete {unit} coverage by control in {path}")


def assert_visual_anchor_equal(visual: pd.DataFrame, adapter: pd.DataFrame, unit: str, fold: int, seed: int) -> None:
    left = visual.loc[visual["control"].eq(VISUAL_CONTROL)].copy()
    right = adapter.loc[adapter["control"].eq(ANCHOR_CONTROL)].copy()
    left["control"] = ANCHOR_CONTROL
    left = left.sort_values(unit).reset_index(drop=True)
    right = right.sort_values(unit).reset_index(drop=True)
    if not left[unit].astype(str).equals(right[unit].astype(str)):
        raise RuntimeError(f"Visual anchor {unit} identity mismatch in fold={fold} seed={seed}")
    exact = [column for column in ("tp", "fp", "fn", "tn", "visual_errors", "adapter_errors", "corrected", "harmed", "net_corrected") if column in left and column in right]
    for column in exact:
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise RuntimeError(f"Visual anchor mismatch for {unit}.{column} in fold={fold} seed={seed}")
    floats = [column for column in ("threshold", "iou", "average_precision", "brier", "accuracy", "f1") if column in left and column in right]
    for column in floats:
        if not np.allclose(left[column].to_numpy(float), right[column].to_numpy(float), rtol=0.0, atol=0.0, equal_nan=True):
            raise RuntimeError(f"Visual anchor mismatch for {unit}.{column} in fold={fold} seed={seed}")


def assert_pair_identity(pair_dir: Path, fold: int, seed: int) -> dict[str, Any]:
    visual_dir, adapter_dir = pair_dir / "visual", pair_dir / "adapter"
    visual = assert_run_artifacts(visual_dir, "visual", fold, seed)
    adapter = assert_run_artifacts(adapter_dir, "adapter", fold, seed)
    visual_identity = visual["checkpoint"]["identity"]
    adapter_identity = adapter["checkpoint"]["identity"]
    if exact_identity_subset(visual_identity) != exact_identity_subset(adapter_identity):
        raise RuntimeError(f"Visual/adapter data identity mismatch in fold={fold} seed={seed}")
    adapter_config = adapter["config"]
    if normalized_path(adapter_config.get("visual_checkpoint")) != str((visual_dir / "checkpoint.pt").resolve()):
        raise RuntimeError(f"Adapter visual-checkpoint path mismatch in fold={fold} seed={seed}")
    if adapter_config.get("visual_checkpoint_identity") != visual_identity:
        raise RuntimeError(f"Adapter matched visual-checkpoint identity mismatch in fold={fold} seed={seed}")
    visual_state = visual_identity.get("visual_state_sha256")
    if not visual_state or adapter["result"].get("visual_state_sha256_before") != visual_state:
        raise RuntimeError(f"Adapter did not start from matched visual state in fold={fold} seed={seed}")
    if adapter["result"].get("visual_state_sha256_after") != visual_state or adapter_identity.get("visual_state_sha256") != visual_state:
        raise RuntimeError(f"Frozen visual state changed in fold={fold} seed={seed}")
    if adapter["checkpoint"].get("threshold_source") != "loaded_matched_visual_checkpoint":
        raise RuntimeError(f"Adapter threshold source mismatch in fold={fold} seed={seed}")
    if float(adapter["checkpoint"].get("threshold")) != float(visual["checkpoint"].get("threshold")):
        raise RuntimeError(f"Adapter/visual threshold mismatch in fold={fold} seed={seed}")

    visual_audit = find_test_audit(visual["result"], visual_dir)
    adapter_audit = find_test_audit(adapter["result"], adapter_dir)
    if not adapter_audit.get("same_sample_identity_and_order"):
        raise RuntimeError(f"Control sample order audit failed in fold={fold} seed={seed}")
    hashes = adapter_audit.get("sample_order_sha256_by_control", {})
    counts = adapter_audit.get("n_samples_by_control", {})
    if set(hashes) != set(ADAPTER_CONTROLS) or set(counts) != set(ADAPTER_CONTROLS):
        raise RuntimeError(f"Incomplete control identity audit in fold={fold} seed={seed}")
    if len(set(hashes.values())) != 1 or len(set(counts.values())) != 1 or next(iter(counts.values()), 0) <= 0:
        raise RuntimeError(f"Control identities/counts differ in fold={fold} seed={seed}")
    visual_hashes = visual_audit.get("sample_order_sha256_by_control", {})
    if visual_hashes.get(VISUAL_CONTROL) != hashes.get(ANCHOR_CONTROL):
        raise RuntimeError(f"Visual/adapter sample-order hash mismatch in fold={fold} seed={seed}")
    if adapter_audit.get("zero_terrain_exact_fallback") is not True or float(adapter_audit.get("zero_terrain_max_abs_logit_delta_from_visual", math.nan)) != 0.0:
        raise RuntimeError(f"Zero-Terrain exact fallback failed in fold={fold} seed={seed}")
    if adapter_audit.get("q_t_zero_exact_fallback") is not True or float(adapter_audit.get("q_t_zero_max_abs_logit_delta_from_visual", math.nan)) != 0.0:
        raise RuntimeError(f"q_T=0 exact fallback failed in fold={fold} seed={seed}")
    if int(adapter_audit.get("other_region_donor_violations", -1)) != 0:
        raise RuntimeError(f"Other-region donor contract failed in fold={fold} seed={seed}")

    csvs: dict[str, dict[str, pd.DataFrame]] = {"visual": {}, "adapter": {}}
    units = {"per_sample.csv": "sample_id", "per_event.csv": "physical_event_id", "per_region.csv": "region_group"}
    for filename, unit in units.items():
        required = (unit, "mode", "fold", "seed", "split", "control", "iou", "average_precision", "brier")
        visual_frame = assert_row_contract(load_nonempty_csv(visual_dir / filename, required), visual_dir / filename, "visual", fold, seed)
        adapter_frame = assert_row_contract(load_nonempty_csv(adapter_dir / filename, required), adapter_dir / filename, "adapter", fold, seed)
        assert_control_coverage(visual_frame, unit, (VISUAL_CONTROL,), visual_dir / filename)
        assert_control_coverage(adapter_frame, unit, ADAPTER_CONTROLS, adapter_dir / filename)
        assert_visual_anchor_equal(visual_frame, adapter_frame, unit, fold, seed)
        csvs["visual"][filename] = visual_frame
        csvs["adapter"][filename] = adapter_frame

    sample_frame = csvs["adapter"]["per_sample.csv"]
    raw_hashes = {
        control: sha256_strings(sample_frame.loc[sample_frame["control"].eq(control), "sample_id"].astype(str).tolist())
        for control in ADAPTER_CONTROLS
    }
    if raw_hashes != hashes:
        raise RuntimeError(f"Control CSV order hashes disagree with result audit in fold={fold} seed={seed}")
    visual_samples = csvs["visual"]["per_sample.csv"]
    visual_raw_hash = sha256_strings(visual_samples.loc[visual_samples["control"].eq(VISUAL_CONTROL), "sample_id"].astype(str).tolist())
    if visual_raw_hash != visual_hashes.get(VISUAL_CONTROL):
        raise RuntimeError(f"Visual CSV order hash disagrees with result audit in fold={fold} seed={seed}")

    return {
        "visual": visual,
        "adapter": adapter,
        "sample": sample_frame,
        "event": csvs["adapter"]["per_event.csv"],
        "region": csvs["adapter"]["per_region.csv"],
        "sample_order_sha256": hashes[ANCHOR_CONTROL],
    }


def discover_pairs(runs_dir: Path, expected_folds: set[int], min_seeds: int, allow_partial: bool) -> dict[int, dict[int, Path]]:
    if not runs_dir.is_dir():
        raise RuntimeError(f"Runs directory does not exist: {runs_dir}")
    discovered: dict[int, dict[int, Path]] = {}
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        match = RUN_PATTERN.fullmatch(path.name)
        if match:
            fold, seed = int(match.group("fold")), int(match.group("seed"))
            discovered.setdefault(fold, {})[seed] = path
    if not discovered:
        raise RuntimeError(f"No fold/seed pair directories under {runs_dir}")
    if allow_partial:
        return discovered
    if set(discovered) != expected_folds:
        raise RuntimeError(f"Fold coverage mismatch: found={sorted(discovered)}, expected={sorted(expected_folds)}")
    for fold in sorted(expected_folds):
        if len(discovered[fold]) < min_seeds:
            raise RuntimeError(f"fold={fold} has {len(discovered[fold])} common visual/adapter seeds; need >= {min_seeds}")
    return discovered


def corpus_test_frame(result: dict[str, Any], fold: int, seed: int) -> pd.DataFrame:
    rows = result.get("corpus_metrics")
    if not isinstance(rows, list):
        raise RuntimeError(f"Missing corpus_metrics in fold={fold} seed={seed}")
    frame = pd.DataFrame(rows)
    required = {"split", "control", "iou", "average_precision", "brier", "rer", "corrected", "harmed", "net_corrected", "visual_errors", "adapter_errors"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Corpus metric columns missing in fold={fold} seed={seed}: {missing}")
    frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if set(frame["control"].astype(str)) != set(ADAPTER_CONTROLS) or frame["control"].duplicated().any():
        raise RuntimeError(f"Corpus control coverage failed in fold={fold} seed={seed}")
    frame["fold"], frame["seed"] = fold, seed
    return frame


def paired_table(frame: pd.DataFrame, unit_columns: Sequence[str]) -> pd.DataFrame:
    keys = ["fold", "seed", *unit_columns]
    rows: list[pd.DataFrame] = []
    aligned = frame.loc[frame["control"].eq("aligned")].copy()
    for comparator in COMPARATORS:
        other = frame.loc[frame["control"].eq(comparator)].copy()
        columns = [column for column in PAIR_METRICS if column in aligned.columns and column in other.columns]
        left = aligned[keys + columns].rename(columns={column: f"aligned_{column}" for column in columns})
        right = other[keys + columns].rename(columns={column: f"comparator_{column}" for column in columns})
        merged = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
        if not merged["_merge"].eq("both").all():
            raise RuntimeError(f"Unpaired rows for aligned vs {comparator} at unit={unit_columns}")
        merged = merged.drop(columns="_merge")
        merged.insert(len(keys), "aligned_control", "aligned")
        merged.insert(len(keys) + 1, "comparator", comparator)
        for metric in columns:
            merged[f"delta_{metric}"] = merged[f"aligned_{metric}"] - merged[f"comparator_{metric}"]
        rows.append(merged)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["comparator", *keys]).reset_index(drop=True)


def bootstrap_mean_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    n = values.size
    means: list[np.ndarray] = []
    remaining = draws
    while remaining:
        count = min(512, remaining)
        indices = rng.integers(0, n, size=(count, n))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(means)
    return float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def sign_flip_p(values: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[float, str, int]:
    observed = abs(float(values.mean()))
    n = values.size
    tolerance = np.finfo(np.float64).eps * max(observed, 1.0) * 8
    if n <= 20 and 2**n <= permutations:
        extreme = 0
        total = 2**n
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            if abs(float(np.dot(values, np.asarray(signs)) / n)) + tolerance >= observed:
                extreme += 1
        return extreme / total, "exact_sign_flip", total
    extreme = 0
    total = permutations
    remaining = total
    while remaining:
        count = min(4096, remaining)
        signs = rng.integers(0, 2, size=(count, n), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs @ values) / n)
        extreme += int(np.count_nonzero(permuted + tolerance >= observed))
        remaining -= count
    return (extreme + 1) / (total + 1), "monte_carlo_sign_flip", total


def paired_stats(values: Iterable[float], bootstrap: int, permutations: int, token: str) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise RuntimeError(f"No finite paired values for {token}")
    seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    ci_low, ci_high = bootstrap_mean_ci(array, bootstrap, rng)
    permutation_p, permutation_method, permutation_n = sign_flip_p(array, permutations, rng)
    try:
        wilcoxon_p = float(wilcoxon(array, alternative="two-sided", zero_method="pratt").pvalue) if np.any(array != 0) else 1.0
    except (ValueError, RuntimeWarning):
        wilcoxon_p = math.nan
    sd = float(array.std(ddof=1)) if array.size > 1 else math.nan
    return {
        "n_independent_units": int(array.size),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95": [ci_low, ci_high],
        "positive_fraction": float(np.mean(array > 0)),
        "permutation_p_two_sided": permutation_p,
        "permutation_method": permutation_method,
        "permutation_draws": permutation_n,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "cohens_dz": float(array.mean() / sd) if math.isfinite(sd) and sd > 0 else math.nan,
    }


def summarize_paired(
    frame: pd.DataFrame,
    unit_columns: Sequence[str],
    independent_columns: Sequence[str],
    bootstrap: int,
    permutations: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for comparator in COMPARATORS:
        selected = frame.loc[frame["comparator"].eq(comparator)]
        comparison: dict[str, Any] = {}
        for metric in SUMMARY_METRICS:
            delta = f"delta_{metric}"
            if delta not in selected.columns:
                continue
            independent = selected.groupby(list(independent_columns), dropna=False)[delta].mean().reset_index()
            comparison[metric] = paired_stats(
                independent[delta], bootstrap, permutations,
                f"{','.join(unit_columns)}|{','.join(independent_columns)}|{comparator}|{metric}",
            )
        summary[f"aligned_vs_{comparator}"] = comparison
    return summary


def control_summary(run_frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for control, group in run_frame.groupby("control", sort=True):
        visual_errors = int(group["visual_errors"].sum())
        net_corrected = int(group["net_corrected"].sum())
        output[str(control)] = {
            "n_fold_seed_runs": int(len(group)),
            "mean_iou": float(group["iou"].mean()),
            "sd_iou": float(group["iou"].std(ddof=1)),
            "mean_average_precision": float(group["average_precision"].mean()),
            "sd_average_precision": float(group["average_precision"].std(ddof=1)),
            "mean_brier": float(group["brier"].mean()),
            "sd_brier": float(group["brier"].std(ddof=1)),
            "pooled_visual_errors_across_runs": visual_errors,
            "pooled_adapter_errors_across_runs": int(group["adapter_errors"].sum()),
            "pooled_corrected_across_runs": int(group["corrected"].sum()),
            "pooled_harmed_across_runs": int(group["harmed"].sum()),
            "pooled_net_corrected_across_runs": net_corrected,
            "pooled_rer_across_runs": net_corrected / max(visual_errors, 1),
        }
    return output


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def clean_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise RuntimeError(f"Refusing to write empty analysis CSV: {path}")
    frame.replace([np.inf, -np.inf], np.nan).to_csv(path, index=False, na_rep="")


def build_report(summary: dict[str, Any]) -> str:
    primary = summary["statistics"]["run_corpus_fold_bootstrap"]["aligned_vs_visual_anchor"]
    lines = [
        "# Sen12 LOGO-5 GeoPhysAdapter formal paired analysis",
        "",
        f"- strict status: `{summary['status']}`",
        f"- complete visual/adapter pairs: `{summary['n_paired_runs']}`",
        f"- folds: `{summary['folds']}`; observed minimum common seeds per fold: "
        f"`{summary['observed_min_seeds_per_fold']}`; formal requirement: `{summary['required_min_seeds']}`",
        "- evaluation split: `test` only; threshold inherited from each matched visual checkpoint.",
        "- corpus CIs resample independent held-out spatial folds; event/region CIs resample physical events/regions after averaging seeds.",
        "",
        "## Aligned Terrain vs matched visual anchor",
        "",
        "| metric | mean paired delta | 95% bootstrap CI | permutation p | Wilcoxon p | Cohen dz |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ("iou", "average_precision", "brier", "rer", "corrected", "harmed"):
        item = primary[metric]
        lines.append(
            f"| {metric} | {fmt(item['mean_delta'])} | [{fmt(item['ci95'][0])}, {fmt(item['ci95'][1])}] | "
            f"{fmt(item['permutation_p_two_sided'])} | {fmt(item['wilcoxon_p_two_sided'])} | {fmt(item['cohens_dz'])} |"
        )
    lines.extend([
        "",
        "## Spatial specificity",
        "",
        "Aligned support must outperform spatially destroyed or mismatched controls; aligned-vs-visual improvement alone is not treated as evidence of spatially specific physical use.",
        "",
        "| comparator | fold-level delta IoU | 95% CI | event-level delta IoU | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ])
    run_stats = summary["statistics"]["run_corpus_fold_bootstrap"]
    event_stats = summary["statistics"]["physical_event_bootstrap"]
    for comparator in ("zero", "roll32", "roll64", "other_region_donor"):
        run_item = run_stats[f"aligned_vs_{comparator}"]["iou"]
        event_item = event_stats[f"aligned_vs_{comparator}"]["iou"]
        lines.append(
            f"| {comparator} | {fmt(run_item['mean_delta'])} | [{fmt(run_item['ci95'][0])}, {fmt(run_item['ci95'][1])}] | "
            f"{fmt(event_item['mean_delta'])} | [{fmt(event_item['ci95'][0])}, {fmt(event_item['ci95'][1])}] |"
        )
    lines.extend([
        "",
        "## Interpretation guardrail",
        "",
        "Positive IoU/AP and RER favor aligned support; negative Brier and harmed deltas favor aligned support. Statistical significance is interpreted only with artifact identity, exact-fallback, and aligned-versus-negative-control gates intact.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    outdir: Path,
    run_paired: pd.DataFrame,
    sample_paired: pd.DataFrame,
    event_paired: pd.DataFrame,
    region_paired: pd.DataFrame,
    summary: dict[str, Any],
    artifact_rows: list[dict[str, Any]],
    allow_partial: bool,
) -> None:
    temp = outdir.with_name(f".{outdir.name}.tmp.{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        clean_csv(run_paired, temp / "per_run_paired.csv")
        clean_csv(sample_paired, temp / "per_sample_paired.csv")
        clean_csv(event_paired, temp / "per_event_paired.csv")
        clean_csv(region_paired, temp / "per_region_paired.csv")
        (temp / "summary.json").write_text(
            json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temp / "report.md").write_text(build_report(summary), encoding="utf-8")
        gate = {
            "status": "DEVELOPMENT_PARTIAL" if allow_partial else "PASS",
            "strict_pass": not allow_partial,
            "n_paired_runs": len(artifact_rows),
            "validated_runs": artifact_rows,
            "outputs": {},
        }
        for name in OUTPUT_FILES[:-1]:
            require_nonempty_file(temp / name)
            gate["outputs"][name] = {"bytes": (temp / name).stat().st_size, "sha256": sha256_file(temp / name)}
        (temp / "artifact_check.json").write_text(
            json.dumps(json_safe(gate), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if outdir.exists():
            shutil.rmtree(outdir)
        temp.rename(outdir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def write_failure_gate(outdir: Path, error: BaseException) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (outdir / name).unlink(missing_ok=True)
    payload = {
        "status": "FAIL",
        "strict_pass": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    (outdir / "artifact_check.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def analyze(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[dict[str, Any]]]:
    runs_dir = args.runs_dir.expanduser().resolve()
    expected_folds = {int(value.strip()) for value in args.expected_folds.split(",") if value.strip()}
    if not expected_folds:
        raise RuntimeError("--expected-folds is empty")
    pairs = discover_pairs(runs_dir, expected_folds, args.min_seeds, args.allow_partial)
    run_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    region_frames: list[pd.DataFrame] = []
    artifacts: list[dict[str, Any]] = []
    validated: dict[int, list[int]] = {}
    for fold, seeds in sorted(pairs.items()):
        for seed, pair_dir in sorted(seeds.items()):
            try:
                pair = assert_pair_identity(pair_dir, fold, seed)
            except Exception:
                if args.allow_partial:
                    continue
                raise
            validated.setdefault(fold, []).append(seed)
            run_frames.append(corpus_test_frame(pair["adapter"]["result"], fold, seed))
            for key, collection in (("sample", sample_frames), ("event", event_frames), ("region", region_frames)):
                frame = pair[key].copy()
                frame["fold"], frame["seed"] = fold, seed
                collection.append(frame)
            artifacts.append({
                "fold": fold,
                "seed": seed,
                "pair_dir": str(pair_dir.resolve()),
                "visual_result_sha256": sha256_file(pair_dir / "visual/result.json"),
                "visual_checkpoint_sha256": sha256_file(pair_dir / "visual/checkpoint.pt"),
                "adapter_result_sha256": sha256_file(pair_dir / "adapter/result.json"),
                "adapter_checkpoint_sha256": sha256_file(pair_dir / "adapter/checkpoint.pt"),
                "test_sample_order_sha256": pair["sample_order_sha256"],
            })
    if not artifacts:
        raise RuntimeError("No valid paired runs")
    if not args.allow_partial:
        if set(validated) != expected_folds:
            raise RuntimeError(f"Validated fold coverage mismatch: {sorted(validated)}")
        for fold in expected_folds:
            if len(validated[fold]) < args.min_seeds:
                raise RuntimeError(f"fold={fold} has only {len(validated[fold])} validated common seeds")
    run_frame = pd.concat(run_frames, ignore_index=True)
    sample_frame = pd.concat(sample_frames, ignore_index=True)
    event_frame = pd.concat(event_frames, ignore_index=True)
    region_frame = pd.concat(region_frames, ignore_index=True)
    paired = {
        "run": paired_table(run_frame, ()),
        "sample": paired_table(sample_frame, ("sample_id", "physical_event_id", "region_group")),
        "event": paired_table(event_frame, ("physical_event_id",)),
        "region": paired_table(region_frame, ("region_group",)),
    }
    statistics = {
        "run_corpus_fold_bootstrap": summarize_paired(paired["run"], (), ("fold",), args.bootstrap, args.permutations),
        "sample_bootstrap": summarize_paired(paired["sample"], ("sample_id",), ("sample_id",), args.bootstrap, args.permutations),
        "physical_event_bootstrap": summarize_paired(paired["event"], ("physical_event_id",), ("physical_event_id",), args.bootstrap, args.permutations),
        "region_bootstrap": summarize_paired(paired["region"], ("region_group",), ("region_group",), args.bootstrap, args.permutations),
    }
    summary = {
        "status": "DEVELOPMENT_PARTIAL" if args.allow_partial else "PASS",
        "strict_pass": not args.allow_partial,
        "runs_dir": str(runs_dir),
        "analysis_split": "test",
        "folds": sorted(validated),
        "seeds_by_fold": {str(fold): sorted(seeds) for fold, seeds in validated.items()},
        "min_seeds": args.min_seeds,
        "required_min_seeds": args.min_seeds,
        "observed_min_seeds_per_fold": min(len(seeds) for seeds in validated.values()),
        "n_paired_runs": len(artifacts),
        "bootstrap_draws": args.bootstrap,
        "permutation_draws_requested": args.permutations,
        "independence_contract": {
            "corpus": "average seed-level paired deltas within held-out spatial fold, then resample folds",
            "sample": "average seed-level paired deltas within sample, then resample samples",
            "event": "average seed-level paired deltas within physical event, then resample physical events",
            "region": "average seed-level paired deltas within region, then resample regions",
            "pixels": "never treated as independent replicates",
        },
        "control_metrics": control_summary(run_frame),
        "statistics": statistics,
        "spatial_specificity_required_comparisons": [f"aligned_vs_{name}" for name in COMPARATORS[1:]],
    }
    return paired, summary, artifacts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = (args.outdir or args.runs_dir / "analysis").expanduser().resolve()
    try:
        paired, summary, artifacts = analyze(args)
        write_outputs(
            outdir,
            paired["run"],
            paired["sample"],
            paired["event"],
            paired["region"],
            summary,
            artifacts,
            args.allow_partial,
        )
    except Exception as error:
        write_failure_gate(outdir, error)
        print(f"[FAIL] {error}", file=sys.stderr)
        if os.environ.get("SEN12_ANALYZER_TRACEBACK") == "1":
            traceback.print_exc()
        return 1
    print(f"[DONE] strict_pass={not args.allow_partial} paired_runs={len(artifacts)} outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
