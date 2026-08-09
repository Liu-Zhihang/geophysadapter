#!/usr/bin/env python3
"""Strict cross-run analysis for the Sen12 role-aware TMR matrix.

The analysis unit is a held-out LOGO spatial fold.  Seed-level deltas are
averaged within fold before bootstrap and exact sign-flip inference.  TMR
aligned outputs are compared with the actual matched Terrain parent and with
role-specific negative controls.  Parent metrics are never reconstructed from
the child run or replaced by a synthetic row.
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RUNS_DEFAULT = PROJECT_ROOT / "experiments/revision2026/sen12_xdomain_tmr_v1"
TERRAIN_RUNS_DEFAULT = PROJECT_ROOT / "experiments/revision2026/sen12_xdomain_geophysadapter_v1"
RUN_PATTERN = re.compile(r"fold(?P<fold>\d+)_seed(?P<seed>\d+)$")
MODES = ("terrain_material", "terrain_trigger", "full_tmr")
TMR_CONTROLS = (
    "visual_anchor",
    "aligned",
    "zero_terrain",
    "terrain_roll64",
    "terrain_donor",
    "material_shuffled",
    "material_donor",
    "material_constant",
    "trigger_event_shuffled",
    "trigger_donor",
    "trigger_wrong_family",
    "trigger_constant",
)
PARENT_CONTROLS = ("visual_anchor", "aligned", "zero", "roll32", "roll64", "other_region_donor")
MODE_COMPARATORS = {
    "terrain_material": ("parent_terrain", "material_shuffled", "material_donor", "material_constant"),
    "terrain_trigger": (
        "parent_terrain",
        "trigger_event_shuffled",
        "trigger_donor",
        "trigger_wrong_family",
        "trigger_constant",
    ),
    "full_tmr": (
        "parent_terrain",
        "material_shuffled",
        "material_donor",
        "material_constant",
        "trigger_event_shuffled",
        "trigger_donor",
        "trigger_wrong_family",
        "trigger_constant",
    ),
}
RUN_ARTIFACTS = (
    "DONE.json",
    "result.json",
    "checkpoint.pt",
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
    "visual_state_sha256",
)
COUNT_KEYS = ("tp", "fp", "fn", "tn")
SHARED_COUNT_KEYS = tuple(f"shared_{key}" for key in COUNT_KEYS)
CORPUS_METRICS = (
    "iou",
    "average_precision",
    "brier",
    "region_macro_iou",
    "region_macro_average_precision",
    "region_macro_brier",
    "event_macro_iou",
    "event_macro_average_precision",
    "event_macro_brier",
    "shared_iou",
    "rer",
    "corrected",
    "harmed",
    "net_corrected",
    "visual_errors",
    "adapter_errors",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DEFAULT)
    parser.add_argument("--terrain-runs-dir", type=Path, default=TERRAIN_RUNS_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--expected-folds", default="0,1,2,3,4")
    parser.add_argument("--min-folds", type=int, default=5)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Development only; relax coverage gates and force DEVELOPMENT_PARTIAL status.",
    )
    args = parser.parse_args(argv)
    if args.min_folds < 1 or args.min_seeds < 1:
        parser.error("--min-folds and --min-seeds must be >=1")
    if args.bootstrap < 100 or args.permutations < 100:
        parser.error("--bootstrap and --permutations must be >=100")
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


def selected_state_sha256(state_dict: dict[str, Any], prefixes: tuple[str, ...]) -> str:
    selected = [(key, value) for key, value in state_dict.items() if key.startswith(prefixes)]
    if not selected:
        raise RuntimeError(f"No checkpoint tensors match prefixes={prefixes}")
    digest = hashlib.sha256()
    for key, tensor in sorted(selected):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"Checkpoint state is not a tensor: {key}")
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty artifact: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def load_checkpoint(path: Path) -> dict[str, Any]:
    require_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("identity"), dict):
        raise RuntimeError(f"Checkpoint identity missing: {path}")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"Checkpoint state missing: {path}")
    return payload


def load_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    require_file(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Empty CSV: {path}")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")
    return frame


def identity_subset(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: identity.get(key) for key in IDENTITY_FIELDS}


def test_audit(result: dict[str, Any], path: Path) -> dict[str, Any]:
    audits = result.get("identity_and_control_audits")
    matches = [item for item in audits or [] if isinstance(item, dict) and item.get("split") == "test"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one test identity audit: {path}")
    return matches[0]


def assert_csv_contract(
    frame: pd.DataFrame,
    path: Path,
    mode: str,
    fold: int,
    seed: int,
    unit: str,
    controls: Sequence[str],
) -> pd.DataFrame:
    required = {"mode", "fold", "seed", "split", "control", unit, "iou", "average_precision", "brier"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")
    test = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if test.empty:
        raise RuntimeError(f"No test rows: {path}")
    if set(test["mode"].astype(str)) != {mode}:
        raise RuntimeError(f"Mode mismatch in {path}")
    if set(pd.to_numeric(test["fold"]).astype(int)) != {fold}:
        raise RuntimeError(f"Fold mismatch in {path}")
    if set(pd.to_numeric(test["seed"]).astype(int)) != {seed}:
        raise RuntimeError(f"Seed mismatch in {path}")
    if set(test["control"].astype(str)) != set(controls):
        raise RuntimeError(f"Control mismatch in {path}")
    if test.duplicated([unit, "control"]).any():
        raise RuntimeError(f"Duplicate {unit}/control rows in {path}")
    identities = {
        control: test.loc[test["control"].eq(control), unit].astype(str).tolist()
        for control in controls
    }
    if len({tuple(values) for values in identities.values()}) != 1 or not next(iter(identities.values())):
        raise RuntimeError(f"{unit} identity/order differs by control in {path}")
    return test


def corpus_rows(result: dict[str, Any], mode: str, fold: int, seed: int, controls: Sequence[str]) -> pd.DataFrame:
    rows = result.get("corpus_metrics")
    if not isinstance(rows, list):
        raise RuntimeError(f"Missing corpus_metrics for mode={mode} fold={fold} seed={seed}")
    frame = pd.DataFrame(rows)
    required = {
        "mode", "fold", "seed", "split", "control", "iou", "average_precision", "brier",
        "region_macro_iou", "region_macro_average_precision", "region_macro_brier",
        "event_macro_iou", "event_macro_average_precision", "event_macro_brier",
        "visual_errors", "adapter_errors", "corrected", "harmed", "net_corrected", "rer",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Corpus columns missing for mode={mode} fold={fold} seed={seed}: {missing}")
    frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if set(frame["control"].astype(str)) != set(controls) or frame["control"].duplicated().any():
        raise RuntimeError(f"Corpus control coverage mismatch for mode={mode} fold={fold} seed={seed}")
    if set(frame["mode"].astype(str)) != {mode} or set(frame["fold"].astype(int)) != {fold} or set(frame["seed"].astype(int)) != {seed}:
        raise RuntimeError(f"Corpus run identity mismatch for mode={mode} fold={fold} seed={seed}")
    return frame


def counts_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (int(counts[key]) for key in COUNT_KEYS)
    denominator = tp + fp + fn
    return {
        "shared_iou": tp / denominator if denominator else 0.0,
        "shared_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def shared_summary(sample: pd.DataFrame, control: str) -> dict[str, Any]:
    selected = sample.loc[sample["control"].eq(control)]
    missing = sorted(set((*SHARED_COUNT_KEYS, "visual_errors", "adapter_errors", "corrected", "harmed")) - set(selected.columns))
    if missing:
        raise RuntimeError(f"Shared-threshold columns missing for control={control}: {missing}")
    counts = {key: int(selected[f"shared_{key}"].sum()) for key in COUNT_KEYS}
    visual_errors = int(selected["visual_errors"].sum())
    adapter_errors = int(selected["adapter_errors"].sum())
    corrected = int(selected["corrected"].sum())
    harmed = int(selected["harmed"].sum())
    output = counts_metrics(counts)
    output.update({
        "visual_errors": visual_errors,
        "adapter_errors": adapter_errors,
        "corrected": corrected,
        "harmed": harmed,
        "net_corrected": corrected - harmed,
        "rer": (visual_errors - adapter_errors) / max(visual_errors, 1),
    })
    return output


def assert_corpus_matches_samples(corpus: pd.Series, shared: dict[str, Any], label: str) -> None:
    for key in ("visual_errors", "adapter_errors", "corrected", "harmed", "net_corrected"):
        if int(corpus[key]) != int(shared[key]):
            raise RuntimeError(f"Corpus/sample mismatch for {label}.{key}")
    if not math.isclose(float(corpus["rer"]), float(shared["rer"]), rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"Corpus/sample mismatch for {label}.rer")


def assert_aggregate_consistency(
    corpus: pd.Series,
    sample: pd.DataFrame,
    event: pd.DataFrame,
    region: pd.DataFrame,
    control: str,
    label: str,
) -> None:
    selected_sample = sample.loc[sample["control"].eq(control)]
    selected_event = event.loc[event["control"].eq(control)]
    selected_region = region.loc[region["control"].eq(control)]
    for key in COUNT_KEYS:
        expected = int(corpus[key])
        totals = {
            "sample": int(selected_sample[key].sum()),
            "event": int(selected_event[key].sum()),
            "region": int(selected_region[key].sum()),
        }
        if any(actual != expected for actual in totals.values()):
            raise RuntimeError(f"Count aggregation mismatch for {label}.{key}: corpus={expected} aggregates={totals}")
    for unit_name, frame, prefix in (
        ("event", selected_event, "event_macro"),
        ("region", selected_region, "region_macro"),
    ):
        for metric in ("iou", "average_precision", "brier"):
            actual = float(frame[metric].mean())
            expected = float(corpus[f"{prefix}_{metric}"])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"{unit_name} macro mismatch for {label}.{metric}: corpus={expected} aggregate={actual}"
                )


def load_artifact_run(run_dir: Path, mode: str, fold: int, seed: int, controls: Sequence[str]) -> dict[str, Any]:
    for name in RUN_ARTIFACTS:
        require_file(run_dir / name)
    done = load_json(run_dir / "DONE.json")
    result = load_json(run_dir / "result.json")
    checkpoint = load_checkpoint(run_dir / "checkpoint.pt")
    expected = {"mode": mode, "fold": fold, "seed": seed}
    for source_name, source in (("DONE", done), ("result", result), ("checkpoint", checkpoint["identity"])):
        mismatch = {key: (value, source.get(key)) for key, value in expected.items() if source.get(key) != value}
        if mismatch:
            raise RuntimeError(f"{source_name} identity mismatch in {run_dir}: {mismatch}")
    if done.get("status") != "complete":
        raise RuntimeError(f"DONE status is not complete: {run_dir}")
    if done.get("result_sha256") != sha256_file(run_dir / "result.json"):
        raise RuntimeError(f"DONE/result hash mismatch: {run_dir}")
    if done.get("checkpoint_sha256") != sha256_file(run_dir / "checkpoint.pt"):
        raise RuntimeError(f"DONE/checkpoint hash mismatch: {run_dir}")
    if result.get("identity") != checkpoint.get("identity"):
        raise RuntimeError(f"result/checkpoint identity mismatch: {run_dir}")

    sample = assert_csv_contract(
        load_csv(run_dir / "per_sample.csv", ("sample_id",)), run_dir / "per_sample.csv",
        mode, fold, seed, "sample_id", controls,
    )
    event = assert_csv_contract(
        load_csv(run_dir / "per_event.csv", ("physical_event_id",)), run_dir / "per_event.csv",
        mode, fold, seed, "physical_event_id", controls,
    )
    region = assert_csv_contract(
        load_csv(run_dir / "per_region.csv", ("region_group",)), run_dir / "per_region.csv",
        mode, fold, seed, "region_group", controls,
    )
    audit = test_audit(result, run_dir / "result.json")
    hashes = audit.get("sample_order_sha256_by_control", {})
    counts = audit.get("n_samples_by_control", {})
    if set(hashes) != set(controls) or set(counts) != set(controls):
        raise RuntimeError(f"Incomplete sample-order audit: {run_dir}")
    csv_hashes = {
        control: sha256_strings(sample.loc[sample["control"].eq(control), "sample_id"].astype(str).tolist())
        for control in controls
    }
    if csv_hashes != hashes or len(set(hashes.values())) != 1 or len(set(counts.values())) != 1:
        raise RuntimeError(f"Sample-order audit mismatch: {run_dir}")
    corpus = corpus_rows(result, mode, fold, seed, controls)
    for control in controls:
        shared = shared_summary(sample, control)
        corpus_row = corpus.loc[corpus["control"].eq(control)].iloc[0]
        assert_corpus_matches_samples(corpus_row, shared, f"{run_dir.name}/{control}")
        assert_aggregate_consistency(corpus_row, sample, event, region, control, f"{run_dir.name}/{control}")
    return {
        "dir": run_dir,
        "done": done,
        "result": result,
        "checkpoint": checkpoint,
        "sample": sample,
        "event": event,
        "region": region,
        "corpus": corpus,
        "sample_order_sha256": next(iter(hashes.values())),
    }


def assert_parent_identity(child: dict[str, Any], parent: dict[str, Any], mode: str, fold: int, seed: int) -> None:
    child_identity = child["result"]["identity"]
    parent_identity = parent["result"]["identity"]
    if identity_subset(child_identity) != identity_subset(parent_identity):
        raise RuntimeError(f"TMR/parent data identity mismatch for mode={mode} fold={fold} seed={seed}")
    result = child["result"]
    if result.get("terrain_parent_tensor_identity") is not True:
        raise RuntimeError(f"Frozen Terrain parent audit is not true for mode={mode} fold={fold} seed={seed}")
    before = result.get("terrain_parent_state_sha256_before")
    after = result.get("terrain_parent_state_sha256_after")
    identity_hash = child_identity.get("terrain_parent_state_sha256")
    actual = selected_state_sha256(
        parent["checkpoint"]["model_state_dict"],
        ("terrain_encoder.", "terrain_direction.", "gate_head."),
    )
    if not before or before != after or before != identity_hash or before != actual:
        raise RuntimeError(
            f"Terrain parent state mismatch for mode={mode} fold={fold} seed={seed}: "
            f"before={before} after={after} identity={identity_hash} actual={actual}"
        )
    child_config = child["checkpoint"].get("config")
    if not isinstance(child_config, dict):
        raise RuntimeError(f"TMR checkpoint config missing for mode={mode} fold={fold} seed={seed}")
    if child_config.get("terrain_checkpoint_sha256") != sha256_file(parent["dir"] / "checkpoint.pt"):
        raise RuntimeError(f"Recorded Terrain checkpoint hash mismatch for mode={mode} fold={fold} seed={seed}")
    if child_config.get("terrain_checkpoint_identity") != parent["checkpoint"]["identity"]:
        raise RuntimeError(f"Recorded Terrain checkpoint identity mismatch for mode={mode} fold={fold} seed={seed}")
    if child["sample_order_sha256"] != parent["sample_order_sha256"]:
        raise RuntimeError(f"TMR/parent sample order mismatch for mode={mode} fold={fold} seed={seed}")
    child_threshold = float(child["result"].get("visual_shared_threshold"))
    parent_threshold = float(parent["result"].get("visual_shared_threshold"))
    if child_threshold != parent_threshold:
        raise RuntimeError(f"Shared visual threshold mismatch for mode={mode} fold={fold} seed={seed}")
    for unit, child_frame, parent_frame in (
        ("sample_id", child["sample"], parent["sample"]),
        ("physical_event_id", child["event"], parent["event"]),
        ("region_group", child["region"], parent["region"]),
    ):
        left = child_frame.loc[child_frame["control"].eq("aligned"), unit].astype(str).tolist()
        right = parent_frame.loc[parent_frame["control"].eq("aligned"), unit].astype(str).tolist()
        if left != right:
            raise RuntimeError(f"TMR/parent {unit} order mismatch for mode={mode} fold={fold} seed={seed}")


def assert_tmr_audits(run: dict[str, Any], mode: str, fold: int, seed: int) -> None:
    audit = test_audit(run["result"], run["dir"] / "result.json")
    required_true = (
        "same_sample_identity_and_order",
        "zero_terrain_exact_fallback",
        "q_t_zero_exact_fallback",
        "q_m_zero_exact_identity",
        "q_r_zero_exact_identity",
        "q_m_q_r_zero_exact_parent_terrain_fallback",
        "inactive_controls_exact_identity",
    )
    failed = [key for key in required_true if audit.get(key) is not True]
    if failed or int(audit.get("other_region_donor_violations", -1)) != 0:
        raise RuntimeError(f"TMR control audit failed for mode={mode} fold={fold} seed={seed}: {failed}")


def discover(runs_dir: Path) -> dict[tuple[int, int], dict[str, Path]]:
    if not runs_dir.is_dir():
        raise RuntimeError(f"Runs directory does not exist: {runs_dir}")
    discovered: dict[tuple[int, int], dict[str, Path]] = {}
    for pair_dir in sorted(runs_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        match = RUN_PATTERN.fullmatch(pair_dir.name)
        if not match:
            continue
        key = (int(match.group("fold")), int(match.group("seed")))
        for mode in MODES:
            if (pair_dir / mode).is_dir():
                discovered.setdefault(key, {})[mode] = pair_dir / mode
    if not discovered:
        raise RuntimeError(f"No TMR fold/seed/mode directories under {runs_dir}")
    return discovered


def validate_coverage(
    discovered: dict[tuple[int, int], dict[str, Path]],
    expected_folds: set[int],
    min_folds: int,
    min_seeds: int,
    allow_partial: bool,
) -> list[tuple[int, int, str, Path]]:
    entries = [
        (fold, seed, mode, modes[mode])
        for (fold, seed), modes in sorted(discovered.items())
        for mode in MODES if mode in modes
    ]
    if allow_partial:
        return entries
    actual_folds = {fold for fold, _, _, _ in entries}
    if len(actual_folds) < min_folds or actual_folds != expected_folds:
        raise RuntimeError(f"Formal fold coverage mismatch: found={sorted(actual_folds)} expected={sorted(expected_folds)}")
    for fold in sorted(expected_folds):
        seed_sets = {
            mode: {seed for f, seed, candidate, _ in entries if f == fold and candidate == mode}
            for mode in MODES
        }
        if any(len(seeds) < min_seeds for seeds in seed_sets.values()):
            raise RuntimeError(f"fold={fold} has insufficient seeds by mode: {seed_sets}")
        if len({frozenset(seeds) for seeds in seed_sets.values()}) != 1:
            raise RuntimeError(f"fold={fold} mode seed sets are not identical: {seed_sets}")
    return entries


def row_metrics(run: dict[str, Any], control: str) -> dict[str, Any]:
    corpus = run["corpus"].loc[run["corpus"]["control"].eq(control)]
    if len(corpus) != 1:
        raise RuntimeError(f"Expected one corpus row for {run['dir']} control={control}")
    row = corpus.iloc[0]
    shared = shared_summary(run["sample"], control)
    output = {metric: float(row[metric]) for metric in CORPUS_METRICS if metric in row and metric not in shared}
    output.update(shared)
    return output


def bootstrap_mean_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    chunks: list[np.ndarray] = []
    remaining = draws
    while remaining:
        count = min(512, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        chunks.append(values[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(chunks)
    return float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def exact_or_mc_sign_flip(values: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[float, str, int]:
    observed = abs(float(values.mean()))
    tolerance = np.finfo(np.float64).eps * max(observed, 1.0) * 8
    if len(values) <= 20 and 2 ** len(values) <= permutations:
        total = 2 ** len(values)
        extreme = sum(
            abs(float(np.dot(values, np.asarray(signs)) / len(values))) + tolerance >= observed
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        )
        return extreme / total, "exact_sign_flip", total
    extreme = 0
    remaining = permutations
    while remaining:
        count = min(4096, remaining)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        extreme += int(np.count_nonzero(np.abs((signs @ values) / len(values)) + tolerance >= observed))
        remaining -= count
    return (extreme + 1) / (permutations + 1), "monte_carlo_sign_flip", permutations


def paired_stats(values: Iterable[float], bootstrap: int, permutations: int, token: str) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise RuntimeError(f"No finite fold deltas for {token}")
    seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    low, high = bootstrap_mean_ci(array, bootstrap, rng)
    p_value, method, draws = exact_or_mc_sign_flip(array, permutations, rng)
    sd = float(array.std(ddof=1)) if len(array) > 1 else math.nan
    return {
        "n_folds": int(len(array)),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95": [low, high],
        "positive_fraction": float(np.mean(array > 0)),
        "permutation_p_two_sided": p_value,
        "permutation_method": method,
        "permutation_draws": draws,
        "cohens_dz": float(array.mean() / sd) if math.isfinite(sd) and sd > 0 else math.nan,
    }


def clean_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    frame.replace([np.inf, -np.inf], np.nan).to_csv(path, index=False, na_rep="")


def fmt(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Sen12 role-aware TMR paired analysis",
        "",
        f"- evidence status: `{summary['status']}`",
        f"- validated TMR runs: `{summary['n_validated_runs']}`",
        f"- folds: `{summary['folds']}`",
        "- AP and Brier are threshold-free. Operational IoU uses each model's validation-frozen threshold.",
        "- `shared_iou`, RER, corrected and harmed use the same matched visual threshold in child and parent.",
        "- Inference averages seed deltas within each held-out fold; pixels and patches are not independent replicates.",
        "",
        "## Increment beyond the matched Terrain parent",
        "",
        "| mode | metric | fold-level delta | 95% CI | exact/MC sign-flip p |",
        "|---|---|---:|---:|---:|",
    ]
    for mode in MODES:
        stats = summary["statistics"].get(mode, {}).get("aligned_vs_parent_terrain", {})
        for metric in ("iou", "average_precision", "brier", "shared_iou", "rer", "event_macro_iou", "region_macro_iou"):
            item = stats.get(metric)
            if item:
                lines.append(
                    f"| {mode} | {metric} | {fmt(item['mean_delta'])} | "
                    f"[{fmt(item['ci95'][0])}, {fmt(item['ci95'][1])}] | {fmt(item['permutation_p_two_sided'])} |"
                )
    lines.extend([
        "",
        "## Role-specific negative controls",
        "",
        "| mode | comparator | delta IoU | 95% CI | delta RER | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        for comparator in MODE_COMPARATORS[mode]:
            if comparator == "parent_terrain":
                continue
            stats = summary["statistics"].get(mode, {}).get(f"aligned_vs_{comparator}", {})
            if not stats:
                continue
            iou, rer = stats["iou"], stats["rer"]
            lines.append(
                f"| {mode} | {comparator} | {fmt(iou['mean_delta'])} | "
                f"[{fmt(iou['ci95'][0])}, {fmt(iou['ci95'][1])}] | {fmt(rer['mean_delta'])} | "
                f"[{fmt(rer['ci95'][0])}, {fmt(rer['ci95'][1])}] |"
            )
    lines.extend([
        "",
        "## Evidence guardrail",
        "",
        "`DEVELOPMENT_PARTIAL` is directional engineering evidence only. A smoke run cannot be promoted to manuscript evidence by accumulating convenient comparisons; formal status additionally requires the preregistered five folds, at least five matched seeds per mode and fold, complete parent identity, and all negative controls.",
    ])
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    runs_dir = args.runs_dir.expanduser().resolve()
    terrain_runs_dir = args.terrain_runs_dir.expanduser().resolve()
    if "smoke" in runs_dir.name.lower() and not args.allow_partial:
        raise RuntimeError("Smoke directories can only be analyzed with --allow-partial")
    expected_folds = {int(value.strip()) for value in args.expected_folds.split(",") if value.strip()}
    if len(expected_folds) < args.min_folds:
        raise RuntimeError("--expected-folds contains fewer folds than --min-folds")
    entries = validate_coverage(
        discover(runs_dir), expected_folds, args.min_folds, args.min_seeds, args.allow_partial
    )
    cache_parent: dict[tuple[int, int], dict[str, Any]] = {}
    loaded: dict[tuple[int, int, str], dict[str, Any]] = {}
    artifact_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for fold, seed, mode, run_dir in entries:
        try:
            child = load_artifact_run(run_dir, mode, fold, seed, TMR_CONTROLS)
            parent_key = (fold, seed)
            if parent_key not in cache_parent:
                parent_dir = terrain_runs_dir / f"fold{fold}_seed{seed}" / "adapter"
                cache_parent[parent_key] = load_artifact_run(parent_dir, "adapter", fold, seed, PARENT_CONTROLS)
            parent = cache_parent[parent_key]
            assert_tmr_audits(child, mode, fold, seed)
            assert_parent_identity(child, parent, mode, fold, seed)
            loaded[(fold, seed, mode)] = child
            artifact_rows.append({
                "fold": fold,
                "seed": seed,
                "mode": mode,
                "run_dir": str(run_dir.resolve()),
                "result_sha256": sha256_file(run_dir / "result.json"),
                "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt"),
                "parent_dir": str(parent["dir"].resolve()),
                "parent_result_sha256": sha256_file(parent["dir"] / "result.json"),
                "parent_checkpoint_sha256": sha256_file(parent["dir"] / "checkpoint.pt"),
                "sample_order_sha256": child["sample_order_sha256"],
            })
        except Exception as error:
            if not args.allow_partial:
                raise
            invalid_rows.append({
                "fold": fold,
                "seed": seed,
                "mode": mode,
                "run_dir": str(run_dir.resolve()),
                "error_type": type(error).__name__,
                "error": str(error),
            })
    if not loaded:
        raise RuntimeError("No artifact-complete TMR runs survived validation")
    if not args.allow_partial:
        validated = {(fold, seed, mode) for fold, seed, mode in loaded}
        expected = {(fold, seed, mode) for fold, seed, mode, _ in entries}
        if validated != expected:
            raise RuntimeError("Formal analysis lost one or more discovered runs during validation")

    run_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for (fold, seed, mode), child in sorted(loaded.items()):
        parent = cache_parent[(fold, seed)]
        available = {"parent_terrain": row_metrics(parent, "aligned")}
        for control in MODE_COMPARATORS[mode]:
            if control != "parent_terrain":
                available[control] = row_metrics(child, control)
        aligned = row_metrics(child, "aligned")
        for control, source, metrics in (
            ("aligned", "tmr", aligned),
            ("parent_terrain", "parent", available["parent_terrain"]),
            *((control, "tmr_control", available[control]) for control in MODE_COMPARATORS[mode] if control != "parent_terrain"),
        ):
            run_rows.append({"mode": mode, "fold": fold, "seed": seed, "control": control, "source": source, **metrics})
        for comparator, comparator_metrics in available.items():
            row = {"mode": mode, "fold": fold, "seed": seed, "aligned": "aligned", "comparator": comparator}
            for metric in CORPUS_METRICS:
                row[f"aligned_{metric}"] = aligned[metric]
                row[f"comparator_{metric}"] = comparator_metrics[metric]
                row[f"delta_{metric}"] = aligned[metric] - comparator_metrics[metric]
            delta_rows.append(row)
    run_level = pd.DataFrame(run_rows).sort_values(["mode", "fold", "seed", "control"]).reset_index(drop=True)
    control_deltas = pd.DataFrame(delta_rows).sort_values(["mode", "comparator", "fold", "seed"]).reset_index(drop=True)
    aggregation = {
        f"delta_{metric}": (f"delta_{metric}", "mean")
        for metric in CORPUS_METRICS
    }
    fold_level = control_deltas.groupby(["mode", "comparator", "fold"], as_index=False).agg(
        n_seeds=("seed", "nunique"), **aggregation
    ).sort_values(["mode", "comparator", "fold"]).reset_index(drop=True)

    statistics: dict[str, Any] = {}
    for mode in MODES:
        mode_stats: dict[str, Any] = {}
        for comparator in MODE_COMPARATORS[mode]:
            selected = fold_level.loc[(fold_level["mode"] == mode) & (fold_level["comparator"] == comparator)]
            if selected.empty:
                continue
            mode_stats[f"aligned_vs_{comparator}"] = {
                metric: paired_stats(
                    selected[f"delta_{metric}"], args.bootstrap, args.permutations, f"{mode}|{comparator}|{metric}"
                )
                for metric in CORPUS_METRICS
            }
        statistics[mode] = mode_stats
    status = "DEVELOPMENT_PARTIAL" if args.allow_partial else "PASS"
    summary = {
        "status": status,
        "strict_pass": not args.allow_partial,
        "manuscript_evidence_eligible": not args.allow_partial,
        "runs_dir": str(runs_dir),
        "terrain_runs_dir": str(terrain_runs_dir),
        "n_validated_runs": len(loaded),
        "folds": sorted({fold for fold, _, _ in loaded}),
        "seeds_by_fold_and_mode": {
            str(fold): {
                mode: sorted(seed for f, seed, candidate in loaded if f == fold and candidate == mode)
                for mode in MODES
            }
            for fold in sorted({fold for fold, _, _ in loaded})
        },
        "min_folds": args.min_folds,
        "min_seeds": args.min_seeds,
        "threshold_contract": {
            "iou": "each model's validation-frozen operational threshold",
            "average_precision_brier": "threshold-free",
            "shared_iou_rer_corrected_harmed": "matched visual validation threshold shared by TMR and Terrain parent",
        },
        "independence_contract": "average matched seed deltas within each held-out spatial fold, then resample/sign-flip folds",
        "smoke_guardrail": "Smoke and --allow-partial analyses are DEVELOPMENT_PARTIAL and cannot become manuscript evidence automatically.",
        "comparisons_by_mode": {mode: list(MODE_COMPARATORS[mode]) for mode in MODES},
        "statistics": statistics,
        "validated_artifacts": artifact_rows,
        "invalid_runs_rejected": invalid_rows,
    }
    return run_level, fold_level, control_deltas, summary


def write_outputs(outdir: Path, run_level: pd.DataFrame, fold_level: pd.DataFrame, control_deltas: pd.DataFrame, summary: dict[str, Any]) -> None:
    temp = outdir.with_name(f".{outdir.name}.tmp.{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        clean_csv(run_level, temp / "run_level.csv")
        clean_csv(fold_level, temp / "fold_level.csv")
        clean_csv(control_deltas, temp / "control_deltas.csv")
        (temp / "summary.json").write_text(
            json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        (temp / "report.md").write_text(build_report(summary), encoding="utf-8")
        for name in ("run_level.csv", "fold_level.csv", "control_deltas.csv", "summary.json", "report.md"):
            require_file(temp / name)
        if outdir.exists():
            shutil.rmtree(outdir)
        temp.rename(outdir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def write_failure(outdir: Path, error: BaseException) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("run_level.csv", "fold_level.csv", "control_deltas.csv", "summary.json", "report.md"):
        (outdir / name).unlink(missing_ok=True)
    (outdir / "analysis_failure.json").write_text(
        json.dumps({"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = (args.outdir or args.runs_dir / "analysis").expanduser().resolve()
    try:
        run_level, fold_level, control_deltas, summary = analyze(args)
        write_outputs(outdir, run_level, fold_level, control_deltas, summary)
    except Exception as error:
        write_failure(outdir, error)
        print(f"[FAIL] {error}", file=sys.stderr)
        if os.environ.get("SEN12_TMR_ANALYZER_TRACEBACK") == "1":
            traceback.print_exc()
        return 1
    print(f"[DONE] status={summary['status']} runs={summary['n_validated_runs']} outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
