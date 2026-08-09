#!/usr/bin/env python3
"""Strict paired analysis for unified PILD/Sen12 role-aware experiments.

Input contract
--------------
Each trained parent lives at ``<runs-root>/<variant>_seed<seed>`` and contains:

* ``DONE.json`` (strict completion marker and artifact hashes),
* ``config.json`` (terrain9 order and Material interaction groups),
* ``result.json`` (run identity),
* ``per_sample.csv`` (aligned and inference-time negative-control receipts), and
* ``per_sample_metrics.csv`` (the primary aligned variant only).

Exactly five checkpoints are trained per seed: V, VT, VTM, VTR, and VTMR.
Negative controls are selected from the appropriate parent's ``per_sample.csv``
and must carry that parent's checkpoint provenance; they are never discovered
or accepted as independent runs.  The analysis fails closed unless every
parent has exactly the same seeds and every selected condition has exactly the
same sample keys.  Seeds are treated as optimization repeats: receipts are
averaged across seeds before source/event/sample inference.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


RUN_SCHEMA = "pild_sen12_roleaware_run.v1"
DONE_SCHEMA = "pild_sen12_roleaware_done.v1"
CONFIG_SCHEMA = "pild_sen12_roleaware_config.v1"
ANALYSIS_SCHEMA = "pild_sen12_roleaware_analysis.v1"

TERRAIN9_CHANNEL_ORDER = (
    "elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "laplacian_curvature",
    "tpi_90m",
    "tpi_300m",
    "ruggedness_90m",
    "local_relief_300m",
)
MATERIAL_INTERACTION_GROUPS = {
    "slope": (1,),
    "curvature": (4,),
    "relief": (5, 6, 7, 8),
}

MAIN_CONDITIONS = ("V", "VT", "VTM", "VTR", "VTMR")
TERRAIN_CONTROLS = ("T_zero", "T_shift", "T_roll", "T_donor")
MATERIAL_CONDITIONS = ("M_aligned", "M_shuffle", "M_zero_q")
TRIGGER_CONDITIONS = (
    "R_aligned",
    "R_wrong_time",
    "R_event_shuffle",
    "R_zero_q",
)
CONDITIONS = (
    *MAIN_CONDITIONS,
    *TERRAIN_CONTROLS,
    *MATERIAL_CONDITIONS,
    *TRIGGER_CONDITIONS,
)

CONDITION_PARENT = {
    "V": "V",
    "VT": "VT",
    "VTM": "VTM",
    "VTR": "VTR",
    "VTMR": "VTMR",
    **{condition: "VT" for condition in TERRAIN_CONTROLS},
    **{condition: "VTM" for condition in MATERIAL_CONDITIONS},
    **{condition: "VTR" for condition in TRIGGER_CONDITIONS},
}

# Conditions physically emitted by each selected checkpoint.  VTMR contains
# additional joint diagnostic rows, but only its aligned row enters this
# analysis; role-specific controls deliberately come from VTM and VTR.
PARENT_CSV_CONDITIONS = {
    "V": ("V",),
    "VT": ("VT", *TERRAIN_CONTROLS),
    "VTM": ("VTM", *MATERIAL_CONDITIONS),
    "VTR": ("VTR", *TRIGGER_CONDITIONS),
    "VTMR": (
        "VTMR",
        "M_shuffle",
        "M_zero_q",
        "R_wrong_time",
        "R_event_shuffle",
        "R_zero_q",
        "VTMR_material-trigger-both-zero-q",
    ),
}

EXPECTED_EVALUATION_CONTEXT = {
    "V": ("aligned", "aligned", "aligned", "aligned"),
    "VT": ("aligned", "aligned", "aligned", "aligned"),
    "VTM": ("aligned", "aligned", "aligned", "aligned"),
    "VTR": ("aligned", "aligned", "aligned", "aligned"),
    "VTMR": ("aligned", "aligned", "aligned", "aligned"),
    "T_zero": ("terrain-zero", "terrain-zero", "aligned", "aligned"),
    "T_shift": (
        "terrain-shift32-zero-pad",
        "terrain-shift32-zero-pad",
        "aligned",
        "aligned",
    ),
    "T_roll": (
        "terrain-roll64-circular",
        "terrain-roll64-circular",
        "aligned",
        "aligned",
    ),
    "T_donor": (
        "terrain-other-source-or-event-donor",
        "terrain-other-source-or-event-donor",
        "aligned",
        "aligned",
    ),
    "M_aligned": ("material-aligned", "aligned", "aligned", "aligned"),
    "M_shuffle": (
        "material-shuffle",
        "aligned",
        "within-source/event-shuffle",
        "aligned",
    ),
    "M_zero_q": ("material-zero-q", "aligned", "zero-q", "aligned"),
    "R_aligned": ("trigger-aligned", "aligned", "aligned", "aligned"),
    "R_wrong_time": ("trigger-wrong-time", "aligned", "aligned", "wrong-time"),
    "R_event_shuffle": (
        "trigger-event-shuffle",
        "aligned",
        "aligned",
        "event-shuffle",
    ),
    "R_zero_q": ("trigger-zero-q", "aligned", "aligned", "zero-q"),
}

# corrected/harmed are defined against these frozen references.  This makes
# error-flow auditable rather than inferred from aggregate IoU.
REFERENCE_CONDITION = {
    "V": "V",
    "VT": "V",
    "VTM": "VT",
    "VTR": "VT",
    "VTMR": "VT",
    "T_zero": "V",
    "T_shift": "V",
    "T_roll": "V",
    "T_donor": "V",
    "M_aligned": "VT",
    "M_shuffle": "VT",
    "M_zero_q": "VT",
    "R_aligned": "VT",
    "R_wrong_time": "VT",
    "R_event_shuffle": "VT",
    "R_zero_q": "VT",
}

# Positive efficacy contrasts and falsification controls.  The direction is
# always left minus right for higher-is-better metrics.
CONTRASTS = {
    "VT_minus_V": ("VT", "V", "all"),
    "VTM_minus_VT": ("VTM", "VT", "material_q_positive"),
    "VTR_minus_VT": ("VTR", "VT", "trigger_q_positive"),
    "VTMR_minus_VT": ("VTMR", "VT", "all"),
    "T_aligned_minus_zero": ("VT", "T_zero", "all"),
    "T_aligned_minus_shift": ("VT", "T_shift", "all"),
    "T_aligned_minus_roll": ("VT", "T_roll", "all"),
    "T_aligned_minus_donor": ("VT", "T_donor", "all"),
    "M_aligned_minus_shuffle": ("M_aligned", "M_shuffle", "material_q_positive"),
    "M_aligned_minus_zero_q": ("M_aligned", "M_zero_q", "material_q_positive"),
    "R_aligned_minus_wrong_time": (
        "R_aligned", "R_wrong_time", "trigger_q_positive"
    ),
    "R_aligned_minus_event_shuffle": (
        "R_aligned", "R_event_shuffle", "trigger_q_positive"
    ),
    "R_aligned_minus_zero_q": ("R_aligned", "R_zero_q", "trigger_q_positive"),
}

KEY_COLUMNS = ("source", "canonical_event_id", "sample_id")
COUNT_COLUMNS = ("tp", "fp", "fn", "tn")
TRIGGER_SUM_COLUMNS = (
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
REQUIRED_COLUMNS = (
    *KEY_COLUMNS,
    "condition",
    "seed",
    "reference_condition",
    "q_material",
    "q_trigger",
    *COUNT_COLUMNS,
    "ap",
    "corrected",
    "harmed",
    *TRIGGER_SUM_COLUMNS,
)
PROVENANCE_COLUMNS = (
    "split",
    "variant",
    "evaluation_context",
    "terrain_context",
    "material_context",
    "trigger_context",
    "checkpoint_sha256",
)
INTEGER_COLUMNS = (
    "seed",
    *COUNT_COLUMNS,
    "corrected",
    "harmed",
    "fixed_fpr_tp",
    "fixed_fpr_fn",
    "fixed_fpr_fp",
    "fixed_fpr_tn",
    "valid_pixel_count",
    "target_positive_count",
)
EPS = 1.0e-12


class AnalysisContractError(RuntimeError):
    """Raised when an input violates the frozen analysis contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AnalysisContractError(f"non-finite JSON constant in {path}: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AnalysisContractError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisContractError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AnalysisContractError(f"JSON root must be an object: {path}")
    return payload


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(
        json_safe(payload), ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
    )
    atomic_write_text(path, text + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    atomic_write_text(path, buffer.getvalue())


def discover_seed_sets(runs_root: Path) -> dict[str, set[int]]:
    if not runs_root.is_dir():
        raise AnalysisContractError(f"runs root is absent: {runs_root}")
    control_pattern = re.compile(
        rf"^(?:{'|'.join(map(re.escape, CONDITIONS[len(MAIN_CONDITIONS):]))})_seed[0-9]+$"
    )
    disguised_controls = sorted(
        path.name
        for path in runs_root.iterdir()
        if path.is_dir() and control_pattern.fullmatch(path.name)
    )
    if disguised_controls:
        raise AnalysisContractError(
            "negative controls must be rows from a parent per_sample.csv, not "
            f"independent runs: {disguised_controls[:5]}"
        )
    result: dict[str, set[int]] = {}
    for variant in MAIN_CONDITIONS:
        pattern = re.compile(rf"^{re.escape(variant)}_seed([0-9]+)$")
        seeds = {
            int(match.group(1))
            for path in runs_root.iterdir()
            if path.is_dir() and (match := pattern.fullmatch(path.name))
        }
        if not seeds:
            raise AnalysisContractError(f"no runs found for required parent variant {variant}")
        result[variant] = seeds
    return result


def resolve_seeds(
    runs_root: Path, requested: tuple[int, ...] | None, min_seeds: int
) -> tuple[int, ...]:
    seed_sets = discover_seed_sets(runs_root)
    if requested is None:
        unique_sets = {tuple(sorted(values)) for values in seed_sets.values()}
        if len(unique_sets) != 1:
            detail = {key: sorted(value) for key, value in seed_sets.items()}
            raise AnalysisContractError(f"parent variant seed sets differ: {detail}")
        seeds = next(iter(unique_sets))
    else:
        if len(set(requested)) != len(requested):
            raise AnalysisContractError("--seeds contains duplicates")
        seeds = tuple(sorted(requested))
        expected = set(seeds)
        differences = {
            condition: sorted(values ^ expected)
            for condition, values in seed_sets.items()
            if values != expected
        }
        if differences:
            raise AnalysisContractError(
                f"discovered runs do not equal requested seed inventory: {differences}"
            )
    if len(seeds) < min_seeds:
        raise AnalysisContractError(
            f"only {len(seeds)} complete seed identities found; require >= {min_seeds}"
        )
    return tuple(int(seed) for seed in seeds)


def _require_hash(done: Mapping[str, Any], key: str, path: Path) -> None:
    expected = done.get(key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise AnalysisContractError(f"missing/invalid {key} in {path.parent / 'DONE.json'}")
    observed = sha256_file(path)
    if observed != expected:
        raise AnalysisContractError(f"artifact hash mismatch for {path}: {observed} != {expected}")


def _require_artifact_hash(done: Mapping[str, Any], name: str, path: Path) -> None:
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise AnalysisContractError(f"DONE artifacts missing {name}: {path.parent}")
    receipt = artifacts[name]
    expected = receipt.get("sha256")
    try:
        expected_size = int(receipt.get("size", -1))
    except (TypeError, ValueError) as error:
        raise AnalysisContractError(f"invalid DONE artifact size for {name}: {path.parent}") from error
    if not isinstance(expected, str) or len(expected) != 64:
        raise AnalysisContractError(f"invalid DONE artifact hash for {name}: {path.parent}")
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise AnalysisContractError(
            f"artifact size mismatch for {path}: {observed_size} != {expected_size}"
        )
    observed = sha256_file(path)
    if observed != expected:
        raise AnalysisContractError(f"artifact hash mismatch for {path}: {observed} != {expected}")


def _coerce_integer_column(frame: pd.DataFrame, column: str, run: Path) -> None:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
        raise AnalysisContractError(f"{column} must contain integers: {run}")
    if column != "seed" and (numeric < 0).any():
        raise AnalysisContractError(f"{column} must be nonnegative: {run}")
    frame[column] = numeric.astype(np.int64)


def _normalize_condition_frame(
    frame: pd.DataFrame,
    *,
    run: Path,
    parent: str,
    condition: str,
    seed: int,
    checkpoint_sha256: str,
) -> pd.DataFrame:
    missing_columns = sorted(
        set((*REQUIRED_COLUMNS, *PROVENANCE_COLUMNS)) - set(frame.columns)
    )
    if missing_columns:
        raise AnalysisContractError(f"missing columns in {run}: {missing_columns}")
    frame = frame.loc[:, list(dict.fromkeys((*REQUIRED_COLUMNS, *PROVENANCE_COLUMNS)))].copy()
    if frame.empty:
        raise AnalysisContractError(f"negative-control or aligned condition has no rows: {run} {condition}")
    if frame.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise AnalysisContractError(f"null sample identity in {run} condition={condition}")
    for column in KEY_COLUMNS:
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise AnalysisContractError(f"blank {column} in {run} condition={condition}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise AnalysisContractError(f"duplicate sample keys in {run} condition={condition}")
    if set(frame["split"].astype(str)) != {"test"}:
        raise AnalysisContractError(f"selected rows must use split=test: {run} {condition}")
    if set(frame["variant"].astype(str)) != {parent}:
        raise AnalysisContractError(f"CSV parent variant identity mismatch: {run} {condition}")
    if set(frame["condition"].astype(str)) != {condition}:
        raise AnalysisContractError(f"CSV condition identity mismatch: {run} {condition}")
    for column in INTEGER_COLUMNS:
        _coerce_integer_column(frame, column, run)
    if set(frame["seed"].tolist()) != {seed}:
        raise AnalysisContractError(f"CSV seed identity mismatch: {run} {condition}")
    expected_reference = REFERENCE_CONDITION[condition]
    if set(frame["reference_condition"].astype(str)) != {expected_reference}:
        raise AnalysisContractError(
            f"reference_condition for {condition} must be {expected_reference}: {run}"
        )
    context_columns = (
        "evaluation_context",
        "terrain_context",
        "material_context",
        "trigger_context",
    )
    expected_context = EXPECTED_EVALUATION_CONTEXT[condition]
    observed_context = tuple(
        next(iter(set(frame[column].astype(str))))
        if frame[column].astype(str).nunique() == 1
        else "<multiple>"
        for column in context_columns
    )
    if observed_context != expected_context:
        raise AnalysisContractError(
            f"evaluation context mismatch for {condition}: {observed_context} != "
            f"{expected_context} in {run}"
        )
    if set(frame["checkpoint_sha256"].astype(str)) != {checkpoint_sha256}:
        raise AnalysisContractError(
            f"same-checkpoint provenance mismatch for {condition}: {run}"
        )
    for column in ("q_material", "q_trigger"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (
            frame[column].isna().any()
            or (~np.isfinite(frame[column])).any()
            or (frame[column] < 0).any()
            or (frame[column] > 1).any()
        ):
            raise AnalysisContractError(f"{column} must be finite in [0,1]: {run}")
    for column in ("brier_sum", "nll_sum", "soft_area_error"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or (~np.isfinite(frame[column])).any():
            raise AnalysisContractError(f"{column} must be finite: {run}")
    if (frame[["brier_sum", "nll_sum"]] < 0).any().any():
        raise AnalysisContractError(f"Brier/NLL sums must be nonnegative: {run}")
    frame["ap"] = pd.to_numeric(frame["ap"], errors="coerce")
    invalid_ap = frame["ap"].notna() & ((frame["ap"] < 0) | (frame["ap"] > 1))
    if invalid_ap.any() or np.isinf(frame["ap"].fillna(0)).any():
        raise AnalysisContractError(f"AP must be in [0,1] or missing: {run}")
    if (frame["valid_pixel_count"] <= 0).any():
        raise AnalysisContractError(f"valid_pixel_count must be positive: {run}")
    if not np.array_equal(
        frame["fixed_fpr_tp"] + frame["fixed_fpr_fn"],
        frame["target_positive_count"],
    ):
        raise AnalysisContractError(f"fixed-FPR positive counts are inconsistent: {run}")
    fixed_total = frame[
        ["fixed_fpr_tp", "fixed_fpr_fn", "fixed_fpr_fp", "fixed_fpr_tn"]
    ].sum(axis=1)
    if not np.array_equal(fixed_total, frame["valid_pixel_count"]):
        raise AnalysisContractError(f"fixed-FPR counts do not sum to valid pixels: {run}")
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def load_parent_run(
    runs_root: Path, parent: str, seed: int
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if parent not in MAIN_CONDITIONS:
        raise AnalysisContractError(f"unknown parent variant: {parent}")
    run = runs_root / f"{parent}_seed{seed}"
    required = [
        run / "DONE.json",
        run / "config.json",
        run / "result.json",
        run / "checkpoint.pt",
        run / "per_sample.csv",
        run / "per_sample_metrics.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AnalysisContractError(
            f"incomplete parent run variant={parent} seed={seed}: {missing}"
        )
    done = strict_json(run / "DONE.json")
    config = strict_json(run / "config.json")
    result = strict_json(run / "result.json")
    if done.get("schema_version") != DONE_SCHEMA or done.get("status") != "complete":
        raise AnalysisContractError(f"invalid DONE schema/status: {run}")
    if result.get("schema_version") != RUN_SCHEMA or result.get("status") != "complete":
        raise AnalysisContractError(f"invalid result schema/status: {run}")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise AnalysisContractError(f"invalid config schema: {run}")
    if result.get("evaluation_split") != "test":
        raise AnalysisContractError(f"result evaluation_split must be test: {run}")
    if result.get("fixed_fpr_threshold_source") != "validation_visual_only":
        raise AnalysisContractError(
            f"fixed-FPR threshold must come from validation_visual_only: {run}"
        )
    for payload, label in ((done, "DONE"), (config, "config"), (result, "result")):
        try:
            payload_seed = int(payload.get("seed", -1))
        except (TypeError, ValueError) as error:
            raise AnalysisContractError(f"{label} seed is invalid: {run}") from error
        if (
            payload.get("condition") != parent
            or payload.get("variant", parent) != parent
            or payload_seed != seed
        ):
            raise AnalysisContractError(f"{label} identity mismatch: {run}")
    config_identity = config.get("identity")
    result_identity = result.get("identity")
    if not isinstance(config_identity, dict) or config_identity != result_identity:
        raise AnalysisContractError(f"config/result identity mismatch: {run}")
    try:
        identity_seed = int(config_identity.get("seed", -1))
    except (TypeError, ValueError) as error:
        raise AnalysisContractError(f"run identity seed is invalid: {run}") from error
    if identity_seed != seed:
        raise AnalysisContractError(f"run identity seed mismatch: {run}")
    observed_order = config.get("terrain_channel_order")
    if not isinstance(observed_order, list) or tuple(map(str, observed_order)) != TERRAIN9_CHANNEL_ORDER:
        raise AnalysisContractError(
            f"terrain9 channel order mismatch in {run}: {observed_order}"
        )
    observed_groups = config.get("material_interaction_groups")
    if not isinstance(observed_groups, dict):
        raise AnalysisContractError(f"material_interaction_groups missing in {run}")
    try:
        normalized_groups = {
            str(name): tuple(int(index) for index in indices)
            for name, indices in observed_groups.items()
        }
    except (TypeError, ValueError) as error:
        raise AnalysisContractError(
            f"material_interaction_groups malformed in {run}"
        ) from error
    if normalized_groups != MATERIAL_INTERACTION_GROUPS:
        raise AnalysisContractError(
            f"Material interaction groups mismatch in {run}: {normalized_groups}"
        )
    try:
        fixed_fpr_threshold = float(result["fixed_fpr_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisContractError(f"fixed_fpr_threshold is missing/invalid: {run}") from error
    if not math.isfinite(fixed_fpr_threshold) or not 0 < fixed_fpr_threshold < 1:
        raise AnalysisContractError(f"fixed_fpr_threshold must lie in (0,1): {run}")
    _require_hash(done, "config_sha256", run / "config.json")
    _require_hash(done, "result_sha256", run / "result.json")
    _require_hash(done, "per_sample_metrics_sha256", run / "per_sample_metrics.csv")
    for artifact in (
        "config.json",
        "result.json",
        "checkpoint.pt",
        "per_sample.csv",
        "per_sample_metrics.csv",
    ):
        _require_artifact_hash(done, artifact, run / artifact)
    checkpoint_sha256 = sha256_file(run / "checkpoint.pt")
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise AnalysisContractError(f"result checkpoint provenance mismatch: {run}")
    try:
        all_rows = pd.read_csv(run / "per_sample.csv")
        primary_rows = pd.read_csv(run / "per_sample_metrics.csv")
    except Exception as error:
        raise AnalysisContractError(f"cannot read per-sample CSVs {run}: {error}") from error
    if all_rows.empty or primary_rows.empty:
        raise AnalysisContractError(f"empty per-sample CSV artifact: {run}")
    if "split" not in all_rows or "condition" not in all_rows:
        raise AnalysisContractError(f"per_sample.csv lacks split/condition columns: {run}")
    test_rows = all_rows.loc[all_rows["split"].astype(str) == "test"].copy()
    observed_conditions = set(test_rows["condition"].astype(str))
    expected_conditions = set(PARENT_CSV_CONDITIONS[parent])
    if observed_conditions != expected_conditions:
        raise AnalysisContractError(
            f"test condition inventory mismatch for parent {parent}: "
            f"observed={sorted(observed_conditions)} expected={sorted(expected_conditions)}"
        )

    selected: dict[str, pd.DataFrame] = {}
    for condition, mapped_parent in CONDITION_PARENT.items():
        if mapped_parent != parent:
            continue
        condition_rows = test_rows.loc[
            test_rows["condition"].astype(str) == condition
        ].copy()
        normalized = _normalize_condition_frame(
            condition_rows,
            run=run,
            parent=parent,
            condition=condition,
            seed=seed,
            checkpoint_sha256=checkpoint_sha256,
        )
        normalized["fixed_fpr_threshold"] = fixed_fpr_threshold
        selected[condition] = normalized

    normalized_primary = _normalize_condition_frame(
        primary_rows,
        run=run,
        parent=parent,
        condition=parent,
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
    )
    normalized_primary["fixed_fpr_threshold"] = fixed_fpr_threshold
    comparison_columns = list(
        dict.fromkeys((*REQUIRED_COLUMNS, *PROVENANCE_COLUMNS, "fixed_fpr_threshold"))
    )
    try:
        pd.testing.assert_frame_equal(
            selected[parent].loc[:, comparison_columns],
            normalized_primary.loc[:, comparison_columns],
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise AnalysisContractError(
            f"per_sample_metrics.csv is not the primary row subset of per_sample.csv: {run}"
        ) from error

    inventory = {
        "variant": parent,
        "seed": seed,
        "identity": json_safe(config_identity),
        "config_sha256": sha256_file(run / "config.json"),
        "result_sha256": sha256_file(run / "result.json"),
        "checkpoint_sha256": checkpoint_sha256,
        "per_sample_sha256": sha256_file(run / "per_sample.csv"),
        "per_sample_metrics_sha256": sha256_file(run / "per_sample_metrics.csv"),
        "analysis_conditions": sorted(selected),
    }
    return selected, inventory


def load_run(runs_root: Path, condition: str, seed: int) -> pd.DataFrame:
    """Load one analysis condition from its frozen trained parent."""

    if condition not in CONDITION_PARENT:
        raise AnalysisContractError(f"unknown analysis condition: {condition}")
    selected, _ = load_parent_run(runs_root, CONDITION_PARENT[condition], seed)
    return selected[condition]


def load_all_runs(
    runs_root: Path, seeds: tuple[int, ...]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    expected_keys: pd.MultiIndex | None = None
    support_reference: pd.DataFrame | None = None
    ground_truth_reference: pd.DataFrame | None = None
    threshold_by_seed: dict[int, float] = {}
    protocol_identity: dict[str, Any] | None = None
    for parent in MAIN_CONDITIONS:
        for seed in seeds:
            selected, parent_inventory = load_parent_run(runs_root, parent, seed)
            current_protocol_identity = {
                key: value
                for key, value in parent_inventory["identity"].items()
                if key != "seed"
            }
            if protocol_identity is None:
                protocol_identity = current_protocol_identity
            elif current_protocol_identity != protocol_identity:
                raise AnalysisContractError(
                    f"parent runs do not share one protocol identity: {parent} seed={seed}"
                )
            inventory.append(parent_inventory)
            for condition, frame in selected.items():
                keys = pd.MultiIndex.from_frame(frame.loc[:, list(KEY_COLUMNS)])
                if expected_keys is None:
                    expected_keys = keys
                elif not keys.equals(expected_keys):
                    missing = expected_keys.difference(keys).tolist()[:5]
                    extra = keys.difference(expected_keys).tolist()[:5]
                    raise AnalysisContractError(
                        f"sample pairing differs for {condition} seed={seed}; "
                        f"missing={missing} extra={extra}"
                    )
                current_support = frame.loc[:, [*KEY_COLUMNS, "q_material", "q_trigger"]]
                if support_reference is None:
                    support_reference = current_support
                elif not np.allclose(
                    current_support[["q_material", "q_trigger"]],
                    support_reference[["q_material", "q_trigger"]],
                    rtol=0,
                    atol=EPS,
                ):
                    raise AnalysisContractError(
                        f"q-material/q-trigger values differ across paired rows: "
                        f"{condition} seed={seed}"
                    )
                current_ground_truth = pd.DataFrame(
                    {
                        "target_positive": frame["tp"] + frame["fn"],
                        "valid_pixel_count": frame["valid_pixel_count"],
                        "trigger_target_positive": frame["target_positive_count"],
                    }
                )
                if ground_truth_reference is None:
                    ground_truth_reference = current_ground_truth
                elif not np.array_equal(current_ground_truth, ground_truth_reference):
                    raise AnalysisContractError(
                        f"ground-truth pixel receipts differ across paired rows: "
                        f"{condition} seed={seed}"
                    )
                threshold = float(frame["fixed_fpr_threshold"].iloc[0])
                if frame["fixed_fpr_threshold"].nunique() != 1:
                    raise AnalysisContractError(
                        f"multiple fixed-FPR thresholds in {condition} seed={seed}"
                    )
                if seed in threshold_by_seed and not math.isclose(
                    threshold, threshold_by_seed[seed], rel_tol=0, abs_tol=EPS
                ):
                    raise AnalysisContractError(
                        f"fixed-FPR threshold differs across conditions for seed {seed}"
                    )
                threshold_by_seed[seed] = threshold
                frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    _validate_error_flow(combined, seeds)
    return combined, inventory


def _validate_error_flow(frame: pd.DataFrame, seeds: tuple[int, ...]) -> None:
    indexed = frame.set_index(["condition", "seed", *KEY_COLUMNS]).sort_index()
    for condition in CONDITIONS:
        current = indexed.xs(condition, level="condition")
        reference_name = REFERENCE_CONDITION[condition]
        reference = indexed.xs(reference_name, level="condition")
        if not current.index.equals(reference.index):
            raise AnalysisContractError(f"reference pairing failed for {condition}")
        current_errors = current["fp"] + current["fn"]
        reference_errors = reference["fp"] + reference["fn"]
        net = current["corrected"] - current["harmed"]
        if not np.array_equal(reference_errors - current_errors, net):
            raise AnalysisContractError(
                f"corrected-harmed identity fails for condition {condition}"
            )
        if condition == "V" and (
            current["corrected"].ne(0).any() or current["harmed"].ne(0).any()
        ):
            raise AnalysisContractError("V self-reference must have zero corrected/harmed")
    observed = tuple(sorted(frame["seed"].unique().astype(int)))
    if observed != seeds:
        raise AnalysisContractError(f"loaded seed inventory differs: {observed} != {seeds}")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > EPS else float("nan")


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    denominator = output["tp"] + output["fp"] + output["fn"]
    output["iou"] = np.where(denominator > 0, output["tp"] / denominator, np.nan)
    output["errors"] = output["fp"] + output["fn"]
    output["brier"] = output["brier_sum"] / output["valid_pixel_count"]
    output["nll"] = output["nll_sum"] / output["valid_pixel_count"]
    output["area_error"] = output["soft_area_error"].abs() / output["valid_pixel_count"]
    recall_denominator = output["fixed_fpr_tp"] + output["fixed_fpr_fn"]
    output["fixed_fpr_recall"] = np.where(
        recall_denominator > 0,
        output["fixed_fpr_tp"] / recall_denominator,
        np.nan,
    )
    fpr_denominator = output["fixed_fpr_fp"] + output["fixed_fpr_tn"]
    output["fixed_fpr"] = np.where(
        fpr_denominator > 0,
        output["fixed_fpr_fp"] / fpr_denominator,
        np.nan,
    )
    return output


def average_seeds(frame: pd.DataFrame, seeds: tuple[int, ...]) -> pd.DataFrame:
    numeric = [
        *COUNT_COLUMNS,
        "q_material",
        "q_trigger",
        "ap",
        "corrected",
        "harmed",
        *TRIGGER_SUM_COLUMNS,
    ]
    grouped = (
        frame.groupby(["condition", *KEY_COLUMNS], sort=True, as_index=False)[numeric]
        .mean()
    )
    counts = frame.groupby(["condition", *KEY_COLUMNS], sort=True)["seed"].nunique()
    if not counts.eq(len(seeds)).all():
        raise AnalysisContractError("some condition/sample cells do not cover every seed")
    grouped["n_seeds"] = len(seeds)
    return add_derived_metrics(grouped)


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    counts = {column: float(frame[column].sum()) for column in COUNT_COLUMNS}
    errors = counts["fp"] + counts["fn"]
    fixed_positive = float(frame["fixed_fpr_tp"].sum() + frame["fixed_fpr_fn"].sum())
    fixed_negative = float(frame["fixed_fpr_fp"].sum() + frame["fixed_fpr_tn"].sum())
    valid = float(frame["valid_pixel_count"].sum())
    target_positive = float(frame["target_positive_count"].sum())
    ap_values = frame["ap"].to_numpy(dtype=float)
    return {
        "n_samples": int(len(frame)),
        "n_sources": int(frame["source"].nunique()),
        "n_events": int(frame[["source", "canonical_event_id"]].drop_duplicates().shape[0]),
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "tn": counts["tn"],
        "iou": _safe_divide(counts["tp"], counts["tp"] + counts["fp"] + counts["fn"]),
        "ap": float(np.nanmean(ap_values)) if np.isfinite(ap_values).any() else float("nan"),
        "errors": errors,
        "corrected": float(frame["corrected"].sum()),
        "harmed": float(frame["harmed"].sum()),
        "brier": _safe_divide(float(frame["brier_sum"].sum()), valid),
        "nll": _safe_divide(float(frame["nll_sum"].sum()), valid),
        "area_error": _safe_divide(float(frame["soft_area_error"].abs().sum()), valid),
        "signed_area_error": _safe_divide(float(frame["soft_area_error"].sum()), valid),
        "fixed_fpr_recall": _safe_divide(float(frame["fixed_fpr_tp"].sum()), fixed_positive),
        "fixed_fpr": _safe_divide(float(frame["fixed_fpr_fp"].sum()), fixed_negative),
        "target_prevalence": _safe_divide(target_positive, valid),
    }


def method_metrics(seed_mean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        current = seed_mean.loc[seed_mean["condition"] == condition]
        metrics = aggregate_metrics(current)
        reference = seed_mean.loc[
            seed_mean["condition"] == REFERENCE_CONDITION[condition]
        ]
        reference_metrics = aggregate_metrics(reference)
        metrics.update(
            {
                "condition": condition,
                "reference_condition": REFERENCE_CONDITION[condition],
                "delta_iou": metrics["iou"] - reference_metrics["iou"],
                "delta_ap": metrics["ap"] - reference_metrics["ap"],
                "rer": _safe_divide(
                    reference_metrics["errors"] - metrics["errors"],
                    reference_metrics["errors"],
                ),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def event_metrics(seed_mean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, source, event), current in seed_mean.groupby(
        ["condition", "source", "canonical_event_id"], sort=True
    ):
        rows.append(
            {
                "condition": condition,
                "source": source,
                "canonical_event_id": event,
                **aggregate_metrics(current),
            }
        )
    return pd.DataFrame(rows)


def _stratum_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    if name == "all":
        return pd.Series(True, index=frame.index)
    if name == "material_q_positive":
        return frame["q_material"] > 0
    if name == "trigger_q_positive":
        return frame["q_trigger"] > 0
    raise AssertionError(name)


def sign_flip_p(values: np.ndarray, *, iterations: int, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = iterations
    while remaining:
        batch = min(remaining, 4096)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(values)))
        means = np.abs((signs * values[None, :]).mean(axis=1))
        exceed += int(np.count_nonzero(means >= observed - EPS))
        remaining -= batch
    return float((exceed + 1) / (iterations + 1))


def hierarchical_bootstrap_metrics(
    paired: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> dict[str, list[float]]:
    """Compute all hierarchical CIs in one shared resampling pass."""

    rng = np.random.default_rng(seed)
    clusters: dict[str, list[np.ndarray]] = {}
    for source, source_frame in paired.groupby("source", sort=False):
        clusters[str(source)] = [
            event_frame.index.to_numpy(dtype=np.int64)
            for _, event_frame in source_frame.groupby("canonical_event_id", sort=False)
        ]
    # Resetting the index makes cluster indices direct NumPy row indices.
    working = paired.reset_index(drop=True)
    clusters = {}
    for source, source_frame in working.groupby("source", sort=False):
        clusters[str(source)] = [
            event_frame.index.to_numpy(dtype=np.int64)
            for _, event_frame in source_frame.groupby("canonical_event_id", sort=False)
        ]
    sources = np.asarray(list(clusters), dtype=object)
    arrays = {
        column: working[column].to_numpy(dtype=np.float64)
        for column in (
            "tp_left", "fp_left", "fn_left", "tp_right", "fp_right", "fn_right",
            "delta_iou", "delta_ap", "brier_sum_left", "brier_sum_right",
            "nll_sum_left", "nll_sum_right", "soft_area_error_left",
            "soft_area_error_right", "valid_pixel_count_left", "valid_pixel_count_right",
            "fixed_fpr_tp_left", "fixed_fpr_fn_left", "fixed_fpr_tp_right",
            "fixed_fpr_fn_right",
        )
    }
    metric_names = (
        "pooled_delta_iou",
        "mean_sample_delta_iou",
        "delta_ap",
        "rer",
        "delta_brier",
        "delta_nll",
        "delta_area_error",
        "delta_fixed_fpr_recall",
    )
    draws = {name: np.empty(n_bootstrap, dtype=np.float64) for name in metric_names}
    for iteration in range(n_bootstrap):
        sampled_indices: list[np.ndarray] = []
        for source in rng.choice(sources, size=len(sources), replace=True):
            events = clusters[str(source)]
            for event_index in rng.integers(0, len(events), size=len(events)):
                event_rows = events[int(event_index)]
                sampled_indices.append(
                    event_rows[rng.integers(0, len(event_rows), size=len(event_rows))]
                )
        index = np.concatenate(sampled_indices)
        tp_left = arrays["tp_left"][index].sum()
        tp_right = arrays["tp_right"][index].sum()
        left_iou_denom = (
            tp_left + arrays["fp_left"][index].sum() + arrays["fn_left"][index].sum()
        )
        right_iou_denom = (
            tp_right + arrays["fp_right"][index].sum() + arrays["fn_right"][index].sum()
        )
        draws["pooled_delta_iou"][iteration] = (
            _safe_divide(tp_left, left_iou_denom)
            - _safe_divide(tp_right, right_iou_denom)
        )
        draws["mean_sample_delta_iou"][iteration] = np.nanmean(arrays["delta_iou"][index])
        delta_ap = arrays["delta_ap"][index]
        draws["delta_ap"][iteration] = (
            np.nanmean(delta_ap) if np.isfinite(delta_ap).any() else np.nan
        )
        left_errors = arrays["fp_left"][index].sum() + arrays["fn_left"][index].sum()
        right_errors = arrays["fp_right"][index].sum() + arrays["fn_right"][index].sum()
        draws["rer"][iteration] = _safe_divide(right_errors - left_errors, right_errors)
        valid_left = arrays["valid_pixel_count_left"][index].sum()
        valid_right = arrays["valid_pixel_count_right"][index].sum()
        draws["delta_brier"][iteration] = (
            _safe_divide(arrays["brier_sum_left"][index].sum(), valid_left)
            - _safe_divide(arrays["brier_sum_right"][index].sum(), valid_right)
        )
        draws["delta_nll"][iteration] = (
            _safe_divide(arrays["nll_sum_left"][index].sum(), valid_left)
            - _safe_divide(arrays["nll_sum_right"][index].sum(), valid_right)
        )
        draws["delta_area_error"][iteration] = (
            _safe_divide(np.abs(arrays["soft_area_error_left"][index]).sum(), valid_left)
            - _safe_divide(np.abs(arrays["soft_area_error_right"][index]).sum(), valid_right)
        )
        left_positive = (
            arrays["fixed_fpr_tp_left"][index].sum()
            + arrays["fixed_fpr_fn_left"][index].sum()
        )
        right_positive = (
            arrays["fixed_fpr_tp_right"][index].sum()
            + arrays["fixed_fpr_fn_right"][index].sum()
        )
        draws["delta_fixed_fpr_recall"][iteration] = (
            _safe_divide(arrays["fixed_fpr_tp_left"][index].sum(), left_positive)
            - _safe_divide(arrays["fixed_fpr_tp_right"][index].sum(), right_positive)
        )
    intervals: dict[str, list[float]] = {}
    for name, values in draws.items():
        finite = values[np.isfinite(values)]
        intervals[name] = (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if finite.size
            else [float("nan"), float("nan")]
        )
    return intervals


def cohens_dz(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    standard_deviation = float(values.std(ddof=1))
    return (
        float(values.mean() / standard_deviation)
        if standard_deviation > EPS
        else float("nan")
    )


def paired_contrast(
    seed_mean: pd.DataFrame,
    name: str,
    left: str,
    right: str,
    stratum: str,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    permutation_iterations: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    columns = [
        *KEY_COLUMNS,
        "q_material",
        "q_trigger",
        *COUNT_COLUMNS,
        "iou",
        "ap",
        "errors",
        "corrected",
        "harmed",
        *TRIGGER_SUM_COLUMNS,
        "brier",
        "nll",
        "area_error",
        "fixed_fpr_recall",
        "fixed_fpr",
    ]
    left_frame = seed_mean.loc[seed_mean["condition"] == left, columns]
    right_frame = seed_mean.loc[seed_mean["condition"] == right, columns]
    paired = left_frame.merge(
        right_frame,
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise AnalysisContractError(f"sample pairing failed for contrast {name}")
    paired = paired.drop(columns="_merge")
    for support in ("q_material", "q_trigger"):
        if not np.allclose(
            paired[f"{support}_left"], paired[f"{support}_right"], rtol=0, atol=EPS
        ):
            raise AnalysisContractError(f"support values differ within contrast {name}")
        paired[support] = paired[f"{support}_left"]
    paired = paired.loc[_stratum_mask(paired, stratum)].copy()
    if paired.empty:
        raise AnalysisContractError(f"predefined stratum {stratum} is empty for {name}")
    for metric in (
        "iou",
        "ap",
        "brier",
        "nll",
        "area_error",
        "fixed_fpr_recall",
        "fixed_fpr",
        "corrected",
        "harmed",
    ):
        paired[f"delta_{metric}"] = paired[f"{metric}_left"] - paired[f"{metric}_right"]
    paired.insert(0, "contrast", name)
    paired.insert(1, "left", left)
    paired.insert(2, "right", right)
    paired.insert(3, "stratum", stratum)

    event_rows: list[dict[str, Any]] = []
    for (source, event), current in paired.groupby(
        ["source", "canonical_event_id"], sort=True
    ):
        left_metrics = aggregate_metrics(
            _paired_side_as_frame(current, "left")
        )
        right_metrics = aggregate_metrics(
            _paired_side_as_frame(current, "right")
        )
        event_rows.append(
            {
                "contrast": name,
                "left": left,
                "right": right,
                "stratum": stratum,
                "source": source,
                "canonical_event_id": event,
                "n_samples": len(current),
                "delta_iou": left_metrics["iou"] - right_metrics["iou"],
                "delta_ap": left_metrics["ap"] - right_metrics["ap"],
                "rer": _safe_divide(
                    right_metrics["errors"] - left_metrics["errors"],
                    right_metrics["errors"],
                ),
                "delta_corrected": left_metrics["corrected"] - right_metrics["corrected"],
                "delta_harmed": left_metrics["harmed"] - right_metrics["harmed"],
                "delta_brier": left_metrics["brier"] - right_metrics["brier"],
                "delta_nll": left_metrics["nll"] - right_metrics["nll"],
                "delta_area_error": left_metrics["area_error"] - right_metrics["area_error"],
                "delta_fixed_fpr_recall": (
                    left_metrics["fixed_fpr_recall"] - right_metrics["fixed_fpr_recall"]
                ),
            }
        )
    events = pd.DataFrame(event_rows)

    left_metrics = aggregate_metrics(_paired_side_as_frame(paired, "left"))
    right_metrics = aggregate_metrics(_paired_side_as_frame(paired, "right"))
    summary: dict[str, Any] = {
        "contrast": name,
        "left": left,
        "right": right,
        "stratum": stratum,
        "n_samples": len(paired),
        "n_sources": paired["source"].nunique(),
        "n_events": events.shape[0],
        "pooled_delta_iou": left_metrics["iou"] - right_metrics["iou"],
        "mean_sample_delta_iou": float(np.nanmean(paired["delta_iou"])),
        "median_sample_delta_iou": float(np.nanmedian(paired["delta_iou"])),
        "positive_sample_rate_iou": float(np.nanmean(paired["delta_iou"] > 0)),
        "sample_iou_sign_flip_p": sign_flip_p(
            paired["delta_iou"].to_numpy(),
            iterations=permutation_iterations,
            seed=bootstrap_seed + 31,
        ),
        "sample_iou_cohens_dz": cohens_dz(paired["delta_iou"].to_numpy()),
        "mean_event_delta_iou": float(np.nanmean(events["delta_iou"])),
        "median_event_delta_iou": float(np.nanmedian(events["delta_iou"])),
        "positive_event_rate_iou": float(np.nanmean(events["delta_iou"] > 0)),
        "event_iou_sign_flip_p": sign_flip_p(
            events["delta_iou"].to_numpy(),
            iterations=permutation_iterations,
            seed=bootstrap_seed + 37,
        ),
        "event_iou_cohens_dz": cohens_dz(events["delta_iou"].to_numpy()),
        "delta_ap": left_metrics["ap"] - right_metrics["ap"],
        "rer": _safe_divide(
            right_metrics["errors"] - left_metrics["errors"], right_metrics["errors"]
        ),
        "corrected_left": left_metrics["corrected"],
        "harmed_left": left_metrics["harmed"],
        "corrected_right": right_metrics["corrected"],
        "harmed_right": right_metrics["harmed"],
        "delta_brier": left_metrics["brier"] - right_metrics["brier"],
        "delta_nll": left_metrics["nll"] - right_metrics["nll"],
        "delta_area_error": left_metrics["area_error"] - right_metrics["area_error"],
        "delta_fixed_fpr_recall": (
            left_metrics["fixed_fpr_recall"] - right_metrics["fixed_fpr_recall"]
        ),
        "delta_fixed_fpr": left_metrics["fixed_fpr"] - right_metrics["fixed_fpr"],
    }
    intervals = hierarchical_bootstrap_metrics(
        paired, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    for output_name, interval in intervals.items():
        summary[f"{output_name}_hierarchical_ci95"] = interval
    return summary, paired, events


def _paired_side_as_frame(paired: pd.DataFrame, side: str) -> pd.DataFrame:
    output = paired.loc[:, list(KEY_COLUMNS)].copy()
    for column in (
        *COUNT_COLUMNS,
        "ap",
        "corrected",
        "harmed",
        *TRIGGER_SUM_COLUMNS,
    ):
        # Trigger sum fields are not retained by paired_contrast because the
        # derived calibration metrics are sufficient for paired inference.
        candidate = f"{column}_{side}"
        if candidate in paired:
            output[column] = paired[candidate]
    # Defensive fallback for externally constructed paired frames.  Normal
    # analysis retains the additive receipts and never takes this branch.
    if "valid_pixel_count" not in output:
        output["valid_pixel_count"] = 1.0
        output["target_positive_count"] = 1.0
        output["brier_sum"] = paired[f"brier_{side}"]
        output["nll_sum"] = paired[f"nll_{side}"]
        output["soft_area_error"] = paired[f"area_error_{side}"]
        recall = paired[f"fixed_fpr_recall_{side}"].fillna(0.0)
        fpr = paired[f"fixed_fpr_{side}"].fillna(0.0)
        output["fixed_fpr_tp"] = recall
        output["fixed_fpr_fn"] = 1.0 - recall
        output["fixed_fpr_fp"] = fpr
        output["fixed_fpr_tn"] = 1.0 - fpr
    return output


def analyze(
    runs_root: Path,
    outdir: Path,
    *,
    seeds: tuple[int, ...] | None,
    min_seeds: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    permutation_iterations: int,
) -> dict[str, Any]:
    if n_bootstrap < 100:
        raise AnalysisContractError("n_bootstrap must be >= 100")
    if permutation_iterations < 100:
        raise AnalysisContractError("permutation_iterations must be >= 100")
    resolved_seeds = resolve_seeds(runs_root, seeds, min_seeds)
    raw, inventory = load_all_runs(runs_root, resolved_seeds)
    seed_mean = average_seeds(raw, resolved_seeds)
    methods = method_metrics(seed_mean)
    events = event_metrics(seed_mean)

    contrast_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for index, (name, (left, right, stratum)) in enumerate(CONTRASTS.items()):
        summary, paired_samples, paired_events = paired_contrast(
            seed_mean,
            name,
            left,
            right,
            stratum,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed + index * 101,
            permutation_iterations=permutation_iterations,
        )
        contrast_rows.append(summary)
        sample_frames.append(paired_samples)
        event_frames.append(paired_events)
    contrasts = pd.DataFrame(contrast_rows)
    paired_samples = pd.concat(sample_frames, ignore_index=True)
    paired_events = pd.concat(event_frames, ignore_index=True)

    strata_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        current = seed_mean.loc[seed_mean["condition"] == condition]
        for stratum in ("all", "material_q_positive", "trigger_q_positive"):
            selected = current.loc[_stratum_mask(current, stratum)]
            if selected.empty:
                continue
            strata_rows.append(
                {"condition": condition, "stratum": stratum, **aggregate_metrics(selected)}
            )
    strata = pd.DataFrame(strata_rows)

    required_aligned_controls = {
        "T": [name for name in CONTRASTS if name.startswith("T_aligned_minus_")],
        "M": [name for name in CONTRASTS if name.startswith("M_aligned_minus_")],
        "R": [name for name in CONTRASTS if name.startswith("R_aligned_minus_")],
    }
    observed_contrasts = set(contrasts["contrast"])
    for role, required in required_aligned_controls.items():
        if not set(required).issubset(observed_contrasts):
            raise AnalysisContractError(f"aligned-versus-negative-control contrasts missing for {role}")

    summary = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "seeds": list(resolved_seeds),
        "n_seeds": len(resolved_seeds),
        "n_parent_variants": len(MAIN_CONDITIONS),
        "n_analysis_conditions": len(CONDITIONS),
        "n_runs": len(MAIN_CONDITIONS) * len(resolved_seeds),
        "n_samples": int(seed_mean.loc[seed_mean["condition"] == "V"].shape[0]),
        "n_sources": int(seed_mean["source"].nunique()),
        "n_canonical_events": int(
            seed_mean[["source", "canonical_event_id"]].drop_duplicates().shape[0]
        ),
        "conditions": list(CONDITIONS),
        "feature_contract": {
            "terrain_channel_order": list(TERRAIN9_CHANNEL_ORDER),
            "material_interaction_groups": {
                name: list(indices) for name, indices in MATERIAL_INTERACTION_GROUPS.items()
            },
            "index_base": 0,
        },
        "contrasts": [json_safe(row) for row in contrast_rows],
        "aligned_control_requirements": required_aligned_controls,
        "aggregation_contract": {
            "seed_role": "optimization_repeat",
            "seed_reduction": "per-sample arithmetic mean before inference",
            "scientific_unit": "canonical_event_nested_within_source",
            "bootstrap": "source_then_event_then_sample_with_replacement",
            "ap": "macro sample AP; missing AP is excluded",
            "lower_is_better": ["brier", "nll", "area_error", "fixed_fpr"],
        },
        "strict_checks": [
            "complete_five_parent_variant_seed_inventory",
            "strict_json_and_artifact_hashes",
            "per_sample_csv_DONE_artifact_hash",
            "same_checkpoint_negative_control_provenance",
            "negative_controls_not_independent_runs",
            "terrain9_order_and_material_interaction_groups",
            "run_csv_identity",
            "exact_paired_sample_keys",
            "support_values_equal_across_runs",
            "all_seed_cells_present",
            "corrected_harmed_error_identity",
            "aligned_versus_negative_controls_present",
        ],
        "input_inventory": inventory,
    }

    report_lines = [
        "# Unified PILD/Sen12 role-aware analysis", "",
        f"Seeds: `{', '.join(map(str, resolved_seeds))}`; trained parent variants: "
        f"`{len(MAIN_CONDITIONS)}`; analysis conditions: `{len(CONDITIONS)}`; "
        f"paired samples: `{summary['n_samples']}`; canonical events: "
        f"`{summary['n_canonical_events']}`.", "",
        "Seeds are optimization repeats. Inference uses paired samples and canonical events, "
        "with a hierarchical source-event-sample bootstrap.", "",
        "## Method metrics", "", methods.to_markdown(index=False), "",
        "## Predefined paired contrasts", "", contrasts.to_markdown(index=False), "",
        "## Interpretation guardrails", "",
        "- Exactly five trained checkpoints are used per seed; every negative control is an inference-time row from its declared parent checkpoint.",
        "- Positive aligned effects are not assumed; every aligned physical condition is compared with its frozen negative controls.",
        "- Material and Trigger headline contrasts use only their preregistered q-positive support strata.",
        "- Trigger Brier, NLL, area error, and fixed-FPR recall are reconstructed from additive receipts.",
        "- Average IoU is reported as a completeness measure alongside corrected-versus-harmed error flow.",
        "",
    ]

    atomic_write_csv(outdir / "method_metrics.csv", methods)
    atomic_write_csv(outdir / "contrast_summary.csv", contrasts)
    atomic_write_csv(outdir / "paired_sample_metrics.csv", paired_samples)
    atomic_write_csv(outdir / "paired_event_metrics.csv", paired_events)
    atomic_write_csv(outdir / "per_event_metrics.csv", events)
    atomic_write_csv(outdir / "support_strata_metrics.csv", strata)
    atomic_write_json(outdir / "summary.json", summary)
    atomic_write_text(outdir / "report.md", "\n".join(report_lines))
    return json_safe(summary)


def parse_seed_argument(value: str) -> tuple[int, ...] | None:
    if not value.strip():
        return None
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds cannot be empty after parsing")
    return seeds


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=root / "experiments/revision2026/pild_sen12_roleaware_v1",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        type=parse_seed_argument,
        default=None,
        help="comma-separated frozen seeds; default requires identical discovered seed sets",
    )
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--permutations", type=int, default=20000)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_seeds < 1:
        raise SystemExit("[FATAL] --min-seeds must be >= 1")
    outdir = args.outdir or args.runs_root / "analysis"
    try:
        summary = analyze(
            args.runs_root,
            outdir,
            seeds=args.seeds,
            min_seeds=args.min_seeds,
            n_bootstrap=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            permutation_iterations=args.permutations,
        )
    except AnalysisContractError as error:
        raise SystemExit(f"[FATAL] {error}") from error
    print(
        json.dumps(
            {
                "status": summary["status"],
                "n_runs": summary["n_runs"],
                "outdir": str(outdir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
