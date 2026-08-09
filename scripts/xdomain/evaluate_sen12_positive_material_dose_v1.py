#!/usr/bin/env python3
"""Nested-OOF evaluation of a positive Material dose on frozen Terrain deltas.

For every target protocol, two inner OOF bundles fit a regularized mapping from
label-free Material factors to the NLL-optimal positive Terrain multiplier.
The third inner bundle is evaluated without using its labels for fitting. This
rotates three times. No target outer-test artifact is opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

import train_sen12_proposal_utility_gate_v3 as gate_protocol
from material_factors_v3 import FACTOR_GROUPS


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "experiments/revision2026/sen12_positive_dose_v1/formal_inputs_v1"
DEFAULT_REGISTRY = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v3/material_factor_registry_v3.csv"
DEFAULT_SCHEMA = PROJECT_ROOT / "processed/hybrid_pinn/sen12_context_v3/material_factor_schema_v3.json"
DEFAULT_OUT = PROJECT_ROOT / "experiments/revision2026/sen12_positive_material_dose_v1"
FEATURE_SETS = {
    "AWC": FACTOR_GROUPS["awc_core"],
    "SOIL": FACTOR_GROUPS["soil_hydraulic"],
    "LITH": FACTOR_GROUPS["lithology_composition"],
    "AWC_SOIL": FACTOR_GROUPS["awc_core"] + FACTOR_GROUPS["soil_hydraulic"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--material-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--material-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--min-multiplier", type=float, default=0.5)
    parser.add_argument("--max-multiplier", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--sample-batch", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def transform(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(values), values, center)
    return np.clip((clean - center) / scale, -5.0, 5.0)


def event_weights(events: Sequence[str]) -> np.ndarray:
    counts = pd.Series(list(events)).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[event] for event in events], dtype=np.float64)
    return weights / weights.mean()


def oracle_log_multiplier(
    bundle: gate_protocol.FoldBundle,
    grid: np.ndarray,
    *,
    device: str,
    sample_batch: int,
) -> np.ndarray:
    """NLL-optimal multiplier per sample, used only as a training target."""

    output = np.zeros(len(bundle.sample_ids), dtype=np.float32)
    grid_tensor = torch.as_tensor(grid, dtype=torch.float32, device=device)[None, :, None]
    for start in range(0, len(bundle.sample_ids), sample_batch):
        stop = min(start + sample_batch, len(bundle.sample_ids))
        visual = torch.as_tensor(
            bundle.visual_logits[start:stop], dtype=torch.float32, device=device
        ).flatten(1)
        delta = torch.as_tensor(
            bundle.terrain_delta[start:stop], dtype=torch.float32, device=device
        ).flatten(1)
        target = torch.as_tensor(
            bundle.mask[start:stop], dtype=torch.float32, device=device
        ).flatten(1)
        valid = torch.as_tensor(
            bundle.valid[start:stop], dtype=torch.float32, device=device
        ).flatten(1)
        logits = visual[:, None, :] + delta[:, None, :] * grid_tensor
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target[:, None, :].expand_as(logits), reduction="none"
        )
        loss = (loss * valid[:, None, :]).sum(dim=2) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        selected = torch.argmin(loss, dim=1).detach().cpu().numpy()
        output[start:stop] = np.log(grid[selected]).astype(np.float32)
    return output


def deterministic_donor_indices(
    sample_ids: Sequence[str],
    events: Sequence[str],
    regions: Sequence[str],
    donor_events: Sequence[str],
    donor_regions: Sequence[str],
    seed: int,
) -> np.ndarray:
    donors = np.zeros(len(sample_ids), dtype=np.int64)
    donor_events = np.asarray(donor_events, dtype=object)
    donor_regions = np.asarray(donor_regions, dtype=object)
    for index, sample_id in enumerate(sample_ids):
        candidates = np.flatnonzero(
            (donor_events != events[index]) & (donor_regions != regions[index])
        )
        if not len(candidates):
            candidates = np.flatnonzero(donor_events != events[index])
        if not len(candidates):
            raise RuntimeError("no mismatched Material donor is available")
        digest = hashlib.sha256(f"{seed}|M-dose|{sample_id}".encode()).digest()
        donors[index] = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
    return donors


def confusion(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    prediction = np.asarray(prediction, bool)
    target = np.asarray(target, bool)
    valid = np.asarray(valid, bool)
    return {
        "tp": int(np.sum(prediction & target & valid)),
        "fp": int(np.sum(prediction & ~target & valid)),
        "fn": int(np.sum(~prediction & target & valid)),
        "tn": int(np.sum(~prediction & ~target & valid)),
    }


def metrics(counts: Mapping[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = [int(counts[key]) for key in ("tp", "fp", "fn", "tn")]
    return {
        **counts,
        "errors": fp + fn,
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def evaluate_bundle(
    bundle: gate_protocol.FoldBundle,
    multiplier: np.ndarray,
    condition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visual = bundle.visual_logits.astype(np.float32)
    delta = bundle.terrain_delta.astype(np.float32)
    target = bundle.mask.astype(bool)
    valid = bundle.valid.astype(bool)
    threshold = bundle.threshold_logit
    logits = visual + delta * multiplier[:, None, None, None].astype(np.float32)
    prediction = logits >= threshold
    terrain_prediction = (visual + delta) >= threshold
    visual_prediction = visual >= threshold
    total = metrics(confusion(prediction, target, valid))
    parent = metrics(confusion(terrain_prediction, target, valid))
    visual_metric = metrics(confusion(visual_prediction, target, valid))
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[valid], -30.0, 30.0)))
    labels = target[valid]
    summary = {
        **total,
        "condition": condition,
        "ap": float(average_precision_score(labels, probability)) if labels.any() else 0.0,
        "vt_iou": parent["iou"],
        "visual_iou": visual_metric["iou"],
        "rer_vs_vt": (parent["errors"] - total["errors"]) / max(parent["errors"], 1),
        "mean_multiplier": float(np.mean(multiplier)),
        "std_multiplier": float(np.std(multiplier)),
    }
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(bundle.sample_ids):
        item = metrics(confusion(prediction[index], target[index], valid[index]))
        item_parent = metrics(confusion(terrain_prediction[index], target[index], valid[index]))
        rows.append({
            "condition": condition,
            "sample_id": sample_id,
            "event_id": bundle.event_ids[index],
            "region_id": bundle.source_ids[index],
            "iou": item["iou"],
            "errors": item["errors"],
            "vt_iou": item_parent["iou"],
            "vt_errors": item_parent["errors"],
            "multiplier": float(multiplier[index]),
        })
    return summary, rows


def main() -> int:
    args = parse_args()
    if not (0 < args.min_multiplier < 1 < args.max_multiplier):
        raise ValueError("multiplier interval must contain one")
    args.outdir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.material_registry, low_memory=False)
    schema = json.loads(args.material_schema.read_text(encoding="utf-8"))
    registry = registry.assign(sample_id=registry.sample_id.astype(str)).set_index("sample_id")
    grid = np.exp(
        np.linspace(np.log(args.min_multiplier), np.log(args.max_multiplier), args.grid_size)
    ).astype(np.float32)
    if not np.any(np.isclose(grid, 1.0)):
        grid = np.sort(np.unique(np.append(grid, 1.0))).astype(np.float32)
    all_target_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    access_log: list[dict[str, Any]] = []

    for target_fold in range(5):
        print(f"[positive-M] target_outer={target_fold}", flush=True)
        manifest = args.formal_input_root / f"target_outer{target_fold}/oof_manifest.json"
        split_csv = args.formal_input_root / f"target_outer{target_fold}/gate_split.csv"
        bundles, _receipt = gate_protocol.load_formal_nested_bundles(
            manifest,
            target_fold=target_fold,
            split_csv=split_csv,
            seed=args.seed,
            access_log=access_log,
        )
        oracle = {
            bundle.fold: oracle_log_multiplier(
                bundle,
                grid,
                device=args.device,
                sample_batch=args.sample_batch,
            )
            for bundle in bundles
        }
        target_dir = args.outdir / f"target_outer{target_fold}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for feature_set, feature_names in FEATURE_SETS.items():
            if not set(feature_names) <= set(schema["model_eligible_features"]):
                raise RuntimeError(f"factor schema lacks {feature_set} features")
            for eval_index, eval_bundle in enumerate(bundles):
                fit_bundles = [bundle for index, bundle in enumerate(bundles) if index != eval_index]
                fit_ids = [sample_id for bundle in fit_bundles for sample_id in bundle.sample_ids]
                fit_events = [event for bundle in fit_bundles for event in bundle.event_ids]
                fit_regions = [region for bundle in fit_bundles for region in bundle.source_ids]
                fit_x_raw = registry.loc[fit_ids, list(feature_names)].to_numpy(dtype=np.float64)
                fit_q = pd.to_numeric(registry.loc[fit_ids, "q_M"], errors="coerce").fillna(0.0).to_numpy()
                fit_y = np.concatenate([oracle[bundle.fold] for bundle in fit_bundles]).astype(np.float64)
                center, scale = fit_scaler(fit_x_raw)
                fit_x = transform(fit_x_raw, center, scale)
                eligible = np.isfinite(fit_x).all(axis=1) & (fit_q > 0)
                weights = event_weights(np.asarray(fit_events, object)[eligible])
                model = Ridge(alpha=args.ridge_alpha)
                model.fit(
                    fit_x[eligible],
                    fit_y[eligible],
                    sample_weight=weights,
                )
                global_log_multiplier = float(
                    np.average(fit_y[eligible], weights=weights)
                )

                eval_ids = list(eval_bundle.sample_ids)
                eval_raw = registry.loc[eval_ids, list(feature_names)].to_numpy(dtype=np.float64)
                eval_q = pd.to_numeric(registry.loc[eval_ids, "q_M"], errors="coerce").fillna(0.0).to_numpy()
                predicted_log = np.clip(
                    model.predict(transform(eval_raw, center, scale)),
                    np.log(args.min_multiplier),
                    np.log(args.max_multiplier),
                )
                aligned_multiplier = np.where(eval_q > 0, np.exp(predicted_log), 1.0)

                donors = deterministic_donor_indices(
                    eval_ids,
                    eval_bundle.event_ids,
                    eval_bundle.source_ids,
                    fit_events,
                    fit_regions,
                    args.seed + target_fold * 101 + eval_index,
                )
                donor_raw = fit_x_raw[donors]
                donor_q = fit_q[donors]
                donor_log = np.clip(
                    model.predict(transform(donor_raw, center, scale)),
                    np.log(args.min_multiplier),
                    np.log(args.max_multiplier),
                )
                shuffled_multiplier = np.where(
                    (eval_q > 0) & (donor_q > 0), np.exp(donor_log), 1.0
                )
                conditions = {
                    "VT_identity": np.ones(len(eval_ids), dtype=np.float64),
                    "VTM_global_dose": np.where(
                        eval_q > 0, np.exp(global_log_multiplier), 1.0
                    ),
                    "VTM_aligned": aligned_multiplier,
                    "VTM_material_shuffle": shuffled_multiplier,
                }
                for condition, multiplier in conditions.items():
                    summary, sample_rows = evaluate_bundle(
                        eval_bundle, multiplier, condition
                    )
                    all_target_rows.append({
                        "target_outer_fold": target_fold,
                        "meta_eval_bundle": int(eval_bundle.fold),
                        "feature_set": feature_set,
                        **summary,
                    })
                    all_sample_rows.extend({
                        "target_outer_fold": target_fold,
                        "meta_eval_bundle": int(eval_bundle.fold),
                        "feature_set": feature_set,
                        **row,
                    } for row in sample_rows)

    target_frame = pd.DataFrame(all_target_rows)
    sample_frame = pd.DataFrame(all_sample_rows)
    target_frame.to_csv(args.outdir / "meta_fold_metrics.csv", index=False)
    sample_frame.to_csv(args.outdir / "meta_sample_metrics.csv", index=False)

    event = (
        sample_frame.groupby(
            ["target_outer_fold", "feature_set", "condition", "event_id"], as_index=False
        )[["errors", "vt_errors"]].sum()
    )
    event["rer_vs_vt"] = (event.vt_errors - event.errors) / event.vt_errors.clip(lower=1)
    event.to_csv(args.outdir / "meta_event_metrics.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        aligned = event[(event.feature_set == feature_set) & (event.condition == "VTM_aligned")]
        shuffled = event[(event.feature_set == feature_set) & (event.condition == "VTM_material_shuffle")]
        global_dose = event[(event.feature_set == feature_set) & (event.condition == "VTM_global_dose")]
        paired = aligned.merge(
            shuffled[["target_outer_fold", "event_id", "errors"]],
            on=["target_outer_fold", "event_id"],
            suffixes=("_aligned", "_shuffle"),
            validate="one_to_one",
        )
        paired = paired.merge(
            global_dose[["target_outer_fold", "event_id", "errors"]].rename(
                columns={"errors": "errors_global"}
            ),
            on=["target_outer_fold", "event_id"],
            validate="one_to_one",
        )
        per_target = paired.groupby("target_outer_fold").agg(
            mean_rer=("rer_vs_vt", "mean"),
            aligned_errors=("errors_aligned", "sum"),
            shuffle_errors=("errors_shuffle", "sum"),
            global_errors=("errors_global", "sum"),
        )
        per_target["aligned_beats_shuffle"] = per_target.aligned_errors < per_target.shuffle_errors
        per_target["aligned_beats_global"] = per_target.aligned_errors < per_target.global_errors
        per_target["pass"] = (
            (per_target.mean_rer > 0)
            & per_target.aligned_beats_shuffle
            & per_target.aligned_beats_global
        )
        summary_rows.append({
            "feature_set": feature_set,
            "n_target_protocols": int(len(per_target)),
            "target_protocols_passed": int(per_target["pass"].sum()),
            "mean_event_rer_vs_vt": float(aligned.rer_vs_vt.mean()),
            "positive_events_vs_vt": int((aligned.rer_vs_vt > 0).sum()),
            "n_event_protocol_rows": int(len(aligned)),
            "aligned_minus_shuffle_errors": int(paired.errors_aligned.sum() - paired.errors_shuffle.sum()),
            "aligned_minus_global_errors": int(paired.errors_aligned.sum() - paired.errors_global.sum()),
            "development_gate_pass": bool(per_target["pass"].sum() >= 3),
        })
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(args.outdir / "summary_table.csv", index=False)
    summary = {
        "status": "complete",
        "scope": "nested-OOF development evidence; target outer-test never opened",
        "formula": "Delta_TM = Delta_T * exp(clamp(f_M(M))); q_M=0 gives exact multiplier one",
        "ridge_alpha": args.ridge_alpha,
        "multiplier_bounds": [args.min_multiplier, args.max_multiplier],
        "material_registry_sha256": sha256(args.material_registry),
        "material_schema_sha256": sha256(args.material_schema),
        "outer_test_labels_loaded": False,
        "access_log": access_log,
        "results": summary_rows,
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Sen12 positive Material dose v1",
        "",
        "Nested-OOF development only; no target outer-test artifact was opened.",
        "",
        "| Factor set | Target protocols passed | Mean event RER vs VT | Positive event rows | Aligned - shuffle errors | Aligned - global errors | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['feature_set']} | {row['target_protocols_passed']}/{row['n_target_protocols']} | "
            f"{row['mean_event_rer_vs_vt']:+.4%} | {row['positive_events_vs_vt']}/{row['n_event_protocol_rows']} | "
            f"{row['aligned_minus_shuffle_errors']:+d} | {row['aligned_minus_global_errors']:+d} | "
            f"{'PASS' if row['development_gate_pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Material is promotable only when at least three target protocols improve over VT and aligned Material beats both mismatched Material and a Material-free global dose. A failed family remains exact VT identity.",
        "",
    ]
    (args.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
