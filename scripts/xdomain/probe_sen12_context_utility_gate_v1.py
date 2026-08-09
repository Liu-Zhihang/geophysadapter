#!/usr/bin/env python3
"""Probe whether Material/Trigger predict when a Terrain correction is useful.

This is a validation-first diagnostic, not the main trained model.  The frozen
visual and Terrain models produce two candidate predictions for each sample.
Only outer-train labels are used to learn sample-level correction utility.
The inference features are label-free Terrain summaries, visual uncertainty,
Material context, and Trigger context.  ``validate`` never opens a test cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONTEXTS = ("T", "TM", "TR", "TMR")
MODEL_SPECS = (
    "ridge_a1",
    "ridge_a10",
    "ridge_a100",
    "hist_l7_r0.05",
    "hist_l15_r0.05",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SampleTable:
    sample_ids: list[str]
    event_ids: list[str]
    base: np.ndarray
    material: np.ndarray
    material_shuffle: np.ndarray
    trigger: np.ndarray
    trigger_wrong: np.ndarray
    trigger_shuffle: np.ndarray
    q_m: np.ndarray
    q_m_shuffle: np.ndarray
    q_r: np.ndarray
    q_r_shuffle: np.ndarray
    utility: np.ndarray
    visual_counts: np.ndarray
    terrain_counts: np.ndarray


def _quantiles(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        values.mean(dim=1),
        torch.quantile(values, 0.75, dim=1),
        torch.quantile(values, 0.90, dim=1),
    )


def build_sample_table(payload: Mapping[str, Any], threshold: float, chunk: int = 128) -> SampleTable:
    sample_ids = list(payload["sample_ids"])
    event_ids = list(payload["event_ids"])
    n_samples = len(sample_ids)
    threshold_logit = float(torch.logit(torch.tensor(threshold)).item())
    base_rows: list[np.ndarray] = []
    utility_rows: list[np.ndarray] = []
    visual_count_rows: list[np.ndarray] = []
    terrain_count_rows: list[np.ndarray] = []
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        visual = payload["visual_logits"][start:stop].float()
        residual = payload["frozen_vt_correction"][start:stop].float()
        terrain = payload["terrain_common9"][start:stop].float()
        target = payload["mask"][start:stop].bool()
        valid = payload["valid"][start:stop].bool()
        visual_prediction = visual >= threshold_logit
        terrain_prediction = visual + residual >= threshold_logit
        visual_probability = torch.sigmoid(visual)
        uncertainty = 1.0 - torch.abs(2.0 * visual_probability - 1.0)
        flat_valid = valid.flatten(1)
        valid_count = flat_valid.sum(dim=1).clamp_min(1)
        uncertainty_flat = uncertainty.flatten(1)
        uncertainty_masked = torch.where(flat_valid, uncertainty_flat, torch.nan)
        uncertainty_mean = torch.nanmean(uncertainty_masked, dim=1)
        # Quantiles are computed after replacing invalid pixels with the valid
        # mean; Sen12 valid masks are normally dense, and validity fraction is
        # included explicitly as a feature.
        fill = uncertainty_mean[:, None]
        uncertainty_dense = torch.where(flat_valid, uncertainty_flat, fill)
        _, uncertainty_q75, uncertainty_q90 = _quantiles(uncertainty_dense)
        residual_flat = residual.flatten(1)
        residual_valid = torch.where(flat_valid, residual_flat, torch.zeros_like(residual_flat))
        support = (residual_valid != 0).sum(dim=1) / valid_count
        abs_mean = residual_valid.abs().sum(dim=1) / valid_count
        positive_mean = residual_valid.clamp_min(0).sum(dim=1) / valid_count
        negative_mean = (-residual_valid.clamp_max(0)).sum(dim=1) / valid_count
        visual_positive = (visual_prediction & valid).flatten(1).sum(dim=1) / valid_count
        terrain_features = []
        for channel in range(terrain.shape[1]):
            current = terrain[:, channel].flatten(1)
            current = torch.where(flat_valid, current, torch.zeros_like(current))
            mean = current.sum(dim=1) / valid_count
            variance = torch.where(flat_valid, (current - mean[:, None]).square(), 0).sum(dim=1) / valid_count
            terrain_features.extend((mean, variance.sqrt()))
        base = torch.stack(
            [
                uncertainty_mean,
                uncertainty_q75,
                uncertainty_q90,
                visual_positive,
                support,
                abs_mean,
                positive_mean,
                negative_mean,
                valid_count.float() / float(valid[0].numel()),
                *terrain_features,
            ],
            dim=1,
        )
        visual_correct = visual_prediction == target
        terrain_correct = terrain_prediction == target
        visual_errors = ((~visual_correct) & valid).flatten(1).sum(dim=1)
        terrain_errors = ((~terrain_correct) & valid).flatten(1).sum(dim=1)
        utility = (visual_errors - terrain_errors).float() / valid_count.float()

        def count_matrix(prediction: torch.Tensor) -> torch.Tensor:
            tp = (prediction & target & valid).flatten(1).sum(dim=1)
            fp = (prediction & ~target & valid).flatten(1).sum(dim=1)
            fn = (~prediction & target & valid).flatten(1).sum(dim=1)
            tn = (~prediction & ~target & valid).flatten(1).sum(dim=1)
            return torch.stack((tp, fp, fn, tn), dim=1)

        base_rows.append(base.cpu().numpy().astype(np.float32))
        utility_rows.append(utility.cpu().numpy().astype(np.float64))
        visual_count_rows.append(count_matrix(visual_prediction).cpu().numpy().astype(np.int64))
        terrain_count_rows.append(count_matrix(terrain_prediction).cpu().numpy().astype(np.int64))
    return SampleTable(
        sample_ids=sample_ids,
        event_ids=event_ids,
        base=np.concatenate(base_rows),
        material=payload["material"].numpy().astype(np.float32),
        material_shuffle=payload["material_shuffle"].numpy().astype(np.float32),
        trigger=payload["trigger"].numpy().astype(np.float32),
        trigger_wrong=payload["trigger_wrong"].numpy().astype(np.float32),
        trigger_shuffle=payload["trigger_shuffle"].numpy().astype(np.float32),
        q_m=payload["q_material"].numpy().astype(np.float32),
        q_m_shuffle=payload["q_material_shuffle"].numpy().astype(np.float32),
        q_r=payload["q_trigger"].numpy().astype(np.float32),
        q_r_shuffle=payload["q_trigger_shuffle"].numpy().astype(np.float32),
        utility=np.concatenate(utility_rows),
        visual_counts=np.concatenate(visual_count_rows),
        terrain_counts=np.concatenate(terrain_count_rows),
    )


def features(table: SampleTable, context: str, control: str = "aligned") -> np.ndarray:
    if context not in CONTEXTS:
        raise ValueError(context)
    blocks = [table.base]
    if "M" in context:
        material = table.material_shuffle if control in ("material_shuffle", "both") else table.material
        q_m = table.q_m_shuffle if control in ("material_shuffle", "both") else table.q_m
        blocks.extend((material * q_m[:, None], q_m[:, None]))
    if "R" in context:
        if control in ("trigger_wrong", "both"):
            trigger, q_r = table.trigger_wrong, table.q_r
        elif control == "trigger_shuffle":
            trigger, q_r = table.trigger_shuffle, table.q_r_shuffle
        else:
            trigger, q_r = table.trigger, table.q_r
        blocks.extend((trigger * q_r[:, None], q_r[:, None]))
    return np.concatenate(blocks, axis=1)


def event_balanced_weights(event_ids: Sequence[str]) -> np.ndarray:
    unique, counts = np.unique(np.asarray(event_ids), return_counts=True)
    count_by_event = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_event[event] for event in event_ids], dtype=np.float64)
    return weights * len(weights) / weights.sum()


def make_model(spec: str, seed: int):
    if spec.startswith("ridge_a"):
        alpha = float(spec.removeprefix("ridge_a"))
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if spec == "hist_l7_r0.05":
        return HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=40, l2_regularization=10.0, random_state=seed,
        )
    if spec == "hist_l15_r0.05":
        return HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=20.0, random_state=seed,
        )
    raise ValueError(spec)


def aggregate_gate(table: SampleTable, score: np.ndarray) -> dict[str, Any]:
    use_terrain = np.asarray(score) > 0.0
    selected = np.where(use_terrain[:, None], table.terrain_counts, table.visual_counts)
    tp, fp, fn, tn = selected.sum(axis=0).tolist()
    visual_errors = int(table.visual_counts[:, 1:3].sum())
    selected_errors = int(fp + fn)
    denominator = max(1, int(tp + fp + fn))
    return {
        "n_samples": len(table.sample_ids),
        "n_events": len(set(table.event_ids)),
        "terrain_selected_samples": int(use_terrain.sum()),
        "coverage": float(use_terrain.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "iou": float(tp / denominator),
        "visual_errors": visual_errors,
        "selected_errors": selected_errors,
        "net_corrected_vs_visual": visual_errors - selected_errors,
        "rer_vs_visual": (visual_errors - selected_errors) / max(visual_errors, 1),
        "utility_score_mean": float(np.mean(score)),
    }


def load_table(cache_path: Path, threshold: float) -> SampleTable:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    return build_sample_table(payload, threshold)


def concatenate_tables(left: SampleTable, right: SampleTable) -> SampleTable:
    return SampleTable(
        sample_ids=left.sample_ids + right.sample_ids,
        event_ids=left.event_ids + right.event_ids,
        base=np.concatenate((left.base, right.base)),
        material=np.concatenate((left.material, right.material)),
        material_shuffle=np.concatenate((left.material_shuffle, right.material_shuffle)),
        trigger=np.concatenate((left.trigger, right.trigger)),
        trigger_wrong=np.concatenate((left.trigger_wrong, right.trigger_wrong)),
        trigger_shuffle=np.concatenate((left.trigger_shuffle, right.trigger_shuffle)),
        q_m=np.concatenate((left.q_m, right.q_m)),
        q_m_shuffle=np.concatenate((left.q_m_shuffle, right.q_m_shuffle)),
        q_r=np.concatenate((left.q_r, right.q_r)),
        q_r_shuffle=np.concatenate((left.q_r_shuffle, right.q_r_shuffle)),
        utility=np.concatenate((left.utility, right.utility)),
        visual_counts=np.concatenate((left.visual_counts, right.visual_counts)),
        terrain_counts=np.concatenate((left.terrain_counts, right.terrain_counts)),
    )


def choose_candidate(
    train: SampleTable,
    validation: SampleTable,
    context: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    candidates = []
    x_train = features(train, context)
    x_validation = features(validation, context)
    weights = event_balanced_weights(train.event_ids)
    for spec in MODEL_SPECS:
        model = make_model(spec, seed)
        model.fit(x_train, train.utility, **({"sample_weight": weights} if not hasattr(model, "steps") else {"ridge__sample_weight": weights}))
        aligned_score = model.predict(x_validation)
        aligned = aggregate_gate(validation, aligned_score)
        controls: dict[str, Any] = {}
        if "M" in context:
            controls["material_shuffle"] = aggregate_gate(
                validation, model.predict(features(validation, context, "material_shuffle"))
            )
        if "R" in context:
            controls["trigger_wrong"] = aggregate_gate(
                validation, model.predict(features(validation, context, "trigger_wrong"))
            )
            controls["trigger_shuffle"] = aggregate_gate(
                validation, model.predict(features(validation, context, "trigger_shuffle"))
            )
        if context == "TMR":
            controls["both"] = aggregate_gate(
                validation, model.predict(features(validation, context, "both"))
            )
        candidates.append({"spec": spec, "aligned": aligned, "controls": controls, "model": model})
    candidates.sort(
        key=lambda row: (
            row["aligned"]["net_corrected_vs_visual"],
            row["aligned"]["iou"],
            -row["aligned"]["coverage"],
        ),
        reverse=True,
    )
    chosen = candidates[0]
    receipt = {
        "context": context,
        "chosen_spec": chosen["spec"],
        "gate_threshold": 0.0,
        "selection_metric": "validation net corrected pixels vs frozen visual; IoU tiebreak",
        "aligned": chosen["aligned"],
        "controls": chosen["controls"],
        "candidate_summaries": [
            {"spec": row["spec"], "aligned": row["aligned"], "controls": row["controls"]}
            for row in candidates
        ],
    }
    return chosen["model"], receipt


def fold_paths(root: Path, seed: int, fold: int) -> dict[str, Path]:
    cache = root / "cache" / f"seed{seed}" / f"fold{fold}"
    result = root / f"seed{seed}" / f"fold{fold}" / "material" / "result.json"
    return {
        "train": cache / "frozen_train_joint_cache.pt",
        "validation": cache / "frozen_outer_val_cache.pt",
        "test": cache / "frozen_test_cache.pt",
        "result": result,
    }


def validate_fold(root: Path, seed: int, fold: int) -> dict[str, Any]:
    paths = fold_paths(root, seed, fold)
    for key in ("train", "validation", "result"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    threshold = float(result["visual_threshold"])
    train = load_table(paths["train"], threshold)
    validation = load_table(paths["validation"], threshold)
    contexts = {}
    for context in CONTEXTS:
        _, receipt = choose_candidate(train, validation, context, seed + fold)
        contexts[context] = receipt
    t_net = contexts["T"]["aligned"]["net_corrected_vs_visual"]
    for context in ("TM", "TR", "TMR"):
        receipt = contexts[context]
        aligned_net = receipt["aligned"]["net_corrected_vs_visual"]
        controls = receipt["controls"]
        control_nets = [item["net_corrected_vs_visual"] for item in controls.values()]
        receipt["increment_over_T"] = aligned_net - t_net
        receipt["aligned_beats_all_controls"] = bool(control_nets and aligned_net > max(control_nets))
        receipt["validation_gate_passed"] = bool(
            aligned_net > t_net and receipt["aligned_beats_all_controls"]
        )
    return {
        "fold": fold,
        "threshold": threshold,
        "train_cache_sha256": sha256_file(paths["train"]),
        "validation_cache_sha256": sha256_file(paths["validation"]),
        "test_cache_accessed": False,
        "contexts": contexts,
    }


def write_validation_csv(path: Path, folds: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "fold", "context", "chosen_spec", "net_corrected_vs_visual", "rer_vs_visual",
        "iou", "coverage", "increment_over_T", "aligned_beats_all_controls",
        "validation_gate_passed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fold in folds:
            for context, receipt in fold["contexts"].items():
                aligned = receipt["aligned"]
                writer.writerow({
                    "fold": fold["fold"], "context": context,
                    "chosen_spec": receipt["chosen_spec"],
                    "net_corrected_vs_visual": aligned["net_corrected_vs_visual"],
                    "rer_vs_visual": aligned["rer_vs_visual"], "iou": aligned["iou"],
                    "coverage": aligned["coverage"],
                    "increment_over_T": receipt.get("increment_over_T", ""),
                    "aligned_beats_all_controls": receipt.get("aligned_beats_all_controls", ""),
                    "validation_gate_passed": receipt.get("validation_gate_passed", ""),
                })


def fit_frozen_spec(table: SampleTable, context: str, spec: str, seed: int):
    model = make_model(spec, seed)
    weights = event_balanced_weights(table.event_ids)
    fit_kwargs = (
        {"ridge__sample_weight": weights}
        if hasattr(model, "steps") else {"sample_weight": weights}
    )
    model.fit(features(table, context), table.utility, **fit_kwargs)
    return model


def pooled_from_fold_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    visual_errors = sum(int(row["visual_errors"]) for row in rows)
    selected_errors = fp + fn
    n_samples = sum(int(row["n_samples"]) for row in rows)
    selected_samples = sum(int(row["terrain_selected_samples"]) for row in rows)
    return {
        "n_folds": len(rows), "n_samples": n_samples,
        "terrain_selected_samples": selected_samples,
        "coverage": selected_samples / max(n_samples, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": tp / max(tp + fp + fn, 1),
        "visual_errors": visual_errors,
        "selected_errors": selected_errors,
        "net_corrected_vs_visual": visual_errors - selected_errors,
        "rer_vs_visual": (visual_errors - selected_errors) / max(visual_errors, 1),
    }


def test_from_receipt(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.outdir / "validation_summary.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    validation_summary = json.loads(receipt_path.read_text(encoding="utf-8"))
    if validation_summary.get("label_contract", {}).get("test_cache_accessed") is not False:
        raise RuntimeError("validation receipt does not certify an unread test cache")
    fold_metrics = []
    sample_rows = []
    for fold_receipt in validation_summary["folds"]:
        fold = int(fold_receipt["fold"])
        paths = fold_paths(args.runs_root, args.seed, fold)
        if sha256_file(paths["train"]) != fold_receipt["train_cache_sha256"]:
            raise RuntimeError(f"fold {fold} train cache changed after validation")
        if sha256_file(paths["validation"]) != fold_receipt["validation_cache_sha256"]:
            raise RuntimeError(f"fold {fold} validation cache changed after validation")
        if not paths["test"].is_file():
            raise FileNotFoundError(paths["test"])
        threshold = float(fold_receipt["threshold"])
        train = load_table(paths["train"], threshold)
        validation = load_table(paths["validation"], threshold)
        refit = concatenate_tables(train, validation)
        selected = fold_receipt["contexts"]["T"]
        model = fit_frozen_spec(refit, "T", selected["chosen_spec"], args.seed + fold)
        test = load_table(paths["test"], threshold)
        score = model.predict(features(test, "T"))
        metric = aggregate_gate(test, score)
        metric.update({
            "fold": fold,
            "chosen_spec_from_validation": selected["chosen_spec"],
            "threshold_from_validation": threshold,
            "gate_threshold": 0.0,
            "test_cache_sha256": sha256_file(paths["test"]),
            "unconditional_visual": aggregate_gate(test, np.full(len(test.sample_ids), -1.0)),
            "unconditional_terrain": aggregate_gate(test, np.full(len(test.sample_ids), 1.0)),
        })
        fold_metrics.append(metric)
        gate = score > 0
        for index, sample_id in enumerate(test.sample_ids):
            sample_rows.append({
                "fold": fold, "sample_id": sample_id, "event_id": test.event_ids[index],
                "predicted_utility": float(score[index]),
                "terrain_selected": int(gate[index]),
                "observed_utility_for_evaluation_only": float(test.utility[index]),
                "visual_errors": int(test.visual_counts[index, 1:3].sum()),
                "selected_errors": int((test.terrain_counts if gate[index] else test.visual_counts)[index, 1:3].sum()),
            })
        del train, validation, refit, test
    pooled = pooled_from_fold_metrics(fold_metrics)
    output = {
        "schema_version": "sen12_context_utility_gate_probe_v1_test",
        "status": "test_complete",
        "context_evaluated": "T",
        "contexts_not_evaluated": [
            context for context, eligible in validation_summary["eligible_for_test"].items()
            if not eligible
        ],
        "selection": "fold-specific model family selected on outer validation; refit on train+validation; fixed score threshold zero",
        "inference_uses_test_labels": False,
        "test_labels_used_only_for_reported_metrics": True,
        "folds": fold_metrics,
        "pooled": pooled,
    }
    atomic_json(args.outdir / "test_T_summary.json", output)
    sample_path = args.outdir / "test_T_per_sample.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "test"), default="validate")
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("experiments/revision2026/sen12_prithvi_roleaware_hierarchical_v2"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--folds", default="0,1,2,3,4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "test":
        output = test_from_receipt(args)
        print(json.dumps(output["pooled"], sort_keys=True))
        return 0
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    receipts = [validate_fold(args.runs_root, args.seed, fold) for fold in folds]
    pass_counts = {
        context: sum(bool(row["contexts"][context].get("validation_gate_passed")) for row in receipts)
        for context in ("TM", "TR", "TMR")
    }
    summary = {
        "schema_version": "sen12_context_utility_gate_probe_v1",
        "status": "validation_complete",
        "scientific_scope": "predictability probe; not a final GeoPhysAdapter result",
        "label_contract": {
            "training_target": "outer-train Terrain-vs-visual net correction utility",
            "inference_features": "label-free visual uncertainty, Terrain summaries, Material, Trigger",
            "validation": "outer validation only",
            "test_cache_accessed": False,
        },
        "gate": "select frozen Terrain correction iff predicted utility > 0",
        "folds": receipts,
        "validation_pass_counts_out_of_folds": pass_counts,
        "expansion_rule": "do not read test unless aligned M/R beats T and all matched controls in at least 3/5 folds",
        "eligible_for_test": {
            context: pass_counts[context] >= 3 for context in pass_counts
        },
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.outdir / "validation_summary.json", summary)
    write_validation_csv(args.outdir / "validation_fold_contexts.csv", receipts)
    print(json.dumps(summary["eligible_for_test"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
