#!/usr/bin/env python3
"""Aggregate the five Sen12 T×M development folds without reopening test choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_RUN_FILES = ("DONE.json", "checkpoint.pt", "result.json", "per_sample_test.csv")
CONDITIONS = ("zero", "aligned", "shuffle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def aggregate_counts(frame: pd.DataFrame) -> dict[str, float | int]:
    tp = int(frame["tp"].sum())
    fp = int(frame["fp"].sum())
    fn = int(frame["fn"].sum())
    denom = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / denom if denom else float("nan")}


def paired_event_bootstrap(
    event_frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    seed: int = 20260722,
    n_boot: int = 20000,
) -> dict[str, Any]:
    wide = event_frame.pivot(index="event_id", columns="condition", values="iou")
    delta = (wide[left] - wide[right]).dropna().to_numpy(float)
    if len(delta) == 0:
        raise ValueError(f"no paired events for {left} vs {right}")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, len(delta)), replace=True)
    permuted = np.abs((signs * delta).mean(axis=1))
    observed = abs(float(delta.mean()))
    return {
        "comparison": f"{left}_minus_{right}",
        "n_events": int(len(delta)),
        "mean_delta_iou": float(delta.mean()),
        "median_delta_iou": float(np.median(delta)),
        "bootstrap_ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "sign_flip_p_two_sided": float((1 + np.count_nonzero(permuted >= observed)) / (n_boot + 1)),
        "n_positive": int(np.count_nonzero(delta > 0)),
        "n_negative": int(np.count_nonzero(delta < 0)),
        "n_zero": int(np.count_nonzero(delta == 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/revision2026/sen12_tm_susceptibility_v1"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/revision2026/sen12_tm_susceptibility_v1_analysis"),
    )
    parser.add_argument("--expected-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = sorted(path for path in args.runs_dir.glob("fold*_seed*") if path.is_dir())
    if len(run_dirs) != args.expected_folds:
        raise SystemExit(f"expected {args.expected_folds} run directories, found {len(run_dirs)}")

    results: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    for run_dir in run_dirs:
        missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise SystemExit(f"{run_dir}: missing {missing}")
        result = json.loads((run_dir / "result.json").read_text())
        done = json.loads((run_dir / "DONE.json").read_text())
        if done.get("status") != "complete":
            raise SystemExit(f"{run_dir}: DONE status is not complete")
        fold = int(result["fold"])
        frame = pd.read_csv(run_dir / "per_sample_test.csv")
        required = {"condition", "event_id", "sample_id", "tp", "fp", "fn"}
        if not required.issubset(frame.columns) or frame.empty:
            raise SystemExit(f"{run_dir}: invalid per_sample_test.csv")
        if set(frame["condition"].unique()) != set(CONDITIONS):
            raise SystemExit(f"{run_dir}: expected conditions {CONDITIONS}")
        counts = frame.groupby("sample_id")["condition"].nunique()
        if not counts.eq(len(CONDITIONS)).all():
            raise SystemExit(f"{run_dir}: controls do not share an identical sample set")
        frame["fold"] = fold
        samples.append(frame)
        results.append(result)
        inputs.append({
            "run_dir": str(run_dir.resolve()),
            "result_sha256": sha256_file(run_dir / "result.json"),
            "per_sample_sha256": sha256_file(run_dir / "per_sample_test.csv"),
            "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt"),
        })

    folds = sorted(int(item["fold"]) for item in results)
    if folds != list(range(args.expected_folds)):
        raise SystemExit(f"fold identities are not 0..{args.expected_folds - 1}: {folds}")
    pooled = pd.concat(samples, ignore_index=True)
    duplicated = pooled.duplicated(["condition", "sample_id"])
    if duplicated.any():
        raise SystemExit("a sample appears in the test partition of more than one fold")

    pooled_metrics = {
        condition: aggregate_counts(pooled.loc[pooled["condition"].eq(condition)])
        for condition in CONDITIONS
    }
    event_rows = []
    for (condition, event_id), group in pooled.groupby(["condition", "event_id"], sort=True):
        event_rows.append({"condition": condition, "event_id": event_id, **aggregate_counts(group)})
    event_frame = pd.DataFrame(event_rows)
    aligned_zero = paired_event_bootstrap(event_frame, "aligned", "zero")
    aligned_shuffle = paired_event_bootstrap(event_frame, "aligned", "shuffle")

    n_gate_pass = sum(bool(item["material_gate_pass"]) for item in results)
    identity_exact = (
        pooled_metrics["zero"]["tp"] == pooled_metrics["aligned"]["tp"]
        and pooled_metrics["zero"]["fp"] == pooled_metrics["aligned"]["fp"]
        and pooled_metrics["zero"]["fn"] == pooled_metrics["aligned"]["fn"]
    )
    pooled_gain = float(pooled_metrics["aligned"]["iou"] - pooled_metrics["zero"]["iou"])
    control_gap = float(pooled_metrics["aligned"]["iou"] - pooled_metrics["shuffle"]["iou"])
    paper_gate = bool(
        n_gate_pass >= 3
        and pooled_gain > 0
        and control_gap > 0
        and aligned_zero["bootstrap_ci95"][0] > 0
    )
    if n_gate_pass == 0 and not identity_exact:
        raise SystemExit("all folds abstained but aligned output is not the exact Terrain parent")

    fold_rows = []
    for item in sorted(results, key=lambda value: int(value["fold"])):
        fold_rows.append({
            "fold": int(item["fold"]),
            "material_gate_pass": bool(item["material_gate_pass"]),
            "selected_epoch": int(item["material_selected_epoch"]),
            "terrain_val_ap": float(item["terrain_best_val_ap"]),
            "aligned_test_ap": float(item["test"]["aligned"]["ap"]),
            "aligned_test_iou": float(item["test"]["aligned"]["metrics"]["iou"]),
            "zero_test_iou": float(item["test"]["zero"]["metrics"]["iou"]),
            "shuffle_test_iou": float(item["test"]["shuffle"]["metrics"]["iou"]),
        })

    summary = json_safe({
        "status": "complete",
        "evidence_role": "development gate; not independent Full-TMR confirmation",
        "n_folds": len(results),
        "n_unique_events": int(event_frame["event_id"].nunique()),
        "n_unique_samples": int(pooled["sample_id"].nunique()),
        "n_material_gate_pass_folds": n_gate_pass,
        "pooled": pooled_metrics,
        "paired_event_stats": {
            "aligned_vs_zero": aligned_zero,
            "aligned_vs_shuffle": aligned_shuffle,
        },
        "pooled_aligned_minus_zero_iou": pooled_gain,
        "pooled_aligned_minus_shuffle_iou": control_gap,
        "all_abstain_exact_identity": bool(n_gate_pass == 0 and identity_exact),
        "paper_material_gate": paper_gate,
        "decision": "retain_material_modulation" if paper_gate else "abstain_material_in_full_tmr",
        "inputs": inputs,
    })

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.outdir / "fold_metrics.csv", index=False)
    event_frame.to_csv(args.outdir / "per_event_metrics.csv", index=False)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = [
        "# Sen12 T×M susceptibility development gate", "",
        f"- Complete folds: {len(results)}; Material-qualified folds: {n_gate_pass}.",
        f"- Pooled aligned-minus-Terrain IoU: {pooled_gain:+.6f}.",
        f"- Pooled aligned-minus-event-shuffled Material IoU: {control_gap:+.6f}.",
        f"- Event-level aligned-minus-Terrain 95% CI: [{aligned_zero['bootstrap_ci95'][0]:+.6f}, {aligned_zero['bootstrap_ci95'][1]:+.6f}].",
        f"- Decision: `{summary['decision']}`.", "",
        "This is a development gate. Test metrics were evaluated once with each fold's frozen validation threshold.",
        "A failed Material gate is an exact abstention, not evidence of a positive Material segmentation effect.",
    ]
    (args.outdir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
