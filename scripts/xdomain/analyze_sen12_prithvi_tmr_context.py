#!/usr/bin/env python3
"""Strictly aggregate Sen12 bounded Material/Trigger modulation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ("material", "trigger", "joint")
FOLDS = tuple(range(5))
COUNT_KEYS = ("tp", "fp", "fn", "tn")
EXPECTED_METHODS = {
    "material": {"visual", "vt", "aligned", "material_shuffled", "zero_q"},
    "trigger": {"visual", "vt", "aligned", "trigger_wrongtime", "zero_q"},
    "joint": {
        "visual", "vt", "aligned", "material_shuffled", "trigger_wrongtime", "zero_q"
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (counts[key] for key in COUNT_KEYS)
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def bootstrap_mean(values: np.ndarray, seed: int, n_bootstrap: int) -> list[float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=root / "experiments/revision2026/sen12_prithvi_tmr_context_v1",
    )
    parser.add_argument("--seed", type=int, default=20260771)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = args.outdir or args.runs_root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    sample_frames = []
    event_frames = []
    result_rows = []
    baseline_by_fold: dict[int, pd.DataFrame] = {}
    for fold in FOLDS:
        for mode in MODES:
            run = args.runs_root / f"fold{fold}" / mode
            required = [run / name for name in ("DONE.json", "result.json", "modulator.pt", "per_sample.csv", "per_event.csv")]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"incomplete fold={fold} mode={mode}: {missing}")
            done = json.loads((run / "DONE.json").read_text())
            expected_hashes = {
                "result_sha256": run / "result.json",
                "checkpoint_sha256": run / "modulator.pt",
                "per_sample_sha256": run / "per_sample.csv",
                "per_event_sha256": run / "per_event.csv",
            }
            for key, path in expected_hashes.items():
                if done.get(key) != sha256(path):
                    raise RuntimeError(f"artifact hash mismatch: {path}")
            result = json.loads((run / "result.json").read_text())
            if (result.get("fold"), result.get("mode"), result.get("seed")) != (fold, mode, args.seed):
                raise RuntimeError(f"run identity mismatch: {run}")
            sample = pd.read_csv(run / "per_sample.csv")
            event = pd.read_csv(run / "per_event.csv")
            methods = set(sample["method"].astype(str))
            if methods != EXPECTED_METHODS[mode]:
                raise RuntimeError(f"method schema mismatch fold={fold} mode={mode}: {methods}")
            if sample.groupby(["sample_id", "method"]).size().ne(1).any():
                raise RuntimeError(f"duplicate sample/method rows: {run}")
            vt = sample.loc[sample["method"] == "vt"].set_index("sample_id").sort_index()
            zero = sample.loc[sample["method"] == "zero_q"].set_index("sample_id").sort_index()
            if not vt.loc[:, list(COUNT_KEYS)].equals(zero.loc[:, list(COUNT_KEYS)]):
                raise RuntimeError(f"zero-q is not exact VT: {run}")
            baseline = sample.loc[
                sample["method"].isin(["visual", "vt"]),
                ["sample_id", "method", *COUNT_KEYS],
            ].sort_values(["sample_id", "method"]).reset_index(drop=True)
            if fold in baseline_by_fold and not baseline.equals(baseline_by_fold[fold]):
                raise RuntimeError(f"frozen baseline differs across modes for fold {fold}")
            baseline_by_fold[fold] = baseline
            sample.insert(0, "fold", fold)
            sample.insert(1, "mode", mode)
            event.insert(0, "fold", fold)
            event.insert(1, "mode", mode)
            sample_frames.append(sample)
            event_frames.append(event)
            result_rows.append({
                "fold": fold,
                "mode": mode,
                "best_epoch": result["best_epoch"],
                "aligned_ap": result["test"]["variants"]["aligned"]["ap"],
                "vt_ap": result["test"]["vt_baseline"]["ap"],
            })

    samples = pd.concat(sample_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    aggregate_rows = []
    for (mode, method), frame in samples.groupby(["mode", "method"], sort=True):
        current = {key: int(frame[key].sum()) for key in COUNT_KEYS}
        row = {"mode": mode, "method": method, **current, **metrics(current)}
        row["errors"] = current["fp"] + current["fn"]
        vt = samples.loc[
            (samples["mode"] == mode) & (samples["method"] == "vt")
        ]
        vt_counts = {key: int(vt[key].sum()) for key in COUNT_KEYS}
        vt_metrics = metrics(vt_counts)
        vt_errors = vt_counts["fp"] + vt_counts["fn"]
        row["delta_iou_vs_vt"] = row["iou"] - vt_metrics["iou"]
        row["rer_vs_vt"] = (vt_errors - row["errors"]) / max(vt_errors, 1)
        row["corrected"] = int(frame["corrected"].sum())
        row["harmed"] = int(frame["harmed"].sum())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    contrasts = []
    contrast_specs = {
        "material": (("aligned", "vt"), ("aligned", "material_shuffled")),
        "trigger": (("aligned", "vt"), ("aligned", "trigger_wrongtime")),
        "joint": (
            ("aligned", "vt"),
            ("aligned", "material_shuffled"),
            ("aligned", "trigger_wrongtime"),
        ),
    }
    for mode, specifications in contrast_specs.items():
        mode_events = events.loc[events["mode"] == mode]
        for left, right in specifications:
            left_frame = mode_events.loc[mode_events["method"] == left].set_index(
                ["fold", "physical_event_id"]
            )
            right_frame = mode_events.loc[mode_events["method"] == right].set_index(
                ["fold", "physical_event_id"]
            )
            paired = left_frame.join(right_frame, lsuffix="_left", rsuffix="_right", how="inner")
            if len(paired) != left_frame.index.nunique() or len(paired) == 0:
                raise RuntimeError(f"event pairing failed: {mode} {left}-{right}")
            delta = (paired["iou_left"] - paired["iou_right"]).to_numpy()
            contrasts.append({
                "mode": mode,
                "contrast": f"{left}_minus_{right}",
                "n_events": len(delta),
                "mean_event_delta_iou": float(np.mean(delta)),
                "median_event_delta_iou": float(np.median(delta)),
                "positive_events": int((delta > 0).sum()),
                "event_bootstrap_mean_ci95": bootstrap_mean(
                    delta, args.seed + len(contrasts), args.bootstrap
                ),
                "pooled_error_difference": int(
                    paired["errors_right"].sum() - paired["errors_left"].sum()
                ),
            })
    contrast_frame = pd.DataFrame(contrasts)
    aggregate.to_csv(outdir / "pooled_method_metrics.csv", index=False)
    contrast_frame.to_csv(outdir / "event_contrasts.csv", index=False)
    pd.DataFrame(result_rows).to_csv(outdir / "fold_selection_metrics.csv", index=False)

    summary = {
        "status": "complete",
        "seed": args.seed,
        "n_runs": len(MODES) * len(FOLDS),
        "strict_checks": [
            "artifact_hashes", "run_identity", "method_schema", "zero_q_exact_vt",
            "shared_frozen_baseline", "complete_event_pairing",
        ],
        "pooled_metrics": aggregate.to_dict("records"),
        "event_contrasts": contrasts,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Sen12 bounded Material/Trigger modulation", "",
        "All runs passed artifact, identity, exact-fallback and paired-event checks.", "",
        "## Pooled metrics", "", aggregate.to_markdown(index=False), "",
        "## Event contrasts", "", contrast_frame.to_markdown(index=False), "",
    ]
    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "n_runs": summary["n_runs"], "outdir": str(outdir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
