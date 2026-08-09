#!/usr/bin/env python3
"""Tests for the strict Sen12 role-aware Material/Trigger analyzer."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import analyze_sen12_prithvi_roleaware_mr_v1 as analyzer


SEEDS = (101, 102)
FOLDS = (0, 1)
SAMPLES = (
    ("sample_a", "event_a", "region_a"),
    ("sample_b", "event_a", "region_a"),
    ("sample_c", "event_b", "region_b"),
)
VT_COUNTS = (5, 3, 2, 90)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "sha256": analyzer.sha256_file(path),
    }


def metric(counts: tuple[int, int, int, int]) -> tuple[float, int]:
    tp, fp, fn, _ = counts
    return tp / max(tp + fp + fn, 1), fp + fn


def counts_for(mode: str, control: str, q_r: float) -> tuple[int, int, int, int]:
    if q_r == 0 and mode == "trigger":
        return VT_COUNTS
    if control == "all_zero_q":
        return VT_COUNTS
    if mode == "material":
        return {
            "aligned": (6, 2, 1, 91),
            "material_shuffle": (6, 3, 1, 90),
            "material_zero_q": VT_COUNTS,
        }[control]
    if mode == "trigger":
        return {
            "aligned": (6, 2, 1, 91),
            "trigger_wrong_time": (6, 3, 1, 90),
            "trigger_event_shuffle": VT_COUNTS,
            "trigger_zero_q": VT_COUNTS,
        }[control]
    return {
        "aligned": (7, 2, 0, 91),
        "material_shuffle": (6, 2, 1, 91),
        "material_zero_q": (6, 3, 1, 90) if q_r > 0 else VT_COUNTS,
        "trigger_wrong_time": (6, 2, 1, 91),
        "trigger_event_shuffle": (6, 3, 1, 90),
        "trigger_zero_q": (6, 3, 1, 90),
        "all_zero_q": VT_COUNTS,
    }[control]


def sample_frame(mode: str, q_r: float, fold: int) -> pd.DataFrame:
    vt_iou, vt_errors = metric(VT_COUNTS)
    rows = []
    for base_sample_id, base_event_id, source_id in SAMPLES:
        sample_id = f"{base_sample_id}_fold{fold}"
        event_id = f"{base_event_id}_fold{fold}"
        for control_index, control in enumerate(analyzer.CONTROLS_BY_MODE[mode]):
            current = counts_for(mode, control, q_r)
            iou, errors = metric(current)
            corrected = max(vt_errors - errors, 0)
            harmed = max(errors - vt_errors, 0)
            if control == "material_shuffle":
                effective_q_m = 1.0
            elif control in ("material_zero_q", "all_zero_q"):
                effective_q_m = 0.0
            else:
                effective_q_m = 1.0
            if control == "trigger_event_shuffle":
                effective_q_r = q_r
            elif control in ("trigger_zero_q", "all_zero_q"):
                effective_q_r = 0.0
            else:
                effective_q_r = q_r
            if control == "aligned":
                applicable = True
            elif control.startswith("material_"):
                applicable = True
            elif control.startswith("trigger_"):
                applicable = q_r > 0
            else:
                applicable = q_r > 0
            tp, fp, fn, tn = current
            baseline_correct = VT_COUNTS[0] + VT_COUNTS[3]
            rows.append({
                "sample_id": sample_id, "event_id": event_id, "source_id": source_id,
                "mode": mode, "control": control, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "iou": iou, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "accuracy": (tp + tn) / 100, "vt_iou": vt_iou,
                "errors": errors, "vt_errors": vt_errors, "corrected": corrected,
                "harmed": harmed, "baseline_condition": "frozen_VT",
                "baseline_correct_count": baseline_correct,
                "preserved_correct_count": baseline_correct - harmed,
                "preservation_rate": (baseline_correct - harmed) / baseline_correct,
                "brier": 0.22 + 0.01 * errors + 0.002 * control_index,
                "nll": 0.45 + 0.01 * errors + 0.002 * control_index,
                "predicted_area": tp + fp, "true_area": 7,
                "fixed_fpr_tp": max(tp - 1, 0), "fixed_fpr_fn": 7 - max(tp - 1, 0),
                "q_M": 1.0, "q_R": q_r, "effective_q_M": effective_q_m,
                "effective_q_R": effective_q_r,
                "material_donor_sample_id": "material_donor" if control == "material_shuffle" else sample_id,
                "material_donor_event_id": "event_donor" if control == "material_shuffle" else event_id,
                "trigger_donor_sample_id": "trigger_donor" if control == "trigger_event_shuffle" else sample_id,
                "trigger_donor_event_id": "event_donor" if control == "trigger_event_shuffle" else event_id,
                "control_applicable": applicable,
                "material_shuffle_pair_applicable": True,
                "trigger_wrongtime_pair_applicable": q_r > 0,
                "trigger_event_shuffle_pair_applicable": q_r > 0,
                "material_delta_abs_mean": 0.1 if effective_q_m > 0 else 0.0,
                "trigger_delta_abs_mean": 0.1 if effective_q_r > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def event_frame(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (control, event_id), group in samples.groupby(["control", "event_id"], sort=True):
        tp, fp, fn, tn = (int(group[name].sum()) for name in ("tp", "fp", "fn", "tn"))
        rows.append({
            "control": control, "event_id": event_id, "n_samples": len(group),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": tp / max(tp + fp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1), "errors": fp + fn,
            "corrected": int(group["corrected"].sum()), "harmed": int(group["harmed"].sum()),
            "baseline_condition": "frozen_VT",
            "baseline_correct_count": int(group["baseline_correct_count"].sum()),
            "preserved_correct_count": int(group["preserved_correct_count"].sum()),
            "brier": float(group["brier"].mean()), "nll": float(group["nll"].mean()),
        })
    return pd.DataFrame(rows)


def control_frame(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in samples.to_dict("records"):
        rows.append({
            "sample_id": row["sample_id"], "event_id": row["event_id"], "mode": row["mode"],
            "control": row["control"],
            "checkpoint_selection": "same-aligned-validation-checkpoint",
            "material_context": row["control"], "trigger_context": row["control"],
            "effective_q_M": row["effective_q_M"], "effective_q_R": row["effective_q_R"],
            "material_donor_sample_id": row["material_donor_sample_id"],
            "material_donor_event_id": row["material_donor_event_id"],
            "trigger_donor_sample_id": row["trigger_donor_sample_id"],
            "trigger_donor_event_id": row["trigger_donor_event_id"],
            "control_applicable": row["control_applicable"],
            "pairing_rule": "aligned-versus-control only where both recipient rows are applicable",
            "material_shuffle_abstain": not row["material_shuffle_pair_applicable"],
            "trigger_shuffle_abstain": not row["trigger_event_shuffle_pair_applicable"],
        })
    return pd.DataFrame(rows)


def receipt_frame(samples: pd.DataFrame) -> pd.DataFrame:
    aligned = samples.loc[samples["control"] == "aligned"].set_index("sample_id")
    rows = []
    for row in samples.loc[
        (samples["control"] != "aligned") & samples["control_applicable"]
    ].to_dict("records"):
        reference = aligned.loc[row["sample_id"]]
        rows.append({
            "sample_id": row["sample_id"], "event_id": row["event_id"],
            "source_id": row["source_id"], "mode": row["mode"], "control": row["control"],
            "pair_applicable": True,
            "checkpoint_selection": "same-aligned-validation-checkpoint",
            "aligned_iou": reference["iou"], "control_iou": row["iou"],
            "delta_iou_aligned_minus_control": reference["iou"] - row["iou"],
            "aligned_errors": reference["errors"], "control_errors": row["errors"],
            "error_reduction_aligned_minus_control": row["errors"] - reference["errors"],
            "effective_q_M_control": row["effective_q_M"],
            "effective_q_R_control": row["effective_q_R"],
            "material_donor_sample_id": row["material_donor_sample_id"],
            "material_donor_event_id": row["material_donor_event_id"],
            "trigger_donor_sample_id": row["trigger_donor_sample_id"],
            "trigger_donor_event_id": row["trigger_donor_event_id"],
        })
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=[
        "sample_id", "event_id", "source_id", "mode", "control", "pair_applicable",
        "checkpoint_selection", "aligned_iou", "control_iou",
        "delta_iou_aligned_minus_control", "aligned_errors", "control_errors",
        "error_reduction_aligned_minus_control", "effective_q_M_control",
        "effective_q_R_control", "material_donor_sample_id", "material_donor_event_id",
        "trigger_donor_sample_id", "trigger_donor_event_id",
    ])


def aggregate_control(samples: pd.DataFrame, control: str) -> dict:
    frame = samples.loc[samples["control"] == control]
    tp, fp, fn, tn = (int(frame[name].sum()) for name in ("tp", "fp", "fn", "tn"))
    errors = fp + fn
    fixed_tp = int(frame["fixed_fpr_tp"].sum())
    fixed_fn = int(frame["fixed_fpr_fn"].sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": tp / max(tp + fp + fn, 1), "ap": 0.55 + 0.01 * (5 - errors),
        "errors": errors, "rer_vs_vt": 0.0,
        "brier_mean": float(frame["brier"].mean()), "nll_mean": float(frame["nll"].mean()),
        "area_abs_error_mean": float((frame["predicted_area"] - frame["true_area"]).abs().mean()),
        "fixed_fpr_recall": fixed_tp / max(fixed_tp + fixed_fn, 1),
        "corrected_vs_frozen_vt": int(frame["corrected"].sum()),
        "harmed_vs_frozen_vt": int(frame["harmed"].sum()),
        "preservation_rate_vs_frozen_vt": float(frame["preservation_rate"].mean()),
    }


def refresh_hashes(run: Path) -> None:
    hashes = {name: analyzer.sha256_file(run / name) for name in analyzer.HASHED_ARTIFACTS}
    write_json(run / "hashes.json", hashes)
    done = json.loads((run / "DONE.json").read_text(encoding="utf-8"))
    done["hashes_sha256"] = analyzer.sha256_file(run / "hashes.json")
    write_json(run / "DONE.json", done)


def write_run(root: Path, parents: Path, seed: int, fold: int, mode: str, q_r: float) -> Path:
    run = root / f"seed{seed}" / f"fold{fold}" / mode
    run.mkdir(parents=True)
    parent_dir = parents / f"seed{seed}" / f"fold{fold}"
    parent_dir.mkdir(parents=True, exist_ok=True)
    visual = parent_dir / "visual.pt"
    terrain = parent_dir / "terrain.pt"
    if not visual.exists():
        visual.write_bytes(f"visual-{seed}-{fold}".encode())
        terrain.write_bytes(f"terrain-{seed}-{fold}".encode())
    parent = {
        "schema_version": analyzer.PARENT_SCHEMA, "fold": fold, "seed": seed,
        "threshold": 0.42, "visual": signature(visual), "terrain": signature(terrain),
        "terrain_embedded_seed": seed, "legacy_seed_resolution": None,
        "direction_semantics": "frozen VT correction direction",
    }
    samples = sample_frame(mode, q_r, fold)
    events = event_frame(samples)
    controls = control_frame(samples)
    receipts = receipt_frame(samples)
    samples.to_csv(run / "per_sample.csv", index=False)
    events.to_csv(run / "per_event.csv", index=False)
    controls.to_csv(run / "control_rows.csv", index=False)
    controls.to_csv(run / "same_checkpoint_controls.csv", index=False)
    receipts.to_csv(run / "paired_control_receipts.csv", index=False)
    (run / "command.txt").write_text("synthetic test command\n", encoding="utf-8")
    (run / "run.log").write_text("complete\n", encoding="utf-8")
    (run / "checkpoint.pt").write_bytes(f"head-{seed}-{fold}-{mode}".encode())
    write_json(run / "config.json", {
        "schema_version": analyzer.CONFIG_SCHEMA, "mode": mode, "seed": seed, "fold": fold,
        "contract": {"training_context": "aligned-only"},
    })
    test = {
        "vt": {"tp": 15, "fp": 9, "fn": 6, "tn": 270, "iou": 0.5},
        "controls": {
            control: aggregate_control(samples, control)
            for control in analyzer.CONTROLS_BY_MODE[mode]
        },
    }
    write_json(run / "result.json", {
        "schema_version": analyzer.RUN_SCHEMA, "status": "complete", "mode": mode,
        "seed": seed, "fold": fold, "test": test, "parent_identity": parent,
        "fold_interpretation": (
            "q_R=0 test: exact VT fallback audit only; excluded from Trigger-effect aggregation"
            if q_r == 0 else "q_R-supported test contributes to Trigger-effect aggregation"
        ),
    })
    write_json(run / "DONE.json", {
        "schema_version": analyzer.DONE_SCHEMA, "status": "complete", "mode": mode,
        "seed": seed, "fold": fold, "same_checkpoint_controls": True,
        "hashes_sha256": "0" * 64,
    })
    refresh_hashes(run)
    return run


def build_fixture(root: Path, parents: Path) -> None:
    for seed in SEEDS:
        for fold in FOLDS:
            q_r = 1.0 if fold == 0 else 0.0
            for mode in analyzer.MODES:
                write_run(root, parents, seed, fold, mode, q_r)


class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        analyzer._FILE_SHA_CACHE.clear()
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.root = base / "runs"
        self.parents = base / "parents"
        self.root.mkdir()
        build_fixture(self.root, self.parents)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_analysis(self) -> tuple[dict, Path]:
        output = Path(self.temporary.name) / "analysis"
        summary = analyzer.analyze(
            self.root, output, seeds=SEEDS, folds=FOLDS, min_seeds=2,
            bootstrap_reps=100, bootstrap_seed=7,
        )
        return summary, output

    def test_complete_matrix_writes_strict_atomic_outputs(self) -> None:
        summary, output = self.run_analysis()
        self.assertEqual(summary["n_runs"], len(SEEDS) * len(FOLDS) * len(analyzer.MODES))
        self.assertEqual(summary["n_optimization_seeds"], 2)
        expected = set(summary["outputs"])
        self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
        parsed = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"),
            parse_constant=lambda value: self.fail(f"nonstandard JSON constant: {value}"),
        )
        self.assertEqual(parsed["schema_version"], analyzer.ANALYSIS_SCHEMA)
        self.assertFalse(any(path.name.startswith(".") for path in output.iterdir()))
        statistics = pd.read_csv(output / "paired_statistics.csv")
        sample_iou = statistics.loc[
            (statistics["contrast"] == "material_aligned_vs_vt")
            & (statistics["level"] == "sample")
            & (statistics["metric"] == "delta_iou")
        ].iloc[0]
        self.assertEqual(int(sample_iou["n"]), len(SAMPLES) * len(FOLDS))
        ap = pd.read_csv(output / "ap_summary.csv")
        self.assertEqual(
            int(ap.set_index("contrast").loc["material_aligned_vs_shuffle", "n_seeds"]),
            len(SEEDS),
        )
        availability = pd.read_csv(output / "metric_availability.csv").set_index("metric")
        self.assertFalse(bool(availability.loc["soft_area", "aligned_vs_control"]))

    def test_missing_fold_is_fatal(self) -> None:
        shutil.rmtree(self.root / f"seed{SEEDS[0]}" / "fold1" / "joint")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "incomplete run"):
            analyzer.load_matrix(self.root, SEEDS, FOLDS)

    def test_control_checkpoint_mismatch_is_fatal(self) -> None:
        run = self.root / f"seed{SEEDS[0]}" / "fold0" / "material"
        frame = pd.read_csv(run / "control_rows.csv")
        frame.loc[0, "checkpoint_selection"] = "different-checkpoint"
        frame.to_csv(run / "control_rows.csv", index=False)
        analyzer._FILE_SHA_CACHE.clear()
        refresh_hashes(run)
        analyzer._FILE_SHA_CACHE.clear()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "checkpoint mismatch"):
            analyzer.load_run(run, "material", SEEDS[0], 0)

    def test_parent_checkpoint_sha_drift_is_fatal(self) -> None:
        run = self.root / f"seed{SEEDS[0]}" / "fold0" / "trigger"
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        result["parent_identity"]["terrain"]["sha256"] = "f" * 64
        write_json(run / "result.json", result)
        analyzer._FILE_SHA_CACHE.clear()
        refresh_hashes(run)
        analyzer._FILE_SHA_CACHE.clear()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "parent terrain checkpoint SHA drift"):
            analyzer.load_run(run, "trigger", SEEDS[0], 0)

    def test_q0_exact_fallback_violation_is_fatal(self) -> None:
        run = self.root / f"seed{SEEDS[0]}" / "fold0" / "material"
        frame = pd.read_csv(run / "per_sample.csv")
        index = frame.index[frame["control"] == "material_zero_q"][0]
        frame.loc[index, ["tp", "fp", "fn", "tn"]] = [6, 3, 1, 90]
        frame.loc[index, "iou"] = 0.6
        frame.loc[index, "errors"] = 4
        frame.loc[index, "corrected"] = 1
        frame.loc[index, "harmed"] = 0
        frame.to_csv(run / "per_sample.csv", index=False)
        analyzer._FILE_SHA_CACHE.clear()
        refresh_hashes(run)
        analyzer._FILE_SHA_CACHE.clear()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "q0 exact fallback"):
            analyzer.load_run(run, "material", SEEDS[0], 0)

    def test_unsupported_trigger_aligned_must_fallback(self) -> None:
        run = self.root / f"seed{SEEDS[0]}" / "fold1" / "trigger"
        frame = pd.read_csv(run / "per_sample.csv")
        index = frame.index[frame["control"] == "aligned"][0]
        frame.loc[index, ["tp", "fp", "fn", "tn"]] = [6, 3, 1, 90]
        frame.loc[index, "iou"] = 0.6
        frame.loc[index, "errors"] = 4
        frame.loc[index, "corrected"] = 1
        frame.loc[index, "harmed"] = 0
        frame.to_csv(run / "per_sample.csv", index=False)
        analyzer._FILE_SHA_CACHE.clear()
        refresh_hashes(run)
        analyzer._FILE_SHA_CACHE.clear()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "effective-q0"):
            analyzer.load_run(run, "trigger", SEEDS[0], 1)

    def test_empty_primary_csv_is_fatal(self) -> None:
        run = self.root / f"seed{SEEDS[0]}" / "fold0" / "joint"
        columns = pd.read_csv(run / "per_sample.csv", nrows=0).columns
        pd.DataFrame(columns=columns).to_csv(run / "per_sample.csv", index=False)
        analyzer._FILE_SHA_CACHE.clear()
        refresh_hashes(run)
        analyzer._FILE_SHA_CACHE.clear()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "CSV is empty"):
            analyzer.load_run(run, "joint", SEEDS[0], 0)

    def test_csv_schema_drift_across_folds_is_fatal(self) -> None:
        run = self.root / f"seed{SEEDS[0]}" / "fold1" / "material"
        frame = pd.read_csv(run / "per_sample.csv")
        frame["unexpected_column"] = 1
        frame.to_csv(run / "per_sample.csv", index=False)
        analyzer._FILE_SHA_CACHE.clear()
        refresh_hashes(run)
        analyzer._FILE_SHA_CACHE.clear()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "per_sample schema drift"):
            analyzer.load_matrix(self.root, SEEDS, FOLDS)

    def test_trigger_q0_fold_is_not_counted_as_efficacy(self) -> None:
        _, output = self.run_analysis()
        sample = pd.read_csv(output / "paired_sample_metrics.csv")
        trigger = sample.loc[sample["contrast"].str.contains("trigger", case=False)]
        self.assertEqual(set(trigger["fold"]), {0})
        inventory = pd.read_csv(output / "run_inventory.csv")
        q0 = inventory.loc[(inventory["mode"] == "trigger") & (inventory["fold"] == 1)]
        self.assertTrue((q0["q_R_positive_samples"] == 0).all())

    def test_nan_is_sanitized_to_standard_json_null(self) -> None:
        payload = analyzer.json_safe({"nan": float("nan"), "inf": np.float64(np.inf)})
        self.assertEqual(payload, {"nan": None, "inf": None})
        encoded = json.dumps(payload, allow_nan=False)
        self.assertEqual(json.loads(encoded), {"nan": None, "inf": None})


if __name__ == "__main__":
    unittest.main(verbosity=2)
