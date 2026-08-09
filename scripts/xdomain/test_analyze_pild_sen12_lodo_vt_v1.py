#!/usr/bin/env python3
"""Contract tests for the full-corpus PILD-XDomain LODO V/VT analyzer."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

import analyze_pild_sen12_lodo_vt_v1 as analyzer
import analyze_pild_sen12_roleaware_v1 as shared


FOLDS = ("lodo_00_A", "lodo_01_B", "lodo_02_C", "lodo_03_D")
SEEDS = (11, 12, 13, 14, 15)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def receipt(path: Path) -> dict[str, object]:
    return {"sha256": shared.sha256_file(path), "size": path.stat().st_size}


def sample_row(fold_index: int, seed: int, variant: str, checkpoint_sha: str) -> dict[str, object]:
    if variant == "V":
        tp, fp, fn, tn = 40 + fold_index, 10, 10, 940 - fold_index
        corrected = harmed = 0
        reference_errors = final_errors = visual_errors = fp + fn
        ap = 0.60 + fold_index * 0.01
    else:
        tp, fp, fn, tn = 43 + fold_index, 9, 7, 941 - fold_index
        corrected, harmed = 4, 0
        reference_errors, final_errors, visual_errors = 20, 16, 20
        ap = 0.63 + fold_index * 0.01
    return {
        "sample_id": f"sample_{fold_index}",
        "dataset_id": chr(ord("A") + fold_index),
        "canonical_event_id": f"event_{fold_index}",
        "split": "test",
        "condition": variant,
        "variant": variant,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "reference_condition": "V",
        "ap": ap,
        "corrected": corrected,
        "harmed": harmed,
        "reference_errors": reference_errors,
        "final_errors": final_errors,
        "visual_errors": visual_errors,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "brier_sum": 10.0 if variant == "V" else 9.0,
        "nll_sum": 20.0 if variant == "V" else 18.0,
        "soft_area_error": 2.0 if variant == "V" else 1.0,
        "fixed_fpr_tp": 40,
        "fixed_fpr_fn": 10,
        "fixed_fpr_fp": 5,
        "fixed_fpr_tn": 945,
        "valid_pixel_count": 1000,
        "target_positive_count": 50,
    }


def write_run(root: Path, fold: str, fold_index: int, seed: int, variant: str) -> Path:
    run = root / fold / f"{variant}_seed{seed}"
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint.pt"
    if variant == "V":
        checkpoint.write_bytes(f"V:{fold}:{seed}".encode())
    else:
        checkpoint.write_bytes(f"VT:{fold}:{seed}".encode())
    checkpoint_sha = shared.sha256_file(checkpoint)
    v_run = root / fold / f"V_seed{seed}"
    identity = {"manifest_sha256": "a" * 64, "split_sha256": "b" * 64, "fold_id": fold, "seed": seed, "prithvi_checkpoint_sha256": "c" * 64}
    visual_hash = f"{fold_index + seed:064x}"[-64:]
    config = {
        "schema_version": "pild_sen12_roleaware_config.v1",
        "variant": variant,
        "condition": variant,
        "seed": seed,
        "identity": identity,
        "parent_checkpoint": str((v_run / "checkpoint.pt").resolve()) if variant == "VT" else None,
    }
    result = {
        "schema_version": analyzer.RUN_SCHEMA,
        "status": "complete",
        "variant": variant,
        "condition": variant,
        "seed": seed,
        "identity": identity,
        "evaluation_split": "test",
        "threshold": 0.5,
        "checkpoint_sha256": checkpoint_sha,
        "component_sha256": {"visual_decoder": visual_hash, **({"terrain_adapter": "d" * 64} if variant == "VT" else {})},
    }
    write_json(run / "config.json", config)
    write_json(run / "result.json", result)
    pd.DataFrame([sample_row(fold_index, seed, variant, checkpoint_sha)]).to_csv(run / "per_sample_metrics.csv", index=False)
    done = {
        "schema_version": analyzer.DONE_SCHEMA,
        "status": "complete",
        "variant": variant,
        "condition": variant,
        "seed": seed,
        "fold_id": fold,
        "artifacts": {name: receipt(run / name) for name in ("checkpoint.pt", "config.json", "result.json", "per_sample_metrics.csv")},
    }
    write_json(run / "DONE.json", done)
    return run


def refresh(run: Path, name: str) -> None:
    done = json.loads((run / "DONE.json").read_text())
    done["artifacts"][name] = receipt(run / name)
    write_json(run / "DONE.json", done)


class FullOOFAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="pild-lodo-analysis-"))
        self.runs = self.temp / "runs"
        split_rows = []
        for fold_index, fold in enumerate(FOLDS):
            split_rows.append({"fold_id": fold, "sample_id": f"sample_{fold_index}", "dataset_id": chr(ord("A") + fold_index), "canonical_event_id": f"event_{fold_index}", "role": "test"})
            for seed in SEEDS:
                write_run(self.runs, fold, fold_index, seed, "V")
                write_run(self.runs, fold, fold_index, seed, "VT")
        self.split = self.temp / "split.csv"
        pd.DataFrame(split_rows).to_csv(self.split, index=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def run_analysis(self) -> dict:
        return analyzer.analyze(self.runs, self.split, self.temp / "analysis", min_seeds=5, bootstrap=200, bootstrap_seed=7)

    def test_full_oof_primary_uses_all_folds(self) -> None:
        result = self.run_analysis()
        self.assertEqual(result["n_samples"], 4)
        self.assertEqual(result["n_datasets"], 4)
        self.assertEqual(result["n_runs"], 40)
        self.assertGreater(result["primary_vt_minus_v"]["delta_iou"]["mean"], 0)
        self.assertEqual(result["primary_vt_minus_v"]["delta_iou"]["positive_seeds"], 5)

    def test_missing_fold_variant_fails_closed(self) -> None:
        shutil.rmtree(self.runs / FOLDS[-1] / f"VT_seed{SEEDS[-1]}")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "seed sets differ"):
            self.run_analysis()

    def test_repeated_test_sample_across_folds_fails(self) -> None:
        split = pd.read_csv(self.split)
        split.loc[1, "sample_id"] = split.loc[0, "sample_id"]
        split.to_csv(self.split, index=False)
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "repeat across folds"):
            self.run_analysis()

    def test_wrong_vt_parent_fails(self) -> None:
        run = self.runs / FOLDS[0] / f"VT_seed{SEEDS[0]}"
        config = json.loads((run / "config.json").read_text())
        config["parent_checkpoint"] = str((self.runs / FOLDS[0] / f"V_seed{SEEDS[1]}" / "checkpoint.pt").resolve())
        write_json(run / "config.json", config)
        refresh(run, "config.json")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "parent path mismatch"):
            self.run_analysis()

    def test_changed_visual_component_fails(self) -> None:
        run = self.runs / FOLDS[0] / f"VT_seed{SEEDS[0]}"
        result = json.loads((run / "result.json").read_text())
        result["component_sha256"]["visual_decoder"] = "f" * 64
        write_json(run / "result.json", result)
        refresh(run, "result.json")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "visual component changed"):
            self.run_analysis()


if __name__ == "__main__":
    unittest.main()
