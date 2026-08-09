#!/usr/bin/env python3
"""Tests for the strict unified role-aware experiment analyzer."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import analyze_pild_sen12_roleaware_v1 as analyzer


SEEDS = (101, 102)
SAMPLES = (
    ("pild", "event_a", "a0", 1.0, 1.0),
    ("pild", "event_a", "a1", 0.0, 1.0),
    ("sen12", "event_b", "b0", 1.0, 0.0),
    ("sen12", "event_c", "c0", 1.0, 1.0),
)

# tp, fp, fn, tn. Every condition preserves seven positive and 93 negative
# target pixels per sample while changing the thresholded error count.
COUNTS = {
    "V": (5, 3, 2, 90),
    "VT": (6, 3, 1, 90),
    "VTM": (6, 2, 1, 91),
    "VTR": (6, 2, 1, 91),
    "VTMR": (7, 2, 0, 91),
    "T_zero": (5, 3, 2, 90),
    "T_shift": (5, 3, 2, 90),
    "T_roll": (5, 4, 2, 89),
    "T_donor": (6, 3, 1, 90),
    "M_aligned": (6, 2, 1, 91),
    "M_shuffle": (6, 3, 1, 90),
    "M_zero_q": (6, 3, 1, 90),
    "R_aligned": (6, 2, 1, 91),
    "R_wrong_time": (6, 3, 1, 90),
    "R_event_shuffle": (5, 3, 2, 90),
    "R_zero_q": (6, 3, 1, 90),
    "VTMR_material-trigger-both-zero-q": (6, 3, 1, 90),
}

EXTRA_CONTEXTS = {
    "VTMR_material-trigger-both-zero-q": (
        "material-trigger-both-zero-q",
        "aligned",
        "zero-q",
        "zero-q",
    ),
}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def reference_for(condition: str, parent: str) -> str:
    return analyzer.REFERENCE_CONDITION.get(
        condition, analyzer.REFERENCE_CONDITION[parent]
    )


def sample_frame(
    condition: str, parent: str, seed: int, checkpoint_sha256: str
) -> pd.DataFrame:
    tp, fp, fn, tn = COUNTS[condition]
    reference = reference_for(condition, parent)
    ref_tp, ref_fp, ref_fn, _ = COUNTS[reference]
    current_errors = fp + fn
    reference_errors = ref_fp + ref_fn
    corrected = max(reference_errors - current_errors, 0)
    harmed = max(current_errors - reference_errors, 0)
    condition_index = list(COUNTS).index(condition)
    context = (
        analyzer.EXPECTED_EVALUATION_CONTEXT[condition]
        if condition in analyzer.EXPECTED_EVALUATION_CONTEXT
        else EXTRA_CONTEXTS[condition]
    )
    rows = []
    for sample_index, (source, event, sample, q_m, q_r) in enumerate(SAMPLES):
        ap = 0.55 + 0.005 * condition_index + 0.001 * (seed - SEEDS[0])
        fixed_tp = min(7, 4 + int(condition in {"VTR", "VTMR", "R_aligned"}))
        fixed_fn = 7 - fixed_tp
        fixed_fp = max(0, 10 - int(condition in {"VTR", "VTMR", "R_aligned"}))
        fixed_tn = 93 - fixed_fp
        rows.append(
            {
                "split": "test",
                "source": source,
                "canonical_event_id": event,
                "sample_id": sample,
                "variant": parent,
                "condition": condition,
                "seed": seed,
                "reference_condition": reference,
                "evaluation_context": context[0],
                "terrain_context": context[1],
                "material_context": context[2],
                "trigger_context": context[3],
                "checkpoint_sha256": checkpoint_sha256,
                "q_material": q_m,
                "q_trigger": q_r,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "ap": ap,
                "corrected": corrected,
                "harmed": harmed,
                "brier_sum": 18.0 - 0.1 * condition_index + sample_index * 0.01,
                "nll_sum": 42.0 - 0.1 * condition_index + sample_index * 0.01,
                "soft_area_error": 3.0 - 0.05 * condition_index,
                "fixed_fpr_tp": fixed_tp,
                "fixed_fpr_fn": fixed_fn,
                "fixed_fpr_fp": fixed_fp,
                "fixed_fpr_tn": fixed_tn,
                "valid_pixel_count": 100,
                "target_positive_count": 7,
            }
        )
    return pd.DataFrame(rows)


def artifact_receipt(path: Path) -> dict[str, object]:
    return {"sha256": analyzer.sha256_file(path), "size": path.stat().st_size}


def write_parent_run(root: Path, parent: str, seed: int) -> Path:
    run = root / f"{parent}_seed{seed}"
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint:{parent}:{seed}".encode("ascii"))
    checkpoint_sha256 = analyzer.sha256_file(checkpoint)
    frames = [
        sample_frame(condition, parent, seed, checkpoint_sha256)
        for condition in analyzer.PARENT_CSV_CONDITIONS[parent]
    ]
    per_sample = pd.concat(frames, ignore_index=True)
    per_sample.to_csv(run / "per_sample.csv", index=False)
    per_sample.loc[per_sample["condition"] == parent].to_csv(
        run / "per_sample_metrics.csv", index=False
    )
    identity = {
        "manifest_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "fold_id": "fixture",
        "seed": seed,
    }
    result = {
        "schema_version": analyzer.RUN_SCHEMA,
        "status": "complete",
        "variant": parent,
        "condition": parent,
        "seed": seed,
        "identity": identity,
        "evaluation_split": "test",
        "fixed_fpr_threshold_source": "validation_visual_only",
        "fixed_fpr_threshold": 0.42 + 0.001 * (seed - SEEDS[0]),
        "checkpoint_sha256": checkpoint_sha256,
    }
    config = {
        "schema_version": analyzer.CONFIG_SCHEMA,
        "variant": parent,
        "condition": parent,
        "seed": seed,
        "identity": identity,
        "terrain_channel_order": list(analyzer.TERRAIN9_CHANNEL_ORDER),
        "material_interaction_groups": {
            name: list(indices)
            for name, indices in analyzer.MATERIAL_INTERACTION_GROUPS.items()
        },
    }
    write_json(run / "config.json", config)
    write_json(run / "result.json", result)
    artifacts = {
        name: artifact_receipt(run / name)
        for name in (
            "config.json",
            "result.json",
            "checkpoint.pt",
            "per_sample.csv",
            "per_sample_metrics.csv",
        )
    }
    done = {
        "schema_version": analyzer.DONE_SCHEMA,
        "status": "complete",
        "variant": parent,
        "condition": parent,
        "seed": seed,
        "config_sha256": analyzer.sha256_file(run / "config.json"),
        "result_sha256": analyzer.sha256_file(run / "result.json"),
        "per_sample_metrics_sha256": analyzer.sha256_file(
            run / "per_sample_metrics.csv"
        ),
        "artifacts": artifacts,
    }
    write_json(run / "DONE.json", done)
    return run


def refresh_artifact_hash(run: Path, name: str) -> None:
    done = json.loads((run / "DONE.json").read_text(encoding="utf-8"))
    done["artifacts"][name] = artifact_receipt(run / name)
    top_level = {
        "config.json": "config_sha256",
        "result.json": "result_sha256",
        "per_sample_metrics.csv": "per_sample_metrics_sha256",
    }
    if name in top_level:
        done[top_level[name]] = analyzer.sha256_file(run / name)
    write_json(run / "DONE.json", done)


def build_fixture(root: Path) -> None:
    for parent in analyzer.MAIN_CONDITIONS:
        for seed in SEEDS:
            write_parent_run(root, parent, seed)


class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "runs"
        self.root.mkdir()
        build_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_analysis(self) -> tuple[dict, Path]:
        output = Path(self.temporary.name) / "analysis"
        summary = analyzer.analyze(
            self.root,
            output,
            seeds=SEEDS,
            min_seeds=2,
            n_bootstrap=100,
            bootstrap_seed=7,
            permutation_iterations=100,
        )
        return summary, output

    def test_complete_matrix_produces_strict_atomic_outputs(self) -> None:
        summary, output = self.run_analysis()
        self.assertEqual(summary["n_runs"], len(analyzer.MAIN_CONDITIONS) * len(SEEDS))
        self.assertEqual(summary["n_analysis_conditions"], len(analyzer.CONDITIONS))
        self.assertEqual(summary["n_canonical_events"], 3)
        expected = {
            "summary.json",
            "report.md",
            "method_metrics.csv",
            "contrast_summary.csv",
            "paired_sample_metrics.csv",
            "paired_event_metrics.csv",
            "per_event_metrics.csv",
            "support_strata_metrics.csv",
        }
        self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
        parsed = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"),
            parse_constant=lambda value: self.fail(
                f"nonstandard JSON constant: {value}"
            ),
        )
        self.assertEqual(parsed["schema_version"], analyzer.ANALYSIS_SCHEMA)
        self.assertEqual(len(parsed["input_inventory"]), 10)
        contrasts = pd.read_csv(output / "contrast_summary.csv")
        self.assertEqual(set(contrasts["contrast"]), set(analyzer.CONTRASTS))
        self.assertGreater(
            contrasts.set_index("contrast").loc["M_aligned_minus_shuffle", "rer"], 0
        )
        self.assertGreater(
            contrasts.set_index("contrast").loc["R_aligned_minus_wrong_time", "rer"], 0
        )
        self.assertFalse(any(path.name.startswith(".") for path in output.iterdir()))

    def test_missing_parent_run_is_fatal(self) -> None:
        shutil.rmtree(self.root / f"VTR_seed{SEEDS[-1]}")
        with self.assertRaisesRegex(
            analyzer.AnalysisContractError,
            "parent variant seed sets differ|requested seed inventory",
        ):
            analyzer.resolve_seeds(self.root, SEEDS, 2)

    def test_negative_control_cannot_masquerade_as_independent_run(self) -> None:
        (self.root / f"T_zero_seed{SEEDS[0]}").mkdir()
        with self.assertRaisesRegex(
            analyzer.AnalysisContractError, "negative controls must be rows"
        ):
            analyzer.resolve_seeds(self.root, SEEDS, 2)

    def test_missing_negative_control_row_is_fatal(self) -> None:
        run = self.root / f"VTM_seed{SEEDS[0]}"
        frame = pd.read_csv(run / "per_sample.csv")
        frame = frame.loc[frame["condition"] != "M_shuffle"]
        frame.to_csv(run / "per_sample.csv", index=False)
        refresh_artifact_hash(run, "per_sample.csv")
        with self.assertRaisesRegex(
            analyzer.AnalysisContractError, "test condition inventory mismatch"
        ):
            analyzer.load_run(self.root, "M_shuffle", SEEDS[0])

    def test_per_sample_hash_drift_is_fatal(self) -> None:
        run = self.root / f"VT_seed{SEEDS[0]}"
        with (run / "per_sample.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "size mismatch|hash mismatch"):
            analyzer.load_run(self.root, "T_zero", SEEDS[0])

    def test_control_checkpoint_provenance_drift_is_fatal(self) -> None:
        run = self.root / f"VTR_seed{SEEDS[0]}"
        frame = pd.read_csv(run / "per_sample.csv")
        frame.loc[frame["condition"] == "R_wrong_time", "checkpoint_sha256"] = "f" * 64
        frame.to_csv(run / "per_sample.csv", index=False)
        refresh_artifact_hash(run, "per_sample.csv")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "same-checkpoint provenance"):
            analyzer.load_run(self.root, "R_wrong_time", SEEDS[0])

    def test_extra_seed_in_one_parent_is_fatal(self) -> None:
        write_parent_run(self.root, "V", 103)
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "requested seed inventory"):
            analyzer.resolve_seeds(self.root, SEEDS, 2)

    def test_missing_csv_schema_is_fatal(self) -> None:
        run = self.root / f"VTM_seed{SEEDS[0]}"
        frame = pd.read_csv(run / "per_sample.csv").drop(columns="q_material")
        frame.to_csv(run / "per_sample.csv", index=False)
        refresh_artifact_hash(run, "per_sample.csv")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "missing columns"):
            analyzer.load_run(self.root, "VTM", SEEDS[0])

    def test_unpaired_sample_key_is_fatal(self) -> None:
        run = self.root / f"VT_seed{SEEDS[0]}"
        frame = pd.read_csv(run / "per_sample.csv")
        mask = frame["condition"] == "T_shift"
        frame.loc[mask.idxmax(), "sample_id"] = "unexpected_sample"
        frame.to_csv(run / "per_sample.csv", index=False)
        refresh_artifact_hash(run, "per_sample.csv")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "sample pairing differs"):
            analyzer.load_all_runs(self.root, SEEDS)

    def test_corrected_harmed_identity_is_fatal(self) -> None:
        run = self.root / f"VTM_seed{SEEDS[0]}"
        frame = pd.read_csv(run / "per_sample.csv")
        mask = frame["condition"] == "M_aligned"
        frame.loc[mask.idxmax(), "corrected"] += 1
        frame.to_csv(run / "per_sample.csv", index=False)
        refresh_artifact_hash(run, "per_sample.csv")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "corrected-harmed identity"):
            analyzer.load_all_runs(self.root, SEEDS)

    def test_wrong_fixed_fpr_contract_is_fatal(self) -> None:
        run = self.root / f"VTR_seed{SEEDS[0]}"
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        result["fixed_fpr_threshold_source"] = "test_labels"
        write_json(run / "result.json", result)
        refresh_artifact_hash(run, "result.json")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "validation_visual_only"):
            analyzer.load_run(self.root, "R_aligned", SEEDS[0])

    def test_config_result_identity_drift_is_fatal(self) -> None:
        run = self.root / f"VTM_seed{SEEDS[0]}"
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        result["identity"]["fold_id"] = "wrong-fold"
        write_json(run / "result.json", result)
        refresh_artifact_hash(run, "result.json")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "config/result identity"):
            analyzer.load_run(self.root, "VTM", SEEDS[0])

    def test_terrain_or_material_channel_semantics_drift_is_fatal(self) -> None:
        run = self.root / f"VTM_seed{SEEDS[0]}"
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        config["material_interaction_groups"]["relief"] = [6, 7, 8]
        write_json(run / "config.json", config)
        refresh_artifact_hash(run, "config.json")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "interaction groups mismatch"):
            analyzer.load_run(self.root, "M_aligned", SEEDS[0])

    def test_terrain9_order_drift_is_fatal(self) -> None:
        run = self.root / f"VT_seed{SEEDS[0]}"
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        config["terrain_channel_order"][0:2] = reversed(
            config["terrain_channel_order"][0:2]
        )
        write_json(run / "config.json", config)
        refresh_artifact_hash(run, "config.json")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "terrain9 channel order"):
            analyzer.load_run(self.root, "VT", SEEDS[0])

    def test_missing_run_config_is_fatal(self) -> None:
        run = self.root / f"V_seed{SEEDS[0]}"
        (run / "config.json").unlink()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "incomplete parent run"):
            analyzer.load_run(self.root, "V", SEEDS[0])

    def test_fixed_fpr_threshold_drift_is_fatal(self) -> None:
        run = self.root / f"VTR_seed{SEEDS[0]}"
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        result["fixed_fpr_threshold"] = 0.91
        write_json(run / "result.json", result)
        refresh_artifact_hash(run, "result.json")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "threshold differs"):
            analyzer.load_all_runs(self.root, SEEDS)

    def test_primary_metrics_must_match_parent_aligned_rows(self) -> None:
        run = self.root / f"VTM_seed{SEEDS[0]}"
        frame = pd.read_csv(run / "per_sample_metrics.csv")
        frame.loc[0, "tp"] += 1
        frame.to_csv(run / "per_sample_metrics.csv", index=False)
        refresh_artifact_hash(run, "per_sample_metrics.csv")
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "primary row subset"):
            analyzer.load_run(self.root, "VTM", SEEDS[0])

    def test_json_safe_maps_nonfinite_to_null(self) -> None:
        payload = analyzer.json_safe({"nan": float("nan"), "inf": np.float64(np.inf)})
        self.assertEqual(payload, {"nan": None, "inf": None})
        encoded = json.dumps(payload, allow_nan=False)
        self.assertEqual(json.loads(encoded), {"nan": None, "inf": None})


if __name__ == "__main__":
    unittest.main(verbosity=2)
