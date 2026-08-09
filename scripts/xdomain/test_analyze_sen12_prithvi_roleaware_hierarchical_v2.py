#!/usr/bin/env python3
"""False-positive tests for the strict Sen12 hierarchical v2 analyzer."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

import pandas as pd
import torch

import analyze_sen12_prithvi_roleaware_hierarchical_v2 as analyzer


SEEDS = (101,)
FOLDS = tuple(range(5))
SAMPLES = (("a", "event_a", "region_a"), ("b", "event_a", "region_a"),
           ("c", "event_b", "region_b"))
VT_COUNTS = (5, 3, 2, 90)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": analyzer.sha256_file(path)}


def metric(counts: tuple[int, int, int, int]) -> tuple[float, int]:
    tp, fp, fn, _ = counts
    return tp / max(tp + fp + fn, 1), fp + fn


def counts_for(mode: str, control: str, q_r: float) -> tuple[int, int, int, int]:
    if mode == "trigger" and q_r == 0:
        return VT_COUNTS
    tables = {
        "material": {"aligned": (6, 2, 1, 91), "material_shuffle": (5, 3, 2, 90),
                     "material_zero_q": VT_COUNTS},
        "trigger": {"aligned": (6, 2, 1, 91), "trigger_wrong_time": (5, 3, 2, 90),
                    "trigger_event_shuffle": (5, 3, 2, 90), "trigger_zero_q": VT_COUNTS},
        "joint": {"aligned": (7, 2, 0, 91), "material_shuffle": (6, 2, 1, 91),
                  "material_zero_q": (6, 2, 1, 91) if q_r > 0 else VT_COUNTS,
                  "trigger_wrong_time": (6, 2, 1, 91),
                  "trigger_event_shuffle": (6, 2, 1, 91),
                  "trigger_zero_q": (6, 2, 1, 91), "all_zero_q": VT_COUNTS},
    }
    return tables[mode][control]


def sample_frame(mode: str, q_r: float, fold: int) -> pd.DataFrame:
    vt_iou, vt_errors = metric(VT_COUNTS)
    rows = []
    for base_sample, base_event, source in SAMPLES:
        sample_id, event_id = f"{base_sample}_f{fold}", f"{base_event}_f{fold}"
        for index, control in enumerate(analyzer.CONTROLS_BY_MODE[mode]):
            current = counts_for(mode, control, q_r)
            iou, errors = metric(current)
            corrected, harmed = max(vt_errors - errors, 0), max(errors - vt_errors, 0)
            effective_q_m = 0.0 if control in ("material_zero_q", "all_zero_q") else 1.0
            effective_q_r = 0.0 if control in ("trigger_zero_q", "all_zero_q") else q_r
            if control == "aligned":
                applicable = True
            elif control.startswith("material_"):
                applicable = True
            else:
                applicable = q_r > 0
            terrain_count = 20
            material_eligible = effective_q_m > 0 and terrain_count > 0
            trigger_eligible = effective_q_r > 0 and terrain_count > 0
            joint_eligible = material_eligible and trigger_eligible
            trigger_active = mode in ("trigger", "joint") and effective_q_r > 0
            changed = (8 if control == "trigger_event_shuffle" else 10) if trigger_active else 0
            tp, fp, fn, tn = current
            baseline_correct = VT_COUNTS[0] + VT_COUNTS[3]
            rows.append({
                "sample_id": sample_id, "event_id": event_id, "source_id": source,
                "mode": mode, "control": control, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "iou": iou, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "accuracy": (tp + tn) / 100, "vt_iou": vt_iou,
                "errors": errors, "vt_errors": vt_errors, "corrected": corrected,
                "harmed": harmed, "baseline_condition": "frozen_VT",
                "baseline_correct_count": baseline_correct,
                "preserved_correct_count": baseline_correct - harmed,
                "preservation_rate": (baseline_correct - harmed) / baseline_correct,
                "brier": 0.2 + 0.01 * errors + 0.001 * index,
                "nll": 0.4 + 0.01 * errors + 0.001 * index,
                "predicted_area": tp + fp, "true_area": 7,
                "fixed_fpr_tp": max(tp - 1, 0), "fixed_fpr_fn": 7 - max(tp - 1, 0),
                "q_M": 1.0, "q_R": q_r, "effective_q_M": effective_q_m,
                "effective_q_R": effective_q_r,
                "visual_uncertainty_mean": 0.3, "visual_uncertainty_q75": 0.5,
                "visual_uncertainty_q90": 0.7, "terrain_support_pixel_count": terrain_count,
                "terrain_support_fraction": 0.2, "material_scalar": 0.1,
                "material_multiplier_abs_deviation_mean": 0.02 if effective_q_m else 0.0,
                "rain_contrast": 1.0 if effective_q_r else 0.0,
                "rain_gain": 0.2 if effective_q_r else 0.0,
                "material_local_effect_eligible": material_eligible,
                "trigger_local_effect_eligible": trigger_eligible,
                "joint_local_effect_eligible": joint_eligible,
                "local_effect_subset_uses_test_label": False,
                "material_donor_sample_id": "m_donor" if control == "material_shuffle" else sample_id,
                "material_donor_event_id": "m_event" if control == "material_shuffle" else event_id,
                "trigger_donor_sample_id": "r_donor" if control == "trigger_event_shuffle" else sample_id,
                "trigger_donor_event_id": "r_event" if control == "trigger_event_shuffle" else event_id,
                "control_applicable": applicable, "material_shuffle_pair_applicable": True,
                "trigger_wrongtime_pair_applicable": q_r > 0,
                "trigger_event_shuffle_pair_applicable": q_r > 0,
                "trigger_event_shuffle_donor_scope": "outer-train-supported-events-only",
                "material_delta_abs_mean": 0.1 if effective_q_m else 0.0,
                "trigger_delta_abs_mean": 0.1 if trigger_active else 0.0,
                "trigger_changed_pixel_count": changed,
                "trigger_terrain_overlap_pixel_count": changed,
                "trigger_terrain_overlap_fraction": 1.0,
                "trigger_support_overlap_100pct": True,
                "trigger_signed_direction_violation_count": 0,
            })
    return pd.DataFrame(rows)


def event_frame(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (control, event), group in samples.groupby(["control", "event_id"], sort=True):
        tp, fp, fn, tn = (int(group[key].sum()) for key in ("tp", "fp", "fn", "tn"))
        rows.append({"control": control, "event_id": event, "n_samples": len(group),
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "iou": tp / max(tp + fp + fn, 1), "errors": fp + fn,
                     "corrected": int(group["corrected"].sum()),
                     "harmed": int(group["harmed"].sum()),
                     "baseline_condition": "frozen_VT",
                     "baseline_correct_count": int(group["baseline_correct_count"].sum()),
                     "preserved_correct_count": int(group["preserved_correct_count"].sum()),
                     "brier": float(group["brier"].mean()), "nll": float(group["nll"].mean())})
    return pd.DataFrame(rows)


def control_frame(samples: pd.DataFrame) -> pd.DataFrame:
    copied = ("q_M", "q_R", "effective_q_M", "effective_q_R", "visual_uncertainty_mean",
              "visual_uncertainty_q75", "visual_uncertainty_q90", "terrain_support_pixel_count",
              "terrain_support_fraction", "material_scalar",
              "material_multiplier_abs_deviation_mean", "rain_contrast", "rain_gain",
              "trigger_changed_pixel_count", "trigger_terrain_overlap_pixel_count",
              "trigger_support_overlap_100pct", "material_local_effect_eligible",
              "trigger_local_effect_eligible", "joint_local_effect_eligible",
              "local_effect_subset_uses_test_label", "material_donor_sample_id",
              "material_donor_event_id", "trigger_donor_sample_id", "trigger_donor_event_id",
              "trigger_event_shuffle_donor_scope", "control_applicable")
    rows = []
    for row in samples.to_dict("records"):
        material_context, trigger_context = analyzer.CONTROL_CONTEXTS[row["control"]]
        receipt = {key: row[key] for key in copied}
        receipt.update({"sample_id": row["sample_id"], "event_id": row["event_id"],
                        "mode": row["mode"], "control": row["control"],
                        "checkpoint_selection": "same-aligned-validation-checkpoint",
                        "material_context": material_context, "trigger_context": trigger_context,
                        "pairing_rule": "aligned-versus-control only where both recipient rows are applicable",
                        "material_shuffle_abstain": not row["material_shuffle_pair_applicable"],
                        "trigger_shuffle_abstain": not row["trigger_event_shuffle_pair_applicable"]})
        rows.append(receipt)
    return pd.DataFrame(rows)


def receipt_frame(samples: pd.DataFrame) -> pd.DataFrame:
    aligned = samples.loc[samples["control"] == "aligned"].set_index("sample_id")
    rows = []
    for row in samples.loc[(samples["control"] != "aligned") & samples["control_applicable"]].to_dict("records"):
        ref = aligned.loc[row["sample_id"]]
        rows.append({"sample_id": row["sample_id"], "event_id": row["event_id"],
                     "source_id": row["source_id"], "mode": row["mode"], "control": row["control"],
                     "pair_applicable": True,
                     "checkpoint_selection": "same-aligned-validation-checkpoint",
                     "aligned_iou": ref["iou"], "control_iou": row["iou"],
                     "delta_iou_aligned_minus_control": ref["iou"] - row["iou"],
                     "aligned_errors": ref["errors"], "control_errors": row["errors"],
                     "error_reduction_aligned_minus_control": row["errors"] - ref["errors"],
                     "effective_q_M_control": row["effective_q_M"],
                     "effective_q_R_control": row["effective_q_R"],
                     "material_donor_sample_id": row["material_donor_sample_id"],
                     "material_donor_event_id": row["material_donor_event_id"],
                     "trigger_donor_sample_id": row["trigger_donor_sample_id"],
                     "trigger_donor_event_id": row["trigger_donor_event_id"],
                     "trigger_event_shuffle_donor_scope": row["trigger_event_shuffle_donor_scope"]})
    return pd.DataFrame(rows, columns=analyzer.RECEIPT_REQUIRED + (
        "source_id", "effective_q_M_control", "effective_q_R_control",
        "material_donor_sample_id", "material_donor_event_id", "trigger_donor_sample_id",
        "trigger_donor_event_id", "trigger_event_shuffle_donor_scope",
    ))


def aggregate_control(samples: pd.DataFrame, control: str) -> dict:
    frame = samples.loc[samples["control"] == control]
    tp, fp, fn, tn = (int(frame[key].sum()) for key in ("tp", "fp", "fn", "tn"))
    errors, fixed_tp, fixed_fn = fp + fn, int(frame["fixed_fpr_tp"].sum()), int(frame["fixed_fpr_fn"].sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "iou": tp / max(tp + fp + fn, 1),
            "ap": 0.6, "errors": errors, "rer_vs_vt": 0.0,
            "brier_mean": float(frame["brier"].mean()), "nll_mean": float(frame["nll"].mean()),
            "area_abs_error_mean": float((frame["predicted_area"] - frame["true_area"]).abs().mean()),
            "fixed_fpr_recall": fixed_tp / max(fixed_tp + fixed_fn, 1),
            "corrected_vs_frozen_vt": int(frame["corrected"].sum()),
            "harmed_vs_frozen_vt": int(frame["harmed"].sum()),
            "preservation_rate_vs_frozen_vt": float(frame["preservation_rate"].mean())}


def refresh_hashes(run: Path) -> None:
    analyzer._FILE_SHA_CACHE.clear()
    write_json(run / "hashes.json", {name: analyzer.sha256_file(run / name)
                                      for name in analyzer.HASHED_ARTIFACTS})
    done = json.loads((run / "DONE.json").read_text(encoding="utf-8"))
    done["hashes_sha256"] = analyzer.sha256_file(run / "hashes.json")
    write_json(run / "DONE.json", done)


def validation_receipt() -> dict:
    return {"pooled_ap": 0.6, "event_macro_ap": 0.6, "event_ap": {"e": 0.6},
            "n_events": 1, "candidate_errors": 4, "epoch0_identity_errors": 5,
            "net_error_reduction_vs_epoch0": 1, "threshold": 0.42,
            "split_usage": "validation-only; test-not-accessed"}


def write_run(root: Path, parents: Path, schema_path: Path, seed: int, fold: int,
              mode: str, q_r: float) -> Path:
    run = root / f"seed{seed}" / f"fold{fold}" / mode
    run.mkdir(parents=True)
    parent_dir = parents / f"seed{seed}" / f"fold{fold}"
    parent_dir.mkdir(parents=True, exist_ok=True)
    visual, terrain = parent_dir / "visual.pt", parent_dir / "terrain.pt"
    if not visual.exists():
        visual.write_bytes(f"visual-{seed}-{fold}".encode())
        terrain.write_bytes(f"terrain-{seed}-{fold}".encode())
    parent = {"schema_version": analyzer.PARENT_SCHEMA, "fold": fold, "seed": seed,
              "threshold": 0.42, "visual": signature(visual), "terrain": signature(terrain),
              "terrain_embedded_seed": seed, "legacy_seed_resolution": None,
              "direction_semantics": "frozen VT correction direction"}
    samples = sample_frame(mode, q_r, fold)
    events, controls, receipts = event_frame(samples), control_frame(samples), receipt_frame(samples)
    samples.to_csv(run / "per_sample.csv", index=False)
    events.to_csv(run / "per_event.csv", index=False)
    controls.to_csv(run / "control_rows.csv", index=False)
    controls.to_csv(run / "same_checkpoint_controls.csv", index=False)
    receipts.to_csv(run / "paired_control_receipts.csv", index=False)
    (run / "command.txt").write_text("synthetic command\n", encoding="utf-8")
    (run / "run.log").write_text("complete\n", encoding="utf-8")
    selected_identity = mode == "trigger" and q_r == 0
    best_epoch = 0 if selected_identity else 1
    identity = {"epoch": 0, "identity_candidate": True, "selection_score": 0.5,
                "selection_score_gain_vs_epoch0": 0.0, "passes_minimum_gain_gate": True,
                "selected_so_far": selected_identity,
                "material_outer_val_receipt": validation_receipt(),
                "trigger_inner_supported_val_receipt": validation_receipt(),
                "selection_contract": "synthetic validation contract"}
    trained = {"epoch": 1, "identity_candidate": False, "selection_score": 0.6,
               "selection_score_gain_vs_epoch0": 0.1, "passes_minimum_gain_gate": True,
               "net_error_gate_passed": True, "selected_so_far": not selected_identity,
               "material_outer_val_receipt": validation_receipt(),
               "trigger_inner_supported_val_receipt": validation_receipt(),
               "selection_contract": "synthetic validation contract"}
    history = [identity, trained]
    gate = {"epoch0_identity_is_candidate": True,
            "minimum_score_gain_vs_epoch0": 0.001,
            "minimum_net_error_reduction_vs_epoch0": 1, "test_used_for_selection": False}
    material_signature = signature(schema_path)
    local_contract = {"test_label_used_for_subgroup_or_threshold": False,
                      "conditional_rows": "aligned and q>0 and nonzero Terrain support",
                      "global_requirement": "no negative transfer versus frozen VT"}
    torch.save({"schema_version": analyzer.CHECKPOINT_SCHEMA, "mode": mode, "seed": seed,
                "fold": fold, "state_dict": {}, "best_epoch": best_epoch,
                "selection": "synthetic validation contract",
                "selected_identity_abstain": selected_identity,
                "epoch0_identity_was_candidate": True, "selection_gate": gate,
                "parent_identity": parent, "material_schema": material_signature,
                "material_feature_names": ["awc", "lithology_entropy"],
                "local_effect_audit_contract": local_contract}, run / "checkpoint.pt")
    write_json(run / "config.json", {"schema_version": analyzer.CONFIG_SCHEMA,
               "mode": mode, "seed": seed, "fold": fold, "selection_gate": gate,
               "contract": {"training_context": "aligned-only"}})
    n = len(SAMPLES)
    test = {"vt": {"tp": VT_COUNTS[0] * n, "fp": VT_COUNTS[1] * n,
                    "fn": VT_COUNTS[2] * n, "tn": VT_COUNTS[3] * n, "iou": 0.5},
            "controls": {control: aggregate_control(samples, control)
                         for control in analyzer.CONTROLS_BY_MODE[mode]}}
    write_json(run / "result.json", {"schema_version": analyzer.RUN_SCHEMA, "status": "complete",
               "mode": mode, "seed": seed, "fold": fold, "best_epoch": best_epoch,
               "selected_identity_abstain": selected_identity, "epoch0_identity_was_candidate": True,
               "selection_gate": gate, "history": history, "test": test,
               "parent_identity": parent, "material_schema": material_signature,
               "material_feature_names": ["awc", "lithology_entropy"],
               "local_effect_audit_contract": local_contract})
    write_json(run / "DONE.json", {"schema_version": analyzer.DONE_SCHEMA, "status": "complete",
               "mode": mode, "seed": seed, "fold": fold, "same_checkpoint_controls": True,
               "selected_identity_abstain": selected_identity, "hashes_sha256": "0" * 64})
    refresh_hashes(run)
    return run


def build_fixture(root: Path, parents: Path) -> Path:
    schema = parents / "material_schema.json"
    parents.mkdir(parents=True, exist_ok=True)
    write_json(schema, {"schema_version": "material.v2", "features": ["awc", "lithology_entropy"]})
    for seed in SEEDS:
        for fold in FOLDS:
            q_r = 1.0 if fold in (0, 2, 4) else 0.0
            for mode in analyzer.MODES:
                write_run(root, parents, schema, seed, fold, mode, q_r)
    return schema


class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        analyzer._FILE_SHA_CACHE.clear()
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.root, self.parents = base / "runs", base / "parents"
        self.root.mkdir()
        self.schema = build_fixture(self.root, self.parents)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_analysis(self) -> tuple[dict, Path]:
        output = Path(self.temporary.name) / "analysis"
        return analyzer.analyze(self.root, output, seeds=SEEDS, folds=FOLDS,
                                min_seeds=1, bootstrap_reps=100, bootstrap_seed=7), output

    def run_path(self, fold: int = 0, mode: str = "trigger") -> Path:
        return self.root / "seed101" / f"fold{fold}" / mode

    def test_complete_matrix_outputs_global_and_conditional_metrics(self) -> None:
        summary, output = self.run_analysis()
        self.assertEqual(summary["n_runs"], 15)
        self.assertTrue(summary["trigger_terrain_support_and_sign_verified"])
        self.assertEqual(len(pd.read_csv(output / "pooled_full_test_metrics.csv")), 3)
        conditional = pd.read_csv(output / "conditional_effect_metrics.csv")
        self.assertFalse(conditional["test_label_or_error_used_for_selection"].any())
        json.loads((output / "summary.json").read_text(encoding="utf-8"),
                   parse_constant=lambda value: self.fail(f"nonfinite {value}"))
        for csv_path in output.glob("*.csv"):
            text = csv_path.read_text(encoding="utf-8").lower()
            self.assertIsNone(re.search(r"(?:^|,)(?:nan|inf|-inf)(?:,|\n)", text))

    def test_missing_fold_is_fatal(self) -> None:
        target = self.root / "seed101" / "fold4" / "joint"
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "incomplete run"):
            self.run_analysis()

    def test_parent_sha_drift_is_fatal(self) -> None:
        result_path = self.run_path() / "result.json"
        result = json.loads(result_path.read_text())
        result["parent_identity"]["terrain"]["sha256"] = "f" * 64
        write_json(result_path, result)
        refresh_hashes(self.run_path())
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "parent terrain checkpoint SHA drift"):
            self.run_analysis()

    def test_material_schema_sha_drift_is_fatal(self) -> None:
        self.schema.write_text("{}\n", encoding="utf-8")
        analyzer._FILE_SHA_CACHE.clear()
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "Material schema SHA drift"):
            self.run_analysis()

    def test_epoch0_identity_receipt_is_required(self) -> None:
        path = self.run_path() / "result.json"
        result = json.loads(path.read_text())
        result["history"][0]["identity_candidate"] = False
        write_json(path, result)
        refresh_hashes(self.run_path())
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "epoch0 identity receipt absent"):
            self.run_analysis()

    def test_test_selection_leakage_is_fatal(self) -> None:
        path = self.run_path() / "config.json"
        config = json.loads(path.read_text())
        config["selection_gate"]["test_used_for_selection"] = True
        write_json(path, config)
        refresh_hashes(self.run_path())
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "selection gate permits test access"):
            self.run_analysis()

    def test_different_checkpoint_control_is_fatal(self) -> None:
        path = self.run_path() / "control_rows.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "checkpoint_selection"] = "different"
        frame.to_csv(path, index=False)
        refresh_hashes(self.run_path())
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "checkpoint mismatch"):
            self.run_analysis()

    def test_q0_fallback_tamper_is_fatal(self) -> None:
        path = self.run_path(fold=1, mode="trigger") / "per_sample.csv"
        frame = pd.read_csv(path)
        row = frame.index[frame["control"] == "trigger_zero_q"][0]
        frame.loc[row, "errors"] += 1
        frame.to_csv(path, index=False)
        refresh_hashes(self.run_path(fold=1, mode="trigger"))
        with self.assertRaises(analyzer.AnalysisContractError):
            self.run_analysis()

    def test_trigger_support_escape_is_fatal(self) -> None:
        path = self.run_path() / "per_sample.csv"
        frame = pd.read_csv(path)
        row = frame.index[frame["control"] == "aligned"][0]
        frame.loc[row, "trigger_terrain_overlap_pixel_count"] -= 1
        frame.loc[row, "trigger_support_overlap_100pct"] = False
        frame.to_csv(path, index=False)
        refresh_hashes(self.run_path())
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "Trigger leaves Terrain support"):
            self.run_analysis()

    def test_trigger_sign_violation_is_fatal(self) -> None:
        path = self.run_path() / "per_sample.csv"
        frame = pd.read_csv(path)
        frame.loc[frame.index[frame["control"] == "aligned"][0],
                  "trigger_signed_direction_violation_count"] = 1
        frame.to_csv(path, index=False)
        refresh_hashes(self.run_path())
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "signed-direction violation"):
            self.run_analysis()

    def test_trigger_shuffle_donor_must_be_outer_train(self) -> None:
        path = self.run_path() / "per_sample.csv"
        frame = pd.read_csv(path)
        frame.loc[frame["control"] == "trigger_event_shuffle",
                  "trigger_event_shuffle_donor_scope"] = "test-events"
        frame.to_csv(path, index=False)
        refresh_hashes(self.run_path())
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "donor escapes outer train"):
            self.run_analysis()

    def test_label_derived_eligibility_is_fatal(self) -> None:
        path = self.run_path(mode="material") / "per_sample.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "local_effect_subset_uses_test_label"] = True
        frame.to_csv(path, index=False)
        refresh_hashes(self.run_path(mode="material"))
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "uses a test label"):
            self.run_analysis()

    def test_eligibility_identity_tamper_is_fatal(self) -> None:
        path = self.run_path(mode="material") / "per_sample.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "material_local_effect_eligible"] = False
        frame.to_csv(path, index=False)
        refresh_hashes(self.run_path(mode="material"))
        with self.assertRaisesRegex(analyzer.AnalysisContractError, "eligibility identity"):
            self.run_analysis()

    def test_trigger_q0_folds_are_abstention_not_efficacy(self) -> None:
        summary, output = self.run_analysis()
        self.assertGreater(summary["n_trigger_q0_fold_abstentions"], 0)
        pairs = pd.read_csv(output / "paired_sample_metrics.csv")
        trigger = pairs[pairs["contrast"].str.startswith("trigger_")]
        self.assertFalse(trigger["fold"].isin((1, 3)).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
