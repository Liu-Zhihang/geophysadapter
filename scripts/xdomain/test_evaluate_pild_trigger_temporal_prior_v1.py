#!/usr/bin/env python3
"""CPU contract and end-to-end tests for the Trigger temporal-prior evaluator."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_pild_trigger_temporal_prior_v1 as evaluator


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_registry(path: Path, prefix: str, event_ids: list[str], supported: list[bool]) -> None:
    rows = []
    for event_index, (event_id, support) in enumerate(zip(event_ids, supported)):
        for sample_index in range(2):
            case = 30.0 + 10.0 * event_index
            rows.append({
                "sample_id": f"{prefix}-s{event_index}-{sample_index}",
                "physical_event_id": event_id,
                "q_R": float(support),
                "rain_d7_antecedent_case_mm": case if support else np.nan,
                "rain_d7_wrong_m56_mm": 2.0 if support else np.nan,
                "rain_d7_wrong_m28_mm": 4.0 if support else np.nan,
                "rain_d7_wrong_p28_mm": 6.0 if support else np.nan,
                "rain_d7_wrong_p56_mm": 8.0 if support else np.nan,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_oof_fixture(root: Path, *, unsafe_threshold: bool = False) -> tuple[Path, Path, Path]:
    pild = root / "pild.csv"
    sen12 = root / "sen12.csv"
    pild_events = ["p-event-0", "p-event-1"]
    sen_events = ["s-event-0", "s-event-1"]
    write_registry(pild, "p", pild_events, [True, False])
    write_registry(sen12, "s", sen_events, [True, True])
    entries = []
    folds = [
        ("fold0", "p-event-0", ["p-s0-0", "p-s0-1"]),
        ("fold1", "p-event-1", ["p-s1-0", "p-s1-1"]),
        ("fold2", "s-event-0", ["s-s0-0", "s-s0-1"]),
        ("fold3", "s-event-1", ["s-s1-0", "s-s1-1"]),
    ]
    all_events = {item[1] for item in folds}
    for fold_index, (fold_id, event_id, samples) in enumerate(folds):
        logits = np.full((2, 1, 3, 3), -1.2, np.float32)
        labels = np.zeros_like(logits)
        labels[:, :, :2, :2] = 1.0
        # Positive global evidence benefits these deliberately under-confident
        # held-out predictions; q_R=0 fold remains exact identity.
        logits[:, :, :2, :2] = -0.2 - 0.05 * fold_index
        valid = np.ones_like(logits, dtype=np.uint8)
        prediction = root / f"{fold_id}.npz"
        np.savez_compressed(
            prediction, sample_ids=np.asarray(samples), event_ids=np.asarray([event_id] * 2),
            logits=logits, labels=labels, valid=valid,
        )
        receipt = root / f"{fold_id}.json"
        receipt.write_text(json.dumps({
            "schema_version": evaluator.PREDICTION_RECEIPT_SCHEMA,
            "fold_id": fold_id,
            "prediction_role": "parent_oof",
            "prediction_value_type": "raw_logits",
            "selection_uses_holdout_labels": False,
            "threshold_uses_holdout_labels": unsafe_threshold,
            "threshold_probability": 0.5,
            "checkpoint_sha256": hashlib.sha256(fold_id.encode()).hexdigest(),
            "training_event_ids": sorted(all_events - {event_id}),
            "held_out_event_ids": [event_id],
        }), encoding="utf-8")
        entries.append({
            "fold_id": fold_id,
            "prediction_path": prediction.name,
            "prediction_sha256": digest(prediction),
            "producer_receipt_path": receipt.name,
            "producer_receipt_sha256": digest(receipt),
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": evaluator.MANIFEST_SCHEMA,
        "selection_uses_labels": False,
        "all_available_parent_oof_folds_included": True,
        "entries": entries,
    }), encoding="utf-8")
    return manifest, pild, sen12


class TriggerTemporalPriorUnitTests(unittest.TestCase):
    def test_q_zero_offset_is_exact_identity(self) -> None:
        evidence = {
            "event": evaluator.TriggerEvidence("event", "PILD", 0.0, 20.0, -10.0, 2),
            "donor": evaluator.TriggerEvidence("donor", "PILD", 1.0, 2.0, 0.0, 2),
        }
        model = evaluator.TemporalPriorModel(0.0, 1.0, 0.8, 2.0, 0.6, 3, "test")
        fold = evaluator.OOFFold(
            "fold", np.asarray(["sample"]), np.asarray(["held"]), np.asarray(["event"]),
            np.zeros((1, 1, 1, 1), np.float32), np.zeros((1, 1, 1, 1), np.float32),
            np.ones((1, 1, 1, 1), bool), 0.5, "a" * 64, Path("x"), "b" * 64,
            Path("y"), "c" * 64,
        )
        registry = pd.DataFrame({"sample_id": ["sample"], "q_R": [0.0]}).set_index("sample_id", drop=False)
        for condition in evaluator.CONDITIONS:
            offset, _ = evaluator.offsets_for_fold(
                fold, registry, evidence, model, condition, maximum=1.0, seed=3
            )
            self.assertTrue(np.array_equal(offset, np.zeros(1)), condition)

    def test_monotone_model_and_offsets_are_bounded(self) -> None:
        evidence = {
            f"e{i}": evaluator.TriggerEvidence(f"e{i}", "PILD", 1.0, float(i), 0.0, 1)
            for i in range(4)
        }
        model = evaluator.fit_temporal_prior(
            {f"e{i}": 0.2 * i for i in range(4)}, evidence,
            maximum=0.5, ridge_alpha=0.0, min_events=3,
        )
        predicted = model.predict(np.asarray([-100.0, 100.0]), 0.5)
        self.assertTrue(np.all(predicted >= 0.0))
        self.assertTrue(np.all(predicted <= 0.5))
        self.assertGreaterEqual(model.coefficient, 0.0)

    def test_event_feature_has_a_real_shifted_time_control(self) -> None:
        aligned, shifted = evaluator._event_feature(100.0, [1.0, 2.0, 3.0, 4.0])
        self.assertGreater(aligned, shifted)
        self.assertTrue(np.isfinite(shifted))

    def test_deterministic_shuffle_never_self_donates(self) -> None:
        for event in ("a", "b", "c"):
            self.assertNotEqual(event, evaluator.deterministic_donor(event, ["a", "b", "c"], 9))


class TriggerTemporalPriorIntegrationTests(unittest.TestCase):
    def test_missing_real_oof_fails_after_writing_coverage_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pild = root / "pild.csv"
            sen12 = root / "sen12.csv"
            write_registry(pild, "p", ["p-event"], [True])
            write_registry(sen12, "s", ["s-event"], [False])
            out = root / "out"
            status = evaluator.main([
                "--oof-manifest", str(root / "missing.json"),
                "--pild-trigger-registry", str(pild),
                "--sen12-trigger-registry", str(sen12),
                "--external-evidence", str(root / "missing-external.json"),
                "--sen12-prior-evaluation", str(root / "missing-prior.json"),
                "--outdir", str(out), "--bootstrap-replicates", "100",
                "--permutation-replicates", "100",
            ])
            self.assertEqual(status, 2)
            self.assertTrue((out / "coverage_audit.json").is_file())
            self.assertTrue((out / "required_parent_oof_contract.json").is_file())
            blocked = json.loads((out / "BLOCKED.json").read_text())
            self.assertFalse(blocked["fabricated_results_created"])
            self.assertFalse((out / "summary.json").exists())
            self.assertFalse((out / "DONE.json").exists())

    def test_unsafe_holdout_threshold_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, pild, sen12 = write_oof_fixture(root, unsafe_threshold=True)
            registry = evaluator.load_trigger_registries(pild, sen12)
            with self.assertRaisesRegex(evaluator.ContractError, "threshold_uses_holdout_labels"):
                evaluator.load_oof_folds(manifest, registry, min_folds=3)

    def test_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, pild, sen12 = write_oof_fixture(root)
            payload = json.loads(manifest.read_text())
            payload["entries"][0]["prediction_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload))
            registry = evaluator.load_trigger_registries(pild, sen12)
            with self.assertRaisesRegex(evaluator.ContractError, "hash mismatch"):
                evaluator.load_oof_folds(manifest, registry, min_folds=3)

    def test_complete_real_oof_fixture_emits_standard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, pild, sen12 = write_oof_fixture(root)
            out = root / "out"
            status = evaluator.main([
                "--oof-manifest", str(manifest),
                "--pild-trigger-registry", str(pild),
                "--sen12-trigger-registry", str(sen12),
                "--external-evidence", str(root / "none.json"),
                "--sen12-prior-evaluation", str(root / "none-prior.json"),
                "--outdir", str(out), "--min-fit-supported-events", "2",
                "--bootstrap-replicates", "200", "--permutation-replicates", "200",
            ])
            self.assertEqual(status, 0)
            for name in (
                "sample_metrics.csv", "event_metrics.csv", "fold_metrics.csv",
                "calibration_receipts.csv", "summary.json", "report.md", "DONE.json",
                "coverage_audit.json", "coverage_event_audit.csv",
                "required_parent_oof_contract.json",
            ):
                self.assertTrue((out / name).is_file(), name)
            samples = pd.read_csv(out / "sample_metrics.csv")
            unsupported = samples[samples.q_R == 0]
            self.assertTrue(np.array_equal(unsupported.offset.to_numpy(), np.zeros(len(unsupported))))
            self.assertEqual(set(samples.condition), set(evaluator.CONDITIONS))
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["guardrails"]["trigger_draws_pixel_boundaries"])
            self.assertFalse(summary["guardrails"]["target_fold_labels_used_for_calibration"])


if __name__ == "__main__":
    unittest.main()
