#!/usr/bin/env python3
"""Synthetic tests for the validation-only Sen12 v3 utility-gate preflight."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import build_sen12_proposal_utility_gate_v3_manifests as aggregate
import preflight_sen12_proposal_utility_gate_v3_validation as preflight
import test_build_sen12_proposal_utility_gate_v3_manifests as fixture
import train_sen12_proposal_utility_gate_v3 as gate


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "watch_sen12_proposal_utility_gate_v3_validation_preflight.sh"


class ValidationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tree = fixture.SyntheticTree(self.root)
        aggregate.aggregate(self.tree.args())
        self.output = self.root / "validation_preflight"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, *, dry_run: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            formal_input_root=self.tree.output_root,
            output_root=self.output,
            seed=self.tree.seed,
            alphas=(1e-4,),
            threshold_grid=(0.5,),
            dry_run=dry_run,
        )

    def test_selection_never_calls_target_outer_loader(self) -> None:
        with mock.patch.object(
            gate,
            "load_fold_bundle",
            side_effect=AssertionError("target outer loader must not be called"),
        ) as target_loader:
            result = preflight.run_preflight(self.args())
        target_loader.assert_not_called()
        self.assertFalse(result["target_outer_cache_loaded"])
        self.assertFalse(result["target_outer_labels_loaded"])
        self.assertEqual(result["n_targets"], 5)
        self.assertTrue((self.output / "DONE.json").is_file())
        lines = (self.output / "validation_preflight_targets.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(lines), 5)
        for line in lines:
            item = json.loads(line)
            self.assertFalse(item["target_outer_cache_loaded"])
            self.assertFalse(item["target_outer_labels_loaded"])
            self.assertEqual(set(item["contexts"]), set(preflight.CONTEXTS))
            for context in preflight.CONTEXTS:
                selected = item["contexts"][context]
                self.assertIn("aligned", selected)
                self.assertIn("proposal_only", selected)
                self.assertIn("controls", selected)
                self.assertIn("alpha", selected)
                self.assertIn("rescue_threshold", selected)
                self.assertIn("veto_threshold", selected)
            self.assertIn("running_role_gate", item)
            self.assertFalse(item["running_role_gate"]["outer_test_started"])

    def test_missing_formal_input_fails_closed_without_output(self) -> None:
        (self.tree.output_root / "target_outer2/oof_manifest.json").unlink()
        with self.assertRaisesRegex(preflight.PreflightError, "inventory mismatch"):
            preflight.run_preflight(self.args())
        self.assertFalse(self.output.exists())

    def test_formal_hash_mismatch_fails_closed_without_output(self) -> None:
        path = self.tree.output_root / "target_outer1/oof_manifest.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(preflight.PreflightError, "hash mismatch"):
            preflight.run_preflight(self.args())
        self.assertFalse(self.output.exists())

    def test_dry_run_validates_nested_inputs_without_selection_output(self) -> None:
        with mock.patch.object(gate, "select_context") as select_context:
            result = preflight.run_preflight(self.args(dry_run=True))
        select_context.assert_not_called()
        self.assertEqual(result["status"], "dry_run_validated")
        self.assertFalse(result["target_outer_cache_loaded"])
        self.assertFalse(self.output.exists())

    def test_three_of_five_is_exact_eligibility_boundary(self) -> None:
        def results(n_pass: int, *, shuffled_target: int | None = None):
            output = []
            for target in preflight.TARGETS:
                contexts = {}
                for context in preflight.CONTEXTS:
                    contexts[context] = {
                        "claim_pass": target < n_pass,
                        "label_shuffle_claim_pass": target == shuffled_target,
                    }
                output.append({"target_outer_fold": target, "contexts": contexts})
            return output

        at_boundary = preflight.aggregate_eligibility(results(3))
        self.assertTrue(at_boundary["eligible_for_outer_test"])
        for context in preflight.CONTEXTS:
            self.assertTrue(at_boundary["contexts"][context]["eligible_for_outer_test"])

        below = preflight.aggregate_eligibility(results(2))
        self.assertFalse(below["eligible_for_outer_test"])
        for context in preflight.CONTEXTS:
            self.assertFalse(below["contexts"][context]["eligible_for_outer_test"])

        shuffled = preflight.aggregate_eligibility(results(4, shuffled_target=4))
        self.assertFalse(shuffled["eligible_for_outer_test"])
        for context in preflight.CONTEXTS:
            self.assertFalse(shuffled["contexts"][context]["eligible_for_outer_test"])

        impossible_after_three = preflight.running_role_gate(results(0)[:3])
        self.assertTrue(impossible_after_three["overall_no_go"])
        for context in preflight.CONTEXTS:
            self.assertEqual(
                impossible_after_three["contexts"][context]["maximum_possible_claim_pass"],
                2,
            )
            self.assertEqual(
                impossible_after_three["contexts"][context]["decision_state"], "no_go"
            )

    def test_done_hashes_cover_published_outputs(self) -> None:
        preflight.run_preflight(self.args())
        done = json.loads((self.output / "DONE.json").read_text(encoding="utf-8"))
        self.assertEqual(done["status"], "complete")
        expected = {
            "validation_preflight_targets.jsonl",
            "validation_preflight_summary.json",
            "validation_preflight_summary.md",
            "validation_preflight_summary.csv",
        }
        self.assertEqual(set(done["artifact_sha256"]), expected)
        for relative, digest in done["artifact_sha256"].items():
            self.assertEqual(preflight.sha256_file(self.output / relative), digest)

    def test_runner_dry_run_stops_before_aggregate_and_outer_test(self) -> None:
        formal = self.root / "runner_formal"
        output = self.root / "runner_preflight"
        environment = {
            **os.environ,
            "PYTHON": os.environ.get("PYTHON", "/home/jinlin/miniconda3/envs/dpl/bin/python"),
            "PROPOSER_ROOT": str(self.tree.input_root),
            "PROTOCOL_ROOT": str(self.tree.protocol_root),
            "FORMAL_INPUT_ROOT": str(formal),
            "PREFLIGHT_OUTPUT_ROOT": str(output),
            "MATERIAL_REGISTRY": str(self.tree.material),
            "TRIGGER_REGISTRY": str(self.tree.trigger),
            "SEED": str(self.tree.seed),
            "DRY_RUN": "1",
            "MAX_WAIT_SECONDS": "0",
        }
        completed = subprocess.run(
            [str(RUNNER)],
            cwd=SCRIPT_DIR.parents[1],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("15/15 proposer", completed.stdout)
        self.assertIn("outer_test_started=0", completed.stdout)
        self.assertIn("gate_trainer_started=0", completed.stdout)
        self.assertFalse(formal.exists())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
