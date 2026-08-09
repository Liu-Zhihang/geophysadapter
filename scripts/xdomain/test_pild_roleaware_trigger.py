#!/usr/bin/env python3
"""CPU contract tests for the standalone PILD role-aware Trigger gate."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pild_roleaware_trigger as trigger


class PILDRoleAwareTriggerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260722)
        self.config = trigger.TriggerGateConfig(
            feature_dim=3,
            hidden_dim=5,
            max_gate_budget=0.4,
            max_abs_logit_prior=0.6,
            min_uncertainty_threshold=0.2,
            max_uncertainty_threshold=0.8,
        )
        self.model = trigger.PILDRoleAwareTrigger(self.config).eval()
        with torch.no_grad():
            self.model.calibrator[-1].bias.copy_(torch.tensor([2.0, 1.5, -1.0]))

    @staticmethod
    def features() -> torch.Tensor:
        return torch.tensor(
            [
                [100.0, 20.0, 80.0],
                [100.0, 20.0, 80.0],
                [40.0, 10.0, 30.0],
                [40.0, 10.0, 30.0],
            ]
        )

    @staticmethod
    def events() -> list[str]:
        return ["event-a", "event-a", "event-b", "event-b"]

    def test_q_zero_is_pointwise_exact_baseline_identity(self) -> None:
        baseline = torch.randn(4, 1, 5, 7)
        features = self.features()
        features[2:] = torch.nan
        output = self.model(
            baseline,
            features,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            self.events(),
        )
        self.assertTrue(torch.equal(output["logits"][2:], baseline[2:]))
        self.assertTrue(torch.equal(output["logit_delta"][2:], torch.zeros_like(baseline[2:])))
        self.assertGreater(float(output["logit_delta"][:2].detach().abs().max()), 0.0)

        zero = self.model(
            baseline,
            features,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            self.events(),
            context="zero-q",
        )
        self.assertTrue(torch.equal(zero["logits"], baseline))

    def test_zero_initialized_calibrator_is_exact_identity(self) -> None:
        model = trigger.PILDRoleAwareTrigger(self.config).eval()
        baseline = torch.randn(4, 1, 5, 7)
        output = model(baseline, self.features(), torch.ones(4), self.events())
        self.assertTrue(torch.equal(output["logits"], baseline))
        self.assertTrue(torch.equal(output["logit_delta"], torch.zeros_like(baseline)))

    def test_all_controls_and_wrong_time_replacement(self) -> None:
        baseline = torch.zeros(4, 1, 3, 3)
        features = self.features()
        q_r = torch.ones(4)
        aligned = self.model(baseline, features, q_r, self.events(), context="aligned")
        wrong = self.model(baseline, features, q_r, self.events(), context="wrong-time")
        expected_wrong = features.clone()
        expected_wrong[:, 0] = features[:, 1]
        expected_wrong[:, 2] = 0.0
        self.assertTrue(torch.equal(wrong["context_features"], expected_wrong))

        shuffled = self.model(
            baseline,
            features,
            q_r,
            self.events(),
            context="event-shuffle",
            donor_by_event={"event-a": "event-b", "event-b": "event-a"},
        )
        self.assertTrue(torch.equal(shuffled["context_features"][:2], features[2:4]))
        self.assertTrue(torch.equal(shuffled["context_features"][2:], features[:2]))
        self.assertEqual(aligned["context"], "aligned")
        self.assertEqual(wrong["context"], "wrong-time")
        self.assertEqual(shuffled["context"], "event-shuffle")

    def test_controls_and_logit_delta_are_bounded(self) -> None:
        baseline = torch.linspace(-4, 4, 4 * 6 * 5).reshape(4, 1, 6, 5)
        output = self.model(baseline, self.features(), torch.ones(4), self.events())
        self.assertTrue(bool((output["gate_budget"] >= 0).all()))
        self.assertLessEqual(float(output["gate_budget"].detach().max()), 0.4)
        self.assertLessEqual(float(output["logit_prior"].detach().abs().max()), 0.6)
        self.assertGreaterEqual(
            float(output["uncertainty_threshold"].detach().min()), 0.2
        )
        self.assertLessEqual(
            float(output["uncertainty_threshold"].detach().max()), 0.8
        )
        self.assertLessEqual(float(output["gate"].detach().max()), 0.4)
        self.assertLessEqual(
            float(output["logit_delta"].detach().abs().max()), 0.4 * 0.6
        )

    def test_same_event_controls_are_identical_and_variation_is_rejected(self) -> None:
        baseline = torch.zeros(4, 1, 4, 4)
        output = self.model(baseline, self.features(), torch.ones(4), self.events())
        for key in ("gate_budget", "logit_prior", "uncertainty_threshold"):
            self.assertTrue(torch.equal(output[key][0], output[key][1]), key)
            self.assertTrue(torch.equal(output[key][2], output[key][3]), key)
        varying = self.features()
        varying[1, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "vary within event"):
            self.model(baseline, varying, torch.ones(4), self.events())

    def test_no_spatial_trigger_leakage(self) -> None:
        baseline = torch.randn(4, 1, 4, 5)
        permutation = torch.randperm(20)
        output = self.model(baseline, self.features(), torch.ones(4), self.events())
        permuted_baseline = baseline.flatten(2)[:, :, permutation].reshape_as(baseline)
        permuted = self.model(
            permuted_baseline, self.features(), torch.ones(4), self.events()
        )
        expected = output["logit_delta"].flatten(2)[:, :, permutation].reshape_as(baseline)
        self.assertTrue(torch.equal(permuted["logit_delta"], expected))
        self.assertFalse(output["audit"]["trigger_dense_direction"])

        constant_visual = torch.zeros(4, 1, 4, 5)
        constant = self.model(
            constant_visual, self.features(), torch.ones(4), self.events()
        )["logit_delta"]
        self.assertTrue(
            torch.equal(constant, constant[:, :, :1, :1].expand_as(constant))
        )
        with self.assertRaisesRegex(ValueError, "spatial Trigger inputs are forbidden"):
            trigger.assert_event_level_broadcast(
                self.events(), torch.ones(4, 3, 4, 5), torch.ones(4)
            )

    def test_gradients_enter_only_trigger_calibrator(self) -> None:
        baseline = torch.randn(4, 1, 4, 5, requires_grad=True)
        features = self.features().requires_grad_()
        q_r = torch.ones(4, requires_grad=True)
        output = self.model(baseline, features, q_r, self.events())
        output["logits"].square().mean().backward()
        self.assertIsNone(baseline.grad)
        self.assertIsNone(features.grad)
        self.assertIsNone(q_r.grad)
        parameters_with_grad = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(parameters_with_grad)
        self.assertTrue(
            all(name.startswith("calibrator.") for name in parameters_with_grad)
        )

    def test_audit_receipts_support_required_metrics(self) -> None:
        baseline = torch.zeros(2, 1, 2, 2)
        logits = torch.tensor(
            [[[[2.0, -2.0], [0.0, 1.0]]], [[[-1.0, 1.0], [2.0, -2.0]]]]
        )
        target = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 1.0], [1.0, 0.0]]]]
        )
        audit = trigger.trigger_audit_quantities(
            logits,
            baseline,
            target=target,
            area_threshold=0.5,
            fixed_fpr_threshold=0.75,
        )
        probability = torch.sigmoid(logits)
        expected_brier = (probability - target).square().flatten(1).sum(dim=1)
        self.assertTrue(torch.allclose(audit["brier_sum"], expected_brier))
        self.assertTrue(torch.equal(audit["valid_pixel_count"], torch.tensor([4, 4])))
        self.assertTrue(
            torch.equal(
                audit["fixed_fpr_recall_denominator"],
                audit["fixed_fpr_tp"] + audit["fixed_fpr_fn"],
            )
        )
        self.assertTrue(torch.isfinite(audit["nll_sum"]).all())
        self.assertTrue(torch.isfinite(audit["soft_area_error"]).all())
        self.assertFalse(math.isnan(float(audit["fixed_fpr_threshold"][0])))


if __name__ == "__main__":
    unittest.main()
