#!/usr/bin/env python3
"""CPU contract tests for the role-aware PILD Material interaction."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pild_roleaware_material as material_module


class MaterialInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20260722)
        self.batch = 4
        self.feature = torch.randn(self.batch, 3, 7, 9, generator=generator)
        self.residual = torch.randn(self.batch, 1, 7, 9, generator=generator)
        self.residual[:, :, 2:4, 3:6] = 0.0
        self.material = torch.randn(
            self.batch,
            material_module.MATERIAL_FEATURE_COUNT,
            generator=generator,
        )
        self.q_m = torch.ones(self.batch)
        self.model = material_module.RoleAwareMaterialInteraction(
            hidden_dim=12, rank=3, modulation_bound=0.25
        )
        with torch.no_grad():
            self.model.interaction_head.response.weight.fill_(2.0)

    def test_multiplier_is_bounded_and_material_cannot_draw_boundaries(self) -> None:
        output, audit = self.model(
            self.feature, self.residual, self.material * 1000.0, self.q_m
        )
        multiplier = audit["material_multiplier_map"]
        self.assertGreaterEqual(float(multiplier.detach().min()), 0.75)
        self.assertLessEqual(float(multiplier.detach().max()), 1.25)
        self.assertTrue(torch.equal(output == 0.0, self.residual == 0.0))
        self.assertTrue(
            torch.equal(torch.sign(output), torch.sign(self.residual))
        )
        self.assertTrue(
            torch.equal(
                audit["material_interaction_map"][self.residual == 0.0],
                torch.zeros_like(self.residual[self.residual == 0.0]),
            )
        )
        self.assertFalse(audit["material_dense_direction"])

    def test_zero_initialized_head_is_exact_terrain_identity(self) -> None:
        model = material_module.RoleAwareMaterialInteraction(hidden_dim=8, rank=2)
        output, audit = model(
            self.feature, self.residual, self.material, torch.ones(self.batch)
        )
        self.assertTrue(torch.equal(output, self.residual))
        self.assertTrue(
            torch.equal(
                audit["material_multiplier_map"],
                torch.ones_like(audit["material_multiplier_map"]),
            )
        )
        self.assertEqual(
            float(audit["material_interaction_map"].detach().abs().max()), 0.0
        )

    def test_zero_q_and_abstain_are_exact_per_sample_terrain_identity(self) -> None:
        q_m = torch.tensor([0.0, 1.0, 1.0, 0.5])
        abstain = torch.tensor([False, True, False, False])
        output, audit = self.model(
            self.feature,
            self.residual,
            self.material,
            q_m,
            abstain=abstain,
        )
        for index in (0, 1):
            self.assertTrue(torch.equal(output[index], self.residual[index]))
            self.assertTrue(
                torch.equal(
                    audit["material_multiplier_map"][index],
                    torch.ones_like(audit["material_multiplier_map"][index]),
                )
            )
            self.assertEqual(
                float(
                    audit["material_interaction_map"][index]
                    .detach()
                    .abs()
                    .max()
                ),
                0.0,
            )
        self.assertFalse(torch.equal(output[2], self.residual[2]))

        zero_q, zero_audit = self.model(
            self.feature,
            self.residual,
            self.material,
            torch.ones_like(q_m),
            context=material_module.CONTEXT_ZERO_Q,
        )
        abstained, abstain_audit = self.model(
            self.feature,
            self.residual,
            self.material,
            torch.ones_like(q_m),
            context=material_module.CONTEXT_ABSTAIN,
        )
        self.assertTrue(torch.equal(zero_q, self.residual))
        self.assertTrue(torch.equal(abstained, self.residual))
        self.assertEqual(float(zero_audit["q_M_effective"].max()), 0.0)
        self.assertEqual(float(abstain_audit["q_M_effective"].max()), 0.0)

    def test_gradients_enter_only_the_interaction_head(self) -> None:
        terrain_parent = nn.Conv2d(3, 4, 1, bias=False)
        terrain_input = torch.randn(2, 3, 5, 6, requires_grad=True)
        parent_feature = terrain_parent(terrain_input)
        parent_residual = parent_feature[:, :1]
        model = material_module.RoleAwareMaterialInteraction(
            {
                "slope_response": (0,),
                "curvature_response": (1,),
                "relief_response": (2, 3),
            },
            hidden_dim=8,
            rank=2,
        )
        with torch.no_grad():
            model.interaction_head.response.weight.fill_(0.5)
        material = torch.randn(
            2, material_module.MATERIAL_FEATURE_COUNT, requires_grad=True
        )
        q_m = torch.ones(2, requires_grad=True)
        output, _ = model(
            parent_feature, parent_residual, material, q_m
        )
        output.square().mean().backward()

        self.assertIsNone(terrain_input.grad)
        self.assertIsNone(terrain_parent.weight.grad)
        self.assertIsNone(material.grad)
        self.assertIsNone(q_m.grad)
        parameters_with_grad = [
            name for name, parameter in model.named_parameters() if parameter.grad is not None
        ]
        self.assertTrue(parameters_with_grad)
        self.assertTrue(
            all(name.startswith("interaction_head.") for name in parameters_with_grad)
        )


class MaterialContextTests(unittest.TestCase):
    def test_shuffle_preserves_recipient_identity_and_stays_cross_event_within_source(self) -> None:
        sample_ids = ("a0", "a1", "a2", "a3", "b0")
        source_ids = ("source_a", "source_a", "source_a", "source_a", "source_b")
        event_ids = ("event_0", "event_0", "event_1", "event_1", "event_2")
        values = torch.arange(
            len(sample_ids) * material_module.MATERIAL_FEATURE_COUNT,
            dtype=torch.float32,
        ).reshape(len(sample_ids), material_module.MATERIAL_FEATURE_COUNT)
        contexts = material_module.build_material_contexts(
            values,
            torch.ones(len(sample_ids)),
            sample_ids,
            source_ids,
            event_ids,
            seed=17,
        )

        for context in contexts.values():
            self.assertEqual(context.sample_ids, sample_ids)
            self.assertEqual(context.source_ids, source_ids)
            self.assertEqual(context.event_ids, event_ids)

        shuffled = contexts[material_module.CONTEXT_SHUFFLED]
        index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
        for position in range(4):
            donor = index[shuffled.donor_sample_ids[position]]
            self.assertEqual(source_ids[position], source_ids[donor])
            self.assertNotEqual(event_ids[position], event_ids[donor])
            self.assertTrue(torch.equal(shuffled.material[position], values[donor]))
        self.assertTrue(bool(shuffled.abstain[4]))
        self.assertEqual(float(shuffled.q_m[4]), 0.0)
        self.assertEqual(shuffled.donor_sample_ids[4], sample_ids[4])

        zero_q = contexts[material_module.CONTEXT_ZERO_Q]
        self.assertEqual(float(zero_q.q_m.abs().max()), 0.0)
        self.assertTrue(torch.equal(zero_q.material, values))


class OuterTrainNormalizerTests(unittest.TestCase):
    def test_fit_ignores_heldout_values_and_blocks_source_only_features(self) -> None:
        generator = np.random.default_rng(20260722)
        values = generator.normal(size=(10, material_module.MATERIAL_FEATURE_COUNT))
        sample_ids = tuple(f"sample_{index}" for index in range(10))
        source_ids = ("a", "a", "a", "a", "b", "b", "b", "b", "held", "held")
        event_ids = ("a0", "a0", "a1", "a1", "b0", "b0", "b1", "b1", "h0", "h1")
        train_ids = sample_ids[:8]
        values[:4, 0] = -10.0
        values[4:8, 0] = 10.0

        first = material_module.OuterTrainMaterialNormalizer.fit(
            values, sample_ids, source_ids, event_ids, train_ids
        )
        changed = values.copy()
        changed[8:] = 1e12
        second = material_module.OuterTrainMaterialNormalizer.fit(
            changed, sample_ids, source_ids, event_ids, train_ids
        )

        np.testing.assert_array_equal(first.impute_mean, second.impute_mean)
        np.testing.assert_array_equal(first.scale, second.scale)
        np.testing.assert_array_equal(first.shortcut_blocked, second.shortcut_blocked)
        self.assertTrue(bool(first.shortcut_blocked[0]))
        transformed = first.transform(values)
        np.testing.assert_array_equal(transformed[:, 0], np.zeros(len(values)))
        self.assertEqual(first.audit()["fit_scope"], "outer-train-only")
        self.assertFalse(first.audit()["source_id_is_model_feature"])


if __name__ == "__main__":
    unittest.main()
