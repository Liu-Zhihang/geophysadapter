#!/usr/bin/env python3
"""Contract tests for the role-pure Sen12 Terrain-v2 core."""

from __future__ import annotations

import unittest

import torch

from sen12_terrain_v2 import BoundedTerrainAdapterV2, SupportOnlyMultiScaleTerrainPyramid


class TerrainV2ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260721)
        self.model = BoundedTerrainAdapterV2(terrain_channels=9, visual_channels=16, alpha_max=2.0)
        self.model.eval()
        self.visual_logits = torch.randn(2, 1, 128, 128)
        self.visual_feature = torch.randn(2, 16, 128, 128)
        self.uncertainty = torch.rand(2, 1, 128, 128)
        self.terrain = torch.randn(2, 9, 128, 128)
        self.q_t = torch.ones(2, 1, 128, 128)

    def test_output_and_pyramid_shapes(self) -> None:
        output, diagnostics = self.model(
            self.visual_logits,
            self.visual_feature,
            self.uncertainty,
            self.terrain,
            self.q_t,
        )
        self.assertEqual(tuple(output.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(diagnostics["terrain_fine_feature"].shape[-2:]), (128, 128))
        self.assertEqual(tuple(diagnostics["terrain_meso_feature"].shape[-2:]), (64, 64))
        self.assertEqual(tuple(diagnostics["terrain_macro_feature"].shape[-2:]), (32, 32))

    def test_zero_terrain_exact_visual_fallback(self) -> None:
        output, diagnostics = self.model(
            self.visual_logits,
            self.visual_feature,
            self.uncertainty,
            torch.zeros_like(self.terrain),
            self.q_t,
        )
        self.assertTrue(torch.equal(output, self.visual_logits))
        self.assertEqual(float(diagnostics["correction"].detach().abs().max()), 0.0)

    def test_zero_support_exact_visual_fallback(self) -> None:
        output, diagnostics = self.model(
            self.visual_logits,
            self.visual_feature,
            self.uncertainty,
            self.terrain,
            torch.zeros_like(self.q_t),
        )
        self.assertTrue(torch.equal(output, self.visual_logits))
        self.assertEqual(float(diagnostics["correction"].detach().abs().max()), 0.0)

    def test_zero_uncertainty_exact_visual_fallback(self) -> None:
        output, diagnostics = self.model(
            self.visual_logits,
            self.visual_feature,
            torch.zeros_like(self.uncertainty),
            self.terrain,
            self.q_t,
        )
        self.assertTrue(torch.equal(output, self.visual_logits))
        self.assertEqual(
            float(diagnostics["visual_reliability_gate"].detach().abs().max()), 0.0
        )

    def test_terrain_direction_is_independent_of_visual_inputs(self) -> None:
        _, first = self.model(
            self.visual_logits,
            self.visual_feature,
            self.uncertainty,
            self.terrain,
            self.q_t,
        )
        _, second = self.model(
            self.visual_logits * 7.0,
            self.visual_feature * -3.0,
            1.0 - self.uncertainty,
            self.terrain,
            self.q_t,
        )
        self.assertTrue(torch.equal(first["raw_terrain_direction"], second["raw_terrain_direction"]))
        self.assertTrue(
            torch.equal(first["bounded_terrain_direction"], second["bounded_terrain_direction"])
        )

    def test_direction_is_bounded(self) -> None:
        _, diagnostics = self.model(
            self.visual_logits,
            self.visual_feature,
            self.uncertainty,
            self.terrain * 1000.0,
            self.q_t,
        )
        self.assertLessEqual(
            float(diagnostics["bounded_terrain_direction"].detach().abs().max()), 2.0
        )


class TerrainScaleValidationTest(unittest.TestCase):
    def test_invalid_channel_count_fails(self) -> None:
        with self.assertRaises(ValueError):
            SupportOnlyMultiScaleTerrainPyramid(terrain_channels=8)


if __name__ == "__main__":
    unittest.main()
