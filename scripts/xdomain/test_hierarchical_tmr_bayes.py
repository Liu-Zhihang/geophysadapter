#!/usr/bin/env python3

import math
import unittest

import torch

from hierarchical_tmr_bayes import (
    BoundedBayesianTMRFusion,
    PositiveMaterialDose,
    TerrainMaterialSusceptibility,
    spatially_center,
)


class HierarchicalTMRContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_spatial_center_is_label_free_zero_mean(self) -> None:
        value = torch.randn(3, 1, 8, 8)
        centered = spatially_center(value)
        self.assertTrue(torch.allclose(centered.mean((-2, -1)), torch.zeros(3, 1), atol=1e-6))

    def test_material_zero_quality_is_exact_terrain_parent(self) -> None:
        model = TerrainMaterialSusceptibility(9, 5, basis_count=4)
        terrain = torch.randn(2, 9, 32, 32)
        material = torch.randn(2, 5)
        output, audit = model(terrain, material, torch.zeros(2))
        self.assertTrue(torch.equal(output, audit["terrain_logit"]))
        self.assertEqual(float(audit["material_delta"].detach().abs().max()), 0.0)

    def test_material_initialization_is_exact_parent(self) -> None:
        model = TerrainMaterialSusceptibility(9, 5, basis_count=4)
        terrain = torch.randn(2, 9, 32, 32)
        material = torch.randn(2, 5)
        output, audit = model(terrain, material, torch.ones(2))
        self.assertTrue(torch.equal(output, audit["terrain_logit"]))

    def test_missing_terrain_is_exact_zero_susceptibility(self) -> None:
        model = TerrainMaterialSusceptibility(9, 5, basis_count=4)
        terrain = torch.randn(2, 9, 32, 32)
        material = torch.randn(2, 5)
        output, audit = model(
            terrain, material, torch.ones(2), q_t=torch.zeros(2)
        )
        self.assertEqual(float(output.detach().abs().max()), 0.0)
        self.assertEqual(float(audit["material_delta"].detach().abs().max()), 0.0)

    def test_material_interaction_is_bounded_and_centered(self) -> None:
        model = TerrainMaterialSusceptibility(
            9, 5, basis_count=4, material_logit_bound=0.3
        )
        torch.nn.init.normal_(model.material_coefficients[-1].weight, std=0.2)
        terrain = torch.randn(2, 9, 32, 32)
        material = torch.randn(2, 5)
        _, audit = model(terrain, material, torch.ones(2))
        self.assertLessEqual(
            float(audit["material_delta"].detach().abs().max()), 0.300001
        )

    def test_trigger_cannot_create_a_spatial_direction(self) -> None:
        fusion = BoundedBayesianTMRFusion(alpha=1.0, trigger_scale=0.5)
        visual = torch.zeros(2, 1, 8, 8)
        susceptibility = torch.ones_like(visual) * 2.0
        output, audit = fusion(
            visual,
            susceptibility,
            torch.ones(2),
            torch.tensor([2.0, -2.0]),
            torch.ones(2),
        )
        self.assertTrue(torch.equal(output, visual))
        self.assertEqual(float(audit["correction"].abs().max()), 0.0)

    def test_unsupported_terrain_is_exact_visual_fallback(self) -> None:
        fusion = BoundedBayesianTMRFusion(alpha=1.0)
        visual = torch.randn(2, 1, 8, 8)
        susceptibility = torch.randn_like(visual)
        output, _ = fusion(
            visual,
            susceptibility,
            torch.zeros(2),
            torch.ones(2),
            torch.ones(2),
        )
        self.assertTrue(torch.equal(output, visual))

    def test_correction_is_bounded(self) -> None:
        bound = math.log(4.0)
        fusion = BoundedBayesianTMRFusion(
            alpha=0.75, correction_bound=bound, uncertainty_power=0.0
        )
        visual = torch.zeros(2, 1, 8, 8)
        susceptibility = torch.randn_like(visual) * 100.0
        _, audit = fusion(
            visual,
            susceptibility,
            torch.ones(2),
            torch.ones(2) * 100.0,
            torch.ones(2),
        )
        self.assertLessEqual(
            float(audit["correction"].abs().max()), 0.75 * bound + 1e-6
        )

    def test_positive_material_dose_initialization_is_exact_identity(self) -> None:
        model = PositiveMaterialDose(5)
        multiplier, audit = model(torch.randn(4, 5), torch.ones(4))
        self.assertTrue(torch.equal(multiplier, torch.ones_like(multiplier)))
        self.assertTrue(torch.equal(audit["material_log_multiplier"], torch.zeros(4)))

    def test_positive_material_dose_zero_quality_is_exact_identity(self) -> None:
        model = PositiveMaterialDose(5)
        with torch.no_grad():
            model.network[-1].weight.fill_(2.0)
            model.network[-1].bias.fill_(1.0)
        multiplier, _ = model(torch.randn(4, 5), torch.zeros(4))
        self.assertTrue(torch.equal(multiplier, torch.ones_like(multiplier)))

    def test_positive_material_dose_is_bounded(self) -> None:
        model = PositiveMaterialDose(3, log_multiplier_bound=math.log(2.0))
        with torch.no_grad():
            model.network[-1].weight.fill_(100.0)
            model.network[-1].bias.fill_(100.0)
        multiplier, _ = model(torch.randn(8, 3), torch.ones(8))
        self.assertTrue(bool(torch.all(multiplier > 0.0)))
        self.assertTrue(bool(torch.all(multiplier <= 2.0 + 1e-6)))
        self.assertTrue(bool(torch.all(multiplier >= 0.5 - 1e-6)))


if __name__ == "__main__":
    unittest.main()
