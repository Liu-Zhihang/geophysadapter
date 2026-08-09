#!/usr/bin/env python3
"""Lightweight tests for the PILD T x M susceptibility protocol."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import evaluate_pild_material_susceptibility_interaction_v1 as target


class MaterialSusceptibilityProtocolTest(unittest.TestCase):
    def test_q_m_zero_makes_interactions_exactly_zero(self) -> None:
        terrain = np.asarray([[1.0, -2.0, 3.0], [0.5, 0.25, -0.75]])
        material = np.arange(12, dtype=np.float64).reshape(2, 6)
        result = target.interaction_features(terrain, material, np.zeros(2))
        self.assertTrue(np.array_equal(result, np.zeros_like(result)))

    def test_t_only_does_not_depend_on_material(self) -> None:
        terrain = np.asarray([[1.0, 2.0, 3.0]])
        material_a = np.zeros((1, 6))
        material_b = np.ones((1, 6)) * 99.0
        a = target.condition_matrix("T_ONLY", terrain, material_a, np.ones(1))
        b = target.condition_matrix("T_ONLY", terrain, material_b, np.ones(1))
        self.assertTrue(np.array_equal(a, b))

    def test_test_shuffled_uses_same_interaction_basis(self) -> None:
        terrain = np.asarray([[1.0, 2.0, 3.0]])
        material = np.asarray([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        aligned = target.condition_matrix("TXM_ALIGNED", terrain, material, np.ones(1))
        shuffled = target.condition_matrix(
            "TXM_TEST_SHUFFLED_SAME_MODEL", terrain, material, np.ones(1)
        )
        self.assertTrue(np.array_equal(aligned, shuffled))
        self.assertNotIn("TXM_TEST_SHUFFLED_SAME_MODEL", target.TRAIN_CONDITIONS)

    def test_event_folds_keep_cross_source_event_together(self) -> None:
        frame = pd.DataFrame(
            {
                "canonical_event_id": ["cross", "cross", "p1", "p2", "s1", "s2"],
                "source_id": ["PILD", "Sen12", "PILD", "PILD", "Sen12", "Sen12"],
            }
        )
        assignment = target.assign_event_folds(frame, folds=2, seed=7)
        self.assertEqual(set(assignment), {"cross", "p1", "p2", "s1", "s2"})
        self.assertIn(assignment["cross"], {0, 1})

    def test_donor_map_stays_inside_provided_event_subset(self) -> None:
        frame = pd.DataFrame(
            {
                "manifest_index": [0, 1, 2, 3],
                "sample_id": ["a0", "a1", "b0", "b1"],
                "canonical_event_id": ["event-a", "event-a", "event-b", "event-b"],
                "source_id": ["PILD", "PILD", "PILD", "PILD"],
            }
        )
        mapping = target.donor_manifest_map(frame, seed=11)
        event_by_index = frame.set_index("manifest_index")["canonical_event_id"].to_dict()
        for recipient, donor in mapping.items():
            self.assertIn(donor, set(frame.manifest_index))
            self.assertNotEqual(event_by_index[recipient], event_by_index[donor])

    def test_material_vector_uses_fixed_low_dimensional_fields(self) -> None:
        row = pd.Series(
            {
                "awc_0_200_aligned_mm": 200.0,
                "awc_0_10_aligned_mm": 20.0,
                "awc_10_30_aligned_mm": 30.0,
                "soil_clay_0_5cm_mean_raw": 100.0,
                "soil_sand_0_5cm_mean_raw": 200.0,
                "soil_soc_0_5cm_mean_raw": 30.0,
                "soil_cec_0_5cm_mean_raw": 40.0,
                "q_M_full": 0.75,
                "sample_id": "sample",
            }
        )
        vector, quality = target.material_vector(row)
        self.assertEqual(vector.shape, (6,))
        self.assertAlmostEqual(vector[1], 0.25)
        self.assertAlmostEqual(quality, 0.75)

    def test_fixed_offset_model_is_exact_t_only_when_q_is_zero(self) -> None:
        terrain = np.asarray(
            [[-1.0, 0.5, 2.0], [0.25, -0.75, 1.0], [1.5, 0.0, -0.5], [0.0, 1.0, 0.5]]
        )
        labels = np.asarray([0, 0, 1, 1])
        weights = np.ones(4)
        base = target.fit_model(terrain, labels, weights, c_value=1.0, max_iter=100)
        material = np.ones((4, 6))
        interactions = target.interaction_features(terrain, material, np.zeros(4))
        fitted = target.fit_fixed_offset_interaction(
            base, terrain, interactions, labels, weights, c_value=1.0, max_iter=100
        )
        combined = np.concatenate([terrain, interactions], axis=1)
        self.assertTrue(
            np.array_equal(fitted.predict_proba(combined), base.predict_proba(terrain))
        )

    def test_event_support_map_collapses_cross_source_aliases(self) -> None:
        frame = pd.DataFrame(
            {
                "fold": [3, 3],
                "source_id": ["PILD", "Sen12Landslides"],
                "canonical_event_id": ["shared", "shared"],
                "q_M_full": [0.0, 1.0],
            }
        )
        self.assertEqual(target.event_support_map(frame), {(3, "shared"): 1.0})


if __name__ == "__main__":
    unittest.main()
