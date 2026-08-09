#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_sen12_trigger_awc_dose_v1 import (
    deterministic_event_donors,
    dose_from_log,
    fit_dose_model,
    _finite_median,
    EventEvidence,
)


class TriggerDoseContractTests(unittest.TestCase):
    def test_zero_trigger_quality_is_exact_identity(self) -> None:
        dose = dose_from_log(np.asarray([-10.0, 10.0]), np.zeros(2), 0.8, 1.25)
        self.assertTrue(np.array_equal(dose, np.ones(2)))

    def test_positive_dose_is_bounded(self) -> None:
        dose = dose_from_log(
            np.asarray([-100.0, 100.0, 0.0]), np.ones(3), 0.8, 1.25
        )
        self.assertTrue(np.all(dose >= 0.8))
        self.assertTrue(np.all(dose <= 1.25))
        self.assertEqual(float(dose[2]), 1.0)

    def test_event_shuffle_never_uses_same_event(self) -> None:
        donors = deterministic_event_donors(["a", "b"], ["a", "b", "c"], 7)
        self.assertNotEqual(donors["a"], "a")
        self.assertNotEqual(donors["b"], "b")

    def test_spatial_rainfall_is_aggregated_to_one_event_value(self) -> None:
        import pandas as pd

        self.assertEqual(_finite_median(pd.Series([1.0, 7.0, 3.0])), 3.0)

    def test_unsupported_events_do_not_fit_dose_model(self) -> None:
        evidence = {
            "a": EventEvidence("a", 0.0, 3.0, 2.0, 1.0, 1, None),
            "b": EventEvidence("b", 0.0, 4.0, 3.0, 2.0, 1, None),
        }
        model = fit_dose_model(evidence, {"a": 0.1, "b": -0.1}, ridge_alpha=10, min_events=2)
        self.assertEqual(model.status, "no_supported_fit_events")
        self.assertEqual(model.global_log_multiplier, 0.0)

    def test_ridge_is_fit_only_from_supported_events(self) -> None:
        evidence = {
            key: EventEvidence(key, quality, value, value - 1, value + 1, 1, None)
            for key, quality, value in (("a", 1.0, 0.0), ("b", 1.0, 1.0), ("c", 1.0, 2.0), ("x", 0.0, 100.0))
        }
        model = fit_dose_model(
            evidence, {"a": -0.1, "b": 0.0, "c": 0.1, "x": 5.0},
            ridge_alpha=0.0, min_events=3,
        )
        self.assertEqual(model.status, "ridge")
        self.assertEqual(model.n_fit_events, 3)
        self.assertGreater(float(model.predict(np.asarray([2.0]))[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
