#!/usr/bin/env python3

import unittest

import numpy as np

from trigger_likelihood_oof import centered_log_odds, match_known_event, score_event


def record(fold, beta, held=(), train=()):
    return {
        "oof_fold": fold,
        "heldout_event_ids": list(held),
        "train_event_ids": list(train),
        "model": {"beta": beta, "mean": 0.0, "std": 1.0},
    }


class TriggerLikelihoodTests(unittest.TestCase):
    def test_known_event_uses_only_heldout_model(self) -> None:
        models = [
            record(0, 1.0, held=("event",)),
            record(1, -10.0, train=("event",)),
        ]
        score = score_event("event", np.asarray([20, 1, 1, 1, 1]), models)
        self.assertEqual(score.scoring_mode, "storm_cluster_oof")
        self.assertEqual(score.model_folds, (0,))
        self.assertGreater(score.aligned_log_bf, 0)

    def test_external_event_uses_fold_ensemble(self) -> None:
        models = [record(0, 1.0), record(1, 1.0)]
        score = score_event("external", np.asarray([20, 1, 1, 1, 1]), models)
        self.assertEqual(score.scoring_mode, "external_five_fold_ensemble")
        self.assertEqual(score.model_folds, (0, 1))

    def test_zero_beta_is_chance_and_zero_log_bf(self) -> None:
        score = score_event(
            "external",
            np.asarray([20, 1, 2, 3, 4]),
            [record(0, 0.0)],
        )
        self.assertAlmostEqual(score.aligned_probability, 0.2)
        self.assertAlmostEqual(score.aligned_log_bf, 0.0)
        self.assertAlmostEqual(centered_log_odds(0.2), 0.0)

    def test_wrong_time_is_fixed_minus_28_anchor(self) -> None:
        score = score_event(
            "external",
            np.asarray([1, 2, 50, 3, 4]),
            [record(0, 1.0)],
        )
        self.assertGreater(score.wrong_time_probability, score.aligned_probability)

    def test_conservative_cross_registry_alias(self) -> None:
        known = [{
            "physical_event_id": "known",
            "canonical_date": "2018-08-16",
            "center_lon": 76.60,
            "center_lat": 10.40,
        }]
        alias = match_known_event("2018-08-07", 76.64, 10.39, known)
        self.assertIsNotNone(alias)
        self.assertEqual(alias.physical_event_id, "known")
        self.assertIsNone(match_known_event("2018-08-07", 80.0, 10.39, known))


if __name__ == "__main__":
    unittest.main()
