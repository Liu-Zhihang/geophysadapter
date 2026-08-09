#!/usr/bin/env python3
"""Unit tests for the PILD event-first Trigger registry contract."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

import build_pild_prithvi_trigger_registry_v1 as registry


class TriggerRegistryContractTest(unittest.TestCase):
    def test_strict_window_excludes_d0(self) -> None:
        anchor = date(2022, 4, 10)
        self.assertEqual(
            registry.strict_window_dates(anchor, 0),
            tuple(date(2022, 4, day) for day in range(9, 2, -1)),
        )
        self.assertEqual(registry.strict_window_dates(anchor, -56)[0], date(2022, 2, 12))
        self.assertEqual(registry.strict_window_dates(anchor, 56)[-1], date(2022, 5, 29))

    def test_q_r_gate_fails_closed_in_required_order(self) -> None:
        cases = [
            ((False, 1, True, True, True), "event_date_invalid"),
            ((True, 2, False, True, True), "event_date_not_unique"),
            ((True, 1, False, True, True), "event_date_canonical_mismatch"),
            ((True, 1, True, False, True), "mechanism_not_rainfall"),
            ((True, 1, True, True, False), "incomplete_chirps_coverage"),
            ((True, 1, True, True, True), "rainfall_strict_d7_complete"),
        ]
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(registry.q_r_reason(*arguments), expected)

    def test_event_values_are_broadcast_unchanged(self) -> None:
        readiness = pd.DataFrame(
            {
                "sample_id": ["sample_b", "sample_a"],
                "event_uid": ["event", "event"],
                "physical_event_id": ["physical", "physical"],
                "dataset_id": ["dataset", "dataset"],
                "event_date": ["2022-04-10", "2022-04-10"],
                "event_date_valid": [1, 1],
            }
        )
        event = {
            "physical_event_id": "physical",
            "registry_build_id": "build",
            "event_record_sha256": "event-hash",
            "canonical_event_date": "2022-04-10",
            "readiness_event_dates": "2022-04-10",
            "n_unique_readiness_dates": 1,
            "date_unique_and_canonical": 1,
            "physical_trigger_family": "hydrometeorological",
            "rainfall_mechanism": 1,
            "event_center_lon": 125.0,
            "event_center_lat": 10.0,
            "rain_d7_case_mm": 100.0,
            "rain_d7_wrong_m56_mm": 10.0,
            "rain_d7_wrong_m28_mm": 20.0,
            "rain_d7_wrong_p28_mm": 30.0,
            "rain_d7_wrong_p56_mm": 40.0,
            "rain_d7_wrongtime_median_mm": 25.0,
            "rain_d7_case_minus_wrongtime_mm": 75.0,
            "case_days_available": 7,
            "wrong_m56_days_available": 7,
            "wrong_m28_days_available": 7,
            "wrong_p28_days_available": 7,
            "wrong_p56_days_available": 7,
            "available_days_total": 35,
            "required_days_total": 35,
            "chirps_coverage_complete": 1,
            "q_R": 1,
            "q_R_reason": "rainfall_strict_d7_complete",
        }
        sample = registry.build_sample_frame(readiness, pd.DataFrame([event]))
        self.assertEqual(sample["sample_id"].tolist(), ["sample_a", "sample_b"])
        for column in registry.EVENT_BROADCAST_COLUMNS:
            self.assertTrue(
                all(registry.scalar_equal(value, event[column]) for value in sample[column]),
                column,
            )

    def test_incomplete_role_never_produces_partial_total(self) -> None:
        event = pd.DataFrame(
            [
                {
                    "physical_event_id": "physical",
                    "canonical_event_date": "2022-04-10",
                    "n_unique_readiness_dates": 1,
                    "all_readiness_dates_valid": 1,
                    "date_unique_and_canonical": 1,
                    "rainfall_mechanism": 1,
                    **{f"rain_d7_{role}_mm": np.nan for role in registry.ANCHOR_SHIFTS},
                    **{f"{role}_days_available": 0 for role in registry.ANCHOR_SHIFTS},
                    "rain_d7_wrongtime_median_mm": np.nan,
                    "rain_d7_case_minus_wrongtime_mm": np.nan,
                    "available_days_total": 0,
                    "required_days_total": 35,
                    "chirps_coverage_complete": 0,
                    "q_R": 0,
                    "q_R_reason": "incomplete_chirps_coverage",
                }
            ]
        )
        rows = []
        for role, shift in registry.ANCHOR_SHIFTS.items():
            for lag in registry.STRICT_LAGS:
                rows.append(
                    {
                        "physical_event_id": "physical",
                        "anchor_role": role,
                        "antecedent_lag_days": lag,
                        "status": "missing_file" if role == "wrong_p56" and lag == 7 else "valid",
                        "rainfall_mm": np.nan if role == "wrong_p56" and lag == 7 else 1.0,
                    }
                )
        output = registry.apply_event_rainfall(event, pd.DataFrame(rows))
        self.assertEqual(int(output.loc[0, "q_R"]), 0)
        self.assertEqual(output.loc[0, "q_R_reason"], "incomplete_chirps_coverage")
        self.assertTrue(pd.isna(output.loc[0, "rain_d7_wrong_p56_mm"]))
        self.assertTrue(pd.isna(output.loc[0, "rain_d7_wrongtime_median_mm"]))


if __name__ == "__main__":
    unittest.main()
