#!/usr/bin/env python3
"""CPU contract tests for the PILD Material registry builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from shapely import make_valid

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_pild_material_registry_v1 as material


class ReadinessContractTest(unittest.TestCase):
    def make_readiness(self) -> pd.DataFrame:
        rows = []
        for index in range(material.EXPECTED_SAMPLES):
            rows.append(
                {
                    "sample_id": f"sample_{index:04d}",
                    "physical_event_id": f"event_{index // 100}",
                    "dataset_id": "source_a",
                    "source_scene_id": f"scene_{index // 10}",
                    "target_crs": "EPSG:32631",
                    "bbox_left": 0.1,
                    "bbox_bottom": 0.1,
                    "bbox_right": 0.2,
                    "bbox_top": 0.2,
                    "target_gsd_m": 10,
                    "footprint_m": 1280,
                    "source_label_path": "/must/not/be/loaded.tif",
                }
            )
        return pd.DataFrame(rows)

    def test_readiness_loads_only_identity_and_geography_whitelist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readiness.csv"
            self.make_readiness().to_csv(path, index=False)
            frame = material.load_readiness(path)
            self.assertEqual(len(frame), material.EXPECTED_SAMPLES)
            self.assertNotIn("source_label_path", frame.columns)
            self.assertEqual(frame.sample_id.nunique(), material.EXPECTED_SAMPLES)

    def test_readiness_rejects_duplicate_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readiness.csv"
            frame = self.make_readiness()
            frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "sample_id"):
                material.load_readiness(path)

    def test_prohibited_source_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction_cache.csv"
            path.write_text("x\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prohibited"):
                material.require_safe_file(path)


class NativeRasterContractTest(unittest.TestCase):
    def test_footprint_stats_preserve_native_cells_and_nodata_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "soil.tif"
            array = np.asarray(
                [
                    [1, 2, 3, 4],
                    [5, 6, -9999, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                dtype=np.int16,
            )
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=1,
                dtype=array.dtype,
                crs="EPSG:4326",
                transform=from_origin(0, 4, 1, 1),
                nodata=-9999,
            ) as target:
                target.write(array, 1)
            frame = pd.DataFrame(
                [
                    {
                        "bbox_left": 1.0,
                        "bbox_bottom": 2.0,
                        "bbox_right": 3.0,
                        "bbox_top": 4.0,
                        "center_lon": 2.0,
                        "center_lat": 3.0,
                    }
                ]
            )
            stats, metadata = material.raster_footprint_stats(frame, path, "soil_test")
            self.assertEqual(int(stats.loc[0, "soil_test_candidate_native_cell_count"]), 4)
            self.assertEqual(int(stats.loc[0, "soil_test_valid_native_cell_count"]), 3)
            self.assertAlmostEqual(float(stats.loc[0, "soil_test_valid_fraction"]), 0.75)
            self.assertGreater(float(stats.loc[0, "soil_test_native_cell_std_raw"]), 0.0)
            self.assertEqual(metadata["native_resolution"], [1.0, 1.0])

    def test_invalid_source_polygon_can_be_repaired_without_dropping_area(self) -> None:
        bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        self.assertFalse(bowtie.is_valid)
        repaired = make_valid(bowtie)
        self.assertTrue(repaired.is_valid)
        self.assertGreater(repaired.area, 0.0)


class QualityContractTest(unittest.TestCase):
    @staticmethod
    def contract_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        sample_ids = [f"sample_{index:04d}" for index in range(material.EXPECTED_SAMPLES)]
        readiness = pd.DataFrame({"sample_id": sample_ids})
        frame = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "q_M_awc": 0.8,
                "q_M_soilgrids": 0.6,
                "q_M_geology": 0.4,
                "q_M_hydraulic": 0.6,
                "q_M": 0.6,
                "q_M_full": 0.4,
                "awc_native_resolution_x_degrees": 0.002083333,
                "soilgrids_native_resolution_x_m": 250.0,
                "lithology_native_scale": "GLiM_polygon_map_approximately_1_to_1M",
                "material_scientific_role": "context_moderator_only",
            }
        )
        return readiness, frame

    def test_q_m_formula_and_role_are_exact(self) -> None:
        frame = pd.DataFrame(
            {
                "q_M_awc": [1.0, 0.2, 0.0],
                "q_M_soilgrids": [0.8, 0.9, 0.0],
                "q_M_geology": [0.4, 0.7, 0.0],
            }
        )
        material.apply_material_quality(frame)
        np.testing.assert_allclose(frame.q_M_hydraulic, [0.8, 0.2, 0.0])
        np.testing.assert_allclose(frame.q_M, [0.8, 0.7, 0.0])
        np.testing.assert_allclose(frame.q_M_full, [0.4, 0.2, 0.0])
        self.assertTrue(frame.material_scientific_role.eq("context_moderator_only").all())
        self.assertTrue(frame.material_multiplier_neutral.eq(1.0).all())

    def test_strict_validator_rejects_sample_order_drift(self) -> None:
        readiness, frame = self.contract_frames()
        frame.loc[[0, 1], "sample_id"] = frame.loc[[1, 0], "sample_id"].to_numpy()
        with self.assertRaisesRegex(RuntimeError, "order/set"):
            material.validate_frame_contract(readiness, frame)

    def test_strict_validator_rejects_q_m_tampering(self) -> None:
        readiness, frame = self.contract_frames()
        frame.loc[0, "q_M"] = 0.9
        with self.assertRaisesRegex(RuntimeError, "q_M formula"):
            material.validate_frame_contract(readiness, frame)


if __name__ == "__main__":
    unittest.main()
