#!/usr/bin/env python3
"""Focused CPU tests for build_sen12_material_context_v2.py."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


MODULE_PATH = Path(__file__).with_name("build_sen12_material_context_v2.py")
SPEC = importlib.util.spec_from_file_location("material_v2", MODULE_PATH)
assert SPEC and SPEC.loader
material_v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(material_v2)


class MaterialV2Tests(unittest.TestCase):
    def test_footprint_stats_uses_all_intersecting_native_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.tif"
            data = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=5,
                height=5,
                count=1,
                dtype="float32",
                crs="EPSG:3857",
                transform=from_origin(0, 5, 1, 1),
                nodata=-9999,
            ) as destination:
                destination.write(data, 1)
            with rasterio.open(path) as source:
                mean, std, coverage, valid_count, footprint_count = material_v2.footprint_stats(
                    source, box(1, 1, 4, 4)
                )
            expected = data[1:4, 1:4]
            self.assertAlmostEqual(mean, float(expected.mean()))
            self.assertAlmostEqual(std, float(expected.std(ddof=0)))
            self.assertEqual(valid_count, 9)
            self.assertEqual(footprint_count, 9)
            self.assertEqual(coverage, 1.0)

    def test_footprint_stats_reports_missing_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.tif"
            data = np.ones((4, 4), dtype=np.float32)
            data[1, 1] = -9999
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=1,
                dtype="float32",
                crs="EPSG:3857",
                transform=from_origin(0, 4, 1, 1),
                nodata=-9999,
            ) as destination:
                destination.write(data, 1)
            with rasterio.open(path) as source:
                _, _, coverage, valid_count, footprint_count = material_v2.footprint_stats(
                    source, box(0, 0, 4, 4)
                )
            self.assertEqual(valid_count, 15)
            self.assertEqual(footprint_count, 16)
            self.assertAlmostEqual(coverage, 15 / 16)

    def test_quality_gate_is_fail_closed(self) -> None:
        frame = pd.DataFrame(index=range(2))
        metadata = {}
        for key, _, eligible in material_v2.AWC_SPECS:
            frame[f"{key}_valid_coverage"] = [1.0, 1.0]
            frame[f"{key}_native_cell_count"] = [9, 9]
            frame[f"{key}_std_mm"] = [1.0, 0.0]
            metadata[key] = {"source_valid": True}
        for prop in material_v2.SOIL_UNITS:
            for depth in ("0_5cm", "5_15cm"):
                key = f"soil_{prop}_{depth}"
                unit = material_v2.SOIL_UNITS[prop][1]
                frame[f"{key}_valid_coverage"] = [1.0, 1.0]
                frame[f"{key}_native_cell_count"] = [9, 9]
                frame[f"{key}_std_{unit}"] = [1.0, 1.0]
                metadata[key] = {"source_valid": True}
        frame["limw_valid_coverage"] = [1.0, 1.0]
        frame["limw_polygon_candidate_count"] = [1, 1]
        frame["limw_broad_class_count"] = [1, 1]
        material_v2.add_quality_gates(frame, metadata, lithology_available=True)
        self.assertEqual(frame.loc[0, "q_M"], 1.0)
        self.assertEqual(frame.loc[1, "q_M"], 0.0)
        self.assertIn("AWC", frame.loc[1, "q_M_status"])

    def test_missing_lithology_forces_q_m_zero(self) -> None:
        frame = pd.DataFrame(index=[0])
        metadata = {}
        for key, _, _ in material_v2.AWC_SPECS:
            frame[f"{key}_valid_coverage"] = 1.0
            frame[f"{key}_native_cell_count"] = 9
            frame[f"{key}_std_mm"] = 1.0
            metadata[key] = {"source_valid": True}
        for prop in material_v2.SOIL_UNITS:
            for depth in ("0_5cm", "5_15cm"):
                key = f"soil_{prop}_{depth}"
                unit = material_v2.SOIL_UNITS[prop][1]
                frame[f"{key}_valid_coverage"] = 1.0
                frame[f"{key}_native_cell_count"] = 9
                frame[f"{key}_std_{unit}"] = 1.0
                metadata[key] = {"source_valid": True}
        material_v2.add_quality_gates(frame, metadata, lithology_available=False)
        self.assertEqual(frame.loc[0, "q_M_continuous"], 1.0)
        self.assertEqual(frame.loc[0, "q_M_lithology"], 0.0)
        self.assertEqual(frame.loc[0, "q_M"], 0.0)

    def test_missing_lithology_features_are_not_model_eligible(self) -> None:
        frame = pd.DataFrame({"continuous": [1.0]})
        for code in material_v2.LITHOLOGY_BROAD_CLASSES:
            frame[f"limw_frac_{code}"] = np.nan
        frame["limw_normalized_entropy"] = np.nan
        schema = material_v2.feature_schema(
            frame,
            ["continuous"],
            [
                *(f"limw_frac_{code}" for code in material_v2.LITHOLOGY_BROAD_CLASSES),
                "limw_normalized_entropy",
            ],
            {},
            "missing",
        )
        self.assertEqual(schema["model_eligible_features"], ["continuous"])
        self.assertNotIn("limw_frac_mt", schema["model_eligible_features"])

    def test_schema_states_local_conditional_effect_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "feature": [1.0],
                "soil_x_std_unit": [0.2],
                "soil_x_valid_coverage": [1.0],
                "soil_x_native_cell_count": [25],
                "limw_valid_coverage": [1.0],
                "limw_polygon_candidate_count": [1],
                "limw_broad_class_count": [1],
                "limw_entropy": [0.0],
                "limw_normalized_entropy": [0.0],
                "awc_nonconstant_feature_count": [1],
                "soil_nonconstant_feature_count": [1],
            }
        )
        schema = material_v2.feature_schema(frame, ["feature"], [], {}, None)
        contract = schema["local_conditional_effect_contract"]
        self.assertIn("not evidence", contract["q_M_does_not_mean"])
        self.assertIn("m_M=1", contract["q_M_equals_0"])
        self.assertIn("soil_x_std_unit", contract["pre_registered_applicability_fields"])
        self.assertIn("No sample", contract["forbidden_selection"])

    def test_lithology_uses_area_proportions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lith.gpkg"
            geology = gpd.GeoDataFrame(
                {"Litho": ["mt____", "scpu__"]},
                geometry=[box(0, 0, 1, 2), box(1, 0, 2, 2)],
                crs="ESRI:54012",
            )
            geology.to_file(path, layer="GLiM_export", driver="GPKG")
            footprints = gpd.GeoDataFrame(
                {"sample_id": ["s1"], "region": ["r1"]},
                geometry=[box(0, 0, 2, 2)],
                crs="ESRI:54012",
            ).to_crs("EPSG:4326")
            output, _, model, _ = material_v2.summarize_lithology(footprints, path)
            self.assertAlmostEqual(output.loc[0, "limw_frac_mt"], 0.5, places=6)
            self.assertAlmostEqual(output.loc[0, "limw_frac_sc"], 0.5, places=6)
            self.assertAlmostEqual(output.loc[0, "limw_valid_coverage"], 1.0, places=6)
            self.assertEqual(output.loc[0, "limw_dominant_broad_class"], "mt")
            self.assertIn("limw_normalized_entropy", model)

    def test_variability_audit_detects_constant_feature(self) -> None:
        frame = pd.DataFrame(
            {
                "physical_event_id": ["e1", "e1", "e2", "e2"],
                "source_id": ["s", "s", "s", "s"],
                "variable": [0.0, 1.0, 2.0, 3.0],
                "constant": [1.0, 1.0, 1.0, 1.0],
            }
        )
        audit, summary = material_v2.variability_audit(frame, ["variable", "constant"])
        decisions = audit.set_index("feature")["availability_decision"].to_dict()
        self.assertEqual(decisions["variable"], "MODEL_ELIGIBLE")
        self.assertNotEqual(decisions["constant"], "MODEL_ELIGIBLE")
        self.assertEqual(summary["features_with_nonzero_global_variation"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
