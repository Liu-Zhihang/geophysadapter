#!/usr/bin/env python3
"""CPU unit tests for the PILD + Sen12 registry, split, loader, and sampler."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_pild_sen12_training_registry_v2 as registry
from pild_sen12_training_loader_v2 import (
    MATERIAL_FEATURE_NAMES,
    ROLE_MATERIAL_FEATURE_NAMES,
    SourceEventPatchBalancedSampler,
    TRIGGER_FEATURE_NAMES,
    UnifiedPILDSen12Dataset,
    harmonize_material,
    harmonize_role_material,
    harmonize_trigger,
)


class IdentityAndSplitTests(unittest.TestCase):
    def test_57_raw_to_56_canonical_requires_only_one_auto_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(42):
                rows.append(
                    {
                        "source_collection": registry.PILD_SOURCE,
                        "source_event_id": f"p{index}",
                        "alias_decision": "distinct",
                        "canonical_physical_event_id": f"c_p{index}",
                        "split_group_id": f"c_p{index}",
                    }
                )
            for index in range(15):
                rows.append(
                    {
                        "source_collection": registry.SEN12_SOURCE,
                        "source_event_id": f"s{index}",
                        "alias_decision": "distinct",
                        "canonical_physical_event_id": f"c_s{index}",
                        "split_group_id": f"c_s{index}",
                    }
                )
            rows[0]["alias_decision"] = "auto-match"
            rows[42]["alias_decision"] = "auto-match"
            rows[42]["canonical_physical_event_id"] = rows[0]["canonical_physical_event_id"]
            rows[42]["split_group_id"] = rows[0]["split_group_id"]
            aliases = root / "aliases.csv"
            pd.DataFrame(rows).to_csv(aliases, index=False)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "source_event_rows": 57,
                        "canonical_events_after_auto_deduplication": 56,
                        "automatic_alias_pairs": 1,
                        "coverage_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            _, mapping = registry.validate_aliases(aliases, summary)
            self.assertEqual(len(mapping), 57)
            self.assertEqual(len(set(mapping.values())), 56)

            broken = pd.read_csv(aliases)
            broken.loc[1, "canonical_physical_event_id"] = broken.loc[2, "canonical_physical_event_id"]
            broken.loc[1, "split_group_id"] = broken.loc[2, "split_group_id"]
            broken.to_csv(aliases, index=False)
            with self.assertRaisesRegex(ValueError, "canonical events|two-source canonical alias"):
                registry.validate_aliases(aliases, summary)

    @staticmethod
    def split_manifest() -> pd.DataFrame:
        rows = []
        specification = {
            "dataset_a": ("PILD", ["e0", "e1", "e2", "e6"]),
            "dataset_b": ("Sen12Landslides", ["e0", "e3", "e4", "e5", "e7"]),
        }
        for dataset, (source, events) in specification.items():
            for event in events:
                for patch in range(2):
                    rows.append(
                        {
                            "sample_id": f"{dataset}_{event}_{patch}",
                            "dataset_id": dataset,
                            "source_id": source,
                            "canonical_event_id": event,
                        }
                    )
        return pd.DataFrame(rows)

    def test_event_isolated_and_lodo_never_leak_canonical_events(self) -> None:
        manifest = self.split_manifest()
        event_split = registry.build_event_isolated_split(manifest, 17, 0.2, 0.25)
        registry.validate_split(event_split, "test event split")
        self.assertTrue({"train", "val", "test"}.issubset(set(event_split["role"])))

        lodo = registry.build_lodo_split(manifest, 17, 0.25)
        for fold_id, fold in lodo.groupby("fold_id"):
            registry.validate_split(fold, fold_id)
            heldout = fold["heldout_dataset_id"].iloc[0]
            self.assertTrue(fold.loc[fold["dataset_id"].eq(heldout), "role"].eq("test").all())
            active = fold[fold["role"].isin(["train", "val", "test"])]
            self.assertTrue((active.groupby("canonical_event_id")["role"].nunique() == 1).all())


class SamplerAndLoaderTests(unittest.TestCase):
    def test_source_event_patch_hierarchy_is_balanced(self) -> None:
        rows = []
        for source, event_sizes in {"a": {"a0": 1, "a1": 7}, "b": {"b0": 2, "b1": 3, "b2": 11}}.items():
            for event, size in event_sizes.items():
                for patch in range(size):
                    rows.append(
                        {"source_id": source, "canonical_event_id": event, "sample_id": f"{event}_{patch}"}
                    )
        frame = pd.DataFrame(rows)
        sampler = SourceEventPatchBalancedSampler(frame, num_samples=600, seed=9)
        indices = list(sampler)
        source_counts = Counter(frame.loc[index, "source_id"] for index in indices)
        self.assertLessEqual(max(source_counts.values()) - min(source_counts.values()), 1)

        event_counts: dict[str, Counter] = defaultdict(Counter)
        patch_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for index in indices:
            row = frame.loc[index]
            event_counts[row.source_id][row.canonical_event_id] += 1
            patch_counts[(row.source_id, row.canonical_event_id)][row.sample_id] += 1
        for counts in event_counts.values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for counts in patch_counts.values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

        self.assertEqual(indices, list(SourceEventPatchBalancedSampler(frame, 600, seed=9)))
        sampler.set_epoch(1)
        self.assertNotEqual(indices, list(sampler))

    def test_loader_refuses_incomplete_protocol_and_audit_mode_cannot_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            row = {column: "" for column in registry.MANIFEST_COLUMNS}
            row.update(
                {
                    "dataset_id": "d",
                    "source_id": "s",
                    "source_event_id": "raw",
                    "canonical_event_id": "canonical",
                    "sample_id": "sample",
                    "base_h5_index": 0,
                    "optical_h5_index": 0,
                    "terrain_h5_index": 0,
                    "terrain_channel_indices": "0",
                    "material_registry_index": -1,
                    "trigger_registry_index": -1,
                    "core_assets_ready": 0,
                    "full_tmr_assets_ready": 0,
                }
            )
            pd.DataFrame([row]).to_csv(manifest, index=False)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "validation_status": "PASS",
                        "readiness": {
                            "core_training_ready": False,
                            "full_tmr_training_ready": False,
                            "blockers": ["cache incomplete"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "loader refused to open"):
                UnifiedPILDSen12Dataset(manifest, summary)
            dataset = UnifiedPILDSen12Dataset(
                manifest, summary, allow_incomplete=True, verify_manifest_hash=False
            )
            with self.assertRaisesRegex(RuntimeError, "audit-only"):
                _ = dataset[0]

    def test_sen12_native_channel_projection_matches_common_schema(self) -> None:
        selected = tuple(
            registry.SEN12_NATIVE_TERRAIN_NAMES[index]
            for index in registry.SEN12_COMMON_TERRAIN_INDICES
        )
        self.assertEqual(selected, registry.COMMON_TERRAIN_NAMES)

    def test_material_schema_harmonizes_sen12_and_pild_variation_names(self) -> None:
        common = {
            "soil_clay_0_5cm_mean_raw": 10.0,
            "q_M_full": 1.0,
            "awc_0_10_aligned_mm": 12.0,
        }
        sen12, sen12_q = harmonize_material(
            {**common, "soil_clay_0_5cm_local_std_raw": 2.0}
        )
        pild, pild_q = harmonize_material(
            {**common, "soil_clay_0_5cm_native_cell_std_raw": 2.0}
        )
        self.assertEqual(len(sen12), len(MATERIAL_FEATURE_NAMES))
        self.assertEqual(sen12.tolist(), pild.tolist())
        self.assertEqual((sen12_q, pild_q), (1.0, 1.0))
        missing, missing_q = harmonize_material(None)
        self.assertEqual(float(missing.sum()), 0.0)
        self.assertEqual(missing_q, 0.0)

    def test_role_material_schema_is_fixed_and_uses_explicit_awc_alias(self) -> None:
        common = {
            "soil_clay_0_5cm_mean_raw": 10.0,
            "q_M_full": 1.0,
        }
        sen12, sen12_q = harmonize_role_material(
            {**common, "awc_0_10_aligned_mm": 12.0}
        )
        pild, pild_q = harmonize_role_material(
            {**common, "awc_0_10_footprint_mean_mm": 12.0}
        )
        self.assertEqual(len(ROLE_MATERIAL_FEATURE_NAMES), 21)
        self.assertEqual(sen12.tolist(), pild.tolist())
        self.assertEqual((sen12_q, pild_q), (1.0, 1.0))

    def test_trigger_schema_harmonizes_case_column_alias(self) -> None:
        common = {
            "rain_d7_wrongtime_median_mm": 20.0,
            "rain_d7_case_minus_wrongtime_mm": 10.0,
            "q_R": 1.0,
        }
        sen12, sen12_q = harmonize_trigger(
            {**common, "rain_d7_antecedent_case_mm": 30.0}
        )
        pild, pild_q = harmonize_trigger({**common, "rain_d7_case_mm": 30.0})
        self.assertEqual(len(sen12), len(TRIGGER_FEATURE_NAMES))
        self.assertEqual(sen12.tolist(), pild.tolist())
        self.assertEqual((sen12_q, pild_q), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
