#!/usr/bin/env python3
"""CPU tests for the strict Sen12 nested OOF protocol generator."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import h5py


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_sen12_nested_oof_protocol_v1 as protocol


class NestedOOFProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.split_csv = self.root / "logo5.csv"
        self.h5_path = self.root / "cache.h5"
        self.samples = []
        for region_index in range(11):
            region = f"r{region_index}"
            event = "event_shared" if region_index in {0, 1} else f"event_{region_index}"
            if region_index == 10:
                event = "event_2"  # Event-linked target leakage outside target region.
            for patch in range(1 + (region_index % 3)):
                self.samples.append((f"sample_{region_index}_{patch}", region, event))
        self._write_h5()
        self._write_split(label_offset=0)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_h5(self) -> None:
        string_type = h5py.string_dtype(encoding="utf-8")
        # Reverse order to prove that identity is joined by sample_id, not index.
        ordered = list(reversed(self.samples))
        with h5py.File(self.h5_path, "w") as handle:
            handle.create_dataset(
                "sample_id", data=[item[0] for item in ordered], dtype=string_type
            )
            handle.create_dataset(
                "physical_event_id", data=[item[2] for item in ordered], dtype=string_type
            )

    def _write_split(self, label_offset: int) -> None:
        fields = [
            "sample_id",
            "region_group",
            "spatial_supergroup",
            "source_id",
            "annotated_pixels",
            "mask_quality_label",
            "outer_fold",
            "role",
            "role_reason",
        ]
        rows = []
        for outer_fold in range(5):
            target_region = f"r{outer_fold + 2}"
            for index, (sample_id, region, _event) in enumerate(self.samples):
                if region == target_region:
                    role = "test"
                    reason = "heldout_target"
                elif region == "r9":
                    role = "val"
                    reason = "source_validation"
                else:
                    role = "train"
                    reason = "source_training"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "region_group": region,
                        "spatial_supergroup": region,
                        "source_id": "synthetic",
                        "annotated_pixels": str(label_offset + index * 17),
                        "mask_quality_label": f"label_{label_offset + index}",
                        "outer_fold": str(outer_fold),
                        "role": role,
                        "role_reason": reason,
                    }
                )
        with self.split_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _build(self, dirname: str) -> tuple[dict, Path]:
        outdir = self.root / dirname
        manifest = protocol.build_protocol(self.split_csv, self.h5_path, outdir)
        return manifest, outdir

    def test_cross_region_same_event_is_one_component_and_never_crosses_roles(self) -> None:
        manifest, outdir = self._build("first")
        rows = self._read_csv(outdir / "sen12_nested_oof_target_outer0_v1.csv")
        shared = [row for row in rows if row["spatial_supergroup"] in {"r0", "r1"}]
        self.assertEqual(len({row["nested_component_id"] for row in shared}), 1)
        for inner_fold in range(3):
            fold = [row for row in shared if row["outer_fold"] == str(inner_fold)]
            self.assertEqual(len({row["role"] for row in fold}), 1)
        component = next(
            item
            for item in manifest["targets"][0]["components"]
            if set(item["spatial_supergroups"]) == {"r0", "r1"}
        )
        self.assertEqual(component["physical_event_ids"], ["event_shared"])

    def test_target_sample_region_and_event_are_excluded(self) -> None:
        manifest, outdir = self._build("target_exclusion")
        rows = self._read_csv(outdir / "sen12_nested_oof_target_outer0_v1.csv")
        sample_ids = {row["sample_id"] for row in rows}
        regions = {row["spatial_supergroup"] for row in rows}
        events = {row["physical_event_id"] for row in rows}
        self.assertNotIn("r2", regions)
        self.assertNotIn("event_2", events)
        self.assertFalse(any(sample_id.startswith("sample_2_") for sample_id in sample_ids))
        self.assertFalse(any(sample_id.startswith("sample_10_") for sample_id in sample_ids))
        target = manifest["targets"][0]
        self.assertTrue(target["audit"]["zero_target_sample_region_event_leakage"])
        self.assertGreater(target["outer_development"]["n_target_linked_samples_excluded"], 0)

    def test_three_folds_are_complete_and_each_development_sample_tests_once(self) -> None:
        manifest, outdir = self._build("complete")
        for target in range(5):
            rows = self._read_csv(
                outdir / f"sen12_nested_oof_target_outer{target}_v1.csv"
            )
            by_fold: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_fold[row["outer_fold"]].append(row)
            self.assertEqual(set(by_fold), {"0", "1", "2"})
            test_counts: Counter[str] = Counter()
            all_samples = {row["sample_id"] for row in rows}
            for fold_rows in by_fold.values():
                self.assertEqual({row["role"] for row in fold_rows}, {"train", "val", "test"})
                by_role = {
                    role: [row for row in fold_rows if row["role"] == role]
                    for role in ("train", "val", "test")
                }
                for row in by_role["test"]:
                    test_counts[row["sample_id"]] += 1
                for identity in ("sample_id", "spatial_supergroup", "physical_event_id", "nested_component_id"):
                    sets = {role: {row[identity] for row in values} for role, values in by_role.items()}
                    self.assertFalse(sets["train"] & sets["val"])
                    self.assertFalse(sets["train"] & sets["test"])
                    self.assertFalse(sets["val"] & sets["test"])
            self.assertEqual(set(test_counts), all_samples)
            self.assertEqual(set(test_counts.values()), {1})
            self.assertTrue(
                manifest["targets"][target]["audit"][
                    "each_development_sample_exactly_once_inner_test"
                ]
            )

    def test_label_columns_do_not_affect_assignment(self) -> None:
        first, first_outdir = self._build("labels_first")
        first_projection = {}
        for target in range(5):
            path = first_outdir / f"sen12_nested_oof_target_outer{target}_v1.csv"
            first_projection[target] = [
                (row["outer_fold"], row["sample_id"], row["role"], row["nested_component_id"])
                for row in self._read_csv(path)
            ]
        self._write_split(label_offset=999_999)
        second, second_outdir = self._build("labels_second")
        for target in range(5):
            path = second_outdir / f"sen12_nested_oof_target_outer{target}_v1.csv"
            projection = [
                (row["outer_fold"], row["sample_id"], row["role"], row["nested_component_id"])
                for row in self._read_csv(path)
            ]
            self.assertEqual(first_projection[target], projection)
            self.assertEqual(
                first["targets"][target]["allocation_sha256"],
                second["targets"][target]["allocation_sha256"],
            )
        self.assertEqual(first["allocation_sha256"], second["allocation_sha256"])
        self.assertEqual(first["contract"]["label_columns_used_for_assignment"], [])

    def test_manifest_hash_and_output_hashes_are_verifiable(self) -> None:
        manifest, outdir = self._build("hashes")
        manifest_path = outdir / "sen12_nested_oof_protocol_v1_manifest.json"
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_payload_hash = persisted.pop("manifest_payload_sha256")
        self.assertEqual(protocol.hash_payload(persisted), expected_payload_hash)
        self.assertEqual(manifest["manifest_payload_sha256"], expected_payload_hash)
        for target in manifest["targets"]:
            self.assertEqual(
                protocol.sha256_file(Path(target["output_csv"])),
                target["output_csv_sha256"],
            )
        self.assertTrue(manifest["all_targets_all_audits_pass"])


if __name__ == "__main__":
    unittest.main()
