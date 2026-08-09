#!/usr/bin/env python3
"""CPU tests for the strict nested OOF proposer cache contract."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_sen12_nested_oof_proposer_cache_v1 as proposer
import build_sen12_nested_oof_protocol_v1 as nested


class ProposerCacheContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_csv = self.root / "source.csv"
        self.h5_path = self.root / "base.h5"
        self.protocol_root = self.root / "protocol"
        self.samples = []
        for region_index in range(10):
            region = f"r{region_index}"
            event = "shared" if region_index in {0, 1} else f"event{region_index}"
            for patch in range(1 + region_index % 2):
                self.samples.append((f"s{region_index}_{patch}", region, event))
        self._write_h5()
        self._write_source()
        self.protocol_manifest = self.protocol_root / "sen12_nested_oof_protocol_v1_manifest.json"
        nested.build_protocol(self.source_csv, self.h5_path, self.protocol_root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_h5(self) -> None:
        text = h5py.string_dtype("utf-8")
        with h5py.File(self.h5_path, "w") as handle:
            handle.create_dataset("sample_id", data=[row[0] for row in self.samples], dtype=text)
            handle.create_dataset(
                "physical_event_id", data=[row[2] for row in self.samples], dtype=text
            )

    def _write_source(self) -> None:
        fields = [
            "sample_id", "region_group", "spatial_supergroup", "source_id",
            "annotated_pixels", "outer_fold", "role", "role_reason",
        ]
        rows = []
        for outer in range(5):
            target = f"r{outer + 2}"
            for index, (sample_id, region, _event) in enumerate(self.samples):
                role = "test" if region == target else ("val" if region == "r9" else "train")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "region_group": region,
                        "spatial_supergroup": region,
                        "source_id": "synthetic",
                        "annotated_pixels": str(index * 13),
                        "outer_fold": str(outer),
                        "role": role,
                        "role_reason": f"source_{role}",
                    }
                )
        with self.source_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _audit(self, target: int = 0, inner: int = 0):
        split = self.protocol_root / f"sen12_nested_oof_target_outer{target}_v1.csv"
        return proposer.audit_nested_split(
            split, self.source_csv, self.protocol_manifest, target, inner, self.h5_path
        )

    def test_valid_split_audit_records_all_roles_and_zero_leakage(self) -> None:
        _rows, roles, audit = self._audit()
        self.assertEqual(set(roles), {"train", "val", "test"})
        self.assertTrue(audit["zero_target_outer_leakage"])
        self.assertTrue(audit["zero_inner_role_sample_region_event_component_leakage"])
        self.assertEqual(audit["label_access_contract"]["inner_test"], "post_selection_paired_cache_export_only")
        for role in ("train", "val", "test"):
            self.assertGreater(audit["roles"][role]["n_samples"], 0)

    def test_tampered_split_hash_is_rejected(self) -> None:
        split = self.protocol_root / "sen12_nested_oof_target_outer0_v1.csv"
        split.write_text(split.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(proposer.ProposerProtocolError, "hash differs"):
            self._audit()

    def test_target_outer_identity_injection_is_rejected_even_with_updated_manifest_hash(self) -> None:
        split = self.protocol_root / "sen12_nested_oof_target_outer0_v1.csv"
        rows = proposer.read_csv(split)
        injected = dict(rows[0])
        target_sample = next(sample for sample, region, _event in self.samples if region == "r2")
        injected["sample_id"] = target_sample
        injected["spatial_supergroup"] = "r2"
        injected["region_group"] = "r2"
        injected["physical_event_id"] = "event2"
        rows.append(injected)
        fields = list(rows[0])
        with split.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest = json.loads(self.protocol_manifest.read_text(encoding="utf-8"))
        manifest["targets"][0]["output_csv_sha256"] = proposer.sha256_file(split)
        self.protocol_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises((proposer.ProposerProtocolError, ValueError)):
            self._audit()

    def test_cache_schema_requires_paired_shapes_and_fixed_correction(self) -> None:
        count = 3
        shape = (count, 1, 8, 8)
        direction = torch.randn(shape).half()
        payload = {
            "identity": {
                "cache_schema_version": proposer.CACHE_SCHEMA_VERSION,
                "export_role": "inner_test_post_selection_only",
            },
            "sample_ids": [f"s{i}" for i in range(count)],
            "physical_event_ids": [f"e{i}" for i in range(count)],
            "spatial_supergroups": [f"r{i}" for i in range(count)],
            "region_groups": [f"r{i}" for i in range(count)],
            "component_ids": [f"c{i}" for i in range(count)],
            "event_ids": [f"e{i}" for i in range(count)],
            "source_ids": [f"r{i}" for i in range(count)],
            "dataset_source_ids": ["synthetic"] * count,
            "visual_logits": torch.randn(shape).half(),
            "terrain_logits": torch.randn(shape).half(),
            "terrain_direction": direction,
            "frozen_vt_correction": (direction.float() * proposer.ROUTING_CONFIG["alpha"]).half(),
            "q_t": torch.ones(shape).half(),
            "mask": torch.zeros(shape, dtype=torch.uint8),
            "valid": torch.ones(shape, dtype=torch.uint8),
        }
        proposer.validate_cache_payload(payload)
        broken = dict(payload)
        broken["terrain_logits"] = torch.randn(count, 1, 4, 4)
        with self.assertRaisesRegex(proposer.ProposerProtocolError, "shape mismatch"):
            proposer.validate_cache_payload(broken)
        broken = dict(payload)
        broken["frozen_vt_correction"] = torch.zeros(shape).half()
        with self.assertRaisesRegex(proposer.ProposerProtocolError, "differs"):
            proposer.validate_cache_payload(broken)

    def test_runner_dry_run_has_fifteen_tasks_and_thirty_trainings(self) -> None:
        runner = SCRIPT_DIR / "run_sen12_nested_oof_proposer_cache_v1.sh"
        completed = subprocess.run(
            ["bash", str(runner)],
            env={"PATH": "/usr/bin:/bin", "DRY_RUN": "1"},
            text=True,
            capture_output=True,
            check=True,
        )
        task_lines = [line for line in completed.stdout.splitlines() if line.startswith("[DRY_RUN task=")]
        self.assertEqual(len(task_lines), 15)
        self.assertIn("tasks=15 proposer_trainings=30 gpus=2", completed.stdout)


if __name__ == "__main__":
    unittest.main()
