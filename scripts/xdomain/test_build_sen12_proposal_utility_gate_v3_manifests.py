#!/usr/bin/env python3
"""Synthetic tests for formal utility-gate aggregation and launch protocol."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import h5py


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_sen12_proposal_utility_gate_v3_manifests as aggregate
import build_sen12_nested_oof_protocol_v1 as nested_protocol
import train_sen12_prithvi_roleaware_hierarchical_v2 as roleaware
import train_sen12_proposal_utility_gate_v3 as gate


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": aggregate.sha256_file(path),
    }


class SyntheticTree:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.seed = 20260751
        self.input_root = root / "producer"
        self.protocol_root = root / "protocol"
        self.output_root = root / "formal_inputs"
        self.gate_out_root = root / "gate_outputs"
        self.material = root / "material.csv"
        self.material_schema = root / "material_schema.json"
        self.trigger = root / "trigger.csv"
        self.source_split = root / "source_outer.csv"
        self.h5_path = root / "frozen.h5"
        self.target_rows: dict[int, list[dict[str, object]]] = {}
        self.all_development_ids: list[str] = []
        self._write_protocol()
        self._write_context_registries()
        self._write_tasks()

    def development_ids(self, target: int) -> list[str]:
        return [f"t{target}_d{index}" for index in range(6)]

    def event(self, sample_id: str) -> str:
        return f"event_{sample_id}"

    def region(self, sample_id: str) -> str:
        return f"region_{sample_id}"

    def _role_rows(self, target: int, inner: int) -> list[dict[str, object]]:
        ids = self.development_ids(target)
        test = set(ids[2 * inner:2 * inner + 2])
        remainder = [sample_id for sample_id in ids if sample_id not in test]
        val = {remainder[0]}
        rows = []
        for sample_id in ids:
            role = "test" if sample_id in test else ("val" if sample_id in val else "train")
            rows.append({
                "patch_id": sample_id,
                "sample_id": sample_id,
                "region_group": self.region(sample_id),
                "spatial_supergroup": self.region(sample_id),
                "source_id": "SYNTHETIC",
                "physical_event_id": self.event(sample_id),
                "target_outer_fold": target,
                "source_outer_fold": target,
                "source_outer_role": "train",
                "nested_component_id": f"component_{sample_id}",
                "outer_fold": inner,
                "role": role,
                "role_reason": f"synthetic_{role}",
            })
        return rows

    def _write_protocol(self) -> None:
        source_rows = []
        targets = []
        self.protocol_root.mkdir(parents=True)
        for target in aggregate.TARGETS:
            outer_test = f"target{target}_outer_test"
            source_rows.append({
                "sample_id": outer_test,
                "outer_fold": target,
                "role": "test",
                "region_group": f"outer_region_{target}",
                "spatial_supergroup": f"outer_region_{target}",
            })
            source_rows.append({
                "sample_id": f"target{target}_outer_missing_from_h5",
                "outer_fold": target,
                "role": "test",
                "region_group": f"outer_region_{target}",
                "spatial_supergroup": f"outer_region_{target}",
            })
            all_rows: list[dict[str, object]] = []
            inner_manifests = []
            for inner in aggregate.INNER_FOLDS:
                rows = self._role_rows(target, inner)
                all_rows.extend(rows)
                role_manifest = {
                    role: {
                        key: value
                        for key, value in aggregate.role_detail_from_rows(rows, role).items()
                        if key not in {
                            "sample_ids", "spatial_supergroups", "region_groups",
                            "physical_event_ids", "component_ids",
                        }
                    }
                    for role in ("train", "val", "test")
                }
                inner_manifests.append({"inner_fold": inner, "roles": role_manifest})
            split_path = self.protocol_root / f"sen12_nested_oof_target_outer{target}_v1.csv"
            write_csv(split_path, all_rows)
            self.target_rows[target] = all_rows
            development = self.development_ids(target)
            self.all_development_ids.extend(development)
            targets.append({
                "target_outer_fold": target,
                "output_csv": str(split_path.resolve()),
                "output_csv_sha256": aggregate.sha256_file(split_path),
                "outer_development": {
                    "n_samples": len(development),
                    "sample_sha256": aggregate.producer_value_hash(development),
                },
                "target_outer_test": {
                    "n_h5_samples": 1,
                    "h5_sample_sha256": aggregate.producer_value_hash([outer_test]),
                },
                "inner_folds": inner_manifests,
            })
        write_csv(self.source_split, source_rows)
        text = h5py.string_dtype("utf-8")
        h5_ids = self.all_development_ids + [f"target{target}_outer_test" for target in aggregate.TARGETS]
        with h5py.File(self.h5_path, "w") as handle:
            handle.create_dataset("sample_id", data=h5_ids, dtype=text)
        protocol = {
            "schema_version": aggregate.PRODUCER_PROTOCOL_SCHEMA,
            "contract": {"n_target_outer_folds": 5, "n_inner_folds": 3},
            "inputs": {
                "split_csv": str(self.source_split.resolve()),
                "split_csv_sha256": aggregate.sha256_file(self.source_split),
                "h5_path": str(self.h5_path.resolve()),
                "h5_sha256": aggregate.sha256_file(self.h5_path),
            },
            "targets": targets,
            "all_targets_all_audits_pass": True,
        }
        protocol["manifest_payload_sha256"] = aggregate.canonical_hash(protocol)
        write_json(self.protocol_root / aggregate.PROTOCOL_MANIFEST_NAME, protocol)

    def _write_context_registries(self) -> None:
        material_columns = [
            roleaware.AWC_SOURCE_COLUMNS.get(name, name)
            for name in roleaware.LEGACY_MATERIAL_FEATURE_NAMES
        ]
        material_rows = []
        trigger_rows = []
        for index, sample_id in enumerate(self.all_development_ids):
            material_row: dict[str, object] = {
                "sample_id": sample_id,
                "region_group": self.region(sample_id),
                "q_M": 1.0,
                "q_M_full": 1.0,
            }
            for column_index, column in enumerate(material_columns):
                material_row[column] = float(index + column_index + 1)
            material_rows.append(material_row)
            trigger_rows.append({
                "sample_id": sample_id,
                "physical_event_id": self.event(sample_id),
                "rain_d7_antecedent_case_mm": float(index + 10),
                "rain_d7_wrongtime_median_mm": float(index + 3),
                "rain_d7_case_minus_wrongtime_mm": 7.0,
                "q_R": 1.0,
            })
        write_csv(self.material, material_rows)
        write_json(
            self.material_schema,
            {
                "model_eligible_features": list(roleaware.LEGACY_MATERIAL_FEATURE_NAMES),
                "model_eligible_dimension": len(roleaware.LEGACY_MATERIAL_FEATURE_NAMES),
            },
        )
        write_csv(self.trigger, trigger_rows)

    def _split_audit(self, target: int, inner: int) -> dict[str, object]:
        protocol_path = self.protocol_root / aggregate.PROTOCOL_MANIFEST_NAME
        split_path = self.protocol_root / f"sen12_nested_oof_target_outer{target}_v1.csv"
        rows = [row for row in self.target_rows[target] if int(row["outer_fold"]) == inner]
        roles = {
            role: aggregate.role_detail_from_rows(rows, role)
            for role in ("train", "val", "test")
        }
        audit = {
            "status": "PASS",
            "target_outer_fold": target,
            "inner_fold": inner,
            "split_csv": signature(split_path),
            "protocol_manifest": signature(protocol_path),
            "roles": roles,
            "leakage": {
                "sample_ids": [], "spatial_supergroups": [],
                "region_groups": [], "physical_event_ids": [],
            },
            "zero_target_outer_leakage": True,
            "zero_inner_role_sample_region_event_component_leakage": True,
            "label_access_contract": {
                "inner_test": "post_selection_paired_cache_export_only",
                "target_outer_test": "never_materialized",
            },
        }
        audit["audit_sha256"] = aggregate.canonical_hash(audit)
        return audit

    def _write_task(self, target: int, inner: int) -> None:
        directory = aggregate.task_dir(self.input_root, target, inner, self.seed)
        directory.mkdir(parents=True)
        audit = self._split_audit(target, inner)
        write_json(directory / "split_audit.json", audit)
        write_json(directory / "config.json", {"target": target, "inner": inner, "seed": self.seed})
        visual_path = directory / "checkpoints/visual_proposer.pt"
        terrain_path = directory / "checkpoints/terrain_proposer.pt"
        visual_path.parent.mkdir(parents=True)
        visual_path.write_bytes(f"visual-{target}-{inner}".encode())
        terrain_path.write_bytes(f"terrain-{target}-{inner}".encode())
        test_ids = audit["roles"]["test"]["sample_ids"]
        n = len(test_ids)
        shape = (n, 1, 2, 2)
        direction = torch.full(shape, 0.25 + inner * 0.01, dtype=torch.float16)
        cache_identity = {
            "cache_schema_version": aggregate.PRODUCER_CACHE_SCHEMA,
            "target_outer_fold": target,
            "inner_fold": inner,
            "seed": self.seed,
            "export_role": "inner_test_post_selection_only",
            "split_audit_sha256": audit["audit_sha256"],
            "split_csv_sha256": audit["split_csv"]["sha256"],
            "visual_checkpoint_sha256": aggregate.sha256_file(visual_path),
            "terrain_checkpoint_sha256": aggregate.sha256_file(terrain_path),
            "visual_threshold": 0.5,
            "routing": {"alpha": 4.0},
            "sample_sha256": audit["roles"]["test"]["sample_sha256"],
        }
        events = [self.event(sample_id) for sample_id in test_ids]
        regions = [self.region(sample_id) for sample_id in test_ids]
        cache = {
            "identity": cache_identity,
            "sample_ids": test_ids,
            "physical_event_ids": events,
            "spatial_supergroups": regions,
            "region_groups": regions,
            "component_ids": [f"component_{sample_id}" for sample_id in test_ids],
            "event_ids": events,
            "source_ids": regions,
            "dataset_source_ids": ["SYNTHETIC"] * n,
            "visual_logits": torch.zeros(shape, dtype=torch.float16),
            "terrain_logits": torch.ones(shape, dtype=torch.float16),
            "terrain_direction": direction,
            "frozen_vt_correction": direction * 4.0,
            "q_t": torch.ones(shape, dtype=torch.float16),
            "mask": torch.zeros(shape, dtype=torch.uint8),
            "valid": torch.ones(shape, dtype=torch.uint8),
        }
        cache_path = directory / "cache/inner_test_proposer_cache.pt"
        cache_path.parent.mkdir(parents=True)
        torch.save(cache, cache_path)
        run_manifest = {
            "schema_version": aggregate.PRODUCER_RUN_SCHEMA,
            "status": "complete",
            "target_outer_fold": target,
            "inner_fold": inner,
            "seed": self.seed,
            "split_audit": audit,
            "config": signature(directory / "config.json"),
            "checkpoints": {"visual": signature(visual_path), "terrain": signature(terrain_path)},
            "selection": {
                "visual": {"threshold": 0.5},
                "inner_test_used_for_selection": False,
                "target_outer_test_used_anywhere": False,
            },
            "cache": signature(cache_path),
            "cache_schema": {
                "paired_same_checkpoint": True,
                "tensor_keys": list(aggregate.PRODUCER_TENSOR_KEYS),
                "metadata_keys": list(aggregate.PRODUCER_METADATA_KEYS),
            },
        }
        run_manifest["manifest_payload_sha256"] = aggregate.canonical_hash(run_manifest)
        write_json(directory / "run_manifest.json", run_manifest)
        done_artifacts = {
            "config.json": aggregate.sha256_file(directory / "config.json"),
            "split_audit.json": aggregate.sha256_file(directory / "split_audit.json"),
            "checkpoints/visual_proposer.pt": aggregate.sha256_file(visual_path),
            "checkpoints/terrain_proposer.pt": aggregate.sha256_file(terrain_path),
            "cache/inner_test_proposer_cache.pt": aggregate.sha256_file(cache_path),
            "run_manifest.json": aggregate.sha256_file(directory / "run_manifest.json"),
        }
        write_json(directory / "DONE.json", {"status": "complete", "artifact_sha256": done_artifacts})

    def _write_tasks(self) -> None:
        for target in aggregate.TARGETS:
            for inner in aggregate.INNER_FOLDS:
                self._write_task(target, inner)

    def args(self, *, dry_run: bool = False, output_root: Path | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            input_root=self.input_root,
            protocol_root=self.protocol_root,
            output_root=self.output_root if output_root is None else output_root,
            material_registry=self.material,
            material_schema=self.material_schema,
            trigger_registry=self.trigger,
            seed=self.seed,
            dry_run=dry_run,
        )

    def refresh_done(self, target: int, inner: int) -> None:
        directory = aggregate.task_dir(self.input_root, target, inner, self.seed)
        done = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
        for relative in list(done["artifact_sha256"]):
            done["artifact_sha256"][relative] = aggregate.sha256_file(directory / relative)
        write_json(directory / "DONE.json", done)


class AggregatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tree = SyntheticTree(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_protocol_hash_accepts_non_ascii_paths_without_weakening_run_hashes(self) -> None:
        payload = {"path": "${WORKSPACE_ROOT}/protocol.csv", "status": "PASS"}
        protocol_manifest = {
            **payload,
            "manifest_payload_sha256": nested_protocol.hash_payload(payload),
        }
        aggregate.validate_payload_hash(
            protocol_manifest,
            "manifest_payload_sha256",
            "protocol manifest",
            hash_fn=aggregate.protocol_canonical_hash,
        )
        self.assertNotEqual(
            aggregate.protocol_canonical_hash(payload), aggregate.canonical_hash(payload)
        )

    def test_dry_run_validates_fifteen_without_output(self) -> None:
        result = aggregate.aggregate(self.tree.args(dry_run=True))
        self.assertEqual(result["validated_tasks"], 15)
        self.assertEqual(result["status"], "dry_run_validated")
        self.assertFalse(self.tree.output_root.exists())

    def test_aggregate_emits_five_validator_accepted_manifests(self) -> None:
        result = aggregate.aggregate(self.tree.args())
        self.assertTrue(result["output_written"])
        self.assertEqual(len(result["targets"]), 5)
        self.assertTrue((self.tree.output_root / "DONE.json").is_file())
        for target in aggregate.TARGETS:
            target_root = self.tree.output_root / f"target_outer{target}"
            access_log: list[dict[str, object]] = []
            bundles, audit = gate.load_formal_nested_bundles(
                target_root / "oof_manifest.json",
                target_fold=target,
                split_csv=target_root / "gate_split.csv",
                seed=self.tree.seed,
                access_log=access_log,
            )
            self.assertEqual(len(bundles), 3)
            self.assertTrue(audit["inner_holdouts_partition_outer_train"])
            manifest_text = (target_root / "oof_manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("frozen_test_cache", manifest_text)
            self.assertNotIn("target_outer1/inner", manifest_text if target == 0 else "")

    def test_aggregated_manifest_preserves_freeze_order_and_control_checkpoint(self) -> None:
        aggregate.aggregate(self.tree.args())
        target_root = self.tree.output_root / "target_outer0"
        shape = (1, 1, 2, 2)

        def target_loader(fold: int, **kwargs: object) -> object:
            access_log = kwargs["access_log"]
            self.assertTrue(any(item["stage"] == "models_frozen" for item in access_log))
            access_log.append({
                "stage": kwargs["purpose"], "fold": fold, "labels_loaded": True
            })
            visual = np.asarray([[[[-1.0, 1.0], [-1.0, 1.0]]]], np.float32)
            delta = np.asarray([[[[2.0, -2.0], [2.0, -2.0]]]], np.float32)
            return gate.FoldBundle(
                fold=fold,
                seed=self.tree.seed,
                sample_ids=("target0_outer_test",),
                event_ids=("outer_event_0",),
                source_ids=("outer_region_0",),
                threshold=0.5,
                visual_logits=visual,
                terrain_delta=delta,
                valid=np.ones(shape, bool),
                mask=np.asarray([[[[1, 0], [1, 0]]]], bool),
                material=np.ones((1, len(roleaware.LEGACY_MATERIAL_FEATURE_NAMES)), np.float32),
                q_material=np.ones(1, np.float32),
                material_shuffle=np.zeros((1, len(roleaware.LEGACY_MATERIAL_FEATURE_NAMES)), np.float32),
                q_material_shuffle=np.ones(1, np.float32),
                trigger=np.ones((1, 3), np.float32),
                trigger_wrong=np.zeros((1, 3), np.float32),
                q_trigger=np.ones(1, np.float32),
                trigger_shuffle=np.zeros((1, 3), np.float32),
                q_trigger_shuffle=np.ones(1, np.float32),
                identity={
                    "split": "test",
                    "visual_checkpoint": {"sha256": "outer-visual"},
                    "terrain_checkpoint": {"sha256": "outer-terrain"},
                },
                cache_path="/synthetic/outer-cache.pt",
                cache_sha256="outer-cache-sha",
                result_path="/synthetic/outer-result.json",
                result_sha256="outer-result-sha",
            )

        arguments = SimpleNamespace(
            protocol_mode="formal_nested_oof",
            oof_manifest=target_root / "oof_manifest.json",
            target_fold=0,
            seed=self.tree.seed,
            cache_root=self.root,
            runs_root=self.root,
            split_csv=target_root / "gate_split.csv",
            alphas=(1e-3,),
            threshold_grid=(0.5,),
        )
        result = gate.run_protocol(arguments, target_loader_fn=target_loader)
        frozen_index = next(
            index for index, item in enumerate(result["access_log"])
            if item["stage"] == "models_frozen"
        )
        target_index = next(
            index for index, item in enumerate(result["access_log"])
            if item["stage"] == "target_evaluation_after_freeze"
        )
        self.assertGreater(target_index, frozen_index)
        for context in gate.CONTEXTS:
            evaluation = result["target_evaluation"][context]
            checkpoint = evaluation["checkpoint"]["checkpoint_sha256"]
            self.assertTrue(evaluation["negative_controls_reuse_same_checkpoint"])
            self.assertTrue(all(
                item["checkpoint_sha256"] == checkpoint
                for item in evaluation["candidate_controls"].values()
            ))

    def test_missing_task_fails_closed_before_output(self) -> None:
        path = aggregate.task_dir(self.tree.input_root, 4, 2, self.tree.seed) / "DONE.json"
        path.unlink()
        with self.assertRaisesRegex(aggregate.AggregateError, "15/15"):
            aggregate.aggregate(self.tree.args())
        self.assertFalse(self.tree.output_root.exists())

    def test_hash_mismatch_fails_closed(self) -> None:
        path = aggregate.task_dir(self.tree.input_root, 0, 0, self.tree.seed) / "cache/inner_test_proposer_cache.pt"
        with path.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(aggregate.AggregateError, "hash mismatch"):
            aggregate.aggregate(self.tree.args(dry_run=True))

    def test_stale_done_fails_closed(self) -> None:
        directory = aggregate.task_dir(self.tree.input_root, 0, 0, self.tree.seed)
        done_mtime = (directory / "DONE.json").stat().st_mtime_ns
        run_manifest = directory / "run_manifest.json"
        os.utime(run_manifest, ns=(done_mtime + 10_000_000, done_mtime + 10_000_000))
        with self.assertRaisesRegex(aggregate.AggregateError, "DONE is stale"):
            aggregate.aggregate(self.tree.args(dry_run=True))

    def test_target_inner_seed_identity_mismatch_fails_closed(self) -> None:
        directory = aggregate.task_dir(self.tree.input_root, 0, 0, self.tree.seed)
        path = directory / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["inner_fold"] = 2
        manifest["manifest_payload_sha256"] = aggregate.canonical_hash({
            key: value for key, value in manifest.items() if key != "manifest_payload_sha256"
        })
        write_json(path, manifest)
        self.tree.refresh_done(0, 0)
        with self.assertRaisesRegex(aggregate.AggregateError, "identity mismatch"):
            aggregate.aggregate(self.tree.args(dry_run=True))

    def test_holdout_overlap_fails_closed(self) -> None:
        protocol_manifest, protocol_targets, source_split, h5_sample_ids = aggregate.protocol_inputs(
            self.tree.protocol_root
        )
        del protocol_manifest
        protocol_path = self.tree.protocol_root / aggregate.PROTOCOL_MANIFEST_NAME
        tasks = []
        split = self.tree.protocol_root / "sen12_nested_oof_target_outer0_v1.csv"
        for inner in aggregate.INNER_FOLDS:
            tasks.append(aggregate.validate_task(
                aggregate.task_dir(self.tree.input_root, 0, inner, self.tree.seed),
                target=0,
                inner=inner,
                seed=self.tree.seed,
                protocol_manifest_path=protocol_path,
                protocol_target=protocol_targets[0],
                protocol_split=split,
            ))
        tasks[1]["payload"]["sample_ids"] = list(tasks[0]["payload"]["sample_ids"])
        with self.assertRaisesRegex(aggregate.AggregateError, "holdout overlap"):
            aggregate.build_formal_target(
                self.root / "stage",
                self.root / "final",
                0,
                self.tree.seed,
                tasks,
                split,
                source_split,
                h5_sample_ids,
                protocol_targets[0],
                self.tree.material,
                self.tree.trigger,
                aggregate.build_context_source(
                    self.tree.material,
                    self.tree.material_schema,
                    self.tree.trigger,
                ),
            )

    def test_runner_dry_run_has_five_targets_and_two_gpu_queues(self) -> None:
        runner = SCRIPT_DIR / "run_sen12_proposal_utility_gate_v3_formal.sh"
        environment = dict(os.environ)
        environment.update({
            "PYTHON": sys.executable,
            "INPUT_ROOT": str(self.tree.input_root),
            "PROTOCOL_ROOT": str(self.tree.protocol_root),
            "FORMAL_INPUT_ROOT": str(self.root / "runner_formal_inputs"),
            "GATE_OUT_ROOT": str(self.tree.gate_out_root),
            "MATERIAL_REGISTRY": str(self.tree.material),
            "MATERIAL_SCHEMA": str(self.tree.material_schema),
            "TRIGGER_REGISTRY": str(self.tree.trigger),
            "GPUS": "3,7",
            "SEED": str(self.tree.seed),
            "DRY_RUN": "1",
        })
        completed = subprocess.run(
            ["bash", str(runner)],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line.startswith("[DRY_RUN target=")]
        self.assertEqual(len(lines), 5)
        self.assertIn("target=0 gpu=3", lines[0])
        self.assertIn("target=1 gpu=7", lines[1])
        self.assertIn("--protocol-mode formal_nested_oof", completed.stdout)
        self.assertIn("real_gate_started=0", completed.stdout)
        self.assertFalse((self.root / "runner_formal_inputs").exists())
        self.assertFalse(self.tree.gate_out_root.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
