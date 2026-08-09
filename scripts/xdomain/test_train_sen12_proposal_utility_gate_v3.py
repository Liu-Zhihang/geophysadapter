#!/usr/bin/env python3
"""Synthetic protocol tests for the strict Sen12 proposal utility gate v3."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)


MODULE_PATH = Path(__file__).with_name("train_sen12_proposal_utility_gate_v3.py")
SPEC = importlib.util.spec_from_file_location("proposal_gate_v3", MODULE_PATH)
assert SPEC and SPEC.loader
V3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V3
SPEC.loader.exec_module(V3)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload_for(
    sample_ids: list[str],
    event_ids: list[str],
    identity: dict[str, object],
    *,
    material_signs: list[float] | None = None,
    source_ids: list[str] | None = None,
) -> dict[str, object]:
    n = len(sample_ids)
    if material_signs is None:
        material_signs = [1.0 if index % 2 == 0 else -1.0 for index in range(n)]
    signs = np.asarray(material_signs, np.float32)
    if source_ids is None:
        source_ids = ["SYNTHETIC"] * n
    visual = np.tile(np.asarray([[-1.0, 1.0], [-1.0, 1.0]], np.float32), (n, 1, 1))
    delta = np.tile(np.asarray([[2.0, -2.0], [2.0, -2.0]], np.float32), (n, 1, 1))
    visual_prediction = visual >= 0
    terrain_prediction = visual + delta >= 0
    mask = np.where(signs[:, None, None] > 0, terrain_prediction, visual_prediction)
    material = signs[:, None]
    trigger = signs[:, None]
    return {
        "identity": identity,
        "sample_ids": sample_ids,
        "event_ids": event_ids,
        "source_ids": source_ids,
        "visual_logits": torch.from_numpy(visual),
        "frozen_vt_correction": torch.from_numpy(delta),
        "valid": torch.ones((n, 2, 2), dtype=torch.bool),
        "mask": torch.from_numpy(mask),
        "material": torch.from_numpy(material),
        "q_material": torch.ones(n),
        "material_shuffle": torch.from_numpy(-material),
        "q_material_shuffle": torch.ones(n),
        "trigger": torch.from_numpy(trigger),
        "trigger_wrong": torch.from_numpy(-trigger),
        "q_trigger": torch.ones(n),
        "trigger_shuffle": torch.from_numpy(-trigger),
        "q_trigger_shuffle": torch.ones(n),
    }


class FormalFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.seed = 20260751
        self.target_fold = 0
        self.split_csv = root / "split.csv"
        self.manifest_path = root / "manifest.json"
        self.regions = {0: "A", 1: "B", 2: "C", 3: "D"}
        self.holdouts = {
            inner: [f"{region.lower()}0", f"{region.lower()}1"]
            for inner, region in self.regions.items()
        }
        self.events = {inner: f"EVENT_{region}" for inner, region in self.regions.items()}
        self._write_split()
        self.rebuild()

    def _write_split(self) -> None:
        fields = ("sample_id", "outer_fold", "role", "region_group")
        rows: list[dict[str, object]] = []
        for inner, region in self.regions.items():
            rows.extend(
                {"sample_id": sample_id, "outer_fold": 0, "role": "train", "region_group": region}
                for sample_id in self.holdouts[inner]
            )
        rows += [
            {"sample_id": "z0", "outer_fold": 0, "role": "test", "region_group": "Z"},
            {"sample_id": "z1", "outer_fold": 0, "role": "test", "region_group": "Z"},
            {"sample_id": "v0", "outer_fold": 0, "role": "val", "region_group": "V"},
        ]
        with self.split_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def rebuild(self) -> None:
        entries = []
        all_train = set().union(*map(set, self.holdouts.values()))
        split_sha = V3.sha256_file(self.split_csv)
        for inner, holdout_ids in self.holdouts.items():
            holdout_set = set(holdout_ids)
            train_ids = sorted(all_train - holdout_set)
            train_regions = sorted({
                self.regions[index]
                for index, ids in self.holdouts.items()
                if set(ids) & set(train_ids)
            })
            holdout_regions = [self.regions[inner]]
            holdout_events = [self.events[inner]]
            train_event_map = {
                sample_id: self.events[index]
                for index, ids in self.holdouts.items()
                for sample_id in ids
                if sample_id in train_ids
            }
            holdout_event_map = {sample_id: self.events[inner] for sample_id in holdout_ids}
            visual_sha = f"visual-{inner:02d}-sha"
            terrain_sha = f"terrain-{inner:02d}-sha"
            receipt = {
                "schema_version": V3.FORMAL_RECEIPT_SCHEMA,
                "target_outer_fold": 0,
                "inner_fold": inner,
                "seed": self.seed,
                "split_csv_sha256": split_sha,
                "proposer_train_sample_ids": train_ids,
                "proposer_train_sample_sha256": V3.sample_hash(train_ids),
                "inner_holdout_sample_ids": holdout_ids,
                "inner_holdout_sample_sha256": V3.sample_hash(sorted(holdout_ids)),
                "proposer_train_regions": train_regions,
                "inner_holdout_regions": holdout_regions,
                "proposer_train_sample_event_ids": train_event_map,
                "inner_holdout_sample_event_ids": holdout_event_map,
                "proposer_train_events": sorted(set(train_event_map.values())),
                "inner_holdout_events": holdout_events,
                "visual_checkpoint_sha256": visual_sha,
                "terrain_checkpoint_sha256": terrain_sha,
                "visual_threshold": 0.5,
            }
            receipt_path = self.root / f"inner{inner}_receipt.json"
            write_json(receipt_path, receipt)
            receipt_sha = V3.sha256_file(receipt_path)
            identity = {
                "schema_version": V3.FORMAL_CACHE_SCHEMA,
                "split": "nested_inner_holdout",
                "target_outer_fold": 0,
                "inner_fold": inner,
                "seed": self.seed,
                "sample_sha256": V3.sample_hash(holdout_ids),
                "holdout_regions": holdout_regions,
                "holdout_events": holdout_events,
                "producer_receipt_sha256": receipt_sha,
                "visual_checkpoint_sha256": visual_sha,
                "terrain_checkpoint_sha256": terrain_sha,
                "visual_threshold": 0.5,
            }
            cache_path = self.root / f"inner{inner}_cache.pt"
            torch.save(
                payload_for(
                    holdout_ids,
                    [self.events[inner]] * len(holdout_ids),
                    identity,
                    source_ids=[self.regions[inner]] * len(holdout_ids),
                ),
                cache_path,
            )
            entries.append({
                "inner_fold": inner,
                "cache_path": cache_path.name,
                "cache_sha256": V3.sha256_file(cache_path),
                "producer_receipt_path": receipt_path.name,
                "producer_receipt_sha256": receipt_sha,
                "holdout_sample_sha256": V3.sample_hash(sorted(holdout_ids)),
                "holdout_regions": holdout_regions,
                "holdout_events": holdout_events,
                "visual_checkpoint_sha256": visual_sha,
                "terrain_checkpoint_sha256": terrain_sha,
                "visual_threshold": 0.5,
            })
        write_json(self.manifest_path, {
            "schema_version": V3.FORMAL_MANIFEST_SCHEMA,
            "target_outer_fold": 0,
            "seed": self.seed,
            "split_csv_sha256": split_sha,
            "entries": entries,
        })

    def manifest(self) -> dict[str, object]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, value: dict[str, object]) -> None:
        write_json(self.manifest_path, value)

    def rewrite_receipt_and_rebind(
        self, manifest: dict[str, object], entry_index: int, receipt: dict[str, object]
    ) -> None:
        entry = manifest["entries"][entry_index]
        receipt_path = self.root / entry["producer_receipt_path"]
        cache_path = self.root / entry["cache_path"]
        write_json(receipt_path, receipt)
        receipt_sha = V3.sha256_file(receipt_path)
        payload = torch.load(cache_path, weights_only=False)
        payload["identity"]["producer_receipt_sha256"] = receipt_sha
        torch.save(payload, cache_path)
        entry["producer_receipt_sha256"] = receipt_sha
        entry["cache_sha256"] = V3.sha256_file(cache_path)
        self.save_manifest(manifest)


def outer_bundle(fold: int, seed: int, prefix: str = "target") -> object:
    sample_ids = [f"{prefix}_{fold}_0", f"{prefix}_{fold}_1"]
    identity = {
        "split": "test",
        "visual_checkpoint": {"sha256": f"visual-{fold}"},
        "terrain_checkpoint": {"sha256": f"terrain-{fold}"},
    }
    payload = payload_for(sample_ids, [f"E_{fold}"] * 2, identity)
    arrays = V3._payload_arrays(payload, 2, "synthetic outer")
    return V3.FoldBundle(
        fold=fold,
        seed=seed,
        sample_ids=tuple(sample_ids),
        event_ids=tuple(payload["event_ids"]),
        source_ids=tuple(payload["source_ids"]),
        threshold=0.5,
        identity=identity,
        cache_path=f"/{prefix}/fold{fold}.pt",
        cache_sha256=f"cache-{prefix}-{fold}",
        result_path=f"/{prefix}/fold{fold}.json",
        result_sha256=f"result-{prefix}-{fold}",
        **arrays,
    )


def args_for(fixture: FormalFixture, mode: str = "formal_nested_oof") -> SimpleNamespace:
    return SimpleNamespace(
        protocol_mode=mode,
        oof_manifest=fixture.manifest_path if mode == "formal_nested_oof" else None,
        target_fold=0,
        seed=fixture.seed,
        cache_root=fixture.root,
        runs_root=fixture.root,
        split_csv=fixture.split_csv,
        alphas=(1e-3,),
        threshold_grid=(0.5,),
    )


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", ConvergenceWarning)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = FormalFixture(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def load_formal(self) -> tuple[list[object], dict[str, object]]:
        return V3.load_formal_nested_bundles(
            self.fixture.manifest_path,
            target_fold=0,
            split_csv=self.fixture.split_csv,
            seed=self.fixture.seed,
            access_log=[],
        )

    def test_formal_manifest_and_delayed_target_access(self) -> None:
        def target_loader(fold: int, **kwargs: object) -> object:
            access_log = kwargs["access_log"]
            self.assertTrue(any(item["stage"] == "models_frozen" for item in access_log))
            access_log.append({"stage": kwargs["purpose"], "fold": fold, "labels_loaded": True})
            return outer_bundle(fold, self.fixture.seed)

        result = V3.run_protocol(args_for(self.fixture), target_loader_fn=target_loader)
        self.assertEqual(result["evidence_status"], "formal_nested_oof")
        target_positions = [
            index for index, item in enumerate(result["access_log"])
            if item["stage"] == "target_evaluation_after_freeze"
        ]
        frozen_position = next(
            index for index, item in enumerate(result["access_log"])
            if item["stage"] == "models_frozen"
        )
        self.assertTrue(target_positions and min(target_positions) > frozen_position)
        self.assertTrue(result["provenance_audit"]["inner_holdouts_partition_outer_train"])

    def test_target_outer_test_event_leakage_is_rejected(self) -> None:
        def target_loader(fold: int, **kwargs: object) -> object:
            kwargs["access_log"].append({
                "stage": kwargs["purpose"], "fold": fold, "labels_loaded": True
            })
            bundle = outer_bundle(fold, self.fixture.seed)
            bundle.event_ids = (self.fixture.events[0],) * len(bundle.sample_ids)
            return bundle

        with self.assertRaisesRegex(RuntimeError, "target outer-test event leakage"):
            V3.run_protocol(args_for(self.fixture), target_loader_fn=target_loader)

    def test_non_oof_cache_identity_is_rejected(self) -> None:
        manifest = self.fixture.manifest()
        entry = manifest["entries"][0]
        cache_path = self.root / entry["cache_path"]
        payload = torch.load(cache_path, weights_only=False)
        payload["identity"]["split"] = "test"
        torch.save(payload, cache_path)
        entry["cache_sha256"] = V3.sha256_file(cache_path)
        self.fixture.save_manifest(manifest)
        with self.assertRaisesRegex(RuntimeError, "cache identity mismatch"):
            self.load_formal()

    def test_target_or_inner_geography_leakage_is_rejected(self) -> None:
        manifest = self.fixture.manifest()
        entry = manifest["entries"][0]
        receipt_path = self.root / entry["producer_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["proposer_train_regions"].append("Z")
        write_json(receipt_path, receipt)
        entry["producer_receipt_sha256"] = V3.sha256_file(receipt_path)
        self.fixture.save_manifest(manifest)
        with self.assertRaisesRegex(RuntimeError, "region receipt|outer-test geography"):
            self.load_formal()

    def test_target_outer_test_sample_leakage_is_rejected(self) -> None:
        manifest = self.fixture.manifest()
        entry = manifest["entries"][0]
        receipt_path = self.root / entry["producer_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["proposer_train_sample_ids"].append("z0")
        receipt["proposer_train_sample_ids"].sort()
        receipt["proposer_train_sample_sha256"] = V3.sample_hash(
            receipt["proposer_train_sample_ids"]
        )
        receipt["proposer_train_regions"].append("Z")
        receipt["proposer_train_sample_event_ids"]["z0"] = "EVENT_Z"
        receipt["proposer_train_events"].append("EVENT_Z")
        self.fixture.rewrite_receipt_and_rebind(manifest, 0, receipt)
        with self.assertRaisesRegex(RuntimeError, "outer-val/test or unknown samples"):
            self.load_formal()

    def test_inner_event_leakage_is_rejected(self) -> None:
        manifest = self.fixture.manifest()
        entry = manifest["entries"][0]
        receipt_path = self.root / entry["producer_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        train_sample = receipt["proposer_train_sample_ids"][0]
        receipt["proposer_train_sample_event_ids"][train_sample] = self.fixture.events[0]
        receipt["proposer_train_events"] = sorted(
            set(receipt["proposer_train_sample_event_ids"].values())
        )
        self.fixture.rewrite_receipt_and_rebind(manifest, 0, receipt)
        with self.assertRaisesRegex(RuntimeError, "train/holdout event leakage"):
            self.load_formal()

    def test_holdouts_must_exactly_partition_outer_train(self) -> None:
        manifest = self.fixture.manifest()
        manifest["entries"] = manifest["entries"][:-1]
        self.fixture.save_manifest(manifest)
        with self.assertRaisesRegex(RuntimeError, "at least three|partition"):
            self.load_formal()

    def test_binary_residual_contract(self) -> None:
        delta = np.asarray([[1.0, -2.0], [3.0, 4.0]], np.float32)
        actionable = np.asarray([[True, True], [False, True]])
        accept = np.asarray([[True, False], [False, False]])
        residual = V3.apply_binary_gate(delta, actionable, accept)
        self.assertTrue(np.all((residual == 0) | (residual == delta)))
        self.assertEqual(float(residual[0, 1]), 0.0)
        self.assertEqual(float(residual[1, 0]), float(delta[1, 0]))

    def test_q0_is_exact_proposal_only_fallback(self) -> None:
        bundle = outer_bundle(0, self.fixture.seed)
        bundle.q_material[:] = 0
        table = V3.build_proposal_table(bundle)
        _, active = V3.make_features(bundle, table, "TM", "aligned")
        proposal = np.arange(len(active)) % 2 == 0
        candidate_probability = np.zeros(len(active), np.float32)
        deployed = V3.context_decisions(
            candidate_probability, active, table, 0.5, 0.5, proposal
        )
        self.assertFalse(active.any())
        np.testing.assert_array_equal(deployed, proposal)

    def test_rescue_and_veto_are_separate_heads(self) -> None:
        bundles, _ = self.load_formal()
        tables = {bundle.fold: V3.build_proposal_table(bundle) for bundle in bundles}
        gate = V3.fit_gate("TMR", bundles, tables, 1e-3, self.fixture.seed)
        self.assertIsNot(gate.rescue, gate.veto)
        self.assertEqual(gate.rescue.proposal_type, "rescue")
        self.assertEqual(gate.veto.proposal_type, "veto")

    def test_negative_controls_reuse_one_checkpoint(self) -> None:
        def target_loader(fold: int, **kwargs: object) -> object:
            kwargs["access_log"].append({
                "stage": kwargs["purpose"], "fold": fold, "labels_loaded": True
            })
            return outer_bundle(fold, self.fixture.seed)

        result = V3.run_protocol(args_for(self.fixture), target_loader_fn=target_loader)
        for context in V3.CONTEXTS:
            evaluation = result["target_evaluation"][context]
            checkpoint = evaluation["checkpoint"]["checkpoint_sha256"]
            self.assertTrue(evaluation["negative_controls_reuse_same_checkpoint"])
            self.assertTrue(all(
                item["checkpoint_sha256"] == checkpoint
                for item in evaluation["candidate_controls"].values()
            ))

    def test_label_shuffle_failure_blocks_claim_and_manuscript_pass(self) -> None:
        def target_loader(fold: int, **kwargs: object) -> object:
            kwargs["access_log"].append({
                "stage": kwargs["purpose"], "fold": fold, "labels_loaded": True
            })
            return outer_bundle(fold, self.fixture.seed)

        with patch.object(V3, "label_shuffle_sanity", return_value=True):
            result = V3.run_protocol(args_for(self.fixture), target_loader_fn=target_loader)
        self.assertFalse(result["manuscript_pass"])
        for context in ("TM", "TR", "TMR"):
            selection = result["target_evaluation"][context]["selection"]
            self.assertFalse(selection["claim_pass"])
            self.assertTrue(selection["label_shuffle_claim_pass"])

    def test_cross_outer_mode_is_permanently_exploratory(self) -> None:
        def loader(fold: int, **kwargs: object) -> object:
            kwargs["access_log"].append({
                "stage": kwargs["purpose"], "fold": fold, "labels_loaded": True
            })
            return outer_bundle(fold, self.fixture.seed, prefix="cross")

        result = V3.run_protocol(
            args_for(self.fixture, "cross_outer_exploratory"), target_loader_fn=loader
        )
        self.assertTrue(result["exploratory_only"])
        self.assertFalse(result["manuscript_pass"])
        self.assertEqual(
            result["manuscript_pass_prohibited_reason"],
            "cross_outer_proposer_geography_not_proven_independent",
        )
        strict = json.dumps(V3.json_safe(result), allow_nan=False)
        self.assertIn('"manuscript_pass": false', strict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
