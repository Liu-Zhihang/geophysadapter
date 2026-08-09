#!/usr/bin/env python3
"""CPU tests for the PILD/Sen12 parent OOF pixel exporter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_pild_sen12_parent_oof_logits_v1 as exporter

class ExporterContractTests(unittest.TestCase):
    def test_formal_selection_requires_five_common_seeds(self) -> None:
        folds = ("fold-a", "fold-b")
        discovered = {
            seed: {fold: Path(f"/{seed}/{fold}") for fold in folds}
            for seed in (1, 2, 3, 4)
        }
        with self.assertRaisesRegex(exporter.ExportContractError, "at least 5"):
            exporter.select_formal_run_groups(discovered, folds, (), 5)

    def test_formal_seed_groups_never_mix(self) -> None:
        folds = ("fold-a", "fold-b")
        discovered = {
            seed: {fold: Path(f"/{seed}/{fold}") for fold in folds}
            for seed in (1, 2, 3, 4, 5)
        }
        groups = exporter.select_formal_run_groups(discovered, folds, (), 5)
        self.assertEqual(set(groups), {1, 2, 3, 4, 5})
        self.assertTrue(all(all(str(seed) in str(path) for path in paths) for seed, paths in groups.items()))

    def test_nonformal_requires_explicit_limit(self) -> None:
        args = SimpleNamespace(
            manifest=Path("missing"), protocol_summary=Path("missing"),
            split=Path("missing"), run_root=Path("missing"), outdir=Path("out"),
            batch_size=1, num_workers=0, min_seeds=5, max_samples=0,
            nonformal=True, run_dir=[Path("run")],
        )
        with self.assertRaisesRegex(ValueError, "max-samples"):
            exporter.validate_args(args)

    def test_physical_identity_loader_does_not_require_q_r(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "trigger.csv"
            pd.DataFrame({
                "sample_id": ["s0", "s1"],
                "physical_event_id": ["p0", "p1"],
            }).to_csv(registry, index=False)
            frame = pd.DataFrame({
                "sample_id": ["s0", "s1"],
                "trigger_registry_path": [str(registry)] * 2,
                "trigger_registry_index": [0, 1],
            })
            observed = exporter._load_physical_identities(frame)
            self.assertEqual(observed.tolist(), ["p0", "p1"])

    def test_compressed_archive_matches_trigger_evaluator_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold.npz"
            logits = np.zeros((2, 1, 3, 4), np.float32)
            labels = np.ones_like(logits, np.uint8)
            valid = np.ones_like(logits, np.uint8)
            exporter._write_prediction_archive(
                path, sample_ids=["s0", "s1"], canonical_event_ids=["c0", "c1"],
                physical_event_ids=["p0", "p1"], logits=logits, labels=labels,
                valid=valid, compression="compressed",
            )
            with np.load(path, allow_pickle=False) as payload:
                self.assertTrue({"sample_ids", "event_ids", "logits", "labels", "valid"} <= set(payload.files))
                self.assertEqual(payload["logits"].shape, (2, 1, 3, 4))
                self.assertEqual(payload["physical_event_ids"].tolist(), ["p0", "p1"])

    def test_pixel_dataset_reads_no_trigger_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = "sample-0"
            base = root / "base.h5"
            optical = root / "optical.h5"
            terrain = root / "terrain.h5"
            with h5py.File(base, "w") as handle:
                handle.create_dataset("sample_id", data=np.asarray([sample], dtype="S16"))
                handle.create_dataset("mask", data=np.zeros((1, 1, 4, 4), np.uint8))
                handle.create_dataset("valid_mask", data=np.ones((1, 1, 4, 4), np.uint8))
            with h5py.File(optical, "w") as handle:
                handle.create_dataset("sample_id", data=np.asarray([sample], dtype="S16"))
                handle.create_dataset("optical", data=np.zeros((1, 6, 4, 4, 4), np.uint16))
                handle.create_dataset("optical_valid", data=np.ones((1, 1, 4, 4), np.uint8))
                handle.create_dataset("temporal_coords", data=np.zeros((1, 4, 2), np.float32))
                handle.create_dataset("location_coords", data=np.zeros((1, 2), np.float32))
            with h5py.File(terrain, "w") as handle:
                handle.create_dataset("sample_id", data=np.asarray([sample], dtype="S16"))
                handle.create_dataset("terrain", data=np.zeros((1, 9, 4, 4), np.float32))
                handle.create_dataset("terrain_valid", data=np.ones((1, 1, 4, 4), np.uint8))
            trigger = root / "trigger.csv"
            # Deliberately no q_R or rainfall columns.
            pd.DataFrame({"sample_id": [sample], "physical_event_id": ["physical-0"]}).to_csv(trigger, index=False)
            manifest = root / "manifest.csv"
            pd.DataFrame({
                "sample_id": [sample], "canonical_event_id": ["canonical-0"],
                "base_h5_path": [str(base)], "base_h5_index": [0],
                "optical_h5_path": [str(optical)], "optical_h5_index": [0],
                "terrain_h5_path": [str(terrain)], "terrain_h5_index": [0],
                "terrain_channel_indices": ["0;1;2;3;4;5;6;7;8"],
                "trigger_registry_path": [str(trigger)], "trigger_registry_index": [0],
                "core_assets_ready": [1],
            }).to_csv(manifest, index=False)
            split = root / "split.csv"
            pd.DataFrame({
                "fold_id": ["fold"], "sample_id": [sample],
                "canonical_event_id": ["canonical-0"], "role": ["test"],
                "role_reason": ["heldout"],
            }).to_csv(split, index=False)
            dataset = exporter.ParentOOFPixelDataset(
                manifest, split, "fold", variant="V", normalization={}, max_samples=0
            )
            item = dataset[0]
            self.assertEqual(item["physical_event_id"], "physical-0")
            self.assertNotIn("q_trigger", item)
            self.assertNotIn("trigger_features", item)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
