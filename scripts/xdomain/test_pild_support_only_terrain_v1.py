#!/usr/bin/env python3
"""Contract tests for the PILD support-only Terrain transfer chain."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pild_support_only_additive_v1 import (  # noqa: E402
    aggregate_samples_to_events,
    fuse_logits,
    select_on_validation,
    validate_terrain_checkpoint,
)
from train_pild_sen12_roleaware_v1 import (  # noqa: E402
    COMMON_TERRAIN9_NAMES,
    COMMON_TERRAIN9_SCALE_GROUPS,
    tensor_sha256,
)
from train_pild_support_only_terrain_v1 import (  # noqa: E402
    TerrainOnlyDataset,
    validate_parent_v_checkpoint,
)
from pild_sen12_training_loader_v2 import sha256_file  # noqa: E402


class ParentCheckpointTests(unittest.TestCase):
    def _write_parent(self, path: Path) -> dict:
        visual_state = {"weight": torch.tensor([[1.0, 2.0]])}
        payload = {
            "variant": "V",
            "identity": {
                "manifest_sha256": "m" * 64,
                "split_sha256": "s" * 64,
                "fold_id": "fold0",
                "seed": 7,
                "prithvi_checkpoint_sha256": "p" * 64,
            },
            "components": {"visual_decoder": visual_state},
            "component_sha256": {
                "visual_decoder": tensor_sha256(visual_state)
            },
            "threshold": 0.41,
            "threshold_source": "visual_validation",
        }
        torch.save(payload, path)
        return payload

    def test_parent_identity_and_component_hash_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.pt"
            payload = self._write_parent(path)
            loaded, receipt = validate_parent_v_checkpoint(
                path,
                manifest_sha256="m" * 64,
                split_sha256="s" * 64,
                fold_id="fold0",
                seed=7,
            )
            self.assertEqual(loaded["threshold"], 0.41)
            self.assertEqual(receipt["checkpoint_sha256"], sha256_file(path))
            payload["components"]["visual_decoder"]["weight"][0, 0] = 99.0
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "tensor hash"):
                validate_parent_v_checkpoint(
                    path,
                    manifest_sha256="m" * 64,
                    split_sha256="s" * 64,
                    fold_id="fold0",
                    seed=7,
                )

    def test_parent_fold_mismatch_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.pt"
            self._write_parent(path)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                validate_parent_v_checkpoint(
                    path,
                    manifest_sha256="m" * 64,
                    split_sha256="s" * 64,
                    fold_id="other",
                    seed=7,
                )


class TerrainOnlyDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_string_dataset(handle: h5py.File, name: str, values: list[str]) -> None:
        dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)

    def test_dataset_does_not_open_optical_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.h5"
            terrain_path = root / "terrain.h5"
            optical_path = root / "must_not_exist.h5"
            with h5py.File(base_path, "w") as handle:
                self._write_string_dataset(handle, "sample_id", ["sample-1"])
                handle.create_dataset(
                    "mask", data=np.ones((1, 8, 8), dtype=np.uint8)
                )
                handle.create_dataset(
                    "valid_mask", data=np.ones((1, 8, 8), dtype=np.uint8)
                )
            with h5py.File(terrain_path, "w") as handle:
                self._write_string_dataset(handle, "sample_id", ["sample-1"])
                handle.create_dataset(
                    "terrain",
                    data=np.arange(9 * 8 * 8, dtype=np.float32).reshape(1, 9, 8, 8),
                )
                handle.create_dataset(
                    "terrain_valid", data=np.ones((1, 8, 8), dtype=np.uint8)
                )
                handle["terrain"][0, 0, 0, 0] = np.nan
            frame = pd.DataFrame(
                [
                    {
                        "sample_id": "sample-1",
                        "dataset_id": "synthetic",
                        "source_id": "source",
                        "source_event_id": "event-source",
                        "canonical_event_id": "event-canonical",
                        "base_h5_path": str(base_path),
                        "base_h5_index": 0,
                        "terrain_h5_path": str(terrain_path),
                        "terrain_h5_index": 0,
                        "terrain_channel_indices": ";".join(map(str, range(9))),
                        "optical_h5_path": str(optical_path),
                    }
                ]
            )
            dataset = TerrainOnlyDataset(SimpleNamespace(frame=frame))
            item = dataset[0]
            self.assertEqual(tuple(item["terrain"].shape), (9, 8, 8))
            self.assertTrue(torch.isfinite(item["terrain"]).all())
            self.assertEqual(float(item["q_t"][0, 0, 0]), 0.0)
            self.assertFalse(optical_path.exists())
            self.assertEqual(set(dataset._h5), {str(base_path), str(terrain_path)})


class FusionContractTests(unittest.TestCase):
    def test_q_zero_is_exact_abstention_and_residual_is_bounded(self) -> None:
        visual = torch.tensor([[[[-2.0, 0.5]]]])
        terrain = torch.tensor([[[[100.0, -100.0]]]])
        q_zero = torch.zeros_like(visual)
        output = fuse_logits(
            visual,
            terrain,
            q_zero,
            alpha=4.0,
            uncertainty_power=2.0,
        )
        torch.testing.assert_close(output, visual, rtol=0, atol=0)
        q_one = torch.ones_like(visual)
        output = fuse_logits(
            visual,
            terrain,
            q_one,
            alpha=4.0,
            uncertainty_power=0.0,
        )
        self.assertTrue(torch.all(torch.abs(output - visual) <= 4.0 + 1e-6))

    def test_terrain_checkpoint_hash_and_protocol_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            summary.write_text(json.dumps({"validation_status": "PASS"}), encoding="utf-8")
            parent_receipt = {
                "checkpoint_sha256": "c" * 64,
                "visual_decoder_sha256": "v" * 64,
                "prithvi_checkpoint_sha256": "p" * 64,
            }
            state = {"weight": torch.ones(1)}
            payload = {
                "schema_version": "pild_support_only_terrain_checkpoint.v1",
                "identity": {
                    "manifest_sha256": "m" * 64,
                    "protocol_summary_sha256": sha256_file(summary),
                    "split_sha256": "s" * 64,
                    "fold_id": "fold0",
                    "seed": 7,
                    "parent_v_checkpoint_sha256": "c" * 64,
                    "parent_visual_decoder_sha256": "v" * 64,
                    "parent_prithvi_checkpoint_sha256": "p" * 64,
                },
                "terrain_state_dict": state,
                "terrain_state_sha256": tensor_sha256(state),
                "terrain_channel_order": list(COMMON_TERRAIN9_NAMES),
                "terrain_scale_groups": {
                    "fine": list(COMMON_TERRAIN9_SCALE_GROUPS.fine),
                    "meso": list(COMMON_TERRAIN9_SCALE_GROUPS.meso),
                    "macro": list(COMMON_TERRAIN9_SCALE_GROUPS.macro),
                },
                "normalization": {
                    "fit_scope": "train-only-valid-terrain-pixels",
                    "feature_names": list(COMMON_TERRAIN9_NAMES),
                    "mean": [0.0] * 9,
                    "scale": [1.0] * 9,
                },
                "best_epoch": 2,
            }
            checkpoint = root / "terrain.pt"
            torch.save(payload, checkpoint)
            _, receipt = validate_terrain_checkpoint(
                checkpoint,
                schema={
                    "manifest_sha256": "m" * 64,
                    "split_sha256": "s" * 64,
                },
                protocol_summary_path=summary,
                fold_id="fold0",
                seed=7,
                parent_receipt=parent_receipt,
            )
            self.assertEqual(receipt["terrain_state_sha256"], tensor_sha256(state))
            payload["terrain_state_dict"]["weight"][0] = 2.0
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "tensor hash"):
                validate_terrain_checkpoint(
                    checkpoint,
                    schema={
                        "manifest_sha256": "m" * 64,
                        "split_sha256": "s" * 64,
                    },
                    protocol_summary_path=summary,
                    fold_id="fold0",
                    seed=7,
                    parent_receipt=parent_receipt,
                )


class TinyValidationDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "optical": torch.zeros(1, 1, 1, 2),
            "temporal_coords": torch.zeros(4, 2),
            "location_coords": torch.zeros(2),
            "terrain": torch.zeros(9, 1, 2),
            "q_t": torch.ones(1, 1, 2),
            "mask": torch.tensor([[[0.0, 1.0]]]),
            "valid_mask": torch.ones(1, 1, 2),
            "sample_id": "sample",
            "dataset_id": "dataset",
            "source_id": "source",
            "source_event_id": "source-event",
            "canonical_event_id": "event",
        }


class FakeVisual(nn.Module):
    def forward(
        self,
        optical: torch.Tensor,
        temporal_coords: torch.Tensor,
        location_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = optical.shape[0]
        logits = torch.tensor([[[[-4.0, 4.0]]]], device=optical.device).repeat(batch, 1, 1, 1)
        return {"logits": logits}


class FakeTerrain(nn.Module):
    def forward(self, terrain: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch = terrain.shape[0]
        logits = torch.tensor([[[[10.0, -10.0]]]], device=terrain.device).repeat(batch, 1, 1, 1)
        return logits, {}


class SelectionAndAggregationTests(unittest.TestCase):
    def test_validation_grid_keeps_identity_when_terrain_is_harmful(self) -> None:
        loader = DataLoader(TinyValidationDataset(), batch_size=1)
        selected, rows = select_on_validation(
            FakeVisual(),
            FakeTerrain(),
            loader,
            threshold=0.5,
            device=torch.device("cpu"),
        )
        self.assertEqual(selected["alpha"], 0.0)
        self.assertEqual(selected["uncertainty_power"], 0.0)
        self.assertEqual(len(rows), 16)
        self.assertTrue(selected["validation_feasible"])

    def test_event_aggregation_sums_paired_counts(self) -> None:
        rows = []
        for sample_id, corrected, harmed in (("a", 3, 1), ("b", 2, 2)):
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_id": "dataset",
                    "source_id": "source",
                    "source_event_id": "raw",
                    "canonical_event_id": "event",
                    "baseline_tp": 5,
                    "baseline_fp": 3,
                    "baseline_fn": 2,
                    "baseline_tn": 10,
                    "adapted_tp": 6,
                    "adapted_fp": 2,
                    "adapted_fn": 1,
                    "adapted_tn": 11,
                    "corrected": corrected,
                    "harmed": harmed,
                    "valid_pixels": 20,
                }
            )
        events = aggregate_samples_to_events(rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["n_samples"], 2)
        self.assertEqual(events[0]["corrected"], 5)
        self.assertEqual(events[0]["harmed"], 3)
        self.assertEqual(events[0]["net_error_reduction"], 2.0)


if __name__ == "__main__":
    unittest.main()
