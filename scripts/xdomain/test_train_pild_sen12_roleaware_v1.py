#!/usr/bin/env python3
"""CPU tests for the unified PILD + Sen12 role-aware trainer."""

from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_pild_sen12_roleaware_v1 as trainer
import analyze_pild_sen12_roleaware_v1 as analyzer
from pild_roleaware_material import MATERIAL_FEATURE_COUNT
from pild_roleaware_trigger import PILDRoleAwareTrigger, TriggerGateConfig
from sen12_terrain_v2 import BoundedTerrainAdapterV2


class DummyVisual(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.decoder = nn.Conv2d(6, channels, 1)
        self.logits = nn.Conv2d(channels, 1, 1)

    def forward(self, optical, temporal_coords, location_coords):
        del temporal_coords, location_coords
        image = optical[:, :, 0]
        feature = torch.tanh(self.decoder(image))
        return {"logits": self.logits(feature), "visual_feature": feature}


def make_batch(batch_size: int = 4, height: int = 12, width: int = 10):
    generator = torch.Generator().manual_seed(20260722)
    features = torch.randn(batch_size, 3, generator=generator)
    features[1] = features[0]
    features[3] = features[2]
    terrain = torch.randn(batch_size, 9, height, width, generator=generator)
    mask = (torch.rand(batch_size, 1, height, width, generator=generator) > 0.8).float()
    return {
        "optical": torch.rand(batch_size, 6, 4, height, width, generator=generator),
        "temporal_coords": torch.zeros(batch_size, 4, 2),
        "location_coords": torch.zeros(batch_size, 2),
        "terrain": terrain,
        "q_t": torch.ones(batch_size, 1, height, width),
        "terrain_donor": terrain.flip(0).clone(),
        "terrain_donor_q": torch.ones(batch_size, 1, height, width),
        "terrain_donor_sample_id": [f"terrain-donor-{index}" for index in range(batch_size)],
        "terrain_donor_event_id": ["event-b", "event-b", "event-a", "event-a"],
        "role_material_features": torch.randn(
            batch_size, MATERIAL_FEATURE_COUNT, generator=generator
        ),
        "material_shuffle_features": torch.randn(
            batch_size, MATERIAL_FEATURE_COUNT, generator=generator
        ),
        "q_material": torch.ones(batch_size),
        "material_shuffle_q": torch.ones(batch_size),
        "material_donor_sample_id": [f"material-donor-{index}" for index in range(batch_size)],
        "trigger_features": features,
        "trigger_wrong_time_features": features.clone(),
        "trigger_shuffle_features": features.flip(0),
        "trigger_shuffle_q": torch.ones(batch_size),
        "trigger_donor_event_id": ["event-b", "event-b", "event-a", "event-a"],
        "q_trigger": torch.ones(batch_size),
        "canonical_event_id": ["event-a", "event-a", "event-b", "event-b"],
        "sample_id": [f"sample-{index}" for index in range(batch_size)],
        "dataset_id": ["dataset"] * batch_size,
        "source_id": ["source"] * batch_size,
        "mask": mask,
        "valid_mask": torch.ones_like(mask),
    }


def build_model(variant: str) -> trainer.RoleAwareGeoPhysAdapter:
    terrain = None
    if variant != "V":
        terrain = BoundedTerrainAdapterV2(
            9, 8, trainer.COMMON_TERRAIN9_SCALE_GROUPS, alpha_max=1.0
        )
    trigger = None
    if "R" in variant:
        trigger = PILDRoleAwareTrigger(
            TriggerGateConfig(feature_dim=3, hidden_dim=5)
        )
    return trainer.RoleAwareGeoPhysAdapter(
        DummyVisual(),
        variant,
        visual_channels=8,
        terrain_adapter=terrain,
        trigger_module=trigger,
    )


class RoleBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_five_variants_have_a_minimal_forward(self):
        batch = make_batch()
        for variant in trainer.VARIANTS:
            with self.subTest(variant=variant):
                model = build_model(variant)
                output = model(batch)
                self.assertEqual(output["logits"].shape, (4, 1, 12, 10))
                self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_trainability_follows_hierarchical_role_contract(self):
        expected_prefixes = {
            "V": ("visual.decoder",),
            "VT": ("terrain_adapter",),
            "VTM": ("material_module",),
            "VTR": ("trigger_module",),
            "VTMR": ("material_module", "trigger_module"),
        }
        for variant, prefixes in expected_prefixes.items():
            model = build_model(variant)
            trainable = [name for name, value in model.named_parameters() if value.requires_grad]
            self.assertTrue(trainable, variant)
            self.assertTrue(
                all(any(name.startswith(prefix) for prefix in prefixes) for name in trainable),
                (variant, trainable),
            )

    def test_material_uses_explicit_common_terrain9_response_groups(self):
        model = build_model("VTM")
        self.assertEqual(
            model.material_module.response_names,
            ("slope", "curvature", "relief"),
        )
        self.assertEqual(
            model.material_module.response_groups,
            ((1,), (4,), (5, 6, 7, 8)),
        )
        self.assertNotEqual(
            model.material_module.response_groups,
            ((0,), (1,), (2,)),
            "the trainer must not fall back to Material module placeholder indices",
        )

    def test_common_terrain9_scale_groups_are_local_complete_and_in_range(self):
        self.assertEqual(
            trainer.COMMON_TERRAIN9_NAMES,
            (
                "elevation",
                "slope_deg",
                "aspect_sin",
                "aspect_cos",
                "laplacian_curvature",
                "tpi_90m",
                "tpi_300m",
                "ruggedness_90m",
                "local_relief_300m",
            ),
        )
        groups = trainer.COMMON_TERRAIN9_SCALE_GROUPS
        indices = groups.fine + groups.meso + groups.macro
        self.assertEqual(set(indices), set(range(9)))
        self.assertEqual(len(indices), len(set(indices)))
        self.assertTrue(all(0 <= index < 9 for index in indices))
        self.assertEqual(groups.fine, (1, 2, 3, 4, 7))
        self.assertEqual(groups.meso, (5, 6))
        self.assertEqual(groups.macro, (0, 8))

    def test_material_cannot_act_without_terrain_residual(self):
        batch = make_batch()
        model = build_model("VTM")
        with torch.no_grad():
            model.material_module.interaction_head.response.weight.fill_(2.0)
            model.terrain_adapter.terrain.output.weight.zero_()
        aligned = model(batch, material_context=trainer.MATERIAL_ALIGNED)
        shuffled = model(batch, material_context=trainer.MATERIAL_SHUFFLE)
        self.assertTrue(torch.equal(aligned["logits"], aligned["visual_logits"]))
        self.assertTrue(torch.equal(shuffled["logits"], shuffled["visual_logits"]))
        self.assertTrue(
            torch.equal(aligned["material_correction"], torch.zeros_like(aligned["material_correction"]))
        )

    def test_material_q_zero_is_exact_vt_identity_per_sample(self):
        batch = make_batch()
        batch["q_material"] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        vt = build_model("VT")
        vtm = build_model("VTM")
        vtm.visual.load_state_dict(vt.visual.state_dict())
        vtm.terrain_adapter.load_state_dict(vt.terrain_adapter.state_dict())
        with torch.no_grad():
            vtm.material_module.interaction_head.response.weight.fill_(1.5)
        baseline = vt(batch)["logits"]
        output = vtm(batch)["logits"]
        self.assertTrue(torch.equal(output[0], baseline[0]))
        self.assertTrue(torch.equal(output[2], baseline[2]))

    def test_trigger_q_zero_is_exact_vt_identity_and_uses_visual_uncertainty(self):
        batch = make_batch()
        batch["q_trigger"] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        vt = build_model("VT")
        vtr = build_model("VTR")
        vtr.visual.load_state_dict(vt.visual.state_dict())
        vtr.terrain_adapter.load_state_dict(vt.terrain_adapter.state_dict())
        with torch.no_grad():
            vtr.trigger_module.calibrator[-1].bias.copy_(torch.tensor([2.0, 1.5, -1.0]))
        baseline = vt(batch)["logits"]
        output = vtr(batch)
        self.assertTrue(torch.equal(output["logits"][:2], baseline[:2]))
        self.assertGreater(
            float(output["trigger_correction"][2:].detach().abs().max()), 0.0
        )
        self.assertEqual(
            output["trigger_audit"]["audit"]["trigger_spatial_source"],
            "detached_visual_uncertainty_only",
        )

    def test_zero_q_controls_are_exact_parent_identity(self):
        batch = make_batch()
        vt = build_model("VT")
        full = build_model("VTMR")
        full.visual.load_state_dict(vt.visual.state_dict())
        full.terrain_adapter.load_state_dict(vt.terrain_adapter.state_dict())
        with torch.no_grad():
            full.material_module.interaction_head.response.weight.fill_(2.0)
            full.trigger_module.calibrator[-1].bias.copy_(torch.tensor([2.0, 1.0, -1.0]))
        parent = vt(batch)["logits"]
        output = full(
            batch,
            material_context=trainer.MATERIAL_ZERO_Q,
            trigger_context=trainer.TRIGGER_ZERO_Q,
        )["logits"]
        self.assertTrue(torch.equal(output, parent))

    def test_terrain_zero_q_is_exact_visual_identity(self):
        batch = make_batch()
        model = build_model("VT")
        output = model(batch, terrain_context=trainer.TERRAIN_ZERO)
        self.assertTrue(torch.equal(output["logits"], output["visual_logits"]))
        self.assertTrue(
            torch.equal(
                output["terrain_correction"],
                torch.zeros_like(output["terrain_correction"]),
            )
        )

    def test_mr_variants_expose_frozen_vt_reference_logits(self):
        batch = make_batch()
        for variant in ("VTM", "VTR", "VTMR"):
            with self.subTest(variant=variant):
                model = build_model(variant)
                output = model(batch)
                expected = output["visual_logits"] + output["terrain_correction"]
                self.assertTrue(torch.equal(output["reference_logits"], expected))

    def test_preservation_penalty_is_invariant_to_v_when_vt_reference_is_fixed(self):
        target = torch.ones(1, 1, 2, 2)
        valid = torch.ones_like(target)
        final = torch.full_like(target, 2.0, requires_grad=True)
        reference = torch.full_like(target, 1.0)
        common = {
            "logits": final,
            "reference_logits": reference,
        }
        first = trainer.segmentation_loss(
            {**common, "visual_logits": torch.full_like(target, -8.0)},
            target,
            valid,
            torch.ones(1, 1, 1, 1),
            "VTM",
            0.5,
        )
        second = trainer.segmentation_loss(
            {**common, "visual_logits": torch.full_like(target, 8.0)},
            target,
            valid,
            torch.ones(1, 1, 1, 1),
            "VTM",
            0.5,
        )
        self.assertTrue(torch.equal(first, second))

    def test_error_flow_for_vtm_is_measured_against_vt_not_v(self):
        class FixedVTM(nn.Module):
            variant = "VTM"
            _terrain_inputs = staticmethod(trainer.RoleAwareGeoPhysAdapter._terrain_inputs)

            def forward(self, batch, **_kwargs):
                target = batch["mask"]
                visual = torch.full_like(target, -10.0)
                reference = torch.where(target > 0, 10.0, -10.0)
                zeros = torch.zeros_like(target)
                return {
                    "logits": reference,
                    "visual_logits": visual,
                    "reference_logits": reference,
                    "terrain_correction": reference - visual,
                    "material_correction": zeros,
                    "trigger_correction": zeros,
                }

        batch = make_batch()
        batch["mask"].zero_()
        batch["mask"][:, :, 0, 0] = 1.0
        rows, _, _ = trainer.evaluate(
            FixedVTM(),
            [batch],
            "cpu",
            0.5,
            0.6,
            "test",
            trainer.EvaluationContext("aligned"),
            7,
            "d" * 64,
            {"component": "e" * 64},
        )
        self.assertGreater(sum(row["visual_errors"] for row in rows), 0)
        self.assertEqual(sum(row["reference_errors"] for row in rows), 0)
        self.assertEqual(sum(row["corrected"] for row in rows), 0)
        self.assertEqual(sum(row["harmed"] for row in rows), 0)
        self.assertEqual({row["reference_condition"] for row in rows}, {"VT"})

    def test_all_negative_control_paths_run(self):
        batch = make_batch()
        model = build_model("VTMR")
        for terrain_context in trainer.TERRAIN_CONTEXTS:
            for material_context in trainer.MATERIAL_CONTEXTS:
                for trigger_context in trainer.TRIGGER_CONTEXTS:
                    with self.subTest(
                        terrain=terrain_context,
                        material=material_context,
                        trigger=trigger_context,
                    ):
                        output = model(
                            batch,
                            terrain_context=terrain_context,
                            material_context=material_context,
                            trigger_context=trigger_context,
                        )
                        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_zero_pad_shift_never_wraps(self):
        value = torch.zeros(1, 1, 6, 6)
        value[..., -1, -1] = 7.0
        shifted = trainer.zero_pad_spatial_shift(value, 2)
        self.assertEqual(float(shifted.sum()), 0.0)
        value.zero_()
        value[..., 0, 0] = 3.0
        shifted = trainer.zero_pad_spatial_shift(value, 2)
        self.assertEqual(float(shifted[..., 2, 2]), 3.0)
        self.assertEqual(float(shifted[..., :2, :].sum()), 0.0)

    def test_terrain_donor_is_deterministic_nonself_and_prefers_same_source(self):
        sample = ("a0", "a1", "b0", "c0")
        source = ("A", "A", "B", "C")
        event = ("e0", "e1", "e2", "e3")
        first = trainer.deterministic_terrain_donors(sample, source, event, 17)
        second = trainer.deterministic_terrain_donors(sample, source, event, 17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(first[0]), 1)
        self.assertEqual(int(first[1]), 0)
        self.assertTrue(all(int(donor) != index for index, donor in enumerate(first)))

    def test_frozen_evaluation_context_inventory(self):
        expected = {
            "V": {"aligned"},
            "VT": {
                "aligned",
                "terrain-zero",
                "terrain-shift32-zero-pad",
                "terrain-roll64-circular",
                "terrain-other-source-or-event-donor",
            },
            "VTM": {
                "aligned",
                "material-aligned",
                "material-shuffle",
                "material-zero-q",
            },
            "VTR": {
                "aligned",
                "trigger-aligned",
                "trigger-wrong-time",
                "trigger-event-shuffle",
                "trigger-zero-q",
            },
            "VTMR": {
                "aligned",
                "material-shuffle",
                "material-zero-q",
                "trigger-wrong-time",
                "trigger-event-shuffle",
                "trigger-zero-q",
                "material-trigger-both-zero-q",
            },
        }
        for variant, names in expected.items():
            contexts = trainer.evaluation_contexts_for_variant(variant)
            self.assertEqual({context.name for context in contexts}, names)
            self.assertEqual(len(contexts), len(names))

    def test_all_inference_contexts_share_checkpoint_and_component_identity(self):
        batch = make_batch()
        model = build_model("VTMR")
        checkpoint_hash = "a" * 64
        component_hashes = {"visual_decoder": "b" * 64, "terrain_adapter": "c" * 64}
        observed = set()
        for context in trainer.evaluation_contexts_for_variant("VTMR"):
            sample_rows, event_rows, corpus = trainer.evaluate(
                model,
                [batch],
                "cpu",
                0.5,
                0.6,
                "test",
                context,
                20260722,
                checkpoint_hash,
                component_hashes,
            )
            self.assertTrue(sample_rows)
            self.assertTrue(event_rows)
            self.assertTrue(set(analyzer.REQUIRED_COLUMNS).issubset(sample_rows[0]))
            for row in [*sample_rows, *event_rows]:
                self.assertEqual(row["checkpoint_sha256"], checkpoint_hash)
                observed.add(row["component_sha256_json"])
                self.assertEqual(row["evaluation_context"], context.name)
                self.assertIn("effective_q", row)
            self.assertEqual(corpus["checkpoint_sha256"], checkpoint_hash)
        self.assertEqual(len(observed), 1)

    def test_train_and_checkpoint_selection_use_aligned_context_only(self):
        class Sampler:
            def set_epoch(self, epoch):
                self.epoch = epoch

        batch = make_batch()
        model = build_model("VTMR")
        calls = []
        original = model.forward

        def recording_forward(*args, **kwargs):
            calls.append(dict(kwargs))
            return original(*args, **kwargs)

        model.forward = recording_forward
        args = SimpleNamespace(
            lr=1.0e-3,
            weight_decay=0.0,
            device="cpu",
            grad_clip=1.0,
            variant="VTMR",
            max_steps=1,
            epochs=1,
        )
        trainer.train_model(
            model,
            [batch],
            Sampler(),
            [batch],
            args,
            1.0,
            0.5,
            lambda _message: None,
        )
        self.assertTrue(calls)
        expected = {
            "terrain_context": trainer.TERRAIN_ALIGNED,
            "material_context": trainer.MATERIAL_ALIGNED,
            "trigger_context": trainer.TRIGGER_ALIGNED,
        }
        self.assertTrue(all(call == expected for call in calls), calls)

    def test_cli_rejects_training_context_controls(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            trainer.parse_args(
                ["--variant", "VTM", "--material-context", trainer.MATERIAL_SHUFFLE]
            )
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            trainer.parse_args(
                ["--variant", "VTR", "--trigger-context", trainer.TRIGGER_WRONG_TIME]
            )


class NormalizationAndSchemaTests(unittest.TestCase):
    @staticmethod
    def _schema_fixture(root: Path) -> tuple[Path, Path]:
        manifest = root / "manifest.csv"
        rows = []
        for index in range(4):
            row = {column: "" for column in trainer.REQUIRED_MANIFEST_COLUMNS}
            row.update(
                {
                    "dataset_id": "dataset",
                    "source_id": "source",
                    "source_event_id": f"source-event-{index}",
                    "canonical_event_id": f"event-{min(index, 2)}",
                    "sample_id": f"sample-{index}",
                    "base_h5_path": str(root / "missing-base.h5"),
                    "base_h5_index": 0,
                    "optical_h5_path": str(root / "missing-optical.h5"),
                    "optical_h5_index": 0,
                    "terrain_h5_path": str(root / "missing-terrain.h5"),
                    "terrain_h5_index": 0,
                    "terrain_channel_indices": "0;1;2;3;4;5;6;7;8",
                    "material_registry_path": "",
                    "material_registry_index": -1,
                    "trigger_registry_path": "",
                    "trigger_registry_index": -1,
                    "core_assets_ready": False,
                    "full_tmr_assets_ready": False,
                }
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv(manifest, index=False)
        summary = root / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "validation_status": "PASS",
                    "terrain_contract": {"names": list(trainer.COMMON_TERRAIN9_NAMES)},
                    "readiness": {
                        "core_training_ready": False,
                        "blockers": ["missing fixture cache"],
                    },
                    "outputs": {},
                }
            ),
            encoding="utf-8",
        )
        return manifest, summary

    def test_lodo_excluded_alias_does_not_count_as_active_event_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, summary = self._schema_fixture(root)
            split = root / "split.csv"
            pd.DataFrame(
                [
                    {"fold_id": "fold", "sample_id": "sample-0", "canonical_event_id": "event-0", "role": "train"},
                    {"fold_id": "fold", "sample_id": "sample-1", "canonical_event_id": "event-1", "role": "val"},
                    {"fold_id": "fold", "sample_id": "sample-2", "canonical_event_id": "event-2", "role": "test"},
                    {"fold_id": "fold", "sample_id": "sample-3", "canonical_event_id": "event-2", "role": "excluded"},
                ]
            ).to_csv(split, index=False)
            result = trainer.validate_protocol_schema(manifest, summary, split, "fold")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["excluded_samples"], 1)

    def test_trigger_is_event_median_aggregated_then_broadcast(self):
        values = np.asarray(
            [[1, 10, -9], [100, 20, 80], [3, 30, -27], [999, 999, 0]],
            np.float32,
        )
        aggregated, quality = trainer.aggregate_trigger_by_event(
            values,
            np.asarray([1, 1, 1, 0], np.float32),
            ["event-a", "event-a", "event-a", "event-b"],
        )
        expected = np.asarray([3, 20, -9], np.float32)
        np.testing.assert_allclose(aggregated[:3], np.tile(expected, (3, 1)))
        np.testing.assert_allclose(aggregated[3], np.zeros(3, np.float32))
        np.testing.assert_array_equal(quality, np.asarray([1, 1, 1, 0], np.float32))

    def test_trigger_normalizer_is_event_balanced_and_outer_only(self):
        values = np.asarray([[1, 0, 1], [1, 0, 1], [5, 2, 3], [999, 999, 0]], np.float32)
        q = np.asarray([1, 1, 1, 0], np.float32)
        normalizer = trainer.OuterTrainTriggerNormalizer.fit(
            values, q, ["a", "a", "b", "heldout"]
        )
        np.testing.assert_allclose(normalizer.mean, np.asarray([3, 1, 2], np.float32))
        self.assertEqual(normalizer.audit()["n_events"], 2)

    def test_validate_only_schema_does_not_open_missing_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            rows = []
            for index, role in enumerate(("train", "val", "test")):
                row = {column: "" for column in trainer.REQUIRED_MANIFEST_COLUMNS}
                row.update(
                    {
                        "dataset_id": "dataset",
                        "source_id": "source",
                        "source_event_id": f"source-event-{index}",
                        "canonical_event_id": f"event-{index}",
                        "sample_id": f"sample-{index}",
                        "base_h5_path": str(root / "missing-base.h5"),
                        "base_h5_index": 0,
                        "optical_h5_path": str(root / "missing-optical.h5"),
                        "optical_h5_index": 0,
                        "terrain_h5_path": str(root / "missing-terrain.h5"),
                        "terrain_h5_index": 0,
                        "terrain_channel_indices": "0;1;2;3;4;5;6;7;8",
                        "material_registry_path": "",
                        "material_registry_index": -1,
                        "trigger_registry_path": "",
                        "trigger_registry_index": -1,
                        "core_assets_ready": False,
                        "full_tmr_assets_ready": False,
                    }
                )
                rows.append(row)
            pd.DataFrame(rows).to_csv(manifest, index=False)
            split = root / "split.csv"
            pd.DataFrame(
                [
                    {
                        "fold_id": "fold",
                        "sample_id": f"sample-{index}",
                        "canonical_event_id": f"event-{index}",
                        "role": role,
                        "role_reason": "test",
                    }
                    for index, role in enumerate(("train", "val", "test"))
                ]
            ).to_csv(split, index=False)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "validation_status": "PASS",
                        "terrain_contract": {"names": list(trainer.COMMON_TERRAIN9_NAMES)},
                        "readiness": {
                            "core_training_ready": False,
                            "blockers": ["missing fixture cache"],
                        },
                        "outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            result = trainer.validate_protocol_schema(manifest, summary, split, "fold")
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(result["readiness"]["core_training_ready"])


if __name__ == "__main__":
    unittest.main()
