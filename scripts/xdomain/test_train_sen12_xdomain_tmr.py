#!/usr/bin/env python3
"""Small CPU contract tests for the role-aware Sen12 TMR trainer."""

from __future__ import annotations

import unittest
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_sen12_xdomain_tmr as tmr
import train_sen12_xdomain_geophysadapter as baseline


class DummyVisual(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(6, hidden, 1)
        self.logits = nn.Conv2d(hidden, 1, 1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        feature = self.projection(torch.cat([pre, post], dim=1))
        return self.logits(feature), feature


class TMRContractTests(unittest.TestCase):
    def test_train_only_statistics_and_unknown_lithology(self) -> None:
        schema = {
            "material": {"continuous_columns": ["m"], "q_columns": ["q_M_any"]},
            "trigger": {"continuous_columns": ["r"], "q_column": "q_R"},
        }
        rows = {
            "train_a": {"m": "1", "r": "10", "q_M_any": "1", "q_R": "1", "lithology_class": "a", "trigger_family": "rainfall"},
            "train_b": {"m": "3", "r": "14", "q_M_any": "1", "q_R": "1", "lithology_class": "b", "trigger_family": "earthquake"},
            "held_out": {"m": "1000", "r": "2000", "q_M_any": "1", "q_R": "1", "lithology_class": "new", "trigger_family": "volcanic"},
        }
        contract = tmr.fit_tmr_feature_contract(rows, schema, ["train_a", "train_b"], "q_M_any", 5.0)
        self.assertEqual(contract.material_mean, (2.0,))
        self.assertEqual(contract.trigger_mean, (12.0,))
        encoded, _ = contract.encode_material(rows["held_out"])
        self.assertEqual(encoded[0], 5.0)
        self.assertTrue(torch.equal(torch.from_numpy(encoded[1:]), torch.zeros(2)))
        trigger, _ = contract.encode_trigger(rows["held_out"])
        self.assertEqual(contract.trigger_family_vocabulary, ("earthquake", "rainfall"))
        self.assertTrue(torch.equal(torch.from_numpy(trigger[1:]), torch.zeros(2)))

    def _model(self, mode: str) -> tmr.RoleAwareTMRAdapter:
        return tmr.RoleAwareTMRAdapter(
            DummyVisual(hidden=8), terrain_channels=2, hidden=8, terrain_base=8,
            alpha_max=2.0, mode=mode, material_dim=4, trigger_dim=3,
            modulator_hidden=4, beta_m=0.5, beta_r=0.5,
        ).eval()

    def test_q_zero_is_exact_identity(self) -> None:
        model = self._model("full_tmr")
        material = torch.randn(3, 4)
        trigger = torch.randn(3, 3)
        g_m, g_r = model.multipliers(material, torch.zeros(3), trigger, torch.zeros(3))
        self.assertTrue(torch.equal(g_m, torch.ones_like(g_m)))
        self.assertTrue(torch.equal(g_r, torch.ones_like(g_r)))

    def test_inactive_controls_are_exact_aligned_identity(self) -> None:
        model = self._model("terrain")
        batch, height, width = 2, 16, 16
        pre, post = torch.rand(batch, 3, height, width), torch.rand(batch, 3, height, width)
        terrain, donor_terrain = torch.rand(batch, 2, height, width), torch.rand(batch, 2, height, width)
        q_t, donor_q_t = torch.ones(batch, 1, height, width), torch.ones(batch, 1, height, width)
        material, material_alt = torch.rand(batch, 4), torch.rand(batch, 4)
        trigger, trigger_alt = torch.rand(batch, 3), torch.rand(batch, 3)
        q = torch.ones(batch)
        outputs = model.forward_controls(
            pre, post, terrain, q_t, donor_terrain, donor_q_t,
            material, q, trigger, q,
            material_alt, q, material_alt, q,
            trigger_alt, q, trigger_alt, q, trigger_alt, q,
        )
        for control in ("material_shuffled", "material_donor", "material_constant",
                        "trigger_event_shuffled", "trigger_donor", "trigger_wrong_family", "trigger_constant"):
            self.assertTrue(torch.equal(outputs[control][0], outputs["aligned"][0]), control)
        self.assertTrue(torch.equal(outputs["zero_terrain"][0], outputs["visual_anchor"][0]))
        self.assertFalse(any(parameter.requires_grad for parameter in model.visual.parameters()))

    def test_zero_modulation_matches_frozen_terrain_parent(self) -> None:
        parent = baseline.TerrainCorrectionAdapter(
            DummyVisual(hidden=8), terrain_channels=2, hidden=8, terrain_base=8, alpha_max=2.0
        ).eval()
        child = self._model("full_tmr")
        missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
        self.assertTrue(all(key.startswith(("material_head.", "trigger_head.")) for key in missing))
        self.assertEqual(unexpected, [])
        pre, post = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
        terrain, q_t = torch.rand(2, 2, 16, 16), torch.ones(2, 1, 16, 16)
        with torch.no_grad():
            parent_logits, _ = parent(pre, post, terrain, q_t)
            child_logits, _ = child(
                pre, post, terrain, q_t, torch.rand(2, 4), torch.zeros(2),
                torch.rand(2, 3), torch.zeros(2),
            )
        self.assertTrue(torch.equal(parent_logits, child_logits))

    def test_beta_cannot_reverse_terrain_direction(self) -> None:
        args = Namespace(
            device="cuda", h5=Path("missing.h5"), split_csv=Path("missing.csv"),
            sidecar=Path("missing-sidecar.csv"), sidecar_schema=Path("missing-schema.json"),
            visual_checkpoint=Path("missing-visual.pt"), terrain_checkpoint=Path("missing-terrain.pt"),
            mode="full_tmr", epochs=1, batch_size=1, alpha_max=2.0,
            beta_m=1.01, beta_r=0.5, modulator_hidden=4, feature_z_clip=5.0,
            lr=1e-3, outdir=Path("unused"),
        )
        with patch.object(tmr.torch.cuda, "is_available", return_value=True), \
             patch.object(tmr.torch.cuda, "device_count", return_value=1), \
             patch.object(Path, "is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, r"must lie in \[0, 1\]"):
                tmr.validate_args(args)


if __name__ == "__main__":
    unittest.main()
