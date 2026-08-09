#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch
from torch import nn

from sen12_prithvi_v2 import PrithviEO2ChangeModel


class FakePatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = (1, 16, 16)


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 32
        self.patch_embed = FakePatchEmbed()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(24))
        self.weight = nn.Parameter(torch.ones(()))

    def forward_features(self, optical, temporal_coords, location_coords):
        batch, _, frames, height, width = optical.shape
        tokens = frames * (height // 16) * (width // 16)
        base = optical.mean(dim=1).reshape(batch, tokens, 16 * 16).mean(dim=-1, keepdim=True)
        base = base.repeat(1, 1, self.embed_dim) * self.weight
        cls = torch.zeros(batch, 1, self.embed_dim, device=optical.device)
        return [torch.cat((cls, base + depth / 100.0), dim=1) for depth in range(24)]


class PrithviChangeTests(unittest.TestCase):
    def make_inputs(self):
        return (
            torch.rand(2, 6, 4, 64, 64) * 10_000,
            torch.tensor([[[2020, 1], [2020, 30], [2020, 60], [2020, 90]]] * 2),
            torch.tensor([[10.0, 20.0], [-15.0, 40.0]]),
        )

    def test_visual_shapes(self):
        model = PrithviEO2ChangeModel(FakeEncoder(), decoder_width=32)
        result = model(*self.make_inputs())
        self.assertEqual(result["logits"].shape, (2, 1, 64, 64))
        self.assertEqual(result["visual_feature"].shape, (2, 32, 64, 64))

    def test_frozen_encoder(self):
        encoder = FakeEncoder()
        PrithviEO2ChangeModel(encoder, decoder_width=32, freeze_encoder=True)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_qt_zero_is_exact_visual(self):
        model = PrithviEO2ChangeModel(
            FakeEncoder(), decoder_width=32, terrain_channels=17, alpha_max=1.5
        )
        optical, temporal, location = self.make_inputs()
        result = model(
            optical,
            temporal,
            location,
            terrain=torch.rand(2, 17, 64, 64),
            q_t=torch.zeros(2),
        )
        self.assertTrue(torch.equal(result["logits"], result["visual_logits"]))

    def test_zero_terrain_is_exact_visual(self):
        model = PrithviEO2ChangeModel(
            FakeEncoder(), decoder_width=32, terrain_channels=17, alpha_max=1.5
        )
        optical, temporal, location = self.make_inputs()
        result = model(
            optical,
            temporal,
            location,
            terrain=torch.zeros(2, 17, 64, 64),
            q_t=torch.ones(2),
        )
        self.assertTrue(torch.equal(result["logits"], result["visual_logits"]))


if __name__ == "__main__":
    unittest.main()
