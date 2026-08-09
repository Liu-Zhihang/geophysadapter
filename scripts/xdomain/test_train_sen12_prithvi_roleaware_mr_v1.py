#!/usr/bin/env python3
"""CPU contract tests for the Sen12 role-aware M/R trainer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_sen12_prithvi_roleaware_mr_v1 as subject
from pild_roleaware_material import MATERIAL_FEATURE_NAMES


def synthetic_batch() -> dict:
    return {
        "visual_logits": torch.randn(4, 1, 8, 8),
        "frozen_vt_correction": torch.randn(4, 1, 8, 8) * 0.1,
        "terrain_common9": torch.randn(4, 9, 8, 8),
        "material": torch.randn(4, 21),
        "q_material": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "material_shuffle": torch.randn(4, 21),
        "q_material_shuffle": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "material_shuffle_abstain": torch.tensor([True, False, False, True]),
        "trigger": torch.tensor([[1.0, 0.0, 1.0]] * 2 + [[0.0, 1.0, -1.0]] * 2),
        "trigger_wrong": torch.zeros(4, 3),
        "q_trigger": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "trigger_shuffle": torch.tensor([[0.0, 1.0, -1.0]] * 2 + [[1.0, 0.0, 1.0]] * 2),
        "q_trigger_shuffle": torch.ones(4),
        "trigger_shuffle_abstain": torch.zeros(4, dtype=torch.bool),
        "event_id": ["a", "a", "b", "b"],
        "sample_id": ["a0", "a1", "b0", "b1"],
        "source_id": ["s", "s", "s", "s"],
        "mask": torch.randint(0, 2, (4, 1, 8, 8), dtype=torch.uint8),
        "valid": torch.ones(4, 1, 8, 8, dtype=torch.uint8),
    }


def test_training_path_is_aligned_only() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "material_context=M_ALIGNED, trigger_context=R_ALIGNED" in source


def test_controls_share_one_model_checkpoint() -> None:
    model = subject.RoleAwareMR("joint")
    identity = id(model)
    batch = synthetic_batch()
    for material, trigger in subject.controls_for_mode("joint").values():
        model(batch, material_context=material, trigger_context=trigger)
        assert id(model) == identity


def test_q0_exact_fallback_to_vt() -> None:
    batch = synthetic_batch()
    for mode in subject.MODES:
        model = subject.RoleAwareMR(mode)
        material = subject.M_ZERO_Q if mode in ("material", "joint") else subject.M_ALIGNED
        trigger = subject.R_ZERO_Q if mode in ("trigger", "joint") else subject.R_ALIGNED
        logits, _ = model(batch, material_context=material, trigger_context=trigger)
        assert torch.equal(logits, batch["visual_logits"] + batch["frozen_vt_correction"])


def test_material_cannot_draw_boundary() -> None:
    model = subject.RoleAwareMR("material")
    batch = synthetic_batch()
    _, audit = model(batch)
    assert audit["material"]["material_dense_direction"] is False
    zero_residual = dict(batch)
    zero_residual["frozen_vt_correction"] = torch.zeros_like(batch["frozen_vt_correction"])
    logits, _ = model(zero_residual)
    assert torch.equal(logits, batch["visual_logits"])


def test_trigger_is_event_broadcast_and_visual_spatial_only() -> None:
    model = subject.RoleAwareMR("trigger")
    _, audit = model(synthetic_batch())
    trigger = audit["trigger"]
    assert trigger["audit"]["trigger_dense_direction"] is False
    assert trigger["audit"]["trigger_spatial_source"] == "detached_visual_uncertainty_only"


def test_checkpoint_parent_identity_rejects_wrong_fold() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        visual_path, terrain_path = root / "visual.pt", root / "terrain.pt"
        torch.save({"identity": {"mode": "visual", "fold": 1, "seed": 7}, "threshold": 0.5}, visual_path)
        torch.save({"result": {"fold": 0, "seed": 7, "visual_threshold": 0.5}}, terrain_path)
        args = type("Args", (), {
            "visual_checkpoint": visual_path,
            "terrain_checkpoint": terrain_path,
            "material_registry": root / "m.csv",
            "trigger_registry": root / "r.csv",
            "fold": 0,
            "mode": "joint",
            "seed": 7,
        })()
        try:
            subject.validate_real_inputs(args)
        except (RuntimeError, FileNotFoundError) as error:
            assert "visual" in str(error) or "missing" in str(error)
        else:
            raise AssertionError("wrong-fold parent was accepted")


def test_material_role_vector_is_exactly_21() -> None:
    assert len(MATERIAL_FEATURE_NAMES) == 21
    assert subject.MATERIAL_FEATURE_COUNT == 21
    assert tuple(subject.RESPONSE_GROUPS) == ("slope", "curvature", "relief")
    assert len(subject.COMMON_TERRAIN9_NAMES) == 9
    assert subject.COMMON_TERRAIN9_INDICES == (0, 1, 2, 3, 6, 7, 8, 16, 12)


def test_trigger_event_broadcast_rejects_variation() -> None:
    features = torch.tensor([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    try:
        subject.assert_event_level_broadcast(["same", "same"], features, torch.ones(2))
    except ValueError as error:
        assert "vary" in str(error)
    else:
        raise AssertionError("within-event Trigger variation was accepted")


def test_same_source_cross_event_material_shuffle() -> None:
    context = object.__new__(subject.RoleContext)
    context.sample_ids = ("a", "b", "c")
    context.source_ids = ("s", "s", "other")
    context.event_ids = ("e1", "e2", "e3")
    context.q_material = np.ones(3, np.float32)
    context.seed = 1
    donors, abstain = context._material_donors([0, 1, 2])
    assert donors[0] == 1 and donors[1] == 0
    assert not abstain[0] and not abstain[1]
    assert abstain[2]


def test_registry_source_column_mapping() -> None:
    mapped = [subject.AWC_SOURCE_COLUMNS.get(name, name) for name in MATERIAL_FEATURE_NAMES]
    assert mapped[:5] == [
        "awc_0_10_aligned_mm", "awc_10_30_aligned_mm", "awc_30_60_aligned_mm",
        "awc_60_100_aligned_mm", "awc_100_200_aligned_mm",
    ]


def test_json_is_strict() -> None:
    assert subject.json_safe(float("nan")) is None
    assert subject.json_safe(float("inf")) is None


def test_trigger_inner_event_selection_is_label_free_and_excluded() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trigger.csv"
        ids = ["a0", "a1", "b0", "b1", "v0", "t0"]
        events = ["a", "a", "b", "b", "v", "t"]
        pd.DataFrame({
            "sample_id": ids,
            "physical_event_id": events,
            "q_R": [1, 1, 1, 1, 0, 0],
        }).to_csv(path, index=False)
        plan = subject.trigger_support_plan(
            path, ids, events, ids[:4], ["v0"], ["t0"], 17
        )
        assert plan["inner_event"] in {"a", "b"}
        assert not (set(plan["inner_ids"]) & set(plan["role_train_ids"]))
        assert plan["outer_val_q_R_positive"] == 0
        assert plan["outer_test_q_R_positive"] == 0
        assert "label-free sha256" in plan["selection_receipt"]["contract"]


def test_unsupported_trigger_fold_publishes_explicit_empty_pair_receipt() -> None:
    rows = []
    for control in ("aligned", "trigger_wrong_time", "trigger_event_shuffle", "trigger_zero_q"):
        rows.append({
            "sample_id": "x0",
            "event_id": "event_x",
            "source_id": "sen12",
            "mode": "trigger",
            "control": control,
            "control_applicable": False,
        })
    assert subject.paired_control_rows(rows) == []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "paired_control_receipts.csv"
        subject.atomic_csv(path, [], empty_fields=subject.PAIRED_CONTROL_FIELDS)
        frame = pd.read_csv(path)
        assert frame.empty
        assert tuple(frame.columns) == tuple(sorted(subject.PAIRED_CONTROL_FIELDS))


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
