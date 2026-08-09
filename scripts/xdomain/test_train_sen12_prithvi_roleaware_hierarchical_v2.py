#!/usr/bin/env python3
"""CPU contract tests for the independent Sen12 hierarchical v2 trainer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import train_sen12_prithvi_roleaware_hierarchical_v2 as subject


def synthetic_batch(*, certain_visual: bool = False) -> dict:
    visual = torch.full((4, 1, 8, 8), 100.0) if certain_visual else torch.randn(4, 1, 8, 8)
    return {
        "visual_logits": visual,
        "frozen_vt_correction": torch.randn(4, 1, 8, 8) * 0.1,
        "terrain_common9": torch.randn(4, 9, 8, 8),
        "material": torch.randn(4, 21),
        "q_material": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "material_shuffle": torch.randn(4, 21),
        "q_material_shuffle": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "material_shuffle_abstain": torch.tensor([True, False, False, True]),
        "material_control_applicable": torch.tensor([False, True, True, False]),
        "material_donor_sample_id": ["a0", "a0", "b0", "b0"],
        "material_donor_event_id": ["a", "a", "b", "b"],
        "trigger": torch.tensor([[1.0, 0.0, 1.0]] * 2 + [[0.0, 1.0, -1.0]] * 2),
        "trigger_wrong": torch.tensor([[0.0, 0.0, 0.0]] * 2 + [[1.0, 1.0, 0.0]] * 2),
        "q_trigger": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "trigger_shuffle": torch.tensor([[0.0, 1.0, -1.0]] * 2 + [[1.0, 0.0, 1.0]] * 2),
        "q_trigger_shuffle": torch.ones(4),
        "trigger_shuffle_abstain": torch.zeros(4, dtype=torch.bool),
        "trigger_control_applicable": torch.ones(4, dtype=torch.bool),
        "trigger_donor_sample_id": ["b0", "b0", "a0", "a0"],
        "trigger_donor_event_id": ["b", "b", "a", "a"],
        "event_id": ["a", "a", "b", "b"],
        "sample_id": ["a0", "a1", "b0", "b1"],
        "source_id": ["s", "s", "s", "s"],
        "mask": torch.randint(0, 2, (4, 1, 8, 8), dtype=torch.uint8),
        "valid": torch.ones(4, 1, 8, 8, dtype=torch.uint8),
    }


def test_hierarchy_order_is_explicit() -> None:
    model = subject.HierarchicalRoleAwareMR("joint")
    _, audit = model(synthetic_batch())
    assert audit["hierarchy"] == (
        "visual", "terrain_residual", "material_modulation", "trigger_dose"
    )


def test_material_and_trigger_have_no_dense_direction() -> None:
    model = subject.HierarchicalRoleAwareMR("joint")
    _, audit = model(synthetic_batch())
    assert audit["material"]["material_dense_direction"] is False
    assert audit["trigger"]["trigger_dense_direction"] is False
    assert audit["trigger"]["trigger_additive_logit_prior"] is False
    assert audit["material"]["terrain_is_only_dense_direction"] is True
    assert audit["trigger"]["terrain_is_only_dense_direction"] is True
    assert bool(audit["trigger"]["trigger_support_overlap_100pct"].all())
    assert int(audit["trigger"]["trigger_signed_direction_violation_count"].sum()) == 0


def test_q0_is_bit_exact_frozen_vt() -> None:
    batch = synthetic_batch()
    expected = batch["visual_logits"] + batch["frozen_vt_correction"]
    for mode in subject.MODES:
        model = subject.HierarchicalRoleAwareMR(mode)
        material = subject.M_ZERO_Q if mode in ("material", "joint") else subject.M_ALIGNED
        trigger = subject.R_ZERO_Q if mode in ("trigger", "joint") else subject.R_ALIGNED
        actual, _ = model(batch, material_context=material, trigger_context=trigger)
        assert torch.equal(actual, expected)


def test_mr_increment_is_restricted_to_visual_uncertainty() -> None:
    batch = synthetic_batch(certain_visual=True)
    model = subject.HierarchicalRoleAwareMR("joint")
    with torch.no_grad():
        model.material.head[-1].bias.fill_(1.0)
        model.trigger.raw_gain.fill_(1.0)
    actual, audit = model(batch)
    expected = batch["visual_logits"] + batch["frozen_vt_correction"]
    assert torch.equal(subject.detached_visual_uncertainty(batch["visual_logits"]), torch.zeros_like(batch["visual_logits"]))
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(audit["material_delta"]) == 0
    assert torch.count_nonzero(audit["trigger_delta"]) == 0


def test_trigger_gain_is_monotonic_and_bounded() -> None:
    module = subject.MonotonicTriggerResidualDose(dose_bound=0.5)
    with torch.no_grad():
        module.raw_gain.fill_(0.8)
        module.raw_slope.fill_(0.2)
        module.bias.zero_()
    contrast = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    gain = module.gain_from_contrast(contrast)
    assert torch.all(gain[1:] >= gain[:-1])
    assert float(gain.detach().min()) >= 0.0
    assert float(gain.detach().max()) <= 0.5


def test_trigger_strengthens_rescue_and_relaxes_veto_without_flip() -> None:
    batch = synthetic_batch()
    batch["visual_logits"] = torch.zeros_like(batch["visual_logits"])
    residual = torch.ones_like(batch["frozen_vt_correction"])
    residual[:, :, :, 4:] = -1.0
    batch["frozen_vt_correction"] = residual
    batch["q_trigger"] = torch.ones(4)
    model = subject.HierarchicalRoleAwareMR("trigger")
    with torch.no_grad():
        model.trigger.raw_gain.fill_(1.0)
    _, audit = model(batch)
    final_residual = audit["final_physical_residual"]
    assert torch.all(final_residual[:, :, :, :4] >= residual[:, :, :, :4])
    assert torch.all(final_residual[:, :, :, 4:] <= 0)
    assert torch.all(final_residual[:, :, :, 4:].abs() <= residual[:, :, :, 4:].abs())
    assert int(audit["trigger"]["trigger_signed_direction_violation_count"].sum()) == 0


def test_wrongtime_uses_same_model_checkpoint() -> None:
    model = subject.HierarchicalRoleAwareMR("trigger")
    identity = id(model)
    batch = synthetic_batch()
    aligned, _ = model(batch, trigger_context=subject.R_ALIGNED)
    wrong, _ = model(batch, trigger_context=subject.R_WRONG_TIME)
    assert id(model) == identity
    assert aligned.shape == wrong.shape
    assert subject.controls_for_mode("trigger")["trigger_wrong_time"][1] == subject.R_WRONG_TIME


def test_trigger_shuffle_donor_is_outer_train_supported() -> None:
    context = object.__new__(subject.RoleContext)
    context.seed = 9
    context.sample_ids = ("train0", "train1", "test0")
    context.event_ids = ("train_e0", "train_e1", "test_e")
    context.q_trigger = torch.ones(3).numpy()
    context.trigger_donor_by_event = {"train_e0": 0, "train_e1": 1}
    context.trigger_donor_events = ("train_e0", "train_e1")
    donors, abstain = context._trigger_donors([2])
    assert int(donors[0]) in (0, 1)
    assert int(donors[0]) != 2
    assert not bool(abstain[0])


def test_training_path_is_aligned_only() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "material_context=M_ALIGNED, trigger_context=R_ALIGNED" in source
    assert "same-selected-checkpoint-inference-only" in source


def test_epoch0_identity_is_selectable() -> None:
    model = subject.HierarchicalRoleAwareMR("material")
    batch = synthetic_batch()
    args = SimpleNamespace(
        lr=1e-3,
        weight_decay=0.0,
        mode="material",
        batch_size=4,
        seed=7,
        device="cpu",
        epochs=1,
        preservation_weight=0.1,
        preservation_confidence=0.9,
        selection_threshold=0.5,
        min_selection_score_gain=0.001,
        min_selection_net_error_reduction=1,
    )
    original_loader = subject.loader
    original_validation = subject.validation_receipt
    subject.loader = lambda payload, batch_size, shuffle, seed: [batch]
    subject.validation_receipt = lambda model, payload, args: {
        "pooled_ap": 0.5,
        "event_macro_ap": 0.5,
        "event_ap": {"e": 0.5},
        "n_events": 1,
        "candidate_errors": 10,
        "epoch0_identity_errors": 10,
        "net_error_reduction_vs_epoch0": 0,
        "threshold": 0.5,
        "split_usage": "validation-only; test-not-accessed",
    }
    try:
        state, history, best_epoch = subject.train_aligned_only(
            model, {}, {}, {}, args, lambda message: None
        )
    finally:
        subject.loader = original_loader
        subject.validation_receipt = original_validation
    assert best_epoch == 0
    assert history[0]["identity_candidate"] is True
    assert history[0]["epoch"] == 0
    assert set(state) == set(model.state_dict())


def test_preservation_regularizer_is_label_independent() -> None:
    baseline = torch.tensor([[[[8.0, 0.0], [-8.0, 0.0]]]])
    logits = baseline + 1.0
    valid = torch.ones_like(baseline, dtype=torch.uint8)
    loss, selected = subject.high_confidence_preservation_loss(logits, baseline, valid, 0.9)
    assert selected == 2
    assert float(loss) > 0
    assert "target" not in subject.high_confidence_preservation_loss.__code__.co_varnames


def test_unsupported_trigger_receipt_is_header_only() -> None:
    rows = [
        {
            "sample_id": "x0",
            "event_id": "event_x",
            "source_id": "sen12",
            "mode": "trigger",
            "control": control,
            "control_applicable": False,
        }
        for control in ("aligned", "trigger_wrong_time", "trigger_event_shuffle", "trigger_zero_q")
    ]
    assert subject.paired_control_rows(rows) == []
    assert "control" in subject.PAIRED_CONTROL_FIELDS


def test_v2_schema_and_outroot_are_independent() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "roleaware_hierarchical" in str(subject.DEFAULT_OUTROOT)
    assert "hierarchical_run.v2" in source
    assert "selected_identity_abstain" in source
    assert "roleaware_mr_v1" not in str(subject.DEFAULT_OUTROOT)
    assert subject.MATERIAL_SCOPE["lithology_present"] is True
    assert "footprint" in subject.MATERIAL_SCOPE["spatial_semantics"]


def test_formal_material_schema_dimension_is_dynamic() -> None:
    names = subject.load_material_feature_names(subject.DEFAULT_MATERIAL_SCHEMA)
    assert len(names) == 55
    model = subject.HierarchicalRoleAwareMR("material", material_feature_count=len(names))
    batch = synthetic_batch()
    batch["material"] = torch.randn(4, len(names))
    batch["material_shuffle"] = torch.randn(4, len(names))
    output, audit = model(batch)
    assert output.shape == batch["visual_logits"].shape
    assert audit["material"]["material_dense_direction"] is False


def test_material_normalizer_accepts_frozen_dynamic_schema() -> None:
    values = torch.arange(18, dtype=torch.float32).reshape(6, 3).numpy()
    normalizer = subject.OuterTrainMaterialNormalizer.fit(
        values,
        [f"s{index}" for index in range(6)],
        ["sen12"] * 6,
        ["e0", "e0", "e1", "e1", "e2", "e2"],
        ["s0", "s1", "s2", "s3"],
        feature_names=("a", "b", "c"),
    )
    transformed = normalizer.transform(values)
    assert transformed.shape == values.shape
    assert normalizer.feature_names == ("a", "b", "c")


def test_selection_receipt_contains_event_macro_and_net_error() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "event_macro_ap" in source
    assert "net_error_reduction_vs_epoch0" in source
    assert "test-not-accessed" in source
    assert subject.DEFAULT_MIN_SELECTION_SCORE_GAIN > 0


def test_validation_receipt_computes_event_macro_without_test() -> None:
    batch = synthetic_batch()
    model = subject.HierarchicalRoleAwareMR("material")
    args = SimpleNamespace(batch_size=4, seed=5, device="cpu", selection_threshold=0.5)
    original_loader = subject.loader
    subject.loader = lambda payload, batch_size, shuffle, seed: [batch]
    try:
        receipt = subject.validation_receipt(model, {}, args)
    finally:
        subject.loader = original_loader
    assert receipt["n_events"] == 2
    assert 0.0 <= receipt["pooled_ap"] <= 1.0
    assert 0.0 <= receipt["event_macro_ap"] <= 1.0
    assert receipt["net_error_reduction_vs_epoch0"] == 0
    assert receipt["split_usage"] == "validation-only; test-not-accessed"


def test_local_effect_fields_are_in_sample_and_control_artifacts() -> None:
    batch = synthetic_batch()
    model = subject.HierarchicalRoleAwareMR("joint")
    args = SimpleNamespace(batch_size=4, seed=5, device="cpu", mode="joint")
    original_loader = subject.loader
    subject.loader = lambda payload, batch_size, shuffle, seed: [batch]
    try:
        result, sample_rows, control_rows = subject.evaluate(
            model, {}, args, threshold=0.5, fixed_fpr_threshold=0.5, with_controls=True
        )
    finally:
        subject.loader = original_loader
    required = {
        "q_M", "q_R", "effective_q_M", "effective_q_R",
        "visual_uncertainty_mean", "visual_uncertainty_q75", "visual_uncertainty_q90",
        "terrain_support_pixel_count", "terrain_support_fraction",
        "material_scalar", "material_multiplier_abs_deviation_mean",
        "rain_contrast", "rain_gain", "trigger_changed_pixel_count",
        "trigger_terrain_overlap_pixel_count", "trigger_support_overlap_100pct",
        "material_local_effect_eligible", "trigger_local_effect_eligible",
        "joint_local_effect_eligible", "local_effect_subset_uses_test_label",
    }
    assert required <= set(sample_rows[0])
    assert required <= set(control_rows[0])
    assert sample_rows[0]["local_effect_subset_uses_test_label"] is False
    assert result["controls"]["aligned"]["global_no_negative_transfer_vs_frozen_vt"] is True
    assert result["global_no_negative_transfer_aligned_vs_frozen_vt"] is True
    assert result["conditional_effect_contract"]["test_label_used_for_subgroup_or_threshold"] is False


def test_strict_json() -> None:
    payload = subject.json_safe({"nan": float("nan"), "inf": float("inf"), "ok": 1.0})
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == {"nan": None, "inf": None, "ok": 1.0}


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
