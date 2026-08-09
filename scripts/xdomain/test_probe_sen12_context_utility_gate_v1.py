#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("probe_sen12_context_utility_gate_v1.py")
SPEC = importlib.util.spec_from_file_location("utility_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_table() -> MODULE.SampleTable:
    n = 4
    visual = np.asarray([[2, 2, 1, 5]] * n, dtype=np.int64)
    terrain = visual.copy()
    terrain[0, 1] -= 1
    terrain[0, 3] += 1
    terrain[1, 2] -= 1
    terrain[1, 3] += 1
    zeros = np.zeros((n, 2), dtype=np.float32)
    q = np.asarray([1, 1, 0, 0], dtype=np.float32)
    return MODULE.SampleTable(
        sample_ids=[f"s{i}" for i in range(n)], event_ids=["a", "a", "b", "b"],
        base=zeros, material=zeros, material_shuffle=zeros,
        trigger=np.zeros((n, 3), np.float32), trigger_wrong=np.zeros((n, 3), np.float32),
        trigger_shuffle=np.zeros((n, 3), np.float32), q_m=q, q_m_shuffle=q,
        q_r=q, q_r_shuffle=q, utility=np.asarray([0.1, 0.1, -0.1, -0.1]),
        visual_counts=visual, terrain_counts=terrain,
    )


def test_gate_counts_and_rer() -> None:
    table = synthetic_table()
    result = MODULE.aggregate_gate(table, np.asarray([1, 1, -1, -1], dtype=float))
    assert result["terrain_selected_samples"] == 2
    assert result["net_corrected_vs_visual"] == 2
    assert result["rer_vs_visual"] > 0


def test_feature_context_dimensions() -> None:
    table = synthetic_table()
    assert MODULE.features(table, "T").shape[1] == 2
    assert MODULE.features(table, "TM").shape[1] == 5
    assert MODULE.features(table, "TR").shape[1] == 6
    assert MODULE.features(table, "TMR").shape[1] == 9


def test_event_weights_are_balanced() -> None:
    weights = MODULE.event_balanced_weights(["a", "a", "a", "b"])
    assert np.isclose(weights[:3].sum(), weights[3:].sum())
