#!/usr/bin/env python3
"""Synthetic artifact-chain tests for the strict Sen12 TMR analyzer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_sen12_xdomain_tmr as analyzer


def metric_values(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    denominator = tp + fp + fn
    total = tp + fp + fn + tn
    iou = tp / denominator if denominator else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / max(total, 1),
    }


class Fixture:
    def __init__(self, root: Path, modes: tuple[str, ...] = analyzer.MODES) -> None:
        self.root = root
        self.tmr = root / "tmr"
        self.parents = root / "parents"
        self.fold = 0
        self.seed = 21
        self.parent_dir = self.parents / "fold0_seed21" / "adapter"
        self.parent_identity = self._identity("adapter")
        self.parent_state = {
            "terrain_encoder.weight": torch.tensor([1.0, 2.0]),
            "terrain_direction.weight": torch.tensor([3.0]),
            "gate_head.weight": torch.tensor([4.0]),
            "visual.weight": torch.tensor([5.0]),
        }
        self.parent_state_hash = analyzer.selected_state_sha256(
            self.parent_state, ("terrain_encoder.", "terrain_direction.", "gate_head.")
        )
        self._write_parent()
        for mode in modes:
            self._write_child(mode)

    def _identity(self, mode: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": mode,
            "fold": self.fold,
            "seed": self.seed,
            "backbone": "synthetic",
            "split_csv_sha256": "split-sha",
            "h5_signature": {"path": "/synthetic/data.h5", "size": 123, "mtime_ns": 456},
            "sample_identity_sha256": {"train": "train", "val": "val", "test": analyzer.sha256_strings(["s1", "s2"])},
            "reflectance_scale": 1.0,
            "image_size": 32,
            "out_indices": [0, 1],
            "hidden": 8,
            "pretrained_backbone": False,
            "visual_state_sha256": "visual-sha",
        }
        if mode in analyzer.MODES:
            payload.update({"tmr_mode": mode, "terrain_parent_state_sha256": self.parent_state_hash})
        return payload

    @staticmethod
    def _sample_rows(mode: str, controls: tuple[str, ...]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for control_index, control in enumerate(controls):
            improvement = 2 if control == "aligned" else (1 if control in {"material_shuffled", "trigger_event_shuffled"} else 0)
            if mode == "adapter" and control == "aligned":
                improvement = 1
            for sample_index, sample_id in enumerate(("s1", "s2")):
                visual_errors = 5
                corrected = 1 + improvement
                harmed = 1
                adapter_errors = visual_errors - corrected + harmed
                tp = 5 + improvement
                fp = 2
                fn = max(3 - improvement, 0)
                tn = 10
                row = {
                    "mode": mode,
                    "fold": 0,
                    "seed": 21,
                    "split": "test",
                    "control": control,
                    "sample_id": sample_id,
                    "physical_event_id": f"event{sample_index + 1}",
                    "region_group": f"region{sample_index + 1}",
                    "threshold": 0.5,
                    "visual_shared_threshold": 0.5,
                    "operational_threshold": 0.5,
                    "average_precision": 0.60 + 0.01 * improvement,
                    "brier": 0.20 - 0.01 * improvement,
                    "visual_errors": visual_errors,
                    "adapter_errors": adapter_errors,
                    "corrected": corrected,
                    "harmed": harmed,
                    "net_corrected": corrected - harmed,
                    "rer": (visual_errors - adapter_errors) / visual_errors,
                    **metric_values(tp, fp, fn, tn),
                }
                row.update({f"shared_{key}": value for key, value in metric_values(tp, fp, fn, tn).items()})
                rows.append(row)
        return rows

    @staticmethod
    def _aggregate_rows(sample_rows: list[dict[str, object]], unit: str) -> list[dict[str, object]]:
        frame = pd.DataFrame(sample_rows)
        output: list[dict[str, object]] = []
        for (mode, fold, seed, split, control, unit_value), group in frame.groupby(
            ["mode", "fold", "seed", "split", "control", unit], sort=False
        ):
            counts = {key: int(group[key].sum()) for key in analyzer.COUNT_KEYS}
            output.append({
                "mode": mode,
                "fold": int(fold),
                "seed": int(seed),
                "split": split,
                "control": control,
                unit: unit_value,
                "average_precision": float(group["average_precision"].mean()),
                "brier": float(group["brier"].mean()),
                **metric_values(**counts),
            })
        return output

    @staticmethod
    def _corpus_rows(sample_rows: list[dict[str, object]], controls: tuple[str, ...]) -> list[dict[str, object]]:
        frame = pd.DataFrame(sample_rows)
        rows: list[dict[str, object]] = []
        for control in controls:
            group = frame.loc[frame["control"].eq(control)]
            counts = {key: int(group[key].sum()) for key in analyzer.COUNT_KEYS}
            visual_errors = int(group["visual_errors"].sum())
            adapter_errors = int(group["adapter_errors"].sum())
            corrected = int(group["corrected"].sum())
            harmed = int(group["harmed"].sum())
            values = metric_values(**counts)
            rows.append({
                "mode": str(group["mode"].iloc[0]),
                "fold": 0,
                "seed": 21,
                "split": "test",
                "control": control,
                "average_precision": float(group["average_precision"].mean()),
                "brier": float(group["brier"].mean()),
                "region_macro_iou": values["iou"],
                "region_macro_average_precision": float(group["average_precision"].mean()),
                "region_macro_brier": float(group["brier"].mean()),
                "event_macro_iou": values["iou"],
                "event_macro_average_precision": float(group["average_precision"].mean()),
                "event_macro_brier": float(group["brier"].mean()),
                "visual_errors": visual_errors,
                "adapter_errors": adapter_errors,
                "corrected": corrected,
                "harmed": harmed,
                "net_corrected": corrected - harmed,
                "rer": (visual_errors - adapter_errors) / max(visual_errors, 1),
                **values,
            })
        return rows

    @staticmethod
    def _audit(sample_rows: list[dict[str, object]], controls: tuple[str, ...], tmr: bool) -> dict[str, object]:
        frame = pd.DataFrame(sample_rows)
        hashes = {
            control: analyzer.sha256_strings(frame.loc[frame["control"].eq(control), "sample_id"].astype(str).tolist())
            for control in controls
        }
        audit: dict[str, object] = {
            "split": "test",
            "n_samples_by_control": {control: 2 for control in controls},
            "sample_order_sha256_by_control": hashes,
            "same_sample_identity_and_order": True,
        }
        if tmr:
            audit.update({
                "zero_terrain_exact_fallback": True,
                "q_t_zero_exact_fallback": True,
                "q_m_zero_exact_identity": True,
                "q_r_zero_exact_identity": True,
                "q_m_q_r_zero_exact_parent_terrain_fallback": True,
                "inactive_controls_exact_identity": True,
                "other_region_donor_violations": 0,
            })
        return audit

    def _write_run(
        self,
        run_dir: Path,
        mode: str,
        controls: tuple[str, ...],
        identity: dict[str, object],
        state: dict[str, torch.Tensor],
        checkpoint_config: dict[str, object],
        tmr: bool,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        sample_rows = self._sample_rows(mode, controls)
        pd.DataFrame(sample_rows).to_csv(run_dir / "per_sample.csv", index=False)
        pd.DataFrame(self._aggregate_rows(sample_rows, "physical_event_id")).to_csv(run_dir / "per_event.csv", index=False)
        pd.DataFrame(self._aggregate_rows(sample_rows, "region_group")).to_csv(run_dir / "per_region.csv", index=False)
        result: dict[str, object] = {
            "identity": identity,
            "mode": mode,
            "fold": 0,
            "seed": 21,
            "visual_shared_threshold": 0.5,
            "corpus_metrics": self._corpus_rows(sample_rows, controls),
            "identity_and_control_audits": [self._audit(sample_rows, controls, tmr)],
        }
        if tmr:
            result.update({
                "terrain_parent_tensor_identity": True,
                "terrain_parent_state_sha256_before": self.parent_state_hash,
                "terrain_parent_state_sha256_after": self.parent_state_hash,
            })
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        torch.save({"identity": identity, "model_state_dict": state, "config": checkpoint_config}, run_dir / "checkpoint.pt")
        done = {
            "status": "complete",
            "mode": mode,
            "fold": 0,
            "seed": 21,
            "result_sha256": analyzer.sha256_file(run_dir / "result.json"),
            "checkpoint_sha256": analyzer.sha256_file(run_dir / "checkpoint.pt"),
        }
        (run_dir / "DONE.json").write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")

    def _write_parent(self) -> None:
        self._write_run(
            self.parent_dir,
            "adapter",
            analyzer.PARENT_CONTROLS,
            self.parent_identity,
            self.parent_state,
            {},
            False,
        )

    def _write_child(self, mode: str) -> None:
        run_dir = self.tmr / "fold0_seed21" / mode
        identity = self._identity(mode)
        config = {
            "terrain_checkpoint_sha256": analyzer.sha256_file(self.parent_dir / "checkpoint.pt"),
            "terrain_checkpoint_identity": self.parent_identity,
        }
        state = {**self.parent_state, f"{mode}_head.weight": torch.tensor([0.0])}
        self._write_run(run_dir, mode, analyzer.TMR_CONTROLS, identity, state, config, True)


class AnalyzerFixtureTests(unittest.TestCase):
    def test_partial_complete_chain_and_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            outdir = Path(temporary) / "analysis"
            code = analyzer.main([
                "--runs-dir", str(fixture.tmr),
                "--terrain-runs-dir", str(fixture.parents),
                "--outdir", str(outdir),
                "--allow-partial",
                "--min-folds", "1",
                "--min-seeds", "1",
                "--bootstrap", "200",
                "--permutations", "200",
            ])
            self.assertEqual(code, 0)
            summary = json.loads(
                (outdir / "summary.json").read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            self.assertEqual(summary["status"], "DEVELOPMENT_PARTIAL")
            self.assertFalse(summary["manuscript_evidence_eligible"])
            self.assertEqual(summary["n_validated_runs"], 3)
            for name in ("run_level.csv", "fold_level.csv", "control_deltas.csv", "report.md"):
                self.assertGreater((outdir / name).stat().st_size, 0)

    def test_formal_coverage_rejects_one_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            code = analyzer.main([
                "--runs-dir", str(fixture.tmr),
                "--terrain-runs-dir", str(fixture.parents),
                "--outdir", str(Path(temporary) / "analysis"),
                "--bootstrap", "200",
                "--permutations", "200",
            ])
            self.assertEqual(code, 1)

    def test_sample_order_corruption_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), modes=("terrain_material",))
            path = fixture.tmr / "fold0_seed21" / "terrain_material" / "per_sample.csv"
            frame = pd.read_csv(path)
            indices = frame.index[frame["control"].eq("material_shuffled")].tolist()
            frame.loc[indices, :] = frame.loc[list(reversed(indices)), :].to_numpy()
            frame.to_csv(path, index=False)
            code = analyzer.main([
                "--runs-dir", str(fixture.tmr),
                "--terrain-runs-dir", str(fixture.parents),
                "--outdir", str(Path(temporary) / "analysis"),
                "--allow-partial", "--min-folds", "1", "--min-seeds", "1",
                "--bootstrap", "200", "--permutations", "200",
            ])
            self.assertEqual(code, 1)

    def test_parent_state_identity_corruption_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), modes=("terrain_trigger",))
            result_path = fixture.tmr / "fold0_seed21" / "terrain_trigger" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["terrain_parent_state_sha256_before"] = "corrupt"
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            done_path = result_path.parent / "DONE.json"
            done = json.loads(done_path.read_text(encoding="utf-8"))
            done["result_sha256"] = analyzer.sha256_file(result_path)
            done_path.write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
            code = analyzer.main([
                "--runs-dir", str(fixture.tmr),
                "--terrain-runs-dir", str(fixture.parents),
                "--outdir", str(Path(temporary) / "analysis"),
                "--allow-partial", "--min-folds", "1", "--min-seeds", "1",
                "--bootstrap", "200", "--permutations", "200",
            ])
            self.assertEqual(code, 1)

    def test_smoke_directory_cannot_pass_formal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            smoke = Path(temporary) / "tmr_smoke_v1"
            fixture.tmr.rename(smoke)
            code = analyzer.main([
                "--runs-dir", str(smoke),
                "--terrain-runs-dir", str(fixture.parents),
                "--outdir", str(Path(temporary) / "analysis"),
                "--min-folds", "1", "--min-seeds", "1", "--expected-folds", "0",
                "--bootstrap", "200", "--permutations", "200",
            ])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
