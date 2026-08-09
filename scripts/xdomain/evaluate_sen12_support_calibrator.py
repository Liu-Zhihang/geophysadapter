#!/usr/bin/env python3
"""Fit a validation-only support calibrator for frozen visual/Terrain experts.

The calibrator is deliberately small and auditable.  Validation samples are
split by sample ID into disjoint fit and threshold-calibration subsets.  The
reported test comparison includes a matched visual-context calibrator and a
spatially rolled Terrain negative control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_sen12_prithvi_terrain_v2 as trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--terrain-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--pixels-per-class-per-sample", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def is_fit_sample(sample_id: str, seed: int) -> bool:
    token = hashlib.sha256(f"{seed}|{sample_id}|support-calibrator".encode()).digest()
    return int.from_bytes(token[:8], "big") % 10 < 6


def deterministic_choice(indices: np.ndarray, limit: int, token: str) -> np.ndarray:
    if indices.size <= limit:
        return indices
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=limit, replace=False)


def spectral_features(optical: torch.Tensor) -> torch.Tensor:
    return trainer.PrithviTerrainVetoCompat.spectral_change_features(optical)


def make_features(
    visual_logits: torch.Tensor,
    terrain_logits: torch.Tensor,
    q_t: torch.Tensor,
    optical: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    visual_logits = visual_logits.float().clamp(-12.0, 12.0)
    terrain_logits = terrain_logits.float().clamp(-12.0, 12.0)
    visual_probability = torch.sigmoid(visual_logits)
    terrain_probability = torch.sigmoid(terrain_logits)
    uncertainty = 1.0 - 2.0 * torch.abs(visual_probability - 0.5)
    spectral = spectral_features(optical).float()
    visual = torch.cat((visual_logits, uncertainty, spectral), dim=1)
    physical = torch.cat(
        (
            visual,
            terrain_logits,
            terrain_probability,
            q_t.float(),
            terrain_logits * uncertainty,
            terrain_logits * spectral[:, :1],
        ),
        dim=1,
    )
    visual_np = visual.permute(0, 2, 3, 1).cpu().numpy()
    physical_np = physical.permute(0, 2, 3, 1).cpu().numpy()
    return visual_np, physical_np


def build_dataset(args, role, all_ids, event_ids, rows, roles, mean, std):
    return trainer.PrithviTerrainDataset(
        trainer.BASE_H5,
        trainer.OPTICAL_H5,
        trainer.TERRAIN_H5,
        all_ids,
        event_ids,
        rows,
        roles[role],
        mean,
        std,
        args.seed,
        roles["train"],
        True,
    )


def load_models(args):
    terrain_payload = torch.load(args.terrain_checkpoint, map_location="cpu", weights_only=False)
    terrain = trainer.SupportOnlyMultiScaleTerrainPyramid(
        17, trainer.NATIVE_TERRAIN_V2_SCALE_GROUPS
    )
    trainer.load_trainable_state(terrain, terrain_payload["trainable_state_dict"])
    terrain = terrain.to(args.device).eval()

    encoder, provenance = trainer.load_prithvi_encoder()
    visual = trainer.PrithviVisualCompat(
        trainer.PrithviEO2ChangeModel(encoder, decoder_width=128, freeze_encoder=True)
    )
    visual_payload = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    trainer.load_trainable_state(visual, visual_payload["trainable_state_dict"])
    visual = visual.to(args.device).eval()
    return visual, terrain, float(visual_payload["threshold"]), provenance


def forward_batch(batch, visual, terrain, device):
    optical = batch["pre"].to(device, non_blocking=True)
    coordinates = batch["post"].to(device, non_blocking=True)
    terrain_input = batch["terrain"].to(device, non_blocking=True)
    q_t = batch["q_t"].to(device, non_blocking=True)
    with torch.inference_mode(), trainer.protocol.autocast_context(True):
        visual_logits, _ = visual(optical, coordinates)
        terrain_logits, _ = terrain(terrain_input)
    return optical, q_t, visual_logits.float(), terrain_logits.float()


def fit_calibrators(args, loader, visual, terrain):
    visual_rows, physical_rows, labels = [], [], []
    fit_samples, calibration_samples = set(), set()
    for batch in loader:
        optical, q_t, visual_logits, terrain_logits = forward_batch(
            batch, visual, terrain, args.device
        )
        visual_feature, physical_feature = make_features(
            visual_logits, terrain_logits, q_t, optical
        )
        target = batch["mask"].numpy()[:, 0] >= 0.5
        valid = batch["valid"].numpy()[:, 0] >= 0.5
        for index, sample_id in enumerate(batch["sample_id"]):
            if not is_fit_sample(sample_id, args.seed):
                calibration_samples.add(sample_id)
                continue
            fit_samples.add(sample_id)
            flat_target = target[index].reshape(-1)
            flat_valid = valid[index].reshape(-1)
            positive = np.flatnonzero(flat_valid & flat_target)
            negative = np.flatnonzero(flat_valid & ~flat_target)
            limit = args.pixels_per_class_per_sample
            positive = deterministic_choice(positive, limit, f"{args.seed}|{sample_id}|pos")
            negative = deterministic_choice(negative, limit, f"{args.seed}|{sample_id}|neg")
            selected = np.concatenate((positive, negative))
            if selected.size == 0:
                continue
            visual_rows.append(visual_feature[index].reshape(-1, visual_feature.shape[-1])[selected])
            physical_rows.append(
                physical_feature[index].reshape(-1, physical_feature.shape[-1])[selected]
            )
            labels.append(flat_target[selected].astype(np.uint8))
    if not fit_samples or not calibration_samples:
        raise RuntimeError("validation sample split produced an empty fit or calibration subset")
    x_visual = np.concatenate(visual_rows)
    x_physical = np.concatenate(physical_rows)
    y = np.concatenate(labels)
    # Explicit interaction features above retain the mechanism while a linear
    # calibrator keeps full-pixel inference cheap and coefficients auditable.
    def estimator():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=200,
                solver="lbfgs",
                random_state=args.seed,
            ),
        )

    visual_calibrator = estimator().fit(x_visual, y)
    physical_calibrator = estimator().fit(x_physical, y)
    return visual_calibrator, physical_calibrator, fit_samples, calibration_samples, int(y.size)


def update_counts(total, prediction, target):
    total["tp"] += int(np.sum(prediction & target))
    total["fp"] += int(np.sum(prediction & ~target))
    total["fn"] += int(np.sum(~prediction & target))
    total["tn"] += int(np.sum(~prediction & ~target))


def calibrate_thresholds(
    args, loader, visual, terrain, visual_calibrator, physical_calibrator, calibration_samples
):
    histograms = {
        "visual_calibrator": trainer.protocol.ProbabilityHistogram(),
        "physical_calibrator": trainer.protocol.ProbabilityHistogram(),
    }
    for batch in loader:
        keep = np.asarray([sample in calibration_samples for sample in batch["sample_id"]])
        if not np.any(keep):
            continue
        optical, q_t, visual_logits, terrain_logits = forward_batch(
            batch, visual, terrain, args.device
        )
        visual_feature, physical_feature = make_features(
            visual_logits, terrain_logits, q_t, optical
        )
        target = batch["mask"].numpy()[:, 0] >= 0.5
        valid = batch["valid"].numpy()[:, 0] >= 0.5
        for index in np.flatnonzero(keep):
            selected = valid[index].reshape(-1)
            y = target[index].reshape(-1)[selected]
            xv = visual_feature[index].reshape(-1, visual_feature.shape[-1])[selected]
            xp = physical_feature[index].reshape(-1, physical_feature.shape[-1])[selected]
            histograms["visual_calibrator"].update(
                visual_calibrator.predict_proba(xv)[:, 1], y
            )
            histograms["physical_calibrator"].update(
                physical_calibrator.predict_proba(xp)[:, 1], y
            )
    selected = {}
    for name, histogram in histograms.items():
        threshold, metrics = trainer.protocol.choose_threshold(histogram)
        selected[name] = {
            "threshold": float(threshold),
            "ap": float(histogram.average_precision),
            "metrics": metrics,
        }
    return selected


def evaluate_test(
    args,
    loader,
    visual,
    terrain,
    raw_threshold,
    visual_calibrator,
    physical_calibrator,
    selected,
):
    totals = {name: defaultdict(int) for name in ("raw_visual", "visual_calibrator", "physical_calibrator", "roll64_control")}
    histograms = {name: trainer.protocol.ProbabilityHistogram() for name in totals}
    transitions = {name: defaultdict(int) for name in ("physical_vs_visual_calibrator", "physical_vs_raw_visual")}
    for batch in loader:
        optical, q_t, visual_logits, terrain_logits = forward_batch(
            batch, visual, terrain, args.device
        )
        visual_feature, physical_feature = make_features(
            visual_logits, terrain_logits, q_t, optical
        )
        rolled_logits = torch.roll(terrain_logits, shifts=(64, 64), dims=(-2, -1))
        rolled_q_t = torch.roll(q_t, shifts=(64, 64), dims=(-2, -1))
        _, rolled_feature = make_features(visual_logits, rolled_logits, rolled_q_t, optical)
        target = batch["mask"].numpy()[:, 0] >= 0.5
        valid = batch["valid"].numpy()[:, 0] >= 0.5
        raw_probability = torch.sigmoid(visual_logits)[:, 0].cpu().numpy()
        for index in range(target.shape[0]):
            keep = valid[index].reshape(-1)
            y = target[index].reshape(-1)[keep]
            xv = visual_feature[index].reshape(-1, visual_feature.shape[-1])[keep]
            xp = physical_feature[index].reshape(-1, physical_feature.shape[-1])[keep]
            xr = rolled_feature[index].reshape(-1, rolled_feature.shape[-1])[keep]
            probabilities = {
                "raw_visual": raw_probability[index].reshape(-1)[keep],
                "visual_calibrator": visual_calibrator.predict_proba(xv)[:, 1],
                "physical_calibrator": physical_calibrator.predict_proba(xp)[:, 1],
                "roll64_control": physical_calibrator.predict_proba(xr)[:, 1],
            }
            thresholds = {
                "raw_visual": raw_threshold,
                "visual_calibrator": selected["visual_calibrator"]["threshold"],
                "physical_calibrator": selected["physical_calibrator"]["threshold"],
                "roll64_control": selected["physical_calibrator"]["threshold"],
            }
            predictions = {}
            for name, probability in probabilities.items():
                histograms[name].update(probability, y)
                prediction = probability >= thresholds[name]
                predictions[name] = prediction
                update_counts(totals[name], prediction, y)
            for key, reference in (
                ("physical_vs_visual_calibrator", "visual_calibrator"),
                ("physical_vs_raw_visual", "raw_visual"),
            ):
                before = predictions[reference] == y
                after = predictions["physical_calibrator"] == y
                transitions[key]["corrected"] += int(np.sum(~before & after))
                transitions[key]["harmed"] += int(np.sum(before & ~after))
    output = {}
    for name, current in totals.items():
        metrics = trainer.protocol.metrics_from_counts(current)
        output[name] = {
            **current,
            **metrics,
            "errors": current["fp"] + current["fn"],
            "ap": float(histograms[name].average_precision),
        }
    for key, value in transitions.items():
        reference = "visual_calibrator" if key.endswith("visual_calibrator") else "raw_visual"
        before = output[reference]
        after = output["physical_calibrator"]
        corrected, harmed = value["corrected"], value["harmed"]
        value.update(
            {
                "delta_iou": after["iou"] - before["iou"],
                "rer": (before["errors"] - after["errors"]) / max(before["errors"], 1),
                "corrected_to_harmed": corrected / max(harmed, 1),
            }
        )
    return output, transitions


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    trainer.set_seed(args.seed)
    all_ids, event_ids = trainer.validate_sidecars(
        trainer.BASE_H5, trainer.OPTICAL_H5, trainer.TERRAIN_H5
    )
    rows, roles, split_regions = trainer.protocol.load_logo_rows(trainer.SPLIT_CSV, args.fold)
    allowed = set(all_ids)
    roles = {role: [sample for sample in values if sample in allowed] for role, values in roles.items()}
    mean, std = trainer.estimate_terrain_stats(trainer.TERRAIN_H5, all_ids, roles["train"])
    loader_args = SimpleNamespace(
        seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = trainer.protocol.make_loader(
        build_dataset(args, "val", all_ids, event_ids, rows, roles, mean, std),
        loader_args,
        shuffle=False,
    )
    test_loader = trainer.protocol.make_loader(
        build_dataset(args, "test", all_ids, event_ids, rows, roles, mean, std),
        loader_args,
        shuffle=False,
    )
    visual, terrain, raw_threshold, provenance = load_models(args)
    visual_calibrator, physical_calibrator, fit_samples, calibration_samples, n_fit = fit_calibrators(
        args, val_loader, visual, terrain
    )
    selected = calibrate_thresholds(
        args,
        val_loader,
        visual,
        terrain,
        visual_calibrator,
        physical_calibrator,
        calibration_samples,
    )
    test_metrics, transitions = evaluate_test(
        args,
        test_loader,
        visual,
        terrain,
        raw_threshold,
        visual_calibrator,
        physical_calibrator,
        selected,
    )
    result = {
        "status": "posthoc_development_after_test_family_was_observed",
        "fold": args.fold,
        "seed": args.seed,
        "test_regions": sorted(set(split_regions["test"])),
        "fit_samples": len(fit_samples),
        "calibration_samples": len(calibration_samples),
        "fit_pixels": n_fit,
        "feature_contract": {
            "visual_calibrator": ["visual_logit", "visual_uncertainty", "dNBR", "dNDVI", "post_NBR", "post_SWIR_contrast"],
            "physical_additions": ["terrain_expert_logit", "terrain_expert_probability", "terrain_validity", "terrain_x_uncertainty", "terrain_x_dNBR"],
            "warning": "dNBR is a burn-like descriptor, not a verified fire label",
        },
        "validation_selection": selected,
        "test_metrics": test_metrics,
        "transitions": transitions,
        "negative_control": "Terrain expert output and validity rolled by 64 pixels at test time",
        "prithvi_provenance": provenance,
    }
    (args.outdir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"validation_selection": selected, "test_metrics": test_metrics, "transitions": transitions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
