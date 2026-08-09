#!/usr/bin/env python3
"""G6c: rescue candidates with the full physical descriptor set and adjacency context.

The first rescue pass used only eleven summaries and reached a directed AUC of 0.684
against a 9.4 percent base rate, so the deployable promoter recovered almost nothing even
though a perfect one could reach delta IoU +0.056. Two things were missing.

First, the veto side ranks bodies with twenty-seven descriptors, including aspect
coherence, downslope elongation and gravity-descent consistency; the rescue side saw none
of them. The same function is reused here so both directions are judged on identical
physical evidence.

Second, a landslide is spatially contiguous. A sub-threshold body that touches an already
detected body is usually the missing tail of that same failure, whereas an isolated
sub-threshold body far from any detection is far more often a bright surface. Adjacency
is therefore recorded explicitly, together with the size and confidence of the neighbour
it attaches to.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from analyze_pild_object_physical_separability_v1 import (  # noqa: E402
    apply_terrain_condition,
    component_features,
)

DEFAULT_CACHE = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)


def adjacency_context(
    footprint: np.ndarray,
    predicted: np.ndarray,
    predicted_labels: np.ndarray,
    label_area: dict[int, int],
    distance: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Relationship between a sub-threshold body and the detections around it."""
    dilated = ndimage.binary_dilation(footprint, iterations=1)
    contact = dilated & predicted
    touching = np.unique(predicted_labels[contact])
    touching = touching[touching > 0]
    neighbour_area = float(sum(label_area.get(int(item), 0) for item in touching))
    rows, cols = np.nonzero(footprint)
    return {
        "touches_detection": float(touching.size > 0),
        "contact_pixels": float(np.count_nonzero(contact)),
        "contact_fraction": float(
            np.count_nonzero(contact) / max(np.count_nonzero(dilated & ~footprint), 1)
        ),
        "neighbour_detection_area": neighbour_area,
        "log_neighbour_area": float(np.log10(neighbour_area + 1.0)),
        "distance_to_detection": float(np.min(distance[rows, cols])),
        "neighbour_mean_probability": float(
            np.mean(probability[contact]) if contact.any() else 0.0
        ),
    }


def analyse_fold(
    cache_path: Path,
    *,
    operating_threshold: float,
    rescue_thresholds: tuple[float, ...],
    min_area: int,
    condition: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    with np.load(cache_path, allow_pickle=False) as handle:
        sample_id = handle["sample_id"]
        dataset_id = handle["dataset_id"]
        event_id = handle["canonical_event_id"]
        probability = handle["visual_probability"]
        target = handle["target"]
        valid = handle["valid"]
        terrain = handle["terrain"]
        terrain_valid = handle["terrain_valid"]
    terrain, _ = apply_terrain_condition(
        terrain, terrain_valid, dataset_id, event_id, condition, seed
    )

    structure = ndimage.generate_binary_structure(2, 2)
    rows_out: list[dict[str, Any]] = []
    totals = {"tp_pixels": 0.0, "fp_pixels": 0.0, "fn_pixels": 0.0}

    for index in range(len(sample_id)):
        keep = valid[index].astype(bool)
        truth = target[index].astype(bool) & keep
        chance = probability[index].astype(np.float32)
        predicted = (chance >= operating_threshold) & keep
        totals["tp_pixels"] += float(np.count_nonzero(predicted & truth))
        totals["fp_pixels"] += float(np.count_nonzero(predicted & ~truth))
        totals["fn_pixels"] += float(np.count_nonzero(~predicted & truth))
        if not keep.any():
            continue

        predicted_labels, n_predicted = ndimage.label(predicted, structure=structure)
        label_area: dict[int, int] = {}
        if n_predicted:
            counts = np.bincount(predicted_labels.ravel())
            label_area = {i: int(counts[i]) for i in range(1, len(counts))}
        distance = (
            ndimage.distance_transform_edt(~predicted)
            if predicted.any()
            else np.full(predicted.shape, float(predicted.shape[-1]), dtype=float)
        )
        sample_terrain = terrain[index]

        for rescue_threshold in rescue_thresholds:
            candidate = (chance >= rescue_threshold) & keep & ~predicted
            if not candidate.any():
                continue
            labels, count = ndimage.label(candidate, structure=structure)
            if count == 0:
                continue
            objects = ndimage.find_objects(labels)
            for label_value in range(1, count + 1):
                window = objects[label_value - 1]
                local = labels[window] == label_value
                area = int(np.count_nonzero(local))
                if area < min_area:
                    continue
                local_rows, local_cols = np.nonzero(local)
                global_rows = local_rows + window[0].start
                global_cols = local_cols + window[1].start
                footprint = np.zeros_like(predicted)
                footprint[global_rows, global_cols] = True
                recovered = int(np.count_nonzero(truth[global_rows, global_cols]))
                features = component_features(
                    global_rows, global_cols, sample_terrain, chance, local
                )
                context = adjacency_context(
                    footprint,
                    predicted,
                    predicted_labels,
                    label_area,
                    distance,
                    chance,
                )
                rows_out.append(
                    {
                        "sample_id": str(sample_id[index]),
                        "dataset_id": str(dataset_id[index]),
                        "canonical_event_id": str(event_id[index]),
                        "rescue_threshold": float(rescue_threshold),
                        "component_id": int(label_value),
                        "recovered_px": recovered,
                        "added_false_px": area - recovered,
                        "purity": float(recovered / area),
                        **features,
                        **context,
                    }
                )
    return rows_out, totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/revision2026/pild_object_rescue_capacity_v2",
    )
    parser.add_argument(
        "--rescue-thresholds", type=float, nargs="+", default=[0.5, 0.3, 0.15]
    )
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument(
        "--terrain-condition",
        choices=("aligned", "zero", "shift32", "roll64", "donor"),
        default="aligned",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    all_rows: list[dict[str, Any]] = []
    totals: dict[str, float] = defaultdict(float)
    for receipt_path in sorted(cache_dir.glob("*_oof_cache_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        rows, fold_totals = analyse_fold(
            cache_dir / f"{receipt['fold_id']}_oof_cache.npz",
            operating_threshold=float(receipt["threshold"]),
            rescue_thresholds=tuple(args.rescue_thresholds),
            min_area=args.min_area,
            condition=args.terrain_condition,
            seed=args.seed,
        )
        for row in rows:
            row["fold_id"] = receipt["fold_id"]
        all_rows.extend(rows)
        for key, value in fold_totals.items():
            totals[key] += value
        print(f"[fold] {receipt['fold_id']}: {len(rows)} candidates", flush=True)

    frame = pd.DataFrame(all_rows)
    frame.to_csv(outdir / "rescue_candidates.csv", index=False)
    tp, fp, fn = totals["tp_pixels"], totals["fp_pixels"], totals["fn_pixels"]
    baseline = tp / (tp + fp + fn)
    limit = baseline / (1.0 + baseline)

    report = []
    for threshold in sorted(frame.rescue_threshold.unique(), reverse=True):
        part = frame[frame.rescue_threshold == threshold]
        useful = part[part.purity > limit]
        recovered = float(useful.recovered_px.sum())
        added = float(useful.added_false_px.sum())
        new_tp, new_fp, new_fn = tp + recovered, fp + added, fn - recovered
        oracle = new_tp / max(new_tp + new_fp + new_fn, 1.0)
        touching = part[part.touches_detection > 0]
        report.append(
            {
                "rescue_threshold": float(threshold),
                "n_candidates": int(len(part)),
                "n_useful": int(len(useful)),
                "useful_rate": float(len(useful) / max(len(part), 1)),
                "addressable_fn_share": float(recovered / max(fn, 1.0)),
                "oracle_delta_iou": float(oracle - baseline),
                "touching_share": float(len(touching) / max(len(part), 1)),
                "useful_rate_touching": float(
                    (touching.purity > limit).mean() if len(touching) else float("nan")
                ),
                "useful_rate_isolated": float(
                    (part[part.touches_detection == 0].purity > limit).mean()
                    if (part.touches_detection == 0).any()
                    else float("nan")
                ),
                "fn_mass_in_touching_useful": float(
                    touching[touching.purity > limit].recovered_px.sum()
                    / max(recovered, 1.0)
                ),
            }
        )
    pd.DataFrame(report).to_csv(outdir / "rescue_capacity.csv", index=False)
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_rescue_capacity.v2",
                "terrain_condition": args.terrain_condition,
                "pixel_totals": {"tp": tp, "fp": fp, "fn": fn},
                "baseline_iou": baseline,
                "promotion_purity_limit": limit,
                "per_threshold": report,
                "elapsed_seconds": round(time.time() - started, 2),
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n=== rescue capacity v2 [{args.terrain_condition}] ===")
    print(
        f"{'thr':>5} {'cands':>8} {'useful%':>8} {'FN addr':>9} {'oracle':>9} "
        f"{'touch%':>8} {'useful|touch':>13} {'useful|iso':>11} {'mass|touch':>11}"
    )
    for row in report:
        print(
            f"{row['rescue_threshold']:5.2f} {row['n_candidates']:8d} "
            f"{row['useful_rate']:8.1%} {row['addressable_fn_share']:9.1%} "
            f"{row['oracle_delta_iou']:+9.5f} {row['touching_share']:8.1%} "
            f"{row['useful_rate_touching']:13.1%} {row['useful_rate_isolated']:11.1%} "
            f"{row['fn_mass_in_touching_useful']:11.1%}"
        )
    print(f"\nartifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
