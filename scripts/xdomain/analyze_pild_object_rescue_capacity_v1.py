#!/usr/bin/env python3
"""G6: how much missed landslide area can an object-level rescue recover?

Every object experiment so far only removed candidate bodies, yet the corpus holds more
missed area than detected area: FN is 1.41 times TP. Rescue is the exact mirror of veto,
because promoting a body with ``i`` true and ``f`` false pixels moves the pooled score to
``(TP+i)/(D+f)``, which improves it precisely when

    i / f  >  TP / D  =  IoU_baseline

so the same purity criterion decides both directions, only the inequality flips.

Candidate bodies are formed by lowering the decision threshold of the frozen visual
model. A component that does not overlap the operating-threshold prediction at all is a
missed body; the visual model already places some probability there but not enough to
cross its own threshold. This diagnostic measures how much false-negative mass sits in
such bodies and what a perfect promoter could recover, mirroring the veto-side result
that 69.8 percent of false-positive mass was addressable at 1.36 percent true-positive
risk.
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
DEFAULT_CACHE = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)

# Same channel order as the veto-side diagnostic.
TERRAIN_INDEX = {
    "elevation": 0,
    "slope_deg": 1,
    "aspect_sin": 2,
    "aspect_cos": 3,
    "tpi_900m": 9,
    "local_relief_300m": 12,
    "valley_depth_900m": 14,
    "ruggedness_90m": 16,
}


def body_statistics(
    rows: np.ndarray,
    cols: np.ndarray,
    terrain: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Compact physical and confidence summary of a candidate rescue body."""
    slope = terrain[TERRAIN_INDEX["slope_deg"], rows, cols].astype(np.float32)
    elevation = terrain[TERRAIN_INDEX["elevation"], rows, cols].astype(np.float32)
    tpi900 = terrain[TERRAIN_INDEX["tpi_900m"], rows, cols].astype(np.float32)
    values = probability[rows, cols].astype(np.float32)
    area = float(rows.size)
    return {
        "area_px": area,
        "mean_slope": float(np.mean(slope)),
        "p90_slope": float(np.percentile(slope, 90)),
        "flat_fraction": float(np.mean(slope < 5.0)),
        "elev_range": float(np.max(elevation) - np.min(elevation)),
        "relative_relief": float(
            (np.max(elevation) - np.min(elevation)) / (np.sqrt(max(area, 1.0)) * 10.0)
        ),
        "mean_tpi_900m": float(np.mean(tpi900)),
        "mean_local_relief_300m": float(
            np.mean(terrain[TERRAIN_INDEX["local_relief_300m"], rows, cols])
        ),
        "mean_ruggedness": float(
            np.mean(terrain[TERRAIN_INDEX["ruggedness_90m"], rows, cols])
        ),
        "mean_probability": float(np.mean(values)),
        "max_probability": float(np.max(values)),
    }


def analyse_fold(
    cache_path: Path,
    *,
    operating_threshold: float,
    rescue_thresholds: tuple[float, ...],
    min_area: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    with np.load(cache_path, allow_pickle=False) as handle:
        sample_id = handle["sample_id"]
        dataset_id = handle["dataset_id"]
        event_id = handle["canonical_event_id"]
        probability = handle["visual_probability"]
        target = handle["target"]
        valid = handle["valid"]
        terrain = handle["terrain"]

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
        if not truth.any() and not keep.any():
            continue
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
                recovered = int(
                    np.count_nonzero(truth[window][local])
                )
                stats = body_statistics(
                    global_rows, global_cols, sample_terrain, chance
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
                        **stats,
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
        / "experiments/revision2026/pild_object_rescue_capacity_v1",
    )
    parser.add_argument(
        "--rescue-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.15, 0.05],
    )
    parser.add_argument("--min-area", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    all_rows: list[dict[str, Any]] = []
    totals = defaultdict(float)
    for receipt_path in sorted(cache_dir.glob("*_oof_cache_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        cache_path = cache_dir / f"{receipt['fold_id']}_oof_cache.npz"
        rows, fold_totals = analyse_fold(
            cache_path,
            operating_threshold=float(receipt["threshold"]),
            rescue_thresholds=tuple(args.rescue_thresholds),
            min_area=args.min_area,
        )
        for row in rows:
            row["fold_id"] = receipt["fold_id"]
        all_rows.extend(rows)
        for key, value in fold_totals.items():
            totals[key] += value
        print(
            f"[fold] {receipt['fold_id']}: {len(rows)} candidate bodies", flush=True
        )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(outdir / "rescue_candidates.csv", index=False)
    tp, fp, fn = totals["tp_pixels"], totals["fp_pixels"], totals["fn_pixels"]
    denominator = tp + fp + fn
    baseline = tp / denominator
    promote_limit = baseline / (1.0 + baseline)

    report: list[dict[str, Any]] = []
    for threshold in sorted(frame.rescue_threshold.unique(), reverse=True):
        part = frame[frame.rescue_threshold == threshold]
        useful = part[part.purity > promote_limit]
        recovered = float(useful.recovered_px.sum())
        added = float(useful.added_false_px.sum())
        new_tp, new_fp, new_fn = tp + recovered, fp + added, fn - recovered
        oracle = new_tp / max(new_tp + new_fp + new_fn, 1.0)
        report.append(
            {
                "rescue_threshold": float(threshold),
                "n_candidates": int(len(part)),
                "n_useful": int(len(useful)),
                "candidate_fn_mass": float(part.recovered_px.sum()),
                "candidate_fn_share": float(part.recovered_px.sum() / max(fn, 1.0)),
                "useful_recovered_px": recovered,
                "useful_added_fp_px": added,
                "addressable_fn_share": float(recovered / max(fn, 1.0)),
                "fp_cost_share": float(added / max(fp, 1.0)),
                "oracle_iou": float(oracle),
                "oracle_delta_iou": float(oracle - baseline),
            }
        )
    table = pd.DataFrame(report)
    table.to_csv(outdir / "rescue_capacity.csv", index=False)

    cumulative = frame[frame.purity > promote_limit]
    best = cumulative.sort_values("purity", ascending=False).drop_duplicates(
        subset=["fold_id", "sample_id", "component_id"], keep="first"
    )
    summary = {
        "schema_version": "pild_object_rescue_capacity.v1",
        "pixel_totals": {"tp": tp, "fp": fp, "fn": fn},
        "baseline_iou": baseline,
        "promotion_purity_limit": promote_limit,
        "per_threshold": report,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )

    print(f"\n=== rescue capacity (baseline IoU {baseline:.5f}) ===")
    print(f"promotion is beneficial when body purity > {promote_limit:.4f}")
    print(
        f"{'thr':>6} {'cands':>8} {'useful':>8} {'FN reach':>9} {'FN addr':>9} "
        f"{'FP cost':>9} {'oracle dIoU':>12}"
    )
    for row in report:
        print(
            f"{row['rescue_threshold']:6.2f} {row['n_candidates']:8d} {row['n_useful']:8d} "
            f"{row['candidate_fn_share']:9.1%} {row['addressable_fn_share']:9.1%} "
            f"{row['fp_cost_share']:9.1%} {row['oracle_delta_iou']:+12.5f}"
        )
    print(f"\nartifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
