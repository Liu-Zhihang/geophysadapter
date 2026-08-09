#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
DATASET_ID="${DATASET_ID:?Set DATASET_ID}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR containing trained seed/fold checkpoints}"
FOLDS="${FOLDS:?Set FOLDS}"
SEEDS="${SEEDS:-20260724}"
GPUS="${GPUS:-0 1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

BASE_H5="$CACHE_DIR/base_p128.h5"
OPTICAL_H5="$CACHE_DIR/prithvi_4t6b_p128.h5"
TERRAIN_H5="$CACHE_DIR/common_terrain9_p128.h5"
SPLIT_CSV="$CACHE_DIR/event_isolated_splits.csv"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_pild_member_terrain_confirm_v1.py"

read -r -a seed_array <<< "$SEEDS"
read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"
tasks=()
for seed in "${seed_array[@]}"; do
  for fold in "${fold_array[@]}"; do
    tasks+=("$seed:$fold")
  done
done

run_task() {
  local seed="$1" fold="$2" gpu="$3"
  local task_dir="$RUNS_DIR/seed${seed}/fold${fold}"
  local visual_checkpoint="$task_dir/visual/checkpoint.pt"
  local terrain_checkpoint="$task_dir/terrain/terrain_expert.pt"
  local val_dir="$task_dir/gate_val_grid"
  local test_dir="$task_dir/gate_test_valselected"
  [[ -f "$visual_checkpoint" && -f "$terrain_checkpoint" ]] || {
    printf 'fatal: missing trained checkpoints in %s\n' "$task_dir" >&2
    exit 2
  }
  mkdir -p "$val_dir" "$test_dir"

  if [[ ! -f "$val_dir/result.json" ]]; then
    printf '[start] dataset=%s seed=%s fold=%s stage=gate_val_grid gpu=%s time=%s\n' \
      "$DATASET_ID" "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$RUNS_DIR/valcal_progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" --terrain-h5 "$TERRAIN_H5" \
      --split-csv "$SPLIT_CSV" --fold "$fold" --seed "$seed" --split val \
      --routing-mode both --visual-checkpoint "$visual_checkpoint" \
      --terrain-checkpoint "$terrain_checkpoint" --outdir "$val_dir" \
      --batch-size 32 --num-workers "$NUM_WORKERS" --device cuda \
      2>&1 | tee "$val_dir/run.log"
  fi

  local selected
  selected="$("$PYTHON" - "$val_dir/result.json" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
row = payload["best_iou"]
print(",".join(str(float(row[key])) for key in ("low_threshold", "high_threshold", "alpha", "visual_margin")))
PY
)"
  printf '%s\n' "$selected" > "$test_dir/validation_selected_config.txt"
  if [[ ! -f "$test_dir/result.json" || ! -f "$test_dir/per_sample.csv" ]]; then
    printf '[start] dataset=%s seed=%s fold=%s stage=gate_test_valselected config=%s gpu=%s time=%s\n' \
      "$DATASET_ID" "$seed" "$fold" "$selected" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$RUNS_DIR/valcal_progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" --terrain-h5 "$TERRAIN_H5" \
      --split-csv "$SPLIT_CSV" --fold "$fold" --seed "$seed" --split test \
      --routing-mode both --fixed-config "$selected" --emit-per-sample \
      --visual-checkpoint "$visual_checkpoint" --terrain-checkpoint "$terrain_checkpoint" \
      --outdir "$test_dir" --batch-size 32 --num-workers "$NUM_WORKERS" --device cuda \
      2>&1 | tee "$test_dir/run.log"
  fi
  printf '[done] dataset=%s seed=%s fold=%s stage=valcal gpu=%s time=%s\n' \
    "$DATASET_ID" "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$RUNS_DIR/valcal_progress.log"
}

worker() {
  local worker_index="$1" gpu="$2" index task seed fold
  for ((index=worker_index; index<${#tasks[@]}; index+=${#gpu_array[@]})); do
    task="${tasks[$index]}"
    seed="${task%%:*}"
    fold="${task##*:}"
    run_task "$seed" "$fold" "$gpu"
  done
}

pids=()
for index in "${!gpu_array[@]}"; do
  worker "$index" "${gpu_array[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

# Validation selects one configuration per fold; aggregate without imposing a single fixed tuple.
"$PYTHON" - "$RUNS_DIR" "$SEEDS" "$FOLDS" "$DATASET_ID" <<'PY'
import csv
import json
import pathlib
import sys
from collections import defaultdict

runs = pathlib.Path(sys.argv[1])
seeds = [int(v) for v in sys.argv[2].split()]
folds = [int(v) for v in sys.argv[3].split()]
dataset_id = sys.argv[4]
rows = []
sample_rows = []
for seed in seeds:
    for fold in folds:
        directory = runs / f"seed{seed}" / f"fold{fold}" / "gate_test_valselected"
        payload = json.loads((directory / "result.json").read_text())
        adapted = payload["grid"][0]
        baseline = payload["baseline"]
        rows.append({
            "seed": seed, "fold": fold,
            "visual_tp": baseline["tp"], "visual_fp": baseline["fp"], "visual_fn": baseline["fn"],
            "visual_errors": baseline["errors"], "adapted_tp": adapted["tp"],
            "adapted_fp": adapted["fp"], "adapted_fn": adapted["fn"],
            "adapted_errors": adapted["errors"], "corrected": adapted["corrected"],
            "harmed": adapted["harmed"], "visual_iou": baseline["iou"],
            "adapted_iou": adapted["iou"], "delta_iou": adapted["delta_iou"],
            "rer": adapted["rer"], "selected_config": [
                adapted["low_threshold"], adapted["high_threshold"],
                adapted["alpha"], adapted["visual_margin"],
            ],
        })
        with (directory / "per_sample.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                sample_rows.append({"seed": seed, **row})

def aggregate(values):
    sums = {key: sum(int(row[key]) for row in values) for key in (
        "visual_tp","visual_fp","visual_fn","visual_errors","adapted_tp","adapted_fp",
        "adapted_fn","adapted_errors","corrected","harmed")}
    viou = sums["visual_tp"] / max(sums["visual_tp"] + sums["visual_fp"] + sums["visual_fn"], 1)
    aiou = sums["adapted_tp"] / max(sums["adapted_tp"] + sums["adapted_fp"] + sums["adapted_fn"], 1)
    return {**sums, "visual_iou": viou, "adapted_iou": aiou, "delta_iou": aiou - viou,
            "rer": (sums["visual_errors"] - sums["adapted_errors"]) / max(sums["visual_errors"], 1),
            "corrected_to_harmed": sums["corrected"] / max(sums["harmed"], 1)}

out = runs / "analysis_valselected"
out.mkdir(parents=True, exist_ok=True)
with (out / "per_seed_fold.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
pooled = aggregate(rows)
summary = {
    "status": "complete", "dataset_id": dataset_id,
    "contract": "member-specific training; per-fold validation-only gate selection; untouched event-isolated tests",
    "seeds": seeds, "folds": folds, "n_seed_folds": len(rows),
    "positive_delta_iou_folds": sum(float(row["delta_iou"]) > 0 for row in rows),
    "positive_rer_folds": sum(float(row["rer"]) > 0 for row in rows),
    "selected_configs": [{"seed": row["seed"], "fold": row["fold"], "config": row["selected_config"]} for row in rows],
    "pooled": pooled,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "report.md").write_text(
    f"# {dataset_id} validation-calibrated Terrain gate\\n\\n"
    f"- Pooled visual IoU: `{pooled['visual_iou']:.6f}`.\\n"
    f"- Pooled adapted IoU: `{pooled['adapted_iou']:.6f}`.\\n"
    f"- DeltaIoU: `{pooled['delta_iou']:+.6f}`.\\n"
    f"- RER: `{pooled['rer']:+.2%}`.\\n"
    f"- Corrected/harmed: `{pooled['corrected_to_harmed']:.3f}`.\\n"
    f"- Positive folds: `{summary['positive_delta_iou_folds']}/{len(rows)}`.\\n\\n"
    "All gate choices were made on validation events before one-shot test evaluation.\\n"
)
print(json.dumps(summary, indent=2))
PY
