#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ID="${DATASET_ID:?Set DATASET_ID}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
FOLDS="${FOLDS:?Set FOLDS}"
SEEDS="${SEEDS:-20260724}"
GPUS="${GPUS:-0 1}"
RUN_TAG="${RUN_TAG:?Set RUN_TAG}"
OUTBASE="${OUTBASE:-$ROOT/experiments/revision2026/$RUN_TAG}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TERRAIN_EPOCHS="${TERRAIN_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FIXED_CONFIG="${FIXED_CONFIG:-0.3,0.7,4.0,1.0}"

BASE_H5="$CACHE_DIR/base_p128.h5"
OPTICAL_H5="$CACHE_DIR/prithvi_4t6b_p128.h5"
TERRAIN_H5="${TERRAIN_H5:-$CACHE_DIR/common_terrain9_p128.h5}"
SPLIT_CSV="$CACHE_DIR/event_isolated_splits.csv"
VISUAL_PARENT_BASE="${VISUAL_PARENT_BASE:-}"
VISUAL_TRAINER="$ROOT/scripts/xdomain/train_sen12_prithvi_terrain_v2.py"
TERRAIN_TRAINER="$ROOT/scripts/xdomain/train_sen12_terrain_expert_fusion.py"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_pild_member_terrain_confirm_v1.py"

for path in "$BASE_H5" "$OPTICAL_H5" "$TERRAIN_H5" "$SPLIT_CSV"; do
  [[ -f "$path" ]] || { printf 'fatal: missing %s\n' "$path" >&2; exit 2; }
done

mkdir -p "$OUTBASE"
printf '%s\n' "$0 $*" > "$OUTBASE/coordinator.command.txt"
printf '{"status":"running","dataset_id":"%s","seeds":"%s","folds":"%s","gpus":"%s","fixed_config":"%s","terrain_h5":"%s","visual_parent_base":"%s"}\n' \
  "$DATASET_ID" "$SEEDS" "$FOLDS" "$GPUS" "$FIXED_CONFIG" "$TERRAIN_H5" \
  "$VISUAL_PARENT_BASE" > "$OUTBASE/run_manifest.json"

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
  local task_dir="$OUTBASE/seed${seed}/fold${fold}"
  local visual_dir="$task_dir/visual"
  if [[ -n "$VISUAL_PARENT_BASE" ]]; then
    visual_dir="$VISUAL_PARENT_BASE/seed${seed}/fold${fold}/visual"
  fi
  local terrain_dir="$task_dir/terrain"
  local gate_dir="$task_dir/gate_test"
  mkdir -p "$terrain_dir" "$gate_dir"
  if [[ -z "$VISUAL_PARENT_BASE" ]]; then
    mkdir -p "$visual_dir"
  fi

  if [[ ! -f "$visual_dir/DONE.json" || ! -f "$visual_dir/checkpoint.pt" || ! -f "$visual_dir/result.json" ]]; then
    if [[ -n "$VISUAL_PARENT_BASE" ]]; then
      printf 'fatal: incomplete frozen visual parent %s\n' "$visual_dir" >&2
      return 3
    fi
    printf '[start] dataset=%s seed=%s fold=%s stage=visual gpu=%s time=%s\n' \
      "$DATASET_ID" "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VISUAL_TRAINER" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" --terrain-h5 "$TERRAIN_H5" \
      --split-csv "$SPLIT_CSV" --mode visual --fold "$fold" --seed "$seed" \
      --outdir "$visual_dir" --epochs "$VISUAL_EPOCHS" --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" --max-train-samples 0 --max-eval-samples 0 \
      --max-steps 0 --evaluation-splits val,test --device cuda
  fi

  if [[ ! -f "$terrain_dir/terrain_expert.pt" || ! -f "$terrain_dir/result.json" ]]; then
    printf '[start] dataset=%s seed=%s fold=%s stage=terrain gpu=%s time=%s\n' \
      "$DATASET_ID" "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TERRAIN_TRAINER" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" --terrain-h5 "$TERRAIN_H5" \
      --split-csv "$SPLIT_CSV" --fold "$fold" --seed "$seed" \
      --visual-checkpoint "$visual_dir/checkpoint.pt" --outdir "$terrain_dir" \
      --epochs "$TERRAIN_EPOCHS" --batch-size 64 --num-workers "$NUM_WORKERS" \
      --device cuda 2>&1 | tee "$terrain_dir/run.log"
  fi

  if [[ ! -f "$gate_dir/DONE.json" || ! -f "$gate_dir/result.json" || ! -f "$gate_dir/per_sample.csv" ]]; then
    printf '[start] dataset=%s seed=%s fold=%s stage=gate_test gpu=%s time=%s\n' \
      "$DATASET_ID" "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" --terrain-h5 "$TERRAIN_H5" \
      --split-csv "$SPLIT_CSV" --fold "$fold" --seed "$seed" --split test \
      --routing-mode both --fixed-config "$FIXED_CONFIG" --emit-per-sample \
      --visual-checkpoint "$visual_dir/checkpoint.pt" \
      --terrain-checkpoint "$terrain_dir/terrain_expert.pt" \
      --outdir "$gate_dir" --batch-size 32 --num-workers "$NUM_WORKERS" --device cuda \
      2>&1 | tee "$gate_dir/run.log"
    "$PYTHON" - "$gate_dir/result.json" "$gate_dir/DONE.json" "$seed" "$fold" "$FIXED_CONFIG" <<'PY'
import json
import pathlib
import sys

result_path, done_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
seed, fold, fixed = int(sys.argv[3]), int(sys.argv[4]), tuple(float(v) for v in sys.argv[5].split(","))
payload = json.loads(result_path.read_text())
row = payload["grid"][0]
observed = tuple(float(row[key]) for key in ("low_threshold", "high_threshold", "alpha", "visual_margin"))
assert payload["status"] == "confirmatory_fixed_configuration"
assert payload["split"] == "test" and int(payload["fold"]) == fold
assert observed == fixed
done_path.write_text(json.dumps({"status": "complete", "seed": seed, "fold": fold}, indent=2) + "\n")
PY
  fi
  printf '[done] dataset=%s seed=%s fold=%s gpu=%s time=%s\n' \
    "$DATASET_ID" "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
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

"$PYTHON" "$ANALYZER" \
  --runs-dir "$OUTBASE" --seeds "$SEEDS" --folds "$FOLDS" \
  --dataset-id "$DATASET_ID" --fixed-config "$FIXED_CONFIG" --outdir "$OUTBASE/analysis"
printf '{"status":"complete","dataset_id":"%s","seeds":"%s","folds":"%s","fixed_config":"%s","terrain_h5":"%s","visual_parent_base":"%s"}\n' \
  "$DATASET_ID" "$SEEDS" "$FOLDS" "$FIXED_CONFIG" "$TERRAIN_H5" \
  "$VISUAL_PARENT_BASE" > "$OUTBASE/run_manifest.json"
