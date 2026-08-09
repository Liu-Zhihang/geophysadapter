#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ID="${DATASET_ID:?Set DATASET_ID}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR}"
CONTROL_TERRAIN_H5="${CONTROL_TERRAIN_H5:?Set CONTROL_TERRAIN_H5}"
CONTROL_NAME="${CONTROL_NAME:?Set CONTROL_NAME}"
FOLDS="${FOLDS:?Set FOLDS}"
SEEDS="${SEEDS:-20260724}"
GPUS="${GPUS:-0 1}"
FIXED_CONFIG="${FIXED_CONFIG:-0.3,0.7,4.0,1.0}"
CONFIG_MODE="${CONFIG_MODE:-fixed}"
NUM_WORKERS="${NUM_WORKERS:-2}"

BASE_H5="$CACHE_DIR/base_p128.h5"
OPTICAL_H5="$CACHE_DIR/prithvi_4t6b_p128.h5"
SPLIT_CSV="$CACHE_DIR/event_isolated_splits.csv"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_pild_member_terrain_confirm_v1.py"
if [[ "$CONFIG_MODE" == "fixed" ]]; then
  GATE_SUBDIR="gate_test_control_${CONTROL_NAME}"
elif [[ "$CONFIG_MODE" == "validation" ]]; then
  GATE_SUBDIR="gate_test_control_${CONTROL_NAME}_valselected"
else
  printf 'fatal: CONFIG_MODE must be fixed or validation\n' >&2
  exit 2
fi

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
  local gate_dir="$task_dir/$GATE_SUBDIR"
  local selected="$FIXED_CONFIG"
  if [[ "$CONFIG_MODE" == "validation" ]]; then
    selected="$(<"$task_dir/gate_test_valselected/validation_selected_config.txt")"
  fi
  mkdir -p "$gate_dir"
  if [[ ! -f "$gate_dir/result.json" || ! -f "$gate_dir/per_sample.csv" ]]; then
    printf '[start] dataset=%s control=%s seed=%s fold=%s gpu=%s time=%s\n' \
      "$DATASET_ID" "$CONTROL_NAME" "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" \
      | tee -a "$RUNS_DIR/control_progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" \
      --terrain-h5 "$CONTROL_TERRAIN_H5" --split-csv "$SPLIT_CSV" \
      --fold "$fold" --seed "$seed" --split test --routing-mode both \
      --fixed-config "$selected" --emit-per-sample \
      --visual-checkpoint "$task_dir/visual/checkpoint.pt" \
      --terrain-checkpoint "$task_dir/terrain/terrain_expert.pt" \
      --outdir "$gate_dir" --batch-size 32 --num-workers "$NUM_WORKERS" --device cuda \
      2>&1 | tee "$gate_dir/run.log"
  fi
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

analysis_args=()
if [[ "$CONFIG_MODE" == "validation" ]]; then
  analysis_args+=(--allow-varying-config)
fi
"$PYTHON" "$ANALYZER" \
  --runs-dir "$RUNS_DIR" --seeds "$SEEDS" --folds "$FOLDS" \
  --dataset-id "${DATASET_ID}_${CONTROL_NAME}" --fixed-config "$FIXED_CONFIG" \
  --gate-subdir "$GATE_SUBDIR" "${analysis_args[@]}" \
  --outdir "$RUNS_DIR/analysis_control_${CONTROL_NAME}_${CONFIG_MODE}"
