#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_dlr_terrain_additive_calibrated.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_dlr_validation_calibrated.py"
PARENT="${PARENT:-$ROOT/experiments/revision2026/dlr_sen12_protocol_transfer_temporalvalid_v2}"
SEED="${SEED:-20260723}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"

CACHE="$ROOT/processed/hybrid_pinn/dlr_sen12_protocol_transfer_v1"
BASE_H5="$CACHE/dlr_base_temporalvalid_p128.h5"
OPTICAL_H5="$CACHE/dlr_prithvi_4t6b_p128.h5"
TERRAIN_H5="$CACHE/dlr_common_terrain9_p128.h5"
SPLIT_CSV="$ROOT/metadata/protocol_assets/dlr_sen12_protocol_transfer_v1/dlr_eventisolated_nested5_v1.csv"

read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"

run_fold() {
  local fold="$1" gpu="$2"
  local task="$PARENT/seed${SEED}/fold${fold}"
  local out="$task/additive_calibrated_test"
  mkdir -p "$out"
  if [[ ! -f "$out/result.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" \
      --terrain-h5 "$TERRAIN_H5" --split-csv "$SPLIT_CSV" \
      --fold "$fold" --seed "$SEED" \
      --visual-checkpoint "$task/visual/checkpoint.pt" \
      --terrain-checkpoint "$task/terrain/terrain_expert.pt" \
      --outdir "$out" --batch-size 32 --num-workers 2 --device cuda \
      2>&1 | tee "$out/run.log"
  fi
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#fold_array[@]}; index+=${#gpu_array[@]})); do
    run_fold "${fold_array[$index]}" "$gpu"
  done
}

pids=()
for index in "${!gpu_array[@]}"; do
  worker "$index" "${gpu_array[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" "$ANALYZER" --runs-dir "$PARENT" \
  --result-glob 'seed*/fold*/additive_calibrated_test/result.json' \
  --outdir "$PARENT/additive_calibrated_analysis"
