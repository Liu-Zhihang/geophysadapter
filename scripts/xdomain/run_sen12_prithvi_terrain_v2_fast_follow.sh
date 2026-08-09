#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAINER="$ROOT/scripts/xdomain/train_sen12_prithvi_terrain_v2.py"
RUN_TAG="${RUN_TAG:-sen12_prithvi_terrain_v2_formal_v1}"
OUTBASE="${OUTBASE:-$ROOT/experiments/revision2026/$RUN_TAG}"
FOLDS="${FOLDS:-1 2 3 4}"
GPUS="${GPUS:-1 0}"
SEED="${SEED:-20260751}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
POLL_SECONDS="${POLL_SECONDS:-15}"

read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"

run_adapter() {
  local fold="$1" gpu="$2"
  local pair_dir="$OUTBASE/fold${fold}_seed${SEED}"
  local visual_dir="$pair_dir/visual"
  local adapter_dir="$pair_dir/adapter"

  while [[ ! -f "$visual_dir/DONE.json" || ! -f "$visual_dir/checkpoint.pt" ]]; do
    sleep "$POLL_SECONDS"
  done
  if [[ -f "$adapter_dir/DONE.json" ]]; then
    return
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAINER" \
    --mode adapter \
    --visual-checkpoint "$visual_dir/checkpoint.pt" \
    --outdir "$adapter_dir" \
    --fold "$fold" --seed "$SEED" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --max-train-samples 0 --max-eval-samples 0 --max-steps 0 --device cuda
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#fold_array[@]}; index+=${#gpu_array[@]})); do
    run_adapter "${fold_array[$index]}" "$gpu"
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

printf '{"status":"complete","run_tag":"%s","folds":"%s","seed":%s,"batch_size":%s}\n' \
  "$RUN_TAG" "$FOLDS" "$SEED" "$BATCH_SIZE" > "$OUTBASE/adapter_fast_follow.complete.json"
