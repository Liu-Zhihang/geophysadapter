#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
TRAINER="$ROOT/scripts/xdomain/train_sen12_prithvi_terrain_v2.py"
RUN_TAG="${RUN_TAG:-sen12_prithvi_terrain_v2_pilot}"
OUTBASE="${OUTBASE:-$ROOT/experiments/revision2026/$RUN_TAG}"
FOLDS="${FOLDS:-0 1}"
SEEDS="${SEEDS:-20260721}"
GPUS="${GPUS:-0 1}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-256}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-128}"
MAX_STEPS="${MAX_STEPS:-0}"
MODES="${MODES:-visual adapter}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a fold_array <<< "$FOLDS"
read -r -a seed_array <<< "$SEEDS"
read -r -a gpu_array <<< "$GPUS"
read -r -a mode_array <<< "$MODES"
if [[ ${#gpu_array[@]} -eq 0 ]]; then
  echo "GPUS must contain at least one device" >&2
  exit 2
fi
for mode in "${mode_array[@]}"; do
  if [[ "$mode" != "visual" && "$mode" != "adapter" ]]; then
    echo "MODES accepts only visual and/or adapter, got: $mode" >&2
    exit 2
  fi
done

jobs=()
for seed in "${seed_array[@]}"; do
  for fold in "${fold_array[@]}"; do
    jobs+=("$fold:$seed")
  done
done

run_one() {
  local fold="$1" seed="$2" gpu="$3"
  local pair_dir="$OUTBASE/fold${fold}_seed${seed}"
  local visual_dir="$pair_dir/visual"
  local adapter_dir="$pair_dir/adapter"
  local visual_checkpoint="$visual_dir/checkpoint.pt"
  local common=(
    --fold "$fold" --seed "$seed" --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
    --max-train-samples "$MAX_TRAIN_SAMPLES" --max-eval-samples "$MAX_EVAL_SAMPLES"
    --max-steps "$MAX_STEPS" --device cuda
  )
  local visual=("$PYTHON" "$TRAINER" --mode visual --outdir "$visual_dir" "${common[@]}")
  local adapter=("$PYTHON" "$TRAINER" --mode adapter --visual-checkpoint "$visual_checkpoint" --outdir "$adapter_dir" "${common[@]}")
  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ " ${mode_array[*]} " == *" visual "* ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"; printf '%q ' "${visual[@]}"; printf '\n'
    fi
    if [[ " ${mode_array[*]} " == *" adapter "* ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"; printf '%q ' "${adapter[@]}"; printf '\n'
    fi
    return
  fi
  if [[ " ${mode_array[*]} " == *" visual "* && ! -f "$visual_dir/DONE.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${visual[@]}"
  fi
  if [[ " ${mode_array[*]} " == *" adapter "* && ! -f "$visual_checkpoint" ]]; then
    echo "adapter requested without complete visual checkpoint: $visual_checkpoint" >&2
    return 3
  fi
  if [[ " ${mode_array[*]} " == *" adapter "* && ! -f "$adapter_dir/DONE.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${adapter[@]}"
  fi
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#jobs[@]}; index+=${#gpu_array[@]})); do
    IFS=: read -r fold seed <<< "${jobs[$index]}"
    run_one "$fold" "$seed" "$gpu"
  done
}

if [[ "$DRY_RUN" == "1" ]]; then
  for index in "${!jobs[@]}"; do
    IFS=: read -r fold seed <<< "${jobs[$index]}"
    gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
    run_one "$fold" "$seed" "$gpu"
  done
  exit 0
fi

pids=()
for index in "${!gpu_array[@]}"; do
  worker "$index" "${gpu_array[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
if [[ "$DRY_RUN" != "1" ]]; then
  printf '{"status":"complete","run_tag":"%s","folds":"%s","seeds":"%s","modes":"%s"}\n' \
    "$RUN_TAG" "$FOLDS" "$SEEDS" "$MODES" > "$OUTBASE/pipeline.complete.json"
fi
