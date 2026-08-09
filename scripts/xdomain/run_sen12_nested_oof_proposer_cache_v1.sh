#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAINER="$SCRIPT_DIR/build_sen12_nested_oof_proposer_cache_v1.py"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$PROJECT_ROOT/metadata/pild_xdomain_v1/sen12_nested_oof_protocol_v1}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_nested_oof_proposer_cache_v1}"
TARGETS="${TARGETS:-0,1,2,3,4}"
INNER_FOLDS="${INNER_FOLDS:-0,1,2}"
GPUS="${GPUS:-0,1}"
SEED="${SEED:-20260751}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TERRAIN_EPOCHS="${TERRAIN_EPOCHS:-20}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a target_values <<< "$TARGETS"
IFS=',' read -r -a inner_values <<< "$INNER_FOLDS"
IFS=',' read -r -a gpu_values <<< "$GPUS"
if (( ${#gpu_values[@]} == 0 )); then
  echo "GPUS must contain at least one device" >&2
  exit 2
fi

commands=()
task_gpus=()
task_index=0
for target in "${target_values[@]}"; do
  split_csv="$PROTOCOL_ROOT/sen12_nested_oof_target_outer${target}_v1.csv"
  for inner in "${inner_values[@]}"; do
    gpu="${gpu_values[$((task_index % ${#gpu_values[@]}))]}"
    outdir="$OUT_ROOT/target_outer${target}/inner_fold${inner}/seed${SEED}"
    command=(
      "$PYTHON" "$TRAINER"
      --target-outer-fold "$target"
      --inner-fold "$inner"
      --seed "$SEED"
      --split-csv "$split_csv"
      --protocol-manifest "$PROTOCOL_ROOT/sen12_nested_oof_protocol_v1_manifest.json"
      --outdir "$outdir"
      --visual-epochs "$VISUAL_EPOCHS"
      --terrain-epochs "$TERRAIN_EPOCHS"
      --device cuda
    )
    printf -v rendered '%q ' env "CUDA_VISIBLE_DEVICES=$gpu" "${command[@]}"
    commands+=("$rendered")
    task_gpus+=("$gpu")
    task_index=$((task_index + 1))
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  for index in "${!commands[@]}"; do
    printf '[DRY_RUN task=%02d gpu=%s] %s\n' "$index" "${task_gpus[$index]}" "${commands[$index]}"
  done
  printf '[DRY_RUN summary] tasks=%d proposer_trainings=%d gpus=%d\n' \
    "${#commands[@]}" "$((2 * ${#commands[@]}))" "${#gpu_values[@]}"
  exit 0
fi

run_gpu_queue() {
  local gpu_index="$1"
  local index
  for index in "${!commands[@]}"; do
    if (( index % ${#gpu_values[@]} != gpu_index )); then
      continue
    fi
    target="${target_values[$((index / ${#inner_values[@]}))]}"
    inner="${inner_values[$((index % ${#inner_values[@]}))]}"
    outdir="$OUT_ROOT/target_outer${target}/inner_fold${inner}/seed${SEED}"
    mkdir -p "$outdir"
    echo "[launch] task=$index gpu=${gpu_values[$gpu_index]} target=$target inner=$inner"
    bash -lc "${commands[$index]}" >"$outdir/launcher.log" 2>&1
  done
}

pids=()
for gpu_index in "${!gpu_values[@]}"; do
  run_gpu_queue "$gpu_index" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
