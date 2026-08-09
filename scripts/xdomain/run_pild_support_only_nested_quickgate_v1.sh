#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
VISUAL="$ROOT/scripts/xdomain/train_pild_sen12_roleaware_v1.py"
TERRAIN="$ROOT/scripts/xdomain/train_pild_support_only_terrain_v1.py"
EVALUATE="$ROOT/scripts/xdomain/evaluate_pild_support_only_additive_v1.py"
CONTROL="$ROOT/scripts/xdomain/audit_pild_support_only_controls_v1.py"
META="${META:-$ROOT/metadata/pild_sen12_training_v2}"
MANIFEST="${MANIFEST:-$META/unified_sample_manifest_v2.csv}"
SUMMARY="${SUMMARY:-$META/protocol_summary_v2.json}"
SPLIT="${SPLIT:?SPLIT is required}"
FOLDS="${FOLDS:?FOLDS is required}"
OUTROOT="${OUTROOT:?OUTROOT is required}"
GPUS="${GPUS:-0 1}"
SEED="${SEED:-20260723}"
EPOCHS_VISUAL="${EPOCHS_VISUAL:-1}"
EPOCHS_TERRAIN="${EPOCHS_TERRAIN:-1}"
MAX_STEPS_VISUAL="${MAX_STEPS_VISUAL:-100}"
MAX_STEPS_TERRAIN="${MAX_STEPS_TERRAIN:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAMPLING_MODE="${SAMPLING_MODE:-balanced}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a fold_array <<<"$FOLDS"
read -r -a gpu_array <<<"$GPUS"
(( ${#fold_array[@]} > 0 )) || { printf 'FOLDS is empty\n' >&2; exit 2; }
(( ${#gpu_array[@]} > 0 )) || { printf 'GPUS is empty\n' >&2; exit 2; }
for path in "$VISUAL" "$TERRAIN" "$EVALUATE" "$CONTROL" "$MANIFEST" "$SUMMARY" "$SPLIT"; do
  [[ -s "$path" ]] || { printf 'missing prerequisite: %s\n' "$path" >&2; exit 2; }
done

run_command() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_fold() {
  local gpu="$1" fold="$2" fold_root visual terrain evaluation controls
  fold_root="$OUTROOT/$fold/seed${SEED}"
  visual="$fold_root/visual"
  terrain="$fold_root/terrain"
  evaluation="$fold_root/additive"
  controls="$fold_root/controls_v2"

  if [[ ! -s "$visual/DONE.json" ]]; then
    printf '[start] gpu=%s fold=%s stage=visual\n' "$gpu" "$fold"
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VISUAL" \
      --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
      --fold-id "$fold" --variant V --seed "$SEED" --outdir "$visual" \
      --epochs "$EPOCHS_VISUAL" --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" --max-steps "$MAX_STEPS_VISUAL" \
      --sampling-mode "$SAMPLING_MODE" --device cuda
  fi
  if [[ "$DRY_RUN" != 1 && ! -s "$visual/checkpoint.pt" ]]; then
    printf 'missing visual checkpoint: %s\n' "$visual/checkpoint.pt" >&2
    return 3
  fi

  if [[ ! -s "$terrain/DONE.json" ]]; then
    printf '[start] gpu=%s fold=%s stage=terrain\n' "$gpu" "$fold"
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TERRAIN" \
      --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
      --fold-id "$fold" --seed "$SEED" --parent-v-checkpoint "$visual/checkpoint.pt" \
      --outdir "$terrain" --epochs "$EPOCHS_TERRAIN" --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" --max-train-steps "$MAX_STEPS_TERRAIN" \
      --sampling-mode "$SAMPLING_MODE" --device cuda
  fi
  if [[ "$DRY_RUN" != 1 && ! -s "$terrain/terrain_expert.pt" ]]; then
    printf 'missing terrain checkpoint: %s\n' "$terrain/terrain_expert.pt" >&2
    return 4
  fi

  if [[ ! -s "$evaluation/DONE.json" ]]; then
    printf '[start] gpu=%s fold=%s stage=evaluate\n' "$gpu" "$fold"
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATE" \
      --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
      --fold-id "$fold" --seed "$SEED" --parent-v-checkpoint "$visual/checkpoint.pt" \
      --terrain-checkpoint "$terrain/terrain_expert.pt" --outdir "$evaluation" \
      --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device cuda
  fi
  if [[ "$DRY_RUN" != 1 && ! -s "$evaluation/result.json" ]]; then
    printf 'evaluation artifact gate failed: %s\n' "$evaluation" >&2
    return 5
  fi

  if [[ ! -s "$controls/DONE.json" ]]; then
    printf '[start] gpu=%s fold=%s stage=controls\n' "$gpu" "$fold"
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$CONTROL" \
      --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
      --fold-id "$fold" --seed "$SEED" --parent-v-checkpoint "$visual/checkpoint.pt" \
      --terrain-checkpoint "$terrain/terrain_expert.pt" \
      --selection-result "$evaluation/result.json" --outdir "$controls" \
      --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device cuda
  fi
  if [[ "$DRY_RUN" != 1 && ! -s "$controls/summary.json" ]]; then
    printf 'control artifact gate failed: %s\n' "$controls" >&2
    return 6
  fi
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#fold_array[@]}; index+=${#gpu_array[@]})); do
    run_fold "$gpu" "${fold_array[$index]}"
  done
}

pids=()
for index in "${!gpu_array[@]}"; do
  worker "$index" "${gpu_array[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"
printf '[complete] folds=%s seed=%s out=%s\n' "${#fold_array[@]}" "$SEED" "$OUTROOT"
