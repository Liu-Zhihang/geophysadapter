#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTROOT="${OUTROOT:-$PROJECT/experiments/revision2026/sen12_prithvi_roleaware_mr_v1}"
CACHEROOT="${CACHEROOT:-$OUTROOT/cache}"
FORMAL_ROOT="${FORMAL_ROOT:-$PROJECT/experiments/revision2026/sen12_prithvi_terrain_v2_formal_v1}"
SEED51_TERRAIN_ROOT="${SEED51_TERRAIN_ROOT:-$PROJECT/experiments/revision2026/sen12_terrain_expert_fusion_v1}"
MULTISEED_ROOT="${MULTISEED_ROOT:-$PROJECT/experiments/revision2026/sen12_terrain_multiseed_confirm_v1}"
SEEDS="${SEEDS:-20260751}"
FOLDS="${FOLDS:-0,1,2,3,4}"
MODES="${MODES:-material,trigger,joint}"
GPUS="${GPUS:-0,1}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DRY_RUN="${DRY_RUN:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a FOLD_ARRAY <<< "$FOLDS"
IFS=',' read -r -a MODE_ARRAY <<< "$MODES"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"

checkpoint_paths() {
  local seed="$1" fold="$2"
  if [[ "$seed" == "20260751" ]]; then
    VISUAL="$FORMAL_ROOT/fold${fold}_seed20260751/visual/checkpoint.pt"
    TERRAIN="$SEED51_TERRAIN_ROOT/fold${fold}_seed20260751/terrain_expert.pt"
  else
    VISUAL="$MULTISEED_ROOT/seed${seed}/fold${fold}/visual/checkpoint.pt"
    TERRAIN="$MULTISEED_ROOT/seed${seed}/fold${fold}/terrain/terrain_expert.pt"
  fi
}

jobs=()
for seed in "${SEED_ARRAY[@]}"; do
  for fold in "${FOLD_ARRAY[@]}"; do
    checkpoint_paths "$seed" "$fold"
    [[ -s "$VISUAL" ]] || { echo "missing visual checkpoint: $VISUAL" >&2; exit 1; }
    [[ -s "$TERRAIN" ]] || { echo "missing Terrain checkpoint: $TERRAIN" >&2; exit 1; }
    jobs+=("$seed|$fold|$VISUAL|$TERRAIN")
  done
done

make_command() {
  local seed="$1" fold="$2" mode="$3" visual="$4" terrain="$5"
  local out="$OUTROOT/seed${seed}/fold${fold}/${mode}"
  local cache="$CACHEROOT/seed${seed}/fold${fold}"
  COMMAND=(
    "$PYTHON" "$SCRIPT_DIR/train_sen12_prithvi_roleaware_mr_v1.py"
    --seed "$seed" --fold "$fold" --mode "$mode"
    --visual-checkpoint "$visual" --terrain-checkpoint "$terrain"
  )
  if [[ "$VALIDATE_ONLY" == "1" ]]; then
    COMMAND+=(--validate-only)
  else
    COMMAND+=(
      --cache-dir "$cache" --outdir "$out" --epochs "$EPOCHS"
      --batch-size "$BATCH_SIZE" --device cuda
    )
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  for job in "${jobs[@]}"; do
    IFS='|' read -r seed fold visual terrain <<< "$job"
    for mode in "${MODE_ARRAY[@]}"; do
      make_command "$seed" "$fold" "$mode" "$visual" "$terrain"
      printf '%q ' "${COMMAND[@]}"
      printf '\n'
    done
  done
  exit 0
fi

mkdir -p "$OUTROOT/launcher_logs"
worker() {
  local worker_index="$1" gpu="${GPU_ARRAY[$1]}" index mode
  local seed fold visual terrain out log
  for index in "${!jobs[@]}"; do
    if (( index % ${#GPU_ARRAY[@]} != worker_index )); then
      continue
    fi
    IFS='|' read -r seed fold visual terrain <<< "${jobs[$index]}"
    for mode in "${MODE_ARRAY[@]}"; do
      out="$OUTROOT/seed${seed}/fold${fold}/${mode}"
      if [[ -s "$out/DONE.json" && -s "$out/checkpoint.pt" && -s "$out/result.json" ]]; then
        echo "[skip] seed=$seed fold=$fold mode=$mode"
        continue
      fi
      make_command "$seed" "$fold" "$mode" "$visual" "$terrain"
      log="$OUTROOT/launcher_logs/seed${seed}_fold${fold}_${mode}.log"
      echo "[start] worker=$worker_index gpu=$gpu seed=$seed fold=$fold mode=$mode time=$(date --iso-8601=seconds)"
      CUDA_VISIBLE_DEVICES="$gpu" "${COMMAND[@]}" 2>&1 | tee "$log"
    done
  done
}

pids=()
for worker_index in "${!GPU_ARRAY[@]}"; do
  worker "$worker_index" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
