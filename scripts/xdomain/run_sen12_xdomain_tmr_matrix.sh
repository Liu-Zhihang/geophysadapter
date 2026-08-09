#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${WORKSPACE_ROOT:-$(pwd)}}"
PROJECT="$ROOT"
PYTHON="${PYTHON:-python}"
TRAINER="$PROJECT/scripts/xdomain/train_sen12_xdomain_tmr.py"
H5="${H5:-$PROJECT/processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5}"
SPLIT="${SPLIT:-$PROJECT/metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv}"
SIDECAR="${SIDECAR:-$PROJECT/metadata/pild_xdomain_v1/sen12_tmr_sample_support_v2/sen12_tmr_sample_support_v1.csv}"
SIDECAR_SCHEMA="${SIDECAR_SCHEMA:-$PROJECT/metadata/pild_xdomain_v1/sen12_tmr_sample_support_v2/schema.json}"
VISUAL_BASE="${VISUAL_BASE:-$PROJECT/experiments/revision2026/sen12_xdomain_geophysadapter_v1}"
OUTBASE="${OUTBASE:-$PROJECT/experiments/revision2026/sen12_xdomain_tmr_smoke_v1}"
FOLDS="${FOLDS:-0 1}"
SEED="${SEED:-20260721}"
MODES="${MODES:-terrain_material terrain_trigger full_tmr}"
GPUS="${GPUS:-0 1}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a GPU_LIST <<< "$GPUS"
if [[ ${#GPU_LIST[@]} -lt 1 ]]; then
  echo "GPUS must contain at least one CUDA index" >&2
  exit 2
fi

required=("$TRAINER" "$H5" "$SPLIT" "$SIDECAR" "$SIDECAR_SCHEMA")
if [[ "$DRY_RUN" != "1" ]]; then
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || { echo "Required input is missing: $path" >&2; exit 2; }
  done
fi

is_complete() {
  local run_dir="$1"
  [[ -s "$run_dir/DONE.json" && -s "$run_dir/result.json" && \
     -s "$run_dir/checkpoint.pt" && -s "$run_dir/per_sample.csv" && \
     -s "$run_dir/per_event.csv" && -s "$run_dir/per_region.csv" ]]
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_fold() {
  local fold="$1" gpu="$2"
  local visual_dir="$VISUAL_BASE/fold${fold}_seed${SEED}/visual"
  local terrain_dir="$VISUAL_BASE/fold${fold}_seed${SEED}/adapter"
  local visual_checkpoint="$visual_dir/checkpoint.pt"
  local terrain_checkpoint="$terrain_dir/checkpoint.pt"
  if [[ "$DRY_RUN" != "1" ]] && { ! is_complete "$visual_dir" || ! is_complete "$terrain_dir"; }; then
    echo "Complete visual/Terrain parent artifact chain is missing: $visual_dir $terrain_dir" >&2
    return 2
  fi
  for mode in $MODES; do
    local run_dir="$OUTBASE/fold${fold}_seed${SEED}/$mode"
    local command=(
      env "CUDA_VISIBLE_DEVICES=$gpu" "$PYTHON" "$TRAINER"
      --h5 "$H5"
      --split-csv "$SPLIT"
      --sidecar "$SIDECAR"
      --sidecar-schema "$SIDECAR_SCHEMA"
      --visual-checkpoint "$visual_checkpoint"
      --terrain-checkpoint "$terrain_checkpoint"
      --mode "$mode"
      --fold "$fold"
      --seed "$SEED"
      --epochs "$EPOCHS"
      --batch-size "$BATCH_SIZE"
      --num-workers "$NUM_WORKERS"
      --max-train-samples "$MAX_TRAIN_SAMPLES"
      --max-eval-samples "$MAX_EVAL_SAMPLES"
      --max-steps "$MAX_STEPS"
      --device cuda
      --outdir "$run_dir"
    )
    if is_complete "$run_dir"; then
      echo "[SKIP] complete fold=$fold seed=$SEED mode=$mode"
    elif [[ "$DRY_RUN" == "1" ]]; then
      print_command "${command[@]}"
    else
      echo "[RUN] fold=$fold seed=$SEED mode=$mode gpu=$gpu"
      "${command[@]}"
    fi
  done
}

jobs=()
gpu_slot=0
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUTBASE"
fi
for fold in $FOLDS; do
  gpu="${GPU_LIST[$gpu_slot]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    run_fold "$fold" "$gpu"
  else
    run_fold "$fold" "$gpu" >"$OUTBASE/fold${fold}_seed${SEED}.runner.log" 2>&1 &
    jobs+=("$!")
  fi
  gpu_slot=$(( (gpu_slot + 1) % ${#GPU_LIST[@]} ))
done
for pid in "${jobs[@]}"; do
  wait "$pid"
done

echo "[DONE] Sen12 two-fold role-aware TMR smoke matrix: $OUTBASE"
