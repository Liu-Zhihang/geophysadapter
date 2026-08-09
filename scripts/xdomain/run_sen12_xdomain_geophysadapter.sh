#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${WORKSPACE_ROOT:-$(pwd)}}"
PYTHON="${PYTHON:-python}"
TRAINER="$ROOT/physics_informed_landslide_dataset/scripts/xdomain/train_sen12_xdomain_geophysadapter.py"
H5="${H5:-$ROOT/physics_informed_landslide_dataset/processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5}"
SPLIT="${SPLIT:-$ROOT/physics_informed_landslide_dataset/metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv}"
OUTBASE="${OUTBASE:-$ROOT/physics_informed_landslide_dataset/experiments/revision2026/sen12_xdomain_geophysadapter_v1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SEEDS="${SEEDS:-20260721 20260722 20260723 20260724 20260725}"
GPUS="${GPUS:-0 1}"
EPOCHS="${EPOCHS:-60}"
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
if [[ "$DRY_RUN" != "1" && ( ! -f "$H5" || ! -f "$SPLIT" ) ]]; then
  echo "Sen12 cache or split is missing: H5=$H5 SPLIT=$SPLIT" >&2
  exit 2
fi

is_complete() {
  local run_dir="$1"
  [[ -s "$run_dir/DONE.json" && -s "$run_dir/result.json" && -s "$run_dir/checkpoint.pt" && -s "$run_dir/per_sample.csv" && -s "$run_dir/per_event.csv" ]]
}

run_pair() {
  local fold="$1" seed="$2" gpu="$3"
  local pair_dir="$OUTBASE/fold${fold}_seed${seed}"
  local visual_dir="$pair_dir/visual"
  local adapter_dir="$pair_dir/adapter"
  mkdir -p "$pair_dir"
  local common=(
    --h5 "$H5"
    --split-csv "$SPLIT"
    --fold "$fold"
    --seed "$seed"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
    --max-eval-samples "$MAX_EVAL_SAMPLES"
    --max-steps "$MAX_STEPS"
    --device cuda
  )
  if ! is_complete "$visual_dir"; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAINER" \
      "${common[@]}" --mode visual --outdir "$visual_dir"
  else
    echo "[SKIP] visual fold=$fold seed=$seed"
  fi
  if ! is_complete "$adapter_dir"; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAINER" \
      "${common[@]}" --mode adapter --visual-checkpoint "$visual_dir/checkpoint.pt" \
      --outdir "$adapter_dir"
  else
    echo "[SKIP] adapter fold=$fold seed=$seed"
  fi
}

jobs=()
gpu_slot=0
for seed in $SEEDS; do
  for fold in $FOLDS; do
    gpu="${GPU_LIST[$gpu_slot]}"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "fold=$fold seed=$seed gpu=$gpu epochs=$EPOCHS max_steps=$MAX_STEPS"
    else
      run_pair "$fold" "$seed" "$gpu" >"$OUTBASE.fold${fold}.seed${seed}.log" 2>&1 &
      jobs+=("$!")
      if [[ ${#jobs[@]} -ge ${#GPU_LIST[@]} ]]; then
        for pid in "${jobs[@]}"; do wait "$pid"; done
        jobs=()
      fi
    fi
    gpu_slot=$(( (gpu_slot + 1) % ${#GPU_LIST[@]} ))
  done
done
for pid in "${jobs[@]}"; do wait "$pid"; done

echo "[DONE] Sen12 LOGO-5 visual/adapter queue completed: $OUTBASE"
