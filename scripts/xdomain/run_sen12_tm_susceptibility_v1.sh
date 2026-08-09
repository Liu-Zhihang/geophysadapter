#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data_hdd/滑坡检测/physics_informed_landslide_dataset}"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
OUTROOT="${OUTROOT:-$ROOT/experiments/revision2026/sen12_tm_susceptibility_v1}"
SEED="${SEED:-20260761}"
TERRAIN_EPOCHS="${TERRAIN_EPOCHS:-12}"
MATERIAL_EPOCHS="${MATERIAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-2}"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local outdir="$OUTROOT/fold${fold}_seed${SEED}"
  if [[ -f "$outdir/DONE.json" ]]; then
    echo "[skip] fold=$fold DONE exists"
    return
  fi
  mkdir -p "$outdir"
  local -a command=(
    "$PYTHON" -u "$ROOT/scripts/xdomain/train_sen12_tm_susceptibility_v1.py"
    --fold "$fold"
    --seed "$SEED"
    --outdir "$outdir"
    --terrain-epochs "$TERRAIN_EPOCHS"
    --material-epochs "$MATERIAL_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device cuda
  )
  printf '%q ' "${command[@]}" >"$outdir/command.txt"
  printf '\n' >>"$outdir/command.txt"
  echo "[start] fold=$fold gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" 2>&1 | tee "$outdir/run.log"
}

worker_zero() {
  run_fold 0 0
  run_fold 2 0
  run_fold 4 0
}

worker_one() {
  run_fold 1 1
  run_fold 3 1
}

mkdir -p "$OUTROOT"
worker_zero &
pid_zero=$!
worker_one &
pid_one=$!
wait "$pid_zero"
wait "$pid_one"
echo "[done] Sen12 T x M susceptibility development matrix complete"
