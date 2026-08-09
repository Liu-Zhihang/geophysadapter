#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data_hdd/滑坡检测/physics_informed_landslide_dataset}"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
SEED="${SEED:-20260725}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
MANIFEST="${MANIFEST:-$ROOT/metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv}"
PROTOCOL="${PROTOCOL:-$ROOT/metadata/pild_geo4_qc_v1/protocol_summary_training_v1.json}"
SPLIT="${SPLIT:-$ROOT/metadata/pild_geo4_qc_v1/source_stratified_event_folds_v1.csv}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/pild_geo4_twin_dinov2_deterministic_20260724}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local fold_id="source_stratified_${fold}"
  local outdir="$OUT_ROOT/$fold_id/seed${SEED}"
  if [[ -s "$outdir/DONE.json" ]]; then
    printf 'skip complete fold %s: %s\n' "$fold" "$outdir"
    return 0
  fi
  [[ ! -e "$outdir" ]] || {
    printf 'refusing non-complete output directory: %s\n' "$outdir" >&2
    return 2
  }
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/scripts/xdomain/train_pild_geo4_twin_dinov2_v1.py" \
    --manifest "$MANIFEST" \
    --protocol-summary "$PROTOCOL" \
    --split "$SPLIT" \
    --fold-id "$fold_id" \
    --seed "$SEED" \
    --outdir "$outdir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --device cuda
}

run_fold 0 "$GPU0" &
pid0=$!
run_fold 1 "$GPU1" &
pid1=$!
wait "$pid0"
wait "$pid1"

run_fold 2 "$GPU0" &
pid2=$!
run_fold 3 "$GPU1" &
pid3=$!
wait "$pid2"
wait "$pid3"

printf 'completed deterministic Twin DINOv2-S baseline\n'
