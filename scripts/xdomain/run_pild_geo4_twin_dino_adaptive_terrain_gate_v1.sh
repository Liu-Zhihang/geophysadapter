#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${PILD_ROOT}}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-20260725}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
MANIFEST="$ROOT/metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv"
PROTOCOL="$ROOT/metadata/pild_geo4_qc_v1/protocol_summary_training_v1.json"
SPLIT="$ROOT/metadata/pild_geo4_qc_v1/source_stratified_event_folds_v1.csv"
VISUAL_ROOT="${VISUAL_ROOT:-$ROOT/experiments/pild_geo4_twin_dinov2_deterministic_20260724}"
TERRAIN_ROOT="${TERRAIN_ROOT:-$ROOT/experiments/pild_geo4_source_stratified_probe_v1}"
OUT_NAME="${OUT_NAME:-adaptive_terrain_gate_v1}"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local fold_id="source_stratified_${fold}"
  local visual_run="$VISUAL_ROOT/$fold_id/seed${SEED}"
  local outdir="$visual_run/$OUT_NAME"
  if [[ -s "$outdir/DONE.json" ]]; then
    printf 'skip complete fold %s\n' "$fold"
    return 0
  fi
  [[ ! -e "$outdir" ]] || {
    printf 'refusing non-complete output: %s\n' "$outdir" >&2
    return 2
  }
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/scripts/xdomain/train_pild_geo4_twin_dino_adaptive_terrain_gate_v1.py" \
    --manifest "$MANIFEST" \
    --protocol-summary "$PROTOCOL" \
    --split "$SPLIT" \
    --fold-id "$fold_id" \
    --seed "$SEED" \
    --visual-checkpoint "$visual_run/checkpoints/best_model.pt" \
    --visual-per-sample "$visual_run/metrics/per_sample_metrics.csv" \
    --terrain-checkpoint "$TERRAIN_ROOT/$fold_id/seed${SEED}/terrain/terrain_expert.pt" \
    --outdir "$outdir" \
    --epochs 12 \
    --epoch-samples 1024 \
    --batch-size 8 \
    --num-workers 8 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --alpha-max 1.0 \
    --gate-penalty 0.02 \
    --min-validation-delta-iou 0.002 \
    --min-validation-rer 0.02 \
    --device cuda
}

run_fold 0 "$GPU0" &
p0=$!
run_fold 1 "$GPU1" &
p1=$!
wait "$p0"
wait "$p1"
run_fold 2 "$GPU0" &
p2=$!
run_fold 3 "$GPU1" &
p3=$!
wait "$p2"
wait "$p3"
printf 'completed adaptive Terrain gate folds\n'
