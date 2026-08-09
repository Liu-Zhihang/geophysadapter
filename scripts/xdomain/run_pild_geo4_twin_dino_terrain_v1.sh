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
VISUAL_ROOT="${VISUAL_ROOT:-$ROOT/experiments/pild_geo4_twin_dinov2_discovery_20260724}"
TERRAIN_ROOT="${TERRAIN_ROOT:-$ROOT/experiments/pild_geo4_source_stratified_probe_v1}"
OUT_NAME="${OUT_NAME:-terrain_decision_margin_v1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_ALPHA="${MAX_ALPHA:-4.0}"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local fold_id="source_stratified_${fold}"
  local visual="$VISUAL_ROOT/$fold_id/seed${SEED}/checkpoints/best_model.pt"
  local visual_per_sample="$VISUAL_ROOT/$fold_id/seed${SEED}/metrics/per_sample_metrics.csv"
  local terrain="$TERRAIN_ROOT/$fold_id/seed${SEED}/terrain/terrain_expert.pt"
  local outdir="$VISUAL_ROOT/$fold_id/seed${SEED}/$OUT_NAME"

  [[ -s "$visual" ]] || { printf 'missing visual checkpoint: %s\n' "$visual" >&2; return 2; }
  [[ -s "$visual_per_sample" ]] || {
    printf 'missing visual per-sample metrics: %s\n' "$visual_per_sample" >&2
    return 2
  }
  [[ -s "$terrain" ]] || { printf 'missing terrain checkpoint: %s\n' "$terrain" >&2; return 2; }
  if [[ -s "$outdir/DONE.json" ]]; then
    printf 'skip complete fold %s: %s\n' "$fold" "$outdir"
    return 0
  fi
  [[ ! -e "$outdir" ]] || {
    printf 'refusing non-complete output directory: %s\n' "$outdir" >&2
    return 2
  }

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/scripts/xdomain/evaluate_pild_geo4_twin_dino_terrain_v1.py" \
    --manifest "$MANIFEST" \
    --protocol-summary "$PROTOCOL" \
    --split "$SPLIT" \
    --fold-id "$fold_id" \
    --seed "$SEED" \
    --visual-checkpoint "$visual" \
    --visual-per-sample "$visual_per_sample" \
    --terrain-checkpoint "$terrain" \
    --outdir "$outdir" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-alpha "$MAX_ALPHA" \
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

printf 'completed all Twin DINOv2-S Terrain folds\n'
