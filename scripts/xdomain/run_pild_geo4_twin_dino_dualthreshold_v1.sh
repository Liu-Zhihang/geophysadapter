#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${PILD_ROOT}}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-20260725}"
VISUAL_ROOT="${VISUAL_ROOT:-$ROOT/experiments/pild_geo4_twin_dinov2_deterministic_20260724}"
TERRAIN_ROOT="${TERRAIN_ROOT:-$ROOT/experiments/pild_geo4_source_stratified_probe_v1}"
OUT_NAME="${OUT_NAME:-terrain_dualthreshold_sen12fixed_v1}"
GPUS=("${GPU0:-0}" "${GPU1:-1}")

run_fold() {
  local fold="$1"
  local gpu="$2"
  local fold_id="source_stratified_${fold}"
  local visual="$VISUAL_ROOT/$fold_id/seed${SEED}"
  local outdir="$visual/$OUT_NAME"
  [[ ! -e "$outdir" ]] || { printf 'output exists: %s\n' "$outdir" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$ROOT/scripts/xdomain/evaluate_pild_geo4_twin_dino_dualthreshold_v1.py" \
    --manifest "$ROOT/metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv" \
    --protocol-summary "$ROOT/metadata/pild_geo4_qc_v1/protocol_summary_training_v1.json" \
    --split "$ROOT/metadata/pild_geo4_qc_v1/source_stratified_event_folds_v1.csv" \
    --fold-id "$fold_id" \
    --seed "$SEED" \
    --visual-checkpoint "$visual/checkpoints/best_model.pt" \
    --visual-per-sample "$visual/metrics/per_sample_metrics.csv" \
    --terrain-checkpoint "$TERRAIN_ROOT/$fold_id/seed${SEED}/terrain/terrain_expert.pt" \
    --outdir "$outdir" \
    --low 0.3 --high 0.7 --alpha 4.0 --visual-margin 1.0 \
    --batch-size 8 --num-workers 8 --device cuda
}

run_fold 0 "${GPUS[0]}" & p0=$!
run_fold 1 "${GPUS[1]}" & p1=$!
wait "$p0"; wait "$p1"
run_fold 2 "${GPUS[0]}" & p2=$!
run_fold 3 "${GPUS[1]}" & p3=$!
wait "$p2"; wait "$p3"
