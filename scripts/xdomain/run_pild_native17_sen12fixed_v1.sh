#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAINER="$ROOT/scripts/xdomain/train_pild_support_only_terrain_v1.py"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_pild_prithvi_dualthreshold_v1.py"
META="${META:-$ROOT/metadata/pild_sen12_training_native17_v1}"
MANIFEST="${MANIFEST:-$META/unified_sample_manifest_native17_v1.csv}"
SUMMARY="${SUMMARY:-$META/protocol_summary_native17_v1.json}"
SPLIT="${SPLIT:-$ROOT/metadata/pild_sen12_training_v2/event_isolated_split_v2.csv}"
VISUAL_ROOT="${VISUAL_ROOT:-$ROOT/experiments/revision2026/pild_native17_eventisolated_vt_v1}"
OUTROOT="${OUTROOT:-$ROOT/experiments/revision2026/pild_native17_sen12fixed_v1}"
SEED="${SEED:-20260724}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"

VISUAL_DIR="$VISUAL_ROOT/event_isolated/seed${SEED}/V"
TERRAIN_DIR="$OUTROOT/event_isolated/seed${SEED}/terrain_expert"
EVAL_DIR="$OUTROOT/event_isolated/seed${SEED}/dualthreshold_test"

for path in "$VISUAL_DIR/checkpoint.pt" "$VISUAL_DIR/per_sample.csv"; do
  [[ -s "$path" ]] || { echo "missing visual parent artifact: $path" >&2; exit 2; }
done

mkdir -p "$(dirname "$TERRAIN_DIR")"
if [[ ! -s "$TERRAIN_DIR/DONE.json" || ! -s "$TERRAIN_DIR/terrain_expert.pt" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$TRAINER" \
    --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
    --fold-id event_isolated --seed "$SEED" \
    --parent-v-checkpoint "$VISUAL_DIR/checkpoint.pt" \
    --outdir "$TERRAIN_DIR" --epochs "$EPOCHS" --batch-size 16 \
    --sampling-mode natural --num-workers "$NUM_WORKERS" --device cuda
fi

if [[ ! -s "$EVAL_DIR/DONE.json" || ! -s "$EVAL_DIR/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$EVALUATOR" \
    --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
    --fold-id event_isolated --seed "$SEED" \
    --visual-checkpoint "$VISUAL_DIR/checkpoint.pt" \
    --visual-per-sample "$VISUAL_DIR/per_sample.csv" \
    --terrain-checkpoint "$TERRAIN_DIR/terrain_expert.pt" \
    --outdir "$EVAL_DIR" --low 0.3 --high 0.7 --alpha 4.0 \
    --visual-margin 1.0 --batch-size 16 --num-workers "$NUM_WORKERS" \
    --device cuda
fi
