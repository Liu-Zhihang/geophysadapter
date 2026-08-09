#!/usr/bin/env bash
# G5: strengthen the visual anchor, then re-test the object-level physical veto on it.
#
# Reviewer 1 objected that the absolute IoU is very low, which is a separate problem from
# whether physics helps. The anchor is therefore improved on its own terms and recorded
# as a visual gain, never as a physics gain: the object veto is re-run afterwards so the
# mechanism has to survive a stronger baseline instead of relying on a weak one.
#
# Two pre-registered arms per fold, both selected only on the fold's validation split:
#   long   : same frozen encoder, twice the schedule and a wider decoder
#   tuned  : additionally opens the last 4 transformer blocks at a reduced encoder step
#
# The frozen-encoder anchors already in
# experiments/revision2026/pild_native17_source_stratified_tempered075_v1 are untouched.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
TRAINER="$ROOT/scripts/xdomain/train_pild_sen12_roleaware_v1.py"
META="${META:-$ROOT/metadata/pild_geo4_qc_native17_v1}"
MANIFEST="${MANIFEST:-$META/unified_sample_manifest_geo4_qc_native17_v1.csv}"
SUMMARY="${SUMMARY:-$META/protocol_summary_geo4_qc_native17_v1.json}"
SPLIT="${SPLIT:-$ROOT/metadata/pild_geo4_qc_v1/source_stratified_event_folds_v1.csv}"
OUT="${OUT:-$ROOT/experiments/revision2026/pild_stronger_anchor_v1}"
SEED="${SEED:-20260724}"
FOLDS="${FOLDS:-source_stratified_0 source_stratified_1 source_stratified_2 source_stratified_3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAMPLING_MODE="${SAMPLING_MODE:-natural}"
LONG_EPOCHS="${LONG_EPOCHS:-60}"
TUNED_EPOCHS="${TUNED_EPOCHS:-40}"
LONG_WIDTH="${LONG_WIDTH:-192}"
TUNED_BLOCKS="${TUNED_BLOCKS:-4}"
ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-0.05}"

mkdir -p "$OUT"
read -r -a fold_array <<< "$FOLDS"

gpu_is_free() {
  local count
  count=$(nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [[ "${count:-1}" -eq 0 ]]
}

wait_for_gpu() {
  while ! gpu_is_free "$1"; do sleep 60; done
  sleep 5
}

train_arm() {
  local gpu="$1" fold="$2" arm="$3"
  local dir="$OUT/$arm/$fold/seed$SEED"
  if [[ -s "$dir/DONE.json" && -s "$dir/checkpoint.pt" ]]; then
    echo "[skip ] $arm/$fold"
    return 0
  fi
  wait_for_gpu "$gpu"
  mkdir -p "$dir"
  local extra=()
  local epochs="$LONG_EPOCHS"
  local width="$LONG_WIDTH"
  if [[ "$arm" == "tuned" ]]; then
    epochs="$TUNED_EPOCHS"
    width=128
    extra=(--unfreeze-encoder-blocks "$TUNED_BLOCKS" --encoder-lr-scale "$ENCODER_LR_SCALE")
  fi
  echo "[start] gpu=$gpu arm=$arm fold=$fold epochs=$epochs width=$width $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$TRAINER" \
    --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
    --fold-id "$fold" --variant V --seed "$SEED" \
    --epochs "$epochs" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --sampling-mode "$SAMPLING_MODE" --decoder-width "$width" \
    "${extra[@]}" --outdir "$dir" > "$OUT/${arm}_${fold}.log" 2>&1
  echo "[done ] gpu=$gpu arm=$arm fold=$fold exit=$? $(date -Is)"
}

queue_long() {
  for fold in "${fold_array[@]}"; do train_arm 0 "$fold" long; done
}

queue_tuned() {
  for fold in "${fold_array[@]}"; do train_arm 1 "$fold" tuned; done
}

queue_long &
queue_tuned &
wait
echo "[anchor] complete $(date -Is)"
