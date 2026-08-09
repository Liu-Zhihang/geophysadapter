#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
VISUAL_TRAINER="$ROOT/scripts/xdomain/train_pild_sen12_roleaware_v1.py"
TERRAIN_TRAINER="$ROOT/scripts/xdomain/train_pild_support_only_terrain_v1.py"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_pild_prithvi_dualthreshold_v1.py"
META="${META:-$ROOT/metadata/pild_geo4_qc_native17_v1}"
MANIFEST="${MANIFEST:-$META/unified_sample_manifest_geo4_qc_native17_v1.csv}"
SUMMARY="${SUMMARY:-$META/protocol_summary_geo4_qc_native17_v1.json}"
SPLIT="${SPLIT:-$ROOT/metadata/pild_geo4_qc_v1/source_stratified_event_folds_v1.csv}"
OUTROOT="${OUTROOT:-$ROOT/experiments/revision2026/pild_native17_source_stratified_sen12fixed_v1}"
FOLDS="${FOLDS:-source_stratified_0 source_stratified_1 source_stratified_2 source_stratified_3}"
GPUS="${GPUS:-0 1}"
SEED="${SEED:-20260724}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TERRAIN_EPOCHS="${TERRAIN_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-0}"
SAMPLING_MODE="${SAMPLING_MODE:-natural}"
DATASET_TEMPERATURE="${DATASET_TEMPERATURE:-0.75}"
EVENT_TEMPERATURE="${EVENT_TEMPERATURE:-0.75}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"
(( ${#fold_array[@]} > 0 )) || { echo "FOLDS is empty" >&2; exit 2; }
(( ${#gpu_array[@]} > 0 )) || { echo "GPUS is empty" >&2; exit 2; }
for path in "$VISUAL_TRAINER" "$TERRAIN_TRAINER" "$EVALUATOR" \
  "$MANIFEST" "$SUMMARY" "$SPLIT"; do
  [[ -s "$path" ]] || { echo "missing prerequisite: $path" >&2; exit 2; }
done

complete_visual() {
  local directory="$1"
  [[ -s "$directory/DONE.json" && -s "$directory/result.json" \
     && -s "$directory/checkpoint.pt" && -s "$directory/per_sample.csv" ]]
}

complete_terrain() {
  local directory="$1"
  [[ -s "$directory/DONE.json" && -s "$directory/result.json" \
     && -s "$directory/terrain_expert.pt" ]]
}

complete_eval() {
  local directory="$1"
  [[ -s "$directory/DONE.json" && -s "$directory/result.json" \
     && -s "$directory/per_sample_metrics.csv" \
     && -s "$directory/per_event_metrics.csv" ]]
}

run_fold() {
  local fold="$1" gpu="$2"
  local fold_root="$OUTROOT/$fold/seed${SEED}"
  local visual="$fold_root/V"
  local terrain="$fold_root/terrain_expert"
  local evaluation="$fold_root/dualthreshold_test"
  local common=(
    --manifest "$MANIFEST"
    --protocol-summary "$SUMMARY"
    --split "$SPLIT"
    --fold-id "$fold"
    --seed "$SEED"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --sampling-mode "$SAMPLING_MODE"
    --dataset-temperature "$DATASET_TEMPERATURE"
    --event-temperature "$EVENT_TEMPERATURE"
    --device cuda
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q %q %q --variant V --outdir %q --epochs %q --max-steps %q ' \
      "$gpu" "$PYTHON" "$VISUAL_TRAINER" "$visual" "$VISUAL_EPOCHS" "$MAX_STEPS"
    printf '%q ' "${common[@]}"
    printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=%q %q %q --parent-v-checkpoint %q --outdir %q --epochs %q ' \
      "$gpu" "$PYTHON" "$TERRAIN_TRAINER" "$visual/checkpoint.pt" "$terrain" "$TERRAIN_EPOCHS"
    printf '%q ' "${common[@]}"
    printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=%q %q %q --visual-checkpoint %q --visual-per-sample %q --terrain-checkpoint %q --outdir %q ' \
      "$gpu" "$PYTHON" "$EVALUATOR" "$visual/checkpoint.pt" \
      "$visual/per_sample.csv" "$terrain/terrain_expert.pt" "$evaluation"
    printf '%q ' "${common[@]}"
    printf '%s\n' '--low 0.3 --high 0.7 --alpha 4.0 --visual-margin 1.0'
    return
  fi

  mkdir -p "$fold_root"
  if ! complete_visual "$visual"; then
    printf '[start] fold=%s stage=V gpu=%s time=%s\n' \
      "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTROOT/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VISUAL_TRAINER" \
      --variant V --outdir "$visual" --epochs "$VISUAL_EPOCHS" \
      --max-steps "$MAX_STEPS" "${common[@]}"
  fi
  complete_visual "$visual" || { echo "visual artifact gate failed: $visual" >&2; return 3; }

  if ! complete_terrain "$terrain"; then
    printf '[start] fold=%s stage=terrain gpu=%s time=%s\n' \
      "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTROOT/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TERRAIN_TRAINER" \
      --parent-v-checkpoint "$visual/checkpoint.pt" --outdir "$terrain" \
      --epochs "$TERRAIN_EPOCHS" "${common[@]}"
  fi
  complete_terrain "$terrain" || { echo "Terrain artifact gate failed: $terrain" >&2; return 4; }

  if ! complete_eval "$evaluation"; then
    printf '[start] fold=%s stage=evaluate gpu=%s time=%s\n' \
      "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTROOT/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
      --fold-id "$fold" --seed "$SEED" \
      --visual-checkpoint "$visual/checkpoint.pt" \
      --visual-per-sample "$visual/per_sample.csv" \
      --terrain-checkpoint "$terrain/terrain_expert.pt" \
      --outdir "$evaluation" --low 0.3 --high 0.7 --alpha 4.0 \
      --visual-margin 1.0 --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" --device cuda
  fi
  complete_eval "$evaluation" || { echo "evaluation artifact gate failed: $evaluation" >&2; return 5; }
  printf '[done] fold=%s gpu=%s time=%s\n' \
    "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTROOT/progress.log"
}

if [[ "$DRY_RUN" == "1" ]]; then
  for index in "${!fold_array[@]}"; do
    run_fold "${fold_array[$index]}" "${gpu_array[$((index % ${#gpu_array[@]}))]}"
  done
  exit 0
fi

mkdir -p "$OUTROOT"
printf '%s\n' "$0 $*" > "$OUTROOT/coordinator.command.txt"
printf '{"status":"running","folds":"%s","seed":%s,"gpus":"%s","terrain_contract":"native17","protocol":"source_stratified_event_isolated","sampling":"%s"}\n' \
  "$FOLDS" "$SEED" "$GPUS" "$SAMPLING_MODE" > "$OUTROOT/run_manifest.json"

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#fold_array[@]}; index+=${#gpu_array[@]})); do
    run_fold "${fold_array[$index]}" "$gpu"
  done
}

pids=()
for index in "${!gpu_array[@]}"; do
  worker "$index" "${gpu_array[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"
printf '{"status":"complete","folds":"%s","seed":%s,"gpus":"%s","terrain_contract":"native17","protocol":"source_stratified_event_isolated","sampling":"%s"}\n' \
  "$FOLDS" "$SEED" "$GPUS" "$SAMPLING_MODE" > "$OUTROOT/run_manifest.json"
