#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAINER="$ROOT/scripts/xdomain/train_pild_sen12_roleaware_v1.py"
ANALYZER="$ROOT/scripts/xdomain/summarize_pild_sen12_crossdataset_quickgate_v1.py"
META="${META:-$ROOT/metadata/pild_sen12_training_v2}"
MANIFEST="${MANIFEST:-$META/unified_sample_manifest_v2.csv}"
SUMMARY="${SUMMARY:-$META/protocol_summary_v2.json}"
SPLIT="${SPLIT:-$META/leave_one_dataset_out_split_v2.csv}"
OUTROOT="${OUTROOT:-$ROOT/experiments/revision2026/pild_sen12_crossdataset_quickgate_v1}"
FOLDS="${FOLDS:-lodo_01_GDCLD lodo_02_GLaD4CD_v1 lodo_03_SEN12LS_HARMONIZED}"
GPUS="${GPUS:-0 1}"
SEED="${SEED:-20260723}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a fold_array <<<"$FOLDS"
read -r -a gpu_array <<<"$GPUS"
(( ${#fold_array[@]} > 0 )) || { printf 'FOLDS is empty\n' >&2; exit 2; }
(( ${#gpu_array[@]} > 0 )) || { printf 'GPUS is empty\n' >&2; exit 2; }

for path in "$TRAINER" "$ANALYZER" "$MANIFEST" "$SUMMARY" "$SPLIT"; do
  [[ -s "$path" ]] || { printf 'missing prerequisite: %s\n' "$path" >&2; exit 2; }
done

complete_run() {
  local directory="$1"
  [[ -s "$directory/DONE.json" && -s "$directory/result.json" \
     && -s "$directory/checkpoint.pt" && -s "$directory/per_sample.csv" ]]
}

run_command() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_fold() {
  local gpu="$1" fold="$2" fold_root visual terrain
  fold_root="$OUTROOT/$fold"
  visual="$fold_root/V_seed${SEED}"
  terrain="$fold_root/VT_seed${SEED}"

  "$PYTHON" "$TRAINER" \
    --manifest "$MANIFEST" \
    --protocol-summary "$SUMMARY" \
    --split "$SPLIT" \
    --fold-id "$fold" \
    --variant V \
    --validate-only >/dev/null

  if ! complete_run "$visual"; then
    printf '[start] gpu=%s fold=%s variant=V seed=%s time=%s\n' \
      "$gpu" "$fold" "$SEED" "$(date --iso-8601=seconds)"
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAINER" \
      --manifest "$MANIFEST" \
      --protocol-summary "$SUMMARY" \
      --split "$SPLIT" \
      --fold-id "$fold" \
      --variant V \
      --seed "$SEED" \
      --outdir "$visual" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-steps "$MAX_STEPS" \
      --device cuda
  else
    printf '[skip] fold=%s variant=V seed=%s\n' "$fold" "$SEED"
  fi

  if [[ "$DRY_RUN" != 1 && ! -s "$visual/checkpoint.pt" ]]; then
    printf 'missing visual checkpoint: %s\n' "$visual/checkpoint.pt" >&2
    return 3
  fi

  if ! complete_run "$terrain"; then
    printf '[start] gpu=%s fold=%s variant=VT seed=%s time=%s\n' \
      "$gpu" "$fold" "$SEED" "$(date --iso-8601=seconds)"
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAINER" \
      --manifest "$MANIFEST" \
      --protocol-summary "$SUMMARY" \
      --split "$SPLIT" \
      --fold-id "$fold" \
      --variant VT \
      --seed "$SEED" \
      --parent-checkpoint "$visual/checkpoint.pt" \
      --outdir "$terrain" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-steps "$MAX_STEPS" \
      --device cuda
  else
    printf '[skip] fold=%s variant=VT seed=%s\n' "$fold" "$SEED"
  fi

  if [[ "$DRY_RUN" != 1 ]] && ! complete_run "$terrain"; then
    printf 'artifact gate failed: %s\n' "$terrain" >&2
    return 4
  fi
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#fold_array[@]}; index+=${#gpu_array[@]})); do
    run_fold "$gpu" "${fold_array[$index]}"
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

run_command "$PYTHON" "$ANALYZER" \
  --runs-root "$OUTROOT" \
  --seed "$SEED" \
  --out "$OUTROOT/quickgate_summary.json"
