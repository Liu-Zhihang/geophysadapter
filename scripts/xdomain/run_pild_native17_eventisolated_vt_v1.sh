#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
TRAINER="$ROOT/scripts/xdomain/train_pild_sen12_roleaware_v1.py"
META="${META:-$ROOT/metadata/pild_sen12_training_native17_v1}"
MANIFEST="${MANIFEST:-$META/unified_sample_manifest_native17_v1.csv}"
SUMMARY="${SUMMARY:-$META/protocol_summary_native17_v1.json}"
SPLIT="${SPLIT:-$ROOT/metadata/pild_sen12_training_v2/event_isolated_split_v2.csv}"
OUTROOT="${OUTROOT:-$ROOT/experiments/revision2026/pild_native17_eventisolated_vt_v1}"
SEEDS="${SEEDS:-20260724 20260725}"
GPUS="${GPUS:-0 1}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a seed_array <<< "$SEEDS"
read -r -a gpu_array <<< "$GPUS"
(( ${#gpu_array[@]} > 0 )) || { echo "GPUS is empty" >&2; exit 2; }

for path in "$TRAINER" "$MANIFEST" "$SUMMARY" "$SPLIT"; do
  [[ -s "$path" ]] || { echo "missing prerequisite: $path" >&2; exit 2; }
done

"$PYTHON" "$TRAINER" \
  --manifest "$MANIFEST" --protocol-summary "$SUMMARY" --split "$SPLIT" \
  --fold-id event_isolated --variant V --validate-only >/dev/null

complete_run() {
  local directory="$1"
  [[ -s "$directory/DONE.json" && -s "$directory/result.json" \
     && -s "$directory/checkpoint.pt" && -s "$directory/per_sample.csv" ]]
}

run_seed() {
  local seed="$1" gpu="$2"
  local seed_root="$OUTROOT/event_isolated/seed${seed}"
  local visual="$seed_root/V"
  local terrain="$seed_root/VT"
  local common=(
    --manifest "$MANIFEST"
    --protocol-summary "$SUMMARY"
    --split "$SPLIT"
    --fold-id event_isolated
    --seed "$seed"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --sampling-mode natural
    --epoch-samples 0
    --max-steps "$MAX_STEPS"
    --device cuda
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q %q %q --variant V --outdir %q ' \
      "$gpu" "$PYTHON" "$TRAINER" "$visual"
    printf '%q ' "${common[@]}"
    printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=%q %q %q --variant VT --parent-checkpoint %q --outdir %q ' \
      "$gpu" "$PYTHON" "$TRAINER" "$visual/checkpoint.pt" "$terrain"
    printf '%q ' "${common[@]}"
    printf '\n'
    return
  fi

  mkdir -p "$seed_root"
  if ! complete_run "$visual"; then
    printf '[start] seed=%s variant=V gpu=%s time=%s\n' \
      "$seed" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTROOT/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAINER" \
      --variant V --outdir "$visual" "${common[@]}"
  fi
  if ! complete_run "$visual"; then
    echo "visual artifact gate failed: $visual" >&2
    return 3
  fi
  if ! complete_run "$terrain"; then
    printf '[start] seed=%s variant=VT gpu=%s time=%s\n' \
      "$seed" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTROOT/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAINER" \
      --variant VT --parent-checkpoint "$visual/checkpoint.pt" \
      --outdir "$terrain" "${common[@]}"
  fi
  if ! complete_run "$terrain"; then
    echo "Terrain artifact gate failed: $terrain" >&2
    return 4
  fi
  printf '[done] seed=%s gpu=%s time=%s\n' \
    "$seed" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTROOT/progress.log"
}

if [[ "$DRY_RUN" == "1" ]]; then
  for index in "${!seed_array[@]}"; do
    run_seed "${seed_array[$index]}" "${gpu_array[$((index % ${#gpu_array[@]}))]}"
  done
  exit 0
fi

mkdir -p "$OUTROOT"
printf '%s\n' "$0 $*" > "$OUTROOT/coordinator.command.txt"
printf '{"status":"running","seeds":"%s","gpus":"%s","terrain_contract":"native17","sampling":"natural"}\n' \
  "$SEEDS" "$GPUS" > "$OUTROOT/run_manifest.json"

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#seed_array[@]}; index+=${#gpu_array[@]})); do
    run_seed "${seed_array[$index]}" "$gpu"
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
printf '{"status":"complete","seeds":"%s","gpus":"%s","terrain_contract":"native17","sampling":"natural"}\n' \
  "$SEEDS" "$GPUS" > "$OUTROOT/run_manifest.json"
