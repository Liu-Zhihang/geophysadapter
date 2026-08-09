#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAINER="$ROOT/scripts/xdomain/train_pild_sen12_roleaware_v1.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_pild_sen12_lodo_vt_v1.py"
META="${META:-$ROOT/metadata/pild_sen12_training_v2}"
MANIFEST="${MANIFEST:-$META/unified_sample_manifest_v2.csv}"
SUMMARY="${SUMMARY:-$META/protocol_summary_v2.json}"
SPLIT="${SPLIT:-$META/leave_one_dataset_out_split_v2.csv}"
OUTROOT="${OUTROOT:-$ROOT/experiments/revision2026/pild_sen12_roleaware_lodo_v1}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$OUTROOT/analysis_full_oof}"
SEEDS="${SEEDS:-20260722 20260723 20260724 20260725 20260726}"
FOLDS="${FOLDS:-lodo_00_DLR_Landslide_Ref_2025 lodo_01_GDCLD lodo_02_GLaD4CD_v1 lodo_03_SEN12LS_HARMONIZED}"
VARIANTS="${VARIANTS:-V VT}"
GPUS="${GPUS:-0 1}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_MATERIAL="${ALLOW_MATERIAL:-0}"
ALLOW_TRIGGER="${ALLOW_TRIGGER:-0}"

read -r -a seed_array <<<"$SEEDS"
read -r -a fold_array <<<"$FOLDS"
read -r -a variant_array <<<"$VARIANTS"
read -r -a gpu_array <<<"$GPUS"

(( ${#seed_array[@]} >= 5 )) || {
  printf 'Formal LODO requires at least five seeds; observed %s\n' "${#seed_array[@]}" >&2
  exit 2
}
(( ${#gpu_array[@]} >= 1 )) || { printf 'GPUS is empty\n' >&2; exit 2; }

contains_variant() {
  local wanted="$1" value
  for value in "${variant_array[@]}"; do
    [[ "$value" == "$wanted" ]] && return 0
  done
  return 1
}

contains_variant V || { printf 'VARIANTS must contain V\n' >&2; exit 2; }
if contains_variant VT || contains_variant VTM || contains_variant VTR || contains_variant VTMR; then
  contains_variant VT || { printf 'Every physical variant requires VT in VARIANTS\n' >&2; exit 2; }
fi
if contains_variant VTM || contains_variant VTMR; then
  [[ "$ALLOW_MATERIAL" == 1 ]] || {
    printf 'Material has not been promoted; set ALLOW_MATERIAL=1 only with a frozen passing receipt\n' >&2
    exit 2
  }
fi
if contains_variant VTR || contains_variant VTMR; then
  [[ "$ALLOW_TRIGGER" == 1 ]] || {
    printf 'Trigger has not been promoted; set ALLOW_TRIGGER=1 only with a frozen passing receipt\n' >&2
    exit 2
  }
fi

for path in "$TRAINER" "$ANALYZER" "$MANIFEST" "$SUMMARY" "$SPLIT"; do
  [[ -s "$path" ]] || { printf 'missing prerequisite: %s\n' "$path" >&2; exit 2; }
done

"$PYTHON" - "$SUMMARY" "$DRY_RUN" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
readiness = payload.get("readiness", {})
if not readiness.get("core_training_ready", False):
    message = f"core training is not ready: {readiness.get('blockers', [])}"
    if sys.argv[2] == "1":
        print(f"[dry-run warning] {message}", file=sys.stderr)
    else:
        raise SystemExit(message)
PY

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

run_chain() {
  local gpu="$1" fold="$2" seed="$3" variant out parent
  local fold_root="$OUTROOT/$fold"
  for variant in "${variant_array[@]}"; do
    out="$fold_root/${variant}_seed${seed}"
    if complete_run "$out"; then
      printf '[skip] fold=%s seed=%s variant=%s\n' "$fold" "$seed" "$variant"
      continue
    fi
    parent=()
    if [[ "$variant" != V ]]; then
      local parent_path="$fold_root/VT_seed${seed}/checkpoint.pt"
      [[ "$variant" == VT ]] && parent_path="$fold_root/V_seed${seed}/checkpoint.pt"
      if [[ "$DRY_RUN" != 1 && ! -s "$parent_path" ]]; then
        printf 'missing parent checkpoint for %s: %s\n' "$variant" "$parent_path" >&2
        return 3
      fi
      parent=(--parent-checkpoint "$parent_path")
    fi
    local command=(
      "$PYTHON" "$TRAINER"
      --manifest "$MANIFEST"
      --protocol-summary "$SUMMARY"
      --split "$SPLIT"
      --fold-id "$fold"
      --variant "$variant"
      --seed "$seed"
      --outdir "$out"
      --epochs "$EPOCHS"
      --batch-size "$BATCH_SIZE"
      --num-workers "$NUM_WORKERS"
      --epoch-samples "$EPOCH_SAMPLES"
      --max-steps "$MAX_STEPS"
      --device cuda
      "${parent[@]}"
    )
    printf '[start] gpu=%s fold=%s seed=%s variant=%s time=%s\n' \
      "$gpu" "$fold" "$seed" "$variant" "$(date --iso-8601=seconds)"
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
    if [[ "$DRY_RUN" != 1 ]] && ! complete_run "$out"; then
      printf 'artifact gate failed: %s\n' "$out" >&2
      return 4
    fi
  done
}

tasks=()
for fold in "${fold_array[@]}"; do
  "$PYTHON" "$TRAINER" --manifest "$MANIFEST" --protocol-summary "$SUMMARY" \
    --split "$SPLIT" --fold-id "$fold" --variant V --validate-only >/dev/null
  for seed in "${seed_array[@]}"; do
    tasks+=("$fold|$seed")
  done
done

worker() {
  local worker_index="$1" gpu="$2" index task fold seed
  for ((index=worker_index; index<${#tasks[@]}; index+=${#gpu_array[@]})); do
    task="${tasks[$index]}"
    IFS='|' read -r fold seed <<<"$task"
    run_chain "$gpu" "$fold" "$seed"
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
printf '[complete] folds=%s seeds=%s variants=%s time=%s\n' \
  "${#fold_array[@]}" "${#seed_array[@]}" "$VARIANTS" "$(date --iso-8601=seconds)"

if [[ "$VARIANTS" == "V VT" ]]; then
  run_command "$PYTHON" "$ANALYZER" \
    --runs-root "$OUTROOT" \
    --split "$SPLIT" \
    --outdir "$ANALYSIS_OUT" \
    --min-seeds 5
fi
