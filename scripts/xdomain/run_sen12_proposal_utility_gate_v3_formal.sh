#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
AGGREGATOR="$SCRIPT_DIR/build_sen12_proposal_utility_gate_v3_manifests.py"
TRAINER="$SCRIPT_DIR/train_sen12_proposal_utility_gate_v3.py"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_nested_oof_proposer_cache_v1}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$PROJECT_ROOT/metadata/pild_xdomain_v1/sen12_nested_oof_protocol_v1}"
FORMAL_INPUT_ROOT="${FORMAL_INPUT_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_proposal_utility_gate_v3/formal_inputs_v1}"
GATE_OUT_ROOT="${GATE_OUT_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_proposal_utility_gate_v3/formal_nested_oof}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_proposal_utility_gate_v3/launcher_logs}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_prithvi_roleaware_hierarchical_v2/cache/seed20260751}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_prithvi_roleaware_hierarchical_v2/seed20260751}"
MATERIAL_REGISTRY="${MATERIAL_REGISTRY:-$PROJECT_ROOT/processed/hybrid_pinn/sen12_context_v2/material_sample_registry_v2.csv}"
MATERIAL_SCHEMA="${MATERIAL_SCHEMA:-$PROJECT_ROOT/processed/hybrid_pinn/sen12_context_v2/material_feature_schema_v2.json}"
TRIGGER_REGISTRY="${TRIGGER_REGISTRY:-$PROJECT_ROOT/processed/hybrid_pinn/sen12_context_v1/trigger_sample_registry_v1.csv}"
SEED="${SEED:-20260751}"
GPUS="${GPUS:-0,1}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a gpu_values <<< "$GPUS"
if (( ${#gpu_values[@]} != 2 )); then
  echo "Formal runner requires exactly two GPU queue identifiers, e.g. GPUS=0,1" >&2
  exit 2
fi

aggregator_command=(
  "$PYTHON" "$AGGREGATOR"
  --input-root "$INPUT_ROOT"
  --protocol-root "$PROTOCOL_ROOT"
  --output-root "$FORMAL_INPUT_ROOT"
  --material-registry "$MATERIAL_REGISTRY"
  --material-schema "$MATERIAL_SCHEMA"
  --trigger-registry "$TRIGGER_REGISTRY"
  --seed "$SEED"
)
if [[ "$DRY_RUN" == "1" ]]; then
  aggregator_command+=(--dry-run)
fi

# This is an executable preflight, not a cosmetic preview. Missing/stale/tampered
# proposer evidence stops both DRY_RUN and real launch before any gate process.
"${aggregator_command[@]}"

if [[ "$DRY_RUN" != "1" ]]; then
  for required in aggregate_summary.json hashes.json DONE.json; do
    [[ -f "$FORMAL_INPUT_ROOT/$required" ]] || {
      echo "Formal aggregate missing $required" >&2
      exit 1
    }
  done
fi

commands=()
task_gpus=()
for target in 0 1 2 3 4; do
  gpu="${gpu_values[$((target % 2))]}"
  manifest="$FORMAL_INPUT_ROOT/target_outer${target}/oof_manifest.json"
  split_csv="$FORMAL_INPUT_ROOT/target_outer${target}/gate_split.csv"
  outdir="$GATE_OUT_ROOT/fold${target}_seed${SEED}"
  if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$manifest" && -f "$split_csv" ]] || {
      echo "Target $target formal manifest/split missing" >&2
      exit 1
    }
    [[ ! -e "$outdir" ]] || {
      echo "Refusing existing gate output (stale/ambiguous): $outdir" >&2
      exit 1
    }
  fi
  command=(
    "$PYTHON" "$TRAINER"
    --target-fold "$target"
    --protocol-mode formal_nested_oof
    --oof-manifest "$manifest"
    --split-csv "$split_csv"
    --cache-root "$CACHE_ROOT"
    --runs-root "$RUNS_ROOT"
    --seed "$SEED"
    --outdir "$outdir"
  )
  printf -v rendered '%q ' env "CUDA_VISIBLE_DEVICES=$gpu" "${command[@]}"
  commands+=("$rendered")
  task_gpus+=("$gpu")
done

if [[ "$DRY_RUN" == "1" ]]; then
  for index in "${!commands[@]}"; do
    printf '[DRY_RUN target=%d gpu=%s] %s\n' "$index" "${task_gpus[$index]}" "${commands[$index]}"
  done
  printf '%s\n' \
    '[DRY_RUN contract] contexts=proposal-only,TM,TR,TMR; all controls reuse each context gate checkpoint; all contexts share one frozen proposer cache' \
    '[DRY_RUN contract] target outer-test is loaded only by the trainer after selection_and_fit_frozen' \
    '[DRY_RUN summary] targets=5 gpu_queues=2 real_gate_started=0'
  exit 0
fi

mkdir -p "$LOG_ROOT"
run_gpu_queue() {
  local queue="$1"
  local index
  for index in "${!commands[@]}"; do
    if (( index % 2 != queue )); then
      continue
    fi
    echo "[launch] target=$index gpu=${task_gpus[$index]}"
    bash -lc "${commands[$index]}" >"$LOG_ROOT/target_outer${index}_seed${SEED}.log" 2>&1
  done
}

pids=()
for queue in 0 1; do
  run_gpu_queue "$queue" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
