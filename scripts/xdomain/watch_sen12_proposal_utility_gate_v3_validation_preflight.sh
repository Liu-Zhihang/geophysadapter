#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
AGGREGATOR="$SCRIPT_DIR/build_sen12_proposal_utility_gate_v3_manifests.py"
PREFLIGHT="$SCRIPT_DIR/preflight_sen12_proposal_utility_gate_v3_validation.py"
PROPOSER_ROOT="${PROPOSER_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_nested_oof_proposer_cache_v1}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$PROJECT_ROOT/metadata/pild_xdomain_v1/sen12_nested_oof_protocol_v1}"
FORMAL_INPUT_ROOT="${FORMAL_INPUT_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_proposal_utility_gate_v3/formal_inputs_v1}"
PREFLIGHT_OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:-$PROJECT_ROOT/experiments/revision2026/sen12_proposal_utility_gate_v3/validation_preflight_v1}"
MATERIAL_REGISTRY="${MATERIAL_REGISTRY:-$PROJECT_ROOT/processed/hybrid_pinn/sen12_context_v2/material_sample_registry_v2.csv}"
MATERIAL_SCHEMA="${MATERIAL_SCHEMA:-$PROJECT_ROOT/processed/hybrid_pinn/sen12_context_v2/material_feature_schema_v2.json}"
TRIGGER_REGISTRY="${TRIGGER_REGISTRY:-$PROJECT_ROOT/processed/hybrid_pinn/sen12_context_v1/trigger_sample_registry_v1.csv}"
SEED="${SEED:-20260751}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-86400}"
DRY_RUN="${DRY_RUN:-0}"

required_paths() {
  local target inner run
  for target in 0 1 2 3 4; do
    for inner in 0 1 2; do
      run="$PROPOSER_ROOT/target_outer${target}/inner_fold${inner}/seed${SEED}"
      printf '%s\n' \
        "$run/run_manifest.json" \
        "$run/cache/inner_test_proposer_cache.pt" \
        "$run/DONE.json"
    done
  done
}

missing_count() {
  local count=0 path
  while IFS= read -r path; do
    [[ -f "$path" ]] || count=$((count + 1))
  done < <(required_paths)
  printf '%d\n' "$count"
}

waited=0
while true; do
  missing="$(missing_count)"
  if [[ "$missing" == "0" ]]; then
    printf '[watcher] proposer artifacts present: 15/15 tasks, 45/45 required files\n'
    break
  fi
  if [[ "$DRY_RUN" == "1" || "$waited" -ge "$MAX_WAIT_SECONDS" ]]; then
    echo "[watcher] fail closed: $missing/45 proposer artifacts missing after ${waited}s" >&2
    exit 1
  fi
  printf '[watcher] waiting: missing=%s/45 elapsed=%ss\n' "$missing" "$waited"
  sleep "$POLL_SECONDS"
  waited=$((waited + POLL_SECONDS))
done

aggregator_command=(
  "$PYTHON" "$AGGREGATOR"
  --input-root "$PROPOSER_ROOT"
  --protocol-root "$PROTOCOL_ROOT"
  --output-root "$FORMAL_INPUT_ROOT"
  --material-registry "$MATERIAL_REGISTRY"
  --material-schema "$MATERIAL_SCHEMA"
  --trigger-registry "$TRIGGER_REGISTRY"
  --seed "$SEED"
)
preflight_command=(
  "$PYTHON" "$PREFLIGHT"
  --formal-input-root "$FORMAL_INPUT_ROOT"
  --output-root "$PREFLIGHT_OUTPUT_ROOT"
  --seed "$SEED"
)

if [[ "$DRY_RUN" == "1" ]]; then
  "${aggregator_command[@]}" --dry-run
  printf '[DRY_RUN aggregator]'; printf ' %q' "${aggregator_command[@]}"; printf '\n'
  printf '[DRY_RUN preflight]'; printf ' %q' "${preflight_command[@]}"; printf '\n'
  printf '%s\n' \
    '[DRY_RUN contract] flow=15/15 proposer -> formal aggregator -> validation-only preflight' \
    '[DRY_RUN contract] target outer-test cache/root is never passed to preflight' \
    '[DRY_RUN summary] outer_test_started=0 gate_trainer_started=0'
  exit 0
fi

if [[ ! -e "$FORMAL_INPUT_ROOT" ]]; then
  "${aggregator_command[@]}"
else
  [[ -f "$FORMAL_INPUT_ROOT/DONE.json" ]] || {
    echo "Existing formal input root lacks DONE.json" >&2
    exit 1
  }
  newest_proposer_done=0
  while IFS= read -r path; do
    [[ "$path" == */DONE.json ]] || continue
    mtime="$(stat -c %Y "$path")"
    (( mtime > newest_proposer_done )) && newest_proposer_done="$mtime"
  done < <(required_paths)
  formal_mtime="$(stat -c %Y "$FORMAL_INPUT_ROOT/DONE.json")"
  if (( formal_mtime < newest_proposer_done )); then
    echo "Existing formal inputs are stale relative to proposer DONE artifacts" >&2
    exit 1
  fi
fi

[[ ! -e "$PREFLIGHT_OUTPUT_ROOT" ]] || {
  echo "Refusing existing validation preflight output: $PREFLIGHT_OUTPUT_ROOT" >&2
  exit 1
}
"${preflight_command[@]}"

"$PYTHON" - "$PREFLIGHT_OUTPUT_ROOT/validation_preflight_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "status": "validation_preflight_complete",
    "eligible_for_outer_test": bool(summary["eligible_for_outer_test"]),
    "context_eligibility": {
        key: bool(value["eligible_for_outer_test"])
        for key, value in summary["eligibility"]["contexts"].items()
    },
    "outer_test_started": False,
    "action": "await_main_agent_review",
}, sort_keys=True))
PY
