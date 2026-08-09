#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_dlr_validation_calibrated.py"
PARENT="${PARENT:?PARENT is required}"
SEED="${SEED:-20260724}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"
SOURCE_TAG="${SOURCE_TAG:-boundedalpha4_calibrated}"
OUTPUT_TAG="${OUTPUT_TAG:-eventconsistent_boundedalpha4}"
MIN_POSITIVE_FRACTION="${MIN_POSITIVE_FRACTION:-0.75}"

CACHE="${CACHE:-$ROOT/processed/hybrid_pinn/dlr_geo4qc_sen12_protocol_v1}"
BASE_H5="${BASE_H5:-$CACHE/dlr_base_temporalvalid_p128.h5}"
OPTICAL_H5="${OPTICAL_H5:-$CACHE/dlr_prithvi_4t6b_p128.h5}"
TERRAIN_H5="${TERRAIN_H5:-$CACHE/dlr_common_terrain9_p128.h5}"
SPLIT_CSV="${SPLIT_CSV:?SPLIT_CSV is required}"

read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"

run_fold() {
  local fold="$1" gpu="$2"
  local task="$PARENT/seed${SEED}/fold${fold}"
  local source_selection="$task/${SOURCE_TAG}_val/selection.json"
  local audit_dir="$task/${OUTPUT_TAG}_val"
  local test_dir="$task/${OUTPUT_TAG}_test"
  mkdir -p "$audit_dir" "$test_dir"
  if [[ ! -f "$source_selection" ]]; then
    echo "Missing validation selection: $source_selection" >&2
    return 2
  fi
  local source_config
  source_config="$(
    "$PYTHON" -c \
      'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["config"])))' \
      "$source_selection"
  )"
  if [[ ! -f "$audit_dir/result.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" \
      --terrain-h5 "$TERRAIN_H5" --split-csv "$SPLIT_CSV" \
      --fold "$fold" --seed "$SEED" --split val --routing-mode both \
      --fixed-config "$source_config" --emit-per-region \
      --visual-checkpoint "$task/visual/checkpoint.pt" \
      --terrain-checkpoint "$task/terrain/terrain_expert.pt" \
      --outdir "$audit_dir" --batch-size 32 --num-workers 2 --device cuda \
      2>&1 | tee "$audit_dir/run.log"
  fi
  "$PYTHON" - \
    "$audit_dir/result.json" "$source_selection" "$audit_dir/selection.json" \
    "$MIN_POSITIVE_FRACTION" <<'PY'
import json
import math
import pathlib
import statistics
import sys

audit_path, source_path, output_path = map(pathlib.Path, sys.argv[1:4])
minimum_fraction = float(sys.argv[4])
audit = json.loads(audit_path.read_text())
source = json.loads(source_path.read_text())
row = audit["grid"][0]
regions = row.get("per_region", {})
if len(regions) < 3:
    raise RuntimeError(f"event consistency requires at least 3 validation events, got {len(regions)}")
delta = [float(value["delta_iou"]) for value in regions.values()]
rer = [float(value["rer"]) for value in regions.values()]
positive = sum(d >= 0.0 and r >= 0.0 for d, r in zip(delta, rer, strict=True))
required = math.ceil(minimum_fraction * len(regions))
macro_delta = statistics.fmean(delta)
macro_rer = statistics.fmean(rer)
source_config = [float(value) for value in source["config"]]
identity = source_config[2] == 0.0
gate_pass = identity or (
    positive >= required
    and macro_delta >= 0.0
    and macro_rer >= 0.0
    and row["delta_iou"] >= 0.0
    and row["rer"] >= 0.0
)
selected = source_config if gate_pass else [0.2, 0.5, 0.0, 0.1]
receipt = {
    "status": "frozen_from_validation_event_consistency",
    "source_config": source_config,
    "config": selected,
    "gate_pass": gate_pass,
    "identity_source": identity,
    "n_validation_events": len(regions),
    "minimum_positive_events": required,
    "positive_events": positive,
    "event_macro_delta_iou": macro_delta,
    "event_macro_rer": macro_rer,
    "pooled_validation_delta_iou": row["delta_iou"],
    "pooled_validation_rer": row["rer"],
    "per_event": regions,
    "source_selection": str(source_path.resolve()),
    "audit_result": str(audit_path.resolve()),
}
output_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(",".join(map(str, selected)))
PY
  local config
  config="$(
    "$PYTHON" -c \
      'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["config"])))' \
      "$audit_dir/selection.json"
  )"
  if [[ ! -f "$test_dir/raw_result.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" \
      --terrain-h5 "$TERRAIN_H5" --split-csv "$SPLIT_CSV" \
      --fold "$fold" --seed "$SEED" --split test --routing-mode both \
      --fixed-config "$config" \
      --visual-checkpoint "$task/visual/checkpoint.pt" \
      --terrain-checkpoint "$task/terrain/terrain_expert.pt" \
      --outdir "$test_dir" --batch-size 32 --num-workers 2 --device cuda \
      2>&1 | tee "$test_dir/run.log"
    mv "$test_dir/result.json" "$test_dir/raw_result.json"
  fi
  "$PYTHON" - \
    "$test_dir/raw_result.json" "$audit_dir/selection.json" \
    "$test_dir/result.json" "$SEED" <<'PY'
import json
import pathlib
import sys

raw_path, selection_path, output_path = map(pathlib.Path, sys.argv[1:4])
seed = int(sys.argv[4])
raw = json.loads(raw_path.read_text())
selection = json.loads(selection_path.read_text())
row = raw["grid"][0]
observed = [
    row[key] for key in ("low_threshold", "high_threshold", "alpha", "visual_margin")
]
if observed != selection["config"]:
    raise RuntimeError(f"test config {observed} != validation selection {selection['config']}")
payload = {
    "status": "test_frozen_from_validation",
    "method": "event_consistent_bounded_terrain_gate",
    "fold": raw["fold"],
    "seed": seed,
    "regions": raw["regions"],
    "validation_selection": selection,
    "baseline": raw["baseline"],
    "adapted": row,
}
output_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

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
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" "$ANALYZER" --runs-dir "$PARENT" \
  --result-glob "seed*/fold*/${OUTPUT_TAG}_test/result.json" \
  --outdir "$PARENT/${OUTPUT_TAG}_analysis"
