#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_dlr_validation_calibrated.py"
PARENT="${PARENT:-$ROOT/experiments/revision2026/dlr_sen12_protocol_transfer_temporalvalid_v2}"
SEED="${SEED:-20260723}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"
CALIBRATION_TAG="${CALIBRATION_TAG:-dualthreshold_calibrated}"
MAX_ALPHA="${MAX_ALPHA:-inf}"
VISUAL_PARENT="${VISUAL_PARENT:-}"
SAMPLE_SUPPORT_DIR="${SAMPLE_SUPPORT_DIR:-}"
ROUTING_MODE="${ROUTING_MODE:-both}"

CACHE="${CACHE:-$ROOT/processed/hybrid_pinn/dlr_sen12_protocol_transfer_v1}"
BASE_H5="${BASE_H5:-$CACHE/dlr_base_temporalvalid_p128.h5}"
OPTICAL_H5="${OPTICAL_H5:-$CACHE/dlr_prithvi_4t6b_p128.h5}"
TERRAIN_H5="${TERRAIN_H5:-$CACHE/dlr_common_terrain9_p128.h5}"
SPLIT_CSV="${SPLIT_CSV:-$ROOT/metadata/protocol_assets/dlr_sen12_protocol_transfer_v1/dlr_eventisolated_nested5_v1.csv}"

read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"

run_fold() {
  local fold="$1" gpu="$2"
  local task="$PARENT/seed${SEED}/fold${fold}"
  local visual_dir="$task/visual"
  if [[ -n "$VISUAL_PARENT" ]]; then
    visual_dir="$VISUAL_PARENT/seed${SEED}/fold${fold}/visual"
  fi
  local support_args=()
  if [[ -n "$SAMPLE_SUPPORT_DIR" ]]; then
    local support_csv="$SAMPLE_SUPPORT_DIR/fold${fold}_sample_support.csv"
    if [[ ! -f "$support_csv" ]]; then
      echo "Missing sample support CSV: $support_csv" >&2
      return 2
    fi
    support_args=(--sample-support-csv "$support_csv")
  fi
  local valdir="$task/${CALIBRATION_TAG}_val"
  local testdir="$task/${CALIBRATION_TAG}_test"
  mkdir -p "$valdir" "$testdir"
  if [[ ! -f "$valdir/result.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" \
      --terrain-h5 "$TERRAIN_H5" --split-csv "$SPLIT_CSV" \
      --fold "$fold" --seed "$SEED" --split val --routing-mode "$ROUTING_MODE" --expanded-grid \
      --visual-checkpoint "$visual_dir/checkpoint.pt" \
      --terrain-checkpoint "$task/terrain/terrain_expert.pt" \
      "${support_args[@]}" \
      --outdir "$valdir" --batch-size 32 --num-workers 2 --device cuda \
      2>&1 | tee "$valdir/run.log"
  fi
  "$PYTHON" - "$valdir/result.json" "$valdir/selection.json" "$MAX_ALPHA" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:3])
max_alpha = float(sys.argv[3])
payload = json.loads(source.read_text())
candidates = [
    row for row in payload["grid"]
    if row["alpha"] <= max_alpha
    and row["delta_iou"] >= 0
    and row["rer"] >= 0
    and row["corrected"] >= row["harmed"]
]
if not candidates:
    raise RuntimeError("validation grid lacks identity/abstention")
selected = max(candidates, key=lambda row: (row["delta_iou"], row["rer"], row["corrected_to_harmed"]))
result = {
    "status":"frozen_from_validation",
    "config":[selected[key] for key in ("low_threshold","high_threshold","alpha","visual_margin")],
    "validation_delta_iou":selected["delta_iou"],
    "validation_rer":selected["rer"],
    "validation_corrected_to_harmed":selected["corrected_to_harmed"],
    "max_alpha":max_alpha,
    "source_result":str(source.resolve()),
}
output.write_text(json.dumps(result, indent=2) + "\n")
print(",".join(str(value) for value in result["config"]))
PY
  local config
  config="$($PYTHON -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["config"])))' "$valdir/selection.json")"
  if [[ ! -f "$testdir/raw_result.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" \
      --terrain-h5 "$TERRAIN_H5" --split-csv "$SPLIT_CSV" \
      --fold "$fold" --seed "$SEED" --split test --routing-mode "$ROUTING_MODE" \
      --fixed-config "$config" \
      --visual-checkpoint "$visual_dir/checkpoint.pt" \
      --terrain-checkpoint "$task/terrain/terrain_expert.pt" \
      "${support_args[@]}" \
      --outdir "$testdir" --batch-size 32 --num-workers 2 --device cuda \
      2>&1 | tee "$testdir/run.log"
    mv "$testdir/result.json" "$testdir/raw_result.json"
  fi
  "$PYTHON" - "$testdir/raw_result.json" "$valdir/selection.json" "$testdir/result.json" "$SEED" <<'PY'
import json, pathlib, sys
raw_path, selection_path, output_path = map(pathlib.Path, sys.argv[1:4])
seed = int(sys.argv[4])
raw = json.loads(raw_path.read_text()); selection = json.loads(selection_path.read_text())
row = raw["grid"][0]
observed = [row[key] for key in ("low_threshold","high_threshold","alpha","visual_margin")]
if observed != selection["config"]:
    raise RuntimeError(f"test config {observed} != validation selection {selection['config']}")
payload = {
    "status":"test_frozen_from_validation", "method":"dual_threshold_terrain_gate",
    "fold":raw["fold"], "seed":seed, "regions":raw["regions"],
    "validation_selection":selection, "baseline":raw["baseline"], "adapted":row,
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
for index in "${!gpu_array[@]}"; do worker "$index" "${gpu_array[$index]}" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" "$ANALYZER" --runs-dir "$PARENT" \
  --result-glob "seed*/fold*/${CALIBRATION_TAG}_test/result.json" \
  --outdir "$PARENT/${CALIBRATION_TAG}_analysis"
