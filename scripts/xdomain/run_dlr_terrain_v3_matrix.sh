#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAINER="$ROOT/scripts/xdomain/train_sen12_terrain_expert_fusion.py"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_dlr_terrain_additive_calibrated.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_dlr_validation_calibrated.py"

VARIANT="${VARIANT:-copdem_native17}"
CACHE_DIR="${CACHE_DIR:-$ROOT/processed/hybrid_pinn/dlr_sen12_protocol_transfer_v3}"
TERRAIN_H5="${TERRAIN_H5:-$CACHE_DIR/dlr_copdem_native17_p128.h5}"
BASE_H5="${BASE_H5:-$CACHE_DIR/dlr_base_temporalvalid_p128.h5}"
OPTICAL_H5="${OPTICAL_H5:-$CACHE_DIR/dlr_prithvi_4t6b_p128.h5}"
SPLIT_CSV="${SPLIT_CSV:-$ROOT/metadata/protocol_assets/dlr_sen12_protocol_transfer_v3/dlr_eventisolated_nested5_v1.csv}"
VISUAL_ROOT="${VISUAL_ROOT:-$ROOT/experiments/revision2026/dlr_sen12_protocol_transfer_temporalvalid_v2}"
OUTBASE="${OUTBASE:-$ROOT/experiments/revision2026/dlr_terrain_v3_${VARIANT}_exploratory_v1}"

SEED="${SEED:-20260723}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"
TERRAIN_EPOCHS="${TERRAIN_EPOCHS:-20}"
POOL_FACTORS="${POOL_FACTORS:-1,2,4,8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

for path in "$BASE_H5" "$OPTICAL_H5" "$TERRAIN_H5" "$SPLIT_CSV"; do
  [[ -f "$path" ]] || { echo "[fatal] missing input: $path" >&2; exit 2; }
done

mkdir -p "$OUTBASE"
printf '%q ' "$0" "$@" > "$OUTBASE/coordinator.command.txt"
printf '\n' >> "$OUTBASE/coordinator.command.txt"
cat > "$OUTBASE/run_manifest.json" <<EOF
{
  "status": "running",
  "scientific_status": "exploratory",
  "variant": "$VARIANT",
  "seed": $SEED,
  "folds": "$FOLDS",
  "terrain_h5": "$TERRAIN_H5",
  "visual_root": "$VISUAL_ROOT",
  "terrain_epochs": $TERRAIN_EPOCHS,
  "direction_pool_factors": "$POOL_FACTORS"
}
EOF

read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"

run_fold() {
  local fold="$1" gpu="$2"
  local visual_checkpoint="$VISUAL_ROOT/seed${SEED}/fold${fold}/visual/checkpoint.pt"
  local fold_dir="$OUTBASE/seed${SEED}/fold${fold}"
  local terrain_dir="$fold_dir/terrain"
  local test_dir="$fold_dir/additive_calibrated_test"
  [[ -f "$visual_checkpoint" ]] || {
    echo "[fatal] missing frozen visual checkpoint: $visual_checkpoint" >&2
    return 2
  }
  mkdir -p "$terrain_dir" "$test_dir"
  if [[ ! -f "$terrain_dir/terrain_expert.pt" || ! -f "$terrain_dir/result.json" ]]; then
    printf '[start] fold=%s stage=terrain variant=%s gpu=%s time=%s\n' \
      "$fold" "$VARIANT" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAINER" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" \
      --terrain-h5 "$TERRAIN_H5" --split-csv "$SPLIT_CSV" \
      --fold "$fold" --seed "$SEED" --visual-checkpoint "$visual_checkpoint" \
      --outdir "$terrain_dir" --epochs "$TERRAIN_EPOCHS" \
      --direction-pool-factors "$POOL_FACTORS" \
      --batch-size 64 --num-workers "$NUM_WORKERS" --device cuda \
      2>&1 | tee "$terrain_dir/run.log"
  fi
  if [[ ! -f "$test_dir/result.json" ]]; then
    printf '[start] fold=%s stage=test variant=%s gpu=%s time=%s\n' \
      "$fold" "$VARIANT" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --base-h5 "$BASE_H5" --optical-h5 "$OPTICAL_H5" \
      --terrain-h5 "$TERRAIN_H5" --split-csv "$SPLIT_CSV" \
      --fold "$fold" --seed "$SEED" --visual-checkpoint "$visual_checkpoint" \
      --terrain-checkpoint "$terrain_dir/terrain_expert.pt" \
      --outdir "$test_dir" --batch-size 32 --num-workers "$NUM_WORKERS" \
      --device cuda 2>&1 | tee "$test_dir/run.log"
  fi
  "$PYTHON" - "$test_dir/result.json" "$SEED" "$fold" "$VARIANT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
assert payload["status"] == "test_frozen_from_validation"
assert int(payload["seed"]) == int(sys.argv[2])
assert int(payload["fold"]) == int(sys.argv[3])
(path.parent / "DONE.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "seed": int(sys.argv[2]),
            "fold": int(sys.argv[3]),
            "variant": sys.argv[4],
        },
        indent=2,
    )
    + "\n"
)
PY
  printf '[done] fold=%s variant=%s gpu=%s time=%s\n' \
    "$fold" "$VARIANT" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
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

"$PYTHON" "$ANALYZER" \
  --runs-dir "$OUTBASE" \
  --result-glob 'seed*/fold*/additive_calibrated_test/result.json' \
  --outdir "$OUTBASE/analysis"

cat > "$OUTBASE/run_manifest.json" <<EOF
{
  "status": "complete",
  "scientific_status": "exploratory",
  "variant": "$VARIANT",
  "seed": $SEED,
  "folds": "$FOLDS",
  "terrain_h5": "$TERRAIN_H5",
  "visual_root": "$VISUAL_ROOT",
  "terrain_epochs": $TERRAIN_EPOCHS,
  "direction_pool_factors": "$POOL_FACTORS"
}
EOF
