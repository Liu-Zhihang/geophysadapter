#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
BUILDER="$ROOT/scripts/xdomain/build_dlr_sen12_protocol_transfer.py"
VISUAL_TRAINER="$ROOT/scripts/xdomain/train_sen12_prithvi_terrain_v2.py"
TERRAIN_TRAINER="$ROOT/scripts/xdomain/train_sen12_terrain_expert_fusion.py"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_dlr_sen12_protocol_transfer.py"

CACHE_DIR="${CACHE_DIR:-$ROOT/processed/hybrid_pinn/dlr_sen12_protocol_transfer_v1}"
METADATA_DIR="${METADATA_DIR:-$ROOT/metadata/protocol_assets/dlr_sen12_protocol_transfer_v1}"
MANIFEST_CSV="${MANIFEST_CSV:-$ROOT/metadata/pild_sen12_training_v2/unified_sample_manifest_v2.csv}"
BASE_H5="${BASE_H5:-$CACHE_DIR/dlr_base_temporalvalid_p128.h5}"
OPTICAL_H5="${OPTICAL_H5:-$CACHE_DIR/dlr_prithvi_4t6b_p128.h5}"
TERRAIN_H5="${TERRAIN_H5:-$CACHE_DIR/dlr_common_terrain9_p128.h5}"
SPLIT_CSV="${SPLIT_CSV:-$METADATA_DIR/dlr_eventisolated_nested5_v1.csv}"

RUN_TAG="${RUN_TAG:-dlr_sen12_protocol_transfer_temporalvalid_v2}"
OUTBASE="${OUTBASE:-$ROOT/experiments/revision2026/$RUN_TAG}"
SEEDS="${SEEDS:-20260723}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TERRAIN_EPOCHS="${TERRAIN_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FIXED_CONFIG="${FIXED_CONFIG:-0.3,0.7,4.0,1.0}"
VISUAL_PARENT="${VISUAL_PARENT:-}"
TERRAIN_INIT_ROOT="${TERRAIN_INIT_ROOT:-}"
TERRAIN_INIT_SEED="${TERRAIN_INIT_SEED:-20260752}"
SAMPLE_SUPPORT_DIR="${SAMPLE_SUPPORT_DIR:-}"

if [[ ! -f "$BASE_H5" || ! -f "$OPTICAL_H5" || ! -f "$TERRAIN_H5" || ! -f "$SPLIT_CSV" ]]; then
  "$PYTHON" "$BUILDER" \
    --manifest-csv "$MANIFEST_CSV" \
    --outdir "$CACHE_DIR" \
    --metadata-dir "$METADATA_DIR"
fi

mkdir -p "$OUTBASE"
printf '%q ' "$0" "$@" > "$OUTBASE/coordinator.command.txt"
printf '\n' >> "$OUTBASE/coordinator.command.txt"
printf '{"status":"running","seeds":"%s","folds":"%s","gpus":"%s","fixed_config":"%s","manifest_csv":"%s","cache_dir":"%s","split_csv":"%s","sample_support_dir":"%s"}\n' \
  "$SEEDS" "$FOLDS" "$GPUS" "$FIXED_CONFIG" "$MANIFEST_CSV" "$CACHE_DIR" "$SPLIT_CSV" "$SAMPLE_SUPPORT_DIR" \
  > "$OUTBASE/run_manifest.json"

read -r -a seed_array <<< "$SEEDS"
read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"
tasks=()
for seed in "${seed_array[@]}"; do
  for fold in "${fold_array[@]}"; do
    tasks+=("$seed:$fold")
  done
done

common_paths=(
  --base-h5 "$BASE_H5"
  --optical-h5 "$OPTICAL_H5"
  --terrain-h5 "$TERRAIN_H5"
  --split-csv "$SPLIT_CSV"
)

run_task() {
  local seed="$1" fold="$2" gpu="$3"
  local task_dir="$OUTBASE/seed${seed}/fold${fold}"
  local visual_dir="$task_dir/visual"
  if [[ -n "$VISUAL_PARENT" ]]; then
    visual_dir="$VISUAL_PARENT/seed${seed}/fold${fold}/visual"
  fi
  local terrain_dir="$task_dir/terrain"
  local gate_dir="$task_dir/gate_test"
  local support_args=()
  if [[ -n "$SAMPLE_SUPPORT_DIR" ]]; then
    local support_csv="$SAMPLE_SUPPORT_DIR/fold${fold}_sample_support.csv"
    if [[ ! -f "$support_csv" ]]; then
      echo "Missing sample support CSV: $support_csv" >&2
      return 2
    fi
    support_args=(--sample-support-csv "$support_csv")
  fi
  mkdir -p "$visual_dir" "$terrain_dir" "$gate_dir"

  if [[ ! -f "$visual_dir/DONE.json" || ! -f "$visual_dir/checkpoint.pt" || ! -f "$visual_dir/result.json" ]]; then
    printf '[start] seed=%s fold=%s stage=visual gpu=%s time=%s\n' "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VISUAL_TRAINER" \
      "${common_paths[@]}" --mode visual --fold "$fold" --seed "$seed" \
      --outdir "$visual_dir" --epochs "$VISUAL_EPOCHS" --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" --max-train-samples 0 --max-eval-samples 0 \
      --max-steps 0 --evaluation-splits val,test --device cuda
  fi

  if [[ ! -f "$terrain_dir/terrain_expert.pt" || ! -f "$terrain_dir/result.json" ]]; then
    local init_args=()
    if [[ -n "$TERRAIN_INIT_ROOT" ]]; then
      local init_checkpoint="$TERRAIN_INIT_ROOT/seed${TERRAIN_INIT_SEED}/fold${fold}/terrain/terrain_expert.pt"
      if [[ ! -f "$init_checkpoint" ]]; then
        echo "Missing Terrain initialization checkpoint: $init_checkpoint" >&2
        return 2
      fi
      init_args=(--init-terrain-checkpoint "$init_checkpoint")
    fi
    printf '[start] seed=%s fold=%s stage=terrain gpu=%s time=%s\n' "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TERRAIN_TRAINER" \
      "${common_paths[@]}" --fold "$fold" --seed "$seed" \
      --visual-checkpoint "$visual_dir/checkpoint.pt" --outdir "$terrain_dir" \
      "${support_args[@]}" \
      "${init_args[@]}" \
      --epochs "$TERRAIN_EPOCHS" --batch-size 64 --num-workers "$NUM_WORKERS" \
      --device cuda 2>&1 | tee "$terrain_dir/run.log"
  fi

  if [[ ! -f "$gate_dir/result.json" ]]; then
    printf '[start] seed=%s fold=%s stage=gate_test gpu=%s time=%s\n' "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      "${common_paths[@]}" --fold "$fold" --seed "$seed" --split test \
      --routing-mode both --fixed-config "$FIXED_CONFIG" \
      --visual-checkpoint "$visual_dir/checkpoint.pt" \
      --terrain-checkpoint "$terrain_dir/terrain_expert.pt" \
      "${support_args[@]}" \
      --outdir "$gate_dir" --batch-size 32 --num-workers "$NUM_WORKERS" \
      --device cuda 2>&1 | tee "$gate_dir/run.log"
  fi
  "$PYTHON" - "$gate_dir/result.json" "$seed" "$fold" <<'PY'
import json, pathlib, sys
path, seed, fold = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
payload = json.loads(path.read_text())
assert payload["status"] == "confirmatory_fixed_configuration"
assert payload["split"] == "test" and payload["fold"] == fold
row = payload["grid"][0]
assert (row["low_threshold"], row["high_threshold"], row["alpha"], row["visual_margin"]) == (0.3, 0.7, 4.0, 1.0)
(path.parent / "DONE.json").write_text(json.dumps({"status":"complete","seed":seed,"fold":fold}, indent=2) + "\n")
PY
  printf '[done] seed=%s fold=%s gpu=%s time=%s\n' "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
}

worker() {
  local worker_index="$1" gpu="$2" index task seed fold
  for ((index=worker_index; index<${#tasks[@]}; index+=${#gpu_array[@]})); do
    task="${tasks[$index]}"
    seed="${task%%:*}"
    fold="${task##*:}"
    run_task "$seed" "$fold" "$gpu"
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

"$PYTHON" "$ANALYZER" --runs-dir "$OUTBASE" --outdir "$OUTBASE/analysis"
printf '{"status":"complete","seeds":"%s","folds":"%s","fixed_config":"%s"}\n' \
  "$SEEDS" "$FOLDS" "$FIXED_CONFIG" > "$OUTBASE/run_manifest.json"
