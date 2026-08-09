#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
VISUAL_TRAINER="$ROOT/scripts/xdomain/train_sen12_prithvi_terrain_v2.py"
TERRAIN_TRAINER="$ROOT/scripts/xdomain/train_sen12_terrain_expert_fusion.py"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_sen12_terrain_multiseed_confirm.py"
RUN_TAG="${RUN_TAG:-sen12_terrain_multiseed_confirm_v1}"
OUTBASE="${OUTBASE:-$ROOT/experiments/revision2026/$RUN_TAG}"
SEEDS="${SEEDS:-20260752 20260753 20260754 20260755}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TERRAIN_EPOCHS="${TERRAIN_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FIXED_CONFIG="${FIXED_CONFIG:-0.3,0.7,4.0,1.0}"
SEED51_DIR="${SEED51_DIR:-$ROOT/experiments/revision2026/sen12_terrain_dual_threshold_global_confirm_v1}"

mkdir -p "$OUTBASE"
printf '%s\n' "$0 $*" > "$OUTBASE/coordinator.command.txt"
printf '{"status":"running","seeds":"%s","folds":"%s","gpus":"%s","fixed_config":"%s"}\n' \
  "$SEEDS" "$FOLDS" "$GPUS" "$FIXED_CONFIG" > "$OUTBASE/run_manifest.json"

read -r -a seed_array <<< "$SEEDS"
read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"

tasks=()
for seed in "${seed_array[@]}"; do
  for fold in "${fold_array[@]}"; do
    tasks+=("$seed:$fold")
  done
done

pos_weight_for_fold() {
  case "$1" in
    0) printf '20.618921510855373\n' ;;
    1) printf '33.64611739222182\n' ;;
    2) printf '31.12955873951306\n' ;;
    3) printf '23.52003837218069\n' ;;
    4) printf '19.851538084376134\n' ;;
    *) return 1 ;;
  esac
}

run_task() {
  local seed="$1" fold="$2" gpu="$3"
  local task_dir="$OUTBASE/seed${seed}/fold${fold}"
  local visual_dir="$task_dir/visual"
  local terrain_dir="$task_dir/terrain"
  local gate_dir="$task_dir/gate_test"
  mkdir -p "$visual_dir" "$terrain_dir" "$gate_dir"

  if [[ ! -f "$visual_dir/DONE.json" || ! -f "$visual_dir/checkpoint.pt" || ! -f "$visual_dir/result.json" ]]; then
    printf '[start] seed=%s fold=%s stage=visual gpu=%s time=%s\n' "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VISUAL_TRAINER" \
      --mode visual --fold "$fold" --seed "$seed" --outdir "$visual_dir" \
      --epochs "$VISUAL_EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
      --max-train-samples 0 --max-eval-samples 0 --max-steps 0 \
      --evaluation-splits val,test --device cuda
  fi

  if [[ ! -f "$terrain_dir/terrain_expert.pt" || ! -f "$terrain_dir/result.json" ]]; then
    printf '[start] seed=%s fold=%s stage=terrain gpu=%s time=%s\n' "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TERRAIN_TRAINER" \
      --fold "$fold" --seed "$seed" --visual-checkpoint "$visual_dir/checkpoint.pt" \
      --outdir "$terrain_dir" --epochs "$TERRAIN_EPOCHS" --batch-size 64 \
      --num-workers "$NUM_WORKERS" --pos-weight "$(pos_weight_for_fold "$fold")" --device cuda \
      2>&1 | tee "$terrain_dir/run.log"
  fi

  if [[ ! -f "$gate_dir/result.json" ]]; then
    printf '[start] seed=%s fold=%s stage=gate_test gpu=%s time=%s\n' "$seed" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --fold "$fold" --seed "$seed" --split test --routing-mode both \
      --fixed-config "$FIXED_CONFIG" \
      --visual-checkpoint "$visual_dir/checkpoint.pt" \
      --terrain-checkpoint "$terrain_dir/terrain_expert.pt" \
      --outdir "$gate_dir" --batch-size 32 --num-workers "$NUM_WORKERS" --device cuda \
      2>&1 | tee "$gate_dir/run.log"
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

"$PYTHON" "$ANALYZER" \
  --runs-dir "$OUTBASE" --seed51-dir "$SEED51_DIR" --outdir "$OUTBASE/analysis"
printf '{"status":"complete","seeds":"%s","folds":"%s","fixed_config":"%s"}\n' \
  "$SEEDS" "$FOLDS" "$FIXED_CONFIG" > "$OUTBASE/run_manifest.json"
