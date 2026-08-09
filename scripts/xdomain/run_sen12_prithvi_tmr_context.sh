#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTROOT="${OUTROOT:-$PROJECT/experiments/revision2026/sen12_prithvi_tmr_context_v1}"
VISROOT="${VISROOT:-$PROJECT/experiments/revision2026/sen12_prithvi_terrain_v2_formal_v1}"
TERROOT="${TERROOT:-$PROJECT/experiments/revision2026/sen12_terrain_expert_fusion_v1}"
SCALING_PATTERN="${SCALING_PATTERN:-run_sen12_data_scaling.sh}"
SEED="${SEED:-20260771}"
EPOCHS="${EPOCHS:-30}"
GPU_LIST="${GPU_LIST:-0,1}"
MODES="${MODES:-material,trigger,joint}"

mkdir -p "$OUTROOT"

if [[ "${WAIT_FOR_SCALING:-1}" == "1" ]]; then
  while pgrep -f "$SCALING_PATTERN" >/dev/null; do
    printf '[wait] data-scaling still owns GPUs at %s\n' "$(date --iso-8601=seconds)"
    sleep 60
  done
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
IFS=',' read -r -a MODE_ARRAY <<< "$MODES"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local visual="$VISROOT/fold${fold}_seed20260751/visual/checkpoint.pt"
  local terrain="$TERROOT/fold${fold}_seed20260751/terrain_expert.pt"
  local cache="$OUTROOT/cache/fold${fold}"
  [[ -s "$visual" ]] || { echo "missing visual checkpoint: $visual" >&2; return 1; }
  [[ -s "$terrain" ]] || { echo "missing terrain checkpoint: $terrain" >&2; return 1; }
  mkdir -p "$cache"
  for mode in "${MODE_ARRAY[@]}"; do
    local out="$OUTROOT/fold${fold}/$mode"
    mkdir -p "$out"
    if [[ -s "$out/DONE.json" && -s "$out/result.json" && -s "$out/modulator.pt" ]]; then
      echo "[skip] fold=$fold mode=$mode"
      continue
    fi
    local -a command=(
      "$PYTHON" "$SCRIPT_DIR/train_sen12_prithvi_tmr_modulator.py"
      --fold "$fold"
      --mode "$mode"
      --seed "$SEED"
      --visual-checkpoint "$visual"
      --terrain-checkpoint "$terrain"
      --outdir "$out"
      --cache-dir "$cache"
      --epochs "$EPOCHS"
      --batch-size 64
      --num-workers 4
      --device cuda
    )
    printf '%q ' "${command[@]}" > "$out/command.txt"
    printf '\n' >> "$out/command.txt"
    echo "[start] fold=$fold mode=$mode gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" 2>&1 | tee "$out/run.log"
    echo "[done] fold=$fold mode=$mode gpu=$gpu time=$(date --iso-8601=seconds)"
  done
}

worker() {
  local worker_index="$1"
  local gpu="${GPUS[$worker_index]}"
  local fold
  for fold in 0 1 2 3 4; do
    if (( fold % ${#GPUS[@]} == worker_index )); then
      run_fold "$fold" "$gpu"
    fi
  done
}

pids=()
for worker_index in "${!GPUS[@]}"; do
  worker "$worker_index" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status == 0 )); then
  "$PYTHON" "$SCRIPT_DIR/analyze_sen12_prithvi_tmr_context.py" \
    --runs-root "$OUTROOT" --seed "$SEED" --outdir "$OUTROOT/analysis"
fi
exit "$status"
