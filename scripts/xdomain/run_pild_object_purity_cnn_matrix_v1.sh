#!/usr/bin/env bash
# G2 experiment matrix for the object-level purity model.
#
# Two queues, one per GPU. Each queue waits for any currently running matrix job on
# its own GPU so this script can be fired while the first two arms are still training.
#
#   GPU 0: appearance-only ablation, then the zero and donor Terrain controls
#   GPU 1: the shift32 and roll64 Terrain controls
#
# Every arm keeps the same crop size, schedule, seed and event-grouped protocol, so the
# only thing that differs between arms is which evidence the model is allowed to see.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
SCRIPT="$ROOT/scripts/xdomain/train_pild_object_purity_cnn_v1.py"
OUT="${OUT:-$ROOT/experiments/revision2026/pild_object_purity_cnn_v1}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-3}"

mkdir -p "$OUT"

wait_for_free_gpu() {
  # Compute-app occupancy is the only reliable signal here: the device index is not
  # visible in the child cmdline, and idle memory can sit just under any fixed limit.
  local gpu="$1" apps
  while true; do
    apps=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    [[ "${apps:-1}" -eq 0 ]] && break
    sleep 30
  done
  sleep 5
}

run_arm() {
  local gpu="$1" tag="$2" condition="$3"
  shift 3
  local groups=("$@")
  if [[ -s "$OUT/summary_${tag}.json" ]]; then
    echo "[skip] $tag already complete"
    return 0
  fi
  echo "[start] gpu=$gpu tag=$tag condition=$condition groups=${groups[*]} $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$SCRIPT" \
    --condition "$condition" \
    --groups "${groups[@]}" \
    --epochs "$EPOCHS" \
    --learning-rate "$LR" \
    --tag "$tag" \
    --outdir "$OUT" > "$OUT/${tag}.log" 2>&1
  echo "[done ] gpu=$gpu tag=$tag exit=$? $(date -Is)"
}

queue_zero() {
  wait_for_free_gpu 0
  run_arm 0 aligned_appearance_only aligned optical visual mask
  run_arm 0 control_zero zero terrain optical visual mask
  run_arm 0 control_donor donor terrain optical visual mask
}

queue_one() {
  wait_for_free_gpu 1
  run_arm 1 control_shift32 shift32 terrain optical visual mask
  run_arm 1 control_roll64 roll64 terrain optical visual mask
}

queue_zero &
queue_one &
wait
echo "[matrix] complete $(date -Is)"
