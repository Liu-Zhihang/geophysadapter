#!/usr/bin/env bash
# G2b: crop encoder fused with whole-body summaries.
#
# Crop-only models generalised worse across events than the G1 scalar model, because a
# fixed window cannot cover a large body and raw radiometry carries source identity.
# Each arm here keeps both views and differs only in which evidence groups are visible,
# so the Terrain controls remain interpretable.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
SCRIPT="$ROOT/scripts/xdomain/train_pild_object_purity_cnn_v1.py"
OUT="${OUT:-$ROOT/experiments/revision2026/pild_object_purity_cnn_v2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-3}"
mkdir -p "$OUT"

gpu_is_free() {
  local count
  count=$(nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [[ "${count:-1}" -eq 0 ]]
}

wait_for_gpu() {
  while ! gpu_is_free "$1"; do sleep 30; done
  sleep 5
}

run_arm() {
  local gpu="$1" tag="$2" condition="$3"
  shift 3
  if [[ -s "$OUT/summary_${tag}.json" ]]; then
    echo "[skip ] $tag"
    return 0
  fi
  wait_for_gpu "$gpu"
  echo "[start] gpu=$gpu tag=$tag condition=$condition groups=$* $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$SCRIPT" \
    --condition "$condition" --groups "$@" \
    --epochs "$EPOCHS" --learning-rate "$LR" \
    --tag "$tag" --outdir "$OUT" > "$OUT/${tag}.log" 2>&1
  echo "[done ] gpu=$gpu tag=$tag exit=$? $(date -Is)"
}

queue_a() {
  run_arm 0 hybrid_physics aligned terrain visual mask
  run_arm 0 hybrid_appearance aligned optical visual mask
  run_arm 0 control_zero zero terrain optical visual mask
  run_arm 0 control_donor donor terrain optical visual mask
}

queue_b() {
  run_arm 1 hybrid_full aligned terrain optical visual mask
  run_arm 1 control_shift32 shift32 terrain optical visual mask
  run_arm 1 control_roll64 roll64 terrain optical visual mask
}

queue_a &
queue_b &
wait
echo "[matrix] complete $(date -Is)"
