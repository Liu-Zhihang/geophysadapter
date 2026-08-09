#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
TRAINER="$ROOT/scripts/xdomain/train_sen12_terrain_expert_fusion.py"
EVALUATOR="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"

BASE_H5="${BASE_H5:-$ROOT/processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5}"
OPTICAL_H5="${OPTICAL_H5:-$ROOT/processed/hybrid_pinn/sen12_s2_xdomain_v2/sen12_prithvi_4t6b_p128.h5}"
COMMON9_H5="${COMMON9_H5:-$BASE_H5}"
NATIVE17_RUN="${NATIVE17_RUN:-$ROOT/experiments/revision2026/sen12_terrain_multiseed_confirm_v1}"
OUTBASE="${OUTBASE:-$ROOT/experiments/revision2026/sen12_terrain_contract_common9_regression_v1}"
SEED="${SEED:-20260752}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"
EPOCHS="${EPOCHS:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FIXED_CONFIG="${FIXED_CONFIG:-0.3,0.7,4.0,1.0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"
if [[ ${#gpu_array[@]} -eq 0 ]]; then
  echo "GPUS must contain at least one device" >&2
  exit 2
fi

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

run_fold() {
  local fold="$1" gpu="$2"
  local visual_checkpoint="$NATIVE17_RUN/seed${SEED}/fold${fold}/visual/checkpoint.pt"
  local fold_dir="$OUTBASE/seed${SEED}/fold${fold}"
  local terrain_dir="$fold_dir/terrain_common9"
  local gate_dir="$fold_dir/gate_test"
  local train_cmd=(
    "$PYTHON" "$TRAINER"
    --base-h5 "$BASE_H5"
    --optical-h5 "$OPTICAL_H5"
    --terrain-h5 "$COMMON9_H5"
    --fold "$fold"
    --seed "$SEED"
    --visual-checkpoint "$visual_checkpoint"
    --outdir "$terrain_dir"
    --epochs "$EPOCHS"
    --batch-size 64
    --num-workers "$NUM_WORKERS"
    --pos-weight "$(pos_weight_for_fold "$fold")"
    --device cuda
  )
  local eval_cmd=(
    "$PYTHON" "$EVALUATOR"
    --base-h5 "$BASE_H5"
    --optical-h5 "$OPTICAL_H5"
    --terrain-h5 "$COMMON9_H5"
    --fold "$fold"
    --seed "$SEED"
    --split test
    --routing-mode both
    --fixed-config "$FIXED_CONFIG"
    --visual-checkpoint "$visual_checkpoint"
    --terrain-checkpoint "$terrain_dir/terrain_expert.pt"
    --outdir "$gate_dir"
    --batch-size 32
    --num-workers "$NUM_WORKERS"
    --device cuda
  )

  if [[ ! -f "$visual_checkpoint" ]]; then
    echo "missing frozen visual checkpoint: $visual_checkpoint" >&2
    return 3
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${train_cmd[@]}"
    printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${eval_cmd[@]}"
    printf '\n'
    return
  fi

  mkdir -p "$terrain_dir" "$gate_dir"
  if [[ ! -f "$terrain_dir/terrain_expert.pt" || ! -f "$terrain_dir/result.json" ]]; then
    printf '[start] fold=%s stage=terrain_common9 gpu=%s time=%s\n' \
      "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}" 2>&1 | tee "$terrain_dir/run.log"
  fi
  if [[ ! -f "$gate_dir/result.json" ]]; then
    printf '[start] fold=%s stage=gate_test gpu=%s time=%s\n' \
      "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "${eval_cmd[@]}" 2>&1 | tee "$gate_dir/run.log"
  fi
  printf '[done] fold=%s gpu=%s time=%s\n' \
    "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
}

if [[ "$DRY_RUN" == "1" ]]; then
  for index in "${!fold_array[@]}"; do
    run_fold "${fold_array[$index]}" "${gpu_array[$((index % ${#gpu_array[@]}))]}"
  done
  exit 0
fi

mkdir -p "$OUTBASE"
printf '%s\n' "$0 $*" > "$OUTBASE/coordinator.command.txt"
printf '{"status":"running","seed":%s,"folds":"%s","terrain_contract":"common9","fixed_config":"%s"}\n' \
  "$SEED" "$FOLDS" "$FIXED_CONFIG" > "$OUTBASE/run_manifest.json"

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

"$PYTHON" - "$OUTBASE" "$NATIVE17_RUN" "$SEED" <<'PY'
import json
import math
import sys
from pathlib import Path

outbase = Path(sys.argv[1])
native_run = Path(sys.argv[2])
seed = int(sys.argv[3])

def read_row(path):
    payload = json.loads(path.read_text())
    row = payload["grid"][0]
    baseline = payload["baseline"]
    return {
        "visual_iou": float(baseline["iou"]),
        "adapted_iou": float(row["iou"]),
        "delta_iou": float(row["delta_iou"]),
        "rer": float(row["rer"]),
        "corrected": int(row["corrected"]),
        "harmed": int(row["harmed"]),
        "baseline_errors": int(baseline["errors"]),
    }

rows = []
for fold in range(5):
    common = read_row(outbase / f"seed{seed}" / f"fold{fold}" / "gate_test/result.json")
    native = read_row(native_run / f"seed{seed}" / f"fold{fold}" / "gate_test/result.json")
    rows.append({"fold": fold, "common9": common, "native17": native})

def pooled(contract):
    corrected = sum(row[contract]["corrected"] for row in rows)
    harmed = sum(row[contract]["harmed"] for row in rows)
    baseline_errors = sum(row[contract]["baseline_errors"] for row in rows)
    return {
        "mean_delta_iou": sum(row[contract]["delta_iou"] for row in rows) / len(rows),
        "mean_rer": sum(row[contract]["rer"] for row in rows) / len(rows),
        "pooled_rer": (corrected - harmed) / max(baseline_errors, 1),
        "corrected": corrected,
        "harmed": harmed,
        "corrected_harmed_ratio": corrected / max(harmed, 1),
        "estimated_baseline_errors": baseline_errors,
    }

summary = {
    "status": "complete",
    "question": "Does reducing Sen12 native17 Terrain to common9 erase the fixed-rule gain?",
    "seed": seed,
    "fixed_config": [0.3, 0.7, 4.0, 1.0],
    "rows": rows,
    "pooled": {"common9": pooled("common9"), "native17": pooled("native17")},
}
(outbase / "contract_regression_summary.json").write_text(
    json.dumps(summary, indent=2, allow_nan=False) + "\n"
)
print(json.dumps(summary["pooled"], indent=2))
PY

printf '{"status":"complete","seed":%s,"folds":"%s","terrain_contract":"common9","fixed_config":"%s"}\n' \
  "$SEED" "$FOLDS" "$FIXED_CONFIG" > "$OUTBASE/run_manifest.json"
