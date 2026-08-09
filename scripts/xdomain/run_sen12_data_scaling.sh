#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
VISUAL="$ROOT/scripts/xdomain/train_sen12_prithvi_terrain_v2.py"
TERRAIN="$ROOT/scripts/xdomain/train_sen12_terrain_expert_fusion.py"
GATE="$ROOT/scripts/xdomain/evaluate_sen12_terrain_dual_threshold.py"
ANALYZER="$ROOT/scripts/xdomain/analyze_sen12_data_scaling.py"
OUTBASE="${OUTBASE:-$ROOT/experiments/revision2026/sen12_data_scaling_v1}"
SEED="${SEED:-20260751}"
FRACTIONS="${FRACTIONS:-0.25 0.50 0.75}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"
FIXED_CONFIG="${FIXED_CONFIG:-0.3,0.7,4.0,1.0}"
FULL_REFERENCE="${FULL_REFERENCE:-$ROOT/experiments/revision2026/sen12_terrain_dual_threshold_global_confirm_v1}"

mkdir -p "$OUTBASE"
printf '{"status":"running","seed":%s,"fractions":"%s","folds":"%s"}\n' "$SEED" "$FRACTIONS" "$FOLDS" > "$OUTBASE/run_manifest.json"
read -r -a fraction_array <<< "$FRACTIONS"
read -r -a fold_array <<< "$FOLDS"
read -r -a gpu_array <<< "$GPUS"
tasks=()
for fraction in "${fraction_array[@]}"; do
  for fold in "${fold_array[@]}"; do tasks+=("$fraction:$fold"); done
done

run_task() {
  local fraction="$1" fold="$2" gpu="$3" tag task_dir visual_dir terrain_dir gate_dir
  tag="fraction$(awk -v f="$fraction" 'BEGIN{printf "%03d",int(f*100+0.5)}')"
  task_dir="$OUTBASE/$tag/seed${SEED}/fold${fold}"
  visual_dir="$task_dir/visual"; terrain_dir="$task_dir/terrain"; gate_dir="$task_dir/gate_test"
  mkdir -p "$visual_dir" "$terrain_dir" "$gate_dir"
  if [[ ! -f "$visual_dir/DONE.json" || ! -f "$visual_dir/checkpoint.pt" ]]; then
    printf '[start] fraction=%s fold=%s stage=visual gpu=%s time=%s\n' "$fraction" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$VISUAL" --mode visual --fold "$fold" --seed "$SEED" \
      --outdir "$visual_dir" --train-fraction "$fraction" --epochs 30 --batch-size 32 --num-workers 4 \
      --evaluation-splits val,test --device cuda
  fi
  if [[ ! -f "$terrain_dir/terrain_expert.pt" || ! -f "$terrain_dir/result.json" ]]; then
    printf '[start] fraction=%s fold=%s stage=terrain gpu=%s time=%s\n' "$fraction" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TERRAIN" --fold "$fold" --seed "$SEED" \
      --visual-checkpoint "$visual_dir/checkpoint.pt" --outdir "$terrain_dir" --train-fraction "$fraction" \
      --epochs 12 --batch-size 64 --num-workers 4 --device cuda 2>&1 | tee "$terrain_dir/run.log"
  fi
  if [[ ! -f "$gate_dir/result.json" ]]; then
    printf '[start] fraction=%s fold=%s stage=gate gpu=%s time=%s\n' "$fraction" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$GATE" --fold "$fold" --seed "$SEED" --split test \
      --routing-mode both --fixed-config "$FIXED_CONFIG" --visual-checkpoint "$visual_dir/checkpoint.pt" \
      --terrain-checkpoint "$terrain_dir/terrain_expert.pt" --outdir "$gate_dir" --batch-size 32 --num-workers 4 --device cuda \
      2>&1 | tee "$gate_dir/run.log"
  fi
  "$PYTHON" - "$gate_dir/result.json" "$fraction" "$fold" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); fraction=float(sys.argv[2]); fold=int(sys.argv[3]); x=json.loads(p.read_text())
assert x['status']=='confirmatory_fixed_configuration' and x['fold']==fold and x['split']=='test'
r=x['grid'][0]; assert (r['low_threshold'],r['high_threshold'],r['alpha'],r['visual_margin'])==(0.3,0.7,4.0,1.0)
(p.parent/'DONE.json').write_text(json.dumps({'status':'complete','fraction':fraction,'fold':fold},indent=2)+'\n')
PY
  printf '[done] fraction=%s fold=%s gpu=%s time=%s\n' "$fraction" "$fold" "$gpu" "$(date --iso-8601=seconds)" | tee -a "$OUTBASE/progress.log"
}

worker() {
  local wi="$1" gpu="$2" i task fraction fold
  for ((i=wi;i<${#tasks[@]};i+=${#gpu_array[@]})); do
    task="${tasks[$i]}"; fraction="${task%%:*}"; fold="${task##*:}"
    run_task "$fraction" "$fold" "$gpu"
  done
}
pids=(); for i in "${!gpu_array[@]}"; do worker "$i" "${gpu_array[$i]}" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PYTHON" "$ANALYZER" --runs-dir "$OUTBASE" --full-reference-dir "$FULL_REFERENCE" --seed "$SEED" --outdir "$OUTBASE/analysis"
printf '{"status":"complete","seed":%s,"fractions":"%s","folds":"%s"}\n' "$SEED" "$FRACTIONS" "$FOLDS" > "$OUTBASE/run_manifest.json"
