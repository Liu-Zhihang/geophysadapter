#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/experiments/revision2026/pild_sen12_roleaware_lodo_v1}"
SPLIT="${SPLIT:-$ROOT/metadata/pild_sen12_training_v2/leave_one_dataset_out_split_v2.csv}"
OUTDIR="${OUTDIR:-$RUNS_ROOT/analysis_full_oof}"
ANALYZER="$ROOT/scripts/xdomain/analyze_pild_sen12_lodo_vt_v1.py"
FOLDS="${FOLDS:-lodo_00_DLR_Landslide_Ref_2025 lodo_01_GDCLD lodo_02_GLaD4CD_v1 lodo_03_SEN12LS_HARMONIZED}"
SEEDS="${SEEDS:-20260722 20260723 20260724 20260725 20260726}"
POLL_SECONDS="${POLL_SECONDS:-300}"

read -r -a fold_array <<<"$FOLDS"
read -r -a seed_array <<<"$SEEDS"

complete=0
expected=$((${#fold_array[@]} * ${#seed_array[@]} * 2))
while (( complete < expected )); do
  complete=0
  for fold in "${fold_array[@]}"; do
    for seed in "${seed_array[@]}"; do
      for variant in V VT; do
        run="$RUNS_ROOT/$fold/${variant}_seed${seed}"
        [[ -s "$run/DONE.json" && -s "$run/result.json" && -s "$run/checkpoint.pt" && -s "$run/per_sample_metrics.csv" ]] && ((complete+=1))
      done
    done
  done
  printf '[wait] complete=%s/%s time=%s\n' "$complete" "$expected" "$(date --iso-8601=seconds)"
  (( complete == expected )) || sleep "$POLL_SECONDS"
done

"$PYTHON" "$ANALYZER" \
  --runs-root "$RUNS_ROOT" \
  --split "$SPLIT" \
  --outdir "$OUTDIR" \
  --min-seeds 5
