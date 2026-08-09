#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/mnt/data_hdd/滑坡检测/physics_informed_landslide_dataset}"
MARKER="${MARKER:-$PROJECT/metadata/pild_xdomain_v1/acquisition/sen12_pipeline.complete}"
LOG="${LOG:-$PROJECT/metadata/pild_xdomain_v1/acquisition/sen12_tmr_smoke_watch.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$(dirname "$LOG")"
echo "[WAIT] $(date -Is) waiting for $MARKER" >>"$LOG"
while [[ ! -f "$MARKER" ]]; do
  sleep "$POLL_SECONDS"
done

echo "[START] $(date -Is) Terrain formal queue complete" >>"$LOG"
cd "$PROJECT"
SMOKE_OUTBASE="${OUTBASE:-$PROJECT/experiments/revision2026/sen12_xdomain_tmr_smoke_v1}"
OUTBASE="$SMOKE_OUTBASE" \
FOLDS="${FOLDS:-0 1}" \
SEED="${SEED:-20260721}" \
MODES="${MODES:-terrain_material terrain_trigger full_tmr}" \
GPUS="${GPUS:-0 1}" \
EPOCHS="${EPOCHS:-10}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
MAX_STEPS="${MAX_STEPS:-0}" \
bash scripts/xdomain/run_sen12_xdomain_tmr_matrix.sh >>"$LOG" 2>&1
echo "[ANALYZE] $(date -Is) TMR smoke training complete" >>"$LOG"
/home/jinlin/miniconda3/envs/dpl/bin/python scripts/xdomain/analyze_sen12_xdomain_tmr.py \
  --runs-dir "$SMOKE_OUTBASE" \
  --terrain-runs-dir "$PROJECT/experiments/revision2026/sen12_xdomain_geophysadapter_v1" \
  --outdir "$SMOKE_OUTBASE/analysis_partial" \
  --allow-partial \
  --min-folds 1 \
  --min-seeds 1 >>"$LOG" 2>&1
echo "[DONE] $(date -Is) TMR smoke and DEVELOPMENT_PARTIAL analysis complete" >>"$LOG"
