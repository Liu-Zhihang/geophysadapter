#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${WORKSPACE_ROOT:-$(pwd)}}"
PYTHON="${PYTHON:-python}"
META="$ROOT/metadata/pild_xdomain_v1"
MANIFEST="$META/acquisition/sen12_s2_manifest.json"
COMPLETE="${MANIFEST%.json}.complete"
LOG="$META/acquisition/sen12_finalize.log"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] waiting for $COMPLETE"
while [[ ! -f "$COMPLETE" ]]; do
  if ! tmux has-session -t pild-xdomain-download-s2 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] S2 downloader exited without completion marker" >&2
    exit 1
  fi
  sleep 60
done

echo "[$(date --iso-8601=seconds)] download complete; auditing local assets"
cd "$ROOT"
"$PYTHON" scripts/xdomain/audit_pild_xdomain_local_assets.py

echo "[$(date --iso-8601=seconds)] extracting and checksum-verifying S2 archives"
"$PYTHON" scripts/xdomain/extract_sen12_s2.py --workers 8

echo "[$(date --iso-8601=seconds)] indexing S2 samples and physical-event groups"
"$PYTHON" scripts/xdomain/index_sen12_s2.py --workers 24

echo "[$(date --iso-8601=seconds)] refreshing cross-source event registry"
"$PYTHON" scripts/xdomain/build_pild_xdomain_event_registry.py

echo "[$(date --iso-8601=seconds)] building the audited 4,979-sample change-view S2/Terrain cache"
"$PYTHON" scripts/xdomain/build_sen12_s2_tmr_cache.py

echo "[$(date --iso-8601=seconds)] running the real-data CUDA visual/adapter smoke"
OUTBASE="$ROOT/experiments/revision2026/sen12_xdomain_geophysadapter_smoke_v1" \
FOLDS="0" SEEDS="20260721" GPUS="0" EPOCHS="2" MAX_STEPS="20" \
NUM_WORKERS="2" \
bash scripts/xdomain/run_sen12_xdomain_geophysadapter.sh

echo "[$(date --iso-8601=seconds)] launching the formal five-fold five-seed dual-GPU matrix"
OUTBASE="$ROOT/experiments/revision2026/sen12_xdomain_geophysadapter_v1" \
FOLDS="0 1 2 3 4" SEEDS="20260721 20260722 20260723 20260724 20260725" \
GPUS="0 1" EPOCHS="60" MAX_STEPS="0" NUM_WORKERS="4" \
bash scripts/xdomain/run_sen12_xdomain_geophysadapter.sh

echo "[$(date --iso-8601=seconds)] running the strict paired LOGO-5 analysis"
"$PYTHON" scripts/xdomain/analyze_sen12_xdomain_geophysadapter.py \
  --runs-dir "$ROOT/experiments/revision2026/sen12_xdomain_geophysadapter_v1"

touch "$META/acquisition/sen12_pipeline.complete"
echo "[$(date --iso-8601=seconds)] Sen12 acquisition through formal training matrix complete"
