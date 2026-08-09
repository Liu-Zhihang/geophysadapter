#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
BASE="${BASE:-$ROOT/processed/hybrid_pinn/pild_prithvi_integration_v1}"
BUILDER="$ROOT/scripts/xdomain/build_pild_prithvi_temporal_cache_v1.py"
MERGER="$ROOT/scripts/xdomain/merge_pild_prithvi_temporal_shards_v1.py"
SHARD_DIR="${SHARD_DIR:-$BASE/shards_v1}"
N_SHARDS="${N_SHARDS:-4}"
READ_TIMEOUT_SECONDS="${READ_TIMEOUT_SECONDS:-90}"
ASSET_MIRROR_MANIFEST="${ASSET_MIRROR_MANIFEST:-$BASE/sentinel2_asset_mirror_v1/asset_mirror_manifest_v1.jsonl}"
ASSET_MIRROR_MARKER="${ASSET_MIRROR_MARKER:-${ASSET_MIRROR_MANIFEST%.jsonl}.complete.json}"

if [[ ! -f "$ASSET_MIRROR_MANIFEST" || ! -f "$ASSET_MIRROR_MARKER" ]]; then
  printf 'Local asset mirror is not complete: manifest=%s marker=%s\n' \
    "$ASSET_MIRROR_MANIFEST" "$ASSET_MIRROR_MARKER" >&2
  exit 2
fi

mkdir -p "$SHARD_DIR"
PLAN="$SHARD_DIR/shard_plan.tsv"

"$PYTHON" - "$BASE/acquisition_availability_v1.csv" "$N_SHARDS" "$PLAN" <<'PY'
import csv
import sys
from pathlib import Path

import pandas as pd

source, n_shards, destination = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
rows = pd.read_csv(source)
required = {"status", "prithvi_temporal_ready", "selection_uses_labels"}
missing = required - set(rows.columns)
if missing:
    raise RuntimeError(f"frozen acquisition registry lacks fields: {sorted(missing)}")
ready = (
    rows["status"].astype(str).eq("ready")
    & rows["prithvi_temporal_ready"].astype(int).eq(1)
    & rows["selection_uses_labels"].astype(int).eq(0)
)
if len(rows) != 49 or not ready.all():
    raise RuntimeError(
        "frozen acquisition registry must contain exactly 49 label-independent ready units"
    )

def weight(row):
    tiles = sum(int(value) for value in str(row.selected_tile_counts).split("|"))
    return int(row.n_windows) * max(4, tiles)

tasks = sorted(
    [(weight(row), str(row.acquisition_unit_id)) for row in rows.itertuples()],
    reverse=True,
)
loads = [0] * n_shards
assignments = [[] for _ in range(n_shards)]
for task_weight, unit_id in tasks:
    shard = min(range(n_shards), key=lambda index: (loads[index], index))
    assignments[shard].append((unit_id, task_weight))
    loads[shard] += task_weight

with destination.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(["shard", "acquisition_unit_id", "weight"])
    for shard, values in enumerate(assignments):
        for unit_id, task_weight in values:
            writer.writerow([shard, unit_id, task_weight])
print({"n_shards": n_shards, "loads": loads, "n_units": [len(x) for x in assignments]})
PY

pids=()
for ((shard=0; shard<N_SHARDS; shard++)); do
  out="$SHARD_DIR/shard${shard}.h5"
  marker="$out.complete.json"
  log="$SHARD_DIR/shard${shard}.log"
  if [[ -f "$out" && -f "$marker" ]]; then
    printf '[skip] completed shard=%s\n' "$shard"
    continue
  fi
  unit_args=()
  while IFS=$'\t' read -r _ unit_id _; do
    [[ "$unit_id" == "acquisition_unit_id" ]] && continue
    unit_args+=(--unit-id "$unit_id")
  done < <(awk -F '\t' -v shard="$shard" 'NR==1 || $1==shard' "$PLAN")
  resume_args=()
  [[ -f "$SHARD_DIR/.shard${shard}.h5.inprogress" ]] && resume_args+=(--resume)
  printf '[start] shard=%s units=%s time=%s\n' "$shard" "$((${#unit_args[@]}/2))" "$(date --iso-8601=seconds)"
  (
    env -u http_proxy -u https_proxy "$PYTHON" "$BUILDER" \
      "${unit_args[@]}" "${resume_args[@]}" --out "$out" \
      --asset-mirror-manifest "$ASSET_MIRROR_MANIFEST" \
      --read-timeout-seconds "$READ_TIMEOUT_SECONDS" --flush-every 8
  ) >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if ((status != 0)); then
  printf 'At least one shard failed; inspect %s/shard*.log and rerun to resume.\n' "$SHARD_DIR" >&2
  exit "$status"
fi

shards=()
for ((shard=0; shard<N_SHARDS; shard++)); do
  shards+=(--shard "$SHARD_DIR/shard${shard}.h5")
done
"$PYTHON" "$MERGER" "${shards[@]}" \
  --out "$BASE/pild_prithvi_4t6b_p128.h5"
printf '[complete] merged PILD Prithvi cache time=%s\n' "$(date --iso-8601=seconds)"
