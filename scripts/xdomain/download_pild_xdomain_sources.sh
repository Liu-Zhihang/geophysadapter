#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data_hdd/滑坡检测}"
PYTHON="${PYTHON:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
MAX_WORKERS="${MAX_WORKERS:-12}"

SEN12_REVISION="40af2dd6b4e568edb6640d6e14dc67ebd01038a4"
SEN12_CODE_REVISION="d26a25edc8e0b69550696cfb97bb5a983eaa2fde"
SEN12_DIR="${ROOT}/data_raw/08_Sen12Landslides"
NASA_DIR="${ROOT}/data_raw/09_NASA_COOLR_Rainfall_Events"
UGLC_DIR="${ROOT}/data_raw/10_UGLC"
META_DIR="${ROOT}/physics_informed_landslide_dataset/metadata/pild_xdomain_v1/acquisition"

mkdir -p "${SEN12_DIR}" "${NASA_DIR}" "${UGLC_DIR}" "${META_DIR}" "${HF_HOME}"

available_kib=$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')
required_kib=$((100 * 1024 * 1024))
if (( available_kib < required_kib )); then
  echo "[FATAL] Less than 100 GiB is available under ${ROOT}." >&2
  exit 2
fi

echo "[1/5] Downloading pinned Sen12Landslides harmonized archives."
HF_HOME="${HF_HOME}" "${PYTHON}" \
  "${ROOT}/physics_informed_landslide_dataset/scripts/xdomain/download_hf_dataset_files.py" \
  paulhoehn/Sen12Landslides \
  --revision "${SEN12_REVISION}" \
  --prefix data_harmonized \
  --prefix inventories.zip \
  --out-dir "${SEN12_DIR}" \
  --workers "${MAX_WORKERS}" \
  --manifest "${META_DIR}/sen12_harmonized_manifest.json"
curl -fL --retry 8 --retry-delay 5 \
  "https://huggingface.co/datasets/paulhoehn/Sen12Landslides/raw/${SEN12_REVISION}/README.md" \
  -o "${SEN12_DIR}/README.upstream.md"

echo "[2/5] Recording the pinned upstream Sen12Landslides code."
if [[ ! -d "${SEN12_DIR}/upstream_code/.git" ]]; then
  git clone --no-checkout https://github.com/PaulH97/Sen12Landslides.git "${SEN12_DIR}/upstream_code"
fi
git -C "${SEN12_DIR}/upstream_code" fetch --depth 1 origin "${SEN12_CODE_REVISION}"
git -C "${SEN12_DIR}/upstream_code" checkout --detach "${SEN12_CODE_REVISION}"

echo "[3/5] Downloading NASA COOLR rainfall-event polygons."
curl -fL --retry 8 --retry-delay 5 \
  https://api.figshare.com/v2/articles/26972467 \
  -o "${NASA_DIR}/figshare_record_26972467.json"
curl -fL -C - --retry 8 --retry-delay 5 \
  https://ndownloader.figshare.com/files/54363830 \
  -o "${NASA_DIR}/nasa_coolr_new_events.zip"
printf '%s  %s\n' \
  "412ccea62ca148299db1fec317719304" \
  "${NASA_DIR}/nasa_coolr_new_events.zip" | md5sum --check --status

echo "[4/5] Downloading UGLC global point and polygon catalogues."
curl -fL --retry 8 --retry-delay 5 \
  https://zenodo.org/api/records/18643456 \
  -o "${UGLC_DIR}/zenodo_record_18643456.json"
"${PYTHON}" - "${UGLC_DIR}/zenodo_record_18643456.json" "${UGLC_DIR}" <<'PY'
import json
import pathlib
import subprocess
import sys

record_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
record = json.loads(record_path.read_text(encoding="utf-8"))
wanted = {"UGLC_point.csv", "UGLC_poly.csv", "UGLC_tile_grid_map.jpeg"}
files = {item["key"]: item for item in record["files"]}
missing = wanted - files.keys()
if missing:
    raise SystemExit(f"Missing expected UGLC files in Zenodo record: {sorted(missing)}")
for name in sorted(wanted):
    item = files[name]
    target = out_dir / name
    subprocess.run(
        [
            "curl", "-fL", "-C", "-", "--retry", "8", "--retry-delay", "5",
            item["links"]["self"], "-o", str(target),
        ],
        check=True,
    )
    expected = item["checksum"].split(":", 1)[1]
    actual = subprocess.check_output(["md5sum", str(target)], text=True).split()[0]
    if actual != expected:
        raise SystemExit(f"Checksum mismatch for {target}: {actual} != {expected}")
PY

echo "[5/5] Writing acquisition receipt."
"${PYTHON}" - "${ROOT}" "${SEN12_REVISION}" "${SEN12_CODE_REVISION}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sen12_revision = sys.argv[2]
code_revision = sys.argv[3]
sources = {
    "sen12": root / "data_raw/08_Sen12Landslides",
    "nasa_coolr": root / "data_raw/09_NASA_COOLR_Rainfall_Events",
    "uglc": root / "data_raw/10_UGLC",
    "usgs": root / "data_raw/07_USGS_Inventory_v3",
    "hr_gldd": root / "data_raw/03_HR_GLDD",
}
payload = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sen12_dataset_revision": sen12_revision,
    "sen12_code_revision": code_revision,
    "sources": {},
}
for source_id, source_root in sources.items():
    files = [path for path in source_root.rglob("*") if path.is_file()]
    payload["sources"][source_id] = {
        "root": str(source_root),
        "n_files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }
target = root / "physics_informed_landslide_dataset/metadata/pild_xdomain_v1/acquisition/acquisition_receipt.json"
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

touch "${META_DIR}/DOWNLOAD_COMPLETE"
echo "[DONE] PILD-XDomain source acquisition is complete."
