# Data and usage guide

This page covers:

1. How to use **our open PILD package**
2. Links to the **upstream open datasets** used to build PILD
3. How to run the main code entry points

Raw satellite imagery is not redistributed here. PILD provides curated metadata, splits, and protocol tables; imagery must be obtained from the original providers.

---

## 1. Our open dataset (PILD)

### 1.1 What PILD is

PILD (Physics-Informed Landslide Dataset) is the event-isolated, four-source corpus used in the GeoPhysAdapter paper:

- **7,890** samples
- **55** canonical events
- four public landslide sources

| `dataset_id` | Samples | Source-level events |
|---|---:|---:|
| `SEN12LS_HARMONIZED` | 4,979 | 15 |
| `GDCLD` | 2,334 | 4 |
| `DLR_Landslide_Ref_2025` | 509 | 22 |
| `GLaD4CD_v1` | 68 | 15 |

Source-level events sum to 56; one GDCLD–Sen12Landslides overlap is merged into a single canonical event (55).

### 1.2 Where to download PILD

| Channel | Link | Contents |
|---|---|---|
| This repository | https://github.com/Liu-Zhihang/geophysadapter | Code + metadata |
| Data archive | https://doi.org/10.5281/zenodo.19430714 | Same metadata family (manifests, splits, protocol notes) |

Key local paths after clone:

```text
metadata/pild_geo4_qc_v1/
  unified_sample_manifest_geo4_qc_v1.csv   # sample index
  event_isolated_split_geo4_qc_v1.csv      # train / val / test by fold
  leave_one_dataset_out_split_geo4_qc_v1.csv
  summary.json                             # retained counts / QC thresholds
metadata/pild_geo4_qc_native17_v1/         # protocol hashes
experiments/revision2026/pild_object_veto_final_v1/summary.json
```

### 1.3 How to use PILD metadata

**Step A — install**

```bash
git clone https://github.com/Liu-Zhihang/geophysadapter.git
cd geophysadapter
conda env create -f environment.yml
conda activate geophysadapter
```

**Step B — load the sample index and a split**

```python
import pandas as pd

manifest = pd.read_csv(
    "metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv"
)
split = pd.read_csv(
    "metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv"
)

print(manifest["dataset_id"].value_counts())
print(split["role"].value_counts())

df = manifest.merge(split, on="sample_id", how="inner")
train = df[df["role"] == "train"]
val = df[df["role"] == "val"]
test = df[df["role"] == "test"]
```

Important columns in the manifest:

| Column | Meaning |
|---|---|
| `sample_id` | Canonical sample ID used by our scripts |
| `source_sample_id` | ID in the upstream dataset (join key) |
| `canonical_event_id` | Event ID used for event-isolated splits |
| `dataset_id` | Upstream source family |
| `primary_qc_included` | Kept in the paper corpus |
| terrain / material / trigger readiness flags | Whether each prior is available for that sample |

Path fields use the placeholder `$PILD_ROOT/...`. After you build local caches, set:

```bash
export PILD_ROOT=/path/to/your/data/root
```

or rewrite those path prefixes to your machine.

**Step C — map PILD samples to upstream tiles**

1. Choose a `dataset_id` (e.g. `SEN12LS_HARMONIZED`).
2. Use `source_sample_id` / `source_event_id` to find the corresponding file in that upstream package.
3. Keep event isolation: never put samples from the same `canonical_event_id` into both train and test of the same fold.

**Step D — build local optical / terrain / material / trigger caches**

Use the helpers under `scripts/xdomain/` (see Section 3). The training script expects caches referenced by the manifest. Rebuild them on your machine from upstream imagery; they are not shipped in this repository.

**Step E — train / evaluate**

```bash
python scripts/xdomain/train_pild_sen12_roleaware_v1.py \
  --manifest metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv \
  --protocol-summary metadata/pild_geo4_qc_v1/summary.json \
  --split metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv \
  --variant full_tmr \
  --outdir experiments/local_run
```

Object-level review (after unit / hydrology feature tables exist):

```bash
python scripts/xdomain/evaluate_pild_object_veto_final_v1.py \
  --outdir experiments/local_object_veto
```

Reported object-level numbers from the paper are archived at:

```text
experiments/revision2026/pild_object_veto_final_v1/summary.json
```

---

## 2. Upstream open datasets used in the paper

Download these if you need the original imagery or labels behind PILD samples. Cite each source you use.

### 2.1 Sen12Landslides (harmonized)

- PILD id: `SEN12LS_HARMONIZED`
- Paper: Höhn et al., 2025, *Scientific Data* — https://doi.org/10.1038/s41597-025-06167-2
- Data: https://huggingface.co/datasets/paulhoehn/Sen12Landslides
- Code helpers: https://github.com/PaulH97/Sen12Landslides

```bash
pip install -U huggingface_hub
hf download paulhoehn/Sen12Landslides \
  --repo-type dataset \
  --local-dir ./data_raw/Sen12Landslides \
  --include "data_harmonized/**"
```

### 2.2 DLR Landslide Reference

- PILD id: `DLR_Landslide_Ref_2025`
- Paper: Orynbaikyzy / Martinis et al., 2025, *GIScience & Remote Sensing* — https://doi.org/10.1080/15481603.2025.2502214
- Data: https://doi.org/10.5281/zenodo.17007637

### 2.3 GDCLD

- PILD id: `GDCLD`
- Paper / access: Fang et al., 2024, *Earth System Science Data* — https://doi.org/10.5194/essd-16-4817-2024
- Follow the article’s data-availability statement. Do not rehost mixed-license commercial imagery.

### 2.4 GLaD4CD v1

- PILD id: `GLaD4CD_v1`
- Data: Leonardi et al., 2024 — https://doi.org/10.5281/zenodo.14226448

### 2.5 Geophysical products (not redistributed as raw stacks)

| Prior | Product | Link |
|---|---|---|
| Terrain | Copernicus DEM GLO-30 | https://dataspace.copernicus.eu/ |
| Material | SoilGrids | https://www.isric.org/explore/soilgrids |
| Trigger | CHIRPS | https://www.chc.ucsb.edu/data/chirps |
| Optional | ESA WorldCover | https://esa-worldcover.org/en/data-access |
| Optional | SMAP / ERA5-Land | https://smap.jpl.nasa.gov/data/ ; ECMWF ERA5-Land page |

### 2.6 CAS Landslide (not in the 7,890-sample corpus)

CAS Landslide (Xu et al., 2024; https://doi.org/10.1038/s41597-023-02847-z, https://doi.org/10.5281/zenodo.10463130) is registered in the broader PILD collection but is **not** part of the spatial corpus used here (missing auditable sample-level CRS / affine transforms).

---

## 3. Main code entry points

| Step | Script |
|---|---|
| Four-source training | `scripts/xdomain/train_pild_sen12_roleaware_v1.py` |
| Terrain encoding | `scripts/xdomain/sen12_terrain_v2.py` |
| Material module | `scripts/xdomain/pild_roleaware_material.py` |
| Trigger module | `scripts/xdomain/pild_roleaware_trigger.py` |
| Pixel utility gate | `scripts/xdomain/evaluate_pild_benefit_gate_v1.py` |
| Object-unit export | `scripts/xdomain/export_pild_subobject_units_v1.py` |
| Spectral feature export | `scripts/xdomain/export_pild_object_spectral_features_v1.py` |
| Hydrology feature export | `scripts/xdomain/export_pild_object_hydrology_features_v1.py` |
| Object-level review | `scripts/xdomain/evaluate_pild_object_veto_final_v1.py` |

Suggested order for a full local rebuild:

1. Download PILD metadata (this repo or the data archive DOI above).
2. Download upstream imagery for the four sources.
3. Build optical / terrain caches with the `sen12_*` / PILD builders under `scripts/xdomain/`.
4. Train with `train_pild_sen12_roleaware_v1.py`.
5. Export object units and features.
6. Run `evaluate_pild_object_veto_final_v1.py`.

Model checkpoints and large OOF tensors are not shipped. Rebuild them locally.

---

## 4. Citation

If you use this repository or PILD, please cite:

1. the GeoPhysAdapter paper
2. the PILD data archive: https://doi.org/10.5281/zenodo.19430714
3. every upstream dataset / product you download
