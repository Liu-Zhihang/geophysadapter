# Data access and usage

This page explains how to obtain the open datasets used by GeoPhysAdapter / PILD, how to cite them, and how to connect them to the metadata in this repository.

## What this repository provides

| Asset | Location | Role |
|---|---|---|
| Sample manifest (7,890 samples) | `metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv` | Sample IDs, source IDs, event IDs, asset readiness flags |
| Event-isolated split | `metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv` | Train / validation / test roles by fold |
| Protocol summary | `metadata/pild_geo4_qc_v1/summary.json` | Retained counts and QC thresholds |
| Native protocol hashes | `metadata/pild_geo4_qc_native17_v1/` | Supplement S6 checksums |
| Object-level summary | `experiments/revision2026/pild_object_veto_final_v1/summary.json` | Reported object-scale metrics |
| Zenodo package | https://doi.org/10.5281/zenodo.19430714 | Curated metadata, split tables, download notes |

Raw satellite imagery is **not** redistributed here. Download it from the original providers below, then join it to the PILD manifests by `dataset_id`, `sample_id`, `source_sample_id`, and `canonical_event_id`.

Corpus size used in the paper: **7,890 samples / 55 canonical events**.

| `dataset_id` | Samples | Events (source-level) |
|---|---:|---:|
| `SEN12LS_HARMONIZED` | 4,979 | 15 |
| `GDCLD` | 2,334 | 4 |
| `DLR_Landslide_Ref_2025` | 509 | 22 |
| `GLaD4CD_v1` | 68 | 15 |

Source-level event counts sum to 56; one GDCLD–Sen12Landslides overlap is merged into a single canonical event, giving 55.

## 1. Upstream landslide datasets (required)

### 1.1 Sen12Landslides (harmonized)

- Role in PILD: primary multi-temporal optical source (`SEN12LS_HARMONIZED`)
- Paper: Höhn et al., 2025, *Scientific Data*  
  https://doi.org/10.1038/s41597-025-06167-2
- Dataset: https://huggingface.co/datasets/paulhoehn/Sen12Landslides
- Code helpers: https://github.com/PaulH97/Sen12Landslides
- Suggested download (harmonized only):

```bash
pip install -U huggingface_hub
hf download paulhoehn/Sen12Landslides \
  --repo-type dataset \
  --local-dir ./data_raw/Sen12Landslides \
  --include "data_harmonized/**"
```

### 1.2 DLR Landslide Reference

- Role in PILD: `DLR_Landslide_Ref_2025`
- Paper: Orynbaikyzy / Martinis et al., 2025, *GIScience & Remote Sensing*  
  https://doi.org/10.1080/15481603.2025.2502214
- Dataset (Zenodo): https://zenodo.org/records/17007637  
  DOI: https://doi.org/10.5281/zenodo.17007637

### 1.3 GDCLD

- Role in PILD: `GDCLD`
- Paper: Fang et al., 2024, *Earth System Science Data*  
  https://doi.org/10.5194/essd-16-4817-2024  
  https://essd.copernicus.org/articles/16/4817/2024/
- Access: follow the article’s data-availability statement. Imagery mixes multiple commercial / agency sources; do not rehost raw tiles through this project.

### 1.4 GLaD4CD v1

- Role in PILD: `GLaD4CD_v1`
- Dataset: Leonardi et al., 2024, Zenodo  
  https://doi.org/10.5281/zenodo.14226448
- Use the v1 package referenced by the paper.

### 1.5 CAS Landslide (registered, not in the geo4 spatial corpus)

- Paper: Xu et al., 2024, *Scientific Data*  
  https://doi.org/10.1038/s41597-023-02847-z
- Dataset: https://doi.org/10.5281/zenodo.10463130
- Note: kept in the broader PILD registry, but excluded from the 7,890-sample spatial attribution corpus because samples lack auditable CRS / affine transforms.

## 2. Geophysical layers used by GeoPhysAdapter

These are fetched from their providers and aligned to sample footprints. They are not shipped as raw rasters in this repository.

| Prior | Product | Official access | Typical use |
|---|---|---|---|
| Terrain (~30 m) | Copernicus DEM GLO-30 | https://dataspace.copernicus.eu/ | DEM / slope and dense spatial support |
| Material (~250 m) | SoilGrids | https://www.isric.org/explore/soilgrids | Soil texture / chemistry amplitude |
| Trigger (~5 km window) | CHIRPS daily rainfall | https://www.chc.ucsb.edu/data/chirps | Event rainfall dose |
| Optional land cover | ESA WorldCover | https://esa-worldcover.org/en/data-access | Auxiliary context |
| Optional moisture | SMAP / ERA5-Land | https://smap.jpl.nasa.gov/data/ ; https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5 | Hydrometeorological support |

## 3. How to use the data with this code

### Step A — Install

```bash
conda env create -f environment.yml
conda activate geophysadapter
```

### Step B — Get PILD metadata

Either clone this repository, or download the Zenodo package:

https://doi.org/10.5281/zenodo.19430714

Key file:

```text
metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv
```

Useful columns:

- `dataset_id`: one of the four sources above
- `sample_id` / `source_sample_id`: join keys to upstream tiles
- `canonical_event_id`: event used for event-isolated splits
- `primary_qc_included`: retained in the paper corpus
- terrain / material / trigger readiness flags

Path columns in the CSV are written as `$PILD_ROOT/...`. Set `PILD_ROOT` to your local data root after you rebuild caches.

### Step C — Download upstream imagery

1. Sen12Landslides harmonized (Hugging Face)
2. DLR Landslide Reference (Zenodo)
3. GDCLD (via Fang et al., 2024)
4. GLaD4CD v1 (Zenodo)

Keep each source in its own directory, for example:

```text
data_raw/
  Sen12Landslides/
  DLR_Landslide_Ref_2025/
  GDCLD/
  GLaD4CD_v1/
```

### Step D — Build local caches

Use the scripts under `scripts/xdomain/` to build optical / terrain / material / trigger caches that match the manifest. Entry points for the paper protocol:

| Task | Script |
|---|---|
| Four-source training | `scripts/xdomain/train_pild_sen12_roleaware_v1.py` |
| Terrain encoding | `scripts/xdomain/sen12_terrain_v2.py` |
| Material module | `scripts/xdomain/pild_roleaware_material.py` |
| Trigger module | `scripts/xdomain/pild_roleaware_trigger.py` |
| Pixel utility gate | `scripts/xdomain/evaluate_pild_benefit_gate_v1.py` |
| Object-level review | `scripts/xdomain/evaluate_pild_object_veto_final_v1.py` |

Example training call after caches exist:

```bash
python scripts/xdomain/train_pild_sen12_roleaware_v1.py \
  --manifest metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv \
  --protocol-summary metadata/pild_geo4_qc_v1/summary.json \
  --split metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv \
  --variant full_tmr \
  --outdir experiments/local_run
```

Exact flags depend on the variant you reproduce (`terrain`, material/trigger roles, or full object review). See Supplement S6 of the paper for the protocol mapping.

### Step E — Evaluate

Pixel-scale and object-scale evaluators write summary JSON/CSV under your `--outdir`. The paper’s object-level numbers are archived at:

```text
experiments/revision2026/pild_object_veto_final_v1/summary.json
```

## 4. Citation checklist

Please cite **all** of the following that you actually use:

1. The GeoPhysAdapter paper (this repository)
2. The PILD Zenodo record: https://doi.org/10.5281/zenodo.19430714
3. Upstream datasets used in your run:
   - Höhn et al., 2025 (Sen12Landslides)
   - Orynbaikyzy / Martinis et al., 2025 (DLR Landslide Reference)
   - Fang et al., 2024 (GDCLD)
   - Leonardi et al., 2024 (GLaD4CD)
4. Any geophysical product you download (Copernicus DEM, SoilGrids, CHIRPS, etc.)

## 5. License boundary

- Metadata, splits, and code in this repository follow the repository license.
- Raw imagery and environmental rasters remain under the license of each provider.
- GDCLD in particular should be obtained through the authors’ published access route; do not mirror mixed-license commercial imagery.
