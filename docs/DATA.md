# Data guide

This page has two parts:

1. **How to use our open PILD package** (metadata, splits, protocol tables)
2. **Where the paper’s upstream open datasets come from** (download links and citations)

Raw satellite imagery is not redistributed in this repository or on Zenodo. PILD tells you *which* samples and events enter the paper experiments; imagery is obtained from the original providers.

---

## Part 1 — Our open dataset (PILD)

### What you get

| Item | Where | What it is |
|---|---|---|
| Zenodo package | https://doi.org/10.5281/zenodo.19430714 | Same metadata family as this repo |
| Unified sample manifest | `metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv` | 7,890 samples with IDs and readiness flags |
| Event-isolated split | `metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv` | Fold roles for train / val / test |
| LODO split | `metadata/pild_geo4_qc_v1/leave_one_dataset_out_split_geo4_qc_v1.csv` | Leave-one-dataset-out roles |
| QC summary | `metadata/pild_geo4_qc_v1/summary.json` | Retained counts and thresholds |
| Protocol hashes | `metadata/pild_geo4_qc_native17_v1/` | Checksums cited in Supplement S6 |
| Object-level metrics | `experiments/revision2026/pild_object_veto_final_v1/summary.json` | Numbers reported for object-scale review |

Paper corpus: **7,890 samples**, **55 canonical events**.

| `dataset_id` | Samples | Source-level events |
|---|---:|---:|
| `SEN12LS_HARMONIZED` | 4,979 | 15 |
| `GDCLD` | 2,334 | 4 |
| `DLR_Landslide_Ref_2025` | 509 | 22 |
| `GLaD4CD_v1` | 68 | 15 |

Source-level events sum to 56; one GDCLD–Sen12Landslides overlap is merged into one canonical event (55).

### How to use PILD

**Step 1 — Get the package**

```bash
git clone https://github.com/Liu-Zhihang/geophysadapter.git
cd geophysadapter
# or download the Zenodo zip: https://doi.org/10.5281/zenodo.19430714
```

**Step 2 — Read the manifest**

```python
import pandas as pd

manifest = pd.read_csv(
    "metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv"
)
split = pd.read_csv(
    "metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv"
)

print(manifest["dataset_id"].value_counts())
print(manifest[["sample_id", "canonical_event_id", "dataset_id"]].head())
```

Important columns:

- `sample_id` / `source_sample_id`: join keys to upstream tiles
- `canonical_event_id`: event identity used for event-isolated splits
- `dataset_id`: upstream source family
- `primary_qc_included`: kept in the paper corpus
- terrain / material / trigger readiness flags

Path fields in the CSV use the placeholder `$PILD_ROOT/...`. After you build local caches, replace that prefix with your data root (or set `PILD_ROOT` in the environment).

**Step 3 — Join a split**

```python
df = manifest.merge(split, on="sample_id", how="inner")
train = df[df["role"] == "train"]  # column name follows the split file
```

**Step 4 — Run the paper scripts**

Install once:

```bash
conda env create -f environment.yml
conda activate geophysadapter
```

Then point the training / evaluation entry points at the PILD files, for example:

```bash
python scripts/xdomain/train_pild_sen12_roleaware_v1.py \
  --manifest metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv \
  --protocol-summary metadata/pild_geo4_qc_v1/summary.json \
  --split metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv \
  --variant full_tmr \
  --outdir experiments/local_run
```

Object-level review:

```bash
python scripts/xdomain/evaluate_pild_object_veto_final_v1.py --help
```

Full entry-point list: README (Supplement S6 table).

---

## Part 2 — Upstream open datasets used in the paper

Download these if you need the original imagery / labels behind PILD samples. Cite each source you use.

### Sen12Landslides (harmonized)

- PILD id: `SEN12LS_HARMONIZED`
- Paper: Höhn et al., 2025, *Scientific Data* — https://doi.org/10.1038/s41597-025-06167-2
- Data: https://huggingface.co/datasets/paulhoehn/Sen12Landslides
- Helpers: https://github.com/PaulH97/Sen12Landslides

```bash
pip install -U huggingface_hub
hf download paulhoehn/Sen12Landslides \
  --repo-type dataset \
  --local-dir ./data_raw/Sen12Landslides \
  --include "data_harmonized/**"
```

### DLR Landslide Reference

- PILD id: `DLR_Landslide_Ref_2025`
- Paper: Orynbaikyzy / Martinis et al., 2025, *GIScience & Remote Sensing* — https://doi.org/10.1080/15481603.2025.2502214
- Data: https://doi.org/10.5281/zenodo.17007637

### GDCLD

- PILD id: `GDCLD`
- Paper / data access: Fang et al., 2024, *Earth System Science Data* — https://doi.org/10.5194/essd-16-4817-2024
- Follow the article’s data-availability statement. Do not rehost mixed-license commercial imagery.

### GLaD4CD v1

- PILD id: `GLaD4CD_v1`
- Data: Leonardi et al., 2024 — https://doi.org/10.5281/zenodo.14226448

### Geophysical products (aligned by us, not redistributed as raw stacks)

| Prior | Product | Link |
|---|---|---|
| Terrain | Copernicus DEM GLO-30 | https://dataspace.copernicus.eu/ |
| Material | SoilGrids | https://www.isric.org/explore/soilgrids |
| Trigger | CHIRPS | https://www.chc.ucsb.edu/data/chirps |
| Optional | ESA WorldCover | https://esa-worldcover.org/en/data-access |
| Optional | SMAP / ERA5-Land | https://smap.jpl.nasa.gov/data/ ; ECMWF ERA5-Land page |

### Note on CAS Landslide

CAS Landslide (Xu et al., 2024; https://doi.org/10.1038/s41597-023-02847-z, https://doi.org/10.5281/zenodo.10463130) is registered in the broader PILD collection but is **not** part of the 7,890-sample spatial corpus in this paper (missing auditable sample-level CRS / affine transforms).

---

## Citation

If you use PILD or this code, cite:

1. the GeoPhysAdapter paper
2. the PILD Zenodo DOI: https://doi.org/10.5281/zenodo.19430714
3. every upstream dataset / product you download
