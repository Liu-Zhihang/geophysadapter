# DATA_SOURCE_REGISTRY

## 1. Purpose

This file answers two questions:

1. Which open or publicly documented source families are currently integrated into this release package.
2. Whether each source is presently stored locally, included in the strict release, used for current training, or reserved for later physics-guided extensions.

## 2. Landslide Remote-Sensing Sources (`raw/datasets`)

| local_id | source_dataset | local_path | data_type | current_status | included_in `PILD_v1_strict_t2` | current_training_role |
|---|---|---|---|---|---|---|
| `01_GDCLD` | GDCLD | `raw/datasets/01_GDCLD` | image + mask | downloaded | yes (8 events) | reserved for multi-dataset expansion |
| `02_CAS_Landslide` | CAS Landslide | `raw/datasets/02_CAS_Landslide` | image / mask / label + AOI | unpacked and cleaned | yes (6 events) | reserved for multi-dataset expansion |
| `03_HR_GLDD` | HR-GLDD | `raw/datasets/03_HR_GLDD` | numpy tensors | downloaded | no | held as an external evaluation candidate |
| `04_Landslide4Sense` | Landslide4Sense | `raw/datasets/04_Landslide4Sense` | h5 / image + annotation | downloaded | no | held as an external evaluation candidate |
| `05_DLR_Landslide_Ref_2025` | DLR Landslide Reference 2025 | `raw/datasets/05_DLR_Landslide_Ref_2025` | GeoTIFF + gpkg + `reference_data/*.h5` | unpacked | yes (25 events) | primary source for the current DLR line |
| `06_GLaD4CD` | GLaD4CD v1/v2 | `raw/datasets/06_GLaD4CD` | pre/post GeoTIFF | expanded | yes (v1, 157 events) | currently used for external zero-shot validation |
| `07_USGS_Inventory_v3` | USGS Inventory v3 | `raw/datasets/07_USGS_Inventory_v3` | csv / gpkg / shp | downloaded | no | reserved for inventory and AOI enrichment |

## 3. Static Geophysical Layers (`raw/static`)

| source | local_path | role | current_status | currently_used_for_training |
|---|---|---|---|---|
| Copernicus DEM GLO-30 | `raw/static/copdem_glo30_2021` | `DEM / slope / curvature / TWI` | downloaded | currently only `DEM / slope` are used indirectly through DLR `reference_data`; the remaining products are reserved |
| ESA WorldCover 2021 | `raw/static/worldcover_v200_2021` | land-cover prior | downloaded | not used in current training |
| SoilGrids | `raw/static/soilgrids` | soil texture / chemistry / organic matter | downloaded (`clay / sand / silt / cec / soc` available) | not used in current training |
| LiMW / GLiM | `raw/static/lithology` | lithology / geology prior | downloaded and converted to `gpkg` | not used in current training |

## 4. Dynamic Weather and Moisture Layers (`raw/weather`)

| source | local_path | role | current_status | currently_used_for_training |
|---|---|---|---|---|
| CHIRPS daily | `raw/weather/chirps_daily_global` | event-window rainfall | downloaded | not used in current training |
| SMAP SPL3SMP | `raw/weather/smap_spl3smp` | antecedent soil moisture | downloaded | not used in current training |
| ERA5-Land | `raw/weather/era5_land/DLR_Landslide_Ref_2025` | DLR-focused hydro-temperature variables | downloaded for the DLR subset | reserved for the higher-fidelity `strict_t3` setting |

## 5. Current Strict-Release Coverage

The current `PILD_v1_strict_t2` release includes the following event sources:

1. `GLaD4CD_v1`: 157 events
2. `DLR_Landslide_Ref_2025`: 25 events
3. `GDCLD`: 8 events
4. `CAS_Landslide`: 6 events

The current `strict_t3` subset includes:

1. `DLR_Landslide_Ref_2025`: 25 events

See also:

1. `DATA_CARD.md`
2. `PROTOCOLS.md`

## 6. What Is Actually Used in Current Training

As of 2026-03-07, the current Stage-1 training scripts use only:

1. pre-event and post-event `Sentinel-2 B02 / B03 / B04 / B08`
2. `DEM`
3. `SLOPE`
4. an `NDVI drop` constraint computed from pre/post spectra

In other words:

- most downloaded geophysical variables are **not yet used directly by the current training backbone**
- they are already part of the `PILD` data system, but several are reserved for later physics-guided extensions

## 7. Recommended Companion Files

1. Package overview: `../README.md`
2. Data card: `DATA_CARD.md`
3. Data dictionary: `DATA_DICTIONARY.md`
4. Protocol summary: `PROTOCOLS.md`
5. Licensing boundary: `SOURCE_LICENSE_MATRIX.md`
