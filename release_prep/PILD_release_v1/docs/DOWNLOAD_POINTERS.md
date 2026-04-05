# PILD Upstream Download Pointers

Purpose: point users to the official acquisition routes for upstream data that are **referenced by** `PILD` but **not fully mirrored in** `PILD_release_v1`

## 1. How to Use This File

`PILD_release_v1` is a public metadata and protocol release. It does not attempt to redistribute every upstream file used during internal experimentation.

Use this document to:

- identify the official source page or DOI for each upstream dataset
- understand whether the source is mirrored in the public `PILD` package
- determine whether a source should be fetched directly from its provider

For redistribution status, consult:

- `SOURCE_LICENSE_MATRIX.md`

## 2. Core Landslide Source Families

| Source family | Role in `PILD` | Official access point | What users should obtain from upstream | Included in `PILD_release_v1` |
|---|---|---|---|---|
| `DLR_Landslide_Ref_2025` | DLR frontier benchmark and `strict_t3` core | https://zenodo.org/records/17007637 | official event package and associated metadata from the DLR Zenodo record | no raw mirror; referenced via metadata only |
| `GLaD4CD` | large external cross-domain source | https://zenodo.org/records/10800338 | official dataset package from the Zenodo record | no raw mirror; referenced via metadata only |
| `CAS Landslide` | cross-domain source with limited event count in `strict_t2` | https://zenodo.org/records/10463130 ; https://www.nature.com/articles/s41597-023-02847-z | the official CAS release and article-linked materials | no raw mirror; referenced via metadata only |
| `GDCLD` | cross-domain source with mixed imagery provenance | https://essd.copernicus.org/articles/16/4817/2024/essd-16-4817-2024.html | official publication and any provider-approved access route described by the authors | no raw mirror; referenced via metadata only |
| `USGS_Inventory_v3` | auxiliary inventory source outside the main public benchmark | official USGS inventory access route used by the project | event inventory files only if needed for auxiliary analyses | not included in the main public release |

## 3. Static Environmental Layers

| Source | Role | Official access point | Suggested release handling |
|---|---|---|---|
| `ESA WorldCover` | land-cover prior | https://esa-worldcover.org/en/data-access | fetch from the official page; do not mirror in the main metadata package |
| `Copernicus DEM GLO-30` | elevation and terrain features | https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM | fetch from the official provider or documented mirrors; keep attribution and license notice |
| `SoilGrids` | soil texture and chemistry features | https://www.isric.org/explore/soilgrids/faq-soilgrids | fetch from ISRIC; pointer-only in the main package |
| `LiMW / GLiM / lithology layers` | lithology prior | source-specific geology providers used by the project | do not mirror unless each source license is individually confirmed |

## 4. Dynamic Environmental Layers

| Source | Role | Official access point | Suggested release handling |
|---|---|---|---|
| `SMAP SPL3SMP` | antecedent soil moisture | https://smap.jpl.nasa.gov/data/ | fetch from the official NASA/JPL channel |
| `ERA5-Land` | weather and hydrology support | https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5 ; https://forum.ecmwf.int/t/cc-by-licence-to-replace-licence-to-use-copernicus-products-on-02-july-2025/13464 | fetch from the official Copernicus/ECMWF route; derived aligned subsets may be released separately later |
| `CHIRPS` | rainfall-window support | https://www.chc.ucsb.edu/data/chirps | fetch from the official CHC page; pointer-only in the main package |

## 5. What `PILD_release_v1` Already Gives You

Even though the package does not mirror all upstream raw files, it already gives you:

- canonical event IDs
- benchmark inclusion flags
- split definitions
- source-family mapping
- protocol summaries
- global coverage visualization

This means users can still understand:

- which events belong to the public benchmark release
- how those events are distributed across source families
- which official upstream records need to be revisited for raw data access

## 6. Recommended Citation Practice

When a user reconstructs experiments from upstream material, they should cite:

1. the `PILD` release DOI
2. the associated manuscript
3. each upstream dataset actually used in reconstruction

## 7. Important Constraint

This file is intentionally conservative. If an upstream source has open access but mirroring is still unnecessary or operationally messy, the recommendation remains:

- fetch from the upstream source
- cite the upstream source
- keep the main `PILD` Zenodo record lightweight and low risk
