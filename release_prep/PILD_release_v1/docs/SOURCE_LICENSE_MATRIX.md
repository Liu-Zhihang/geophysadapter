# PILD Source License Matrix

Scope: operational release guidance for the public `PILD` Zenodo package

## 1. Purpose

`PILD` aggregates event metadata, benchmark protocols, and experiment-facing pointers from multiple upstream datasets and environmental products. Those upstream sources do **not** share a single redistribution policy.

This matrix is therefore used to answer two release questions:

1. Can the upstream content be publicly mirrored by us?
2. Should the upstream content enter the main public Zenodo package?

This document is an operational release note, not legal advice. When in doubt, the public release should fall back to metadata, source pointers, checksums, and fetch scripts rather than redistributing raw files.

## 2. Status Labels

- `green`: redistribution appears explicitly permitted by the upstream source; inclusion is legally plausible if attribution is preserved
- `yellow`: access/use appears open or partly open, but mirroring is unnecessary, size-heavy, license-mixed, or still needs one more source-specific check; keep as pointer-only in the main package
- `red`: do not redistribute in the public `PILD` package; provide citations, official links, and acquisition notes only

## 3. Release Matrix

| Source family | Role in `PILD` | Upstream license / access note | Status | Recommended action in `PILD_release_v1` | Primary source |
|---|---|---|---|---|---|
| `PILD` event metadata, splits, manifests, docs, figures, scripts | benchmark definition, release metadata | project-authored release assets | `green` | include directly in the main Zenodo package | internal release assets |
| `DLR Landslide Reference 2025` | core DLR benchmark source | Zenodo record indicates `CC BY 4.0` | `green` | do not mirror in the main metadata package; cite and optionally place into a second open-assets companion record if size is acceptable | https://zenodo.org/records/17007637 |
| `GLaD4CD` | external cross-domain source | Zenodo record indicates `CC BY 4.0` | `green` | same handling as `DLR`: cite directly; optional companion release only if exact upstream package identity is preserved | https://zenodo.org/records/10800338 |
| `CAS Landslide` official package | external cross-domain source | official Zenodo package is distributed under `CC BY-NC 4.0`; raw imagery provenance in the paper still depends on upstream providers | `yellow` | cite official package and paper; do not re-mirror raw files in the main `PILD` release | https://zenodo.org/records/10463130 ; https://www.nature.com/articles/s41597-023-02847-z |
| `GDCLD` | external cross-domain source | paper describes imagery from `Map World`, `Gaofen-6`, `UAV`, and `Planet` sources with mixed access conditions | `red` | do not redistribute raw imagery through `PILD`; provide source citation, acquisition note, and event mapping only | https://essd.copernicus.org/articles/16/4817/2024/essd-16-4817-2024.html |
| `ESA WorldCover` | land-cover prior | official access page states `CC BY 4.0` | `green` | prefer pointer-only or fetch-script handling in the main package because of size; okay for a separate open companion | https://esa-worldcover.org/en/data-access |
| `Copernicus DEM GLO-30` | DEM / terrain features | license allows broad use and redistribution with attribution and notice preservation | `yellow` | keep pointer-only in the main package; add fetch guidance rather than mirroring tiles | https://docs.sentinel-hub.com/api/latest/static/files/data/dem/resources/license/License-COPDEM-30.pdf |
| `SoilGrids` | soil features | official FAQ states `CC BY 4.0` | `green` | pointer-only in the main package; companion release or scripted fetch is acceptable | https://www.isric.org/explore/soilgrids/faq-soilgrids |
| `SMAP SPL3SMP` | antecedent soil moisture | NASA Earth science data are distributed under open science / open access policy | `yellow` | pointer-only in the main package; use official source links and fetch notes | https://smap.jpl.nasa.gov/data/ ; https://science.nasa.gov/researchers/science-data/science-information-policy/ |
| `ERA5-Land` | weather / hydrology support | ECMWF announced `CC BY` for Copernicus products from 2025-07-02 | `green` | pointer-only or scripted fetch in the main package; companion release possible for derived aligned subsets | https://forum.ecmwf.int/t/cc-by-licence-to-replace-licence-to-use-copernicus-products-on-02-july-2025/13464 |
| `CHIRPS` | rainfall-window support | publicly accessible climate product, but redistribution terms should be preserved from the official provider | `yellow` | keep pointer-only in the main package unless a source-specific redistribution statement is archived in the release docs | https://www.chc.ucsb.edu/data/chirps |
| `LiMW / GLiM / lithology layers` | lithology prior | license provenance should be checked source-by-source before redistribution | `yellow` | pointer-only in the main package; do not mirror until each source notice is captured | source-specific documentation required |
| local cache tensors, processed training shards, teacher assets, local mirrors | internal experiment acceleration | contain local paths, mixed third-party content, and non-release artifacts | `red` | exclude from all public Zenodo uploads | internal only |

## 4. What Goes Into the Main Zenodo Package

The main public Zenodo package should contain only:

- event-level tables
- benchmark splits
- protocol manifests
- data card
- data dictionary
- source registry
- release notes
- global coverage figure
- release scripts that do not embed local paths

The following should stay out of the main package:

- raw imagery mirrors
- large third-party raster collections
- cache tensors and processed training shards
- teacher assets or tensors that embed mixed-license data

## 5. Practical Rule

If a source is not clearly `green`, the default `PILD` action is:

1. cite the upstream paper or DOI
2. link to the official acquisition page
3. describe how the source was filtered or aligned in `PILD`
4. avoid redistributing the raw files

## 6. Recommended Two-Record Strategy

### Record A: main public DOI

`PILD metadata and protocol release`

Contents:
- metadata
- splits
- manifests
- docs
- figures
- lightweight scripts

### Record B: optional companion DOI

`PILD open-assets companion`

Contents:
- only those upstream assets whose redistribution is clearly permitted
- or release-time fetch manifests and checksums for open sources

This keeps the manuscript-facing dataset release clean, auditable, and low risk.
