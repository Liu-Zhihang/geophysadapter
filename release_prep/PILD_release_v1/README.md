# Physics-Informed Landslide Dataset (PILD) Release README

## 1. Overview

The Physics-Informed Landslide Dataset (`PILD`) is a multi-source event-centered dataset system for cross-domain landslide mapping from remote sensing. The project combines event metadata, split manifests, protocol definitions, and physics-support availability summaries for a heterogeneous benchmark whose main public subset is assembled from four source families:

- `GLaD4CD_v1`
- `DLR_Landslide_Ref_2025`
- `GDCLD`
- `CAS_Landslide`

The broader `event_master.csv` index also contains a small auxiliary `USGS_Inventory_v3` component outside the main `strict_t2` release.

The current public release is designed to support the associated manuscript on cross-domain landslide mapping with geophysical-context adaptation:

`GeoPhysAdapter: Physically Structured Adaptation of Foundation Models for Cross-Domain Landslide Mapping`

This public release is **not** a raw data mirror. It is a curated, public-facing release that exposes the event index, benchmark splits, protocol manifests, and reproducibility metadata required to understand, audit, and partially reconstruct the reported experiments without redistributing third-party raw imagery or derived products whose licenses remain controlled by upstream providers.

## 2. What This Release Contains

The release is centered on three layers of metadata.

### 2.1 Event-level master index

`event_master.csv` records the canonical event inventory used by the project, including:
- `event_uid`
- source dataset identifier
- event date and trigger type
- event-level point or bounding-box geolocation
- availability flags for static and weather support
- notes on event provenance

### 2.2 Benchmark protocol manifests

The release provides protocol-facing manifests for:
- `strict_t2`: heterogeneous multi-source benchmark with static + weather + SMAP support
- `strict_t3`: DLR-centered high-fidelity subset with richer physical support
- `in_domain`
- `LODO` (leave-one-dataset-out)
- `cross_trigger`

### 2.3 Release documentation

The release includes:
- a data card
- a data dictionary
- a source registry
- a source license matrix
- upstream download pointers
- split files
- release notes
- figure assets that summarize geographic coverage

## 3. What This Release Does Not Contain

This release does **not** redistribute the full raw imagery, label rasters, or all third-party physical layers used during experimentation.

Excluded or pointer-only content includes:
- original multi-source optical imagery
- original label masks where redistribution is not explicitly permitted
- third-party topography, land-cover, weather, soil-moisture, soil, and lithology source files
- local raw mirrors used during internal experimentation
- large cache-backed tensors containing local absolute paths

The public package therefore emphasizes **traceability and reproducibility of the benchmark definition**, not unrestricted redistribution of all upstream files.

## 4. Dataset Statistics

At the current project stage:

- `event_master.csv` contains `224` indexed events
- `strict_t2` contains `196` events
- `strict_t3` contains `25` events

### event_master source composition

- `GLaD4CD_v1`: `174`
- `DLR_Landslide_Ref_2025`: `28`
- `CAS_Landslide`: `11`
- `GDCLD`: `9`
- `USGS_Inventory_v3`: `2`

### strict_t2 source composition

- `GLaD4CD_v1`: `157`
- `DLR_Landslide_Ref_2025`: `25`
- `GDCLD`: `8`
- `CAS_Landslide`: `6`

### strict_t3 source composition

- `DLR_Landslide_Ref_2025`: `25`

## 5. Geographic Coverage

The benchmark is globally distributed rather than centered on a single country or single provider. Geographic coordinates are stored at the event level in `event_master.csv`:

- if an event-level point is available, the release uses `lat/lon`
- otherwise the release uses the center of the recorded event bounding box (`min_lon`, `min_lat`, `max_lon`, `max_lat`)

The release includes a public global coverage map:

- `figures/pild_global_event_coverage_t2.png`

This figure is intended to communicate the benchmark's cross-domain scope before model-level results are discussed.

## 6. Directory Layout

Current release layout:

```text
PILD_release_v1/
  README.md
  LICENSE.md
  RELEASE_NOTES.md
  CITATION.cff
  data/
    event_master.csv
    event_index_v1_strict_t2.csv
    split_in_domain.csv
    split_lodo.csv
    split_cross_trigger.csv
    strict_t2_supervised_ready_event_summary_v1.csv
    dlr_strict_t3_hybrid_events_v1.csv
  docs/
    DATA_CARD.md
    DATA_DICTIONARY.md
    DATA_SOURCE_REGISTRY.md
    PROTOCOLS.md
    SOURCE_LICENSE_MATRIX.md
    DOWNLOAD_POINTERS.md
  figures/
    pild_global_event_coverage_t2.png
  scripts/
    render_pild_global_event_map.py
```

## 7. Key Files

### Event and split metadata in this package

- `data/event_master.csv`
- `data/event_index_v1_strict_t2.csv`
- `data/strict_t2_supervised_ready_event_summary_v1.csv`
- `data/dlr_strict_t3_hybrid_events_v1.csv`
- `data/split_in_domain.csv`
- `data/split_lodo.csv`
- `data/split_cross_trigger.csv`

### Documentation in this package

- `docs/DATA_CARD.md`
- `docs/DATA_DICTIONARY.md`
- `docs/DATA_SOURCE_REGISTRY.md`
- `docs/PROTOCOLS.md`
- `docs/SOURCE_LICENSE_MATRIX.md`
- `docs/DOWNLOAD_POINTERS.md`

### Additional release assets in this package

- `figures/pild_global_event_coverage_t2.png`
- `scripts/render_pild_global_event_map.py`
- `LICENSE.md`
- `RELEASE_NOTES.md`
- `CITATION.cff`

## 8. Source Provenance and Licensing

`PILD` aggregates metadata derived from multiple upstream datasets and environmental data products. Licensing, redistribution rights, and attribution requirements are therefore source dependent.

At a minimum, the release should clearly acknowledge the following source families:

- `DLR_Landslide_Ref_2025`
- `GLaD4CD_v1`
- `GDCLD`
- `CAS_Landslide`
- `CopDEM GLO-30`
- `ESA WorldCover`
- `CHIRPS`
- `ERA5-Land`
- `SMAP SPL3SMP`
- `SoilGrids`
- `LiMW / GLiM`

Important release rule:

The public package should redistribute only those files for which redistribution is allowed. For all other assets, provide:
- source citation
- official download link if available
- a manifest entry showing how the file was used

For the current package, that decision boundary is documented in:

- `docs/SOURCE_LICENSE_MATRIX.md`
- `docs/DOWNLOAD_POINTERS.md`

## 9. Protocol Definitions

### strict_t2

`strict_t2` is the main heterogeneous benchmark. It requires:
- static support
- weather-window support
- SMAP availability

It is used to study multi-source transfer under stronger heterogeneity.

### strict_t3

`strict_t3` is the DLR-centered high-fidelity subset. It requires:
- static support
- DLR-specific ERA5 support

It is used to anchor the manuscript's high-fidelity reference analyses.

## 10. Recommended Citation

Once the release receives a DOI, cite both:

1. the data release DOI
2. the associated journal article

Suggested placeholder format:

```text
Author(s), 2026. Physics-Informed Landslide Dataset (PILD), version v1.0. [Data set]. DOI: ...
```

The package also includes a fill-in `CITATION.cff` template that should be updated with the final author metadata before DOI minting.

## 11. Known Limitations

- Source coverage is heterogeneous across datasets.
- Trigger labels are incomplete for many events and include an `unknown` category.
- The public release is metadata-complete, not raw-data-complete.
- Some event locations are represented by bounding-box centers rather than precise mapped initiation points.
- `strict_t3` is DLR-only and should not be interpreted as a globally balanced benchmark on its own.

## 12. Recommended Data Availability Wording

Suggested manuscript-facing wording:

> The study builds on public source datasets whose raw imagery, labels, and derived physical layers remain subject to the licenses and distribution policies of their original providers. The public PILD release therefore focuses on event-level metadata, protocol manifests, split definitions, source registries, and reproducibility assets rather than redistributing all upstream raw data.

## 13. Contact

Please include before DOI minting:
- corresponding author name
- institutional affiliation
- contact e-mail
- project or laboratory webpage if available

## 14. Final Release Checklist

Before minting a DOI, confirm:

- all local absolute paths are removed or replaced
- all private mirrors are excluded
- all redistributed files have confirmed licenses
- all upstream sources are acknowledged
- README, data card, dictionary, and source registry are internally consistent
- the geographic coverage figure matches the released event table
- author metadata are finalized for the DOI landing page
