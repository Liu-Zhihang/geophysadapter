# DATA_CARD (PILD_v1_strict_T2)

## 1. Overview
- Name: `Physics-Informed Landslide Dataset (PILD)`
- Version: `v1_strict_t2`
- Unit: event-level index for multi-source landslide study
- Size: 196 events

## 2. Composition
- Dataset sources:
dataset_id
GLaD4CD_v1                157
DLR_Landslide_Ref_2025     25
GDCLD                       8
CAS_Landslide               6

- Detailed source registry:
  `DATA_SOURCE_REGISTRY.md`

- Trigger distribution:
trigger_type
unknown       164
earthquake     14
rainfall       10
complex         4
snowmelt        2
storm           2

## 3. Intended Use
- Cross-domain landslide mapping and physics-informed segmentation.
- Benchmark protocols: in-domain, LODO, cross-trigger.

## 4. Not Intended Use
- Not for direct hazard warning deployment without regional validation.
- Not for legal or regulatory decision as-is.

## 5. Data Processing
- Built from strict whitelist and event index.
- Static/meteorological layers require alignment in downstream preprocessing.
- Raw-source provenance, current training usage, and static/weather source mapping are documented in:
  `DATA_SOURCE_REGISTRY.md`

## 6. Quality / Limitations
- Some source tiles are unavailable from upstream providers (HTTP 404).
- Trigger labels include many `unknown` entries (not all events have reliable trigger metadata).

## 7. Licensing Note
- This package aggregates multi-source datasets.
- Redistribution must follow each original provider license.
