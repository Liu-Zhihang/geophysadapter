# PILD Protocol Summary

## 1. Main Release Scope

The main public release is centered on the heterogeneous benchmark family used in the associated dataset and code release.

The key protocol distinction is:

- `strict_t2`: heterogeneous multi-source benchmark
- `strict_t3`: DLR-centered high-fidelity subset

## 2. `strict_t2`

`strict_t2` is the main cross-domain benchmark used to study transfer across source families with mismatched sensing conditions and incomplete physics support.

### Admission rule

An event enters `strict_t2` when it has:

- static support
- weather-window support
- `SMAP` support

### Current source composition

- `GLaD4CD_v1`: `157`
- `DLR_Landslide_Ref_2025`: `25`
- `GDCLD`: `8`
- `CAS_Landslide`: `6`

### Main use in the manuscript

- heterogeneous transfer study
- `post_rgb` vs `change_rgb` protocol analysis
- boundary analysis for physics-aware gains

## 3. `strict_t3`

`strict_t3` is a smaller but higher-fidelity subset centered on `DLR Landslide Reference 2025`.

### Admission rule

An event enters `strict_t3` when it has:

- static support
- DLR-specific `ERA5` support

### Current source composition

- `DLR_Landslide_Ref_2025`: `25`

### Main use in the manuscript

- frontier-model validation
- richer physics support
- teacher and structured-prior development

## 4. Split Families

The release also includes split tables for:

- `in_domain`
- `LODO` (leave-one-dataset-out)
- `cross_trigger`

These split files are distributed as metadata tables rather than raw sample bundles.

## 5. Important Interpretation Rule

`strict_t3` should not be interpreted as a globally balanced benchmark on its own. It is a high-fidelity subset used to anchor the frontier line, while `strict_t2` is the main multi-source transfer benchmark.
