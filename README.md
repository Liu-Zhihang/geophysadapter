# GeoPhysAdapter

Code and release-safe metadata for **GeoPhysAdapter**, a scale-matched geophysical adapter for cross-domain landslide mapping with vision foundation models.

Paper:

**GeoPhysAdapter: Scale-Matched Geophysical Adaptation for Cross-Domain Landslide Mapping with Vision Foundation Models**

## Overview

GeoPhysAdapter keeps a vision foundation model frozen (primary anchor: Prithvi-EO-2.0-300M-TL) and applies a constrained physical update around its prediction:

- Terrain provides a densely aligned correction direction
- Material modulates regional amplitude within a bounded range
- Trigger supplies an event-level intervention dose

Updates operate at the pixel scale and at the candidate landslide-body scale. When physical support is invalid or insufficient, the method returns the frozen visual prediction unchanged.

## PILD metadata in this repository

The public release includes metadata for the four-source PILD corpus used in the paper:

- 7,890 samples
- 55 canonical events
- event-isolated splits and protocol summaries

Main paths:

- `metadata/pild_geo4_qc_v1/`
- `metadata/pild_geo4_qc_native17_v1/` (protocol summary and hashes; see Supplement S6)

Raw imagery and mixed-license environmental rasters are **not** redistributed here. Obtain source data from the original providers; the Zenodo package lists download pointers and license notes.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/xdomain/` | Training, evaluation, and role-aware modules |
| `scripts/` | Figure exporters and auxiliary analysis scripts |
| `metadata/` | Manifests, splits, and protocol tables |
| `experiments/revision2026/` | Lightweight numeric summaries (no checkpoints) |
| `docs/assets/` | Paper figures shown below |
| `release_prep/` | Zenodo packaging notes |

## Setup

```bash
conda env create -f environment.yml
conda activate geophysadapter
```

## Reproduction entry points (Supplement S6)

| Step | Script / artifact |
|---|---|
| Four-source training | `scripts/xdomain/train_pild_sen12_roleaware_v1.py` |
| Terrain encoding | `scripts/xdomain/sen12_terrain_v2.py` |
| Material module | `scripts/xdomain/pild_roleaware_material.py` |
| Trigger module | `scripts/xdomain/pild_roleaware_trigger.py` |
| Pixel utility gate | `scripts/xdomain/evaluate_pild_benefit_gate_v1.py` |
| Object-level review | `scripts/xdomain/evaluate_pild_object_veto_final_v1.py` |
| Object-level summary | `experiments/revision2026/pild_object_veto_final_v1/summary.json` |
| Protocol hashes | `metadata/pild_geo4_qc_native17_v1/protocol_summary_geo4_qc_native17_v1.json` |

Checkpoints and large OOF tensors are not included. Rebuild local caches from the scripts above and the source imagery referenced in Zenodo.

## Data package

- Zenodo: https://doi.org/10.5281/zenodo.19430714

## Figures

### Figure 1. Conceptual overview

![Figure 1](docs/assets/figure1_scale_matched_overview.png)

### Figure 2. Global distribution of the 55 canonical events

![Figure 2](docs/assets/figure2_global_source_distribution.png)

### Figure 3. Technical framework

![Figure 3](docs/assets/figure3_scale_matched_framework.png)

### Figure 4. Object structure of cross-domain visual errors

![Figure 4](docs/assets/figure4_visual_error_structure.png)

### Figure 5. Pixel-scale adaptation and terrain-content controls

![Figure 5](docs/assets/figure5_pixel_scale_capability.png)

### Figure 6. Native-task evidence for the three priors

![Figure 6](docs/assets/figure6_native_task_evidence.png)

### Figure 7. Object-scale correction cases

![Figure 7](docs/assets/figure7_object_scale_correction.png)

### Figure 8. Cross-anchor reproduction and UGCoP landscape

![Figure 8](docs/assets/figure8_attribution_and_reproduction.png)

### Supplementary Figure S1. Object-scale behavior spectrum

![Figure S1](docs/assets/figureS1_object_scale_behavior_spectrum.png)

## Citation

Please cite the paper and the Zenodo record above if you use this repository or the PILD release assets.

## License and contact

See the repository license file and the Zenodo record for release terms. Questions about the public release can be opened as GitHub issues.
