# GeoPhysAdapter

Official code for:

**GeoPhysAdapter: Scale-Matched Geophysical Adaptation for Cross-Domain Landslide Mapping with Vision Foundation Models**

## Overview

GeoPhysAdapter freezes a vision foundation model (primary anchor: Prithvi-EO-2.0-300M-TL) and updates its prediction with three geophysical priors:

- Terrain: densely aligned correction direction
- Material: bounded regional amplitude modulation
- Trigger: event-level intervention dose

The update runs at both the pixel scale and the candidate landslide-body scale. If physical support is missing or invalid, the frozen visual prediction is kept as is.

## Dataset (PILD)

This repository includes the metadata used by the paper for the four-source PILD corpus:

- 7,890 samples
- 55 canonical events
- event-isolated splits and protocol summaries

- `metadata/pild_geo4_qc_v1/`
- `metadata/pild_geo4_qc_native17_v1/` (protocol summary and hashes; Supplement S6)

Raw imagery remains with the original data providers. Download pointers and license notes are in the Zenodo record below.

## Repository structure

| Path | Contents |
|---|---|
| `scripts/xdomain/` | Training and evaluation code |
| `scripts/` | Figure and analysis utilities |
| `metadata/` | Sample manifests and splits |
| `experiments/revision2026/` | Summary metrics reported in the paper |
| `docs/assets/` | Figures |

## Installation

```bash
conda env create -f environment.yml
conda activate geophysadapter
```

## Reproducing Supplement S6

| Step | Path |
|---|---|
| Four-source training | `scripts/xdomain/train_pild_sen12_roleaware_v1.py` |
| Terrain encoding | `scripts/xdomain/sen12_terrain_v2.py` |
| Material module | `scripts/xdomain/pild_roleaware_material.py` |
| Trigger module | `scripts/xdomain/pild_roleaware_trigger.py` |
| Pixel utility gate | `scripts/xdomain/evaluate_pild_benefit_gate_v1.py` |
| Object-level review | `scripts/xdomain/evaluate_pild_object_veto_final_v1.py` |
| Object-level summary | `experiments/revision2026/pild_object_veto_final_v1/summary.json` |
| Protocol hashes | `metadata/pild_geo4_qc_native17_v1/protocol_summary_geo4_qc_native17_v1.json` |

Model checkpoints are not shipped with the repository. Train or evaluate locally with the scripts above after preparing the source imagery listed on Zenodo.

## Data

Zenodo: https://doi.org/10.5281/zenodo.19430714

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

If you use this code or the PILD assets, please cite the paper and the Zenodo DOI above.

## Contact

Please open a GitHub issue for questions about the code or data links.
