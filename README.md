# GeoPhysAdapter

**GeoPhysAdapter: Scale-Matched Geophysical Adaptation for Cross-Domain Landslide Mapping with Vision Foundation Models**

GeoPhysAdapter freezes a vision foundation model (primary anchor: Prithvi-EO-2.0-300M-TL) and updates its prediction with three geophysical priors:

- Terrain: densely aligned correction direction
- Material: bounded regional amplitude modulation
- Trigger: event-level intervention dose

Updates are applied at the pixel scale and at the candidate landslide-body scale. If physical support is missing or invalid, the frozen visual prediction is kept unchanged.

## Data

Two things are open:

1. **PILD** — our curated metadata package (manifests, splits, protocol tables) for the paper corpus (**7,890 samples / 55 events**)
2. **Upstream sources** — the public landslide datasets from which PILD samples are drawn (imagery stays with those providers)

How to download and use both: **[docs/DATA.md](docs/DATA.md)**

PILD package (Zenodo): https://doi.org/10.5281/zenodo.19430714

| Upstream source | Samples in PILD | Link |
|---|---:|---|
| Sen12Landslides (harmonized) | 4,979 | https://huggingface.co/datasets/paulhoehn/Sen12Landslides |
| GDCLD | 2,334 | https://doi.org/10.5194/essd-16-4817-2024 |
| DLR Landslide Reference | 509 | https://doi.org/10.5281/zenodo.17007637 |
| GLaD4CD v1 | 68 | https://doi.org/10.5281/zenodo.14226448 |

Local metadata used by the scripts:

- `metadata/pild_geo4_qc_v1/`
- `metadata/pild_geo4_qc_native17_v1/`

## Install

```bash
conda env create -f environment.yml
conda activate geophysadapter
```

## Reproduce the main protocol (Supplement S6)

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

Example after PILD metadata and local caches are ready:

```bash
python scripts/xdomain/train_pild_sen12_roleaware_v1.py \
  --manifest metadata/pild_geo4_qc_v1/unified_sample_manifest_geo4_qc_v1.csv \
  --protocol-summary metadata/pild_geo4_qc_v1/summary.json \
  --split metadata/pild_geo4_qc_v1/event_isolated_split_geo4_qc_v1.csv \
  --variant full_tmr \
  --outdir experiments/local_run
```

## Repository layout

| Path | Contents |
|---|---|
| `scripts/xdomain/` | Training and evaluation |
| `scripts/` | Figure / table helpers |
| `metadata/` | PILD manifests and splits |
| `experiments/revision2026/` | Numeric summaries from the paper |
| `docs/` | Data guide and figures |

## Example figures

Three paper figures are included below for a quick look at the method, the corpus, and a qualitative result. The full figure set is in the manuscript.

**Figure 1 — Method overview.**  
Multi-source optical inputs and three geophysical priors enter a frozen vision foundation model; GeoPhysAdapter applies bounded pixel-scale correction and object-scale review, with exact fallback where physical support is invalid.

![Figure 1](docs/assets/figure1_scale_matched_overview.png)

**Figure 2 — PILD event map.**  
Global distribution of the 55 canonical events used in the paper. Marker color is the upstream source; marker size scales with samples per event.

![Figure 2](docs/assets/figure2_global_source_distribution.png)

**Figure 7 — Object-scale correction cases.**  
Post-event image, reference inventory, frozen visual prediction, and GeoPhysAdapter output on representative tiles. Teal / coral / pale gold mark TP / FP / missed pixels; slate outlines mark wholly vetoed candidates.

![Figure 7](docs/assets/figure7_object_scale_correction.png)

## Citation

Please cite the paper, the PILD Zenodo record, and any upstream datasets you download (see [docs/DATA.md](docs/DATA.md)).

## Contact

Open a GitHub issue for questions about the code or data.
