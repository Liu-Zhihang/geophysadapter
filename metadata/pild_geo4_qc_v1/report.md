# PILD-GEO4 QC summary

- Retained samples: `7,890/7,916`.
- DLR retained events: `22/23`.
- CAS is excluded from this training manifest; source files are unchanged.
- DLR hard exclusion: `CA0002` only.
- DLR role eligibility: Terrain `22`, Material `20`, Trigger `8`, Full-TMR `6` events.

## Hard exclusion

`CA0002` has only `3/26` four-date-quality-pass samples and an event mean optical-valid fraction of approximately `0.284`. Labels, predictions, IoU, and model errors were not used for this decision.

## Role-specific abstention

Material abstention events:

- `JP0001`: q_M_full_mean=`0.7738`.
- `PH0001;PH0003;PH0004`: q_M_full_mean=`0.7990`.

Trigger abstention events:

- `CA0001`: `mechanism_not_rainfall`.
- `CO0001`: `mechanism_not_rainfall`.
- `PK0001`: `mechanism_not_rainfall`.
- `IT0001`: `mechanism_not_rainfall`.
- `BR0001`: `mechanism_not_rainfall`.
- `IR0002`: `mechanism_not_rainfall`.
- `NZ0001`: `mechanism_not_rainfall`.
- `CL0002`: `mechanism_not_rainfall`.
- `IE0001`: `incomplete_chirps_coverage`.
- `IS0001`: `incomplete_chirps_coverage`.
- `IR0001`: `mechanism_not_rainfall`.
- `KG0001`: `mechanism_not_rainfall`.
- `IN0001`: `mechanism_not_rainfall`.
- `IN0004`: `mechanism_not_rainfall`.

Role abstention does not remove an event from Terrain experiments. Unsupported branches return the parent prediction.
