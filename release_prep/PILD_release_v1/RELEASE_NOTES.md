# Release Notes for `PILD_release_v1`

## Version Summary

`PILD_release_v1` is the first public metadata and protocol release for the dataset and codebase:

`GeoPhysAdapter: Physically Structured Foundation-Model Learning for Cross-Domain Landslide Mapping`

## Included

- event-level master index
- `strict_t2` event index
- benchmark split files
- `strict_t2` and `strict_t3` event summaries
- data card
- data dictionary
- source registry
- protocol summary
- source license matrix
- global event coverage figure
- figure rendering script for the coverage map

## Excluded

- raw imagery mirrors
- complete upstream label mirrors
- large third-party physical layers
- processed training shards
- local cache tensors
- teacher assets and path-bearing experiment artifacts

## Rationale

The release is intended to maximize auditability and manuscript support while minimizing license risk. It is metadata-complete for the benchmark definition, but not raw-data-complete for all upstream assets.

## Next Actions Before DOI Minting

- fill author and contact metadata if needed
- finalize the citation entry
- verify that no local absolute paths remain in redistributed text files
- confirm that all copied files match the final public release protocol
