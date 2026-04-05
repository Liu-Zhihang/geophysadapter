# DATA_DICTIONARY

## 1. event_index_v1_strict_t2.csv
- `event_uid`: unique event identifier
- `dataset_id`: source dataset
- `event_date`: event date (YYYY-MM-DD)
- `date_quality`: exact / inferred_name / partial / missing
- `has_bbox`: whether bbox exists (0/1)
- `has_dem_core`: CopDEM core availability (0/1)
- `has_worldcover_core`: WorldCover core availability (0/1)
- `has_static_core`: static strict availability (0/1)
- `has_weather_window`: valid weather window flag (0/1)
- `has_smap_pool`: SMAP-in-window availability (0/1)
- `has_era5_dlr`: ERA5-DLR availability (0/1)
- `strict_t1_static`: T1 membership flag (0/1)
- `strict_t2_static_weather_smap`: T2 membership flag (0/1)
- `strict_t3_static_era5_dlr`: T3 membership flag (0/1)
- `missing_reasons`: pipe-separated missing reasons
- `trigger_type`: trigger class label
- `download_group`: core/usgs group tag
- `n_samples`: per-event sample count hint
- `release_tag`: release identifier

## 2. split_in_domain.csv
- `protocol`: in_domain
- `fold_id`: ID01
- `role`: train / val / test
- Other columns inherited from event index.

## 3. split_lodo.csv
- `protocol`: lodo
- `fold_id`: LODO_XX
- `held_out_dataset`: dataset used as test in this fold
- `role`: train / val / test

## 4. split_cross_trigger.csv
- `protocol`: cross_trigger
- `fold_id`: CT01 / CT02 / CT03
- `rule`: test trigger rule (`rainfall`, `earthquake`, `unknown`)
- `role`: train / val / test
