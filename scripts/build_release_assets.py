#!/usr/bin/env python3
"""Build release assets for PILD_v1_strict_T2."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class SplitCfg:
    fold_id: str
    rule: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--seed", type=int, default=20260305)
    return ap.parse_args()


def role_counts(n: int, val_ratio: float = 0.15, test_ratio: float = 0.15) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) with at least one train sample."""
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    while n - n_test - n_val < 1:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
    return n_train, n_val, n_test


def select_val_indices(df: pd.DataFrame, seed: int, ratio: float = 0.10) -> set[int]:
    """Select validation rows from a train pool with per-dataset balancing."""
    val_ids: set[int] = set()
    for i, (ds, g) in enumerate(sorted(df.groupby("dataset_id")), start=1):
        n = len(g)
        if n <= 1:
            continue
        k = max(1, round(n * ratio))
        if n - k < 1:
            k = n - 1
        if k <= 0:
            continue
        gg = g.sample(frac=1.0, random_state=seed + i)
        val_ids.update(gg.head(k).index.tolist())
    return val_ids


def build_in_domain(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for i, (ds, g) in enumerate(sorted(df.groupby("dataset_id")), start=1):
        n = len(g)
        n_train, n_val, n_test = role_counts(n)
        gg = g.sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
        roles = (["test"] * n_test) + (["val"] * n_val) + (["train"] * n_train)
        gg["role"] = roles[:n]
        gg["protocol"] = "in_domain"
        gg["fold_id"] = "ID01"
        gg["rule"] = "stratified by dataset_id"
        gg["seed"] = seed
        rows.append(gg)
    out = pd.concat(rows, ignore_index=True)
    return out


def build_lodo(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    folds = []
    datasets = sorted(df["dataset_id"].unique().tolist())
    for i, held in enumerate(datasets, start=1):
        test = df[df["dataset_id"] == held].copy()
        pool = df[df["dataset_id"] != held].copy()
        val_ids = select_val_indices(pool, seed + i, ratio=0.10)
        pool["role"] = ["val" if idx in val_ids else "train" for idx in pool.index]
        test["role"] = "test"
        fold = pd.concat([pool, test], ignore_index=True)
        fold["protocol"] = "lodo"
        fold["fold_id"] = f"LODO_{i:02d}"
        fold["held_out_dataset"] = held
        fold["rule"] = f"test=dataset_id:{held}"
        fold["seed"] = seed
        folds.append(fold)
    return pd.concat(folds, ignore_index=True)


def build_cross_trigger(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    d = df.copy()
    d["trigger_norm"] = d["trigger_type"].fillna("unknown").astype(str).str.strip().str.lower()

    cfgs = [
        SplitCfg("CT01", "test=rainfall"),
        SplitCfg("CT02", "test=earthquake"),
        SplitCfg("CT03", "test=unknown"),
    ]
    out = []
    for i, cfg in enumerate(cfgs, start=1):
        test_tag = cfg.rule.split("=")[1]
        test = d[d["trigger_norm"] == test_tag].copy()
        pool = d[d["trigger_norm"] != test_tag].copy()
        if len(test) == 0 or len(pool) <= 1:
            continue
        val_ids = select_val_indices(pool, seed + 100 + i, ratio=0.10)
        pool["role"] = ["val" if idx in val_ids else "train" for idx in pool.index]
        test["role"] = "test"
        fold = pd.concat([pool, test], ignore_index=True)
        fold["protocol"] = "cross_trigger"
        fold["fold_id"] = cfg.fold_id
        fold["rule"] = cfg.rule
        fold["seed"] = seed
        out.append(fold)
    if not out:
        return d.iloc[0:0].copy()
    return pd.concat(out, ignore_index=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(pattern))


def manifest_completion(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    d = pd.read_csv(path)
    ok = sum(1 for tp in d["target_path"] if Path(tp).exists())
    total = len(d)
    return ok, total, total - ok


def write_data_card(root: Path, df: pd.DataFrame) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    out = docs / "DATA_CARD.md"
    txt = f"""# DATA_CARD (PILD_v1_strict_T2)

## 1. Overview
- Name: `Physics-Informed Landslide Dataset (PILD)`
- Version: `v1_strict_t2`
- Unit: event-level index for multi-source landslide study
- Size: {len(df)} events

## 2. Composition
- Dataset sources:
{df['dataset_id'].value_counts().to_string()}

- Trigger distribution:
{df['trigger_type'].fillna('unknown').value_counts().to_string()}

## 3. Intended Use
- Cross-domain landslide mapping and physics-informed segmentation.
- Benchmark protocols: in-domain, LODO, cross-trigger.

## 4. Not Intended Use
- Not for direct hazard warning deployment without regional validation.
- Not for legal or regulatory decision as-is.

## 5. Data Processing
- Built from strict whitelist and event index.
- Static/meteorological layers require alignment in downstream preprocessing.

## 6. Quality / Limitations
- Some source tiles are unavailable from upstream providers (HTTP 404).
- Trigger labels include many `unknown` entries (not all events have reliable trigger metadata).

## 7. Licensing Note
- This package aggregates multi-source datasets.
- Redistribution must follow each original provider license.
"""
    out.write_text(txt, encoding="utf-8")


def write_data_dictionary(root: Path) -> None:
    out = root / "docs" / "DATA_DICTIONARY.md"
    txt = """# DATA_DICTIONARY

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
"""
    out.write_text(txt, encoding="utf-8")


def write_release_notes(root: Path) -> None:
    out = root / "docs" / "RELEASE_NOTES_v1.md"
    txt = """# RELEASE_NOTES_v1

## Included
- Strict event whitelists (T1/T2/T3)
- Primary event index: `event_index_v1_strict_t2.csv`
- Standard splits:
  - `split_in_domain.csv`
  - `split_lodo.csv`
  - `split_cross_trigger.csv`
- Integrity report and checksums
- Data card and data dictionary

## Entry
- Default raw entry: `raw -> raw_fullcopy`
"""
    out.write_text(txt, encoding="utf-8")


def write_integrity_report(root: Path, df: pd.DataFrame, split_dir: Path) -> None:
    reports = root / "metadata" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / "integrity_report_v1.md"

    proj = root.parent
    mani = proj / "data_external" / "manifests"
    c_era5 = manifest_completion(mani / "era5_land_dlr_manifest.csv")
    c_c_dlr = manifest_completion(mani / "chirps_daily_dlr_manifest.csv")
    c_c_lite = manifest_completion(mani / "chirps_daily_corelite_manifest.csv")
    c_c_full = manifest_completion(mani / "chirps_daily_corefull_manifest.csv")
    c_dem = manifest_completion(mani / "copdem_core_manifest.csv")
    c_wc = manifest_completion(mani / "worldcover_core_manifest.csv")

    raw = root / "raw"
    n_chirps = count_files(raw / "weather" / "chirps_daily_global", "*.tif.gz")
    n_era5 = count_files(raw / "weather" / "era5_land", "*.grib")
    n_smap = count_files(raw / "weather" / "smap_spl3smp", "*.h5")
    n_cop = count_files(raw / "static" / "copdem_glo30_2021", "*.tif")
    n_wc = count_files(raw / "static" / "worldcover_v200_2021", "*.tif")

    lines = [
        "# Integrity Report v1",
        "",
        "## 1. Event Index",
        f"- rows: {len(df)}",
        f"- unique event_uid: {df['event_uid'].nunique()}",
        f"- duplicates(event_uid): {len(df) - df['event_uid'].nunique()}",
        "",
        "## 2. Source Completion (project manifests)",
        f"- ERA5 DLR: {c_era5[0]}/{c_era5[1]} (missing={c_era5[2]})",
        f"- CHIRPS DLR: {c_c_dlr[0]}/{c_c_dlr[1]} (missing={c_c_dlr[2]})",
        f"- CHIRPS corelite: {c_c_lite[0]}/{c_c_lite[1]} (missing={c_c_lite[2]})",
        f"- CHIRPS corefull: {c_c_full[0]}/{c_c_full[1]} (missing={c_c_full[2]})",
        f"- CopDEM core: {c_dem[0]}/{c_dem[1]} (missing={c_dem[2]})",
        f"- WorldCover core: {c_wc[0]}/{c_wc[1]} (missing={c_wc[2]})",
        "",
        "## 3. Raw Package Counts (PILD entry)",
        f"- chirps_daily_global (*.tif.gz): {n_chirps}",
        f"- era5_land (*.grib): {n_era5}",
        f"- smap_spl3smp (*.h5): {n_smap}",
        f"- copdem_glo30_2021 (*.tif): {n_cop}",
        f"- worldcover_v200_2021 (*.tif): {n_wc}",
        "",
        "## 4. Split Files",
    ]
    for sp in sorted(split_dir.glob("split_*.csv")):
        d = pd.read_csv(sp)
        lines.append(f"- {sp.name}: rows={len(d)}")
        if "role" in d.columns:
            vc = d["role"].value_counts().to_dict()
            lines.append(f"  - role_counts={vc}")
    lines.extend(
        [
            "",
            "## 5. Notes",
            "- Some static tiles are unavailable from source (HTTP 404), reflected in strict whitelist.",
            "- Cleanup details are recorded in `docs/CLEANUP_LOG_20260305.md`.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_checksums(root: Path, files: Iterable[Path]) -> None:
    out = root / "metadata" / "checksums.sha256"
    rows = []
    for p in sorted(files):
        if not p.exists() or not p.is_file():
            continue
        h = sha256_file(p)
        rel = p.relative_to(root)
        rows.append(f"{h}  {rel}")
    out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    meta = root / "metadata"
    split_dir = meta / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    event_idx = meta / "manifests" / "event_index_v1_strict_t2.csv"
    df = pd.read_csv(event_idx)

    # T30: split generation
    id_split = build_in_domain(df, args.seed)
    lodo_split = build_lodo(df, args.seed)
    ct_split = build_cross_trigger(df, args.seed)

    base_cols = list(df.columns)
    out_cols = base_cols + ["protocol", "fold_id", "role", "rule", "seed"]

    id_split[out_cols].to_csv(split_dir / "split_in_domain.csv", index=False, encoding="utf-8")
    lodo_cols = out_cols + ["held_out_dataset"]
    lodo_split[lodo_cols].to_csv(split_dir / "split_lodo.csv", index=False, encoding="utf-8")
    ct_split[out_cols].to_csv(split_dir / "split_cross_trigger.csv", index=False, encoding="utf-8")

    # T31: integrity report + checksum
    write_integrity_report(root, df, split_dir)

    # T32: docs
    write_data_card(root, df)
    write_data_dictionary(root)
    write_release_notes(root)

    # checksums for key release assets (metadata + docs + scripts)
    checksum_targets = []
    checksum_targets.extend((meta / "whitelists").glob("*.csv"))
    checksum_targets.extend((meta / "manifests").glob("*.csv"))
    checksum_targets.extend((meta / "manifests").glob("*.md"))
    checksum_targets.extend((meta / "splits").glob("*.csv"))
    checksum_targets.extend((meta / "reports").glob("*.md"))
    checksum_targets.extend((root / "docs").glob("*.md"))
    checksum_targets.append(root / "README.md")
    checksum_targets.extend((root / "scripts").glob("*.py"))
    build_checksums(root, checksum_targets)

    print("wrote splits:", [p.name for p in sorted(split_dir.glob("split_*.csv"))])
    print("wrote report:", meta / "reports" / "integrity_report_v1.md")
    print("wrote checksums:", meta / "checksums.sha256")
    print("wrote docs: DATA_CARD.md, DATA_DICTIONARY.md, RELEASE_NOTES_v1.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
