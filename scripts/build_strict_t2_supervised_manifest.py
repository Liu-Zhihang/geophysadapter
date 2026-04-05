#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CAS_EVENT_ALIASES = {
    "CAS_Hokkaido": ["Hokkaido Iburi-Tobu"],
    "CAS_Jiuzhai_valley": ["Jiuzhai valley (UAV-0.2m)", "Jiuzhai valley (UAV-0.5m)"],
    "CAS_Lombokt": ["Lombok"],
    "CAS_Palu": ["palu"],
    "CAS_Tiburon Peninsula (Planet)": ["Tiburon Peninsula (planet)"],
    "CAS_Tiburon_Peninsula_(Sentinel)t": ["Tiburon Peninsula (Sentinel)"],
}

GDCLD_EVENT_SCENES = {
    "GDCLD_Lushan": [("Lushan", "Lushan_1.tif"), ("Lushan", "Lushan_2.tif"), ("Lushan", "Lushan_3.tif")],
    "GDCLD_Mesetas": [("Mesetas", "Mesetas.tif")],
    "GDCLD_Palu": [("Palu", "Palu_1.tif"), ("Palu", "Palu_2.tif")],
    "GDCLD_Sumatra": [("Sumarta", "Sumarta.tif")],
}

FIELDNAMES = [
    "protocol",
    "fold_id",
    "role",
    "event_uid",
    "dataset_id",
    "event_date",
    "trigger_type",
    "sample_id",
    "sample_kind",
    "asset_status",
    "source_split",
    "image_path",
    "pre_path",
    "post_path",
    "label_path",
    "valid_mask_path",
    "h5_path",
    "h5_sample_index",
    "num_channels",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def detect_raw_root(root: Path) -> Path:
    for cand in (root / "raw_fullcopy", root / "raw"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"unable to find raw dataset root under {root}")


def detect_index_root(root: Path) -> Path:
    for cand in (root / "raw_fullcopy" / "indexes", root / "raw" / "indexes"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"unable to find indexes under {root}")


def remap_legacy_data_raw_path(path_str: str, project_root: Path, dataset_root: Path) -> str:
    if not path_str:
        return ""
    legacy_root = project_root / "data_raw"
    p = Path(path_str)
    try:
        rel = p.relative_to(legacy_root)
        mapped = dataset_root / "datasets" / rel
        return str(mapped)
    except Exception:
        return str(p)


def load_split_events(split_csv: Path, protocol: str, fold_id: str) -> dict[str, dict[str, str]]:
    events: dict[str, dict[str, str]] = {}
    for row in read_csv(split_csv):
        if row.get("protocol") != protocol or row.get("fold_id") != fold_id:
            continue
        events[row["event_uid"]] = row
    if not events:
        raise ValueError(f"no rows matched protocol={protocol} fold_id={fold_id} in {split_csv}")
    return events


def merge_event_meta(
    split_rows: dict[str, dict[str, str]],
    event_index_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    index_map = {row["event_uid"]: row for row in event_index_rows}
    merged: dict[str, dict[str, str]] = {}
    for event_uid, split_row in split_rows.items():
        base = dict(split_row)
        base.update(index_map.get(event_uid, {}))
        merged[event_uid] = base
    return merged


def make_row(meta: dict[str, str], **extra: str) -> dict[str, str]:
    row = {
        "protocol": meta.get("protocol", ""),
        "fold_id": meta.get("fold_id", ""),
        "role": meta.get("role", ""),
        "event_uid": meta.get("event_uid", ""),
        "dataset_id": meta.get("dataset_id", ""),
        "event_date": meta.get("event_date", ""),
        "trigger_type": meta.get("trigger_type", ""),
        "sample_id": "",
        "sample_kind": "",
        "asset_status": "",
        "source_split": "",
        "image_path": "",
        "pre_path": "",
        "post_path": "",
        "label_path": "",
        "valid_mask_path": "",
        "h5_path": "",
        "h5_sample_index": "",
        "num_channels": "",
        "notes": "",
    }
    row.update(extra)
    return row


def resolve_cas_samples(
    event_meta: dict[str, dict[str, str]],
    index_root: Path,
    project_root: Path,
    dataset_root: Path,
) -> list[dict[str, str]]:
    whitelist_path = index_root / "cas_pair_whitelist.csv"
    rows = read_csv(whitelist_path)
    rows_by_event = defaultdict(list)
    for row in rows:
        rows_by_event[row["event"]].append(row)

    out: list[dict[str, str]] = []
    for event_uid, aliases in CAS_EVENT_ALIASES.items():
        meta = event_meta.get(event_uid)
        if meta is None:
            continue
        matched = 0
        for alias in aliases:
            for row in rows_by_event.get(alias, []):
                matched += 1
                out.append(
                    make_row(
                        meta,
                        sample_id=f"{event_uid}::{row['sample_id']}",
                        sample_kind="cas_single_rgb",
                        asset_status="supervised_ready",
                        image_path=remap_legacy_data_raw_path(row.get("img_path", ""), project_root, dataset_root),
                        label_path=remap_legacy_data_raw_path(row.get("label_path", ""), project_root, dataset_root),
                        valid_mask_path=remap_legacy_data_raw_path(row.get("mask_path", ""), project_root, dataset_root),
                        num_channels="3",
                        notes=f"alias={alias}; labels use RGB 0/255 and valid mask uses 0/1",
                    )
                )
        if matched == 0:
            out.append(
                make_row(
                    meta,
                    sample_id=f"{event_uid}::UNRESOLVED",
                    sample_kind="cas_single_rgb",
                    asset_status="event_unresolved",
                    notes=f"no whitelist samples matched aliases={json.dumps(aliases, ensure_ascii=False)}",
                )
            )
    return out


def resolve_gdcld_samples(
    event_meta: dict[str, dict[str, str]],
    dataset_root: Path,
) -> list[dict[str, str]]:
    gd_root = dataset_root / "datasets" / "01_GDCLD"
    out: list[dict[str, str]] = []
    for event_uid, meta in event_meta.items():
        if meta.get("dataset_id") != "GDCLD":
            continue
        scenes = GDCLD_EVENT_SCENES.get(event_uid, [])
        if not scenes:
            out.append(
                make_row(
                    meta,
                    sample_id=f"{event_uid}::UNRESOLVED",
                    sample_kind="gdcld_single_rgb",
                    asset_status="event_unresolved",
                    notes="strict event exists, but no stable event->scene mapping beyond the georeferenced test scenes",
                )
            )
            continue
        for folder, name in scenes:
            img = gd_root / "test_data" / "test_data" / folder / name
            lab = gd_root / "test_data" / "test_label" / folder / name
            out.append(
                make_row(
                    meta,
                    sample_id=f"{event_uid}::{folder}/{name}",
                    sample_kind="gdcld_single_rgb",
                    asset_status="supervised_ready",
                    image_path=str(img),
                    label_path=str(lab),
                    num_channels="3",
                    notes="GDCLD test label uses nodata=3 -> ignore; train/val patches are not event-resolved here",
                )
            )
    return out


def resolve_glad_samples(
    event_meta: dict[str, dict[str, str]],
    dataset_root: Path,
) -> list[dict[str, str]]:
    glad_root = dataset_root / "datasets" / "06_GLaD4CD" / "GLaD4CD_v1_unpacked" / "LANDSLIDE_DATASET"
    out: list[dict[str, str]] = []
    for event_uid, meta in event_meta.items():
        if meta.get("dataset_id") != "GLaD4CD_v1":
            continue
        file_idx = int(meta["event_uid"].split("_")[1]) - 1
        stem = str(file_idx)
        train_pre = glad_root / "TRAINING" / "A" / f"{stem}.tif"
        train_post = glad_root / "TRAINING" / "B" / f"{stem}.tif"
        val_pre = glad_root / "VALIDATION" / "A" / f"{stem}.tif"
        val_post = glad_root / "VALIDATION" / "B" / f"{stem}.tif"
        val_label = glad_root / "VALIDATION" / "LABEL" / f"{stem}.tif"
        if val_pre.exists() and val_post.exists() and val_label.exists():
            out.append(
                make_row(
                    meta,
                    sample_id=f"{event_uid}::{stem}",
                    sample_kind="glad_pre_post",
                    asset_status="supervised_ready",
                    source_split="VALIDATION",
                    pre_path=str(val_pre),
                    post_path=str(val_post),
                    label_path=str(val_label),
                    num_channels="5",
                    notes=f"file_index={stem}; GLaD TIFF numbering is zero-based relative to GLADV1 ids",
                )
            )
        elif train_pre.exists() and train_post.exists():
            out.append(
                make_row(
                    meta,
                    sample_id=f"{event_uid}::{stem}",
                    sample_kind="glad_pre_post",
                    asset_status="unlabeled_pair_only",
                    source_split="TRAINING",
                    pre_path=str(train_pre),
                    post_path=str(train_post),
                    num_channels="5",
                    notes=f"file_index={stem}; optical pair exists but no segmentation label was unpacked for v1 TRAINING",
                )
            )
        else:
            out.append(
                make_row(
                    meta,
                    sample_id=f"{event_uid}::{stem}",
                    sample_kind="glad_pre_post",
                    asset_status="event_unresolved",
                    notes=f"expected GLaD v1 files not found for zero-based index {stem}",
                )
            )
    return out


def resolve_dlr_samples(
    event_meta: dict[str, dict[str, str]],
    root: Path,
) -> list[dict[str, str]]:
    subset_dir = root / "processed" / "hybrid_pinn" / "dlr_strict_t3_reference_subset_v1"
    sample_manifest = subset_dir / "sample_manifest.csv"
    sample_rows = read_csv(sample_manifest)
    split_counters = Counter()
    out: list[dict[str, str]] = []
    for row in sample_rows:
        event_uid = row["event_uid"]
        meta = event_meta.get(event_uid)
        if meta is None:
            continue
        source_split = row["split"]
        local_idx = split_counters[source_split]
        split_counters[source_split] += 1
        h5_path = subset_dir / f"{source_split}_n3_s1s2.h5"
        out.append(
            make_row(
                meta,
                sample_id=row["sample_id"],
                sample_kind="dlr_h5_patch",
                asset_status="supervised_ready",
                source_split=source_split,
                h5_path=str(h5_path),
                h5_sample_index=str(local_idx),
                num_channels="pre4+post4+terrain2",
                notes=f"sample_sid={row['sample_sid']}; label is stored as None_MASK in the same H5 record",
            )
        )
    return out


def build_report(rows: list[dict[str, str]], output_path: Path) -> None:
    per_dataset_status = defaultdict(Counter)
    per_dataset_roles = defaultdict(Counter)
    sample_counts = Counter()
    event_status = defaultdict(lambda: defaultdict(set))
    event_count = Counter()
    for row in rows:
        dataset = row["dataset_id"]
        status = row["asset_status"]
        per_dataset_status[dataset][status] += 1
        per_dataset_roles[dataset][row["role"]] += 1
        if status == "supervised_ready":
            sample_counts[dataset] += 1
        event_status[dataset][status].add(row["event_uid"])
        event_count[dataset].add(row["event_uid"]) if False else None
    # Counter cannot hold sets, compute explicitly.
    event_totals = defaultdict(set)
    for row in rows:
        event_totals[row["dataset_id"]].add(row["event_uid"])

    lines = [
        "# strict_t2 Supervised Readiness Report",
        "",
        "This report summarizes which `strict_t2` events currently resolve to supervised segmentation assets.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | strict_events_seen | supervised_ready_events | supervised_ready_samples | unlabeled_pair_only_events | unresolved_events |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(event_totals):
        lines.append(
            "| {dataset} | {strict_events} | {ready_events} | {ready_samples} | {unlabeled_events} | {unresolved_events} |".format(
                dataset=dataset,
                strict_events=len(event_totals[dataset]),
                ready_events=len(event_status[dataset].get("supervised_ready", set())),
                ready_samples=sample_counts[dataset],
                unlabeled_events=len(event_status[dataset].get("unlabeled_pair_only", set())),
                unresolved_events=len(event_status[dataset].get("event_unresolved", set())),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `DLR_Landslide_Ref_2025`: fully supervised at patch level; current gap is split/loader generalization, not missing labels.",
            "- `CAS_Landslide`: the 6 strict events resolve cleanly through `cas_pair_whitelist.csv` and are supervision-ready.",
            "- `GLaD4CD_v1`: most strict events only have unlabeled `TRAINING/A,B` pairs; only the small labeled `VALIDATION` subset is directly usable for supervised segmentation.",
            "- `GDCLD`: only the georeferenced `test_data` scenes can be event-resolved here; several strict events still lack a stable event-to-scene mapping in the current local indexes.",
            "",
            "## By Status",
            "",
            "| dataset | asset_status | rows |",
            "|---|---|---:|",
        ]
    )
    for dataset in sorted(per_dataset_status):
        for status, count in sorted(per_dataset_status[dataset].items()):
            lines.append(f"| {dataset} | {status} | {count} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Build a strict_t2 supervised asset manifest across DLR/CAS/GDCLD/GLaD4CD v1")
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="PILD root",
    )
    p.add_argument(
        "--split-csv",
        default="",
        help="default: metadata/splits/split_in_domain.csv",
    )
    p.add_argument("--protocol", default="in_domain")
    p.add_argument("--fold-id", default="ID01")
    p.add_argument(
        "--event-index",
        default="",
        help="default: metadata/manifests/event_index_v1_strict_t2.csv",
    )
    p.add_argument(
        "--output",
        default="",
        help="default: metadata/manifests/strict_t2_supervised_asset_manifest_v1.csv",
    )
    p.add_argument(
        "--report",
        default="",
        help="default: metadata/reports/strict_t2_supervised_readiness_report.md",
    )
    args = p.parse_args()

    root = Path(args.root)
    project_root = root.parent
    dataset_root = detect_raw_root(root)
    index_root = detect_index_root(root)
    split_csv = Path(args.split_csv) if args.split_csv.strip() else root / "metadata" / "splits" / "split_in_domain.csv"
    event_index = Path(args.event_index) if args.event_index.strip() else root / "metadata" / "manifests" / "event_index_v1_strict_t2.csv"
    output_csv = Path(args.output) if args.output.strip() else root / "metadata" / "manifests" / "strict_t2_supervised_asset_manifest_v1.csv"
    report_md = Path(args.report) if args.report.strip() else root / "metadata" / "reports" / "strict_t2_supervised_readiness_report.md"

    split_rows = load_split_events(split_csv, protocol=args.protocol, fold_id=args.fold_id)
    event_meta = merge_event_meta(split_rows, read_csv(event_index))

    rows: list[dict[str, str]] = []
    rows.extend(resolve_cas_samples(event_meta, index_root, project_root, dataset_root))
    rows.extend(resolve_gdcld_samples(event_meta, dataset_root))
    rows.extend(resolve_glad_samples(event_meta, dataset_root))
    rows.extend(resolve_dlr_samples(event_meta, root))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    build_report(rows, report_md)

    status_counts = Counter(row["asset_status"] for row in rows)
    payload = {
        "protocol": args.protocol,
        "fold_id": args.fold_id,
        "rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "output_csv": str(output_csv),
        "report_md": str(report_md),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
