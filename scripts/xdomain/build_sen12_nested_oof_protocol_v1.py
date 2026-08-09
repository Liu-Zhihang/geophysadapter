#!/usr/bin/env python3
"""Build leakage-safe nested OOF protocols for the frozen Sen12 LOGO5 data.

The outer test partition is used only to define exclusions.  Nested training,
validation, and test roles are formed exclusively from the corresponding
outer-development samples.  Physical events and spatial supergroups are joined
into connected components before assignment, so neither identity can cross an
inner role boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py


N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
REQUIRED_SPLIT_COLUMNS = {
    "sample_id",
    "outer_fold",
    "role",
    "region_group",
    "spatial_supergroup",
    "source_id",
}
ACTIVE_OUTER_DEVELOPMENT_ROLES = {"train", "val"}


class ProtocolError(RuntimeError):
    """Raised when the source identities cannot support a strict protocol."""


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Lexical root selection makes component construction order irrelevant.
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def decode_h5_strings(values: Iterable[Any]) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def hash_values(values: Iterable[str]) -> str:
    return hash_payload(sorted({str(value) for value in values}))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_split_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = REQUIRED_SPLIT_COLUMNS - set(fieldnames)
        if missing:
            raise ProtocolError(f"Split CSV misses columns: {sorted(missing)}")
        rows = [{key: str(value) for key, value in row.items()} for row in reader]
    if not rows:
        raise ProtocolError("Split CSV is empty")
    return fieldnames, rows


def read_h5_event_map(path: Path) -> dict[str, str]:
    with h5py.File(path, "r") as handle:
        missing = {"sample_id", "physical_event_id"} - set(handle.keys())
        if missing:
            raise ProtocolError(f"H5 misses identity datasets: {sorted(missing)}")
        sample_ids = decode_h5_strings(handle["sample_id"][:])
        event_ids = decode_h5_strings(handle["physical_event_id"][:])
    if len(sample_ids) != len(event_ids):
        raise ProtocolError("H5 sample_id and physical_event_id lengths differ")
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted(key for key, count in Counter(sample_ids).items() if count > 1)
        raise ProtocolError(f"H5 sample_id is not unique: {duplicates[:10]}")
    if any(not sample_id or not event_id for sample_id, event_id in zip(sample_ids, event_ids)):
        raise ProtocolError("H5 contains an empty sample or physical-event identity")
    return dict(zip(sample_ids, event_ids))


def rows_for_outer_fold(rows: Sequence[dict[str, str]], target: int) -> list[dict[str, str]]:
    selected = [row for row in rows if row["outer_fold"] == str(target)]
    if not selected:
        raise ProtocolError(f"No source rows for target outer fold {target}")
    sample_ids = [row["sample_id"] for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ProtocolError(f"Duplicate sample_id within target outer fold {target}")
    unknown_roles = sorted({row["role"] for row in selected} - {"train", "val", "test"})
    if unknown_roles:
        raise ProtocolError(f"Unknown source roles in outer fold {target}: {unknown_roles}")
    return selected


def connected_components(
    rows: Sequence[dict[str, str]], event_by_sample: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    union = UnionFind()
    for row in rows:
        region_node = f"region\x1f{row['spatial_supergroup']}"
        event_node = f"event\x1f{event_by_sample[row['sample_id']]}"
        union.union(region_node, event_node)

    samples_by_root: dict[str, list[str]] = defaultdict(list)
    regions_by_root: dict[str, set[str]] = defaultdict(set)
    events_by_root: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        region_node = f"region\x1f{row['spatial_supergroup']}"
        root = union.find(region_node)
        samples_by_root[root].append(row["sample_id"])
        regions_by_root[root].add(row["spatial_supergroup"])
        events_by_root[root].add(event_by_sample[row["sample_id"]])

    components = []
    sample_to_component: dict[str, str] = {}
    for root in sorted(samples_by_root):
        sample_ids = sorted(samples_by_root[root])
        spatial_supergroups = sorted(regions_by_root[root])
        physical_event_ids = sorted(events_by_root[root])
        identity = {
            "sample_ids": sample_ids,
            "spatial_supergroups": spatial_supergroups,
            "physical_event_ids": physical_event_ids,
        }
        component_id = f"NCOMP_{hash_payload(identity)[:16]}"
        component = {
            "component_id": component_id,
            "n_samples": len(sample_ids),
            "n_spatial_supergroups": len(spatial_supergroups),
            "n_physical_events": len(physical_event_ids),
            "sample_sha256": hash_values(sample_ids),
            "spatial_supergroup_sha256": hash_values(spatial_supergroups),
            "physical_event_sha256": hash_values(physical_event_ids),
            **identity,
        }
        components.append(component)
        for sample_id in sample_ids:
            sample_to_component[sample_id] = component_id
    components.sort(key=lambda item: item["component_id"])
    if len(components) < N_INNER_FOLDS:
        raise ProtocolError(
            f"Need at least {N_INNER_FOLDS} development components, found {len(components)}"
        )
    return components, sample_to_component


def assign_inner_test_components(components: Sequence[dict[str, Any]]) -> list[set[str]]:
    """Deterministic LPT allocation using component sample counts only."""
    buckets: list[set[str]] = [set() for _ in range(N_INNER_FOLDS)]
    loads = [0] * N_INNER_FOLDS
    ordered = sorted(components, key=lambda item: (-item["n_samples"], item["component_id"]))
    for component in ordered:
        fold = min(
            range(N_INNER_FOLDS),
            key=lambda index: (loads[index], len(buckets[index]), index),
        )
        buckets[fold].add(component["component_id"])
        loads[fold] += int(component["n_samples"])
    if any(not bucket for bucket in buckets):
        raise ProtocolError("At least one inner-test fold is empty")
    return buckets


def choose_validation_components(
    components: Sequence[dict[str, Any]], test_components: Sequence[set[str]]
) -> list[str]:
    """Choose one small deterministic non-test component and maximize train size."""
    component_sizes = {item["component_id"]: int(item["n_samples"]) for item in components}
    all_components = set(component_sizes)
    selected: list[str] = []
    val_use_count: Counter[str] = Counter()
    for fold, test_ids in enumerate(test_components):
        candidates = []
        for component_id in all_components - test_ids:
            remaining_train = all_components - test_ids - {component_id}
            if remaining_train:
                candidates.append(component_id)
        if not candidates:
            raise ProtocolError(f"Inner fold {fold} cannot form non-empty train and val roles")
        # Size is primary: a single small validation component leaves maximal
        # training support. Reuse count and ID provide balance and stable ties.
        chosen = min(
            candidates,
            key=lambda item: (component_sizes[item], val_use_count[item], item),
        )
        selected.append(chosen)
        val_use_count[chosen] += 1
    return selected


def output_fieldnames(source_fieldnames: Sequence[str]) -> list[str]:
    renamed = [name for name in source_fieldnames if name not in {"outer_fold", "role", "role_reason"}]
    additions = [
        "physical_event_id",
        "target_outer_fold",
        "source_outer_fold",
        "source_outer_role",
        "source_role_reason",
        "nested_component_id",
        "outer_fold",
        "role",
        "role_reason",
    ]
    return renamed + [name for name in additions if name not in renamed]


def csv_text(fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def role_audit(
    output_rows: Sequence[dict[str, str]], event_by_sample: Mapping[str, str]
) -> dict[str, Any]:
    role_samples: dict[str, set[str]] = defaultdict(set)
    role_regions: dict[str, set[str]] = defaultdict(set)
    role_events: dict[str, set[str]] = defaultdict(set)
    role_components: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        role = row["role"]
        role_samples[role].add(row["sample_id"])
        role_regions[role].add(row["spatial_supergroup"])
        role_events[role].add(event_by_sample[row["sample_id"]])
        role_components[role].add(row["nested_component_id"])
    expected_roles = {"train", "val", "test"}
    if set(role_samples) != expected_roles or any(not role_samples[role] for role in expected_roles):
        raise ProtocolError("Nested fold contains an empty role")
    role_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in role_pairs:
        if role_samples[left] & role_samples[right]:
            raise ProtocolError(f"Sample leakage between {left} and {right}")
        if role_regions[left] & role_regions[right]:
            raise ProtocolError(f"Spatial-supergroup leakage between {left} and {right}")
        if role_events[left] & role_events[right]:
            raise ProtocolError(f"Physical-event leakage between {left} and {right}")
        if role_components[left] & role_components[right]:
            raise ProtocolError(f"Component leakage between {left} and {right}")
    return {
        role: {
            "n_samples": len(role_samples[role]),
            "n_spatial_supergroups": len(role_regions[role]),
            "n_physical_events": len(role_events[role]),
            "n_components": len(role_components[role]),
            "sample_sha256": hash_values(role_samples[role]),
            "spatial_supergroup_sha256": hash_values(role_regions[role]),
            "physical_event_sha256": hash_values(role_events[role]),
            "component_sha256": hash_values(role_components[role]),
        }
        for role in ("train", "val", "test")
    }


def build_target_protocol(
    target: int,
    source_rows: Sequence[dict[str, str]],
    source_fieldnames: Sequence[str],
    event_by_sample: Mapping[str, str],
    outdir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    fold_rows = rows_for_outer_fold(source_rows, target)
    h5_ids = set(event_by_sample)
    target_rows = [row for row in fold_rows if row["role"] == "test"]
    if not target_rows:
        raise ProtocolError(f"Target outer fold {target} has no test rows")

    target_sample_ids = {row["sample_id"] for row in target_rows}
    target_h5_sample_ids = target_sample_ids & h5_ids
    target_spatial_supergroups = {row["spatial_supergroup"] for row in target_rows}
    target_region_groups = {row["region_group"] for row in target_rows}
    target_events = {event_by_sample[sample_id] for sample_id in target_h5_sample_ids}

    source_development = [
        row for row in fold_rows if row["role"] in ACTIVE_OUTER_DEVELOPMENT_ROLES
    ]
    protocol_only_development = [row for row in source_development if row["sample_id"] not in h5_ids]
    h5_development = [row for row in source_development if row["sample_id"] in h5_ids]
    excluded_target_linked = [
        row
        for row in h5_development
        if row["sample_id"] in target_sample_ids
        or row["spatial_supergroup"] in target_spatial_supergroups
        or row["region_group"] in target_region_groups
        or event_by_sample[row["sample_id"]] in target_events
    ]
    development = [
        row
        for row in h5_development
        if row["sample_id"] not in target_sample_ids
        and row["spatial_supergroup"] not in target_spatial_supergroups
        and row["region_group"] not in target_region_groups
        and event_by_sample[row["sample_id"]] not in target_events
    ]
    if not development:
        raise ProtocolError(f"Target outer fold {target} has no development samples after exclusion")

    development_ids = {row["sample_id"] for row in development}
    development_regions = {row["spatial_supergroup"] for row in development}
    development_region_groups = {row["region_group"] for row in development}
    development_events = {event_by_sample[row["sample_id"]] for row in development}
    zero_target_leakage = (
        not (development_ids & target_sample_ids)
        and not (development_regions & target_spatial_supergroups)
        and not (development_region_groups & target_region_groups)
        and not (development_events & target_events)
    )
    if not zero_target_leakage:
        raise ProtocolError(f"Target identities leak into development for outer fold {target}")

    components, sample_to_component = connected_components(development, event_by_sample)
    test_components = assign_inner_test_components(components)
    val_components = choose_validation_components(components, test_components)
    source_by_id = {row["sample_id"]: row for row in development}
    all_component_ids = {item["component_id"] for item in components}
    nested_rows: list[dict[str, str]] = []
    fold_audits = []
    inner_test_counts: Counter[str] = Counter()

    for inner_fold in range(N_INNER_FOLDS):
        test_ids = test_components[inner_fold]
        val_id = val_components[inner_fold]
        train_ids = all_component_ids - test_ids - {val_id}
        if not train_ids or not test_ids:
            raise ProtocolError(f"Empty train/test component set in inner fold {inner_fold}")
        fold_output = []
        for sample_id in sorted(development_ids):
            source = source_by_id[sample_id]
            component_id = sample_to_component[sample_id]
            if component_id in test_ids:
                role = "test"
                reason = "nested_oof_component_test"
                inner_test_counts[sample_id] += 1
            elif component_id == val_id:
                role = "val"
                reason = "nested_oof_single_small_component_validation"
            elif component_id in train_ids:
                role = "train"
                reason = "nested_oof_remaining_components_train"
            else:
                raise ProtocolError(f"Unassigned component {component_id}")
            row = {
                key: value
                for key, value in source.items()
                if key not in {"outer_fold", "role", "role_reason"}
            }
            row.update(
                {
                    "physical_event_id": event_by_sample[sample_id],
                    "target_outer_fold": str(target),
                    "source_outer_fold": str(target),
                    "source_outer_role": source["role"],
                    "source_role_reason": source.get("role_reason", ""),
                    "nested_component_id": component_id,
                    "outer_fold": str(inner_fold),
                    "role": role,
                    "role_reason": reason,
                }
            )
            fold_output.append(row)
        audit = role_audit(fold_output, event_by_sample)
        fold_audits.append(
            {
                "inner_fold": inner_fold,
                "test_component_ids": sorted(test_ids),
                "validation_component_id": val_id,
                "train_component_ids": sorted(train_ids),
                "roles": audit,
            }
        )
        nested_rows.extend(fold_output)

    if set(inner_test_counts) != development_ids or set(inner_test_counts.values()) != {1}:
        bad = sorted(
            sample_id for sample_id in development_ids if inner_test_counts[sample_id] != 1
        )
        raise ProtocolError(f"Development samples do not occur exactly once as inner test: {bad[:10]}")

    fields = output_fieldnames(source_fieldnames)
    output_path = outdir / f"sen12_nested_oof_target_outer{target}_v1.csv"
    atomic_write_text(output_path, csv_text(fields, nested_rows))
    allocation_records = sorted(
        [
            [
                int(row["outer_fold"]),
                row["sample_id"],
                row["role"],
                row["nested_component_id"],
            ]
            for row in nested_rows
        ]
    )
    target_manifest = {
        "target_outer_fold": target,
        "output_csv": str(output_path.resolve()),
        "output_csv_sha256": sha256_file(output_path),
        "allocation_sha256": hash_payload(allocation_records),
        "target_outer_test": {
            "n_source_samples": len(target_sample_ids),
            "n_h5_samples": len(target_h5_sample_ids),
            "n_h5_physical_events": len(target_events),
            "n_spatial_supergroups": len(target_spatial_supergroups),
            "n_region_groups": len(target_region_groups),
            "sample_sha256": hash_values(target_sample_ids),
            "h5_sample_sha256": hash_values(target_h5_sample_ids),
            "physical_event_sha256": hash_values(target_events),
            "spatial_supergroup_sha256": hash_values(target_spatial_supergroups),
            "region_group_sha256": hash_values(target_region_groups),
            "source_samples_missing_from_h5": sorted(target_sample_ids - h5_ids),
        },
        "outer_development": {
            "n_source_train_val_samples": len(source_development),
            "n_source_samples_missing_from_h5": len(protocol_only_development),
            "source_samples_missing_from_h5": sorted(
                row["sample_id"] for row in protocol_only_development
            ),
            "n_target_linked_samples_excluded": len(excluded_target_linked),
            "target_linked_sample_sha256": hash_values(
                row["sample_id"] for row in excluded_target_linked
            ),
            "n_samples": len(development_ids),
            "n_physical_events": len(development_events),
            "n_spatial_supergroups": len(development_regions),
            "n_components": len(components),
            "sample_sha256": hash_values(development_ids),
            "physical_event_sha256": hash_values(development_events),
            "spatial_supergroup_sha256": hash_values(development_regions),
            "component_sha256": hash_values(item["component_id"] for item in components),
        },
        "components": components,
        "inner_folds": fold_audits,
        "audit": {
            "zero_target_sample_region_event_leakage": zero_target_leakage,
            "each_development_sample_exactly_once_inner_test": True,
            "inner_test_count_min": min(inner_test_counts.values()),
            "inner_test_count_max": max(inner_test_counts.values()),
            "allocation_uses_only_component_sample_counts": True,
            "label_columns_used_for_assignment": [],
        },
    }
    return target_manifest, nested_rows


def build_protocol(
    split_csv: Path,
    h5_path: Path,
    outdir: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    split_csv = split_csv.resolve()
    h5_path = h5_path.resolve()
    outdir = outdir.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else outdir / "sen12_nested_oof_protocol_v1_manifest.json"
    )
    source_fieldnames, source_rows = read_split_rows(split_csv)
    event_by_sample = read_h5_event_map(h5_path)
    csv_sample_ids = {row["sample_id"] for row in source_rows}
    h5_sample_ids = set(event_by_sample)
    source_outer_folds = sorted({row["outer_fold"] for row in source_rows})
    expected_outer_folds = [str(index) for index in range(N_OUTER_FOLDS)]
    if source_outer_folds != expected_outer_folds:
        raise ProtocolError(
            f"Expected source outer folds {expected_outer_folds}, found {source_outer_folds}"
        )
    if h5_sample_ids - csv_sample_ids:
        raise ProtocolError(
            f"H5 samples absent from source CSV: {sorted(h5_sample_ids - csv_sample_ids)[:10]}"
        )

    targets = []
    for target in range(N_OUTER_FOLDS):
        target_manifest, _ = build_target_protocol(
            target, source_rows, source_fieldnames, event_by_sample, outdir
        )
        targets.append(target_manifest)

    manifest: dict[str, Any] = {
        "schema_version": "sen12-nested-oof-protocol-v1",
        "contract": {
            "target_outer_test_is_exclusion_only": True,
            "outer_development_source_roles": sorted(ACTIVE_OUTER_DEVELOPMENT_ROLES),
            "component_identity": "connected components of spatial_supergroup and physical_event_id",
            "inner_test_assignment": "deterministic LPT using component sample count only",
            "inner_validation_assignment": "one smallest deterministic non-test component",
            "n_target_outer_folds": N_OUTER_FOLDS,
            "n_inner_folds": N_INNER_FOLDS,
            "label_columns_used_for_assignment": [],
        },
        "inputs": {
            "split_csv": str(split_csv),
            "split_csv_sha256": sha256_file(split_csv),
            "h5_path": str(h5_path),
            "h5_sha256": sha256_file(h5_path),
            "n_split_rows": len(source_rows),
            "n_split_unique_samples": len(csv_sample_ids),
            "n_h5_samples": len(h5_sample_ids),
            "split_samples_missing_from_h5": sorted(csv_sample_ids - h5_sample_ids),
            "h5_samples_missing_from_split": sorted(h5_sample_ids - csv_sample_ids),
            "split_sample_sha256": hash_values(csv_sample_ids),
            "h5_sample_sha256": hash_values(h5_sample_ids),
            "h5_physical_event_sha256": hash_values(event_by_sample.values()),
        },
        "targets": targets,
    }
    manifest["all_targets_all_audits_pass"] = all(
        target["audit"]["zero_target_sample_region_event_leakage"]
        and target["audit"]["each_development_sample_exactly_once_inner_test"]
        for target in targets
    )
    manifest["allocation_sha256"] = hash_payload(
        [target["allocation_sha256"] for target in targets]
    )
    manifest["manifest_payload_sha256"] = hash_payload(manifest)
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=repo / "metadata/pild_xdomain_v1/sen12_s2_logo5_v1.csv",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=repo / "processed/hybrid_pinn/sen12_s2_xdomain_v1/sen12_s2_tmr_p128.h5",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo / "metadata/pild_xdomain_v1/sen12_nested_oof_protocol_v1",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_protocol(args.split_csv, args.h5, args.outdir, args.manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "targets": len(manifest["targets"]),
                "allocation_sha256": manifest["allocation_sha256"],
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
