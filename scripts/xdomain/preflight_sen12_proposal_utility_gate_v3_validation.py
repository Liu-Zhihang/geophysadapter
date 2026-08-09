#!/usr/bin/env python3
"""Select Sen12 v3 utility gates using formal nested OOF evidence only.

This validation preflight never imports or loads a target outer-test cache.  It
stops after proposal/context meta-CV selection and label-shuffle sanity checks.
Its only admissible inputs are the formal nested inner-holdout manifests emitted
by ``build_sen12_proposal_utility_gate_v3_manifests.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import train_sen12_proposal_utility_gate_v3 as gate


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_FORMAL_ROOT = (
    PROJECT_ROOT / "experiments/revision2026/sen12_proposal_utility_gate_v3/formal_inputs_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments/revision2026/sen12_proposal_utility_gate_v3/validation_preflight_v1"
)
TARGETS = tuple(range(5))
CONTEXTS = ("TM", "TR", "TMR")
SCHEMA_VERSION = "sen12_proposal_utility_gate_validation_preflight.v1"
DONE_SCHEMA = "sen12_proposal_utility_gate_validation_preflight_done.v1"
MIN_PASS_TARGETS = 3


class PreflightError(RuntimeError):
    """A formal input or validation invariant failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        gate.json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, name: str) -> Any:
    if not path.is_file():
        raise PreflightError(f"missing {name}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid {name}: {path}: {exc}") from exc


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(gate.json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _safe_member(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise PreflightError(f"absolute path forbidden in formal hashes: {relative}")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise PreflightError(f"path escapes formal input root: {relative}")
    return path


def validate_formal_root(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise PreflightError(f"formal input root missing: {root}")
    done_path = root / "DONE.json"
    hashes_path = root / "hashes.json"
    summary_path = root / "aggregate_summary.json"
    done = read_json(done_path, "formal DONE")
    hashes = read_json(hashes_path, "formal hashes")
    summary = read_json(summary_path, "formal aggregate summary")
    if done.get("status") != "complete":
        raise PreflightError("formal aggregate is not complete")
    if int(done.get("expected_tasks", -1)) != 15 or int(done.get("validated_tasks", -1)) != 15:
        raise PreflightError("formal aggregate does not prove 15/15 tasks")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise PreflightError("formal hashes.json hash mismatch")
    if done.get("aggregate_summary_sha256") != sha256_file(summary_path):
        raise PreflightError("formal aggregate_summary.json hash mismatch")
    if not isinstance(hashes, dict) or not hashes:
        raise PreflightError("formal hashes.json is empty or malformed")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"DONE.json", "hashes.json"}
    }
    if actual != set(hashes):
        raise PreflightError(
            f"formal artifact inventory mismatch: missing={sorted(set(hashes)-actual)[:10]} "
            f"extra={sorted(actual-set(hashes))[:10]}"
        )
    latest_dependency = max(hashes_path.stat().st_mtime_ns, summary_path.stat().st_mtime_ns)
    for relative, expected in hashes.items():
        path = _safe_member(root, str(relative))
        if not path.is_file() or sha256_file(path) != expected:
            raise PreflightError(f"formal artifact hash mismatch: {relative}")
        latest_dependency = max(latest_dependency, path.stat().st_mtime_ns)
    if done_path.stat().st_mtime_ns < latest_dependency:
        raise PreflightError("formal DONE is stale")
    if summary.get("status") != "complete" or int(summary.get("validated_tasks", -1)) != 15:
        raise PreflightError("formal aggregate summary is not complete")
    payload_hash = summary.get("summary_payload_sha256")
    payload = {key: value for key, value in summary.items() if key != "summary_payload_sha256"}
    if payload_hash != canonical_hash(payload):
        raise PreflightError("formal aggregate summary canonical hash mismatch")
    targets = {int(item.get("target_outer_fold", -1)): item for item in summary.get("targets", [])}
    if set(targets) != set(TARGETS):
        raise PreflightError("formal aggregate must contain exactly target folds 0..4")
    for target in TARGETS:
        target_root = root / f"target_outer{target}"
        expected_manifest = target_root / "oof_manifest.json"
        expected_split = target_root / "gate_split.csv"
        item = targets[target]
        if Path(str(item.get("oof_manifest", ""))).resolve() != expected_manifest.resolve():
            raise PreflightError(f"target {target} manifest path identity mismatch")
        if Path(str(item.get("gate_split", ""))).resolve() != expected_split.resolve():
            raise PreflightError(f"target {target} split path identity mismatch")
        if item.get("oof_manifest_sha256") != sha256_file(expected_manifest):
            raise PreflightError(f"target {target} manifest summary hash mismatch")
        if item.get("gate_split_sha256") != sha256_file(expected_split):
            raise PreflightError(f"target {target} split summary hash mismatch")
        if int(item.get("n_inner_folds", -1)) != 3:
            raise PreflightError(f"target {target} does not contain three inner folds")
    return summary


def metric_view(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "iou": float(value["iou"]),
        "errors": int(value["errors"]),
        "proposal_count": int(value.get("proposal_count", 0)),
        "proposal_accepted": int(value.get("proposal_accepted", 0)),
        "proposal_rejected": int(value.get("proposal_rejected", 0)),
    }


def selection_view(selection: gate.Selection, *, label_shuffle: bool = False) -> dict[str, Any]:
    proposal = selection.controls.get("proposal_only", selection.meta_metrics)
    controls = {
        name: metric_view(value)
        for name, value in selection.controls.items()
        if name not in {"aligned", "proposal_only"}
    }
    return {
        "aligned": metric_view(selection.controls.get("aligned", selection.meta_metrics)),
        "proposal_only": metric_view(proposal),
        "controls": controls,
        "claim_pass": bool(selection.claim_pass),
        "label_shuffle_claim_pass": bool(label_shuffle),
        "fallback": selection.fallback,
        "alpha": float(selection.alpha),
        "rescue_threshold": float(selection.rescue_threshold),
        "veto_threshold": float(selection.veto_threshold),
    }


def select_target(
    formal_root: Path,
    target: int,
    seed: int,
    alphas: Sequence[float],
    threshold_grid: Sequence[float],
) -> dict[str, Any]:
    target_root = formal_root / f"target_outer{target}"
    access_log: list[dict[str, Any]] = []
    bundles, provenance = gate.load_formal_nested_bundles(
        target_root / "oof_manifest.json",
        target_fold=target,
        split_csv=target_root / "gate_split.csv",
        seed=seed,
        access_log=access_log,
    )
    if len(bundles) != 3 or len(access_log) != 3:
        raise PreflightError(f"target {target} did not load exactly three nested OOF bundles")
    if any(item.get("identity_role") != "nested_inner_holdout_oof" for item in access_log):
        raise PreflightError(f"target {target} accessed a non-nested OOF identity")
    tables = {bundle.fold: gate.build_proposal_table(bundle) for bundle in bundles}
    proposal_selection, proposal_decisions = gate.select_proposal_only(
        bundles, tables, alphas, threshold_grid, seed
    )
    contexts: dict[str, Any] = {}
    for context in CONTEXTS:
        selection = gate.select_context(
            context,
            bundles,
            tables,
            alphas,
            threshold_grid,
            seed,
            proposal_selection,
            proposal_decisions,
        )
        raw_claim_pass = bool(selection.claim_pass)
        shuffled_pass = gate.label_shuffle_sanity(
            context,
            selection,
            bundles,
            tables,
            seed,
            proposal_selection,
            proposal_decisions,
        )
        if shuffled_pass:
            selection.claim_pass = False
            selection.fallback = "proposal_only; label-shuffle sanity failed"
        view = selection_view(selection, label_shuffle=shuffled_pass)
        view["raw_claim_pass_before_label_shuffle"] = raw_claim_pass
        contexts[context] = view
    proposal_view = selection_view(proposal_selection)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "selection_complete",
        "target_outer_fold": target,
        "seed": seed,
        "evidence_role": "target_outer_train_nested_inner_holdout_meta_cv_only",
        "target_outer_cache_loaded": False,
        "target_outer_labels_loaded": False,
        "proposal_only": proposal_view,
        "contexts": contexts,
        "nested_loader_audit_sha256": canonical_hash(provenance),
        "nested_access_log": access_log,
    }


def aggregate_eligibility(target_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if {int(item["target_outer_fold"]) for item in target_results} != set(TARGETS):
        raise PreflightError("eligibility requires exactly five unique targets")
    context_status: dict[str, Any] = {}
    for context in CONTEXTS:
        passed = [
            int(item["target_outer_fold"])
            for item in target_results
            if item["contexts"][context]["claim_pass"]
            and not item["contexts"][context]["label_shuffle_claim_pass"]
        ]
        shuffled = [
            int(item["target_outer_fold"])
            for item in target_results
            if item["contexts"][context]["label_shuffle_claim_pass"]
        ]
        eligible = len(passed) >= MIN_PASS_TARGETS and not shuffled
        context_status[context] = {
            "n_targets": len(TARGETS),
            "n_claim_pass": len(passed),
            "claim_pass_targets": passed,
            "n_label_shuffle_claim_pass": len(shuffled),
            "label_shuffle_claim_pass_targets": shuffled,
            "minimum_required": MIN_PASS_TARGETS,
            "eligible_for_outer_test": eligible,
        }
    return {
        "contexts": context_status,
        "eligible_for_outer_test": all(
            value["eligible_for_outer_test"] for value in context_status.values()
        ),
        "decision_rule": (
            "TM, TR, and TMR must each pass >=3/5 target protocols, and each context "
            "must have zero label-shuffle claim passes across all five targets"
        ),
    }


def running_role_gate(target_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the live lower/upper pass bounds after each completed target."""
    evaluated = len(target_results)
    remaining = len(TARGETS) - evaluated
    contexts: dict[str, Any] = {}
    for context in CONTEXTS:
        passed = sum(
            bool(item["contexts"][context]["claim_pass"])
            and not bool(item["contexts"][context]["label_shuffle_claim_pass"])
            for item in target_results
        )
        shuffled = sum(
            bool(item["contexts"][context]["label_shuffle_claim_pass"])
            for item in target_results
        )
        maximum_possible = passed + remaining
        no_go = shuffled > 0 or maximum_possible < MIN_PASS_TARGETS
        if no_go:
            state = "no_go"
        elif passed >= MIN_PASS_TARGETS:
            state = "minimum_reached_pending_remaining_label_shuffle"
        else:
            state = "pending"
        contexts[context] = {
            "n_evaluated": evaluated,
            "n_remaining": remaining,
            "n_claim_pass": passed,
            "maximum_possible_claim_pass": maximum_possible,
            "n_label_shuffle_claim_pass": shuffled,
            "decision_state": state,
            "no_go": no_go,
        }
    return {
        "contexts": contexts,
        "overall_no_go": any(item["no_go"] for item in contexts.values()),
        "outer_test_started": False,
    }


def csv_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        target = int(result["target_outer_fold"])
        for context in ("proposal_only", *CONTEXTS):
            item = result["proposal_only"] if context == "proposal_only" else result["contexts"][context]
            rows.append({
                "target_outer_fold": target,
                "context": context,
                "aligned_iou": item["aligned"]["iou"],
                "aligned_errors": item["aligned"]["errors"],
                "proposal_only_iou": item["proposal_only"]["iou"],
                "proposal_only_errors": item["proposal_only"]["errors"],
                "claim_pass": item["claim_pass"],
                "label_shuffle_claim_pass": item["label_shuffle_claim_pass"],
                "fallback": item["fallback"],
                "alpha": item["alpha"],
                "rescue_threshold": item["rescue_threshold"],
                "veto_threshold": item["veto_threshold"],
                "controls_json": json.dumps(item["controls"], sort_keys=True, allow_nan=False),
            })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sen12 proposal utility gate validation preflight",
        "",
        "No target outer-test cache or label was loaded. Eligibility is based only on formal nested OOF selection.",
        "",
        "| Context | Passed targets | Label-shuffle passes | Eligible for outer test |",
        "|---|---:|---:|---:|",
    ]
    for context in CONTEXTS:
        item = summary["eligibility"]["contexts"][context]
        lines.append(
            f"| {context} | {item['n_claim_pass']}/5 | "
            f"{item['n_label_shuffle_claim_pass']}/5 | {str(item['eligible_for_outer_test']).lower()} |"
        )
    lines.extend([
        "",
        f"**Overall eligible for outer test:** {str(summary['eligible_for_outer_test']).lower()}",
        "",
        f"Decision rule: {summary['eligibility']['decision_rule']}",
        "",
    ])
    return "\n".join(lines)


def collect_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "DONE.json"
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = args.formal_input_root.resolve()
    formal_summary = validate_formal_root(formal_root)
    if args.dry_run:
        audits = []
        for target in TARGETS:
            target_root = formal_root / f"target_outer{target}"
            access_log: list[dict[str, Any]] = []
            bundles, audit = gate.load_formal_nested_bundles(
                target_root / "oof_manifest.json",
                target_fold=target,
                split_csv=target_root / "gate_split.csv",
                seed=args.seed,
                access_log=access_log,
            )
            if len(bundles) != 3 or len(access_log) != 3:
                raise PreflightError(f"target {target} nested bundle dry-run validation failed")
            audits.append(canonical_hash(audit))
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run_validated",
            "targets": list(TARGETS),
            "nested_loader_audit_sha256": audits,
            "target_outer_cache_loaded": False,
            "output_written": False,
        }
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise PreflightError(f"refusing existing output root: {output_root}")
    progress = output_root.with_name(f"{output_root.name}.progress.jsonl")
    stage = output_root.with_name(f".{output_root.name}.stage-{os.getpid()}")
    if progress.exists() or stage.exists():
        raise PreflightError("stale progress or staging output exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    try:
        with progress.open("x", encoding="utf-8") as handle:
            for target in TARGETS:
                result = select_target(
                    formal_root, target, args.seed, args.alphas, args.threshold_grid
                )
                results.append(result)
                result["running_role_gate"] = running_role_gate(results)
                line = json.dumps(gate.json_safe(result), sort_keys=True, allow_nan=False)
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(line, flush=True)
        eligibility = aggregate_eligibility(results)
        no_go_targets = [
            int(item["target_outer_fold"])
            for item in results
            if item["running_role_gate"]["overall_no_go"]
        ]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "seed": args.seed,
            "formal_input_root": str(formal_root),
            "formal_aggregate_summary_sha256": canonical_hash(formal_summary),
            "target_outer_cache_loaded": False,
            "target_outer_labels_loaded": False,
            "n_targets": len(results),
            "eligibility": eligibility,
            "eligible_for_outer_test": eligibility["eligible_for_outer_test"],
            "no_go": not eligibility["eligible_for_outer_test"],
            "no_go_first_detected_after_target": no_go_targets[0] if no_go_targets else None,
            "targets": results,
        }
        summary["summary_payload_sha256"] = canonical_hash(summary)
        stage.mkdir(parents=True)
        os.replace(progress, stage / "validation_preflight_targets.jsonl")
        atomic_json(stage / "validation_preflight_summary.json", summary)
        atomic_text(stage / "validation_preflight_summary.md", render_markdown(summary))
        write_csv(stage / "validation_preflight_summary.csv", csv_rows(results))
        hashes = collect_hashes(stage)
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "complete",
            "n_targets": len(results),
            "eligible_for_outer_test": eligibility["eligible_for_outer_test"],
            "artifact_sha256": hashes,
        }
        atomic_json(stage / "DONE.json", done)
        os.replace(stage, output_root)
        return {**summary, "output_written": True, "output_root": str(output_root)}
    except Exception:
        progress.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(not math.isfinite(item) or item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated finite values")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-input-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260751)
    parser.add_argument("--alphas", type=parse_float_list, default=(1e-5, 1e-4, 1e-3))
    parser.add_argument("--threshold-grid", type=parse_float_list, default=(0.35, 0.50, 0.65))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if any(not 0.0 <= item <= 1.0 for item in args.threshold_grid):
        parser.error("threshold grid values must be in [0,1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_preflight(args)
    except Exception as exc:
        print(json.dumps({
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "target_outer_cache_loaded": False,
        }, sort_keys=True), flush=True)
        return 1
    if args.dry_run:
        print(json.dumps(gate.json_safe(result), sort_keys=True, allow_nan=False), flush=True)
    else:
        print(json.dumps({
            "status": "complete",
            "eligible_for_outer_test": result["eligible_for_outer_test"],
            "output_root": result["output_root"],
            "outer_test_started": False,
        }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
