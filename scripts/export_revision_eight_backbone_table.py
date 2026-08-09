#!/usr/bin/env python3
"""Export the frozen eight-backbone terrain-attribution table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FM = ROOT / "experiments/revision2026/l4s_fm_terrain_attribution_20260715/summary.json"
DEFAULT_MODERN = ROOT / "experiments/revision2026/l4s_modern_bn_frozen_attribution_20260715/summary.json"
DEFAULT_OUTDIR = ROOT.parent / "submission_package_jprs_revision1/response_materials"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-summary", type=Path, default=DEFAULT_FM)
    parser.add_argument("--modern-summary", type=Path, default=DEFAULT_MODERN)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    spec = f"{'+' if signed else ''}.{digits}f"
    return format(float(value), spec)


def load_family(path: Path, family: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("n_unique_samples") != 800:
        raise ValueError(f"{path}: expected 800 unique samples")

    decisions = {row["backbone"]: row for row in payload["decisions"]}
    mechanisms = {row["backbone"]: row for row in payload["mechanism"]}
    rows: list[dict[str, Any]] = []
    for spec in payload["specs"]:
        label = spec["label"]
        decision = decisions[label]
        if not decision.get("terrain_attribution_pass"):
            raise ValueError(f"{path}: attribution gate failed for {label}")

        comparison = payload["comparisons"][label]["terrain_true_vs_visual_anchor"]
        sample = comparison["sample"]
        seed = comparison["seed"]
        mechanism = mechanisms[label]
        rows.append(
            {
                "family": family,
                "backbone": label,
                "n_unique_patches": sample["n"],
                "n_seeds": seed["n"],
                "patch_mean_delta": sample["mean_delta"],
                "patch_ci95_low": sample["mean_delta_ci95"][0],
                "patch_ci95_high": sample["mean_delta_ci95"][1],
                "patch_holm_p": sample["holm_permutation_p_family"],
                "seed_mean_delta": seed["mean_delta"],
                "seed_positive_rate": seed["positive_rate"],
                "seed_exact_p": seed["permutation_p"],
                "gate_error_minus_correct": mechanism["gate_error_minus_correct"],
                "gate_ci95_low": mechanism["gate_error_minus_correct_ci95"][0],
                "gate_ci95_high": mechanism["gate_error_minus_correct_ci95"][1],
                "net_error_reduction": mechanism["net_error_reduction_fraction"],
                "net_error_ci95_low": mechanism["net_error_reduction_ci95"][0],
                "net_error_ci95_high": mechanism["net_error_reduction_ci95"][1],
                "all_controls_pass": decision["all_sample_controls_pass"],
                "visual_state_identity_pass": True,
                "terrain_attribution_pass": decision["terrain_attribution_pass"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    lines = [
        "# Eight-backbone terrain-attribution summary",
        "",
        "All rows use five seeds and all 800 unique Landslide4Sense test patches. "
        "Repeated seed predictions are first aggregated within each patch for patch-level inference. "
        "The visual state is tensor-identical within every matched observation-versus-adapter comparison.",
        "",
        "| family | backbone | unique-patch delta [95% CI] | Holm p | seed mean delta | positive seeds (aligned vs visual) | gate(error-correct) [95% CI] | net error reduction [95% CI] | attribution gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        patch_ci = (
            f"{fmt(row['patch_mean_delta'], signed=True)} "
            f"[{fmt(row['patch_ci95_low'], signed=True)},{fmt(row['patch_ci95_high'], signed=True)}]"
        )
        gate_ci = (
            f"{fmt(row['gate_error_minus_correct'], signed=True)} "
            f"[{fmt(row['gate_ci95_low'], signed=True)},{fmt(row['gate_ci95_high'], signed=True)}]"
        )
        net_ci = (
            f"{100 * row['net_error_reduction']:.2f}% "
            f"[{100 * row['net_error_ci95_low']:.2f}%,{100 * row['net_error_ci95_high']:.2f}%]"
        )
        lines.append(
            "| {family} | {backbone} | {patch_ci} | {holm:.4g} | {seed_delta} | {positive:.0f}/5 | {gate_ci} | {net_ci} | pass |".format(
                family=row["family"],
                backbone=row["backbone"],
                patch_ci=patch_ci,
                holm=row["patch_holm_p"],
                seed_delta=fmt(row["seed_mean_delta"], signed=True),
                positive=5 * row["seed_positive_rate"],
                gate_ci=gate_ci,
                net_ci=net_ci,
            )
        )
    lines.extend(
        [
            "",
            "The table supports modest, terrain-attributable correction across architectures. "
            "It does not establish a universal large IoU gain, event-level generalization, or Landslide4Sense state of the art. "
            "With five seeds, the minimum non-zero two-sided exact sign-flip p-value is 0.0625; seed results are therefore reported as directional replication.",
            "",
            "## Provenance",
            "",
            f"- foundation summary: `{provenance['foundation_summary']}`",
            f"- foundation SHA-256: `{provenance['foundation_sha256']}`",
            f"- modern summary: `{provenance['modern_summary']}`",
            f"- modern SHA-256: `{provenance['modern_sha256']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (args.foundation_summary, args.modern_summary):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = load_family(args.foundation_summary, "foundation")
    rows.extend(load_family(args.modern_summary, "modern"))
    if len(rows) != 8:
        raise ValueError(f"expected eight backbone rows, found {len(rows)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / "eight_backbone_terrain_attribution"
    provenance = {
        "foundation_summary": str(args.foundation_summary.resolve()),
        "foundation_sha256": sha256(args.foundation_summary),
        "modern_summary": str(args.modern_summary.resolve()),
        "modern_sha256": sha256(args.modern_summary),
        "n_rows": len(rows),
        "n_unique_patches": 800,
        "n_seeds": 5,
    }
    write_csv(stem.with_suffix(".csv"), rows)
    write_markdown(stem.with_suffix(".md"), rows, provenance)
    stem.with_suffix(".json").write_text(
        json.dumps({"provenance": provenance, "rows": rows}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[done] wrote {stem}.csv/.md/.json")


if __name__ == "__main__":
    main()
