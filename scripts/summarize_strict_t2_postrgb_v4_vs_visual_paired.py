#!/usr/bin/env python3
"""Summarize paired strict_t2 post_rgb v4 versus DeepLabV3 visual evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS = [
    "sample_id",
    "event_uid",
    "dataset_id",
    "tp",
    "fp",
    "fn",
    "stability_proxy",
]


def per_sample_iou(df: pd.DataFrame) -> pd.DataFrame:
    out = df[KEY_COLS].copy()
    denom = out["tp"] + out["fp"] + out["fn"]
    out["iou"] = np.where(denom > 0, out["tp"] / denom, 0.0)
    return out[["sample_id", "event_uid", "dataset_id", "stability_proxy", "iou"]]


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(repeats, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def choose_case_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []

    stable_v4 = df[df["v4_wins"] >= 2].sort_values(["mean_delta", "v4_iou_mean"], ascending=[False, False])
    stable_visual = df[df["visual_wins"] >= 2].sort_values(["mean_delta", "visual_iou_mean"], ascending=[True, False])
    consensus_hard = df.sort_values(["max_iou", "abs_mean_delta"], ascending=[True, True])

    dlr_pos = stable_v4[stable_v4["dataset_id"] == "DLR_Landslide_Ref_2025"].head(1)
    cas_pos = stable_v4[stable_v4["dataset_id"] == "CAS_Landslide"].head(1)
    if not dlr_pos.empty:
        row = dlr_pos.iloc[0].copy()
        row["panel_tag"] = "Panel A"
        row["panel_note"] = "stable DLR gain for v4 over matched visual"
        rows.append(row)
    if not cas_pos.empty:
        row = cas_pos.iloc[0].copy()
        row["panel_tag"] = "Panel B"
        row["panel_note"] = "stable CAS gain for v4 over matched visual"
        rows.append(row)

    for _, src_row in stable_visual.iterrows():
        sample_id = str(src_row["sample_id"])
        if sample_id in {str(r["sample_id"]) for r in rows}:
            continue
        row = src_row.copy()
        row["panel_tag"] = "Panel C"
        row["panel_note"] = "matched visual still wins on a visually favorable patch"
        rows.append(row)
        break

    for _, src_row in consensus_hard.iterrows():
        sample_id = str(src_row["sample_id"])
        if sample_id in {str(r["sample_id"]) for r in rows}:
            continue
        row = src_row.copy()
        row["panel_tag"] = "Panel D"
        row["panel_note"] = "hard case where both same-backbone variants remain weak"
        rows.append(row)
        break

    if len(rows) < 4:
        for _, src_row in stable_v4.iterrows():
            sample_id = str(src_row["sample_id"])
            if sample_id in {str(r["sample_id"]) for r in rows}:
                continue
            row = src_row.copy()
            row["panel_tag"] = f"Panel {chr(ord('A') + len(rows))}"
            row["panel_note"] = "additional v4-positive case"
            rows.append(row)
            if len(rows) == 4:
                break

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize paired v4-vs-visual strict_t2 post_rgb evidence")
    p.add_argument("--visual-csv", action="append", required=True, help="per-sample physical-consistency CSV for DeepLabV3 visual")
    p.add_argument("--v4-csv", action="append", required=True, help="per-sample physical-consistency CSV for v4")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap-repeats", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260311)
    p.add_argument("--tie-eps", type=float, default=0.001)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.visual_csv) != len(args.v4_csv):
        raise SystemExit("the number of --visual-csv and --v4-csv inputs must match")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    sample_frames = []
    dataset_frames = []

    for idx, (visual_csv, v4_csv) in enumerate(zip(args.visual_csv, args.v4_csv), start=1):
        visual_path = Path(visual_csv)
        v4_path = Path(v4_csv)
        visual = per_sample_iou(pd.read_csv(visual_path)).rename(columns={"iou": "visual_iou", "stability_proxy": "visual_stability_proxy"})
        v4 = per_sample_iou(pd.read_csv(v4_path)).rename(columns={"iou": "v4_iou", "stability_proxy": "v4_stability_proxy"})
        merged = visual.merge(v4, on=["sample_id", "event_uid", "dataset_id"], how="inner")
        merged["seed_pair"] = idx
        merged["delta_iou"] = merged["v4_iou"] - merged["visual_iou"]
        merged["stability_proxy"] = merged["v4_stability_proxy"]
        per_seed_rows.append(
            {
                "seed_pair": idx,
                "visual_csv": str(visual_path),
                "v4_csv": str(v4_path),
                "n_samples": int(len(merged)),
                "visual_iou_mean": float(merged["visual_iou"].mean()),
                "v4_iou_mean": float(merged["v4_iou"].mean()),
                "delta_iou_mean": float(merged["delta_iou"].mean()),
                "v4_win_rate": float((merged["delta_iou"] > args.tie_eps).mean()),
            }
        )
        sample_frames.append(merged[["sample_id", "event_uid", "dataset_id", "stability_proxy", "seed_pair", "visual_iou", "v4_iou", "delta_iou"]])
        by_dataset = merged.groupby("dataset_id", as_index=False).agg(
            visual_iou_mean=("visual_iou", "mean"),
            v4_iou_mean=("v4_iou", "mean"),
            delta_iou_mean=("delta_iou", "mean"),
        )
        by_dataset["seed_pair"] = idx
        dataset_frames.append(by_dataset)

    paired = pd.concat(sample_frames, ignore_index=True)
    per_seed = pd.DataFrame(per_seed_rows)
    per_dataset_seed = pd.concat(dataset_frames, ignore_index=True)

    sample_mean = paired.groupby(["sample_id", "event_uid", "dataset_id"], as_index=False).agg(
        stability_proxy=("stability_proxy", "mean"),
        visual_iou_mean=("visual_iou", "mean"),
        v4_iou_mean=("v4_iou", "mean"),
        mean_delta=("delta_iou", "mean"),
        std_delta=("delta_iou", "std"),
        v4_wins=("delta_iou", lambda s: int((s > args.tie_eps).sum())),
        visual_wins=("delta_iou", lambda s: int((s < -args.tie_eps).sum())),
        ties=("delta_iou", lambda s: int((s.abs() <= args.tie_eps).sum())),
    )
    sample_mean["std_delta"] = sample_mean["std_delta"].fillna(0.0)
    sample_mean["abs_mean_delta"] = sample_mean["mean_delta"].abs()
    sample_mean["max_iou"] = sample_mean[["visual_iou_mean", "v4_iou_mean"]].max(axis=1)

    mean_delta = float(sample_mean["mean_delta"].mean())
    ci_low, ci_high = bootstrap_ci(sample_mean["mean_delta"].to_numpy(dtype=np.float64), seed=args.seed, repeats=args.bootstrap_repeats)
    win_count = int((sample_mean["mean_delta"] > args.tie_eps).sum())
    loss_count = int((sample_mean["mean_delta"] < -args.tie_eps).sum())
    tie_count = int((sample_mean["mean_delta"].abs() <= args.tie_eps).sum())

    dataset_mean = sample_mean.groupby("dataset_id", as_index=False).agg(
        n_samples=("sample_id", "count"),
        visual_iou_mean=("visual_iou_mean", "mean"),
        v4_iou_mean=("v4_iou_mean", "mean"),
        mean_delta=("mean_delta", "mean"),
        v4_win_rate=("mean_delta", lambda s: float((s > args.tie_eps).mean())),
    ).sort_values("mean_delta", ascending=False)

    sample_mean_path = out_dir / "paired_sample_mean_diff.csv"
    per_seed_path = out_dir / "paired_seed_summary.csv"
    per_dataset_seed_path = out_dir / "paired_dataset_seed_summary.csv"
    shortlist_path = out_dir / "paired_case_shortlist.csv"
    report_path = out_dir / "report.md"
    json_path = out_dir / "summary.json"

    sample_mean.sort_values("mean_delta", ascending=False).to_csv(sample_mean_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    per_dataset_seed.to_csv(per_dataset_seed_path, index=False)
    shortlist = choose_case_rows(sample_mean)
    shortlist.to_csv(shortlist_path, index=False)

    summary = {
        "n_samples": int(len(sample_mean)),
        "n_seed_pairs": int(len(per_seed)),
        "visual_mean_iou": float(sample_mean["visual_iou_mean"].mean()),
        "v4_mean_iou": float(sample_mean["v4_iou_mean"].mean()),
        "mean_delta_iou": mean_delta,
        "bootstrap_ci95": [ci_low, ci_high],
        "tie_eps": float(args.tie_eps),
        "v4_wins": win_count,
        "visual_wins": loss_count,
        "ties": tie_count,
        "sample_mean_csv": str(sample_mean_path),
        "seed_summary_csv": str(per_seed_path),
        "dataset_seed_csv": str(per_dataset_seed_path),
        "shortlist_csv": str(shortlist_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# strict_t2 post_rgb paired v4 vs DeepLabV3 visual robustness v1",
        "",
        "## Overall",
        "",
        f"- `n_samples = {len(sample_mean)}`",
        f"- `n_seed_pairs = {len(per_seed)}`",
        f"- `DeepLabV3 visual mean IoU = {sample_mean['visual_iou_mean'].mean():.6f}`",
        f"- `DeepLabV3 v4 mean IoU = {sample_mean['v4_iou_mean'].mean():.6f}`",
        f"- `mean delta IoU (v4 - visual) = {mean_delta:.6f}`",
        f"- `bootstrap 95% CI = [{ci_low:.6f}, {ci_high:.6f}]`",
        f"- `sample-level wins / ties / losses = {win_count} / {tie_count} / {loss_count}`",
        "",
        "## Seed Pairs",
        "",
        "| seed_pair | visual_iou_mean | v4_iou_mean | delta_iou_mean | v4_win_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in per_seed.itertuples(index=False):
        lines.append(
            f"| {row.seed_pair} | {row.visual_iou_mean:.6f} | {row.v4_iou_mean:.6f} | {row.delta_iou_mean:.6f} | {row.v4_win_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "## By Dataset",
            "",
            "| dataset | n_samples | visual_iou_mean | v4_iou_mean | mean_delta | v4_win_rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_mean.itertuples(index=False):
        lines.append(
            f"| {row.dataset_id} | {row.n_samples} | {row.visual_iou_mean:.6f} | {row.v4_iou_mean:.6f} | {row.mean_delta:.6f} | {row.v4_win_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Representative Shortlist",
            "",
            "| panel | sample_id | dataset | stability_proxy | visual_iou_mean | v4_iou_mean | mean_delta | v4_wins | visual_wins | note |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in shortlist.itertuples(index=False):
        lines.append(
            f"| {row.panel_tag} | {row.sample_id} | {row.dataset_id} | {row.stability_proxy:.6f} | "
            f"{row.visual_iou_mean:.6f} | {row.v4_iou_mean:.6f} | {row.mean_delta:.6f} | "
            f"{row.v4_wins} | {row.visual_wins} | {row.panel_note} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            f"- `summary.json`: `{json_path}`",
            f"- `paired_sample_mean_diff.csv`: `{sample_mean_path}`",
            f"- `paired_case_shortlist.csv`: `{shortlist_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
