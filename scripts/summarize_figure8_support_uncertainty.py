#!/usr/bin/env python3
"""Summarize support/uncertainty statistics for the residual-prior Figure 8 mechanism."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from render_strict_t2_postrgb_case_panels import load_model


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUT_DIR = EXPERIMENTS / "strict_t2_postrgb_v4_support_uncertainty_summary_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_SUMMARIES = [
    EXPERIMENTS / "strict_t2_postrgb_deeplabv3_resnet50_v4_pilot_e3_sp05_lb025_distill_v1" / "summary.json",
    EXPERIMENTS / "strict_t2_postrgb_deeplabv3_resnet50_v4_pilot_e3_sp05_lb025_distill_seed20260312" / "summary.json",
    EXPERIMENTS / "strict_t2_postrgb_deeplabv3_resnet50_v4_pilot_e3_sp05_lb025_distill_seed20260313" / "summary.json",
]
PAIRED_CSV = EXPERIMENTS / "strict_t2_postrgb_v4_vs_visual_paired_v1" / "paired_sample_mean_diff.csv"


def decode_bytes(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def source_family(sample_id: str) -> str:
    if sample_id.startswith("EID_"):
        return "DLR"
    if sample_id.startswith("CAS_Palu::"):
        return "CAS_Palu"
    if sample_id.startswith("GLADV1_"):
        return "GLaD4CD_v1"
    if sample_id.startswith("GDCLD"):
        return "GDCLD"
    return "Other"


def summary_test_cache(summary_json: Path) -> Path:
    summary = json.loads(summary_json.read_text())
    cache = summary.get("resolved_cache_h5", {}).get("test")
    if not cache:
        raise RuntimeError(f"missing test cache path in {summary_json}")
    return Path(str(cache))


def infer_seed_scalars(summary_json: Path, sample_filter: set[str], batch_size: int = 32) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(summary_json, device)
    cache_h5 = summary_test_cache(summary_json)

    with h5py.File(cache_h5, "r") as f:
        sample_ids = decode_bytes(f["sample_id"][:])
        event_uids = decode_bytes(f["event_uid"][:])
        keep_idx = [idx for idx, sid in enumerate(sample_ids) if sid in sample_filter]
        rows: list[dict[str, float | str]] = []

        for start in range(0, len(keep_idx), batch_size):
            batch_idx = keep_idx[start : start + batch_size]
            batch_images = torch.from_numpy(np.asarray(f["image"][batch_idx], dtype=np.float32)).to(device)
            batch_physics = []
            batch_meta = []
            batch_sample_ids = []
            batch_event_uids = []
            for idx in batch_idx:
                sid = sample_ids[idx]
                eid = event_uids[idx]
                zero_phys = np.zeros_like(bundle.mean, dtype=np.float32)
                zero_meta = np.zeros((18,), dtype=np.float32)
                vec = bundle.sample_map.get(sid, bundle.event_map.get(eid, zero_phys))
                vec = np.nan_to_num((vec - bundle.mean) / bundle.std, nan=0.0, posinf=0.0, neginf=0.0).astype(
                    np.float32,
                    copy=False,
                )
                meta = bundle.sample_meta.get(sid, bundle.event_meta.get(eid, zero_meta)).astype(np.float32, copy=False)
                batch_physics.append(vec)
                batch_meta.append(meta)
                batch_sample_ids.append(sid)
                batch_event_uids.append(eid)

            physics_t = torch.from_numpy(np.stack(batch_physics)).to(device)
            meta_t = torch.from_numpy(np.stack(batch_meta)).to(device)
            with torch.no_grad():
                out = bundle.model(image=batch_images, physics=physics_t, meta=meta_t)
            support = out["support"][:, 0].detach().cpu().numpy()
            uncertainty = out["uncertainty"][:, 0].detach().cpu().numpy()
            gate = out["gate"][:, 0].detach().cpu().numpy()

            for sid, eid, s, w, g in zip(batch_sample_ids, batch_event_uids, support, uncertainty, gate):
                rows.append(
                    {
                        "sample_id": sid,
                        "event_uid": eid,
                        "support": float(s),
                        "uncertainty": float(w),
                        "gate": float(g),
                    }
                )

    df = pd.DataFrame(rows)
    if len(df) != len(sample_filter):
        missing = sorted(sample_filter - set(df["sample_id"]))
        raise RuntimeError(f"missing support/uncertainty for {len(missing)} samples, e.g. {missing[:5]}")
    return df.sort_values("sample_id").reset_index(drop=True)


def quartile_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    ranked = df[key].rank(method="first")
    q = pd.qcut(ranked, q=4, labels=labels)
    work = df.assign(quartile=q.astype(str))
    out = (
        work.groupby("quartile", sort=False)
        .agg(
            n=("sample_id", "size"),
            strat_mean=(key, "mean"),
            support_mean=("support_mean", "mean"),
            uncertainty_mean=("uncertainty_mean", "mean"),
            gate_mean=("gate_mean", "mean"),
            delta_mean=("mean_delta", "mean"),
            delta_median=("mean_delta", "median"),
            positive_rate=("positive_delta", "mean"),
        )
        .reset_index()
    )
    out["positive_rate"] = out["positive_rate"] * 100.0
    return out


def outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["group"] = np.where(work["mean_delta"] > 0.0, "positive-delta", "non-positive-delta")
    out = (
        work.groupby("group", sort=False)
        .agg(
            n=("sample_id", "size"),
            support_mean=("support_mean", "mean"),
            uncertainty_mean=("uncertainty_mean", "mean"),
            gate_mean=("gate_mean", "mean"),
            correction_weight_mean=("correction_weight_mean", "mean"),
            delta_mean=("mean_delta", "mean"),
            delta_median=("mean_delta", "median"),
        )
        .reset_index()
    )
    return out


def format_table(df: pd.DataFrame, float_cols: list[str], percent_cols: list[str] | None = None) -> str:
    percent_cols = percent_cols or []
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---:|" * len(df.columns)
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if col in float_cols:
                vals.append(f"{float(val):.4f}")
            elif col in percent_cols:
                vals.append(f"{float(val):.1f}%")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    paired = pd.read_csv(PAIRED_CSV)
    paired["source_family"] = paired["sample_id"].map(source_family)
    sample_filter = set(paired["sample_id"])

    seed_frames = []
    for idx, summary_json in enumerate(SEED_SUMMARIES, start=1):
        seed_df = infer_seed_scalars(summary_json, sample_filter=sample_filter)
        seed_df = seed_df.rename(
            columns={
                "support": f"support_seed{idx}",
                "uncertainty": f"uncertainty_seed{idx}",
                "gate": f"gate_seed{idx}",
            }
        )
        seed_frames.append(seed_df)

    merged = seed_frames[0]
    for seed_df in seed_frames[1:]:
        merged = merged.merge(seed_df, on=["sample_id", "event_uid"], how="inner")

    support_cols = [c for c in merged.columns if c.startswith("support_seed")]
    uncertainty_cols = [c for c in merged.columns if c.startswith("uncertainty_seed")]
    gate_cols = [c for c in merged.columns if c.startswith("gate_seed")]
    merged["support_mean"] = merged[support_cols].mean(axis=1)
    merged["support_std"] = merged[support_cols].std(axis=1, ddof=0)
    merged["uncertainty_mean"] = merged[uncertainty_cols].mean(axis=1)
    merged["uncertainty_std"] = merged[uncertainty_cols].std(axis=1, ddof=0)
    merged["gate_mean"] = merged[gate_cols].mean(axis=1)
    merged["gate_std"] = merged[gate_cols].std(axis=1, ddof=0)
    merged["correction_weight_mean"] = merged["gate_mean"] * merged["support_mean"] * (1.0 - merged["uncertainty_mean"])

    full = paired.merge(merged, on=["sample_id", "event_uid"], how="inner")
    full["positive_delta"] = (full["mean_delta"] > 0.0).astype(float)

    support_quartiles = quartile_table(full, "support_mean")
    uncertainty_quartiles = quartile_table(full, "uncertainty_mean")
    correction_quartiles = quartile_table(full, "correction_weight_mean")
    outcome = outcome_table(full)

    per_sample_csv = OUT_DIR / "per_sample_support_uncertainty.csv"
    support_q_csv = OUT_DIR / "support_quartiles.csv"
    uncertainty_q_csv = OUT_DIR / "uncertainty_quartiles.csv"
    correction_q_csv = OUT_DIR / "correction_weight_quartiles.csv"
    outcome_csv = OUT_DIR / "delta_outcome_groups.csv"
    report_md = OUT_DIR / "report.md"

    full.to_csv(per_sample_csv, index=False)
    support_quartiles.to_csv(support_q_csv, index=False)
    uncertainty_quartiles.to_csv(uncertainty_q_csv, index=False)
    correction_quartiles.to_csv(correction_q_csv, index=False)
    outcome.to_csv(outcome_csv, index=False)

    report_lines = [
        "# strict_t2 post_rgb support/uncertainty summary for Figure 8",
        "",
        "Representative v4 residual-prior mechanism statistics over the shared `post_rgb` test pool, averaged over the three `v4 distill` seed checkpoints.",
        "",
        f"- n_samples = {len(full)}",
        f"- mean support = {full['support_mean'].mean():.4f}",
        f"- mean uncertainty = {full['uncertainty_mean'].mean():.4f}",
        f"- mean gate = {full['gate_mean'].mean():.4f}",
        f"- mean effective prior weight = {full['correction_weight_mean'].mean():.4f}",
        f"- mean delta IoU = {full['mean_delta'].mean():.4f}",
        f"- positive-delta rate = {(100.0 * full['positive_delta'].mean()):.1f}%",
        "",
        "## Effective prior-weight quartiles",
        "",
        format_table(
            correction_quartiles,
            float_cols=["strat_mean", "support_mean", "uncertainty_mean", "gate_mean", "delta_mean", "delta_median"],
            percent_cols=["positive_rate"],
        ),
        "",
        "## Support quartiles",
        "",
        format_table(
            support_quartiles,
            float_cols=["strat_mean", "support_mean", "uncertainty_mean", "gate_mean", "delta_mean", "delta_median"],
            percent_cols=["positive_rate"],
        ),
        "",
        "## Uncertainty quartiles",
        "",
        format_table(
            uncertainty_quartiles,
            float_cols=["strat_mean", "support_mean", "uncertainty_mean", "gate_mean", "delta_mean", "delta_median"],
            percent_cols=["positive_rate"],
        ),
        "",
        "## Positive vs non-positive delta groups",
        "",
        format_table(
            outcome,
            float_cols=["support_mean", "uncertainty_mean", "gate_mean", "correction_weight_mean", "delta_mean", "delta_median"],
        ),
        "",
        "Artifacts:",
        f"- per-sample csv: `{per_sample_csv}`",
        f"- correction-weight quartiles csv: `{correction_q_csv}`",
        f"- support quartiles csv: `{support_q_csv}`",
        f"- uncertainty quartiles csv: `{uncertainty_q_csv}`",
        f"- delta outcome csv: `{outcome_csv}`",
    ]
    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote {per_sample_csv}")
    print(f"Wrote {support_q_csv}")
    print(f"Wrote {uncertainty_q_csv}")
    print(f"Wrote {outcome_csv}")
    print(f"Wrote {report_md}")
    print(report_md.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()
