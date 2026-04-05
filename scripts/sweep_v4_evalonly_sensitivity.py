from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_strict_t2_postrgb_v4_pilot.py"
OUT_ROOT = ROOT / "experiments" / "strict_t2_postrgb_v4_evalonly_sensitivity_v1"


@dataclass(frozen=True)
class RunConfig:
    tag: str
    prior_fusion_scale: float
    interaction_scale: float


SEED_TO_CKPT = {
    20260311: ROOT / "experiments" / "strict_t2_postrgb_deeplabv3_resnet50_v4_pilot_e3_sp05_lb025_distill_v1" / "best_model.pt",
    20260312: ROOT / "experiments" / "strict_t2_postrgb_deeplabv3_resnet50_v4_pilot_e3_sp05_lb025_distill_seed20260312" / "best_model.pt",
    20260313: ROOT / "experiments" / "strict_t2_postrgb_deeplabv3_resnet50_v4_pilot_e3_sp05_lb025_distill_seed20260313" / "best_model.pt",
}

CONFIGS = [
    RunConfig("pf05_is01_traincfg", 0.5, 0.1),
    RunConfig("pf10_is01", 1.0, 0.1),
    RunConfig("pf15_is01", 1.5, 0.1),
    RunConfig("pf05_is00", 0.5, 0.0),
    RunConfig("pf05_is025", 0.5, 0.25),
    RunConfig("pf10_is025_default", 1.0, 0.25),
]


def run_one(seed: int, cfg: RunConfig) -> dict[str, object]:
    ckpt = SEED_TO_CKPT[seed]
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    outdir = OUT_ROOT / cfg.tag / f"seed{seed}"
    summary_path = outdir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--root",
        str(ROOT),
        "--outdir",
        str(outdir),
        "--eval-only-ckpt",
        str(ckpt),
        "--lambda-distill",
        "0",
        "--workers",
        "0",
        "--device",
        "auto",
        "--tune-threshold-on-val",
        "--prior-fusion-scale",
        str(cfg.prior_fusion_scale),
        "--interaction-scale",
        str(cfg.interaction_scale),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def mean_sd(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return "nan"
    if arr.size == 1:
        return f"{arr.mean():.4f}"
    return f"{arr.mean():.4f} +- {arr.std(ddof=1):.4f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_reports(rows: list[dict[str, object]]) -> None:
    path_csv = OUT_ROOT / "per_run_metrics.csv"
    write_csv(path_csv, rows)

    baseline_rows = [row for row in rows if row["config"] == "pf05_is01_traincfg"]
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for config, cfg_rows in grouped.items():
        cfg_rows = sorted(cfg_rows, key=lambda item: int(item["seed"]))
        test_ious = [float(row["test_iou"]) for row in cfg_rows]
        test_f1s = [float(row["test_f1"]) for row in cfg_rows]
        thresholds = [float(row["eval_threshold"]) for row in cfg_rows]
        deltas = []
        for row in cfg_rows:
            base = baseline_by_seed[int(row["seed"])]
            deltas.append(float(row["test_iou"]) - float(base["test_iou"]))
        summary_rows.append(
            {
                "config": config,
                "prior_fusion_scale": cfg_rows[0]["prior_fusion_scale"],
                "interaction_scale": cfg_rows[0]["interaction_scale"],
                "test_iou_mean_sd": mean_sd(test_ious),
                "test_f1_mean_sd": mean_sd(test_f1s),
                "eval_threshold_mean_sd": mean_sd(thresholds),
                "delta_vs_traincfg_iou_mean_sd": mean_sd(deltas),
                "mean_test_iou": float(np.mean(test_ious)),
                "mean_test_f1": float(np.mean(test_f1s)),
            }
        )

    summary_rows.sort(key=lambda row: float(row["mean_test_iou"]), reverse=True)
    write_csv(OUT_ROOT / "summary_table.csv", summary_rows)

    lines = [
        "# strict_t2 post_rgb v4 eval-only sensitivity v1",
        "",
        "## Scope",
        "",
        "- Three released distill multiseed checkpoints (`20260311`, `20260312`, `20260313`).",
        "- `eval-only` inference on the shared post-event test pool.",
        "- Validation-threshold sweep enabled for every configuration.",
        "- Distillation loss disabled during eval-only replay (`--lambda-distill 0`).",
        "",
        "## Configurations",
        "",
        "| config | prior_fusion_scale | interaction_scale |",
        "|---|---:|---:|",
    ]
    for cfg in CONFIGS:
        lines.append(f"| `{cfg.tag}` | {cfg.prior_fusion_scale:.2f} | {cfg.interaction_scale:.2f} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| config | test IoU (mean +- sd) | test F1 (mean +- sd) | eval threshold | delta vs train config IoU |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| `{config}` | {test_iou_mean_sd} | {test_f1_mean_sd} | {eval_threshold_mean_sd} | {delta_vs_traincfg_iou_mean_sd} |".format(
                **row
            )
        )
    best = summary_rows[0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The strongest mean test IoU comes from `{best['config']}` at `{best['mean_test_iou']:.4f}`.",
            "- The sweep tests whether the heterogeneous residual-prior result is narrowly tied to one hand-tuned coefficient setting.",
            "- Reviewer-facing conclusion should remain conservative: directional robustness across reasonable settings matters more than maximizing a single point estimate.",
            "",
            "## Per-run outputs",
            "",
            "- `per_run_metrics.csv`: one row per seed/configuration.",
            "- `summary_table.csv`: table-ready aggregate summary.",
        ]
    )
    (OUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for cfg in CONFIGS:
        for seed in sorted(SEED_TO_CKPT):
            summary = run_one(seed, cfg)
            rows.append(
                {
                    "config": cfg.tag,
                    "seed": seed,
                    "prior_fusion_scale": cfg.prior_fusion_scale,
                    "interaction_scale": cfg.interaction_scale,
                    "best_epoch": summary.get("best_epoch", ""),
                    "eval_threshold": float(summary.get("eval_threshold", 0.0)),
                    "test_iou": float(summary["test_metrics"]["iou"]),
                    "test_f1": float(summary["test_metrics"]["f1"]),
                    "default_test_iou": float(summary["test_metrics_default"]["iou"]),
                    "default_test_f1": float(summary["test_metrics_default"]["f1"]),
                    "cas_test_iou": float(summary["test_by_dataset"]["CAS_Landslide"]["iou"]),
                    "dlr_test_iou": float(summary["test_by_dataset"]["DLR_Landslide_Ref_2025"]["iou"]),
                    "ckpt": str(SEED_TO_CKPT[seed]),
                    "outdir": summary.get("outdir", ""),
                }
            )
    build_reports(rows)


if __name__ == "__main__":
    main()
