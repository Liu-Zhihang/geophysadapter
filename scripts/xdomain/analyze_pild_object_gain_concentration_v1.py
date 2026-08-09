#!/usr/bin/env python3
"""Test whether event-level addressable false-positive structure predicts ΔIoU.

For each event, compute addressable_share, large_share, large_pure_share, and
fp_to_tp_ratio from candidate-body purity/area statistics, then relate them to
event-level ΔIoU. Thresholds follow the paper definitions (purity ≤ 0.10,
area ≥ 200 px).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_FINAL = PROJECT_ROOT / "experiments/revision2026/pild_object_veto_final_v1"

PURITY_CUT = 0.10
LARGE_AREA_PX = 200


def per_event_structure(components: pd.DataFrame) -> pd.DataFrame:
    """逐事件的误差结构描述子，全部为占比，不受事件绝对大小影响。"""
    rows = []
    for name, block in components.groupby("canonical_event_id"):
        fp = float(block.false_px.sum())
        tp = float(block.intersection_px.sum())
        if fp <= 0:
            continue
        pure = block.purity <= PURITY_CUT
        large = block.area_px >= LARGE_AREA_PX
        rows.append(
            {
                "canonical_event_id": name,
                "dataset_id": block.dataset_id.iloc[0],
                "n_units": int(len(block)),
                "fp_mass": fp,
                "tp_mass": tp,
                "fp_to_tp_ratio": fp / max(tp, 1.0),
                "addressable_share": float(block.false_px[pure].sum() / fp),
                "large_share": float(block.false_px[large].sum() / fp),
                "large_pure_share": float(block.false_px[pure & large].sum() / fp),
            }
        )
    return pd.DataFrame(rows)


def ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = ((y - y.mean()) ** 2).sum()
    return float(1.0 - (residual**2).sum() / total) if total > 0 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_gain_concentration_v1",
    )
    args = parser.parse_args()
    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    components = pd.read_parquet(args.final / "component_decisions.parquet")
    events = pd.read_csv(args.final / "source_conditioned_by_event.csv")
    structure = per_event_structure(components)
    table = structure.merge(
        events[["canonical_event_id", "delta_iou", "rer"]],
        on="canonical_event_id", how="inner",
    )
    print(
        f"事件 {len(table)}  来源 {table.dataset_id.nunique()}  "
        f"（近纯假阳性定义 purity<={PURITY_CUT}，大体定义 area>={LARGE_AREA_PX}px）\n"
    )

    predictors = ["addressable_share", "large_share", "large_pure_share", "fp_to_tp_ratio"]
    correlations = []
    for column in predictors:
        rho, p_value = stats.spearmanr(table[column], table.delta_iou)
        rho_rer, p_rer = stats.spearmanr(table[column], table.rer)
        correlations.append(
            {
                "predictor": column,
                "spearman_vs_delta_iou": float(rho),
                "p_vs_delta_iou": float(p_value),
                "spearman_vs_rer": float(rho_rer),
                "p_vs_rer": float(p_rer),
            }
        )
        print(
            f"  {column:20s} vs ΔIoU  rho={rho:+.3f} (p={p_value:.4g})   "
            f"vs RER  rho={rho_rer:+.3f} (p={p_rer:.4g})"
        )

    # 源内相关：排除"关系只是数据源差异的投影"这一解释
    print("\n源内 Spearman（addressable_share vs ΔIoU）：")
    within = []
    for source, block in table.groupby("dataset_id"):
        if len(block) < 5:
            print(f"  {source:26s} n={len(block):2d}  事件数不足，跳过")
            continue
        rho, p_value = stats.spearmanr(block.addressable_share, block.delta_iou)
        within.append(
            {"dataset_id": source, "n_events": int(len(block)),
             "spearman": float(rho), "p": float(p_value)}
        )
        print(f"  {source:26s} n={len(block):2d}  rho={rho:+.3f} (p={p_value:.4g})")

    # 增量解释力：控制住误差结构后，数据源是否还有额外解释力
    y = table.delta_iou.to_numpy(dtype=float)
    structure_matrix = table[["addressable_share", "large_share", "fp_to_tp_ratio"]].to_numpy(float)
    source_dummies = pd.get_dummies(table.dataset_id, drop_first=True).to_numpy(float)
    r2 = {
        "structure_only": ols_r2(y, structure_matrix),
        "source_only": ols_r2(y, source_dummies),
        "structure_plus_source": ols_r2(y, np.hstack([structure_matrix, source_dummies])),
    }
    r2["source_increment_over_structure"] = r2["structure_plus_source"] - r2["structure_only"]
    r2["structure_increment_over_source"] = r2["structure_plus_source"] - r2["source_only"]
    print(
        "\n增量解释力（逐事件 ΔIoU 的 R²）：\n"
        f"  仅误差结构        {r2['structure_only']:.4f}\n"
        f"  仅数据源哑变量    {r2['source_only']:.4f}\n"
        f"  两者              {r2['structure_plus_source']:.4f}\n"
        f"  → 数据源在结构之上的增量  {r2['source_increment_over_structure']:+.4f}\n"
        f"  → 结构在数据源之上的增量  {r2['structure_increment_over_source']:+.4f}"
    )

    print("\n逐源误差结构（解释为何增益集中）：")
    by_source = table.groupby("dataset_id").agg(
        n_events=("canonical_event_id", "count"),
        addressable_share=("addressable_share", "mean"),
        large_share=("large_share", "mean"),
        fp_to_tp_ratio=("fp_to_tp_ratio", "median"),
        delta_iou=("delta_iou", "mean"),
    ).reset_index()
    print(by_source.to_string(index=False))

    table.to_csv(args.outdir / "event_structure_vs_gain.csv", index=False)
    by_source.to_csv(args.outdir / "by_source_structure.csv", index=False)

    primary = next(c for c in correlations if c["predictor"] == "addressable_share")
    verdict = {
        "primary_spearman": primary["spearman_vs_delta_iou"],
        "primary_p": primary["p_vs_delta_iou"],
        "mechanism_supported": bool(
            primary["spearman_vs_delta_iou"] >= 0.4 and primary["p_vs_delta_iou"] < 0.05
        ),
        "source_is_redundant_given_structure": bool(
            r2["source_increment_over_structure"] < 0.05
        ),
        "within_source_all_positive": bool(all(item["spearman"] > 0 for item in within)) if within else None,
    }
    print("\n裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))

    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_gain_concentration.v1",
                "evidence_status": "explanatory diagnostic; uses labels, not a deployable predictor",
                "prespecified_definitions": {
                    "near_pure_false_positive": f"purity <= {PURITY_CUT}（G0 既有定义）",
                    "large_body": f"area >= {LARGE_AREA_PX} px（光谱消融既有定义）",
                },
                "n_events": int(len(table)),
                "correlations": correlations,
                "within_source": within,
                "variance_explained": r2,
                "by_source": by_source.to_dict("records"),
                "verdict": verdict,
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n写出 {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
