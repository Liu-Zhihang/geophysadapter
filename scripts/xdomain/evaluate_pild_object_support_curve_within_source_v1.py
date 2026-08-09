#!/usr/bin/env python3
"""支持完整性与性能的关系：源内去混杂对照。

整池上的"支持缺失处收益反而更大"几乎肯定是**来源混杂**：Material 缺失主要发生在
GDCLD（覆盖 10.7%），而 GDCLD 恰好是对象级审查收益最集中的来源。因此整池关联
不能作为"支持完整性—性能"曲线，必须在同一来源内部比较。

只有同时存在有支持与无支持分量、且两侧样本量足够的来源才进入对照。
每个分层仍使用自身基线 IoU 作参照系与解析判据，与全部对象级实验同一纪律。

这仍是自然实验：分配机制未知，支持可用性本身可能与地质或地形类型相关。
因此结论只能写成"源内关联"，不得写成因果曲线。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pild_object_role_support_strata_v1 import (  # noqa: E402
    CONFIDENCE,
    DEFAULT_CACHE,
    DEFAULT_HYDRO,
    DEFAULT_UNITS,
    TERRAIN,
    load_role_context,
    sample_pixel_counts,
    score_arm,
    subset_baseline_iou,
)

MIN_UNITS = 400
MIN_EVENTS = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--hydrology", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 101, 2029, 55555])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_support_curve_within_source_v1",
    )
    args = parser.parse_args()
    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    whole = pd.read_parquet(args.units / "units_whole.parquet")
    hydro = pd.read_parquet(args.hydrology)
    frame = whole.merge(hydro, on=["sample_id", "component_id"], how="inner")
    frame = frame.merge(load_role_context(args.cache), on="sample_id", how="left")
    frame = frame.reset_index(drop=True)

    spec_cols = [c for c in frame.columns if c.startswith("spec_")]
    hyd_cols = [c for c in frame.columns if c.startswith("hyd_")]
    feature_cols = TERRAIN + CONFIDENCE + spec_cols + hyd_cols
    counts = sample_pixel_counts(args.cache)

    rows = []
    for role in ("material", "trigger"):
        flag = frame[f"q_{role}"] > 0
        for source, block in frame.groupby("dataset_id"):
            present = block.loc[flag.loc[block.index]]
            absent = block.loc[~flag.loc[block.index]]
            if min(len(present), len(absent)) < MIN_UNITS:
                continue
            if min(
                present.canonical_event_id.nunique(), absent.canonical_event_id.nunique()
            ) < MIN_EVENTS:
                print(
                    f"[skip] {role} / {source}：两侧事件数 "
                    f"{present.canonical_event_id.nunique()}/{absent.canonical_event_id.nunique()} 不足",
                    flush=True,
                )
                continue
            print(f"=== {role} / {source} 源内对照 ===", flush=True)
            for label, stratum in (("present", present), ("absent", absent)):
                stratum = stratum.reset_index(drop=True)
                base_iou = subset_baseline_iou(counts, stratum.sample_id.unique())
                result = score_arm(
                    stratum, None, base_iou, args.seeds, args.n_splits,
                    args.max_iter, args.max_leaf_nodes, feature_cols,
                )
                result.update(
                    {
                        "role": role,
                        "dataset_id": source,
                        "support": label,
                        "n_events": int(stratum.canonical_event_id.nunique()),
                    }
                )
                rows.append(result)
                print(
                    f"  {label:8s} n={result['n_units']:6,} 事件={result['n_events']:3d}  "
                    f"基线={base_iou:.5f}  ΔIoU={result['delta_iou']:+.5f}  "
                    f"RER={result['rer']:+.4f}",
                    flush=True,
                )
            print(flush=True)

    if not rows:
        print("没有来源同时满足两侧样本量与事件数要求；源内去混杂不可识别。")

    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / "within_source_support_curve.csv", index=False)

    verdict = {}
    for (role, source), block in table.groupby(["role", "dataset_id"]) if rows else []:
        present = block[block.support == "present"].iloc[0]
        absent = block[block.support == "absent"].iloc[0]
        verdict[f"{role}__{source}"] = {
            "present_baseline_iou": float(present.baseline_iou),
            "absent_baseline_iou": float(absent.baseline_iou),
            "present_delta_iou": float(present.delta_iou),
            "absent_delta_iou": float(absent.delta_iou),
            "present_minus_absent": float(present.delta_iou - absent.delta_iou),
            "present_rer": float(present.rer),
            "absent_rer": float(absent.rer),
        }
    print("\n裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))

    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_support_curve_within_source.v1",
                "evidence_status": "development: event-grouped OOF on already-opened folds",
                "design_note": (
                    "自然实验，分配机制未知；支持可用性本身可能与地质或地形类型相关。"
                    "结论只能写成源内关联，不得写成因果曲线。"
                ),
                "min_units": MIN_UNITS,
                "min_events": MIN_EVENTS,
                "rows": rows,
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
