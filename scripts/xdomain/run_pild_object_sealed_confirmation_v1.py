#!/usr/bin/env python3
"""对象级物理审查的预冻结单次复算。

为什么需要这一步
----------------
2026-07-25 收敛出的最终配置（92 维描述子 + 源独热 + 五种子 HGB + 解析判据）
是在整个语料上反复读取结果之后定下来的。事件分组 OOF 保证了**拟合**没有事件泄漏，
但**配置选择**看过全部事件的结果，因此这些数字只能标为 development，不能用
"independent confirmation" 这类措辞。

本脚本消除的正是"分析者自由度"这一项，做法有三条：

1. 协议先冻结后执行。特征清单、学习器超参、种子、判据公式、分区规则、输入文件
   内容哈希全部写入 `protocol_freeze.json` 并取 SHA-256；重跑时若哈希不一致直接终止。
2. 事件按外部确定性规则劈成 design / sealed 两个分区。模型只在 design 分区拟合，
   sealed 分区在本次运行中从未参与任何拟合；判据阈值也只由 design 分区的基线 IoU 决定，
   不读取 sealed 的任何统计量。
3. 错配阶梯（aligned / shift32 / roll64 / donor）在同一次执行内跑完，
   因此"空间对齐特异性"也获得单次射击的数字，而不是又一轮可挑选的对照。

诚实边界
--------
sealed 分区的标签在历史上被打开过，因此本结果的正确标签是
`PRE_REGISTERED_SINGLE_SHOT`，**不是** `PROSPECTIVE_CONFIRMATION`。
后者只能来自合作者持有标签的新事件队列或第三方封存评估。
脚本写出运行锁；再次运行必须显式声明重开，且会在收据中留痕。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pild_object_veto_mismatch_controls_v1 import (  # noqa: E402
    CONFIDENCE,
    MIN_AREA,
    RING_RADIUS,
    TERRAIN,
    build_condition_table,
)

DEFAULT_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
DEFAULT_SPECTRAL = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_spectral_features_v1/object_spectral_features.parquet"
)
FOLD_IDS = [f"source_stratified_{i}" for i in range(4)]

# 分区规则的盐值属于冻结协议的一部分，不得在看到结果后更换。
PARTITION_SALT = "pild_object_sealed_confirmation.v1"


# --------------------------------------------------------------------------- #
# 1. 分区与基线：只依赖事件标识与 OOF 缓存，不依赖任何调参产物
# --------------------------------------------------------------------------- #
def partition_of(event_id: str) -> str:
    """外部确定性规则：事件标识加盐哈希的奇偶决定分区。"""
    digest = hashlib.sha256(f"{PARTITION_SALT}|{event_id}".encode("utf-8")).hexdigest()
    return "sealed" if int(digest[:16], 16) % 2 else "design"


def pooled_counts_by_event(cache_dir: Path) -> pd.DataFrame:
    """逐事件统计视觉锚点的 TP/FP/FN，用于给每个分区算自己的基线 IoU。"""
    records: dict[str, list[float]] = {}
    for fold_id in FOLD_IDS:
        receipt = json.loads(
            (cache_dir / f"{fold_id}_oof_cache_receipt.json").read_text(encoding="utf-8")
        )
        threshold = float(receipt["threshold"])
        with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
            probability = handle["visual_probability"]
            target = handle["target"]
            valid = handle["valid"]
            event = handle["canonical_event_id"]
            for index in range(probability.shape[0]):
                keep = valid[index].astype(bool)
                truth = (target[index] > 0) & keep
                predicted = (probability[index].astype(np.float32) >= threshold) & keep
                slot = records.setdefault(str(event[index]), [0.0, 0.0, 0.0])
                slot[0] += float(np.count_nonzero(predicted & truth))
                slot[1] += float(np.count_nonzero(predicted & ~truth))
                slot[2] += float(np.count_nonzero(~predicted & truth))
    frame = pd.DataFrame(
        [{"canonical_event_id": k, "tp": v[0], "fp": v[1], "fn": v[2]} for k, v in records.items()]
    )
    frame["partition"] = frame.canonical_event_id.map(partition_of)
    return frame


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 2. 冻结协议：先写盘取哈希，再执行
# --------------------------------------------------------------------------- #
def build_protocol(args, cache_dir: Path, spectral: Path) -> dict:
    inputs = {str(spectral.relative_to(PROJECT_ROOT)): sha256_of(spectral)}
    for fold_id in FOLD_IDS:
        for name in (f"{fold_id}_oof_cache_receipt.json",):
            path = cache_dir / name
            inputs[str(path.relative_to(PROJECT_ROOT))] = sha256_of(path)
    return {
        "schema_version": "pild_object_sealed_confirmation.v1",
        "frozen_before_execution": True,
        "visual_anchor": args.anchor,
        "decision_unit": "connected component of the frozen visual prediction",
        "feature_contract": {
            "terrain": TERRAIN,
            "confidence": CONFIDENCE,
            "spectral": "all columns prefixed spec_ in the frozen spectral parquet",
            "hydrology": "all columns prefixed hyd_ recomputed per terrain condition",
            "source_onehot": "dataset_id one-hot with categories fixed by sorted order",
        },
        "learner": {
            "estimator": "HistGradientBoostingRegressor",
            "target": "component purity",
            "max_iter": args.max_iter,
            "max_leaf_nodes": args.max_leaf_nodes,
            "learning_rate": 0.06,
            "l2_regularization": 1.0,
            "seeds": args.seeds,
            "fit_scope": "design partition only; no cross-validation inside the sealed partition",
        },
        "decision_rule": {
            "formula": "remove component when predicted purity < IoU_design / (1 + IoU_design)",
            "iou_source": "pooled baseline IoU of the design partition only",
            "tunable_parameters": 0,
        },
        "partition_rule": {
            "salt": PARTITION_SALT,
            "rule": "sealed if int(sha256(salt|canonical_event_id)[:16], 16) % 2 else design",
            "declared_before_execution": True,
        },
        "conditions": args.conditions,
        "control_seed": args.control_seed,
        "component_filter": {"min_area_px": MIN_AREA, "ring_radius_px": RING_RADIUS},
        "input_hashes": inputs,
        "evidence_label": "PRE_REGISTERED_SINGLE_SHOT",
        "honest_boundary": (
            "sealed 分区的标签在历史上被打开过；本结果消除的是配置选择自由度，"
            "不等于前瞻性独立确认。后者需要合作者持有标签的新事件队列。"
        ),
    }


# --------------------------------------------------------------------------- #
# 3. 单次执行：design 拟合 -> sealed 预测 -> 逐条件记账
# --------------------------------------------------------------------------- #
def fit_predict(design_x, design_y, sealed_x, seeds, max_iter, max_leaf_nodes):
    total = np.zeros(len(sealed_x), dtype=float)
    for seed in seeds:
        model = HistGradientBoostingRegressor(
            max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, learning_rate=0.06,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(design_x, design_y)
        total += np.clip(model.predict(sealed_x), 0.0, 1.0)
    return total / len(seeds)


def outcome_of(frame, remove, base_iou):
    """与最终配置逐字一致的记账口径：FN 由该分区自身基线 IoU 反推。"""
    tp = float(frame.intersection_px.sum())
    fp = float(frame.false_px.sum())
    fn = tp / base_iou - tp - fp
    i_px = frame.intersection_px.to_numpy(dtype=float)
    f_px = frame.false_px.to_numpy(dtype=float)
    lost = float(i_px[remove].sum())
    cleared = float(f_px[remove].sum())
    base_err = fp + fn
    new_err = (fp - cleared) + (fn + lost)
    purity = frame.purity.to_numpy(dtype=float)
    return {
        "n_units": int(len(frame)),
        "n_removed": int(remove.sum()),
        "baseline_iou": base_iou,
        "delta_iou": float((tp - lost) / (tp + fp + fn - cleared) - base_iou),
        "iou_after": float((tp - lost) / (tp + fp + fn - cleared)),
        "rer": float((base_err - new_err) / base_err),
        "lost_tp": lost,
        "cleared_fp": cleared,
        "fp_mass_captured": float(cleared / fp),
        "tp_mass_lost": float(lost / tp),
        "corrected_to_harmed": float(cleared / max(lost, 1.0)),
        "removal_precision": float(
            1.0 - (remove & (purity >= base_iou)).sum() / max(remove.sum(), 1)
        ),
    }


def event_macro(frame, remove, base_iou, n_boot=5000, seed=0):
    work = frame.assign(_rm=remove)
    rows = []
    for name, block in work.groupby("canonical_event_id"):
        e_tp = float(block.intersection_px.sum())
        e_fp = float(block.false_px.sum())
        if e_tp <= 0:
            continue
        e_fn = max(e_tp / base_iou - e_tp - e_fp, 0.0)
        lost = float(block.intersection_px[block._rm].sum())
        cleared = float(block.false_px[block._rm].sum())
        denom = e_tp + e_fp + e_fn
        base_err = e_fp + e_fn
        new_err = (e_fp - cleared) + (e_fn + lost)
        rows.append(
            {
                "canonical_event_id": name,
                "n_units": int(len(block)),
                "delta_iou": (e_tp - lost) / max(denom - cleared, 1.0) - e_tp / denom,
                "rer": (base_err - new_err) / base_err if base_err > 0 else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(table), size=(n_boot, len(table)))
    d_means = table.delta_iou.to_numpy()[draws].mean(axis=1)
    r_means = table.rer.to_numpy()[draws].mean(axis=1)
    return table, {
        "event_macro_delta_iou": float(table.delta_iou.mean()),
        "event_macro_delta_ci": [
            float(np.percentile(d_means, 2.5)), float(np.percentile(d_means, 97.5))
        ],
        "event_macro_rer": float(table.rer.mean()),
        "event_macro_rer_ci": [
            float(np.percentile(r_means, 2.5)), float(np.percentile(r_means, 97.5))
        ],
        "events_positive": int((table.delta_iou > 0).sum()),
        "n_events": int(len(table)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--spectral", type=Path, default=DEFAULT_SPECTRAL)
    parser.add_argument("--anchor", default="prithvi_eo2_300m_tl")
    parser.add_argument(
        "--conditions", nargs="+", default=["aligned", "shift32", "roll64", "donor"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 101, 2029, 55555])
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--control-seed", type=int, default=20260725)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_sealed_confirmation_v1",
    )
    parser.add_argument(
        "--reopen", action="store_true",
        help="显式声明重开一次已完成的封存运行；会在收据中留痕",
    )
    args = parser.parse_args()
    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- 运行锁：封存复算的价值就在于只射一次 ---
    lock = args.outdir / "sealed_receipt.json"
    reopen_count = 0
    if lock.is_file():
        previous = json.loads(lock.read_text(encoding="utf-8"))
        reopen_count = int(previous.get("reopen_count", 0)) + 1
        if not args.reopen:
            print(f"[abort] 已存在封存收据 {lock}；重开必须显式加 --reopen")
            return 2
        print(f"[warn] 这是第 {reopen_count} 次重开，收据将记录该事实")

    # --- 冻结协议：先落盘取哈希，再动数据 ---
    protocol = build_protocol(args, args.cache, args.spectral)
    protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True)
    protocol_hash = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    freeze_path = args.outdir / "protocol_freeze.json"
    if freeze_path.is_file():
        existing = freeze_path.read_text(encoding="utf-8")
        existing_hash = hashlib.sha256(existing.encode("utf-8")).hexdigest()
        if existing_hash != protocol_hash and not args.reopen:
            print("[abort] 协议哈希与已冻结版本不一致，拒绝执行")
            return 3
    freeze_path.write_text(protocol_text, encoding="utf-8")
    print(f"[freeze] protocol SHA-256 = {protocol_hash}")

    # --- 分区与两个分区各自的基线 IoU ---
    counts = pooled_counts_by_event(args.cache)
    grouped = counts.groupby("partition")[["tp", "fp", "fn"]].sum()
    baselines = {
        name: float(row.tp / (row.tp + row.fp + row.fn)) for name, row in grouped.iterrows()
    }
    design_events = set(counts.loc[counts.partition == "design", "canonical_event_id"])
    sealed_events = set(counts.loc[counts.partition == "sealed", "canonical_event_id"])
    cut = baselines["design"] / (1.0 + baselines["design"])
    print(
        f"[partition] design {len(design_events)} 事件 基线 IoU={baselines['design']:.5f}  |  "
        f"sealed {len(sealed_events)} 事件 基线 IoU={baselines['sealed']:.5f}\n"
        f"[rule] 判据阈值由 design 分区决定：purity < {cut:.5f}（sealed 未参与）"
    )
    counts.to_csv(args.outdir / "event_partition.csv", index=False)

    spectral = pd.read_parquet(args.spectral)
    spec_cols = [c for c in spectral.columns if c.startswith("spec_")]

    rows = []
    for condition in args.conditions:
        stage = time.time()
        frame = build_condition_table(
            args.cache, condition, args.control_seed, MIN_AREA, RING_RADIUS
        )
        frame = frame.merge(spectral, on=["sample_id", "component_id"], how="inner")
        hyd_cols = [c for c in frame.columns if c.startswith("hyd_")]
        feature_cols = TERRAIN + CONFIDENCE + spec_cols + hyd_cols

        categories = sorted(frame.dataset_id.unique())
        onehot = pd.get_dummies(
            pd.Categorical(frame.dataset_id, categories=categories)
        ).to_numpy(dtype=float)
        x_all = np.hstack([frame[feature_cols].to_numpy(dtype=float), onehot])
        y_all = frame.purity.to_numpy(dtype=float)
        is_sealed = frame.canonical_event_id.isin(sealed_events).to_numpy()

        score = fit_predict(
            x_all[~is_sealed], y_all[~is_sealed], x_all[is_sealed],
            args.seeds, args.max_iter, args.max_leaf_nodes,
        )
        sealed_frame = frame.loc[is_sealed].reset_index(drop=True)
        remove = score < cut
        result = outcome_of(sealed_frame, remove, baselines["sealed"])
        events, macro = event_macro(sealed_frame, remove, baselines["sealed"])
        result.update(macro)
        result.update(
            {
                "condition": condition,
                "spearman": float(
                    pd.Series(score).corr(pd.Series(sealed_frame.purity), method="spearman")
                ),
                "n_design_units": int((~is_sealed).sum()),
                "elapsed_seconds": round(time.time() - stage, 1),
            }
        )
        rows.append(result)
        events.to_csv(args.outdir / f"sealed_by_event_{condition}.csv", index=False)
        print(
            f"{condition:9s} sealed ΔIoU={result['delta_iou']:+.5f}  RER={result['rer']:+.4f}  "
            f"rho={result['spearman']:.3f}  c/h={result['corrected_to_harmed']:.2f}  "
            f"事件宏观={result['event_macro_delta_iou']:+.5f} "
            f"[{macro['event_macro_delta_ci'][0]:+.5f}, {macro['event_macro_delta_ci'][1]:+.5f}]  "
            f"{result['events_positive']}/{result['n_events']} 正  ({result['elapsed_seconds']:.0f}s)"
        )

    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / "sealed_conditions.csv", index=False)
    aligned = table[table.condition == "aligned"].iloc[0]
    controls = table[table.condition != "aligned"]
    verdict = {
        "sealed_aligned_delta_iou": float(aligned.delta_iou),
        "sealed_aligned_rer": float(aligned.rer),
        "sealed_event_macro_delta_iou": float(aligned.event_macro_delta_iou),
        "sealed_event_macro_delta_ci": list(aligned.event_macro_delta_ci),
        "best_control_delta_iou": float(controls.delta_iou.max()) if len(controls) else None,
        "aligned_minus_best_control": (
            float(aligned.delta_iou - controls.delta_iou.max()) if len(controls) else None
        ),
        "beats_all_controls": (
            bool((aligned.delta_iou > controls.delta_iou).all()) if len(controls) else None
        ),
    }
    print("\n裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))

    lock.write_text(
        json.dumps(
            {
                "schema_version": "pild_object_sealed_confirmation.v1",
                "evidence_label": "PRE_REGISTERED_SINGLE_SHOT",
                "protocol_sha256": protocol_hash,
                "visual_anchor": args.anchor,
                "reopen_count": reopen_count,
                "partition": {
                    "design_events": len(design_events),
                    "sealed_events": len(sealed_events),
                    "design_baseline_iou": baselines["design"],
                    "sealed_baseline_iou": baselines["sealed"],
                    "analytic_cut_from_design": cut,
                },
                "conditions": rows,
                "verdict": verdict,
                "honest_boundary": protocol["honest_boundary"],
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
