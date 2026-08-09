#!/usr/bin/env python3
"""Material 与 Trigger 的支持分层条件效应，以及缺失-性能曲线。

为什么必须分层评估
------------------
用整池 ΔIoU 判定 M/R 存在一个结构性错配：覆盖率与误差质量是**反相关**的。
GDCLD 承载最大的可寻址假阳性质量，而它的 Material 覆盖仅 10.7%、Trigger 为 0%。
用整池平均去评价一个只够得着约一半分量的修正，等于让它为它够不到的地方负责。
条件效应 + 覆盖率同时报告是标准做法，不是包装——前提是分层必须由既有的
可用性掩膜定义，而不是看到结果后挑出来的。

本脚本回答两个问题
------------------
Q1  在支持确实存在的分量上，对齐的 M / R 是否改进了对象级决策？
    判据是双重的：既要胜过同一受限集合上的 T-only，也要胜过自己的错配控制。
Q2  性能如何随支持完整性退化？（这正是 R3.12 问的那条曲线。在像元尺度上
    监督池内核心完整性没有自然变异所以不可识别，但在对象尺度上覆盖率
    天然从 10.7% 变到 100%，曲线因此可画。）

预先固定的规则（看到结果之后不得更改）
--------------------------------------
1. 分层由既有可用性掩膜 `q_material > 0` / `q_trigger > 0` 定义，逐样本，非事后挑选。
2. 每个分层用**自己的**整池基线 IoU 作参照系与解析判据，与换锚点时同一条纪律。
3. M/R 直接作为协变量进入，由错配控制裁决。这是合法的，因为
   **同源内打乱恰好中和了来源身份**：若模型只是把 M 学成了数据源标签，
   打乱臂会拿到相同分数，于是 aligned − shuffle 恰好隔离出事件特异的内容。
4. 采纳判据：M（或 R）只有在其支持分层上同时满足
   `ΔIoU(T+role) > ΔIoU(T)` 且 `ΔIoU(T+role) > ΔIoU(T+role_mismatch)` 才可写为正效应。
5. 分层结果不得脱离覆盖率单独报告。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_UNITS = PROJECT_ROOT / "experiments/revision2026/pild_subobject_units_v1"
DEFAULT_HYDRO = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_hydrology_features_v1/object_hydrology_features.parquet"
)
DEFAULT_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
FOLD_IDS = [f"source_stratified_{i}" for i in range(4)]

CONFIDENCE = ["mean_probability", "max_probability", "p90_probability"]
TERRAIN = [
    "area_px", "log_area", "mean_slope", "p10_slope", "p90_slope", "flat_fraction",
    "steep_fraction", "elev_range", "relative_relief", "aspect_coherence", "elongation",
    "downslope_alignment", "descent_consistency", "slope_decline", "divide_straddle",
    "tpi900_range", "mean_tpi_90m", "mean_tpi_300m", "mean_tpi_900m",
    "valley_bottom_fraction", "mean_valley_depth", "mean_ridge_height", "mean_ruggedness",
    "mean_local_relief_300m", "mean_plan_curvature", "mean_profile_curvature", "compactness",
]
TRIGGER_NAMES = [
    "rain_d7_antecedent_case_mm",
    "rain_d7_wrongtime_median_mm",
    "rain_d7_case_minus_wrongtime_mm",
]


# --------------------------------------------------------------------------- #
# 数据装配
# --------------------------------------------------------------------------- #
def load_role_context(cache_dir: Path) -> pd.DataFrame:
    """逐样本的 Material / Trigger 上下文及其可用性标记。"""
    frames = []
    for fold_id in FOLD_IDS:
        with np.load(cache_dir / f"{fold_id}_optical_cache.npz", allow_pickle=False) as handle:
            material = handle["material_features"]
            trigger = handle["trigger_features"]
            frame = pd.DataFrame(
                {
                    "sample_id": [str(item) for item in handle["sample_id"]],
                    "q_material": handle["q_material"].astype(float),
                    "q_trigger": handle["q_trigger"].astype(float),
                }
            )
        for index in range(material.shape[1]):
            frame[f"material_{index:02d}"] = material[:, index].astype(float)
        for index, name in enumerate(TRIGGER_NAMES[: trigger.shape[1]]):
            frame[name] = trigger[:, index].astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sample_pixel_counts(cache_dir: Path) -> pd.DataFrame:
    """逐样本 TP/FP/FN，用于给任意子集算它自己的基线 IoU。"""
    rows = []
    for fold_id in FOLD_IDS:
        receipt = json.loads(
            (cache_dir / f"{fold_id}_oof_cache_receipt.json").read_text(encoding="utf-8")
        )
        threshold = float(receipt["threshold"])
        with np.load(cache_dir / f"{fold_id}_oof_cache.npz", allow_pickle=False) as handle:
            probability = handle["visual_probability"]
            target = handle["target"]
            valid = handle["valid"]
            sample_id = [str(item) for item in handle["sample_id"]]
            for index in range(probability.shape[0]):
                keep = valid[index].astype(bool)
                truth = (target[index] > 0) & keep
                predicted = (probability[index].astype(np.float32) >= threshold) & keep
                rows.append(
                    {
                        "sample_id": sample_id[index],
                        "tp": float(np.count_nonzero(predicted & truth)),
                        "fp": float(np.count_nonzero(predicted & ~truth)),
                        "fn": float(np.count_nonzero(~predicted & truth)),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 错配控制
# --------------------------------------------------------------------------- #
def shuffle_across_events_within_source(
    values: np.ndarray, dataset: np.ndarray, event: np.ndarray, seed: int
) -> np.ndarray:
    """把每个事件的上下文换成同源另一事件的上下文。

    保留边缘分布与来源身份，只破坏事件与空间的对应关系。正因为来源身份被保留，
    这个控制才能隔离出事件特异的内容，而不是把"模型学到了数据源"也一起打掉。
    """
    rng = np.random.default_rng(seed)
    output = values.copy()
    for source in np.unique(dataset):
        members = np.nonzero(dataset == source)[0]
        events = np.unique(event[members])
        if events.size < 2:
            continue
        permuted = rng.permutation(events)
        donor = {}
        for position, original in enumerate(events):
            replacement = permuted[position]
            if replacement == original:
                replacement = events[(position + 1) % events.size]
            donor[original] = replacement
        pool = {name: np.nonzero((dataset == source) & (event == name))[0] for name in events}
        for index in members:
            candidates = pool[donor[event[index]]]
            if candidates.size:
                output[index] = values[rng.choice(candidates)]
    return output


def wrong_time_trigger(frame: pd.DataFrame) -> np.ndarray:
    """把事件时段降雨换成同地点错时窗口，量级保留、事件特异性破坏。"""
    wrongtime = frame["rain_d7_wrongtime_median_mm"].to_numpy(dtype=float)
    return np.column_stack([wrongtime, wrongtime, np.zeros_like(wrongtime)])


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def ensemble_oof(x, y, groups, seeds, n_splits, max_iter, max_leaf_nodes):
    total = np.zeros(len(y), dtype=float)
    for seed in seeds:
        out = np.zeros(len(y), dtype=float)
        for train, test in GroupKFold(n_splits).split(x, y, groups=groups):
            model = HistGradientBoostingRegressor(
                max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, learning_rate=0.06,
                l2_regularization=1.0, random_state=seed,
            )
            model.fit(x[train], y[train])
            out[test] = model.predict(x[test])
        total += np.clip(out, 0.0, 1.0)
    return total / len(seeds)


def score_arm(frame, extra, base_iou, seeds, n_splits, max_iter, max_leaf_nodes, feature_cols):
    """在给定分层上跑一条臂，返回与最终配置同口径的记账。"""
    cut = base_iou / (1.0 + base_iou)
    y = frame.purity.to_numpy(dtype=float)
    groups = frame.canonical_event_id.to_numpy()
    onehot = pd.get_dummies(
        pd.Categorical(frame.dataset_id, categories=sorted(frame.dataset_id.unique()))
    ).to_numpy(dtype=float)
    blocks = [frame[feature_cols].to_numpy(dtype=float), onehot]
    if extra is not None:
        blocks.append(extra)
    x = np.hstack(blocks)

    score = ensemble_oof(x, y, groups, seeds, n_splits, max_iter, max_leaf_nodes)
    remove = score < cut

    tp = float(frame.intersection_px.sum())
    fp = float(frame.false_px.sum())
    fn = tp / base_iou - tp - fp
    lost = float(frame.intersection_px.to_numpy(dtype=float)[remove].sum())
    cleared = float(frame.false_px.to_numpy(dtype=float)[remove].sum())
    base_err = fp + fn
    new_err = (fp - cleared) + (fn + lost)
    return {
        "n_units": int(len(frame)),
        "n_removed": int(remove.sum()),
        "baseline_iou": base_iou,
        "delta_iou": float((tp - lost) / (tp + fp + fn - cleared) - base_iou),
        "rer": float((base_err - new_err) / base_err),
        "spearman": float(pd.Series(score).corr(pd.Series(y), method="spearman")),
        "corrected_to_harmed": float(cleared / max(lost, 1.0)),
        "fp_mass_captured": float(cleared / fp),
        "tp_mass_lost": float(lost / tp),
    }


def subset_baseline_iou(counts: pd.DataFrame, sample_ids) -> float:
    block = counts[counts.sample_id.isin(set(sample_ids))]
    tp, fp, fn = float(block.tp.sum()), float(block.fp.sum()), float(block.fn.sum())
    return tp / (tp + fp + fn)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--hydrology", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 101, 2029, 55555])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--control-seed", type=int, default=20260726)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_role_support_strata_v1",
    )
    args = parser.parse_args()
    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    whole = pd.read_parquet(args.units / "units_whole.parquet")
    hydro = pd.read_parquet(args.hydrology)
    frame = whole.merge(hydro, on=["sample_id", "component_id"], how="inner")
    context = load_role_context(args.cache)
    frame = frame.merge(context, on="sample_id", how="left", validate="many_to_one")
    if frame.q_material.isna().any():
        raise RuntimeError("部分分量缺少角色上下文")
    frame = frame.reset_index(drop=True)

    spec_cols = [c for c in frame.columns if c.startswith("spec_")]
    hyd_cols = [c for c in frame.columns if c.startswith("hyd_")]
    feature_cols = TERRAIN + CONFIDENCE + spec_cols + hyd_cols
    material_cols = [c for c in frame.columns if c.startswith("material_")]

    counts = sample_pixel_counts(args.cache)
    dataset = frame.dataset_id.to_numpy()
    event = frame.canonical_event_id.to_numpy()

    coverage = {
        "n_components": int(len(frame)),
        "material_fraction": float((frame.q_material > 0).mean()),
        "trigger_fraction": float((frame.q_trigger > 0).mean()),
        "material_by_source": frame.groupby("dataset_id")
        .q_material.apply(lambda s: float((s > 0).mean())).to_dict(),
        "trigger_by_source": frame.groupby("dataset_id")
        .q_trigger.apply(lambda s: float((s > 0).mean())).to_dict(),
    }
    print("覆盖率：" + json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", flush=True)

    rows = []

    # ---- Q1：支持分层上的条件效应 ---------------------------------------- #
    roles = [
        ("material", frame.q_material > 0, material_cols, "shuffle"),
        ("trigger", frame.q_trigger > 0, TRIGGER_NAMES, "wrong_time"),
    ]
    for role, mask, columns, control_kind in roles:
        stratum = frame.loc[mask].reset_index(drop=True)
        if stratum.canonical_event_id.nunique() < args.n_splits:
            print(f"[skip] {role} 分层事件数不足")
            continue
        base_iou = subset_baseline_iou(counts, stratum.sample_id.unique())
        aligned = stratum[columns].to_numpy(dtype=float)
        if control_kind == "shuffle":
            mismatched = shuffle_across_events_within_source(
                aligned,
                stratum.dataset_id.to_numpy(),
                stratum.canonical_event_id.to_numpy(),
                args.control_seed,
            )
        else:
            mismatched = wrong_time_trigger(stratum)

        print(
            f"=== {role} 支持分层：{len(stratum):,} 分量 / "
            f"{stratum.canonical_event_id.nunique()} 事件 / 基线 IoU={base_iou:.5f} ===",
            flush=True,
        )
        for arm, extra in (
            (f"T_only__{role}_stratum", None),
            (f"T_{role}_aligned", aligned),
            (f"T_{role}_{control_kind}", mismatched),
        ):
            result = score_arm(
                stratum, extra, base_iou, args.seeds, args.n_splits,
                args.max_iter, args.max_leaf_nodes, feature_cols,
            )
            result.update({"arm": arm, "role": role, "stratum": f"{role}_available"})
            rows.append(result)
            print(
                f"  {arm:34s} ΔIoU={result['delta_iou']:+.5f}  RER={result['rer']:+.4f}  "
                f"rho={result['spearman']:.3f}  c/h={result['corrected_to_harmed']:.2f}",
                flush=True,
            )
        print(flush=True)

    # ---- Q2：缺失-性能曲线（T-only 在不同覆盖分层上的表现） --------------- #
    print("=== 缺失-性能曲线：T-only 配置在支持有无两侧 ===", flush=True)
    for role in ("material", "trigger"):
        flag = frame[f"q_{role}"] > 0
        for label, mask in ((f"{role}_present", flag), (f"{role}_absent", ~flag)):
            stratum = frame.loc[mask].reset_index(drop=True)
            if len(stratum) == 0 or stratum.canonical_event_id.nunique() < args.n_splits:
                print(f"  [skip] {label} 事件数不足")
                continue
            base_iou = subset_baseline_iou(counts, stratum.sample_id.unique())
            result = score_arm(
                stratum, None, base_iou, args.seeds, args.n_splits,
                args.max_iter, args.max_leaf_nodes, feature_cols,
            )
            result.update({"arm": f"curve__{label}", "role": role, "stratum": label})
            rows.append(result)
            print(
                f"  {label:20s} n={result['n_units']:6,}  基线={base_iou:.5f}  "
                f"ΔIoU={result['delta_iou']:+.5f}  RER={result['rer']:+.4f}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / "role_support_strata.csv", index=False)

    lookup = {row["arm"]: row for row in rows}
    verdict = {}
    for role, control_kind in (("material", "shuffle"), ("trigger", "wrong_time")):
        base = lookup.get(f"T_only__{role}_stratum")
        aligned = lookup.get(f"T_{role}_aligned")
        control = lookup.get(f"T_{role}_{control_kind}")
        if not (base and aligned and control):
            continue
        verdict[role] = {
            "coverage": coverage[f"{role}_fraction"],
            "stratum_baseline_iou": base["baseline_iou"],
            "delta_iou_T_only": base["delta_iou"],
            "delta_iou_aligned": aligned["delta_iou"],
            "delta_iou_control": control["delta_iou"],
            "aligned_minus_T_only": aligned["delta_iou"] - base["delta_iou"],
            "aligned_minus_control": aligned["delta_iou"] - control["delta_iou"],
            "promoted": bool(
                aligned["delta_iou"] > base["delta_iou"]
                and aligned["delta_iou"] > control["delta_iou"]
            ),
        }
    print("\n裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))

    (args.outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pild_object_role_support_strata.v1",
                "evidence_status": "development: event-grouped OOF on already-opened folds",
                "prespecified_rules": [
                    "分层由既有可用性掩膜定义，逐样本，非事后挑选",
                    "每个分层使用自身基线 IoU 作参照系与解析判据",
                    "M/R 直接作为协变量，由同源内打乱 / 错时窗口裁决",
                    "采纳需同时胜过同一分层上的 T-only 与自身错配控制",
                    "分层结果不得脱离覆盖率单独报告",
                ],
                "coverage": coverage,
                "seeds": args.seeds,
                "arms": rows,
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
