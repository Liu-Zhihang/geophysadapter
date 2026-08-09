#!/usr/bin/env python3
"""Material 与 Trigger 是否"辅助地形"：三条预先声明的假设。

背景
----
M/R 作为精度贡献者已被反复证否，最后一次是在它们自己的支持分层上、
用当前最强的 T 配置：Material 输给 T-only（−0.00072），Trigger 连自己的
错时控制都输（−0.00364）。覆盖率解释因此被排除，失败原因是**相对于地形的信息冗余**。

但"不能提高精度"与"没有作用"是两回事。本脚本检验三个不同的作用形式，
全部预先声明，跑完即停，负结果照实写。

H1  适用域：支持可用性是否标记了地形判据更有效的区域？
    已观察到源内三个对照方向一致（present 优于 absent）。这里补做置换检验，
    判断该分离是否超过随机可用性掩膜。Trigger 按构造是事件恒定的，因此
    事件级置换是精确的；Material 在事件内空间变化，其源内对照样本量不足，
    仅作描述性报告。

H2  操作点：Material 是否移动了地形判据的临界坡角？
    若不同岩性在不同坡角失稳，则 M 解释的是 T 的**参数**，而不是与 T 竞争。
    统计量为各材料簇临界坡角的极差，零分布由同源内打乱材料向量给出。

H3  可靠性条件：M/R 是否预测地形纯度模型**在哪里出错**？
    注意这与"M/R 能否改进纯度预测"是不同的问题，后者已判否。模型可以在
    平均意义上用足地形，同时误差是异方差的。若 M/R 能预测残差，它们就获得
    一个可部署角色：在低可靠区弃权，这正是已冻结合同里 m_M、τ_R 的语义。
    可部署形式测两种：有界残差校正，以及弃权门。均须胜过同源打乱控制。

共享步骤：先用最终配置（92 维 + 源独热、五种子、事件分组 OOF）算出全局纯度分数，
三条假设都复用它，因此不存在"每条假设各自拟合出不同模型"的比较污染。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pild_object_role_support_strata_v1 import (  # noqa: E402
    CONFIDENCE,
    DEFAULT_CACHE,
    DEFAULT_HYDRO,
    DEFAULT_UNITS,
    TERRAIN,
    TRIGGER_NAMES,
    ensemble_oof,
    load_role_context,
    sample_pixel_counts,
    shuffle_across_events_within_source,
)

POOLED_BASELINE_IOU = 0.21819164482792633
N_PERMUTATIONS = 500
N_MATERIAL_CLUSTERS = 4


# --------------------------------------------------------------------------- #
# 记账：给任意分量子集算 ΔIoU，参照系为该子集自身基线
# --------------------------------------------------------------------------- #
def subset_outcome(frame, remove, counts_lookup):
    ids = frame.sample_id.unique()
    tp_all = sum(counts_lookup[s][0] for s in ids)
    fp_all = sum(counts_lookup[s][1] for s in ids)
    fn_all = sum(counts_lookup[s][2] for s in ids)
    base_iou = tp_all / (tp_all + fp_all + fn_all)

    tp = float(frame.intersection_px.sum())
    fp = float(frame.false_px.sum())
    fn = tp / base_iou - tp - fp
    lost = float(frame.intersection_px.to_numpy(dtype=float)[remove].sum())
    cleared = float(frame.false_px.to_numpy(dtype=float)[remove].sum())
    base_err = fp + fn
    new_err = (fp - cleared) + (fn + lost)
    return {
        "n_units": int(len(frame)),
        "baseline_iou": float(base_iou),
        "delta_iou": float((tp - lost) / (tp + fp + fn - cleared) - base_iou),
        "rer": float((base_err - new_err) / base_err),
    }


# --------------------------------------------------------------------------- #
# H1：支持可用性是否标记地形判据的适用域
# --------------------------------------------------------------------------- #
def hypothesis_one(frame, remove, counts_lookup, rng, n_permutations):
    """Trigger 按构造事件恒定，事件级置换精确；只在两侧事件数充足的来源上做。"""
    results = []
    flag = (frame.q_trigger > 0).to_numpy()
    for source in sorted(frame.dataset_id.unique()):
        positions = np.nonzero((frame.dataset_id == source).to_numpy())[0]
        block = frame.iloc[positions]
        block_flag = flag[positions]
        block_remove = remove[positions]
        events = block.canonical_event_id.to_numpy()
        unique_events = np.unique(events)

        # 事件级可用性；若事件内不一致则记录，不静默平均
        event_flag = {}
        inconsistent = 0
        for name in unique_events:
            member = block_flag[events == name]
            event_flag[name] = bool(member.mean() >= 0.5)
            if 0 < member.mean() < 1:
                inconsistent += 1
        present_events = [e for e in unique_events if event_flag[e]]
        absent_events = [e for e in unique_events if not event_flag[e]]
        if min(len(present_events), len(absent_events)) < 3:
            continue

        def separation(assign_present: set) -> float:
            mask = np.isin(events, list(assign_present))
            if mask.sum() < 50 or (~mask).sum() < 50:
                return np.nan
            a = subset_outcome(block[mask], block_remove[mask], counts_lookup)
            b = subset_outcome(block[~mask], block_remove[~mask], counts_lookup)
            return a["delta_iou"] - b["delta_iou"]

        observed = separation(set(present_events))
        null = []
        for _ in range(n_permutations):
            permuted = rng.permutation(unique_events)
            null.append(separation(set(permuted[: len(present_events)])))
        null = np.asarray([v for v in null if np.isfinite(v)], dtype=float)
        p_value = float((np.sum(null >= observed) + 1) / (null.size + 1))
        results.append(
            {
                "role": "trigger",
                "dataset_id": source,
                "n_events_present": len(present_events),
                "n_events_absent": len(absent_events),
                "events_with_mixed_availability": inconsistent,
                "observed_separation": float(observed),
                "null_mean": float(null.mean()),
                "null_p95": float(np.percentile(null, 95)),
                "permutation_p": p_value,
                "n_permutations_valid": int(null.size),
            }
        )
        print(
            f"  [H1] trigger / {source:24s} 观测分离={observed:+.5f}  "
            f"零分布均值={null.mean():+.5f}  p95={np.percentile(null, 95):+.5f}  "
            f"p={p_value:.4f}  (事件 {len(present_events)}present/{len(absent_events)}absent, "
            f"混合 {inconsistent})",
            flush=True,
        )
    return results


# --------------------------------------------------------------------------- #
# H2：Material 是否移动地形判据的临界坡角
# --------------------------------------------------------------------------- #
def critical_slope(purity, slope, cut, min_count=200, n_bins=12):
    """局部平均纯度跨过解析判据的坡角，即该子群的判据操作点。"""
    if purity.size < min_count:
        return np.nan
    order = np.argsort(slope)
    slope_sorted = slope[order]
    purity_sorted = purity[order]
    edges = np.linspace(0, purity.size, n_bins + 1).astype(int)
    centres, means = [], []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end - start < 10:
            continue
        centres.append(float(np.median(slope_sorted[start:end])))
        means.append(float(purity_sorted[start:end].mean()))
    if len(centres) < 4:
        return np.nan
    centres = np.asarray(centres)
    means = np.asarray(means)
    crossings = np.nonzero((means[:-1] < cut) & (means[1:] >= cut))[0]
    if crossings.size == 0:
        return np.nan
    index = int(crossings[0])
    lo, hi = means[index], means[index + 1]
    weight = (cut - lo) / max(hi - lo, 1e-9)
    return float(centres[index] + weight * (centres[index + 1] - centres[index]))


def hypothesis_two(frame, cut, rng, n_permutations, material_cols):
    support = frame.loc[frame.q_material > 0].reset_index(drop=True)
    values = support[material_cols].to_numpy(dtype=float)
    centre = values.mean(axis=0)
    spread = values.std(axis=0)
    spread[spread < 1e-6] = 1.0
    purity = support.purity.to_numpy(dtype=float)
    slope = support.mean_slope.to_numpy(dtype=float)
    datasets = support.dataset_id.to_numpy()
    events = support.canonical_event_id.to_numpy()

    def cluster_spread(matrix) -> tuple[float, list[float]]:
        labels = KMeans(
            n_clusters=N_MATERIAL_CLUSTERS, n_init=4, random_state=0
        ).fit_predict((matrix - centre) / spread)
        thresholds = [
            critical_slope(purity[labels == k], slope[labels == k], cut)
            for k in range(N_MATERIAL_CLUSTERS)
        ]
        finite = [t for t in thresholds if np.isfinite(t)]
        if len(finite) < 2:
            return np.nan, thresholds
        return float(max(finite) - min(finite)), thresholds

    observed, thresholds = cluster_spread(values)
    null = []
    for index in range(n_permutations):
        shuffled = shuffle_across_events_within_source(
            values, datasets, events, int(rng.integers(0, 2**31 - 1))
        )
        value, _ = cluster_spread(shuffled)
        if np.isfinite(value):
            null.append(value)
    null = np.asarray(null, dtype=float)
    p_value = float((np.sum(null >= observed) + 1) / (null.size + 1)) if null.size else np.nan
    print(
        f"  [H2] 材料簇临界坡角：{[round(t, 2) if np.isfinite(t) else None for t in thresholds]}\n"
        f"       观测极差={observed:.3f}°  零分布均值={null.mean():.3f}°  "
        f"p95={np.percentile(null, 95):.3f}°  p={p_value:.4f}",
        flush=True,
    )
    return {
        "cluster_thresholds_deg": [float(t) if np.isfinite(t) else None for t in thresholds],
        "observed_spread_deg": float(observed),
        "null_mean_deg": float(null.mean()) if null.size else None,
        "null_p95_deg": float(np.percentile(null, 95)) if null.size else None,
        "permutation_p": p_value,
        "n_permutations_valid": int(null.size),
    }


# --------------------------------------------------------------------------- #
# H3：M/R 是否预测地形纯度模型的残差，并能否据此弃权
# --------------------------------------------------------------------------- #
def hypothesis_three(frame, score, cut, counts_lookup, seeds, role, columns, control_kind):
    mask = (frame[f"q_{role}"] > 0).to_numpy()
    stratum = frame.loc[mask].reset_index(drop=True)
    local_score = score[mask]
    residual = stratum.purity.to_numpy(dtype=float) - local_score
    groups = stratum.canonical_event_id.to_numpy()

    aligned = stratum[columns].to_numpy(dtype=float)
    if control_kind == "shuffle":
        control = shuffle_across_events_within_source(
            aligned, stratum.dataset_id.to_numpy(), groups, 20260726
        )
    else:
        wrongtime = stratum["rain_d7_wrongtime_median_mm"].to_numpy(dtype=float)
        control = np.column_stack([wrongtime, wrongtime, np.zeros_like(wrongtime)])

    rows = []
    for label, matrix in (("aligned", aligned), (f"control_{control_kind}", control)):
        # 残差有正有负，不能复用会裁剪到 [0,1] 的 ensemble_oof
        predicted = np.zeros(len(residual), dtype=float)
        for seed in seeds:
            out = np.zeros(len(residual), dtype=float)
            for train, test in GroupKFold(5).split(matrix, residual, groups=groups):
                model = HistGradientBoostingRegressor(
                    max_iter=300, max_leaf_nodes=31, learning_rate=0.06,
                    l2_regularization=1.0, random_state=seed,
                )
                model.fit(matrix[train], residual[train])
                out[test] = model.predict(matrix[test])
            predicted += out
        predicted /= len(seeds)

        rho = float(pd.Series(predicted).corr(pd.Series(residual), method="spearman"))
        rho_abs = float(
            pd.Series(np.abs(predicted)).corr(pd.Series(np.abs(residual)), method="spearman")
        )

        # 可部署形式 A：有界残差校正后再套同一判据
        bound = 0.05
        corrected = np.clip(local_score + bound * np.tanh(predicted / bound), 0.0, 1.0)
        outcome_correct = subset_outcome(stratum, corrected < cut, counts_lookup)

        # 可部署形式 B：弃权门。预测残差为正（模型低估纯度）说明该体更可能是真滑坡，
        # 此时放弃移除，逐位退回视觉。
        base_remove = local_score < cut
        gated = base_remove & ~(predicted > bound)
        outcome_gate = subset_outcome(stratum, gated, counts_lookup)

        rows.append(
            {
                "role": role,
                "arm": label,
                "residual_spearman": rho,
                "abs_residual_spearman": rho_abs,
                "delta_iou_correction": outcome_correct["delta_iou"],
                "delta_iou_gate": outcome_gate["delta_iou"],
                "n_units": int(len(stratum)),
            }
        )
        print(
            f"  [H3] {role:8s} {label:18s} rho(residual)={rho:+.4f}  "
            f"rho(|residual|)={rho_abs:+.4f}  "
            f"ΔIoU 校正={outcome_correct['delta_iou']:+.5f}  "
            f"ΔIoU 弃权门={outcome_gate['delta_iou']:+.5f}",
            flush=True,
        )

    baseline = subset_outcome(stratum, local_score < cut, counts_lookup)
    print(f"  [H3] {role:8s} {'未加 M/R 的参照':18s} ΔIoU={baseline['delta_iou']:+.5f}", flush=True)
    return {"reference": baseline, "arms": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--hydrology", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 101, 2029, 55555])
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument(
        "--outdir", type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_object_role_assist_terrain_v1",
    )
    args = parser.parse_args()
    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    prespecified = {
        "declared_before_execution": True,
        "H1": "支持可用性是否标记地形判据更有效的区域；Trigger 事件级置换检验",
        "H2": "Material 是否移动地形判据的临界坡角；同源打乱置换检验",
        "H3": "M/R 是否预测地形纯度模型的残差，并能否据此做有界校正或弃权门",
        "stopping_rule": "三条跑完即停，不追加变体；负结果照实写入",
        "shared_score": "最终配置（92 维 + 源独热、五种子、事件分组 OOF），三条假设复用同一分数",
        "analytic_cut": POOLED_BASELINE_IOU / (1.0 + POOLED_BASELINE_IOU),
    }
    (args.outdir / "prespecification.json").write_text(
        json.dumps(prespecified, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    whole = pd.read_parquet(args.units / "units_whole.parquet")
    hydro = pd.read_parquet(args.hydrology)
    frame = whole.merge(hydro, on=["sample_id", "component_id"], how="inner")
    frame = frame.merge(load_role_context(args.cache), on="sample_id", how="left")
    frame = frame.reset_index(drop=True)

    spec_cols = [c for c in frame.columns if c.startswith("spec_")]
    hyd_cols = [c for c in frame.columns if c.startswith("hyd_")]
    feature_cols = TERRAIN + CONFIDENCE + spec_cols + hyd_cols
    material_cols = [c for c in frame.columns if c.startswith("material_")]

    counts = sample_pixel_counts(args.cache)
    counts_lookup = {
        row.sample_id: (row.tp, row.fp, row.fn) for row in counts.itertuples(index=False)
    }

    cut = POOLED_BASELINE_IOU / (1.0 + POOLED_BASELINE_IOU)
    print("计算最终配置的全局纯度分数（三条假设共用）…", flush=True)
    onehot = pd.get_dummies(
        pd.Categorical(frame.dataset_id, categories=sorted(frame.dataset_id.unique()))
    ).to_numpy(dtype=float)
    x = np.hstack([frame[feature_cols].to_numpy(dtype=float), onehot])
    score = ensemble_oof(
        x, frame.purity.to_numpy(dtype=float), frame.canonical_event_id.to_numpy(),
        args.seeds, 5, 800, 63,
    )
    remove = score < cut
    print(f"  全局分数完成，移除 {int(remove.sum()):,} / {len(frame):,} 个候选体\n", flush=True)

    rng = np.random.default_rng(20260726)
    print("H1 适用域：", flush=True)
    h1 = hypothesis_one(frame, remove, counts_lookup, rng, args.permutations)

    print("\nH2 操作点：", flush=True)
    h2 = hypothesis_two(frame, cut, rng, min(args.permutations, 200), material_cols)

    print("\nH3 可靠性条件：", flush=True)
    h3 = {
        "material": hypothesis_three(
            frame, score, cut, counts_lookup, args.seeds, "material", material_cols, "shuffle"
        ),
        "trigger": hypothesis_three(
            frame, score, cut, counts_lookup, args.seeds, "trigger", TRIGGER_NAMES, "wrong_time"
        ),
    }

    summary = {
        "schema_version": "pild_object_role_assist_terrain.v1",
        "evidence_status": "development: event-grouped OOF on already-opened folds",
        "prespecified": prespecified,
        "H1_validity_domain": h1,
        "H2_operating_point": h2,
        "H3_reliability_conditioning": h3,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n写出 {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
