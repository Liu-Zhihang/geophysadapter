#!/usr/bin/env python3
"""G7-a：物理变量作为跨事件判据校准器的判决实验。

背景（实测，见 analyze_pild_object_headroom_v1）：
    事件分折 OOF + 最优全局阈值        ΔIoU = +0.0239
    事件分折 OOF + 逐事件最优阈值      ΔIoU = +0.0382
    逐事件最优阈值中位 0.152，全距 [0.007, 0.488]，10/55 事件最优动作是完全不移除。
    数据源分组分折时上界坍到 +0.0006。

即：地形排序本身信息足够，损失全部发生在"判据无法跨事件迁移"。这正是 Material 与
Trigger 在其原生尺度上唯一不可替代的工作——Material 决定该立地条件下多陡才算失稳，
Trigger 决定该次事件应当多宽容。本脚本检验这一预测。

协议要点：
1. 纯度模型只用地形+置信度特征，事件分组 OOF，与 veto_gate v2 同口径；
2. 逐事件最优阈值 tau*_e 由该事件自身 OOF 分数与标签解析求得，作为回归目标；
3. 阈值预测严格 leave-one-event-out，测试事件的 tau* 从不进入训练；
4. 部署评价用预测阈值重算整池与事件级 ΔIoU/RER，而非只看回归 R^2；
5. 采纳必须同时胜过：全局阈值、纯视觉事件描述子、地形聚合事件描述子、打乱/错时窗控制。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_DECISIONS = (
    PROJECT_ROOT
    / "experiments/revision2026/pild_object_veto_gate_v2/aligned_component_decisions.csv"
)
DEFAULT_CACHE = (
    PROJECT_ROOT / "experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
FOLD_IDS = [f"source_stratified_{i}" for i in range(4)]

TRIGGER_NAMES = (
    "rain_d7_antecedent_case_mm",
    "rain_d7_wrongtime_median_mm",
    "rain_d7_case_minus_wrongtime_mm",
)

# 对象级纯度模型的特征（与 veto_gate v2 一致）
OBJECT_FEATURES = [
    "area_px", "log_area", "mean_slope", "p10_slope", "p90_slope", "flat_fraction",
    "steep_fraction", "elev_range", "relative_relief", "aspect_coherence", "elongation",
    "downslope_alignment", "descent_consistency", "slope_decline", "divide_straddle",
    "tpi900_range", "mean_tpi_90m", "mean_tpi_300m", "mean_tpi_900m",
    "valley_bottom_fraction", "mean_valley_depth", "mean_ridge_height", "mean_ruggedness",
    "mean_local_relief_300m", "mean_plan_curvature", "mean_profile_curvature", "compactness",
    "terrain_support_fraction", "mean_probability", "max_probability", "p90_probability",
]

# 事件级"纯视觉"描述子：不含任何物理量，用于证伪"任何事件协变量都行"
VISUAL_EVENT_FEATURES = [
    "ev_n_objects", "ev_log_n_objects", "ev_mean_probability", "ev_p90_probability",
    "ev_max_probability", "ev_mean_log_area", "ev_median_log_area", "ev_total_pred_px",
    "ev_log_total_pred_px", "ev_pred_area_fraction", "ev_n_samples",
]

# 事件级地形聚合描述子：检验 Material 是否只是地形均值的代理
TERRAIN_EVENT_FEATURES = [
    "ev_mean_slope", "ev_p90_slope", "ev_steep_fraction", "ev_flat_fraction",
    "ev_relative_relief", "ev_mean_ruggedness", "ev_valley_bottom_fraction",
    "ev_mean_local_relief_300m", "ev_terrain_support_fraction",
]


# --------------------------------------------------------------------------------------
# 数据装载
# --------------------------------------------------------------------------------------
def load_role_context(cache_dir: Path) -> pd.DataFrame:
    """逐样本的 Material / Trigger 上下文与可用性标记。"""
    frames = []
    for fold_id in FOLD_IDS:
        path = cache_dir / f"{fold_id}_optical_cache.npz"
        with np.load(path, allow_pickle=False) as handle:
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


# --------------------------------------------------------------------------------------
# 第一层：地形纯度模型的事件分组 OOF
# --------------------------------------------------------------------------------------
def terrain_oof_scores(frame: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """事件分组交叉预测的对象纯度。分数越低越应被移除。"""
    x = frame[OBJECT_FEATURES].to_numpy(dtype=float)
    y = frame.purity.to_numpy(dtype=float)
    groups = frame.canonical_event_id.to_numpy()
    scores = np.zeros(len(frame), dtype=float)
    for train, test in GroupKFold(n_splits).split(x, y, groups=groups):
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(x[train], y[train])
        scores[test] = model.predict(x[test])
    return np.clip(scores, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# 第二层：逐事件最优阈值
# --------------------------------------------------------------------------------------
def event_optimal_threshold(
    score: np.ndarray, i_px: np.ndarray, f_px: np.ndarray, base_iou: float
) -> tuple[float, float]:
    """事件内最优移除阈值与对应增益。

    移除一个对象对整池 IoU 的一阶贡献正比于 (base_iou * f - i)，
    因此事件内按分数升序累加该量，取最大处的分数为阈值。
    若最大值 <= 0，则该事件的最优动作是完全不移除，阈值记为 0。
    """
    order = np.argsort(score)
    gains = np.cumsum(base_iou * f_px[order] - i_px[order])
    if len(gains) == 0 or gains.max() <= 0:
        return 0.0, 0.0
    k = int(np.argmax(gains))
    # 阈值取被移除的最后一个对象与下一个对象分数的中点，避免边界抖动
    upper = score[order][k + 1] if k + 1 < len(order) else 1.0
    return float(0.5 * (score[order][k] + upper)), float(gains[k])


# --------------------------------------------------------------------------------------
# 事件级描述子
# --------------------------------------------------------------------------------------
def build_event_table(
    frame: pd.DataFrame, score: np.ndarray, base_iou: float, context: pd.DataFrame
) -> pd.DataFrame:
    """每个事件一行：回归目标 + 三类描述子。"""
    work = frame.copy()
    work["score"] = score
    material_cols = [c for c in context.columns if c.startswith("material_")]

    rows = []
    for event, block in work.groupby("canonical_event_id", sort=True):
        tau, gain = event_optimal_threshold(
            block.score.to_numpy(),
            block.intersection_px.to_numpy(dtype=float),
            block.false_px.to_numpy(dtype=float),
            base_iou,
        )
        samples = context[context.sample_id.isin(set(block.sample_id))]
        m_active = samples[samples.q_material > 0]
        r_active = samples[samples.q_trigger > 0]
        row = {
            "canonical_event_id": event,
            "dataset_id": block.dataset_id.iloc[0],
            "tau_star": tau,
            "event_gain": gain,
            "abstain": int(tau <= 0.0),
            "n_objects": len(block),
            "tp_px": float(block.intersection_px.sum()),
            "fp_px": float(block.false_px.sum()),
            # --- 纯视觉事件描述子 ---
            "ev_n_objects": float(len(block)),
            "ev_log_n_objects": float(np.log10(len(block) + 1)),
            "ev_mean_probability": float(block.mean_probability.mean()),
            "ev_p90_probability": float(block.p90_probability.mean()),
            "ev_max_probability": float(block.max_probability.mean()),
            "ev_mean_log_area": float(block.log_area.mean()),
            "ev_median_log_area": float(block.log_area.median()),
            "ev_total_pred_px": float(block.area_px.sum()),
            "ev_log_total_pred_px": float(np.log10(block.area_px.sum() + 1)),
            "ev_n_samples": float(block.sample_id.nunique()),
            # --- 地形聚合事件描述子 ---
            "ev_mean_slope": float(block.mean_slope.mean()),
            "ev_p90_slope": float(block.p90_slope.mean()),
            "ev_steep_fraction": float(block.steep_fraction.mean()),
            "ev_flat_fraction": float(block.flat_fraction.mean()),
            "ev_relative_relief": float(block.relative_relief.mean()),
            "ev_mean_ruggedness": float(block.mean_ruggedness.mean()),
            "ev_valley_bottom_fraction": float(block.valley_bottom_fraction.mean()),
            "ev_mean_local_relief_300m": float(block.mean_local_relief_300m.mean()),
            "ev_terrain_support_fraction": float(block.terrain_support_fraction.mean()),
            # --- 支持覆盖率 ---
            "q_material_cov": float((samples.q_material > 0).mean()) if len(samples) else 0.0,
            "q_trigger_cov": float((samples.q_trigger > 0).mean()) if len(samples) else 0.0,
        }
        row["ev_pred_area_fraction"] = row["ev_total_pred_px"] / max(
            row["ev_n_samples"] * 128 * 128, 1.0
        )
        # --- Material 事件聚合：只在有支持的样本上取均值 ---
        for col in material_cols:
            row[f"ev_{col}"] = float(m_active[col].mean()) if len(m_active) else np.nan
        # --- Trigger 事件聚合 ---
        for name in TRIGGER_NAMES:
            row[f"ev_{name}"] = float(r_active[name].mean()) if len(r_active) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 第三层：leave-one-event-out 阈值预测与部署评价
# --------------------------------------------------------------------------------------
def predict_thresholds(
    events: pd.DataFrame, feature_cols: list[str], seed: int
) -> np.ndarray:
    """留一事件预测 tau*。样本量仅 55，使用强正则 Ridge，alpha 由训练事件内选。"""
    if not feature_cols:
        # 常数臂：训练事件的中位阈值
        out = np.zeros(len(events))
        for k in range(len(events)):
            out[k] = np.median(np.delete(events.tau_star.to_numpy(), k))
        return out

    x = events[feature_cols].to_numpy(dtype=float)
    # 缺失支持用训练集均值补，避免把"无支持"泄漏成特殊取值
    y = events.tau_star.to_numpy(dtype=float)
    out = np.zeros(len(events))
    alphas = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    rng = np.random.default_rng(seed)

    for k in range(len(events)):
        train = np.ones(len(events), dtype=bool)
        train[k] = False
        x_tr, y_tr = x[train], y[train]
        col_mean = np.nanmean(x_tr, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        x_tr = np.where(np.isfinite(x_tr), x_tr, col_mean)
        x_te = np.where(np.isfinite(x[k]), x[k], col_mean).reshape(1, -1)
        scaler = StandardScaler().fit(x_tr)
        x_tr_s, x_te_s = scaler.transform(x_tr), scaler.transform(x_te)

        # 内层再留一，选 alpha
        best_alpha, best_err = alphas[-1], np.inf
        for alpha in alphas:
            err = 0.0
            for j in range(len(x_tr_s)):
                inner = np.ones(len(x_tr_s), dtype=bool)
                inner[j] = False
                model = Ridge(alpha=alpha).fit(x_tr_s[inner], y_tr[inner])
                err += (model.predict(x_tr_s[j : j + 1])[0] - y_tr[j]) ** 2
            if err < best_err:
                best_alpha, best_err = alpha, err
        model = Ridge(alpha=best_alpha).fit(x_tr_s, y_tr)
        out[k] = float(model.predict(x_te_s)[0])
    del rng
    return np.clip(out, 0.0, 1.0)


def deploy(
    frame: pd.DataFrame,
    score: np.ndarray,
    events: pd.DataFrame,
    tau_pred: np.ndarray,
    tp: float,
    fp: float,
    fn: float,
) -> dict:
    """按预测的逐事件阈值执行移除，返回整池与事件级指标。"""
    tau_map = dict(zip(events.canonical_event_id, tau_pred))
    thresholds = frame.canonical_event_id.map(tau_map).to_numpy(dtype=float)
    remove = score < thresholds
    lost = float(frame.intersection_px.to_numpy()[remove].sum())
    cleared = float(frame.false_px.to_numpy()[remove].sum())
    base_iou = tp / (tp + fp + fn)
    new_iou = (tp - lost) / (tp + fp + fn - cleared)
    base_err = fp + fn
    new_err = (fp - cleared) + (fn + lost)

    # 事件级：每个事件用自己的 TP/FP/FN 重算
    macro = []
    for event, block in frame.assign(_rm=remove).groupby("canonical_event_id"):
        e_tp = float(block.intersection_px.sum())
        e_fp = float(block.false_px.sum())
        e_row = events[events.canonical_event_id == event].iloc[0]
        e_fn = max(e_tp / base_iou - e_tp - e_fp, 0.0) if e_tp > 0 else 0.0
        e_lost = float(block.intersection_px[block._rm].sum())
        e_clear = float(block.false_px[block._rm].sum())
        denom = e_tp + e_fp + e_fn
        if denom <= 0:
            continue
        b = e_tp / denom
        a = (e_tp - e_lost) / max(denom - e_clear, 1.0)
        macro.append(a - b)
        del e_row
    macro_arr = np.array(macro)
    return {
        "n_removed": int(remove.sum()),
        "delta_iou": float(new_iou - base_iou),
        "rer": float((base_err - new_err) / base_err),
        "lost_tp": lost,
        "cleared_fp": cleared,
        "removal_precision": float(
            1.0 - (remove & (frame.purity.to_numpy() >= base_iou)).sum() / max(remove.sum(), 1)
        ),
        "event_macro_delta_iou": float(macro_arr.mean()) if len(macro_arr) else 0.0,
        "event_positive_fraction": float((macro_arr > 0).mean()) if len(macro_arr) else 0.0,
    }


def shuffle_within_dataset(values: np.ndarray, datasets: np.ndarray, seed: int) -> np.ndarray:
    """在数据源内打乱事件级取值，保留源级分布、破坏事件级对应。"""
    rng = np.random.default_rng(seed)
    out = values.copy()
    for ds in np.unique(datasets):
        idx = np.where(datasets == ds)[0]
        if len(idx) > 1:
            out[idx] = values[rng.permutation(idx)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--baseline-iou", type=float, default=0.21819164482792633)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "experiments/revision2026/pild_event_criterion_calibration_v1",
    )
    args = parser.parse_args()
    started = time.time()

    frame = pd.read_csv(args.decisions)
    context = load_role_context(args.cache)
    tp = float(frame.intersection_px.sum())
    fp = float(frame.false_px.sum())
    fn = tp / args.baseline_iou - tp - fp
    base_iou = args.baseline_iou

    print(f"对象 {len(frame):,}  事件 {frame.canonical_event_id.nunique()}  源 {frame.dataset_id.nunique()}")
    score = terrain_oof_scores(frame, args.n_splits, args.seed)
    events = build_event_table(frame, score, base_iou, context)
    events = events.sort_values("canonical_event_id").reset_index(drop=True)
    print(
        f"逐事件最优阈值：中位 {events.tau_star.median():.3f}  "
        f"弃权事件 {int(events.abstain.sum())}/{len(events)}"
    )

    material_cols = [c for c in events.columns if c.startswith("ev_material_")]
    trigger_cols = [f"ev_{name}" for name in TRIGGER_NAMES]

    # 打乱 / 错时窗控制：在数据源内打乱事件级物理取值
    ds = events.dataset_id.to_numpy()
    events_shuf = events.copy()
    for col in material_cols + trigger_cols:
        events_shuf[col] = shuffle_within_dataset(
            events[col].to_numpy(dtype=float), ds, args.seed + 7
        )
    events_wrongtime = events.copy()
    events_wrongtime["ev_rain_d7_antecedent_case_mm"] = events["ev_rain_d7_wrongtime_median_mm"]
    events_wrongtime["ev_rain_d7_case_minus_wrongtime_mm"] = 0.0

    arms = [
        ("global_median", events, []),
        ("visual_event", events, VISUAL_EVENT_FEATURES),
        ("terrain_aggregate", events, TERRAIN_EVENT_FEATURES),
        ("material_only", events, material_cols + ["q_material_cov"]),
        ("trigger_only", events, trigger_cols + ["q_trigger_cov"]),
        ("material_trigger", events, material_cols + trigger_cols + ["q_material_cov", "q_trigger_cov"]),
        ("material_trigger_shuffled", events_shuf, material_cols + trigger_cols + ["q_material_cov", "q_trigger_cov"]),
        ("trigger_wrongtime", events_wrongtime, trigger_cols + ["q_trigger_cov"]),
        ("visual_plus_physics", events, VISUAL_EVENT_FEATURES + material_cols + trigger_cols),
    ]

    # 参考臂：单一全局最优阈值（事后），以及逐事件 oracle 阈值
    order = np.argsort(score)
    cum_i = np.cumsum(frame.intersection_px.to_numpy()[order])
    cum_f = np.cumsum(frame.false_px.to_numpy()[order])
    k = int(np.argmax((tp - cum_i) / (tp + fp + fn - cum_f)))
    global_best_tau = float(score[order][k])
    ref_global = deploy(
        frame, score, events, np.full(len(events), global_best_tau), tp, fp, fn
    )
    ref_oracle = deploy(frame, score, events, events.tau_star.to_numpy(), tp, fp, fn)

    rows = []
    for name, table, cols in arms:
        tau_pred = predict_thresholds(table, cols, args.seed)
        res = deploy(frame, score, events, tau_pred, tp, fp, fn)
        target = events.tau_star.to_numpy()
        res["arm"] = name
        res["n_features"] = len(cols)
        res["tau_spearman"] = float(
            pd.Series(tau_pred).corr(pd.Series(target), method="spearman")
        )
        res["tau_r2"] = float(
            1.0 - np.sum((tau_pred - target) ** 2) / np.sum((target - target.mean()) ** 2)
        )
        rows.append(res)
        print(
            f"{name:28s} ΔIoU={res['delta_iou']:+.5f}  RER={res['rer']:+.4f}  "
            f"macroΔ={res['event_macro_delta_iou']:+.5f}  "
            f"rho(tau)={res['tau_spearman']:+.3f}  R2={res['tau_r2']:+.3f}"
        )

    print(
        f"\n参考：事后最优全局阈值 ΔIoU={ref_global['delta_iou']:+.5f}   "
        f"逐事件 oracle 阈值 ΔIoU={ref_oracle['delta_iou']:+.5f}"
    )

    table = pd.DataFrame(rows)
    best = table.loc[table[table.arm.isin(["material_only", "trigger_only", "material_trigger"])]
                     .delta_iou.idxmax()]
    visual = table[table.arm == "visual_event"].iloc[0]
    terrain = table[table.arm == "terrain_aggregate"].iloc[0]
    shuffled = table[table.arm == "material_trigger_shuffled"].iloc[0]
    verdict = {
        "best_physics_arm": str(best.arm),
        "best_physics_delta_iou": float(best.delta_iou),
        "beats_visual_event": bool(best.delta_iou > visual.delta_iou),
        "beats_terrain_aggregate": bool(best.delta_iou > terrain.delta_iou),
        "beats_shuffled": bool(best.delta_iou > shuffled.delta_iou),
        "beats_global_threshold": bool(best.delta_iou > ref_global["delta_iou"]),
        "reaches_target_delta_iou": bool(best.delta_iou >= 0.03),
        "oracle_event_delta_iou": float(ref_oracle["delta_iou"]),
        "global_threshold_delta_iou": float(ref_global["delta_iou"]),
    }
    verdict["adopt"] = bool(
        verdict["beats_visual_event"]
        and verdict["beats_terrain_aggregate"]
        and verdict["beats_shuffled"]
        and verdict["beats_global_threshold"]
    )
    print("\n裁决：" + json.dumps(verdict, ensure_ascii=False, indent=2))

    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "arms.csv", index=False)
    events.to_csv(args.outdir / "event_table.csv", index=False)
    payload = {
        "schema_version": "pild_event_criterion_calibration.v1",
        "evidence_status": "development: leave-one-event-out on already-opened folds",
        "n_objects": int(len(frame)),
        "n_events": int(len(events)),
        "baseline_iou": base_iou,
        "abstain_events": int(events.abstain.sum()),
        "tau_star_quantiles": {
            q: float(events.tau_star.quantile(v))
            for q, v in {"p25": 0.25, "p50": 0.5, "p75": 0.75}.items()
        },
        "reference": {"global_threshold": ref_global, "event_oracle": ref_oracle},
        "arms": rows,
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n写出 {args.outdir}")


if __name__ == "__main__":
    main()
