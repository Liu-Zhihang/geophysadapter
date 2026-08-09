#!/usr/bin/env bash
# 替代视觉锚点矩阵：三个非 Prithvi 骨干 x 四折，双卡并行。
#
# 目的是让"对象级物理审查是否依赖特定视觉锚点"成为可判决的问题。
# 数据契约、划分、种子、采样温度、优化预算、阈值规则全部与 Prithvi 锚点一致。
#
# 权重已在本地 HuggingFace 缓存中，故强制离线，避免代理导致的失败。
set -uo pipefail

ROOT="/mnt/data_hdd/滑坡检测/physics_informed_landslide_dataset"
PY="/home/jinlin/miniconda3/envs/dpl/bin/python"
SCRIPT="${ROOT}/scripts/xdomain/train_pild_alt_visual_anchor_v1.py"
LOGDIR="${ROOT}/experiments/revision2026/pild_alt_anchor_runs_v1"
mkdir -p "${LOGDIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_backbone () {
  local gpu="$1" backbone="$2"
  echo "[start] gpu=${gpu} backbone=${backbone} $(date -Is)" >> "${LOGDIR}/matrix.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT}" \
    --backbone "${backbone}" \
    --epochs 30 \
    > "${LOGDIR}/${backbone}.log" 2>&1
  echo "[done ] gpu=${gpu} backbone=${backbone} exit=$? $(date -Is)" >> "${LOGDIR}/matrix.log"
}

# GPU 0 串行跑两个，GPU 1 跑一个，总时长由 GPU 0 决定
(
  run_backbone 0 hiera_small_mae_fpn
  run_backbone 0 fcmae_convnextv2_tiny_fpn
) &
(
  run_backbone 1 dinov2_s_fpn
) &
wait
echo "[matrix] complete $(date -Is)" >> "${LOGDIR}/matrix.log"
