#!/bin/bash
# 训练融合网络的完整流程脚本
# 依次训练：lidar -> 拷贝 lidar 权重 -> cam 加载 lidar 预训练 -> 合并权重 -> fusion

set -e  # 遇到错误立即退出

# 配置路径
BASE_DIR="${LVFUSION_OCC_ROOT:?Set LVFUSION_OCC_ROOT to the repository root}"
CONFIG_DIR="${BASE_DIR}/configs/hascolv/fusion_config/small_range/r18/r18_105.6m"
WORK_DIRS="${BASE_DIR}/work_dirs"

# 配置文件路径（cam / lidar / fusion）
CAM_CONFIG="${CONFIG_DIR}/hasco_sr_cam_r18_105.6m_s1080x1920_v015_lvfusion_fpn_centerhead.py"
LIDAR_CONFIG="${CONFIG_DIR}/hasco_sr_lidar_r18_105.6m_s1080x1920_v015_lvfusion_fpn_centerhead.py"
FUSION_CONFIG="${CONFIG_DIR}/hasco_sr_fusion_r18_105.6m_s1080x1920_v015_lvfusion_fpn_centerhead.py"

# 权重路径
CAM_WORK_SUBDIR="hasco_sr_cam_r18_105.6m_s1080x1920_v015_lvfusion_fpn_centerhead"
LIDAR_WORK_SUBDIR="hasco_sr_lidar_r18_105.6m_s1080x1920_v015_lvfusion_fpn_centerhead"
MAX_EPOCH=24
LIDAR_FIXED_WEIGHT="${CONFIG_DIR}/lidar_pretrain.pth"
FUSION_PRETRAIN="${CONFIG_DIR}/fusion_pretrain.pth"

# 等待 rank0 写完 checkpoint（大文件 + 分布式退出时偶发可见性延迟）
wait_for_checkpoint() {
    local work_subdir="$1"
    local preferred_epoch="$2"
    local ckpt_dir="${WORK_DIRS}/${work_subdir}"
    local preferred="${ckpt_dir}/epoch_${preferred_epoch}.pth"
    local latest="${ckpt_dir}/latest.pth"
    local max_wait=120
    local elapsed=0

    while [ "${elapsed}" -lt "${max_wait}" ]; do
        if [ -f "${preferred}" ]; then
            echo "${preferred}"
            return 0
        fi
        if [ -e "${latest}" ]; then
            echo "${latest}"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo "错误: 在 ${ckpt_dir} 未找到 epoch_${preferred_epoch}.pth 或 latest.pth（已等待 ${max_wait}s）" >&2
    return 1
}

# 融合权重脚本
MERGE_SCRIPT="${BASE_DIR}/tools/merge_fusion_weights.py"

# 训练参数
CUDA_DEVICE=0
NUM_GPUS=1
MASTER_PORT=29513

echo "=========================================="
echo "开始训练融合网络流程"
echo "=========================================="
echo ""

# 步骤1: 训练点云分支 (lidar)
echo "=========================================="
echo "步骤 1/4: 训练点云分支 (lidar)"
echo "=========================================="
echo "配置文件: ${LIDAR_CONFIG}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    tools/train.py \
    ${LIDAR_CONFIG} \
    --gpus ${NUM_GPUS} \
    --launcher="pytorch"

LIDAR_WEIGHT=$(wait_for_checkpoint "${LIDAR_WORK_SUBDIR}" "${MAX_EPOCH}") || exit 1

cp "${LIDAR_WEIGHT}" "${LIDAR_FIXED_WEIGHT}"

echo ""
echo "✓ 点云分支训练完成，权重保存在: ${LIDAR_WEIGHT}"
echo "✓ 点云分支权重已复制到固定路径: ${LIDAR_FIXED_WEIGHT}"
echo ""

# 步骤2: 训练图像分支 (cam)
echo "=========================================="
echo "步骤 2/4: 训练图像分支 (cam)"
echo "=========================================="
echo "配置文件: ${CAM_CONFIG}"
echo "预训练权重: ${LIDAR_FIXED_WEIGHT}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    tools/train.py \
    ${CAM_CONFIG} \
    --gpus ${NUM_GPUS} \
    --launcher="pytorch"

CAM_WEIGHT=$(wait_for_checkpoint "${CAM_WORK_SUBDIR}" "${MAX_EPOCH}") || exit 1

echo ""
echo "✓ 图像分支训练完成，权重保存在: ${CAM_WEIGHT}"
echo ""

# 步骤3: 合并权重
echo "=========================================="
echo "步骤 3/4: 合并图像和点云权重"
echo "=========================================="
echo "图像权重: ${CAM_WEIGHT}"
echo "点云权重: ${LIDAR_FIXED_WEIGHT}"
echo "输出路径: ${FUSION_PRETRAIN}"
echo ""

python ${MERGE_SCRIPT} \
    --img-checkpoint ${CAM_WEIGHT} \
    --lidar-checkpoint ${LIDAR_FIXED_WEIGHT} \
    --output ${FUSION_PRETRAIN} \
    --fusion-config ${FUSION_CONFIG}

if [ ! -f "${FUSION_PRETRAIN}" ]; then
    echo "错误: 融合权重文件生成失败: ${FUSION_PRETRAIN}"
    exit 1
fi

echo ""
echo "✓ 权重合并完成，融合权重保存在: ${FUSION_PRETRAIN}"
echo ""

# 步骤4: 训练融合网络 (fusion)
echo "=========================================="
echo "步骤 4/4: 训练融合网络 (fusion)"
echo "=========================================="
echo "配置文件: ${FUSION_CONFIG}"
echo "预训练权重: ${FUSION_PRETRAIN}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    tools/train.py \
    ${FUSION_CONFIG} \
    --gpus ${NUM_GPUS} \
    --launcher="pytorch"

echo ""
echo "=========================================="
echo "✓ 所有训练步骤完成！"
echo "=========================================="
echo ""
echo "训练结果:"
echo "  图像分支权重: ${CAM_WEIGHT}"
echo "  点云分支权重: ${LIDAR_WEIGHT}"
echo "  点云固定预训练权重: ${LIDAR_FIXED_WEIGHT}"
echo "  融合预训练权重: ${FUSION_PRETRAIN}"
echo "  融合网络权重: ${WORK_DIRS}/hasco_sr_fusion_r18_105.6m_s1080x1920_v015_lvfusion_fpn_centerhead/"
echo ""
