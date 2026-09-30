#!/bin/bash
# 训练融合网络的完整流程脚本
# 依次训练：cam -> lidar -> 合并权重 -> fusion

set -e  # 遇到错误立即退出

# 配置路径 (small_range: ResNet50 backbone)
BASE_DIR="${LVFUSION_OCC_ROOT:?Set LVFUSION_OCC_ROOT to the repository root}"
CONFIG_DIR="${BASE_DIR}/configs/hascolv/fusion_config/small_range"
WORK_DIRS="${BASE_DIR}/work_dirs"

# 配置文件路径（small_range 下的 3 个网络配置: cam / lidar / fusion）
CAM_CONFIG="${CONFIG_DIR}/hasco_sr_cam_r50_s1080x1920_v015_lvfusion_fpn_centerhead.py"
LIDAR_CONFIG="${CONFIG_DIR}/hasco_sr_lidar_r50_s1080x1920_v015_lvfusion_fpn_centerhead.py"
FUSION_CONFIG="${CONFIG_DIR}/hasco_sr_fusion_r50_s1080x1920_v015_lvfusion_fpn_centerhead.py"

# 权重路径
CAM_WEIGHT="${WORK_DIRS}/hasco_sr_cam_r50_s1080x1920_v015_lvfusion_fpn_centerhead/latest.pth"
LIDAR_WEIGHT="${WORK_DIRS}/hasco_sr_lidar_r50_s1080x1920_v015_lvfusion_fpn_centerhead/latest.pth"
FUSION_PRETRAIN="${CONFIG_DIR}/fusion_pretrain.pth"

# 融合权重脚本
MERGE_SCRIPT="${BASE_DIR}/tools/merge_fusion_weights.py"

# 训练参数
CUDA_DEVICE=0
NUM_GPUS=1
MASTER_PORT=29503

echo "=========================================="
echo "开始训练融合网络流程"
echo "=========================================="
echo ""

# 步骤1: 训练图像分支 (cam)
echo "=========================================="
echo "步骤 1/4: 训练图像分支 (cam)"
echo "=========================================="
echo "配置文件: ${CAM_CONFIG}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    tools/train.py \
    ${CAM_CONFIG} \
    --gpus ${NUM_GPUS} \
    --launcher="pytorch"

if [ ! -f "${CAM_WEIGHT}" ]; then
    echo "错误: 图像分支权重文件不存在: ${CAM_WEIGHT}"
    echo "请检查训练是否成功完成，并确认 epoch_12.pth 已生成"
    exit 1
fi

echo ""
echo "✓ 图像分支训练完成，权重保存在: ${CAM_WEIGHT}"
echo ""

# 步骤2: 训练点云分支 (lidar)
echo "=========================================="
echo "步骤 2/4: 训练点云分支 (lidar)"
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

if [ ! -f "${LIDAR_WEIGHT}" ]; then
    echo "错误: 点云分支权重文件不存在: ${LIDAR_WEIGHT}"
    echo "请检查训练是否成功完成，并确认 epoch_12.pth 已生成"
    exit 1
fi

echo ""
echo "✓ 点云分支训练完成，权重保存在: ${LIDAR_WEIGHT}"
echo ""

# 步骤3: 合并权重
echo "=========================================="
echo "步骤 3/4: 合并图像和点云权重"
echo "=========================================="
echo "图像权重: ${CAM_WEIGHT}"
echo "点云权重: ${LIDAR_WEIGHT}"
echo "输出路径: ${FUSION_PRETRAIN}"
echo ""

python ${MERGE_SCRIPT} \
    --img-checkpoint ${CAM_WEIGHT} \
    --lidar-checkpoint ${LIDAR_WEIGHT} \
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
echo "  融合预训练权重: ${FUSION_PRETRAIN}"
echo "  融合网络权重: ${WORK_DIRS}/hasco_sr_fusion_r50_s1080x1920_v015_lvfusion_fpn_centerhead/"
echo ""
