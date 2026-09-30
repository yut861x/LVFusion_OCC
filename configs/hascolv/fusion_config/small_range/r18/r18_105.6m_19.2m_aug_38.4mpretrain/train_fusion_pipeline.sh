#!/bin/bash
# 完整训练流程脚本（38.4m 预训练 -> 19.2m fusion）
# 阶段一 (38.4m): lidar -> cam -> merge -> fusion
# 阶段二: 拷贝 38.4m fusion epoch_24 为 fusion_38.4m.pth
# 阶段三 (19.2m): fusion（加载 fusion_38.4m.pth 预训练权重）

set -e  # 遇到错误立即退出

BASE_DIR="${LVFUSION_OCC_ROOT:?Set LVFUSION_OCC_ROOT to the repository root}"
WORK_DIRS="${BASE_DIR}/work_dirs"

# 38.4m 配置目录
CONFIG_DIR_38_4M="${BASE_DIR}/configs/hascolv/fusion_config/small_range/r18/r18_105.6m_38.4m_aug"
CAM_CONFIG_38_4M="${CONFIG_DIR_38_4M}/hasco_sr_cam_r18_105.6m_38.4m_aug_s1080x1920_v015_lvfusion_fpn_centerhead.py"
LIDAR_CONFIG_38_4M="${CONFIG_DIR_38_4M}/hasco_sr_lidar_r18_105.6m_38.4m_aug_s1080x1920_v015_lvfusion_fpn_centerhead.py"
FUSION_CONFIG_38_4M="${CONFIG_DIR_38_4M}/hasco_sr_fusion_r18_105.6m_38.4m_aug_s1080x1920_v015_lvfusion_fpn_centerhead.py"

CAM_WORK_SUBDIR_38_4M="hasco_sr_cam_r18_105.6m_38.4m_aug_s1080x1920_v015_lvfusion_fpn_centerhead"
LIDAR_WORK_SUBDIR_38_4M="hasco_sr_lidar_r18_105.6m_38.4m_aug_s1080x1920_v015_lvfusion_fpn_centerhead"
FUSION_WORK_SUBDIR_38_4M="hasco_sr_fusion_r18_105.6m_38.4m_aug_s1080x1920_v015_lvfusion_fpn_centerhead"
LIDAR_FIXED_WEIGHT_38_4M="${CONFIG_DIR_38_4M}/lidar_pretrain.pth"
FUSION_PRETRAIN_38_4M="${CONFIG_DIR_38_4M}/fusion_pretrain.pth"

# 19.2m fusion（38.4m 预训练）配置目录
CONFIG_DIR="${BASE_DIR}/configs/hascolv/fusion_config/small_range/r18/r18_105.6m_19.2m_aug_38.4mpretrain"
FUSION_CONFIG="${CONFIG_DIR}/hasco_sr_fusion_r18_105.6m_19.2m_aug_38.4mpretrain_s1080x1920_v015_lvfusion_fpn_centerhead.py"
FUSION_WORK_SUBDIR="hasco_sr_fusion_r18_105.6m_19.2m_aug_38.4mpretrain_s1080x1920_v015_lvfusion_fpn_centerhead"
FUSION_38_4M_PRETRAIN="${CONFIG_DIR}/fusion_38.4m.pth"

MAX_EPOCH=24
MERGE_SCRIPT="${BASE_DIR}/tools/merge_fusion_weights.py"

# 训练参数
CUDA_DEVICE=0,1,3,4,6,7
NUM_GPUS=6
MASTER_PORT_38_4M=29588
MASTER_PORT_19_2M=29589

cd "${BASE_DIR}"

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

echo "=========================================="
echo "开始完整训练流程 (38.4m 预训练 -> 19.2m fusion)"
echo "=========================================="
echo ""

# ============================================================
# 阶段一: 38.4m 训练流程 (lidar -> cam -> merge -> fusion)
# ============================================================

echo "##########################################"
echo "# 阶段一: 38.4m 训练"
echo "##########################################"
echo ""

# 步骤 1/4: 训练 38.4m 点云分支
echo "=========================================="
echo "阶段一 步骤 1/4: 训练 38.4m 点云分支 (lidar)"
echo "=========================================="
echo "配置文件: ${LIDAR_CONFIG_38_4M}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT_38_4M} \
    tools/train.py \
    ${LIDAR_CONFIG_38_4M} \
    --gpus ${NUM_GPUS} \
    --launcher="pytorch"

LIDAR_WEIGHT_38_4M=$(wait_for_checkpoint "${LIDAR_WORK_SUBDIR_38_4M}" "${MAX_EPOCH}") || exit 1
cp "${LIDAR_WEIGHT_38_4M}" "${LIDAR_FIXED_WEIGHT_38_4M}"

echo ""
echo "✓ 38.4m 点云分支训练完成，权重: ${LIDAR_WEIGHT_38_4M}"
echo "✓ 已复制到: ${LIDAR_FIXED_WEIGHT_38_4M}"
echo ""

# 步骤 2/4: 训练 38.4m 图像分支
echo "=========================================="
echo "阶段一 步骤 2/4: 训练 38.4m 图像分支 (cam)"
echo "=========================================="
echo "配置文件: ${CAM_CONFIG_38_4M}"
echo "预训练权重: ${LIDAR_FIXED_WEIGHT_38_4M}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT_38_4M} \
    tools/train.py \
    ${CAM_CONFIG_38_4M} \
    --gpus ${NUM_GPUS} \
    --launcher="pytorch"

CAM_WEIGHT_38_4M=$(wait_for_checkpoint "${CAM_WORK_SUBDIR_38_4M}" "${MAX_EPOCH}") || exit 1

echo ""
echo "✓ 38.4m 图像分支训练完成，权重: ${CAM_WEIGHT_38_4M}"
echo ""

# 步骤 3/4: 合并 38.4m 图像和点云权重
echo "=========================================="
echo "阶段一 步骤 3/4: 合并 38.4m 图像和点云权重"
echo "=========================================="
echo "图像权重: ${CAM_WEIGHT_38_4M}"
echo "点云权重: ${LIDAR_FIXED_WEIGHT_38_4M}"
echo "输出路径: ${FUSION_PRETRAIN_38_4M}"
echo ""

python ${MERGE_SCRIPT} \
    --img-checkpoint ${CAM_WEIGHT_38_4M} \
    --lidar-checkpoint ${LIDAR_FIXED_WEIGHT_38_4M} \
    --output ${FUSION_PRETRAIN_38_4M} \
    --fusion-config ${FUSION_CONFIG_38_4M}

if [ ! -f "${FUSION_PRETRAIN_38_4M}" ]; then
    echo "错误: 38.4m 融合权重文件生成失败: ${FUSION_PRETRAIN_38_4M}"
    exit 1
fi

echo ""
echo "✓ 38.4m 权重合并完成: ${FUSION_PRETRAIN_38_4M}"
echo ""

# 步骤 4/4: 训练 38.4m 融合网络
echo "=========================================="
echo "阶段一 步骤 4/4: 训练 38.4m 融合网络 (fusion)"
echo "=========================================="
echo "配置文件: ${FUSION_CONFIG_38_4M}"
echo "预训练权重: ${FUSION_PRETRAIN_38_4M}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT_38_4M} \
    tools/train.py \
    ${FUSION_CONFIG_38_4M} \
    --gpus ${NUM_GPUS} \
    --launcher="pytorch"

FUSION_WEIGHT_38_4M=$(wait_for_checkpoint "${FUSION_WORK_SUBDIR_38_4M}" "${MAX_EPOCH}") || exit 1

echo ""
echo "✓ 38.4m 融合网络训练完成，权重: ${FUSION_WEIGHT_38_4M}"
echo ""

# ============================================================
# 阶段二: 拷贝 38.4m fusion 权重作为 19.2m fusion 预训练
# ============================================================

echo "##########################################"
echo "# 阶段二: 准备 19.2m fusion 预训练权重"
echo "##########################################"
echo ""

cp "${FUSION_WEIGHT_38_4M}" "${FUSION_38_4M_PRETRAIN}"

if [ ! -f "${FUSION_38_4M_PRETRAIN}" ]; then
    echo "错误: 预训练权重拷贝失败: ${FUSION_38_4M_PRETRAIN}"
    exit 1
fi

echo "✓ 已将 38.4m fusion epoch_${MAX_EPOCH} 拷贝到: ${FUSION_38_4M_PRETRAIN}"
echo ""

# ============================================================
# 阶段三: 19.2m fusion 训练（加载 fusion_38.4m.pth）
# ============================================================

echo "##########################################"
echo "# 阶段三: 19.2m fusion 训练 (38.4m 预训练)"
echo "##########################################"
echo ""

echo "=========================================="
echo "阶段三 步骤 1/1: 训练 19.2m 融合网络 (fusion)"
echo "=========================================="
echo "配置文件: ${FUSION_CONFIG}"
echo "预训练权重: ${FUSION_38_4M_PRETRAIN}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT_19_2M} \
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
echo "  [38.4m] 点云分支权重: ${LIDAR_WEIGHT_38_4M}"
echo "  [38.4m] 图像分支权重: ${CAM_WEIGHT_38_4M}"
echo "  [38.4m] 融合预训练权重: ${FUSION_PRETRAIN_38_4M}"
echo "  [38.4m] 融合网络权重: ${FUSION_WEIGHT_38_4M}"
echo "  [19.2m] 38.4m 预训练权重: ${FUSION_38_4M_PRETRAIN}"
echo "  [19.2m] 融合网络权重: ${WORK_DIRS}/${FUSION_WORK_SUBDIR}/"
echo ""
