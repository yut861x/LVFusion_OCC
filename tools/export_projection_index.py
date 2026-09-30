# Copyright (c) OpenMMLab. All rights reserved.
"""
导出投影索引参数（valid, x, y）用于模型量化部署
这些参数是固定的，基于配置的voxel大小和图像尺寸预计算

python tools/export_projection_index.py \
    --config configs/hascolv/mono_2class_fastbev1v_m0_r18_s256x704_v200x200x4_c192_d2.py \
    --checkpoint ./configs/hascolv/weights/mono_2class_fastbev1v_m0_r18_s256x704_v200x200x4_c192_d2.pth \
    --out-dir ./output/onnx/index \
    --sample-idx 0
"""
import argparse
import os
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models.detectors.fastbev1v import get_points, backproject_inplace


def parse_args():
    parser = argparse.ArgumentParser(description='Export projection indices for FastBEV1V')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file (optional, for model initialization)')
    parser.add_argument('--out-dir', type=str, default='./output/onnx/index', help='output directory for indices')
    parser.add_argument('--sample-idx', type=int, default=0, help='sample index to use for computing projection')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 704], help='input image size [H, W]')
    parser.add_argument('--feat-size', type=int, nargs=2, default=None, help='feature map size [H, W] (auto-computed if not specified)')
    return parser.parse_args()


def compute_projection_indices(model, img_meta, img_size, feat_size, device='cuda'):
    """
    计算投影索引
    
    Args:
        model: FastBEV1V模型
        img_meta: 图像元数据（包含lidar2img信息）
        img_size: 输入图像尺寸 [H, W]
        feat_size: 特征图尺寸 [H, W]
        device: 计算设备
    
    Returns:
        valid: [1, N] bool tensor，表示哪些voxel是有效的
        x: [1, N] int tensor，2D特征图的x坐标
        y: [1, N] int tensor，2D特征图的y坐标
    """
    # 获取模型配置
    n_voxels = model.n_voxels
    voxel_size = model.voxel_size
    
    # 确保n_voxels和voxel_size是列表格式，然后转换为tensor
    if isinstance(n_voxels, list):
        if isinstance(n_voxels[0], list):
            n_voxels = n_voxels[0]  # 取第一个元素
    if isinstance(voxel_size, list):
        if isinstance(voxel_size[0], list):
            voxel_size = voxel_size[0]  # 取第一个元素
    
    n_voxels_tensor = torch.tensor(n_voxels, device=device)
    voxel_size_tensor = torch.tensor(voxel_size, device=device)
    origin = torch.tensor(img_meta["lidar2img"]["origin"], device=device)
    
    # 计算stride
    stride = img_size[1] / feat_size[1]  # 使用宽度计算stride
    
    # 计算投影矩阵
    intrinsic = torch.tensor(
        img_meta["lidar2img"]["intrinsic"][:3, :3], device=device)
    intrinsic[:2] /= stride
    extrinsic = torch.tensor(
        img_meta["lidar2img"]["extrinsic"][0], device=device)  # CAM_FRONT for FastBEV1V
    projection = (intrinsic @ extrinsic[:3]).unsqueeze(0)  # [1, 3, 4]
    
    # 生成3D点云
    # 注意：get_points函数内部使用torch.arange创建grid，默认在CPU上
    # 我们需要确保所有参数都在同一设备上
    # 先获取n_voxels的值用于创建grid
    n_x, n_y, n_z = n_voxels_tensor.cpu().tolist()
    
    # 在目标设备上创建grid
    grid = torch.stack(torch.meshgrid(
        torch.arange(n_x, device=device),
        torch.arange(n_y, device=device),
        torch.arange(n_z, device=device)
    ))
    
    # 计算new_origin
    new_origin = origin - n_voxels_tensor.float() / 2 * voxel_size_tensor
    
    # 计算points
    points = grid * voxel_size_tensor.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)  # [3, X, Y, Z]
    
    # 计算投影
    n_channels = 64  # 特征通道数（neck_fuse输出）
    n_x_voxels, n_y_voxels, n_z_voxels = n_voxels_tensor.cpu().tolist()
    n_points = n_x_voxels * n_y_voxels * n_z_voxels
    
    # 准备点云 [1, 3, N]
    points_flat = points.view(1, 3, -1)  # [1, 3, N]
    # 添加齐次坐标 [1, 4, N]
    points_homo = torch.cat([points_flat, torch.ones(1, 1, n_points, device=device)], dim=1)
    
    # 投影到2D [1, 3, N]
    points_2d = torch.bmm(projection, points_homo)
    
    # 计算像素坐标
    x = (points_2d[:, 0] / points_2d[:, 2]).round().long()  # [1, N]
    y = (points_2d[:, 1] / points_2d[:, 2]).round().long()  # [1, N]
    z = points_2d[:, 2]  # [1, N]
    
    # 计算有效掩码
    height, width = feat_size
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [1, N]
    
    return valid, x, y


def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 构建数据集（用于获取img_meta）
    dataset = build_dataset(cfg.data.test)
    
    # 构建模型
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载checkpoint（如果提供）
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 获取样本数据
    if args.sample_idx >= len(dataset):
        print(f"Warning: sample_idx {args.sample_idx} >= dataset size {len(dataset)}, using sample 0")
        args.sample_idx = 0
    
    sample = dataset[args.sample_idx]
    img_meta = sample['img_metas'].data if hasattr(sample['img_metas'], 'data') else sample['img_metas']
    
    # 如果img_meta是list，取第一个
    if isinstance(img_meta, list):
        img_meta = img_meta[0]
    
    print(f"Using sample {args.sample_idx}")
    print(f"Image size: {args.img_size}")
    
    # 计算特征图尺寸（如果未指定）
    if args.feat_size is None:
        # 根据图像 backbone/neck 的 stride 计算
        # 对 LVFusion：使用 img_backbone / img_neck / img_neck_fuse
        print("Computing feature map size from model...")
        dummy_img = torch.randn(1, 3, args.img_size[0], args.img_size[1]).to(device)
        with torch.no_grad():
            # 兼容 LVFusion：优先使用 img_backbone/img_neck/img_neck_fuse
            if hasattr(model, 'img_backbone') and hasattr(model, 'img_neck'):
                feats = model.img_backbone(dummy_img)
                feats = model.img_neck(feats)
                feat = feats[0]
                from mmseg.ops import resize
                for i in range(1, len(feats)):
                    feat = torch.cat([
                        feat,
                        resize(feats[i], size=feat.shape[-2:], mode='bilinear')
                    ], dim=1)
                neck_fuse = getattr(model, 'img_neck_fuse', None)
                if neck_fuse is not None:
                    feat = neck_fuse(feat)
            else:
                # 兼容旧 FastBEV1V：使用 backbone/neck/neck_fuse
                feats = model.backbone(dummy_img)
                feats = model.neck(feats)
                feat = feats[0]
                from mmseg.ops import resize
                for i in range(1, len(feats)):
                    feat = torch.cat([
                        feat,
                        resize(feats[i], size=feat.shape[-2:], mode='bilinear')
                    ], dim=1)
                feat = model.neck_fuse(feat)

            feat_h, feat_w = feat.shape[-2:]
            args.feat_size = [feat_h, feat_w]
        print(f"Auto-computed feature size: {args.feat_size}")
    else:
        print(f"Using specified feature size: {args.feat_size}")
    
    # 计算投影索引
    print("Computing projection indices...")
    valid, x, y = compute_projection_indices(
        model, img_meta, args.img_size, args.feat_size, device=device
    )
    
    # 转换为numpy并保存
    valid_np = valid.cpu().numpy()
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    
    # 保存索引文件
    valid_path = os.path.join(args.out_dir, 'valid.npy')
    x_path = os.path.join(args.out_dir, 'x.npy')
    y_path = os.path.join(args.out_dir, 'y.npy')
    
    np.save(valid_path, valid_np)
    np.save(x_path, x_np)
    np.save(y_path, y_np)
    
    print(f"\nProjection indices saved to:")
    print(f"  valid: {valid_path} (shape: {valid_np.shape})")
    print(f"  x: {x_path} (shape: {x_np.shape})")
    print(f"  y: {y_path} (shape: {y_np.shape})")
    
    # 打印统计信息
    n_voxels = model.n_voxels
    total_voxels = n_voxels[0] * n_voxels[1] * n_voxels[2]
    valid_count = valid_np.sum()
    print(f"\nStatistics:")
    print(f"  Total voxels: {total_voxels}")
    print(f"  Valid voxels: {valid_count} ({100*valid_count/total_voxels:.2f}%)")
    print(f"  Invalid voxels: {total_voxels - valid_count} ({100*(total_voxels-valid_count)/total_voxels:.2f}%)")
    
    # 保存配置信息（用于参考）
    config_info = {
        'img_size': args.img_size,
        'feat_size': args.feat_size,
        'n_voxels': n_voxels,
        'voxel_size': model.voxel_size,
        'sample_idx': args.sample_idx,
    }
    import json
    config_path = os.path.join(args.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    print(f"\nConfig saved to: {config_path}")


if __name__ == '__main__':
    main()

