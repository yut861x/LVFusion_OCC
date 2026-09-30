#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成2D到3D投影的索引文件（用于C++部署）
生成文件：
- gather_new_0.bin: gather索引（从2D特征图提取）
- scatter_nd_new_0.bin: scatter索引（投影到3D BEV空间）
- featurePointLength.bin: 每个相机的有效投影点数量
适配：单相机单帧模型
"""
import argparse
import os
import sys
import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
def parse_args():
    parser = argparse.ArgumentParser(description='Generate projection indices for C++ deployment')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--output-dir', type=str, default='output/projection_indices', 
                        help='Output directory for index files')
    parser.add_argument('--sample-idx', type=int, default=0, 
                        help='Sample index to use for generating indices (default: 0)')
    return parser.parse_args()
def get_points_python(n_voxels, voxel_size, origin):
    """生成BEV网格点"""
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ],
            indexing='ij'
        )
    )
    new_origin = origin - torch.tensor(n_voxels) / 2.0 * torch.tensor(voxel_size)
    points = points * torch.tensor(voxel_size).view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points
def compute_projection_python(img_meta, stride, noise=0):
    """计算投影矩阵"""
    projection = []
    intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
    intrinsic[:2] /= stride
    extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
    for extrinsic in extrinsics:
        if noise > 0:
            projection.append(intrinsic @ extrinsic[:3] + noise)
        else:
            projection.append(intrinsic @ extrinsic[:3])
    return torch.stack(projection)
def generate_indices(points, projection, height, width, n_voxels):
    """
    生成gather和scatter索引
    Args:
        points: [3, n_x, n_y, n_z] BEV网格点
        projection: [1, 3, 4] 投影矩阵（单相机）
        height: 特征图高度
        width: 特征图宽度
        n_voxels: [n_x, n_y, n_z] voxel数量
    Returns:
        gather_indices: [N_valid] 2D特征图的线性索引
        scatter_indices: [N_valid] 3D volume的线性索引
        n_valid: 有效投影点数量
    """
    n_x, n_y, n_z = n_voxels
    # 将points reshape为 [3, N_total]
    points_3d = points.view(3, -1)  # [3, n_x*n_y*n_z]
    # 转换为齐次坐标 [4, N_total]
    points_homo = torch.cat([points_3d, torch.ones(1, points_3d.shape[1])], dim=0)  # [4, N_total]
    # 投影到图像平面 [1, 3, 4] @ [4, N_total] -> [1, 3, N_total]
    # projection: [1, 3, 4], points_homo: [4, N_total]
    projection = projection.squeeze(0)  # [3, 4]
    points_2d_3 = projection @ points_homo  # [3, 4] @ [4, N_total] -> [3, N_total]
    # 计算2D坐标
    x = (points_2d_3[0] / points_2d_3[2]).round().long()  # [N_total]
    y = (points_2d_3[1] / points_2d_3[2]).round().long()  # [N_total]
    z = points_2d_3[2]  # [N_total]
    # 计算有效mask
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    # 获取有效点的索引
    valid_indices = torch.where(valid)[0]  # [N_valid]
    n_valid = valid_indices.shape[0]
    if n_valid == 0:
        print("警告: 没有有效的投影点!")
        return None, None, 0
    # 计算gather索引（2D特征图的线性索引）
    x_valid = x[valid_indices]  # [N_valid]
    y_valid = y[valid_indices]  # [N_valid]
    gather_indices = y_valid * width + x_valid  # [N_valid]
    # 计算scatter索引（3D volume的线性索引）
    # 需要找到每个有效点在原始volume中的位置
    # points_3d的索引顺序是: x, y, z (meshgrid的ij模式)
    # volume的reshape顺序是: [C, n_x, n_y, n_z] -> [C, n_x*n_y*n_z]
    # 所以线性索引是: idx = x + y*n_x + z*n_x*n_y
    # 从valid_indices恢复x, y, z的voxel坐标
    # valid_indices是points_3d的线性索引
    # points_3d的索引: idx = x_vox + y_vox*n_x + z_vox*n_x*n_y
    x_vox = valid_indices % n_x
    y_vox = (valid_indices // n_x) % n_y
    z_vox = valid_indices // (n_x * n_y)
    # 计算3D volume的线性索引
    # 注意：C++代码中的volume是 [C, n_x, n_y, n_z]，reshape为 [C, n_x*n_y*n_z]
    # 索引顺序: idx = x_vox + y_vox*n_x + z_vox*n_x*n_y
    scatter_indices = x_vox + y_vox * n_x + z_vox * n_x * n_y  # [N_valid]
    return gather_indices.cpu().numpy().astype(np.int32), scatter_indices.cpu().numpy().astype(np.int32), n_valid
def main():
    args = parse_args()
    # 加载配置
    cfg = Config.fromfile(args.config)
    # 获取配置参数
    n_voxels = cfg.model.get('n_voxels', [[200, 200, 4]])[0]
    voxel_size = cfg.model.get('voxel_size', [[0.5, 0.5, 1.5]])[0]
    print(f"配置信息:")
    print(f"  n_voxels: {n_voxels}")
    print(f"  voxel_size: {voxel_size}")
    # 加载数据集（获取一个样本的img_meta）
    dataset = build_dataset(cfg.data.test)
    if args.sample_idx >= len(dataset):
        print(f"错误: 样本索引 {args.sample_idx} 超出范围 (数据集大小: {len(dataset)})")
        return
    # 获取样本数据
    data = dataset[args.sample_idx]
    if 'img_metas' in data:
        img_metas = [data['img_metas'].data] if hasattr(data['img_metas'], 'data') else [data['img_metas']]
        img_meta = img_metas[0]
    else:
        print("错误: 数据中没有找到 'img_metas'")
        return
    # 从img_meta中获取实际图像尺寸
    # 先尝试从配置中获取
    img_height = 256
    img_width = 704
    # 尝试从配置中获取input_size
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'test'):
        test_pipeline = cfg.data.test.get('pipeline', [])
        for transform in test_pipeline:
            if isinstance(transform, dict) and transform.get('type') == 'Resize':
                if 'size' in transform:
                    size = transform['size']
                    if isinstance(size, (list, tuple)) and len(size) >= 2:
                        img_height = int(size[0])
                        img_width = int(size[1])
                        print(f"  从配置Resize中获取图像尺寸: {img_height}x{img_width}")
                        break
    # 尝试从img_meta中获取
    img_shape = img_meta.get('img_shape', None)
    if img_shape is not None:
        try:
            # 递归展开嵌套结构，直到找到实际的尺寸值
            while True:
                if isinstance(img_shape, list):
                    if len(img_shape) == 0:
                        break
                    img_shape = img_shape[0]
                elif isinstance(img_shape, tuple):
                    if len(img_shape) == 0:
                        break
                    # 如果tuple的第一个元素也是tuple/list，继续展开
                    if isinstance(img_shape[0], (tuple, list)):
                        img_shape = img_shape[0]
                    else:
                        # 已经展开到实际的尺寸值
                        break
                elif isinstance(img_shape, np.ndarray):
                    img_shape = img_shape.tolist()
                else:
                    break
            
            # 现在img_shape应该是包含两个数字的tuple或list
            if isinstance(img_shape, (list, tuple, np.ndarray)):
                shape_list = list(img_shape)
                if len(shape_list) >= 2:
                    img_height = int(shape_list[0])
                    img_width = int(shape_list[1])
                    print(f"  从img_meta中获取图像尺寸: {img_height}x{img_width}")
                elif len(shape_list) == 1:
                    # 如果只有一个元素，可能是正方形图像
                    img_height = img_width = int(shape_list[0])
                    print(f"  从img_meta中获取图像尺寸（正方形）: {img_height}x{img_width}")
        except (IndexError, TypeError, ValueError) as e:
            print(f"  警告: 无法从img_meta解析img_shape: {e}, 使用默认值 {img_height}x{img_width}")
    # 如果还是默认值，尝试从其他字段获取
    if img_height == 256 and img_width == 704:
        # 尝试从img_meta的其他字段获取
        if 'ori_shape' in img_meta:
            ori_shape = img_meta['ori_shape']
            try:
                # 递归展开嵌套结构
                while True:
                    if isinstance(ori_shape, list):
                        if len(ori_shape) == 0:
                            break
                        ori_shape = ori_shape[0]
                    elif isinstance(ori_shape, tuple):
                        if len(ori_shape) == 0:
                            break
                        if isinstance(ori_shape[0], (tuple, list)):
                            ori_shape = ori_shape[0]
                        else:
                            break
                    elif isinstance(ori_shape, np.ndarray):
                        ori_shape = ori_shape.tolist()
                    else:
                        break
                
                # 现在ori_shape应该是包含两个数字的tuple或list
                if isinstance(ori_shape, (list, tuple, np.ndarray)):
                    shape_list = list(ori_shape)
                    if len(shape_list) >= 2:
                        img_height = int(shape_list[0])
                        img_width = int(shape_list[1])
                        print(f"  从ori_shape中获取图像尺寸: {img_height}x{img_width}")
            except (IndexError, TypeError, ValueError) as e:
                print(f"  警告: 无法从ori_shape解析图像尺寸: {e}")
    stride = 4
    feat_height = img_height // stride  # 64
    feat_width = img_width // stride   # 176
    print(f"  图像尺寸: {img_height}x{img_width}")
    print(f"  特征图尺寸: {feat_height}x{feat_width}")
    print(f"使用样本 {args.sample_idx} 的相机参数")
    print(f"  相机: {img_meta.get('filename', 'N/A')}")
    # 计算投影矩阵
    projection_all = compute_projection_python(img_meta, stride, noise=0)
    print(f"投影矩阵形状（所有相机）: {projection_all.shape}")
    # 单相机模式：只使用第一个相机的投影矩阵
    # 如果img_meta中只有单个相机，projection_all可能是[1, 3, 4]
    # 如果是多相机，取第一个
    if projection_all.shape[0] > 1:
        projection_single = projection_all[0:1]  # [1, 3, 4]
    else:
        projection_single = projection_all  # 已经是[1, 3, 4]
    print(f"投影矩阵形状（单相机）: {projection_single.shape}")
    # 生成BEV网格点
    origin = torch.tensor(img_meta["lidar2img"]["origin"])
    points = get_points_python(n_voxels, voxel_size, origin)
    print(f"BEV网格点形状: {points.shape}")
    # 生成索引（单相机）
    gather_indices, scatter_indices, n_valid = generate_indices(
        points, projection_single, feat_height, feat_width, n_voxels
    )
    if n_valid == 0:
        print("错误: 没有生成有效的索引!")
        return
    print(f"有效投影点数量: {n_valid}")
    print(f"gather索引范围: [{gather_indices.min()}, {gather_indices.max()}]")
    print(f"scatter索引范围: [{scatter_indices.min()}, {scatter_indices.max()}]")
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 保存featurePointLength.bin
    # 格式: int32数组，长度为BATCH_NUM（单相机为1）
    feature_point_length = np.array([n_valid], dtype=np.int32)
    feature_point_length_path = os.path.join(args.output_dir, 'featurePointLength.bin')
    feature_point_length.tofile(feature_point_length_path)
    print(f"✓ 保存 featurePointLength.bin: {n_valid} 个有效点")
    # 保存gather_new_0.bin
    gather_path = os.path.join(args.output_dir, 'gather_new_0.bin')
    gather_indices.tofile(gather_path)
    print(f"✓ 保存 gather_new_0.bin: {len(gather_indices)} 个索引")
    # 保存scatter_nd_new_0.bin
    scatter_path = os.path.join(args.output_dir, 'scatter_nd_new_0.bin')
    scatter_indices.tofile(scatter_path)
    print(f"✓ 保存 scatter_nd_new_0.bin: {len(scatter_indices)} 个索引")
    # 验证文件大小
    gather_size = os.path.getsize(gather_path)
    scatter_size = os.path.getsize(scatter_path)
    feature_size = os.path.getsize(feature_point_length_path)
    print(f"\n文件大小验证:")
    print(f"  featurePointLength.bin: {feature_size} bytes (期望: {4} bytes)")
    print(f"  gather_new_0.bin: {gather_size} bytes (期望: {n_valid * 4} bytes)")
    print(f"  scatter_nd_new_0.bin: {scatter_size} bytes (期望: {n_valid * 4} bytes)")
    print(f"\n✓ 索引文件生成完成!")
    print(f"输出目录: {args.output_dir}")
if __name__ == '__main__':
    main()