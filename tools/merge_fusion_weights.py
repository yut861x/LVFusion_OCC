#!/usr/bin/env python3
"""
合并图像分支和点云分支的权重，生成融合网络的预训练权重。

用法:
    python tools/merge_fusion_weights.py
"""

import torch
import argparse
from pathlib import Path


def merge_fusion_weights(
    img_checkpoint_path,
    lidar_checkpoint_path,
    output_path,
    fusion_config_path=None
):
    """
    合并图像和点云权重为融合网络权重。
    
    Args:
        img_checkpoint_path: 图像分支权重路径
        lidar_checkpoint_path: 点云分支权重路径
        output_path: 输出权重路径
        fusion_config_path: 融合网络配置文件路径（可选，用于验证）
    """
    print(f"Loading image checkpoint from: {img_checkpoint_path}")
    img_ckpt = torch.load(img_checkpoint_path, map_location='cpu')
    img_state_dict = img_ckpt.get('state_dict', img_ckpt)
    
    print(f"Loading lidar checkpoint from: {lidar_checkpoint_path}")
    lidar_ckpt = torch.load(lidar_checkpoint_path, map_location='cpu')
    lidar_state_dict = lidar_ckpt.get('state_dict', lidar_ckpt)
    
    # 创建融合网络的state_dict
    fusion_state_dict = {}
    
    # 1. 从图像权重中提取图像分支的权重 (img_*)
    print("\nExtracting image branch weights...")
    img_keys = [k for k in img_state_dict.keys() if k.startswith('img_')]
    for key in img_keys:
        fusion_state_dict[key] = img_state_dict[key].clone()
        print(f"  Added: {key}")
    
    # 2. 从点云权重中提取点云分支的权重 (pts_*)
    print("\nExtracting point cloud branch weights...")
    pts_keys = [k for k in lidar_state_dict.keys() if k.startswith('pts_')]
    for key in pts_keys:
        fusion_state_dict[key] = lidar_state_dict[key].clone()
        print(f"  Added: {key}")
    
    # 3. 处理融合层 (fusion_layer)
    # 融合网络的fusion_layer输入通道是 256+256=512，输出是256
    # 如果图像权重中有fusion_layer，需要检查其输入通道数
    print("\nProcessing fusion layer...")
    fusion_layer_key = 'fusion_layer.weight'
    if fusion_layer_key in img_state_dict:
        img_fusion = img_state_dict[fusion_layer_key]
        # 检查输入通道数
        if img_fusion.shape[1] == 256:
            # 图像网络的fusion_layer输入是256（只有图像），需要扩展为512
            print(f"  Found fusion_layer in image checkpoint with input channels: {img_fusion.shape[1]}")
            print(f"  Expanding to 512 input channels for fusion network...")
            # 初始化新的融合层：输入512，输出256
            # 将图像分支的权重放在前256通道，点云分支的权重初始化为零或复制
            new_fusion_weight = torch.zeros(256, 512, 1, 1)
            new_fusion_weight[:, :256, :, :] = img_fusion  # 图像分支权重
            new_fusion_weight[:, 256:, :, :] = img_fusion * 0.5  # 点云分支权重（初始化为图像权重的一半）
            fusion_state_dict[fusion_layer_key] = new_fusion_weight
            print(f"  Created fusion_layer: {new_fusion_weight.shape}")
        elif img_fusion.shape[1] == 512:
            # 已经是512通道，直接使用
            fusion_state_dict[fusion_layer_key] = img_fusion.clone()
            print(f"  Using fusion_layer from image checkpoint: {img_fusion.shape}")
        else:
            print(f"  Warning: Unexpected fusion_layer input channels: {img_fusion.shape[1]}")
            # 初始化新的融合层
            new_fusion_weight = torch.zeros(256, 512, 1, 1)
            # 使用Xavier初始化
            torch.nn.init.xavier_uniform_(new_fusion_weight)
            fusion_state_dict[fusion_layer_key] = new_fusion_weight
            print(f"  Initialized new fusion_layer: {new_fusion_weight.shape}")
    else:
        # 没有找到fusion_layer，需要初始化
        print("  No fusion_layer found, initializing new one...")
        new_fusion_weight = torch.zeros(256, 512, 1, 1)
        torch.nn.init.xavier_uniform_(new_fusion_weight)
        fusion_state_dict[fusion_layer_key] = new_fusion_weight
        print(f"  Initialized fusion_layer: {new_fusion_weight.shape}")
    
    # 处理fusion_layer的bias（如果有）
    fusion_bias_key = 'fusion_layer.bias'
    if fusion_bias_key in img_state_dict:
        fusion_state_dict[fusion_bias_key] = img_state_dict[fusion_bias_key].clone()
        print(f"  Added fusion_layer bias")
    else:
        fusion_state_dict[fusion_bias_key] = torch.zeros(256)
        print(f"  Initialized fusion_layer bias")
    
    # 4. 处理检测头 (pts_bbox_head)
    # 融合网络的检测头输入是256（融合后的特征）
    # 图像和点云网络的检测头输入也是256，所以可以直接使用其中一个
    print("\nProcessing bbox head...")
    bbox_head_keys = [k for k in img_state_dict.keys() if k.startswith('pts_bbox_head.')]
    if not bbox_head_keys:
        # 如果图像权重中没有，尝试从点云权重中获取
        bbox_head_keys = [k for k in lidar_state_dict.keys() if k.startswith('pts_bbox_head.')]
        source_dict = lidar_state_dict
        print("  Using bbox_head from lidar checkpoint")
    else:
        source_dict = img_state_dict
        print("  Using bbox_head from image checkpoint")

    # bbox_head_keys = [k for k in lidar_state_dict.keys() if k.startswith('pts_bbox_head.')]
    # # bbox_head_keys = None
    # if not bbox_head_keys:
    #     # 如果点云权重中没有，尝试从图像权重中获取
    #     bbox_head_keys = [k for k in img_state_dict.keys() if k.startswith('pts_bbox_head.')]
    #     source_dict = img_state_dict
    #     print("  Using bbox_head from image checkpoint")
    # else:
    #     source_dict = lidar_state_dict
    #     print("  Using bbox_head from lidar checkpoint")
    
    for key in bbox_head_keys:
        fusion_state_dict[key] = source_dict[key].clone()
        print(f"  Added: {key}")
    
    # 5. 保存合并后的权重
    print(f"\nSaving merged checkpoint to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建新的checkpoint，保留元数据
    merged_ckpt = {
        'state_dict': fusion_state_dict,
        'meta': {
            'img_checkpoint': str(img_checkpoint_path),
            'lidar_checkpoint': str(lidar_checkpoint_path),
            'merged_keys': {
                'img_keys': len(img_keys),
                'pts_keys': len(pts_keys),
                'bbox_head_keys': len(bbox_head_keys),
                'total_keys': len(fusion_state_dict)
            }
        }
    }
    
    # 如果有epoch信息，也保留
    if 'epoch' in img_ckpt:
        merged_ckpt['epoch'] = img_ckpt['epoch']
    elif 'epoch' in lidar_ckpt:
        merged_ckpt['epoch'] = lidar_ckpt['epoch']
    
    torch.save(merged_ckpt, output_path)
    
    print(f"\n✓ Successfully merged weights!")
    print(f"  Total keys in merged checkpoint: {len(fusion_state_dict)}")
    print(f"  Image branch keys: {len(img_keys)}")
    print(f"  Point cloud branch keys: {len(pts_keys)}")
    print(f"  Bbox head keys: {len(bbox_head_keys)}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge image and lidar weights for fusion network'
    )
    parser.add_argument(
        '--img-checkpoint',
        type=str,
        default='./work_dirs/hasco_cam_r50_s1080x1920_v04_lvfusion_fpn_centerhead/epoch_24.pth',
        help='Path to image branch checkpoint'
    )
    parser.add_argument(
        '--lidar-checkpoint',
        type=str,
        default='./work_dirs/hasco_lidar_r50_s1080x1920_v04_lvfusion_fpn_centerhead/epoch_24.pth',
        help='Path to lidar branch checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./configs/hascolv/fusion_config/fusion_pretrain.pth',
        help='Path to save merged checkpoint'
    )
    parser.add_argument(
        '--fusion-config',
        type=str,
        default=None,
        help='Path to fusion network config file (optional)'
    )
    
    args = parser.parse_args()
    
    merge_fusion_weights(
        args.img_checkpoint,
        args.lidar_checkpoint,
        args.output,
        args.fusion_config
    )


if __name__ == '__main__':
    main()
