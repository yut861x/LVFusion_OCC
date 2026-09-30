# Copyright (c) OpenMMLab. All rights reserved.
"""
ONNX导出脚本，参考fastbev_v0.0.1的实现
支持FastBEV1V模型的2D和3D部分分别导出
"""
import argparse
import os
import torch
import warnings
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model

import onnx
import onnxruntime
try:
    from onnxsim import simplify
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False
    warnings.warn("onnxsim not installed, will skip model simplification")

from mmdet3d.models import build_model
from mmseg.ops import resize


def parse_args():
    parser = argparse.ArgumentParser(description='Export FastBEV1V model to ONNX')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file')
    parser.add_argument('--out-dir', type=str, default='./output/onnx', help='output directory for ONNX models')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    parser.add_argument(
        '--export-2d',
        action='store_true',
        default=True,
        help='Export 2D feature extraction model (backbone + neck + neck_fuse)')
    parser.add_argument(
        '--export-3d',
        action='store_true',
        default=True,
        help='Export 3D processing model (neck_3d + bbox_head)')
    parser.add_argument(
        '--simplify',
        action='store_true',
        default=True,
        help='Simplify ONNX model using onnxsim')
    parser.add_argument(
        '--opset-version',
        type=int,
        default=13,
        help='ONNX opset version')
    return parser.parse_args()


def export_onnx_2d(model, config, output_path, simplify_model=True, opset_version=13):
    """
    导出2D特征提取模型 (backbone + neck + neck_fuse)
    输入: [1, 3, H, W] 单张图像
    输出: [1, 64, H', W'] 2D特征
    """
    print("=" * 60)
    print("Exporting 2D feature extraction model...")
    print("=" * 60)
    
    model.eval()
    
    # 确保模型和输入类型一致（如果模型是fp16，输入也应该是fp16）
    # 检查模型第一个参数的类型
    first_param = next(model.parameters())
    model_dtype = first_param.dtype
    
    # 准备输入：单张图像 [1, 3, H, W]
    input_size = config.data_config['input_size']  # (256, 704)
    print(f"Input size: {input_size}")
    print(f"Model dtype: {model_dtype}")
    dummy_img = torch.randn(1, 3, input_size[0], input_size[1]).cuda()
    # 确保输入类型与模型权重类型一致
    if model_dtype == torch.float16:
        dummy_img = dummy_img.half()
    else:
        dummy_img = dummy_img.float()
    
    # 创建包装器类，直接调用onnx_export_2d方法
    # 需要禁用fp16自动转换，确保输入类型与模型权重类型一致
    class ModelWrapper2D(torch.nn.Module):
        """兼容 FastBEV1V 与 LVFusion 的 2D 导出包装器。

        - FastBEV1V: 使用 backbone / neck / neck_fuse
        - LVFusion: 使用 img_backbone / img_neck / img_neck_fuse
        """

        def __init__(self, model):
            super().__init__()
            # 兼容 LVFusion：优先使用 img_* 分支
            if hasattr(model, 'img_backbone') and hasattr(model, 'img_neck'):
                self.backbone = model.img_backbone
                self.neck = model.img_neck
                self.neck_fuse = getattr(model, 'img_neck_fuse', None)
            else:
                # 兼容旧 FastBEV1V
                self.backbone = model.backbone
                self.neck = model.neck
                self.neck_fuse = getattr(model, 'neck_fuse', None)

        def forward(self, img):
            # 确保输入是float32（模型权重是float32）
            img = img.float()

            # 直接调用backbone和neck，绕过auto_fp16装饰器
            x = self.backbone(img)
            # 确保backbone输出是float32
            if isinstance(x, (list, tuple)):
                x = [item.float() if isinstance(item, torch.Tensor) else item for item in x]
            elif isinstance(x, torch.Tensor):
                x = x.float()

            feats = self.neck(x)
            # 确保neck输出是float32（neck可能有@auto_fp16装饰器）
            if isinstance(feats, (list, tuple)):
                feats = [item.float() if isinstance(item, torch.Tensor) else item for item in feats]
            elif isinstance(feats, torch.Tensor):
                feats = feats.float()

            # multi-level fuse (only use level-0)
            feat = feats[0].float() if isinstance(feats[0], torch.Tensor) else feats[0]
            for i in range(1, len(feats)):
                feat_i = feats[i].float() if isinstance(feats[i], torch.Tensor) else feats[i]
                feat = torch.cat([
                    feat,
                    resize(feat_i, size=feat.shape[-2:], mode='bilinear', align_corners=False)
                ], dim=1)

            if self.neck_fuse is not None:
                feat = feat.float()
                feat = self.neck_fuse(feat)

            if bool(os.getenv("DEPLOY", False)):
                feat = feat.permute(0, 2, 3, 1)
                return feat

            return feat
    
    wrapped_model = ModelWrapper2D(model)
    wrapped_model.eval()
    
    # 导出ONNX模型
    print(f"Exporting to: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_img,
            output_path,
            input_names=['img'],
            output_names=['feat_2d'],
            dynamic_axes={
                'img': {0: 'batch', 2: 'height', 3: 'width'},
                'feat_2d': {0: 'batch', 2: 'height', 3: 'width'}
            },
            opset_version=opset_version,
            verbose=True,
            do_constant_folding=True,
        )
    
    # 检查ONNX模型
    print("Checking ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # 简化模型
    if simplify_model and HAS_ONNXSIM:
        print("Simplifying ONNX model...")
        simplified_model, check_ok = simplify(onnx_model)
        if check_ok:
            simplified_path = output_path.replace('.onnx', '_simplified.onnx')
            onnx.save_model(simplified_model, simplified_path)
            print(f"✓ Simplified model saved to: {simplified_path}")
        else:
            print("⚠ Model simplification failed, using original model")
    
    print(f"✓ 2D ONNX model saved to: {output_path}\n")


def export_onnx_3d(model, config, output_path, simplify_model=True, opset_version=13):
    """
    导出3D处理模型 (neck_3d + bbox_head)
    输入: [1, 64, X, Y, Z] 或 [1, 256, X, Y] (多个输入)
    输出: [cls_score, bbox_pred, dir_cls_preds]
    """
    print("=" * 60)
    print("Exporting 3D processing model...")
    print("=" * 60)
    
    model.eval()
    
    # 确保模型和输入类型一致
    first_param = next(model.parameters())
    model_dtype = first_param.dtype
    
    # 准备输入：3D volume特征
    # 根据FastBEV1V的实现，3D输入应该是 [1, 64, X, Y, Z]
    n_voxels = config.model['n_voxels'][0]  # [200, 200, 4]
    print(f"Voxel size: {n_voxels}")
    print(f"Model dtype: {model_dtype}")
    
    # 使用5D输入格式 [1, 64, X, Y, Z]
    dummy_volume = torch.randn(1, 64, n_voxels[0], n_voxels[1], n_voxels[2]).cuda()
    # 确保输入类型与模型权重类型一致
    if model_dtype == torch.float16:
        dummy_volume = dummy_volume.half()
    else:
        dummy_volume = dummy_volume.float()
    print(f"Input shape: {dummy_volume.shape}, dtype: {dummy_volume.dtype}")
    
    # 创建包装器类，直接调用onnx_export_3d方法并处理输出
    class ModelWrapper3D(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, volume):
            # 设置DEPLOY环境变量（neck_3d需要这个来正确reshape）
            os.environ['DEPLOY'] = '1'
            try:
                outputs = self.model.onnx_export_3d(volume, None)
                # outputs是 [cls_score, bbox_pred, dir_cls_pred]
                # 其中cls_score是list of tensors，需要展平
                if isinstance(outputs, list) and len(outputs) >= 2:
                    cls_score = outputs[0]
                    bbox_pred = outputs[1]
                    dir_cls_pred = outputs[2] if len(outputs) > 2 else None
                    
                    # 展平cls_score list
                    if isinstance(cls_score, list):
                        cls_score = torch.cat([s for s in cls_score], dim=1)
                    
                    if dir_cls_pred is not None:
                        return cls_score, bbox_pred, dir_cls_pred
                    else:
                        return cls_score, bbox_pred
                else:
                    return outputs[0] if isinstance(outputs, list) else outputs
            finally:
                os.environ.pop('DEPLOY', None)
    
    wrapped_model = ModelWrapper3D(model)
    wrapped_model.eval()
    
    # 导出ONNX模型
    print(f"Exporting to: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_volume,
            output_path,
            input_names=['volume_3d'],
            output_names=['cls_score', 'bbox_pred', 'dir_cls_pred'],
            dynamic_axes={
                'volume_3d': {0: 'batch'},
            },
            opset_version=opset_version,
            verbose=True,
            do_constant_folding=True,
        )
    
    # 检查ONNX模型
    print("Checking ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # 简化模型
    if simplify_model and HAS_ONNXSIM:
        print("Simplifying ONNX model...")
        simplified_model, check_ok = simplify(onnx_model)
        if check_ok:
            simplified_path = output_path.replace('.onnx', '_simplified.onnx')
            onnx.save_model(simplified_model, simplified_path)
            print(f"✓ Simplified model saved to: {simplified_path}")
        else:
            print("⚠ Model simplification failed, using original model")
    
    print(f"✓ 3D ONNX model saved to: {output_path}\n")


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 60)
    print("FastBEV1V ONNX Export")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.out_dir}")
    print()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # 构建模型
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 处理fp16（但ONNX导出时禁用fp16，确保类型一致）
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        # 禁用fp16以确保ONNX导出时类型一致
        if hasattr(model, 'fp16_enabled'):
            model.fp16_enabled = False
    
    # 融合conv和bn
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    # 移动到GPU并转换为float32（确保ONNX导出时类型一致）
    model = model.cuda()
    model = model.float()  # 确保模型是float32
    model.eval()
    
    # 导出2D模型
    if args.export_2d:
        output_path_2d = os.path.join(args.out_dir, 'fastbev1v_2d.onnx')
        try:
            export_onnx_2d(model, cfg, output_path_2d, args.simplify, args.opset_version)
        except Exception as e:
            print(f"✗ Failed to export 2D model: {e}")
            import traceback
            traceback.print_exc()
    
    # 导出3D模型
    if args.export_3d:
        output_path_3d = os.path.join(args.out_dir, 'fastbev1v_3d.onnx')
        try:
            export_onnx_3d(model, cfg, output_path_3d, args.simplify, args.opset_version)
        except Exception as e:
            print(f"✗ Failed to export 3D model: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("ONNX export completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

