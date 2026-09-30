# Copyright (c) OpenMMLab. All rights reserved.
"""
使用ONNX模型进行推理测试的示例脚本

python tools/test_onnx.py \
    --config configs/hascolv/mono_2class_fastbev1v_m0_r18_s256x704_v200x200x4_c192_d2.py \
    --onnx-2d ./output/onnx/fastbev1v_2d.onnx \
    --onnx-3d ./output/onnx/fastbev1v_3d.onnx \
    --eval mAP \
    --cpu
"""
import argparse
import os
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import scatter, collate
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test FastBEV1V with ONNX models')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file (not used in ONNX mode)')
    parser.add_argument('--onnx-2d', type=str, default='./output/onnx/fastbev1v_2d.onnx',
                       help='Path to 2D ONNX model')
    parser.add_argument('--onnx-3d', type=str, default='./output/onnx/fastbev1v_3d.onnx',
                       help='Path to 3D ONNX model')
    parser.add_argument('--eval', type=str, nargs='+', help='evaluation metrics')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', type=str, help='directory where results will be saved')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU (avoid CUDA extension issues)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置ONNX模型路径环境变量
    os.environ['ONNX_2D_MODEL_PATH'] = args.onnx_2d
    os.environ['ONNX_3D_MODEL_PATH'] = args.onnx_3d
    
    # 检查ONNX模型文件是否存在
    if not os.path.exists(args.onnx_2d):
        raise FileNotFoundError(f"2D ONNX model not found: {args.onnx_2d}")
    if not os.path.exists(args.onnx_3d):
        raise FileNotFoundError(f"3D ONNX model not found: {args.onnx_3d}")
    
    print("=" * 60)
    print("FastBEV1V ONNX Inference Test")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"2D ONNX Model: {args.onnx_2d}")
    print(f"3D ONNX Model: {args.onnx_3d}")
    print()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 设置test_mode为test_onnx，并传递ONNX模型路径
    if cfg.get('model', {}).get('test_cfg', None) is not None:
        cfg.model.test_cfg.test_mode = 'test_onnx'
        cfg.model.test_cfg.onnx_2d_path = args.onnx_2d
        cfg.model.test_cfg.onnx_3d_path = args.onnx_3d
    else:
        cfg.model.test_cfg = dict(
            test_mode='test_onnx',
            onnx_2d_path=args.onnx_2d,
            onnx_3d_path=args.onnx_3d
        )
    
    # 构建数据集
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    
    # 构建模型（不需要加载checkpoint，因为使用ONNX模型）
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 如果有checkpoint，可以加载用于初始化（但推理时使用ONNX）
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint for initialization: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 根据参数决定使用CPU还是GPU
    if args.cpu:
        print("Running on CPU mode (avoiding CUDA extension issues)")
        device = torch.device('cpu')
        model = model.cpu()
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model = model.cuda()
            print("Model moved to GPU")
        else:
            print("WARNING: CUDA not available, falling back to CPU")
            device = torch.device('cpu')
            model = model.cpu()
    
    model.eval()
    
    # 运行测试
    print("Starting ONNX inference...")
    results = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # 将数据移到正确的设备
            if device.type == 'cuda':
                # scatter data to GPU
                data = scatter(data, [device.index])[0]
            else:
                # CPU模式：解包DataContainer
                from mmcv.parallel import DataContainer
                if isinstance(data.get('img'), DataContainer):
                    data['img'] = data['img'].data
                if isinstance(data.get('img_metas'), DataContainer):
                    data['img_metas'] = data['img_metas'].data
            
            result = model(return_loss=False, **data)
            results.extend(result)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data_loader)} batches")
    
    print(f"\nCompleted! Processed {len(results)} samples")
    
    # 评估
    if args.eval:
        print(f"\nEvaluating with metrics: {args.eval}")
        eval_results = dataset.evaluate(results, metric=args.eval)
        print("Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()

