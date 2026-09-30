#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NuScenes CAM_FRONT (NuScenesCAMFrontDataset) 评估脚本。

适用于 configs/hascolv/fusion_config/small_range/nus 下三个配置：
- nus_cam_r50_s512x1408_v015_lvfusion_fpn_centerhead.py
- nus_lidar_r50_s512x1408_v015_lvfusion_fpn_centerhead.py
- nus_fusion_r50_s512x1408_v015_lvfusion_fpn_centerhead.py

用法:
  # 从 config + checkpoint 跑推理并评估:
  python tools/nus_frontcam_eval.py CONFIG CHECKPOINT

  # 从已有结果 pkl 仅做评估（跳过推理）:
  python tools/nus_frontcam_eval.py CONFIG --result RESULT.pkl

  # 指定输出目录保存 NuScenes 格式 JSON 与 metrics:
  python tools/nus_frontcam_eval.py CONFIG CHECKPOINT --out ./nus_eval_output
"""

import argparse
import logging
import os
import os.path as osp

import mmcv
import torch

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis import single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='NuScenes CAM_FRONT (NuScenesCAMFrontDataset) 3D 检测评估')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='权重文件（使用 --result 时可省略）')
    parser.add_argument(
        '--result',
        help='已有推理结果 pkl 路径，指定则跳过推理直接评估')
    parser.add_argument(
        '--out',
        help='结果与 metrics 输出目录（含 results_nusc.json、metrics_summary.json）')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox'],
        help='评估指标，默认 bbox（NuScenes 协议）')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='覆盖配置项，例如 data_root=/path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='传给 dataset.evaluate() 的额外参数')
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.checkpoint or args.result, (
        '必须指定 checkpoint（跑推理+评估）或 --result pkl（仅评估）')

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.model.train_cfg = None
    cfg.model.pop('pretrained', None)

    # 使用 test 数据集（NuScenes val 划分）
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    else:
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test)
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    set_random_seed(args.seed)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    if args.result:
        print(f'从已有结果加载: {args.result}')
        outputs = mmcv.load(args.result)
    else:
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        outputs = single_gpu_test(model, data_loader)

    # 输出目录
    if args.out:
        config_basename = osp.splitext(osp.basename(args.config))[0]
        out_dir = osp.join(args.out, config_basename)
        os.makedirs(out_dir, exist_ok=True)
        jsonfile_prefix = osp.join(out_dir, 'results')
    else:
        import tempfile
        tmp = tempfile.mkdtemp()
        jsonfile_prefix = osp.join(tmp, 'results')

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule'):
        eval_kwargs.pop(key, None)
    eval_kwargs.update(metric=args.eval)
    if args.eval_options:
        eval_kwargs.update(args.eval_options)

    print('Running NuScenes evaluation (pts_bbox)...')
    # 传入 logger 以启用 range-filtered 评估的 verbose 打印（含 per-class 明细）
    logger = logging.getLogger(__name__)
    results_dict = dataset.evaluate(
        outputs,
        jsonfile_prefix=jsonfile_prefix,
        logger=logger,
        **eval_kwargs)

    # 打印主要指标（NuScenes 协议）
    print('\n===== NuScenes 评估结果 =====')
    for key in sorted(results_dict.keys()):
        if 'mAP' in key or 'NDS' in key:
            print(f'  {key}: {results_dict[key]:.4f}')
    print('  (全部指标见上方或 metrics_summary.json)')
    if args.out:
        print(f'  结果与 metrics 已写入: {out_dir}')

    return results_dict


if __name__ == '__main__':
    main()
