#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone HASCO 3D detection evaluation script.

Usage:
  # Evaluate from config + checkpoint (run inference then eval):
  python tools/hasco_eval.py CONFIG CHECKPOINT

  # Evaluate from existing result pkl:
  python tools/hasco_eval.py CONFIG --result RESULT.pkl

  # With custom data root / ann file:
  python tools/hasco_eval.py CONFIG CHECKPOINT --options data_root=/path/to/hasco
"""

import argparse
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
from mmdet3d.core.evaluation import hasco_detection_eval


def parse_args():
    parser = argparse.ArgumentParser(description='HASCO 3D detection evaluation')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', nargs='?', default=None, help='Checkpoint file (optional if --result)')
    parser.add_argument('--result', help='Path to result pkl from previous test (skip inference)')
    parser.add_argument('--out', help='Output directory for results and metrics')
    parser.add_argument('--eval', action='store_true', default=True, help='Run evaluation')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Override config (e.g. data_root=/path)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.checkpoint or args.result, 'Need either checkpoint (for inference) or --result pkl'

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.model.train_cfg = None
    cfg.model.pop('pretrained', None)  # LVFusion does not accept pretrained

    # Use val split for evaluation
    if isinstance(cfg.data.val, dict):
        cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    class_names = dataset.CLASSES
    data_root = dataset.data_root
    ann_file = dataset.ann_file
    gt_path = ann_file if osp.isabs(ann_file) else osp.join(data_root, ann_file)

    if args.result:
        print(f'Loading results from {args.result}')
        outputs = mmcv.load(args.result)
    else:
        set_random_seed(args.seed)
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        outputs = single_gpu_test(model, data_loader)

    # Format and save results to JSON
    if args.out:
        # Create subfolder by config filename, e.g. hasco_eval_output/hasco_sr_fusion_r18_xxx/
        config_basename = osp.splitext(osp.basename(args.config))[0]
        out_dir = osp.join(args.out, config_basename)
        os.makedirs(out_dir, exist_ok=True)
        jsonfile_prefix = osp.join(out_dir, 'results')
    else:
        import tempfile
        tmp = tempfile.mkdtemp()
        jsonfile_prefix = osp.join(tmp, 'results')

    result_files, tmp_dir = dataset.format_results(outputs, jsonfile_prefix=jsonfile_prefix)
    result_path = result_files

    print(f'Formatted results saved to {result_path}')

    # Run evaluation
    point_cloud_range = getattr(cfg, 'point_cloud_range', None) or getattr(dataset, 'point_cloud_range', None)
    class_range = getattr(cfg, 'class_range', None) or getattr(dataset, 'class_range', None)
    eval_result = hasco_detection_eval(
        gt_path=gt_path,
        result_path=result_path,
        class_names=class_names,
        output_dir=osp.dirname(result_path),
        point_cloud_range=point_cloud_range,
        class_range=class_range,
        verbose=True)

    print('\n===== Summary =====')
    print(f"mAP: {eval_result['mean_ap']:.4f}")
    print(f"DetectionScore: {eval_result['detection_score']:.4f}")
    for k, v in eval_result.get('tp_errors', {}).items():
        print(f"  {k}: {v:.4f}")

    return eval_result


if __name__ == '__main__':
    main()
