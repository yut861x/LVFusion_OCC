#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize Hasco training data with the full train augmentation pipeline.

Outputs per-frame images with:
  - augmented camera image
  - projected LiDAR points (50% transparency)
  - GT 3D bounding boxes projected onto the image
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lvfusion_python_inference_visualization import (  # noqa: E402
    draw_bbox,
    draw_points_on_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize training data augmentation (image + point cloud + GT boxes)')
    parser.add_argument(
        '--config',
        type=str,
        default=(
            'configs/hascolv/fusion_config/small_range/r18/'
            'r18_105.6m_19.2m_aug_38.4mpretrain/'
            'hasco_sr_fusion_r18_105.6m_19.2m_aug_38.4mpretrain_s1080x1920_v015_lvfusion_fpn_centerhead.py'
        ),
        help='training config file path',
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='hasco_vis/train_aug',
        help='output directory for visualization images',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='number of frames to visualize',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed for dataloader shuffle',
    )
    parser.add_argument(
        '--point-overlap',
        type=float,
        default=0.5,
        help='point cloud overlay alpha (0.5 = 50%% transparency)',
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
        help='shuffle training samples (default: True)',
    )
    parser.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='iterate training samples in order',
    )
    return parser.parse_args()


def _unwrap_dc(x):
    """Unwrap mmcv DataContainer."""
    return x.data if hasattr(x, 'data') else x


def _first_item(x):
    """Take the first element from nested batch structures."""
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            return x
        x = x[0]
    return x


def _extract_image(img):
    img = _unwrap_dc(img)
    img = _first_item(img)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = np.asarray(img)
    while img.ndim > 3:
        img = np.squeeze(img, axis=0)
    return img


def _denormalize_image(img_chw, img_norm_cfg):
    mean = np.array(img_norm_cfg.get('mean', [123.675, 116.28, 103.53]), dtype=np.float32)
    std = np.array(img_norm_cfg.get('std', [58.395, 57.12, 57.375]), dtype=np.float32)
    img_chw = _extract_image(img_chw)
    if img_chw.ndim == 3 and img_chw.shape[0] in (1, 3):
        img_hwc = img_chw.transpose(1, 2, 0).astype(np.float32) * std + mean
    else:
        raise ValueError(f'Unexpected image shape after squeeze: {img_chw.shape}')
    img_bgr = np.clip(img_hwc[:, :, ::-1], 0, 255).astype(np.uint8)
    return img_bgr


def _extract_points(points):
    pts = _unwrap_dc(points)
    pts = _first_item(pts)
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    pts = np.asarray(pts, dtype=np.float32)
    return pts[:, :3] if pts.ndim == 2 and pts.shape[1] >= 3 else pts.reshape(-1, 4)[:, :3]


def _extract_gt_boxes(gt_bboxes_3d):
    boxes = _unwrap_dc(gt_bboxes_3d)
    boxes = _first_item(boxes)
    if boxes is None:
        return np.zeros((0, 7), dtype=np.float32)
    if hasattr(boxes, 'tensor'):
        tensor = boxes.tensor
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        return tensor[:, :7].astype(np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    return arr[:, :7] if arr.ndim == 2 else np.zeros((0, 7), dtype=np.float32)


def _extract_lidar2img(img_metas):
    meta = _unwrap_dc(img_metas)
    meta = _first_item(meta)
    if not isinstance(meta, dict):
        return np.eye(4, dtype=np.float64)
    lidar2img_info = meta.get('lidar2img', {})
    if isinstance(lidar2img_info, dict):
        extrinsic = lidar2img_info.get('extrinsic', [np.eye(4)])
        return np.asarray(extrinsic[0], dtype=np.float64)
    return np.asarray(lidar2img_info, dtype=np.float64)


def _extract_filename(img_metas, index):
    meta = _unwrap_dc(img_metas)
    meta = _first_item(meta)
    if isinstance(meta, dict):
        filename = meta.get('filename', '')
        if isinstance(filename, (list, tuple)):
            filename = filename[0] if filename else ''
        if filename:
            return os.path.basename(str(filename))
    return f'sample_{index:04d}.jpg'


def _build_train_dataset(cfg):
    train_cfg = cfg.data.train
    return build_dataset(train_cfg, default_args=dict(filter_empty_gt=False))


def visualize_sample(data, index, point_overlap):
    img_metas = data.get('img_metas')
    meta = _first_item(_unwrap_dc(img_metas))
    img_norm_cfg = meta.get('img_norm_cfg', {}) if isinstance(meta, dict) else {}

    img = data.get('img')
    image = _denormalize_image(img, img_norm_cfg)

    points = _extract_points(data.get('points'))
    gt_boxes = _extract_gt_boxes(data.get('gt_bboxes_3d'))
    lidar2img = _extract_lidar2img(img_metas)

    vis = draw_points_on_image(
        image,
        points,
        lidar2img,
        draw_range=(3, 100),
        point_size=2,
        overlap=point_overlap,
    )
    if gt_boxes.shape[0] > 0:
        vis = draw_bbox(vis, lidar2img, gt_boxes, color=(0, 255, 0), thickness=2)

    label = f'Train Aug | frame {index:03d} | GT={gt_boxes.shape[0]}'
    cv2.putText(
        vis,
        label,
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        4,
    )
    cv2.putText(
        vis,
        label,
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )
    return vis, _extract_filename(img_metas, index)


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0

    dataset = _build_train_dataset(cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    total = min(args.num_samples, len(dataset))
    print(f'Dataset size: {len(dataset)}, visualizing {total} frames -> {args.save_path}')

    saved = 0
    for batch_idx, data in enumerate(data_loader):
        if saved >= total:
            break

        vis, filename = visualize_sample(data, saved, args.point_overlap)
        stem = Path(filename).stem
        out_name = f'{saved:04d}_{stem}.jpg'
        out_path = os.path.join(args.save_path, out_name)
        cv2.imwrite(out_path, vis)
        saved += 1
        if saved % 10 == 0 or saved == total:
            print(f'[{saved}/{total}] saved: {out_path}')

    print(f'Done. {saved} images saved to {os.path.abspath(args.save_path)}')


if __name__ == '__main__':
    main()
