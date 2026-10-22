#!/usr/bin/env python3
"""Run OCC inference and render image/point-cloud/GT/prediction panels."""
import argparse
from bisect import bisect_right
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mmcv
import numpy as np
import torch
from scipy import ndimage
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


CLASS_NAMES = ['unknown', 'truck', 'large_vehicle',
               'construction_vehicle', 'bus', 'car', 'motorcycle',
               'bicycle', 'tricycle', 'pedestrian', 'other', 'free']
# Keep colors identical to lv_annotation_tools. In particular, car is blue
# and ground is green; the previous near-identical orange colors were invalid.
PALETTE = np.array([
    [32, 40, 50], [239, 155, 35], [174, 106, 32], [236, 207, 53],
    [189, 66, 91], [25, 145, 220], [146, 80, 211], [53, 176, 147],
    [89, 99, 200], [239, 91, 154], [115, 120, 128], [98, 214, 146],
], dtype=np.float32) / 255.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=12)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--score-thr', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-road-filter', action='store_true',
                        help='Visualize raw OCC predictions without predicted-road filtering')
    parser.add_argument('--lidar-fov-deg', type=float, default=33.0,
                        help='Keep predictions inside this forward LiDAR FOV (default: 33)')
    parser.add_argument('--no-ray-mask', action='store_true',
                        help='Do not suppress predictions outside the LiDAR FOV')
    return parser.parse_args()


def dc_value(value):
    """Unwrap MMCV DataContainer and one-sample collation levels."""
    if hasattr(value, 'data') and value.__class__.__name__ == 'DataContainer':
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def to_numpy(value):
    value = dc_value(value)
    if hasattr(value, 'tensor'):
        value = value.tensor
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def get_data_info(dataset, index):
    """Read sample metadata from a single dataset or a ConcatDataset."""
    if hasattr(dataset, 'datasets') and hasattr(dataset, 'cumulative_sizes'):
        part = bisect_right(dataset.cumulative_sizes, index)
        offset = 0 if part == 0 else dataset.cumulative_sizes[part - 1]
        return get_data_info(dataset.datasets[part], index - offset)
    return dataset.get_data_info(index)


def top_surface(volume, confidence=None, threshold=0.0):
    """Return the class at the highest non-background Z voxel for each XY."""
    valid = (volume > 0) & (volume != 12) & (volume != 255)
    if confidence is not None:
        valid &= confidence >= threshold
    any_valid = valid.any(axis=0)
    # Reverse Z so argmax locates the highest valid voxel.
    rev_index = valid[::-1].argmax(axis=0)
    z_index = volume.shape[0] - 1 - rev_index
    surface = np.take_along_axis(volume, z_index[None], axis=0)[0]
    surface[~any_valid] = 0
    return surface


def render_occ(ax, volume, title, pc_range, confidence=None, threshold=0.0):
    surface = top_surface(volume, confidence, threshold)
    extent = [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]
    ax.imshow(surface, origin='lower', extent=extent, interpolation='nearest',
              cmap=ListedColormap(PALETTE), vmin=0, vmax=len(PALETTE) - 1,
              aspect='equal')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.grid(alpha=.15)


def confusion_update(confusion, pred, gt, valid, num_classes):
    keep = valid & (gt >= 0) & (gt < num_classes)
    ids = gt[keep].astype(np.int64) * num_classes + pred[keep].astype(np.int64)
    confusion += np.bincount(ids, minlength=num_classes ** 2).reshape(
        num_classes, num_classes)


def filter_to_predicted_road(pred, pc_range, dilation_m=1.2):
    """Suppress obstacle classes outside the road inferred from predicted free."""
    seed = (pred == 11).any(axis=0)
    vy = (pc_range[4] - pc_range[1]) / pred.shape[1]
    vx = (pc_range[3] - pc_range[0]) / pred.shape[2]
    radius = max(1, int(np.ceil(dilation_m / min(vx, vy))))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    road = ndimage.binary_closing(seed, structure=disk)
    road = ndimage.binary_dilation(road, structure=disk)
    road = ndimage.binary_fill_holes(road)
    filtered = pred.copy()
    obstacle = (filtered >= 1) & (filtered <= 10)
    filtered[obstacle & ~road[None]] = 0
    return filtered


def forward_fov_mask(shape_zyx, pc_range, fov_deg):
    """Return a ZYX mask using the same voxel-center FOV rule as GT generation."""
    if not 0.0 < float(fov_deg) < 180.0:
        raise ValueError('--lidar-fov-deg must be in (0, 180)')
    _, ny, nx = shape_zyx
    vx = (pc_range[3] - pc_range[0]) / nx
    vy = (pc_range[4] - pc_range[1]) / ny
    x = pc_range[0] + (np.arange(nx, dtype=np.float32) + .5) * vx
    y = pc_range[1] + (np.arange(ny, dtype=np.float32) + .5) * vy
    planar = ((x[None, :] >= 0.0)
              & (np.abs(np.arctan2(y[:, None], x[None, :]))
                 <= np.deg2rad(float(fov_deg) * .5)))
    return np.broadcast_to(planar[None, :, :], shape_zyx)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.to(args.device), device_ids=[0])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(range(args.start,
                       min(len(dataset), args.start + args.num_samples * args.stride),
                       args.stride))
    confusion = np.zeros((cfg.model.occ_head.num_classes,) * 2, dtype=np.int64)
    records = []
    cmap = ListedColormap(PALETTE)

    for index, data in enumerate(loader):
        if index > max(wanted):
            break
        if index not in wanted:
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        pred = to_numpy(result['occ_pred']).astype(np.uint8)
        score = to_numpy(result['occ_score']).astype(np.float32)
        gt = to_numpy(data['occ_gt']).astype(np.uint8)
        valid = to_numpy(data['occ_valid_mask']).astype(bool)
        if gt.ndim == 4:
            gt, valid = gt[0], valid[0]
        pc_range = np.asarray(cfg.get('occ_point_cloud_range',
                                      cfg.point_cloud_range), dtype=np.float32)
        if not args.no_road_filter:
            pred = filter_to_predicted_road(pred, pc_range)
        if not args.no_ray_mask:
            ray_mask = forward_fov_mask(pred.shape, pc_range, args.lidar_fov_deg)
            pred = pred.copy()
            score = score.copy()
            pred[~ray_mask] = 255
            score[~ray_mask] = 0.0
            valid &= ray_mask
        confusion_update(confusion, pred, gt, valid, cfg.model.occ_head.num_classes)
        accuracy = float((pred[valid] == gt[valid]).mean()) if valid.any() else float('nan')

        sample = get_data_info(dataset, index)
        image_path = sample['img_info'][0]['filename']
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = to_numpy(data['points'])
        if points.ndim == 3:
            points = points[0]

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f'Camera | validation #{index}', fontweight='bold')
        axes[0, 0].axis('off')
        color = points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))
        axes[0, 1].scatter(points[:, 0], points[:, 1], c=color, cmap='turbo',
                           s=.25, linewidths=0)
        axes[0, 1].set_xlim(pc_range[0], pc_range[3])
        axes[0, 1].set_ylim(pc_range[1], pc_range[4])
        axes[0, 1].set_aspect('equal')
        axes[0, 1].set_title(f'LiDAR | {len(points):,} points', fontweight='bold')
        axes[0, 1].set_xlabel('X / m'); axes[0, 1].set_ylabel('Y / m')
        render_occ(axes[1, 0], gt, 'GT OCC (highest occupied voxel)', pc_range)
        render_occ(axes[1, 1], pred,
                   f'Pred OCC | confidence >= {args.score_thr:.2f} | valid acc={accuracy:.3f}',
                   pc_range, score, args.score_thr)
        handles = [plt.Line2D([0], [0], marker='s', linestyle='', color=cmap(i),
                              label=f'{i}:{CLASS_NAMES[i]}')
                   for i in range(1, len(PALETTE))]
        fig.legend(handles=handles, title='OCC class id', loc='lower center',
                   ncol=len(handles), fontsize=8)
        output = out_dir / f'{index:04d}.png'
        fig.savefig(output, dpi=140)
        plt.close(fig)
        records.append({'index': index, 'image': image_path,
                        'valid_voxel_accuracy': accuracy, 'output': str(output),
                        'ray_mask_fov_deg': None if args.no_ray_mask
                        else float(args.lidar_fov_deg)})
        print(f'[{len(records)}/{len(wanted)}] {output} acc={accuracy:.4f}', flush=True)

    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0)
    metrics = {'samples': records, 'confusion_matrix': confusion.tolist(),
               'iou_per_class': iou.tolist(),
               'miou_present_classes': float(np.nanmean(iou)),
               'voxel_accuracy': float(intersection.sum() / max(confusion.sum(), 1))}
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'samples' and
                      k != 'confusion_matrix'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
