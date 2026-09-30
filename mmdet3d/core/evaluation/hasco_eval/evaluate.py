# Copyright (c) OpenMMLab. All rights reserved.
"""HASCO 3D detection evaluation.

Reference: nuScenes detection evaluation.
- Excludes velocity (vel_err) and attribute (attr_err)
- DetectionScore replaces NDS
"""

import json
import os
from typing import Dict, Any, Optional

import mmcv
import numpy as np

from .config import HascoDetectionConfig
from .data_classes import HascoDetectionMetrics
from .loaders import (load_hasco_gt, load_hasco_predictions,
                      add_center_dist_hasco, filter_eval_boxes_hasco)
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionMetricData


def hasco_detection_eval(gt_path: str,
                        result_path: str,
                        class_names: tuple,
                        output_dir: Optional[str] = None,
                        class_range: Optional[Dict[str, int]] = None,
                        point_cloud_range: Optional[list] = None,
                        verbose: bool = True) -> Dict[str, Any]:
    """Run HASCO 3D detection evaluation.

    Args:
        gt_path: Path to hasco_infos_val.pkl
        result_path: Path to results JSON (nuScenes-like format)
        class_names: Tuple of class names, e.g. ('car', 'truck', 'bus', ...)
        output_dir: Directory to save metrics. If None, uses dir of result_path.
        class_range: Max eval distance per class. Default 50m for all.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] for ObjectRangeFilter.
            If None, skips ObjectRangeFilter (not aligned with val_pipeline).
        verbose: Print progress

    Returns:
        Dict with mean_ap, detection_score, tp_errors, label_aps, etc.
    """
    assert os.path.exists(result_path), f'Result file not found: {result_path}'
    assert os.path.exists(gt_path), f'GT file not found: {gt_path}'

    if output_dir is None:
        output_dir = os.path.dirname(result_path)
    os.makedirs(output_dir, exist_ok=True)

    cfg = HascoDetectionConfig(
        class_names=class_names,
        class_range=class_range,
    )

    # Load predictions
    pred_boxes, meta = load_hasco_predictions(
        result_path, cfg.max_boxes_per_sample, class_names, verbose=verbose)

    # Load GT
    gt_boxes = load_hasco_gt(gt_path, class_names, verbose=verbose)

    # Ensure same sample set
    pred_tokens = set(pred_boxes.sample_tokens)
    gt_tokens = set(gt_boxes.sample_tokens)
    if pred_tokens != gt_tokens:
        missing_pred = gt_tokens - pred_tokens
        missing_gt = pred_tokens - gt_tokens
        if missing_pred:
            for t in missing_pred:
                pred_boxes.add_boxes(t, [])
        if missing_gt and verbose:
            print(f"Warning: predictions have samples not in GT: {len(missing_gt)}")

    # Add ego distance (HASCO: ego at origin)
    pred_boxes = add_center_dist_hasco(pred_boxes)
    gt_boxes = add_center_dist_hasco(gt_boxes)

    # Filter: ObjectNameFilter, ObjectRangeFilter, DistanceFilter (aligned with val_pipeline)
    pred_boxes = filter_eval_boxes_hasco(
        pred_boxes, cfg.class_range, class_names=tuple(cfg.class_names),
        point_cloud_range=point_cloud_range, verbose=verbose, label='predictions')
    gt_boxes = filter_eval_boxes_hasco(
        gt_boxes, cfg.class_range, class_names=tuple(cfg.class_names),
        point_cloud_range=point_cloud_range, verbose=verbose, label='ground truth')

    # Accumulate and compute metrics
    if verbose:
        print("Accumulating metric data...")
    metric_data_list = {}
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(
                gt_boxes, pred_boxes, class_name,
                cfg.dist_fcn_callable, dist_th, verbose=False)
            metric_data_list[(class_name, dist_th)] = md

    if verbose:
        print("Calculating metrics...")
    metrics = HascoDetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(md, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)

        for metric_name in cfg.TP_METRICS:
            md = metric_data_list[(class_name, cfg.dist_th_tp)]
            tp = calc_tp(md, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    # Serialize and save
    err_name_mapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
    }
    summary = {
        'mean_ap': metrics.mean_ap,
        'detection_score': metrics.detection_score,
        'tp_errors': {err_name_mapping[k]: v for k, v in metrics.tp_errors.items()},
        'label_aps': dict(metrics._label_aps),
        'mean_dist_aps': metrics.mean_dist_aps,
        'label_tp_errors': dict(metrics._label_tp_errors),
        'meta': meta,
    }
    summary_path = os.path.join(output_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n===== HASCO 3D Detection Results =====")
        print(f"{'Class':<25} {'AP':>8} {'mATE':>8} {'mASE':>8} {'mAOE':>8}")
        for name in cfg.class_names:
            ap = metrics.mean_dist_aps.get(name, 0.0)
            tps = metrics._label_tp_errors.get(name, {})
            ate = tps.get('trans_err', 0.0)
            ase = tps.get('scale_err', 0.0)
            aoe = tps.get('orient_err', 0.0)
            print(f"{name:<25} {ap:>8.4f} {ate:>8.4f} {ase:>8.4f} {aoe:>8.4f}")
        print(f"Results saved to {summary_path}")

    return summary
