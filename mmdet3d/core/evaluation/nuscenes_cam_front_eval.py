# Copyright (c) OpenMMLab. All rights reserved.
"""NuScenes CAM_FRONT 前向范围评估：使用与 val_pipeline 一致的 GT 过滤（ObjectRangeFilter + ObjectNameFilter），
与官方全场景 NuScenes 评估独立，不影响原始 nuscenes 数据集验证。
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from pyquaternion import Quaternion

from .hasco_eval.config import HascoDetectionConfig
from .hasco_eval.data_classes import HascoDetectionBox, HascoDetectionMetrics
from .hasco_eval.loaders import (
    add_center_dist_hasco,
    filter_eval_boxes_hasco,
)
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp


def _get_ann_info_in_range(dataset, index: int, point_cloud_range: List[float]):
    """与 val_pipeline 一致：从 data_infos 读 NuScenes 结构 + ObjectRangeFilter + ObjectNameFilter。
    不调用 dataset.get_ann_info，避免走到 CocoDataset（需要 data_infos['id']）。
    """
    info = dataset.data_infos[index]
    if 'gt_boxes' not in info or len(info['gt_boxes']) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=np.int64)
    use_valid_flag = getattr(dataset, 'use_valid_flag', False)
    mask = info['valid_flag'] if use_valid_flag else (info['num_lidar_pts'] > 0)
    gt_boxes = np.asarray(info['gt_boxes'][mask], dtype=np.float32)  # (N, 7) or (N, 9)
    gt_names = info['gt_names'][mask]
    cat2id = getattr(dataset, 'cat2id', {c: i for i, c in enumerate(dataset.CLASSES)})
    gt_labels_3d = np.array([cat2id.get(c, -1) for c in gt_names], dtype=np.int64)
    valid = gt_labels_3d >= 0
    if not np.any(valid):
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=np.int64)
    gt_boxes = gt_boxes[valid]
    gt_labels_3d = gt_labels_3d[valid]
    # 取前 7 列 x,y,z,l,w,h,yaw（忽略 velocity）
    if gt_boxes.shape[1] > 7:
        gt_boxes = gt_boxes[:, :7].copy()
    centers = gt_boxes[:, :3]
    pcr = np.asarray(point_cloud_range, dtype=np.float32)
    x_min, y_min, z_min, x_max, y_max, z_max = pcr[0], pcr[1], pcr[2], pcr[3], pcr[4], pcr[5]
    in_range = (
        (centers[:, 0] >= x_min) & (centers[:, 0] <= x_max) &
        (centers[:, 1] >= y_min) & (centers[:, 1] <= y_max) &
        (centers[:, 2] >= z_min) & (centers[:, 2] <= z_max)
    )
    if not np.any(in_range):
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=np.int64)
    gt_boxes = gt_boxes[in_range]
    gt_labels_3d = gt_labels_3d[in_range]
    return gt_boxes, gt_labels_3d


def load_gt_from_dataset(dataset, point_cloud_range: List[float], verbose: bool = False) -> EvalBoxes:
    """从 NuScenesCAMFrontDataset 按 point_cloud_range 与 CLASSES 加载 GT（与 val_pipeline 一致）。"""
    class_names = tuple(dataset.CLASSES)
    all_gt = EvalBoxes()
    it = range(len(dataset))
    if verbose:
        import mmcv
        it = mmcv.track_iter_progress(it)
    for i in it:
        sample_token = dataset.data_infos[i]['token']
        boxes_3d, labels_3d = _get_ann_info_in_range(dataset, i, point_cloud_range)
        if len(labels_3d) == 0:
            all_gt.add_boxes(sample_token, [])
            continue
        sample_boxes = []
        # boxes_3d: (N, 7) x,y,z,l,w,h,yaw (NuScenes pkl 格式)
        centers = boxes_3d[:, :3]
        dims = boxes_3d[:, 3:6]
        yaw = boxes_3d[:, 6]
        for k in range(len(labels_3d)):
            name = class_names[int(labels_3d[k])]
            x, y, z = float(centers[k, 0]), float(centers[k, 1]), float(centers[k, 2])
            l, w, h = float(dims[k, 0]), float(dims[k, 1]), float(dims[k, 2])
            yaw_k = float(yaw[k])
            quat = Quaternion(axis=[0, 0, 1], radians=yaw_k)
            rotation = quat.elements.tolist()
            sample_boxes.append(HascoDetectionBox(
                sample_token=sample_token,
                translation=(x, y, z),
                size=(l, w, h),
                rotation=tuple(rotation),
                velocity=(0.0, 0.0),
                ego_translation=(x, y, z),
                num_pts=-1,
                detection_name=name,
                detection_score=-1.0,
                attribute_name='',
                class_names=class_names,
            ))
        all_gt.add_boxes(sample_token, sample_boxes)
    return all_gt


def build_pred_eval_boxes_from_results(
    results: List[dict],
    dataset,
    max_boxes_per_sample: int = 500,
) -> EvalBoxes:
    """从模型输出 results（pts_bbox，LiDAR 系）构建 EvalBoxes，与 dataset 样本顺序一致。"""
    class_names = tuple(dataset.CLASSES)
    all_pred = EvalBoxes()
    for i in range(len(results)):
        sample_token = dataset.data_infos[i]['token']
        r = results[i].get('pts_bbox', results[i])
        if r is None or len(r.get('scores_3d', [])) == 0:
            all_pred.add_boxes(sample_token, [])
            continue
        boxes_3d = r['boxes_3d']
        scores_3d = r['scores_3d']
        labels_3d = r['labels_3d']
        if hasattr(scores_3d, 'numpy'):
            scores_3d = scores_3d.numpy()
        if hasattr(labels_3d, 'numpy'):
            labels_3d = labels_3d.numpy()
        if hasattr(boxes_3d, 'gravity_center'):
            centers = boxes_3d.gravity_center.numpy()
            dims = boxes_3d.dims.numpy()
            yaw = boxes_3d.yaw.numpy()
        else:
            t = boxes_3d.tensor.numpy()
            centers = t[:, :3]
            dims = t[:, 3:6]
            yaw = t[:, 6]
        n = min(len(scores_3d), max_boxes_per_sample)
        sample_boxes = []
        for k in range(n):
            label = int(labels_3d[k])
            if label < 0 or label >= len(class_names):
                continue
            name = class_names[label]
            x, y, z = float(centers[k, 0]), float(centers[k, 1]), float(centers[k, 2])
            l, w, h = float(dims[k, 0]), float(dims[k, 1]), float(dims[k, 2])
            quat = Quaternion(axis=[0, 0, 1], radians=float(yaw[k]))
            rotation = quat.elements.tolist()
            sample_boxes.append(HascoDetectionBox(
                sample_token=sample_token,
                translation=(x, y, z),
                size=(l, w, h),
                rotation=tuple(rotation),
                velocity=(0.0, 0.0),
                ego_translation=(x, y, z),
                num_pts=-1,
                detection_name=name,
                detection_score=float(scores_3d[k]),
                attribute_name='',
                class_names=class_names,
            ))
        all_pred.add_boxes(sample_token, sample_boxes)
    return all_pred


def nuscenes_cam_front_detection_eval(
    dataset,
    results: List[dict],
    point_cloud_range: List[float],
    class_range: Optional[Dict[str, float]] = None,
    output_dir: Optional[str] = None,
    jsonfile_prefix: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """使用与 val_pipeline 一致的 GT（范围+类别过滤）评估 NuScenes CAM_FRONT 检测结果。

    不调用官方 NuScenesEval，不影响全场景 nuscenes 评估。

    Args:
        dataset: NuScenesCAMFrontDataset 实例（已 load_annotations）
        results: 模型输出 list[dict]，每项含 pts_bbox
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        class_range: 每类最大评估距离，None 则全类 50m
        output_dir: 保存 metrics_summary.json 的目录
        jsonfile_prefix: 若提供则同时写入预测 JSON（兼容原流程）
        verbose: 是否打印

    Returns:
        detail: 与 NuScenes 评估相同结构的指标 dict（pts_bbox_NuScenes/mAP 等），便于日志与可视化
    """
    class_names = tuple(dataset.CLASSES)
    if class_range is None:
        class_range = {n: 50.0 for n in class_names}
    else:
        # eval_detection_configs 可能是 10 类，只保留当前 CLASSES 的 key，保证与 class_names 一致
        class_range = {n: class_range.get(n, 50.0) for n in class_names}

    cfg = HascoDetectionConfig(class_names=class_names, class_range=class_range)

    if verbose:
        print('[NuScenes CAM_FRONT] Loading GT (filtered by point_cloud_range + CLASSES)...')
    gt_boxes = load_gt_from_dataset(dataset, point_cloud_range, verbose=verbose)

    if verbose:
        print('[NuScenes CAM_FRONT] Building predictions from results...')
    pred_boxes = build_pred_eval_boxes_from_results(
        results, dataset, max_boxes_per_sample=cfg.max_boxes_per_sample)

    pred_tokens = set(pred_boxes.sample_tokens)
    gt_tokens = set(gt_boxes.sample_tokens)
    for t in gt_tokens - pred_tokens:
        pred_boxes.add_boxes(t, [])

    pred_boxes = add_center_dist_hasco(pred_boxes)
    gt_boxes = add_center_dist_hasco(gt_boxes)

    pred_boxes = filter_eval_boxes_hasco(
        pred_boxes, cfg.class_range, class_names,
        point_cloud_range=point_cloud_range, verbose=verbose, label='predictions')
    gt_boxes = filter_eval_boxes_hasco(
        gt_boxes, cfg.class_range, class_names,
        point_cloud_range=point_cloud_range, verbose=verbose, label='ground truth')

    if verbose:
        print('Accumulating metric data...')
    metric_data_list = {}
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(
                gt_boxes, pred_boxes, class_name,
                cfg.dist_fcn_callable, dist_th, verbose=False)
            metric_data_list[(class_name, dist_th)] = md

    if verbose:
        print('Calculating metrics...')
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
        'mean_dist_aps': dict(metrics.mean_dist_aps),
        'label_tp_errors': dict(metrics._label_tp_errors),
    }

    if output_dir is None and jsonfile_prefix is not None:
        output_dir = os.path.dirname(jsonfile_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, 'metrics_summary_nus_cam_front.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f'Metrics saved to {summary_path}')

    # 返回与 NuScenes 评估一致的 detail 结构，便于 EvalHook 和可视化
    result_name = 'pts_bbox_NuScenes'
    detail = {}
    for name in class_names:
        for k, v in metrics._label_aps[name].items():
            detail[f'{result_name}/{name}_AP_dist_{k}'] = round(float(v), 4)
        for k, v in metrics._label_tp_errors[name].items():
            detail[f'{result_name}/{name}_{k}'] = round(float(v), 4)
    for k, v in metrics.tp_errors.items():
        detail[f'{result_name}/{err_name_mapping[k]}'] = round(float(v), 4)
    detail[f'{result_name}/NDS'] = round(float(metrics.detection_score), 4)
    detail[f'{result_name}/mAP'] = round(float(metrics.mean_ap), 4)

    if verbose:
        print('\n===== NuScenes CAM_FRONT (range-filtered GT) =====')
        print(f"mAP: {metrics.mean_ap:.4f}")
        print(f"NDS (DetectionScore): {metrics.detection_score:.4f}")
        for k, v in metrics.tp_errors.items():
            print(f"  {err_name_mapping[k]}: {v:.4f}")
        print('\nPer-class results:')
        print(f"{'Object Class':<25} {'AP':>8} {'mATE':>8} {'mASE':>8} {'mAOE':>8}")
        for name in class_names:
            ap = metrics.mean_dist_aps.get(name, 0.0)
            tps = metrics._label_tp_errors.get(name, {})
            ate = tps.get('trans_err', np.nan)
            ase = tps.get('scale_err', np.nan)
            aoe = tps.get('orient_err', np.nan)
            ap_s = f'{ap:.4f}' if not np.isnan(ap) else 'nan'
            ate_s = f'{ate:.4f}' if not np.isnan(ate) else 'nan'
            ase_s = f'{ase:.4f}' if not np.isnan(ase) else 'nan'
            aoe_s = f'{aoe:.4f}' if not np.isnan(aoe) else 'nan'
            print(f"{name:<25} {ap_s:>8} {ate_s:>8} {ase_s:>8} {aoe_s:>8}")

    return detail
