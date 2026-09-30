# Copyright (c) OpenMMLab. All rights reserved.
"""HASCO GT/prediction loaders.

HASCO data is in LiDAR (ego) frame. Ego is at origin, so ego_translation = translation.

Filter order (aligned with val_pipeline):
  1. ObjectNameFilter: keep only classes in class_names
  2. ObjectRangeFilter: keep only boxes within point_cloud_range [x_min,y_min,z_min,x_max,y_max,z_max]
  3. DistanceFilter: keep only boxes within class_range (ego_dist < max_dist per class)
"""

import json
from typing import Dict, List, Optional, Tuple

import mmcv
import numpy as np
from pyquaternion import Quaternion

from .data_classes import HascoDetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from .class_aliases import HASCO_CLASS_ALIASES


def load_hasco_gt(ann_file: str, class_names: tuple, verbose: bool = False) -> EvalBoxes:
    """Load ground truth from HASCO pkl file.

    Args:
        ann_file: Path to hasco_infos_val.pkl
        class_names: Tuple of class names
        verbose: Print progress

    Returns:
        EvalBoxes with HascoDetectionBox
    """
    data = mmcv.load(ann_file)
    infos = data['infos']
    infos = sorted(infos, key=lambda e: e['timestamp'])
    class_set = set(class_names)
    # Build mapping from pkl names (e.g. 'Car', 'Large_Vehicle') to eval names
    def _map_name(n):
        n = n if isinstance(n, str) else n.decode()
        if n in class_set:
            return n
        n_lower = n.lower().replace(' ', '_').strip()
        alias_target = HASCO_CLASS_ALIASES.get(n_lower)
        if alias_target is not None and alias_target in class_set:
            return alias_target
        for c in class_names:
            if c.lower() == n_lower or n_lower.replace('_', '') == c.lower().replace('_', ''):
                return c
        return None

    all_annotations = EvalBoxes()
    iterator = mmcv.track_iter_progress(infos) if verbose else infos
    for info in iterator:
        sample_token = info['token']
        if 'gt_boxes' not in info or len(info['gt_boxes']) == 0:
            all_annotations.add_boxes(sample_token, [])
            continue

        gt_boxes = info['gt_boxes']  # (N, 7) [x,y,z,l,w,h,yaw]
        gt_names = info['gt_names']

        sample_boxes = []
        for i in range(len(gt_boxes)):
            name = _map_name(gt_names[i])
            if name is None:
                continue

            x, y, z = gt_boxes[i, :3]
            l, w, h = gt_boxes[i, 3:6]
            yaw_hasco = gt_boxes[i, 6]
            # Convert HASCO yaw to mmdet3d convention (same as model output)
            yaw = yaw_hasco - np.pi / 2

            # mmdet3d yaw to quaternion (rotation around z)
            quat = Quaternion(axis=[0, 0, 1], radians=yaw)
            rotation = quat.elements.tolist()  # [w,x,y,z] for pyquaternion

            # In LiDAR frame, ego is at origin
            ego_translation = (float(x), float(y), float(z))

            sample_boxes.append(HascoDetectionBox(
                sample_token=sample_token,
                translation=(float(x), float(y), float(z)),
                size=(float(l), float(w), float(h)),
                rotation=tuple(rotation),
                velocity=(0.0, 0.0),
                ego_translation=ego_translation,
                num_pts=-1,
                detection_name=name,
                detection_score=-1.0,
                attribute_name='',
                class_names=class_names))
        all_annotations.add_boxes(sample_token, sample_boxes)

    return all_annotations


def load_hasco_predictions(result_path: str,
                           max_boxes_per_sample: int,
                           class_names: tuple,
                           verbose: bool = False) -> Tuple[EvalBoxes, Dict]:
    """Load predictions from JSON (nuScenes-like format).

    Format: {'results': {sample_token: [box_dict, ...]}, 'meta': {...}}
    """
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'No results field in result file.'

    all_results = EvalBoxes()
    for sample_token, boxes_list in data['results'].items():
        sample_boxes = []
        for box_dict in boxes_list[:max_boxes_per_sample]:
            try:
                # class_names=None to allow loading; ObjectNameFilter applied later
                box = HascoDetectionBox.deserialize(box_dict, class_names=None)
                if box.detection_name in class_names:
                    sample_boxes.append(box)
            except (ValueError, KeyError):
                continue
        all_results.add_boxes(sample_token, sample_boxes)

    if verbose:
        print(f"Loaded results from {result_path}. Found detections for {len(all_results.sample_tokens)} samples.")
    return all_results, data.get('meta', {})


def add_center_dist_hasco(eval_boxes: EvalBoxes) -> EvalBoxes:
    """HASCO: boxes are in ego frame, ego at origin. ego_translation = translation."""
    for sample_token in eval_boxes.sample_tokens:
        for box in eval_boxes[sample_token]:
            box.ego_translation = box.translation
    return eval_boxes


def filter_by_object_name(eval_boxes: EvalBoxes,
                          class_names: Tuple[str, ...],
                          class_field: str = 'detection_name',
                          verbose: bool = False) -> Tuple[EvalBoxes, int, int]:
    """ObjectNameFilter: keep only boxes whose detection_name is in class_names."""
    class_set = set(class_names)
    total = 0
    after = 0
    for sample_token in eval_boxes.sample_tokens:
        boxes = eval_boxes[sample_token]
        total += len(boxes)
        filtered = [b for b in boxes if b.__getattribute__(class_field) in class_set]
        eval_boxes.boxes[sample_token] = filtered
        after += len(filtered)
    return eval_boxes, total, after


def filter_by_point_cloud_range(eval_boxes: EvalBoxes,
                                point_cloud_range: List[float],
                                verbose: bool = False) -> Tuple[EvalBoxes, int, int]:
    """ObjectRangeFilter: keep only boxes whose center is within point_cloud_range.

    point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    Uses BEV range [x_min, y_min, x_max, y_max] for center (x, y).
    """
    pcd = np.array(point_cloud_range, dtype=np.float32)
    x_min, y_min, z_min, x_max, y_max, z_max = pcd[0], pcd[1], pcd[2], pcd[3], pcd[4], pcd[5]
    total = 0
    after = 0
    for sample_token in eval_boxes.sample_tokens:
        boxes = eval_boxes[sample_token]
        total += len(boxes)
        filtered = []
        for b in boxes:
            x, y, z = b.translation[0], b.translation[1], b.translation[2]
            if (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                filtered.append(b)
        eval_boxes.boxes[sample_token] = filtered
        after += len(filtered)
    return eval_boxes, total, after


def filter_by_class_range(eval_boxes: EvalBoxes,
                          class_range: Dict[str, float],
                          class_field: str = 'detection_name',
                          verbose: bool = False) -> Tuple[EvalBoxes, int, int]:
    """DistanceFilter: keep only boxes within class_range (ego_dist < max_dist per class)."""
    total = 0
    after = 0
    for sample_token in eval_boxes.sample_tokens:
        boxes = eval_boxes[sample_token]
        total += len(boxes)
        filtered = [b for b in boxes if b.ego_dist < class_range[b.__getattribute__(class_field)]]
        eval_boxes.boxes[sample_token] = filtered
        after += len(filtered)
    return eval_boxes, total, after


def filter_eval_boxes_hasco(eval_boxes: EvalBoxes,
                            class_range: Dict[str, float],
                            class_names: Tuple[str, ...],
                            point_cloud_range: Optional[List[float]] = None,
                            class_field: str = 'detection_name',
                            verbose: bool = False,
                            label: str = 'boxes') -> EvalBoxes:
    """Apply ObjectNameFilter, ObjectRangeFilter, DistanceFilter (aligned with val_pipeline)."""
    if verbose:
        print(f"\n--- Filtering {label} ---")
        print(f"  ObjectNameFilter: classes={list(class_names)}")
        if point_cloud_range is not None:
            print(f"  ObjectRangeFilter: point_cloud_range=[x:{point_cloud_range[0]},{point_cloud_range[3]}, "
                  f"y:{point_cloud_range[1]},{point_cloud_range[4]}, z:{point_cloud_range[2]},{point_cloud_range[5]}]")
        print(f"  DistanceFilter: class_range={class_range}")

    # 1. ObjectNameFilter (for pred; GT already filtered in load_hasco_gt)
    eval_boxes, n0, n1 = filter_by_object_name(eval_boxes, class_names, class_field, verbose=False)
    if verbose:
        print(f"  => After ObjectNameFilter: {n0} -> {n1}")

    # 2. ObjectRangeFilter
    if point_cloud_range is not None:
        eval_boxes, n0, n1 = filter_by_point_cloud_range(eval_boxes, point_cloud_range, verbose=False)
        if verbose:
            print(f"  => After ObjectRangeFilter: {n0} -> {n1}")

    # 3. DistanceFilter (class_range)
    eval_boxes, n0, n1 = filter_by_class_range(eval_boxes, class_range, class_field, verbose=False)
    if verbose:
        print(f"  => After DistanceFilter: {n0} -> {n1}")

    return eval_boxes
