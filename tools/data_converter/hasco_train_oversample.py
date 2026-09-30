# Copyright (c) OpenMMLab. All rights reserved.
"""Train-split frame oversampling for Hasco infos (build-time).

Three independent rules; each hit adds (REPEAT_TIMES - 1) extra copies.
Total appearances for a frame:
    total = 1 + sum(1 for each enabled matching rule)  # REPEAT_TIMES == 2
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from mmdet3d.core.evaluation.hasco_eval.class_aliases import HASCO_CLASS_ALIASES

# Hard-coded defaults (not exposed via CLI).
REPEAT_TIMES = 2
OVERSAMPLE_CLASSES = frozenset({
    'bicycle',
    'motorcycle',
    'tricycle',
    'pedestrian',
})
YAW_ABS_MIN = 0.523  # ~30 deg
YAW_ABS_MAX = 2.616  # ~150 deg
AZIMUTH_ABS_MIN_DEG = 17.0


def normalize_class_name(name: Any) -> str:
    """Normalize raw gt name with lowercase + HASCO aliases."""
    if name is None:
        return ''
    name_lower = str(name).lower().strip()
    alias_target = HASCO_CLASS_ALIASES.get(name_lower)
    if alias_target is not None:
        return alias_target
    return name_lower


def _get_gt_boxes_names(info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    gt_boxes = info.get('gt_boxes', None)
    gt_names = info.get('gt_names', None)
    if gt_boxes is None or gt_names is None:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=object)
    gt_boxes = np.asarray(gt_boxes)
    gt_names = np.asarray(gt_names)
    if gt_boxes.size == 0 or len(gt_names) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=object)
    return gt_boxes, gt_names


def match_class(info: Dict[str, Any]) -> bool:
    """True if any box belongs to OVERSAMPLE_CLASSES."""
    _, gt_names = _get_gt_boxes_names(info)
    for name in gt_names:
        if normalize_class_name(name) in OVERSAMPLE_CLASSES:
            return True
    return False


def match_yaw(info: Dict[str, Any]) -> bool:
    """True if any box has YAW_ABS_MIN < |yaw| < YAW_ABS_MAX (Hasco raw yaw)."""
    gt_boxes, _ = _get_gt_boxes_names(info)
    if len(gt_boxes) == 0:
        return False
    yaw_abs = np.abs(gt_boxes[:, 6].astype(np.float64))
    return bool(np.any((yaw_abs > YAW_ABS_MIN) & (yaw_abs < YAW_ABS_MAX)))


def match_azimuth(info: Dict[str, Any]) -> bool:
    """True if any box center has |atan2(y, x)| > AZIMUTH_ABS_MIN_DEG degrees."""
    gt_boxes, _ = _get_gt_boxes_names(info)
    if len(gt_boxes) == 0:
        return False
    x = gt_boxes[:, 0].astype(np.float64)
    y = gt_boxes[:, 1].astype(np.float64)
    deg = np.degrees(np.arctan2(y, x))
    return bool(np.any(np.abs(deg) > AZIMUTH_ABS_MIN_DEG))


def expand_train_infos(
    infos: Sequence[Dict[str, Any]],
    enable_class: bool = False,
    enable_yaw: bool = False,
    enable_azimuth: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Expand train infos by enabled oversample rules.

    Args:
        infos: Original train info list (one entry per unique frame).
        enable_class: Enable class-based oversampling.
        enable_yaw: Enable yaw-based oversampling.
        enable_azimuth: Enable azimuth-based oversampling.

    Returns:
        expanded_infos, stats dict.
    """
    extra_per_hit = REPEAT_TIMES - 1
    assert extra_per_hit >= 0

    if not (enable_class or enable_yaw or enable_azimuth):
        stats = dict(
            before=len(infos),
            after=len(infos),
            added=0,
            class_hits=0,
            yaw_hits=0,
            azimuth_hits=0,
            enable_class=enable_class,
            enable_yaw=enable_yaw,
            enable_azimuth=enable_azimuth,
            repeat_times=REPEAT_TIMES,
        )
        return list(infos), stats

    expanded: List[Dict[str, Any]] = []
    class_hits = 0
    yaw_hits = 0
    azimuth_hits = 0
    added = 0

    for info in infos:
        expanded.append(info)
        extras = 0

        if enable_class and match_class(info):
            class_hits += 1
            extras += extra_per_hit
        if enable_yaw and match_yaw(info):
            yaw_hits += 1
            extras += extra_per_hit
        if enable_azimuth and match_azimuth(info):
            azimuth_hits += 1
            extras += extra_per_hit

        base_token = info.get('token', '')
        for k in range(extras):
            rep = copy.deepcopy(info)
            if base_token:
                rep['token'] = f'{base_token}_rep{k + 1}'
            expanded.append(rep)
            added += 1

    stats = dict(
        before=len(infos),
        after=len(expanded),
        added=added,
        class_hits=class_hits,
        yaw_hits=yaw_hits,
        azimuth_hits=azimuth_hits,
        enable_class=enable_class,
        enable_yaw=enable_yaw,
        enable_azimuth=enable_azimuth,
        repeat_times=REPEAT_TIMES,
        oversample_classes=sorted(OVERSAMPLE_CLASSES),
        yaw_abs_range=(YAW_ABS_MIN, YAW_ABS_MAX),
        azimuth_abs_min_deg=AZIMUTH_ABS_MIN_DEG,
    )
    return expanded, stats


def print_oversample_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print oversample statistics."""
    print('\n[Train oversample]')
    print(f'  enabled: class={stats["enable_class"]}, '
          f'yaw={stats["enable_yaw"]}, azimuth={stats["enable_azimuth"]}')
    print(f'  repeat_times={stats["repeat_times"]} '
          f'(extra per hit={stats["repeat_times"] - 1})')
    if stats.get('enable_class'):
        print(f'  class rule: {stats.get("oversample_classes")} '
              f'hits={stats["class_hits"]}')
    if stats.get('enable_yaw'):
        print(f'  yaw rule: {stats.get("yaw_abs_range")} '
              f'hits={stats["yaw_hits"]}')
    if stats.get('enable_azimuth'):
        print(f'  azimuth rule: |deg|>{stats.get("azimuth_abs_min_deg")} '
              f'hits={stats["azimuth_hits"]}')
    print(f'  train size: {stats["before"]} -> {stats["after"]} '
          f'(+{stats["added"]})')
