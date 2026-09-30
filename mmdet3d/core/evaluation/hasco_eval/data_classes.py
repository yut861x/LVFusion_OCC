# Copyright (c) OpenMMLab. All rights reserved.
"""HASCO detection data classes.

Same as nuScenes DetectionBox but accepts custom class names (no DETECTION_NAMES check).
"""

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from nuscenes.eval.common.data_classes import EvalBox


class HascoDetectionBox(EvalBox):
    """Detection box for HASCO. Same as DetectionBox but accepts custom class names."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),
                 num_pts: int = -1,
                 detection_name: str = 'car',
                 detection_score: float = -1.0,
                 attribute_name: str = '',
                 class_names: tuple = None):
        super().__init__(
            sample_token=sample_token,
            translation=translation,
            size=size,
            rotation=rotation,
            velocity=velocity,
            ego_translation=ego_translation,
            num_pts=num_pts)

        if class_names is not None and detection_name not in class_names:
            raise ValueError(f'detection_name {detection_name} not in class_names {class_names}')
        assert type(detection_score) == float, 'detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'detection_score may not be NaN!'

        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def serialize(self) -> dict:
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict, class_names: tuple = None):
        return cls(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=tuple(content.get('ego_translation', (0, 0, 0))),
            num_pts=int(content.get('num_pts', -1)),
            detection_name=content['detection_name'],
            detection_score=float(content.get('detection_score', -1.0)),
            attribute_name=content.get('attribute_name', ''),
            class_names=class_names)


class HascoDetectionMetrics:
    """Metrics for HASCO. Uses only trans_err, scale_err, orient_err (no vel, attr).
    DetectionScore = (5*mAP + sum(3 TP scores)) / 8
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        return {name: np.mean(list(d.values())) for name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """Mean TP error across classes for each metric (trans, scale, orient only)."""
        errors = {}
        for metric_name in self.cfg.TP_METRICS:
            class_errors = []
            for name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(name, metric_name))
            errors[metric_name] = float(np.nanmean(class_errors))
        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        """1 - error, clamped to [0, 1]."""
        scores = {}
        for metric_name, err in self.tp_errors.items():
            score = max(0.0, 1.0 - err)
            scores[metric_name] = score
        return scores

    @property
    def detection_score(self) -> float:
        """DetectionScore (replaces NDS for HASCO, no vel/attr).

        Formula:
          TP_trans_score  = max(0, 1 - mATE)
          TP_scale_score  = max(0, 1 - mASE)
          TP_orient_score = max(0, 1 - mAOE)
          DetectionScore = (5 * mAP + TP_trans_score + TP_scale_score + TP_orient_score) / 8
        """
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores))
        return total
