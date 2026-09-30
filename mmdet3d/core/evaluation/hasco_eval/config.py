# Copyright (c) OpenMMLab. All rights reserved.
"""HASCO detection evaluation config.

Reference: nuScenes detection_cvpr_2019, but:
- No velocity prediction -> exclude vel_err from TP metrics
- No attribute prediction -> exclude attr_err from TP metrics
- NDS renamed to DetectionScore
"""


class HascoDetectionConfig:
    """Detection config for HASCO dataset.

    Same structure as nuScenes DetectionConfig but accepts custom class_names.
    Excludes velocity and attribute from TP metrics.
    """

    # TP metrics for HASCO: no vel_err, no attr_err
    TP_METRICS = ['trans_err', 'scale_err', 'orient_err']

    def __init__(self,
                 class_names,
                 class_range=None,
                 dist_fcn='center_distance',
                 dist_ths=None,
                 dist_th_tp=2.0,
                 min_recall=0.1,
                 min_precision=0.1,
                 max_boxes_per_sample=500,
                 mean_ap_weight=5):
        if dist_ths is None:
            dist_ths = [0.5, 1.0, 2.0, 4.0]
        if class_range is None:
            # Default: 50m for vehicles, 40m for others
            class_range = {name: 50 for name in class_names}

        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."
        assert set(class_range.keys()) == set(class_names), "class_range keys must match class_names."

        self.class_names = list(class_names)
        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

    @property
    def dist_fcn_callable(self):
        """Return the distance function corresponding to dist_fcn."""
        if self.dist_fcn == 'center_distance':
            from nuscenes.eval.common.utils import center_distance
            return center_distance
        raise ValueError(f'Unknown distance function {self.dist_fcn}!')
