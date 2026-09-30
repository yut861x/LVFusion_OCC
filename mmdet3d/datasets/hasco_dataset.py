# -*- coding: utf-8 -*-
import numpy as np
import tempfile
from mmdet.datasets import DATASETS
from .nuscenes_mono_dataset import NuScenesMonoDataset
from .nuscenes_dataset import NuScenesDataset
import mmcv
import cv2
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes, get_box_type
import os.path as osp
from pathlib import Path


from mmdet3d.core.evaluation.hasco_eval.class_aliases import HASCO_CLASS_ALIASES


@DATASETS.register_module()
class HascoDataset(NuScenesMonoDataset):
    """
    Hasco single-view (CAM_FRONT only) dataset.
    Similar to NuScenesCAMFrontDataset but adapted for Hasco data format.
    """
    
    CLASSES = ('car', 'pedestrian')  # Default classes, can be overridden in config

    def __init__(self,
                 with_box2d=False,
                 point_cloud_range=None,
                 class_range=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.point_cloud_range = point_cloud_range
        self.class_range = class_range  # For eval DistanceFilter (max ego_dist per class)

        # Initialize CLASSES and cat2id
        # CLASSES should be set from config file via kwargs['classes'] or self.CLASSES
        # After super().__init__, self.CLASSES should be set by the parent class
        if not hasattr(self, 'CLASSES') or self.CLASSES is None:
            # Fallback to default classes
            self.CLASSES = ('car', 'pedestrian')
        
        # Initialize cat2id mapping (name -> id)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        
        # Use version from metadata (loaded in load_annotations)
        if hasattr(self, 'metadata') and self.metadata is not None:
            version = self.metadata.get('version', 'v1.0')
        else:
            version = getattr(self, 'version', 'v1.0')
        
        self.version = version
        self.with_box2d = with_box2d

    def _map_gt_name_to_cat_id(self, name):
        """Map raw pkl label name to class id in self.cat2id."""
        if name in self.cat2id:
            return self.cat2id[name]

        name_lower = name.lower().strip()
        alias_target = HASCO_CLASS_ALIASES.get(name_lower)
        if alias_target is not None and alias_target in self.cat2id:
            return self.cat2id[alias_target]

        for cat_name, cat_id in self.cat2id.items():
            cat_name_lower = cat_name.lower().strip()
            if name_lower == cat_name_lower:
                return cat_id
            # Truncated label is shorter than class name, e.g. "Large_Vehi" -> "large_vehicle".
            if (len(name_lower) < len(cat_name_lower)
                    and cat_name_lower.startswith(name_lower)):
                return cat_id
        return None

    def _filter_imgs(self, min_size=32):
        """Filter images without ground truths or with invalid size.
        
        Override CocoDataset's _filter_imgs to avoid COCO API dependency.
        For Hasco format, we don't need to filter based on COCO annotations.
        """
        # For Hasco format, we keep all images
        # The filtering will be done in the pipeline if needed
        if not hasattr(self, 'data_infos') or self.data_infos is None:
            return []
        valid_inds = list(range(len(self.data_infos)))
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        
        Override CocoDataset's _set_group_flag to avoid dependency on COCO format.
        For 3D datasets, all images are treated the same, thus are all zeros.
        """
        if not hasattr(self, 'data_infos') or self.data_infos is None:
            self.flag = np.zeros(0, dtype=np.uint8)
        else:
            self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        # Use load_interval if available, otherwise default to 1
        load_interval = getattr(self, 'load_interval', 1)
        data_infos = data_infos[::load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        
        # Log dataset info
        logger.info(f'Loaded {len(data_infos)} samples from {ann_file}')
        logger.info(f'Dataset version: {self.version}')
        logger.info(f'Load interval: {load_interval}')
        
        # Check how many samples have CAM_FRONT
        cam_front_count = sum(1 for info in data_infos if 'CAM_FRONT' in info.get('cams', {}))
        logger.info(f'Samples with CAM_FRONT: {cam_front_count}/{len(data_infos)}')
        
        return data_infos

    def get_data_info(self, index):
        """
        Get data info from Hasco format pkl file and convert to MultiView-style format (list length = 1)
        
        Similar to NuScenesCAMFrontDataset but adapted for Hasco data format.
        """
        info = self.data_infos[index]
        
        # Get CAM_FRONT data
        # Note: The pkl file should contain 'cams' and 'CAM_FRONT' keys
        # which are generated by hasco_converter.py
        if 'cams' not in info:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Sample {index} (token: {info.get('token', 'unknown')}): Missing 'cams' key in info. Available keys: {list(info.keys())}")
            return None
        
        if 'CAM_FRONT' not in info['cams']:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Sample {index} (token: {info.get('token', 'unknown')}): Missing 'CAM_FRONT' key in cams. Available cam keys: {list(info['cams'].keys())}")
            return None
        
        cam_info = info['cams']['CAM_FRONT']
        
        # Get image path (relative to data_root)
        img_filename = osp.join(self.data_root, cam_info['data_path'])
        
        # Get lidar path
        pts_filename = osp.join(self.data_root, info['lidar_path'])
        
        # Get camera intrinsic
        cam_intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
        
        # Get sensor2lidar transformation (camera to lidar)
        # sensor2lidar_rotation and sensor2lidar_translation are in cam_info
        sensor2lidar_rotation = cam_info.get('sensor2lidar_rotation', [1.0, 0.0, 0.0, 0.0])
        sensor2lidar_translation = cam_info.get('sensor2lidar_translation', [0.0, 0.0, 0.0])
        
        # Convert quaternion to rotation matrix
        if isinstance(sensor2lidar_rotation, list):
            quat = Quaternion(sensor2lidar_rotation)
            R_cam2lidar = quat.rotation_matrix
        else:
            R_cam2lidar = np.array(sensor2lidar_rotation, dtype=np.float32)
        
        T_cam2lidar = np.array(sensor2lidar_translation, dtype=np.float32).reshape(3, 1)
        
        # Build cam2lidar matrix (4x4)
        cam2lidar = np.eye(4, dtype=np.float32)
        cam2lidar[:3, :3] = R_cam2lidar
        cam2lidar[:3, 3] = T_cam2lidar.reshape(3)
        
        # Build lidar2cam matrix (inverse of cam2lidar)
        lidar2cam = np.linalg.inv(cam2lidar)
        
        # Build lidar2img matrix: K @ [R|t]
        lidar2img = cam_intrinsic @ lidar2cam[:3, :]
        
        # Build lidar2img_aug in the format expected by RandomAugFrontImage
        # rts2proj expects a dict with 'rot', 'tran', 'intrin' keys
        # 'rot' should be sensor2lidar_rotation (camera to lidar rotation)
        # 'tran' should be sensor2lidar_translation (camera to lidar translation)
        # 'intrin' should be camera intrinsic matrix
        lidar2img_aug = {
            'intrin': cam_intrinsic,
            'rot': R_cam2lidar,  # Camera to lidar rotation matrix
            'tran': T_cam2lidar.reshape(3),  # Camera to lidar translation
            'post_rot': np.eye(3),
            'post_tran': np.zeros(3),
        }
        
        # Get sample token
        sample_token = info['token']
        
        # Get annotation info
        ann_info = None
        if not self.test_mode and 'gt_boxes' in info and len(info['gt_boxes']) > 0:
            gt_boxes = info['gt_boxes']  # Shape: (N, 7) [x, y, z, l, w, h, yaw]
            gt_names = info['gt_names']  # Shape: (N,)
            
            # Ensure cat2id is initialized
            if not hasattr(self, 'cat2id') or self.cat2id is None:
                self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
            
            # Map class names to class IDs
            # Handle case-insensitive matching and truncated names (e.g., "Large_Vehi" -> "large_vehicle")
            gt_labels_3d = []
            valid_mask = []
            
            for name in gt_names:
                cat_id = self._map_gt_name_to_cat_id(name)
                if cat_id is not None:
                    gt_labels_3d.append(cat_id)
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            
            if len(gt_labels_3d) > 0:
                gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)
                valid_mask = np.array(valid_mask, dtype=bool)
                
                # Filter valid boxes
                gt_boxes = gt_boxes[valid_mask]
                gt_names = gt_names[valid_mask]
                
                # Convert to LiDARInstance3DBoxes format
                # Format: [x, y, z, x_size, y_size, z_size, yaw, vx, vy]
                # 
                # Hasco data format:
                #   - Label format: x y z l w h yaw id class
                #   - l: 沿x轴的长度（front方向）
                #   - w: 沿y轴的长度（left方向）
                #   - h: 沿z轴的长度（up方向）
                #   - yaw: l和x轴的夹角
                #
                # mmdet3d format:
                #   - [x, y, z, x_size, y_size, z_size, yaw]
                #   - x_size: 沿x轴的长度（front方向）
                #   - y_size: 沿y轴的长度（left方向）
                #   - z_size: 沿z轴的长度（up方向）
                #
                # Size mapping: l -> x_size, w -> y_size, h -> z_size (direct mapping)
                boxes_with_velocity = np.zeros((len(gt_boxes), 9), dtype=np.float32)
                boxes_with_velocity[:, :3] = gt_boxes[:, :3]  # x, y, z
                boxes_with_velocity[:, 3:6] = gt_boxes[:, 3:6]  # l -> x_size, w -> y_size, h -> z_size
                
                # Convert yaw from Hasco format to mmdet3d format
                # Hasco yaw definition: yaw is the angle between l (length along x-axis) and x-axis
                #   - yaw=0: l aligned with x-axis, facing +x (forward)
                #   - yaw=π/2: l aligned with y-axis, facing +y (left)
                #   - yaw=-π/2: l aligned with -y-axis, facing -y (right)
                # mmdet3d LiDAR yaw definition:
                #   - yaw=0: facing -y (left)
                #   - yaw=-π/2: facing +x (forward)
                #   - yaw=π/2: facing +y (right)
                # Conversion: mmdet3d_yaw = hasco_yaw - π/2
                #   - hasco yaw=0 (朝向 +x) -> mmdet3d yaw=-π/2 (朝向 +x) ✓
                #   - hasco yaw=π/2 (朝向 +y) -> mmdet3d yaw=0 (朝向 -y) ✓
                #   - hasco yaw=-π/2 (朝向 -y) -> mmdet3d yaw=-π (朝向 +y) ✓
                boxes_with_velocity[:, 6] = gt_boxes[:, 6] - np.pi / 2  # yaw (mmdet3d format)
                boxes_with_velocity[:, 7:9] = info.get('gt_velocity', np.zeros((len(gt_boxes), 2)))[valid_mask]  # velocity
                
                gt_bboxes_3d = LiDARInstance3DBoxes(
                    boxes_with_velocity,
                    box_dim=9,
                    origin=(0.5, 0.5, 0.5)
                )
                
                ann_info = dict(
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    gt_names=gt_names
                )
            else:
                # No valid labels, create empty ann_info
                ann_info = dict(
                    gt_bboxes_3d=LiDARInstance3DBoxes(
                        np.zeros((0, 9), dtype=np.float32),
                        box_dim=9,
                        origin=(0.5, 0.5, 0.5)
                    ),
                    gt_labels_3d=np.array([], dtype=np.int64),
                    gt_names=np.array([], dtype='<U10')
                )
        else:
            # No annotations or test mode
            ann_info = dict(
                gt_bboxes_3d=LiDARInstance3DBoxes(
                    np.zeros((0, 9), dtype=np.float32),
                    box_dim=9,
                    origin=(0.5, 0.5, 0.5)
                ),
                gt_labels_3d=np.array([], dtype=np.int64),
                gt_names=np.array([], dtype='<U10')
            )
        
        # 包装成 MultiView 结构（只有 CAM_FRONT）
        # lidar2img_aug should be a dict format for RandomAugFrontImage.rts2proj
        new_info = dict(
            sample_idx=sample_token,
            img_prefix=[None],
            img_info=[dict(filename=img_filename)],
            lidar2img=dict(
                extrinsic=[lidar2img.astype(np.float32)],
                intrinsic=np.eye(4, dtype=np.float32),
                lidar2img_aug=[lidar2img_aug],  # Dict format with 'rot', 'tran', 'intrin'
                lidar2img_extra=None,
            )
        )
        
        # Add pts_filename for LoadPointsFromFile
        new_info['pts_filename'] = pts_filename

        # Prefer the path recorded by the converter. Derive it for older PKLs
        # that follow <scene>/pcds/<frame>.pcd -> <scene>/npz/<frame>.npz.
        occ_path = info.get('occ_path')
        if occ_path is None:
            lidar_rel = Path(info['lidar_path'])
            occ_path = str(lidar_rel.parent.parent / 'npz' / f'{lidar_rel.stem}.npz')
        new_info['occ_filename'] = (
            occ_path if osp.isabs(occ_path) else osp.join(self.data_root, occ_path)
        )
        
        # Always provide ann_info
        new_info['ann_info'] = ann_info
        
        return new_info

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        try:
            input_dict = self.get_data_info(index)
            if input_dict is None:
                return None
            
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            # Skip frame if point cloud is empty after filtering (PointsEmptyCheck sets this)
            if example is not None and example.get('skip_sample', False):
                return None
            # Check if filter_empty_gt is enabled and if there are valid labels
            if hasattr(self, 'filter_empty_gt') and self.filter_empty_gt:
                if example is None:
                    return None
                # Check if there are valid 3D labels
                if 'gt_labels_3d' in example:
                    if hasattr(example['gt_labels_3d'], '_data'):
                        valid_mask = example['gt_labels_3d']._data != -1
                        if not valid_mask.any():
                            return None
                    elif isinstance(example['gt_labels_3d'], (list, np.ndarray)):
                        if len(example['gt_labels_3d']) == 0:
                            return None
                        gt_labels_array = np.array(example['gt_labels_3d'])
                        if (gt_labels_array == -1).all():
                            return None
            return example
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f'Sample {index}: Exception in prepare_train_data: {type(e).__name__}: {str(e)}', exc_info=True)
            return None

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)

        # Add protection against infinite loop
        max_retries = 100
        retry_count = 0
        
        while retry_count < max_retries:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                retry_count += 1
                continue
            return data
        
        raise RuntimeError(
            f'Failed to load valid data after {max_retries} retries.'
        )

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert pts_bbox results to HASCO evaluation JSON format.

        Args:
            results: List of dict with 'pts_bbox': {boxes_3d, scores_3d, labels_3d}
            jsonfile_prefix: Output prefix for JSON file.

        Returns:
            str: Path to the output JSON file.
        """
        from mmdet3d.core.evaluation.hasco_eval.loaders import HascoDetectionBox
        from nuscenes.eval.common.data_classes import EvalBoxes

        nusc_annos = {}
        mapped_class_names = self.CLASSES

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            if 'pts_bbox' in det:
                det = det['pts_bbox']
            annos = []
            boxes = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            if hasattr(scores, 'numpy'):
                scores = scores.cpu().numpy()
            if hasattr(labels, 'numpy'):
                labels = labels.cpu().numpy()

            box_gravity_center = boxes.gravity_center.cpu().numpy()
            box_dims = boxes.dims.cpu().numpy()
            box_yaw = boxes.yaw.cpu().numpy()

            sample_token = self.data_infos[sample_id]['token']
            for i in range(len(boxes)):
                quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
                name = mapped_class_names[int(labels[i])]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat.elements.tolist(),
                    velocity=[0.0, 0.0],
                    ego_translation=box_gravity_center[i].tolist(),
                    num_pts=-1,
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name='')
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': getattr(self, 'modality', dict(use_lidar=True, use_camera=True)),
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format results for HASCO evaluation."""
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: {len(results)} != {len(self)}')

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if 'pts_bbox' in results[0]:
            result_files = self._format_bbox(
                [out['pts_bbox'] for out in results], jsonfile_prefix)
        else:
            result_files = self._format_bbox(results, jsonfile_prefix)

        return result_files, tmp_dir

    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None,
                 result_names=['pts_bbox'], show=False, out_dir=None, pipeline=None, **kwargs):
        """Evaluation in HASCO protocol (no velocity, no attribute)."""
        from mmdet3d.core.evaluation import hasco_detection_eval

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        result_path = result_files

        # GT path: ann_file may be full path or relative
        gt_path = self.ann_file
        if not osp.isabs(gt_path):
            gt_path = osp.join(self.data_root, gt_path)

        eval_result = hasco_detection_eval(
            gt_path=gt_path,
            result_path=result_path,
            class_names=self.CLASSES,
            output_dir=osp.dirname(result_path),
            point_cloud_range=getattr(self, 'point_cloud_range', None),
            class_range=getattr(self, 'class_range', None),
            verbose=True)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        # Build detail dict for compatibility
        detail = {}
        metric_prefix = 'pts_bbox_HASCO'
        for name in self.CLASSES:
            if name in eval_result.get('label_aps', {}):
                for k, v in eval_result['label_aps'][name].items():
                    detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = round(float(v), 4)
            if name in eval_result.get('label_tp_errors', {}):
                for k, v in eval_result['label_tp_errors'][name].items():
                    detail[f'{metric_prefix}/{name}_{k}'] = round(float(v), 4)
        for k, v in eval_result.get('tp_errors', {}).items():
            detail[f'{metric_prefix}/{k}'] = round(float(v), 4)
        detail[f'{metric_prefix}/DetectionScore'] = round(float(eval_result['detection_score']), 4)
        detail[f'{metric_prefix}/mAP'] = round(float(eval_result['mean_ap']), 4)

        return detail
