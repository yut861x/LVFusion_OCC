# -*- coding: utf-8 -*-
import numpy as np
import tempfile
from os import path as osp
from mmdet.datasets import DATASETS
from .nuscenes_mono_dataset import NuScenesMonoDataset
from .nuscenes_dataset import (
    NuScenesDataset,
    output_to_nusc_box,
    lidar_nusc_box_to_global,
)

import mmcv
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

try:
    from tools.data_converter.nuscenes_converter import get_2d_boxes
except:
    print('import error')


@DATASETS.register_module()
class NuScenesCAMFrontDataset(NuScenesMonoDataset):
    """
    NuScenes single-view (CAM_FRONT only) dataset.
    Output format keeps MultiView-style img_metas for compatibility.
    """

    def __init__(self,
                 with_box2d=False,
                 point_cloud_range=None,
                 **kwargs):
        super().__init__(**kwargs)
        # 前向范围，用于 range-filtered 评估（与 val_pipeline 的 ObjectRangeFilter 一致）
        self.point_cloud_range = point_cloud_range

        # Use version from metadata (loaded in load_annotations)
        # NuScenesMonoDataset.__init__ may override self.version with default 'v1.0-trainval'
        # So we need to restore it from metadata if available
        if hasattr(self, 'metadata') and self.metadata is not None:
            version = self.metadata.get('version', 'v1.0-mini')
        else:
            # Fallback to 'v1.0-mini' if metadata is not available
            version = getattr(self, 'version', 'v1.0-mini')
        
        # Update self.version to ensure consistency
        self.version = version
        
        self.nusc = NuScenes(
            version=version,
            dataroot=self.data_root,
            verbose=True
        )

        # Check if BEV segmentation is needed by checking pipeline
        self.with_bev_seg = False
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            for transform in self.pipeline.transforms:
                if hasattr(transform, 'with_bev_seg') and transform.with_bev_seg:
                    self.with_bev_seg = True
                    break
        
        # Only load maps if BEV segmentation is needed
        if self.with_bev_seg:
            self.scene2map = get_scene2map(self.nusc)
            self.maps = get_nusc_maps(map_folder=self.data_root)
        else:
            self.scene2map = None
            self.maps = None

        self.with_box2d = with_box2d

        # Ensure eval_detection_configs has class_range for all CLASSES (e.g. large_vehicle)
        if hasattr(self, 'eval_detection_configs') and self.eval_detection_configs is not None:
            if hasattr(self.eval_detection_configs, 'class_range'):
                for c in self.CLASSES:
                    if c not in self.eval_detection_configs.class_range:
                        self.eval_detection_configs.class_range[c] = 50.0

        # BEV map config (same as Map_Dataset2)
        xbound = [-50, 50, 0.5]
        ybound = [-50, 50, 0.5]
        zbound = [-10, 10, 20.0]

        self.nx = np.array(
            [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]],
            dtype=np.int64
        )
        self.dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        self.bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])

        self.lane_thickness = 2

    def _filter_imgs(self, min_size=32):
        """Filter images without ground truths or with invalid size.
        
        Override CocoDataset's _filter_imgs to avoid COCO API dependency.
        For NuScenes format, we don't need to filter based on COCO annotations.
        """
        # For NuScenes format, we keep all images
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
            ann_file (str): Path of the annotation file.

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
            
            # Debug: Check input_dict before pipeline
            if not hasattr(self, '_debug_pipeline') and index < 3:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f'Sample {index} input_dict keys before pipeline: {list(input_dict.keys())}')
                if 'ann_info' in input_dict:
                    ann = input_dict['ann_info']
                    logger.info(f'Sample {index} ann_info keys: {list(ann.keys())}')
                    if 'gt_labels_3d' in ann:
                        logger.info(f'Sample {index} gt_labels_3d shape/len: {ann["gt_labels_3d"].shape if hasattr(ann["gt_labels_3d"], "shape") else len(ann["gt_labels_3d"])}')
                if index == 2:
                    self._debug_pipeline = True
            
            example = self.pipeline(input_dict)
            
            # Check if filter_empty_gt is enabled and if there are valid labels
            if hasattr(self, 'filter_empty_gt') and self.filter_empty_gt:
                if example is None:
                    if not hasattr(self, '_warned_pipeline_none'):
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f'Sample {index}: pipeline returned None')
                        self._warned_pipeline_none = True
                    return None
                # Check if there are valid 3D labels
                if 'gt_labels_3d' in example:
                    if hasattr(example['gt_labels_3d'], '_data'):
                        # Check if there are any valid labels (not all -1)
                        valid_mask = example['gt_labels_3d']._data != -1
                        if not valid_mask.any():
                            if not hasattr(self, '_warned_empty_gt'):
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(f'Sample {index}: filtered out due to empty gt_labels_3d (all -1)')
                                self._warned_empty_gt = True
                            return None
                    elif isinstance(example['gt_labels_3d'], (list, np.ndarray)):
                        if len(example['gt_labels_3d']) == 0:
                            if not hasattr(self, '_warned_empty_gt'):
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(f'Sample {index}: filtered out due to empty gt_labels_3d (length 0)')
                                self._warned_empty_gt = True
                            return None
                        # Check if there are any valid labels (not all -1)
                        gt_labels_array = np.array(example['gt_labels_3d'])
                        if (gt_labels_array == -1).all():
                            if not hasattr(self, '_warned_empty_gt'):
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(f'Sample {index}: filtered out due to empty gt_labels_3d (all -1)')
                                self._warned_empty_gt = True
                            return None
            return example
        except Exception as e:
            if not hasattr(self, '_warned_exception'):
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f'Sample {index}: Exception in prepare_train_data: {type(e).__name__}: {str(e)}', exc_info=True)
                self._warned_exception = True
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
            # If CAM_FRONT data is missing, return None
            # The caller should handle this appropriately
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
        failed_indices = []
        
        while retry_count < max_retries:
            data = self.prepare_train_data(idx)
            if data is None:
                failed_indices.append(idx)
                idx = self._rand_another(idx)
                retry_count += 1
                continue
            return data
        
        # If we've exhausted retries, raise an error with debug info
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f'Failed to load valid data after {max_retries} retries.')
        logger.error(f'Failed indices (first 10): {failed_indices[:10]}')
        logger.error(f'Total dataset size: {len(self.data_infos)}')
        
        # Try to get more info about why it failed
        if len(failed_indices) > 0:
            test_idx = failed_indices[0]
            info = self.data_infos[test_idx]
            logger.error(f'Sample {test_idx} info keys: {list(info.keys())}')
            if 'cams' in info:
                logger.error(f'Sample {test_idx} cams keys: {list(info["cams"].keys())}')
                if 'CAM_FRONT' in info['cams']:
                    cam_info = info['cams']['CAM_FRONT']
                    logger.error(f'Sample {test_idx} CAM_FRONT keys: {list(cam_info.keys())}')
                    logger.error(f'Sample {test_idx} data_path: {cam_info.get("data_path", "NOT FOUND")}')
        
        raise RuntimeError(
            f'Failed to load valid data after {max_retries} retries. '
            f'This may indicate that all samples in the dataset are invalid or missing CAM_FRONT data. '
            f'Check logs above for details.'
        )

    def get_data_info(self, index):
        """
        Get data info from NuScenes format pkl file and convert to MultiView-style format (list length = 1)
        
        This method calls NuScenesDataset.get_data_info to get the full multi-view data,
        then extracts only CAM_FRONT (index 0) to match NuScenesMultiViewDataset format.
        """
        # Create a minimal object to call NuScenesDataset.get_data_info without full initialization
        # We create an object instance without calling __init__ to avoid load_annotations
        temp_dataset = object.__new__(NuScenesDataset)
        
        # Set only the necessary attributes for get_data_info to work
        temp_dataset.data_infos = self.data_infos
        temp_dataset.modality = getattr(self, 'modality', {'use_camera': True, 'use_lidar': True})
        temp_dataset.test_mode = self.test_mode
        temp_dataset.use_valid_flag = getattr(self, 'use_valid_flag', False)
        temp_dataset.with_velocity = getattr(self, 'with_velocity', True)
        temp_dataset.sequential = getattr(self, 'sequential', False)
        temp_dataset.n_times = getattr(self, 'n_times', 1)
        temp_dataset.prev_only = getattr(self, 'prev_only', False)
        temp_dataset.next_only = getattr(self, 'next_only', False)
        temp_dataset.train_adj_ids = getattr(self, 'train_adj_ids', None)
        temp_dataset.test_adj_ids = getattr(self, 'test_adj_ids', None)
        temp_dataset.min_interval = getattr(self, 'min_interval', 0)
        temp_dataset.max_interval = getattr(self, 'max_interval', 3)
        temp_dataset.verbose = getattr(self, 'verbose', False)
        temp_dataset.test_adj = getattr(self, 'test_adj', 'next')
        temp_dataset.fix_direction = getattr(self, 'fix_direction', False)
        temp_dataset.test_time_id = getattr(self, 'test_time_id', None)
        temp_dataset.speed_mode = getattr(self, 'speed_mode', 'abs_velo')
        
        # Set box_type_3d and box_mode_3d (required by get_ann_info)
        from mmdet3d.core.bbox import get_box_type
        box_type_3d = getattr(self, 'box_type_3d', 'LiDAR')
        if not isinstance(box_type_3d, str):
            box_type_3d = 'LiDAR'
        temp_dataset.box_type_3d, temp_dataset.box_mode_3d = get_box_type(box_type_3d)
        
        # Set CLASSES and cat2id (required by get_ann_info for label mapping)
        temp_dataset.CLASSES = getattr(self, 'CLASSES', None)
        if temp_dataset.CLASSES is None:
            # Fallback to default NuScenes classes if not set
            temp_dataset.CLASSES = NuScenesDataset.CLASSES
        temp_dataset.cat2id = {name: i for i, name in enumerate(temp_dataset.CLASSES)}
        
        # Call NuScenesDataset.get_data_info to get full multi-view data
        data_info = temp_dataset.get_data_info(index)
        
        # Extract CAM_FRONT (index 0) from the multi-view data
        cam_idx = 0  # CAM_FRONT is always the first camera
        
        # Get CAM_FRONT data
        img_filename = data_info['img_filename'][cam_idx]
        lidar2img = data_info['lidar2img'][cam_idx]
        lidar2img_aug = data_info['lidar2img_aug'][cam_idx]
        lidar2img_extra = data_info.get('lidar2img_extra', [None] * len(data_info['img_filename']))[cam_idx] if 'lidar2img_extra' in data_info else None
        
        # Get sample_idx
        sample_token = data_info['sample_idx']
        
        # Get annotation info (same for all cameras)
        # In test_mode, NuScenesDataset.get_data_info doesn't load ann_info by default
        # But we need it for visualization, so load it manually if missing
        ann_info = data_info.get('ann_info', None)
        if ann_info is None:
            # Load ann_info manually for test mode visualization
            # temp_dataset already has all necessary attributes set above
            ann_info = temp_dataset.get_ann_info(index)
        
        # Filter out invalid labels (-1) to match NuScenesMultiViewDataset behavior
        # This is critical to avoid errors in torch.sparse_coo_tensor which doesn't accept negative indices
        if ann_info is not None:
            gt_labels_3d = ann_info.get('gt_labels_3d', None)
            if gt_labels_3d is not None and len(gt_labels_3d) > 0:
                # Filter out labels with value -1 (invalid/ignored classes)
                mask = gt_labels_3d >= 0
                if mask.any():
                    # Only keep valid labels
                    ann_info['gt_bboxes_3d'] = ann_info['gt_bboxes_3d'][mask]
                    ann_info['gt_labels_3d'] = gt_labels_3d[mask]
                    if 'gt_names' in ann_info:
                        ann_info['gt_names'] = ann_info['gt_names'][mask]
                else:
                    # If all labels are invalid, create empty ann_info
                    # ObjectNameFilter will handle empty arrays correctly
                    # Use self.box_type_3d instead of hardcoded LiDARInstance3DBoxes
                    ann_info['gt_bboxes_3d'] = self.box_type_3d(
                        np.zeros((0, 9), dtype=np.float32),
                        box_dim=9,
                        origin=(0.5, 0.5, 0.5)
                    )
                    ann_info['gt_labels_3d'] = np.array([], dtype=np.int64)
                    if 'gt_names' in ann_info:
                        ann_info['gt_names'] = np.array([], dtype='<U10')
        
        # Get cam_info from original info for 2D boxes (if needed)
        info = self.data_infos[index]
        cam_info = info['cams']['CAM_FRONT'] if 'cams' in info and 'CAM_FRONT' in info['cams'] else None

        # Get pts_filename from data_info (for LoadPointsFromFile)
        pts_filename = data_info.get('pts_filename', None)
        if pts_filename is None and 'lidar_path' in info:
            # Fallback to original info if not in data_info
            pts_filename = info['lidar_path']

        # 包装成 MultiView 结构（只有 CAM_FRONT）
        # Match NuScenesMultiViewDataset format exactly
        new_info = dict(
            sample_idx=sample_token,
            img_prefix=[None],
            img_info=[dict(filename=img_filename)],
            lidar2img=dict(
                extrinsic=[lidar2img.astype(np.float32)],
                intrinsic=np.eye(4, dtype=np.float32),
                lidar2img_aug=[lidar2img_aug],  # List with single element, matching NuScenesMultiViewDataset
                lidar2img_extra=[lidar2img_extra] if lidar2img_extra is not None else None,
            )
        )
        
        # Add pts_filename for LoadPointsFromFile (even if dummy=True, it's good to have it)
        if pts_filename is not None:
            new_info['pts_filename'] = pts_filename

        # Always provide ann_info, even if empty (pipeline expects it)
        if ann_info is not None:
            ann = ann_info.copy()
        else:
            # Create empty ann_info for samples without annotations
            # Use self.box_type_3d instead of hardcoded CameraInstance3DBoxes
            ann = dict(
                gt_bboxes_3d=self.box_type_3d(
                    np.zeros((0, 9), dtype=np.float32),
                    box_dim=9,
                    origin=(0.5, 0.5, 0.5)
                ),
                gt_labels_3d=np.array([], dtype=np.int64),
                gt_names=np.array([], dtype='<U10')
            )

        # BEV segmentation GT (only if needed)
        if self.with_bev_seg:
            bev_seg_gt = self._get_map_by_sample_token(sample_token)
            ann['gt_bev_seg'] = bev_seg_gt

        # optional 2D boxes (only CAM_FRONT)
        if self.with_box2d and cam_info is not None:
            gt_bboxes, gt_labels, gt_bboxes_ignore = [], [], []
            sample_data_token = cam_info.get('sample_data_token', None)
            if sample_data_token is not None:
                try:
                    coco_infos = get_2d_boxes(
                        self.nusc,
                        sample_data_token,
                        visibilities=['', '1', '2', '3', '4'],
                        mono3d=False
                    )
                except Exception as e:
                    # If 2D boxes cannot be loaded, continue without them
                    coco_infos = []
            else:
                coco_infos = []

            for coco_info in coco_infos:
                if coco_info is None or coco_info.get('ignore', False):
                    continue
                x1, y1, w, h = coco_info['bbox']
                if w < 1 or h < 1 or coco_info['area'] <= 0:
                    continue
                if coco_info['category_id'] < 0 or coco_info['category_id'] > 9:
                    continue

                bbox = [x1, y1, x1 + w, y1 + h]
                if coco_info.get('iscrowd', False):
                    gt_bboxes_ignore.append(bbox)
                else:
                    gt_bboxes.append(bbox)
                    gt_labels.append(coco_info['category_id'])

            ann['bboxes'] = [np.array(gt_bboxes, dtype=np.float32)]
            ann['labels'] = [np.array(gt_labels, dtype=np.int64)]
            ann['bboxes_ignore'] = [np.array(gt_bboxes_ignore, dtype=np.float32)]

        new_info['ann_info'] = ann

        return new_info

    # ======================= Evaluation (pts_bbox, 1 sample per result) =======================

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert pts_bbox results to NuScenes submission format.

        One result per sample (no 6-camera merge). Uses LiDAR box path like NuScenesDataset.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES
        print('Start to convert detection format (pts_bbox, front-cam single sample)...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            # output_to_nusc_box from nuscenes_dataset (LiDAR); sequential=False for single frame
            boxes = output_to_nusc_box(
                det,
                sequential=False,
                n_times=1,
                info=self.data_infos[sample_id],
                speed_mode='abs_velo',
                min_interval=0,
                max_interval=3,
                test_adj='next',
                test_time_id=None,
                fix_direction=False,
                test_adj_ids=None,
                verbose=False,
            )
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.eval_detection_configs,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                            'car', 'construction_vehicle', 'bus', 'truck',
                            'trailer', 'large_vehicle',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute.get(
                            name, 'vehicle.parked')
                else:
                    if name == 'pedestrian':
                        attr = 'pedestrian.standing'
                    elif name == 'bus':
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute.get(
                            name, 'vehicle.parked')
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 vis_mode=False):
        """评估：若配置了 point_cloud_range 则使用与 val_pipeline 一致的 GT 过滤（前向范围），
        否则走父类官方 NuScenes 全场景评估。"""
        if result_names is None:
            result_names = (
                ['pts_bbox'] if results and 'pts_bbox' in results[0]
                else ['img_bbox'])
        # 使用前向范围过滤 GT 的独立评估（不调用官方 NuScenesEval）
        if getattr(self, 'point_cloud_range', None) is not None:
            from mmdet3d.core.evaluation import nuscenes_cam_front_detection_eval
            if jsonfile_prefix is None:
                import tempfile
                tmp = tempfile.mkdtemp()
                jsonfile_prefix = osp.join(tmp, 'results')
            out_dir = osp.dirname(jsonfile_prefix) if jsonfile_prefix else None
            class_range = None
            if hasattr(self, 'eval_detection_configs') and self.eval_detection_configs is not None:
                if hasattr(self.eval_detection_configs, 'class_range'):
                    class_range = dict(self.eval_detection_configs.class_range)
            return nuscenes_cam_front_detection_eval(
                self,
                results,
                point_cloud_range=self.point_cloud_range,
                class_range=class_range,
                output_dir=out_dir,
                jsonfile_prefix=jsonfile_prefix,
                verbose=(logger is not None),
            )
        return super().evaluate(
            results,
            metric=metric,
            logger=logger,
            jsonfile_prefix=jsonfile_prefix,
            result_names=result_names,
            show=show,
            out_dir=out_dir,
            pipeline=pipeline,
            vis_mode=vis_mode,
        )

    # ======================= BEV MAP =======================

    def coord_transform(self, pts, pose):
        pts = convert_points_to_local(pts, pose)
        pts = np.round((pts - self.bx[:2]) / self.dx[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        return pts

    def _get_map_by_sample_token(self, sample_token):
        egopose = self.nusc.get(
            'ego_pose',
            self.nusc.get(
                'sample_data',
                self.nusc.get('sample', sample_token)['data']['LIDAR_TOP']
            )['ego_pose_token']
        )

        rot = Quaternion(egopose['rotation']).rotation_matrix
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        pose = [
            egopose['translation'][0],
            egopose['translation'][1],
            np.cos(yaw),
            np.sin(yaw)
        ]

        bev_seg_gt_road = np.zeros((self.nx[0], self.nx[1]))
        bev_seg_gt_lane = np.zeros((self.nx[0], self.nx[1]))

        scene_name = self.nusc.get(
            'scene',
            self.nusc.get('sample', sample_token)['scene_token']
        )['name']
        map_name = self.scene2map[scene_name]
        nmap = self.maps[map_name]

        # road
        for record in nmap.drivable_area:
            polygons = [nmap.extract_polygon(p) for p in record['polygon_tokens']]
            for poly in polygons:
                ext = self.coord_transform(np.array(poly.exterior.coords), pose)
                bev_seg_gt_road = cv2.fillPoly(bev_seg_gt_road, [ext], 1)

        # lane
        for layer_name in ['road_divider', 'lane_divider']:
            for record in getattr(nmap, layer_name):
                line = nmap.extract_line(record['line_token'])
                if line.is_empty:
                    continue
                line = self.coord_transform(np.array(line.xy).T, pose)
                bev_seg_gt_lane = cv2.polylines(
                    bev_seg_gt_lane,
                    [line],
                    isClosed=False,
                    color=1,
                    thickness=self.lane_thickness
                )

        bev_seg_gt_road = np.flip(bev_seg_gt_road, axis=1).copy()
        bev_seg_gt_lane = np.flip(bev_seg_gt_lane, axis=1).copy()
        return np.stack([bev_seg_gt_road, bev_seg_gt_lane], axis=-1)


# ======================= utils =======================

def get_scene2map(nusc):
    scene2map = {}
    for rec in nusc.scene:
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    return scene2map


def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def convert_points_to_local(points, pose):
    points = points - pose[:2]
    rot = get_rot(np.arctan2(pose[3], pose[2])).T
    return np.dot(points, rot)


def get_nusc_maps(map_folder='./data/nuscenes'):
    return {
        map_name: NuScenesMap(dataroot=map_folder, map_name=map_name)
        for map_name in [
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
            "singapore-onenorth",
        ]
    }
