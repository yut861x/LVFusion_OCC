import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose, RandomFlip, LoadImageFromFile
import ipdb


@PIPELINES.register_module()
class MultiViewPipeline:
    def __init__(self, transforms, n_images, n_times=2, sequential=False):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.n_times = n_times
        self.sequential = sequential

    def __sort_list(self, old_list, order):
        new_list = []
        for i in order:
            new_list.append(old_list[i])
        return new_list

    def __call__(self, results):
        imgs = []
        extrinsics = []
        if not self.sequential:
            assert len(results['img_info']) == 6
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            ids_list = sorted(ids)  # sort & tolist
        else:
            assert len(results['img_info']) == 6 * self.n_times, f'img info: {len(results["img_info"])}, n_times: {self.n_times}'
            ids_list = np.arange(len(results['img_info'])).tolist()
        for i in ids_list:
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        # resort 2d box by random ids
        if 'gt_bboxes' in results.keys():
            gt_bboxes = self.__sort_list(results['gt_bboxes'], ids_list)
            gt_labels = self.__sort_list(results['gt_labels'], ids_list)
            gt_bboxes_ignore = self.__sort_list(results['gt_bboxes_ignore'], ids_list)
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels
            results['gt_bboxes_ignore'] = gt_bboxes_ignore

        results['lidar2img']['extrinsic'] = extrinsics
        return results


@PIPELINES.register_module()
class RandomShiftOrigin:
    def __init__(self, std):
        self.std = std

    def __call__(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results


@PIPELINES.register_module()
class RandomShiftPointCloudOrigin:
    """Randomly shift the local LiDAR origin along the vehicle Y axis.

    The input point cloud and 3D boxes are expressed in the shifted local
    frame, while the camera image remains unchanged.  ``lidar2img`` is
    therefore composed with the local-to-car translation so that it projects
    the shifted points onto the original image correctly.
    """

    def __init__(self, max_shift_y=10.0):
        if max_shift_y < 0:
            raise ValueError(f'max_shift_y must be non-negative, got {max_shift_y}')
        self.max_shift_y = float(max_shift_y)

    def __call__(self, results):
        shift_y = np.random.uniform(-self.max_shift_y, self.max_shift_y)
        # p_local = p_car - shift, so p_car = T_local_to_car @ p_local.
        local_to_car = np.eye(4, dtype=np.float32)
        local_to_car[1, 3] = shift_y
        local_shift = np.array([0.0, -shift_y, 0.0], dtype=np.float32)

        results['points'].translate(local_shift)
        if 'gt_bboxes_3d' in results:
            results['gt_bboxes_3d'].translate(local_shift)

        extrinsics = results['lidar2img']['extrinsic']
        results['lidar2img']['extrinsic'] = [
            (np.asarray(extrinsic) @ local_to_car).astype(np.float32)
            for extrinsic in extrinsics
        ]
        results['lidar2img']['origin_shift'] = np.array(
            [0.0, shift_y, 0.0], dtype=np.float32)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(max_shift_y={self.max_shift_y})'


@PIPELINES.register_module()
class KittiSetOrigin:
    def __init__(self, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def __call__(self, results):
        results['lidar2img']['origin'] = self.origin.copy()
        return results


@PIPELINES.register_module()
class KittiRandomFlip:
    def __call__(self, results):
        if results['flip']:
            results['lidar2img']['intrinsic'][0, 2] = -results['lidar2img']['intrinsic'][0, 2] + \
                                                      results['ori_shape'][1]
            flip_matrix_0 = np.eye(4, dtype=np.float32)
            flip_matrix_0[0, 0] *= -1
            flip_matrix_1 = np.eye(4, dtype=np.float32)
            flip_matrix_1[1, 1] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0]
            extrinsic = flip_matrix_0 @ extrinsic @ flip_matrix_1.T
            results['lidar2img']['extrinsic'][0] = extrinsic
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 0], -center[:, 1]) - alpha
            center_flip = center
            center_flip[:, 1] *= -1
            alpha_flip = np.arctan2(center_flip[:, 0], -center_flip[:, 1]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results


@PIPELINES.register_module()
class SunRgbdSetOrigin:
    def __call__(self, results):
        intrinsic = results['lidar2img']['intrinsic'][:3, :3]
        extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
        projection = intrinsic @ extrinsic
        h, w, _ = results['ori_shape']
        center_2d_3 = np.array([w / 2, h / 2, 1], dtype=np.float32)
        center_2d_3 *= 3
        origin = np.linalg.inv(projection) @ center_2d_3
        results['lidar2img']['origin'] = origin
        return results


@PIPELINES.register_module()
class SunRgbdTotalLoadImageFromFile(LoadImageFromFile):
    def __call__(self, results):
        file_name = results['img_info']['filename']
        flip = file_name.endswith('_flip.jpg')
        if flip:
            results['img_info']['filename'] = file_name.replace('_flip.jpg', '.jpg')
        results = super().__call__(results)
        if flip:
            results['img'] = results['img'][:, ::-1]
        return results


@PIPELINES.register_module()
class LoadFrontImageFromFile(LoadImageFromFile):
    """Load single front camera image from file.
    
    This is designed for single-view (CAM_FRONT only) datasets.
    Expects results['img_info'] to be a list with one element.
    
    Args:
        file_client_args (dict): Config dict of file clients.
    """
    
    def __call__(self, results):
        """Call function to load front camera image from file.
        
        Args:
            results (dict): Result dict containing image info.
                - img_info (list): List with one dict containing 'filename' key.
                - img_prefix (list): List with one prefix (can be None).
        
        Returns:
            dict: The result dict containing the loaded image data.
                - img (list): List with one image array (compatible with MultiView format).
                - img_shape, ori_shape, pad_shape, scale_factor: Image metadata.
        """
        # Extract single image info from list format
        assert len(results['img_info']) == 1, f'Expected 1 camera, got {len(results["img_info"])}'
        img_info = results['img_info'][0]
        img_prefix = results['img_prefix'][0] if results['img_prefix'] else None
        
        # Prepare dict for LoadImageFromFile
        _results = dict(
            img_info=img_info,
            img_prefix=img_prefix
        )
        
        # Load image using parent class
        _results = super().__call__(_results)
        
        # Format as list (compatible with MultiViewPipeline output format)
        results['img'] = [_results['img']]
        # img_shape should be a list for MultiView format
        if _results.get('img_shape') is not None:
            results['img_shape'] = [_results['img_shape']]
        else:
            results['img_shape'] = [results['img'][0].shape]
        results['ori_shape'] = _results.get('ori_shape', None)
        results['pad_shape'] = _results.get('pad_shape', None)
        results['scale_factor'] = _results.get('scale_factor', 1.0)
        results['img_norm_cfg'] = _results.get('img_norm_cfg', None)
        
        # Keep other keys from _results if needed
        for key in ['filename', 'flip', 'flip_direction']:
            if key in _results:
                results[key] = _results[key]
        
        return results


@PIPELINES.register_module()
class SunRgbdRandomFlip:
    def __call__(self, results):
        if results['flip']:
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
            results['lidar2img']['extrinsic'][0][:3, :3] = flip_matrix @ extrinsic @ flip_matrix.T
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 1], center[:, 0]) - alpha
            center_flip = center @ flip_matrix
            alpha_flip = np.arctan2(center_flip[:, 1], center_flip[:, 0]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results
