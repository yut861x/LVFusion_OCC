"""Occupancy ground-truth loading for Hasco datasets."""
from io import BytesIO

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadOccupancyFromFile:
    """Load semantic OCC while preserving on-disk class IDs 1..11."""

    def __init__(self, file_client_args=dict(backend='disk'),
                 expected_range=None, expected_voxel_size=None,
                 unknown_ids=(12, 255)):
        self.file_client = mmcv.FileClient(**file_client_args)
        self.expected_range = expected_range
        self.expected_voxel_size = expected_voxel_size
        self.unknown_ids = tuple(int(value) for value in unknown_ids)

    def __call__(self, results):
        filename = results.get('occ_filename')
        if not filename:
            raise KeyError('样本缺少 occ_filename；请重新生成包含 occ_path 的 PKL')
        with np.load(BytesIO(self.file_client.get(filename)), allow_pickle=False) as data:
            required = {'occ_gt', 'point_cloud_range', 'voxel_size', 'axis_order',
                        'class_names', 'background_meaning', 'format_version'}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f'{filename} 缺少 OCC 字段: {sorted(missing)}')
            occ = data['occ_gt']
            point_range = np.asarray(data['point_cloud_range'], dtype=np.float32)
            voxel_size = np.asarray(data['voxel_size'], dtype=np.float32)
            axis_order = str(data['axis_order'].item())
            class_names = tuple(str(name) for name in data['class_names'].tolist())
            background_meaning = str(data['background_meaning'].item())
            format_version = int(data['format_version'].item())
            stored_valid_mask = (data['valid_mask'].copy()
                                 if 'valid_mask' in data.files else None)

        if occ.dtype != np.uint8 or occ.ndim != 3:
            raise ValueError(f'{filename}: occ_gt 必须是 uint8 三维数组，实际 {occ.dtype} {occ.shape}')
        if point_range.shape != (6,) or voxel_size.shape != (3,) or axis_order != 'ZYX':
            raise ValueError(f'{filename}: OCC 几何元数据无效')
        grid_xyz = np.rint((point_range[3:] - point_range[:3]) / voxel_size).astype(int)
        if occ.shape != tuple(grid_xyz[::-1]):
            raise ValueError(f'{filename}: occ_gt={occ.shape} 与范围/分辨率不一致')
        if self.expected_range is not None and not np.allclose(
                point_range, self.expected_range, atol=1e-5, rtol=0):
            raise ValueError(f'{filename}: point_cloud_range 与训练配置不一致')
        if self.expected_voxel_size is not None and not np.allclose(
                voxel_size, self.expected_voxel_size, atol=1e-5, rtol=0):
            raise ValueError(f'{filename}: voxel_size 与训练配置不一致')
        if format_version not in (2, 3, 4):
            raise ValueError(f'{filename}: 不支持 OCC format_version={format_version}')
        valid_ids = np.arange(len(class_names), dtype=np.uint8)
        if not np.isin(occ, np.r_[valid_ids, np.uint8(255)]).all():
            raise ValueError(f'{filename}: occ_gt 包含 class_names 范围外的类别 ID')

        occ = np.ascontiguousarray(occ)
        if stored_valid_mask is not None:
            if (stored_valid_mask.shape != occ.shape
                    or stored_valid_mask.dtype != np.bool_):
                raise ValueError(f'{filename}: valid_mask形状或类型无效')
            valid_mask = stored_valid_mask
        else:
            valid_mask = np.ones(occ.shape, dtype=np.bool_)
        for value in self.unknown_ids:
            valid_mask &= occ != value
        results['occ_gt'] = occ
        results['occ_valid_mask'] = np.ascontiguousarray(valid_mask)
        results['occ_point_cloud_range'] = point_range
        results['occ_voxel_size'] = voxel_size
        results['occ_axis_order'] = axis_order
        results['occ_class_names'] = class_names
        results['occ_background_meaning'] = background_meaning
        results['occ_format_version'] = format_version
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(unknown_ids={self.unknown_ids}, '
                f'expected_range={self.expected_range}, '
                f'expected_voxel_size={self.expected_voxel_size})')
