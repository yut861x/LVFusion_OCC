# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile


@PIPELINES.register_module()
class LoadPointsFromPCD(LoadPointsFromFile):
    """Load points from ASCII PCD file format.
    
    This class extends LoadPointsFromFile to handle ASCII PCD files.
    It reads PCD files with format:
    FIELDS x y z row colunm intensity reflectivity confidence noise timestamp id
    and extracts x, y, z, intensity. Optionally loads id for PointsIDFilter.
    
    Args:
        coord_type (str): The type of coordinates of points cloud.
        load_dim (int): The dimension of the loaded points. Defaults to 4 (x, y, z, intensity).
        use_dim (list[int]): Which dimensions of the points to be used. Defaults to [0, 1, 2, 3].
        load_id (bool): Whether to load the id field for PointsIDFilter. Defaults to False.
        file_client_args (dict): Config dict of file clients.
    """

    def __init__(self, *args, load_id=False, **kwargs):
        if load_id:
            # Override load_dim and use_dim to include id (index 10 in PCD fields)
            kwargs['load_dim'] = 5
            kwargs['use_dim'] = [0, 1, 2, 3, 4]  # x, y, z, intensity, id
        super().__init__(*args, **kwargs)
        self.load_id = load_id

    def _load_points(self, pts_filename):
        """Load points from ASCII PCD file.
        
        Args:
            pts_filename (str): Path to PCD file.
            
        Returns:
            np.ndarray: Point cloud data flattened as 1D array (will be reshaped later).
        """
        if pts_filename.endswith('.pcd'):
            return self._load_pcd_ascii(pts_filename)
        else:
            # Fallback to parent class for other formats
            return super()._load_points(pts_filename)
    
    def _load_pcd_ascii(self, pcd_path):
        """Load ASCII PCD file and extract x, y, z, intensity (and optionally id).
        
        Args:
            pcd_path (str): Path to PCD file (can be absolute or relative).
            
        Returns:
            np.ndarray: Point cloud data flattened as 1D array (will be reshaped later).
        """
        # Check file existence and read directly
        # PCD files are ASCII text, so we read them as text files
        mmcv.check_file_exist(pcd_path)
        with open(pcd_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find DATA section and FIELDS
        data_start = -1
        fields = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('FIELDS'):
                # Parse fields: FIELDS x y z row colunm intensity reflectivity confidence noise timestamp id
                fields = line.split()[1:]  # Skip 'FIELDS'
            elif line.startswith('DATA'):
                data_start = i + 1
                break
        
        if data_start < 0:
            raise ValueError(f"Cannot find DATA section in PCD file: {pcd_path}")
        
        if len(fields) == 0:
            raise ValueError(f"Cannot find FIELDS in PCD file: {pcd_path}")
        
        # Find indices of x, y, z, intensity, and optionally id
        try:
            x_idx = fields.index('x')
            y_idx = fields.index('y')
            z_idx = fields.index('z')
            intensity_idx = fields.index('intensity')
        except ValueError as e:
            raise ValueError(f"Cannot find required fields (x, y, z, intensity) in PCD file: {pcd_path}. Available fields: {fields}")
        
        id_idx = None
        if self.load_id:
            try:
                id_idx = fields.index('id')
            except ValueError:
                raise ValueError(f"Cannot find 'id' field in PCD file: {pcd_path}. Available fields: {fields}")
        
        max_idx = max(x_idx, y_idx, z_idx, intensity_idx)
        if id_idx is not None:
            max_idx = max(max_idx, id_idx)
        
        # Read data lines
        points = []
        intensities = []  # Collect intensities for normalization
        
        for line in lines[data_start:]:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split()
            if len(parts) < max_idx + 1:
                continue  # Skip incomplete lines
            
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
                intensity = float(parts[intensity_idx])
                row = [x, y, z, intensity]
                if id_idx is not None:
                    row.append(float(parts[id_idx]))
                points.append(row)
                intensities.append(intensity)
            except (ValueError, IndexError) as e:
                # Skip lines with invalid data
                continue
        
        if len(points) == 0:
            raise ValueError(f"No valid points found in PCD file: {pcd_path}")
        
        # Convert to numpy array
        points_array = np.array(points, dtype=np.float32)
        
        # Normalize intensity to [0, 1] range
        intensities_array = np.array(intensities, dtype=np.float32)
        intensity_min = intensities_array.min()
        intensity_max = intensities_array.max()
        
        if intensity_max > intensity_min:
            normalized_intensity = (intensities_array - intensity_min) / (intensity_max - intensity_min)
        else:
            normalized_intensity = np.zeros_like(intensities_array)
        
        points_array[:, 3] = normalized_intensity
        
        # Return flattened array (will be reshaped in __call__)
        return points_array.flatten()

