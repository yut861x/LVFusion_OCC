# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os
import json
from os import path as osp
from pyquaternion import Quaternion
from pathlib import Path
import glob
from tqdm import tqdm

try:
    import open3d as o3d
except ImportError:
    print("Warning: open3d not installed, will use numpy to read PCD files")


def load_pcd_file(pcd_path):
    """Load PCD file and extract x, y, z, intensity.
    
    Args:
        pcd_path (str): Path to PCD file.
        
    Returns:
        np.ndarray: Point cloud data with shape (N, 4) containing [x, y, z, intensity].
    """
    try:
        # Try using open3d first
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            # If has colors, use intensity from colors (assuming grayscale)
            colors = np.asarray(pcd.colors)
            intensity = colors[:, 0] if colors.shape[1] >= 1 else np.zeros(len(points))
        elif pcd.has_intensities():
            intensity = np.asarray(pcd.intensities)
        else:
            intensity = np.zeros(len(points))
        
        # Combine x, y, z, intensity
        if len(intensity.shape) == 1:
            intensity = intensity.reshape(-1, 1)
        points_with_intensity = np.hstack([points, intensity])
        return points_with_intensity
    except:
        # Fallback: read as binary or ASCII
        # Try to read as binary first (KITTI format)
        try:
            points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
            return points
        except:
            # Try ASCII format
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            # Find DATA section
            data_start = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('DATA'):
                    data_start = i + 1
                    break
            if data_start > 0:
                data_lines = lines[data_start:]
                points = []
                for line in data_lines:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        intensity = float(parts[3]) if len(parts) > 3 else 0.0
                        points.append([x, y, z, intensity])
                return np.array(points, dtype=np.float32)
            else:
                raise ValueError(f"Cannot parse PCD file: {pcd_path}")


def load_label_file(label_path):
    """Load label file and extract annotations.
    
    Args:
        label_path (str): Path to label file.
        
    Returns:
        tuple: (gt_boxes, gt_names) where:
            gt_boxes: np.ndarray with shape (N, 7) containing [x, y, z, l, w, h, yaw]
            gt_names: np.ndarray with shape (N,) containing class names
    """
    gt_boxes = []
    gt_names = []
    
    if not os.path.exists(label_path):
        return np.array([]).reshape(0, 7), np.array([], dtype='<U30')
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        
        # Extract: x, y, z, l, w, h, yaw (first 7 values)
        # Coordinate system definitions:
        #   - hasco LiDAR: x朝前, y朝左, z朝上
        #   - mmdet3d LiDAR: x朝前, y朝左, z朝上
        #   These are the SAME coordinate system!
        # 
        # Label format: x y z l w h yaw id class
        #   - xyz: 底面中心坐标
        #   - l: 沿x轴的长度（front方向）
        #   - w: 沿y轴的长度（left方向）
        #   - h: 沿z轴的长度（up方向）
        #   - yaw: l和x轴的夹角
        #
        # mmdet3d format: [x, y, z, x_size, y_size, z_size, yaw]
        #   - x_size: 沿x轴的长度（front方向）
        #   - y_size: 沿y轴的长度（left方向）
        #   - z_size: 沿z轴的长度（up方向）
        #
        # Mapping: l -> x_size, w -> y_size, h -> z_size (no swap needed!)
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        l, w, h = float(parts[3]), float(parts[4]), float(parts[5])
        yaw = float(parts[6])
        # parts[7] is ID (skip)
        # parts[8] is type (class name)
        class_name = parts[8]
        
        # Convert to standard format: [x, y, z, x_size, y_size, z_size, yaw]
        # Direct mapping: l -> x_size, w -> y_size, h -> z_size
        gt_boxes.append([x, y, z, l, w, h, yaw])
        gt_names.append(class_name)
    
    if len(gt_boxes) == 0:
        return np.array([]).reshape(0, 7), np.array([], dtype='<U30')
    
    return np.array(gt_boxes, dtype=np.float32), np.array(gt_names, dtype='<U30')


def load_camera_intrinsic(intrinsic_path):
    """Load camera intrinsic parameters.
    
    Args:
        intrinsic_path (str): Path to cam_intrinsic.json.
        
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix.
    """
    with open(intrinsic_path, 'r') as f:
        data = json.load(f)
    
    cam_K = np.array(data['param']['cam_K']['data'], dtype=np.float32)
    return cam_K


def load_lidar2cam(lidar2cam_path):
    """Load lidar to camera transformation.
    
    Args:
        lidar2cam_path (str): Path to lidar2cam.json.
        
    Returns:
        np.ndarray: 4x4 transformation matrix from lidar to camera.
    """
    with open(lidar2cam_path, 'r') as f:
        data = json.load(f)
    
    # Use float64 so rotation→quaternion stays numerically orthogonal under
    # pyquaternion's strict check (float32 SVD can leave ~1e-8 residuals).
    transform = np.array(data['param']['sensor_calib']['data'], dtype=np.float64)
    return transform


def create_hasco_infos(
    root_path,
    info_prefix,
    version='v1.0',
    train_ratio=0.8,
    skip_empty_labels=False,
    oversample_class=False,
    oversample_yaw=False,
    oversample_azimuth=False,
    require_occ=False,
    occ_dir_name='npz',
):
    """Create info file for Hasco dataset.
    
    This function processes all date folders in the root_path. Each date folder
    should have its own parameters (cam_intrinsic.json and lidar2cam.json) in
    the params/ subdirectory. Parameters are loaded once per folder and shared
    by all samples in that folder.
    
    To add new data:
    1. Create a new date folder (e.g., 20251220) in root_path
    2. Add pcds/, imgs/, labels/, and params/ subdirectories
    3. Place your data files and parameter files
    4. Re-run this function - it will automatically discover and process the new folder
    
    Args:
        root_path (str): Path of the data root. Should contain date folders.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data. Default: 'v1.0'
        train_ratio (float): Ratio of training data. Default: 0.8
        skip_empty_labels (bool): If True, samples with existing but empty label
            files will be skipped. Samples without label files are always skipped.
            Default: False.
        oversample_class (bool): Oversample train frames with rare classes.
        oversample_yaw (bool): Oversample train frames with large |yaw|.
        oversample_azimuth (bool): Oversample train frames with side azimuth.
        require_occ (bool): Skip samples without a matching npz OCC label.
        
    Folder structure expected:
        root_path/
            date_folder_1/  (e.g., 20251215)
                pcds/
                imgs/
                labels/
                params/
                    cam_intrinsic.json
                    lidar2cam.json
            date_folder_2/  (e.g., 20251217_1)
                pcds/
                imgs/
                labels/
                params/
                    cam_intrinsic.json  # Can be different from date_folder_1
                    lidar2cam.json      # Can be different from date_folder_1
            ...
    """
    root_path = Path(root_path)
    
    # Accept either a parent containing multiple date folders or one date
    # folder directly. The latter is convenient for a small OCC pilot set.
    if (root_path / 'pcds').is_dir() and (root_path / 'imgs').is_dir():
        date_folders = [root_path]
    else:
        date_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    if len(date_folders) == 0:
        raise ValueError(f"No date folders found in {root_path}")
    
    print(f"Found {len(date_folders)} date folders: {[d.name for d in date_folders]}")
    
    # Collect all samples
    all_infos = []
    samples_by_folder = {}
    
    for date_folder in tqdm(date_folders, desc="Processing date folders"):
        date_name = date_folder.name
        
        # Get paths
        pcds_dir = date_folder / 'pcds'
        imgs_dir = date_folder / 'imgs'
        labels_dir = date_folder / 'labels'
        params_dir = date_folder / 'params'
        occ_dir = date_folder / occ_dir_name
        
        if not pcds_dir.exists() or not imgs_dir.exists():
            print(f"Warning: Missing pcds or imgs in {date_name}, skipping folder")
            continue
        
        # Load parameters for this date folder (each folder has its own parameters)
        intrinsic_path = params_dir / 'cam_intrinsic.json'
        lidar2cam_path = params_dir / 'lidar2cam.json'
        
        if not intrinsic_path.exists() or not lidar2cam_path.exists():
            print(f"Warning: Missing params in {date_name}, skipping folder")
            continue
        
        # Load camera intrinsic and lidar2cam for this date folder
        # These parameters are specific to this date folder and will be used
        # for all samples in this folder
        try:
            cam_intrinsic = load_camera_intrinsic(str(intrinsic_path))
            lidar2cam = load_lidar2cam(str(lidar2cam_path))
            print(f"Loaded parameters for {date_name}: intrinsic shape {cam_intrinsic.shape}, lidar2cam shape {lidar2cam.shape}")
        except Exception as e:
            print(f"Error loading parameters for {date_name}: {e}, skipping folder")
            continue
        
        # Pre-compute cam2lidar transformation for this folder (used for all samples)
        lidar2cam_mat = lidar2cam
        
        # Extract rotation and translation from lidar2cam
        R_lidar2cam = lidar2cam_mat[:3, :3]
        T_lidar2cam = lidar2cam_mat[:3, 3]
        
        # For rotation matrices, inverse = transpose
        R_cam2lidar = R_lidar2cam.T
        
        # For translation: cam2lidar translation = -R_cam2lidar @ T_lidar2cam
        T_cam2lidar = -R_cam2lidar @ T_lidar2cam
        
        # Orthogonalize rotation only (translation kept as-is from calibration).
        # float64 SVD avoids false "non-orthogonal" failures in pyquaternion.
        R_cam2lidar = np.asarray(R_cam2lidar, dtype=np.float64)
        U, _, Vt = np.linalg.svd(R_cam2lidar)
        R_cam2lidar_ortho = U @ Vt
        
        # Ensure determinant is +1 (proper rotation)
        if np.linalg.det(R_cam2lidar_ortho) < 0:
            U[:, -1] *= -1
            R_cam2lidar_ortho = U @ Vt
        
        # Convert to quaternion
        quat = Quaternion(matrix=R_cam2lidar_ortho)
        quat_elements = quat.elements  # Returns [w, x, y, z]
        
        # Get all PCD files in this date folder
        pcd_files = sorted(pcds_dir.glob('*.pcd'))
        
        if len(pcd_files) == 0:
            print(f"Warning: No PCD files found in {date_name}, skipping folder")
            continue
        
        print(f"Processing {len(pcd_files)} samples in {date_name}")
        
        for pcd_file in pcd_files:
            # Get base name (without extension)
            base_name = pcd_file.stem
            
            # Find corresponding image and label
            img_file = imgs_dir / f"{base_name}.png"
            if not img_file.exists():
                img_file = imgs_dir / f"{base_name}.jpg"
            if not img_file.exists():
                print(f"Warning: No image found for {base_name} in {date_name}, skipping")
                continue
            
            label_file = labels_dir / f"{base_name}.txt"
            if not label_file.exists():
                print(f"Warning: No label found for {base_name} in {date_name}, skipping")
                continue

            occ_file = occ_dir / f"{base_name}.npz"
            if require_occ and not occ_file.exists():
                print(f"Warning: No OCC found for {base_name} in {date_name}, skipping")
                continue
            
            # Create info dict (similar to nuscenes format)
            # Use relative path from root_path
            lidar_path = str(pcd_file.relative_to(root_path))
            img_path = str(img_file.relative_to(root_path))
            
            # Create token (unique identifier)
            token = f"{date_name}_{base_name}"
            
            # Get timestamp from filename if possible
            timestamp = 0
            try:
                # Try to extract timestamp from filename (e.g., "1764768033_042059")
                parts = base_name.split('_')
                if len(parts) >= 2:
                    timestamp = int(parts[0]) * 1000000 + int(parts[1])
            except:
                pass
            
            # Build info dict
            info = {
                'lidar_path': lidar_path,
                'token': token,
                'timestamp': timestamp,
                'sweeps': [],  # No sweeps for now
                'cams': dict(),
                # Since we don't have ego pose, use identity transformations
                'lidar2ego_translation': [0.0, 0.0, 0.0],
                'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],  # Quaternion [w, x, y, z]
                'ego2global_translation': [0.0, 0.0, 0.0],
                'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
            }
            if occ_file.exists():
                info['occ_path'] = str(occ_file.relative_to(root_path))
            
            # Add camera info (only CAM_FRONT for now)
            # Use the pre-computed cam2lidar transformation for this date folder
            # All samples in this folder share the same parameters
            cam_info = {
                'data_path': img_path,
                'sample_data_token': token,
                'sensor2lidar_translation': T_cam2lidar.tolist(),
                'sensor2lidar_rotation': quat_elements.tolist(),
                'cam_intrinsic': cam_intrinsic.tolist(),
            }
            
            info['cams']['CAM_FRONT'] = cam_info
            
            gt_boxes, gt_names = load_label_file(str(label_file))
            has_valid_label = len(gt_boxes) > 0

            if not has_valid_label and skip_empty_labels:
                continue

            # Convert yaw to match nuscenes format (SECOND format: -yaw - pi/2)
            # But first, let's keep original format and adjust in dataset class if needed
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = gt_names
            # Add velocity (zeros for now, since we don't have it)
            info['gt_velocity'] = np.zeros((len(gt_boxes), 2), dtype=np.float32)
            # Add num_lidar_pts (zeros for now)
            info['num_lidar_pts'] = np.zeros(len(gt_boxes), dtype=np.int32)
            info['num_radar_pts'] = np.zeros(len(gt_boxes), dtype=np.int32)
            info['valid_flag'] = np.ones(len(gt_boxes), dtype=bool)
            info['has_valid_label'] = has_valid_label
            
            all_infos.append(info)
            samples_by_folder.setdefault(date_name, []).append(info)
    
    print(f"\nTotal samples collected: {len(all_infos)}")
    
    if len(all_infos) == 0:
        raise ValueError("No valid samples found. Please check your data structure.")
    
    # Split into train and val by date folder
    # Each date folder is split independently with train_ratio (default 80% train, 20% val)
    # This ensures balanced distribution across all date folders
    train_infos = []
    val_infos = []
    
    print(f"\nSplitting samples by date folder (train_ratio={train_ratio:.1%}):")
    
    # Split each folder independently
    np.random.seed(42)  # Fixed seed for reproducibility
    for folder_name, folder_samples in sorted(samples_by_folder.items()):
        n_samples = len(folder_samples)
        if n_samples == 0:
            continue
        
        # Shuffle samples in this folder
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * train_ratio)

        # Split this folder's samples
        folder_train_indices = indices[:split_idx]
        folder_val_indices = indices[split_idx:]

        folder_train = [folder_samples[i] for i in folder_train_indices]
        folder_val = [folder_samples[i] for i in folder_val_indices]
        
        train_infos.extend(folder_train)
        val_infos.extend(folder_val)
        
        # Print folder statistics
        train_count = len(folder_train)
        val_count = len(folder_val)
        train_pct = train_count / n_samples * 100 if n_samples > 0 else 0
        val_pct = val_count / n_samples * 100 if n_samples > 0 else 0
        nonempty_label_count = sum(1 for s in folder_samples if s.get('has_valid_label', False))
        empty_label_count = n_samples - nonempty_label_count
        print(
            f"  {folder_name}: {n_samples} samples "
            f"(nonempty_labels={nonempty_label_count}, empty_labels={empty_label_count}) -> "
            f"train={train_count} ({train_pct:.1f}%), val={val_count} ({val_pct:.1f}%)"
        )
    
    print(f"\nTotal: Train samples={len(train_infos)}, Val samples={len(val_infos)}")
    print(f"Overall ratio: Train={len(train_infos)/len(all_infos)*100:.1f}%, Val={len(val_infos)/len(all_infos)*100:.1f}%")
    
    # Save train info
    metadata = dict(version=version)
    for info in train_infos:
        info.pop('has_valid_label', None)
    for info in val_infos:
        info.pop('has_valid_label', None)

    if oversample_class or oversample_yaw or oversample_azimuth:
        from tools.data_converter.hasco_train_oversample import (
            expand_train_infos,
            print_oversample_stats,
        )
        train_infos, oversample_stats = expand_train_infos(
            train_infos,
            enable_class=oversample_class,
            enable_yaw=oversample_yaw,
            enable_azimuth=oversample_azimuth,
        )
        print_oversample_stats(oversample_stats)

    data = dict(infos=train_infos, metadata=metadata)
    info_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    mmcv.dump(data, info_path)
    print(f"Saved train info to {info_path}")
    
    # Save val info
    data['infos'] = val_infos
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    mmcv.dump(data, info_val_path)
    print(f"Saved val info to {info_val_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hasco data converter')
    parser.add_argument('--root-path', type=str, required=True, help='Path to data root')
    parser.add_argument('--info-prefix', type=str, default='hasco', help='Prefix of info file')
    parser.add_argument('--version', type=str, default='v1.0', help='Dataset version')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument(
        '--skip-empty-labels',
        action='store_true',
        help='Skip samples whose label files exist but contain no valid objects',
    )
    parser.add_argument(
        '--oversample-class',
        action='store_true',
        help='Oversample train frames containing bicycle/motorcycle/tricycle/pedestrian',
    )
    parser.add_argument(
        '--oversample-yaw',
        action='store_true',
        help='Oversample train frames with objects whose |yaw| is in (0.523, 2.616)',
    )
    parser.add_argument(
        '--oversample-azimuth',
        action='store_true',
        help='Oversample train frames with objects outside |azimuth|<=17 deg',
    )
    parser.add_argument(
        '--require-occ',
        action='store_true',
        help='Only include samples with a matching npz/<frame>.npz OCC label',
    )
    parser.add_argument('--occ-dir-name', default='npz',
                        help='OCC label directory inside each scene (default: npz)')
    
    args = parser.parse_args()
    
    create_hasco_infos(
        root_path=args.root_path,
        info_prefix=args.info_prefix,
        version=args.version,
        train_ratio=args.train_ratio,
        skip_empty_labels=args.skip_empty_labels,
        oversample_class=args.oversample_class,
        oversample_yaw=args.oversample_yaw,
        oversample_azimuth=args.oversample_azimuth,
        require_occ=args.require_occ,
        occ_dir_name=args.occ_dir_name,
    )
