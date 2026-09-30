#!/usr/bin/env python
"""Continue nuScenes data preparation from existing info files.

This script skips the info file creation step and only runs:
1. export_2d_annotation for train and val
2. create_groundtruth_database
"""
import argparse
from os import path as osp
from tools.data_converter import nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database

def continue_nuscenes_prep(root_path, info_prefix, version, out_dir):
    """Continue nuScenes data preparation from existing info files.
    
    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version (e.g., 'v1.0-trainval').
        out_dir (str): Output directory of the groundtruth database info.
    """
    train_version = f'{version}-trainval'
    
    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    
    # Check if info files exist
    import os
    if not os.path.exists(info_train_path):
        print(f'Warning: {info_train_path} not found. Skipping 2D annotation export for train.')
    else:
        print(f'Exporting 2D annotations for train from {info_train_path}...')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=train_version)
    
    if not os.path.exists(info_val_path):
        print(f'Warning: {info_val_path} not found. Skipping 2D annotation export for val.')
    else:
        print(f'Exporting 2D annotations for val from {info_val_path}...')
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=train_version)
    
    # Create groundtruth database
    if os.path.exists(info_train_path):
        print(f'Creating groundtruth database from {info_train_path}...')
        create_groundtruth_database(
            'NuScenesDataset', root_path, info_prefix,
            f'{out_dir}/{info_prefix}_infos_train.pkl')
    else:
        print(f'Warning: {info_train_path} not found. Skipping groundtruth database creation.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continue nuScenes data preparation')
    parser.add_argument('--root-path', type=str, required=True,
                       help='specify the root path of dataset')
    parser.add_argument('--out-dir', type=str, required=True,
                       help='output directory')
    parser.add_argument('--extra-tag', type=str, default='nuscenes',
                       help='prefix of info files')
    parser.add_argument('--version', type=str, default='v1.0',
                       help='dataset version')
    args = parser.parse_args()
    
    continue_nuscenes_prep(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        version=args.version,
        out_dir=args.out_dir
    )

