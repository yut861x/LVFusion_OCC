#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External sequence dataset for LVFusion inference visualization."""

import copy
import json
from pathlib import Path

import numpy as np

from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _get_nested(data, keys):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load_matrix_from_json(path, candidates, shape):
    data = _read_json(path)
    value = None
    for keys in candidates:
        value = _get_nested(data, keys)
        if value is not None:
            break
    if value is None:
        value = data

    mat = np.asarray(value, dtype=np.float32)
    if mat.size == int(np.prod(shape)):
        mat = mat.reshape(shape)
    if shape == (4, 4) and mat.shape == (3, 4):
        padded = np.eye(4, dtype=np.float32)
        padded[:3, :4] = mat
        mat = padded
    if mat.shape != shape:
        raise ValueError(f"Invalid matrix shape in {path}: got {mat.shape}, expected {shape}")
    return mat


def _find_external_pairs(root):
    img_dir = root / "imgs"
    pcd_dir = root / "pcds"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing external image folder: {img_dir}")
    if not pcd_dir.is_dir():
        raise FileNotFoundError(f"Missing external point cloud folder: {pcd_dir}")

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pcd_exts = {".pcd", ".bin", ".npy"}
    images = {p.stem: p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in img_exts}
    pcds = {p.stem: p for p in sorted(pcd_dir.iterdir()) if p.suffix.lower() in pcd_exts}
    stems = sorted(set(images) & set(pcds))
    if not stems:
        raise RuntimeError(f"No matched image/point-cloud pairs found under {root}")
    missing_pcd = sorted(set(images) - set(pcds))
    missing_img = sorted(set(pcds) - set(images))
    print(
        f"[ext_input] Found images={len(images)}, point_clouds={len(pcds)}, "
        f"matched_pairs={len(stems)}, skipped={len(missing_pcd) + len(missing_img)}"
    )
    if missing_pcd:
        print(f"[ext_input] Warning: {len(missing_pcd)} images have no matched point cloud")
        print(f"[ext_input]   image-only examples: {', '.join(missing_pcd[:5])}")
    if missing_img:
        print(f"[ext_input] Warning: {len(missing_img)} point clouds have no matched image")
        print(f"[ext_input]   point-cloud-only examples: {', '.join(missing_img[:5])}")
    return [(stem, images[stem], pcds[stem]) for stem in stems]


class ExternalLVFusionDataset:
    """Minimal external-sequence dataset that reuses the config test pipeline."""

    def __init__(self, root, pipeline, box_type_3d="LiDAR"):
        self.root = Path(root)
        self.pipeline = Compose(pipeline)
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.data_infos = self._build_data_infos()

    def _build_data_infos(self):
        params_dir = self.root / "params"
        cam_intrinsic = _load_matrix_from_json(
            params_dir / "cam_intrinsic.json",
            [
                ("param", "cam_K", "data"),
                ("cam_K", "data"),
                ("cam_intrinsic",),
                ("intrinsic",),
                ("K",),
            ],
            (3, 3),
        )
        lidar2cam = _load_matrix_from_json(
            params_dir / "lidar2cam.json",
            [
                ("param", "sensor_calib", "data"),
                ("sensor_calib", "data"),
                ("lidar2cam",),
                ("lidar2cam_rt",),
                ("extrinsic",),
            ],
            (4, 4),
        )

        r_lidar2cam = lidar2cam[:3, :3]
        t_lidar2cam = lidar2cam[:3, 3]
        r_cam2lidar = r_lidar2cam.T
        t_cam2lidar = -r_cam2lidar @ t_lidar2cam

        lidar2img = np.eye(4, dtype=np.float32)
        lidar2img[:3, :4] = cam_intrinsic @ lidar2cam[:3, :]
        lidar2img_aug = dict(
            intrin=cam_intrinsic,
            rot=r_cam2lidar,
            tran=t_cam2lidar,
            post_rot=np.eye(3, dtype=np.float32),
            post_tran=np.zeros(3, dtype=np.float32),
        )

        infos = []
        for stem, img_path, pcd_path in _find_external_pairs(self.root):
            infos.append(dict(
                sample_idx=stem,
                img_prefix=[None],
                img_info=[dict(filename=str(img_path))],
                pts_filename=str(pcd_path),
                file_name=str(pcd_path),
                lidar2img=dict(
                    extrinsic=[lidar2img.copy()],
                    intrinsic=np.eye(4, dtype=np.float32),
                    lidar2img_aug=[copy.deepcopy(lidar2img_aug)],
                    lidar2img_extra=None,
                ),
            ))
        return infos

    def __len__(self):
        return len(self.data_infos)

    def pre_pipeline(self, results):
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def __getitem__(self, index):
        results = copy.deepcopy(self.data_infos[index])
        self.pre_pipeline(results)
        return self.pipeline(results)


def build_external_dataset(cfg, ext_input, data_split="val"):
    """Build ExternalLVFusionDataset using the test pipeline from config."""
    test_cfg = cfg.data.get("test", None)
    pipeline = (
        test_cfg.get("pipeline", cfg.get("test_pipeline"))
        if test_cfg is not None
        else cfg.get("test_pipeline")
    )
    if pipeline is None:
        raise RuntimeError("Cannot find test pipeline in config for --ext-input")
    box_type_3d = test_cfg.get("box_type_3d", "LiDAR") if test_cfg is not None else "LiDAR"
    dataset = ExternalLVFusionDataset(ext_input, pipeline, box_type_3d=box_type_3d)
    print(f"[ext_input] Loaded {len(dataset)} external samples from {ext_input}")
    return dataset
