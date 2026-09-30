#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LVFusion ONNX inference with BEV + image visualization.

Based on jiesongz_debug/onnx_inference.ipynb and lvfusion_infer_onnx.py
Requires pre-exported ONNX models and backprojection params.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from tools.export_lvfusion_onnx import _convert_syncbn
from tools.lvfusion_external_dataset import build_external_dataset


def parse_args():
    p = argparse.ArgumentParser(description="LVFusion ONNX inference visualization")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="for voxelize (separate) or pad (merged)")
    p.add_argument("--onnx-dir", type=str, required=True, help="dir with export_2d_model, export_lidar*, export_3d_model")
    p.add_argument("--backproj-params", type=str, required=True)
    p.add_argument(
        "--cam-only",
        action="store_true",
        help="cam-only inference path: export_2d_model + export_cam_3d_model, no lidar ONNX needed",
    )
    p.add_argument("--save-path", type=str, required=True)
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="limit to first N samples (default: all)",
    )
    p.add_argument("--score-thr", type=float, default=0.4)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--merged-vfe",
        action="store_true",
        help="use export_lidar_vfe_model.onnx (voxels in)",
    )
    p.add_argument("--data-split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--ext-input",
        type=str,
        default=None,
        help="external sequence folder with imgs/, pcds/, params/cam_intrinsic.json and params/lidar2cam.json",
    )
    p.add_argument(
        "--show-input",
        action="store_true",
        help="show input (Camera+Lidar point cloud projection) on top",
    )
    p.add_argument(
        "--bev_white_background",
        action="store_true",
        help="use white background for BEV view (default: black)",
    )
    p.add_argument(
        "--draw-range",
        nargs="?",
        const=-0.35,
        type=float,
        default=None,
        metavar="GROUND_Z",
        help="draw XY perception range on PRED/BEV; ground height in body frame "
        "(default -0.35 when flag is set without value)",
    )
    p.add_argument(
        "--save-txt",
        action="store_true",
        help="save filtered predictions as txt under <save-path>/result/ "
        "(hasco label format + confidence)",
    )
    return p.parse_args()


def _load_session(path: str, providers=None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers is None:
        providers = ["CPUExecutionProvider"]
    avail = set(ort.get_available_providers())
    providers = [p for p in providers if p in avail]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def external_backproject_volume_numpy(feat_2d: np.ndarray, params: Dict) -> np.ndarray:
    """feat_2d: [B,C,Hf,Wf] -> [B,C,X,Y,Z]"""
    gather = params["gather_indices"].astype(np.int64)
    scatter = params["scatter_indices"].astype(np.int64)
    x, y, z = [int(v) for v in params["volume_xyz"].tolist()]
    b, c, hf, wf = feat_2d.shape
    feat_hw = params.get("feat_hw")
    if feat_hw is not None:
        exp_h, exp_w = int(feat_hw[0]), int(feat_hw[1])
        if (hf, wf) != (exp_h, exp_w):
            raise ValueError(
                f"Feature map (Hf={hf}, Wf={wf}) != params feat_hw=(Hf={exp_h}, Wf={exp_w})"
            )
    feat_flat = feat_2d.reshape(b, c, hf * wf)
    picked = feat_flat[:, :, gather]
    vol_flat = np.zeros((b, c, x * y * z), dtype=feat_2d.dtype)
    vol_flat[:, :, scatter] = picked
    return vol_flat.reshape(b, c, x, y, z)


def compute_pts_spatial_features(model, points_tensor: torch.Tensor) -> torch.Tensor:
    voxels, num_points, coors = model.voxelize([points_tensor])
    voxel_features = model.pts_voxel_encoder(voxels, num_points, coors)
    batch_size = int(coors[-1, 0].item() + 1)
    return model.pts_middle_encoder(voxel_features, coors, batch_size)


def pad_voxels_for_onnx(
    voxels: torch.Tensor,
    num_points: torch.Tensor,
    coors: torch.Tensor,
    max_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = voxels.shape[0]
    if n >= max_voxels:
        voxels, num_points, coors = voxels[:max_voxels], num_points[:max_voxels], coors[:max_voxels]
    else:
        pad_n = max_voxels - n
        voxels = torch.cat([voxels, torch.zeros(pad_n, *voxels.shape[1:], device=voxels.device, dtype=voxels.dtype)], dim=0)
        num_points = torch.cat([num_points, torch.zeros(pad_n, dtype=num_points.dtype, device=num_points.device)], dim=0)
        coors = torch.cat([coors, torch.zeros(pad_n, 4, dtype=coors.dtype, device=coors.device)], dim=0)
    return (
        voxels.cpu().numpy().astype(np.float32),
        num_points.cpu().numpy().astype(np.int32),
        coors.cpu().numpy().astype(np.int32),
    )


def get_lidar_3d_box_corners(box):
    x, y, z, l, w, h, yaw = box
    yaw = yaw + np.pi / 2
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [0, 0, 0, 0, h, h, h, h]
    corners = np.vstack([x_corners, y_corners, z_corners])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = R @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners.T


def project_lidar_to_img(points_3d, lidar2img, min_depth=1e-3):
    pts_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    pts_img = pts_homo @ lidar2img.T
    depth = pts_img[:, 2]
    mask = depth > min_depth
    pts_2d = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float64)
    pts_2d[mask] = pts_img[mask, :2] / depth[mask, None]
    return pts_2d, mask


def clip_line_to_image(p1, p2, width, height):
    """Clip a float line segment to image bounds using Liang-Barsky."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    xmin, ymin = 0.0, 0.0
    xmax, ymax = float(width - 1), float(height - 1)
    u1, u2 = 0.0, 1.0

    for p, q in (
        (-dx, x1 - xmin),
        (dx, xmax - x1),
        (-dy, y1 - ymin),
        (dy, ymax - y1),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > u2:
                return None
            u1 = max(u1, r)
        else:
            if r < u1:
                return None
            u2 = min(u2, r)

    q1 = np.array([x1 + u1 * dx, y1 + u1 * dy], dtype=np.float64)
    q2 = np.array([x1 + u2 * dx, y1 + u2 * dy], dtype=np.float64)
    return q1, q2


def draw_projected_edge(img, proj1, proj2, color, thickness=2, min_depth=1e-3):
    """Draw one projected 3D edge with near-plane and image-bound clipping."""
    h, w = img.shape[:2]
    p1 = np.asarray(proj1, dtype=np.float64).copy()
    p2 = np.asarray(proj2, dtype=np.float64).copy()
    d1, d2 = p1[2], p2[2]

    if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
        return
    if d1 <= min_depth and d2 <= min_depth:
        return
    if d1 <= min_depth or d2 <= min_depth:
        denom = d2 - d1
        if abs(denom) < 1e-12:
            return
        t = (min_depth - d1) / denom
        t = np.clip(t, 0.0, 1.0)
        clipped = p1 + t * (p2 - p1)
        if d1 <= min_depth:
            p1 = clipped
        else:
            p2 = clipped

    q1 = p1[:2] / p1[2]
    q2 = p2[:2] / p2[2]
    if not (np.isfinite(q1).all() and np.isfinite(q2).all()):
        return

    clipped_line = clip_line_to_image(q1, q2, w, h)
    if clipped_line is None:
        return
    q1, q2 = clipped_line
    p1_int = tuple(np.round(q1).astype(np.int32))
    p2_int = tuple(np.round(q2).astype(np.int32))
    cv2.line(img, p1_int, p2_int, color, thickness)


def _gen_color_map():
    """Color map from jsutils/misc/colormap (BGR): blue->cyan->green->yellow->red->magenta."""
    color_data = np.zeros((256, 3), dtype=np.float64)
    idx = 0
    B, G, R = 0, 0, 0
    for B in range(128, 255, 4):
        color_data[idx] = [B, G, R]
        idx += 1
    B = 255
    for G in range(0, 255, 4):
        color_data[idx] = [B, G, R]
        idx += 1
    G = 255
    for R in range(0, 255, 4):
        B = 255 - R
        color_data[idx] = [B, G, R]
        idx += 1
    B, R = 0, 255
    for G in range(255, 0, -4):
        color_data[idx] = [B, G, R]
        idx += 1
    G = 0
    for i in range(idx, 256):
        R = R - 4
        color_data[i] = [B, G, R]
    return color_data.astype(np.uint8)


def draw_points_on_image(
    img,
    points_xyz,
    lidar2img,
    draw_range=(3, 100),
    point_size=2,
    overlap=0.6,
    max_points=80000,
):
    """Project 3D points onto image, color by distance. Match jsutils draw_image_point."""
    img_orig = img.copy()
    img_draw = img.copy()
    H, W = img.shape[:2]
    min_range = min(abs(draw_range[0]), abs(draw_range[1]))
    max_range = max(abs(draw_range[0]), abs(draw_range[1]))
    pts = points_xyz[:, :3].astype(np.float64)
    dist = np.sqrt(np.sum(pts ** 2, axis=1)).reshape(-1, 1)
    pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_2d = pts_homo @ lidar2img.T
    valid = pts_2d[:, 2] > 1e-5
    pts_2d = pts_2d[valid]
    dist = dist[valid]
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    pts_2d_dist = np.hstack([pts_2d, dist])
    pts_2d_dist = pts_2d_dist[(pts_2d_dist[:, 2] > min_range) & (pts_2d_dist[:, 2] < max_range)]
    if len(pts_2d_dist) == 0:
        return img
    if len(pts_2d_dist) > max_points:
        step = len(pts_2d_dist) // max_points
        pts_2d_dist = pts_2d_dist[::step]
    sort_idx = np.lexsort((-pts_2d_dist[:, 2],))
    color_data = _gen_color_map()
    for (x, y, d) in pts_2d_dist[sort_idx].astype(np.float64):
        gray_color = int(255 - (d - min_range) / (max_range - min_range + 1e-6) * 255)
        gray_color = np.clip(gray_color, 0, 255)
        color = tuple(int(c) for c in color_data[gray_color])
        ix, iy = int(np.round(x)), int(np.round(y))
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(img_draw, (ix, iy), point_size, color, -1)
    img_draw = cv2.addWeighted(img_draw, overlap, img_orig, 1 - overlap, 0)
    return img_draw


def draw_dashed_line_2d(img, pt1, pt2, color, thickness=3, dash_len=18, gap_len=12):
    """Draw a dashed 2D line segment."""
    p1 = np.asarray(pt1, dtype=np.float64)
    p2 = np.asarray(pt2, dtype=np.float64)
    vec = p2 - p1
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return
    direction = vec / dist
    pos = 0.0
    draw = True
    while pos < dist - 1e-6:
        seg_len = dash_len if draw else gap_len
        end = min(pos + seg_len, dist)
        if draw:
            start = p1 + direction * pos
            stop = p1 + direction * end
            cv2.line(
                img,
                tuple(np.round(start).astype(np.int32)),
                tuple(np.round(stop).astype(np.int32)),
                color,
                thickness,
            )
        pos = end
        draw = not draw


def project_polyline_to_img(points_3d, lidar2img, min_depth=1e-3):
    """Project a 3D polyline to image. Return list of valid (u, v) points in order."""
    pts_homo = np.hstack([points_3d.astype(np.float64), np.ones((len(points_3d), 1))])
    proj = pts_homo @ lidar2img.T
    pts_2d = []
    for p in proj:
        if p[2] <= min_depth:
            continue
        uv = p[:2] / p[2]
        if np.isfinite(uv).all():
            pts_2d.append(uv)
    return pts_2d


def draw_perception_range_xy(
    img,
    lidar2img,
    point_cloud_range,
    z=-0.35,
    color=(0, 165, 255),
    thickness=3,
    alpha=0.5,
    num_samples=80,
):
    """Draw left/right Y boundaries of perception range on image (XY only, dashed orange)."""
    xmin, ymin, _, xmax, ymax, _ = point_cloud_range
    xs = np.linspace(xmin, xmax, num_samples)
    left_3d = np.column_stack([xs, np.full(num_samples, ymin), np.full(num_samples, z)])
    right_3d = np.column_stack([xs, np.full(num_samples, ymax), np.full(num_samples, z)])

    overlay = img.copy()
    h, w = img.shape[:2]
    for pts_3d in (left_3d, right_3d):
        pts_2d = project_polyline_to_img(pts_3d, lidar2img)
        for i in range(len(pts_2d) - 1):
            clipped = clip_line_to_image(pts_2d[i], pts_2d[i + 1], w, h)
            if clipped is None:
                continue
            draw_dashed_line_2d(overlay, clipped[0], clipped[1], color, thickness=thickness)

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    return img


def xyz_to_bev_pixel(x, y, xmin, ymin, meter_per_pixel, bev_h):
    px = (x - xmin) / meter_per_pixel
    py = (y - ymin) / meter_per_pixel
    py = bev_h - py - 1
    return np.array([px, py], dtype=np.float64)


def draw_perception_range_bev(
    bev_img,
    point_cloud_range,
    bev_range,
    meter_per_pixel=0.1,
    color=(0, 165, 255),
    thickness=3,
    alpha=0.5,
):
    """Draw left/right Y boundaries of perception range on BEV (dashed orange)."""
    pcr_xmin, pcr_ymin, _, pcr_xmax, pcr_ymax, _ = point_cloud_range
    xmin, ymin, _, _, _, _ = bev_range
    h, w = bev_img.shape[:2]

    left_start = xyz_to_bev_pixel(pcr_xmin, pcr_ymin, xmin, ymin, meter_per_pixel, h)
    left_end = xyz_to_bev_pixel(pcr_xmax, pcr_ymin, xmin, ymin, meter_per_pixel, h)
    right_start = xyz_to_bev_pixel(pcr_xmin, pcr_ymax, xmin, ymin, meter_per_pixel, h)
    right_end = xyz_to_bev_pixel(pcr_xmax, pcr_ymax, xmin, ymin, meter_per_pixel, h)

    overlay = bev_img.copy()
    draw_dashed_line_2d(overlay, left_start, left_end, color, thickness=thickness)
    draw_dashed_line_2d(overlay, right_start, right_end, color, thickness=thickness)
    cv2.addWeighted(overlay, alpha, bev_img, 1.0 - alpha, 0, bev_img)
    return bev_img


def draw_bbox(img, lidar2img, boxes_3d, color=(0, 255, 0), thickness=2):
    img = img.copy()
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    front_edges = [(0, 5), (1, 4)]
    for box in boxes_3d:
        corners_3d = get_lidar_3d_box_corners(box)
        corners_homo = np.hstack([corners_3d, np.ones((corners_3d.shape[0], 1))])
        corners_proj = corners_homo @ lidar2img.T
        for e in edges:
            draw_projected_edge(
                img, corners_proj[e[0]], corners_proj[e[1]], color, thickness)
        for e in front_edges:
            draw_projected_edge(
                img, corners_proj[e[0]], corners_proj[e[1]], (50, 100, 200), 3)
    return img


def draw_bev_box(bev_img, box7, xmin, ymin, meter_per_pixel, color=(255, 0, 0), thickness=2):
    x, y, z, l, w, h, yaw = box7
    yaw = yaw + np.pi / 2
    H, W = bev_img.shape[:2]
    cx = (x - xmin) / meter_per_pixel
    cy = (y - ymin) / meter_per_pixel
    cy = H - cy - 1
    l_pix = l / meter_per_pixel
    w_pix = w / meter_per_pixel
    x_corners = np.array([l_pix / 2, l_pix / 2, -l_pix / 2, -l_pix / 2])
    y_corners = np.array([w_pix / 2, -w_pix / 2, -w_pix / 2, w_pix / 2])
    corners = np.stack([x_corners, y_corners], axis=1)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = corners @ R
    corners[:, 0] += cx
    corners[:, 1] += cy
    corners = corners.astype(np.int32)
    for i in range(4):
        cv2.line(bev_img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, thickness)
    cv2.line(bev_img, tuple(corners[0]), tuple(corners[1]), (0, 128, 255), thickness + 1)


def draw_bev(
    points_xyz,
    pred_boxes,
    bev_range,
    gt_boxes=None,
    meter_per_pixel=0.1,
    white_background=False,
):
    xmin, ymin, zmin, xmax, ymax, zmax = bev_range
    W = int((xmax - xmin) / meter_per_pixel)
    H = int((ymax - ymin) / meter_per_pixel)
    if white_background:
        bev_img = np.full((H, W, 3), 255, dtype=np.uint8)
    else:
        bev_img = np.zeros((H, W, 3), dtype=np.uint8)
    mask = (
        (points_xyz[:, 0] >= xmin) & (points_xyz[:, 0] <= xmax)
        & (points_xyz[:, 1] >= ymin) & (points_xyz[:, 1] <= ymax)
        & (points_xyz[:, 2] >= zmin) & (points_xyz[:, 2] <= zmax)
    )
    pts = points_xyz[mask]
    if pts.shape[0] > 0:
        px = ((pts[:, 0] - xmin) / meter_per_pixel).astype(np.int32)
        py = ((pts[:, 1] - ymin) / meter_per_pixel).astype(np.int32)
        py = H - py - 1
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)
        z_norm = (pts[:, 2] - zmin) / (zmax - zmin + 1e-6)
        z_norm = np.clip(z_norm, 0.0, 1.0)
        density = {}
        for x, y in zip(px, py):
            density[(x, y)] = density.get((x, y), 0) + 1
        max_density = max(density.values()) or 1
        for i in range(len(px)):
            d = density[(px[i], py[i])] / max_density
            h = z_norm[i]
            color = np.array([int(255 * d), int(255 * h), int(255 * (1 - h))])
            if white_background:
                bev_img[py[i], px[i]] = np.minimum(bev_img[py[i], px[i]], color)
            else:
                bev_img[py[i], px[i]] = np.maximum(bev_img[py[i], px[i]], color)
    if pred_boxes is not None:
        for box in pred_boxes:
            draw_bev_box(bev_img, box, xmin, ymin, meter_per_pixel, color=(255, 0, 0), thickness=2)
    if gt_boxes is not None:
        for box in gt_boxes:
            draw_bev_box(bev_img, box, xmin, ymin, meter_per_pixel, color=(0, 255, 0), thickness=2)
    return bev_img


def resize_to_fit(img, target_w, target_h, pad_color=(0, 0, 0)):
    """Resize image to fit inside target size and pad to exact canvas."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _resolve_class_names(cfg, dataset, data_split):
    if hasattr(dataset, "CLASSES") and dataset.CLASSES:
        return list(dataset.CLASSES)
    if cfg.get("class_names"):
        return list(cfg.class_names)
    split_cfg = getattr(cfg.data, data_split, cfg.data.val)
    if isinstance(split_cfg, dict) and split_cfg.get("classes"):
        return list(split_cfg["classes"])
    if hasattr(split_cfg, "classes") and split_cfg.classes:
        return list(split_cfg.classes)
    raise RuntimeError("Cannot resolve class_names from config or dataset")


def _mmdet3d_boxes_to_hasco_lines(boxes, labels, scores, class_names):
    """Convert mmdet3d boxes to hasco txt lines: x y z l w h yaw id class [score].

    mmdet3d LiDAR boxes store bottom-center z; hasco labels use object-center z.
    """
    lines = []
    for obj_id, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x, y, z, l, w, h, yaw = box[:7]
        hasco_z = float(z + h / 2)
        hasco_yaw = float(yaw + np.pi / 2)
        class_name = class_names[int(label)]
        lines.append(
            f"{x:.6f} {y:.6f} {hasco_z:.6f} {l:.6f} {w:.6f} {h:.6f} "
            f"{hasco_yaw:.6f} {obj_id} {class_name} {float(score):.6f}"
        )
    return lines


def _save_pred_txt(txt_path, lines):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def add_panel_label(img, text, white_background=False):
    """Add a readable label after the panel has been resized/rotated."""
    h, w = img.shape[:2]
    font_scale = max(0.8, min(2.0, min(h, w) / 400.0))
    thickness = max(2, int(round(font_scale * 2)))
    org = (max(16, int(w * 0.03)), max(38, int(h * 0.08)))
    color = (0, 0, 0) if white_background else (255, 255, 255)
    outline = (255, 255, 255) if white_background else (0, 0, 0)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline, thickness + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    cfg.model.train_cfg = None
    torch_model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(torch_model, args.checkpoint, map_location="cpu")
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(torch_model)
        if hasattr(torch_model, "fp16_enabled"):
            torch_model.fp16_enabled = False
    torch_model = _convert_syncbn(torch_model)
    device = torch.device(args.device)
    torch_model = torch_model.to(device).float().eval()

    params = dict(np.load(args.backproj_params, allow_pickle=True))

    max_voxels = None
    if args.cam_only and args.merged_vfe:
        raise ValueError("--cam-only and --merged-vfe are mutually exclusive")
    if args.merged_vfe:
        voxel_params_path = os.path.join(args.onnx_dir, "voxel_params.npz")
        if not os.path.isfile(voxel_params_path):
            raise FileNotFoundError(f"Merged mode requires {voxel_params_path}")
        voxel_params = dict(np.load(voxel_params_path, allow_pickle=True))
        max_voxels = int(voxel_params["max_voxels"].item())

    def _resolve(name):
        p = os.path.join(args.onnx_dir, name)
        if os.path.isfile(p):
            return p
        simp = p.replace(".onnx", "_simplified.onnx")
        if os.path.isfile(simp):
            return simp
        raise FileNotFoundError(f"ONNX not found: {p}")

    stage1_path = _resolve("export_2d_model.onnx")
    if args.cam_only:
        stage2_path = None
        stage3_path = _resolve("export_cam_3d_model.onnx")
    else:
        stage2_path = _resolve("export_lidar_vfe_model.onnx" if args.merged_vfe else "export_lidar_model.onnx")
        stage3_path = _resolve("export_3d_model.onnx")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    if not providers:
        providers = ["CPUExecutionProvider"]

    sess1 = _load_session(stage1_path, providers=providers)
    sess2 = _load_session(stage2_path, providers=providers) if stage2_path is not None else None
    sess3 = _load_session(stage3_path, providers=providers)

    head_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    num_tasks = len(cfg.model["pts_bbox_head"]["tasks"])
    output_names = [f"task{t}_{k}" for t in range(num_tasks) for k in head_keys]

    pts_bbox_head = torch_model.pts_bbox_head
    box_type_3d, _ = get_box_type("LiDAR")

    split = getattr(cfg.data, args.data_split, cfg.data.test)
    if args.ext_input:
        dataset = build_external_dataset(cfg, args.ext_input, args.data_split)
    else:
        dataset = build_dataset(split)
    n = len(dataset) if args.num_samples is None else min(args.num_samples, len(dataset))
    bev_range = list(cfg.model.get("point_cloud_range", [0, -9.6, -5, 105.6, 9.6, 4]))
    point_cloud_range = list(bev_range)
    bev_range = list(map(lambda x, y: x + y, bev_range, [0, -10, 0, 10, 10, 0]))

    class_names = _resolve_class_names(cfg, dataset, args.data_split)
    result_txt_dir = os.path.join(args.save_path, "result") if args.save_txt else None
    if args.save_txt:
        os.makedirs(result_txt_dir, exist_ok=True)

    for i in range(n):
        data = dataset[i]

        def _unpack(x):
            if hasattr(x, "data"):
                return x.data
            return x

        img_raw = _unpack(data["img"])
        if isinstance(img_raw, torch.Tensor):
            img = img_raw.cpu().numpy().astype(np.float32)
        else:
            img = np.array(img_raw, dtype=np.float32)
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        pts = _unpack(data["points"])
        if isinstance(pts, torch.Tensor):
            pts = pts.to(device).float()
        else:
            pts = torch.from_numpy(np.asarray(pts)).to(device).float()

        img_metas_raw = _unpack(data.get("img_metas", {}))
        if isinstance(img_metas_raw, (list, tuple)):
            img_meta = img_metas_raw[0] if img_metas_raw else {}
        else:
            img_meta = img_metas_raw if isinstance(img_metas_raw, dict) else {}
        if "box_type_3d" not in img_meta:
            img_meta["box_type_3d"] = box_type_3d
        img_metas_list = [img_meta]

        img_feat_2d = sess1.run(None, {"img": img})[0]
        img_volume = external_backproject_volume_numpy(img_feat_2d, params)

        if args.cam_only:
            outs = sess3.run(None, {"img_volume": img_volume.astype(np.float32)})
        else:
            if args.merged_vfe:
                with torch.no_grad():
                    voxels, num_points, coors = torch_model.voxelize([pts])
                    v, np_pts, c = pad_voxels_for_onnx(voxels, num_points, coors, max_voxels)
                pts_bev = sess2.run(None, {"voxels": v, "num_points": np_pts, "coors": c})[0]
            else:
                with torch.no_grad():
                    pts_spatial = compute_pts_spatial_features(torch_model, pts).detach().cpu().numpy().astype(np.float32)
                pts_bev = sess2.run(None, {"pts_spatial_features": pts_spatial})[0]

            outs = sess3.run(None, {"img_volume": img_volume.astype(np.float32), "pts_bev_feat": pts_bev.astype(np.float32)})
        onnx_dict = {name: arr for name, arr in zip(output_names, outs)}

        preds_dicts = []
        for t in range(num_tasks):
            d = {}
            for k in head_keys:
                key = f"task{t}_{k}"
                if key in onnx_dict:
                    d[k] = torch.from_numpy(onnx_dict[key]).float().to(device)
            preds_dicts.append([d])

        with torch.no_grad():
            bbox_list = pts_bbox_head.get_bboxes(preds_dicts, img_metas_list, rescale=False)

        bboxes, scores, labels = bbox_list[0]
        bboxes = bboxes.tensor.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        keep = scores > args.score_thr
        pred_bboxs = bboxes[keep][:, :7]
        pred_scores = scores[keep]
        pred_labels = labels[keep]

        img_norm_cfg = img_meta.get("img_norm_cfg", {})
        mean = np.array(img_norm_cfg.get("mean", [123.675, 116.28, 103.53]))
        std = np.array(img_norm_cfg.get("std", [58.395, 57.12, 57.375]))
        input_img = img[0].transpose(1, 2, 0) * std + mean
        input_img = np.clip(input_img[:, :, ::-1], 0, 255).astype(np.uint8)

        pts_np = pts.cpu().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts)
        input_points = pts_np[:, :3] if pts_np.ndim >= 2 else pts_np.reshape(-1, 4)[:, :3]

        lidar2img_info = img_meta.get("lidar2img", {})
        if isinstance(lidar2img_info, dict):
            extr = lidar2img_info.get("extrinsic", lidar2img_info.get("rt", [np.eye(4)]))
            lidar2img = np.array(extr[0] if isinstance(extr, (list, tuple)) else extr)
        else:
            lidar2img = np.eye(4)

        img_draw = draw_bbox(input_img.copy(), lidar2img, pred_bboxs, (255, 0, 0))
        if args.draw_range is not None:
            draw_perception_range_xy(
                img_draw, lidar2img, point_cloud_range, z=args.draw_range)
        cv2.putText(img_draw, "ONNX PRED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2)

        bev_img = draw_bev(
            input_points,
            pred_bboxs,
            bev_range,
            white_background=args.bev_white_background,
        )
        if args.draw_range is not None:
            draw_perception_range_bev(bev_img, point_cloud_range, bev_range)

        if args.show_input:
            input_vis = draw_points_on_image(
                input_img.copy(), input_points, lidar2img,
                draw_range=(3, 100), point_size=2, overlap=0.4
            )
            cv2.putText(input_vis, "Camera+Lidar", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 255, 0), 2)

            pred_h, pred_w = img_draw.shape[:2]
            bottom_h = pred_h // 2
            bottom_w = pred_w // 2
            input_panel = cv2.resize(input_vis, (bottom_w, bottom_h))
            bev_panel = resize_to_fit(
                bev_img,
                bottom_w,
                bottom_h,
                pad_color=(255, 255, 255) if args.bev_white_background else (0, 0, 0),
            )
            bev_panel = add_panel_label(bev_panel, "BEV", args.bev_white_background)
            concat = np.vstack([img_draw, np.hstack([input_panel, bev_panel])])
        else:
            pred_h = img_draw.shape[0]
            bev_rot = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bev_h, bev_w = bev_rot.shape[:2]
            bev_w_new = int(bev_w * pred_h / bev_h)
            bev_panel = cv2.resize(bev_rot, (bev_w_new, pred_h))
            bev_panel = add_panel_label(bev_panel, "BEV", args.bev_white_background)
            concat = np.hstack([img_draw, bev_panel])

        filename = img_meta.get("filename", f"sample_{i:04d}.jpg")
        filename = os.path.basename(str(filename))
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            filename = f"{filename}.jpg"
        out_path = os.path.join(args.save_path, filename)
        cv2.imwrite(out_path, concat)
        print(f"[{i + 1}/{n}] Saved: {out_path}")

        if args.save_txt:
            txt_name = f"{Path(filename).stem}.txt"
            txt_path = os.path.join(result_txt_dir, txt_name)
            pred_lines = _mmdet3d_boxes_to_hasco_lines(
                pred_bboxs, pred_labels, pred_scores, class_names)
            _save_pred_txt(txt_path, pred_lines)
            print(f"[{i + 1}/{n}] Saved txt: {txt_path}")

    print(f"Done. Results saved to {args.save_path}")
    if args.save_txt:
        print(f"Prediction txt saved to {result_txt_dir}")


if __name__ == "__main__":
    main()
