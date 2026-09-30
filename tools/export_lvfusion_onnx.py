#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export LVFusion to ONNX in 3 stages (方案A, backprojection 外置):

Stage 1 (img_2d.onnx):
  img -> img_feat_2d
  - img_backbone + img_neck (+ img_neck_fuse)

Stage 2 (pts_bev.onnx):
  pts_spatial_features -> pts_bev_feat
  - pts_backbone + pts_neck (+ pts_neck_fuse)
  Note: pts_voxel_layer / pts_voxel_encoder / pts_middle_encoder are EXTERNAL.

  With --include-vfe: extends Stage 2 to include HardVFE + PointPillarsScatter.
  Input: voxels [max_voxels, max_points, C], num_points [max_voxels], coors [max_voxels, 4]
  Padding: voxels 不足时补零，num_points 补 0，coors 补 (0,0,0,0)，使用 scatter_max 避免覆盖有效体素。

Stage 3 (fusion_head.onnx):
  img_volume + pts_bev_feat -> raw head outputs
  - img_neck_3d + fusion_layer + CenterHead forward (NO decode/NMS)

Backprojection is done OUTSIDE the ONNX graphs. This script does not modify model code.
"""

import argparse
import os
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model

import onnx
import onnxruntime
import numpy as np

try:
    from onnxsim import simplify

    HAS_ONNXSIM = True
except Exception:
    HAS_ONNXSIM = False
    warnings.warn("onnxsim not installed, will skip model simplification")

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmseg.ops import resize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export LVFusion to ONNX (3-stage, backprojection external)"
    )
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file path")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./output/lvfusion_onnx",
        help="output directory for ONNX models",
    )
    parser.add_argument("--opset-version", type=int, default=13, help="ONNX opset")
    parser.add_argument(
        "--dummy-img-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="override dummy image size for stage1 export (default from cfg.data_config['input_size'])",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="simplify ONNX models with onnxsim (if installed)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_false",
        dest="simplify",
        help="disable onnxsim simplify",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="which stages to export (default: 1 2 3). Use 1 3 to skip stage2 if export fails.",
    )
    parser.add_argument(
        "--export-backproj-params",
        action="store_true",
        default=True,
        help="also export backprojection params (default: True)",
    )
    parser.add_argument(
        "--no-export-backproj-params",
        action="store_false",
        dest="export_backproj_params",
        help="disable exporting backprojection params",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="dataset sample index used for backprojection params (default: 0)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"]
    )
    parser.add_argument(
        "--include-vfe",
        action="store_true",
        default=False,
        help="include HardVFE + PointPillarsScatter in stage2 ONNX (fixed voxel count, pad with zeros)",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=None,
        help="max voxels for VFE export (default: from cfg pts_voxel_layer.max_voxels test value)",
    )
    return parser.parse_args()


def _maybe_simplify(onnx_path: str) -> str:
    if not HAS_ONNXSIM:
        return onnx_path
    onnx_model = onnx.load(onnx_path)
    simplified_model, ok = simplify(onnx_model)
    if not ok:
        print(f"[WARN] onnxsim simplify failed, keep original: {onnx_path}")
        return onnx_path
    out_path = onnx_path.replace(".onnx", "_simplified.onnx")
    onnx.save_model(simplified_model, out_path)
    return out_path


def _check_onnx(onnx_path: str):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)


def _convert_syncbn(model: nn.Module) -> nn.Module:
    """Convert SyncBN to BN for ONNX export."""
    try:
        from mmcv.cnn import convert_sync_batchnorm

        return convert_sync_batchnorm(model)
    except Exception as e:
        print(f"[WARN] mmcv convert_sync_batchnorm not available/failed: {e}")

    # Fallback: manually replace SyncBN with standard BN.
    # SyncBatchNorm (2D) in img_backbone/img_neck/img_neck_3d
    # NaiveSyncBN1d (1D) in pts_voxel_encoder (HardVFE)
    def _replace(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(
                    child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                )
                if child.affine:
                    with torch.no_grad():
                        bn.weight.copy_(child.weight)
                        bn.bias.copy_(child.bias)
                bn.running_mean = child.running_mean
                bn.running_var = child.running_var
                bn.num_batches_tracked = child.num_batches_tracked
                setattr(module, name, bn)
            elif type(child).__name__ == "NaiveSyncBN1d":
                bn = nn.BatchNorm1d(
                    child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                )
                if child.affine:
                    with torch.no_grad():
                        bn.weight.copy_(child.weight)
                        bn.bias.copy_(child.bias)
                bn.running_mean = child.running_mean
                bn.running_var = child.running_var
                bn.num_batches_tracked = child.num_batches_tracked
                setattr(module, name, bn)
            else:
                _replace(child)

    _replace(model)
    return model


class LVFusionImg2D(nn.Module):
    """Stage-1: image 2D feature extraction."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.img_backbone = model.img_backbone
        self.img_neck = model.img_neck
        self.img_neck_fuse = getattr(model, "img_neck_fuse", None)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Accept [B, 3, H, W] or [B, 1, 3, H, W]
        if img.dim() == 5 and img.size(1) == 1:
            img = img.squeeze(1)
        if img.dim() != 4:
            raise RuntimeError(f"Expected img as 4D/5D tensor, got {img.shape}")

        feats = self.img_backbone(img)
        feats = self.img_neck(feats)

        feat0 = feats[0]
        if self.img_neck_fuse is None:
            return feat0

        feat = feat0
        for i in range(1, len(feats)):
            feat = torch.cat(
                [
                    feat,
                    resize(
                        feats[i],
                        size=feat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                dim=1,
            )
        feat = self.img_neck_fuse(feat)
        return feat


def _scatter_max_to_canvas(
    canvas: torch.Tensor, src: torch.Tensor, indices: torch.Tensor
) -> None:
    """Scatter src to canvas. Uses scatter_reduce(amax) if available, else scatter (last wins)."""
    index_exp = indices.unsqueeze(0).expand(src.shape[0], -1)
    if hasattr(canvas, "scatter_reduce_"):
        canvas.scatter_reduce_(1, index_exp, src, reduce="amax")
    else:
        # PyTorch < 1.12: scatter (last wins). Padding voxels at same coors overwrite; use unique coors for padding.
        canvas.scatter_(1, index_exp, src)


class PointPillarsScatterForONNX(nn.Module):
    """ONNX-exportable scatter. Uses scatter_max when available (PyTorch>=1.12).

    With scatter_max: padding voxels (zeros) do not overwrite real features.
    Fallback: direct scatter (padding at coors 0,0,0,0 may overwrite cell (0,0)).
    Supports batch_size=1 only for ONNX export.
    """

    def __init__(self, in_channels: int, ny: int, nx: int):
        super().__init__()
        self.in_channels = in_channels
        self.ny = ny
        self.nx = nx

    def forward(
        self, voxel_features: torch.Tensor, coors: torch.Tensor
    ) -> torch.Tensor:
        # voxel_features: [N, C], coors: [N, 4] (batch, z, y, x)
        indices = coors[:, 2] * self.nx + coors[:, 3]
        indices = indices.long().clamp(0, self.ny * self.nx - 1)

        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device,
        )
        src = voxel_features.t()  # [C, N]
        _scatter_max_to_canvas(canvas, src, indices)
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return canvas


class LVFusionPtsVFE2BEV(nn.Module):
    """Stage-2 with VFE: voxels -> pts_bev_feat (HardVFE + Scatter + backbone + neck).

    Input (fixed size, pad with zeros when voxel count < max_voxels):
      voxels: [max_voxels, max_points, C]  (C=4 for x,y,z,intensity)
      num_points: [max_voxels]  (0 for padding; internally clamped to avoid NaN)
      coors: [max_voxels, 4]  (batch, z, y, x; padding can use (0,0,0,0))
    """

    def __init__(self, model: nn.Module, max_voxels: int, ny: int, nx: int):
        super().__init__()
        self.pts_voxel_encoder = model.pts_voxel_encoder
        self.max_voxels = max_voxels
        self.ny = ny
        self.nx = nx
        in_channels = model.pts_middle_encoder.in_channels
        self.scatter = PointPillarsScatterForONNX(in_channels, ny, nx)
        self.pts_backbone = model.pts_backbone
        self.pts_neck = model.pts_neck
        self.pts_neck_fuse = getattr(model, "pts_neck_fuse", None)
        self._num_outs = getattr(model.pts_neck, "num_outs", 3)

    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
    ) -> torch.Tensor:
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        # Zero out padded voxels (num_points==0) before scatter. Padded coors (0,0,0,0)
        # would otherwise scatter to cell (0,0) and corrupt it. VFE output is typically
        # non-negative (ReLU), so scatter_max(real, 0) preserves real values.
        valid_mask = (num_points > 0).float().unsqueeze(1)
        voxel_features = voxel_features * valid_mask
        spatial_features = self.scatter(voxel_features, coors)
        x = self.pts_backbone(spatial_features)
        x = self.pts_neck(x)
        if self.pts_neck_fuse is None:
            return x[0]
        target_size = x[0].shape[-2:]
        fuse_feats = [x[0]]
        for i in range(1, self._num_outs):
            fuse_feats.append(
                resize(x[i], size=target_size, mode="bilinear", align_corners=False)
            )
        fused_feat = torch.cat(fuse_feats, dim=1)
        return self.pts_neck_fuse(fused_feat)


class LVFusionPtsBEV(nn.Module):
    """Stage-2: point cloud BEV feature extraction, starting from dense spatial_features.

    Input is expected to be the output of pts_middle_encoder (e.g., PointPillarsScatter):
      spatial_features: [B, 64, Y, X]  (for this repo configs: [B, 64, 128, 704])

    Uses static loop (num_outs=3) to avoid TracerWarning for ONNX export.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.pts_backbone = model.pts_backbone
        self.pts_neck = model.pts_neck
        self.pts_neck_fuse = getattr(model, "pts_neck_fuse", None)
        self._num_outs = getattr(
            model.pts_neck, "num_outs", 3
        )  # FPN num_outs, default 3

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        x = self.pts_backbone(spatial_features)  # tuple of multi-scale features
        x = self.pts_neck(x)  # list of features
        if self.pts_neck_fuse is None:
            return x[0]

        target_size = x[0].shape[-2:]
        fuse_feats = [x[0]]
        for i in range(1, self._num_outs):
            fuse_feats.append(
                resize(
                    x[i],
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fused_feat = torch.cat(fuse_feats, dim=1)
        return self.pts_neck_fuse(fused_feat)


class LVFusionFusionHead(nn.Module):
    """Stage-3: img_neck_3d + fusion_layer + CenterHead forward (raw outputs).

    Inputs:
      img_volume: [B, 64, X, Y, Z]  (produced by external backprojection)
      pts_bev_feat: [B, 256, Y/2, X/2]  (from stage-2)
    Outputs:
      Per-task raw tensors in fixed order:
        task{k}_{head} for head in [heatmap, reg, height, dim, rot, vel]
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.img_neck_3d = model.img_neck_3d
        self.fusion_layer = model.fusion_layer
        self.bbox_head = model.pts_bbox_head

        # Cache task count / head keys for deterministic outputs
        # SeparateHead keys are fixed by config: reg, height, dim, rot, vel, heatmap
        self._head_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]

    def forward(
        self, img_volume: torch.Tensor, pts_bev_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        img_feats = self.img_neck_3d(img_volume)
        if isinstance(img_feats, (list, tuple)):
            img_bev = img_feats[0]
        else:
            img_bev = img_feats

        # Align spatial sizes (always resize to avoid data-dependent Python branching in tracing)
        img_bev = resize(
            img_bev,
            size=pts_bev_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat([pts_bev_feat, img_bev], dim=1)
        fused = self.fusion_layer(fused)

        outs = self.bbox_head([fused])
        # In this repo, CenterHead.forward uses multi_apply on a list return,
        # which makes the return structure:
        #   outs: tuple(num_tasks) where each element is list(num_levels) of dict(head->tensor)
        if not isinstance(outs, tuple):
            raise RuntimeError(f"Unexpected bbox_head output type: {type(outs)}")

        flat: List[torch.Tensor] = []
        for task_id, task_levels in enumerate(outs):
            task_dict = task_levels[0]  # level-0 only
            for k in self._head_keys:
                if k not in task_dict:
                    raise RuntimeError(
                        f"Missing key '{k}' in task{task_id} outputs. Got keys: {list(task_dict.keys())}"
                    )
                flat.append(task_dict[k])
        return tuple(flat)


class LVFusionCamHead(nn.Module):
    """Stage-3 (cam-only): img_neck_3d + CenterHead forward (raw outputs).

    Input:
      img_volume: [B, 64, X, Y, Z]  (produced by external backprojection)
    Outputs:
      Per-task raw tensors in fixed order:
        task{k}_{head} for head in [heatmap, reg, height, dim, rot, vel]
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.img_neck_3d = model.img_neck_3d
        self.bbox_head = model.pts_bbox_head
        self._head_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]

    def forward(self, img_volume: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        img_feats = self.img_neck_3d(img_volume)
        if isinstance(img_feats, (list, tuple)):
            img_bev = img_feats[0]
        else:
            img_bev = img_feats

        outs = self.bbox_head([img_bev])
        if not isinstance(outs, tuple):
            raise RuntimeError(f"Unexpected bbox_head output type: {type(outs)}")

        flat: List[torch.Tensor] = []
        for task_id, task_levels in enumerate(outs):
            task_dict = task_levels[0]
            for k in self._head_keys:
                if k not in task_dict:
                    raise RuntimeError(
                        f"Missing key '{k}' in task{task_id} outputs. Got keys: {list(task_dict.keys())}"
                    )
                flat.append(task_dict[k])
        return tuple(flat)


def _infer_bev_shapes(cfg: Config) -> Dict[str, Tuple[int, ...]]:
    """Infer key tensor shapes from config for dummy inputs.

    - CAM-only 配置：可能没有 pts_middle_encoder/img_neck_3d，此时只需要返回 img 形状即可。
    - Fusion/LiDAR 配置：需要 pts_spatial_features 和 img_volume 的形状。
    """
    # Image input size
    input_h, input_w = cfg.data_config["input_size"]

    shapes: Dict[str, Tuple[int, ...]] = {
        "img": (1, 3, int(input_h), int(input_w)),
    }

    # Image n_voxels is stored as [ [X,Y,Z] ] in LVFusion configs
    if "img_neck_3d" in cfg.model and "n_voxels" in cfg.model:
        n_voxels = cfg.model["n_voxels"][0]
        vx, vy, vz = int(n_voxels[0]), int(n_voxels[1]), int(n_voxels[2])
        shapes["img_volume"] = (
            1,
            int(cfg.model["img_neck_3d"]["in_channels"]),
            vx,
            vy,
            vz,
        )

    # CAM-only config may not have pts_middle_encoder
    if "pts_middle_encoder" not in cfg.model:
        return shapes

    # PointPillarsScatter output_shape is [Y, X]
    out_shape = cfg.model["pts_middle_encoder"]["output_shape"]
    ny, nx = int(out_shape[0]), int(out_shape[1])
    shapes["pts_spatial_features"] = (
        1,
        int(cfg.model["pts_middle_encoder"]["in_channels"]),
        ny,
        nx,
    )

    return shapes


def _infer_voxel_shapes(cfg: Config, max_voxels: int) -> Dict[str, Tuple[int, ...]]:
    """Infer voxel input shapes for HardVFE export.

    HardVFE expects raw point features [M, N, 4] (x,y,z,intensity) from voxelization.
    """
    vl = cfg.model["pts_voxel_layer"]
    max_num_points = int(vl["max_num_points"])
    ve = cfg.model.get("pts_voxel_encoder", {})
    in_channels = int(ve.get("in_channels", 4))  # x, y, z, intensity
    return {
        "voxels": (max_voxels, max_num_points, in_channels),
        "num_points": (max_voxels,),
        "coors": (max_voxels, 4),
    }

def export_backprojection_params(
    cfg: Config,
    model_for_feat: nn.Module,
    out_dir: str,
    sample_idx: int = 0,
):
    """Export backprojection parameters using dataset sample (default first frame).

    This generates a portable parameter file for external backprojection:
      - gather_indices: indices into flattened 2D feature map (Hf*Wf)
      - scatter_indices: indices into flattened 3D volume (X*Y*Z)
      - shapes & meta

    Notes:
      - Uses cfg.data.test pipeline to obtain deterministic test-time aug extrinsic + origin.
      - Projection is derived from img_meta['lidar2img']['extrinsic'][0] (4x4 lidar2img).
        We scale the first two rows by 1/stride to map to feature coordinates.
    """
    os.makedirs(out_dir, exist_ok=True)

    dataset = build_dataset(cfg.data.test)
    if len(dataset) == 0:
        raise RuntimeError("cfg.data.test dataset is empty; cannot export backproj params.")
    if sample_idx >= len(dataset):
        sample_idx = 0

    sample = dataset[sample_idx]
    img = sample["img"].data  # [1, 3, H, W]
    img_meta = sample["img_metas"].data  # dict

    if not isinstance(img_meta, dict):
        raise RuntimeError(f"Unexpected img_metas type: {type(img_meta)}")

    # Compute feature size from stage1 torch module (ensures consistent stride)
    model_for_feat.eval()
    with torch.no_grad():
        feat_2d = model_for_feat(img.float())
    _, c, hf, wf = feat_2d.shape
    _, _, h, w = img.shape
    stride = float(w) / float(wf)

    # Get projection matrix (lidar2img) in pixel space, then scale to feature space.
    extr = np.array(img_meta["lidar2img"]["extrinsic"][0], dtype=np.float32)  # [4,4]
    proj = extr[:3, :].copy()  # [3,4]
    proj[0:2, :] /= stride

    origin = np.array(img_meta["lidar2img"]["origin"], dtype=np.float32).reshape(3)
    n_voxels = cfg.model["n_voxels"][0]
    voxel_size = cfg.model["voxel_size"][0]
    n_x, n_y, n_z = int(n_voxels[0]), int(n_voxels[1]), int(n_voxels[2])
    vx, vy, vz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])

    # Build points grid [3, X, Y, Z]
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(n_x),
            torch.arange(n_y),
            torch.arange(n_z),
            indexing="ij",
        )
    ).float()
    new_origin = torch.from_numpy(origin) - torch.tensor([n_x, n_y, n_z]).float() / 2.0 * torch.tensor([vx, vy, vz]).float()
    points = grid * torch.tensor([vx, vy, vz]).view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)  # [3,X,Y,Z]

    # Project all points to feature plane
    pts3 = points.view(3, -1)  # [3, N]
    pts4 = torch.cat([pts3, torch.ones(1, pts3.shape[1])], dim=0).numpy().astype(np.float32)  # [4,N]
    pts2d3 = proj @ pts4  # [3,N]
    x = np.rint(pts2d3[0] / pts2d3[2]).astype(np.int64)
    y = np.rint(pts2d3[1] / pts2d3[2]).astype(np.int64)
    z = pts2d3[2]
    valid = (x >= 0) & (y >= 0) & (x < wf) & (y < hf) & (z > 0)
    valid_idx = np.where(valid)[0].astype(np.int64)

    gather_indices = (y[valid_idx] * wf + x[valid_idx]).astype(np.int32)
    scatter_indices = valid_idx.astype(np.int32)  # matches points flatten order for [X,Y,Z]

    # ---- Save index format (valid, x, y, volume) like FastBEV export_2d_to_3d ----
    index_dir = os.path.join(out_dir, "index")
    os.makedirs(index_dir, exist_ok=True)
    n_total = x.shape[0]
    valid_full = np.zeros(n_total, dtype=np.bool_)
    valid_full[valid_idx] = True
    x_full = np.zeros(n_total, dtype=np.int32)
    y_full = np.zeros(n_total, dtype=np.int32)
    x_full[valid_idx] = x[valid_idx]
    y_full[valid_idx] = y[valid_idx]
    np.save(os.path.join(index_dir, "valid.npy"), valid_full)
    np.save(os.path.join(index_dir, "x.npy"), x_full)
    np.save(os.path.join(index_dir, "y.npy"), y_full)
    np.save(os.path.join(index_dir, "volume.npy"), np.array([n_x, n_y, n_z], dtype=np.int32))
    # Sample features for reference (from stage1 output)
    feat_sample = feat_2d.cpu().numpy().astype(np.float32)
    np.save(os.path.join(index_dir, "features.npy"), feat_sample)
    print(f"[Backproj] Saved index (valid, x, y, volume, features): {index_dir}/")

    # ---- Save numpy params ----
    params_path = os.path.join(out_dir, f"backprojection_params_sample{sample_idx}.npz")
    np.savez(
        params_path,
        gather_indices=gather_indices,
        scatter_indices=scatter_indices,
        feat_hw=np.array([hf, wf], dtype=np.int32),
        img_hw=np.array([h, w], dtype=np.int32),
        volume_xyz=np.array([n_x, n_y, n_z], dtype=np.int32),
        channels=np.array([c], dtype=np.int32),
        stride=np.array([stride], dtype=np.float32),
        projection=proj.astype(np.float32),
        origin=origin.astype(np.float32),
        sample_idx=np.array([sample_idx], dtype=np.int32),
    )
    print(f"[Backproj] Saved params: {params_path}")

    # ---- Additionally save C++-friendly binary index files (.bin) ----
    # 1) featurePointLength.bin: int32 N_valid
    n_valid = np.array([gather_indices.shape[0]], dtype=np.int32)
    length_path = os.path.join(out_dir, "featurePointLength.bin")
    n_valid.tofile(length_path)

    # 2) gather_indices.bin: int32 linear indices into flattened 2D feature map (Hf*Wf)
    gather_bin_path = os.path.join(out_dir, "gather_indices.bin")
    gather_indices.astype(np.int32).tofile(gather_bin_path)

    # 3) scatter_indices.bin: int32 linear indices into flattened 3D volume (X*Y*Z)
    scatter_bin_path = os.path.join(out_dir, "scatter_indices.bin")
    scatter_indices.astype(np.int32).tofile(scatter_bin_path)

    print(
        f"[Backproj] Saved binary indices: "
        f"{os.path.basename(length_path)}, "
        f"{os.path.basename(gather_bin_path)}, "
        f"{os.path.basename(scatter_bin_path)}"
    )


def export_stage1(model: nn.Module, cfg: Config, out_path: str, opset: int):
    wrapper = LVFusionImg2D(model).eval()
    shapes = _infer_bev_shapes(cfg)
    dummy = torch.randn(*shapes["img"], device=next(wrapper.parameters()).device)

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["img"],
        output_names=["img_feat_2d"],
        dynamic_axes={
            "img": {0: "batch", 2: "height", 3: "width"},
            "img_feat_2d": {0: "batch", 2: "feat_h", 3: "feat_w"},
        },
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    _check_onnx(out_path)


def export_stage2(model: nn.Module, cfg: Config, out_path: str, opset: int):
    wrapper = LVFusionPtsBEV(model).eval()
    shapes = _infer_bev_shapes(cfg)
    dummy = torch.randn(
        *shapes["pts_spatial_features"], device=next(wrapper.parameters()).device
    )

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["pts_spatial_features"],
        output_names=["pts_bev_feat"],
        dynamic_axes={
            "pts_spatial_features": {0: "batch"},
            "pts_bev_feat": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    _check_onnx(out_path)


def export_stage2_with_vfe(
    model: nn.Module, cfg: Config, out_path: str, opset: int, max_voxels: int
):
    """Export Stage 2 including HardVFE + PointPillarsScatter (fixed voxel count)."""
    shapes = _infer_bev_shapes(cfg)
    out_shape = cfg.model["pts_middle_encoder"]["output_shape"]
    ny, nx = int(out_shape[0]), int(out_shape[1])

    wrapper = LVFusionPtsVFE2BEV(model, max_voxels, ny, nx).eval()
    voxel_shapes = _infer_voxel_shapes(cfg, max_voxels)
    dev = next(wrapper.parameters()).device

    # Dummy: padding uses num_points=0 (HardVFE now handles this via clamp in mmdet3d)
    dummy_voxels = torch.zeros(*voxel_shapes["voxels"], device=dev)
    dummy_num_points = torch.zeros(
        voxel_shapes["num_points"][0], dtype=torch.int32, device=dev
    )
    dummy_coors = torch.zeros(*voxel_shapes["coors"], dtype=torch.int32, device=dev)
    # Valid voxels for tracing
    dummy_voxels[0, :5, :] = torch.randn(5, voxel_shapes["voxels"][2], device=dev) * 0.1
    dummy_num_points[0] = 5
    dummy_coors[0] = torch.tensor([0, 0, 10, 20], dtype=torch.int32, device=dev)

    torch.onnx.export(
        wrapper,
        (dummy_voxels, dummy_num_points, dummy_coors),
        out_path,
        input_names=["voxels", "num_points", "coors"],
        output_names=["pts_bev_feat"],
        dynamic_axes={"pts_bev_feat": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    _check_onnx(out_path)


def export_stage3(model: nn.Module, cfg: Config, out_path: str, opset: int):
    wrapper = LVFusionFusionHead(model).eval()
    shapes = _infer_bev_shapes(cfg)
    dev = next(wrapper.parameters()).device
    dummy_img_vol = torch.randn(*shapes["img_volume"], device=dev)

    # pts_bev_feat channel/out spatial: assume stage-2 output is 256 channels at (ny/2, nx/2)
    out_size_factor = int(cfg.model["pts_bbox_head"]["bbox_coder"]["out_size_factor"])
    ny = shapes["pts_spatial_features"][2] // out_size_factor
    nx = shapes["pts_spatial_features"][3] // out_size_factor
    dummy_pts_bev = torch.randn(1, 256, ny, nx, device=dev)

    # Build output names (task0_heatmap, task0_reg, ...)
    output_names: List[str] = []
    head_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    num_tasks = len(cfg.model["pts_bbox_head"]["tasks"])
    for t in range(num_tasks):
        for k in head_keys:
            output_names.append(f"task{t}_{k}")

    torch.onnx.export(
        wrapper,
        (dummy_img_vol, dummy_pts_bev),
        out_path,
        input_names=["img_volume", "pts_bev_feat"],
        output_names=output_names,
        dynamic_axes={
            "img_volume": {0: "batch"},
            "pts_bev_feat": {0: "batch"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    _check_onnx(out_path)


def export_stage3_cam(model: nn.Module, cfg: Config, out_path: str, opset: int):
    wrapper = LVFusionCamHead(model).eval()
    shapes = _infer_bev_shapes(cfg)
    if "img_volume" not in shapes:
        raise RuntimeError("CAM stage3 export requires img_volume shape. Check cfg.model.img_neck_3d and cfg.model.n_voxels.")
    dev = next(wrapper.parameters()).device
    dummy_img_vol = torch.randn(*shapes["img_volume"], device=dev)

    output_names: List[str] = []
    head_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    num_tasks = len(cfg.model["pts_bbox_head"]["tasks"])
    for t in range(num_tasks):
        for k in head_keys:
            output_names.append(f"task{t}_{k}")

    torch.onnx.export(
        wrapper,
        dummy_img_vol,
        out_path,
        input_names=["img_volume"],
        output_names=output_names,
        dynamic_axes={
            "img_volume": {0: "batch"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    _check_onnx(out_path)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    else:
        warnings.warn(
            "No --checkpoint provided. Exporting with randomly initialized weights."
        )
        if hasattr(model, "init_weights"):
            model.init_weights()

    # handle fp16 config (keep fp32 for export)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        if hasattr(model, "fp16_enabled"):
            model.fp16_enabled = False

    model = _convert_syncbn(model)
    model.eval()

    device = torch.device(args.device)
    model = model.to(device).float()

    stages = set(args.stages)

    # Export backprojection params first (needs dataset + stage1 torch module for feat size)
    # 仅在模型存在图像分支时才导出（LiDAR-only 配置没有 img_backbone）
    if args.export_backproj_params and getattr(model, "img_backbone", None) is not None:
        try:
            stage1_torch = LVFusionImg2D(model).to("cpu").float().eval()
            export_backprojection_params(
                cfg=cfg,
                model_for_feat=stage1_torch,
                out_dir=args.out_dir,
                sample_idx=args.sample_idx,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to export backprojection params: {e}") from e
    elif args.export_backproj_params:
        print("[Backproj] Skipped: model has no image backbone (LiDAR-only config).")

    if 1 in stages:
        out_path = os.path.join(args.out_dir, "export_2d_model.onnx")
        print(f"[Stage1] Exporting to {out_path}")
        cfg_stage1 = cfg
        if args.dummy_img_size is not None:
            h, w = int(args.dummy_img_size[0]), int(args.dummy_img_size[1])
            # Patch cfg input_size only for dummy shape generation
            cfg_stage1 = cfg.copy()
            cfg_stage1.data_config = dict(cfg.data_config)
            cfg_stage1.data_config["input_size"] = (h, w)
        export_stage1(model, cfg_stage1, out_path, args.opset_version)
        if args.simplify:
            simp = _maybe_simplify(out_path)
            if simp != out_path:
                print(f"[Stage1] Simplified: {simp}")

    if 2 in stages:
        # 分开VFE: export_lidar_model.onnx (pts_spatial_features -> pts_bev_feat)
        out_path_separate = os.path.join(args.out_dir, "export_lidar_model.onnx")
        print(f"[Stage2-分开VFE] Exporting to {out_path_separate}")
        try:
            export_stage2(model, cfg, out_path_separate, args.opset_version)
        except Exception as e:
            print(f"[Stage2] Export failed: {e}")
            print("[Stage2] Tip: If you hit TracerWarning (dynamic control flow), try: --stages 1 3")
            raise
        if args.simplify:
            simp = _maybe_simplify(out_path_separate)
            if simp != out_path_separate:
                print(f"[Stage2] Simplified: {simp}")

        # 合并VFE: export_lidar_vfe_model.onnx (voxels, num_points, coors -> pts_bev_feat)
        if args.include_vfe:
            max_voxels = args.max_voxels
            if max_voxels is None:
                mv = cfg.model["pts_voxel_layer"].get("max_voxels", 20000)
                max_voxels = mv[1] if isinstance(mv, (list, tuple)) else int(mv)
            out_path_merged = os.path.join(args.out_dir, "export_lidar_vfe_model.onnx")
            print(f"[Stage2-合并VFE] Exporting to {out_path_merged} (max_voxels={max_voxels})")
            export_stage2_with_vfe(
                model, cfg, out_path_merged, args.opset_version, max_voxels
            )
            # Save voxel params for preprocessing (external voxelization)
            voxel_shapes = _infer_voxel_shapes(cfg, max_voxels)
            vl = cfg.model["pts_voxel_layer"]
            params_path = os.path.join(args.out_dir, "voxel_params.npz")
            np.savez(
                params_path,
                max_voxels=np.array([max_voxels], dtype=np.int32),
                max_num_points=np.array([vl["max_num_points"]], dtype=np.int32),
                voxel_shape=np.array(voxel_shapes["voxels"], dtype=np.int32),
                point_cloud_range=np.array(vl["point_cloud_range"], dtype=np.float32),
                voxel_size=np.array(vl["voxel_size"], dtype=np.float32),
            )
            print(f"[Stage2-合并VFE] Saved voxel params: {params_path}")
            if args.simplify:
                simp = _maybe_simplify(out_path_merged)
                if simp != out_path_merged:
                    print(f"[Stage2-合并VFE] Simplified: {simp}")

    if 3 in stages:
        if getattr(model, "with_pts_backbone", False):
            out_path = os.path.join(args.out_dir, "export_3d_model.onnx")
            print(f"[Stage3] Exporting to {out_path}")
            export_stage3(model, cfg, out_path, args.opset_version)
        else:
            out_path = os.path.join(args.out_dir, "export_cam_3d_model.onnx")
            print(f"[Stage3-CAM] Exporting to {out_path}")
            export_stage3_cam(model, cfg, out_path, args.opset_version)
        if args.simplify:
            simp = _maybe_simplify(out_path)
            if simp != out_path:
                print(f"[Stage3] Simplified: {simp}")

    print("[DONE] LVFusion ONNX export completed.")


if __name__ == "__main__":
    main()

