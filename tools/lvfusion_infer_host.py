#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Host (PyTorch) inference for LVFusion using external backprojection and exporting raw head outputs.

This script DOES NOT do dataset-level evaluation; it saves raw outputs for later comparison.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset

from tools.export_lvfusion_onnx import (
    LVFusionImg2D,
    LVFusionPtsBEV,
    LVFusionFusionHead,
    _convert_syncbn,
)


def parse_args():
    p = argparse.ArgumentParser(description="LVFusion host inference (PyTorch)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--backproj-params", type=str, required=True, help="npz params from export script")
    p.add_argument("--out", type=str, required=True, help="output npz file path")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="device for voxelize/VFE/scatter")
    p.add_argument(
        "--export-device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="device for stage1/2/3 reference compute (recommend cpu to match onnxruntime cpu)",
    )
    return p.parse_args()


def external_backproject_volume(feat_2d: torch.Tensor, params: Dict) -> torch.Tensor:
    """feat_2d: [B,C,Hf,Wf] -> volume: [B,C,X,Y,Z] using gather/scatter indices."""
    gather = torch.from_numpy(params["gather_indices"].astype(np.int64)).to(feat_2d.device)
    scatter = torch.from_numpy(params["scatter_indices"].astype(np.int64)).to(feat_2d.device)
    x, y, z = [int(v) for v in params["volume_xyz"].tolist()]
    b, c, hf, wf = feat_2d.shape
    feat_flat = feat_2d.view(b, c, hf * wf)
    picked = feat_flat.index_select(2, gather)  # [B,C,Nv]
    vol_flat = torch.zeros((b, c, x * y * z), device=feat_2d.device, dtype=feat_2d.dtype)
    vol_flat.index_copy_(2, scatter, picked)
    return vol_flat.view(b, c, x, y, z)


def compute_pts_spatial_features(model, points_tensor: torch.Tensor) -> torch.Tensor:
    """Compute dense spatial_features [B,64,Y,X] using voxelize/vfe/scatter."""
    # model.voxelize expects list of tensors
    voxels, num_points, coors = model.voxelize([points_tensor])
    voxel_features = model.pts_voxel_encoder(voxels, num_points, coors)
    batch_size = int(coors[-1, 0].item() + 1)
    spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)
    return spatial_features


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        if hasattr(model, "fp16_enabled"):
            model.fp16_enabled = False

    # If reference export-device is CPU, convert SyncBN to BN to avoid runtime error.
    if args.export_device == "cpu":
        model = _convert_syncbn(model)

    device = torch.device(args.device)
    model = model.to(device).float().eval()

    # wrappers
    export_device = torch.device(args.export_device)
    stage1 = LVFusionImg2D(model).to(export_device).float().eval()
    stage2 = LVFusionPtsBEV(model).to(export_device).float().eval()
    stage3 = LVFusionFusionHead(model).to(export_device).float().eval()

    params = dict(np.load(args.backproj_params, allow_pickle=True))

    dataset = build_dataset(cfg.data.test)
    n = min(args.num_samples, len(dataset))

    all_out: Dict[str, List[np.ndarray]] = {}
    sample_indices: List[int] = []

    with torch.no_grad():
        for i in range(n):
            data = dataset[i]
            img = data["img"].data.to(export_device).float()  # [1,3,H,W]
            pts = data["points"].data.to(device).float()  # [N,4] keep on voxelize device

            img_feat_2d = stage1(img)
            img_volume = external_backproject_volume(img_feat_2d, params)

            pts_spatial = compute_pts_spatial_features(model, pts).to(export_device).float()
            pts_bev = stage2(pts_spatial)

            outs = stage3(img_volume, pts_bev)  # tuple of tensors
            head_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
            num_tasks = len(cfg.model["pts_bbox_head"]["tasks"])
            output_names = [f"task{t}_{k}" for t in range(num_tasks) for k in head_keys]
            if len(output_names) != len(outs):
                raise RuntimeError(
                    f"Unexpected outputs length: got {len(outs)}, expect {len(output_names)}"
                )

            for name, t in zip(output_names, outs):
                key = name
                all_out.setdefault(key, []).append(t.detach().cpu().numpy())

            sample_indices.append(i)

    np.savez(
        args.out,
        sample_indices=np.array(sample_indices, dtype=np.int32),
        **{k: np.stack(v, axis=0) for k, v in all_out.items()},
    )
    print(f"[HOST] Saved raw outputs: {args.out}")


if __name__ == "__main__":
    main()

