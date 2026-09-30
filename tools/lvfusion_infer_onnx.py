#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNXRuntime inference for LVFusion (3-stage) + automatic comparison with host outputs.

Notes:
  - backprojection is external and uses the exported params npz.
  - voxelize/VFE/scatter are kept in PyTorch for now (same as host script), so we only
    validate the exported ONNX stages numerically.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset


def parse_args():
    p = argparse.ArgumentParser(description="LVFusion ONNX inference + compare")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="for voxelize when separate mode")
    p.add_argument("--onnx-dir", type=str, required=True, help="dir containing 3 onnx files")
    p.add_argument("--backproj-params", type=str, required=True)
    p.add_argument("--ref-npz", type=str, required=True, help="host outputs npz")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--providers", type=str, nargs="+", default=None, help="onnxruntime providers override")
    p.add_argument(
        "--merged-vfe",
        action="store_true",
        help="stage2 ONNX has HardVFE+Scatter inside (input: voxels, num_points, coors)",
    )
    return p.parse_args()


def _load_session(path: str, providers=None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers is None:
        providers = ["CPUExecutionProvider"]
    # filter providers by availability to avoid warnings
    avail = set(ort.get_available_providers())
    providers = [p for p in providers if p in avail]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def external_backproject_volume_numpy(feat_2d: np.ndarray, params: Dict) -> np.ndarray:
    """feat_2d: [B,C,Hf,Wf] -> [B,C,X,Y,Z]

    feat_2d Hf,Wf MUST match params['feat_hw'] (from backprojection_params). The params
    are tied to the image size used during export (e.g. 1080x1920 -> 270x480).
    If you use different input_size, re-export backproj params or use --dummy-img-size.
    """
    gather = params["gather_indices"].astype(np.int64)
    scatter = params["scatter_indices"].astype(np.int64)
    x, y, z = [int(v) for v in params["volume_xyz"].tolist()]
    b, c, hf, wf = feat_2d.shape
    feat_hw = params.get("feat_hw")
    if feat_hw is not None:
        exp_h, exp_w = int(feat_hw[0]), int(feat_hw[1])
        if (hf, wf) != (exp_h, exp_w):
            raise ValueError(
                f"Stage1 feature map shape (Hf={hf}, Wf={wf}) does not match backprojection params "
                f"feat_hw=(Hf={exp_h}, Wf={exp_w}). Backproj params were exported for a specific "
                f"image size (e.g. 1080x1920 -> 270x480). Use the same input_size, or re-export "
                f"with --export-backproj-params and your target image size."
            )
    feat_flat = feat_2d.reshape(b, c, hf * wf)
    picked = feat_flat[:, :, gather]  # [B,C,Nv]
    vol_flat = np.zeros((b, c, x * y * z), dtype=feat_2d.dtype)
    vol_flat[:, :, scatter] = picked
    return vol_flat.reshape(b, c, x, y, z)


def compute_pts_spatial_features(model, points_tensor: torch.Tensor) -> torch.Tensor:
    voxels, num_points, coors = model.voxelize([points_tensor])
    voxel_features = model.pts_voxel_encoder(voxels, num_points, coors)
    batch_size = int(coors[-1, 0].item() + 1)
    spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)
    return spatial_features


def pad_voxels_for_onnx(
    voxels: torch.Tensor,
    num_points: torch.Tensor,
    coors: torch.Tensor,
    max_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad voxels/num_points/coors to fixed max_voxels for merged ONNX stage2."""
    n = voxels.shape[0]
    if n >= max_voxels:
        voxels = voxels[:max_voxels]
        num_points = num_points[:max_voxels]
        coors = coors[:max_voxels]
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


def _shapes_compatible(actual: tuple, expected: list) -> bool:
    """Check if actual shape matches expected (None in expected = any value)."""
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if e is not None and a != e:
            return False
    return True


def compare_arrays(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = np.abs(a - b)
    return float(diff.max()), float(diff.mean())


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # Build PyTorch model ONLY for voxelize/VFE/scatter parts.
    cfg.model.train_cfg = None
    torch_model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(torch_model, args.checkpoint, map_location="cpu")
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(torch_model)
        if hasattr(torch_model, "fp16_enabled"):
            torch_model.fp16_enabled = False
    device = torch.device(args.device)
    torch_model = torch_model.to(device).float().eval()

    # Load dataset
    dataset = build_dataset(cfg.data.test)
    n = min(args.num_samples, len(dataset))

    params = dict(np.load(args.backproj_params, allow_pickle=True))

    # Load host reference outputs
    ref = dict(np.load(args.ref_npz, allow_pickle=True))
    ref_indices = ref["sample_indices"].astype(np.int64).tolist()

    # Merged mode: need voxel_params for max_voxels
    max_voxels = None
    if args.merged_vfe:
        voxel_params_path = os.path.join(args.onnx_dir, "voxel_params.npz")
        if not os.path.isfile(voxel_params_path):
            raise FileNotFoundError(
                f"Merged mode requires {voxel_params_path}. Export with --include-vfe."
            )
        voxel_params = dict(np.load(voxel_params_path, allow_pickle=True))
        max_voxels = int(voxel_params["max_voxels"].item())

    # ONNX sessions (naming matches FastBEV: export_2d_model, export_lidar*, export_3d_model)
    def _resolve_onnx(name: str) -> str:
        p = os.path.join(args.onnx_dir, name)
        if os.path.isfile(p):
            return p
        simp = p.replace(".onnx", "_simplified.onnx")
        if os.path.isfile(simp):
            return simp
        raise FileNotFoundError(f"ONNX not found: {p} or {simp}")

    stage1_path = _resolve_onnx("export_2d_model.onnx")
    stage2_path = _resolve_onnx(
        "export_lidar_vfe_model.onnx" if args.merged_vfe else "export_lidar_model.onnx"
    )
    stage3_path = _resolve_onnx("export_3d_model.onnx")
    providers = args.providers
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]

    sess1 = _load_session(stage1_path, providers=providers)
    sess2 = _load_session(stage2_path, providers=providers)
    sess3 = _load_session(stage3_path, providers=providers)

    head_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    num_tasks = len(cfg.model["pts_bbox_head"]["tasks"])
    output_names = [f"task{t}_{k}" for t in range(num_tasks) for k in head_keys]

    # Compare
    summary: List[str] = []
    for i in range(n):
        if i not in ref_indices:
            raise RuntimeError(f"Sample {i} not found in ref npz indices: {ref_indices[:10]}...")

        data = dataset[i]
        img = data["img"].data.cpu().numpy().astype(np.float32)  # [1,3,H,W]
        pts = data["points"].data.to(device).float()

        # stage1 onnx
        img_feat_2d = sess1.run(None, {"img": img})[0]  # [1,64,Hf,Wf]
        img_volume = external_backproject_volume_numpy(img_feat_2d, params)  # [1,64,X,Y,Z]

        # stage2 onnx
        if args.merged_vfe:
            with torch.no_grad():
                voxels, num_points, coors = torch_model.voxelize([pts])
                v, np_pts, c = pad_voxels_for_onnx(voxels, num_points, coors, max_voxels)
            pts_bev = sess2.run(None, {"voxels": v, "num_points": np_pts, "coors": c})[0]
        else:
            with torch.no_grad():
                pts_spatial = compute_pts_spatial_features(torch_model, pts).detach().cpu().numpy().astype(np.float32)
            pts_bev = sess2.run(None, {"pts_spatial_features": pts_spatial})[0]  # [1,256,*,*]

        # stage3 onnx: validate input shapes to avoid cryptic ONNX Runtime errors
        try:
            def _onnx_shape_to_tuple(shape_obj) -> tuple:
                out = []
                for d in shape_obj:
                    v = getattr(d, "dim_value", None)
                    out.append(v if (v is not None and v > 0) else None)
                return tuple(out)
            exp_img = _onnx_shape_to_tuple(sess3.get_inputs()[0].shape)
            exp_pts = _onnx_shape_to_tuple(sess3.get_inputs()[1].shape)
            if exp_img and exp_pts and (
                not _shapes_compatible(img_volume.shape, list(exp_img))
                or not _shapes_compatible(pts_bev.shape, list(exp_pts))
            ):
                raise ValueError(
                    f"Stage3 input shape mismatch. "
                    f"img_volume: got {tuple(img_volume.shape)}, expected (batch,64,X,Y,Z). "
                    f"pts_bev_feat: got {tuple(pts_bev.shape)}, expected (batch,256,ny,nx). "
                    "Ensure config input_size and pts_middle_encoder.output_shape match export."
                )
        except (AttributeError, TypeError, IndexError):
            pass  # skip if shape parsing fails

        # stage3 onnx
        outs_onnx = sess3.run(None, {"img_volume": img_volume.astype(np.float32), "pts_bev_feat": pts_bev.astype(np.float32)})
        onnx_dict = {name: arr for name, arr in zip(output_names, outs_onnx)}

        # compare with ref
        for name in output_names:
            a = onnx_dict[name]
            b = ref[name][ref_indices.index(i)]  # ref saved as [N,B?,...]; we stacked as [N,1,C,H,W]
            maxe, meane = compare_arrays(a, b)
            summary.append(f"sample{i} {name}: max={maxe:.6g} mean={meane:.6g}")

    # Print concise summary (top-k worst by max error)
    worst = []
    for line in summary:
        parts = line.split("max=")[-1]
        maxe = float(parts.split()[0])
        worst.append((maxe, line))
    worst.sort(key=lambda x: x[0], reverse=True)

    print("[ONNX] Comparison completed. Worst 20 by max error:")
    for _, line in worst[:20]:
        print("  " + line)


if __name__ == "__main__":
    main()

