# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize
from mmcv.parallel import DataContainer
from mmcv.runner import force_fp32

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.core.points import BasePoints
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from .. import builder
from .base import Base3DDetector


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    """Generate 3D points grid for backprojection.
    
    Args:
        n_voxels: [X, Y, Z] number of voxels
        voxel_size: [vx, vy, vz] voxel size
        origin: [ox, oy, oz] origin point
        
    Returns:
        points: [3, X, Y, Z] 3D points grid
    """
    grid = torch.stack(torch.meshgrid(
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ))
    new_origin = origin - n_voxels / 2 * voxel_size
    return grid * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)


def backproject_inplace(features, points, projection):
    """Backproject 2D features to 3D volume (inplace version).
    
    Args:
        features: [N, C, H, W] 2D features
        points: [3, X, Y, Z] 3D points grid
        projection: [N, 3, 4] projection matrix
        
    Returns:
        volume: [C, X, Y, Z] 3D volume
    """
    n, c, h, w = features.shape
    pts = points.view(1, 3, -1).expand(n, 3, -1)
    pts = torch.cat([pts, torch.ones_like(pts[:, :1])], 1)
    pts_2d = torch.bmm(projection, pts)
    x = (pts_2d[:, 0] / pts_2d[:, 2]).round().long()
    y = (pts_2d[:, 1] / pts_2d[:, 2]).round().long()
    z = pts_2d[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < w) & (y < h) & (z > 0)

    vol = torch.zeros(c, pts.shape[-1], device=features.device, dtype=features.dtype)
    for i in range(n):
        vol[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    return vol.view(c, *points.shape[-3:])


@DETECTORS.register_module()
class LVFusion(Base3DDetector):
    """LVFusion: LiDAR-Vision Fusion 3D Detector.
    
    This detector supports:
    - Image-only detection (if only img_backbone/img_neck configured)
    - Point cloud-only detection (if only pts_backbone/pts_neck configured)
    - Multi-modal fusion (if both img and pts branches configured)
    
    Args:
        pts_voxel_layer: Voxelization layer config for point cloud
        pts_voxel_encoder: Voxel encoder config for point cloud
        pts_middle_encoder: Middle encoder config for point cloud
        pts_backbone: Backbone config for point cloud
        pts_neck: Neck config for point cloud
        pts_neck_fuse: Neck fusion config for point cloud (to fuse multi-scale features)
        img_backbone: Backbone config for image
        img_neck: Neck config for image
        img_neck_fuse: Neck fusion config for image (to fuse multi-scale features)
        img_neck_3d: 3D neck config for image (after backprojection)
        fusion_layer: Fusion layer config (only used when both img and pts are present)
        pts_bbox_head: Bbox head config
        n_voxels: List of [X, Y, Z] voxel numbers for image backprojection
        voxel_size: List of [vx, vy, vz] voxel sizes for image backprojection
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        train_cfg: Training config
        test_cfg: Testing config
        backproject: 'inplace' or 'vanilla' for backprojection method
        init_cfg: Initialization config
    """
    
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_neck_fuse=None,
                 img_backbone=None,
                 img_neck=None,
                 img_neck_fuse=None,
                 img_neck_3d=None,
                 fusion_layer=None,
                 pts_bbox_head=None,
                 occ_head=None,
                 n_voxels=None,
                 voxel_size=None,
                 point_cloud_range=None,
                 train_cfg=None,
                 test_cfg=None,
                 backproject='inplace',
                 init_cfg=None,
                 pretrained=None):
        super(LVFusion, self).__init__(init_cfg=init_cfg)
        
        # Point cloud branch
        if pts_voxel_layer is not None:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        else:
            self.pts_voxel_layer = None
            
        if pts_voxel_encoder is not None:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        else:
            self.pts_voxel_encoder = None
            
        if pts_middle_encoder is not None:
            self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        else:
            self.pts_middle_encoder = None
            
        if pts_backbone is not None:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        else:
            self.pts_backbone = None
            
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        else:
            self.pts_neck = None
            
        # Point cloud neck fusion (for multi-scale features)
        if pts_neck_fuse is not None:
            if isinstance(pts_neck_fuse.get('in_channels'), list):
                in_channels = sum(pts_neck_fuse['in_channels'])
            else:
                in_channels = pts_neck_fuse['in_channels']
            out_channels = pts_neck_fuse['out_channels']
            kernel_size = pts_neck_fuse.get('kernel_size', 3)
            padding = pts_neck_fuse.get('padding', 1)
            self.pts_neck_fuse = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding)
            self.pts_multi_scale_id = pts_neck_fuse.get('multi_scale_id', [0])
            if isinstance(self.pts_multi_scale_id, int):
                self.pts_multi_scale_id = [self.pts_multi_scale_id]
        else:
            self.pts_neck_fuse = None
            self.pts_multi_scale_id = None
        
        # Image branch
        if img_backbone is not None:
            self.img_backbone = build_backbone(img_backbone)
        else:
            self.img_backbone = None
            
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        else:
            self.img_neck = None
            
        # Image neck fusion (for multi-scale features)
        if img_neck_fuse is not None:
            if isinstance(img_neck_fuse.get('in_channels'), list):
                in_channels = img_neck_fuse['in_channels'][0]  # Use first level
            else:
                in_channels = img_neck_fuse['in_channels']
            out_channels = img_neck_fuse['out_channels'][0] if isinstance(img_neck_fuse.get('out_channels'), list) else img_neck_fuse['out_channels']
            self.img_neck_fuse = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.img_neck_fuse = None
            
        if img_neck_3d is not None:
            self.img_neck_3d = build_neck(img_neck_3d)
        else:
            self.img_neck_3d = None
        
        # Fusion layer (only when both img and pts are present)
        if fusion_layer is not None and self.with_pts_backbone and self.with_img_backbone:
            self.fusion_layer = nn.Conv2d(
                fusion_layer['in_channels'],
                fusion_layer['out_channels'],
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.fusion_layer = None
        
        # Bbox head
        if pts_bbox_head is not None:
            # Extract pts config from train_cfg and test_cfg
            pts_train_cfg = train_cfg.pts if train_cfg and hasattr(train_cfg, 'pts') else train_cfg
            pts_test_cfg = test_cfg.pts if test_cfg and hasattr(test_cfg, 'pts') else test_cfg
            
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        else:
            self.pts_bbox_head = None

        # Optional semantic occupancy head. It consumes the same fused 2D BEV
        # feature used by CenterHead and keeps height in the output channels.
        self.occ_head = builder.build_head(occ_head) if occ_head is not None else None
        
        # Image backprojection parameters
        self.n_voxels = n_voxels[0] if n_voxels is not None else None
        self.voxel_size = voxel_size[0] if voxel_size is not None else None
        self.point_cloud_range = point_cloud_range
        self.backproject = backproject
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a point cloud backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None
    
    @property
    def with_img_backbone(self):
        """bool: Whether the detector has an image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None
    
    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a point cloud neck."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None
    
    @property
    def with_img_neck(self):
        """bool: Whether the detector has an image neck."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
    
    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None
    
    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self, 'fusion_layer') and self.fusion_layer is not None

    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None
    
    @staticmethod
    def _compute_projection(img_meta, stride):
        """Compute projection matrix from lidar to image.
        
        Args:
            img_meta: Image metadata dict
            stride: Feature stride
            
        Returns:
            projection: [3, 4] projection matrix
        """
        intrinsic = torch.tensor(
            img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsic = torch.tensor(
            img_meta["lidar2img"]["extrinsic"][0])  # CAM_FRONT
        return intrinsic @ extrinsic[:3]
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def extract_pts_feat(self, points, img_metas=None):
        """Extract features from point cloud.
        
        Args:
            points: List of point cloud tensors
            img_metas: List of image metadata dicts
            
        Returns:
            pts_feats: List of point cloud features (or None if not configured)
        """
        if not self.with_pts_backbone:
            return None
        
        # Handle different input formats
        if isinstance(points, DataContainer):
            points = points.data
        
        if not isinstance(points, list):
            points = [points]
        
        # Process points: unpack nested lists and DataContainers
        processed_points = []
        for p in points:
            while isinstance(p, list) and len(p) == 1 and isinstance(p[0], torch.Tensor):
                p = p[0]
            if isinstance(p, DataContainer):
                p = p.data
            if isinstance(p, BasePoints):
                p = p.tensor
            if not isinstance(p, torch.Tensor):
                raise TypeError(f'points element must be torch.Tensor, but got {type(p)}')
            if p.dim() != 2:
                raise ValueError(f'points element must be 2D tensor [N, C], but got shape {p.shape}')
            processed_points.append(p)
        
        voxels, num_points, coors = self.voxelize(processed_points)
        
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        
        if self.pts_neck is not None:
            x = self.pts_neck(x)  # x is a list of multi-scale features
            
            # Fuse multi-scale features if needed
            if self.pts_neck_fuse is not None and isinstance(x, (list, tuple)):
                if self.pts_multi_scale_id is not None and len(self.pts_multi_scale_id) > 0:
                    target_level = self.pts_multi_scale_id[0]
                    target_size = x[target_level].shape[2:]
                else:
                    target_size = x[0].shape[2:]
                
                fuse_feats = [x[target_level]]
                for i in range(target_level + 1, len(x)):
                    resized_feat = resize(
                        x[i],
                        size=target_size,
                        mode='bilinear',
                        align_corners=False)
                    fuse_feats.append(resized_feat)
                
                if len(fuse_feats) > 1:
                    fused_feat = torch.cat(fuse_feats, dim=1)
                else:
                    fused_feat = fuse_feats[0]
                
                x = self.pts_neck_fuse(fused_feat)
                x = [x]  # Convert to list for compatibility
        
        return x
    
    def extract_img_feat(self, img, img_metas):
        """Extract features from images and backproject to BEV.
        
        Args:
            img: [B, N, C, H, W] or [B, C, H, W] image tensor
            img_metas: List of image metadata dicts
            
        Returns:
            img_feats: List of BEV features (or None if not configured)
        """
        if not self.with_img_backbone or img is None:
            return None
        
        bs = img.shape[0]
        # Handle different input formats (similar to fastbev1v)
        if img.dim() == 5:
            # [B, N, C, H, W] -> [B*N, C, H, W]
            if img.size(1) == 1:
                # Single view: [B, 1, C, H, W] -> [B, C, H, W]
                img = img.squeeze(1)
            else:
                # Multi-view: [B, N, C, H, W] -> [B*N, C, H, W]
                img = img.view(-1, *img.shape[2:])
        elif img.dim() == 4:
            # Already [B, C, H, W], keep as is
            pass
        
        # 2D feature extraction
        img_feats = self.img_backbone(img)  # List of multi-scale features
        img_feats = self.img_neck(img_feats)  # FPN outputs
        
        # Multi-scale fusion (use level 0)
        if self.img_neck_fuse is not None:
            feat = img_feats[0]
            for i in range(1, len(img_feats)):
                feat = torch.cat([
                    feat,
                    resize(img_feats[i], size=feat.shape[-2:], mode='bilinear')
                ], dim=1)
            feat = self.img_neck_fuse(feat)
        else:
            feat = img_feats[0]
        
        # Backproject to 3D
        volumes = []
        if isinstance(img_metas, dict):
            img_metas_list = [img_metas] * bs
        elif isinstance(img_metas, list):
            img_metas_list = img_metas
        else:
            raise TypeError(f"img_metas must be dict or list, got {type(img_metas)}")
        
        for b in range(bs):
            img_meta = img_metas_list[b] if b < len(img_metas_list) else img_metas_list[0]
            stride = math.ceil(img.shape[-1] / feat.shape[-1])
            
            proj = self._compute_projection(img_meta, stride).to(feat.device)
            points = get_points(
                torch.tensor(self.n_voxels),
                torch.tensor(self.voxel_size),
                torch.tensor(img_meta["lidar2img"]["origin"])
            ).to(feat.device)
            
            if self.backproject == 'inplace':
                vol = backproject_inplace(
                    feat[b:b+1], points, proj[None])
            else:
                # Use vanilla backprojection (sum over views)
                vol, _ = self._backproject_vanilla(
                    feat[b:b+1], points, proj[None])
                vol = vol.sum(0)
            
            volumes.append(vol)
        
        # Stack volumes: [B, C, X, Y, Z]
        x = torch.stack(volumes)
        
        # 3D processing
        if self.img_neck_3d is not None:
            x = self.img_neck_3d(x)  # Returns list of features
        
        # Ensure return format is list
        if not isinstance(x, list):
            x = [x]
        
        return x
    
    def _backproject_vanilla(self, features, points, projection):
        """Vanilla backprojection (returns volume and valid mask)."""
        n_images, n_channels, height, width = features.shape
        n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
        points = points.view(1, 3, -1).expand(n_images, 3, -1)
        points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
        points_2d_3 = torch.bmm(projection, points)
        x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
        y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
        z = points_2d_3[:, 2]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
        volume = torch.zeros(
            (n_images, n_channels, points.shape[-1]), device=features.device
        ).type_as(features)
        for i in range(n_images):
            volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
        volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
        valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
        return volume, valid
    
    def extract_feat(self, points=None, img=None, img_metas=None):
        """Extract features from both point cloud and images.
        
        Args:
            points: List of point cloud tensors
            img: Image tensor
            img_metas: List of image metadata dicts
            
        Returns:
            tuple: (img_feats, pts_feats) or (feats, None) depending on configuration
        """
        img_feats = self.extract_img_feat(img, img_metas) if self.with_img_backbone else None
        pts_feats = self.extract_pts_feat(points, img_metas) if self.with_pts_backbone else None
        
        # Fusion if both branches are present
        if img_feats is not None and pts_feats is not None and self.with_fusion:
            # Ensure both are lists
            if not isinstance(img_feats, list):
                img_feats = [img_feats]
            if not isinstance(pts_feats, list):
                pts_feats = [pts_feats]
            
            # Concatenate features along channel dimension
            # Both should have the same spatial size (same point_cloud_range)
            fuse_feats = []
            for img_feat, pts_feat in zip(img_feats, pts_feats):
                # Ensure same spatial size
                if img_feat.shape[2:] != pts_feat.shape[2:]:
                    # Resize to match (use pts_feat size as target)
                    img_feat = resize(
                        img_feat,
                        size=pts_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False)
                fused = torch.cat([pts_feat, img_feat], dim=1)
                fused = self.fusion_layer(fused)
                fuse_feats.append(fused)
            
            return (fuse_feats, None)  # Return as single branch after fusion
        elif img_feats is not None:
            if not isinstance(img_feats, list):
                img_feats = [img_feats]
            return (img_feats, None)
        elif pts_feats is not None:
            if not isinstance(pts_feats, list):
                pts_feats = [pts_feats]
            return (None, pts_feats)
        else:
            raise ValueError("At least one of img_backbone or pts_backbone must be configured")
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_bboxes_ignore=None,
                      img=None,
                      occ_gt=None,
                      occ_valid_mask=None,
                      **kwargs):
        """Training forward function.
        
        Args:
            points: List of point cloud tensors
            img_metas: List of image metadata dicts
            gt_bboxes_3d: List of ground truth 3D boxes
            gt_labels_3d: List of ground truth labels
            gt_bboxes_ignore: List of boxes to ignore
            img: Image tensor
            **kwargs: Additional arguments
            
        Returns:
            dict: Losses
        """
        img_feats, pts_feats = self.extract_feat(points, img, img_metas)
        
        # Determine which features to use
        if img_feats is not None:
            feats = img_feats
        elif pts_feats is not None:
            feats = pts_feats
        else:
            raise ValueError("No features extracted")
        
        losses = {}
        if self.with_occ_head:
            if occ_gt is None or occ_valid_mask is None:
                raise ValueError('occ_head requires occ_gt and occ_valid_mask')
            occ_logits = self.occ_head(feats)
            losses.update(self.occ_head.loss(occ_logits, occ_gt, occ_valid_mask))

        if self.with_pts_bbox:
            outs = self.pts_bbox_head(feats)
            head_type = type(self.pts_bbox_head).__name__
            if head_type == 'CenterHead':
                det_losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
            else:
                loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
                det_losses = self.pts_bbox_head.loss(
                    *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(det_losses)
        if not losses:
            raise RuntimeError('LVFusion has neither an OCC head nor a bbox head')
        return losses
    
    def simple_test(self, points=None, img_metas=None, img=None, rescale=False, **kwargs):
        """Test function without augmentation.
        
        Args:
            points: List of point cloud tensors
            img_metas: List of image metadata dicts
            img: Image tensor
            rescale: Whether to rescale results
            **kwargs: Additional arguments (filtered for gt_bboxes_3d, gt_labels_3d)
            
        Returns:
            list[dict]: Prediction results
        """
        # Filter out gt_bboxes_3d and gt_labels_3d from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['gt_bboxes_3d', 'gt_labels_3d']}
        
        img_feats, pts_feats = self.extract_feat(points, img, img_metas)
        
        # Determine which features to use
        if img_feats is not None:
            feats = img_feats
        elif pts_feats is not None:
            feats = pts_feats
        else:
            raise ValueError("No features extracted")
        
        occ_results = None
        if self.with_occ_head:
            occ_logits = self.occ_head(feats)
            occ_labels, occ_scores = self.occ_head.predict(occ_logits)
            occ_results = [
                {'occ_pred': labels, 'occ_score': scores}
                for labels, scores in zip(occ_labels, occ_scores)
            ]

        if not self.with_pts_bbox:
            if occ_results is None:
                raise RuntimeError('LVFusion has neither an OCC head nor a bbox head')
            return occ_results

        outs = self.pts_bbox_head(feats)
        
        # Handle img_metas format
        if isinstance(img_metas, list) and len(img_metas) > 0:
            if isinstance(img_metas[0], list):
                img_metas = img_metas[0] if len(img_metas) == 1 else [item for sublist in img_metas for item in sublist]
        elif isinstance(img_metas, dict):
            img_metas = [img_metas]
        
        # Check head type
        head_type = type(self.pts_bbox_head).__name__
        if head_type == 'CenterHead':
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
        else:
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)
        
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        if occ_results is not None:
            for bbox_result, occ_result in zip(bbox_results, occ_results):
                bbox_result.update(occ_result)
        return bbox_results
    
    def aug_test(self, points=None, img_metas=None, imgs=None, rescale=False):
        """Test function with augmentation."""
        # For simplicity, only support single sample augmentation
        feats_list = []
        for pts, img_meta, img_single in zip(points, img_metas, imgs if imgs else [None] * len(points)):
            img_feats, pts_feats = self.extract_feat(pts, img_single, img_meta)
            if img_feats is not None:
                feats_list.append(img_feats)
            elif pts_feats is not None:
                feats_list.append(pts_feats)
        
        aug_bboxes = []
        for feats, img_meta in zip(feats_list, img_metas):
            outs = self.pts_bbox_head(feats)
            
            if isinstance(img_meta, list) and len(img_meta) > 0:
                if isinstance(img_meta[0], list):
                    img_meta = img_meta[0] if len(img_meta) == 1 else [item for sublist in img_meta for item in sublist]
            elif isinstance(img_meta, dict):
                img_meta = [img_meta]
            
            head_type = type(self.pts_bbox_head).__name__
            if head_type == 'CenterHead':
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_meta, rescale=rescale)
            else:
                bbox_list = self.pts_bbox_head.get_bboxes(
                    *outs, img_meta, rescale=rescale)
            
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])
        
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return [merged_bboxes]
