# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize
from mmcv.parallel import DataContainer
from mmcv.runner import force_fp32

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.core.points import BasePoints
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class HascoPts(Base3DDetector):
    """HascoPts detector for point cloud 3D detection.
    
    Single-modal 3D detector based on point cloud, with neck_fuse to fuse multi-scale
    FPN features into a single feature map. It supports both Anchor3DHead and CenterHead.
    """

    def __init__(self,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_neck=None,
                 pts_neck_fuse=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(HascoPts, self).__init__(init_cfg=init_cfg)
        
        # Build voxelization layer
        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        
        # Build voxel encoder
        self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        
        # Build middle encoder
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        
        # Build backbone
        self.pts_backbone = builder.build_backbone(pts_backbone)
        
        # Build neck (e.g., FPN)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        else:
            self.pts_neck = None
        
        # Build neck_fuse layer to fuse multi-scale FPN features
        if pts_neck_fuse is not None:
            if isinstance(pts_neck_fuse['in_channels'], list):
                # Multiple input channels (for multi-scale fusion)
                in_channels = sum(pts_neck_fuse['in_channels'])
            else:
                in_channels = pts_neck_fuse['in_channels']
            
            out_channels = pts_neck_fuse['out_channels']
            kernel_size = pts_neck_fuse.get('kernel_size', 3)
            padding = pts_neck_fuse.get('padding', 1)
            
            self.pts_neck_fuse = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,  # No stride, keep same size
                padding=padding)
            
            # Get multi_scale_id to specify which feature map level to use as target size
            self.multi_scale_id = pts_neck_fuse.get('multi_scale_id', [0])
            if isinstance(self.multi_scale_id, int):
                self.multi_scale_id = [self.multi_scale_id]
        else:
            self.pts_neck_fuse = None
            self.multi_scale_id = None
        
        # Build bbox head
        if pts_bbox_head is not None:
            # Extract pts config from train_cfg and test_cfg (similar to MVXTwoStageDetector)
            # train_cfg is typically a ConfigDict with .pts attribute
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_test_cfg = test_cfg.pts if test_cfg else None
            
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        else:
            self.pts_bbox_head = None
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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

    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        # Handle different input formats:
        # 1. Training: DataContainer(tensor) -> tensor
        # 2. Testing: DataContainer([[tensor]]) -> [[tensor]] -> [tensor]
        # 3. Direct: [tensor] or tensor
        
        # Unpack DataContainer if needed
        if isinstance(points, DataContainer):
            points = points.data
        
        # Ensure points is a list
        if not isinstance(points, list):
            points = [points]
        
        # Handle nested lists (from test-time augmentation)
        # e.g., [[tensor]] -> [tensor]
        processed_points = []
        for p in points:
            # Unpack nested lists
            while isinstance(p, list) and len(p) == 1 and isinstance(p[0], torch.Tensor):
                p = p[0]
            # Unpack DataContainer if still present
            if isinstance(p, DataContainer):
                p = p.data
            # Convert BasePoints to tensor if needed
            if isinstance(p, BasePoints):
                p = p.tensor
            # Ensure it's a torch.Tensor
            if not isinstance(p, torch.Tensor):
                raise TypeError(f'points element must be torch.Tensor, but got {type(p)}')
            # Ensure it's 2D [N, C]
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
            
            # Fuse multi-scale features into a single feature map
            if self.pts_neck_fuse is not None and isinstance(x, (list, tuple)):
                # Use multi_scale_id to specify which feature map level to use as target size
                # multi_scale_id=[0] means use the first (largest) feature map size
                if self.multi_scale_id is not None and len(self.multi_scale_id) > 0:
                    target_level = self.multi_scale_id[0]
                    target_size = x[target_level].shape[2:]  # Use specified level's size
                else:
                    # Default: use the first (largest) feature map size
                    target_size = x[0].shape[2:]
                
                # Resize all feature maps to target_size and concatenate
                # Start from target_level and resize all subsequent levels to target_size
                fuse_feats = [x[target_level]]  # Start with target level feature
                for i in range(target_level + 1, len(x)):
                    resized_feat = resize(
                        x[i],
                        size=target_size,
                        mode='bilinear',
                        align_corners=False)
                    fuse_feats.append(resized_feat)
                
                # Concatenate all features along channel dimension
                if len(fuse_feats) > 1:
                    fused_feat = torch.cat(fuse_feats, dim=1)
                else:
                    fused_feat = fuse_feats[0]
                
                # Apply fusion convolution (no stride, keep same size)
                x = self.pts_neck_fuse(fused_feat)
                # Convert to list for compatibility with both head types
                x = [x]
        
        return x

    def extract_feats(self, points, img_metas=None):
        """Extract features for multiple point cloud augmentations."""
        return [self.extract_feat(pts, img_metas) for pts in points]

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """Test forward function.
        
        Filter out gt_bboxes_3d and gt_labels_3d from kwargs since 
        simple_test() doesn't accept them, but they may be present 
        in test pipeline for visualization purposes.
        """
        # Filter out gt_bboxes_3d and gt_labels_3d from kwargs
        # These are not needed for inference but may be in test pipeline
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['gt_bboxes_3d', 'gt_labels_3d']}
        # Call parent's forward_test with filtered kwargs
        return super().forward_test(points, img_metas, img=img, **filtered_kwargs)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None,
                      img=None,
                      **kwargs):
        """Training forward function.
        
        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sample
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            img (torch.Tensor, optional): Images (not used for point cloud detector).
            **kwargs: Additional arguments (ignored).
        
        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.pts_bbox_head(x)
        
        # Check head type by class name
        head_type = type(self.pts_bbox_head).__name__
        if head_type == 'CenterHead':
            # CenterHead format: loss(gt_bboxes_3d, gt_labels_3d, preds_dicts)
            losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
        else:
            # Anchor3DHead format: loss(cls_scores, bbox_preds, dir_cls_preds, gt_bboxes, gt_labels, img_metas, ...)
            loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
            losses = self.pts_bbox_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentation.
        
        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample.
            img (list[torch.Tensor], optional): Images (not used for point cloud detector).
            rescale (bool): Whether to rescale results.
        
        Returns:
            list[dict]: Prediction results.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.pts_bbox_head(x)
        
        # Ensure img_metas is a list of dicts (get_bboxes expects list[dict])
        # Handle nested lists from test-time augmentation
        if isinstance(img_metas, list) and len(img_metas) > 0:
            # If first element is also a list, it means nested structure [[{...}]]
            # Unpack to [{...}]
            if isinstance(img_metas[0], list):
                # Flatten nested list: [[{...}]] -> [{...}]
                img_metas = img_metas[0] if len(img_metas) == 1 else [item for sublist in img_metas for item in sublist]
            # If first element is a dict, img_metas is already [{...}] format
        elif isinstance(img_metas, dict):
            # Single dict, wrap in list
            img_metas = [img_metas]
        
        # Check head type by class name
        head_type = type(self.pts_bbox_head).__name__
        if head_type == 'CenterHead':
            # CenterHead format
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
        else:
            # Anchor3DHead format
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)
        
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentation."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            
            # Handle img_meta format (same as simple_test)
            if isinstance(img_meta, list) and len(img_meta) > 0:
                if isinstance(img_meta[0], list):
                    img_meta = img_meta[0] if len(img_meta) == 1 else [item for sublist in img_meta for item in sublist]
            elif isinstance(img_meta, dict):
                img_meta = [img_meta]
            
            # Check head type
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

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)

        return [merged_bboxes]
