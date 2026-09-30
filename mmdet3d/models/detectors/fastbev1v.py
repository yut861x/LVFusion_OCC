# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import copy

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import auto_fp16

try:
    import onnxruntime
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False
    import warnings
    warnings.warn("onnxruntime not installed, ONNX inference will not work")


@DETECTORS.register_module()
class FastBEV1V(Base3DDetector):
    """
    FastBEV single-view (CAM_FRONT only, no temporal)
    """

    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        bbox_head,
        seg_head,
        n_voxels,
        voxel_size,
        train_cfg=None,
        test_cfg=None,
        with_cp=False,
        backproject='inplace',
        style='v1',
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)

        self.neck_fuse = nn.Conv2d(
            neck_fuse['in_channels'][0],
            neck_fuse['out_channels'][0],
            kernel_size=3,
            padding=1)

        if bbox_head is not None:
            # For CenterHead, train_cfg and test_cfg should be passed directly
            # For other heads, they might be nested under 'pts' key
            if 'CenterHead' in bbox_head.get('type', ''):
                # Extract pts config if exists, otherwise use train_cfg/test_cfg directly
                if train_cfg is not None and 'pts' in train_cfg:
                    bbox_head.update(train_cfg=train_cfg['pts'])
                else:
                    bbox_head.update(train_cfg=train_cfg)
                if test_cfg is not None and 'pts' in test_cfg:
                    bbox_head.update(test_cfg=test_cfg['pts'])
                else:
                    bbox_head.update(test_cfg=test_cfg)
            else:
                bbox_head.update(train_cfg=train_cfg)
                bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            # CenterHead doesn't need voxel_size attribute, only FreeAnchor3DHead needs it
            if 'CenterHead' not in bbox_head.get('type', ''):
                self.bbox_head.voxel_size = voxel_size[0]
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        self.n_voxels = n_voxels[0]
        self.voxel_size = voxel_size[0]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.with_cp = with_cp
        self.backproject = backproject
        self.style = style

    @staticmethod
    def _compute_projection(img_meta, stride):
        intrinsic = torch.tensor(
            img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsic = torch.tensor(
            img_meta["lidar2img"]["extrinsic"][0])  # CAM_FRONT
        return intrinsic @ extrinsic[:3]

    def extract_feat(self, img, img_metas, mode='train'):
        """
        img: [B, 1, 3, H, W]
        mode: 'train', 'test', or 'test_onnx'
        """
        bs = img.shape[0]
        img = img[:, 0]  # [B, 3, H, W]

        ###################### 2D feature extraction ######################
        if mode in ['train', 'test']:
            feats = self.backbone(img)
            feats = self.neck(feats)

            # multi-level fuse (only use level-0)
            feat = feats[0]
            for i in range(1, len(feats)):
                feat = torch.cat([
                    feat,
                    resize(feats[i], size=feat.shape[-2:], mode='bilinear')
                ], dim=1)
            feat = self.neck_fuse(feat)
        
        elif mode == 'test_onnx':
            # 使用ONNX Runtime进行2D特征提取
            if not HAS_ONNXRUNTIME:
                raise RuntimeError("onnxruntime is required for ONNX inference")
            
            # 获取ONNX模型路径（优先级：模型属性 > test_cfg > 环境变量 > 默认路径）
            onnx_2d_path = None
            if hasattr(self, 'onnx_2d_path') and self.onnx_2d_path:
                onnx_2d_path = self.onnx_2d_path
            elif self.test_cfg and self.test_cfg.get('onnx_2d_path'):
                onnx_2d_path = self.test_cfg.get('onnx_2d_path')
            else:
                onnx_2d_path = os.getenv('ONNX_2D_MODEL_PATH', './output/onnx/fastbev1v_2d.onnx')
            
            if not os.path.exists(onnx_2d_path):
                raise FileNotFoundError(f"ONNX 2D model not found: {onnx_2d_path}")
            
            # 优先使用GPU，如果没有则使用CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess_2d = onnxruntime.InferenceSession(onnx_2d_path, providers=providers)
            
            # 检查实际使用的provider
            actual_provider = sess_2d.get_providers()[0]
            print(f"ONNX 2D using provider: {actual_provider}")
            
            all_outputs = []
            for i in range(img.size(0)):
                input_data = img[i].unsqueeze(0).cpu().float().numpy()  # [1, 3, H, W]
                input_name = sess_2d.get_inputs()[0].name
                onnx_input = {input_name: input_data}
                onnx_output = sess_2d.run(None, onnx_input)
                # 从numpy转换为tensor，确保在正确的设备上
                output_data = torch.from_numpy(onnx_output[0]).float()
                # 使用img.device（与输入数据保持一致）
                output_data = output_data.to(img.device)
                all_outputs.append(output_data)
            
            feat = torch.cat(all_outputs, dim=0)  # [B, 64, H', W']
            # 如果ONNX输出是 [B, H', W', 64] 格式（DEPLOY模式），需要转换
            if len(feat.shape) == 4 and feat.shape[-1] == 64:
                feat = feat.permute(0, 3, 1, 2)  # [B, H', W', 64] -> [B, 64, H', W']
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        ###################### 2D feature extraction ######################

        ###################### 2D -> 3D backproject ######################
        volumes = []
        # 处理img_metas：可能是list或dict
        if isinstance(img_metas, dict):
            # 如果是单个dict，转换为list
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
                vol, _ = backproject_vanilla(
                    feat[b:b+1], points, proj[None])
                vol = vol.sum(0)

            volumes.append(vol)
        ###################### 2D -> 3D backproject ######################

        x = torch.stack(volumes)  # [B, C, X, Y, Z]

        ###################### 3D processing ######################
        if mode in ['train', 'test']:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(self.neck_3d, x)
            else:
                x = self.neck_3d(x)
        
        elif mode == 'test_onnx':
            # 使用ONNX Runtime进行3D处理
            if not HAS_ONNXRUNTIME:
                raise RuntimeError("onnxruntime is required for ONNX inference")
            
            # 获取ONNX模型路径（优先级：模型属性 > test_cfg > 环境变量 > 默认路径）
            onnx_3d_path = None
            if hasattr(self, 'onnx_3d_path') and self.onnx_3d_path:
                onnx_3d_path = self.onnx_3d_path
            elif self.test_cfg and self.test_cfg.get('onnx_3d_path'):
                onnx_3d_path = self.test_cfg.get('onnx_3d_path')
            else:
                onnx_3d_path = os.getenv('ONNX_3D_MODEL_PATH', './output/onnx/fastbev1v_3d.onnx')
            
            if not os.path.exists(onnx_3d_path):
                raise FileNotFoundError(f"ONNX 3D model not found: {onnx_3d_path}")
            
            # 优先使用GPU，如果没有则使用CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess_3d = onnxruntime.InferenceSession(onnx_3d_path, providers=providers)
            
            # 检查实际使用的provider
            actual_provider = sess_3d.get_providers()[0]
            print(f"ONNX 3D using provider: {actual_provider}")
            
            # 准备输入：将 [B, C, X, Y, Z] 转换为ONNX模型期望的格式
            # 对于FastBEV1V，可能需要reshape为 [B, C*Z, X, Y] 或保持 [B, C, X, Y, Z]
            x_input = x[0]  # [C, X, Y, Z]
            # 转换为 [1, C, X, Y, Z] 或根据ONNX模型输入格式调整
            if len(x_input.shape) == 4:  # [C, X, Y, Z]
                x_input = x_input.unsqueeze(0)  # [1, C, X, Y, Z]
            
            # 记录原始设备
            original_device = x_input.device
            
            # 尝试不同的输入格式
            try:
                # 格式1: 5D tensor [1, C, X, Y, Z]
                input_data = x_input.cpu().float().numpy()
                input_name = sess_3d.get_inputs()[0].name
                onnx_input = {input_name: input_data}
                onnx_outputs = sess_3d.run(None, onnx_input)
            except Exception:
                # 格式2: 4D tensor [1, C*Z, X, Y] 或 [1, C, X, Y]
                # 根据配置reshape
                C, X, Y, Z = x_input.shape[1:] if len(x_input.shape) == 5 else (x_input.shape[1], *x_input.shape[2:], 1)
                if len(x_input.shape) == 5:
                    x_input_4d = x_input.permute(0, 4, 1, 2, 3).reshape(1, C*Z, X, Y)  # [1, C*Z, X, Y]
                else:
                    x_input_4d = x_input
                input_data = x_input_4d.cpu().float().numpy()
                input_name = sess_3d.get_inputs()[0].name
                onnx_input = {input_name: input_data}
                onnx_outputs = sess_3d.run(None, onnx_input)
            
            # 转换输出为PyTorch tensor
            # ONNX输出格式: [cls_score, bbox_pred, dir_cls_pred]
            # bbox_head.forward返回格式: (list[cls_score], list[bbox_pred], list[dir_cls_pred])
            # 每个tensor应该是4D: (N, C, H, W)
            if len(onnx_outputs) >= 3:
                cls_score_np = onnx_outputs[0]
                bbox_pred_np = onnx_outputs[1]
                dir_cls_pred_np = onnx_outputs[2]
                
                # 转换为PyTorch tensor，并确保是float32类型（避免Half类型在CPU上的问题）
                # ONNX Runtime在CPU上运行，输出是CPU tensor
                cls_score_tensor = torch.from_numpy(cls_score_np).float()
                bbox_pred_tensor = torch.from_numpy(bbox_pred_np).float()
                dir_cls_pred_tensor = torch.from_numpy(dir_cls_pred_np).float() if dir_cls_pred_np is not None else None
                
                # 确定目标设备：使用img.device（与输入数据保持一致）
                # 如果img在CPU上，就保持在CPU；如果img在GPU上，就移到GPU
                target_device = img.device
                
                cls_score_tensor = cls_score_tensor.to(target_device)
                bbox_pred_tensor = bbox_pred_tensor.to(target_device)
                if dir_cls_pred_tensor is not None:
                    dir_cls_pred_tensor = dir_cls_pred_tensor.to(target_device)
                
                # 确保格式正确：bbox_head期望list格式，每个元素是4D tensor (N, C, H, W)
                # get_bboxes_single期望: dir_cls_pred.permute(1, 2, 0)，说明应该是4D (N, C, H, W)
                # 如果维度不对，尝试修复
                if len(cls_score_tensor.shape) != 4:
                    print(f"WARNING: cls_score shape {cls_score_tensor.shape} is not 4D, attempting to reshape")
                    # 可能需要根据实际情况调整
                if len(bbox_pred_tensor.shape) != 4:
                    print(f"WARNING: bbox_pred shape {bbox_pred_tensor.shape} is not 4D, attempting to reshape")
                if dir_cls_pred_tensor is not None:
                    if len(dir_cls_pred_tensor.shape) != 4:
                        print(f"WARNING: dir_cls_pred shape {dir_cls_pred_tensor.shape} is not 4D, attempting to reshape")
                        # 如果是3D (N, C, H*W)，需要reshape为4D (N, C, H, W)
                        # 但这需要知道H和W，可能需要从cls_score推断
                        if len(dir_cls_pred_tensor.shape) == 3:
                            # 假设与cls_score有相同的H和W
                            N, C, HW = dir_cls_pred_tensor.shape
                            if len(cls_score_tensor.shape) == 4:
                                _, _, H, W = cls_score_tensor.shape
                                if HW == H * W:
                                    dir_cls_pred_tensor = dir_cls_pred_tensor.view(N, C, H, W)
                                    print(f"Reshaped dir_cls_pred to {dir_cls_pred_tensor.shape}")
                
                # 包装为list格式（bbox_head期望list）
                cls_score = [cls_score_tensor]
                bbox_pred = [bbox_pred_tensor]
                dir_cls_pred = [dir_cls_pred_tensor] if dir_cls_pred_tensor is not None else None
                
                x = (cls_score, bbox_pred, dir_cls_pred)
            elif len(onnx_outputs) == 2:
                cls_score_np = onnx_outputs[0]
                bbox_pred_np = onnx_outputs[1]
                
                cls_score_tensor = torch.from_numpy(cls_score_np).float().to(img.device)
                bbox_pred_tensor = torch.from_numpy(bbox_pred_np).float().to(img.device)
                
                # 检查维度
                if len(cls_score_tensor.shape) != 4:
                    raise ValueError(f"Expected 4D cls_score tensor, got shape {cls_score_tensor.shape}")
                if len(bbox_pred_tensor.shape) != 4:
                    raise ValueError(f"Expected 4D bbox_pred tensor, got shape {bbox_pred_tensor.shape}")
                
                cls_score = [cls_score_tensor]
                bbox_pred = [bbox_pred_tensor]
                x = (cls_score, bbox_pred, None)
            else:
                # 如果只有一个输出，可能是neck_3d的输出
                x = [torch.from_numpy(onnx_outputs[0]).float().to(img.device)]
        ###################### 3D processing ######################

        return x

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Forward function for training and testing.
        
        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            # 根据输入形状判断导出2D还是3D模型（参考FastBEV的实现）
            if isinstance(img, list):
                img_shape = img[0].shape
            else:
                img_shape = img.shape
            
            # 2D导出：输入为 [1, 3, H, W] 的单张图像
            if len(img_shape) == 4 and img_shape[1] == 3:
                if isinstance(img, list):
                    return self.onnx_export_2d(img[0], img_metas)
                else:
                    return self.onnx_export_2d(img, img_metas)
            # 3D导出：输入为 [1, C, X, Y] 的3D volume特征
            elif len(img_shape) == 4 and img_shape[1] in [64, 256]:
                if isinstance(img, list):
                    # 如果是列表，需要转换为5D tensor [1, C, X, Y, Z]
                    # 对于FastBEV1V，输入应该是 [1, 64, X, Y, Z]
                    # 但ONNX导出时可能是 [1, 64, X, Y] 需要reshape
                    volume = img[0] if len(img) == 1 else torch.stack(img, dim=0)
                    # 如果输入是4D，需要reshape为5D
                    if len(volume.shape) == 4:
                        # [1, 64, X, Y] -> [1, 64, X, Y, 1] 或根据配置确定Z维度
                        volume = volume.unsqueeze(-1)
                    return self.onnx_export_3d(volume, img_metas)
            else:
                raise NotImplementedError(f"Unsupported input shape for ONNX export: {img_shape}")
        
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        return self.forward_test(img, img_metas)

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs):
        x = self.extract_feat(img, img_metas, mode='train')

        losses = dict()
        if self.bbox_head:
            outs = self.bbox_head(x)
            # CenterHead has different loss signature: loss(gt_bboxes_3d, gt_labels_3d, preds_dicts)
            # FreeAnchor3DHead has: loss(*args, gt_bboxes_3d, gt_labels_3d, img_metas)
            if 'CenterHead' in type(self.bbox_head).__name__:
                losses.update(
                    self.bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs))
            else:
                losses.update(
                    self.bbox_head.loss(
                        *outs, gt_bboxes_3d, gt_labels_3d, img_metas))

        if self.seg_head:
            seg = self.seg_head(x)
            losses.update(
                self.seg_head.losses(seg, gt_bev_seg[0][None].long()))

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """
        Args:
            img: list[torch.Tensor] or DataContainer
            img_metas: list[list[dict]] or DataContainer
        """
        # 处理DataContainer
        from mmcv.parallel import DataContainer
        if isinstance(img, DataContainer):
            img = img.data
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        
        # 确保img是list格式
        if not isinstance(img, list):
            img = [img]
        if not isinstance(img_metas, list):
            img_metas = [img_metas]
        
        if self.test_cfg is None or not self.test_cfg.get('use_tta', False):
            test_mode = self.test_cfg.get('test_mode', 'test') if self.test_cfg else 'test'
            # simple_test期望单个batch的数据
            return self.simple_test(img[0], img_metas[0], mode=test_mode)
        return self.aug_test(img, img_metas)

    def simple_test(self, img, img_metas, mode='test'):
        """
        Args:
            img: torch.Tensor [B, 1, 3, H, W] or [B, 3, H, W]
            img_metas: list[dict] or dict
        """
        # 确保img_metas是列表格式（get_bboxes期望列表）
        if isinstance(img_metas, dict):
            img_metas_list = [img_metas]
        elif isinstance(img_metas, list):
            img_metas_list = img_metas
        else:
            raise TypeError(f"img_metas must be dict or list, got {type(img_metas)}")
        
        if mode == 'test_onnx':
            # ONNX推理模式：extract_feat已经返回bbox_head的输出格式
            x = self.extract_feat(img, img_metas, mode='test_onnx')
            # x已经是 (cls_score, bbox_pred, dir_cls_pred) 格式
            if 'CenterHead' in type(self.bbox_head).__name__:
                bbox_list = self.bbox_head.get_bboxes(x, img_metas_list, rescale=False)
            else:
                bbox_list = self.bbox_head.get_bboxes(*x, img_metas_list)
        else:
            # 正常推理模式
            x = self.extract_feat(img, img_metas, mode='test')
            outs = self.bbox_head(x)
            # CenterHead.get_bboxes() 签名: get_bboxes(preds_dicts, img_metas, img=None, rescale=False)
            # FreeAnchor3DHead.get_bboxes() 签名: get_bboxes(cls_scores, bbox_preds, dir_cls_preds, input_metas, ...)
            if 'CenterHead' in type(self.bbox_head).__name__:
                bbox_list = self.bbox_head.get_bboxes(outs, img_metas_list, rescale=False)
            else:
                bbox_list = self.bbox_head.get_bboxes(*outs, img_metas_list)
        
        return [
            bbox3d2result(b, s, l)
            for b, s, l in bbox_list
        ]

    def aug_test(self, imgs, img_metas):
        """Test with augmentation for single-view FastBEV."""
        # For single-view, we can implement a simple version
        # that just calls simple_test if TTA is not needed
        # or implement proper TTA if needed
        return self.simple_test(imgs, img_metas)

    def onnx_export_2d(self, img, img_metas):
        """
        ONNX export for 2D feature extraction (backbone + neck + neck_fuse).
        Input: [1, 3, H, W] (single camera image)
        Output: [1, 64, H', W'] (2D features)
        """
        x = self.backbone(img)
        feats = self.neck(x)
        
        # multi-level fuse (only use level-0)
        feat = feats[0]
        for i in range(1, len(feats)):
            feat = torch.cat([
                feat,
                resize(feats[i], size=feat.shape[-2:], mode='bilinear', align_corners=False)
            ], dim=1)
        feat = self.neck_fuse(feat)
        
        if bool(os.getenv("DEPLOY", False)):
            feat = feat.permute(0, 2, 3, 1)
            return feat
        
        return feat

    def onnx_export_3d(self, x, _):
        """
        ONNX export for 3D processing (neck_3d + bbox_head).
        Input: [1, 64, X, Y, Z] (3D volume features) 或 [1, C, X, Y] (需要reshape)
        Output: [cls_score, bbox_pred, dir_cls_preds]
        """
        # 如果输入是4D [1, C, X, Y]，需要reshape为5D [1, C, X, Y, Z]
        if len(x.shape) == 4:
            # 根据配置确定Z维度，默认为4
            z_dim = self.n_voxels[2] if hasattr(self, 'n_voxels') and isinstance(self.n_voxels, list) else 4
            x = x.unsqueeze(-1).expand(-1, -1, -1, -1, z_dim)
        
        # x: [1, 64, X, Y, Z] for single camera
        # neck_3d在DEPLOY模式下期望输入是 (N, X, Y, Z, C)，但传入的是 (N, C, X, Y, Z)
        # 所以不设置DEPLOY，使用正常模式的reshape
        # 或者先permute输入到DEPLOY格式
        deploy_mode = os.getenv("DEPLOY", False)
        if deploy_mode:
            # DEPLOY模式：需要先permute (N, C, X, Y, Z) -> (N, X, Y, Z, C)
            x = x.permute(0, 2, 3, 4, 1)
        
        if self.style == "v1":
            x = self.neck_3d(x)  # [[1, 192, Y, X], ] 或类似格式
        else:
            raise NotImplementedError(f"Style {self.style} not implemented for ONNX export")
        
        if self.bbox_head is not None:
            # CenterHead.forward() 返回 tuple(list[dict])，格式与 FreeAnchor3DHead 不同
            if 'CenterHead' in type(self.bbox_head).__name__:
                output_dicts = self.bbox_head(x)
                
                # 将 CenterHead 的输出转换为 ONNX 导出格式
                cls_score = []
                bbox_pred = []
                
                for task_id, task_pred in enumerate(output_dicts):
                    assert len(task_pred) == 1
                    task_pred = task_pred[0]
                    
                    # heatmap
                    hm = task_pred['heatmap']
                    cls_score.append(hm)
                    
                    # bbox components: reg, height, dim, rot, vel
                    reg = task_pred['reg']
                    height = task_pred['height']
                    dim = task_pred['dim']
                    rot = task_pred['rot']
                    vel = task_pred.get('vel', None)
                    
                    if vel is not None:
                        boxes = torch.cat([reg, height, dim, rot, vel], dim=1)
                    else:
                        boxes = torch.cat([reg, height, dim, rot], dim=1)
                    bbox_pred.append(boxes)
                
                # 合并所有任务的输出
                cls_score = torch.cat(cls_score, dim=1)  # [B, num_classes_all, H, W]
                bbox_pred = torch.cat(bbox_pred, dim=1)  # [B, code_size_all, H, W]
                dir_cls_preds = None
            else:
                cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
                cls_score = [item.sigmoid() for item in cls_score]
            
            # ONNX导出时，自动设置DEPLOY模式以返回正确格式
            if torch.onnx.is_in_onnx_export() or os.getenv("DEPLOY", False):
                if dir_cls_preds is None:
                    return [cls_score, bbox_pred]
                else:
                    return [cls_score, bbox_pred, dir_cls_preds]
        
        return x


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    grid = torch.stack(torch.meshgrid(
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ))
    new_origin = origin - n_voxels / 2 * voxel_size
    return grid * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [1, 64, H, W]
        points: [3, X, Y, Z]
        projection: [1, 3, 4]
    output:
        volume: [1, 64, X, Y, Z]
        valid: [1, 1, X, Y, Z]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, X, Y, Z] -> [1, 3, N] -> [1, 3, N]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [1, 3, N] -> [1, 4, N]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [1, 3, 4] * [1, 4, N] -> [1, 3, N]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [1, N]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [1, N]
    z = points_2d_3[:, 2]  # [1, N]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [1, N]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [1, 64, N]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [1, 64, N] -> [1, 64, X, Y, Z]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [1, N] -> [1, 1, X, Y, Z]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


def backproject_inplace(features, points, projection):
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
