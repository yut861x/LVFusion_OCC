"""Lightweight 2D BEV head for height-aware semantic occupancy."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS


@HEADS.register_module()
class BEVOccHead(nn.Module):
    """Predict ``[B, K, Z, Y, X]`` logits from a 2D BEV feature map."""

    def __init__(self, in_channels=256, hidden_channels=256, num_classes=12,
                 num_z=18, loss_weight=1.0, ignore_index=255,
                 score_threshold=0.0, unknown_sample_ratio=3.0,
                 output_size=None, class_weight=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_z = int(num_z)
        self.loss_weight = float(loss_weight)
        self.ignore_index = int(ignore_index)
        self.score_threshold = float(score_threshold)
        self.unknown_sample_ratio = float(unknown_sample_ratio)
        self.output_size = tuple(output_size) if output_size is not None else None
        if class_weight is None:
            weight = None
        else:
            weight = torch.as_tensor(class_weight, dtype=torch.float32)
            if weight.ndim != 1 or len(weight) != self.num_classes:
                raise ValueError(
                    f"class_weight must contain {self.num_classes} values")
        self.register_buffer("class_weight", weight, persistent=False)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(
            hidden_channels, self.num_classes * self.num_z, 1)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features):
        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError('BEVOccHead received an empty feature list')
            features = features[0]
        if features.ndim != 4:
            raise ValueError(f'BEV feature must be BCHW, got {features.shape}')
        logits = self.classifier(self.shared(features))
        if self.output_size is not None:
            out_h, out_w = self.output_size
            if out_h > logits.shape[-2] or out_w > logits.shape[-1]:
                raise ValueError(
                    f'OCC output_size={self.output_size} exceeds BEV feature '
                    f'size={tuple(logits.shape[-2:])}')
            # X starts at the ego origin, so retain the nearest forward cells.
            # Y is symmetric and already uses the full [-19.2, 19.2] extent.
            logits = logits[..., :out_h, :out_w]
        batch, _, height, width = logits.shape
        return logits.view(batch, self.num_classes, self.num_z, height, width)

    def loss(self, logits, occ_gt, occ_valid_mask):
        if occ_gt.ndim != 4 or occ_valid_mask.shape != occ_gt.shape:
            raise ValueError(
                f'OCC target/mask must be BZYX with equal shape, got '
                f'{occ_gt.shape} and {occ_valid_mask.shape}')
        if occ_gt.shape[1] != self.num_z:
            raise ValueError(f'OCC target Z={occ_gt.shape[1]}, expected {self.num_z}')
        if logits.shape[-2:] != occ_gt.shape[-2:]:
            batch, classes, depth = logits.shape[:3]
            flat = logits.reshape(batch, classes * depth, *logits.shape[-2:])
            flat = F.interpolate(flat, size=occ_gt.shape[-2:], mode='bilinear', align_corners=False)
            logits = flat.view(batch, classes, depth, *occ_gt.shape[-2:])

        valid = occ_valid_mask.bool() & (occ_gt != self.ignore_index)
        known = valid & (occ_gt != 0)
        unknown = valid & (occ_gt == 0)
        known_count = int(known.sum().item())
        max_unknown = int(max(1, known_count) * self.unknown_sample_ratio)
        unknown_indices = unknown.nonzero(as_tuple=False)
        if len(unknown_indices) > max_unknown:
            chosen = unknown_indices[torch.randperm(
                len(unknown_indices), device=unknown_indices.device)[:max_unknown]]
            sampled_unknown = torch.zeros_like(unknown)
            sampled_unknown[tuple(chosen.t())] = True
            valid = known | sampled_unknown
        valid_count = valid.sum()
        if valid_count.item() == 0:
            return {'loss_occ': logits.sum() * 0, 'occ_valid_voxels': valid_count.float()}
        # Move classes behind spatial dimensions so boolean indexing produces
        # [N_valid, K] logits and [N_valid] integer targets.
        selected_logits = logits.permute(0, 2, 3, 4, 1)[valid]
        selected_target = occ_gt[valid].long()
        loss = F.cross_entropy(selected_logits, selected_target,
                               weight=self.class_weight, reduction='mean')
        accuracy = (selected_logits.argmax(dim=1) == selected_target).float().mean()
        return {
            'loss_occ': loss * self.loss_weight,
            'occ_accuracy': accuracy.detach(),
            'occ_valid_voxels': valid_count.float().detach(),
            'occ_known_voxels': known.sum().float().detach(),
        }

    @torch.no_grad()
    def predict(self, logits):
        probabilities = logits.softmax(dim=1)
        scores, labels = probabilities.max(dim=1)
        if self.score_threshold > 0:
            labels = labels.masked_fill(scores < self.score_threshold, 0)
        return labels.to(torch.uint8), scores
