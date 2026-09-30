# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models import BACKBONES


@BACKBONES.register_module()
class HASCO_PTS_BACKBONE(BaseModule):
    """HASCO Point Cloud Backbone network.

    This backbone includes downsampling blocks and outputs multi-scale features
    compatible with FPN neck.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for each stage.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
        init_cfg (dict): Initialization config dict.
        pretrained (str): Path to pretrained weights.
    """

    def __init__(self,
                 in_channels=64,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super(HASCO_PTS_BACKBONE, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        self._layer_strides = layer_strides
        self._num_filters = out_channels
        self._layer_nums = layer_nums
        self._num_input_features = in_channels
        self._norm_cfg = norm_cfg

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []

        # Build downsampling blocks
        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def _make_layer(self, inplanes, planes, num_blocks, stride=1,
                    norm_cfg=None, conv_cfg=None):
        """Build a layer with downsampling and residual blocks.

        Args:
            inplanes (int): Input channels.
            planes (int): Output channels.
            num_blocks (int): Number of blocks in this layer.
            stride (int): Stride of the first conv.
            norm_cfg (dict): Normalization config.
            conv_cfg (dict): Convolution config.

        Returns:
            nn.Sequential: The built layer.
            int: Output channels.
        """
        block = [
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes,
                3,
                stride=stride,
                padding=1),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=False),  # Use inplace=False to avoid gradient issues
        ]

        for j in range(num_blocks):
            block.append(
                build_conv_layer(
                    conv_cfg,
                    planes,
                    planes,
                    3,
                    padding=1))
            block.append(build_norm_layer(norm_cfg, planes)[1])
            block.append(nn.ReLU(inplace=False))  # Use inplace=False to avoid gradient issues

        block = nn.Sequential(*block)
        return block, planes

    @property
    def downsample_factor(self):
        """Calculate the overall downsample factor."""
        return np.prod(self._layer_strides)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features from blocks, compatible with FPN.
                Returns tuple of features from each block stage.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)

        # Return multi-scale features (similar to SECOND backbone)
        # This allows compatibility with FPN neck
        return tuple(outs)
