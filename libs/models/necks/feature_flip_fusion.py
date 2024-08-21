import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import NECKS
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

from ..utils.dilated_blocks import DilatedBlocks


class DCNV2_Ref(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(DCNV2_Ref, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels * 2,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(DCNV2_Ref, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, ref):
        """
        :param x: (B, C, H, W)
        :param ref: (B, C, H, W)
        :return:
        """
        concat = torch.cat([x, ref], dim=1)
        out = self.conv_offset(concat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@NECKS.register_module
class FeatureFlipFusion(nn.Module):
    def __init__(self, channels, dilated_blocks=None):
        super(FeatureFlipFusion, self).__init__()
        self.proj1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.proj2_conv = DCNV2_Ref(channels, channels, kernel_size=(3, 3), padding=1)
        self.proj2_norm = nn.BatchNorm2d(channels)

        if dilated_blocks is not None:
            self.dilated_blocks = DilatedBlocks(**dilated_blocks)
        else:
            self.dilated_blocks = None

    def forward(self, features):
        """
        :param features: [(B, C, H, W), ...]
        :return:

        """
        f = features[0]
        if self.dilated_blocks is not None:
            x = self.dilated_blocks(f)

        flipped = torch.flip(x, dims=(-1, ))  # 将图像特征图进行翻转, (B, C, H, W)
        x = self.proj1(x)      # (B, C, H, W)
        flipped = self.proj2_conv(flipped, x)     # (B, C, H, W)
        flipped = self.proj2_norm(flipped)      # (B, C, H, W)

        return F.relu(x + flipped), f
