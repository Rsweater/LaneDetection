from mmcv.cnn.bricks.registry import ATTENTION
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from einops import rearrange

from .fusion_blocks import AFFM, iAFFM, DAF


@ATTENTION.register_module()
class AddFusion(nn.Module):
    """
    CLRNet ROIGather module to process pooled features
    and make them interact with global information.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/roi_gather.py
    """

    def __init__(
        self,
        in_channel,
        num_priors,
        sample_points,
        fc_hidden_dim,
        cross_attention_weight=1.0,
    ):
        """
        Args:
            in_channels (int): Number of input feature channels.
            num_priors (int): Number of priors (anchors).
            sample_points (int): Number of pooling sample points (rows).
            fc_hidden_dim (int): FC middle channel number.
            refine_layers (int): The number of refine levels.
            mid_channels (int): The number of input channels to catconv.
            cross_attention_weight (float): Weight to fuse cross attention result.
        """
        super(AddFusion, self).__init__()
        self.in_channel = in_channel
        self.num_priors = num_priors
        self.cross_attention_weight = cross_attention_weight

        # feature enhance
        self.local_att = ConvModule(
                            fc_hidden_dim,
                            fc_hidden_dim,
                            9,
                            padding=4,
                            bias=False,
                            conv_cfg=dict(type='Conv1d'),
                            norm_cfg=dict(type='BN1d'),
                        )
        if self.cross_attention_weight > 0:
            self.global_att = FocusedLinearAttention(fc_hidden_dim)
            
        self.fusion_blocks = DAF()

        self.local_fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)
        self.local_fc_norm = nn.LayerNorm(fc_hidden_dim)

        self.glocal_fc = nn.Linear(sample_points * in_channel, fc_hidden_dim)
        self.glocal_fc_norm = nn.LayerNorm(fc_hidden_dim)

    def forward(self, x, bs):
        """
        ROIGather forward function.
        Args:
            roi_features (List[torch.Tensor]): List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, Cin, Ns, 1).
            fmap_pyramid (List[torch.Tensor]): Multi-level feature pyramid.
                Each tensor has a shape (B, Cin, H_i, W_i) where i is the pyramid level.
            layer_index (int): The current refine layer index.
        Returns:
            roi (torch.Tensor): Output feature tensors, shape (B, Np, Ch).
        B: batch size, Np: number of priors (anchors), Ns: number of sample points (rows),
        Cin: input channel number, Ch: hidden channel number.
        """
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            fmap_pyramid: feature map pyramid
            layer_index: currently on which layer to refine
        Return:
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        # [B * Np, Cin, Ns] -> [B * Np, Cin * Ns] -> [B * Np, Ch] -> [B, Np, Ch]
        local_feat = self.local_att(x[1])
        local_feat = local_feat.contiguous().view(bs * self.num_priors, -1)
        local_feat = F.relu(self.local_fc_norm(self.local_fc(local_feat)))
        local_feat = local_feat.view(bs, self.num_priors, -1) # [B, Np, Ch]

        # [B * Np, Cin, Ns] -> [B * Np, Cin * Ns] -> [B * Np, Ch]
        glocal_feat = x[0].contiguous().view(bs * self.num_priors, -1)
        glocal_feat = F.relu(self.glocal_fc_norm(self.glocal_fc(glocal_feat)))
        glocal_feat = glocal_feat.view(bs, self.num_priors, -1) # [B, Np, Ch]
        global_feat = self.global_att(glocal_feat)

        fusion_feat = self.fusion_blocks(local_feat, global_feat)

        # if self.cross_attention_weight > 0:
        #     context = self.attention(roi)
        #     roi = roi + self.cross_attention_weight * F.dropout(
        #         context, p=0.1, training=self.training
        #     )

        return fusion_feat

class FocusedLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size=[20, 20], num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # x = x.flatten(2).transpose(1, 2)
        # Features pooled by priors, shape (B, Np, C), so do not need flatten
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3).contiguous()
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding[:, :k.shape[1], :]
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])

        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)

        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)

        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num).contiguous()


        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")

        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    