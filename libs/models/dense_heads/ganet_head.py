
import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmdet.models.builder import build_loss

from libs.models.layers.ctnet_head import CtnetHead


@HEADS.register_module
class GAHead(nn.Module):
    def __init__(
            self,
            in_channels=64, 
            num_classes=1, 
            hm_idx=0,
            loss_heatmap=None,
            loss_kp_offset=None,
            loss_sp_offset=None,
            loss_aux=None,
            train_cfg=None,
            test_cfg=None
    ):
        super(GAHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hm_idx = hm_idx

        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_kp_offset = build_loss(loss_kp_offset)
        self.loss_sp_offset = build_loss(loss_sp_offset)
        self.loss_aux = build_loss(loss_aux)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_layers(in_channels)

    def _init_layers(self, in_channels):
        self.keypts_head = CtnetHead(
            in_channels=in_channels,
            heads_dict={
                'hm': {'out_channels': 1, 'num_conv': 2},
            },
            final_kernel=1,
            init_bias=-2.19,
            use_bias=True
        )

        self.kp_offset_head = CtnetHead(
            in_channels=in_channels,
            heads_dict={
                'offset': {'out_channels': 2, 'num_conv': 2},
            },
            final_kernel=1,
            init_bias=-2.19,
            use_bias=True
        )

        self.sp_offset_head = CtnetHead(
            in_channels=in_channels,
            heads_dict={
                'offset': {'out_channels': 2, 'num_conv': 2},
            },
            final_kernel=1,
            init_bias=-2.19,
            use_bias=True
        )
    
    def forward_train(self, x, img_metas, **kwargs):
        predictions = self(x)
        # head_dict: {
        #     kpts_hm: (B, C=1, H3, W3)
        #     kp_offset: (B, 2, H3, W3)
        #     sp_offset: (B, 2, H3, W3)
        # }
        deform_points = x['deform_points'][0]      # (B, num_points*2, H3, W3)
        predictions['deform_points'] = deform_points
        losses = self.loss(predictions, img_metas)

        return losses

    def simple_test(self, x, img_metas):
        predictions = self(x)
        # head_dict: {
        #     kpts_hm: (B, C=1, H3, W3)
        #     kp_offset: (B, 2, H3, W3)
        #     sp_offset: (B, 2, H3, W3)
        # }
        results_list = self.get_lanes(
            predictions, img_metas=img_metas)
        return results_list
    
    def forward(self, x):
        f_hm = x['features'][self.hm_idx]  # (B, C, H3, W3)
        aux_feat = x['aux_feat']           # (B, C=64, H3, W3)

        # (B, C=64, H3, W3) --> (B, C=64, H3, W3) --> (B, C=1, H3, W3)
        kpts_hm = self.keypts_head(f_hm)['hm']
        kpts_hm = torch.clamp(kpts_hm.sigmoid(), min=1e-4, max=1 - 1e-4)  # (B, C=1, H3, W3)

        if aux_feat is not None:
            f_hm = aux_feat
        
        # (B, C=64, H3, W3) --> (B, 2, H3, W3)
        kp_offset = self.kp_offset_head(f_hm)['offset']
        sp_offset = self.sp_offset_head(f_hm)['offset']
        
        return dict(
            kpts_hm=kpts_hm,
            kp_offset=kp_offset, 
            sp_offset=sp_offset
        )
