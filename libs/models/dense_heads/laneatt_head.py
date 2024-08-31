
import math
import torch
import numpy as np
import torch.nn as nn

from mmdet.models.builder import build_loss
from mmdet.models.builder import HEADS
from nms import nms


@HEADS.register_module
class LaneATTHead(nn.Module):
    # Anchor angles, same ones used in Line-CNN
    left_angles = [72., 60., 49., 39., 30., 22.]
    right_angles = [108., 120., 131., 141., 150., 158.]
    bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

    def __init__(
        self, 
        in_channels,
        num_points,
        img_w=800,
        img_h=320,
        stride=32,
        anchor_feat_channels=64,
        anchors_freq_path=None,
        topk_anchors=None,
        return_attention_matrix=True,
        loss_cls=None,
        loss_reg=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super(LaneATTHead, self).__init__()
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.img_w, self.img_h = img_w, img_h
        self.stride = stride
        self.fmap_w, self.fmap_h = self.img_w // stride,  self.img_h // stride
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels

        self.return_attention_matrix = return_attention_matrix

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            # Sort the samples according to the number of positive matches, and keep only the topk_anchors for speed efficiency.
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]

        # Pre compute indices for the anchor pooling
        # anchor_coord: (n_anchors， H_f, 2)   2: (v, u)
        # invalid_mask: (n_anchors, H_f)
        self.anchor_coord, self.invalid_mask = self.compute_anchor_cut_indices(
            self.fmap_w, self.fmap_h)

        # Setup and initialize layers
        self.conv1 = nn.Conv2d(in_channels, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.to_cuda()
        self.init_weights()

    def generate_anchors(self, lateral_n, bottom_n):
        """generate anchors for left, right, and bottom lanes.
        Args:
            lateral_n (int): number of lateral anchors
            bottom_n (int): number of bottom anchors
        Returns:
            anchors (Tensor): (n_anchors, 2)
            anchors_cut (Tensor): (n_anchors, 2)
        """
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        """
        Args:
            angles: Anchor angles
            nb_origins: Number of lane anchors generated at this boundary.
        Returns:
            anchors: (n_anchors, 3+S)
            anchors_cut: (n_anchors, 3+H_f)
        """
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)   # 该边界处产生的Anchors总数.

        # each row, first for x and second for y:
        # 1 start_y, start_x, 1 lenght, S coordinates
        anchors = torch.zeros((n_anchors, 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut
    
    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys # len = H_f: [1, 0.9, 0.8, ..., 0.1]
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys # len = S: [1, ..., 0]
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y  # anchor[2] is the start y of the lane, 0 for bottom, 1 for top.
        anchor[3] = start_x  # normalized
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def compute_anchor_cut_indices(self, fmaps_w, fmaps_h):
        """
        Args:
            fmaps_w: W_f
            fmaps_h: H_f
        Returns:
            anchor_coord: (n_anchors， H_f, 2)   2: (v, u)
            invalid_mask: (n_anchors, H_f)
        """
        # definitions
        n_proposals = len(self.anchors_cut)     # n_anchors = N_origins*N_angles

        # indexing
        # corresponding x coordinates for anchor_ys, anchor_ys(len=H_f) bottom-->top（1-->0）
        # flip, bottom-->top:  (n_anchors， H_f)
        unclamped_xs = torch.flip((self.anchors_cut[:, 3:] / self.stride).round().long(), dims=(1,))
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)   # (n_anchors, H_f), Limit x coordinate range
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)    # Out-of-image mask (n_anchors， H_f)

        cut_ys = torch.arange(0, fmaps_h)   # (H_f, )
        cut_ys = cut_ys[None, :].repeat(n_proposals, 1)     # (n_anchors， H_f)
        anchor_coord = torch.stack([cut_ys, cut_xs], dim=-1)    # (n_anchors， H_f, 2)   2: (v, u)

        return anchor_coord, invalid_mask

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def init_weights(self):
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def to_cuda(self):
        self.anchors = self.anchors.cuda()
        self.anchor_ys = self.anchor_ys.cuda()
        self.anchors_cut = self.anchors_cut.cuda()
        self.invalid_mask = self.invalid_mask.cuda()
    
    def forward_train(self, x, img_metas, **kwargs):
        """Forward function for training mode.
        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred_dict = self(x)
        losses = self.loss(pred_dict, img_metas)
        return losses
    
    def simple_test(self, feats):
        """Test without augmentation."""
        pred_dict = self(feats)
        lanes, scores = self.get_lanes(pred_dict, as_lane=self.test_cfg.as_lane)
        result_dict = {
            "lanes": lanes,
            "scores": scores,
        }
        return result_dict

    def forward(self, x, param):
        """Take a feature map as input and output the predicted lanes and their probabilities.
        Args:
            x (Tensor): (B, C, H, W)
        Returns:
            cls_pred (Tensor): (B, n_anchors, 2)
            reg_pred (Tensor): (B, n_anchors, n_offsets+1)
            attention_matrix (Tensor): (B, n_anchors-1, H, W)
        """
        # init proposal features
        feat = self.conv1(x[-1])
        batch_size = x[-1].shape[0]  
        # (B, C, H, W) -> (B, anchor_feat_channels,H, W)
        proposal_feats = self.cut_anchor_features(feat)

        # Apply attention
        # Join proposals from all images into a single proposals features batch
        # (B, n_anchors, C, fH) --> (B*n_anchors, C*fH)
        proposal_feats = proposal_feats.view(-1, self.anchor_feat_channels * self.fmap_h)
        # Add attention features
        softmax = nn.Softmax(dim=1)
        # (B*n_anchors, C*H) --> (B*n_anchors, n_anchors-1)
        scores = self.attention_layer(proposal_feats)
        # (B*n_anchors, n_anchors-1)  --> (B, n_anchors, n_anchors-1)
        attention = softmax(scores).reshape(batch_size, len(self.anchors), -1)
        # (n_anchors, n_anchors) --> (B, n_anchors, n_anchors)
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(batch_size, 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()

        # (B*n_anchors, C * H) --> (B, n_anchors, C*H)
        proposal_feats = proposal_feats.reshape(batch_size, len(self.anchors), -1)
        # (B, n_anchors, n_anchors) @ (B, n_anchors, C*fH) --> (B, n_anchors, C*fH)
        attention_features = torch.bmm(
            torch.transpose(attention, 1, 2), 
            torch.transpose(proposal_feats, 1, 2)).transpose(1, 2)
        # (B, n_anchors, C*fH) --> (B*n_anchors, C*fH)    paper中的global feature
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        # (B, n_anchors, C*H) --> (B*n_anchors, C*H)    paper中的local feature
        proposal_feats = proposal_feats.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        # (B*n_anchors, 2*C*H)
        proposal_feats = torch.cat((attention_features, proposal_feats), dim=1)

        # cls and reg heads
        # (B*n_anchors, 2*C*H) --> (B*n_anchors, 2)
        cls_logits = self.cls_layer(proposal_feats)
        # (B*n_anchors, 2*C*H) --> (B*n_anchors,  S(n_offsets)+1(l))
        reg = self.reg_layer(proposal_feats)
        # (B*n_anchors, 2) --> (B, n_anchors, 2)
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])
        # (B*n_anchors, S(n_offsets)+1) --> (B, n_anchors, S(n_offsets)+1)
        reg = reg.reshape(batch_size, -1, reg.shape[1])

        # reg processing
        # (B, n_anchors, 3+S)  1 start_y, 1 start_x, 1 length, S coordinates
        lanes_preds = self.anchors.repeat(x.shape[0], 1, 1).clone()
        lanes_preds[:, :, 2] = reg[:, :, 0]      # l
        lanes_preds[:, :, 3:] += reg[:, :, 1:]    # x + x_offsets

        pred_dict = {
            'cls_logits': cls_logits,   # (B, n_anchors, 2)
            'lanes_preds': lanes_preds,     # (B, n_anchors, 3+S)
        }
        if self.return_attention_matrix:
            pred_dict['attention_matrix'] = attention_matrix

        return pred_dict
    
    def cut_anchor_features(self, features):
        batch_size, C = features.shape[:2]
        # (B, C, n_anchors, H)
        batch_anchor_features = features[:, :, self.anchor_coord[:, :, 0], self.anchor_coord[:, :, 1]]
        # (n_anchors, H) --> (B, C, n_anchors, H)
        invalid_mask = self.invalid_mask[None, None, ...].repeat(batch_size, C, 1, 1)
        batch_anchor_features[invalid_mask] = 0

        # (B, C, n_anchors, H) --> (B, n_anchors, C, H)
        batch_anchor_features = batch_anchor_features.permute(0, 2, 1, 3).contiguous()
        return batch_anchor_features
    
    def loss(self, pred_dict, img_metas):
        """Compute losses of the LaneATT head.
        Args:
            pred_dict (dict): A dictionary of predicted features.
                - cls_pred (Tensor): (B, n_anchors, 2)
                - reg_pred (Tensor): (B, n_anchors, n_offsets+1)
                - attention_matrix (Tensor): (B, n_anchors-1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = len(img_metas)
        device = pred_dict['cls_logits'].device
        cls_loss = torch.tensor(0.0).to(device)
        reg_loss = torch.tensor(0.0).to(device)
        # applay nms to remove redundant predictions
        # cls_logits (B, N_keep, )
        # lane_preds (B, N_keep, 3+S)
        # anchors (B, N_keep, 3+S)
        pred_proposal_list = self.nms(pred_dict['cls_logits'], pred_dict['lanes_preds'], self.train_cfg)

        for (cls_preds, lane_preds, anchors), img_meta in zip(pred_proposal_list, img_metas):
            gt_lanes = img_meta["gt_lanes"].clone().to(device)  # [n_lanes, 78]
            gt_lanes = gt_lanes[gt_lanes[:, 1] == 1]
            cls_targets = cls_preds.new_zeros(cls_preds.shape[0]).long()

            if gt_lanes.size(0) == 0:
                # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                cls_loss = (
                    cls_loss + self.loss_cls(cls_preds, cls_targets).sum()
                )
                continue

            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                (
                    matched_row_inds,
                    matched_col_inds,
                ) = self.assigner.assign(
                    pred_dict, gt_lanes.clone(), img_meta
                )
            
            # classification loss
            cls_targets[matched_row_inds, :] = 1
            cls_loss = self.loss_cls(cls_preds, cls_targets)
            
            # regression loss
            pos_preds = lane_preds[matched_row_inds] # (N_matched, 3+S)
            pos_targets = gt_lanes[matched_col_inds] # (N_matched, 3+S) # TODO: 标签转化；统一变量名
            reg_pred = pos_preds[:, 2:]  # (N_pos, 1+S)
            with torch.no_grad():
                positive_starts = (pos_preds[:, 0] * self.n_strips).round().long()   # (N_pos, ) proposal start_y
                target_starts = (pos_targets[:, 0] * self.n_strips).round().long()   # (N_pos, ) target start_y
                pos_targets[:, 2] -= positive_starts - target_starts
                ends = (positive_starts + pos_targets[:, 2] - 1).round().long()  # 其实就是end_gt

                # (N_pos, 1+S+1)    # 1+S+1: length + S + pad
                invalid_offsets_mask = lane_preds.new_zeros((num_total_pos, 1 + self.n_offsets + 1),
                                                            dtype=torch.int)
                all_indices = torch.arange(num_total_pos, dtype=torch.long)
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False

                reg_target = pos_targets[:, 2:]  # (N_pos, 1+S)
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

    def nms(self, batch_cls_logits, batch_lanes_preds, cfg):
        proposals_list = []
        softmax = nn.Softmax(dim=1)
        for cls_logits, lane_preds in zip(batch_cls_logits, batch_lanes_preds):
            # cls_logits: (n_anchors, 2)
            # lane_preds: (n_anchors, 3+S)

            with torch.no_grad():
                scores = softmax(cls_logits)[:, 1]    # (n_anchors, )  fg score
                keep, num_to_keep, _ = nms(
                    lane_preds.clone(),    # (N, 3+S)
                    scores,             # (N, )
                    overlap=cfg.nms_thres,
                    top_k=cfg.max_lanes,
                )
                keep = keep[:num_to_keep]       # (N_keep, )

            cls_logits = cls_logits[keep]   # (N_keep, )
            lane_preds = lane_preds[keep]   # (N_keep, 3+S)
            anchors = self.anchors[keep]    # (N_keep, 3+S)

            proposals_list.append((cls_logits, lane_preds, anchors))

        return proposals_list