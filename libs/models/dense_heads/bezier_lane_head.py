
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.core import build_assigner
from mmdet.models.builder import HEADS
from mmdet.models.builder import build_loss

from libs.core.lane.bezier_curve import BezierCurve
from libs.core.lane.lane_utils import Lane

@HEADS.register_module
class BezierLaneHead(nn.Module):
    def __init__(self,
        in_channels=256,
        fc_hidden_dim=256,
        stacked_convs=2,
        order=3,
        num_sample_points=100,
        loss_cls=None,
        loss_dist=None,
        loss_seg=None,
        train_cfg=None,
        test_cfg=None,
        ):
        super(BezierLaneHead, self).__init__()
        self.bezier_curve = BezierCurve(order=order)
        self.num_sample_points = num_sample_points

        self.loss_cls = build_loss(loss_cls)
        self.loss_dist = build_loss(loss_dist)
        self.loss_seg = (
            build_loss(loss_seg) if loss_seg["loss_weight"] > 0 else None
        )
        # Auxiliary head
        if self.loss_seg:
            self.seg_conv = ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
            self.seg_conv_out = nn.Conv2d(in_channels, loss_seg.num_classes, kernel_size=1, bias=False)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])

        self._init_layers(in_channels, stacked_convs, fc_hidden_dim, order)

    def _init_layers(self, in_channels, stacked_convs, fc_hidden_dim, order):
        """Initialize layers of the head."""
        # full connected layers
        shared_branchs = []
        for i in range(stacked_convs):
            chn = in_channels if i == 0 else fc_hidden_dim
            branch = ConvModule(
                in_channels=chn,
                out_channels=fc_hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d'),
                act_cfg=dict(type='ReLU')
            )
            shared_branchs.append(branch)
        self.shared_branchs = nn.Sequential(*shared_branchs)

        # task specific branch
        self.cls_layer = nn.Conv1d(in_channels=fc_hidden_dim,
                                out_channels=1,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True,
                                )

        self.num_control_points = order + 1
        self.reg_layer = nn.Conv1d(in_channels=fc_hidden_dim,
                                out_channels=self.num_control_points * 2,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True
                                )

    def init_weights(self):
            """Initialize weights of the head."""
            pass

    def forward(self, x, **kwargs):
        """Forward function for inference mode.
        Args:
            x (list[Tensor]): Features from neck.
            f (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of prediction results.
        """
        batch_size, _, H, W = x.size()
        self.avg_pool = nn.AvgPool2d((H, 1), stride=1, padding=0)
        x = self.avg_pool(x).squeeze(dim=2) # (B, C, W)

        # shared fc branch
        x = self.shared_branchs(x) # (B, C, W)
        # classification branch
        logits = self.cls_layer(x) # (B, 1, W)
        logits = logits.permute(0, 2, 1).contiguous() # (B, W, 1)
        # regression branch
        reg = self.reg_layer(x) # (B, 2 * num_control_points, W)
        reg = reg.permute(0, 2, 1).contiguous() # (B, W, 2 * num_control_points)
        reg = reg.view(batch_size, W, self.num_control_points, 2) # (B, W, num_control_points, 2)

        pred_dict = {
            'cls_logits': logits,    # (B, W, 1)
            'pred_control_points': reg,   # (B, W, 4, 2)
        }
        return pred_dict

    def loss(self, out_dict, img_metas):
        """Compute losses of the head.
        Args:
            out_dict (Dict[torch.Trnsor]): A dictionary of prediction results, including:
                - cls_logits (Tensor): Classification logits, shape (B, W, 1).
                - pred_control_points (Tensor): Predicted control points, shape (B, W, 4, 2).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = len(img_metas)
        device = out_dict["predictions"]["cls_logits"].device
        cls_loss = torch.tensor(0.0).to(device)
        dist_loss = torch.tensor(0.0).to(device)

        for b, img_meta in enumerate(img_metas):
            # get prediction results
            pred_dict = {
                k: v[b] for k, v in out_dict["predictions"].items()
            }
            cls_pred = pred_dict['cls_logits'] # (W, 1)
            reg_pred = pred_dict['pred_control_points'] # (W, 4, 2)
            # get target results
            gt_lanes = img_meta['gt_lanes'].clone().to(device) # (N_lanes, 4, 2)
            cls_target = torch.zeros_like(cls_pred) # (W, 1)

            # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
            if gt_lanes.size(0) == 0:
                cls_loss = (
                    cls_loss + self.loss_cls(cls_pred, cls_target).sum()
                )
                continue

            with torch.no_grad():
                (
                    matched_row_inds,
                    matched_col_inds,
                ) = self.assigner.assign(
                    pred_dict, gt_lanes.clone(), img_meta
                )
            
            # classification loss
            cls_target[matched_row_inds, :] = 1
            cls_loss = self.loss_cls(cls_pred, cls_target)
            # regression loss
            pred_control_points = reg_pred[matched_row_inds] # (N_matched, 4, 2)
            gt_control_points = gt_lanes[matched_col_inds] # (N_matched, 4, 2)
            pred_sample_points = self.bezier_curve.get_sample_points(
                pred_control_points, num_sample_points=self.num_sample_points
            )
            gt_sample_points = self.bezier_curve.get_sample_points(
                gt_control_points, num_sample_points=self.num_sample_points
            )
            dist_loss = (
                dist_loss + self.loss_dist(pred_sample_points, gt_sample_points).mean()
            )

        cls_loss = cls_loss / batch_size
        dist_loss = dist_loss / batch_size
        loss_dict = {
            "loss_cls": cls_loss,
            "loss_dist": dist_loss,
        }

        if self.loss_seg:
            tgt_masks = np.array([img_meta["gt_masks"].data[0] for img_meta in img_metas])
            tgt_masks = torch.tensor(tgt_masks).long().to(device) # (B, img_H, img_W)
            pred_masks = F.interpolate(
                out_dict["seg"], mode="bilinear", align_corners=False,
                size=[tgt_masks.shape[1], tgt_masks.shape[2]], 
            ) # (B, n, H, W) -> (B, n, img_H, img_W)
            loss_dict["loss_seg"] = self.loss_seg(pred_masks, tgt_masks)

        return loss_dict

    def forward_train(self, x, f, img_metas, **kwargs):
        """Forward function for training mode.
        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        predictions = self(x)
        out_dict = {"predictions": predictions}
        if self.loss_seg:
            out_dict["seg"] = self.forward_seg(f)

        losses = self.loss(out_dict, img_metas)
        return losses
    
    def forward_seg(self, x):
        """Forward function for training mode.
        Args:
            x (list[torch.tensor]): Features from backbone.
        Returns:
            torch.tensor: segmentation maps, shape (B, C, H, W), where
            B: batch size, C: segmentation channels, H and W: the largest feature's spatial shape.
        """
        x = self.seg_conv(x)
        x = self.seg_conv_out(x)
        return x

    def get_lanes(self, pred_dict, as_lane=True):
        """Get lanes from prediction results.
        Args:
            pred_dict (dict): A dictionary of prediction results.
            as_lane (bool): Whether to return lanes as a list of points or as a list of Bezier curves.
        Returns:
            list: A list of lanes.
        """
        assert (
            len(pred_dict["cls_logits"]) == 1
        ), "Only single-image prediction is available!"
        # filter out the conf lower than conf threshold
        scores = pred_dict["cls_logits"].squeeze() # (W)
        scores = scores.sigmoid() # (W)
        existences = scores > self.test_cfg.conf_threshold

        pred_control_points = pred_dict["pred_control_points"].squeeze(dim=0) # (W, 4, 2)
        num_pred = pred_control_points.shape[0]

        if self.test_cfg.window_size > 0:
            _, max_indices = F.max_pool1d(
                scores.unsqueeze(0).unsqueeze(0).contiguous(),
                kernel_size=self.test_cfg.window_size,
                stride=1,
                padding=(self.test_cfg.window_size - 1) // 2,
                return_indices=True
            ) # (1, 1, W)
            max_indices = max_indices.squeeze(dim=1) # (1, W)
            indices = torch.arange(0, num_pred, dtype=scores.dtype, 
                device=scores.device).unsqueeze(dim=0).expand_as(max_indices) 
            local_maximas = (max_indices == indices) # (B, W)
            existences = existences * local_maximas

        valid_score = scores * existences[0]
        sorted_score, sorted_indices = torch.sort(valid_score, dim=0, descending=True)
        valid_indices = torch.nonzero(sorted_score, as_tuple=True)[0][:self.test_cfg.max_num_lanes]

        keep_index = sorted_indices[valid_indices] # (N_lanes, )
        scores = scores[keep_index] # (N_lanes, )
        pred_control_points = pred_control_points[keep_index] # (N_lanes, 4, 2)

        if len(keep_index) == 0:
            return [], []
        
        preds = self.predictions_to_lanes(scores, pred_control_points, as_lane)

        return preds, scores

    def predictions_to_lanes(self, scores, pred_control_points, as_lane=True):
        """Convert predictions to lanes.
        Args:
            pred_control_points (torch.Tensor): Predicted control points, shape (N_lanes, 4, 2).
            scores (torch.Tensor): Predicted scores, shape (N_lanes, ).
            as_lane (bool): Whether to return lanes as a list of points or as a list of Bezier curves.
        Returns:
            list: A list of lanes.
        """
        dataset = self.test_cfg.get("dataset", None)

        lanes = []
        for score, control_point in zip(scores, pred_control_points):
            # score: (1, )
            # control_point: (4, 2)
            score = score.detach().cpu().numpy()
            control_point = control_point.detach().cpu().numpy()

            if dataset == 'tusimple':
                ppl = 56
                gap = 10
                bezier_threshold = 5.0 / self.test_cfg.ori_img_h
                h_samples = np.array(
                    [1.0 - (ppl - i) * gap / self.test_cfg.ori_img_h for i in range(ppl)], dtype=np.float32
                )   # (56, )

                sample_point = self.bezier_curve.get_sample_points(
                    control_points_matrix=control_point,
                    num_sample_points=self.test_cfg.ori_img_h)  # (N_sample_points-720, 2)  2: (x, y)

                ys = (
                    sample_point[:, 1] * (self.test_cfg.ori_img_h - self.test_cfg.cut_height)
                     + self.test_cfg.cut_height
                ) / self.test_cfg.ori_img_h   # (720, )
                dis = np.abs(h_samples.reshape(ppl, -1) - ys)    # (56, 720)
                idx = np.argmin(dis, axis=-1)  # (56, )
                temp = []
                for i in range(ppl):
                    h = self.test_cfg.ori_img_h - (ppl - i) * gap
                    if dis[i][idx[i]] > bezier_threshold or sample_point[idx[i]][0] > 1 \
                            or sample_point[idx[i]][0] < 0:
                        temp.append([-2, h])
                    else:
                        temp.append([sample_point[idx[i]][0] * self.test_cfg.ori_img_w, h])
                temp = np.array(temp, dtype=np.float32)
                lanes.append(temp)
            else:
                sample_point = self.bezier_curve.get_sample_points(
                    control_points_matrix=control_point,
                    num_sample_points=self.test_cfg['num_sample_points'])      # (N_sample_points, 2)  2: (x, y)

                lane_xs = sample_point[:, 0]      # 由上向下
                lane_ys = sample_point[:, 1]

                x_mask = np.logical_and(lane_xs >= 0, lane_xs < 1)
                y_mask = np.logical_and(lane_ys >= 0, lane_ys < 1)
                mask = np.logical_and(x_mask, y_mask)

                lane_xs = lane_xs[mask]
                lane_ys = lane_ys[mask]
                lane_ys = (
                    lane_ys * (self.test_cfg.ori_img_h - self.test_cfg.cut_height) 
                    + self.test_cfg.cut_height
                ) / self.test_cfg.ori_img_h
                if len(lane_xs) <= 1:
                    continue
                points = np.stack((lane_xs, lane_ys), axis=1)  # (N_sample_points, 2)  normalized

                points = sorted(points, key=lambda x: x[1])
                filtered_points = []
                used = set()
                for p in points:
                    if p[1] not in used:
                        filtered_points.append(p)
                        used.add(p[1])
                points = np.array(filtered_points)
                if as_lane:
                    lane = Lane(points=points,
                                metadata={
                                    'conf': score,
                                })
                else:
                    lane = points
                lanes.append(lane)
        return lanes

    def simple_test(self, feats):
        """Test without augmentation."""
        pred_dict = self(feats)
        lanes, scores = self.get_lanes(pred_dict, as_lane=self.test_cfg.as_lane)
        result_dict = {
            "lanes": lanes,
            "scores": scores,
        }
        return result_dict