import torch
from scipy.optimize import linear_sum_assignment
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core import AssignResult
from ..bezier_curve import BezierCurve



@BBOX_ASSIGNERS.register_module()
class BezierHungarianAssigner(BaseAssigner):
    def __init__(self, order=3, num_sample_points=100, alpha=0.8, window_size=0):
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.bezier_curve = BezierCurve(order=order)
        self.window_size = window_size

    def assign(self, pred_dict, gt_lanes, img_meta):
        """
        computes hungarian matching based on the costs, including cls_cost and curve sampling cost.
        Args:
            pred_dict (Dict[torch.Trnsor]): predictions predicted by each batch, including: 
                - cls_logits (Tensor): Classification logits, shape (Np, 1).
                - pred_control_points (Tensor): Predicted control points, shape (Np, 4, 2).
            gt_lanes (List[Tensor]): lane targets, shape: (Ng, ncp, 2).
            img_meta (dict): meta dict that includes per-image information such as image shape.
        Returns:
            matched_row_inds (Tensor): matched predictions, shape: (num_targets).
            matched_col_inds (Tensor): matched targets, shape: (num_targets).
        Np: number of priors (anchors), Ng: number of GT lanes, ncp: number of control points per lane.
        """
        Np = pred_dict['cls_logits'].shape[0]
        Ng = gt_lanes.shape[0]

        # 1. compute the costs
        # cls_cost
        cls_score = pred_dict['cls_logits'].sigmoid()    # (Np, n_cls=1)

        # Local maxima prior
        if self.window_size > 0:
            _, max_indices = torch.nn.functional.max_pool1d(cls_score.unsqueeze(0).permute(0, 2, 1),
                                                            kernel_size=self.window_size, stride=1,
                                                            padding=(self.window_size - 1) // 2, return_indices=True)
            max_indices = max_indices.squeeze()   # (Np, )
            indices = torch.arange(0, Np, dtype=cls_score.dtype, device=cls_score.device)
            local_maxima = (max_indices == indices)     # (Np)
        else:
            local_maxima = cls_score.new_ones((Np, ))

        cls_score = cls_score.squeeze(dim=1)     # (Np, )
        cls_cost = (local_maxima * cls_score).unsqueeze(dim=1).repeat(1, Ng)   # (Np, Ng)

        # curve sampling cost
        # (Np, N_sample_points, 2)
        pred_sample_points = self.bezier_curve.get_sample_points(control_points_matrix=pred_dict['pred_control_points'],
                                                                 num_sample_points=100)
        # (Ng, N_sample_points, 2)
        gt_sample_points = self.bezier_curve.get_sample_points(control_points_matrix=gt_lanes,
                                                               num_sample_points=100)

        # (Np, N_sample_points, 2) --> (Np, N_sample_points*2)
        pred_sample_points = pred_sample_points.flatten(start_dim=-2)
        # (Ng, N_sample_points, 2) --> (Ng, N_sample_points*2)
        gt_sample_points = gt_sample_points.flatten(start_dim=-2)
        reg_cost = 1 - torch.cdist(pred_sample_points, gt_sample_points, p=1) / self.num_sample_points  # (Nq, Ng)
        reg_cost = reg_cost.clamp(min=0, max=1)

        cost = -cls_cost ** (1 - self.alpha) * reg_cost ** self.alpha

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(cls_score.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(cls_score.device)

        return matched_row_inds, matched_col_inds
