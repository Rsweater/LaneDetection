import torch 
from mmdet.models.builder import LOSSES
 
 
def normalized_chamfer_distance(lane1, lane2, r=1e-6): 
    """
    Args: 
        lane1:(N_lanes, N_sample_points, 2)
        lane2:(N_lanes, N_sample_points, 2)
        r: r of lane
    """
    dist = torch.cdist(lane1,  lane2, p=2)  # 形状: (N1, N2, N_sample_points)
 
    # 计算lane1到lane2的最小距离, ()
    min_dist_lane1_to_lane2 = torch.min(dist,  dim=-1)[0]   
 
    # 计算CIoU 
    numerator = 2 * r - min_dist_lane1_to_lane2 
    denominator = 2 * r + min_dist_lane1_to_lane2 
    ciou_per_point = numerator / denominator 
    ciou = torch.mean(ciou_per_point,  dim=1) 
 
    return ciou 

def chamfer_loss(pc_pred, pc_gt, r=10): 
    """ 
    Args: 
        pc_pred: (N_lanes, N_sample_points, 2) 预测点集 
        pc_gt: (N_lanes, N_sample_points, 2) 真实点集 
    Returns: 
        loc_loss: (N_lanes, ) 每个样本的L_loc损失 
    """ 
    ciou_pred_to_gt = normalized_chamfer_distance(pc_pred, pc_gt, r=r) 
    ciou_gt_to_pred = normalized_chamfer_distance(pc_gt, pc_pred, r=r) 
 
    loc_loss_value = 1 - 0.5 * (ciou_pred_to_gt + ciou_gt_to_pred) 
    return loc_loss_value 
 

@LOSSES.register_module
class ChamferLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, r=10):
        super(ChamferLoss, self).__init__()
        self.loss_weight = loss_weight
        self.r = r

    def forward(self, pred, target):
        """
        Calculate the Chamfer loss based on the predictions and targets.

        Args:
            pred: lane predictions, shape: (N_lanes, N_sample_points, 2), relative coordinate.
            target: ground truth, shape: (N_lanes, N_sample_points, 2), relative coordinate.

        Returns:
            torch.Tensor: Chamfer loss value.

        Nl: number of lanes, Nr: number of rows.
        """
        assert (
            pred.shape == target.shape
        ), "pred and target should have the same shape, but got {} and {}".format(
            pred.shape, target.shape
        )

        return self.loss_weight * chamfer_loss(pred, target, r=self.r).mean()

 
