import torch 
import torch.nn.functional  as F 
from mmdet.models.builder import LOSSES
 
 
def L_len(len_pred, len_gt): 
    ratio = len_pred / len_gt 
    return F.l1_loss(ratio, torch.ones_like(ratio))  

def lane_length(traj): 
    """ 
    Args: 
        traj: (B, N, 2) 点集（Batch, Num_points, 2 
    Returns: 
        lengths: (B,) 每个样本的车道线长度 
    """ 
    # 计算相邻点之间的距离 
    diff = traj[:, 1:] - traj[:, :-1] 
    dist = torch.norm(diff,  dim=2)  # (B, N-1) 
 
    # 求和得到每个样本的车道线长度 
    lengths = torch.sum(dist,  dim=1) 
    return lengths 

def length_loss(pred, gt):
    len_pred = lane_length(pred)
    len_gt = lane_length(gt)
    ratio = len_pred / len_gt
    return F.l1_loss(ratio, torch.ones_like(ratio))

def endpoint_loss(pred_traj, gt_traj):
    """
    计算端点损失函数 
    :param pred_traj: 预测轨迹，形状为 (B, N, 2)
    :param gt_traj:    真实轨迹，形状为 (B, N, 2)
    :return: 标量损失值 
    """
    # 提取第一个点和最后一个点 
    p1_pred = pred_traj[:, 0, :]   # (B, 2)
    p1_gt = gt_traj[:, 0, :]       # (B, 2)
    p_end_pred = pred_traj[:, -1, :]  # (B, 2)
    p_end_gt = gt_traj[:, -1, :]      # (B, 2)
 
    # 计算L2损失（默认使用均值平均）
    loss_p1 = F.mse_loss(p1_pred,  p1_gt)
    loss_p_end = F.mse_loss(p_end_pred,  p_end_gt)
 
    # 平均两部分损失 
    total_loss = (loss_p1 + loss_p_end) * 0.5 
    return total_loss


@LOSSES.register_module
class LengthLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0):
        super(LengthLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N, 2) 预测点集
            target: (B, N, 2) 真实点集
        Returns:
            loss: 车道线长度损失
        """
        assert (
            pred.shape == target.shape
        ), "pred and target should have the same shape, but got {} and {}".format(
            pred.shape, target.shape
        )

        return self.loss_weight * length_loss(pred, target).mean()


@LOSSES.register_module
class EndpointLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0):
        super(EndpointLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N, D) 预测点集
            target: (B, N, D) 真实点集
        Returns:
            loss: 端点损失
        """
        assert (
            pred.shape == target.shape
        ), "pred and target should have the same shape, but got {} and {}".format(
            pred.shape, target.shape
        )

        return self.loss_weight * endpoint_loss(pred, target).mean()

