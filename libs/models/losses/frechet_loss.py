
import torch
from mmdet.models.builder import LOSSES

import torch

def frechet_dist(exp_data, num_data, p=2):
    r"""
    Compute the discrete Frechet distance using PyTorch

    Compute the Discrete Frechet Distance between two N-D curves according to
    [1]_. The Frechet distance has been defined as the walking dog problem.
    From Wikipedia: "In mathematics, the Frechet distance is a measure of
    similarity between curves that takes into account the location and
    ordering of the points along the curves. It is named after Maurice Frechet.
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance

    Parameters
    ----------
    exp_data : torch.Tensor
        Curve from your experimental data. exp_data is of (M, N) shape, where
        M is the number of data points, and N is the number of dimensions
    num_data : torch.Tensor
        Curve from your numerical data. num_data is of (P, N) shape, where P
        is the number of data points, and N is the number of dimensions
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use. Default is p=2 (Euclidean).
        The Manhattan distance is p=1.

    Returns
    -------
    df : float
        discrete Frechet distance

    References
    ----------
    .. [1] Thomas Eiter and Heikki Mannila. Computing discrete Frechet
        distance. Technical report, 1994.
        http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.937&rep=rep1&type=pdf

    Notes
    -----
    Your x locations of data points should be exp_data[:, 0], and the y
    locations of the data points should be exp_data[:, 1]. Same for num_data.

    Thanks to Arbel Amir for the issue, and Sen ZHANG for the iterative code
    https://github.com/cjekel/similarity_measures/issues/6

    Examples
    --------
    >>> # Generate random experimental data
    >>> x = torch.rand(100)
    >>> y = torch.rand(100)
    >>> exp_data = torch.zeros((100, 2))
    >>> exp_data[:, 0] = x
    >>> exp_data[:, 1] = y
    >>> # Generate random numerical data
    >>> x = torch.rand(100)
    >>> y = torch.rand(100)
    >>> num_data = torch.zeros((100, 2))
    >>> num_data[:, 0] = x
    >>> num_data[:, 1] = y
    >>> df = frechet_dist(exp_data, num_data)

    """
    n = exp_data.shape[0]
    m = num_data.shape[0]
    c = torch.cdist(exp_data, num_data, p=p)
    ca = torch.full((n, m), -1.0, dtype=exp_data.dtype, device=exp_data.device)
    ca[0, 0] = c[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], c[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], c[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]), c[i, j])
    return ca[n-1, m-1].item()



@LOSSES.register_module
class FrechetLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, p=2):
        super(FrechetLoss, self).__init__()
        self.loss_weight = loss_weight
        self.p = p

    def forward(self, pred, target):
        """
        Calculate the Frechet loss based on the predictions and targets.

        Args:
            pred: lane predictions, shape: (Nl, Nr), relative coordinate.
            target: ground truth, shape: (Nl, Nr), relative coordinate.

        Returns:
            torch.Tensor: Frechet loss value.

        Nl: number of lanes, Nr: number of rows.
        """
        assert (
            pred.shape == target.shape
        ), "pred and target should have the same shape, but got {} and {}".format(
            pred.shape, target.shape
        )

        loss = frechet_dist(pred, target, p=self.p)
        return self.loss_weight * loss