import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, splev, splrep, splprep


class Lane:
    # Lane instance structure.
    # Adapted from:
    # https://github.com/lucastabelini/LaneATT/blob/main/lib/lane.py
    # Copyright (c) 2021 Lucas Tabelini
    def __init__(self, points=None, invalid_value=-2.0, metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(
            points[:, 1], points[:, 0], k=min(3, len(points) - 1)
        )
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return "[Lane]\n" + str(self.points) + "\n[/Lane]"

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs
    
    def to_array(self, sample_y, img_size):
        img_h, img_w = img_size
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                              axis=1)
        return lane

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration


def interp(points, n=50):
    """
    Adapted from:
    https://github.com/lucastabelini/LaneATT/blob/main/utils/culane_metric.py
    Copyright (c) 2021 Lucas Tabelini
    Args:
        points (List[tuple]): List of lane point tuples (x, y).
        n (int): number of interpolated points
    Returns:
        output (np.ndarray): Interpolated N lane points with shape (N, 2).
    """
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=8, k=min(3, len(points) - 1))

    u = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
    output = np.array(splev(u, tck)).T
    return output


def sample_lane(points, sample_ys, img_w, extrapolate=True):
    """
    Sample lane points on the horizontal grids.
    Adapted from:
    https://github.com/lucastabelini/LaneATT/blob/main/lib/datasets/lane_dataset.py

    Args:
        points (List[numpy.float64]): lane point coordinate list (length = Np * 2).
          The values are treated as [(x0, y0), (x1, y1), ...., (xp-1, yp-1)].
          y0 ~ yp-1 must be sorted in ascending order (y1 > y0).
        sample_ys (numpy.ndarray): shape (Nr,).
        img_w (int): image width.
        extrapolate (bool): Whether to extrapolate lane points to the bottom of the image.

    Returns:
        numpy.ndarray: x coordinates outside the image, shape (No,).
        numpy.ndarray: x coordinates inside the image, shape (Ni,).
    Np: number of input lane points, Nr: number of rows,
    No and Ni: number of x coordinates outside and inside image.
    """
    points = np.array(points)
    points = points[points[:, 1].argsort()[::-1]]  # sort points by y in descending order
    if not np.all(points[1:, 1] < points[:-1, 1]):
        print(points)
        raise Exception("Annotaion points have to be sorted")
    x, y = points[:, 0], points[:, 1]

    # interpolate points inside domain
    assert len(points) > 1
    interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
    domain_min_y = y.min()
    domain_max_y = y.max()
    mask_inside_domain = (sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)
    sample_ys_inside_domain = sample_ys[mask_inside_domain]
    assert len(sample_ys_inside_domain) > 0
    interp_xs = interp(sample_ys_inside_domain)

    # extrapolate lane to the bottom of the image with a straight line using the some points closest to the bottom
    if extrapolate:
        n_closest = max(len(points) // 5, 2)
        two_closest_points = points[:n_closest]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))
    else:
        all_xs = interp_xs

    # separate between inside and outside points
    inside_mask = (all_xs >= 0) & (all_xs < img_w)
    xs_inside_image = all_xs[inside_mask]
    xs_outside_image = all_xs[~inside_mask]
    return xs_outside_image, xs_inside_image

def interp_extrap(points, sample_ys, img_w):
    """
    Interpolate and extrapolate lane points on the horizontal grids.
    """
    points = np.array(points)
    if not np.all(points[1:, 1] > points[:-1, 1]):
        print(points)
        raise Exception("Annotaion points have to be sorted")
    x, y = points[:, 0], points[:, 1]
    assert len(points) > 1
    f = splrep(y, x, k=1, s=50) 
    """ Error:
    untimeWarning: The maximal number of iterations (20) allowed for finding smoothing
    spline with fp=s has been reached. Probable cause: s too small.
    """
    new_x_pts = splev(sample_ys, f)
    
    inside_mask = (new_x_pts >= 0) & (new_x_pts < img_w)
    xs_inside_image = new_x_pts[inside_mask]
    xs_outside_image = new_x_pts[~inside_mask]
    return xs_outside_image, xs_inside_image