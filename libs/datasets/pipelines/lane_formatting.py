import math
import numpy as np
from scipy.interpolate import splprep, splev
from libs.core.lane import BezierCurve, sample_lane
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import Collect, to_tensor


@PIPELINES.register_module
class CollectBeizerInfo(Collect):
    """Adapted from mmLane"""
    def __init__(
        self,
        keys=None,
        meta_keys=None,
        interpolate=False,
        fix_endpoints=False,
        order=3,
        norm=True,
        num_sample_points=100,
    ):
        self.norm = norm
        self.keys = keys
        self.order = order
        self.meta_keys = meta_keys
        self.interpolate = interpolate
        self.fix_endpoints = fix_endpoints
        self.bezier_curve = BezierCurve(order=order)
        self.num_sample_points = num_sample_points

    def normalize_points(self, points, img_shape):
        h, w = img_shape
        points[..., 0] = points[..., 0] / w
        points[..., 1] = points[..., 1] / h
        return points

    def denormalize_points(self, points, img_shape):
        h, w = img_shape
        points[..., 0] = points[..., 0] * w
        points[..., 1] = points[..., 1] * h
        return points

    def get_valid_points(self, points):
        return (points[..., 0] >= 0) * (points[..., 0] < 1) * (points[..., 1] >= 0) * (points[..., 1] < 1)

    def cubic_bezier_curve_segment(self, control_points, sample_points):
        """ 
            控制点在做增广时可能会超出图像边界，因此使用DeCasteljau算法裁剪贝塞尔曲线段.
            具体做法是：通过判断采样点是否超出边界来确定t的边界值，即最小边界t0和最大边界t1.
            然后便可以通过文章公式（10）计算裁剪后的控制点坐标, 并以此作为label.
        """
        if len(control_points) == 0:
            return control_points

        N_lanes, N_sample_points = sample_points.shape[:-1]
        valid_mask = self.get_valid_points(sample_points)  # (N_lanes, num_sample_points)
        min_id = np.argmax(valid_mask + np.flip(np.arange(0, N_sample_points), axis=0), axis=-1)    # (N_lanes, )
        max_id = np.argmax(valid_mask + np.arange(0, N_sample_points), axis=-1)     # (N_lanes, )

        t = np.linspace(0.0, 1.0, num=N_sample_points, dtype=np.float32)
        t0 = t[min_id]
        t1 = t[max_id]

        # Generate transform matrix (old control points -> new control points = linear transform)
        u0 = 1 - t0  # (N_lanes, )
        u1 = 1 - t1  # (N_lanes, )
        transform_matrix_c = [np.stack([u0 ** (3 - i) * u1 ** i for i in range(4)], axis=-1),  # (N_lanes, 4)
                              np.stack([3 * t0 * u0 ** 2,
                                        2 * t0 * u0 * u1 + u0 ** 2 * t1,
                                        t0 * u1 ** 2 + 2 * u0 * u1 * t1,
                                        3 * t1 * u1 ** 2], axis=-1),
                              np.stack([3 * t0 ** 2 * u0,
                                        t0 ** 2 * u1 + 2 * t0 * t1 * u0,
                                        2 * t0 * t1 * u1 + t1 ** 2 * u0,
                                        3 * t1 ** 2 * u1], axis=-1),
                              np.stack([t0 ** (3 - i) * t1 ** i for i in range(4)], axis=-1)]
        transform_matrix_c = np.stack(transform_matrix_c, axis=-1)   # (N_lanes, 4, 4)
        res = np.matmul(transform_matrix_c, control_points)

        return res

    def cubic_bezier_curve_segmentv2(self, control_points, sample_points):
        """
            控制点在做增广时可能会超出图像边界，因此需要裁剪贝塞尔曲线段.
            具体做法是：通过判断采样点是否超出边界来找到有效的采样点，然后对这些采样点重新进行拟合.(高阶的DeCasteljau太难推)
        :param control_points: (N_lanes, N_controls, 2)
        :param sample_points: (N_lanes, num_sample_points, 2)
        :return:
            res: (N_lanes, N_controls, 2)
        """
        if len(control_points) == 0:
            return control_points,

        N_lanes, N_sample_points = sample_points.shape[:-1]
        valid_mask = self.get_valid_points(sample_points)  # (N_lanes, num_sample_points)
        min_id = np.argmax(np.flip(np.arange(0, N_sample_points), axis=0) * valid_mask, axis=-1)    # (N_lanes, )
        max_id = np.argmax(np.arange(0, N_sample_points) * valid_mask, axis=-1)     # (N_lanes, )

        control_points_list = []
        keep_indices = []
        for lane_id in range(N_lanes):
            cur_min_id = min_id[lane_id]
            cur_max_id = max_id[lane_id]
            if cur_max_id - cur_min_id < 2:
                continue
            if (cur_max_id - cur_min_id + 1) == N_sample_points:
                new_control_points = control_points[lane_id]
            else:
                valid_sample_points = sample_points[lane_id][cur_min_id:cur_max_id+1, :]      # (N_valid, 2)
                new_control_points = self.bezier_curve.get_control_points_with_fixed_endpoints(valid_sample_points)
            control_points_list.append(new_control_points)
            keep_indices.append(lane_id)

        if len(control_points_list):
            res = np.stack(control_points_list, axis=0)
        else:
            res = np.zeros((0, self.order+1, 2), dtype=np.float32)
        return res, keep_indices

    def intepolate(self, lane, n=100):
        # Spline interpolation of a lane. Used on the predictions
        x, y = lane[:, 0], lane[:, 1]
        tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))

        u = np.linspace(0., 1., n)
        return np.stack(splev(u, tck), axis=-1)

    def convert_targets(self, results):
        h, w = results["img_shape"][:2]
        old_lanes = results["gt_points"]
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 2, old_lanes)

        control_points_list = []
        for lane in old_lanes:
            lane = np.array(lane, dtype=np.float32)
            if self.interpolate:
                lane = self.intepolate(lane)
            if self.fix_endpoints:
                control_points = self.bezier_curve.get_control_points_with_fixed_endpoints(lane, to_list=True)
            else:
                control_points = self.bezier_curve.get_control_points(lane, to_list=True)
            control_points_list.append(control_points)

        # (Ng, num_control_points, 2)
        control_points = np.array(control_points_list, dtype=np.float32)   
        if len(control_points) > 0:
            if self.norm:
                control_points = self.normalize_points(control_points, (h, w))

            # (Ng, N_sample_points, 2)   2: (x, y)
            sample_points = self.bezier_curve.get_sample_points(control_points_matrix=control_points,
                                                                num_sample_points=self.num_sample_points)

            if self.order == 3:
                control_points = self.cubic_bezier_curve_segment(control_points, sample_points)
            else:
                control_points, keep_indices = self.cubic_bezier_curve_segmentv2(control_points, sample_points)
        
        results["gt_lanes"] = to_tensor(control_points)
        return results

    def __call__(self, results):
        data = {}
        img_meta = {}
        if "gt_lanes" in self.meta_keys:  # training
            results = self.convert_targets(results)
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_metas"] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data


@PIPELINES.register_module
class CollectCLRNet(Collect):
    def __init__(
        self,
        keys=None,
        meta_keys=None,
        max_lanes=4,
        extrapolate=True,
        num_points=72,
        img_w=800,
        img_h=320,
    ):
        self.keys = keys
        self.extrapolate = extrapolate
        self.meta_keys = meta_keys
        self.max_lanes = max_lanes
        self.n_offsets = num_points
        self.n_strips = num_points - 1
        self.strip_size = img_h / self.n_strips
        self.offsets_ys = np.arange(img_h, -1, -self.strip_size)
        self.img_w = img_w

    def convert_targets(self, results):
        old_lanes = results["gt_points"]
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 2, old_lanes)

        lanes = (
            np.ones((self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32)
            * -1e5
        )
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = sample_lane(
                    lane, self.offsets_ys, self.img_w, extrapolate=self.extrapolate
                )
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:  # to calculate theta
                continue
            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = (
                    math.atan(
                        i
                        * self.strip_size
                        / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)
                    )
                    / math.pi
                )
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            theta_far = sum(thetas) / len(thetas)

            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = (
                1 - len(xs_outside_image) / self.n_strips
            )  # y0, relative
            lanes[lane_idx, 3] = xs_inside_image[0]  # x0, absolute
            lanes[lane_idx, 4] = theta_far  # theta
            lanes[lane_idx, 5] = len(xs_inside_image)  # length
            lanes[lane_idx, 6 : 6 + len(all_xs)] = all_xs  # xs, absolute

        results["gt_lanes"] = to_tensor(lanes)
        return results

    def __call__(self, results):
        data = {}
        img_meta = {}
        if "gt_lanes" in self.meta_keys:  # training
            results = self.convert_targets(results)
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_metas"] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data
