import os
import math
import numpy as np
from scipy.interpolate import splprep, splrep, splev
from shapely.geometry import Polygon, LineString, MultiLineString
from libs.core.lane import BezierCurve, sample_lane
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import Collect, to_tensor

@PIPELINES.register_module()
class VideoLaneFormatBundle:
    """Video lane formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'imgs' in results:
            imgs = results['imgs']
            if self.img_to_float is True and img[0].dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                imgs = imgs.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['imgs'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'

def get_vil100_img_clip(img_path):
    clip_dir = os.path.dirname(img_path)
    frame_name = os.path.basename(img_path)
    frame_number = int(frame_name.split(".")[0])
    return clip_dir, frame_number

def image_sequence(filename, time_window_size):
    clips = {}
    for anno_idx, img_path in enumerate(filename):
        img_clip, frame_idx = get_vil100_img_clip(img_path)
        if img_clip not in clips:
            clips[img_clip] = [(frame_idx, anno_idx)]
        else:
            clips[img_clip].append((frame_idx, anno_idx))
    sequences_idxs = []
    for clip in clips:
        clip_data = sorted(clips[clip], key=lambda x: x[0])
        clip_anno_idxs = [item[1] for item in clip_data]
        for i in range(time_window_size - 1):
            sequences_idxs.append([clip_anno_idxs[i]] * time_window_size)
        for i in range(len(clip_anno_idxs) - time_window_size + 1):
            sequences_idxs.append(clip_anno_idxs[i : i + time_window_size])
    return sequences_idxs


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
        img_h, img_w = results["img_shape"][:2]
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
                control_points = self.normalize_points(control_points, (img_h, img_w))

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
class CollectCLRInfo(Collect):
    def __init__(
        self,
        keys=None,
        meta_keys=None,
        max_lanes=4,
        extrapolate=True,
        num_points=72,
    ):
        self.keys = keys
        self.extrapolate = extrapolate
        self.meta_keys = meta_keys
        self.max_lanes = max_lanes
        self.n_offsets = num_points
        self.n_strips = num_points - 1

    def convert_targets(self, results):
        img_h, img_w = results['img_shape'][:2]
        strip_size = img_h / self.n_strips
        offsets_ys = np.arange(img_h, -1, -strip_size)
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
                    lane, offsets_ys, img_w, extrapolate=self.extrapolate
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
                        * strip_size
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


@PIPELINES.register_module
class CollectGAinfo(Collect):
    def __init__(self,
                 keys=None,
                 meta_keys=None,
                 radius=2,
                 max_lanes=4,
                 fpn_cfg=dict(
                     hm_idx=0,
                     fpn_down_sacle=[8, 16, 32],
                     sample_per_lane=[41, 21, 11]
                 )):
        self.keys = keys
        self.radius = radius
        self.max_lanes = max_lanes
        self.meta_keys = meta_keys
        self.hm_idx = fpn_cfg.get('hm_idx')
        self.fpn_down_scale = fpn_cfg.get('fpn_down_scale')
        self.sample_per_lane = fpn_cfg.get('sample_per_lane')
        self.hm_down_scale = self.fpn_down_scale[self.hm_idx]
        self.fpn_layer_num = len(self.fpn_down_scale)

    def ploy_fitting_cube(self, line, h, w, sample_num=100):
        line_coords = np.array(line).reshape((-1, 2))  # (N, 2)
        # The y-coordinates are arranged from small to large, 
        # meaning the lane goes from the top to the bottom of the image.
        line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))

        lane_x = line_coords[:, 0]
        lane_y = line_coords[:, 1]

        if len(lane_y) < 2:
            return None
        new_y = np.linspace(max(lane_y[0], 0), min(lane_y[-1], h), sample_num)

        sety = set()
        nX, nY = [], []
        for (x, y) in zip(lane_x, lane_y):
            if y in sety:
                continue
            sety.add(x)
            nX.append(x)
            nY.append(y)
        if len(nY) < 2:
            return None

        if len(nY) > 3:
            ipo3 = splrep(nY, nX, k=3)
            ix3 = splev(new_y, ipo3)
        else:
            ipo3 = splrep(nY, nX, k=1)
            ix3 = splev(new_y, ipo3)
        return np.stack((ix3, new_y), axis=-1)
    
    def downscale_lane(self, lane, downscale):
        downscale_lane = []
        for point in lane:
            downscale_lane.append((point[0] / downscale, point[1] / downscale))
        return downscale_lane
    
    def clip_line(self, pts, h, w):
        pts_x = np.clip(pts[:, 0], 0, w - 1)[:, None]
        pts_y = np.clip(pts[:, 1], 0, h - 1)[:, None]
        return np.concatenate([pts_x, pts_y], axis=-1)
    
    def clamp_line(self, line, box, min_length=0):
        left, top, right, bottom = box
        loss_box = Polygon([[left, top], [right, top], [right, bottom],
                            [left, bottom]])
        line_coords = np.array(line).reshape((-1, 2))
        if line_coords.shape[0] < 2:
            return None
        try:
            line_string = LineString(line_coords)
            I = line_string.intersection(loss_box)
            if I.is_empty:
                return None
            if I.length < min_length:
                return None
            if isinstance(I, LineString):
                pts = list(I.coords)
                return pts
            elif isinstance(I, MultiLineString):
                pts = []
                Istrings = list(I)
                for Istring in Istrings:
                    pts += list(Istring.coords)
                return pts
        except:
            return None

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        """
        Args:
            heatmap: (hm_h, hm_w)   1/16
            center: (x0', y0'),  1/16
            radius: float
        Returns:
            heatmap: (hm_h, hm_w)
        """
        def gaussian2D(shape, sigma=1):
            """
            Args:
                shape: (diameter=2*r+1, diameter=2*r+1)
            Returns:
                h: (diameter, diameter)
            """
            m, n = [(ss - 1.) / 2. for ss in shape]
            # y: (1, diameter)    x: (diameter, 1)
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            # (diameter, diameter)
            h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            return h

        diameter = 2 * radius + 1
        # (diameter, diameter)
        gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def convert_targets(self, results):
        img_h, img_w = results["img_shape"][:2]
        gt_lanes = results["gt_points"] # List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]

        # traverse the FPN levels 
        # to find the corresponding sampling points 
        # of each lane line on the feature map at that level.
        gt_hm_lanes = {}
        for l in range(self.fpn_layer_num):
            lane_points = []
            fpn_down_scale = self.fpn_down_scale[l]
            f_h = img_h // fpn_down_scale
            f_w = img_w // fpn_down_scale
            for i, lane in enumerate(gt_lanes):
                # downscaled lane: List[(x0, y0), (x1, y1), ...]
                lane = self.downscale_lane(lane, downscale=self.fpn_down_scale[l])
                # Arrange the lane from the bottom to the top of the image (y from large to small).
                lane = sorted(lane, key=lambda x: x[1], reverse=True)
                pts = self.ploy_fitting_cube(lane, f_h, f_w, self.sample_per_lane[l])  # (N_sample, 2)
                if pts is not None:
                    pts_f = self.clip_line(pts, f_h, f_w)  # (N_sample, 2)
                    pts = np.int32(pts_f)
                    lane_points.append(pts[:, ::-1])  # (N_sample, 2)   2： (y, x)

            # (max_lane_num,  N_sample, 2)  2： (y, x)
            lane_points_align = -1 * np.ones((self.max_lanes, self.sample_per_lane[l], 2))
            if len(lane_points) != 0:
                lane_points_align[:len(lane_points)] = np.stack(lane_points, axis=0)    # (num_lanes, N_sample, 2)
            gt_hm_lanes[l] = lane_points_align
        
        # Generate heatmap and offset maps.
        # gt initialization
        hm_h = img_h // self.hm_down_scale
        hm_w = img_w // self.hm_down_scale
        kpts_hm = np.zeros((1, hm_h, hm_w), np.float32)     # (1, hm_H, hm_W)
        kp_offset = np.zeros((2, hm_h, hm_w), np.float32)   # (2, hm_H, hm_W)
        sp_offset = np.zeros((2, hm_h, hm_w), np.float32)   # (2, hm_H, hm_W)  key points -> start points
        kp_offset_mask = np.zeros((2, hm_h, hm_w), np.float32)  # (2, hm_H, hm_W)
        sp_offset_mask = np.zeros((2, hm_h, hm_w), np.float32)  # (2, hm_H, hm_W)

        start_points = []
        for lane in gt_lanes:
            # downscaled lane: List[(x0, y0), (x1, y1), ...]
            lane = self.downscale_lane(lane, downscale=self.hm_down_scale)
            if len(lane) < 2:
                continue

            # (N_sample=int(360 / self.hm_down_scale), 2)
            lane = self.ploy_fitting_cube(lane, hm_h, hm_w, int(360 / self.hm_down_scale))
            if lane is None:
                continue

            # Arrange the lane from the bottom to the top of the image (y from large to small).
            lane = sorted(lane, key=lambda x: x[1], reverse=True)
            lane = self.clamp_line(lane, box=[0, 0, hm_w - 1, hm_h - 1], min_length=1)
            if lane is None:
                continue

            start_point, end_point = lane[0], lane[-1]    # (2, ),  (2, )
            start_points.append(start_point)
            for pt in lane:
                pt_int = (int(pt[0]), int(pt[1]))   # (x, y)
                # Draw heatmap
                self.draw_umich_gaussian(kpts_hm[0], pt_int, radius=self.radius)

                # Generate key point offset map，compensation for quantization error.
                offset_x = pt[0] - pt_int[0]
                offset_y = pt[1] - pt_int[1]
                kp_offset[0, pt_int[1], pt_int[0]] = offset_x
                kp_offset[1, pt_int[1], pt_int[0]] = offset_y
                # Generate kp_offset_mask, Only the position of the key point is 1.
                kp_offset_mask[:, pt_int[1], pt_int[0]] = 1

                # Offset from the sample point to the start point.
                offset_x = start_point[0] - pt_int[0]
                offset_y = start_point[1] - pt_int[1]
                sp_offset[0, pt_int[1], pt_int[0]] = offset_x
                sp_offset[1, pt_int[1], pt_int[0]] = offset_y
                # Generate sample point offset mask, Only the position of the key point is 1.
                sp_offset_mask[:, pt_int[1], pt_int[0]] = 1

        results['gt_hm_lanes'] = gt_hm_lanes
        results['gt_kpts_hm'] = kpts_hm
        results['gt_kp_offset'] = kp_offset
        results['gt_sp_offset'] = sp_offset
        results['kp_offset_mask'] = kp_offset_mask
        results['sp_offset_mask'] = sp_offset_mask

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