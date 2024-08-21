"""
Alaug interface for albumentations transformations.
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/pipelines/alaug.py
"""
import collections
import copy

import albumentations as al
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module
class Alaug(object):
    def __init__(self, transforms, cut_y_duplicated=False, need_resorted=False):
        assert isinstance(transforms, collections.abc.Sequence)
        # init as None
        self.__augmentor = None
        # put transforms in a list
        self.transforms = []
        self.keypoint_params = None
        self.cut_y_duplicated = cut_y_duplicated
        self.need_resorted = need_resorted

        for transform in transforms:
            if isinstance(transform, dict):
                if transform["type"] == "Compose":
                    self.get_al_params(transform["params"])
                else:
                    transform = self.build_transforms(transform)
                    if transform is not None:
                        self.transforms.append(transform)
            else:
                raise TypeError("transform must be a dict")
        self.build()

    def get_al_params(self, compose):
        if compose["keypoints"]:
            self.keypoint_params = al.KeypointParams(
                format="xy", remove_invisible=False
            )

    def build_transforms(self, transform):
        if transform["type"] == "OneOf":
            transforms = transform["transforms"]
            choices = []
            for t in transforms:
                parmas = {
                    key: value for key, value in t.items() if key != "type"
                }
                choice = getattr(al, t["type"])(**parmas)
                choices.append(choice)
            return getattr(al, "OneOf")(transforms=choices, p=transform["p"])

        parmas = {
            key: value for key, value in transform.items() if key != "type"
        }
        return getattr(al, transform["type"])(**parmas)

    def build(self):
        if len(self.transforms) == 0:
            return
        self.__augmentor = al.Compose(
            self.transforms,
            keypoint_params=self.keypoint_params,
        )

    def cal_sum_list(self, itmes, index):
        sum = 0
        for i in range(index):
            sum += itmes[i]
        return sum

    def is_sorted(self, points):
        for lane in points:
            lane_y = np.array([coord[1] for coord in lane])
            lane_x = np.array([coord[0] for coord in lane])
            # check if y-coordinates are sorted in ascending order
            diff = np.diff(lane_y)
            if not np.all(diff > 0):
                # Find the indices where the condition is not met
                non_increasing_indices = np.where(diff <= 0.)[0]
                for idx in non_increasing_indices:
                    print(f"Element at index {idx + 1} is not strictly greater than the previous element:")
                    print(f"Previous coordinate: ({lane_x[idx]}, {lane_y[idx]})")
                    print(f"Current coordinate: ({lane_x[idx + 1]}, {lane_y[idx + 1]})")
                return False
        return True

    def __call__(self, data):
        data_org = copy.deepcopy(data)
        for i in range(30):
            # Duplicate points exist for VIL100
            # print(f"begin augmentation {i+1} {self.is_sorted(data['gt_points'])}")
            data_aug = self.aug(data)
            data = copy.deepcopy(data_org)
            if self.cut_y_duplicated: 
                # avoid lane sampling errors for sharp curve lanes
                data_aug["gt_points"] = [list({point[1]: point for point in lane}.values()) for lane in data_aug["gt_points"]]
            if self.need_resorted: # augmentation may change the order of lanes
                # sort lanes by their y-coordinates in ascending order for interpolation
                data_aug["gt_points"] = [sorted(lane, key=lambda x: x[1]) for lane in data_aug["gt_points"]]
            if self.is_sorted(data_aug["gt_points"]):
                return data_aug
        raise ValueError("Cannot find a valid result of augmentation!")

    def aug(self, data):
        if self.__augmentor is None:
            return data
        img = data["img"]
        masks = None
        if "gt_masks" in data:
            masks = data["gt_masks"]
        else:
            masks = None

        if "gt_points" in data:
            lane_points = data["gt_points"] # xy format, all lanes
            # run aug
            lane_points_index = []
            for pts in lane_points:
                lane_points_index.append(len(pts))

            points_val = []
            for pts in lane_points:
                num = len(pts)
                for i in range(num):
                    points_val.append(list(pts[i]))

        aug = self.__augmentor(
            image=img,
            keypoints=points_val,
            bboxes=None,
            mask=masks,
            bbox_labels=None,
        )
        data["img"] = aug["image"]
        data["img_shape"] = data["img"].shape
        if "gt_masks" in data:
            data["gt_masks"] = [np.array(aug["mask"])]
    
        if "gt_points" in data:
            start_idx = 0
            points = aug["keypoints"][start_idx:]
            kp_list = [[] for i in range(len(lane_points_index))]
            for lane_id in range(len(lane_points_index)):
                for i in range(lane_points_index[lane_id]):
                    kp_list[lane_id].append(points[self.cal_sum_list(lane_points_index, lane_id) + i])
            data["gt_points"] = kp_list

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
