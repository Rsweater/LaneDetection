"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/culane_dataset.py
"""
import shutil
from pathlib import Path

import cv2
import json
import numpy as np
from PIL import Image
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.utils import get_root_logger
from tqdm import tqdm

from libs.datasets.metrics.vil100_metric import eval_predictions
from libs.datasets.pipelines import Compose


@DATASETS.register_module
class VIL100Dataset(CustomDataset):
    """VIL100 Dataset class."""

    def __init__(
        self,
        data_root,
        data_list,
        pipeline,
        diff_file=None,
        diff_thr=15,
        test_mode=True,
        y_step=2,
    ):
        """
        Args:
            data_root (str): Dataset root path.
            data_list (str): Dataset list file path.
            pipeline (List[mmcv.utils.config.ConfigDict]):
                Data transformation pipeline configs.
            test_mode (bool): Test flag.
            y_step (int): Row interval (in the original image's y scale)
                to sample the predicted lanes for evaluation.

        """
        # self.logger = get_root_logger(log_level="INFO")
        self.img_prefix = data_root
        self.test_mode = test_mode
        # read image list
        self.diffs = (
            np.load(diff_file)["data"] if diff_file is not None else []
        )
        self.diff_thr = diff_thr
        (
            self.img_infos,
            self.annotations,
            self.mask_paths,
        ) = self.parse_datalist(data_list)
        print(len(self.img_infos), "data are loaded")
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # build data pipeline
        self.pipeline = Compose(pipeline)
        self.result_dir = "tmp"
        self.y_step = y_step

    def parse_datalist(self, data_list):
        """
        Read image data list.
        Args:
            data_list (str): Data list file path.
        Returns:
            List[str]: List of image paths.
        """
        print("Geting VIL100 dataset...")
        with open(data_list, "r") as img_list_file:
            img_infos = [
                img_name.strip()[1:] for img_name in img_list_file.readlines()
            ]
        if not self.test_mode:
            annotations = [
                img_path.replace("JPEGImages", "Json") + ".json" for img_path in img_infos
            ]
            mask_paths = [
                img_path[:-3].replace("JPEGImages", "Annotations") + "png" for img_path in img_infos
            ]
        else:
            annotations = ["testing..."]
            mask_paths = ["testing..."]
        return img_infos, annotations, mask_paths

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def prepare_train_img(self, idx):
        """
        Read and process the image through the transform pipeline for training.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        img_name = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
        sub_img_name = self.img_infos[idx]
        
        img_tmp = np.array(Image.open(img_name))
        ori_shape = img_tmp.shape
        cut_height = img_tmp.shape[0] // 3
        img = img_tmp[cut_height:, ...]
        img_shape = crop_shape = img.shape

        kps, id_classes, id_instances = self.load_labels(idx, cut_height)
        results = dict(
            filename=img_name,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=kps,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=img_shape,
            ori_shape=ori_shape,
            eval_shape=(
                crop_shape[0],
                crop_shape[1],
            ),  # Used for LaneIoU calculation. Static for CULane dataset.
            crop_shape=crop_shape,
            gt_masks=None,
        )
        if self.mask_paths[0]:
            mask = self.load_mask(idx)
            mask = mask[cut_height:, :]
            assert mask.shape[:2] == crop_shape[:2]
            results["gt_masks"] = mask

        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """
        Read and process the image through the transform pipeline for test.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        img_name = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
        sub_img_name = self.img_infos[idx]
        img_tmp = np.array(Image.open(img_name))
        ori_shape = img_tmp.shape
        cut_height = img_tmp.shape[0] // 3
        img = img_tmp[cut_height:, ...]
        img_shape = crop_shape = img.shape
        crop_offset = [0, cut_height]
        results = dict(
            filename=img_name,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=img_shape,
            ori_shape=ori_shape,
            crop_offset=crop_offset,
            crop_shape=crop_shape,
        )
        return self.pipeline(results)

    def load_mask(self, idx):
        """
        Read a segmentation mask for training.
        Args:
            idx (int): Data index.
        Returns:
            numpy.ndarray: segmentation mask.
        """
        maskname = str(Path(self.img_prefix).joinpath(self.mask_paths[idx]))
        mask = np.array(Image.open(maskname))
        return mask

    def load_labels(self, idx, cut_height=0):
        """
        Read a ground-truth lane from an annotation file.
        Args:
            idx (int): Data index.
        Returns:
            List[list]: list of lane point lists.
            list: class id (=1) for lane instances.
            list: instance id (start from 1) for lane instances.
        """
        anno_dir = str(Path(self.img_prefix).joinpath(self.annotations[idx]))
        
        with open(anno_dir, "r") as anno_file:
            lanes = [
                lane["points"] for lane in json.load(anno_file)["annotations"]["lane"]
            ]
        # point of lane, y of lane, y+offset_y
        # lanes: [[(x_00,y0), (x_01,y1), ...], [(x_10,y0), (x_11,y1), ...], ...]
        lanes = [[(point[0], point[1] - cut_height) for point in lane] for lane in lanes]
        # remove duplicated points in each lane
        lanes = [list(set(lane)) for lane in lanes]  
        # remove lanes with less than 2 points 
        lanes = [lane for lane in lanes if len(lane) > 1] 
        # sort lanes by their y-coordinates in ascending order for interpolation
        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes] 

        id_classes = [1 for i in range(len(lanes))]
        id_instances = [i + 1 for i in range(len(lanes))]
        return lanes, id_classes, id_instances

    def evaluate(self, results, metric="F1", logger=None):
        """
        Write prediction to txt files for evaluation and
        evaluate them with labels.
        Args:
            results (List[dict]): All inference results containing:
                result (dict): contains 'lanes' and 'scores'.
                meta (dict): contains meta information.
            metric (str): Metric type to evaluate. (not used)
        Returns:
            dict: Evaluation result dict containing
                F1, precision, recall, etc. on the specified IoU thresholds.

        """
        for result in tqdm(results):
            lanes = result["result"]["lanes"]
            ori_shape = result["meta"]["ori_shape"]
            dst_path = (
                Path(self.result_dir)
                .joinpath(result["meta"]["sub_img_name"].replace(
                        "JPEGImages", "Json").replace(
                        ".jpg", ".jpg.json"))
            )
            dst_path.parents[0].mkdir(parents=True, exist_ok=True)
            lanes = self.get_prediction_as_points(lanes, ori_shape)
            # save output in my format
            # if len(lanes) > 0:
            with open(dst_path, "w") as out_file:
                output = {
                    "ori_shape": ori_shape,
                    "lanes": lanes,
                }
                json.dump(output, out_file)
        # TODO: 修改eval_predictions函数，使其支持VIL100数据集
        results = eval_predictions(
            self.result_dir,
            self.img_prefix,
            self.img_infos,
            # self.annotations,
            logger=get_root_logger(log_level="INFO"),
        )
        shutil.rmtree(self.result_dir)
        return results

    def get_prediction_as_points(self, pred, ori_shape):
        ys = np.arange(0, ori_shape[0], self.y_step) / ori_shape[0]
        lanes = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * ori_shape[1]
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * ori_shape[0]
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane = list(zip(lane_xs, lane_ys))
            if len(lane) > 1:
                lanes.append(lane)

        return lanes
