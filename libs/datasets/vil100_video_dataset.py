"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/culane_dataset.py
"""
import shutil
from pathlib import Path

import os
import json
import yaml
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from mmdet.datasets.builder import DATASETS
from mmdet.utils import get_root_logger
from tqdm import tqdm

from libs.datasets.metrics.vil100_metric import eval_predictions
from libs.datasets.pipelines import Compose


@DATASETS.register_module
class VIL100SeqDataset(Dataset):
    """VIL100 Dataset class."""

    def __init__(
        self,
        data_root,
        data_list,
        pipeline,
        diff_file=None,
        diff_thr=15,
        test_mode=True,
        y_step=1,
        time_window_size=3,
        sampled_frames=3, max_skip=5, increment=5, samples_per_video=10
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
        self.time_window_size = time_window_size
        self.sequence_idxs = self.get_sequence_idxs()

        dbfile = os.path.join(data_root, 'data', 'db_info.yaml')
        self.imgdir = os.path.join(data_root, 'JPEGImages')
        self.annodir = os.path.join(data_root, 'Annotations')

        # extract annotation information
        with open(dbfile, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']

            targetset = 'test' if self.test_mode else 'train'
            # targetset = 'training'
            self.info = db
            self.videos = [info['name'] for info in db if info['set'] == targetset]

        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames
        self.length = samples_per_video * len(self.videos)
        self.max_skip = max_skip
        self.increment = increment
        self.max_obj = 8

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        """
        Read and process the image through the transform pipeline for training and test.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        sequence_idxs = self.sequence_idxs[idx]
        sub_imgs_name = [self.img_infos[idx] for idx in sequence_idxs]
        imgs_path = [str(Path(self.img_prefix).joinpath(self.img_infos[idx])) for idx in sequence_idxs]
        imgs = [np.array(Image.open(img_path)) for img_path in imgs_path]
        target_idx = sequence_idxs[-1]
        target_img = imgs[-1]
        ori_shape = target_img.shape
        cut_height = target_img.shape[0] // 3
        img = target_img[cut_height:, ...]
        img_shape = crop_shape = img.shape
        crop_offset = [0, cut_height]
        results = dict(
            filename=imgs_path[-1],
            sub_img_name=sub_imgs_name[-1],
            img=target_img,
            imgs=imgs,
            img_shape=img_shape,
            ori_shape=ori_shape,
            crop_offset=crop_offset,
            crop_shape=crop_shape,
        )
        if not self.test_mode:
            kps, id_classes, id_instances = self.load_labels(target_idx, cut_height)
            results["gt_points"] = kps
            results["id_classes"] = id_classes
            results["id_instances"] = id_instances
            results["eval_shape"] = (
                crop_shape[0],
                crop_shape[1],
            )  # Used for LaneIoU calculation for VIL100 dataset.
            if self.mask_paths[0]:
                mask = self.load_mask(target_idx)
                mask = mask[cut_height:, :]
                assert mask.shape[:2] == crop_shape[:2]
                results["gt_masks"] = mask

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
        # # remove duplicated points in each lane
        # lanes = [list(set(lane)) for lane in lanes]  
        # # remove lanes with less than 2 points 
        # lanes = [lane for lane in lanes if len(lane) > 1] 
        # sort lanes by their y-coordinates in ascending order for interpolation
        # lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes] 
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
