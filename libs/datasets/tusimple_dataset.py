"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/culane_dataset.py
"""
import shutil
from pathlib import Path

import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from mmdet.datasets.builder import DATASETS
from mmdet.utils import get_root_logger
from tqdm import tqdm

from libs.datasets.metrics.tusimple_metric import LaneEval
from libs.datasets.pipelines import Compose


@DATASETS.register_module
class TuSimpleDataset(Dataset):
    """TuSimple Dataset class."""

    def __init__(
        self,
        data_root,
        data_list,
        pipeline,
        test_mode=True,
    ):
        """
        Args:
            data_root (str): Dataset root path.
            data_list (str): Dataset list file path.
            pipeline (List[mmcv.utils.config.ConfigDict]):
                Data transformation pipeline configs.
            test_mode (bool): Test flag.
        """
        self.img_prefix = data_root
        self.data_list = data_list
        self.test_mode = test_mode
        self.ori_w, self.ori_h = 1280, 720
        # read image list
        self.img_infos = self.parse_datalist(data_list)
        print(len(self.img_infos), "data are loaded")
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # build data pipeline
        self.pipeline = Compose(pipeline)
        self.result_dir = "tmp"
        self.list_path = data_list

    def parse_datalist(self, data_list):
        """
        Read image data list.
        Args:
            data_list (str): Data list file path.
        Returns:
            List[str]: List of image paths.
        """
        img_infos = []
        for anno_file in data_list:
            json_gt = [json.loads(line) for line in open(anno_file)]
            img_infos += json_gt
        return img_infos

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        sub_img_name = self.img_infos[idx]["raw_file"]
        img_name = str(Path(self.img_prefix).joinpath(sub_img_name))
        h_samples = self.img_infos[idx]['h_samples']
        img = np.array(Image.open(img_name))
        ori_shape = img.shape
        results = dict(
            filename=img_name,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=ori_shape,
            ori_shape=ori_shape,
            crop_offset=0,
            h_samples=h_samples,
            img_info=self.img_infos[idx]
        )

        if not self.test_mode:
            kps, id_classes, id_instances = self.load_labels(idx)
            results["gt_points"] = kps
            results["id_classes"] = id_classes
            results["id_instances"] = id_instances
            results["eval_shape"] = (
                720-160,
                1280,
            )
            results["gt_masks"] = self.load_mask(idx)
        return self.pipeline(results)

    def load_mask(self, idx):
        """
        Read a segmentation mask for training.
        Args:
            idx (int): Data index.
        Returns:
            numpy.ndarray: segmentation mask.
        """
        mask_path = self.img_infos[idx]["raw_file"].replace('clips',
                                                     'seg_label')[:-3] + 'png'
        maskname = str(Path(self.img_prefix).joinpath(mask_path))
        mask = np.array(Image.open(maskname))
        return mask

    def load_labels(self, idx, offset_y=0):
        """
        Read a ground-truth lane from an annotation file.
        Args:
            idx (int): Data index.
        Returns:
            List[list]: list of lane point lists.
            list: class id (=1) for lane instances.
            list: instance id (start from 1) for lane instances.
        """
        shapes = []
        for lane in self.img_infos[idx]['lanes']:
            coords = []
            for coord_x, coord_y in zip(lane, self.img_infos[idx]['h_samples']):
                if coord_x >= 0:
                    coord_x = float(coord_x)
                    coord_y = float(coord_y) - offset_y
                    coords.append((coord_x, coord_y))
            if len(coords) > 3:
                shapes.append(coords)
        id_classes = [1 for i in range(len(shapes))]
        id_instances = [i + 1 for i in range(len(shapes))]
        return shapes, id_classes, id_instances
        
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
        # save predictions to json
        dst_path = Path(self.result_dir).joinpath('tusimple_predictions.json')
        dst_path.parents[0].mkdir(parents=True, exist_ok=True)
        f_pr = open(dst_path,'w')
        for idx, result in tqdm(enumerate(results), desc="Processing results"):
            lanes = result["result"]["lanes"]
            sample_xs = self.get_prediction_string(lanes, self.img_infos[idx]['h_samples'])
            save_dict = dict(
                h_samples=self.img_infos[idx]['h_samples'],
                lanes=sample_xs,
                run_time=1000,
                raw_file=self.img_infos[idx]['raw_file'],
            )
            json.dump(save_dict, f_pr) 
            f_pr.write('\n')
        f_pr.close()
        print(f"\nWriting tusimple results to {dst_path}")

        results = LaneEval.bench_one_submit(dst_path, self.data_list[0])
        shutil.rmtree(self.result_dir)
        return results

    def get_prediction_string(self, lanes, h_samples):
        """
        Convert lane instance structure to prediction strings.
        Args:
            lanes (List[Lane]): List of lane instances in `Lane` structure.
        Returns:
            lanes_xs (str): Prediction x-coordinates for each lane instance.
        """

        ys = np.array(h_samples) / self.ori_h
        lanes_xs = []
        for lane in lanes:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.ori_w).astype(int)
            lane[invalid_mask] = -2
            lanes_xs.append(lane.tolist())

        return lanes_xs
