"""
Adapted from:
https://github.com/lucastabelini/LaneATT/blob/main/utils/culane_metric.py
Copyright (c) 2021 Lucas Tabelini
"""

import os
import json
from pathlib import Path
from functools import partial

import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.optimize import linear_sum_assignment

from mmcv.utils import print_log

from libs.core.lane.visualizer import draw_lane
from libs.core.lane.lane_utils import interp

def get_macro_measure(img_list, measures):
    sequences = {}
    for img_path, iou in zip(img_list, measures):
        sequence = os.path.dirname(img_path)
        if sequence in sequences:
            sequences[sequence].append(iou)
        else:
            sequences[sequence] = [iou]
    macro_measure = 0
    for sequence in sequences:
        macro_measure += np.mean(sequences[sequence])
    return macro_measure / len(sequences)

def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    """
    Args:
        xs (np.ndarray): Array containing lane coordinate arrays with different lengths.
        ys (np.ndarray): Array containing lane coordinate arrays with different lengths.
        width (int): Lane drawing width to calculate IoU.
        img_shape (tuple): Image shape.
    Returns:
        ious (np.ndarray): IoU matrix.

    """
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]
    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def culane_metric(
    pred,
    anno,
    img_shape,
    width=30,
    iou_thresholds=[0.5],
):
    """
    Calculate the CULane metric for given pred and anno lanes of one image.
    Example of an IoU matrix and assignment:
     ious = [[0.85317694 0.         0.        ]
             [0.         0.49573853 0.        ]]
     (row_ind, col_ind) = (0, 0), (1, 1)
    Args:
        pred (List[List[tuple]]): Prediction result for one image.
        anno (List[List[tuple]]): Lane labels for one image.
        cat (str): Category name.
        width (int): Lane drawing width to calculate IoU.
        iou_thresholds (list): IoU thresholds for evaluation.
        img_shape (tuple): Image shape.
    Returns:
        results (dict) containing:
            n_gt (int): number of annotations
            hits (List[np.ndarray]): bool array (TP or not)
    """
    interp_pred = np.array(
        [interp(pred_lane, n=5) for pred_lane in pred], dtype=object
    )
    interp_anno = np.array(
        [interp(anno_lane, n=5) for anno_lane in anno], dtype=object
    )

    ious = discrete_cross_iou(
        interp_pred, interp_anno, width=width, img_shape=img_shape
    )

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    pred_ious = np.zeros(len(pred))
    pred_ious[row_ind] = ious[row_ind, col_ind]
    hits = [pred_ious > thr for thr in iou_thresholds]

    results = {
        'n_gt': len(anno),
        'hits': hits,
        'iou':pred_ious
    }

    return results

def load_vil100_prediction(prediction_path):
    with open(prediction_path, "r") as prediction_file:
        prediction = json.load(prediction_file)
    return prediction

def load_vil100_annotation(annotation_path):
    with open(annotation_path, "r") as annotation_file:
        annotation = json.load(annotation_file)
    lanes = [lane["points"] for lane in annotation["annotations"]["lane"]]
    lanes = remove_repeated(lanes)
    lanes = [lane for lane in lanes if len(lane) > 0]
    return lanes

def remove_repeated(lanes):
    filtered_lanes = []
    for lane in lanes:
        xs = [p[0] for p in lane]
        ys = [p[1] for p in lane]
        ys, indices = np.unique(ys, return_index=True)
        xs = np.array(xs)[indices]
        filtered_lanes.append(list(zip(xs, ys)))
    return filtered_lanes

def load_vil100_data(pred_dir, anno_dir, img_sub_paths, logger=None):
    """
    Load prediction and annotation data for evaluation.
    Args:
        pred_dir (str): Directory where the prediction json files are stored.
        anno_dir (str): Directory where the test labels are stored.
        anno_sub_paths (str): anno sub list path.
    Returns:
        predictions (List[List[tuple]]): List of lane coordinates for all the images.
        annotations (List[List[tuple]]): List of lane coordinates for all the images.
    """
    print_log('Loading prediction data...', logger=logger)
    anno_sub_paths = [img_path.replace("JPEGImages", "Json") + ".json" for img_path in img_sub_paths]
    pred_dirs = [
        Path(pred_dir).joinpath(anno_sub_path) for anno_sub_path in anno_sub_paths
    ]
    pred_dicts = [load_vil100_prediction(pred_dir) for pred_dir in pred_dirs]
    predictions = [pre_dict["lanes"] for pre_dict in pred_dicts]
    img_shapes = [pre_dict["ori_shape"] for pre_dict in pred_dicts]
    print_log('Loading annotation data...', logger=logger)
    anno_dirs = [
        Path(anno_dir).joinpath(anno_sub_path) for anno_sub_path in anno_sub_paths
    ]
    annotations = [load_vil100_annotation(anno_dir) for anno_dir in anno_dirs]

    return annotations, predictions, img_shapes

def eval_predictions(
    pred_dir,
    anno_dir,
    img_sub_paths,
    iou_thresholds=[0.5, 0.8],
    width=30,
    sequential=False,
    logger=None,
):
    """
    Evaluate predictions on CULane dataset.
    Args:
        pred_dir (str): Directory where the prediction txt files are stored.
        anno_dir (str): Directory where the test labels are stored.
        img_sub_paths (str): Test set data list path.
        iou_thresholds (list): IoU threshold list for TP counting.
        width (int): Lane drawing width to calculate IoU.
        sequential (bool): Evaluate image-level results sequentially.
        logger (logging.Logger): Print to the mmcv log if not None.
    Returns:
        result_dict (dict): Evaluation result dict containing
            F1, precision, recall, etc. on the specified IoU thresholds.
    """

    print_log('Loading prediction data...', logger=logger)
    anno_sub_paths = [img_path.replace("JPEGImages", "Json") + ".json" for img_path in img_sub_paths]
    pred_dirs = [
        Path(pred_dir).joinpath(anno_sub_path) for anno_sub_path in anno_sub_paths
    ]
    pred_dicts = [load_vil100_prediction(pred_dir) for pred_dir in pred_dirs]
    predictions = [pre_dict["lanes"] for pre_dict in pred_dicts]
    img_shapes = [pre_dict["ori_shape"] for pre_dict in pred_dicts]

    print_log('Loading annotation data...', logger=logger)
    anno_dirs = [
        Path(anno_dir).joinpath(anno_sub_path) for anno_sub_path in anno_sub_paths
    ]
    annotations = [load_vil100_annotation(anno_dir) for anno_dir in anno_dirs]

    # annotations, predictions, img_shapes = load_vil100_data(
    #     pred_dir, anno_dir, img_sub_paths, logger=logger
    # )
    print_log(
        'Calculating metric {}...'.format(
            'sequentially' if sequential else 'in parallel'
        ),
        logger=logger,
    )
    eps = 1e-8
    if sequential:
        results = t_map(
            partial(
                culane_metric,
                width=width,
                iou_thresholds=iou_thresholds,
            ),
            predictions,
            annotations,
            img_shapes,
        )
    else:
        results = p_map(
            partial(
                culane_metric,
                width=width,
                iou_thresholds=iou_thresholds,
            ),
            predictions,
            annotations,
            img_shapes,
        )

    result_dict = {}
    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    for k, iou_thr in enumerate(iou_thresholds):
        print_log(f"Evaluation results for IoU threshold = {iou_thr}", logger=logger)
        # category = categories if i == 0 else [categories[i - 1]]
        n_gt_list = [r['n_gt'] for r in results]
        ious = [r['iou'].max() if len(r['iou']) > 0 else 0 for r in results]
        # n_category = len([r for r in results])
        # if n_category == 0:
        #     continue
        n_gts = sum(n_gt_list)
        hits = np.concatenate(
            [r['hits'][k] for r in results]
        )
        tp = np.sum(hits)
        fp = len(hits) - np.sum(hits)
        prec = tp / (tp + fp + eps)
        rec = tp / (n_gts + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        miou = get_macro_measure(img_sub_paths, ious)

        result_dict.update(
            {
                f"TP{iou_thr}": tp,
                f"FP{iou_thr}": fp,
                f"FN{iou_thr}": n_gts - tp,
                f"Precision{iou_thr}": prec,
                f"Recall{iou_thr}": rec,
                f"F1_{iou_thr}": f1,
                f"IoU_{iou_thr}": miou,
            }
        )
        print_log(
            f"TP: {tp:5}, FP: {fp:5}, FN: {n_gts - tp:5}, "
            f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, mIoU: {miou:.4f}", 
            logger=logger,
        )

        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += prec / len(iou_thresholds)
        mean_recall += rec / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += (n_gts - tp)
    if len(iou_thresholds) > 2:
        print_log(
            f"Mean result, total_tp: {total_tp}, total_fp: {total_fp}, total_fn: {total_fn}, "
            f"precision: {mean_prec}, recall: {mean_recall}, f1: {mean_f1}",
            logger=logger,
        )
        result_dict.update(
            {
                'mean_TP': total_tp,
                'mean_FP': total_fp,
                'mean_FN': total_fn,
                'mean_Precision': mean_prec,
                'mean_Recall': mean_recall,
                'mean_F1': mean_f1,
            }
        )

    return result_dict
