# CLRerNet: Improving Confidence of Lane Detection with LaneIoU

([Arxiv 2305](https://arxiv.org/abs/2305.08366)) CLRerNet: Improving Confidence of Lane Detection with LaneIoU. [Hiroto Honda](https://dblp.uni-trier.de/pid/20/8792.html) et al. [WACV2024](https://doi.org/10.1109/WACV57701.2024.00121). [Code](https://github.com/hirotomusiker/CLRerNet)![Stars](https://img.shields.io/github/stars/hirotomusiker/CLRerNet)

## Abstract

> Lane marker detection is a crucial component of the autonomous driving and driver assistance systems. Modern deep lane detection methods with row-based lane representation exhibit excellent performance on lane detection benchmarks. Through preliminary oracle experiments, we firstly disentangle the lane representation components to determine the direction of our approach. We show that correct lane positions are already among the predictions of an existing row-based detector, and the confidence scores that accurately represent intersection-over-union (IoU) with ground truths are the most beneficial. Based on the finding, we propose LaneIoU that better correlates with the metric, by taking the local lane angles into consideration. We develop a novel detector coined CLRerNet featuring LaneIoU for the target assignment cost and loss functions aiming at the improved quality of confidence scores. Through careful and fair benchmark including cross validation, we demonstrate that CLRerNet outperforms the state-of-the-art by a large margin - enjoying F1 score of 81.43% compared with 80.47% of the existing method on CULane, and 86.47% compared with 86.10% on CurveLanes.

## Results and Models

### Results on CuLane

| Architecture | Backbone | F1@50 | F1@75 | mF1 | Config | LogFile | Download |
| :----------: | :-------: | :----: | ----- | --- | ------ | ------- | -------- |
|   CLRerNet   | ResNet-18 | 73.67* |       |     |        |         |          |
|   CLRerNet   | ResNet-34 | 75.57* |       |     |        |         |          |
|   CLRerNet   |  DLA-34  |        |       |     |        |         |          |

### Results on TuSimple

| Architecture | Backbone | F1@50 | F1@75 | mF1 | Config | LogFile | Download |
| :----------: | :-------: | :----: | ----- | --- | ------ | ------- | -------- |
|   CLRerNet   | ResNet-18 | 95.41* |       |     |        |         |          |
|   CLRerNet   | ResNet-34 | 95.65* |       |     |        |         |          |
|   CLRerNet   |  DLA-34  |        |       |     |        |         |          |

### Results on VIL-100

| Architecture | Backbone | F1@50 | F1@75 | mF1 | Config | LogFile | Download |
| :----------: | :-------: | :---: | ----- | --- | ------ | ------- | :------: |
|   CLRerNet   | ResNet-18 | 85.92 |       |     |        |         |          |
|   CLRerNet   | ResNet-34 |      |       |     |        |         |          |
|   CLRerNet   |  DLA-34  |      |       |     |        |         |          |

### Results on CurveLanes

| Architecture | Backbone | F1@50 | F1@75 | mF1 | Config | LogFile | Download |
| :----------: | :-------: | ----- | ----- | --- | ------ | ------- | -------- |
|   CLRerNet   | ResNet-18 |       |       |     |        |         |          |
|   CLRerNet   | ResNet-34 |       |       |     |        |         |          |
|   CLRerNet   |  DLA-34  |       |       |     |        |         |          |
