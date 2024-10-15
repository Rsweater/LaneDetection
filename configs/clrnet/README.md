# **CLRNet:** Cross Layer Refinement Network for Lane Detection.

([Arxiv 2203](https://arxiv.org/abs/2203.10350)) **CLRNet:** Cross Layer Refinement Network for Lane Detection. [Tu Zheng](https://dblp.uni-trier.de/pid/229/4199.html) et al. [CVPR2022](https://doi.org/10.1109/CVPR52688.2022.00097). [Code](https://github.com/Turoad/CLRNet)![Stars](https://img.shields.io/github/stars/Turoad/CLRNet)
![img](../_base_/figures/clrnet.png)

## Abstract

> Lane is critical in the vision navigation system of the intelligent vehicle. Naturally, lane is a traffic sign with high-level semantics, whereas it owns the specific local pattern which needs detailed low-level features to localize accurately. Using different feature levels is of great importance for accurate lane detection, but it is still under-explored. In this work, we present Cross Layer Refinement Network (CLRNet) aiming at fully utilizing both high-level and low-level features in lane detection. In particular, it first detects lanes with high-level semantic features then performs refinement based on low-level features. In this way, we can exploit more contextual information to detect lanes while leveraging local detailed lane features to improve localization accuracy. We present ROIGather to gather global context, which further enhances the feature representation of lanes. In addition to our novel network design, we introduce Line IoU loss which regresses the lane line as a whole unit to improve the localization accuracy. Experiments demonstrate that the proposed method greatly outperforms the state-of-the-art lane detection approaches.

## Results and Models

### Results on CuLane

| Architecture | Backbone | F1@50 | F1@75 | mF1 | Config | LogFile | Download |
| :----------: | :-------: | :----: | ----- | --- | ------ | ------- | -------- |
|    CLRNet    | ResNet-18 | 73.67* |       |     |        |         |          |
|    CLRNet    | ResNet-34 | 75.57* |       |     |        |         |          |
|    CLRNet    |  DLA-34  |        |       |     |        |         |          |

### Results on TuSimple

| Architecture | Backbone | F1@50 | F1@75 | mF1 | Config | LogFile | Download |
| :----------: | :-------: | :----: | ----- | --- | ------ | ------- | -------- |
|    CLRNet    | ResNet-18 | 95.41* |       |     |        |         |          |
|    CLRNet    | ResNet-34 | 95.65* |       |     |        |         |          |
|    CLRNet    |  DLA-34  |        |       |     |        |         |          |

### Results on VIL-100

| Architecture | Backbone | F1@50 | F1@75 | mF1 | Config | LogFile | Download |
| :----------: | :-------: | :---: | ----- | --- | ------ | ------- | :------: |
|    CLRNet    | ResNet-18 | 85.92 |       |     |        |         |          |
|    CLRNet    | ResNet-34 |      |       |     |        |         |          |
|    CLRNet    |  DLA-34  |      |       |     |        |         |          |

### Results on CurveLanes

| Architecture | Backbone | F1@50 | F1@75 | mF1 | Config | LogFile | Download |
| :----------: | :-------: | ----- | ----- | --- | ------ | ------- | -------- |
|    CLRNet    | ResNet-18 |       |       |     |        |         |          |
|    CLRNet    | ResNet-34 |       |       |     |        |         |          |
|    CLRNet    |  DLA-34  |       |       |     |        |         |          |
