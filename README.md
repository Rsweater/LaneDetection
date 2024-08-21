[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrernet-improving-confidence-of-lane/lane-detection-on-culane)](https://paperswithcode.com/sota/lane-detection-on-culane?p=clrernet-improving-confidence-of-lane)

# LaneDetection

LaneDetection is an lane detection library, which is based on  *MMdetection.*

## What's New

## Method

## Performance

## Install

```bash
conda create -n LaneDetection python=3.8 -y
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -U openmim
mim install mmcv-full==1.5.1
pip install mmdet==2.25.3
pip install -r requirements.txt
cd libs/models/layers/nms/ # 确保本地cuda版本与conda中一致
python setup.py install
cd ../../../../ # TODO:
```

See [Installation Tips](docs/INSTALL.md) for more details.

## Inference

Run the following command to detect the lanes from the image and visualize them:

```bash
python demo/image_demo.py demo/demo.jpg configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth --out-file=result.png
```

## Test

Run the following command to evaluate the model on CULane dataset:

```bash
python tools/test.py configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth
```

For dataset preparation, please refer to [Dataset Preparation](docs/DATASETS.md).

## Frame Difference Calculation

Filtering out redundant frames during training helps the model avoid overfitting to them. We provide a simple calculator that outputs an npz file containing frame difference values.

```bash
python tools/calculate_frame_diff.py [culane_root_path]
```

Also you can find the npz file [[here]](https://github.com/hirotomusiker/CLRerNet/releases/download/v0.2.0/train_diffs.npz).

## Train

Make sure that the frame difference npz file is prepared as `dataset/culane/list/train_diffs.npz`.`<br>`
Run the following command to train a model on CULane dataset:

```bash
python tools/train.py configs/clrernet/culane/clrernet_culane_dla34.py
```

### Train on CurveLanes

Draw segmentation images for CurveLanes for auxiliary loss.

```bash
python tools/make_seg.py configs/clrernet/curvelanes/clrernet_curvelanes_dla34.py
```

Run the following command to train a model on CurveLanes dataset:

```bash
python tools/train.py configs/clrernet/curvelanes/clrernet_curvelanes_dla34.py
```

## Citation

```BibTeX
@article{honda2023clrernet,
      title={CLRerNet: Improving Confidence of Lane Detection with LaneIoU},
      author={Hiroto Honda and Yusuke Uchida},
      journal={arXiv preprint arXiv:2305.08366},
      year={2023},
}
```

## References

* [Turoad/CLRNet](https://github.com/Turoad/CLRNet/)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
* [CULane Dataset](https://xingangpan.github.io/projects/CULane.html)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [optn-mmlab/mmcv](https://github.com/open-mmlab/mmcv)
