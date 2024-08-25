[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrernet-improving-confidence-of-lane/lane-detection-on-culane)](https://paperswithcode.com/sota/lane-detection-on-culane?p=clrernet-improving-confidence-of-lane)

# LaneDetection

LaneDetection is an lane detection library, which is based on  *MMdetection.*  为了增加可读性，我们的dataset结构、data读取与增强、标签转换、分割标签、模型等等均采用了统一的代码风格。尽量的进行mmdetection各个组件的解耦，尽可能的精简代码，增加可读性。使用Pillow的P模式进行分割图像的存放，以便直观的观察到分割结果。

## What's New

## Method

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>模型</b>
      </td>
      <td colspan="2">
        <b>框架组件</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>Segmentation based</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/resa">RESA</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/main/configs/scnn">SCNN</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/erfnet">ERFNet</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/deeplabv3p">DeepLabV3+</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v4/configs/rtformer">RTFormer</a></li>
          </ul>
        </details>
        <details><summary><b>Detection based</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v6/configs/ufld">UFLD</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v6/configs/condlane">CondLane</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v6/configs/clrnet">CLRNet</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v6/configs/adnet">ADNet</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v6/pplanedet/model/losses/line_iou.py">CLRerNet</a></li>
          </ul>
        </details>
      </td>
      <td>
        <details><summary><b>Backbones</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v3/pplanedet/model/backbones/resnet.py">ResNet</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v4/pplanedet/model/backbones/convnext.py">ConvNext</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v4/pplanedet/model/backbones/mobilenet.py">MobileNetV3</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v4/pplanedet/model/backbones/cspresnet.py">CSPResNet</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v5/pplanedet/model/backbones/shufflenet.py">ShuffleNet</a></li>
          </ul>
        </details>
        <details><summary><b>Necks</b></summary>
          <ul>
            <li><a href="url">FPN</a></li>
            <li><a href="url">Feature Flip Fusion</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v5/pplanedet/model/necks/csprepbifpn.py">CSPRepbifpn</a></li>
          </ul>
        </details>
        <details><summary><b>Losses</b></summary>
          <ul>
            <li><a href="url">Binary CE Loss</a></li>
            <li><a href="url">Cross Entropy Loss</a></li>
            <li><a href="url">Focal Loss</a></li>
            <li><a href="url">MultiClassFocal Loss</a></li>
            <li><a href="url">RegL1KpLoss</a></li>
          </ul>
        </details>
        <details><summary><b>Metrics</b></summary>
          <ul>
            <li>Accuracy</li>
            <li>FP</li>
            <li>FN</li>
	    <li>F1@0.1</li>
	    <li>F1@0.5</li>
	    <li>F1@0.75</li>
          </ul>  
        </details>
      </td>
      <td>
        <details><summary><b>Datasets</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/datasets/tusimple.py">Tusimple</a></li>  
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/datasets/culane.py">CULane</a></li>
          </ul>
        </details>
        <details><summary><b>Label transformer</b></summary>
          <ul>
            <li>RandomLROffsetLABEL</li>  
          </ul>
        </details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## Install

```bash
# Clone the repo
git clone https://github.com/Rsweater/LaneDetection_mm.git

# Create&Activate environment
conda create -n LaneDetection python=3.8 -y
conda activate LaneDetection

# Install dependencies
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -U openmim
mim install mmcv-full==1.5.1
pip install mmdet==2.25.3
pip install -r requirements.txt

# complie ops
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
