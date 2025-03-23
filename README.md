# LaneDetection

LaneDetection is a comprehensive and stylistically unified lane detection library aimed at accelerating the progress of algorithm research and reproduction in scientific and industrial applications. Some of the reproduced methods achieve accuracy that surpasses the original paper results.

## Features

<details>
<summary>Data Design</summary>

* **Multi-Dataset Compatibility:** Supports a wide range of mainstream datasets, including CuLane, TuSimple, VIL-100, and CurveLanes.
* **Versatile Lane Modeling:** Offers a variety of lane modeling methods tailored to different use cases, ensuring flexibility and adaptability.
* **Unified Code Structure:** Designed with a consistent dataset structure and standardized external interface for easy integration and use.
* **Efficient Storage of Segmentation Label:** Utilizes a space-efficient storage method for segmentation labels, simplifying the review and analysis of segmentation results.
* **Data Augmentation:** Integrates albumetation for consistent and robust model training, and supports multi-scale image cropping to enhance model generalization.

</details>

<details>
<summary>Module Design</summary>

* **State-of-the-Art (SOTA) Methods Support:** Incorporates a comprehensive selection of classic and cutting-edge state-of-the-art algorithms.
* **Streamlined Code Design:** Utilizes encapsulation and inheritance to provide a unified external interface and a well-structured design. This approach facilitates the rapid implementation of diverse algorithms and enhances the understanding of their differences.

</details>

<details>

<summary>Training Design</summary>

* **Comprehensive Lifecycle Management:** Features a robust module lifecycle management system (Runner) for efficiently tracking and managing data and modules.
* **Structured Logging:** Ensures a well-organized log storage system that facilitates easy querying and analysis, complemented by a logically structured configuration file system.
* **Version Control and Environment Setup:** Provides tools for easy code version comparison and supports bash scripts for rapid environment setup with single-line commands, ensuring consistency and reproducibility.

</details>

## Getting Started

### Install

<details>
<summary>Regular Method</summary>

```bash
# Clone the repo
git clone https://github.com/Rsweater/LaneDetection.git

# Create&Activate environment
conda create -n LaneDetection python=3.8 -y
conda activate LaneDetection

# Install dependencies
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -U openmim
mim install mmcv-full==1.5.1
pip install mmdet==2.25.3
pip install -r requirements.txt

# Complie ops
cd libs/models/layers/nms/ # 确保本地cuda版本与conda中一致
python setup.py install
cd ../../../../ # TODO: 直接导入
```

</details>

<details>
<summary>Bash Script</summary>

```bash
source install.sh
```

</details>

### Dataset Preparation

<details>
<summary>CULane</summary>

[\[Introduction\]](https://xingangpan.github.io/projects/CULane.html)

Download the tar.gz files from the [official gdrive](https://drive.google.com/open?id=1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu)
, or use gdown as follows (cited from [LaneATT&#39;s docs](https://github.com/lucastabelini/LaneATT/blob/main/DATASETS.md#culane)):

```bash
# <CULANE.BASE_DIR>
mkdir -p <path>/datasets/culane
cd <path>/datasets/culane
# if you don't have gdown, install it with pip
# pip install gdown
# train & validation images (~30 GB)
gdown "https://drive.google.com/uc?id=14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL"  # driver_23_30frame.tar.gz
gdown "https://drive.google.com/uc?id=1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk"  # driver_161_90frame.tar.gz
gdown "https://drive.google.com/uc?id=1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL"  # driver_182_60frame.tar.gz

# test images (~10 GB)
gdown "https://drive.google.com/uc?id=1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7"  # driver_37_30frame.tar.gz
gdown "https://drive.google.com/uc?id=1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8"  # driver_100_30frame.tar.gz
gdown "https://drive.google.com/uc?id=1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV"  # driver_193_90frame.tar.gz

# all annotations
gdown "https://drive.google.com/uc?id=1MlL1oSiRu6ZRU-62E39OZ7izljagPycH"  # laneseg_label_w16.tar.gz
gdown "https://drive.google.com/uc?id=18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd"  # list.tar.gz
```

Then extract the downloaded files in the dataset directory:

```bash
# extract train & validation images
tar xvf driver_23_30frame.tar.gz
tar xvf driver_161_90frame.tar.gz
tar xvf driver_182_30frame.tar.gz
# extract test images
tar xvf driver_37_30frame.tar.gz
tar xvf driver_100_30frame.tar.gz
tar xvf driver_193_90frame.tar.gz
# extract all annotations
tar xvf laneseg_label_w16.tar.gz
tar xvf list.tar.gz
```

Finally the dataset folder would look like:

```
  <CULANE.BASE_DIR>
     ├─ driver_100_30frame  
     ├─ driver_161_90frame  
     ├─ driver_182_30frame  
     ├─ driver_193_90frame
     ├─ driver_23_30frame
     ├─ driver_37_30frame
     ├─ laneseg_label_w16
     └─ list
```

**Note.** Data storage method:

1. `<path>/datasets/culane` where `<path>` directly points to the datasets directory in the project.
2. **[Recommended]** Store separately, for example, `~/lane/dataset/culane`, and then create a soft link `ln -s ~/lane/dataset/culane <LANEDETECTION.BASE_DIR>/datasets/culane`.

</details>

<details>
<summary>TuSimple</summary>

[\[Introduction\]](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection)

Download the dataset from the [\[Download page\]](https://openxlab.org.cn/datasets/OpenDataLab/tusimple_lane), Provided by openxlab.

It is recommended to use the CLI method for downloading and to create an independent python environment for the openxlab library.

```python
# <TUSIMPLE.BASE_DIR>
mkdir -p <path>/datasets/tusimple
cd <path>/datasets/tusimple
# install openxlab
conda create -n download python -y
conda activate download
pip install openxlab

# login to openxlab
openxlab login
# view dataset
openxlab dataset info --dataset-repo OpenDataLab/tusimple_lane
# download dataset
openxlab dataset download --dataset-repo OpenDataLab/tusimple_lane --source-path /raw/tusimple_lane.tar.gz
# Note. If you encounter '[info] speed degradation, restarting...' for a long time, 
# please quit and re-run the command to continue downloading.
```

Then extract the downloaded files in the dataset directory:

```bash
# extract all files & move them to the parent directory
tar xvf OpenDataLab___tusimple_lane/raw/tusimple_lane.tar.gz
mv tusimple_lane/lane_detection/* .

# extract train & validation zip
unzip train_set.zip
unzip test_set.zip
```

The dataset folder would look like:

```
    <TUSIMPLE.BASE_DIR>
      ├─ clips
      ├─ label_data_0313.json
      ├─ label_data_0531.json
      ├─ label_data_0601.json
      ├─ test_tasks_0627.json
      ├─ test_baseline.json
      └─ test_label.json

```

**Note.** Data storage method:

1. `<path>/datasets/tusimple` where `<path>` directly points to the datasets directory in the project.
2. **[Recommended]** Store separately, for example, `~/lane/dataset/tusimple`, and then create a soft link `ln -s ~/lane/dataset/tusimple <LANEDETECTION.BASE_DIR>/datasets/tusimple`.

**Optional**, IF you want to generate segmentation labels for Tusimple, run the following command:

```bash
python tools/generate_seg/generate_seg_tusimple.py <TUSIMPLE.BASE_DIR>
```

</details>

<details>
<summary>VIL-100</summary>

[\[Introduction\]](https://github.com/yujun0-0/mma-net)

Download the dataset from the [\[Google Drive\]](https://drive.google.com/file/d/1EqdCV-8QKccQ0m3mSd7HuEPefTK7dzXS/view?usp=sharing) or [\[Baidu NetDisk\]](https://pan.baidu.com/s/1hFPKt4az6AiMmsV4c9Odmg?pwd=yyl7).

```bash
# <VIL100.BASE_DIR>
mkdir -p <path>/datasets/vil100
cd <path>/datasets/vil100
# download dataset
gdown "https://drive.google.com/uc?id=1EqdCV-8QKccQ0m3mSd7HuEPefTK7dzXS"
```

Then extract the downloaded files in the dataset directory:

```bash
unrar x dataset.rar
mv dataset/VIL100/* .
```

Finally the dataset folder would look like:

```
    <VIL100.BASE_DIR>
      ├─ Annotations
      ├─ data
      ├─ JPEGImages
      └─ Json
```

**Note.** Data storage method:

1. `<path>/datasets/vil100` where `<path>` directly points to the datasets directory in the project.
2. **[Recommended]** Store separately, for example, `~/lane/dataset/vil100`, and then create a soft link `ln -s ~/lane/dataset/vil100 <LANEDETECTION.BASE_DIR>/datasets/vil100`.

</details>

<details>
<summary>CurveLanes</summary>

[\[Introduction\]](https://github.com/SoulmateB/CurveLanes)

Download the dataset from the [\[Download Page\]](https://github.com/SoulmateB/CurveLanes).
**Note.** If the download link for CurveLanes.part1.rar is invalid, you can use
[\[Baidu NetDisk\]](https://pan.baidu.com/s/1-nmUOCrU0twBZtOe_neuLw?pwd=m67c)

```bash
# <CurveLanes.BASE_DIR>
mkdir -p <path>/datasets/curvelanes
cd <path>/datasets/curvelanes
# download datset
gdown "https://drive.google.com/uc?id=1nTB2Cdyd0cY3nVB1rZ6Z00YjhKLvzIqr" # CurveLanes.part1.rar
gdown "https://drive.google.com/uc?id=1iv-2Z9B6cfncogRhFPHKqNlt-u7hQnZd" # CurveLanes.part2.rar
gdown "https://drive.google.com/uc?id=1n2sFDdy2KAaw-7siO7HWuwxUeVb6SXfN" # CurveLanes.part3.rar
gdown "https://drive.google.com/uc?id=1xiz2oD4A0rlt3TGFdz5uzU1s-a0SbsX8" # CurveLanes.part4.rar
gdown "https://drive.google.com/uc?id=1vpFSytqlsJA-rzfuY2lyXmZvEKpaovjX" # CurveLanes.part5.rar
gdown "https://drive.google.com/uc?id=1NZLvaBWj0Mnuo07bxKT7shxqi9upSegJ" # CurveLanes.part6.rar
```

Then extract the downloaded files in the dataset directory:

```bash
unrar x Curvelanes.part1.rar
```

The dataset folder would look like:

```
    <VIL100.BASE_DIR>
    ├── test
    │   └── images
    ├── train
    │   └── images
    │   └── labels
    │   └── train.txt
    └── valid
        └── images
        └── labels
        └── valid.txt
```

**Note.** Data storage method:

1. `<path>/datasets/curvelanes` where `<path>` directly points to the datasets directory in the project.
2. **[Recommended]** Store separately, for example, `~/lane/dataset/curvelanes`, and then create a soft link `ln -s ~/lane/dataset/curvelanes <LANEDETECTION.BASE_DIR>/datasets/curvelanes`.

**Optional**, IF you want to generate segmentation labels for Tusimple, run the following command:

```bash
python tools/generate_seg/generate_seg_curvelanes.py <CURVELANES.BASE_DIR>
```

</details>

### User Guides

<details>
<summary>Inference</summary>

Run the following command to detect the lanes from the image and visualize them:

```bash
# <config path> consists of four levels of directories: configs, method, dataset, and backbone.
python demo/image_demo.py demo/demo.jpg <config path> <checkpoint path> --out-file=result.png
e.g.
python demo/image_demo.py demo/demo.jpg configs/bezierlanenet/culane/dla34.py work_dirs/bezierlanenet_culane_dla34/<timestamp>/latest.pth --out-file=result.png
```

</details>

<details>
<summary>Training & Test</summary>

```bash
# single gpu training
python tools/train.py <config file path> \ 
            --gpu-id <gpu id> # Optional
# distributed training
bash tools/dist_train.sh <config file path> <gpu number>

# single gpu testing
python tools/test.py <config file path> <checkpoint path> \ 
            --gpu-id <gpu id> # Optional
# distributed testing
bash tools/dist_test.sh <config file path> <checkpoint path> <gpu number>
```

</details>

<details>
<summary>Speed Test</summary>

Filtering out redundant frames during training helps the model avoid overfitting to them. We provide a simple calculator that outputs an npz file containing frame difference values.

```bash
python tools/analysis_tools/calculate_frame_diff.py [culane_root_path]
```

</details>

<!-- ## Support Models

<table align="center">
    <tr>
        <th><b>Paradigm</b></th>
        <th><b>TODO</b></th>
    </tr>
    <tr>
        <td>Traditional Semantic Segmentation</td>
        <td>☐SCNN  ☐RESA</td>
    </tr>
    <tr>
        <td>Grid Semantic Segmentation</td>
        <td>☐UFLD ☐UFLDv2 ☐CondLaneNet</td>
    </tr>
    <tr>
        <td>Instance Segmentation</td>
        <td>☐CondLaneNet</td>
    </tr>
    <tr>
        <td>Keypoints Detection</td>
        <td>☐GANet</td>
    </tr>
    <tr>
        <td>Parametric Curve Detection</td>
        <td>☑<a href="./configs/bezierlanenet/README.md">BézierLaneNet</a> ☐BSNet</td>
    </tr>
    <tr>
        <td>Line-Proposal&Line-Anchor Detection</td>
        <td>☐LaneATT ☑<a href="./configs/clrnet/README.md">CLRNet</a> ☑<a href="./configs/clrernet/README.md">CLRerNet</a> ☐O2SFormer ☐ADNet</td>
    </tr>
</table> -->

## Citation

## References

* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [Wolfwjs/GANet](https://github.com/Wolfwjs/GANet)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [Yzichen/mmLaneDet](https://github.com/Yzichen/mmLaneDet)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [voldemortX/pytorch-auto-drive](https://github.com/voldemortX/pytorch-auto-drive)
