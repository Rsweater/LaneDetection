# LaneDetection

LaneDetection is a comprehensive and stylistically unified lane detection library aimed at accelerating the progress of algorithm research and reproduction in scientific and industrial applications. Some of the reproduced methods achieve accuracy that surpasses the original paper results.

## Features

### **Data Design**

* **Multi-Dataset Support:** Compatible with mainstream datasets including CuLane, TuSimple, VIL-100, and CurveLanes.
* **Versatile Lane Modeling:** Supports various lane modeling methods tailored to different use cases.
* **Unified Structure:** Designed with a consistent dataset structure and external interface for ease of use.
* **Efficient Label Storage:** Utilizes a space-efficient storage method for segmentation labels that simplifies the review of segmentation results.
* **Data Augmentation:** Integrates Albumetation for consistent model training and supports multi-scale image cropping.

### **Module Design**

* **SOTA Methods Support:** Includes a wide range of classic state-of-the-art algorithms.
* **Streamlined Code:** Employs encapsulation and inheritance for a unified external interface and structured design, facilitating rapid implementation of diverse algorithms and enhancing understanding of their differences.

### **Training Design**

* **Lifecycle Management:** Features a comprehensive module lifecycle management Runner for efficient tracking.
* **Organized Logging:** Ensures a structured log storage system that allows for easy querying, alongside a logically organized configuration file structure.
* **Version Control:** Provides functionality for easy code version comparison and supports bash scripts for quick environment configuration with single-line commands.

## Support Models

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
        <td>☑BézierLaneNet ☐BSNet</td>
    </tr>
    <tr>
        <td>Line-Proposal&Line-Anchor Detection</td>
        <td>☐LaneATT ☑CLRNet ☑CLRerNet ☐O2SFormer ☐ADNet</td>
    </tr>
</table>

## Install

### Regular Method

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
cd ../../../../ # TODO: 直接导入
```

### Bash Script

```bash
source install.sh
```

## Getting Started

## Citation

## References

* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [Wolfwjs/GANet](https://github.com/Wolfwjs/GANet)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [Yzichen/mmLaneDet](https://github.com/Yzichen/mmLaneDet)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [voldemortX/pytorch-auto-drive](https://github.com/voldemortX/pytorch-auto-drive)
