# Dataset Preparation

## CULane

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

## TuSimple

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

Finally the dataset folder would look like:

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

## VIL100

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

## CurveLanes

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

Finally the dataset folder would look like:

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
