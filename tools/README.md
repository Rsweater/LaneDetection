# LaneDetection Tools

## Test

Run the following command to evaluate the model on CULane dataset:

```bash
python tools/test.py <config path> <checkpoint path>
e.g.
python tools/test.py configs/bezierlanenet/culane/dla34.py work_dirs/bezierlanenet_culane_dla34/<timestamp>/latest.pth
```

## Frame Difference Calculation

Filtering out redundant frames during training helps the model avoid overfitting to them. We provide a simple calculator that outputs an npz file containing frame difference values.

```bash
python tools/analysis_tools/calculate_frame_diff.py [culane_root_path]
```

## Train

Make sure that the frame difference npz file is prepared as `dataset/culane/list/train_diffs.npz`.
Run the following command to train a model on CULane dataset:

```bash
python tools/train.py configs/bezierlanenet/culane/dla34.py 
```

## Generate Segmentation Label

Draw segmentation images for CurveLanes、Tusimple for auxiliary loss.

```bash
# for CurveLanes
python tools/generate_seg/generate_seg_curvelanes.py configs/bezierlanenet/curvelanes/dla34.py
# for Tusimple
python tools/generate_seg/generate_seg_tusimple.py <path>/datasets/tusimple
```
