## Inference

Run the following command to detect the lanes from the image and visualize them:

```bash
python demo/image_demo.py demo/demo.jpg <config path> <checkpoint path> --out-file=result.png
e.g.
python demo/image_demo.py demo/demo.jpg configs/bezierlanenet/culane/dla34.py work_dirs/bezierlanenet_culane_dla34/<timestamp>/latest.pth --out-file=result.png
```