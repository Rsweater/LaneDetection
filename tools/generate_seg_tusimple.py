import json
import numpy as np
import cv2
import os
import argparse
from PIL import Image, ImageDraw
from mmcv import ProgressBar

TRAIN_SET = ['label_data_0313.json', 'label_data_0601.json']
VAL_SET = ['label_data_0531.json']
TRAIN_VAL_SET = TRAIN_SET + VAL_SET
TEST_SET = ['test_label.json']


def gen_label_for_json(args, image_set):
    H, W = 720, 1280
    SEG_WIDTH = 30
    save_dir = args.savedir

    os.makedirs(os.path.join(args.root, args.savedir, "list"), exist_ok=True)
    list_f = open(
        os.path.join(args.root, args.savedir, "list",
                     "{}_gt.txt".format(image_set)), "w")

    json_path = os.path.join(args.root, args.savedir,
                             "{}.json".format(image_set))
    
    # 获取文件长度
    with open(json_path) as f:
        file_length = sum(1 for _ in f)

    with open(json_path) as f:
        bar = ProgressBar(file_length)
        for line in f:
            label = json.loads(line)
            # ---------- clean and sort lanes -------------
            lanes = []
            _lanes = []
            slope = [
            ]  # identify 0th, 1st, 2nd, 3rd, 4th, 5th lane through slope
            for i in range(len(label['lanes'])):
                l = [(x, y)
                     for x, y in zip(label['lanes'][i], label['h_samples'])
                     if x >= 0]
                if (len(l) > 1):
                    _lanes.append(l)
                    slope.append(
                        np.arctan2(l[-1][1] - l[0][1], l[0][0] - l[-1][0]) /
                        np.pi * 180)
            _lanes = [_lanes[i] for i in np.argsort(slope)]
            slope = [slope[i] for i in np.argsort(slope)]

            idx = [None for i in range(6)]
            for i in range(len(slope)):
                if slope[i] <= 90:
                    idx[2] = i
                    idx[1] = i - 1 if i > 0 else None
                    idx[0] = i - 2 if i > 1 else None
                else:
                    idx[3] = i
                    idx[4] = i + 1 if i + 1 < len(slope) else None
                    idx[5] = i + 2 if i + 2 < len(slope) else None
                    break
            for i in range(6):
                lanes.append([] if idx[i] is None else _lanes[idx[i]])

            # ---------------------------------------------

            img_path = label['raw_file']
            seg_img = Image.new('P', (W, H))
            palette = [
                0, 0, 0,  # 0: 黑色（背景）
                255, 0, 0,  # 1: 红色
                0, 255, 0,  # 2: 绿色
                0, 0, 255,  # 3: 蓝色
                255, 255, 0,  # 4: 黄色
                255, 0, 255,  # 5: 品红色
                0, 255, 255,  # 6: 青色
            ]
            seg_img.putpalette(palette)
            list_str = []  # str to be written to list.txt
            draw = ImageDraw.Draw(seg_img)
            for i in range(len(lanes)):
                coords = lanes[i]
                if len(coords) < 4:
                    list_str.append('0')
                    continue
                for j in range(len(coords) - 1):
                    draw.line([coords[j], coords[j + 1]], fill=i + 1, width=SEG_WIDTH // 2)
                list_str.append('1')

            seg_path = img_path.split("/")
            seg_path, img_name = os.path.join(args.root, args.savedir,
                                              seg_path[1],
                                              seg_path[2]), seg_path[3]
            os.makedirs(seg_path, exist_ok=True)
            seg_path = os.path.join(seg_path, img_name[:-3] + "png")
            seg_img.save(seg_path)

            seg_path = "/".join([
                args.savedir, *img_path.split("/")[1:3], img_name[:-3] + "png"
            ])
            if seg_path[0] != '/':
                seg_path = '/' + seg_path
            if img_path[0] != '/':
                img_path = '/' + img_path
            list_str.insert(0, seg_path)
            list_str.insert(0, img_path)
            list_str = " ".join(list_str) + "\n"
            list_f.write(list_str)
            bar.update()


def generate_json_file(save_dir, json_file, image_set):
    with open(os.path.join(save_dir, json_file), "w") as outfile:
        for json_name in (image_set):
            with open(os.path.join(args.root, json_name)) as infile:
                for line in infile:
                    outfile.write(line)


def generate_label(args):
    save_dir = os.path.join(args.root, args.savedir)
    os.makedirs(save_dir, exist_ok=True)
    generate_json_file(save_dir, "train_val.json", TRAIN_VAL_SET)
    generate_json_file(save_dir, "test.json", TEST_SET)

    print("generating train_val set...")
    gen_label_for_json(args, 'train_val')
    print("\ngenerating test set...")
    gen_label_for_json(args, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',
                        help='The root of the Tusimple dataset')
    parser.add_argument('--savedir',
                        type=str,
                        default='seg_label',
                        help='The root of the Tusimple dataset')
    args = parser.parse_args()

    generate_label(args)
