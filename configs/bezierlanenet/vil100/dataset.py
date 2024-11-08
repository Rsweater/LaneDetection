"""
    config file of the CULane dataset for CondLaneNet
    Adapted from:
    https://github.com/aliyun/conditional-lane-detection/blob/master/configs/condlanenet/curvelanes/curvelanes_medium_train.py
"""

dataset_type = "VIL100Dataset"
data_root = "datasets/vil100"
img_scale = (640, 360)
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False
)
compose_cfg = dict(keypoints=True, masks=True)

# data pipeline settings
train_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    dict(type="HorizontalFlip", p=0.5),
    dict(type="ChannelShuffle", p=0.1),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=0.04,
        contrast_limit=0.15,
        p=0.6,
    ),
    dict(
        type="HueSaturationValue",
        hue_shift_limit=(-10, 10),
        sat_shift_limit=(-10, 10),
        val_shift_limit=(-10, 10),
        p=0.7,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="MotionBlur", blur_limit=5, p=1.0),
            dict(type="MedianBlur", blur_limit=5, p=1.0),
        ],
        p=0.2,
    ),
    dict(
        type="IAAAffine",
        scale=(0.8, 1.2),
        rotate=(-10.0, 10.0),  # this sometimes breaks lane sorting
        translate_percent=0.1,
        p=0.7,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type="albumentation", pipelines=train_al_pipeline, 
            cut_y_duplicated=True, need_resorted=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectBeizerInfo",
        keys=["img"], interpolate=False, fix_endpoints=False,
        order=3, norm=True, num_sample_points=100,
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "eval_shape",
            "img_shape",
            "img_norm_cfg",
            "ori_shape",
            "img_shape",
            "gt_points",
            "gt_masks",
            "gt_lanes",
        ],
    ),
]

val_pipeline = [
    dict(type="albumentation", pipelines=val_al_pipeline, 
            cut_y_duplicated=True, need_resorted=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectBeizerInfo",
        keys=["img"], interpolate=False, fix_endpoints=False,
        order=3, norm=True, num_sample_points=100,
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "ori_shape",
            "img_shape",
            "gt_points",
            "crop_shape",
            "crop_offset",
        ],
    ),
]

data = dict(
    samples_per_gpu=32,  # medium
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + "/data/train.txt",
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + "/data/test.txt",
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + "/data/test.txt",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
