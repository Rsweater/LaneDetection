_base_ = [
    "../base_model.py", "dataset.py",
    "../../_base_/default_runtime.py",
]

# custom imports
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.lane",
        "libs.core.anchor",
        "libs.core.hook",
    ],
    allow_failed_imports=False,
)

cfg_name = "clrernet_tusimple_r18.py"

model = dict(
    type="CLRerNet",
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    lane_head=dict(
        type="CLRerHead",
        loss_cls=dict(type="KorniaFocalLoss", alpha=0.25, gamma=2, loss_weight=6.0, reduction="mean"),
        loss_bbox=dict(type="SmoothL1Loss", reduction="none", loss_weight=0.5),
        loss_iou=dict(
            type="LaneIoULoss",
            # lane_width=9 / 800,
            loss_weight=4.0,
        ),
        loss_seg=dict(
            loss_weight=1.0,
            num_classes=7,  # 6 lane + 1 background
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            iou_dynamick=dict(
                type="LaneIoUCost",
                # lane_width=9 / 800,
                use_pred_start_end=False,
                use_giou=True,
            ),
            iou_cost=dict(
                type="LaneIoUCost",
                # lane_width=15 / 800,
                use_pred_start_end=True,
                use_giou=True,
            ),
        )
    ),
    test_cfg=dict(
        conf_threshold=0.35,
        use_nms=True,
        as_lanes=True,
        nms_thres=50,
        nms_topk=7,
        ori_img_w=1280,
        ori_img_h=720,
        cut_height=160,
    ),
)

total_epochs = 150
evaluation = dict(start=3, interval=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)


data = dict(samples_per_gpu=24)  # single GPU setting

# optimizer
optimizer = dict(type="AdamW", lr=6e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0.0, by_epoch=False)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
