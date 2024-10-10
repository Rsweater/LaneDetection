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

cfg_name = "clrnet_curvelanes_dla34.py"

model = dict(
    backbone=dict(
        type="DLANet",
        dla="dla34",
        pretrained=True,
    ),
    lane_head=dict(
        type="CLRerHead",
        loss_iou=dict(
            type="LaneIoULoss",
            # lane_width=2.5 / 800,
            loss_weight=4.0,
        ),
        loss_seg=dict(
            loss_weight=2.0,
            num_classes=9,  # 8 lane + 1 background
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            iou_dynamick=dict(
                type="CLRNetIoUCost",
                # lane_width=2.5 / 800,
            ),
            iou_cost=dict(
                type="CLRNetIoUCost",
                # lane_width=10 / 800,
            ),
        )
    ),
    test_cfg=dict(
        conf_threshold=0.43,
        use_nms=True,
        as_lanes=True,
        nms_thres=50,
        nms_topk=8,
    ),
)

total_epochs = 150
evaluation = dict(start=3, interval=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)


data = dict(samples_per_gpu=16)  # single GPU setting

# optimizer
optimizer = dict(type="AdamW", lr=6e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0.0, by_epoch=False)

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
