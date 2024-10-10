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

cfg_name = "bezierlanenet_vil100_r18.py"

model = dict(
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    lane_head=dict(
        loss_seg=dict(
            loss_weight=0.75,
            num_classes=9,  # 8 lane + 1 background
        )
    ),
    test_cfg=dict(
        # dataset info
        ori_img_w="no fixed size",
        ori_img_h="no fixed size",
        cut_height="no fixed size",
        # inference settings
        conf_threshold=0.4,
        window_size=5,
        max_num_lanes=8,
        num_sample_points=50,
    ),
)

total_epochs = 400
evaluation = dict(start=10, interval=10)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)


data = dict(samples_per_gpu=48, workers_per_gpu=8)  # single GPU setting

# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-3,
    paramwise_cfg=dict(
        custom_keys={
            'conv_offset': dict(lr_mult=0.1),
        }),
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    by_epoch=True
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
