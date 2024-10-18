model = dict(
    type="Detector",
    backbone=dict(type="DLANet"),
    neck=dict(
        type="FeatureFlipFusion",
        channels=256,
        dilated_blocks=dict(
            in_channels=256,
            mid_channels=64,
            dilations=[4, 8]
        ),
    ),
    lane_head=dict(
        type="BezierLaneHead",
        in_channels=256,
        fc_hidden_dim=256,
        stacked_convs=2,
        order=3,
        num_sample_points=100,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            class_weight=1.0 / 0.4,
            reduction='mean',
            loss_weight=0.1
        ),
        loss_dist=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=1.0,
        ),
        loss_seg=dict(
            type="CLRNetSegLoss",
            loss_weight=0.75,
            num_classes=5,  # 4 lanes + 1 background
            ignore_label=255,
            bg_weight=0.4,
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="BezierHungarianAssigner",
            order=3,
            num_sample_points=100,
            alpha=0.8,
            window_size=9
        )
    ),
    test_cfg=dict(
        # dataset info
        dataset="culane",
        ori_img_w=1640,
        ori_img_h=590,
        cut_height=270,
        # inference settings
        conf_threshold=0.95,
        window_size=9,
        max_num_lanes=4,
        num_sample_points=50,
        as_lane=True
    ),
)
