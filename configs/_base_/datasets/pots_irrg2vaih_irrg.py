# ---------------------------------------------------------------
# Copyright (c) 2022-2023 TUM, Fahong Zhang. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

source_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(576, 576), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

target_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPseudoLabelsV2',
         # pseudo_labels_dir='work_dirs/pseudo_labels/base_deeplabv3plus_r50-d8_928x928_40k_potsdam2vaihingen',
         pseudo_labels_dir=None,
         reduce_zero_label=False,
         load_feats=False,
         pseudo_ratio=0.0),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='StrongAugmentation'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_strong_aug', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'ISPRSDataset'
data_root_pots = '../../Datasets/Dataset4EO/ISPRS/Potsdam_IRRG_1024_mmlab'
data_root_vaih = '../../Datasets/Dataset4EO/ISPRS/Vaihingen_IRRG_1024_mmlab'
gt_seg_map_loader_cfg=dict(reduce_zero_label=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type=dataset_type,
            data_root=data_root_pots,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
            pipeline=source_pipeline),
        target=dict(
            type=dataset_type,
            data_root=data_root_vaih,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
            pipeline=target_pipeline),
        rare_class_sampling=None
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root_vaih,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root_vaih,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=test_pipeline),
    train_dataloader=dict(
        persistent_workers=False),
    val_dataloader=dict(
        persistent_workers=False))
