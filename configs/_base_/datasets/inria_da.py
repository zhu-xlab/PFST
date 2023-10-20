# dataset settings
dataset_type = 'EODataset'
datapipe = 'inria_clipped'
data_root = '../../Datasets/Dataset4EO'
reduce_zero_label = False
gt_seg_map_loader_cfg=dict(reduce_zero_label = reduce_zero_label)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

source_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
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
         pseudo_labels_dir=None,
         load_feats=False,
         reduce_zero_label=False,
         pseudo_ratio=0.3
    ),
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
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

src_crf = dict(
    type=dataset_type,
    data_root=data_root,
    datapipe=datapipe,
    datapipe_cfg=dict(city_names=['austin', 'chicago', 'kitsap']),
    reduce_zero_label=reduce_zero_label,
    split='train',
    gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
    pipeline=source_pipeline)

trg_crf = dict(
    type=dataset_type,
    data_root=data_root,
    datapipe=datapipe,
    datapipe_cfg=dict(city_names=['vienna', 'tyrol-w']),
    reduce_zero_label=reduce_zero_label,
    split='train',
    gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
    pipeline=target_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='UDADataset',
        source=src_crf,
        target=trg_crf
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        datapipe_cfg=dict(city_names=['vienna', 'tyrol-w']),
        reduce_zero_label=reduce_zero_label,
        split='train',
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        datapipe_cfg=dict(city_names=['vienna', 'tyrol-w']),
        # datapipe_cfg=dict(city_names=['tyrol-w']),
        reduce_zero_label=reduce_zero_label,
        split='val',
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=test_pipeline),
    train_dataloader=dict(
        persistent_workers=False),
    val_dataloader=dict(
        persistent_workers=False)
)

