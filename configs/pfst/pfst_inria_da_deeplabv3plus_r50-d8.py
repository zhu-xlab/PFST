# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/deeplabv3plus_r50-d8.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/inria_da.py',
    # Basic UDA Self-Training
    '../_base_/uda/pfst.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw_40k.py',
]
expr_name='pfst_inria_da_deeplabv3plus_r50-d8'

# Random Seed
seed = 0

model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
)

# Call optimizer within train_step
optimizer_config = None
optimizer = dict(
    lr=6e-05,
)

# Modifications to Basic UDA
uda = dict(
    aux_losses=[
        dict(
            type='PFGSTLoss',
            kernel_size=3,
            dilation=2,
            top_k=3,
            weights={'src_pos': 0.1, 'src_neg': 0.1, 'sim_pos': 0.1,
                     'sim_neg': 0.1, 'src_pos_std': 0.1, 'src_neg_std': 0.1},
            sim_type='cosine',
            feat_level=None,
            detach_unfold=True,
            downscale=0.5
        ),
    ],
    alpha=0.999,
    thre_type='all',
    pseudo_threshold=0.98,
    trg_loss_weight=1.,
    use_decoded_feats=True,
)

init_kwargs = dict(
    project='rsi_dass',
    entity='tum-tanmlh',
    name=expr_name,
    resume='never'
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='WandbHookSeg',
        #      init_kwargs=init_kwargs,
        #      interval=201),
        # dict(type='MMSegWandbHook',
        #      init_kwargs=init_kwargs,
        #      interval=201,
        #      num_eval_images=20),
        # dict(type='PlotStatisticsHook',
        #      log_dir=f'work_dirs/plots/{name}',
        #      sim_feat_cfg=dict(kernel_size=3, dilation=2,
        #                        sigma=21.41786553,
        #                        feat_level=3,
        #                        mean_sim=[0.5, 0.55, 0.6, 0.65, 0.7],
        #                        top_k=9),
        #      data_cfg=data,
        #      interval=1),
        # dict(type='PseudoLabelingHook',
        #      log_dir='work_dirs/pseudo_labels/deeplabv3plus_r50-d8_512x512_80k_loveda_r2u',
        #      interval=1),
    ])
