# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='PFGST',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    print_grad_magnitude=False,
    aux_losses=[
        dict(
            type='PFGSTLoss',
            kernel_size=3,
            dilation=4,
            top_k=3,
            sigma=40.79496302,
            weights=[0.01, 0.01],
            sim_type='cosine',
            feat_level=None
        ),
    ]
)
use_ddp_wrapper = True

