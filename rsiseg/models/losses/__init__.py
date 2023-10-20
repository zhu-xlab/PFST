# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .feat_sim_loss import FeatSimLoss, FeatSimLossV2, AdaptiveFeatSimLoss,\
    MultiScaleAdaptiveFeatSimLoss, AdaptiveFeatSimLossV2, AdaptiveFeatSimLossV3
from .adv_loss import AdvLoss
from .entropy_loss import EntropyLoss
from .pseudo_label_loss import PseudoLabelLoss
from .pfst_loss import PFSTLoss
from .pfgst_loss import PFGSTLoss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss', 'FeatSimLoss', 'FeatSimLossV2', 'AdvLoss', 'EntropyLoss',
    'PseudoLabelLoss', 'AdaptiveFeatSimLoss', 'MultiScaleAdaptiveFeatSimLoss',
    'AdaptiveFeatSimLossV2', 'AdaptiveFeatSimLossV3', 'PFSTLoss', 'PFGSTLoss'
]
