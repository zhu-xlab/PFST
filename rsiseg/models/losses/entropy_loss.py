# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from ..builder import LOSSES

@LOSSES.register_module()
class EntropyLoss(nn.Module):
    """
    Minimize Entropy Loss.
    """

    def __init__(self,
                 loss_type='entropy',
                 weights=None,
                 **kwards):
        super(EntropyLoss, self).__init__()
        self.loss_type = loss_type
        self._loss_name = f'loss_{loss_type}'
        self.weights = weights

    def prob2ent(self, prob):
        B, C, H, W = prob.shape
        ent = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(C)
        return ent

    def forward(self, tensors):
        losses = dict()

        logits_trg = tensors['logits_trg']
        prob_trg = F.softmax(logits_trg, dim=1)

        if self.loss_type == 'entropy':
            ent_trg = self.prob2ent(prob_trg)
            losses['loss_ent'] = ent_trg.sum(dim=1).mean() * self.weights['loss_ent']
        elif self.loss_type == 'max_square':
            losses['loss_max_square'] = - (prob_trg ** 2).mean() / 2 * self.weights['loss_max_square']
        else:
            raise ValueError()

        return losses



    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
