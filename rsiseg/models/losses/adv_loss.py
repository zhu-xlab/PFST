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
class AdvLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 loss_type='adv_ent',
                 net_type='gen',
                 weights=None,
                 **kwards):
        super(AdvLoss, self).__init__()
        self.loss_type = loss_type
        self.net_type = net_type
        self._loss_name = f'adv_loss_{loss_type}_{net_type}'
        self.weights = weights

    def prob2ent(self, prob):
        B, C, H, W = prob.shape
        ent = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(C)
        return ent

    def bce_loss(self, y_pred, y_label):
        y_truth_tensor = torch.FloatTensor(y_pred.size())
        y_truth_tensor.fill_(y_label)
        y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
        return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

    def l1_loss(self, y_pred, y_label):
        y_truth_tensor = torch.FloatTensor(y_pred.size())
        y_truth_tensor.fill_(y_label)
        y_truth_tensor = y_truth_tensor.to(y_pred.get_device())

        return F.l1_loss(y_pred, y_truth_tensor, reduction='mean')


    def forward(self, discriminator, tensors):
        losses = dict()

        if self.loss_type == 'advent':
            src_label = 0
            trg_label = 1

            if self.net_type == 'disc':
                logits_src = tensors['logits_src'].detach()
                logits_trg = tensors['logits_trg'].detach()

                prob_src = F.softmax(logits_src, dim=1)
                prob_trg = F.softmax(logits_trg, dim=1)

                ent_src = self.prob2ent(prob_src)
                ent_trg = self.prob2ent(prob_trg)

                d_out_src = discriminator(ent_src)
                d_out_trg = discriminator(ent_trg)

                # loss_disc_src = self.bce_loss(d_out_src, src_label)
                # loss_disc_trg = self.bce_loss(d_out_trg, trg_label)
                loss_disc_src = self.l1_loss(d_out_src, src_label)
                loss_disc_trg = self.l1_loss(d_out_trg, trg_label)
                losses['loss_disc_src'] = loss_disc_src * self.weights['loss_disc_src']
                losses['loss_disc_trg'] = loss_disc_trg * self.weights['loss_disc_trg']

            elif self.net_type == 'gen':

                logits_trg = tensors['logits_trg']
                prob_trg = F.softmax(logits_trg, dim=1)
                ent_trg = self.prob2ent(prob_trg)
                d_out_trg = discriminator(ent_trg)

                # loss_gen = self.bce_loss(d_out_trg, src_label)
                loss_gen = self.l1_loss(d_out_trg, src_label)
                losses['loss_gen'] = loss_gen * self.weights['loss_gen']

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
