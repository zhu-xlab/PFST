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
class PseudoLabelLoss(nn.Module):
    """
    Minimize Entropy Loss.
    """

    def __init__(self,
                 loss_type='entropy',
                 weights=None,
                 **kwards):
        super(PseudoLabelLoss, self).__init__()
        self.loss_type = loss_type
        self._loss_name = f'loss_{loss_type}'
        self.weights = weights

    def forward(self, tensors):
        losses = dict()

        img_trg = tensors['img_trg']
        img_metas_trg = tensors['img_metas_trg']

        logits_trg = tensors['logits_trg'].detach()
        aux_seg_net = tensors['aux_seg_net']
        aux_seg_net.eval()
        pdb.set_trace()
        with torch.no_grad():
            logits_pseudo = aux_seg_net([img_trg], [img_metas_trg], return_loss=False)

        logits_pseudo = self.transform_by_metas(logits_pseudo, 1.0)
        pseudo_labels = logits_pseudo.max(dim=1)[1]
        losses['loss_pseudo'] = F.cross_entropy_loss(logits_trg, pseudo_labels) * self.weights['loss_pseudo']

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

    def proportional_crop(self, data, crop_bbox, scale):
        """Crop from ``img``"""
        rescale = lambda x: int(x * scale)
        crop_y1, crop_y2, crop_x1, crop_x2 = map(rescale, crop_bbox)
        data = data[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        return data

    def transform_by_metas(self, data, metas, scale=1/8.):
        # data: (H, W, ...)

        if 'scale_factor' in metas:
            w_scale, h_scale, _, _ = metas['scale_factor']
            # H, W, C = metas['ori_shape']
            # new_h, new_w = int(H * h_scale), int(W * w_scale)
            # data = F.interpolate(data, size=(new_h, new_w), mode='nearest')
            data = F.interpolate(data, scale_factor=(w_scale, h_scale), mode='bilinear')

        if 'crop_bbox' in metas:
            w_scale, h_scale, _, _ = metas['scale_factor']
            assert w_scale == h_scale
            data = self.proportional_crop(data, metas['crop_bbox'], scale)

            H, W, C = metas['ori_shape']
            new_h, new_w = int(H * h_scale), int(W * w_scale)

            data_h, data_w = data.shape[-2:]

        if 'rotate_k' in metas:
            data = torch.rot90(data, metas['rotate_k'], dims=[2,3])

        if metas['flip']:
            if 'horizontal' in metas['flip_direction']:
                data = data.flip(dims=[3])

            if 'vertical' in metas['flip_direction']:
                data = data.flip(dims=[2])

        if 'pad_shape' in metas:

            _, _, H, W = data.shape
            pad_H, pad_W = metas['pad_shape'][:2]
            pad_H = int(pad_H * scale)
            pad_W = int(pad_W * scale)

            if pad_H != H or pad_W != W:
                data = F.pad(data, (0, pad_W - W, 0, pad_W - W), 'constant', -1) # ignore negative value when process the data

        return data


