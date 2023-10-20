# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
import pdb
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch.nn as nn
import torch.nn.functional as F

from rsiseg.core import add_prefix
from rsiseg.models import UDA, build_segmentor, builder
from rsiseg.models.uda.uda_decorator import UDADecorator, get_module
from rsiseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class PFST(UDADecorator):

    def __init__(self, **cfg):
        super(PFST, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.trg_loss_weight = cfg.get('trg_loss_weight', 1.)
        self.feat_level = cfg.get('feat_level', 2)
        assert self.mix == 'class'

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        aux_losses = cfg.get('aux_losses', None)
        self.apply_aux = False
        if aux_losses is not None:
            self.apply_aux = True
            if not type(aux_losses) == list:
                aux_losses = [aux_losses]

            aux_losses = [builder.build_loss(loss) for loss in aux_losses]
            self.aux_losses = nn.ModuleList(aux_losses)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars, vis_states = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=vis_states
        )
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'rsiseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'rsiseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'rsiseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'rsiseg')
        return torch.mean(pw_feat_dist)

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas, target_img_strong_aug):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsiseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        vis_states = {}
        total_loss = 0
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feats=True, return_logits=True)
        src_feats = clean_losses.pop('features')

        src_logits = clean_losses.pop('logits')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)

        total_loss += clean_loss
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'rsiseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits, ema_states = self.get_ema_model().encode_decode(
            target_img, target_img_metas)
        ema_feats = ema_states['feats']

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)

        # Don't trust pseudo-labels in regions with potential
        # rectification artifacts. This can lead to a pseudo-label
        # drift from sky towards building or traffic light.
        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img_strong_aug[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i]))
            )
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i]))
            )
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feats=True, return_logits=True)
        mixed_feats = mix_losses.pop('features')
        mixed_logits = mix_losses.pop('logits')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        total_loss += mix_loss * self.trg_loss_weight

        tensors = dict(
            img_src=img,
            img_src_metas=img_metas,
            # img_trg=target_img,
            img_trg=mixed_img,
            img_mixed=mixed_img,
            img_metas_trg=target_img_metas,
            gt_src=gt_semantic_seg,
            x_src=src_feats,
            x_ema=ema_feats,
            # x_ema = mixed_ema_feats,
            # x_trg = trg_feats,
            x_trg = mixed_feats,
            logits_src=src_logits,
            logits_trg=mixed_logits,
            logits_ema=ema_logits,
            mix_masks=torch.cat(mix_masks, dim=0)
            # logits_trg=trg_logits
        )

        if self.apply_aux:
            aux_losses = self._get_aux_losses(tensors=tensors)
            vis_states.update({name: value for name, value in aux_losses.items() if name.startswith('vis|')})
            for name in vis_states.keys(): aux_losses.pop(name)

            aux_losses, aux_log_vars = self._parse_losses(aux_losses)
            log_vars.update(aux_log_vars)
            # aux_losses.backward()

            total_loss += aux_losses

        total_loss.backward()

        vis_mask_mix = mixed_lbl
        vis_pseudo_weight = F.interpolate(pseudo_weight.unsqueeze(1), mixed_lbl.shape[2:])
        vis_mask_mix = torch.where(vis_pseudo_weight > 0.5, vis_mask_mix, 255)
        vis_states.update({
            'vis|seg_mask_src': (img, gt_semantic_seg, src_logits.max(dim=1)[1].unsqueeze(1)),
            'vis|seg_mask_mix': (mixed_img, vis_mask_mix, mixed_logits.max(dim=1)[1].unsqueeze(1).float()),
        })

        self.local_iter += 1

        return log_vars, vis_states

    def _get_aux_losses(self, tensors):

        aux_losses = dict()
        for loss_module in self.aux_losses:
            loss_ = loss_module(tensors)
            if loss_ is None:
                continue
            aux_losses.update(loss_)


        return aux_losses
