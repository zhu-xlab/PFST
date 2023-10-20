# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
import pdb

@LOSSES.register_module()
class PFGSTLoss(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='gaussian', num_bins=100, apply_ignore=False, src_perc=None,
                 proj_net_cfg=None, src_loss_type='mean_std', margin=[0.5, 0.5],
                 detach_unfold=False, cross_prob_type='trg', downscale=None):
        super(PFGSTLoss, self).__init__()

        self.top_k = top_k
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weights = weights
        self.sim_type = sim_type
        self.feat_level = feat_level
        self.num_bins = num_bins
        self.sigma = sigma
        self.unfold_fun = nn.Unfold(kernel_size=self.kernel_size,
                                    padding=self.kernel_size // 2 * self.dilation,
                                    dilation=self.dilation)
        self.apply_ignore = apply_ignore
        self.src_perc = src_perc
        self.proj_net = None
        if proj_net_cfg is not None:
            self.proj_net = nn.Conv2d(proj_net_cfg['in_channels'], proj_net_cfg['out_channels'], kernel_size=1)
        self.src_loss_type = src_loss_type
        self.margin=margin
        self.detach_unfold = detach_unfold
        self.cross_prob_type = cross_prob_type
        self.downscale=downscale

    # def forward(self, ori_feats_list, seg_logits):
    def forward(self, tensors):

        logits_trg = tensors['logits_trg']
        logits_ema = tensors['logits_ema']

        gt_src = tensors['gt_src']
        x_ema = tensors['x_ema'][self.feat_level] if self.feat_level is not None else tensors['x_ema']
        x_src = tensors['x_src'][self.feat_level] if self.feat_level is not None else tensors['x_src']
        # x_trg = tensors['x_trg'][self.feat_level] if self.feat_level is not None else tensors['x_trg']
        img_trg = tensors['img_trg']
        losses = {}

        if self.downscale is not None:
            logits_trg = F.interpolate(logits_trg, scale_factor=(self.downscale, self.downscale))
            x_ema = F.interpolate(x_ema, size=logits_trg.shape[2:])
            x_src = F.interpolate(x_src, size=logits_trg.shape[2:])

        B, C, H, W = logits_trg.shape
        gt_src_ = F.interpolate(gt_src.float(), size=(H, W), mode='nearest')
        # ignore_mask_src = gt_src_ != 255 if self.apply_ignore else None
        ignore_mask_src = gt_src_ != 255

        ignore_mask_trg = 1 - tensors['mix_masks']
        ignore_mask_trg = F.interpolate(ignore_mask_trg.float(), size=(H, W), mode='nearest') > 0.5

        unf_ignore_mask_trg = self.unfold_fun(ignore_mask_trg.float())
        unf_ignore_mask_trg = unf_ignore_mask_trg.view(-1, self.kernel_size**2, H, W).long()
        ignore_mask_trg = unf_ignore_mask_trg.sum(dim=1).unsqueeze(1) == self.kernel_size ** 2

        if self.proj_net is not None:
            x_src = self.proj_net(x_src)
            x_ema = self.proj_net(x_ema)

        if self.cross_prob_type == 'trg':
            trg_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_trg) # (B, C, H, W, k)
        elif self.cross_prob_type == 'ema':
            trg_cross_prob_map_diag = self.get_cross_prob_map_diag_ema(logits_trg, logits_ema) # (B, C, H, W, k)

        x_ema, ema_sim_feat = self.get_sim_feat(x_ema, size=(H, W)) # (B, k, H, W)
        _, src_sim_feat = self.get_sim_feat(x_src, size=(H, W)) 

        unf_gt_src = self.unfold_fun(gt_src_.float())
        unf_gt_src = unf_gt_src.view(-1, self.kernel_size**2, H, W).long()
        rep_gt_src = gt_src_.repeat(1, self.kernel_size**2, 1, 1)

        pos_gt_pair = unf_gt_src == rep_gt_src
        neg_gt_pair = unf_gt_src != rep_gt_src

        if ignore_mask_src is None:
            src_pos_sim = src_sim_feat[pos_gt_pair]
            src_neg_sim = src_sim_feat[neg_gt_pair]
        else:
            src_pos_sim = src_sim_feat[pos_gt_pair & ignore_mask_src.repeat(1, pos_gt_pair.shape[1], 1, 1)]
            src_neg_sim = src_sim_feat[neg_gt_pair & ignore_mask_src.repeat(1, neg_gt_pair.shape[1], 1, 1)]
            if self.src_perc is not None:
                sorted_src_pos_sim = src_pos_sim.sort()[0]
                sorted_src_neg_sim = src_neg_sim.sort(descending=True)[0]
                src_pos_sim = sorted_src_pos_sim[:int(sorted_src_pos_sim.shape[0] * self.src_perc)]
                src_neg_sim = sorted_src_neg_sim[:int(sorted_src_neg_sim.shape[0] * self.src_perc)]

        loss_sim_pos, loss_sim_neg = self.get_sim_losses(x_ema, ema_sim_feat,
                                                         trg_cross_prob_map_diag,
                                                         ignore_mask_src & ignore_mask_trg)

        if self.src_loss_type == 'mean_std':
            losses.update({
                'loss_src_pos_mean': - src_pos_sim.mean() * self.weights['src_pos'],
                'loss_src_neg_mean': src_neg_sim.mean() * self.weights['src_neg'],
                'loss_src_pos_std': src_pos_sim.std() * self.weights['src_pos_std'],
                'loss_src_neg_std': src_neg_sim.std() * self.weights['src_neg_std'],

            })
        elif self.src_loss_type == 'margin':
            loss_src_pos_sim = F.relu(self.margin[0] - src_pos_sim).mean()
            loss_src_neg_sim = F.relu(src_neg_sim - self.margin[1]).mean()

            losses.update({
                'loss_src_pos': loss_src_pos_sim * self.weights['src_pos'],
                'loss_src_neg': loss_src_neg_sim * self.weights['src_neg'],
            })
        elif self.src_loss_type == 'margin2':
            loss_src_pos_sim = (F.relu(self.margin[0] - src_pos_sim) ** 2).mean()
            loss_src_neg_sim = (F.relu(src_neg_sim - self.margin[1]) ** 2).mean()

            losses.update({
                'loss_src_pos': loss_src_pos_sim * self.weights['src_pos'],
                'loss_src_neg': loss_src_neg_sim * self.weights['src_neg'],
            })

        losses.update({
            'loss_sim_pos': loss_sim_pos * self.weights['sim_pos'],
            'loss_sim_neg': loss_sim_neg * self.weights['sim_neg'],
            'vis|density_sim_feat': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1),
                                     ignore_mask_trg)
        })

        return losses

    def get_cross_prob_map_diag(self, logits):

        B, C, H, W = logits.shape
        prob_map = F.softmax(logits, dim=1)
        unf_prob_map = self.unfold_fun(prob_map)
        if self.detach_unfold:
            unf_prob_map = unf_prob_map.detach()
        unf_prob_map = unf_prob_map.view(B, -1, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

        p = prob_map.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2)
        q = unf_prob_map # (B, c, h, w, k)

        sim_prob_map = torch.gather(p, 1, q.max(dim=1, keepdim=True)[1])
        sim_prob_map = sim_prob_map.squeeze(1).permute(0, 3, 1, 2) ** 2

        cross_prob_map_diag = (p * q) # (B, C, h, w, k)

        return cross_prob_map_diag

    def get_cross_prob_map_diag_ema(self, logits_trg, logits_ema):

        B, C, H, W = logits_trg.shape
        prob_map_trg = F.softmax(logits_trg, dim=1)
        prob_map_ema = F.softmax(logits_ema, dim=1)

        unf_prob_map_ema = self.unfold_fun(prob_map_ema)
        unf_prob_map_ema = unf_prob_map_ema.view(B, -1, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

        p = prob_map_trg.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2)
        q = unf_prob_map_ema # (B, c, h, w, k)

        # sim_prob_map = torch.gather(p, 1, q.max(dim=1, keepdim=True)[1])
        # sim_prob_map = sim_prob_map.squeeze(1).permute(0, 3, 1, 2) ** 2

        cross_prob_map_diag = (p * q) # (B, C, h, w, k)

        return cross_prob_map_diag


    def get_sim_feat(self, x, size=None):

        B, channels, _, _ = x.shape
        if size is not None:
            feats = F.interpolate(x, size=size, mode='nearest')
        unf_feats = self.unfold_fun(feats)
        unf_feats = unf_feats.view(B, channels, self.kernel_size**2, size[0], size[1]).permute(0, 1, 3, 4, 2)

        if self.sim_type == 'gaussian':
            temp_dis = ((unf_feats - feats.unsqueeze(4)) ** 2).sum(dim=1)
            sim_feat = torch.exp(- temp_dis  / self.sigma ** 2).permute(0, 3, 1, 2)

        elif self.sim_type == 'cosine':

            sim_feat = F.cosine_similarity(unf_feats, feats.unsqueeze(4), dim=1)
            sim_feat = sim_feat.permute(0, 3, 1, 2)

        else:
            raise ValueError()

        return feats, sim_feat

    def get_sim_losses(self, feats, sim_feat, cross_prob_map_diag, ignore_mask=None):

        cross_prob_pos = (cross_prob_map_diag).sum(dim=1).permute(0,3,1,2)
        cross_prob_neg = 1 - cross_prob_pos

        if self.top_k is not None:
            _, top_idx_max = torch.topk(sim_feat, self.top_k+1, dim=1)
            _, top_idx_min = torch.topk(sim_feat, self.top_k, dim=1, largest=False)

            max_sim_feat = torch.gather(sim_feat, 1, top_idx_max)
            min_sim_feat = torch.gather(sim_feat, 1, top_idx_min)

            cross_prob_pos_gather = torch.gather(cross_prob_pos, 1, top_idx_max)
            cross_prob_neg_gather = torch.gather(cross_prob_neg, 1, top_idx_min)

            loc_pos = max_sim_feat * (- cross_prob_pos_gather)
            loc_neg = (1 - min_sim_feat) * (- cross_prob_neg_gather)
        else:
            loc_pos = sim_feat * (- cross_prob_pos)
            loc_neg = (1 - sim_feat) * (- cross_prob_neg)

        if ignore_mask is not None:
            loss_sim_pos = torch.zeros(1).cuda()
            loss_sim_neg = torch.zeros(1).cuda()
            if ignore_mask.sum() > 1:
                loss_sim_pos = loc_pos[ignore_mask.repeat(1, loc_pos.shape[1], 1, 1)].mean()
                loss_sim_neg = loc_neg[ignore_mask.repeat(1, loc_neg.shape[1], 1, 1)].mean()
        else:
            loss_sim_pos = loc_pos.mean()
            loss_sim_neg = loc_neg.mean()

        return loss_sim_pos, loss_sim_neg
