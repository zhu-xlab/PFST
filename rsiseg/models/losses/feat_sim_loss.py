# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
import pdb


@LOSSES.register_module()
class FeatSimLoss(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, sigmas, weights, sim_type='gaussian'):
        super(FeatSimLoss, self).__init__()

        self.top_k = top_k
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.sigmas = sigmas
        self.weights = weights
        self.sim_type = sim_type
        self.unfold_fun = nn.Unfold(kernel_size=self.kernel_size,
                                    padding=self.kernel_size // 2 * self.dilation,
                                    dilation=self.dilation)

    def forward(self, ori_feats_list, seg_logits):
        B, C, H, W = seg_logits.shape

        prob_map = F.softmax(seg_logits, dim=1)
        unf_prob_map = self.unfold_fun(prob_map)
        unf_prob_map = unf_prob_map.view(B, -1, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

        p = prob_map.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2)
        q = unf_prob_map # (B, c, h, w, k)

        sim_prob_map = torch.gather(p, 1, q.max(dim=1, keepdim=True)[1])
        sim_prob_map = sim_prob_map.squeeze(1).permute(0, 3, 1, 2) ** 2

        cross_prob_map = p.unsqueeze(2) * q.unsqueeze(1) # (B, C, C, h, w, k)
        cross_prob_map = cross_prob_map.permute(0, 5, 3, 4, 1, 2)
        cross_prob_map_diag = (p * q) # (B, C, h, w, k)

        # diag_mask = torch.eye(C).view(1,1,1,1,C,C).repeat(B,
        #                                                   self.kernel_size**2,
        #                                                   H, W, 1, 1).bool().to(cross_prob_map.device)

        # cross_prob_pos_mat = cross_prob_map[diag_mask].view(B, self.kernel_size**2, H, W, -1)
        # cross_prob_neg_mat = cross_prob_map[diag_mask == False].view(B, self.kernel_size**2, H, W, -1)
        # cross_prob_pos = (cross_prob_pos_mat).sum(dim=-1)
        # cross_prob_neg = (cross_prob_neg_mat).sum(dim=-1)
        cross_prob_pos = (cross_prob_map_diag).sum(dim=1).permute(0,3,1,2)
        cross_prob_neg = cross_prob_map.sum(dim=[-2,-1]) - cross_prob_pos

        losses = {}
        for idx, ori_feats in enumerate(ori_feats_list):

            channels = ori_feats.shape[1]
            feats = F.interpolate(ori_feats, size=(H, W), mode='nearest')
            unf_feats = self.unfold_fun(feats)
            unf_feats = unf_feats.view(B, channels, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

            if self.sim_type == 'gaussian':
                temp_dis = ((unf_feats - feats.unsqueeze(4)) ** 2).sum(dim=1)
                sim_feat = torch.exp(- temp_dis  / self.sigmas[idx] ** 2).permute(0, 3, 1, 2)

            elif self.sim_type == 'cosine':

                sim_feat = F.cosine_similarity(unf_feats, feats.unsqueeze(4), dim=1)
                sim_feat = sim_feat.permute(0, 3, 1, 2)

            else:
                raise ValueError()

            _, top_idx_max = torch.topk(sim_feat, self.top_k+1, dim=1)
            _, top_idx_min = torch.topk(sim_feat, self.top_k, dim=1, largest=False)

            max_sim_feat = torch.gather(sim_feat, 1, top_idx_max)
            min_sim_feat = torch.gather(sim_feat, 1, top_idx_min)

            cross_prob_pos_gather = torch.gather(cross_prob_pos, 1, top_idx_max)
            cross_prob_neg_gather = torch.gather(cross_prob_neg, 1, top_idx_min)

            loc_pos = max_sim_feat * (- cross_prob_pos_gather)
            loc_neg = (1 - min_sim_feat) * (- cross_prob_neg_gather)

            mask = feats[:, 0, :, :] > 0 # (B, H, W)
            pos_mask = mask.unsqueeze(1).repeat(1, self.top_k+1, 1, 1)
            neg_mask = mask.unsqueeze(1).repeat(1, self.top_k, 1, 1)

            loss_sim_pos = loc_pos[pos_mask].mean() * self.weights[idx][0]
            loss_sim_neg = loc_neg[neg_mask].mean() * self.weights[idx][1]

            losses.update({f'loss_sim_pos_{idx}': loss_sim_pos,
                          f'loss_sim_neg_{idx}': loss_sim_neg})

        states = dict(
            sim_feat=sim_feat.mean(dim=1).detach())

        return losses, states

@LOSSES.register_module()
class FeatSimLossV2(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, sigmas, weights):
        super(FeatSimLossV2, self).__init__()

        self.top_k = top_k
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.sigmas = sigmas
        self.weights = weights
        self.unfold_fun = nn.Unfold(kernel_size=self.kernel_size,
                                    padding=self.kernel_size // 2 * self.dilation,
                                    dilation=self.dilation)

    def forward(self, ori_sim_feats_list, seg_logits):
        B, C, H, W = seg_logits.shape

        prob_map = F.softmax(seg_logits, dim=1)
        unf_prob_map = self.unfold_fun(prob_map)
        unf_prob_map = unf_prob_map.view(B, -1, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

        p = prob_map.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2)
        q = unf_prob_map # (B, c, h, w, k)

        sim_prob_map = torch.gather(p, 1, q.max(dim=1, keepdim=True)[1])
        sim_prob_map = sim_prob_map.squeeze(1).permute(0, 3, 1, 2) ** 2

        cross_prob_map = p.unsqueeze(2) * q.unsqueeze(1) # (B, C, C, h, w, k)
        cross_prob_map = cross_prob_map.permute(0, 5, 3, 4, 1, 2)
        diag_mask = torch.eye(C).view(1,1,1,1,C,C).repeat(B,
                                                          self.kernel_size**2,
                                                          H, W, 1, 1).bool().to(cross_prob_map.device)

        cross_prob_pos_mat = cross_prob_map[diag_mask].view(B, self.kernel_size**2, H, W, -1)
        cross_prob_neg_mat = cross_prob_map[diag_mask == False].view(B, self.kernel_size**2, H, W, -1)
        cross_prob_pos = (cross_prob_pos_mat).sum(dim=-1)
        cross_prob_neg = (cross_prob_neg_mat).sum(dim=-1)

        loss_loc_pos = 0
        loss_loc_neg = 0
        losses = {}

        for idx, ori_sim_feats in enumerate(ori_sim_feats_list):

            # channels = ori_feats.shape[1]

            sim_feat = F.interpolate(ori_sim_feats, size=(H, W), mode='nearest')

            # unf_feats = self.unfold_fun(feats)
            # unf_feats = unf_feats.view(B, channels, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

            # temp_dis = ((unf_feats - feats.unsqueeze(4)) ** 2).sum(dim=1)
            # sim_feat = torch.exp(- temp_dis  / self.sigmas[idx] ** 2).permute(0, 3, 1, 2)

            _, top_idx_max = torch.topk(sim_feat, self.top_k+1, dim=1)
            _, top_idx_min = torch.topk(sim_feat, self.top_k, dim=1, largest=False)

            max_sim_feat = torch.gather(sim_feat, 1, top_idx_max)
            min_sim_feat = torch.gather(sim_feat, 1, top_idx_min)

            cur_cross_prob_pos = torch.gather(cross_prob_pos, 1, top_idx_max)
            cur_cross_prob_neg = torch.gather(cross_prob_neg, 1, top_idx_min)

            loc_pos = max_sim_feat * (- cur_cross_prob_pos)
            loc_neg = (1 - min_sim_feat) * (- cur_cross_prob_neg)

            # loss_loc_pos += loc_pos.mean() * self.weight[idx]
            # loss_loc_neg += loc_neg.mean() * self.weight[idx]
            mask = sim_feat[:, 0, :, :] > 0 # (B, H, W)
            pos_mask = mask.unsqueeze(1).repeat(1, self.top_k+1, 1, 1)
            neg_mask = mask.unsqueeze(1).repeat(1, self.top_k, 1, 1)

            loss_sim_pos = loc_pos[pos_mask].mean() * self.weights[idx][0]
            loss_sim_neg = loc_neg[neg_mask].mean() * self.weights[idx][1]

            losses.update({f'loss_sim_pos_{idx}': loss_sim_pos,
                          f'loss_sim_neg_{idx}': loss_sim_neg})

        states = dict(
            sim_feat=ori_sim_feats_list[0].mean(dim=1).detach())

        return losses, states

@LOSSES.register_module()
class AdaptiveFeatSimLoss(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='gaussian', num_bins=100, apply_ignore=False):
        super(AdaptiveFeatSimLoss, self).__init__()

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

    # def forward(self, ori_feats_list, seg_logits):
    def forward(self, tensors):

        logits_src = tensors['logits_src'].detach()
        logits_trg = tensors['logits_trg']
        gt_src = tensors['gt_src']
        x_ema = tensors['x_ema'][self.feat_level] if self.feat_level is not None else tensors['x_ema']
        x_src = tensors['x_src'][self.feat_level] if self.feat_level is not None else tensors['x_src']
        # x_trg = tensors['x_trg'][self.feat_level] if self.feat_level is not None else tensors['x_trg']
        img_trg = tensors['img_trg']
        losses = {}
        B, C, H, W = logits_trg.shape
        gt_src_ = F.interpolate(gt_src.float(), size=(H, W), mode='nearest')
        ignore_mask = gt_src_ != 255 if self.apply_ignore else None


        src_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_src) # (B, C, H, W, k)
        trg_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_trg) # (B, C, H, W, k)

        x_ema, ema_sim_feat = self.get_sim_feat(x_ema, size=(H, W)) # (B, k, H, W)
        _, src_sim_feat = self.get_sim_feat(x_src, size=(H, W)) 
        unf_gt_src = self.unfold_fun(gt_src_.float())
        unf_gt_src = unf_gt_src.view(-1, self.kernel_size**2, H, W).long()
        rep_gt_src = gt_src_.repeat(1, self.kernel_size**2, 1, 1)

        pos_gt_pair = unf_gt_src == rep_gt_src
        neg_gt_pair = unf_gt_src != rep_gt_src

        if ignore_mask is None:
            src_pos_sim = src_sim_feat[pos_gt_pair]
            src_neg_sim = src_sim_feat[neg_gt_pair]
        else:
            src_pos_sim = src_sim_feat[pos_gt_pair & ignore_mask.repeat(1, pos_gt_pair.shape[1], 1, 1)]
            src_neg_sim = src_sim_feat[neg_gt_pair & ignore_mask.repeat(1, neg_gt_pair.shape[1], 1, 1)]


        loss_sim_pos, loss_sim_neg = self.get_sim_losses(x_ema, ema_sim_feat, trg_cross_prob_map_diag, ignore_mask)

        losses.update({
            'loss_src_pos': - src_pos_sim.mean() * self.weights['src_pos'],
            'loss_src_neg': src_neg_sim.mean() * self.weights['src_neg'],
            'loss_sim_pos': loss_sim_pos * self.weights['sim_pos'],
            'loss_sim_neg': loss_sim_neg * self.weights['sim_neg'],
            # 'vis|hist_sim': (['pos', 'neg'], [src_pos_sim.detach().cpu(), src_neg_sim.detach().cpu()]),
            'vis|density_sim_feat': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1))

        })

        return losses

    def get_cross_prob_map_diag(self, logits):

        B, C, H, W = logits.shape
        prob_map = F.softmax(logits, dim=1)
        unf_prob_map = self.unfold_fun(prob_map)
        unf_prob_map = unf_prob_map.view(B, -1, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

        p = prob_map.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2)
        q = unf_prob_map # (B, c, h, w, k)

        sim_prob_map = torch.gather(p, 1, q.max(dim=1, keepdim=True)[1])
        sim_prob_map = sim_prob_map.squeeze(1).permute(0, 3, 1, 2) ** 2

        # cross_prob_map = p.unsqueeze(2) * q.unsqueeze(1) # (B, C, C, h, w, k)
        # cross_prob_map = cross_prob_map.permute(0, 5, 3, 4, 1, 2)
        cross_prob_map_diag = (p * q) # (B, C, h, w, k)

        # diag_mask = torch.eye(C).view(1,1,1,1,C,C).repeat(B,
        #                                                   self.kernel_size**2,
        #                                                   H, W, 1, 1).bool().to(cross_prob_map.device)
        # cross_prob_pos_mat = cross_prob_map[diag_mask].view(B, self.kernel_size**2, H, W, -1)
        # cross_prob_neg_mat = cross_prob_map[diag_mask == False].view(B, self.kernel_size**2, H, W, -1)
        # cross_prob_pos = (cross_prob_pos_mat).sum(dim=-1)
        # cross_prob_neg = (cross_prob_neg_mat).sum(dim=-1)
        # cross_prob_pos = (cross_prob_map_diag).sum(dim=1).permute(0,3,1,2)
        # cross_prob_neg = cross_prob_map.sum(dim=[-2,-1]) - cross_prob_pos

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
            loss_sim_pos = loc_pos[ignore_mask.repeat(1, loc_pos.shape[1], 1, 1)].mean()
            loss_sim_neg = loc_neg[ignore_mask.repeat(1, loc_neg.shape[1], 1, 1)].mean()
        else:
            loss_sim_pos = loc_pos.mean()
            loss_sim_neg = loc_neg.mean()

        return loss_sim_pos, loss_sim_neg

        """
        _, top_idx_max = torch.topk(sim_feat, self.top_k+1, dim=1)
        _, top_idx_min = torch.topk(sim_feat, self.top_k, dim=1, largest=False)

        max_sim_feat = torch.gather(sim_feat, 1, top_idx_max)
        min_sim_feat = torch.gather(sim_feat, 1, top_idx_min)

        cross_prob_pos_gather = torch.gather(cross_prob_pos, 1, top_idx_max)
        cross_prob_neg_gather = torch.gather(cross_prob_neg, 1, top_idx_min)

        loc_pos = max_sim_feat * (- cross_prob_pos_gather)
        loc_neg = (1 - min_sim_feat) * (- cross_prob_neg_gather)

        mask = feats[:, 0, :, :] > 0 # (B, H, W)
        pos_mask = mask.unsqueeze(1).repeat(1, self.top_k+1, 1, 1)
        neg_mask = mask.unsqueeze(1).repeat(1, self.top_k, 1, 1)

        loss_sim_pos = loc_pos[pos_mask].mean() * self.weights[0]
        loss_sim_neg = loc_neg[neg_mask].mean() * self.weights[1]

        losses.update({'loss_sim_pos': loss_sim_pos,
                       'loss_sim_neg': loss_sim_neg,
                       'vis|density_sim_feat': (img_trg, 1-sim_feat.mean(dim=1).detach().unsqueeze(1))})

        """

@LOSSES.register_module()
class MultiScaleAdaptiveFeatSimLoss(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=[2],
                 sim_type='gaussian', num_bins=100, apply_ignore=False):
        super(MultiScaleAdaptiveFeatSimLoss, self).__init__()

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

    # def forward(self, ori_feats_list, seg_logits):
    def forward(self, tensors):

        logits_src = tensors['logits_src'].detach()
        logits_trg = tensors['logits_trg']
        gt_src = tensors['gt_src']
        # x_trg = tensors['x_trg'][self.feat_level] if self.feat_level is not None else tensors['x_trg']
        img_trg = tensors['img_trg']
        losses = {}
        B, C, H, W = logits_trg.shape
        gt_src_ = F.interpolate(gt_src.float(), size=(H, W), mode='nearest')
        ignore_mask = gt_src_ != 255 if self.apply_ignore else None


        src_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_src) # (B, C, H, W, k)
        trg_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_trg) # (B, C, H, W, k)
        unf_gt_src = self.unfold_fun(gt_src_.float())
        unf_gt_src = unf_gt_src.view(-1, self.kernel_size**2, H, W).long()
        rep_gt_src = gt_src_.repeat(1, self.kernel_size**2, 1, 1)
        pos_gt_pair = unf_gt_src == rep_gt_src
        neg_gt_pair = unf_gt_src != rep_gt_src

        for level in self.feat_level:

            x_ema = tensors['x_ema'][level] if level is not None else tensors['x_ema']
            x_src = tensors['x_src'][level] if level is not None else tensors['x_src']

            x_ema, ema_sim_feat = self.get_sim_feat(x_ema, size=(H, W)) # (B, k, H, W)
            _, src_sim_feat = self.get_sim_feat(x_src, size=(H, W)) 


            if ignore_mask is None:
                src_pos_sim = src_sim_feat[pos_gt_pair]
                src_neg_sim = src_sim_feat[neg_gt_pair]
            else:
                src_pos_sim = src_sim_feat[pos_gt_pair & ignore_mask.repeat(1, pos_gt_pair.shape[1], 1, 1)]
                src_neg_sim = src_sim_feat[neg_gt_pair & ignore_mask.repeat(1, neg_gt_pair.shape[1], 1, 1)]


            loss_sim_pos, loss_sim_neg = self.get_sim_losses(x_ema, ema_sim_feat, trg_cross_prob_map_diag, ignore_mask)

            losses.update({
                f'loss_src_pos_{level}': - src_pos_sim.mean() * self.weights['src_pos'],
                f'loss_src_neg_{level}': src_neg_sim.mean() * self.weights['src_neg'],
                f'loss_sim_pos_{level}': loss_sim_pos * self.weights['sim_pos'],
                f'loss_sim_neg_{level}': loss_sim_neg * self.weights['sim_neg'],
                # 'vis|hist_sim': (['pos', 'neg'], [src_pos_sim.detach().cpu(), src_neg_sim.detach().cpu()]),
                f'vis|density_sim_feat_{level}': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1))

            })

        return losses

    def get_cross_prob_map_diag(self, logits):

        B, C, H, W = logits.shape
        prob_map = F.softmax(logits, dim=1)
        unf_prob_map = self.unfold_fun(prob_map)
        unf_prob_map = unf_prob_map.view(B, -1, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

        p = prob_map.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2)
        q = unf_prob_map # (B, c, h, w, k)

        sim_prob_map = torch.gather(p, 1, q.max(dim=1, keepdim=True)[1])
        sim_prob_map = sim_prob_map.squeeze(1).permute(0, 3, 1, 2) ** 2

        # cross_prob_map = p.unsqueeze(2) * q.unsqueeze(1) # (B, C, C, h, w, k)
        # cross_prob_map = cross_prob_map.permute(0, 5, 3, 4, 1, 2)
        cross_prob_map_diag = (p * q) # (B, C, h, w, k)

        # diag_mask = torch.eye(C).view(1,1,1,1,C,C).repeat(B,
        #                                                   self.kernel_size**2,
        #                                                   H, W, 1, 1).bool().to(cross_prob_map.device)
        # cross_prob_pos_mat = cross_prob_map[diag_mask].view(B, self.kernel_size**2, H, W, -1)
        # cross_prob_neg_mat = cross_prob_map[diag_mask == False].view(B, self.kernel_size**2, H, W, -1)
        # cross_prob_pos = (cross_prob_pos_mat).sum(dim=-1)
        # cross_prob_neg = (cross_prob_neg_mat).sum(dim=-1)
        # cross_prob_pos = (cross_prob_map_diag).sum(dim=1).permute(0,3,1,2)
        # cross_prob_neg = cross_prob_map.sum(dim=[-2,-1]) - cross_prob_pos

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
            loss_sim_pos = loc_pos[ignore_mask.repeat(1, loc_pos.shape[1], 1, 1)].mean()
            loss_sim_neg = loc_neg[ignore_mask.repeat(1, loc_neg.shape[1], 1, 1)].mean()
        else:
            loss_sim_pos = loc_pos.mean()
            loss_sim_neg = loc_neg.mean()

        return loss_sim_pos, loss_sim_neg

@LOSSES.register_module()
class AdaptiveFeatSimLossV2(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='gaussian', num_bins=100, apply_ignore=False):
        super(AdaptiveFeatSimLossV2, self).__init__()

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

    # def forward(self, ori_feats_list, seg_logits):
    def forward(self, tensors):

        logits_src = tensors['logits_src'].detach()
        logits_trg = tensors['logits_trg']
        gt_src = tensors['gt_src']
        x_ema = tensors['x_ema'][self.feat_level] if self.feat_level is not None else tensors['x_ema']
        x_src = tensors['x_src'][self.feat_level] if self.feat_level is not None else tensors['x_src']
        # x_trg = tensors['x_trg'][self.feat_level] if self.feat_level is not None else tensors['x_trg']
        img_trg = tensors['img_trg']
        losses = {}
        B, C, H, W = logits_trg.shape
        gt_src_ = F.interpolate(gt_src.float(), size=(H, W), mode='nearest')
        ignore_mask_src = gt_src_ != 255 if self.apply_ignore else None

        ignore_mask_trg = 1 - tensors['mix_masks']
        ignore_mask_trg = F.interpolate(ignore_mask_trg.float(), size=(H, W), mode='nearest') > 0.5

        trg_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_trg) # (B, C, H, W, k)

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


        loss_sim_pos, loss_sim_neg = self.get_sim_losses(x_ema, ema_sim_feat,
                                                         trg_cross_prob_map_diag,
                                                         ignore_mask_src & ignore_mask_trg)

        losses.update({
            'loss_src_pos': - src_pos_sim.mean() * self.weights['src_pos'],
            'loss_src_neg': src_neg_sim.mean() * self.weights['src_neg'],
            'loss_sim_pos': loss_sim_pos * self.weights['sim_pos'],
            'loss_sim_neg': loss_sim_neg * self.weights['sim_neg'],
            # 'vis|hist_sim': (['pos', 'neg'], [src_pos_sim.detach().cpu(), src_neg_sim.detach().cpu()]),
            'vis|density_sim_feat': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1),
                                     ignore_mask_trg)

        })

        return losses

    def get_cross_prob_map_diag(self, logits):

        B, C, H, W = logits.shape
        prob_map = F.softmax(logits, dim=1)
        unf_prob_map = self.unfold_fun(prob_map)
        unf_prob_map = unf_prob_map.view(B, -1, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

        p = prob_map.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2)
        q = unf_prob_map # (B, c, h, w, k)

        sim_prob_map = torch.gather(p, 1, q.max(dim=1, keepdim=True)[1])
        sim_prob_map = sim_prob_map.squeeze(1).permute(0, 3, 1, 2) ** 2

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
            loss_sim_pos = loc_pos[ignore_mask.repeat(1, loc_pos.shape[1], 1, 1)].mean()
            loss_sim_neg = loc_neg[ignore_mask.repeat(1, loc_neg.shape[1], 1, 1)].mean()
        else:
            loss_sim_pos = loc_pos.mean()
            loss_sim_neg = loc_neg.mean()

        return loss_sim_pos, loss_sim_neg

@LOSSES.register_module()
class AdaptiveFeatSimLossV3(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='cosine'):
        super(AdaptiveFeatSimLossV3, self).__init__()

        self.top_k = top_k
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weights = weights
        self.sim_type = sim_type
        self.feat_level = feat_level
        self.sigma = sigma
        self.unfold_fun = nn.Unfold(kernel_size=self.kernel_size,
                                    padding=self.kernel_size // 2 * self.dilation,
                                    dilation=self.dilation)

    # def forward(self, ori_feats_list, seg_logits):
    def forward(self, tensors):

        logits_trg = tensors['logits_trg']
        gt_src = tensors['gt_src']
        x_ema = tensors['x_ema'][self.feat_level] if self.feat_level is not None else tensors['x_ema']
        x_src = tensors['x_src'][self.feat_level] if self.feat_level is not None else tensors['x_src']
        # x_trg = tensors['x_trg'][self.feat_level] if self.feat_level is not None else tensors['x_trg']
        img_trg = tensors['img_trg']

        losses = {}
        B, C, H, W = logits_trg.shape
        gt_src_ = F.interpolate(gt_src.float(), size=(H, W), mode='nearest')
        ignore_mask_src = gt_src_ != 255
        ignore_mask_trg = 1 - tensors['mix_masks']
        ignore_mask_trg = F.interpolate(ignore_mask_trg.float(), size=(H, W), mode='nearest') > 0.5

        # trg_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_trg) # (B, C, H, W, k)

        # Increasing the similarity margins for the source domain
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

        # Similarity loss on the target domain
        unf_logits_trg = self.unfold_fun(logits_trg).view(B, C, self.kernel_size**2, H, W) # (B, C, k, H, W)
        x_ema, ema_sim_feat = self.get_sim_feat(x_ema, size=(H, W)) # (B, k, H, W)

        pdb.set_trace()
        loss_sim_pos, loss_sim_neg = self.get_sim_losses(
            x_ema, ema_sim_feat, unf_logits_trg,
            ignore_mask_trg
        )

        losses.update({
            'loss_src_pos': - src_pos_sim.mean() * self.weights['src_pos'],
            'loss_src_neg': src_neg_sim.mean() * self.weights['src_neg'],
            'loss_sim_pos': loss_sim_pos * self.weights['sim_pos'],
            'loss_sim_neg': loss_sim_neg * self.weights['sim_neg'],
            # 'vis|hist_sim': (['pos', 'neg'], [src_pos_sim.detach().cpu(), src_neg_sim.detach().cpu()]),
            'vis|density_sim_feat': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1),
                                     ignore_mask_trg)

        })

        return losses

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

    def get_sim_losses(self, feats, sim_feat, unf_logits, ignore_mask=None):

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

        # ddf

        loc_pos = top_aug_sim_feat ** self.expo * (- top_sim_prob_map)
        loc_neg = (1 - top_aug_sim_feat) ** self.expo * top_sim_prob_map


        if ignore_mask is not None:
            loss_sim_pos = loc_pos[ignore_mask.repeat(1, loc_pos.shape[1], 1, 1)].mean()
            loss_sim_neg = loc_neg[ignore_mask.repeat(1, loc_neg.shape[1], 1, 1)].mean()
        else:
            loss_sim_pos = loc_pos.mean()
            loss_sim_neg = loc_neg.mean()

        return loss_sim_pos, loss_sim_neg


@LOSSES.register_module()
class AdaptiveFeatSimLossV4(nn.Module):
    # calculate src loss based on mixed features

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='gaussian', num_bins=100, apply_ignore=False):
        super(AdaptiveFeatSimLossV2, self).__init__()

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

    # def forward(self, ori_feats_list, seg_logits):
    def forward(self, tensors):

        logits_src = tensors['logits_src'].detach()
        logits_trg = tensors['logits_trg']
        gt_src = tensors['gt_src']
        x_ema = tensors['x_ema'][self.feat_level] if self.feat_level is not None else tensors['x_ema']
        x_src = tensors['x_src'][self.feat_level] if self.feat_level is not None else tensors['x_src']
        # x_trg = tensors['x_trg'][self.feat_level] if self.feat_level is not None else tensors['x_trg']
        img_trg = tensors['img_trg']
        losses = {}
        B, C, H, W = logits_trg.shape
        gt_src_ = F.interpolate(gt_src.float(), size=(H, W), mode='nearest')
        ignore_mask_src = gt_src_ != 255 if self.apply_ignore else None

        ignore_mask_trg = 1 - tensors['mix_masks']
        ignore_mask_trg = F.interpolate(ignore_mask_trg.float(), size=(H, W), mode='nearest') > 0.5

        trg_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_trg) # (B, C, H, W, k)

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


        loss_sim_pos, loss_sim_neg = self.get_sim_losses(x_ema, ema_sim_feat,
                                                         trg_cross_prob_map_diag,
                                                         ignore_mask_src & ignore_mask_trg)

        losses.update({
            'loss_src_pos': - src_pos_sim.mean() * self.weights['src_pos'],
            'loss_src_neg': src_neg_sim.mean() * self.weights['src_neg'],
            'loss_sim_pos': loss_sim_pos * self.weights['sim_pos'],
            'loss_sim_neg': loss_sim_neg * self.weights['sim_neg'],
            # 'vis|hist_sim': (['pos', 'neg'], [src_pos_sim.detach().cpu(), src_neg_sim.detach().cpu()]),
            'vis|density_sim_feat': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1),
                                     ignore_mask_trg)

        })

        return losses

    def get_cross_prob_map_diag(self, logits):

        B, C, H, W = logits.shape
        prob_map = F.softmax(logits, dim=1)
        unf_prob_map = self.unfold_fun(prob_map)
        unf_prob_map = unf_prob_map.view(B, -1, self.kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

        p = prob_map.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2)
        q = unf_prob_map # (B, c, h, w, k)

        sim_prob_map = torch.gather(p, 1, q.max(dim=1, keepdim=True)[1])
        sim_prob_map = sim_prob_map.squeeze(1).permute(0, 3, 1, 2) ** 2

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
            loss_sim_pos = loc_pos[ignore_mask.repeat(1, loc_pos.shape[1], 1, 1)].mean()
            loss_sim_neg = loc_neg[ignore_mask.repeat(1, loc_neg.shape[1], 1, 1)].mean()
        else:
            loss_sim_pos = loc_pos.mean()
            loss_sim_neg = loc_neg.mean()

        return loss_sim_pos, loss_sim_neg
