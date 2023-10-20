import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
import pdb

@LOSSES.register_module()
class PFSTLoss(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='cosine'):
        super(PFSTLoss, self).__init__()

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
        self.loss_fun = torch.nn.BCEWithLogitsLoss(reduction='none')

    # def forward(self, ori_feats_list, seg_logits):
    def forward(self, tensors):

        logits_trg = tensors['logits_trg']
        # logits_ema = tensors['logits_ema']
        gt_src = tensors['gt_src']
        x_ema = tensors['x_ema'][self.feat_level] if self.feat_level is not None else tensors['x_ema']
        x_src = tensors['x_src'][self.feat_level] if self.feat_level is not None else tensors['x_src']
        # x_trg = tensors['x_trg'][self.feat_level] if self.feat_level is not None else tensors['x_trg']
        img_trg = tensors['img_trg']

        losses = {}
        B, C, H, W = logits_trg.shape
        # gt_src_ = F.interpolate(gt_src.float(), size=(H, W), mode='nearest')
        # ignore_mask_src = gt_src_ != 255
        ignore_mask_trg = 1 - tensors['mix_masks']
        ignore_mask_trg = F.interpolate(ignore_mask_trg.float(), size=(H, W), mode='nearest') > 0.5
        logits_ema = F.interpolate(tensors['logits_ema'], size=(H, W), mode='bilinear')

        # trg_cross_prob_map_diag = self.get_cross_prob_map_diag(logits_trg) # (B, C, H, W, k)
        # Increasing the similarity margins for the source domain

        # _, src_sim_feat = self.get_sim_feat(x_src, size=(H, W))
        # _, src_sim_feat = self.get_sim_feat(x_src, None)

        # unf_gt_src = self.unfold_fun(gt_src_.float())
        # unf_gt_src = unf_gt_src.view(-1, self.kernel_size**2, H, W).long()
        # rep_gt_src = gt_src_.repeat(1, self.kernel_size**2, 1, 1)

        # pos_gt_pair = unf_gt_src == rep_gt_src
        # neg_gt_pair = unf_gt_src != rep_gt_src

        # if ignore_mask_src is None:
        #     src_pos_sim = src_sim_feat[pos_gt_pair]
        #     src_neg_sim = src_sim_feat[neg_gt_pair]
        # else:
        #     src_pos_sim = src_sim_feat[pos_gt_pair & ignore_mask_src.repeat(1, pos_gt_pair.shape[1], 1, 1)]
        #     src_neg_sim = src_sim_feat[neg_gt_pair & ignore_mask_src.repeat(1, neg_gt_pair.shape[1], 1, 1)]

        # Similarity loss on the target domain
        unf_logits_ema = self.unfold_fun(logits_ema).view(B, C, self.kernel_size**2, H, W) # (B, C, k, H, W)
        x_ema, ema_sim_feat = self.get_sim_feat(x_ema, size=(H, W)) # (B, k, H, W)

        pseudo_labels_sim_pos, pseudo_labels_sim_neg = self.get_sim_pseudo_labels(
            x_ema, ema_sim_feat, unf_logits_ema,
        )

        loss_sim_pos = self.loss_fun(logits_trg, pseudo_labels_sim_pos)
        loss_sim_neg = - self.loss_fun(logits_trg, pseudo_labels_sim_neg)

        loss_sim_pos = loss_sim_pos[ignore_mask_trg.repeat(1, C, 1, 1)].mean()
        loss_sim_neg = loss_sim_neg[ignore_mask_trg.repeat(1, C, 1, 1)].mean()

        losses.update({
            # 'loss_src_pos': - src_pos_sim.mean() * self.weights['src_pos'],
            # 'loss_src_neg': src_neg_sim.mean() * self.weights['src_neg'],
            'loss_sim_pos': loss_sim_pos * self.weights['sim_pos'],
            'loss_sim_neg': loss_sim_neg * self.weights['sim_neg'],
            # 'vis|hist_sim': (['pos', 'neg'], [src_pos_sim.detach().cpu(), src_neg_sim.detach().cpu()]),
            'vis|density_sim_feat': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1)),
            'vis|seg_mask_sim_pseudo_labels': (img_trg,
                                               pseudo_labels_sim_pos.max(dim=1)[1].unsqueeze(1),
                                               pseudo_labels_sim_neg.max(dim=1)[1].unsqueeze(1)),

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

    def get_sim_pseudo_labels(self, feats, sim_feat, unf_logits):

        num_classes = unf_logits.shape[1]
        unf_probs = F.softmax(unf_logits, dim=1)
        unf_preds = unf_logits.max(dim=1)[1]

        _, top_idx_max = torch.topk(sim_feat, self.top_k+1, dim=1)
        _, top_idx_min = torch.topk(sim_feat, self.top_k, dim=1, largest=False)

        max_sim_feat = torch.gather(sim_feat, 1, top_idx_max)
        min_sim_feat = torch.gather(sim_feat, 1, top_idx_min)

        # unf_pseudo_labels_sim_pos = torch.gather(unf_preds, 1, top_idx_max)
        # unf_pseudo_labels_sim_neg = torch.gather(unf_preds, 1, top_idx_min)

        unf_pseudo_logits_pos = torch.gather(unf_logits, 2,
                                             top_idx_max.unsqueeze(1).repeat(1, num_classes, 1, 1, 1))
        unf_pseudo_logits_neg = torch.gather(unf_logits, 2,
                                             top_idx_min.unsqueeze(1).repeat(1, num_classes, 1, 1, 1))

        weighted_unf_pseudo_logits_pos = max_sim_feat.unsqueeze(1) * unf_pseudo_logits_pos
        weighted_unf_pseudo_logits_neg = min_sim_feat.unsqueeze(1) * unf_pseudo_logits_neg

        weighted_pseudo_logits_pos = weighted_unf_pseudo_logits_pos.sum(dim=2)
        weighted_pseudo_logits_neg = weighted_unf_pseudo_logits_neg.sum(dim=2)

        pseudo_prob_pos = F.softmax(weighted_pseudo_logits_pos, dim=1)
        pseudo_prob_neg = F.softmax(weighted_pseudo_logits_neg, dim=1)

        return pseudo_prob_pos, pseudo_prob_neg

@LOSSES.register_module()
class PFSTLossV2(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='gaussian', tau_pos=0.25, tau_neg=0.75, border_margin=None):
        super(PFSTLossV2, self).__init__()

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
        self.border_margin = border_margin
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

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
        ignore_mask_src = gt_src_ != 255
        logits_ema = F.interpolate(tensors['logits_ema'], size=(H, W), mode='bilinear')

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


        loss_sim_pos, loss_sim_neg, pos_sim_mask, neg_sim_mask = self.get_sim_losses(
            x_ema, ema_sim_feat,
            logits_trg, trg_cross_prob_map_diag,
            ignore_mask_src & ignore_mask_trg
        )

        losses.update({
            'loss_src_pos': - src_pos_sim.mean() * self.weights['src_pos'],
            'loss_src_neg': src_neg_sim.mean() * self.weights['src_neg'],
            'loss_sim_pos': loss_sim_pos * self.weights['sim_pos'],
            'loss_sim_neg': loss_sim_neg * self.weights['sim_neg'],
            # 'vis|hist_sim': (['pos', 'neg'], [src_pos_sim.detach().cpu(), src_neg_sim.detach().cpu()]),
            'vis|density_sim_feat': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1)),
            'vis|seg_mask_sim': (img_trg, (pos_sim_mask.sum(dim=1).unsqueeze(1) > 0).long(),
                                 (neg_sim_mask.sum(dim=1).unsqueeze(1) > 0).long())
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

    def get_sim_losses(self, feats, sim_feat, logits, cross_prob_map_diag, ignore_mask=None):

        B, C, H, W = logits.shape
        pred = logits.max(dim=1)[1].unsqueeze(1)
        unf_pred = self.unfold_fun(pred.float())
        unf_pred = unf_pred.view(-1, self.kernel_size**2, H, W).long()
        rep_pred = pred.repeat(1, self.kernel_size**2, 1, 1)

        pos_pred_pair = unf_pred == rep_pred
        neg_pred_pair = unf_pred != rep_pred

        cross_prob_pos = (cross_prob_map_diag).sum(dim=1).permute(0,3,1,2)
        cross_prob_neg = 1 - cross_prob_pos

        pos_sim_feat = torch.where(pos_pred_pair, sim_feat, 255)
        neg_sim_feat = torch.where(neg_pred_pair, sim_feat, -255)

        pos_sim_mask = (sim_feat < self.tau_pos) & pos_pred_pair
        neg_sim_mask = (sim_feat > self.tau_neg) & neg_pred_pair

        if ignore_mask is not None:
            unf_ignore_mask = self.unfold_fun(ignore_mask.float())
            unf_ignore_mask = unf_ignore_mask.view(-1, self.kernel_size**2, H, W).long()
            ignore_mask = unf_ignore_mask.sum(dim=1).unsqueeze(1) == self.kernel_size ** 2
            if self.border_margin is not None:
                ignore_mask[:, :, :self.border_margin, :self.border_margin] = False
                ignore_mask[:, :, -self.border_margin:, -self.border_margin:] = False

            pos_sim_mask = pos_sim_mask & ignore_mask
            neg_sim_mask = neg_sim_mask & ignore_mask

        loss_sim_pos = torch.zeros(1).cuda()
        loss_sim_neg = torch.zeros(1).cuda()

        if pos_sim_mask.sum() > 0:
            loss_sim_pos = cross_prob_pos[pos_sim_mask].mean()

        if neg_sim_mask.sum() > 0:
            loss_sim_neg = - cross_prob_pos[neg_sim_mask].mean()

        return loss_sim_pos, loss_sim_neg, pos_sim_mask, neg_sim_mask

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
class PFSTLossV4(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='gaussian', tau_pos=0.25, tau_neg=0.75):
        super(PFSTLossV2, self).__init__()

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
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

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
        ignore_mask_src = gt_src_ != 255
        logits_ema = F.interpolate(tensors['logits_ema'], size=(H, W), mode='bilinear')

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


        loss_sim_pos, loss_sim_neg, pos_sim_mask, neg_sim_mask = self.get_sim_losses(
            x_ema, ema_sim_feat,
            logits_trg, trg_cross_prob_map_diag,
            ignore_mask_src & ignore_mask_trg
        )

        losses.update({
            'loss_src_pos': - src_pos_sim.mean() * self.weights['src_pos'],
            'loss_src_neg': src_neg_sim.mean() * self.weights['src_neg'],
            'loss_sim_pos': loss_sim_pos * self.weights['sim_pos'],
            'loss_sim_neg': loss_sim_neg * self.weights['sim_neg'],
            # 'vis|hist_sim': (['pos', 'neg'], [src_pos_sim.detach().cpu(), src_neg_sim.detach().cpu()]),
            'vis|density_sim_feat': (img_trg, 1-ema_sim_feat.mean(dim=1).detach().unsqueeze(1)),
            'vis|seg_mask_sim': (img_trg, (pos_sim_mask.sum(dim=1).unsqueeze(1) > 0).long(),
                                 (neg_sim_mask.sum(dim=1).unsqueeze(1) > 0).long())
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

    def get_sim_losses(self, feats, sim_feat, logits, cross_prob_map_diag, ignore_mask=None):

        B, C, H, W = logits.shape
        pred = logits.max(dim=1)[1].unsqueeze(1)
        unf_pred = self.unfold_fun(pred.float())
        unf_pred = unf_pred.view(-1, self.kernel_size**2, H, W).long()
        rep_pred = pred.repeat(1, self.kernel_size**2, 1, 1)

        pos_pred_pair = unf_pred == rep_pred
        neg_pred_pair = unf_pred != rep_pred

        cross_prob_pos = (cross_prob_map_diag).sum(dim=1).permute(0,3,1,2)
        cross_prob_neg = 1 - cross_prob_pos

        pos_sim_feat = torch.where(pos_pred_pair, sim_feat, 255)
        neg_sim_feat = torch.where(neg_pred_pair, sim_feat, -255)

        pos_sim_mask = (sim_feat < self.tau_pos) & pos_pred_pair
        neg_sim_mask = (sim_feat > self.tau_neg) & neg_pred_pair

        if ignore_mask is not None:
            unf_ignore_mask = self.unfold_fun(ignore_mask.float())
            unf_ignore_mask = unf_ignore_mask.view(-1, self.kernel_size**2, H, W).long()
            ignore_mask = unf_ignore_mask.sum(dim=1).unsqueeze(1) == self.kernel_size ** 2

            pos_sim_mask = pos_sim_mask & ignore_mask
            neg_sim_mask = neg_sim_mask & ignore_mask

        loss_sim_pos = torch.zeros(1).cuda()
        loss_sim_neg = torch.zeros(1).cuda()

        if pos_sim_mask.sum() > 0:
            loss_sim_pos = cross_prob_pos[pos_sim_mask].mean()

        if neg_sim_mask.sum() > 0:
            loss_sim_neg = - cross_prob_pos[neg_sim_mask].mean()

        return loss_sim_pos, loss_sim_neg, pos_sim_mask, neg_sim_mask

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


