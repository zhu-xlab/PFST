import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
import pdb


@LOSSES.register_module()
class LocalPseudoFeatLoss(nn.Module):

    def __init__(self, top_k, dilation, kernel_size, weights,
                 num_classes, sigma=30, mean_sim=0.6, feat_level=2,
                 sim_type='cosine'):
        super(LocalPseudoFeatLoss, self).__init__()

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
        self.loc_pseudo_net = nn.Linear(num_classes + self.kernel_size ** 2, 1)

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


