# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import os.path as osp
import sys
import warnings
import pdb
import h5py
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook
from mmcv.runner.hooks import Hook
from mmcv.utils import digit_version

from rsiseg.ops import resize
from rsiseg.core import DistEvalHook, EvalHook
# from rsiseg.core.mask.structures import polygon_to_bitmap


@HOOKS.register_module()
class PlotStatisticsHook(Hook):

    def __init__(self,
                 log_dir,
                 interval=50,
                 data_cfg=None,
                 sim_feat_cfg=None,
                 **kwargs):
        self.log_dir = log_dir
        self.interval = interval
        self.data_cfg = data_cfg
        self.sim_feat_cfg = sim_feat_cfg

        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)


    @master_only
    def before_run(self, runner):
        super(PlotStatisticsHook, self).before_run(runner)
        from rsiseg.datasets import build_dataloader, build_dataset 
        test_dataset = build_dataset(self.data_cfg.test)

        loader_cfg = dict(
            # cfg.gpus will be ignored if distributed
            num_gpus=1,
            dist=False,
            shuffle=False)

        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in self.data_cfg.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader'
            ]
        })

        test_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **self.data_cfg.get('test_dataloader', {}),
        }

        self.test_dataloader = build_dataloader(test_dataset, **test_loader_cfg)



    @master_only
    def after_train_iter(self, runner):
        super(PlotStatisticsHook, self).after_train_iter(runner)

        use_decoded_feats = self.sim_feat_cfg.get('use_decoded_feats', False)
        if self.every_n_iters(runner, self.interval):

            model = runner.model
            model.eval()
            # dataloader = self.eval_hook.dataloader
            # dataset = self.eval_hook.dataloader.dataset
            dataloader = self.test_dataloader
            dataset = self.test_dataloader.dataset

            feat_level = self.sim_feat_cfg['feat_level']

            prog_bar = mmcv.ProgressBar(len(dataset))
            loader_indices = dataloader.batch_sampler

            cnt = 0
            loc_dis_list = []

            for batch_indices, data in zip(loader_indices, dataloader):

                with torch.no_grad():
                    result, states = model(return_loss=False, **data)

                img_metas = data['img_metas'][0].data[0]
                for x, y in zip(states, img_metas):
                    x['img_metas'] = y

                for i, state in enumerate(states):
                    index = batch_indices[i]
                    gt = dataset.get_gt_seg_map_by_idx(index)

                    img_metas = state['img_metas']
                    # feats = state['feats']
                    feats = state['feats'] if not use_decoded_feats else state['decoded_feats']
                    seg_logits = state['seg_logits']
                    img_name = img_metas['filename'].split('/')[-1].split('.')[0]

                    self._add_loc_sim(seg_logits, gt, feats, feat_level)

                    batch_size = len(result)
                    for _ in range(batch_size):
                        prog_bar.update()

            self.plot_sim_hist()


            raise ValueError('Succesfully process the statistics, stop training.')

    def _add_loc_sim(self, seg_logits, gt, feats, feat_level):


        kernel_size = self.sim_feat_cfg['kernel_size']
        sigma = self.sim_feat_cfg['sigma']
        dilation = self.sim_feat_cfg['dilation']
        top_k = self.sim_feat_cfg['top_k']
        sim_type = self.sim_feat_cfg['sim_type']

        unfold_fun = torch.nn.Unfold(kernel_size=kernel_size,
                                     padding=kernel_size // 2 * dilation,
                                     dilation=dilation)

        seg_logits = torch.tensor(seg_logits).unsqueeze(0).cuda()
        gt = torch.tensor(gt).unsqueeze(0).cuda()
        # feat = feats[feat_level].cuda().unsqueeze(0)
        feat = feats[feat_level].cuda().unsqueeze(0) if feat_level is not None else feats.cuda()

        B, H, W = gt.shape
        _, c, h, w = feat.shape

        seg_logits = F.interpolate(seg_logits, (h, w))
        preds = seg_logits.max(dim=1)[1]
        gt = F.interpolate(gt.unsqueeze(1).to(torch.uint8), (h, w), mode='nearest')


        unf_feat = unfold_fun(feat)

        unf_feat = unf_feat.view(-1, c, kernel_size**2, h, w).permute(0, 1, 3, 4, 2)

        if sim_type == 'gaussian':
            temp_dis = ((unf_feat - feat.unsqueeze(4)) ** 2).sum(dim=1)
            sim_feat = torch.exp(- temp_dis  / sigma ** 2).permute(0, 3, 1, 2)

        elif sim_type == 'cosine':
            sim_feat = F.cosine_similarity(unf_feat, feat.unsqueeze(4), dim=1)
            sim_feat = sim_feat.permute(0, 3, 1, 2)

        unf_logits = unfold_fun(preds.unsqueeze(1).float())
        unf_logits = unf_logits.view(-1, kernel_size**2, h, w).long()
        rep_logits = preds.unsqueeze(1).repeat(1, kernel_size**2, 1, 1)

        unf_gt = unfold_fun(gt.float())
        unf_gt = unf_gt.view(-1, kernel_size**2, h, w).long()
        rep_gt = gt.repeat(1, kernel_size**2, 1, 1)

        mask = gt == preds

        mask_tp = (unf_logits == rep_logits) & (unf_gt == rep_gt)
        mask_tn = (unf_logits == rep_logits) & (unf_gt != rep_gt)
        mask_fn = (unf_logits != rep_logits) & (unf_gt != rep_gt)
        mask_fp = (unf_logits != rep_logits) & (unf_gt == rep_gt)

        mask_diag = torch.ones_like(mask_fp)
        diag = torch.zeros(B, 1, h, w).fill_(kernel_size ** 2 // 2).long().cuda()
        mask_diag.scatter_(1, diag, 0)
        mask_diag = mask_diag.bool()

        # _, top_idx_min = torch.topk(sim_feat, self.top_k, dim=1, largest=False)
        _, top_idx_max = torch.topk(sim_feat, top_k, dim=1)
        top_idx_max = top_idx_max[:, 1:, :, :]

        mask_top_max = torch.zeros_like(mask_fp)
        mask_top_max.scatter_(1, top_idx_max, 1).bool


        # lr_ref_prob = F.softmax(seg_logits, dim=1)
        # thres = self.thres[lr_pse_label]
        # mask_thre = (lr_ref_prob.max(dim=1)[0] > thres.cuda()).unsqueeze(1)

        # mask_top_min = torch.zeros_like(mask_fp)
        # mask_top_min.scatter_(1, top_idx_min, 1).bool

        # sim_tp = sim_feat[mask & mask_tp & mask_diag & mask_top_max]
        # sim_tn = sim_feat[mask & mask_tn & mask_diag & mask_top_max]
        # sim_fn = sim_feat[mask & mask_fn & mask_diag & mask_top_min]
        # sim_fp = sim_feat[mask & mask_fp & mask_diag & mask_top_min]

        # sim_tp = sim_feat[mask & mask_tp & mask_diag & mask_thre]
        # sim_tn = sim_feat[mask & mask_tn & mask_diag & mask_thre]
        # sim_fn = sim_feat[mask & mask_fn & mask_diag & mask_thre]
        # sim_fp = sim_feat[mask & mask_fp & mask_diag & mask_thre]

        sim_tp = sim_feat[mask & mask_tp & mask_diag]
        sim_tn = sim_feat[mask & mask_tn & mask_diag]
        sim_fn = sim_feat[mask & mask_fn & mask_diag]
        sim_fp = sim_feat[mask & mask_fp & mask_diag]

        loc_rank = []
        for i in range(8):
            mask_top_max = torch.zeros_like(mask_fp)
            mask_top_max.scatter_(1, top_idx_max[:, i:i+1], 1).bool
            temp = []
            temp.append((unf_gt == rep_gt)[mask_top_max].sum().item())
            temp.append((unf_gt != rep_gt)[mask_top_max].sum().item())
            loc_rank.append(temp)

        self.add_sim_hist([sim_tp, sim_tn, sim_fn, sim_fp])
        self.add_loc_hist(loc_rank)


    def add_sim_hist(self, sim_dist):
        num_bins = 25
        if not hasattr(self, 'sim_hist'):
            self.sim_hist = np.zeros((4, num_bins))
        for i in range(4):
            self.sim_hist[i] += np.histogram(sim_dist[i].cpu().numpy(), bins = num_bins, range=(0, 1))[0]

    def add_loc_hist(self, loc_rank):
        if not hasattr(self, 'loc_hist'):
            self.loc_hist = np.zeros((len(loc_rank), 2))

        for i, rank in enumerate(loc_rank):
            self.loc_hist[i][0] += rank[0]
            self.loc_hist[i][1] += rank[1]


    def plot_sim_hist(self):
        num_bins = 25
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 3)

        ax.bar(range(num_bins), self.sim_hist[0] / self.sim_hist[0].sum(),
                   color=(228/255.0, 26/255.0, 28/255.0, 0.8), label='Case 1a')
        ax.bar(range(num_bins), self.sim_hist[1] / self.sim_hist[1].sum(),
                   color=(55/255.0, 126/255.0, 184/255.0, 0.8), label='Case 1b')
        ax.legend()
        ax.set_xticks(np.linspace(0, num_bins, 5))
        ax.set_xticklabels(np.linspace(0, num_bins, 5) / num_bins)
        ax.set(xlabel='Similarity', ylabel='Frequency')
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, 'sim_hist_true.pdf'))

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 3)

        ax.bar(range(num_bins), self.sim_hist[3] / self.sim_hist[3].sum(),
                   color=(228/255.0, 26/255.0, 28/255.0, 0.8), label='Case 2a')
        ax.bar(range(num_bins), self.sim_hist[2] / self.sim_hist[2].sum(),
                   color=(55/255.0, 126/255.0, 184/255.0, 0.8), label='Case 2b')

        ax.legend()
        ax.set_xticks(np.linspace(0, num_bins, 5))
        ax.set_xticklabels(np.linspace(0, num_bins, 5) / num_bins)
        ax.set(xlabel='Similarity', ylabel='Frequency')
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, 'sim_hist_false.pdf'))

        loc_hist = self.loc_hist
        loc_hist[:, 0] = loc_hist[:, 0] / loc_hist[:, 0].sum()
        loc_hist[:, 1] = loc_hist[:, 1] / loc_hist[:, 1].sum()

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 3)
        label = [str(i+1) for i in range(loc_hist.shape[0])]
        x = np.arange(len(label))

        width = 0.4
        ax.bar(x - width / 2, loc_hist[:, 0], width, label='Case 1a & 2a',
               color=(228/255.0, 26/255.0, 28/255.0, 0.8))
        ax.bar(x + width / 2, loc_hist[:, 1], width, label='Case 1b & 2b',
               color=(55/255.0, 126/255.0, 184/255.0, 0.8))

        ax.legend()
        ax.set(xlabel='Local Rank', ylabel='Frequency')
        fig.tight_layout()
        ax.set_xticks(x)
        ax.set_xticklabels(x + 1)
        fig.savefig(os.path.join(self.log_dir, 'local_rank.pdf'))

        # axs[0].bar(range(num_bins), self.sim_hist[0])
        # axs[0].bar(range(num_bins), self.sim_hist[1])
        # axs[1].bar(range(num_bins), self.sim_hist[2])
        # axs[1].bar(range(num_bins), self.sim_hist[3])
