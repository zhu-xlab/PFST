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
class PseudoLabelingHookV2(Hook):
    """Enhanced Wandb logger hook for MMDetection.
    """

    def __init__(self,
                 log_dir,
                 cls_thre_ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
                 interval=50,
                 sim_feat_cfg=None,
                 down_scale=1.0,
                 **kwargs):
        self.log_dir = log_dir
        self.interval = interval
        self.cls_thre_ratios=cls_thre_ratios
        self.sim_feat_cfg = sim_feat_cfg
        self.down_scale = down_scale

        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)


    @master_only
    def before_run(self, runner):
        super(PseudoLabelingHookV2, self).before_run(runner)

        # Check if EvalHook and CheckpointHook are available.
        for hook in runner.hooks:

            if isinstance(hook, EvalHook):
                from rsiseg.apis import single_gpu_test
                self.eval_hook = hook
                self.test_fn = single_gpu_test
            if isinstance(hook, DistEvalHook):
                from rsiseg.apis import multi_gpu_test
                self.eval_hook = hook
                self.test_fn = multi_gpu_test

    @master_only
    def after_train_iter(self, runner):
        super(PseudoLabelingHookV2, self).after_train_iter(runner)

        if self.every_n_iters(runner, self.interval):

            model = runner.model
            model.eval()
            dataloader = self.eval_hook.dataloader
            logits = []
            img_names = []

            dataset = self.eval_hook.dataloader.dataset
            prog_bar = mmcv.ProgressBar(len(dataset))
            loader_indices = dataloader.batch_sampler

            cnt = 0
            for batch_indices, data in zip(loader_indices, dataloader):
                with torch.no_grad():
                    result, states = model(return_loss=False, **data)

                img_metas = data['img_metas'][0].data[0]
                for x, y in zip(states, img_metas):
                    x['img_metas'] = y

                for state in states:
                    img_metas = state['img_metas']
                    feats = state['feats']
                    seg_logits = state['seg_logits']
                    img_name = img_metas['filename'].split('/')[-1].split('.')[0]

                    logits.append(seg_logits)
                    img_names.append(img_name)

                    gaussian_sim_feats = self._cal_sim_feat(feats, type='gaussian')
                    cosine_sim_feats = self._cal_sim_feat(feats, type='cosine')

                    with h5py.File(osp.join(self.log_dir, f'{img_name}.h5'), 'w') as hf:
                        hf.create_dataset('seg_logits', data=seg_logits)

                        for i, gaussian_sim_feat in enumerate(gaussian_sim_feats):
                            hf.create_dataset(f'gaussian_sim_feat_{i}', data=gaussian_sim_feat)

                        for i, cosine_sim_feat in enumerate(cosine_sim_feats):
                            hf.create_dataset(f'cosine_sim_feat_{i}', data=cosine_sim_feat)

                        hf.close()

                    batch_size = len(result)
                    for _ in range(batch_size):
                        prog_bar.update()

                # cnt += 1
                # if cnt == 10:
                #     break

            all_seg_logits = [logit for logit in logits]
            cls_thre_map = self._cal_threshold(torch.stack(all_seg_logits, dim=0),
                                               down_scale=self.down_scale)

            for img_name in img_names:
                with h5py.File(osp.join(self.log_dir, f'{img_name}.h5'), 'a') as hf:
                    for key, value in cls_thre_map.items():
                        hf.create_dataset(key, data=value)
                    hf.close()

    @master_only
    def after_train_epoch(self, runner):
        pdb.set_trace()
        super(PseudoLabelingHookV2, self).after_train_epoch(runner)
        pass

    def _cal_threshold(self, seg_logits, down_scale=1):

        if down_scale < 1:
            seg_logits = F.interpolate(seg_logits, scale_factor=(down_scale, down_scale))

        num_classes = seg_logits.shape[1]
        prob_maps = F.softmax(seg_logits, dim=1)
        pred_maps = prob_maps.argmax(dim=1)
        ent_maps = (- prob_maps * torch.log(prob_maps)).sum(dim=1)

        thre_map = {}
        for cls_thre_ratio in self.cls_thre_ratios:
            thre_map[f'thre@{cls_thre_ratio}'] = []

        for cls in range(num_classes):
            if (pred_maps == cls).sum() == 0:
                for cls_thre_ratio in self.cls_thre_ratios:
                    thre_map[f'thre@{cls_thre_ratio}'].append(0)

            else:
                sorted_map = np.sort(ent_maps[pred_maps==cls].reshape(-1))
                for cls_thre_ratio in self.cls_thre_ratios:
                    thre_rank = int(len(sorted_map) * cls_thre_ratio)
                    cur_thre = sorted_map[thre_rank]
                    thre_map[f'thre@{cls_thre_ratio}'].append(cur_thre)

        return thre_map

    def _cal_sim_feat(self, feats, type='gaussian'):
        kernel_size = self.sim_feat_cfg['kernel_size']
        sigmas = self.sim_feat_cfg['sigmas']
        dilation = self.sim_feat_cfg['dilation']

        self.unfold_fun = torch.nn.Unfold(kernel_size=kernel_size,
                                          padding=kernel_size // 2 * dilation,
                                          dilation=dilation)

        sim_feats = []
        for i, feat in enumerate(feats):
            # feats = F.interpolate(ori_feats, size=(H, W), mode='nearest')
            C, H, W = feat.shape
            feat = feat.unsqueeze(0).cuda()
            unf_feat = self.unfold_fun(feat)
            unf_feat = unf_feat.view(1, C, kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

            if type == 'gaussian':
                temp_dis = ((unf_feat - feat.unsqueeze(4)) ** 2).sum(dim=1)
                sim_feat = torch.exp(- temp_dis  / sigmas[i] ** 2).permute(0, 3, 1, 2)
            elif type == 'cosine':
                sim_feat = F.cosine_similarity(unf_feat, feat.unsqueeze(4), dim=1)
                sim_feat = sim_feat.permute(0, 3, 1, 2)
            else:
                raise ValueError('No such type of similarity calculation!')

            sim_feats.append(sim_feat.squeeze(0).cpu())

        return sim_feats


