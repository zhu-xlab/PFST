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
import tqdm

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
class PseudoLabelingHookV4(Hook):
    """Enhanced Wandb logger hook for MMDetection.
    """

    def __init__(self,
                 log_dir,
                 cls_thre_ratios=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                 interval=50,
                 down_scale=1.0,
                 thre_sample_ratio=1.0,
                 data_cfg=None,
                 sim_feat_cfg=None,
                 **kwargs):
        self.log_dir = log_dir
        self.interval = interval
        self.cls_thre_ratios=cls_thre_ratios
        self.down_scale = down_scale
        self.thre_sample_ratio = thre_sample_ratio # to accelerate the sorting process
        self.data_cfg = data_cfg
        self.sim_feat_cfg = sim_feat_cfg

        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)


    @master_only
    def before_run(self, runner):
        super(PseudoLabelingHookV4, self).before_run(runner)
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
        super(PseudoLabelingHookV4, self).after_train_iter(runner)


        if self.every_n_iters(runner, self.interval):

            model = runner.model
            model.eval()
            # dataloader = self.eval_hook.dataloader
            # dataset = self.eval_hook.dataloader.dataset
            dataloader = self.test_dataloader
            dataset = self.test_dataloader.dataset
            dataset.test_mode = True


            logits = []
            img_names = []
            feat_level = self.sim_feat_cfg['feat_level'] if self.sim_feat_cfg is not None else []

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

                for state in states:
                    img_metas = state['img_metas']
                    feats = state['feats']
                    seg_logits = state['seg_logits']
                    img_name = img_metas['filename'].split('/')[-1].split('.')[0]

                    if self.sim_feat_cfg is not None:
                        loc_dis_list.append(self._cal_loc_dis(feats))

                    logits.append(seg_logits)
                    img_names.append(img_name)

                    with h5py.File(osp.join(self.log_dir, f'{img_name}.h5'), 'w') as hf:

                        seg_logits_down = F.interpolate(seg_logits.unsqueeze(0),
                                                        scale_factor=(self.down_scale, self.down_scale))

                        hf.create_dataset('seg_logits', data=seg_logits_down.squeeze(0))
                        # for i, feat in enumerate(feats):
                        for i in feat_level:
                            feat = feats[i]
                            feat_down = F.interpolate(feat.unsqueeze(0),
                                                      scale_factor=(self.down_scale, self.down_scale))
                            hf.create_dataset(f'feat_{i}', data=feat_down.squeeze(0))

                        hf.close()

                    batch_size = len(result)
                    for _ in range(batch_size):
                        prog_bar.update()

            all_seg_logits = [logit for logit in logits]
            cls_thre_map = self._cal_threshold(torch.stack(all_seg_logits, dim=0),
                                               sample_ratio=self.thre_sample_ratio)
            if self.sim_feat_cfg is not None:
                sigmas = self._cal_sigmas(loc_dis_list, sample_ratio=self.thre_sample_ratio)

            for idx, img_name in enumerate(img_names):
                with h5py.File(osp.join(self.log_dir, f'{img_name}.h5'), 'a') as hf:
                    for key, value in cls_thre_map.items():
                        hf.create_dataset(key, data=value)
                    if self.sim_feat_cfg is not None:
                        for key, value in sigmas.items():
                            hf.create_dataset(key, data=value)
                    hf.close()

            raise ValueError('Succesfully generated the pseudo labels, stop training.')

    @master_only
    def after_train_epoch(self, runner):
        super(PseudoLabelingHookV4, self).after_train_epoch(runner)
        pass

    def _cal_threshold(self, seg_logits, sample_ratio):

        B, num_classes, H, W = seg_logits.shape
        seg_logits = seg_logits.permute(0,2,3,1).contiguous().view(-1, num_classes)
        num_samples, _ = seg_logits.shape


        idx = np.random.permutation(num_samples)[:int(num_samples * sample_ratio)-1]
        seg_logits = seg_logits[idx, :]
        # if down_scale < 1:
        #     seg_logits = F.interpolate(seg_logits, scale_factor=(down_scale, down_scale))

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


    def _cal_loc_dis(self, feats):

        kernel_size = self.sim_feat_cfg['kernel_size']
        sigmas = self.sim_feat_cfg['sigmas']
        dilations = self.sim_feat_cfg['dilation']
        if type(dilations) != list:
            dilations = [dilations]

        loc_dis = dict()
        for level, feat in enumerate(feats):
            C, H, W = feat.shape
            feat = feat.unsqueeze(0).cuda()
            for dila in dilations:
                self.unfold_fun = torch.nn.Unfold(kernel_size=kernel_size,
                                                  padding=kernel_size // 2 * dila,
                                                  dilation=dila)
                unf_feat = self.unfold_fun(feat)
                unf_feat = unf_feat.view(1, -1, kernel_size**2, H, W).permute(0, 1, 3, 4, 2)

                temp_dis = ((unf_feat - feat.unsqueeze(4)) ** 2).sum(dim=1)
                loc_dis[f'level{level}_dila@{dila}'] = temp_dis.cpu()
                # sim_feat = torch.exp(- temp_dis  / sigmas[i] ** 2).permute(0, 3, 1, 2)

        return loc_dis

    def _cal_sigmas(self, loc_dis_list, sample_ratio):
        mean_sims = self.sim_feat_cfg['mean_sim']
        dilations = self.sim_feat_cfg['dilation']
        feat_level = self.sim_feat_cfg['feat_level']
        if type(dilations) != list:
            dilations = [dilations]

        if type(mean_sims) != list:
            mean_sims = [mean_sims]

        loc_dis_tensor = dict()
        for level in feat_level:
            for dila in dilations:
                # loc_dis_tensor[f'level{level}_dila@{dila}'] = []
                cur_tensor = []
                for loc_dis in loc_dis_list:
                    cur_tensor.append(loc_dis[f'level{level}_dila@{dila}'])

                # loc_dis_tensor[f'level{level}_dila@{dila}'] = torch.cat(loc_dis_tensor[f'level{level}_dila@{dila}'], dim=0)
                cur_tensor = torch.cat(cur_tensor, dim=0) # B, 9, H, W
                B, H, W, C = cur_tensor.shape

                cur_tensor = cur_tensor.view(-1, C)
                num_samples = cur_tensor.shape[0]
                idx = np.random.permutation(num_samples)[:int(num_samples * sample_ratio)-1]
                loc_dis_tensor[f'level{level}_dila@{dila}'] = cur_tensor[idx, :]

        sigmas = dict()
        for key, dis in loc_dis_tensor.items():

            for mean_sim in tqdm.tqdm(mean_sims):
                # start binary search
                left = 0
                right = 1000
                while abs(left - right) > 1e-6:
                    sigma = (left + right) / 2
                    sim_feat = torch.exp(- dis  / sigma ** 2)
                    if sim_feat.mean() < mean_sim:
                        left = sigma
                    else:
                        right = sigma

                sigmas[f'{key}_mean@{mean_sim}'] = left

        return sigmas

