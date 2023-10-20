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
import json

import mmcv
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm
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
class RareClassSamplingHook(Hook):

    def __init__(self,
                 log_dir,
                 num_classes,
                 data_cfg=None,
                 cfg=None,
                 overwrite=False,
                 **kwargs):
        self.log_dir = log_dir
        self.data_cfg = data_cfg
        self.cfg = cfg
        self.num_classes = num_classes

        self.skip = False
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if osp.exists(osp.join(self.log_dir, 'sample_class_stats.json')):
            self.skip = not overwrite

    @master_only
    def before_run(self, runner):
        super(RareClassSamplingHook, self).before_run(runner)
        if self.skip:
            return

        from rsiseg.datasets import build_dataloader, build_dataset 
        dataset = build_dataset(self.data_cfg.test)

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

        dataloader = build_dataloader(dataset, **test_loader_cfg)
        loader_indices = dataloader.batch_sampler
        sample_class_stats = []
        for batch_indices, data in tqdm(zip(loader_indices, dataloader)):
            for i, index in enumerate(batch_indices):
                # gt = dataset.get_gt_seg_map_by_idx(index)
                # img_metas = data['img_metas'][i].data[0]
                gt_results = dataset.get_gt_results_by_idx(index)
                gt_path = gt_results['ann_info']['seg_map']
                if 'img_prefix' in gt_results and gt_results['img_prefix'] is not None:
                    gt_path = os.path.join(gt_results['img_prefix'], gt_path)

                cur_stat = self._get_sample_class_stats(gt_results['gt_semantic_seg'], gt_path)
                sample_class_stats.append(cur_stat)

        self.save_class_stats(sample_class_stats)
        raise ValueError('Successfully logged rare class information!')

    def _get_sample_class_stats(self, gt, gt_name):
        sample_class_stats = {}
        for class_id in range(self.num_classes):
            mask = gt == class_id
            n = int(np.sum(mask))
            if n > 0:
                sample_class_stats[class_id] = n

        sample_class_stats['file'] = gt_name
        return sample_class_stats

    def save_class_stats(self, sample_class_stats):
        out_dir = self.log_dir

        with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
            json.dump(sample_class_stats, of, indent=2)

        sample_class_stats_dict = {}
        for stats in sample_class_stats:
            f = stats.pop('file')
            sample_class_stats_dict[f] = stats

        with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
            json.dump(sample_class_stats_dict, of, indent=2)

        samples_with_class = {}
        for file, stats in sample_class_stats_dict.items():
            for c, n in stats.items():
                if c not in samples_with_class:
                    samples_with_class[c] = [(file, n)]
                else:
                    samples_with_class[c].append((file, n))
        with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
            json.dump(samples_with_class, of, indent=2)



