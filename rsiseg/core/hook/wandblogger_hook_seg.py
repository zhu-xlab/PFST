# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import sys
import warnings
import pdb
import torch
import cv2

import mmcv
import numpy as np
import pycocotools.mask as mask_util
import matplotlib.pyplot as plt

from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook
from mmcv.utils import digit_version

from rsiseg.ops import resize
from rsiseg.core import DistEvalHook, EvalHook
# from rsiseg.core.mask.structures import polygon_to_bitmap


@HOOKS.register_module()
class WandbHookSeg(WandbLoggerHook):
    """Enhanced Wandb logger hook for MMDetection.

    Comparing with the :cls:`mmcv.runner.WandbLoggerHook`, this hook can not
    only automatically log all the metrics but also log the following extra
    information - saves model checkpoints as W&B Artifact, and
    logs model prediction as interactive W&B Tables.

    - Metrics: The MMDetWandbHook will automatically log training
        and validation metrics along with system metrics (CPU/GPU).

    - Checkpointing: If `log_checkpoint` is True, the checkpoint saved at
        every checkpoint interval will be saved as W&B Artifacts.
        This depends on the : class:`mmcv.runner.CheckpointHook` whose priority
        is higher than this hook. Please refer to
        https://docs.wandb.ai/guides/artifacts/model-versioning
        to learn more about model versioning with W&B Artifacts.

    - Checkpoint Metadata: If evaluation results are available for a given
        checkpoint artifact, it will have a metadata associated with it.
        The metadata contains the evaluation metrics computed on validation
        data with that checkpoint along with the current epoch. It depends
        on `EvalHook` whose priority is more than MMDetWandbHook.

    - Evaluation: At every evaluation interval, the `MMDetWandbHook` logs the
        model prediction as interactive W&B Tables. The number of samples
        logged is given by `num_eval_images`. Currently, the `MMDetWandbHook`
        logs the predicted bounding boxes along with the ground truth at every
        evaluation interval. This depends on the `EvalHook` whose priority is
        more than `MMDetWandbHook`. Also note that the data is just logged once
        and subsequent evaluation tables uses reference to the logged data
        to save memory usage. Please refer to
        https://docs.wandb.ai/guides/data-vis to learn more about W&B Tables.

    For more details check out W&B's MMDetection docs:
    https://docs.wandb.ai/guides/integrations/rsidetection

    ```
    Example:
        log_config = dict(
            ...
            hooks=[
                ...,
                dict(type='MMDetWandbHook',
                     init_kwargs={
                         'entity': "YOUR_ENTITY",
                         'project': "YOUR_PROJECT_NAME"
                     },
                     interval=50,
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=100,
                     bbox_score_thr=0.3)
            ])
    ```

    Args:
        init_kwargs (dict): A dict passed to wandb.init to initialize
            a W&B run. Please refer to https://docs.wandb.ai/ref/python/init
            for possible key-value pairs.
        interval (int): Logging interval (every k iterations). Defaults to 50.
        log_checkpoint (bool): Save the checkpoint at every checkpoint interval
            as W&B Artifacts. Use this for model versioning where each version
            is a checkpoint. Defaults to False.
        log_checkpoint_metadata (bool): Log the evaluation metrics computed
            on the validation data with the checkpoint, along with current
            epoch as a metadata to that checkpoint.
            Defaults to True.
        num_eval_images (int): The number of validation images to be logged.
            If zero, the evaluation won't be logged. Defaults to 100.
        bbox_score_thr (float): Threshold for bounding box scores.
            Defaults to 0.3.
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=50,
                 num_eval_images=10,
                 BGR2RGB=False,
                 **kwargs):
        super(WandbHookSeg, self).__init__(init_kwargs, interval, **kwargs)

        self.num_eval_images = num_eval_images
        self.BGR2RGB = BGR2RGB

    @master_only
    def before_run(self, runner):
        super(WandbHookSeg, self).before_run(runner)

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        CLASSES = model.CLASSES
        class_labels = {}
        for i, class_name in enumerate(CLASSES):
            class_labels[i] = class_name

        self.class_labels = class_labels

    @master_only
    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            # An ugly patch. The iter-based eval hook will call the
            # `after_train_iter` method of all logger hooks before evaluation.
            # Use this trick to skip that call.
            # Don't call super method at first, it will clear the log_buffer
            return super(WandbHookSeg, self).after_train_iter(runner)
        else:
            super(WandbHookSeg, self).after_train_iter(runner)

        # if self.by_epoch:
        #     return

        if self.every_n_iters(runner, self.interval):
        # if True:
            # results = self.eval_hook.latest_results
            results = runner.outputs
            if 'states' in results:
                self._process_vis_data(runner, results['states'])

    def _process_vis_data(self, runner, state):

        for key, value in state.items():
            if key.startswith('vis|'):
                if key.split('|')[1].startswith('seg_mask'):
                    self._vis_seg_mask(key.split('|')[1], value)

                elif key.split('|')[1].startswith('density'):
                    self._vis_density(key.split('|')[1], value)

                elif key.split('|')[1].startswith('hist'):
                    self._vis_hist(key.split('|')[1], value[0], value[1])

    def _vis_density(self, prefix, data):
        img = data[0]
        img_log = {}

        for i in range(img.shape[0]):

            cur_img = img[i].permute(1,2,0).cpu().numpy()
            if self.BGR2RGB:
                cur_img = cur_img[:, :, [2,1,0]]

            for idx, mask in enumerate(data[1:]):

                cur_mask = (mask * 255.).to(torch.uint8)

                cur_mask = resize(
                    input=cur_mask,
                    size=img.shape[2:],
                    mode='nearest')

                cur_mask = cur_mask[i].squeeze(0).cpu().numpy()
                cur_mask = cv2.applyColorMap(cur_mask, colormap=cv2.COLORMAP_JET)


                if f'{prefix}_{idx}' not in img_log:
                    img_log[f'{prefix}_{idx}'] = []

                img_log[f'{prefix}_{idx}'].append(self.wandb.Image(cur_img))
                img_log[f'{prefix}_{idx}'].append(self.wandb.Image(cur_mask))


        self.wandb.log(img_log)



    def _vis_seg_mask(self, prefix, data):

        img = data[0]
        img_log = {}

        for i in range(img.shape[0]):

            cur_img = img[i].permute(1,2,0).cpu().numpy()
            if self.BGR2RGB:
                cur_img = cur_img[:, :, [2,1,0]]

            for idx, mask in enumerate(data[1:]):
                if mask.dtype == torch.int64:
                    mask = mask.to(torch.uint8)
                cur_mask = resize(
                    input=mask,
                    size=img.shape[2:],
                    mode='nearest')

                cur_mask = cur_mask[i].squeeze(0).cpu().numpy()

                # cur_mask[cur_mask == 0] = 255
                # cur_mask = cur_mask - 1
                # cur_mask[cur_mask == 254] = 255

                if f'{prefix}_{idx}' not in img_log:
                    img_log[f'{prefix}_{idx}'] = []

                vis_mask = dict(mask_data=cur_mask,
                                class_labels=self.class_labels)

                img_log[f'{prefix}_{idx}'].append(self.wandb.Image(cur_img, masks={'mask': vis_mask}))

        self.wandb.log(img_log)


    def _visualize_train_data(self, runner, vis_data):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        CLASSES = model.CLASSES
        class_labels = {}
        for i, class_name in enumerate(CLASSES):
            class_labels[i] = class_name

        img_log = {'gt': [], 'pred': []}
        for i in range(vis_data['img'].shape[0]):
            vis_data['seg_logits'] = resize(
                input=vis_data['seg_logits'],
                size=vis_data['gt'].shape[2:],
                mode='bilinear',
                align_corners=False)

            img = vis_data['img'][i].permute(1,2,0).cpu().numpy()
            gt = vis_data['gt'][i].squeeze(0).cpu().numpy()
            pred = vis_data['seg_logits'][i].argmax(dim=0).cpu().numpy()
            if self.BGR2RGB:
                img = img[:, :, [2,1,0]]


            gt_masks = {
                'mask_data': gt,
                'class_labels': class_labels}
            pred_masks = {
                'mask_data': pred,
                'class_labels': class_labels}
            img_log['gt'].append(self.wandb.Image(img, masks={'ground_truth': gt_masks}))
            img_log['pred'].append(self.wandb.Image(img, masks={'predition': pred_masks}))

        self.wandb.log(img_log)

    def _vis_hist(self, name, column_names, data_list):
        tables = []
        for i, data in enumerate(data_list):
            data = data.numpy()
            hist, bins = np.histogram(data, bins=20, range=(0,1))
            hist = hist / hist.sum()
            bars = [[freq, value] for freq, value in zip(hist, bins)]

            table = self.wandb.Table(data=bars, columns=['freq', 'probability'])
            # tables.append(table)

            self.wandb.log({
                column_names[i]: self.wandb.plot.bar(table, "a", 'b', title='sim_hist')
            })

    @master_only
    def after_train_epoch(self, runner):
        pdb.set_trace()
        super(WandbHookSeg, self).after_train_epoch(runner)
        pass

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

