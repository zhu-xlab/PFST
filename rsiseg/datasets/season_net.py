# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import EODataset


@DATASETS.register_module()
class SeasonNetDataset(EODataset):
    """ISPRS dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to False. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
