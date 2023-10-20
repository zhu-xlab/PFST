import os.path as osp

import mmcv
import numpy as np
from cv2 import imread
import tifffile as tiff
import h5py
import pdb
import time

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['ori_img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        if not 'img_fields' in results:
            results['img_fields'] = []
        results['img_fields'].append('img')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')

        for key, value in results['ann_info'].items():
            if key != 'seg_map':
                results[key] = value

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class AnnotationMapperInria(object):
    def __call__(self, results):
        gt_semantic_seg = results['gt_semantic_seg']
        gt_semantic_seg[gt_semantic_seg==255] = 1

        return results


@PIPELINES.register_module()
class LoadAnnotationsGTA(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        #img_bytes = self.file_client.get(filename)
        gt_semantic_seg = imread(filename, 2) / 100.
        #gt_semantic_seg = imread(filename, 2)
        gt_semantic_seg = np.clip(gt_semantic_seg, 0, 500)
        if np.isnan(gt_semantic_seg.sum()):
            gt_semantic_seg = np.where(np.isnan(gt_semantic_seg), np.full_like(gt_semantic_seg, 0), gt_semantic_seg)
        # modify if custom classes
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



@PIPELINES.register_module()
class LoadAnnotationsDepth(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        #img_bytes = self.file_client.get(filename)
        
        #filename = filename[:-7]+'.png'
        filename = filename.replace('RGB','AGL')
        
        gt_semantic_seg = imread(filename, 2)
        #gt_semantic_seg = imread(filename, 2) / 100.
        gt_semantic_seg[gt_semantic_seg>400] = 0
        #gt_semantic_seg = mmcv.imread(filename,2)
        gt_semantic_seg = np.clip(gt_semantic_seg, 0, 400)
        # If these is NaN value
        #if np.isnan(gt_semantic_seg.sum()):
        #    gt_semantic_seg = np.where(np.isnan(gt_semantic_seg), np.full_like(gt_semantic_seg, 0), gt_semantic_seg)
        '''gt_semantic_seg = mmcv.imfrombytes(
            iimg_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()'''
        # modify if custom classes
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotationsPseudoLabels(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 pseudo_labels_dir,
                 load_feats=False,
                 reduce_zero_label=False,
                 # file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        # self.file_client_args = file_client_args.copy()
        self.load_feats = load_feats
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.pseudo_labels_dir = pseudo_labels_dir

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        filename = filename.split('/')[-1].split('.')[0]
        file_path = osp.join(self.pseudo_labels_dir, filename + '.h5')

        with h5py.File(file_path, 'r') as f:
            logits = np.array(f['seg_logits'])
            feats = np.array(f['feats_2'])
            thres = np.array(f['cls_thres'])
            f.close()

        preds = logits.argmax(axis=0)
        probs = np.exp(logits) / np.exp(logits).sum(axis=0)
        ent_map = - (probs * np.log(probs + 1e-8)).sum(axis=0)
        thre_map = thres[preds]
        mask = ent_map < thre_map

        pse_labels = np.where(mask, preds, 255)

        if self.reduce_zero_label:
            # avoid using underflow conversion
            pse_labels[pse_labels == 0] = 255
            pse_labels = pse_labels - 1
            pse_labels[pse_labels == 254] = 255

        results['gt_semantic_seg'] = pse_labels.astype(np.uint8)
        results['seg_fields'].append('gt_semantic_seg')

        if self.load_feats:
            results['feats'] = feats.transpose(1,2,0)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotationsPseudoLabelsV2(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 pseudo_labels_dir=None,
                 updated_pseudo_labels_dir=None,
                 pseudo_ratio=0.5,
                 load_feats=False,
                 reduce_zero_label=False,
                 # file_client_args=dict(backend='disk'),
                 sim_feat_names=['gaussian_sim_feat_2'],
                 imdecode_backend='pillow',
                 filename_mapper_type=None):
        self.reduce_zero_label = reduce_zero_label
        # self.file_client_args = file_client_args.copy()
        self.load_feats = load_feats
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.pseudo_labels_dir = pseudo_labels_dir
        self.pseudo_ratio = pseudo_ratio
        self.sim_feat_names = sim_feat_names
        self.updated_pseudo_labels_dir = updated_pseudo_labels_dir
        self.filename_mapper_type = filename_mapper_type


    def filename_mapper(self, filename):
        if self.filename_mapper_type == 'season_net':
            return filename.replace('labels', '10m_RGB')

        return filename

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)

        # t1 = time.time()
        if 'ann_info' in results:
            ann_path = results['ann_info']['seg_map']
        else:
            # In pseudo labeling case, names of the image and the pseudo label should match
            ann_path = results['img_info']['filename']

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'], ann_path)
        else:
            filename = ann_path

        filename = filename.split('/')[-1].split('.')[0]
        filename = self.filename_mapper(filename)

        if self.pseudo_labels_dir is None:
            results['gt_semantic_seg'] = np.zeros(results['img_shape']).astype(np.uint8)
            results['gt_semantic_seg'].fill(255)
            results['seg_fields'].append('gt_semantic_seg')

            return results

        file_path = osp.join(self.pseudo_labels_dir, filename + '.h5')
        if self.updated_pseudo_labels_dir is not None:
            updated_file_path = osp.join(self.updated_pseudo_labels_dir, filename + '.h5')
            if osp.exists(updated_file_path):
                file_path = updated_file_path

        with h5py.File(file_path, 'r') as f:
            logits = np.array(f['seg_logits'])
            thres = np.array(f[f'thre@{self.pseudo_ratio}'])


            preds = logits.argmax(axis=0)
            probs = np.exp(logits) / np.exp(logits).sum(axis=0)
            ent_map = - (probs * np.log(probs + 1e-8)).sum(axis=0)
            thre_map = thres[preds]
            mask = ent_map < thre_map

            pse_labels = np.where(mask, preds, 255)

            if self.reduce_zero_label:
                # avoid using underflow conversion
                pse_labels[pse_labels == 0] = 255
                pse_labels = pse_labels - 1
                pse_labels[pse_labels == 254] = 255

            results['gt_semantic_seg'] = pse_labels.astype(np.uint8)
            results['seg_fields'].append('gt_semantic_seg')

            if self.load_feats:
                gaussian_feats = []
                cosine_feats = []

                for name in self.sim_feat_names:
                    results[name] = np.array(f[name])
                    # results[name] = np.zeros((9, 128, 128))

                # for level in range(4):
                #     gaussian_feat = np.array(f[f'gaussian_sim_feat_{level}'])
                #     # gaussian_feats.append(gaussian_feat)

                #     cosine_feat = np.array(f[f'cosine_sim_feat_{level}'])
                #     # cosine_feats.append(cosine_feat)

                #     results[f'gaussian_sim_feat_{level}'] = gaussian_feat
                #     results[f'cosine_sim_feat_{level}'] = cosine_feat

            f.close()

        # t2 = time.time()
        # print('loading time: {}'.format(t2-t1))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
