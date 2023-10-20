import torch.nn.functional as F
import torch

def proportional_crop(data, crop_bbox, scale):
    """Crop from ``img``"""
    rescale = lambda x: int(x * scale)
    crop_y1, crop_y2, crop_x1, crop_x2 = map(rescale, crop_bbox)
    data = data[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    return data

def transform_by_metas(data, metas, scale=1/8.):
    # data: (H, W, ...)

    if 'scale_factor' in metas:
        w_scale, h_scale, _, _ = metas['scale_factor']
        # H, W, C = metas['ori_shape']
        # new_h, new_w = int(H * h_scale), int(W * w_scale)
        # data = F.interpolate(data, size=(new_h, new_w), mode='nearest')
        data = F.interpolate(data, scale_factor=(w_scale, h_scale), mode='bilinear')

    if 'crop_bbox' in metas:
        w_scale, h_scale, _, _ = metas['scale_factor']
        assert w_scale == h_scale
        data = proportional_crop(data, metas['crop_bbox'], 1/8.)

        H, W, C = metas['ori_shape']
        new_h, new_w = int(H * h_scale), int(W * w_scale)

        data_h, data_w = data.shape[-2:]

    if 'rotate_k' in metas:
        data = torch.rot90(data, metas['rotate_k'], dims=[2,3])

    if metas['flip']:
        if 'horizontal' in metas['flip_direction']:
            data = data.flip(dims=[3])

        if 'vertical' in metas['flip_direction']:
            data = data.flip(dims=[2])

    if 'pad_shape' in metas:

        _, _, H, W = data.shape
        pad_H, pad_W = metas['pad_shape'][:2]
        pad_H = int(pad_H * scale)
        pad_W = int(pad_W * scale)

        if pad_H != H or pad_W != W:
            data = F.pad(data, (0, pad_W - W, 0, pad_W - W), 'constant', -1) # ignore negative value when process the data

    return data

