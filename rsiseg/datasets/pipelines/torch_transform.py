import torch

def transform_by_metas(self, data, metas):
    # data: (H, W, ...)

    if 'scale_factor' in metas:
        w_scale, h_scale = metas['scale_factor']
        H, W = metas['ori_shape']
        new_h, new_w = int(H * h_scale), int(W * w_scale)
        data = F.interpolate(data, size=(new_h, new_w), mode='nearest')

    if 'crop_bbox' in metas:
        crop_y1, crop_y2, crop_x1, crop_x2 = metas['crop_bbox']
        data = data[crop_y1:crop_y2, crop_x1:crop_x2, ...]

    if 'rotate_k' in metas:
        data = torch.rot90(data, metas['rotate_k'])

    if 'flip_horizontal' in metas and metas['flip_horizontal']:
        data = data.flip(dims=[1])

    if 'flip_vertical' in metas and metas['flip_vertical']:
        data = data.flip(dims=[0])

    return data
