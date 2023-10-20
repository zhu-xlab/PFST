import numpy as np
import seaborn as sns
import pdb
import cv2
import os
import matplotlib as mpl

# PALETTE = (np.array(sns.color_palette('bright', 33)) * 255).astype(np.uint8).tolist()
# PALETTE = (np.array(sns.color_palette('Spectral', 33)) * 255).astype(np.uint8).tolist()
PALETTE1 = mpl.cm.get_cmap('tab20')
PALETTE2 = mpl.cm.get_cmap('tab20b')

PALETTE = [PALETTE1(i/20) for i in range(20)] + [PALETTE2(i/20) for i in range(20)]
PALETTE = PALETTE[:33]
PALETTE = (np.array(PALETTE)[:, :3] * 255).astype(np.uint8).tolist()

# PALETTE = sns.color_palette("Spectral", 33)

H, W = 20, 100
img_path = 'work_dirs/color_palette/season_net/color_palette_{}.png'
img_dir = 'work_dirs/color_palette/season_net/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

for i, color in enumerate(PALETTE):
    mat = np.expand_dims(np.array(color), [0,1])
    mat = mat.repeat(H, axis=0).repeat(W, axis=1)
    mat = mat[:, :, [2,1,0]]
    cv2.imwrite(img_path.format(i), mat)
