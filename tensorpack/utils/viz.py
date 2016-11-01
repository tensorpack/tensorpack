#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Credit: zxytim

import numpy as np
import io
import cv2

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

__all__ = ['pyplot2img', 'build_patch_list', 'pyplot_viz']

def pyplot2img(plt):
    buf = io.BytesIO()
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    rawbuf = np.frombuffer(buf.getvalue(), dtype='uint8')
    im = cv2.imdecode(rawbuf, cv2.IMREAD_COLOR)
    buf.close()
    return im

def pyplot_viz(img, shape=None):
    """ use pyplot to visualize the image
        Note: this is quite slow. and the returned image will have a border
    """
    plt.clf()
    plt.axes([0,0,1,1])
    plt.imshow(img)
    ret = pyplot2img(plt)
    if shape is not None:
        ret = cv2.resize(ret, shape)
    return ret

def minnone(x, y):
    if x is None: x = y
    elif y is None: y = x
    return min(x, y)

def build_patch_list(patch_list,
        nr_row=None, nr_col=None, border=5,
        max_width=1000, max_height=1000,
        shuffle=False, bgcolor=255):
    """
    patch_list: bhw or bhwc
    """
    patch_list = np.asarray(patch_list)
    if patch_list.ndim == 3:
        patch_list = patch_list[:,:,:,np.newaxis]
    assert patch_list.ndim == 4 and patch_list.shape[3] in [1, 3], patch_list.shape
    if shuffle:
        np.random.shuffle(patch_list)
    ph, pw = patch_list.shape[1:3]
    mh, mw = max(max_height, ph + border), max(max_width, pw + border)
    if nr_row is None:
        nr_row = minnone(nr_row, max_height / (ph + border))
    if nr_col is None:
        nr_col = minnone(nr_col, max_width / (pw + border))

    canvas = np.zeros((nr_row * (ph + border) - border,
             nr_col * (pw + border) - border,
             patch_list.shape[3]), dtype='uint8')

    def draw_patch(plist):
        cur_row, cur_col = 0, 0
        canvas.fill(bgcolor)
        for patch in plist:
            r0 = cur_row * (ph + border)
            c0 = cur_col * (pw + border)
            canvas[r0:r0+ph, c0:c0+pw] = patch
            cur_col += 1
            if cur_col == nr_col:
                cur_col = 0
                cur_row += 1

    nr_patch = nr_row * nr_col
    start = 0
    while True:
        end = start + nr_patch
        cur_list = patch_list[start:end]
        if not len(cur_list):
            return
        draw_patch(cur_list)
        yield canvas
        start = end

if __name__ == '__main__':
    import cv2
    imglist = []
    for i in range(100):
        fname = "{:03d}.png".format(i)
        imglist.append(cv2.imread(fname))
    for idx, patch in enumerate(build_patch_list(
            imglist, max_width=500, max_height=200)):
        of = "patch{:02d}.png".format(idx)
        cv2.imwrite(of, patch)
