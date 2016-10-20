#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np

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
    nr_row = minnone(nr_row, max_height / (ph + border))
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
