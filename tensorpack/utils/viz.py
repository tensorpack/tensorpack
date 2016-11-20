#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Credit: zxytim

import numpy as np
import os, sys
import io
import cv2
from .fs import mkdir_p
from .argtools import shape2d

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

__all__ = ['pyplot2img', 'build_patch_list', 'pyplot_viz',
        'dump_dataflow_images', 'interactive_imshow']

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

def interactive_imshow(img, lclick_cb=None, rclick_cb=None, **kwargs):
    """
    :param lclick_cb: a callback(img, x, y) for left click
    :param kwargs: can be {key_cb_a ... key_cb_z: callback(img)}
    """
    name = 'random_window_name'
    cv2.imshow(name, img)

    def mouse_cb(event, x, y, *args):
        if event == cv2.EVENT_LBUTTONUP and lclick_cb is not None:
            lclick_cb(img, x, y)
        elif event == cv2.EVENT_RBUTTONUP and rclick_cb is not None:
            rclick_cb(img, x, y)
    cv2.setMouseCallback(name, mouse_cb)
    key = chr(cv2.waitKey(-1) & 0xff)
    cb_name = 'key_cb_' + key
    if cb_name in kwargs:
        kwargs[cb_name](img)
    elif key == 'q':
        cv2.destroyWindow(name)
    elif key == 'x':
        sys.exit()
    elif key == 's':
        cv2.imwrite('out.png', img)

def build_patch_list(patch_list,
        nr_row=None, nr_col=None, border=None,
        max_width=1000, max_height=1000,
        shuffle=False, bgcolor=255,
        viz=False, lclick_cb=None):
    """
    Generate patches.
    :param patch_list: bhw or bhwc
    :param border: defaults to 0.1 * max(image_width, image_height)
    :param nr_row, nr_col: rows and cols of the grid
    :parma max_width, max_height: if nr_row/col are not given, use this to infer the rows and cols
    :param shuffle: shuffle the images
    :param bgcolor: background color
    :param viz: use interactive imshow to visualize the results
    :param lclick_cb: only useful when viz=True. a callback(patch, idx)
    """
    # setup parameters
    patch_list = np.asarray(patch_list)
    if patch_list.ndim == 3:
        patch_list = patch_list[:,:,:,np.newaxis]
    assert patch_list.ndim == 4 and patch_list.shape[3] in [1, 3], patch_list.shape
    if shuffle:
        np.random.shuffle(patch_list)
    if lclick_cb is not None:
        viz = True
    ph, pw = patch_list.shape[1:3]
    if border is None:
        border = int(0.1 * max(ph, pw))
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

    def lclick_callback(img, x, y):
        if lclick_cb is None:
            return
        x = x // (pw + border)
        y = y // (pw + border)
        idx = start + y * nr_col + x
        if idx < end:
            lclick_cb(patch_list[idx], idx)

    while True:
        end = start + nr_patch
        cur_list = patch_list[start:end]
        if not len(cur_list):
            return
        draw_patch(cur_list)
        if viz:
            interactive_imshow(canvas, lclick_cb=lclick_callback)
        yield canvas
        start = end

def dump_dataflow_images(df, index=0, batched=True,
        number=300, output_dir=None,
        scale=1, resize=None, viz=None,
        flipRGB=False, exit_after=True):
    """
    :param df: a DataFlow
    :param index: the index of the image component
    :param batched: whether the component contains batched images or not
    :param number: how many datapoint to take from the DataFlow
    :param output_dir: output directory to save images, default to not save.
    :param scale: scale the value, usually either 1 or 255
    :param resize: (h, w) or Nne, resize the images
    :param viz: (h, w) or None, visualize the images in grid with imshow
    :param flipRGB: apply a RGB<->BGR conversion or not
    :param exit_after: exit the process after this function
    """
    if output_dir:
        mkdir_p(output_dir)
    if viz is not None:
        viz = shape2d(viz)
        vizsize = viz[0] * viz[1]
    if resize is not None:
        resize = tuple(shape2d(resize))
    vizlist = []

    df.reset_state()
    cnt = 0
    while True:
        for dp in df.get_data():
            if not batched:
                imgbatch = [dp[index]]
            else:
                imgbatch = dp[index]
            for img in imgbatch:
                cnt += 1
                if cnt == number:
                    if exit_after:
                        sys.exit()
                    else:
                        return
                if scale != 1:
                    img = img * scale
                if resize is not None:
                    img = cv2.resize(img, resize)
                if flipRGB:
                    img = img[:,:,::-1]
                if output_dir:
                    fname = os.path.join(output_dir, '{:03d}.jpg'.format(cnt))
                    cv2.imwrite(fname, img)
                if viz is not None:
                    vizlist.append(img)
            if viz is not None and len(vizlist) >= vizsize:
                patch = next(build_patch_list(
                    vizlist[:vizsize],
                    nr_row=viz[0], nr_col=viz[1], viz=True))
                vizlist = vizlist[vizsize:]


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
