#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Credit: zxytim

import numpy as np
import os
import sys
import io
import cv2
from .fs import mkdir_p
from .argtools import shape2d

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


__all__ = ['pyplot2img', 'interactive_imshow', 'build_patch_list',
           'pyplot_viz', 'dump_dataflow_images', 'intensity_to_rgb', 'stack_images']


def pyplot2img(plt):
    """ Convert a pyplot instance to image """
    buf = io.BytesIO()
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    rawbuf = np.frombuffer(buf.getvalue(), dtype='uint8')
    im = cv2.imdecode(rawbuf, cv2.IMREAD_COLOR)
    buf.close()
    return im


def pyplot_viz(img, shape=None):
    """ Use pyplot to visualize the image. e.g., when input is grayscale, the result
    will automatically have a colormap.

    Returns:
        np.ndarray: an image.
    Note:
        this is quite slow. and the returned image will have a border
    """
    plt.clf()
    plt.axes([0, 0, 1, 1])
    plt.imshow(img)
    ret = pyplot2img(plt)
    if shape is not None:
        ret = cv2.resize(ret, shape)
    return ret


def minnone(x, y):
    if x is None:
        x = y
    elif y is None:
        y = x
    return min(x, y)


def interactive_imshow(img, lclick_cb=None, rclick_cb=None, **kwargs):
    """
    Args:
        img (np.ndarray): an image (expect BGR) to show.
        lclick_cb: a callback func(img, x, y) for left click event.
        kwargs: can be {key_cb_a: callback_img, key_cb_b: callback_img}, to
            specify a callback func(img) for keypress.

    Some existing keypress event handler:

    * q: destroy the current window
    * x: execute ``sys.exit()``
    * s: save image to "out.png"
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
    Stacked patches into grid, to produce visualizations like the following:

    .. image:: https://github.com/ppwwyyxx/tensorpack/raw/master/examples/GAN/demo/CelebA-samples.jpg

    Args:
        patch_list(np.ndarray): NHW or NHWC images in [0,255].
        nr_row(int), nr_col(int): rows and cols of the grid.
        border(int): border length between images.
            Defaults to ``0.1 * min(image_w, image_h)``.
        max_width(int), max_height(int): Maximum allowed size of the
            visualization image. If ``nr_row/nr_col`` are not given, will use this to infer the rows and cols.
        shuffle(bool): shuffle the images inside ``patch_list``.
        bgcolor(int or 3-tuple): background color in [0, 255]. Either an int
            or a BGR tuple.
        viz(bool): whether to use :func:`interactive_imshow` to visualize the results.
        lclick_cb: A callback function to get called when ``viz==True`` and an
            image get clicked. It takes the image patch and its index in
            ``patch_list`` as arguments. (The index is invalid when
            ``shuffle==True``.)

    Yields:
        np.ndarray: the visualization image.
    """
    # setup parameters
    patch_list = np.asarray(patch_list)
    if patch_list.ndim == 3:
        patch_list = patch_list[:, :, :, np.newaxis]
    assert patch_list.ndim == 4 and patch_list.shape[3] in [1, 3], patch_list.shape
    if shuffle:
        np.random.shuffle(patch_list)
    if lclick_cb is not None:
        viz = True
    ph, pw = patch_list.shape[1:3]
    if border is None:
        border = int(0.1 * min(ph, pw))
    if nr_row is None:
        nr_row = minnone(nr_row, max_height / (ph + border))
    if nr_col is None:
        nr_col = minnone(nr_col, max_width / (pw + border))

    if isinstance(bgcolor, int):
        bgchannel = 1
    else:
        bgchannel = 3
    canvas_channel = max(patch_list.shape[3], bgchannel)
    canvas = np.zeros((nr_row * (ph + border) - border,
                       nr_col * (pw + border) - border,
                       canvas_channel), dtype='uint8')

    def draw_patch(plist):
        cur_row, cur_col = 0, 0
        if bgchannel == 1:
            canvas.fill(bgcolor)
        else:
            canvas[:, :, :] = bgcolor
        for patch in plist:
            r0 = cur_row * (ph + border)
            c0 = cur_col * (pw + border)
            canvas[r0:r0 + ph, c0:c0 + pw] = patch
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
                         number=1000, output_dir=None,
                         scale=1, resize=None, viz=None,
                         flipRGB=False):
    """
    Dump or visualize images of a :class:`DataFlow`.

    Args:
        df (DataFlow): the DataFlow.
        index (int): the index of the image component.
        batched (bool): whether the component contains batched images (NHW or
            NHWC) or not (HW or HWC).
        number (int): how many datapoint to take from the DataFlow.
        output_dir (str): output directory to save images, default to not save.
        scale (float): scale the value, usually either 1 or 255.
        resize (tuple or None): tuple of (h, w) to resize the images to.
        viz (tuple or None): tuple of (h, w) determining the grid size to use
            with :func:`build_patch_list` for visualization. No visualization will happen by
            default.
        flipRGB (bool): apply a RGB<->BGR conversion or not.
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
                    return
                if scale != 1:
                    img = img * scale
                if resize is not None:
                    img = cv2.resize(img, resize)
                if flipRGB:
                    img = img[:, :, ::-1]
                if output_dir:
                    fname = os.path.join(output_dir, '{:03d}.jpg'.format(cnt))
                    cv2.imwrite(fname, img)
                if viz is not None:
                    vizlist.append(img)
            if viz is not None and len(vizlist) >= vizsize:
                next(build_patch_list(
                    vizlist[:vizsize],
                    nr_row=viz[0], nr_col=viz[1], viz=True))
                vizlist = vizlist[vizsize:]


def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.

    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.

    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def stack_images(imgs, vertical=False):
    """Stack images with different shapes and different number of channels.

    Args:
        imgs (np.array): imgage
        vertical (bool, optional): stack images vertically

    Returns:
        np.array: stacked images
    """
    rows = [x.shape[0] for x in imgs]
    cols = [x.shape[1] for x in imgs]

    if vertical:
        if len(imgs[0].shape) == 2:
            out = np.zeros((np.sum(rows), max(cols)), dtype='uint8')
        else:
            out = np.zeros((np.sum(rows), max(cols), 3), dtype='uint8')
    else:
        if len(imgs[0].shape) == 2:
            out = np.zeros((max(rows), np.sum(cols)), dtype='uint8')
        else:
            out = np.zeros((max(rows), np.sum(cols), 3), dtype='uint8')

    offset = 0
    for i, img in enumerate(imgs):
        assert img.max() > 1, "expect images within range [0, 255]"
        if vertical:
            out[offset:offset + rows[i], :cols[i]] = img
            offset += rows[i]
        else:
            out[:rows[i], offset:offset + cols[i]] = img
            offset += cols[i]
    return out


if __name__ == '__main__':
    imglist = []
    for i in range(100):
        fname = "{:03d}.png".format(i)
        imglist.append(cv2.imread(fname))
    for idx, patch in enumerate(build_patch_list(
            imglist, max_width=500, max_height=200)):
        of = "patch{:02d}.png".format(idx)
        cv2.imwrite(of, patch)
