# -*- coding: utf-8 -*-
# File: viz.py
# Credit: zxytim

import numpy as np
import os
import sys

from ..utils.develop import create_dummy_func  # noqa
from .argtools import shape2d
from .fs import mkdir_p

try:
    import cv2
except ImportError:
    pass


__all__ = ['interactive_imshow',
           'stack_patches', 'gen_stack_patches',
           'dump_dataflow_images', 'intensity_to_rgb',
           'draw_boxes']


def interactive_imshow(img, lclick_cb=None, rclick_cb=None, **kwargs):
    """
    Args:
        img (np.ndarray): an image (expect BGR) to show.
        lclick_cb, rclick_cb: a callback ``func(img, x, y)`` for left/right click event.
        kwargs: can be {key_cb_a: callback_img, key_cb_b: callback_img}, to
            specify a callback ``func(img)`` for keypress.

    Some existing keypress event handler:

    * q: destroy the current window
    * x: execute ``sys.exit()``
    * s: save image to "out.png"
    """
    name = 'tensorpack_viz_window'
    cv2.imshow(name, img)

    def mouse_cb(event, x, y, *args):
        if event == cv2.EVENT_LBUTTONUP and lclick_cb is not None:
            lclick_cb(img, x, y)
        elif event == cv2.EVENT_RBUTTONUP and rclick_cb is not None:
            rclick_cb(img, x, y)
    cv2.setMouseCallback(name, mouse_cb)
    key = cv2.waitKey(-1)
    while key >= 128:
        key = cv2.waitKey(-1)
    key = chr(key & 0xff)
    cb_name = 'key_cb_' + key
    if cb_name in kwargs:
        kwargs[cb_name](img)
    elif key == 'q':
        cv2.destroyWindow(name)
    elif key == 'x':
        sys.exit()
    elif key == 's':
        cv2.imwrite('out.png', img)
    elif key in ['+', '=']:
        img = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
        interactive_imshow(img, lclick_cb, rclick_cb, **kwargs)
    elif key == '-':
        img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
        interactive_imshow(img, lclick_cb, rclick_cb, **kwargs)


def _preprocess_patch_list(plist):
    plist = np.asarray(plist)
    assert plist.dtype != np.object
    if plist.ndim == 3:
        plist = plist[:, :, :, np.newaxis]
    assert plist.ndim == 4 and plist.shape[3] in [1, 3], plist.shape
    return plist


def _pad_patch_list(plist, bgcolor):
    if isinstance(bgcolor, int):
        bgcolor = (bgcolor, bgcolor, bgcolor)

    def _pad_channel(plist):
        ret = []
        for p in plist:
            if len(p.shape) == 2:
                p = p[:, :, np.newaxis]
            if p.shape[2] == 1:
                p = np.repeat(p, 3, 2)
            ret.append(p)
        return ret

    plist = _pad_channel(plist)
    shapes = [x.shape for x in plist]
    ph = max(s[0] for s in shapes)
    pw = max(s[1] for s in shapes)

    ret = np.zeros((len(plist), ph, pw, 3), dtype=plist[0].dtype)
    ret[:, :, :] = bgcolor
    for idx, p in enumerate(plist):
        s = p.shape
        sh = (ph - s[0]) // 2
        sw = (pw - s[1]) // 2
        ret[idx, sh:sh + s[0], sw:sw + s[1], :] = p
    return ret


class Canvas(object):
    def __init__(self, ph, pw,
                 nr_row, nr_col,
                 channel, border, bgcolor):
        self.ph = ph
        self.pw = pw
        self.nr_row = nr_row
        self.nr_col = nr_col

        if border is None:
            border = int(0.05 * min(ph, pw))
        self.border = border

        if isinstance(bgcolor, int):
            bgchannel = 1
        else:
            bgchannel = 3
        self.bgcolor = bgcolor
        self.channel = max(channel, bgchannel)

        self.canvas = np.zeros((nr_row * (ph + border) - border,
                               nr_col * (pw + border) - border,
                               self.channel), dtype='uint8')

    def draw_patches(self, plist):
        assert self.nr_row * self.nr_col >= len(plist), \
            "{}*{} < {}".format(self.nr_row, self.nr_col, len(plist))
        if self.channel == 3 and plist.shape[3] == 1:
            plist = np.repeat(plist, 3, axis=3)
        cur_row, cur_col = 0, 0
        if self.channel == 1:
            self.canvas.fill(self.bgcolor)
        else:
            self.canvas[:, :, :] = self.bgcolor
        for patch in plist:
            r0 = cur_row * (self.ph + self.border)
            c0 = cur_col * (self.pw + self.border)
            self.canvas[r0:r0 + self.ph, c0:c0 + self.pw] = patch
            cur_col += 1
            if cur_col == self.nr_col:
                cur_col = 0
                cur_row += 1

    def get_patchid_from_coord(self, x, y):
        x = x // (self.pw + self.border)
        y = y // (self.pw + self.border)
        idx = y * self.nr_col + x
        return idx


def stack_patches(
        patch_list, nr_row, nr_col, border=None,
        pad=False, bgcolor=255, viz=False, lclick_cb=None):
    """
    Stacked patches into grid, to produce visualizations like the following:

    .. image:: https://github.com/tensorpack/tensorpack/raw/master/examples/GAN/demo/BEGAN-CelebA-samples.jpg

    Args:
        patch_list(list[ndarray] or ndarray): NHW or NHWC images in [0,255].
        nr_row(int), nr_col(int): rows and cols of the grid.
            ``nr_col * nr_row`` must be no less than ``len(patch_list)``.
        border(int): border length between images.
            Defaults to ``0.05 * min(patch_width, patch_height)``.
        pad (boolean): when `patch_list` is a list, pad all patches to the maximum height and width.
            This option allows stacking patches of different shapes together.
        bgcolor(int or 3-tuple): background color in [0, 255]. Either an int
            or a BGR tuple.
        viz(bool): whether to use :func:`interactive_imshow` to visualize the results.
        lclick_cb: A callback function ``f(patch, patch index in patch_list)``
            to get called when a patch get clicked in imshow.

    Returns:
        np.ndarray: the stacked image.
    """
    if pad:
        patch_list = _pad_patch_list(patch_list, bgcolor)
    patch_list = _preprocess_patch_list(patch_list)

    if lclick_cb is not None:
        viz = True
    ph, pw = patch_list.shape[1:3]

    canvas = Canvas(ph, pw, nr_row, nr_col,
                    patch_list.shape[-1], border, bgcolor)

    if lclick_cb is not None:
        def lclick_callback(img, x, y):
            idx = canvas.get_patchid_from_coord(x, y)
            lclick_cb(patch_list[idx], idx)
    else:
        lclick_callback = None

    canvas.draw_patches(patch_list)
    if viz:
        interactive_imshow(canvas.canvas, lclick_cb=lclick_callback)
    return canvas.canvas


def gen_stack_patches(patch_list,
                      nr_row=None, nr_col=None, border=None,
                      max_width=1000, max_height=1000,
                      bgcolor=255, viz=False, lclick_cb=None):
    """
    Similar to :func:`stack_patches` but with a generator interface.
    It takes a much-longer list and yields stacked results one by one.
    For example, if ``patch_list`` contains 1000 images and ``nr_row==nr_col==10``,
    this generator yields 10 stacked images.

    Args:
        nr_row(int), nr_col(int): rows and cols of each result.
        max_width(int), max_height(int): Maximum allowed size of the
            stacked image. If ``nr_row/nr_col`` are None, this number
            will be used to infer the rows and cols. Otherwise the option is
            ignored.
        patch_list, border, viz, lclick_cb: same as in :func:`stack_patches`.

    Yields:
        np.ndarray: the stacked image.
    """
    # setup parameters
    patch_list = _preprocess_patch_list(patch_list)
    if lclick_cb is not None:
        viz = True
    ph, pw = patch_list.shape[1:3]

    if border is None:
        border = int(0.05 * min(ph, pw))
    if nr_row is None:
        nr_row = int(max_height / (ph + border))
    if nr_col is None:
        nr_col = int(max_width / (pw + border))
    canvas = Canvas(ph, pw, nr_row, nr_col, patch_list.shape[-1], border, bgcolor)

    nr_patch = nr_row * nr_col
    start = 0

    if lclick_cb is not None:
        def lclick_callback(img, x, y):
            idx = canvas.get_patchid_from_coord(x, y)
            idx = idx + start
            if idx < end:
                lclick_cb(patch_list[idx], idx)
    else:
        lclick_callback = None

    while True:
        end = start + nr_patch
        cur_list = patch_list[start:end]
        if not len(cur_list):
            return
        canvas.draw_patches(cur_list)
        if viz:
            interactive_imshow(canvas.canvas, lclick_cb=lclick_callback)
        yield canvas.canvas
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
            with :func:`gen_stack_patches` for visualization. No visualization will happen by
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
        for dp in df:
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
                stack_patches(
                    vizlist[:vizsize],
                    nr_row=viz[0], nr_col=viz[1], viz=True)
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


def draw_text(img, pos, text, color, font_scale=0.4):
    """
    Draw text on an image.

    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
        color (tuple): a 3-tuple BGR color in [0, 255]
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.15 * text_h) < 0:
        y0 = int(1.15 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, color, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.25 * text_h)
    cv2.putText(img, text, text_bottomleft, font, font_scale, (222, 222, 222), lineType=cv2.LINE_AA)
    return img


def draw_boxes(im, boxes, labels=None, color=None):
    """
    Args:
        im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
        boxes (np.ndarray): a numpy array of shape Nx4 where each row is [x1, y1, x2, y2].
        labels: (list[str] or None)
        color: a 3-tuple BGR color (in range [0, 255])

    Returns:
        np.ndarray: a new image.
    """
    boxes = np.asarray(boxes, dtype='int32')
    if labels is not None:
        assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)    # draw large ones first
    assert areas.min() > 0, areas.min()
    # allow equal, because we are not very strict about rounding error here
    assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
        and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
        "Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

    im = im.copy()
    if color is None:
        color = (15, 128, 15)
    if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for i in sorted_inds:
        box = boxes[i, :]
        if labels is not None:
            im = draw_text(im, (box[0], box[1]), labels[i], color=color)
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                      color=color, thickness=1)
    return im


try:
    import matplotlib.pyplot as plt
except (ImportError, RuntimeError):
    intensity_to_rgb = create_dummy_func('intensity_to_rgb', 'matplotlib')    # noqa

if __name__ == '__main__':
    if False:
        imglist = []
        for i in range(100):
            fname = "{:03d}.png".format(i)
            imglist.append(cv2.imread(fname))
        for idx, patch in enumerate(gen_stack_patches(
                imglist, max_width=500, max_height=200)):
            of = "patch{:02d}.png".format(idx)
            cv2.imwrite(of, patch)
    if False:
        imglist = []
        img = cv2.imread('out.png')
        img2 = cv2.resize(img, (300, 300))
        viz = stack_patches([img, img2], 1, 2, pad=True, viz=True)

    if False:
        img = cv2.imread('cat.jpg')
        boxes = np.asarray([
            [10, 30, 200, 100],
            [20, 80, 250, 250]
        ])
        img = draw_boxes(img, boxes, ['asdfasdf', '11111111111111'])
        interactive_imshow(img)
