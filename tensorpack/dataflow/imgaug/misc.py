# -*- coding: UTF-8 -*-
# File: misc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import cv2

from .base import ImageAugmentor
from ...utils import logger
from ...utils.argtools import shape2d
from .transform import ResizeTransform, TransformAugmentorBase

__all__ = ['Flip', 'Resize', 'RandomResize', 'ResizeShortestEdge', 'Transpose']


class Flip(ImageAugmentor):
    """
    Random flip the image either horizontally or vertically.
    """
    def __init__(self, horiz=False, vert=False, prob=0.5):
        """
        Args:
            horiz (bool): use horizontal flip.
            vert (bool): use vertical flip.
            prob (float): probability of flip.
        """
        super(Flip, self).__init__()
        if horiz and vert:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        elif horiz:
            self.code = 1
        elif vert:
            self.code = 0
        else:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        return (do, h, w)

    def _augment(self, img, param):
        do, _, _ = param
        if do:
            ret = cv2.flip(img, self.code)
            if img.ndim == 3 and ret.ndim == 2:
                ret = ret[:, :, np.newaxis]
        else:
            ret = img
        return ret

    def _augment_coords(self, coords, param):
        do, h, w = param
        if do:
            if self.code == 0:
                coords[:, 1] = h - coords[:, 1]
            elif self.code == 1:
                coords[:, 0] = w - coords[:, 0]
        return coords


class Resize(TransformAugmentorBase):
    """ Resize image to a target size"""

    def __init__(self, shape, interp=cv2.INTER_LINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: cv2 interpolation method
        """
        shape = tuple(shape2d(shape))
        self._init(locals())

    def _get_augment_params(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1],
            self.shape[0], self.shape[1], self.interp)


class ResizeShortestEdge(TransformAugmentorBase):
    """
    Resize the shortest edge to a certain number while
    keeping the aspect ratio.
    """

    def __init__(self, size, interp=cv2.INTER_LINEAR):
        """
        Args:
            size (int): the size to resize the shortest edge to.
        """
        size = int(size)
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, int(scale * w + 0.5)
        else:
            newh, neww = int(scale * h + 0.5), self.size
        return ResizeTransform(
            h, w, newh, neww, self.interp)


class RandomResize(TransformAugmentorBase):
    """ Randomly rescale width and height of the image."""

    def __init__(self, xrange, yrange, minimum=(0, 0), aspect_ratio_thres=0.15,
                 interp=cv2.INTER_LINEAR):
        """
        Args:
            xrange (tuple): a (min, max) tuple. If is floating point, the
                tuple defines the range of scaling ratio of new width, e.g. (0.9, 1.2).
                If is integer, the tuple defines the range of new width in pixels, e.g. (200, 350).
            yrange (tuple): similar to xrange, but for height.
            minimum (tuple): (xmin, ymin) in pixels. To avoid scaling down too much.
            aspect_ratio_thres (float): discard samples which change aspect ratio
                larger than this threshold. Set to 0 to keep aspect ratio.
            interp: cv2 interpolation method
        """
        super(RandomResize, self).__init__()
        assert aspect_ratio_thres >= 0
        if aspect_ratio_thres == 0:
            assert xrange == yrange
        self._init(locals())

        def is_float(tp):
            return isinstance(tp[0], float) or isinstance(tp[1], float)
        assert is_float(xrange) == is_float(yrange), "xrange and yrange has different type!"
        self._is_scale = is_float(xrange)

    def _get_augment_params(self, img):
        cnt = 0
        h, w = img.shape[:2]

        def get_dest_size():
            if self._is_scale:
                sx = self._rand_range(*self.xrange)
                if self.aspect_ratio_thres == 0:
                    sy = sx
                else:
                    sy = self._rand_range(*self.yrange)
                destX = max(sx * w, self.minimum[0])
                destY = max(sy * h, self.minimum[1])
            else:
                sx = self._rand_range(*self.xrange)
                if self.aspect_ratio_thres == 0:
                    sy = sx * 1.0 / w * h
                else:
                    sy = self._rand_range(*self.yrange)
                destX = max(sx, self.minimum[0])
                destY = max(sy, self.minimum[1])
            return (int(destX + 0.5), int(destY + 0.5))

        while True:
            destX, destY = get_dest_size()
            if self.aspect_ratio_thres > 0:  # don't check when thres == 0
                oldr = w * 1.0 / h
                newr = destX * 1.0 / destY
                diff = abs(newr - oldr) / oldr
                if diff >= self.aspect_ratio_thres + 1e-5:
                    cnt += 1
                    if cnt > 50:
                        logger.warn("RandomResize failed to augment an image")
                        return ResizeTransform(h, w, h, w, self.interp)
                    continue
            return ResizeTransform(h, w, destY, destX, self.interp)


class Transpose(ImageAugmentor):
    """
    Random transpose the image
    """
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of transpose.
        """
        super(Transpose, self).__init__()
        self.prob = prob
        self._init()

    def _get_augment_params(self, img):
        return self._rand_range() < self.prob

    def _augment(self, img, do):
        ret = img
        if do:
            ret = cv2.transpose(img)
            if img.ndim == 3 and ret.ndim == 2:
                ret = ret[:, :, np.newaxis]
        return ret

    def _augment_coords(self, coords, do):
        if do:
            coords = coords[:, ::-1]
        return coords
