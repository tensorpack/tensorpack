# -*- coding: UTF-8 -*-
# File: noname.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
from ...utils import logger
from ...utils.argtools import shape2d
import numpy as np
import cv2

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
            raise ValueError("Are you kidding?")
        self.prob = prob
        self._init()

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


class Resize(ImageAugmentor):
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
        h, w = img.shape[:2]
        return (h, w)

    def _augment(self, img, _):
        ret = cv2.resize(
            img, self.shape[::-1],
            interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def _augment_coords(self, coords, param):
        h, w = param
        coords[:, 0] = coords[:, 0] * (self.shape[1] * 1.0 / w)
        coords[:, 1] = coords[:, 1] * (self.shape[0] * 1.0 / h)
        return coords


class ResizeShortestEdge(ImageAugmentor):
    """
    Resize the shortest edge to a certain number while
    keeping the aspect ratio.
    """

    def __init__(self, size, interp=cv2.INTER_LINEAR):
        """
        Args:
            size (int): the size to resize the shortest edge to.
        """
        size = size * 1.0
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        scale = self.size / min(h, w)
        newh, neww = map(int, [scale * h, scale * w])
        return (h, w, newh, neww)

    def _augment(self, img, param):
        _, _, newh, neww = param
        ret = cv2.resize(img, (neww, newh), interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def _augment_coords(self, coords, param):
        h, w, newh, neww = param
        coords[:, 0] = coords[:, 0] * (neww * 1.0 / w)
        coords[:, 1] = coords[:, 1] * (newh * 1.0 / h)
        return coords


class RandomResize(ImageAugmentor):
    """ Randomly rescale w and h of the image"""

    def __init__(self, xrange, yrange, minimum=(0, 0), aspect_ratio_thres=0.15,
                 interp=cv2.INTER_LINEAR):
        """
        Args:
            xrange (tuple): (min, max) range of scaling ratio for w, e.g. (0.9, 1.2)
            yrange (tuple): (min, max) range of scaling ratio for h
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

    def _get_augment_params(self, img):
        cnt = 0
        h, w = img.shape[:2]
        while True:
            sx = self._rand_range(*self.xrange)
            if self.aspect_ratio_thres == 0:
                sy = sx
            else:
                sy = self._rand_range(*self.yrange)
            destX = max(sx * w, self.minimum[0])
            destY = max(sy * h, self.minimum[1])
            oldr = w * 1.0 / h
            newr = destX * 1.0 / destY
            diff = abs(newr - oldr) / oldr
            if diff <= self.aspect_ratio_thres + 1e-5:
                return (h, w, int(destY), int(destX))
            cnt += 1
            if cnt > 50:
                logger.warn("RandomResize failed to augment an image")
                return (h, w, h, w)

    def _augment(self, img, param):
        _, _, newh, neww = param
        ret = cv2.resize(img, (neww, newh), interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def _augment_coords(self, coords, param):
        h, w, newh, neww = param
        coords[:, 0] = coords[:, 0] * (neww * 1.0 / w)
        coords[:, 1] = coords[:, 1] * (newh * 1.0 / h)
        return coords


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
