# -*- coding: UTF-8 -*-
# File: noname.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
from ...utils import logger
from ...utils.argtools import shape2d
import numpy as np
import cv2

__all__ = ['Flip', 'Resize', 'RandomResize', 'ResizeShortestEdge']


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
        return self._rand_range() < self.prob

    def _augment(self, img, do):
        if do:
            img = cv2.flip(img, self.code)
        return img

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()


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

    def _augment(self, img, _):
        ret = cv2.resize(
            img, self.shape[::-1],
            interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret


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

    def _augment(self, img, _):
        h, w = img.shape[:2]
        scale = self.size / min(h, w)
        desSize = map(int, [scale * w, scale * h])
        ret = cv2.resize(img, tuple(desSize), interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret


class RandomResize(ImageAugmentor):
    """ Randomly rescale w and h of the image"""

    def __init__(self, xrange, yrange, minimum=(0, 0), aspect_ratio_thres=0.15,
                 interp=cv2.INTER_LINEAR):
        """
        Args:
            xrange (tuple): (min, max) range of scaling ratio for w
            yrange (tuple): (min, max) range of scaling ratio for h
            minimum (tuple): (xmin, ymin). avoid scaling down too much.
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
        while True:
            sx = self._rand_range(*self.xrange)
            if self.aspect_ratio_thres == 0:
                sy = sx
            else:
                sy = self._rand_range(*self.yrange)
            destX = max(sx * img.shape[1], self.minimum[0])
            destY = max(sy * img.shape[0], self.minimum[1])
            oldr = img.shape[1] * 1.0 / img.shape[0]
            newr = destX * 1.0 / destY
            diff = abs(newr - oldr) / oldr
            if diff <= self.aspect_ratio_thres + 1e-5:
                return (int(destX), int(destY))
            cnt += 1
            if cnt > 50:
                logger.warn("RandomResize failed to augment an image")
                return img.shape[1], img.shape[0]

    def _augment(self, img, dsize):
        ret = cv2.resize(img, dsize, interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret
