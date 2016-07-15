# -*- coding: UTF-8 -*-
# File: noname.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
from ...utils import logger
import numpy as np
import cv2

__all__ = ['Flip', 'Resize', 'RandomResize']

class Flip(ImageAugmentor):
    """
    Random flip.
    """
    def __init__(self, horiz=False, vert=False, prob=0.5):
        """
        Only one of horiz, vert can be set.

        :param horiz: whether or not apply horizontal flip.
        :param vert: whether or not apply vertical flip.
        :param prob: probability of flip.
        """
        if horiz and vert:
            raise ValueError("Please use two Flip instead.")
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
    def __init__(self, shape):
        """
        :param shape: shape in (h, w)
        """
        self._init(locals())

    def _augment(self, img, _):
        return cv2.resize(
            img, self.shape[::-1],
            interpolation=cv2.INTER_CUBIC)

class RandomResize(ImageAugmentor):
    """ randomly rescale w and h of the image"""
    def __init__(self, xrange, yrange, minimum=(0,0), aspect_ratio_thres=0.15):
        """
        :param xrange: (min, max) scaling ratio
        :param yrange: (min, max) scaling ratio
        :param minimum: (xmin, ymin). Avoid scaling down too much.
        :param aspect_ratio_thres: at most change k=20% aspect ratio
        """
        self._init(locals())

    def _get_augment_params(self, img):
        cnt = 0
        while True:
            sx = self._rand_range(*self.xrange)
            sy = self._rand_range(*self.yrange)
            destX = int(max(sx * img.shape[1], self.minimum[0]))
            destY = int(max(sy * img.shape[0], self.minimum[1]))
            oldr = img.shape[1] * 1.0 / img.shape[0]
            newr = destX * 1.0 / destY
            diff = abs(newr - oldr) / oldr
            if diff <= self.aspect_ratio_thres:
                return (destX, destY)
            cnt += 1
            if cnt > 50:
                logger.warn("RandomResize failed to augment an image")

    def _augment(self, img, dsize):
        return cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)

