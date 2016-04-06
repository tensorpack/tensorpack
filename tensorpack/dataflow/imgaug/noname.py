# -*- coding: UTF-8 -*-
# File: noname.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
import numpy as np
import cv2

__all__ = ['Flip', 'MapImage', 'Resize']

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
            raise ValueError("Please use two Flip, with both 0.5 prob")
        elif horiz:
            self.code = 1
        elif vert:
            self.code = 0
        else:
            raise ValueError("Are you kidding?")
        self.prob = prob
        self._init()

    def _augment(self, img):
        if self._rand_range() < self.prob:
            img.arr = cv2.flip(img.arr, self.code)
            if img.coords:
                raise NotImplementedError()


class MapImage(ImageAugmentor):
    """
    Map the image array by a function.
    """
    def __init__(self, func):
        """
        :param func: a function which takes a image array and return a augmented one
        """
        self.func = func

    def _augment(self, img):
        img.arr = self.func(img.arr)


class Resize(ImageAugmentor):
    """ Resize image to a target size"""
    def __init__(self, shape):
        """
        :param shape: shape in (h, w)
        """
        self._init(locals())

    def _augment(self, img):
        img.arr = cv2.resize(
            img.arr, self.shape[::-1],
            interpolation=cv2.INTER_CUBIC)
