# -*- coding: UTF-8 -*-
# File: crop.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor

import numpy as np
from abc import abstractmethod

__all__ = ['RandomCrop', 'CenterCrop', 'FixedCrop', 'CenterPaste']

class RandomCrop(ImageAugmentor):
    """ Randomly crop the image into a smaller one """
    def __init__(self, crop_shape):
        """
        Args:
            crop_shape: shape in (h, w)
        """
        self._init(locals())

    def _augment(self, img):
        orig_shape = img.arr.shape
        h0 = self.rng.randint(0, orig_shape[0] - self.crop_shape[0])
        w0 = self.rng.randint(0, orig_shape[1] - self.crop_shape[1])
        img.arr = img.arr[h0:h0+self.crop_shape[0],w0:w0+self.crop_shape[1]]
        if img.coords:
            raise NotImplementedError()

class CenterCrop(ImageAugmentor):
    """ Crop the image in the center"""
    def __init__(self, crop_shape):
        self._init(locals())

    def _augment(self, img):
        orig_shape = img.arr.shape
        h0 = (orig_shape[0] - self.crop_shape[0]) * 0.5
        w0 = (orig_shape[1] - self.crop_shape[1]) * 0.5
        img.arr = img.arr[h0:h0+self.crop_shape[0],w0:w0+self.crop_shape[1]]
        if img.coords:
            raise NotImplementedError()

class FixedCrop(ImageAugmentor):
    """ Crop a rectangle at a given location"""
    def __init__(self, rangex, rangey):
        self._init(locals())

    def _augment(self, img):
        orig_shape = img.arr.shape
        img.arr = img.arr[self.rangey[0]:self.rangey[1],
                          self.rangex[0]:self.rangex[1]]
        if img.coords:
            raise NotImplementedError()

class BackgroundFiller(object):
    @abstractmethod
    def fill(background_shape, img):
        """
        return a proper background image of background_shape, given img
        """

class ConstantBackgroundFiller(BackgroundFiller):
    def __init__(self, value):
        self.value = value

    def fill(self, background_shape, img):
        assert img.ndim in [3, 1]
        if img.ndim == 3:
            return_shape = background_shape + (3,)
        else:
            return_shape = background_shape
        return np.zeros(return_shape) + self.value

class CenterPaste(ImageAugmentor):
    """
    Paste the image onto center of a background
    """
    def __init__(self, background_shape, background_filler=None):
        if background_filler is None:
            background_filler = ConstantBackgroundFiller(0)

        self._init(locals())

    def _augment(self, img):
        img_shape = img.arr.shape[:2]
        assert self.background_shape[0] > img_shape[0] and self.background_shape[1] > img_shape[1]

        background = self.background_filler.fill(
            self.background_shape, img.arr)
        h0 = (self.background_shape[0] - img_shape[0]) * 0.5
        w0 = (self.background_shape[1] - img_shape[1]) * 0.5
        background[h0:h0+img_shape[0], w0:w0+img_shape[1]] = img.arr
        img.arr = background
        if img.coords:
            raise NotImplementedError()


