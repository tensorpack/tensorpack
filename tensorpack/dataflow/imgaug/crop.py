# -*- coding: UTF-8 -*-
# File: crop.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor

import numpy as np
from abc import abstractmethod

__all__ = ['RandomCrop', 'CenterCrop', 'FixedCrop', 'CenterPaste',
           'ConstantBackgroundFiller']

class RandomCrop(ImageAugmentor):
    """ Randomly crop the image into a smaller one """
    def __init__(self, crop_shape):
        """
        :param crop_shape: a shape like (h, w)
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
    """ Crop the image at the center"""
    def __init__(self, crop_shape):
        """
        :param crop_shape: a shape like (h, w)
        """
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
        """
        Two arguments defined the range in both axes to crop, min inclued, max excluded.

        :param rangex: like (xmin, xmax).
        :param rangey: like (ymin, ymax).
        """
        self._init(locals())

    def _augment(self, img):
        orig_shape = img.arr.shape
        img.arr = img.arr[self.rangey[0]:self.rangey[1],
                          self.rangex[0]:self.rangex[1]]
        if img.coords:
            raise NotImplementedError()

class BackgroundFiller(object):
    """ Base class for all BackgroundFiller"""
    def fill(self, background_shape, img):
        """
        Return a proper background image of background_shape, given img

        :param background_shape: a shape of [h, w]
        :param img: an image
        :returns: a background image
        """
        return self._fill(background_shape, img)

    @abstractmethod
    def _fill(self, background_shape, img):
        pass

class ConstantBackgroundFiller(BackgroundFiller):
    """ Fill the background by a constant """
    def __init__(self, value):
        """
        :param value: the value to fill the background.
        """
        self.value = value

    def _fill(self, background_shape, img):
        assert img.ndim in [3, 1]
        if img.ndim == 3:
            return_shape = background_shape + (3,)
        else:
            return_shape = background_shape
        return np.zeros(return_shape) + self.value

class CenterPaste(ImageAugmentor):
    """
    Paste the image onto the center of a background canvas.
    """
    def __init__(self, background_shape, background_filler=None):
        """
        :param background_shape: shape of the background canvas.
        :param background_filler: a `BackgroundFiller` instance. Default to zero-filler.
        """
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


