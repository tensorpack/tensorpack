#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: paste.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import ImageAugmentor

from abc import abstractmethod
import numpy as np

__all__ = [ 'CenterPaste', 'ConstantBackgroundFiller']


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


