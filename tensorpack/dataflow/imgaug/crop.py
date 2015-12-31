#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: crop.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor

__all__ = ['RandomCrop']

class RandomCrop(ImageAugmentor):
    def __init__(self, crop_shape):
        """
        Randomly crop the image into a smaller one
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


