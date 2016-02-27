#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: crop.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor

__all__ = ['RandomCrop', 'CenterCrop', 'FixedCrop']

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
