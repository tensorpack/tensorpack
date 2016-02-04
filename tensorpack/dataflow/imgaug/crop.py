#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: crop.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
import cv2

__all__ = ['RandomCrop', 'CenterCrop', 'Resize']

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

class Resize(ImageAugmentor):
    """Resize image to a target size"""
    def __init__(self, shape):
        """
        Args:
            shape: (h, w)
        """
        self._init(locals())

    def _augment(self, img):
        img.arr = cv2.resize(
            img.arr, self.shape[::-1],
            interpolation=cv2.INTER_CUBIC)
