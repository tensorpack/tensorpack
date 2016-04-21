# -*- coding: UTF-8 -*-
# File: imgproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
import numpy as np

__all__ = ['Brightness', 'Contrast', 'MeanVarianceNormalize']

class Brightness(ImageAugmentor):
    """
    Random adjust brightness.
    """
    def __init__(self, delta, clip=True):
        """
        Randomly add a value within [-delta,delta], and clip in [0,255] if clip is True.
        """
        assert delta > 0
        self._init(locals())

    def _augment(self, img):
        v = self._rand_range(-self.delta, self.delta)
        img.arr += v
        if self.clip:
            img.arr = np.clip(img.arr, 0, 255)

class Contrast(ImageAugmentor):
    """
    Apply x = (x - mean) * contrast_factor + mean to each channel
    and clip to [0, 255]
    """
    def __init__(self, factor_range, clip=True):
        """
        :param factor_range: an interval to random sample the `contrast_factor`.
        :param clip: boolean.
        """
        self._init(locals())

    def _augment(self, img):
        arr = img.arr
        r = self._rand_range(*self.factor_range)
        mean = np.mean(arr, axis=(0,1), keepdims=True)
        img.arr = (arr - mean) * r + mean
        if self.clip:
            img.arr = np.clip(img.arr, 0, 255)

class MeanVarianceNormalize(ImageAugmentor):
    """
    Linearly scales image to have zero mean and unit norm.
    x = (x - mean) / adjusted_stddev
    where adjusted_stddev = max(stddev, 1.0/sqrt(num_pixels * channels))
    """
    def __init__(self, all_channel=True):
        """
        :param all_channel: if True, normalize all channels together. else separately.
        """
        self.all_channel = all_channel

    def _augment(self, img):
        if self.all_channel:
            mean = np.mean(img.arr)
            std = np.std(img.arr)
        else:
            mean = np.mean(img.arr, axis=(0,1), keepdims=True)
            std = np.std(img.arr, axis=(0,1), keepdims=True)
        std = np.maximum(std, 1.0 / np.sqrt(np.prod(img.arr.shape)))
        img.arr = (img.arr - mean) / std
