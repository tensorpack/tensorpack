# -*- coding: UTF-8 -*-
# File: imgproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
import numpy as np
import cv2

__all__ = ['Brightness', 'Contrast', 'MeanVarianceNormalize', 'GaussianBlur',
        'Gamma', 'Clip']

class Brightness(ImageAugmentor):
    """
    Random adjust brightness.
    """
    def __init__(self, delta, clip=True):
        """
        Randomly add a value within [-delta,delta], and clip in [0,255] if clip is True.
        """
        super(Brightness, self).__init__()
        assert delta > 0
        self._init(locals())

    def _get_augment_params(self, img):
        v = self._rand_range(-self.delta, self.delta)
        return v

    def _augment(self, img, v):
        img += v
        if self.clip:
            img = np.clip(img, 0, 255)
        return img

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
        super(Contrast, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        return self._rand_range(*self.factor_range)

    def _augment(self, img, r):
        mean = np.mean(img, axis=(0,1), keepdims=True)
        img = (img - mean) * r + mean
        if self.clip:
            img = np.clip(img, 0, 255)
        return img

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

    def _augment(self, img, _):
        if self.all_channel:
            mean = np.mean(img)
            std = np.std(img)
        else:
            mean = np.mean(img, axis=(0,1), keepdims=True)
            std = np.std(img, axis=(0,1), keepdims=True)
        std = np.maximum(std, 1.0 / np.sqrt(np.prod(img.shape)))
        img = (img - mean) / std
        return img


class GaussianBlur(ImageAugmentor):
    def __init__(self, max_size=3):
        """:params max_size: (maximum kernel size-1)/2"""
        super(GaussianBlur, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        sx, sy = self.rng.randint(self.max_size, size=(2,))
        sx = sx * 2 + 1
        sy = sy * 2 + 1
        return sx, sy

    def _augment(self, img, s):
        return cv2.GaussianBlur(img, s, sigmaX=0, sigmaY=0,
                borderType=cv2.BORDER_REPLICATE)


class Gamma(ImageAugmentor):
    def __init__(self, range=(-0.5, 0.5)):
        super(Gamma, self).__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        return self._rand_range(*self.range)

    def _augment(self, img, gamma):
        lut = ((np.arange(256, dtype='float32') / 255) ** (1. / (1. + gamma)) * 255).astype('uint8')
        img = np.clip(img, 0, 255).astype('uint8')
        img = cv2.LUT(img, lut).astype('float32')
        return img

class Clip(ImageAugmentor):
    def __init__(self, min=0, max=255):
        assert delta > 0
        self._init(locals())

    def _augment(self, img, _):
        img = np.clip(img, self.min, self.max)
        return img
