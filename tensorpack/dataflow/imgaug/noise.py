#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: noise.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import ImageAugmentor
import numpy as np
import cv2

__all__ = ['JpegNoise', 'GaussianNoise', 'SaltPepperNoise']


class JpegNoise(ImageAugmentor):
    """ Random Jpeg noise. """

    def __init__(self, quality_range=(40, 100)):
        """
        Args:
            quality_range (tuple): range to sample Jpeg quality
        """
        super(JpegNoise, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        return self.rng.randint(*self.quality_range)

    def _augment(self, img, q):
        enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        return cv2.imdecode(enc, 1).astype(img.dtype)


class GaussianNoise(ImageAugmentor):
    """
    Add random Gaussian noise N(0, sigma^2) of the same shape to img.
    """
    def __init__(self, sigma=1, clip=True):
        """
        Args:
            sigma (float): stddev of the Gaussian distribution.
            clip (bool): clip the result to [0,255] in the end.
        """
        super(GaussianNoise, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        return self.rng.randn(*img.shape)

    def _augment(self, img, noise):
        old_dtype = img.dtype
        ret = img + noise * self.sigma
        if self.clip or old_dtype == np.uint8:
            ret = np.clip(ret, 0, 255)
        return ret.astype(old_dtype)


class SaltPepperNoise(ImageAugmentor):
    """ Salt and pepper noise.
        Randomly set some elements in img to 0 or 255, regardless of its channels.
    """

    def __init__(self, white_prob=0.05, black_prob=0.05):
        """
        Args:
            white_prob (float), black_prob (float): probabilities setting an element to 255 or 0.
        """
        assert white_prob + black_prob <= 1, "Sum of probabilities cannot be greater than 1"
        super(SaltPepperNoise, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        return self.rng.uniform(low=0, high=1, size=img.shape)

    def _augment(self, img, param):
        img[param > (1 - self.white_prob)] = 255
        img[param < self.black_prob] = 0
        return img
