#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: noise.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import ImageAugmentor
import numpy as np
import cv2

__all__ = ['JpegNoise', 'GaussianNoise', 'SaltPepperNoise']

class JpegNoise(ImageAugmentor):
    def __init__(self, quality_range=(40, 100)):
        super(JpegNoise, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        return self.rng.randint(*self.quality_range)

    def _augment(self, img, q):
        enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        return cv2.imdecode(enc, 1)


class GaussianNoise(ImageAugmentor):
    def __init__(self, sigma=1, clip=True):
        """
        Add a gaussian noise N(0, sigma^2) of the same shape to img.
        """
        super(GaussianNoise, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        return self.rng.randn(*img.shape)

    def _augment(self, img, noise):
        ret = img + noise * self.sigma
        if self.clip:
            ret = np.clip(ret, 0, 255)
        return ret

class SaltPepperNoise(ImageAugmentor):
    def __init__(self, white_prob=0.05, black_prob=0.05):
        """ Salt and pepper noise.
            Randomly set some elements in img to 0 or 255, regardless of its channels.
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
