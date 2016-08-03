#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: noise.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import ImageAugmentor
import numpy as np
import cv2

__all__ = ['JpegNoise', 'GaussianNoise']

class JpegNoise(ImageAugmentor):
    def __init__(self, quality_range=(40, 100)):
        self._init(locals())

    def _get_augment_params(self, img):
        return self.rng.randint(*self.quality_range)

    def _augment(self, img, q):
        enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        return cv2.imdecode(enc, 1)


class GaussianNoise(ImageAugmentor):
    def __init__(self, scale=10, clip=True):
        self._init(locals())

    def _get_augment_params(self, img):
        return self.rng.randn(*img.shape)

    def _augment(self, img, noise):
        ret = img + noise
        if self.clip:
            ret = np.clip(ret, 0, 255)
        return ret
