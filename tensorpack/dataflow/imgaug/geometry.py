#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: geometry.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import ImageAugmentor
import cv2
import numpy as np

__all__ = ['Rotation']

class Rotation(ImageAugmentor):
    """ Random rotate the image w.r.t a random center"""
    def __init__(self, max_deg, center_range=(0,1),
            interp=cv2.INTER_CUBIC,
            border=cv2.BORDER_REPLICATE):
        """
        :param max_deg: max abs value of the rotation degree
        :param center_range: the location of the rotation center
        """
        self._init(locals())

    def _get_augment_params(self, img):
        center = img.shape[1::-1] * self._rand_range(
                self.center_range[0], self.center_range[1], (2,))
        deg = self._rand_range(-self.max_deg, self.max_deg)
        return cv2.getRotationMatrix2D(tuple(center), deg, 1)

    def _augment(self, img, rot_m):
        ret = cv2.warpAffine(img, rot_m, img.shape[1::-1],
                flags=self.interp, borderMode=self.border)
        return ret

