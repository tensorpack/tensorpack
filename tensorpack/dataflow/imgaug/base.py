#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import abstractmethod, ABCMeta
from ...utils import get_rng

__all__ = ['Image', 'ImageAugmentor', 'AugmentorList']

class Image(object):
    """ An image with attributes, for augmentor to operate on
        Attributes (such as coordinates) have to be augmented acoordingly, if necessary
    """
    def __init__(self, arr, coords=None):
        self.arr = arr
        self.coords = coords

class ImageAugmentor(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.rng = get_rng(self)

    def _init(self, params=None):
        self.rng = get_rng(self)
        if params:
            for k, v in params.iteritems():
                if k != 'self':
                    setattr(self, k, v)

    def augment(self, img):
        """
        Note: will both modify `img` in-place and return `img`
        """
        self._augment(img)
        return img

    @abstractmethod
    def _augment(self, img):
        """
        Augment the image in-place. Will always make it float32 array.
        Args:
            img: the input Image instance
                img.arr must be of shape [h, w] or [h, w, c]
        """

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size == None:
            size = []
        return low + self.rng.rand(*size) * (high - low)

class AugmentorList(ImageAugmentor):
    """
    Augment by a list of augmentors
    """
    def __init__(self, augmentors):
        self.augs = augmentors

    def _augment(self, img):
        img.arr = img.arr.astype('float32') / 255.0
        for aug in self.augs:
            aug.augment(img)
