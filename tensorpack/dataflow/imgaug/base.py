# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import abstractmethod, ABCMeta
from ...utils import get_rng

__all__ = ['Image', 'ImageAugmentor', 'AugmentorList']

class Image(object):
    """ An image class with attributes, for augmentor to operate on.
        Attributes (such as coordinates) have to be augmented acoordingly by
        the augmentor, if necessary.
    """
    def __init__(self, arr, coords=None):
        """
        :param arr: the image array. Expected to be of [h, w, c] or [h, w]
        :param coords: keypoint coordinates.
        """
        self.arr = arr
        self.coords = coords

class ImageAugmentor(object):
    """ Base class for an image augmentor"""
    __metaclass__ = ABCMeta

    def __init__(self):
        self.reset_state()

    def _init(self, params=None):
        self.reset_state()
        if params:
            for k, v in params.items():
                if k != 'self':
                    setattr(self, k, v)

    def reset_state(self):
        self.rng = get_rng(self)

    def augment(self, img):
        """
        Perform augmentation on the image in-place.
        :param img: an `Image` instance.
        :returns: the augmented `Image` instance. arr will always be of type
        'float32' after augmentation.
        """
        self._augment(img)
        return img

    @abstractmethod
    def _augment(self, img):
        pass

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
        """
        :param augmentors: list of `ImageAugmentor` instance to be applied
        """
        self.augs = augmentors

    def _augment(self, img):
        assert img.arr.ndim in [2, 3], img.arr.ndim
        img.arr = img.arr.astype('float32')
        for aug in self.augs:
            aug.augment(img)

    def reset_state(self):
        """ Will reset state of each augmentor """
        for a in self.augs:
            a.reset_state()
