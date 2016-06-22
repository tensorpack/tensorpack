# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import abstractmethod, ABCMeta
from ...utils import get_rng
from six.moves import zip

__all__ = ['ImageAugmentor', 'AugmentorList', 'AugmentWithFunc']

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
        :param img: an [h,w] or [h,w,c] image
        :returns: the augmented image, always of type 'float32'
        """
        img, params = self._augment_return_params(img)
        return img

    def _augment_return_params(self, img):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(img)
        return (self._augment(img, prms), prms)

    @abstractmethod
    def _augment(self, img, param):
        """
        augment with the given param and return the new image
        """

    def _get_augment_params(self, img):
        """
        get the augmentor parameters
        """
        return None

    def _fprop_coord(self, coord, param):
        return coord

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size == None:
            size = []
        return low + self.rng.rand(*size) * (high - low)

class AugmentWithFunc(ImageAugmentor):
    """ func: takes an image and return an image"""
    def __init__(self, func):
        self.func = func

    def _augment(self, img, _):
        return self.func(img)

class AugmentorList(ImageAugmentor):
    """
    Augment by a list of augmentors
    """
    def __init__(self, augmentors):
        """
        :param augmentors: list of `ImageAugmentor` instance to be applied
        """
        self.augs = augmentors
        super(AugmentorList, self).__init__()

    def _get_augment_params(self, img):
        raise RuntimeError("Cannot simply get parameters of a AugmentorList!")

    def _augment_return_params(self, img):
        assert img.ndim in [2, 3], img.ndim
        img = img.astype('float32')

        prms = []
        for a in self.augs:
            img, prm = a._augment_return_params(img)
            prms.append(prm)
        return img, prms

    def _augment(self, img, param):
        assert img.ndim in [2, 3], img.ndim
        img = img.astype('float32')
        for aug, prm in zip(self.augs, param):
            img = aug._augment(img, prm)
        return img

    def reset_state(self):
        """ Will reset state of each augmentor """
        for a in self.augs:
            a.reset_state()


