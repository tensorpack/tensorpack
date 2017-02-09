# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import abstractmethod, ABCMeta
from ...utils import get_rng
import six
from six.moves import zip

__all__ = ['Augmentor', 'ImageAugmentor', 'AugmentorList']


@six.add_metaclass(ABCMeta)
class Augmentor(object):
    """ Base class for an augmentor"""

    def __init__(self):
        self.reset_state()

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self':
                    setattr(self, k, v)

    def reset_state(self):
        """ reset rng and other state """
        self.rng = get_rng(self)

    def augment(self, d):
        """
        Perform augmentation on the data.
        """
        d, params = self._augment_return_params(d)
        return d

    def _augment_return_params(self, d):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(d)
        return (self._augment(d, prms), prms)

    @abstractmethod
    def _augment(self, d, param):
        """
        augment with the given param and return the new image
        """

    def _get_augment_params(self, d):
        """
        get the augmentor parameters
        """
        return None

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)


class ImageAugmentor(Augmentor):

    def augment(self, img):
        """
        Perform augmentation on the image (possibly) in-place.

        Args:
            img (np.ndarray): an [h,w] or [h,w,c] image.

        Returns:
            np.ndarray: the augmented image, always of type float32.
        """
        img, params = self._augment_return_params(img)
        return img

    def _fprop_coord(self, coord, param):
        return coord


class AugmentorList(ImageAugmentor):
    """
    Augment by a list of augmentors
    """

    def __init__(self, augmentors):
        """
        Args:
            augmentors (list): list of :class:`ImageAugmentor` instance to be applied.
        """
        self.augs = augmentors
        super(AugmentorList, self).__init__()

    def _get_augment_params(self, img):
        # the next augmentor requires the previous one to finish
        raise RuntimeError("Cannot simply get parameters of a AugmentorList!")

    def _augment_return_params(self, img):
        assert img.ndim in [2, 3], img.ndim

        prms = []
        for a in self.augs:
            img, prm = a._augment_return_params(img)
            prms.append(prm)
        return img, prms

    def _augment(self, img, param):
        assert img.ndim in [2, 3], img.ndim
        for aug, prm in zip(self.augs, param):
            img = aug._augment(img, prm)
        return img

    def reset_state(self):
        """ Will reset state of each augmentor """
        for a in self.augs:
            a.reset_state()
