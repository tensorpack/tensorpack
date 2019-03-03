# -*- coding: utf-8 -*-
# File: base.py

import os
import inspect
import pprint
from abc import ABCMeta, abstractmethod
import six
from six.moves import zip
import weakref

from ...utils.argtools import log_once
from ...utils.utils import get_rng
from ..image import check_dtype

__all__ = ['Augmentor', 'ImageAugmentor', 'AugmentorList']


def _reset_augmentor_after_fork(aug_ref):
    aug = aug_ref()
    if aug:
        aug.reset_state()


@six.add_metaclass(ABCMeta)
class Augmentor(object):
    """ Base class for an augmentor"""

    def __init__(self):
        self.reset_state()

        # only available on Unix after Python 3.7
        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(
                after_in_child=lambda: _reset_augmentor_after_fork(weakref.ref(self)))

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
                    setattr(self, k, v)

    def reset_state(self):
        """
        Reset rng and other state of the augmentor.

        Similar to :meth:`DataFlow.reset_state`, the caller of Augmentor
        is responsible for calling this method (once or more times) in the **process that uses the augmentor**
        before using it.

        If you use tensorpack's built-in augmentation dataflow (:class:`AugmentImageComponent`, etc),
        this method will be called in the dataflow's own `reset_state` method.

        If you use Python≥3.7 on Unix, this method will be automatically called after fork,
        and you do not need to bother calling it.
        """
        self.rng = get_rng(self)

    def augment(self, d):
        """
        Perform augmentation on the data.

        Args:
            d: input data

        Returns:
            augmented data
        """
        d, params = self._augment_return_params(d)
        return d

    def augment_return_params(self, d):
        """
        Augment the data and return the augmentation parameters.
        If the augmentation is non-deterministic (random),
        the returned parameters can be used to augment another data with the identical transformation.
        This can be used for, e.g. augmenting image, masks, keypoints altogether with the
        same transformation.

        Returns:
            (augmented data, augmentation params)
        """
        return self._augment_return_params(d)

    def _augment_return_params(self, d):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(d)
        return (self._augment(d, prms), prms)

    def augment_with_params(self, d, param):
        """
        Augment the data with the given param.

        Args:
            d: input data
            param: augmentation params returned by :meth:`augment_return_params`

        Returns:
            augmented data
        """
        return self._augment(d, param)

    @abstractmethod
    def _augment(self, d, param):
        """
        Augment with the given param and return the new data.
        The augmentor is allowed to modify data in-place.
        """

    def _get_augment_params(self, d):
        """
        Get the augmentor parameters.
        """
        return None

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "imgaug.MyAugmentor(field1={self.field1}, field2={self.field2})"
        """
        try:
            argspec = inspect.getargspec(self.__init__)
            assert argspec.varargs is None, "The default __repr__ doesn't work for varargs!"
            assert argspec.keywords is None, "The default __repr__ doesn't work for kwargs!"
            fields = argspec.args[1:]
            index_field_has_default = len(fields) - (0 if argspec.defaults is None else len(argspec.defaults))

            classname = type(self).__name__
            argstr = []
            for idx, f in enumerate(fields):
                assert hasattr(self, f), \
                    "Attribute {} not found! Default __repr__ only works if attributes match the constructor.".format(f)
                attr = getattr(self, f)
                if idx >= index_field_has_default:
                    if attr is argspec.defaults[idx - index_field_has_default]:
                        continue
                argstr.append("{}={}".format(f, pprint.pformat(attr)))
            return "imgaug.{}({})".format(classname, ', '.join(argstr))
        except AssertionError as e:
            log_once(e.args[0], 'warn')
            return super(Augmentor, self).__repr__()

    __str__ = __repr__


class ImageAugmentor(Augmentor):
    """
    ImageAugmentor should take images of type uint8 in range [0, 255], or
    floating point images in range [0, 1] or [0, 255].
    """
    def augment_coords(self, coords, param):
        """
        Augment the coordinates given the param.

        By default, an augmentor keeps coordinates unchanged.
        If a subclass of :class:`ImageAugmentor` changes coordinates but couldn't implement this method,
        it should ``raise NotImplementedError()``.

        Args:
            coords: Nx2 floating point numpy array where each row is (x, y)
            param: augmentation params returned by :meth:`augment_return_params`

        Returns:
            new coords
        """
        return self._augment_coords(coords, param)

    def _augment_coords(self, coords, param):
        return coords


class AugmentorList(ImageAugmentor):
    """
    Augment an image by a list of augmentors
    """

    def __init__(self, augmentors):
        """
        Args:
            augmentors (list): list of :class:`ImageAugmentor` instance to be applied.
        """
        assert isinstance(augmentors, (list, tuple)), augmentors
        self.augmentors = augmentors
        super(AugmentorList, self).__init__()

    def _get_augment_params(self, img):
        # the next augmentor requires the previous one to finish
        raise RuntimeError("Cannot simply get all parameters of a AugmentorList without running the augmentation!")

    def _augment_return_params(self, img):
        check_dtype(img)
        assert img.ndim in [2, 3], img.ndim

        prms = []
        for a in self.augmentors:
            img, prm = a._augment_return_params(img)
            prms.append(prm)
        return img, prms

    def _augment(self, img, param):
        check_dtype(img)
        assert img.ndim in [2, 3], img.ndim
        for aug, prm in zip(self.augmentors, param):
            img = aug._augment(img, prm)
        return img

    def _augment_coords(self, coords, param):
        for aug, prm in zip(self.augmentors, param):
            coords = aug._augment_coords(coords, prm)
        return coords

    def reset_state(self):
        """ Will reset state of each augmentor """
        super(AugmentorList, self).reset_state()
        for a in self.augmentors:
            a.reset_state()
