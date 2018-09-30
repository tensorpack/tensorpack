# -*- coding: utf-8 -*-
# File: base.py


import inspect
import pprint
from abc import abstractmethod, ABCMeta
import six
from six.moves import zip

from ...utils.utils import get_rng
from ...utils.argtools import log_once
from ..image import check_dtype

__all__ = ['Augmentor', 'ImageAugmentor', 'AugmentorList']


@six.add_metaclass(ABCMeta)
class Augmentor(object):
    """ Base class for an augmentor"""

    def __init__(self):
        self.reset_state()

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
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

    def augment_return_params(self, d):
        """
        Returns:
            augmented data
            augmentation params
        """
        return self._augment_return_params(d)

    def _augment_return_params(self, d):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(d)
        return (self._augment(d, prms), prms)

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
        If a subclass changes coordinates but couldn't implement this method,
        it should ``raise NotImplementedError()``.

        Args:
            coords: Nx2 floating point numpy array where each row is (x, y)
        Returns:
            new coords
        """
        return self._augment_coords(coords, param)

    def _augment_coords(self, coords, param):
        return coords


class AugmentorList(ImageAugmentor):
    """
    Augment by a list of augmentors
    """

    def __init__(self, augmentors):
        """
        Args:
            augmentors (list): list of :class:`ImageAugmentor` instance to be applied.
        """
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
        for a in self.augmentors:
            a.reset_state()
