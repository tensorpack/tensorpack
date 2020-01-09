# -*- coding: utf-8 -*-
# File: base.py

import os
import inspect
import pprint
from collections import namedtuple
import weakref

from ...utils.argtools import log_once
from ...utils.utils import get_rng
from ...utils.develop import deprecated
from ..image import check_dtype

# Cannot import here if we want to keep backward compatibility.
# Because this causes circular dependency
# from .transform import TransformList, PhotometricTransform, TransformFactory

__all__ = ['Augmentor', 'ImageAugmentor', 'AugmentorList', 'PhotometricAugmentor']


def _reset_augmentor_after_fork(aug_ref):
    aug = aug_ref()
    if aug:
        aug.reset_state()


def _default_repr(self):
    """
    Produce something like:
    "imgaug.MyAugmentor(field1={self.field1}, field2={self.field2})"

    It assumes that the instance `self` contains attributes that match its constructor.
    """
    classname = type(self).__name__
    argspec = inspect.getfullargspec(self.__init__)
    assert argspec.varargs is None, "The default __repr__ in {} doesn't work for varargs!".format(classname)
    assert argspec.varkw is None, "The default __repr__ in {} doesn't work for kwargs!".format(classname)
    defaults = {}

    fields = argspec.args[1:]
    defaults_pos = argspec.defaults
    if defaults_pos is not None:
        for f, d in zip(fields[::-1], defaults_pos[::-1]):
            defaults[f] = d

    for k in argspec.kwonlyargs:
        fields.append(k)
        if k in argspec.kwonlydefaults:
            defaults[k] = argspec.kwonlydefaults[k]

    argstr = []
    for f in fields:
        assert hasattr(self, f), \
            "Attribute {} in {} not found! Default __repr__ only works if " \
            "the instance has attributes that match the constructor.".format(f, classname)
        attr = getattr(self, f)
        if f in defaults and attr is defaults[f]:
            continue
        argstr.append("{}={}".format(f, pprint.pformat(attr)))
    return "imgaug.{}({})".format(classname, ', '.join(argstr))


ImagePlaceholder = namedtuple("ImagePlaceholder", ["shape"])


class ImageAugmentor(object):
    """
    Base class for an augmentor

    ImageAugmentor should take images of type uint8 in range [0, 255], or
    floating point images in range [0, 1] or [0, 255].

    Attributes:
        rng: a numpy :class:`RandomState`
    """

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

        If you use a built-in augmentation dataflow (:class:`AugmentImageComponent`, etc),
        this method will be called in the dataflow's own `reset_state` method.

        If you use Python≥3.7 on Unix, this method will be automatically called after fork,
        and you do not need to bother calling it.
        """
        self.rng = get_rng(self)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Generate uniform float random number between low and high using `self.rng`.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)

    def __str__(self):
        try:
            return _default_repr(self)
        except AssertionError as e:
            log_once(e.args[0], 'warn')
            return super(Augmentor, self).__repr__()

    __repr__ = __str__

    def get_transform(self, img):
        """
        Instantiate a :class:`Transform` object to be used given the input image.
        Subclasses should implement this method.

        The :class:`ImageAugmentor` often has random policies which generate deterministic transform.
        Any of those random policies should happen inside this method and instantiate
        an actual deterministic transform to be performed.
        The returned :class:`Transform` object should perform deterministic transforms
        through its :meth:`apply_*` method.

        In this way, the returned :class:`Transform` object can be used to transform not only the
        input image, but other images or coordinates associated with the image.

        Args:
            img (ndarray): see notes of this class on the requirements.

        Returns:
            Transform
        """
        # This should be an abstract method
        # But we provide an implementation that uses the old interface,
        # for backward compatibility
        log_once("The old augmentor interface was deprecated. "
                 "Please implement {} with `get_transform` instead!".format(self.__class__.__name__),
                 "warning")

        def legacy_augment_coords(self, coords, p):
            try:
                return self._augment_coords(coords, p)
            except AttributeError:
                pass
            try:
                return self.augment_coords(coords, p)
            except AttributeError:
                pass
            return coords  # this is the old default

        p = None  # the default return value for this method
        try:
            p = self._get_augment_params(img)
        except AttributeError:
            pass
        try:
            p = self.get_augment_params(img)
        except AttributeError:
            pass

        from .transform import BaseTransform, TransformFactory
        if isinstance(p, BaseTransform):  # some old augs return Transform already
            return p

        return TransformFactory(name="LegacyConversion -- " + str(self),
                                apply_image=lambda img: self._augment(img, p),
                                apply_coords=lambda coords: legacy_augment_coords(self, coords, p))

    def augment(self, img):
        """
        Create a transform, and apply it to augment the input image.

        This can save you one line of code, when you only care the augmentation of "one image".
        It will not return the :class:`Transform` object to you
        so you won't be able to apply the same transformation on
        other data associated with the image.

        Args:
            img (ndarray): see notes of this class on the requirements.

        Returns:
            img: augmented image.
        """
        check_dtype(img)
        t = self.get_transform(img)
        return t.apply_image(img)

    # ###########################
    # Legacy interfaces:
    # ###########################
    @deprecated("Please use `get_transform` instead!", "2020-06-06", max_num_warnings=3)
    def augment_return_params(self, d):
        t = self.get_transform(d)
        return t.apply_image(d), t

    @deprecated("Please use `transform.apply_image` instead!", "2020-06-06", max_num_warnings=3)
    def augment_with_params(self, d, param):
        return param.apply_image(d)

    @deprecated("Please use `transform.apply_coords` instead!", "2020-06-06", max_num_warnings=3)
    def augment_coords(self, coords, param):
        return param.apply_coords(coords)


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

    def reset_state(self):
        """ Will reset state of each augmentor """
        super(AugmentorList, self).reset_state()
        for a in self.augmentors:
            a.reset_state()

    def get_transform(self, img):
        check_dtype(img)
        assert img.ndim in [2, 3], img.ndim

        from .transform import LazyTransform, TransformList
        # The next augmentor requires the previous one to finish.
        # So we have to use LazyTransform
        tfms = []
        for idx, a in enumerate(self.augmentors):
            if idx == 0:
                t = a.get_transform(img)
            else:
                t = LazyTransform(a.get_transform)

            if isinstance(t, TransformList):
                tfms.extend(t.tfms)
            else:
                tfms.append(t)
        return TransformList(tfms)

    def __str__(self):
        repr_each_aug = ",\n".join(["  " + repr(x) for x in self.augmentors])
        return "imgaug.AugmentorList([\n{}])".format(repr_each_aug)

    __repr__ = __str__


Augmentor = ImageAugmentor
"""
Legacy name. Augmentor and ImageAugmentor are now the same thing.
"""


class PhotometricAugmentor(ImageAugmentor):
    """
    A base class for ImageAugmentor which only affects pixels.

    Subclass should implement `_get_params(img)` and `_impl(img, params)`.
    """
    def get_transform(self, img):
        p = self._get_augment_params(img)
        from .transform import PhotometricTransform
        return PhotometricTransform(func=lambda img: self._augment(img, p),
                                    name="from " + str(self))

    def _get_augment_params(self, _):
        return None
