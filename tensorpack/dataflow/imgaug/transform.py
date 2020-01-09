# -*- coding: utf-8 -*-
# File: transform.py

import numpy as np
import cv2

from ...utils.argtools import log_once

from .base import ImageAugmentor, _default_repr

TransformAugmentorBase = ImageAugmentor
"""
Legacy alias. Please don't use.
"""
# This legacy augmentor requires us to import base from here, causing circular dependency.
# Should remove this in the future.

__all__ = ["Transform", "ResizeTransform", "CropTransform", "FlipTransform",
           "TransformList", "TransformFactory"]


# class WrappedImgFunc(object):
#     def __init__(self, func, need_float=False, cast_back=True, fix_ndim=True):
#         self.func = func
#         self.need_float = need_float
#         self.cast_back = cast_back

#     def __call__(self, img):
#         old_dtype = img.dtype
#         old_ndim = img.ndim

#         if self.need_float:
#             img = img.astype("float32")

#         img = self.func(img)

#         if self.cast_back and old_dtype == np.uint8 and img.dtype != np.uint8:
#             img = np.clip(img, 0, 255.)

#         if self.cast_back:
#             img = img.astype(old_dtype)

#         if self.fix_ndim and old_ndim == 3 and img.ndim == 2:
#             img = img[:, :, np.newaxis]
#         return img


class BaseTransform(object):
    """
    Base class for all transforms, for type-check only.

    Users should never interact with this class.
    """
    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
                    setattr(self, k, v)


class Transform(BaseTransform):
    """
    A deterministic image transformation, used to implement
    the (probably random) augmentors.

    This class is also the place to provide a default implementation to any
    :meth:`apply_xxx` method.
    The current default is to raise NotImplementedError in any such methods.

    All subclasses should implement `apply_image`.
    The image should be of type uint8 in range [0, 255], or
    floating point images in range [0, 1] or [0, 255]

    Some subclasses may implement `apply_coords`, when applicable.
    It should take and return a numpy array of Nx2, where each row is the (x, y) coordinate.

    The implementation of each method may choose to modify its input data
    in-place for efficient transformation.
    """

    def __init__(self):
        # provide an empty __init__, so that __repr__ will work nicely
        pass

    def __getattr__(self, name):
        if name.startswith("apply_"):

            def f(x):
                raise NotImplementedError("{} does not implement method {}".format(self.__class__.__name__, name))

            return f
        raise AttributeError("Transform object has no attribute {}".format(name))

    def __repr__(self):
        try:
            return _default_repr(self)
        except AssertionError as e:
            log_once(e.args[0], 'warn')
            return super(Transform, self).__repr__()

    __str__ = __repr__


class ResizeTransform(Transform):
    """
    Resize the image.
    """
    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int):
            new_h, new_w (int):
            interp (int): cv2 interpolation method
        """
        super(ResizeTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        assert img.shape[:2] == (self.h, self.w)
        ret = cv2.resize(
            img, (self.new_w, self.new_h),
            interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords


class CropTransform(Transform):
    """
    Crop a subimage from an image.
    """
    def __init__(self, y0, x0, h, w):
        super(CropTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]

    def apply_coords(self, coords):
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords


class WarpAffineTransform(Transform):
    def __init__(self, mat, dsize, interp=cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_CONSTANT, borderValue=0):
        super(WarpAffineTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        ret = cv2.warpAffine(img, self.mat, self.dsize,
                             flags=self.interp,
                             borderMode=self.borderMode,
                             borderValue=self.borderValue)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1), dtype='f4')), axis=1)
        coords = np.dot(coords, self.mat.T)
        return coords


class FlipTransform(Transform):
    """
    Flip the image.
    """
    def __init__(self, h, w, horiz=True):
        """
        Args:
            h, w (int):
            horiz (bool): whether to flip horizontally or vertically.
        """
        self._init(locals())

    def apply_image(self, img):
        if self.horiz:
            return img[:, ::-1]
        else:
            return img[::-1]

    def apply_coords(self, coords):
        if self.horiz:
            coords[:, 0] = self.w - coords[:, 0]
        else:
            coords[:, 1] = self.h - coords[:, 1]
        return coords


class TransposeTransform(Transform):
    """
    Transpose the image.
    """
    def apply_image(self, img):
        ret = cv2.transpose(img)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        return coords[:, ::-1]


class NoOpTransform(Transform):
    """
    A Transform that does nothing.
    """
    def __getattr__(self, name):
        if name.startswith("apply_"):
            return lambda x: x
        raise AttributeError("NoOpTransform object has no attribute {}".format(name))


class PhotometricTransform(Transform):
    """
    A transform which only has `apply_image` but does nothing in `apply_coords`.
    """
    def __init__(self, func, name=None):
        """
        Args:
            func (img -> img): a function to be used for :meth:`apply_image`
            name (str, optional): the name of this transform
        """
        self._func = func
        self._name = name

    def apply_image(self, img):
        return self._func(img)

    def apply_coords(self, coords):
        return coords

    def __repr__(self):
        return "imgaug.PhotometricTransform({})".format(self._name if self._name else "")

    __str__ = __repr__


class TransformFactory(Transform):
    """
    Create a :class:`Transform` from user-provided functions.
    """
    def __init__(self, name=None, **kwargs):
        """
        Args:
            name (str, optional): the name of this transform
            **kwargs: mapping from `'apply_xxx'` to implementation of such functions.
        """
        for k, v in kwargs.items():
            if k.startswith('apply_'):
                setattr(self, k, v)
            else:
                raise KeyError("Unknown argument '{}' in TransformFactory!".format(k))
        self._name = name

    def __str__(self):
        return "imgaug.TransformFactory({})".format(self._name if self._name else "")

    __repr__ = __str__


"""
Some meta-transforms:
they do not perform actual transformation, but delegate to another Transform.
"""


class TransformList(BaseTransform):
    """
    Apply a list of transforms sequentially.
    """
    def __init__(self, tfms):
        """
        Args:
            tfms (list[Transform]):
        """
        for t in tfms:
            assert isinstance(t, BaseTransform), t
        self.tfms = tfms

    def _apply(self, x, meth):
        for t in self.tfms:
            x = getattr(t, meth)(x)
        return x

    def __getattr__(self, name):
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        raise AttributeError("TransformList object has no attribute {}".format(name))

    def __str__(self):
        repr_each_tfm = ",\n".join(["  " + repr(x) for x in self.tfms])
        return "imgaug.TransformList([\n{}])".format(repr_each_tfm)

    def __add__(self, other):
        other = other.tfms if isinstance(other, TransformList) else [other]
        return TransformList(self.tfms + other)

    def __iadd__(self, other):
        other = other.tfms if isinstance(other, TransformList) else [other]
        self.tfms.extend(other)
        return self

    def __radd__(self, other):
        other = other.tfms if isinstance(other, TransformList) else [other]
        return TransformList(other + self.tfms)

    __repr__ = __str__


class LazyTransform(BaseTransform):
    """
    A transform that's instantiated at the first call to `apply_image`.
    """
    def __init__(self, get_transform):
        """
        Args:
            get_transform (img -> Transform): a function which will be used to instantiate a Transform.
        """
        self.get_transform = get_transform
        self._transform = None

    def apply_image(self, img):
        if not self._transform:
            self._transform = self.get_transform(img)
        return self._transform.apply_image(img)

    def _apply(self, x, meth):
        assert self._transform is not None, \
            "LazyTransform.{} can only be called after the transform has been applied on an image!"
        return getattr(self._transform, meth)(x)

    def __getattr__(self, name):
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        raise AttributeError("TransformList object has no attribute {}".format(name))

    def __repr__(self):
        if self._transform is None:
            return "LazyTransform(get_transform={})".format(str(self.get_transform))
        else:
            return repr(self._transform)

    __str__ = __repr__

    def apply_coords(self, coords):
        return self._apply(coords, "apply_coords")


if __name__ == '__main__':
    shape = (100, 100)
    center = (10, 70)
    mat = cv2.getRotationMatrix2D(center, 20, 1)
    trans = WarpAffineTransform(mat, (130, 130))

    def draw_points(img, pts):
        for p in pts:
            try:
                img[int(p[1]), int(p[0])] = 0
            except IndexError:
                pass

    image = cv2.imread('cat.jpg')
    image = cv2.resize(image, shape)
    orig_image = image.copy()
    coords = np.random.randint(100, size=(20, 2))

    draw_points(orig_image, coords)
    print(coords)

    for _ in range(1):
        coords = trans.apply_coords(coords)
        image = trans.apply_image(image)
    print(coords)
    draw_points(image, coords)

    # viz = cv2.resize(viz, (1200, 600))
    orig_image = cv2.resize(orig_image, (600, 600))
    image = cv2.resize(image, (600, 600))
    viz = np.concatenate((orig_image, image), axis=1)
    cv2.imshow("mat", viz)
    cv2.waitKey()
