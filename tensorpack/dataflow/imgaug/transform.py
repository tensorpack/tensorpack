# -*- coding: utf-8 -*-
# File: transform.py

import numpy as np
import pprint
import inspect
import cv2

from ...utils.argtools import log_once

from .base import ImageAugmentor

TransformAugmentorBase = ImageAugmentor
"""
Legacy alias. Please don't use TransformAugmentorBase.
"""

__all__ = []


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


class Transform(object):
    """
    A deterministic image transformation, used to implement
    the (probably random) augmentors.

    This way the deterministic part
    (the actual transformation which may be common between augmentors)
    can be separated from the random part
    (the random policy which is different between augmentors).

    The implementation of each method may choose to modify its input data
    in-place for efficient transformation.
    """
    def __init__(self):
        pass

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
                    setattr(self, k, v)

    def apply_image(self, img):
        raise NotImplementedError(self.__class__)

    def apply_coords(self, coords):
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        """
        Produce something like:
        "imgaug.MyTransform(field1={self.field1}, field2={self.field2})"
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
                    "Attribute {} in {} not found! Default __repr__ only works if " \
                    "attributes match the constructor.".format(f, classname)
                attr = getattr(self, f)
                if idx >= index_field_has_default:
                    if attr is argspec.defaults[idx - index_field_has_default]:
                        continue
                argstr.append("{}={}".format(f, pprint.pformat(attr)))
            return "imgaug.{}({})".format(classname, ', '.join(argstr))
        except AssertionError as e:
            log_once(e.args[0], 'warn')
            return super(Transform, self).__repr__()

    __str__ = __repr__


class TransformList(Transform):
    def __init__(self, tfms):
        for t in tfms:
            assert isinstance(t, Transform), t
        self.tfms = tfms

    def _apply(self, x, meth):
        for t in self.tfms:
            x = getattr(t, meth)(x)
        return x

    def __getattr__(self, name):
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        raise AttributeError("TransformList object has no attribute {}".format(name))

    def apply_image(self, img):
        return self._apply(img, 'apply_image')

    def apply_coords(self, coords):
        return self._apply(coords, 'apply_coords')


class NoOpTransform(Transform):
    def __getattr__(self, name):
        if name.startswith("apply_"):
            return lambda x: x
        raise AttributeError("NoOpTransform object has no attribute {}".format(name))

    def apply_image(self, img):
        return img

    def apply_coords(self, coords):
        return coords


class ResizeTransform(Transform):
    def __init__(self, h, w, new_h, new_w, interp):
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
    def __init__(self, h, w, horiz=True):
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
    def apply_image(self, img):
        ret = cv2.transpose(img)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        return coords[:, ::-1]


class PhotometricTransform(NoOpTransform):
    def __init__(self, func, name=None):
        self._func = func
        self._name = name

    def apply_image(self, img):
        return self._func(img)

    def __repr__(self):
        return "imgaug.PhotometricTransform({})".format(self._name if self._name else "")


class TransformFactory(Transform):
    def __init__(self, name=None, **kwargs):
        """
        Args:
            func (img -> img):
        """
        for k, v in kwargs.items():
            if k.startswith('apply_'):
                setattr(self, k, v)
        self._name = name

    def __str__(self):
        return "imgaug.TransformFactory({})".format(self._name if self._name else "")


class LazyTransform(Transform):
    def __init__(self, get_transform):
        self.get_transform = get_transform
        self._transform = None

    def apply_image(self, img):
        if not self._transform:
            self._transform = self.get_transform(img)
        return self._transform.apply_image(img)

    def _apply(self, x, meth):
        assert self._transform is not None, \
            "LazyTransform.{} can only be called after the transform has been initialized by an image!"
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

    for k in range(1):
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
