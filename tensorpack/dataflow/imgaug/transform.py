# -*- coding: utf-8 -*-
# File: transform.py

from abc import abstractmethod, ABCMeta
import six
import cv2
import numpy as np

from .base import ImageAugmentor

__all__ = []


class TransformAugmentorBase(ImageAugmentor):
    """
    Base class of augmentors which use :class:`ImageTransform`
    for the actual implementation of the transformations.

    It assumes that :meth:`_get_augment_params` should
    return a :class:`ImageTransform` instance, and it will use
    this instance to augment both image and coordinates.
    """
    def _augment(self, img, t):
        return t.apply_image(img)

    def _augment_coords(self, coords, t):
        return t.apply_coords(coords)


@six.add_metaclass(ABCMeta)
class ImageTransform(object):
    """
    A deterministic image transformation, used to implement
    the (probably random) augmentors.

    This way the deterministic part
    (the actual transformation which may be common between augmentors)
    can be separated from the random part
    (the random policy which is different between augmentors).
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self':
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img):
        pass

    @abstractmethod
    def apply_coords(self, coords):
        pass


class ResizeTransform(ImageTransform):
    def __init__(self, h, w, newh, neww, interp):
        super(ResizeTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        assert img.shape[:2] == (self.h, self.w)
        ret = cv2.resize(
            img, (self.neww, self.newh),
            interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.neww * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.newh * 1.0 / self.h)
        return coords


class CropTransform(ImageTransform):
    def __init__(self, h0, w0, h, w):
        super(CropTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        return img[self.h0:self.h0 + self.h, self.w0:self.w0 + self.w]

    def apply_coords(self, coords):
        coords[:, 0] -= self.w0
        coords[:, 1] -= self.h0
        return coords


class WarpAffineTransform(ImageTransform):
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
