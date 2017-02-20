# -*- coding: UTF-8 -*-
# File: crop.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
from ...utils.rect import Rect
from ...utils.argtools import shape2d

from six.moves import range
import numpy as np

__all__ = ['RandomCrop', 'CenterCrop', 'FixedCrop',
           'perturb_BB', 'RandomCropAroundBox', 'RandomCropRandomShape']


class RandomCrop(ImageAugmentor):
    """ Randomly crop the image into a smaller one """

    def __init__(self, crop_shape):
        """
        Args:
            crop_shape: (h, w) tuple or a int
        """
        crop_shape = shape2d(crop_shape)
        super(RandomCrop, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        orig_shape = img.shape
        assert orig_shape[0] >= self.crop_shape[0] \
            and orig_shape[1] >= self.crop_shape[1], orig_shape
        diffh = orig_shape[0] - self.crop_shape[0]
        h0 = 0 if diffh == 0 else self.rng.randint(diffh)
        diffw = orig_shape[1] - self.crop_shape[1]
        w0 = 0 if diffw == 0 else self.rng.randint(diffw)
        return (h0, w0)

    def _augment(self, img, param):
        h0, w0 = param
        return img[h0:h0 + self.crop_shape[0], w0:w0 + self.crop_shape[1]]

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()


class CenterCrop(ImageAugmentor):
    """ Crop the image at the center"""

    def __init__(self, crop_shape):
        """
        Args:
            crop_shape: (h, w) tuple or a int
        """
        crop_shape = shape2d(crop_shape)
        self._init(locals())

    def _augment(self, img, _):
        orig_shape = img.shape
        h0 = int((orig_shape[0] - self.crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - self.crop_shape[1]) * 0.5)
        return img[h0:h0 + self.crop_shape[0], w0:w0 + self.crop_shape[1]]

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()


class FixedCrop(ImageAugmentor):
    """ Crop a rectangle at a given location"""

    def __init__(self, rect):
        """
        Args:
            rect(Rect): min included, max excluded.
        """
        self._init(locals())

    def _augment(self, img, _):
        return img[self.rect.y0: self.rect.y1 + 1,
                   self.rect.x0: self.rect.x0 + 1]

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()


def perturb_BB(image_shape, bb, max_perturb_pixel,
               rng=None, max_aspect_ratio_diff=0.3,
               max_try=100):
    """
    Perturb a bounding box.

    Args:
        image_shape: [h, w]
        bb (Rect): original bounding box
        max_perturb_pixel: perturbation on each coordinate
        max_aspect_ratio_diff: result can't have an aspect ratio too different from the original
        max_try: if cannot find a valid bounding box, return the original
    Returns:
        new bounding box
    """
    orig_ratio = bb.h * 1.0 / bb.w
    if rng is None:
        rng = np.random.RandomState()
    for _ in range(max_try):
        p = rng.randint(-max_perturb_pixel, max_perturb_pixel, [4])
        newbb = bb.copy()
        newbb.x += p[0]
        newbb.y += p[1]
        newx1 = bb.x1 + p[2]
        newy1 = bb.y1 + p[3]
        newbb.w = newx1 - newbb.x
        newbb.h = newy1 - newbb.y
        if not newbb.validate(image_shape):
            continue
        new_ratio = newbb.h * 1.0 / newbb.w
        diff = abs(new_ratio - orig_ratio)
        if diff / orig_ratio > max_aspect_ratio_diff:
            continue
        return newbb
    return bb


class RandomCropAroundBox(ImageAugmentor):
    """
    Crop a box around a bounding box by some random perturbation
    """

    def __init__(self, perturb_ratio, max_aspect_ratio_diff=0.3):
        """
        Args:
            perturb_ratio (float): perturb distance will be in
                ``[0, perturb_ratio * sqrt(w * h)]``
            max_aspect_ratio_diff (float): keep aspect ratio difference within the range
        """
        super(RandomCropAroundBox, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        shape = img.shape[:2]
        box = Rect(0, 0, shape[1] - 1, shape[0] - 1)
        dist = self.perturb_ratio * np.sqrt(shape[0] * shape[1])
        newbox = perturb_BB(shape, box, dist,
                            self.rng, self.max_aspect_ratio_diff)
        return newbox

    def _augment(self, img, newbox):
        return newbox.roi(img)

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()


class RandomCropRandomShape(ImageAugmentor):
    """ Random crop with a random shape"""

    def __init__(self, wmin, hmin,
                 wmax=None, hmax=None,
                 max_aspect_ratio=None):
        """
        Randomly crop a box of shape (h, w), sampled from [min, max] (both inclusive).
        If max is None, will use the input image shape.

        Args:
            wmin, hmin, wmax, hmax: range to sample shape.
            max_aspect_ratio (float): the upper bound of ``max(w,h)/min(w,h)``.
        """
        if max_aspect_ratio is None:
            max_aspect_ratio = 9999999
        self._init(locals())

    def _get_augment_params(self, img):
        hmax = self.hmax or img.shape[0]
        wmax = self.wmax or img.shape[1]
        h = self.rng.randint(self.hmin, hmax + 1)
        w = self.rng.randint(self.wmin, wmax + 1)
        diffh = img.shape[0] - h
        diffw = img.shape[1] - w
        assert diffh >= 0 and diffw >= 0
        y0 = 0 if diffh == 0 else self.rng.randint(diffh)
        x0 = 0 if diffw == 0 else self.rng.randint(diffw)
        return (y0, x0, h, w)

    def _augment(self, img, param):
        y0, x0, h, w = param
        return img[y0:y0 + h, x0:x0 + w]


if __name__ == '__main__':
    print(perturb_BB([100, 100], Rect(3, 3, 50, 50), 50))
