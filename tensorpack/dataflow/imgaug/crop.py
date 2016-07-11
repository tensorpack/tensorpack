# -*- coding: UTF-8 -*-
# File: crop.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
from ...utils.rect import Rect

from six.moves import range
import numpy as np

__all__ = ['RandomCrop', 'CenterCrop', 'FixedCrop', 'RandomCropRandomShape']

class RandomCrop(ImageAugmentor):
    """ Randomly crop the image into a smaller one """
    def __init__(self, crop_shape):
        """
        :param crop_shape: a shape like (h, w)
        """
        self._init(locals())

    def _get_augment_params(self, img):
        orig_shape = img.shape
        assert orig_shape[0] >= self.crop_shape[0] \
                and orig_shape[1] >= self.crop_shape[1], orig_shape
        diffh = orig_shape[0] - self.crop_shape[0]
        if diffh == 0:
            h0 = 0
        else:
            h0 = self.rng.randint(diffh)
        diffw = orig_shape[1] - self.crop_shape[1]
        if diffw == 0:
            w0 = 0
        else:
            w0 = self.rng.randint(diffw)
        return (h0, w0)

    def _augment(self, img, param):
        h0, w0 = param
        return img[h0:h0+self.crop_shape[0],w0:w0+self.crop_shape[1]]

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()

class CenterCrop(ImageAugmentor):
    """ Crop the image at the center"""
    def __init__(self, crop_shape):
        """
        :param crop_shape: a shape like (h, w)
        """
        self._init(locals())

    def _augment(self, img, _):
        orig_shape = img.shape
        h0 = int((orig_shape[0] - self.crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - self.crop_shape[1]) * 0.5)
        return img[h0:h0+self.crop_shape[0],w0:w0+self.crop_shape[1]]

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()

class FixedCrop(ImageAugmentor):
    """ Crop a rectangle at a given location"""
    def __init__(self, rect):
        """
        Two arguments defined the range in both axes to crop, min inclued, max excluded.

        :param rect: a `Rect` instance
        """
        self._init(locals())

    def _augment(self, img, _):
        orig_shape = img.shape
        return img[self.rect.y0: self.rect.y1+1,
                   self.rect.x0: self.rect.x0+1]

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()

def perturb_BB(image_shape, bb, max_pertub_pixel,
        rng=None, max_aspect_ratio_diff=0.3,
        max_try=100):
    """
    Perturb a bounding box.
    :param image_shape: [h, w]
    :param bb: a `Rect` instance
    :param max_pertub_pixel: pertubation on each coordinate
    :param max_aspect_ratio_diff: result can't have an aspect ratio too different from the original
    :param max_try: if cannot find a valid bounding box, return the original
    :returns: new bounding box
    """
    orig_ratio = bb.h * 1.0 / bb.w
    if rng is None:
        rng = np.random.RandomState()
    for _ in range(max_try):
        p = rng.randint(-max_pertub_pixel, max_pertub_pixel, [4])
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


class RandomCropRandomShape(ImageAugmentor):
    """
    Crop a box around a bounding box
    """
    def __init__(self, perturb_ratio, max_aspect_ratio_diff=0.3):
        """
        :param perturb_ratio: perturb distance will be in [0, perturb_ratio * sqrt(w * h)]
        :param max_aspect_ratio_diff: keep aspect ratio within the range
        """
        self._init(locals())

    def _get_augment_params(self, img):
        shape = img.shape[:2]
        box = Rect(0, 0, shape[1] - 1, shape[0] - 1)
        dist = self.perturb_ratio * np.sqrt(shape[0]*shape[1])
        newbox = perturb_BB(shape, box, dist,
                self.rng, self.max_aspect_ratio_diff)
        return newbox

    def _augment(self, img, newbox):
        return newbox.roi(img)

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()

if __name__ == '__main__':
    print(perturb_BB([100, 100], Rect(3, 3, 50, 50), 50))
