# -*- coding: utf-8 -*-
# File: crop.py

import numpy as np
import cv2

from ...utils.argtools import shape2d
from .base import ImageAugmentor
from .transform import CropTransform, TransformAugmentorBase
from .misc import ResizeShortestEdge

__all__ = ['RandomCrop', 'CenterCrop', 'RandomCropRandomShape', 'GoogleNetRandomCropAndResize']


class RandomCrop(TransformAugmentorBase):
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
        return CropTransform(h0, w0, self.crop_shape[0], self.crop_shape[1])


class CenterCrop(TransformAugmentorBase):
    """ Crop the image at the center"""

    def __init__(self, crop_shape):
        """
        Args:
            crop_shape: (h, w) tuple or a int
        """
        crop_shape = shape2d(crop_shape)
        self._init(locals())

    def _get_augment_params(self, img):
        orig_shape = img.shape
        assert orig_shape[0] >= self.crop_shape[0] \
            and orig_shape[1] >= self.crop_shape[1], orig_shape
        h0 = int((orig_shape[0] - self.crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - self.crop_shape[1]) * 0.5)
        return CropTransform(h0, w0, self.crop_shape[0], self.crop_shape[1])


class RandomCropRandomShape(TransformAugmentorBase):
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
        return CropTransform(y0, x0, h, w)


class GoogleNetRandomCropAndResize(ImageAugmentor):
    """
    The random crop and resize augmentation proposed in
    Sec. 6 of `Going Deeper with Convolutions` by Google.
    This implementation follows the details in `fb.resnet.torch`.

    It attempts to crop a random rectangle with 8%~100% area of the original image,
    and keep the aspect ratio between 3/4 to 4/3. Then it resize this crop to the target shape.
    If such crop cannot be found in 10 iterations, it will do a ResizeShortestEdge + CenterCrop.
    """
    def __init__(self, crop_area_fraction=(0.08, 1.),
                 aspect_ratio_range=(0.75, 1.333),
                 target_shape=224, interp=cv2.INTER_LINEAR):
        """
        Args:
            crop_area_fraction (tuple(float)): Defaults to crop 8%-100% area.
            aspect_ratio_range (tuple(float)): Defaults to make aspect ratio in 3/4-4/3.
            target_shape (int): Defaults to 224, the standard ImageNet image shape.
        """
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(*self.crop_area_fraction) * area
            aspectR = self.rng.uniform(*self.aspect_ratio_range)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=self.interp)
                return out
        out = ResizeShortestEdge(self.target_shape, interp=self.interp).augment(img)
        out = CenterCrop(self.target_shape).augment(out)
        return out

    def _augment_coords(self, coords, param):
        raise NotImplementedError()
