# -*- coding: utf-8 -*-
# File: crop.py

import numpy as np
import cv2

from ...utils.argtools import shape2d
from ...utils.develop import log_deprecated
from .base import ImageAugmentor, ImagePlaceholder
from .transform import CropTransform, TransformList, ResizeTransform
from .misc import ResizeShortestEdge

__all__ = ['RandomCrop', 'CenterCrop', 'RandomCropRandomShape', 'GoogleNetRandomCropAndResize']


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

    def get_transform(self, img):
        orig_shape = img.shape
        assert orig_shape[0] >= self.crop_shape[0] \
            and orig_shape[1] >= self.crop_shape[1], orig_shape
        diffh = orig_shape[0] - self.crop_shape[0]
        h0 = 0 if diffh == 0 else self.rng.randint(diffh)
        diffw = orig_shape[1] - self.crop_shape[1]
        w0 = 0 if diffw == 0 else self.rng.randint(diffw)
        return CropTransform(h0, w0, self.crop_shape[0], self.crop_shape[1])


class CenterCrop(ImageAugmentor):
    """ Crop the image at the center"""

    def __init__(self, crop_shape):
        """
        Args:
            crop_shape: (h, w) tuple or a int
        """
        crop_shape = shape2d(crop_shape)
        self._init(locals())

    def get_transform(self, img):
        orig_shape = img.shape
        assert orig_shape[0] >= self.crop_shape[0] \
            and orig_shape[1] >= self.crop_shape[1], orig_shape
        h0 = int((orig_shape[0] - self.crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - self.crop_shape[1]) * 0.5)
        return CropTransform(h0, w0, self.crop_shape[0], self.crop_shape[1])


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
            max_aspect_ratio (float): this argument has no effect and is deprecated.
        """
        super(RandomCropRandomShape, self).__init__()
        if max_aspect_ratio is not None:
            log_deprecated("RandomCropRandomShape(max_aspect_ratio)", "It is never implemented!", "2020-06-06")
        self._init(locals())

    def get_transform(self, img):
        hmax = self.hmax or img.shape[0]
        wmax = self.wmax or img.shape[1]
        h = self.rng.randint(self.hmin, hmax + 1)
        w = self.rng.randint(self.wmin, wmax + 1)
        diffh = img.shape[0] - h
        diffw = img.shape[1] - w
        assert diffh >= 0 and diffw >= 0, str(diffh) + ", " + str(diffw)
        y0 = 0 if diffh == 0 else self.rng.randint(diffh)
        x0 = 0 if diffw == 0 else self.rng.randint(diffw)
        return CropTransform(y0, x0, h, w)


class GoogleNetRandomCropAndResize(ImageAugmentor):
    """
    The random crop and resize augmentation proposed in
    Sec. 6 of "Going Deeper with Convolutions" by Google.
    This implementation follows the details in ``fb.resnet.torch``.

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
        super(GoogleNetRandomCropAndResize, self).__init__()
        self._init(locals())

    def get_transform(self, img):
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
                return TransformList([
                    CropTransform(y1, x1, hh, ww),
                    ResizeTransform(hh, ww, self.target_shape, self.target_shape, interp=self.interp)
                ])
        tfm1 = ResizeShortestEdge(self.target_shape, interp=self.interp).get_transform(img)
        out_shape = (tfm1.new_h, tfm1.new_w)
        tfm2 = CenterCrop(self.target_shape).get_transform(ImagePlaceholder(shape=out_shape))
        return TransformList([tfm1, tfm2])
