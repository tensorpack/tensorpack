# -*- coding: UTF-8 -*-
# File: crop.py


from ...utils.argtools import shape2d
from .transform import TransformAugmentorBase, CropTransform


__all__ = ['RandomCrop', 'CenterCrop', 'RandomCropRandomShape']


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
