#!/usr/bin/env python

import numpy as np

from .base import ImageAugmentor
from .transform import Transform

__all__ = ['IAAugmentor', 'Albumentations']


class IAATransform(Transform):
    def __init__(self, aug, img_shape):
        self._init(locals())

    def apply_image(self, img):
        return self.aug.augment_image(img)

    def apply_coords(self, coords):
        import imgaug as IA
        points = [IA.Keypoint(x=x, y=y) for x, y in coords]
        points = IA.KeypointsOnImage(points, shape=self.img_shape)
        augmented = self.aug.augment_keypoints([points])[0].keypoints
        return np.asarray([[p.x, p.y] for p in augmented])


class IAAugmentor(ImageAugmentor):
    """
    Wrap an augmentor form the IAA library: https://github.com/aleju/imgaug.
    Both images and coordinates are supported.

    Note:
        1. It's NOT RECOMMENDED
           to use coordinates because the IAA library does not handle coordinates accurately.

        2. Only uint8 images are supported by the IAA library.

        3. The IAA library can only produces images of the same shape.

    Example:

    .. code-block:: python

        from imgaug import augmenters as iaa  # this is the aleju/imgaug library
        from tensorpack import imgaug  # this is not the aleju/imgaug library
        # or from dataflow import imgaug  # if you're using the standalone version of dataflow
        myaug = imgaug.IAAugmentor(
            iaa.Sequential([
                iaa.Sharpen(alpha=(0, 1), lightness=(0.75, 1.5)),
                iaa.Fliplr(0.5),
                iaa.Crop(px=(0, 100)),
            ])
    """

    def __init__(self, augmentor):
        """
        Args:
            augmentor (iaa.Augmenter):
        """
        super(IAAugmentor, self).__init__()
        self._aug = augmentor

    def get_transform(self, img):
        return IAATransform(self._aug.to_deterministic(), img.shape)


class AlbumentationsTransform(Transform):
    def __init__(self, aug, param):
        self._init(locals())

    def apply_image(self, img):
        return self.aug.apply(img, **self.param)


class Albumentations(ImageAugmentor):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Coordinate augmentation is not supported by the library.

    Example:

    .. code-block:: python

        from tensorpack import imgaug
        # or from dataflow import imgaug  # if you're using the standalone version of dataflow
        import albumentations as AB
        myaug = imgaug.Albumentations(AB.RandomRotate90(p=1))
    """
    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        super(Albumentations, self).__init__()
        self._aug = augmentor

    def get_transform(self, img):
        return AlbumentationsTransform(self._aug, self._aug.get_params())
