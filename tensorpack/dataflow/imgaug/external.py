#!/usr/bin/env python

import numpy as np

from .base import ImageAugmentor


__all__ = ['IAAugmentor', 'Albumentations']


class IAAugmentor(ImageAugmentor):
    """
    Wrap an augmentor form the IAA library: https://github.com/aleju/imgaug
    Both images and coordinates are supported.

    Note:
        1. It's NOT RECOMMENDED
           to use coordinates because the IAA library does not handle coordinates accurately.

        2. Only uint8 images are supported by the IAA library.

        3. The IAA library can only produces images of the same shape.
    """

    def __init__(self, augmentor):
        """
        Args:
            augmentor (iaa.Augmenter):
        """
        super(IAAugmentor, self).__init__()
        self._aug = augmentor

    def _get_augment_params(self, img):
        return (self._aug.to_deterministic(), img.shape)

    def _augment(self, img, param):
        aug, _ = param
        return aug.augment_image(img)

    def _augment_coords(self, coords, param):
        import imgaug as IA
        aug, shape = param
        points = [IA.Keypoint(x=x, y=y) for x, y in coords]
        points = IA.KeypointsOnImage(points, shape=shape)
        augmented = aug.augment_keypoints([points])[0].keypoints
        return np.asarray([[p.x, p.y] for p in augmented])


class Albumentations(ImageAugmentor):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations
    Coordinate augmentation is not supported by the library.
    """
    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        super(Albumentations, self).__init__()
        self._aug = augmentor

    def _get_augment_params(self, img):
        return self._aug.get_params()

    def _augment(self, img, param):
        return self._aug.apply(img, **param)

    def _augment_coords(self, coords, param):
        raise NotImplementedError()
