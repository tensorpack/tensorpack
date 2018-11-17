#!/usr/bin/env python

import numpy as np

from .base import ImageAugmentor


__all__ = ['IAAugmentor']


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
        aug, shape = param
        points = [IA.Keypoint(x=x, y=y) for x, y in coords]
        points = IA.KeypointsOnImage(points, shape=shape)
        augmented = aug.augment_keypoints([points])[0].keypoints
        return np.asarray([[p.x, p.y] for p in augmented])


from ...utils.develop import create_dummy_class   # noqa
try:
    import imgaug as IA
except ImportError:
    IAAugmentor = create_dummy_class('IAAugmentor', 'imgaug')  # noqa
