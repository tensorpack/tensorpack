# -*- coding: utf-8 -*-
# File: _test.py


import sys
import numpy as np
import cv2
import unittest

from .base import ImageAugmentor, AugmentorList
from .imgproc import Contrast
from .noise import SaltPepperNoise
from .misc import Flip, Resize


def _rand_image(shape=(20, 20)):
    return np.random.rand(*shape).astype("float32")


class LegacyBrightness(ImageAugmentor):
    def __init__(self, delta, clip=True):
        super(LegacyBrightness, self).__init__()
        assert delta > 0
        self._init(locals())

    def _get_augment_params(self, _):
        v = self._rand_range(-self.delta, self.delta)
        return v

    def _augment(self, img, v):
        old_dtype = img.dtype
        img = img.astype('float32')
        img += v
        if self.clip or old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class LegacyFlip(ImageAugmentor):
    def __init__(self, horiz=False, vert=False, prob=0.5):
        super(LegacyFlip, self).__init__()
        if horiz and vert:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        elif horiz:
            self.code = 1
        elif vert:
            self.code = 0
        else:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        return (do, h, w)

    def _augment(self, img, param):
        do, _, _ = param
        if do:
            ret = cv2.flip(img, self.code)
            if img.ndim == 3 and ret.ndim == 2:
                ret = ret[:, :, np.newaxis]
        else:
            ret = img
        return ret

    def _augment_coords(self, coords, param):
        do, h, w = param
        if do:
            if self.code == 0:
                coords[:, 1] = h - coords[:, 1]
            elif self.code == 1:
                coords[:, 0] = w - coords[:, 0]
        return coords


class ImgAugTest(unittest.TestCase):
    def _get_augs(self):
        return AugmentorList([
            Contrast((0.8, 1.2)),
            Flip(horiz=True),
            Resize((30, 30)),
            SaltPepperNoise()
        ])

    def _get_augs_with_legacy(self):
        return AugmentorList([
            LegacyBrightness(0.5),
            LegacyFlip(horiz=True),
            Resize((30, 30)),
            SaltPepperNoise()
        ])

    def test_augmentors(self):
        augmentors = self._get_augs()

        img = _rand_image()
        orig = img.copy()
        tfms = augmentors.get_transform(img)

        # test printing
        print(augmentors)
        print(tfms)

        newimg = tfms.apply_image(img)
        print(tfms)  # lazy ones will instantiate after the first apply

        newimg2 = tfms.apply_image(orig)
        self.assertTrue(np.allclose(newimg, newimg2))
        self.assertEqual(newimg2.shape[0], 30)

        coords = np.asarray([[0, 0], [10, 12]], dtype="float32")
        tfms.apply_coords(coords)

    def test_legacy_usage(self):
        augmentors = self._get_augs()

        img = _rand_image()
        orig = img.copy()
        newimg, tfms = augmentors.augment_return_params(img)
        newimg2 = augmentors.augment_with_params(orig, tfms)
        self.assertTrue(np.allclose(newimg, newimg2))
        self.assertEqual(newimg2.shape[0], 30)

        coords = np.asarray([[0, 0], [10, 12]], dtype="float32")
        augmentors.augment_coords(coords, tfms)

    def test_legacy_augs_new_usage(self):
        augmentors = self._get_augs_with_legacy()

        img = _rand_image()
        orig = img.copy()
        tfms = augmentors.get_transform(img)
        newimg = tfms.apply_image(img)
        newimg2 = tfms.apply_image(orig)
        self.assertTrue(np.allclose(newimg, newimg2))
        self.assertEqual(newimg2.shape[0], 30)

        coords = np.asarray([[0, 0], [10, 12]], dtype="float32")
        tfms.apply_coords(coords)

    def test_legacy_augs_legacy_usage(self):
        augmentors = self._get_augs_with_legacy()

        img = _rand_image()
        orig = img.copy()
        newimg, tfms = augmentors.augment_return_params(img)
        newimg2 = augmentors.augment_with_params(orig, tfms)
        self.assertTrue(np.allclose(newimg, newimg2))
        self.assertEqual(newimg2.shape[0], 30)

        coords = np.asarray([[0, 0], [10, 12]], dtype="float32")
        augmentors.augment_coords(coords, tfms)


if __name__ == '__main__':
    anchors = [(0.2, 0.2), (0.7, 0.2), (0.8, 0.8), (0.5, 0.5), (0.2, 0.5)]
    augmentors = AugmentorList([
        Contrast((0.8, 1.2)),
        Flip(horiz=True),
        # RandomCropRandomShape(0.3),
        SaltPepperNoise()
    ])

    img = cv2.imread(sys.argv[1])
    newimg, prms = augmentors._augment_return_params(img)
    cv2.imshow(" ", newimg.astype('uint8'))
    cv2.waitKey()

    newimg = augmentors._augment(img, prms)
    cv2.imshow(" ", newimg.astype('uint8'))
    cv2.waitKey()
