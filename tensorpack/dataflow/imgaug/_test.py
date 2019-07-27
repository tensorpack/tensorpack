# -*- coding: utf-8 -*-
# File: _test.py


import sys
import numpy as np
import cv2
import unittest

from . import AugmentorList
from .crop import *
from .deform import *
from .imgproc import *
from .noise import *
from .misc import *


def _rand_image(shape=(20, 20)):
    return np.random.rand(*shape).astype("float32")


class ImgAugTest(unittest.TestCase):
    def test_augmentors(self):
        augmentors = AugmentorList([
            Contrast((0.8, 1.2)),
            Flip(horiz=True),
            Resize((30, 30)),
            SaltPepperNoise()
        ])

        img = _rand_image()
        orig = img.copy()
        newimg, tfms = augmentors.apply(img)
        print(augmentors, tfms)  # TODO better print
        newimg2 = tfms.apply_image(orig)
        self.assertTrue(np.allclose(newimg, newimg2))
        self.assertEqual(newimg2.shape[0], 30)

        coords = np.asarray([[0, 0], [10, 12]], dtype="float32")
        new_coords = tfms.apply_coords(coords)


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
