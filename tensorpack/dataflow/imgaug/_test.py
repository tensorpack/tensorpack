#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: _test.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import sys
import cv2
from . import AugmentorList, Flip, GaussianDeform, Image


anchors = [(0.2, 0.2), (0.8, 0.8), (0.5, 0.5), (0.2, 0.5)]
augmentors = AugmentorList([
    #Contrast((0.2,1.8)),
    #Flip(horiz=True),
    GaussianDeform(anchors, (360,480), 1, randrange=10)
])

while True:
    img = cv2.imread(sys.argv[1])
    img = Image(img)
    augmentors.augment(img)
    cv2.imshow(" ", img.arr.astype('uint8'))
    cv2.waitKey()
