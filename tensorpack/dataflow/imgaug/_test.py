#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: _test.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
from . import *

augmentors = AugmentorList([
    Contrast((0.2,1.8)),
    Flip(horiz=True)
])

img = cv2.imread('cat.jpg')
img = Image(img)
augmentors.augment(img)
cv2.imshow(" ", img.arr)
cv2.waitKey()
