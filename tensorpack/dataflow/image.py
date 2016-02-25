#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: image.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import cv2
import copy
from .base import DataFlow, ProxyDataFlow
from .common import MapDataComponent
from .imgaug import AugmentorList, Image

__all__ = ['ImageFromFile', 'AugmentImageComponent']

class ImageFromFile(DataFlow):
    """ generate rgb images from files """
    def __init__(self, files, channel=3, resize=None):
        """ files: list of file path
            channel: 1 or 3 channel
            resize: a (h, w) tuple. If given, will force a resize
        """
        assert len(files)
        self.files = files
        self.channel = int(channel)
        self.resize = resize

    def size(self):
        return len(self.files)

    def get_data(self):
        for f in self.files:
            im = cv2.imread(
                f, cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR)
            if self.channel == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                im = cv2.resize(im, self.resize[::-1])
            yield [im]


class AugmentImageComponent(MapDataComponent):
    """
    Augment image in each data point
    Args:
        ds: a DataFlow dataset instance
        augmentors: a list of ImageAugmentor instance
        index: the index of image in each data point. default to be 0
    """
    def __init__(self, ds, augmentors, index=0):
        self.augs = AugmentorList(augmentors)
        super(AugmentImageComponent, self).__init__(
            ds, lambda x: self.augs.augment(Image(x)).arr, index)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()
