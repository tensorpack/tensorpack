# -*- coding: UTF-8 -*-
# File: image.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import cv2
import copy
from .base import DataFlow, ProxyDataFlow
from .common import MapDataComponent, MapData
from .imgaug import AugmentorList

__all__ = ['ImageFromFile', 'AugmentImageComponent', 'AugmentImageComponents']

class ImageFromFile(DataFlow):
    def __init__(self, files, channel=3, resize=None):
        """
        Generate rgb images from list of files
        :param files: list of file paths
        :param channel: 1 or 3 channel
        :param resize: a (h, w) tuple. If given, will force a resize
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
    def __init__(self, ds, augmentors, index=0):
        """
        Augment the image component of datapoints
        :param ds: a `DataFlow` instance.
        :param augmentors: a list of `ImageAugmentor` instance to be applied in order.
        :param index: the index (or list of indices) of the image component in the produced datapoints by `ds`. default to be 0
        """
        self.augs = AugmentorList(augmentors)
        super(AugmentImageComponent, self).__init__(
            ds, lambda x: self.augs.augment(x), index)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()


class AugmentImageComponents(MapData):
    def __init__(self, ds, augmentors, index=(0,1)):
        """ Augment a list of images of the same shape, with the same parameters
        :param ds: a `DataFlow` instance.
        :param augmentors: a list of `ImageAugmentor` instance to be applied in order.
        :param index: tuple of indices of the image components
        """
        self.augs = AugmentorList(augmentors)
        self.ds = ds

        def func(dp):
            im = dp[index[0]]
            im, prms = self.augs._augment_return_params(im)
            dp[index[0]] = im
            for idx in index[1:]:
                dp[idx] = self.augs._augment(dp[idx], prms)
            return dp

        super(AugmentImageComponents, self).__init__(ds, func)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()
