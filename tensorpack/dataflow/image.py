# -*- coding: UTF-8 -*-
# File: image.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import cv2
from .base import RNGDataFlow
from .common import MapDataComponent, MapData
from .imgaug import AugmentorList
from ..utils import logger

__all__ = ['ImageFromFile', 'AugmentImageComponent', 'AugmentImageComponents']


class ImageFromFile(RNGDataFlow):
    """ Produce images read from a list of files. """
    def __init__(self, files, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
            resize (tuple): (h, w). If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        self.resize = resize
        self.shuffle = shuffle

    def size(self):
        return len(self.files)

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.files)
        for f in self.files:
            im = cv2.imread(f, self.imread_mode)
            if self.channel == 3:
                im = im[:, :, ::-1]
            if self.resize is not None:
                im = cv2.resize(im, self.resize[::-1])
            if self.channel == 1:
                im = im[:, :, np.newaxis]
            yield [im]


class AugmentImageComponent(MapDataComponent):
    """
    Apply image augmentors on 1 component.
    """
    def __init__(self, ds, augmentors, index=0):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` to be applied in order.
            index (int): the index of the image component to be augmented.
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)

        self._nr_error = 0

        def func(x):
            try:
                ret = self.augs.augment(x)
            except KeyboardInterrupt:
                raise
            except Exception:
                self._nr_error += 1
                if self._nr_error % 1000 == 0 or self._nr_error < 10:
                    logger.exception("Got {} augmentation errors.".format(self._nr_error))
                return None
            return ret

        super(AugmentImageComponent, self).__init__(
            ds, func, index)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()


class AugmentImageComponents(MapData):
    """
    Apply image augmentors on several components, with shared augmentation parameters.
    """

    def __init__(self, ds, augmentors, index=(0, 1)):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` instance to be applied in order.
            index: tuple of indices of components.
        """
        self.augs = AugmentorList(augmentors)
        self.ds = ds
        self._nr_error = 0

        def func(dp):
            try:
                im = dp[index[0]]
                im, prms = self.augs._augment_return_params(im)
                dp[index[0]] = im
                for idx in index[1:]:
                    dp[idx] = self.augs._augment(dp[idx], prms)
                return dp
            except KeyboardInterrupt:
                raise
            except Exception:
                self._nr_error += 1
                if self._nr_error % 1000 == 0 or self._nr_error < 10:
                    logger.exception("Got {} augmentation errors.".format(self._nr_error))
                return None

        super(AugmentImageComponents, self).__init__(ds, func)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()
