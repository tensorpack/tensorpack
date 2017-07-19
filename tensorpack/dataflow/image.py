# -*- coding: UTF-8 -*-
# File: image.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import copy as copy_mod
from .base import RNGDataFlow
from .common import MapDataComponent, MapData
from ..utils import logger
from ..utils.argtools import shape2d

__all__ = ['ImageFromFile', 'AugmentImageComponent', 'AugmentImageCoordinates', 'AugmentImageComponents']


class ImageFromFile(RNGDataFlow):
    """ Produce images read from a list of files. """
    def __init__(self, files, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
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
                im = cv2.resize(im, tuple(self.resize[::-1]))
            if self.channel == 1:
                im = im[:, :, np.newaxis]
            yield [im]


class AugmentImageComponent(MapDataComponent):
    """
    Apply image augmentors on 1 component.
    """
    def __init__(self, ds, augmentors, index=0, copy=True):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` to be applied in order.
            index (int): the index of the image component to be augmented.
            copy (bool): Some augmentors modify the input images. When copy is
                True, a copy will be made before any augmentors are applied,
                to keep the original images not modified.
                Turn it off to save time when you know it's OK.
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)

        self._nr_error = 0

        def func(x):
            try:
                if copy:
                    x = copy_mod.deepcopy(x)
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


class AugmentImageCoordinates(MapData):
    """
    Apply image augmentors on an image and set of coordinates.
    """
    def __init__(self, ds, augmentors, img_index=0, coords_index=1, copy=True):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` to be applied in order.
            img_index (int): the index of the image component to be augmented.
            coords_index (int): the index of the coordinate component to be augmented.
            copy (bool): Some augmentors modify the input images. When copy is
                True, a copy will be made before any augmentors are applied,
                to keep the original images not modified.
                Turn it off to save time when you know it's OK.
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)
        self._nr_error = 0

        def func(dp):
            try:
                img, coords = dp[img_index], dp[coords_index]
                if copy:
                    img, coords = copy_mod.deepcopy((img, coords))
                img, prms = self.augs._augment_return_params(img)
                dp[img_index] = img
                coords = self.augs._augment_coords(coords, prms)
                dp[coords_index] = coords
                return dp
            except KeyboardInterrupt:
                raise
            except Exception:
                self._nr_error += 1
                if self._nr_error % 1000 == 0 or self._nr_error < 10:
                    logger.exception("Got {} augmentation errors.".format(self._nr_error))
                return None

        super(AugmentImageCoordinates, self).__init__(ds, func)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()


class AugmentImageComponents(MapData):
    """
    Apply image augmentors on several components, with shared augmentation parameters.
    """

    def __init__(self, ds, augmentors, index=(0, 1), copy=True):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` instance to be applied in order.
            index: tuple of indices of components.
            copy (bool): Some augmentors modify the input images. When copy is
                True, a copy will be made before any augmentors are applied,
                to keep the original images not modified.
                Turn it off to save time when you know it's OK.
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)
        self.ds = ds
        self._nr_error = 0

        def func(dp):
            dp = copy_mod.copy(dp)  # always do a shallow copy, make sure the list is intact
            copy_func = copy_mod.deepcopy if copy else lambda x: x  # noqa
            try:
                im = copy_func(dp[index[0]])
                im, prms = self.augs._augment_return_params(im)
                dp[index[0]] = im
                for idx in index[1:]:
                    dp[idx] = self.augs._augment(copy_func(dp[idx]), prms)
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


try:
    import cv2
    from .imgaug import AugmentorList
except ImportError:
    from ..utils.develop import create_dummy_class
    ImageFromFile = create_dummy_class('ImageFromFile', 'cv2')  # noqa
    AugmentImageComponent = create_dummy_class('AugmentImageComponent', 'cv2')  # noqa
    AugmentImageCoordinates = create_dummy_class('AugmentImageCoordinates', 'cv2') # noqa
    AugmentImageComponents = create_dummy_class('AugmentImageComponents', 'cv2')  # noqa
