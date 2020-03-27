# -*- coding: utf-8 -*-
# File: image.py


import copy as copy_mod
import numpy as np
from contextlib import contextmanager

from ..utils import logger
from ..utils.argtools import shape2d
from .base import RNGDataFlow
from .common import MapData, MapDataComponent

__all__ = ['ImageFromFile', 'AugmentImageComponent', 'AugmentImageCoordinates', 'AugmentImageComponents']


def check_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentor] Needs an numpy array, but got a {}!".format(type(img))
    assert not isinstance(img.dtype, np.integer) or (img.dtype == np.uint8), \
        "[Augmentor] Got image of type {}, use uint8 or floating points instead!".format(img.dtype)


def validate_coords(coords):
    assert coords.ndim == 2, coords.ndim
    assert coords.shape[1] == 2, coords.shape
    assert np.issubdtype(coords.dtype, np.float), coords.dtype


class ExceptionHandler:
    def __init__(self, catch_exceptions=False):
        self._nr_error = 0
        self.catch_exceptions = catch_exceptions

    @contextmanager
    def catch(self):
        try:
            yield
        except Exception:
            self._nr_error += 1
            if not self.catch_exceptions:
                raise
            else:
                if self._nr_error % 100 == 0 or self._nr_error < 10:
                    logger.exception("Got {} augmentation errors.".format(self._nr_error))


class ImageFromFile(RNGDataFlow):
    """ Produce images read from a list of files as (h, w, c) arrays. """
    def __init__(self, files, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.files)
        for f in self.files:
            im = cv2.imread(f, self.imread_mode)
            assert im is not None, f
            if self.channel == 3:
                im = im[:, :, ::-1]
            if self.resize is not None:
                im = cv2.resize(im, tuple(self.resize[::-1]))
            if self.channel == 1:
                im = im[:, :, np.newaxis]
            yield [im]


class AugmentImageComponent(MapDataComponent):
    """
    Apply image augmentors on 1 image component.
    """

    def __init__(self, ds, augmentors, index=0, copy=True, catch_exceptions=False):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` to be applied in order.
            index (int or str): the index or key of the image component to be augmented in the datapoint.
            copy (bool): Some augmentors modify the input images. When copy is
                True, a copy will be made before any augmentors are applied,
                to keep the original images not modified.
                Turn it off to save time when you know it's OK.
            catch_exceptions (bool): when set to True, will catch
                all exceptions and only warn you when there are too many (>100).
                Can be used to ignore occasion errors in data.
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)
        self._copy = copy

        self._exception_handler = ExceptionHandler(catch_exceptions)
        super(AugmentImageComponent, self).__init__(ds, self._aug_mapper, index)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()

    def _aug_mapper(self, x):
        check_dtype(x)
        with self._exception_handler.catch():
            if self._copy:
                x = copy_mod.deepcopy(x)
            return self.augs.augment(x)


class AugmentImageCoordinates(MapData):
    """
    Apply image augmentors on an image and a list of coordinates.
    Coordinates must be a Nx2 floating point array, each row is (x, y).
    """

    def __init__(self, ds, augmentors, img_index=0, coords_index=1, copy=True, catch_exceptions=False):

        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` to be applied in order.
            img_index (int or str): the index/key of the image component to be augmented.
            coords_index (int or str): the index/key of the coordinate component to be augmented.
            copy, catch_exceptions: same as in :class:`AugmentImageComponent`
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)

        self._img_index = img_index
        self._coords_index = coords_index
        self._copy = copy
        self._exception_handler = ExceptionHandler(catch_exceptions)

        super(AugmentImageCoordinates, self).__init__(ds, self._aug_mapper)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()

    def _aug_mapper(self, dp):
        with self._exception_handler.catch():
            img, coords = dp[self._img_index], dp[self._coords_index]
            check_dtype(img)
            validate_coords(coords)
            if self._copy:
                img, coords = copy_mod.deepcopy((img, coords))
            tfms = self.augs.get_transform(img)
            dp[self._img_index] = tfms.apply_image(img)
            dp[self._coords_index] = tfms.apply_coords(coords)
            return dp


class AugmentImageComponents(MapData):
    """
    Apply image augmentors on several components, with shared augmentation parameters.

    Example:

        .. code-block:: python

            ds = MyDataFlow()   # produce [image(HWC), segmask(HW), keypoint(Nx2)]
            ds = AugmentImageComponents(
                ds, augs,
                index=(0,1), coords_index=(2,))

    """

    def __init__(self, ds, augmentors, index=(0, 1), coords_index=(), copy=True, catch_exceptions=False):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` instance to be applied in order.
            index: tuple of indices of the image components.
            coords_index: tuple of indices of the coordinates components.
            copy, catch_exceptions: same as in :class:`AugmentImageComponent`
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)
        self.ds = ds
        self._exception_handler = ExceptionHandler(catch_exceptions)
        self._copy = copy
        self._index = index
        self._coords_index = coords_index

        super(AugmentImageComponents, self).__init__(ds, self._aug_mapper)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()

    def _aug_mapper(self, dp):
        dp = copy_mod.copy(dp)  # always do a shallow copy, make sure the list is intact
        copy_func = copy_mod.deepcopy if self._copy else lambda x: x  # noqa
        with self._exception_handler.catch():
            major_image = self._index[0]  # image to be used to get params. TODO better design?
            im = copy_func(dp[major_image])
            check_dtype(im)
            tfms = self.augs.get_transform(im)
            dp[major_image] = tfms.apply_image(im)
            for idx in self._index[1:]:
                check_dtype(dp[idx])
                dp[idx] = tfms.apply_image(copy_func(dp[idx]))
            for idx in self._coords_index:
                coords = copy_func(dp[idx])
                validate_coords(coords)
                dp[idx] = tfms.apply_coords(coords)
            return dp


try:
    import cv2
    from .imgaug import AugmentorList
except ImportError:
    from ..utils.develop import create_dummy_class
    ImageFromFile = create_dummy_class('ImageFromFile', 'cv2')  # noqa
    AugmentImageComponent = create_dummy_class('AugmentImageComponent', 'cv2')  # noqa
    AugmentImageCoordinates = create_dummy_class('AugmentImageCoordinates', 'cv2') # noqa
    AugmentImageComponents = create_dummy_class('AugmentImageComponents', 'cv2')  # noqa
