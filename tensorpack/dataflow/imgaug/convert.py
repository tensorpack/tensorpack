# -*- coding: utf-8 -*-
# File: convert.py

import numpy as np
import cv2

from .base import PhotometricAugmentor

__all__ = ['ColorSpace', 'Grayscale', 'ToUint8', 'ToFloat32']


class ColorSpace(PhotometricAugmentor):
    """ Convert into another color space.  """

    def __init__(self, mode, keepdims=True):
        """
        Args:
            mode: OpenCV color space conversion code (e.g., ``cv2.COLOR_BGR2HSV``)
            keepdims (bool): keep the dimension of image unchanged if OpenCV
                changes it.
        """
        super(ColorSpace, self).__init__()
        self._init(locals())

    def _augment(self, img, _):
        transf = cv2.cvtColor(img, self.mode)
        if self.keepdims:
            if len(transf.shape) is not len(img.shape):
                transf = transf[..., None]
        return transf


class Grayscale(ColorSpace):
    """ Convert RGB or BGR image to grayscale. """

    def __init__(self, keepdims=True, rgb=False, keepshape=False):
        """
        Args:
            keepdims (bool): return image of shape [H, W, 1] instead of [H, W]
            rgb (bool): interpret input as RGB instead of the default BGR
            keepshape (bool): whether to duplicate the gray image into 3 channels
                so the result has the same shape as input.
        """
        mode = cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY
        if keepshape:
            assert keepdims, "keepdims must be True when keepshape==True"
        super(Grayscale, self).__init__(mode, keepdims)
        self.keepshape = keepshape
        self.rgb = rgb

    def _augment(self, img, _):
        ret = super()._augment(img, _)
        if self.keepshape:
            return np.concatenate([ret] * 3, axis=2)
        else:
            return ret


class ToUint8(PhotometricAugmentor):
    """ Clip and convert image to uint8. Useful to reduce communication overhead. """
    def _augment(self, img, _):
        return np.clip(img, 0, 255).astype(np.uint8)


class ToFloat32(PhotometricAugmentor):
    """ Convert image to float32, may increase quality of the augmentor. """
    def _augment(self, img, _):
        return img.astype(np.float32)
