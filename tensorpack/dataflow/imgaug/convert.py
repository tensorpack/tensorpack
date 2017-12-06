#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: convert.py

from .base import ImageAugmentor
from .meta import MapImage
import numpy as np
import cv2

__all__ = ['ColorSpace', 'Grayscale', 'ToUint8', 'ToFloat32']


class ColorSpace(ImageAugmentor):
    """ Convert into another colorspace.  """

    def __init__(self, mode, keepdims=True):
        """
        Args:
            mode: opencv colorspace conversion code (e.g., `cv2.COLOR_BGR2HSV`)
            keepdims (bool): keep the dimension of image unchanged if opencv
                changes it.
        """
        self._init(locals())

    def _augment(self, img, _):
        transf = cv2.cvtColor(img, self.mode)
        if self.keepdims:
            if len(transf.shape) is not len(img.shape):
                transf = transf[..., None]
        return transf


class Grayscale(ColorSpace):
    """ Convert image to grayscale.  """

    def __init__(self, keepdims=True, rgb=False):
        """
        Args:
            keepdims (bool): return image of shape [H, W, 1] instead of [H, W]
            rgb (bool): interpret input as RGB instead of the default BGR
        """
        mode = cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY
        super(Grayscale, self).__init__(mode, keepdims)


class ToUint8(MapImage):
    """ Convert image to uint8. Useful to reduce communication overhead. """
    def __init__(self):
        super(ToUint8, self).__init__(lambda x: np.clip(x, 0, 255).astype(np.uint8), lambda x: x)


class ToFloat32(MapImage):
    """ Convert image to float32, may increase quality of the augmentor. """
    def __init__(self):
        super(ToFloat32, self).__init__(lambda x: x.astype(np.float32), lambda x: x)
