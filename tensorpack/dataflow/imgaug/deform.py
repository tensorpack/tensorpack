# -*- coding: UTF-8 -*-
# File: deform.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
from ...utils import logger
import numpy as np

__all__ = ['GaussianDeform']


class GaussianMap(object):
    """ Generate gaussian weighted deformation map"""
    # TODO really needs speedup

    def __init__(self, image_shape, sigma=0.5):
        assert len(image_shape) == 2
        self.shape = image_shape
        self.sigma = sigma

    def get_gaussian_weight(self, anchor):
        """
        Args:
            anchor: coordinate of the center
        """
        ret = np.zeros(self.shape, dtype='float32')

        y, x = np.mgrid[:self.shape[0], :self.shape[1]]
        y = y.astype('float32') / ret.shape[0] - anchor[0]
        x = x.astype('float32') / ret.shape[1] - anchor[1]
        g = np.exp(-(x**2 + y ** 2) / self.sigma)
        # cv2.imshow(" ", g)
        # cv2.waitKey()
        return g


def np_sample(img, coords):
    # a numpy implementation of ImageSample layer
    coords = np.maximum(coords, 0)
    coords = np.minimum(coords, np.array([img.shape[0] - 1, img.shape[1] - 1]))

    lcoor = np.floor(coords).astype('int32')
    ucoor = lcoor + 1
    ucoor = np.minimum(ucoor, np.array([img.shape[0] - 1, img.shape[1] - 1]))
    diff = coords - lcoor
    neg_diff = 1.0 - diff

    lcoory, lcoorx = np.split(lcoor, 2, axis=2)
    ucoory, ucoorx = np.split(ucoor, 2, axis=2)
    diff = np.repeat(diff, 3, 2).reshape((diff.shape[0], diff.shape[1], 2, 3))
    neg_diff = np.repeat(neg_diff, 3, 2).reshape((diff.shape[0], diff.shape[1], 2, 3))
    diffy, diffx = np.split(diff, 2, axis=2)
    ndiffy, ndiffx = np.split(neg_diff, 2, axis=2)

    ret = img[lcoory, lcoorx, :] * ndiffx * ndiffy + \
        img[ucoory, ucoorx, :] * diffx * diffy + \
        img[lcoory, ucoorx, :] * ndiffy * diffx + \
        img[ucoory, lcoorx, :] * diffy * ndiffx
    return ret[:, :, 0, :]


class GaussianDeform(ImageAugmentor):
    """
    Some kind of slow deformation.
    """
    # TODO input/output with different shape

    def __init__(self, anchors, shape, sigma=0.5, randrange=None):
        """
        Args:
            anchors (list): list of center coordinates in range [0,1].
            shape(list or tuple): image shape in [h, w].
            sigma (float): sigma for Gaussian weight
            randrange (int): offset range. Defaults to shape[0] / 8
        """
        logger.warn("GaussianDeform is slow. Consider using it with 4 or more prefetching processes.")
        super(GaussianDeform, self).__init__()
        self.anchors = anchors
        self.K = len(self.anchors)
        self.shape = shape
        self.grid = np.mgrid[0:self.shape[0], 0:self.shape[1]].transpose(1, 2, 0)
        self.grid = self.grid.astype('float32')  # HxWx2

        gm = GaussianMap(self.shape, sigma=sigma)
        self.gws = np.array([gm.get_gaussian_weight(ank)
                             for ank in self.anchors], dtype='float32')  # KxHxW
        self.gws = self.gws.transpose(1, 2, 0)  # HxWxK
        if randrange is None:
            self.randrange = self.shape[0] / 8
        else:
            self.randrange = randrange

    def _get_augment_params(self, img):
        v = self.rng.rand(self.K, 2).astype('float32') - 0.5
        v = v * 2 * self.randrange
        return v

    def _augment(self, img, v):
        grid = self.grid + np.dot(self.gws, v)
        return np_sample(img, grid)

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()
