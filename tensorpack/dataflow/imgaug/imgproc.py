# -*- coding: UTF-8 -*-
# File: imgproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
import numpy as np
import cv2

__all__ = ['Brightness', 'Contrast', 'MeanVarianceNormalize', 'GaussianBlur',
           'Gamma', 'Clip', 'Saturation', 'Lighting']


class Brightness(ImageAugmentor):
    """
    Randomly adjust brightness.
    """
    def __init__(self, delta, clip=True):
        """
        Randomly add a value within [-delta,delta], and clip in [0,255] if clip is True.
        """
        super(Brightness, self).__init__()
        assert delta > 0
        self._init(locals())

    def _get_augment_params(self, img):
        v = self._rand_range(-self.delta, self.delta)
        return v

    def _augment(self, img, v):
        img += v
        if self.clip:
            img = np.clip(img, 0, 255)
        return img


class Contrast(ImageAugmentor):
    """
    Apply ``x = (x - mean) * contrast_factor + mean`` to each channel.
    """

    def __init__(self, factor_range, clip=True):
        """
        Args:
            factor_range (list or tuple): an interval to randomly sample the `contrast_factor`.
            clip (bool): clip to [0, 255] if True.
        """
        super(Contrast, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        return self._rand_range(*self.factor_range)

    def _augment(self, img, r):
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * r + mean
        if self.clip:
            img = np.clip(img, 0, 255)
        return img


class MeanVarianceNormalize(ImageAugmentor):
    """
    Linearly scales the image to have zero mean and unit norm.
    ``x = (x - mean) / adjusted_stddev``
    where ``adjusted_stddev = max(stddev, 1.0/sqrt(num_pixels * channels))``
    """

    def __init__(self, all_channel=True):
        """
        Args:
            all_channel (bool): if True, normalize all channels together. else separately.
        """
        self.all_channel = all_channel

    def _augment(self, img, _):
        if self.all_channel:
            mean = np.mean(img)
            std = np.std(img)
        else:
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            std = np.std(img, axis=(0, 1), keepdims=True)
        std = np.maximum(std, 1.0 / np.sqrt(np.prod(img.shape)))
        img = (img - mean) / std
        return img


class GaussianBlur(ImageAugmentor):
    """ Gaussian blur the image with random window size"""

    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible Gaussian window size would be 2 * max_size + 1
        """
        super(GaussianBlur, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        sx, sy = self.rng.randint(self.max_size, size=(2,))
        sx = sx * 2 + 1
        sy = sy * 2 + 1
        return sx, sy

    def _augment(self, img, s):
        return cv2.GaussianBlur(img, s, sigmaX=0, sigmaY=0,
                                borderType=cv2.BORDER_REPLICATE)


class Gamma(ImageAugmentor):
    """ Randomly adjust gamma """
    def __init__(self, range=(-0.5, 0.5)):
        """
        Args:
            range(list or tuple): gamma range
        """
        super(Gamma, self).__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        return self._rand_range(*self.range)

    def _augment(self, img, gamma):
        lut = ((np.arange(256, dtype='float32') / 255) ** (1. / (1. + gamma)) * 255).astype('uint8')
        img = np.clip(img, 0, 255).astype('uint8')
        img = cv2.LUT(img, lut).astype('float32')
        return img


class Clip(ImageAugmentor):
    """ Clip the pixel values """

    def __init__(self, min=0, max=255):
        """
        Args:
            min, max: the clip range
        """
        self._init(locals())

    def _augment(self, img, _):
        img = np.clip(img, self.min, self.max)
        return img


class Saturation(ImageAugmentor):
    """ Randomly adjust saturation.
        Follows the implementation in `fb.resnet.torch
        <https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua#L218>`__.
    """

    def __init__(self, alpha=0.4):
        """
        Args:
            alpha(float): maximum saturation change.
        """
        super(Saturation, self).__init__()
        assert alpha < 1
        self._init(locals())

    def _get_augment_params(self, _):
        return 1 + self._rand_range(-self.alpha, self.alpha)

    def _augment(self, img, v):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img * v + (grey * (1 - v))[:, :, np.newaxis]


class Lighting(ImageAugmentor):
    """ Lighting noise, as in the paper
        `ImageNet Classification with Deep Convolutional Neural Networks
        <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_.
        The implementation follows `fb.resnet.torch
        <https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua#L184>`__.
    """

    def __init__(self, std, eigval, eigvec):
        """
        Args:
            std (float): maximum standard deviation
            eigval: a vector of (3,). The eigenvalues of 3 channels.
            eigvec: a 3x3 matrix. Each column is one eigen vector.
        """
        eigval = np.asarray(eigval)
        eigvec = np.asarray(eigvec)
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self._init(locals())

    def _get_augment_params(self, img):
        assert img.shape[2] == 3
        return self.rng.randn(3) * self.std

    def _augment(self, img, v):
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img += inc
        return img
