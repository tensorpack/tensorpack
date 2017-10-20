# -*- coding: UTF-8 -*-
# File: imgproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import ImageAugmentor
from ...utils import logger
import numpy as np
import cv2

__all__ = ['Hue', 'Brightness', 'BrightnessScale', 'Contrast', 'MeanVarianceNormalize',
           'GaussianBlur', 'Gamma', 'Clip', 'Saturation', 'Lighting', 'MinMaxNormalize']


class Hue(ImageAugmentor):
    """ Randomly change color hue.
    """

    def __init__(self, range=(0, 180), rgb=None):
        """
        Args:
            range(list or tuple): hue range
            rgb (bool): whether input is RGB or BGR.
        """
        super(Hue, self).__init__()
        if rgb is None:
            logger.warn("Hue() now assumes rgb=False, but will by default use rgb=True in the future!")
            rgb = False
        rgb = bool(rgb)
        self._init(locals())

    def _get_augment_params(self, _):
        return self._rand_range(*self.range)

    def _augment(self, img, hue):
        m = cv2.COLOR_BGR2HSV if not self.rgb else cv2.COLOR_RGB2HSV
        hsv = cv2.cvtColor(img, m)
        # Note, OpenCV used 0-179 degree instead of 0-359 degree
        hsv[..., 0] = (hsv[..., 0] + hue) % 180

        m = cv2.COLOR_HSV2BGR if not self.rgb else cv2.COLOR_HSV2RGB
        img = cv2.cvtColor(hsv, m)
        return img


class Brightness(ImageAugmentor):
    """
    Adjust brightness by adding a random number.
    """
    def __init__(self, delta, clip=True):
        """
        Args:
            delta (float): Randomly add a value within [-delta,delta]
            clip (bool): clip results to [0,255].
        """
        super(Brightness, self).__init__()
        assert delta > 0
        self._init(locals())

    def _get_augment_params(self, _):
        v = self._rand_range(-self.delta, self.delta)
        return v

    def _augment(self, img, v):
        old_dtype = img.dtype
        img = img.astype('float32')
        img += v
        if self.clip or old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class BrightnessScale(ImageAugmentor):
    """
    Adjust brightness by scaling by a random factor.
    """
    def __init__(self, range, clip=True):
        """
        Args:
            range (tuple): Randomly scale the image by a factor in (range[0], range[1])
            clip (bool): clip results to [0,255].
        """
        super(BrightnessScale, self).__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        v = self._rand_range(*self.range)
        return v

    def _augment(self, img, v):
        old_dtype = img.dtype
        img = img.astype('float32')
        img *= v
        if self.clip or old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


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
        old_dtype = img.dtype
        img = img.astype('float32')
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * r + mean
        if self.clip or old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class MeanVarianceNormalize(ImageAugmentor):
    """
    Linearly scales the image to have zero mean and unit norm.
    ``x = (x - mean) / adjusted_stddev``
    where ``adjusted_stddev = max(stddev, 1.0/sqrt(num_pixels * channels))``

    This augmentor always returns float32 images.
    """

    def __init__(self, all_channel=True):
        """
        Args:
            all_channel (bool): if True, normalize all channels together. else separately.
        """
        self._init(locals())

    def _augment(self, img, _):
        img = img.astype('float32')
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
        return np.reshape(cv2.GaussianBlur(img, s, sigmaX=0, sigmaY=0,
                                           borderType=cv2.BORDER_REPLICATE), img.shape)


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
        old_dtype = img.dtype
        lut = ((np.arange(256, dtype='float32') / 255) ** (1. / (1. + gamma)) * 255).astype('uint8')
        img = np.clip(img, 0, 255).astype('uint8')
        ret = cv2.LUT(img, lut).astype(old_dtype)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret


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

    def __init__(self, alpha=0.4, rgb=None):
        """
        Args:
            alpha(float): maximum saturation change.
            rgb (bool): whether input is RGB or BGR.
        """
        super(Saturation, self).__init__()
        if rgb is None:
            logger.warn("Saturation() now assumes rgb=False, but will by default use rgb=True in the future!")
            rgb = False
        rgb = bool(rgb)
        assert alpha < 1
        self._init(locals())

    def _get_augment_params(self, _):
        return 1 + self._rand_range(-self.alpha, self.alpha)

    def _augment(self, img, v):
        old_dtype = img.dtype
        m = cv2.COLOR_RGB2GRAY if self.rgb else cv2.COLOR_BGR2GRAY
        grey = cv2.cvtColor(img, m)
        ret = img * v + (grey * (1 - v))[:, :, np.newaxis]
        return ret.astype(old_dtype)


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
        old_dtype = img.dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class MinMaxNormalize(ImageAugmentor):
    """
    Linearly scales the image to the range [min, max].

    This augmentor always returns float32 images.
    """
    def __init__(self, min=0, max=255, all_channel=True):
        """
        Args:
            max (float): The new maximum value
            min (float): The new minimum value
            all_channel (bool): if True, normalize all channels together. else separately.
        """
        self._init(locals())

    def _augment(self, img, _):
        img = img.astype('float32')
        if self.all_channel:
            minimum = np.min(img)
            maximum = np.max(img)
        else:
            minimum = np.min(img, axis=(0, 1), keepdims=True)
            maximum = np.max(img, axis=(0, 1), keepdims=True)
        img = (self.max - self.min) * (img - minimum) / (maximum - minimum) + self.min
        return img
