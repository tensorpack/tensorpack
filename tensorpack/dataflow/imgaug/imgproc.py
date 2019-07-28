# -*- coding: utf-8 -*-
# File: imgproc.py


import numpy as np
import cv2

from .base import PhotometricAugmentor

__all__ = ['Hue', 'Brightness', 'BrightnessScale', 'Contrast', 'MeanVarianceNormalize',
           'GaussianBlur', 'Gamma', 'Clip', 'Saturation', 'Lighting', 'MinMaxNormalize']


class Hue(PhotometricAugmentor):
    """ Randomly change color hue.
    """

    def __init__(self, range=(0, 180), rgb=True):
        """
        Args:
            range(list or tuple): range from which the applied hue offset is selected (maximum [-90,90] or [0,180])
            rgb (bool): whether input is RGB or BGR.
        """
        super(Hue, self).__init__()
        rgb = bool(rgb)
        self._init(locals())

    def _get_augment_params(self, _):
        return self._rand_range(*self.range)

    def _augment(self, img, hue):
        m = cv2.COLOR_BGR2HSV if not self.rgb else cv2.COLOR_RGB2HSV
        hsv = cv2.cvtColor(img, m)
        # https://docs.opencv.org/3.2.0/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
        if hsv.dtype.itemsize == 1:
            # OpenCV uses 0-179 for 8-bit images
            hsv[..., 0] = (hsv[..., 0] + hue) % 180
        else:
            # OpenCV uses 0-360 for floating point images
            hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
        m = cv2.COLOR_HSV2BGR if not self.rgb else cv2.COLOR_HSV2RGB
        img = cv2.cvtColor(hsv, m)
        return img


class Brightness(PhotometricAugmentor):
    """
    Adjust brightness by adding a random number.
    """
    def __init__(self, delta, clip=True):
        """
        Args:
            delta (float): Randomly add a value within [-delta,delta]
            clip (bool): clip results to [0,255] even when data type is not uint8.
        """
        super(Brightness, self).__init__()
        assert delta > 0
        self._init(locals())

    def _get_augment_params(self, _):
        return self._rand_range(-self.delta, self.delta)

    def _augment(self, img, v):
        old_dtype = img.dtype
        img = img.astype('float32')
        img += v
        if self.clip or old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class BrightnessScale(PhotometricAugmentor):
    """
    Adjust brightness by scaling by a random factor.
    """
    def __init__(self, range, clip=True):
        """
        Args:
            range (tuple): Randomly scale the image by a factor in (range[0], range[1])
            clip (bool): clip results to [0,255] even when data type is not uint8.
        """
        super(BrightnessScale, self).__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        return self._rand_range(*self.range)

    def _augment(self, img, v):
        old_dtype = img.dtype
        img = img.astype('float32')
        img *= v
        if self.clip or old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class Contrast(PhotometricAugmentor):
    """
    Apply ``x = (x - mean) * contrast_factor + mean`` to each channel.
    """

    def __init__(self, factor_range, rgb=None, clip=True):
        """
        Args:
            factor_range (list or tuple): an interval to randomly sample the `contrast_factor`.
            rgb (bool or None): if None, use the mean per-channel.
            clip (bool): clip to [0, 255] even when data type is not uint8.
        """
        super(Contrast, self).__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        return self._rand_range(*self.factor_range)

    def _augment(self, img, r):
        old_dtype = img.dtype

        if img.ndim == 3:
            if self.rgb is not None:
                m = cv2.COLOR_RGB2GRAY if self.rgb else cv2.COLOR_BGR2GRAY
                grey = cv2.cvtColor(img.astype('float32'), m)
                mean = np.mean(grey)
            else:
                mean = np.mean(img, axis=(0, 1), keepdims=True)
        else:
            mean = np.mean(img)

        img = img * r + mean * (1 - r)
        if self.clip or old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class MeanVarianceNormalize(PhotometricAugmentor):
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


class GaussianBlur(PhotometricAugmentor):
    """ Gaussian blur the image with random window size"""

    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible Gaussian window size would be 2 * max_size + 1
        """
        super(GaussianBlur, self).__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        sx, sy = self.rng.randint(self.max_size, size=(2,))
        sx = sx * 2 + 1
        sy = sy * 2 + 1
        return sx, sy

    def _augment(self, img, s):
        return np.reshape(cv2.GaussianBlur(img, s, sigmaX=0, sigmaY=0,
                                           borderType=cv2.BORDER_REPLICATE), img.shape)


class Gamma(PhotometricAugmentor):
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


class Clip(PhotometricAugmentor):
    """ Clip the pixel values """

    def __init__(self, min=0, max=255):
        """
        Args:
            min, max: the clip range
        """
        self._init(locals())

    def _augment(self, img, _):
        return np.clip(img, self.min, self.max)


class Saturation(PhotometricAugmentor):
    """ Randomly adjust saturation.
        Follows the implementation in `fb.resnet.torch
        <https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua#L218>`__.
    """

    def __init__(self, alpha=0.4, rgb=True):
        """
        Args:
            alpha(float): maximum saturation change.
            rgb (bool): whether input is RGB or BGR.
        """
        super(Saturation, self).__init__()
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
        if old_dtype == np.uint8:
            ret = np.clip(ret, 0, 255)
        return ret.astype(old_dtype)


class Lighting(PhotometricAugmentor):
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
        super(Lighting, self).__init__()
        eigval = np.asarray(eigval)
        eigvec = np.asarray(eigvec)
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self._init(locals())

    def _get_augment_params(self, img):
        assert img.shape[2] == 3
        return (self.rng.randn(3) * self.std).astype("float32")

    def _augment(self, img, v):
        old_dtype = img.dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class MinMaxNormalize(PhotometricAugmentor):
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
