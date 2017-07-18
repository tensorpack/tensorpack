#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: geometry.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import ImageAugmentor
import math
import cv2
import numpy as np

__all__ = ['Shift', 'Rotation', 'RotationAndCropValid']


class Shift(ImageAugmentor):
    """ Random horizontal and vertical shifts """

    def __init__(self, horiz_frac=0, vert_frac=0,
                 border=cv2.BORDER_REPLICATE, border_value=0):
        """
        Args:
            horiz_frac (float): max abs fraction for horizontal shift
            vert_frac (float): max abs fraction for horizontal shift
            border: cv2 border method
            border_value: cv2 border value for border=cv2.BORDER_CONSTANT
        """
        assert horiz_frac < 1.0 and vert_frac < 1.0
        super(Shift, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        max_dx = self.horiz_frac * img.shape[1]
        max_dy = self.vert_frac * img.shape[0]
        dx = np.round(self._rand_range(-max_dx, max_dx))
        dy = np.round(self._rand_range(-max_dy, max_dy))
        return np.float32(
            [[1, 0, dx], [0, 1, dy]])

    def _augment(self, img, shift_m):
        ret = cv2.warpAffine(img, shift_m, img.shape[1::-1],
                             borderMode=self.border, borderValue=self.border_value)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def _augment_coords(self, coords, param):
        raise NotImplementedError()


class Rotation(ImageAugmentor):
    """ Random rotate the image w.r.t a random center"""

    def __init__(self, max_deg, center_range=(0, 1),
                 interp=cv2.INTER_LINEAR,
                 border=cv2.BORDER_REPLICATE, step_deg=None, border_value=0):
        """
        Args:
            max_deg (float): max abs value of the rotation angle (in degree).
            center_range (tuple): (min, max) range of the random rotation center.
            interp: cv2 interpolation method
            border: cv2 border method
            step_deg (float): if not None, the stepping of the rotation
                angle. The rotation angle will be a multiple of step_deg. This
                option requires ``max_deg==180`` and step_deg has to be a divisor of 180)
            border_value: cv2 border value for border=cv2.BORDER_CONSTANT
        """
        assert step_deg is None or (max_deg == 180 and max_deg % step_deg == 0)
        super(Rotation, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        center = img.shape[1::-1] * self._rand_range(
            self.center_range[0], self.center_range[1], (2,))
        deg = self._rand_range(-self.max_deg, self.max_deg)
        if self.step_deg:
            deg = deg // self.step_deg * self.step_deg
        return cv2.getRotationMatrix2D(tuple(center - 0.5), deg, 1)

    def _augment(self, img, rot_m):
        ret = cv2.warpAffine(img, rot_m, img.shape[1::-1],
                             flags=self.interp, borderMode=self.border, borderValue=self.border_value)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def _augment_coords(self, coords, param):
        raise NotImplementedError()


class RotationAndCropValid(ImageAugmentor):
    """ Random rotate and then crop the largest possible rectangle.
        Note that this will produce images of different shapes.
    """

    def __init__(self, max_deg, interp=cv2.INTER_LINEAR, step_deg=None):
        """
        Args:
            max_deg, interp, step_deg: same as :class:`Rotation`
        """
        assert step_deg is None or (max_deg == 180 and max_deg % step_deg == 0)
        super(RotationAndCropValid, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        deg = self._rand_range(-self.max_deg, self.max_deg)
        if self.step_deg:
            deg = deg // self.step_deg * self.step_deg
        return deg

    def _augment(self, img, deg):
        center = (img.shape[1] * 0.5, img.shape[0] * 0.5)
        rot_m = cv2.getRotationMatrix2D((center[0] - 0.5, center[1] - 0.5), deg, 1)
        ret = cv2.warpAffine(img, rot_m, img.shape[1::-1],
                             flags=self.interp, borderMode=cv2.BORDER_CONSTANT)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
        neww = min(neww, ret.shape[1])
        newh = min(newh, ret.shape[0])
        newx = int(center[0] - neww * 0.5)
        newy = int(center[1] - newh * 0.5)
        # print(ret.shape, deg, newx, newy, neww, newh)
        return ret[newy:newy + newh, newx:newx + neww]

    def _augment_coords(self, coords, param):
        raise NotImplementedError()

    @staticmethod
    def largest_rotated_rect(w, h, angle):
        """
        Get largest rectangle after rotation.
        http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        angle = angle / 180.0 * math.pi
        if w <= 0 or h <= 0:
            return 0, 0

        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2. * sin_a * cos_a * side_long:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
        return int(np.round(wr)), int(np.round(hr))
