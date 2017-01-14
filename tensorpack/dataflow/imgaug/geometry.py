#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: geometry.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import ImageAugmentor
import math
import cv2

__all__ = ['Rotation', 'RotationAndCropValid']


class Rotation(ImageAugmentor):
    """ Random rotate the image w.r.t a random center"""

    def __init__(self, max_deg, center_range=(0, 1),
                 interp=cv2.INTER_LINEAR,
                 border=cv2.BORDER_REPLICATE):
        """
        Args:
            max_deg (float): max abs value of the rotation degree (in angle).
            center_range (tuple): (min, max) range of the random rotation center.
            interp: cv2 interpolation method
            border: cv2 border method
        """
        super(Rotation, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        center = img.shape[1::-1] * self._rand_range(
            self.center_range[0], self.center_range[1], (2,))
        deg = self._rand_range(-self.max_deg, self.max_deg)
        return cv2.getRotationMatrix2D(tuple(center), deg, 1)

    def _augment(self, img, rot_m):
        ret = cv2.warpAffine(img, rot_m, img.shape[1::-1],
                             flags=self.interp, borderMode=self.border)
        return ret


class RotationAndCropValid(ImageAugmentor):
    """ Random rotate and then crop the largest possible rectangle.
        Note that this will produce images of different shapes.
    """

    def __init__(self, max_deg, interp=cv2.INTER_LINEAR):
        """
        Args:
            max_deg (float): max abs value of the rotation degree (in angle).
            interp: cv2 interpolation method
        """
        super(RotationAndCropValid, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        deg = self._rand_range(-self.max_deg, self.max_deg)
        return deg

    def _augment(self, img, deg):
        center = (img.shape[1] * 0.5, img.shape[0] * 0.5)
        rot_m = cv2.getRotationMatrix2D(center, deg, 1)
        ret = cv2.warpAffine(img, rot_m, img.shape[1::-1],
                             flags=self.interp, borderMode=cv2.BORDER_CONSTANT)
        neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
        neww = min(neww, ret.shape[1])
        newh = min(newh, ret.shape[0])
        newx = int(center[0] - neww * 0.5)
        newy = int(center[1] - newh * 0.5)
        # print(ret.shape, deg, newx, newy, neww, newh)
        return ret[newy:newy + newh, newx:newx + neww]

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

        return int(wr), int(hr)
