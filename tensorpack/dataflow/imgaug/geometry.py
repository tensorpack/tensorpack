#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: geometry.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import math
import cv2
import numpy as np

from .base import ImageAugmentor
from .transform import TransformAugmentorBase, WarpAffineTransform

__all__ = ['Shift', 'Rotation', 'RotationAndCropValid', 'Affine']


class Shift(TransformAugmentorBase):
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

        mat = np.array([[1, 0, dx], [0, 1, dy]], dtype='float32')
        return WarpAffineTransform(
            mat, img.shape[1::-1],
            borderMode=self.border, borderValue=self.border_value)


class Rotation(TransformAugmentorBase):
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
        """
        The correct center is shape*0.5-0.5 This can be verified by:

        SHAPE = 7
        arr = np.random.rand(SHAPE, SHAPE)
        orig = arr
        c = SHAPE * 0.5 - 0.5
        c = (c, c)
        for k in range(4):
            mat = cv2.getRotationMatrix2D(c, 90, 1)
            arr = cv2.warpAffine(arr, mat, arr.shape)
        assert np.all(arr == orig)
        """
        mat = cv2.getRotationMatrix2D(tuple(center - 0.5), deg, 1)
        return WarpAffineTransform(
            mat, img.shape[1::-1], interp=self.interp,
            borderMode=self.border, borderValue=self.border_value)


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


class Affine(TransformAugmentorBase):
    """
    Random affine transform of the image w.r.t to the image center.
    Transformations involve:
        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)
    """

    def __init__(self, scale=None, translate_frac=None, rotate_max_deg=0.0, shear=0.0,
                 interp=cv2.INTER_LINEAR, border=cv2.BORDER_REPLICATE, border_value=0):
        """
        Args:
            scale (tuple of 2 floats): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep
                original scale by default.
            translate_frac (tuple of 2 floats): tuple of max abs fraction for horizontal
                and vertical translation. For example translate_frac=(a, b), then horizontal shift
                is randomly sampled in the range 0 < dx < img_width * a and vertical shift is
                randomly sampled in the range 0 < dy < img_height * b. Will
                not translate by default.
            shear (float): max abs shear value in degrees between 0 to 180
            interp: cv2 interpolation method
            border: cv2 border method
            border_value: cv2 border value for border=cv2.BORDER_CONSTANT
        """
        if scale is not None:
            assert isinstance(scale, tuple) and len(scale) == 2, \
                "Argument scale should be a tuple of two floats, e.g (a, b)"

        if translate_frac is not None:
            assert isinstance(translate_frac, tuple) and len(translate_frac) == 2, \
                "Argument translate_frac should be a tuple of two floats, e.g (a, b)"

        assert shear >= 0.0, "Argument shear should be between 0.0 and 180.0"

        super(Affine, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        if self.scale is not None:
            scale = self._rand_range(self.scale[0], self.scale[1])
        else:
            scale = 1.0

        if self.translate_frac is not None:
            max_dx = self.translate_frac[0] * img.shape[1]
            max_dy = self.translate_frac[1] * img.shape[0]
            dx = np.round(self._rand_range(-max_dx, max_dx))
            dy = np.round(self._rand_range(-max_dy, max_dy))
        else:
            dx = 0
            dy = 0

        if self.shear > 0.0:
            shear = self._rand_range(-self.shear, self.shear)
            sin_shear = math.sin(math.radians(shear))
            cos_shear = math.cos(math.radians(shear))
        else:
            sin_shear = 0.0
            cos_shear = 1.0

        center = (img.shape[1::-1] * np.array((0.5, 0.5))) - 0.5
        deg = self._rand_range(-self.rotate_max_deg, self.rotate_max_deg)

        transform_matrix = cv2.getRotationMatrix2D(tuple(center), deg, scale)

        # Apply shear :
        if self.shear > 0.0:
            m00 = transform_matrix[0, 0]
            m01 = transform_matrix[0, 1]
            m10 = transform_matrix[1, 0]
            m11 = transform_matrix[1, 1]
            transform_matrix[0, 1] = m01 * cos_shear + m00 * sin_shear
            transform_matrix[1, 1] = m11 * cos_shear + m10 * sin_shear
            # Add correction term to keep the center unchanged
            tx = center[0] * (1.0 - m00) - center[1] * transform_matrix[0, 1]
            ty = -center[0] * m10 + center[1] * (1.0 - transform_matrix[1, 1])
            transform_matrix[0, 2] = tx
            transform_matrix[1, 2] = ty

        # Apply shift :
        transform_matrix[0, 2] += dx
        transform_matrix[1, 2] += dy
        return WarpAffineTransform(transform_matrix, img.shape[1::-1],
                                   self.interp, self.border, self.border_value)
