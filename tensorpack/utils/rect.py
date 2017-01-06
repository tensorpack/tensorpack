#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rect.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np


class Rect(object):
    """
    A rectangle class.

    Note that x1 = x + w, not x+w-1 or something else.
    """
    __slots__ = ['x', 'y', 'w', 'h']

    def __init__(self, x=0, y=0, w=0, h=0, allow_neg=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        if not allow_neg:
            assert min(self.x, self.y, self.w, self.h) >= 0

    @property
    def x0(self):
        return self.x

    @property
    def y0(self):
        return self.y

    @property
    def x1(self):
        return self.x + self.w

    @property
    def y1(self):
        return self.y + self.h

    def copy(self):
        new = type(self)()
        for i in self.__slots__:
            setattr(new, i, getattr(self, i))
        return new

    def __str__(self):
        return 'Rect(x={}, y={}, w={}, h={})'.format(self.x, self.y, self.w, self.h)

    def area(self):
        return self.w * self.h

    def validate(self, shape=None):
        """
        Check that this rect is a valid bounding box within this shape.
        Args:
            shape: [h, w]
        Returns:
            bool
        """
        if min(self.x, self.y) < 0:
            return False
        if min(self.w, self.h) <= 0:
            return False
        if shape is None:
            return True
        if self.x1 > shape[1] - 1:
            return False
        if self.y1 > shape[0] - 1:
            return False
        return True

    def roi(self, img):
        assert self.validate(img.shape[:2]), "{} vs {}".format(self, img.shape[:2])
        return img[self.y0:self.y1 + 1, self.x0:self.x1 + 1]

    def expand(self, frac):
        assert frac > 1.0, frac
        neww = self.w * frac
        newh = self.h * frac
        newx = self.x - (neww - self.w) * 0.5
        newy = self.y - (newh - self.h) * 0.5
        return Rect(*(map(int, [newx, newy, neww, newh])), allow_neg=True)

    def roi_zeropad(self, img):
        shp = list(img.shape)
        shp[0] = self.h
        shp[1] = self.w
        ret = np.zeros(tuple(shp), dtype=img.dtype)

        xstart = 0 if self.x >= 0 else -self.x
        ystart = 0 if self.y >= 0 else -self.y

        xmin = max(self.x0, 0)
        ymin = max(self.y0, 0)
        xmax = min(self.x1, img.shape[1])
        ymax = min(self.y1, img.shape[0])
        patch = img[ymin:ymax, xmin:xmax]
        ret[ystart:ystart + patch.shape[0], xstart:xstart + patch.shape[1]] = patch
        return ret

    __repr__ = __str__


if __name__ == '__main__':
    x = Rect(2, 1, 3, 3, allow_neg=True)

    img = np.random.rand(3, 3)
    print(img)
    print(x.roi_zeropad(img))
