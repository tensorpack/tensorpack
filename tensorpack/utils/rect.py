#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rect.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np

__all__ = ['IntBox', 'FloatBox']


class BoxBase(object):
    __slots__ = ['x1', 'y1', 'x2', 'y2']

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def copy(self):
        new = type(self)()
        for i in self.__slots__:
            setattr(new, i, getattr(self, i))
        return new

    def __str__(self):
        return '{}(x1={}, y1={}, x2={}, y2={})'.format(
            type(self).__name__, self.x1, self.y1, self.x2, self.y2)

    __repr__ = __str__

    def area(self):
        return self.w * self.h

    def is_box(self):
        return self.area() > 0


class IntBox(BoxBase):
    def __init__(self, x1, y1, x2, y2):
        for k in [x1, y1, x2, y2]:
            assert isinstance(k, int)
        super(IntBox, self).__init__(x1, y1, x2, y2)

    @property
    def w(self):
        return self.x2 - self.x1 + 1

    @property
    def h(self):
        return self.y2 - self.y1 + 1

    def is_valid_box(self, shape):
        """
        Check that this rect is a valid bounding box within this shape.

        Args:
            shape: int [h, w] or None.
        Returns:
            bool
        """
        if min(self.x1, self.y1) < 0:
            return False
        if min(self.w, self.h) <= 0:
            return False
        if self.x2 >= shape[1]:
            return False
        if self.y2 >= shape[0]:
            return False
        return True

    def roi(self, img):
        assert self.validate(img.shape[:2]), "{} vs {}".format(self, img.shape[:2])
        return img[self.y1:self.y2 + 1, self.x1:self.x2 + 1]

    # def expand(self, frac):
    #     assert frac > 1.0, frac
    #     neww = self.w * frac
    #     newh = self.h * frac
    #     newx = self.x - (neww - self.w) * 0.5
    #     newy = self.y - (newh - self.h) * 0.5
    #     return Rect(*(map(int, [newx, newy, neww, newh])), allow_neg=True)

    # def roi_zeropad(self, img):
    #     shp = list(img.shape)
    #     shp[0] = self.h
    #     shp[1] = self.w
    #     ret = np.zeros(tuple(shp), dtype=img.dtype)

    #     xstart = 0 if self.x >= 0 else -self.x
    #     ystart = 0 if self.y >= 0 else -self.y

    #     xmin = max(self.x0, 0)
    #     ymin = max(self.y0, 0)
    #     xmax = min(self.x1, img.shape[1])
    #     ymax = min(self.y1, img.shape[0])
    #     patch = img[ymin:ymax, xmin:xmax]
    #     ret[ystart:ystart + patch.shape[0], xstart:xstart + patch.shape[1]] = patch
    #     return ret


class FloatBox(BoxBase):
    def __init__(self, x1, y1, x2, y2):
        for k in [x1, y1, x2, y2]:
            assert isinstance(k, float)
        super(FloatBox, self).__init__(x1, y1, x2, y2)

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @staticmethod
    def from_intbox(intbox):
        return FloatBox(intbox.x1, intbox.y1,
                        intbox.x2 + 1, intbox.y2 + 1)


if __name__ == '__main__':
    x = IntBox(2, 1, 3, 3)

    img = np.random.rand(3, 3)
    print(img)
