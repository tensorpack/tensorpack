#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rect.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


class Rect(object):
    """
    A Rectangle.
    Note that x1 = x+w, not x+w-1 or something
    """
    __slots__ = ['x', 'y', 'w', 'h']

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
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
        Is a valid bounding box within this shape
        :param shape: [h, w]
        :returns: boolean
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
        assert self.validate(img.shape[:2])
        return img[self.y0:self.y1+1, self.x0:self.x1+1]

    __repr__ = __str__
