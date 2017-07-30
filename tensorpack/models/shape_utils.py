#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: shape_utils.py

import tensorflow as tf

__all__ = []


class StaticDynamicAxis(object):
    def __init__(self, static, dynamic):
        self.static = static
        self.dynamic = dynamic

    def apply(self, f):
        try:
            st = f(self.static)
            return StaticDynamicAxis(st, st)
        except:
            return StaticDynamicAxis(None, f(self.dynamic))

    def __str__(self):
        return "S={}, D={}".format(str(self.static), str(self.dynamic))


def DynamicLazyAxis(shape, idx):
    return lambda: shape[idx]


def StaticLazyAxis(dim):
    return lambda: dim


class StaticDynamicShape(object):
    def __init__(self, tensor):
        assert isinstance(tensor, tf.Tensor), tensor
        ndims = tensor.shape.ndims
        self.static = tensor.shape.as_list()
        if tensor.shape.is_fully_defined():
            self.dynamic = self.static[:]
        else:
            dynamic = tf.shape(tensor)
            self.dynamic = [DynamicLazyAxis(dynamic, k) for k in range(ndims)]

        for k in range(ndims):
            if self.static[k] is not None:
                self.dynamic[k] = StaticLazyAxis(self.static[k])

    def apply(self, axis, f):
        if self.static[axis] is not None:
            try:
                st = f(self.static[axis])
                self.static[axis] = st
                self.dynamic[axis] = StaticLazyAxis(st)
                return
            except:
                pass
        self.static[axis] = None
        dyn = self.dynamic[axis]
        self.dynamic[axis] = lambda: f(dyn())

    def get_static(self):
        return self.static

    @property
    def ndims(self):
        return len(self.static)

    def get_dynamic(self, axis=None):
        if axis is None:
            return [self.dynamic[k]() for k in range(self.ndims)]
        return self.dynamic[axis]()


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=[None, 3, None, 10])
    shape = StaticDynamicShape(x)
    shape.apply(1, lambda x: x * 3)
    shape.apply(2, lambda x: x + 5)
    print(shape.get_static())
    print(shape.get_dynamic())
