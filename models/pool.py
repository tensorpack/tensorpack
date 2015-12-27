#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: pool.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from ._common import *
import tensorflow as tf

__all__ = ['MaxPooling']

@layer_register()
def MaxPooling(x, shape, stride=None, padding='VALID'):
    """
        shape, stride: int or list/tuple of length 2
        if stride is None, use shape by default
        padding: 'VALID' or 'SAME'
    """
    padding = padding.upper()
    shape = shape4d(shape)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride)

    return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)

