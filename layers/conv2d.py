#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import math
from ._common import layer_register

__all__ = ['Conv2D']

@layer_register()
def Conv2D(x, out_channel, kernel_shape,
           padding='VALID', stride=None,
           W_init=None, b_init=None):
    """
    kernel_shape: (h, w) or a int
    stride: (h, w) or a int
    padding: 'valid' or 'same'
    """
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[-1]

    if type(kernel_shape) == int:
        kernel_shape = [kernel_shape, kernel_shape]
    padding = padding.upper()

    filter_shape = kernel_shape + [in_channel, out_channel]

    if stride is None:
        stride = [1, 1, 1, 1]
    elif type(stride) == int:
        stride = [1, stride, stride, 1]
    elif type(stride) in [list, tuple]:
        assert len(stride) == 2
        stride = [1] + list(stride) + [1]

    if W_init is None:
        W_init = tf.truncated_normal_initializer(stddev=0.04)
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init) # TODO collections
    b = tf.get_variable('b', [out_channel], initializer=b_init)

    conv = tf.nn.conv2d(x, W, stride, padding)
    return tf.nn.bias_add(conv, b)

