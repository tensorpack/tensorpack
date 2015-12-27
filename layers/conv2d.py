#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import math
from ._common import *

__all__ = ['Conv2D']

@layer_register(summary_activation=True)
def Conv2D(x, out_channel, kernel_shape,
           padding='VALID', stride=1,
           W_init=None, b_init=None, nl=tf.nn.relu):
    """
    kernel_shape: (h, w) or a int
    stride: (h, w) or a int
    padding: 'valid' or 'same'
    """
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[-1]

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel, out_channel]
    stride = shape4d(stride)

    if W_init is None:
        W_init = tf.truncated_normal_initializer(stddev=0.04)
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init) # TODO collections
    b = tf.get_variable('b', [out_channel], initializer=b_init)

    conv = tf.nn.conv2d(x, W, stride, padding)
    return nl(tf.nn.bias_add(conv, b))

