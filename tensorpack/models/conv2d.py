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
           W_init=None, b_init=None,
           nl=tf.nn.relu, split=1):
    """
    kernel_shape: (h, w) or a int
    stride: (h, w) or a int
    padding: 'valid' or 'same'
    split: split channels. used in Alexnet
    """
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[-1]
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride)

    if W_init is None:
        W_init = tf.truncated_normal_initializer(stddev=1e-2)
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    b = tf.get_variable('b', [out_channel], initializer=b_init)

    if split == 1:
        conv = tf.nn.conv2d(x, W, stride, padding)
    else:
        inputs = tf.split(3, split, x)
        kernels = tf.split(3, split, W)
        outputs = [tf.nn.conv2d(i, k, stride, padding)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(3, outputs)
    return nl(tf.nn.bias_add(conv, b))


