#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf
import math
from ._common import layer_register, shape2d, shape4d
from ..utils import map_arg, logger

__all__ = ['Conv2D', 'Deconv2D']

@layer_register()
def Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           nl=None, split=1, use_bias=True):
    """
    2D convolution on 4D inputs.

    :param input: a tensor of shape NHWC
    :param out_channel: number of output channel
    :param kernel_shape: (h, w) or a int
    :param stride: (h, w) or a int. default to 1
    :param padding: 'valid' or 'same'. default to 'same'
    :param split: split channels as used in Alexnet. Default to 1 (no split)
    :param W_init: initializer for W. default to `xavier_initializer_conv2d`.
    :param b_init: initializer for b. default to zero initializer.
    :param nl: nonlinearity
    :param use_bias: whether to use bias. a boolean default to True
    :returns: a NHWC tensor
    """
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[-1]
    assert in_channel is not None, "Input to Conv2D cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride)

    if W_init is None:
        W_init = tf.contrib.layers.xavier_initializer_conv2d()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    if split == 1:
        conv = tf.nn.conv2d(x, W, stride, padding)
    else:
        inputs = tf.split(3, split, x)
        kernels = tf.split(3, split, W)
        outputs = [tf.nn.conv2d(i, k, stride, padding)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(3, outputs)
    if nl is None:
        logger.warn("[DEPRECATED] Default ReLU nonlinearity for Conv2D and FullyConnected will be deprecated. Please use argscope instead.")
        nl = tf.nn.relu
    return nl(tf.nn.bias_add(conv, b) if use_bias else conv, name='output')

@layer_register()
def Deconv2D(x, out_shape, kernel_shape,
           stride, padding='SAME',
           W_init=None, b_init=None,
           nl=tf.identity, use_bias=True):
    """
    2D deconvolution on 4D inputs.

    :param input: a tensor of shape NHWC
    :param out_shape: either (h, w, channel), or just channel,
        then h, w will calculated by input_shape * stride
    :param kernel_shape: (h, w) or a int
    :param stride: (h, w) or a int
    :param padding: 'valid' or 'same'. default to 'same'
    :param W_init: initializer for W. default to `xavier_initializer_conv2d`.
    :param b_init: initializer for b. default to zero initializer.
    :param nl: nonlinearity.
    :param use_bias: whether to use bias. a boolean default to True
    :returns: a NHWC tensor
    """
    in_shape = x.get_shape().as_list()[1:]
    assert None not in in_shape, "Input to Deconv2D cannot have unknown shape!"
    in_channel = in_shape[-1]
    kernel_shape = shape2d(kernel_shape)
    stride2d = shape2d(stride)
    stride4d = shape4d(stride)
    padding = padding.upper()

    if isinstance(out_shape, int):
        out_channel = out_shape
        shape3 = [stride2d[0] * in_shape[0], stride2d[1] * in_shape[1], out_shape]
    else:
        out_channel = out_shape[-1]
        shape3 = out_shape
    filter_shape = kernel_shape + [out_channel, in_channel]

    if W_init is None:
        W_init = tf.contrib.layers.xavier_initializer_conv2d()
    if b_init is None:
        b_init = tf.constant_initializer()
    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    out_shape = tf.pack([tf.shape(x)[0]] + shape3)
    conv = tf.nn.conv2d_transpose(x, W, out_shape, stride4d, padding=padding)
    conv.set_shape(tf.TensorShape([None] + shape3))
    return nl(tf.nn.bias_add(conv, b) if use_bias else conv, name='output')
