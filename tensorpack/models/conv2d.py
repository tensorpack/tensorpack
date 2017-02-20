#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from .common import layer_register
from ..utils.argtools import shape2d, shape4d

__all__ = ['Conv2D', 'Deconv2D']


@layer_register()
def Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           nl=tf.identity, split=1, use_bias=True,
           data_format='NHWC'):
    """
    2D convolution on 4D inputs.

    Args:
        x (tf.Tensor): a 4D tensor.
            Must have known number of channels, but can have other unknown dimensions.
        out_channel (int): number of output channel.
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        split (int): Split channels as used in Alexnet. Defaults to 1 (no split).
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor named ``output``.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride, data_format=data_format)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    if split == 1:
        conv = tf.nn.conv2d(x, W, stride, padding, data_format=data_format)
    else:
        inputs = tf.split(x, split, channel_axis)
        kernels = tf.split(W, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding, data_format=data_format)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)
    return nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')


class StaticDynamicShape(object):
    def __init__(self, static, dynamic):
        self.static = static
        self.dynamic = dynamic

    def apply(self, f):
        try:
            st = f(self.static)
            return StaticDynamicShape(st, st)
        except:
            return StaticDynamicShape(None, f(self.dynamic))


@layer_register()
def Deconv2D(x, out_shape, kernel_shape,
             stride, padding='SAME',
             W_init=None, b_init=None,
             nl=tf.identity, use_bias=True,
             data_format='NHWC'):
    """
    2D deconvolution on 4D inputs.

    Args:
        x (tf.Tensor): a tensor of shape NHWC.
            Must have known number of channels, but can have other unknown dimensions.
        out_shape: (h, w, channel) tuple, or just a integer channel,
            then (h, w) will be calculated by input_shape * stride
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor: a NHWC tensor named ``output``.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"
    kernel_shape = shape2d(kernel_shape)
    stride2d = shape2d(stride)
    stride4d = shape4d(stride, data_format=data_format)
    padding = padding.upper()
    in_shape_dyn = tf.shape(x)

    if isinstance(out_shape, int):
        out_channel = out_shape
        if data_format == 'NHWC':
            shp3_0 = StaticDynamicShape(in_shape[1], in_shape_dyn[1]).apply(lambda x: stride2d[0] * x)
            shp3_1 = StaticDynamicShape(in_shape[2], in_shape_dyn[2]).apply(lambda x: stride2d[1] * x)
            shp3_dyn = [shp3_0.dynamic, shp3_1.dynamic, out_channel]
            shp3_static = [shp3_0.static, shp3_1.static, out_channel]
        else:
            shp3_0 = StaticDynamicShape(in_shape[2], in_shape_dyn[2]).apply(lambda x: stride2d[0] * x)
            shp3_1 = StaticDynamicShape(in_shape[3], in_shape_dyn[3]).apply(lambda x: stride2d[1] * x)
            shp3_dyn = [out_channel, shp3_0.dynamic, shp3_1.dynamic]
            shp3_static = [out_channel, shp3_0.static, shp3_1.static]
    else:
        for k in out_shape:
            if not isinstance(k, int):
                raise ValueError("[Deconv2D] out_shape {} is invalid!".format(k))
        out_channel = out_shape[channel_axis - 1]   # out_shape doesn't have batch
        shp3_static = shp3_dyn = out_shape
    filter_shape = kernel_shape + [out_channel, in_channel]

    if W_init is None:
        W_init = tf.contrib.layers.xavier_initializer_conv2d()
    if b_init is None:
        b_init = tf.constant_initializer()
    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    out_shape_dyn = tf.stack([tf.shape(x)[0]] + shp3_dyn)
    conv = tf.nn.conv2d_transpose(
        x, W, out_shape_dyn, stride4d, padding=padding, data_format=data_format)
    conv.set_shape(tf.TensorShape([None] + shp3_static))
    return nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
