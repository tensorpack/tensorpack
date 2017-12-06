#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from .common import layer_register, VariableHolder, rename_get_variable
from ..utils.argtools import shape2d, shape4d
from ..utils.develop import log_deprecated

__all__ = ['Conv2D', 'Deconv2D']


@layer_register(log_shape=True)
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
        tf.Tensor named ``output`` with attribute `variables`.

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

    ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret


@layer_register(log_shape=True)
def Deconv2D(x, out_channel, kernel_shape,
             stride, padding='SAME',
             W_init=None, b_init=None,
             nl=tf.identity, use_bias=True,
             data_format='NHWC'):
    """
    2D deconvolution on 4D inputs.

    Args:
        x (tf.Tensor): a tensor of shape NHWC.
            Must have known number of channels, but can have other unknown dimensions.
        out_channel: the output number of channel.
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor: a NHWC tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

    out_shape = out_channel
    if isinstance(out_shape, int):
        out_channel = out_shape
    else:
        log_deprecated("Deconv2D(out_shape=[...])",
                       "Use an integer 'out_channel' instead!", "2017-11-18")
        for k in out_shape:
            if not isinstance(k, int):
                raise ValueError("[Deconv2D] out_shape {} is invalid!".format(k))
        out_channel = out_shape[channel_axis - 1]   # out_shape doesn't have batch

    if W_init is None:
        W_init = tf.contrib.layers.xavier_initializer_conv2d()
    if b_init is None:
        b_init = tf.constant_initializer()

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv2DTranspose(
            out_channel, kernel_shape,
            strides=stride, padding=padding,
            data_format='channels_last' if data_format == 'NHWC' else 'channels_first',
            activation=lambda x: nl(x, name='output'),
            use_bias=use_bias,
            kernel_initializer=W_init,
            bias_initializer=b_init,
            trainable=True)
        ret = layer.apply(x, scope=tf.get_variable_scope())

    # Check that we only supports out_shape = in_shape * stride
    out_shape3 = ret.get_shape().as_list()[1:]
    if not isinstance(out_shape, int):
        assert list(out_shape) == out_shape3, "{} != {}".format(out_shape, out_shape3)

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret
