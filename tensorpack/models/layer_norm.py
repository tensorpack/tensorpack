#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layer_norm.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from .common import layer_register


@layer_register(log_shape=False)
def LayerNorm(x, epsilon=1e-5, use_bias=True, use_scale=True, data_format='NHWC'):
    """
    Layer Normalization layer, as described in the paper:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`_.

    Args:
        x (tf.Tensor): a 4D or 2D tensor. When 4D, the layout should match data_format.
        epsilon (float): epsilon to avoid divide-by-zero.
        use_scale, use_bias (bool): whether to use the extra affine transformation or not.
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]

    mean, var = tf.nn.moments(x, list(range(1, len(shape))), keep_dims=True)

    if data_format == 'NCHW':
        chan = shape[1]
        new_shape = [1, chan, 1, 1]
    else:
        chan = shape[-1]
        new_shape = [1, 1, 1, chan]
    if ndims == 2:
        new_shape = [1, chan]

    if use_bias:
        beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
    else:
        beta = tf.zeros([1] * ndims, name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [chan], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
    else:
        gamma = tf.ones([1] * ndims, name='gamma')

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')
