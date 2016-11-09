#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: fc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import math

from ._common import layer_register
from ..tfutils import symbolic_functions as symbf

__all__ = ['FullyConnected']

@layer_register()
def FullyConnected(x, out_dim,
                   W_init=None, b_init=None,
                   nl=None, use_bias=True):
    """
    Fully-Connected layer.

    :param input: a tensor to be flattened except the first dimension.
    :param out_dim: output dimension
    :param W_init: initializer for W. default to `xavier_initializer_conv2d`.
    :param b_init: initializer for b. default to zero initializer.
    :param nl: nonlinearity
    :param use_bias: whether to use bias. a boolean default to True
    :returns: a 2D tensor
    """
    x = symbf.batch_flatten(x)
    in_dim = x.get_shape().as_list()[1]

    if W_init is None:
        #W_init = tf.truncated_normal_initializer(stddev=1 / math.sqrt(float(in_dim)))
        W_init = tf.uniform_unit_scaling_initializer(factor=1.43)
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', [in_dim, out_dim], initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_dim], initializer=b_init)
    prod = tf.nn.xw_plus_b(x, W, b) if use_bias else tf.matmul(x, W)
    if nl is None:
        logger.warn("[DEPRECATED] Default ReLU nonlinearity for Conv2D and FullyConnected will be deprecated. Please use argscope instead.")
        nl = tf.nn.relu
    return nl(prod, name='output')
