#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: fc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from .common import layer_register
from ..tfutils import symbolic_functions as symbf

__all__ = ['FullyConnected']


@layer_register()
def FullyConnected(x, out_dim,
                   W_init=None, b_init=None,
                   nl=tf.identity, use_bias=True):
    """
    Fully-Connected layer. Takes a N>1D tensor and returns a 2D tensor.

    Args:
        x (tf.Tensor): a tensor to be flattened except for the first dimension.
        out_dim (int): output dimension
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor: a NC tensor named ``output``.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    x = symbf.batch_flatten(x)
    in_dim = x.get_shape().as_list()[1]

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', [in_dim, out_dim], initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_dim], initializer=b_init)
    prod = tf.nn.xw_plus_b(x, W, b) if use_bias else tf.matmul(x, W)
    return nl(prod, name='output')
