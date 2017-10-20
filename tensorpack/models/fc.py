#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: fc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from .common import layer_register, rename_get_variable, VariableHolder
from ..tfutils import symbolic_functions as symbf

__all__ = ['FullyConnected']


@layer_register(log_shape=True)
def FullyConnected(x, out_dim,
                   W_init=None, b_init=None,
                   nl=tf.identity, use_bias=True):
    """
    Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
    It is an equivalent of `tf.layers.dense` except for naming conventions.

    Args:
        x (tf.Tensor): a tensor to be flattened except for the first dimension.
        out_dim (int): output dimension
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor: a NC tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """
    x = symbf.batch_flatten(x)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Dense(
            out_dim, activation=lambda x: nl(x, name='output'), use_bias=use_bias,
            kernel_initializer=W_init, bias_initializer=b_init,
            trainable=True)
        ret = layer.apply(x, scope=tf.get_variable_scope())

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret
