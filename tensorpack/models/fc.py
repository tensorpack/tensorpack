# -*- coding: utf-8 -*-
# File: fc.py


import tensorflow as tf
import numpy as np

from .common import layer_register, VariableHolder
from .tflayer import convert_to_tflayer_args, rename_get_variable

__all__ = ['FullyConnected']


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['units'],
    name_mapping={'out_dim': 'units'})
def FullyConnected(
        inputs,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Dense`.
    One difference to maintain backward-compatibility:
    Default weight initializer is variance_scaling_initializer(2.0).

    Variable Names:

    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """

    inputs = batch_flatten(inputs)
    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            _reuse=tf.get_variable_scope().reuse)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret
