#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: nonlin.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from .common import layer_register, VariableHolder
from .batch_norm import BatchNorm

__all__ = ['Maxout', 'PReLU', 'LeakyReLU', 'BNReLU']


@layer_register(use_scope=None)
def Maxout(x, num_unit):
    """
    Maxout as in the paper `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor. Channel has to be known.
        num_unit (int): a int. Must be divisible by C.

    Returns:
        tf.Tensor: of shape NHW(C/num_unit) named ``output``.
    """
    input_shape = x.get_shape().as_list()
    ndim = len(input_shape)
    assert ndim == 4 or ndim == 2
    ch = input_shape[-1]
    assert ch is not None and ch % num_unit == 0
    if ndim == 4:
        x = tf.reshape(x, [-1, input_shape[1], input_shape[2], ch / num_unit, num_unit])
    else:
        x = tf.reshape(x, [-1, ch / num_unit, num_unit])
    return tf.reduce_max(x, ndim, name='output')


@layer_register()
def PReLU(x, init=0.001, name='output'):
    """
    Parameterized ReLU as in the paper `Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification
    <http://arxiv.org/abs/1502.01852>`_.

    Args:
        x (tf.Tensor): input
        init (float): initial value for the learnable slope.
        name (str): name of the output.

    Variable Names:

    * ``alpha``: learnable slope.
    """
    init = tf.constant_initializer(init)
    alpha = tf.get_variable('alpha', [], initializer=init)
    x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    ret = tf.multiply(x, 0.5, name=name)

    ret.variables = VariableHolder(alpha=alpha)
    return ret


@layer_register(use_scope=None)
def LeakyReLU(x, alpha, name='output'):
    """
    Leaky ReLU as in paper `Rectifier Nonlinearities Improve Neural Network Acoustic
    Models
    <http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_.

    Args:
        x (tf.Tensor): input
        alpha (float): the slope.
    """
    return tf.maximum(x, alpha * x, name=name)


@layer_register(use_scope=None)
def BNReLU(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU.
    """
    x = BatchNorm('bn', x)
    x = tf.nn.relu(x, name=name)
    return x
