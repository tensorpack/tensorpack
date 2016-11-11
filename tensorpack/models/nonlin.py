#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: nonlin.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from copy import copy

from ._common import layer_register
from .batch_norm import BatchNorm

__all__ = ['Maxout', 'PReLU', 'LeakyReLU', 'BNReLU']

@layer_register()
def Maxout(x, num_unit):
    """
    Maxout as in `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.

    :param input: a NHWC or NC tensor.
    :param num_unit: a int. must be divisible by C.
    :returns: a NHW(C/num_unit) tensor
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

@layer_register(log_shape=False)
def PReLU(x, init=tf.constant_initializer(0.001), name=None):
    """
    Parameterized relu as in `Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification
    <http://arxiv.org/abs/1502.01852>`_.

    :param input: any tensor.
    :param init: initializer for the p. default to 0.001.
    """
    alpha = tf.get_variable('alpha', [], initializer=init)
    x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    if name is None:
        name = 'output'
    return tf.mul(x, 0.5, name=name)

@layer_register(log_shape=False)
def LeakyReLU(x, alpha, name=None):
    """
    Leaky relu as in `Rectifier Nonlinearities Improve Neural Network Acoustic
    Models
    <http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_.

    :param input: any tensor.
    :param alpha: the negative slope.
    """
    if name is None:
        name = 'output'
    return tf.maximum(x, alpha * x, name=name)
    #alpha = float(alpha)
    #x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    #return tf.mul(x, 0.5, name=name)

def BNReLU(x, name=None):
    x = BatchNorm('bn', x, use_local_stat=None)
    x = tf.nn.relu(x, name=name)
    return x
