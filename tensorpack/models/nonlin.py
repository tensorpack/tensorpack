#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: nonlin.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from copy import copy

from ._common import *
from .batch_norm import BatchNorm

__all__ = ['Maxout', 'PReLU', 'LeakyReLU', 'BNReLU']

@layer_register()
def Maxout(x, num_unit):
    """
    Maxout networks as in `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.

    :param input: a NHWC tensor.
    :param num_unit: a int. must be divisible by C.
    :returns: a NHW(C/num_unit) tensor
    """
    input_shape = x.get_shape().as_list()
    assert len(input_shape) == 4
    ch = input_shape[3]
    assert ch % num_unit == 0
    x = tf.reshape(x, [-1, input_shape[1], input_shape[2], ch / 3, 3])
    return tf.reduce_max(x, 4, name='output')

@layer_register()
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
        return x * 0.5
    else:
        return tf.mul(x, 0.5, name=name)

@layer_register()
def LeakyReLU(x, alpha, name=None):
    """
    Leaky relu as in `Rectifier Nonlinearities Improve Neural Network Acoustic
    Models
    <http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_.

    :param input: any tensor.
    :param alpha: the negative slope.
    """
    alpha = float(alpha)
    x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    if name is None:
        return x * 0.5
    else:
        return tf.mul(x, 0.5, name=name)


def BNReLU(is_training):
    """
    :returns: a activation function that performs BN + ReLU (a too common combination)
    """
    def f(x, name=None):
        with tf.variable_scope('bn'):
            x = BatchNorm.f(x, is_training)
        x = tf.nn.relu(x, name=name)
        return x
    return f
