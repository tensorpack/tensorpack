#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: nonlin.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from copy import copy

from ._common import *

__all__ = ['Maxout', 'PReLU', 'LeakyReLU']

@layer_register()
def Maxout(x, num_unit):
    input_shape = x.get_shape().as_list()
    assert len(input_shape) == 4
    ch = input_shape[3]
    assert ch % num_unit == 0
    x = tf.reshape(x, [-1, input_shape[1], input_shape[2], ch / 3, 3])
    return tf.reduce_max(x, 4, name='output')

@layer_register()
def PReLU(x, init=tf.constant_initializer(0.001), name=None):
    """ allow name to be compatible to other builtin nonlinearity function"""
    alpha = tf.get_variable('alpha', [], initializer=init)
    x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    if name is None:
        return x * 0.5
    else:
        return tf.mul(x, 0.5, name=name)

@layer_register()
def LeakyReLU(x, alpha, name=None):
    alpha = float(alpha)
    x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    if name is None:
        return x * 0.5
    else:
        return tf.mul(x, 0.5, name=name)
