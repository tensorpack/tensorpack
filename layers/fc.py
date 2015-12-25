#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: fc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


from ._common import layer_register
import tensorflow as tf
import math

__all__ = ['FullyConnected']

@layer_register()
def FullyConnected(x, out_dim, W_init=None, b_init=None):
    """
    x: matrix of bxn
    """
    in_dim = x.get_shape().as_list()[1]

    if W_init is None:
        W_init = lambda shape: tf.truncated_normal(
            shape, stddev=1.0 / math.sqrt(float(in_dim)))
    if b_init is None:
        b_init = tf.zeros

    W = tf.Variable(W_init([in_dim, out_dim]), name='W')
    b = tf.Variable(b_init([out_dim]), name='b')
    return tf.matmul(x, W) + b
