#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: fc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import math

from ._common import layer_register
from ..utils.symbolic_functions import *

__all__ = ['FullyConnected']

@layer_register(summary_activation=True)
def FullyConnected(x, out_dim, W_init=None, b_init=None, nl=tf.nn.relu):
    x = batch_flatten(x)
    in_dim = x.get_shape().as_list()[1]

    if W_init is None:
        W_init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(in_dim)))
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', [in_dim, out_dim], initializer=W_init)
    b = tf.get_variable('b', [out_dim], initializer=b_init)
    return nl(tf.matmul(x, W) + b)
