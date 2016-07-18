#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: softmax.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from ._common import layer_register

__all__ = ['SoftMax']

@layer_register()
def SoftMax(x, use_temperature=False, temperature_init=1.0):
    """
    A SoftMax layer (no linear projection) with optional temperature
    :param x: a 2D tensor
    """
    if use_temperature:
        t = tf.get_variable('invtemp', [],
                initializer=tf.constant_initializer(1.0 / float(temperature_init)))
        x = x * t
    return tf.nn.softmax(x, name='output')
