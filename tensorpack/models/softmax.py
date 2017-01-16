#!/usr/bin/env python
# -*- coding: utf-8 -*- File: softmax.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from .common import layer_register

__all__ = ['SoftMax']


@layer_register()
def SoftMax(x, use_temperature=False, temperature_init=1.0):
    """
    A SoftMax layer (w/o linear projection) with optional temperature, as
    defined in the paper `Distilling the Knowledge in a Neural Network
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        x (tf.Tensor): input of any dimension. Softmax will be performed on
            the last dimension.
        use_temperature (bool): use a learnable temperature or not.
        temperature_init (float): initial value of the temperature.

    Returns:
        tf.Tensor: a tensor of the same shape named ``output``.

    Variable Names:

    * ``invtemp``: 1.0/temperature.
    """
    if use_temperature:
        t = tf.get_variable('invtemp', [],
                            initializer=tf.constant_initializer(1.0 / float(temperature_init)))
        x = x * t
    return tf.nn.softmax(x, name='output')
