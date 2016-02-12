#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: batch_norm.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from ._common import layer_register

__all__ = ['BatchNorm']


# http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
# Only work for 4D tensor right now: #804
@layer_register()
def BatchNorm(x, is_training):
    """
    x: BHWC tensor
    is_training: bool
    """
    is_training = bool(is_training)
    shape = x.get_shape().as_list()
    assert len(shape) == 4

    n_out = shape[-1]  # channel
    beta = tf.get_variable('beta', [n_out])
    gamma = tf.get_variable('gamma', [n_out],
                        initializer=tf.constant_initializer(1.0))
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    if is_training:
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = mean_var_with_update()
    else:
        mean, var = ema_mean, ema_var

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-4, True)
    return normed

