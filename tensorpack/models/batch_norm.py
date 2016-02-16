#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: batch_norm.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from ._common import layer_register

__all__ = ['BatchNorm']


# http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
# TF batch_norm only works for 4D tensor right now: #804
@layer_register()
def BatchNorm(x, is_training, gamma_init=1.0):
    """
    Batch normalization layer as described in:
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    http://arxiv.org/abs/1502.03167
    Notes:
    Whole-population mean/variance is calculated by a running-average mean/variance, with decay rate 0.999
    Epsilon for variance is set to 1e-5, as is torch/nn: https://github.com/torch/nn/blob/master/BatchNormalization.lua

    x: BHWC tensor or a vector
    is_training: bool
    """
    EPS = 1e-5
    is_training = bool(is_training)
    shape = x.get_shape().as_list()
    if len(shape) == 2:
        x = tf.reshape(x, [-1, 1, 1, shape[1]])
        shape = x.get_shape().as_list()
    assert len(shape) == 4

    n_out = shape[-1]  # channel
    beta = tf.get_variable('beta', [n_out])
    gamma = tf.get_variable('gamma', [n_out],
                            initializer=tf.constant_initializer(gamma_init))
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    if is_training:
        with tf.control_dependencies([ema_apply_op]):
            mean, var = tf.identity(batch_mean), tf.identity(batch_var)
    else:
        batch = tf.cast(tf.shape(x)[0], tf.float32)
        mean, var = ema_mean, ema_var * batch / (batch - 1) # unbiased variance estimator

    normed = tf.nn.batch_norm_with_global_normalization(
        x, mean, var, beta, gamma, EPS, True)
    return normed

