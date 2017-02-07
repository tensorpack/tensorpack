#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: batch_norm.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from .common import layer_register

__all__ = ['BatchNorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


# Deprecated. Only kept for future reference.
@layer_register(log_shape=False)
def BatchNormV1(x, use_local_stat=None, decay=0.9, epsilon=1e-5):
    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]

    n_out = shape[-1]  # channel
    assert n_out is not None
    beta = tf.get_variable('beta', [n_out],
                           initializer=tf.constant_initializer())
    gamma = tf.get_variable('gamma', [n_out],
                            initializer=tf.constant_initializer(1.0))

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0], keep_dims=False)
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
    # just to make a clear name.
    batch_mean = tf.identity(batch_mean, 'mean')
    batch_var = tf.identity(batch_var, 'variance')

    emaname = 'EMA'
    ctx = get_current_tower_context()
    if use_local_stat is None:
        use_local_stat = ctx.is_training
    if use_local_stat != ctx.is_training:
        logger.warn("[BatchNorm] use_local_stat != is_training")

    if use_local_stat:
        # training tower
        if ctx.is_training:
            # reuse = tf.get_variable_scope().reuse
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # BatchNorm in reuse scope can be tricky! Moving mean/variance are not reused
                with tf.name_scope(None):  # https://github.com/tensorflow/tensorflow/issues/2740
                    # TODO if reuse=True, try to find and use the existing statistics
                    # how to use multiple tensors to update one EMA? seems impossbile
                    ema = tf.train.ExponentialMovingAverage(decay=decay, name=emaname)
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
                    if ctx.is_main_training_tower:
                        # inside main training tower
                        add_model_variable(ema_mean)
                        add_model_variable(ema_var)
    else:
        # no apply() is called here, no magic vars will get created,
        # no reuse issue will happen
        assert not ctx.is_training
        with tf.name_scope(None):
            ema = tf.train.ExponentialMovingAverage(decay=decay, name=emaname)
            mean_var_name = ema.average_name(batch_mean)
            var_var_name = ema.average_name(batch_var)
            if ctx.is_main_tower:
                # main tower, but needs to use global stat. global stat must be from outside
                # TODO when reuse=True, the desired variable name could
                # actually be different, because a different var is created
                # for different reuse tower
                ema_mean = tf.get_variable('mean/' + emaname, [n_out])
                ema_var = tf.get_variable('variance/' + emaname, [n_out])
            else:
                # use statistics in another tower
                G = tf.get_default_graph()
                ema_mean = ctx.find_tensor_in_main_tower(G, mean_var_name + ':0')
                ema_var = ctx.find_tensor_in_main_tower(G, var_var_name + ':0')

    if use_local_stat:
        batch = tf.cast(tf.shape(x)[0], tf.float32)
        mul = tf.where(tf.equal(batch, 1.0), 1.0, batch / (batch - 1))
        batch_var = batch_var * mul  # use unbiased variance estimator in training

        with tf.control_dependencies([ema_apply_op] if ctx.is_training else []):
            # only apply EMA op if is_training
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, gamma, epsilon, 'output')
    else:
        return tf.nn.batch_normalization(
            x, ema_mean, ema_var, beta, gamma, epsilon, 'output')


@layer_register(log_shape=False)
def BatchNorm(x, use_local_stat=None, decay=0.9, epsilon=1e-5):
    """
    Batch normalization layer, as described in the paper:
    `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariance Shift <http://arxiv.org/abs/1502.03167>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor.
        use_local_stat (bool): whether to use mean/var of the current batch or the moving average.
            Defaults to True in training and False in inference.
        decay (float): decay rate of moving average.
        epsilon (float): epsilon to avoid divide-by-zero.

    Returns:
        tf.Tensor: a tensor named ``output`` with the same shape of x.

    Variable Names:

    * ``beta``: the bias term.
    * ``gamma``: the scale term. Input will be transformed by ``x * gamma + beta``.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.

    Note:
        In multi-tower training, only the first training tower maintains a moving average.
        This is consistent with most frameworks.
    """
    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]  # channel
    assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
    if len(shape) == 2:
        x = tf.reshape(x, [-1, 1, 1, n_out])

    beta = tf.get_variable('beta', [n_out],
                           initializer=tf.constant_initializer())
    gamma = tf.get_variable('gamma', [n_out],
                            initializer=tf.constant_initializer(1.0))
    # x * gamma + beta

    ctx = get_current_tower_context()
    if use_local_stat is None:
        use_local_stat = ctx.is_training
    if use_local_stat != ctx.is_training:
        logger.warn("[BatchNorm] use_local_stat != is_training")

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(), trainable=False)

    if use_local_stat:
        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                                                           epsilon=epsilon, is_training=True)

        # maintain EMA only in the main training tower
        if ctx.is_main_training_tower:
            # TODO a way to use debias in multitower.
            update_op1 = moving_averages.assign_moving_average(
                moving_mean, batch_mean, decay, zero_debias=False,
                name='mean_ema_op')
            update_op2 = moving_averages.assign_moving_average(
                moving_var, batch_var, decay, zero_debias=False,
                name='var_ema_op')
            add_model_variable(moving_mean)
            add_model_variable(moving_var)
    else:
        assert not ctx.is_training, "In training, local statistics has to be used!"
        # TODO do I need to add_model_variable.
        # consider some fixed-param tasks, such as load model and fine tune one layer

        # fused seems slower in inference
        # xn, _, _ = tf.nn.fused_batch_norm(x, gamma, beta,
        #   moving_mean, moving_var,
        #   epsilon=epsilon, is_training=False, name='output')
        xn = tf.nn.batch_normalization(
            x, moving_mean, moving_var, beta, gamma, epsilon)

    if len(shape) == 2:
        xn = tf.squeeze(xn, [1, 2])

    # TODO for other towers, maybe can make it depend some op later
    # TODO update it later (similar to slim) might be faster?
    if ctx.is_main_training_tower:
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')
    else:
        return tf.identity(xn, name='output')
