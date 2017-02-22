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

__all__ = ['BatchNorm', 'BatchRenorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


# XXX This is deprecated. Only kept for future reference.
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
                    # if reuse=True, try to find and use the existing statistics
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
                # when reuse=True, the desired variable name could
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


def get_bn_variables(n_out, use_scale, use_bias):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer())
    else:
        beta = tf.zeros([n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [n_out], initializer=tf.constant_initializer(1.0))
    else:
        gamma = tf.ones([n_out], name='gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(), trainable=False)
    return beta, gamma, moving_mean, moving_var


def update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay):
    # TODO is there a way to use zero_debias in multi-GPU?
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')
    add_model_variable(moving_mean)
    add_model_variable(moving_var)
    with tf.control_dependencies([update_op1, update_op2]):
        return tf.identity(xn, name='output')


@layer_register(log_shape=False)
def BatchNorm(x, use_local_stat=None, decay=0.9, epsilon=1e-5,
              use_scale=True, use_bias=True, data_format='NHWC'):
    """
    Batch Normalization layer, as described in the paper:
    `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariance Shift <http://arxiv.org/abs/1502.03167>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor.
        use_local_stat (bool): whether to use mean/var of the current batch or the moving average.
            Defaults to True in training and False in inference.
        decay (float): decay rate of moving average.
        epsilon (float): epsilon to avoid divide-by-zero.
        use_scale, use_bias (bool): whether to use the extra affine transformation or not.

    Returns:
        tf.Tensor: a tensor named ``output`` with the same shape of x.

    Variable Names:

    * ``beta``: the bias term.
    * ``gamma``: the scale term. Input will be transformed by ``x * gamma + beta``.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.

    Note:
        In multi-GPU training, moving averages across GPUs are not aggregated.
        This is consistent with most frameworks.

        However, all GPUs use the moving averages on the first GPU (instead of
        their own), this is inconsistent with most frameworks (but consistent
        with the official inceptionv3 example).
    """
    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]
    if len(shape) == 2:
        data_format = 'NCHW'
    if data_format == 'NCHW':
        n_out = shape[1]
    else:
        n_out = shape[-1]  # channel
    if len(shape) == 2:
        x = tf.reshape(x, [-1, n_out, 1, 1])
    assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"

    beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_scale, use_bias)

    ctx = get_current_tower_context()
    if use_local_stat is None:
        use_local_stat = ctx.is_training
    if use_local_stat != ctx.is_training:
        # we allow the use of local_stat in testing (only print warnings)
        # because it is useful to certain applications.
        logger.warn("[BatchNorm] use_local_stat != is_training")

    if use_local_stat:
        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
            x, gamma, beta, epsilon=epsilon,
            is_training=True, data_format=data_format)
    else:
        assert not ctx.is_training, "In training, local statistics has to be used!"
        if data_format == 'NCHW':
            # fused is slower in inference, but support NCHW
            xn, _, _ = tf.nn.fused_batch_norm(
                x, gamma, beta,
                moving_mean, moving_var,
                epsilon=epsilon, is_training=False, data_format=data_format)
        else:
            xn = tf.nn.batch_normalization(
                x, moving_mean, moving_var, beta, gamma, epsilon)

    if len(shape) == 2:
        axis = [2, 3] if data_format == 'NCHW' else [1, 2]
        xn = tf.squeeze(xn, axis)

    # maintain EMA only on one GPU.
    if ctx.is_main_training_tower:
        return update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay)
    else:
        return tf.identity(xn, name='output')


@layer_register(log_shape=False)
def BatchRenorm(x, rmax, dmax, decay=0.9, epsilon=1e-5,
                use_scale=True, use_bias=True):
    """
    Batch Renormalization layer, as described in the paper:
    `Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models
    <https://arxiv.org/abs/1702.03275>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor.
        rmax, dmax (tf.Tensor): a scalar tensor, the maximum allowed corrections.
        decay (float): decay rate of moving average.
        epsilon (float): epsilon to avoid divide-by-zero.
        use_scale, use_bias (bool): whether to use the extra affine transformation or not.

    Returns:
        tf.Tensor: a tensor named ``output`` with the same shape of x.

    Variable Names:

    * ``beta``: the bias term.
    * ``gamma``: the scale term. Input will be transformed by ``x * gamma + beta``.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.
    """

    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]
    if len(shape) == 2:
        x = tf.reshape(x, [-1, 1, 1, n_out])
    beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_scale, use_bias)

    ctx = get_current_tower_context()
    use_local_stat = ctx.is_training
    # for BatchRenorm, use_local_stat should always be is_training, unless a
    # different usage comes out in the future.

    if use_local_stat:
        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                                                           epsilon=epsilon, is_training=True)
        inv_sigma = tf.rsqrt(moving_var, 'inv_sigma')
        r = tf.stop_gradient(tf.clip_by_value(
            tf.sqrt(batch_var) * inv_sigma, 1.0 / rmax, rmax))
        d = tf.stop_gradient(tf.clip_by_value(
            (batch_mean - moving_mean) * inv_sigma,
            -dmax, dmax))
        xn = xn * r + d
    else:
        xn = tf.nn.batch_normalization(
            x, moving_mean, moving_var, beta, gamma, epsilon)

    if len(shape) == 2:
        xn = tf.squeeze(xn, [1, 2])
    if ctx.is_main_training_tower:
        return update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay)
    else:
        return tf.identity(xn, name='output')
