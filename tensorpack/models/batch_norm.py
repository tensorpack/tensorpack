#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: batch_norm.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

from ..utils import logger
from ..tfutils.tower import get_current_tower_context
from ..tfutils.collection import backup_collection, restore_collection
from .common import layer_register, VariableHolder

__all__ = ['BatchNorm', 'BatchRenorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


def get_bn_variables(n_out, use_scale, use_bias, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer())
    else:
        beta = tf.zeros([n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([n_out], name='gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(1.0), trainable=False)
    return beta, gamma, moving_mean, moving_var


def update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay):
    # TODO is there a way to use zero_debias in multi-GPU?
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')
    # Only add to model var when we update them
    add_model_variable(moving_mean)
    add_model_variable(moving_var)

    # TODO add an option, and maybe enable it for replica mode?
    # with tf.control_dependencies([update_op1, update_op2]):
    # return tf.identity(xn, name='output')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
    return xn


def reshape_for_bn(param, ndims, chan, data_format):
    if ndims == 2:
        shape = [1, chan]
    else:
        shape = [1, 1, 1, chan] if data_format == 'NHWC' else [1, chan, 1, 1]
    return tf.reshape(param, shape)


@layer_register()
def BatchNorm(x, use_local_stat=None, decay=0.9, epsilon=1e-5,
              use_scale=True, use_bias=True,
              gamma_init=tf.constant_initializer(1.0), data_format='NHWC'):
    """
    Batch Normalization layer, as described in the paper:
    `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariance Shift <http://arxiv.org/abs/1502.03167>`_.

    Args:
        x (tf.Tensor): a 4D or 2D tensor. When 4D, the layout should match data_format.
        use_local_stat (bool): whether to use mean/var of the current batch or the moving average.
            Defaults to True in training and False in inference.
        decay (float): decay rate of moving average.
        epsilon (float): epsilon to avoid divide-by-zero.
        use_scale, use_bias (bool): whether to use the extra affine transformation or not.
        gamma_init: initializer for gamma (the scale).

    Returns:
        tf.Tensor: a tensor named ``output`` with the same shape of x.

    Variable Names:

    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default. Input will be transformed by ``x * gamma + beta``.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.

    Note:
        1. About multi-GPU training: moving averages across GPUs are not aggregated.
           Batch statistics are computed indepdently.  This is consistent with most frameworks.
        2. Combinations of ``use_local_stat`` and ``ctx.is_training``:
            * ``use_local_stat == is_training``: standard BN, EMA are
                maintained during training and used during inference.
            * ``use_local_stat and not is_training``: still use local (batch)
                statistics in inference.
            * ``not use_local_stat and is_training``: use EMA to normalize in
                training. This is useful when you load a pre-trained BN and
                don't want to fine tune the EMA. EMA will not be updated in
                this case.
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    if ndims == 2:
        data_format = 'NHWC'
    if data_format == 'NCHW':
        n_out = shape[1]
    else:
        n_out = shape[-1]  # channel
    assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
    beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_scale, use_bias, gamma_init)

    ctx = get_current_tower_context()
    if use_local_stat is None:
        use_local_stat = ctx.is_training
    use_local_stat = bool(use_local_stat)

    if use_local_stat:
        if ndims == 2:
            x = tf.reshape(x, [-1, 1, 1, n_out])    # fused_bn only takes 4D input
            # fused_bn has error using NCHW? (see #190)

        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
            x, gamma, beta, epsilon=epsilon,
            is_training=True, data_format=data_format)

        if ndims == 2:
            xn = tf.squeeze(xn, [1, 2])
    else:
        if ctx.is_training:
            if ctx.index == 0:  # only warn in first tower
                logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
            # Using moving_mean/moving_variance in training, which means we
            # loaded a pre-trained BN and only fine-tuning the affine part.
            xn, _, _ = tf.nn.fused_batch_norm(
                x, gamma, beta,
                mean=moving_mean, variance=moving_var, epsilon=epsilon,
                data_format=data_format, is_training=False)
        else:
            # non-fused op is faster for inference  # TODO test if this is still true
            if ndims == 4 and data_format == 'NCHW':
                [g, b, mm, mv] = [reshape_for_bn(_, ndims, n_out, data_format)
                                  for _ in [gamma, beta, moving_mean, moving_var]]
                xn = tf.nn.batch_normalization(x, mm, mv, b, g, epsilon)
            else:
                # avoid the reshape if possible (when channel is the last dimension)
                xn = tf.nn.batch_normalization(
                    x, moving_mean, moving_var, beta, gamma, epsilon)

    # maintain EMA only on one GPU is OK, even in replicated mode.
    # because training time doesn't use EMA
    if ctx.is_main_training_tower and use_local_stat:
        ret = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay)
    else:
        ret = tf.identity(xn, name='output')

    vh = ret.variables = VariableHolder(mean=moving_mean, variance=moving_var)
    if use_scale:
        vh.gamma = gamma
    if use_bias:
        vh.beta = beta
    return ret


@layer_register()
def BatchRenorm(x, rmax, dmax, decay=0.9, epsilon=1e-5,
                use_scale=True, use_bias=True, data_format='NHWC'):
    """
    Batch Renormalization layer, as described in the paper:
    `Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models
    <https://arxiv.org/abs/1702.03275>`_.
    This implementation is a wrapper around `tf.layers.batch_normalization`.

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
    * ``moving_mean, renorm_mean, renorm_mean_weight``: See TF documentation.
    * ``moving_variance, renorm_stddev, renorm_stddev_weight``: See TF documentation.
    """

    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    if ndims == 2:
        data_format = 'NHWC'    # error using NCHW? (see #190)
        x = tf.reshape(x, [-1, 1, 1, shape[1]])
    if data_format == 'NCHW':
        n_out = shape[1]
    else:
        n_out = shape[-1]  # channel
    assert n_out is not None, "Input to BatchRenorm cannot have unknown channels!"

    ctx = get_current_tower_context()
    coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
    layer = tf.layers.BatchNormalization(
        axis=1 if data_format == 'NCHW' else 3,
        momentum=decay, epsilon=epsilon,
        center=use_bias, scale=use_scale,
        renorm=True,
        renorm_clipping={
            'rmin': 1.0 / rmax,
            'rmax': rmax,
            'dmax': dmax},
        renorm_momentum=0.99,
        fused=False)
    xn = layer.apply(x, training=ctx.is_training, scope=tf.get_variable_scope())

    if ctx.has_own_variables:
        # Only apply update in this case.
        # Add these EMA to model_variables so that they will be synced
        # properly by replicated trainers.
        for v in layer.non_trainable_variables:
            add_model_variable(v)
    else:
        # Don't need update if we are sharing variables from an existing tower
        restore_collection(coll_bk)

    if ndims == 2:
        xn = tf.squeeze(xn, [1, 2])
    ret = tf.identity(xn, name='output')

    # TODO not sure whether to add moving_mean/moving_var to VH now
    vh = ret.variables = VariableHolder()
    if use_scale:
        vh.gamma = layer.gamma
    if use_bias:
        vh.beta = layer.beta
    return ret
