# -*- coding: utf-8 -*-
# File: _old_batch_norm.py

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

from ..tfutils.common import get_tf_version_tuple
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ..utils.argtools import get_data_format
from .common import VariableHolder, layer_register
from .tflayer import convert_to_tflayer_args


"""
Old Custom BN Implementation, Kept Here For Future Reference
"""


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


def update_bn_ema(xn, batch_mean, batch_var,
                  moving_mean, moving_var, decay, internal_update):
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')

    if internal_update:
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')
    else:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        return tf.identity(xn, name='output')


@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'use_bias': 'center',
        'use_scale': 'scale',
        'gamma_init': 'gamma_initializer',
        'decay': 'momentum',
        'use_local_stat': 'training'
    })
def BatchNorm(inputs, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              gamma_initializer=tf.ones_initializer(),
              data_format='channels_last',
              internal_update=False):
    """
    Mostly equivalent to `tf.layers.batch_normalization`, but difference in
    the following:
    1. Accepts `data_format` rather than `axis`. For 2D input, this argument will be ignored.
    2. Default value for `momentum` and `epsilon` is different.
    3. Default value for `training` is automatically obtained from `TowerContext`.
    4. Support the `internal_update` option.
    Args:
        internal_update (bool): if False, add EMA update ops to
            `tf.GraphKeys.UPDATE_OPS`. If True, update EMA inside the layer
            by control dependencies.
    Variable Names:
    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default. Input will be transformed by ``x * gamma + beta``.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.
    Note:
        1. About multi-GPU training: moving averages across GPUs are not aggregated.
           Batch statistics are computed independently.  This is consistent with most frameworks.
        2. Combinations of ``training`` and ``ctx.is_training``:
            * ``training == ctx.is_training``: standard BN, EMA are
                maintained during training and used during inference. This is
                the default.
            * ``training and not ctx.is_training``: still use batch statistics in inference.
            * ``not training and ctx.is_training``: use EMA to normalize in
                training. This is useful when you load a pre-trained BN and
                don't want to fine tune the EMA. EMA will not be updated in
                this case.
    """
    data_format = get_data_format(data_format, keras_mode=False)
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    if ndims == 2:
        data_format = 'NHWC'
    if data_format == 'NCHW':
        n_out = shape[1]
    else:
        n_out = shape[-1]  # channel
    assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
    beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, scale, center, gamma_initializer)

    ctx = get_current_tower_context()
    use_local_stat = training
    if use_local_stat is None:
        use_local_stat = ctx.is_training
    use_local_stat = bool(use_local_stat)

    if use_local_stat:
        if ndims == 2:
            inputs = tf.reshape(inputs, [-1, 1, 1, n_out])    # fused_bn only takes 4D input
            # fused_bn has error using NCHW? (see #190)

        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
            inputs, gamma, beta, epsilon=epsilon,
            is_training=True, data_format=data_format)

        if ndims == 2:
            xn = tf.squeeze(xn, [1, 2])
    else:
        if ctx.is_training:
            assert get_tf_version_tuple() >= (1, 4), \
                "Fine tuning a BatchNorm model with fixed statistics is only " \
                "supported after https://github.com/tensorflow/tensorflow/pull/12580 "
            if ctx.is_main_training_tower:  # only warn in first tower
                logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
            # Using moving_mean/moving_variance in training, which means we
            # loaded a pre-trained BN and only fine-tuning the affine part.
            xn, _, _ = tf.nn.fused_batch_norm(
                inputs, gamma, beta,
                mean=moving_mean, variance=moving_var, epsilon=epsilon,
                data_format=data_format, is_training=False)
        else:
            if ndims == 4:
                xn, _, _ = tf.nn.fused_batch_norm(
                    inputs, gamma, beta,
                    mean=moving_mean, variance=moving_var, epsilon=epsilon,
                    data_format=data_format, is_training=False)
            else:
                xn = tf.nn.batch_normalization(
                    inputs, moving_mean, moving_var, beta, gamma, epsilon)

    # maintain EMA only on one GPU is OK, even in replicated mode.
    # because training time doesn't use EMA
    if ctx.is_main_training_tower:
        add_model_variable(moving_mean)
        add_model_variable(moving_var)
    if ctx.is_main_training_tower and use_local_stat:
        ret = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, momentum, internal_update)
    else:
        ret = tf.identity(xn, name='output')

    vh = ret.variables = VariableHolder(mean=moving_mean, variance=moving_var)
    if scale:
        vh.gamma = gamma
    if center:
        vh.beta = beta
    return ret
