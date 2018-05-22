# -*- coding: utf-8 -*-
# File: batch_norm.py


import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable

from ..utils import logger
from ..utils.argtools import get_data_format
from ..tfutils.tower import get_current_tower_context
from ..tfutils.common import get_tf_version_number
from ..tfutils.collection import backup_collection, restore_collection
from .common import layer_register, VariableHolder
from .tflayer import convert_to_tflayer_args, rename_get_variable

__all__ = ['BatchNorm', 'BatchRenorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


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
def BatchNorm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_last',
              internal_update=False):
    """
    Mostly equivalent to `tf.layers.batch_normalization`, but different in
    the following:

    1. Accepts `data_format` when `axis` is None. For 2D input, this argument will be ignored.
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
    # parse shapes
    data_format = get_data_format(data_format, tfmode=False)
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4], ndims

    if axis is None:
        if ndims == 2:
            data_format = 'NHWC'
            axis = 1
        else:
            axis = 1 if data_format == 'NCHW' else 3

    # parse training/ctx
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    training = bool(training)
    TF_version = get_tf_version_number()
    if not training and ctx.is_training:
        assert TF_version >= 1.4, \
            "Fine tuning a BatchNorm model with fixed statistics is only " \
            "supported after https://github.com/tensorflow/tensorflow/pull/12580 "
        if ctx.is_main_training_tower:  # only warn in first tower
            logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
        # Using moving_mean/moving_variance in training, which means we
        # loaded a pre-trained BN and only fine-tuning the affine part.

    coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
    with rename_get_variable(
            {'moving_mean': 'mean/EMA',
             'moving_variance': 'variance/EMA'}):
        if TF_version >= 1.5:
            layer = tf.layers.BatchNormalization(
                axis=axis,
                momentum=momentum, epsilon=epsilon,
                center=center, scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                virtual_batch_size=virtual_batch_size,
                fused=True,
                _reuse=tf.get_variable_scope().reuse
            )
        else:
            assert virtual_batch_size is None, "Feature not supported in this version of TF!"
            layer = tf.layers.BatchNormalization(
                axis=axis,
                momentum=momentum, epsilon=epsilon,
                center=center, scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                fused=True,
                _reuse=tf.get_variable_scope().reuse
            )
        xn = layer.apply(inputs, training=training, scope=tf.get_variable_scope())

    # maintain EMA only on one GPU is OK, even in replicated mode.
    # because training time doesn't use EMA
    if ctx.is_main_training_tower:
        for v in layer.non_trainable_variables:
            add_model_variable(v)
    if not ctx.is_main_training_tower or internal_update:
        restore_collection(coll_bk)

    if training and internal_update:
        assert layer.updates
        with tf.control_dependencies(layer.updates):
            ret = tf.identity(xn, name='output')
    else:
        ret = tf.identity(xn, name='output')

    vh = ret.variables = VariableHolder(
        moving_mean=layer.moving_mean,
        mean=layer.moving_mean,  # for backward-compatibility
        moving_variance=layer.moving_variance,
        variance=layer.moving_variance)  # for backward-compatibility
    if scale:
        vh.gamma = layer.gamma
    if center:
        vh.beta = layer.beta
    return ret


@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'use_bias': 'center',
        'use_scale': 'scale',
        'gamma_init': 'gamma_initializer',
        'decay': 'momentum'
    })
def BatchRenorm(x, rmax, dmax, momentum=0.9, epsilon=1e-5,
                center=True, scale=True, gamma_initializer=None,
                data_format='channels_last'):
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
        data_format = 'channels_first'

    ctx = get_current_tower_context()
    coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
    layer = tf.layers.BatchNormalization(
        axis=1 if data_format == 'channels_first' else 3,
        momentum=momentum, epsilon=epsilon,
        center=center, scale=scale,
        renorm=True,
        renorm_clipping={
            'rmin': 1.0 / rmax,
            'rmax': rmax,
            'dmax': dmax},
        renorm_momentum=0.99,
        gamma_initializer=gamma_initializer,
        fused=False,
        _reuse=tf.get_variable_scope().reuse)
    xn = layer.apply(x, training=ctx.is_training, scope=tf.get_variable_scope())

    if ctx.is_main_training_tower:
        for v in layer.non_trainable_variables:
            add_model_variable(v)
    else:
        # only run UPDATE_OPS in the first tower
        restore_collection(coll_bk)

    if ndims == 2:
        xn = tf.squeeze(xn, [1, 2])
    ret = tf.identity(xn, name='output')

    # TODO not sure whether to add moving_mean/moving_var to VH now
    vh = ret.variables = VariableHolder()
    if scale:
        vh.gamma = layer.gamma
    if center:
        vh.beta = layer.beta
    return ret
