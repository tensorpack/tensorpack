# -*- coding: utf-8 -*-
# File: batch_norm.py


import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages
import re
import six

from ..utils import logger
from ..utils.argtools import get_data_format
from ..tfutils.tower import get_current_tower_context
from ..tfutils.common import get_tf_version_tuple
from ..tfutils.collection import backup_collection, restore_collection
from .common import layer_register, VariableHolder
from .tflayer import convert_to_tflayer_args, rename_get_variable

__all__ = ['BatchNorm', 'BatchRenorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


def get_bn_variables(n_out, use_scale, use_bias, beta_init, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=beta_init)
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
def BatchNorm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_last',
              internal_update=False,
              sync_statistics=None):
    """
    Almost equivalent to `tf.layers.batch_normalization`, but different (and more powerful)
    in the following:

    1. Accepts an alternative `data_format` option when `axis` is None. For 2D input, this argument will be ignored.
    2. Default value for `momentum` and `epsilon` is different.
    3. Default value for `training` is automatically obtained from tensorpack's `TowerContext`, but can be overwritten.
    4. Support the `internal_update` option, which enables the use of BatchNorm layer inside conditionals.
    5. Support the `sync_statistics` option, which is very useful in small-batch models.

    Args:
        internal_update (bool): if False, add EMA update ops to
          `tf.GraphKeys.UPDATE_OPS`. If True, update EMA inside the layer by control dependencies.
          They are very similar in speed, but `internal_update=True` can be used
          when you have conditionals in your model, or when you have multiple networks to train.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/14699
        sync_statistics (str or None): one of None, "nccl", or "horovod".

          By default (None), it uses statistics of the input tensor to normalize.
          This is the standard way BatchNorm was done in most frameworks.

          When set to "nccl", this layer must be used under tensorpack's multi-GPU trainers.
          It uses the aggregated statistics of the whole batch (across all GPUs) to normalize.

          When set to "horovod", this layer must be used under tensorpack's :class:`HorovodTrainer`.
          It uses the aggregated statistics of the whole batch (across all MPI ranks) to normalize.
          Note that on single machine this is significantly slower than the "nccl" implementation.

          This implementation averages the per-GPU E[x] and E[x^2] among GPUs to compute
          global mean & variance. Therefore each GPU needs to have the same batch size.

          This option has no effect when not training.

          This option is also known as "Cross-GPU BatchNorm" as mentioned in:
          `MegDet: A Large Mini-Batch Object Detector <https://arxiv.org/abs/1711.07240>`_.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/18222.

    Variable Names:

    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.

    Note:
        Combinations of ``training`` and ``ctx.is_training``:

        * ``training == ctx.is_training``: standard BN, EMA are maintained during training
          and used during inference. This is the default.
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
    if sync_statistics is not None:
        sync_statistics = sync_statistics.lower()
    assert sync_statistics in [None, 'nccl', 'horovod'], sync_statistics

    if axis is None:
        if ndims == 2:
            data_format = 'NHWC'
            axis = 1
        else:
            axis = 1 if data_format == 'NCHW' else 3
    else:
        data_format = 'NCHW' if axis == 1 else 'NHWC'
    num_chan = shape[axis]

    # parse training/ctx
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    training = bool(training)
    TF_version = get_tf_version_tuple()
    freeze_bn_backward = not training and ctx.is_training
    if freeze_bn_backward:
        assert TF_version >= (1, 4), \
            "Fine tuning a BatchNorm model with fixed statistics needs TF>=1.4!"
        if ctx.is_main_training_tower:  # only warn in first tower
            logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
        # Using moving_mean/moving_variance in training, which means we
        # loaded a pre-trained BN and only fine-tuning the affine part.

    if sync_statistics is None or not (training and ctx.is_training):
        coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
        with rename_get_variable(
                {'moving_mean': 'mean/EMA',
                    'moving_variance': 'variance/EMA'}):
            tf_args = dict(
                axis=axis,
                momentum=momentum, epsilon=epsilon,
                center=center, scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                # https://github.com/tensorflow/tensorflow/issues/10857#issuecomment-410185429
                fused=(ndims == 4 and axis in [1, 3] and not freeze_bn_backward),
                _reuse=tf.get_variable_scope().reuse)
            if TF_version >= (1, 5):
                tf_args['virtual_batch_size'] = virtual_batch_size
            else:
                assert virtual_batch_size is None, "Feature not supported in this version of TF!"
            layer = tf.layers.BatchNormalization(**tf_args)
            xn = layer.apply(inputs, training=training, scope=tf.get_variable_scope())

        # maintain EMA only on one GPU is OK, even in replicated mode.
        # because during training, EMA isn't used
        if ctx.is_main_training_tower:
            for v in layer.non_trainable_variables:
                if isinstance(v, tf.Variable):
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
    else:
        red_axis = [0] if ndims == 2 else ([0, 2, 3] if axis == 1 else [0, 1, 2])

        new_shape = None  # don't need to reshape unless ...
        if ndims == 4 and axis == 1:
            new_shape = [1, num_chan, 1, 1]

        batch_mean = tf.reduce_mean(inputs, axis=red_axis)
        batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axis)

        if sync_statistics == 'nccl':
            if six.PY3 and TF_version <= (1, 9) and ctx.is_main_training_tower:
                logger.warn("A bug in TensorFlow<=1.9 will cause cross-GPU BatchNorm to fail. "
                            "Upgrade or apply this patch manually: https://github.com/tensorflow/tensorflow/pull/20360")

            from tensorflow.contrib.nccl.ops import gen_nccl_ops
            shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
            num_dev = ctx.total
            if num_dev == 1:
                logger.warn("BatchNorm(sync_statistics='nccl') is used with only one tower!")
            else:
                batch_mean = gen_nccl_ops.nccl_all_reduce(
                    input=batch_mean,
                    reduction='sum',
                    num_devices=num_dev,
                    shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
                batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                    input=batch_mean_square,
                    reduction='sum',
                    num_devices=num_dev,
                    shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
        elif sync_statistics == 'horovod':
            # Require https://github.com/uber/horovod/pull/331
            import horovod
            hvd_version = tuple(map(int, horovod.__version__.split('.')))
            assert hvd_version >= (0, 13, 6), "sync_statistics needs horovod>=0.13.6 !"
            import horovod.tensorflow as hvd
            if hvd.size() == 1:
                logger.warn("BatchNorm(sync_statistics='horovod') is used with only one process!")
            else:
                batch_mean = hvd.allreduce(batch_mean, average=True)
                batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
        batch_var = batch_mean_square - tf.square(batch_mean)
        batch_mean_vec = batch_mean
        batch_var_vec = batch_var

        beta, gamma, moving_mean, moving_var = get_bn_variables(
            num_chan, scale, center, beta_initializer, gamma_initializer)
        if new_shape is not None:
            batch_mean = tf.reshape(batch_mean, new_shape)
            batch_var = tf.reshape(batch_var, new_shape)
            # Using fused_batch_norm(is_training=False) is actually slightly faster,
            # but hopefully this call will be JITed in the future.
            xn = tf.nn.batch_normalization(
                inputs, batch_mean, batch_var,
                tf.reshape(beta, new_shape),
                tf.reshape(gamma, new_shape), epsilon)
        else:
            xn = tf.nn.batch_normalization(
                inputs, batch_mean, batch_var,
                beta, gamma, epsilon)

        if ctx.is_main_training_tower:
            ret = update_bn_ema(
                xn, batch_mean_vec, batch_var_vec, moving_mean, moving_var,
                momentum, internal_update)
        else:
            ret = tf.identity(xn, name='output')

        vh = ret.variables = VariableHolder(
            moving_mean=moving_mean,
            mean=moving_mean,  # for backward-compatibility
            moving_variance=moving_var,
            variance=moving_var)  # for backward-compatibility
        if scale:
            vh.gamma = gamma
        if center:
            vh.beta = beta
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
            if isinstance(v, tf.Variable):
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
