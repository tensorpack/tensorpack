# -*- coding: utf-8 -*-
# File: batch_norm.py


import re
from ..compat import tfv1 as tf  # this should be avoided first in model code
from tensorflow.python.training import moving_averages

from ..tfutils.collection import backup_collection, restore_collection
from ..tfutils.common import get_tf_version_tuple
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ..utils.argtools import get_data_format, log_once
from ..utils.develop import log_deprecated
from .common import VariableHolder, layer_register
from .tflayer import convert_to_tflayer_args, rename_get_variable
from .utils import disable_autograph

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

    if get_current_tower_context().is_main_training_tower:
        for v in [moving_mean, moving_var]:
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
    return beta, gamma, moving_mean, moving_var


def internal_update_bn_ema(xn, batch_mean, batch_var,
                           moving_mean, moving_var, decay):
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')

    # When sync_statistics is True, always enable internal_update.
    # Otherwise the update ops (only executed on main tower)
    # will hang when some BatchNorm layers are unused (https://github.com/tensorpack/tensorpack/issues/1078)
    with tf.control_dependencies([update_op1, update_op2]):
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
@disable_autograph()
def BatchNorm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_last',
              ema_update='default',
              sync_statistics=None,
              internal_update=None):
    """
    A more powerful version of `tf.layers.batch_normalization`. It differs from
    the offical one in the following aspects:

    1. Accepts an alternative ``data_format`` option when ``axis`` is None. For 2D input, this argument will be ignored.
    2. Default value for ``momentum`` and ``epsilon`` is different.
    3. Default value for ``training`` is automatically obtained from tensorpack's ``TowerContext``.
       User-provided value can overwrite this behavior.
    4. Support the ``ema_update`` option, which covers broader use cases than the standard EMA update.
    5. Support the ``sync_statistics`` option, which implements "SyncBN" and is very useful in small-batch models.

    Args:
        training (bool): if True, use per-batch statistics to normalize. Otherwise, use stored EMA
            to normalize. By default, it is equal to `get_current_tower_context().is_training`.
            This is not a good argument name, but it is what the Tensorflow layer uses.
        ema_update (str): Only effective when ``training=True``. It has the following options:

          * "default": same as "collection". Because this is the default behavior in TensorFlow.
          * "skip": do not update EMA. This can be useful when you reuse a batch norm layer in several places
            but do not want them to all update your EMA.
          * "collection": Add EMA update ops to collection `tf.GraphKeys.UPDATE_OPS`.
            The ops in the collection will be run automatically by the callback :class:`RunUpdateOps`, along with
            your training iterations. This can waste compute if your training iterations do not always depend
            on the BatchNorm layer.
          * "internal": EMA is updated inside this layer itself by control dependencies.
            In standard scenarios, it has similar speed to "collection". But it has some more benefits:

            1. BatchNorm is used inside dynamic control flow.
               The collection-based update does not support dynamic control flows.
            2. BatchNorm layer is sometimes unused (e.g., in GANs you have two networks to train alternatively).
               Putting all update ops into a single collection will waste a lot of compute.
            3. Other part of the model relies on the "updated" EMA. The collection-based method does not update
               EMA immediately.
            4. It has less chance to cause TensorFlow bugs in a graph with complicated control flow.

            Therefore this option is preferred over TensorFlow default.
            Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/14699
        sync_statistics (str or None): one of None, "nccl", or "horovod". It determines how to compute the
          "per-batch statistics" when ``training==True``.

          * None: it uses statistics of the input tensor to normalize during training.
            This is the standard way BatchNorm was implemented in most frameworks.

          * "nccl": this layer must be used under tensorpack's multi-GPU trainers.
            It uses the aggregated statistics of the whole batch (across all GPUs) to normalize.

          * "horovod": this layer must be used under tensorpack's :class:`HorovodTrainer`.
            It uses the aggregated statistics of the whole batch (across all MPI ranks) to normalize.
            Note that on single machine this is significantly slower than the "nccl" implementation.

          When not None, each GPU computes its own E[x] and E[x^2],
          which are then averaged among all GPUs to compute global mean & variance.
          Therefore each GPU needs to have the same batch size.

          The synchronization is based on the current variable scope + the name of the layer
          (`BatchNorm('name', input)`). Therefore, you need to make sure that:

          1. The BatchNorm layer on different GPUs needs to have the same name, so that
             statistics can be synchronized. If names do not match, this layer will hang.
          2. A BatchNorm layer cannot be reused within one tower.
          3. A BatchNorm layer needs to be executed for the same number of times by all GPUs.
             If different GPUs execute one BatchNorm layer for different number of times
             (e.g., if some GPUs do not execute it), this layer may hang.

          This option is also known as "SyncBN" or "Cross-GPU BatchNorm" as mentioned in:
          `MegDet: A Large Mini-Batch Object Detector <https://arxiv.org/abs/1711.07240>`_.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/18222.

          When `sync_statistics` is enabled, `ema_update` is set to "internal" automatically.
          This is to avoid running `UPDATE_OPS`, which requires synchronization.

        internal_update: deprecated option. Don't use.

    Variable Names:

    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.

    Note:
        This layer is more flexible than the standard "BatchNorm" layer and provides more features:

        1. No matter whether you're doing training or not, you can set the ``training`` argument
           to use batch statistics or EMA statistics.
           i.e., you can use batch statistics during inference, or use EMA statistics during training.
           Using EMA statistics in training is useful when you load a pre-trained BN and
           don't want to update it.
        2. As long as `training=True`, `sync_statistics` and `ema_update` option will take effect.
    """
    # parse training/ctx
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    training = bool(training)

    # parse shapes
    data_format = get_data_format(data_format, keras_mode=False)
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4], ndims
    if sync_statistics is not None:
        sync_statistics = sync_statistics.lower()
    assert sync_statistics in [None, 'nccl', 'horovod'], sync_statistics

    assert ema_update in ["default", "collection", "internal", "skip"]
    if internal_update is not None:
        log_deprecated("BatchNorm(internal_update=)", "Use ema_update='internal' instead!", "2020-01-01")
        assert ema_update == 'default', \
            "Do not use internal_update and ema_update together! internal_update is deprecated"
        ema_update = "internal" if internal_update else "collection"
    if ema_update == "default":
        ema_update = "collection"
    # Logic:
    # 1. EMA update is possible only when we compute batch statistics (training=True)
    # 2. We know that in training, non-main training tower does not need EMA
    #    update (unless you need, e.g., inference during training on all towers)
    #    We don't know about what to do in prediction context, so be conservative and do the update.
    # 3. User can explicit disable update by "skip".
    do_ema_update = training and \
        (ctx.is_main_training_tower or not ctx.is_training) \
        and (ema_update != "skip")

    if axis is None:
        if ndims == 2:
            axis = 1
        else:
            axis = 1 if data_format == 'NCHW' else 3
    assert axis in [1, 3], axis
    num_chan = shape[axis]

    TF_version = get_tf_version_tuple()

    freeze_bn_backward = not training and ctx.is_training
    if freeze_bn_backward:
        assert TF_version >= (1, 4), \
            "Fine tuning a BatchNorm model with fixed statistics needs TF>=1.4!"
        if ctx.is_main_training_tower:  # only warn in first tower
            log_once("Some BatchNorm layer uses moving_mean/moving_variance in training.", func='warn')
        # Using moving_mean/moving_variance in training, which means we
        # loaded a pre-trained BN and only fine-tuning the affine part.

    do_sync_bn = (sync_statistics is not None) and training

    if not do_sync_bn:
        # Use the builtin layer for anything except for sync-bn
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
            use_fp16 = inputs.dtype == tf.float16
            if use_fp16:
                # non-fused does not support fp16; fused does not support all layouts.
                # we made our best guess here
                tf_args['fused'] = True
            layer = tf.layers.BatchNormalization(**tf_args)
            xn = layer.apply(inputs, training=training, scope=tf.get_variable_scope())

        # Add EMA variables to the correct collection
        if ctx.is_main_training_tower:
            for v in layer.non_trainable_variables:
                if isinstance(v, tf.Variable):
                    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)

        if not do_ema_update:
            restore_collection(coll_bk)
        if do_ema_update and ema_update == "internal":
            # Implement "internal" update.
            restore_collection(coll_bk)
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
            num_dev = ctx.total
            if num_dev == 1:
                logger.warn("BatchNorm(sync_statistics='nccl') is used with only one tower!")
            else:
                assert TF_version >= (1, 10), \
                    "Cross-GPU BatchNorm is only supported in TF>=1.10 ." \
                    "Upgrade TF or apply this patch manually: https://github.com/tensorflow/tensorflow/pull/20360"

                if TF_version <= (1, 12):
                    try:
                        from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so  # deprecated
                    except Exception:
                        pass
                    else:
                        _validate_and_load_nccl_so()
                    from tensorflow.contrib.nccl.ops import gen_nccl_ops  # deprecated
                else:
                    from tensorflow.python.ops import gen_nccl_ops
                shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
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
            import horovod.tensorflow as hvd
            if hvd.size() == 1:
                logger.warn("BatchNorm(sync_statistics='horovod') is used with only one process!")
            else:
                import horovod
                hvd_version = tuple(map(int, horovod.__version__.split('.')[:3]))
                assert hvd_version >= (0, 13, 6), "sync_statistics=horovod needs horovod>=0.13.6 !"

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

        if do_ema_update:
            ret = internal_update_bn_ema(
                xn, batch_mean_vec, batch_var_vec, moving_mean, moving_var, momentum)
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
                tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
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
