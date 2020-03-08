# -*- coding: utf-8 -*-
# File: layer_norm.py


from ..compat import tfv1 as tf  # this should be avoided first in model code

from ..utils.argtools import get_data_format
from ..utils.develop import log_deprecated
from .common import VariableHolder, layer_register
from .tflayer import convert_to_tflayer_args

__all__ = ['LayerNorm', 'InstanceNorm']


@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'use_bias': 'center',
        'use_scale': 'scale',
        'gamma_init': 'gamma_initializer',
    })
def LayerNorm(
        x, epsilon=1e-5, *,
        center=True, scale=True,
        gamma_initializer=tf.ones_initializer(),
        data_format='channels_last'):
    """
    Layer Normalization layer, as described in the paper:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`_.

    Args:
        x (tf.Tensor): a 4D or 2D tensor. When 4D, the layout should match data_format.
        epsilon (float): epsilon to avoid divide-by-zero.
        center, scale (bool): whether to use the extra affine transformation or not.
    """
    data_format = get_data_format(data_format, keras_mode=False)
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]

    mean, var = tf.nn.moments(x, list(range(1, len(shape))), keep_dims=True)

    if data_format == 'NCHW':
        chan = shape[1]
        new_shape = [1, chan, 1, 1]
    else:
        chan = shape[-1]
        new_shape = [1, 1, 1, chan]
    if ndims == 2:
        new_shape = [1, chan]

    if center:
        beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
    else:
        beta = tf.zeros([1] * ndims, name='beta')
    if scale:
        gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
        gamma = tf.reshape(gamma, new_shape)
    else:
        gamma = tf.ones([1] * ndims, name='gamma')

    ret = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')

    vh = ret.variables = VariableHolder()
    if scale:
        vh.gamma = gamma
    if center:
        vh.beta = beta
    return ret


@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'gamma_init': 'gamma_initializer',
    })
def InstanceNorm(x, epsilon=1e-5, *, center=True, scale=True,
                 gamma_initializer=tf.ones_initializer(),
                 data_format='channels_last', use_affine=None):
    """
    Instance Normalization, as in the paper:
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`_.

    Args:
        x (tf.Tensor): a 4D tensor.
        epsilon (float): avoid divide-by-zero
        center, scale (bool): whether to use the extra affine transformation or not.
        use_affine: deprecated. Don't use.
    """
    data_format = get_data_format(data_format, keras_mode=False)
    shape = x.get_shape().as_list()
    assert len(shape) == 4, "Input of InstanceNorm has to be 4D!"

    if use_affine is not None:
        log_deprecated("InstanceNorm(use_affine=)", "Use center= or scale= instead!", "2020-06-01")
        center = scale = use_affine

    if data_format == 'NHWC':
        axis = [1, 2]
        ch = shape[3]
        new_shape = [1, 1, 1, ch]
    else:
        axis = [2, 3]
        ch = shape[1]
        new_shape = [1, ch, 1, 1]
    assert ch is not None, "Input of InstanceNorm require known channel!"

    mean, var = tf.nn.moments(x, axis, keep_dims=True)

    if center:
        beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
    else:
        beta = tf.zeros([1, 1, 1, 1], name='beta', dtype=x.dtype)
    if scale:
        gamma = tf.get_variable('gamma', [ch], initializer=gamma_initializer)
        gamma = tf.reshape(gamma, new_shape)
    else:
        gamma = tf.ones([1, 1, 1, 1], name='gamma', dtype=x.dtype)
    ret = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')

    vh = ret.variables = VariableHolder()
    if scale:
        vh.gamma = gamma
    if center:
        vh.beta = beta
    return ret
