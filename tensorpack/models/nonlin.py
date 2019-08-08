# -*- coding: utf-8 -*-
# File: nonlin.py


import tensorflow as tf

from ..utils.develop import log_deprecated
from ..compat import tfv1
from .batch_norm import BatchNorm
from .common import VariableHolder, layer_register
from .utils import disable_autograph

__all__ = ['Maxout', 'PReLU', 'BNReLU']


@layer_register(use_scope=None)
def Maxout(x, num_unit):
    """
    Maxout as in the paper `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor. Channel has to be known.
        num_unit (int): a int. Must be divisible by C.

    Returns:
        tf.Tensor: of shape NHW(C/num_unit) named ``output``.
    """
    input_shape = x.get_shape().as_list()
    ndim = len(input_shape)
    assert ndim == 4 or ndim == 2
    ch = input_shape[-1]
    assert ch is not None and ch % num_unit == 0
    if ndim == 4:
        x = tf.reshape(x, [-1, input_shape[1], input_shape[2], ch / num_unit, num_unit])
    else:
        x = tf.reshape(x, [-1, ch / num_unit, num_unit])
    return tf.reduce_max(x, ndim, name='output')


@layer_register()
@disable_autograph()
def PReLU(x, init=0.001, name=None):
    """
    Parameterized ReLU as in the paper `Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification
    <http://arxiv.org/abs/1502.01852>`_.

    Args:
        x (tf.Tensor): input
        init (float): initial value for the learnable slope.
        name (str): deprecated argument. Don't use

    Variable Names:

    * ``alpha``: learnable slope.
    """
    if name is not None:
        log_deprecated("PReLU(name=...)", "The output tensor will be named `output`.")
    init = tfv1.constant_initializer(init)
    alpha = tfv1.get_variable('alpha', [], initializer=init)
    x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    ret = tf.multiply(x, 0.5, name=name or None)

    ret.variables = VariableHolder(alpha=alpha)
    return ret


@layer_register(use_scope=None)
def BNReLU(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU.

    Args:
        x (tf.Tensor): the input
        name: deprecated, don't use.
    """
    if name is not None:
        log_deprecated("BNReLU(name=...)", "The output tensor will be named `output`.")

    x = BatchNorm('bn', x)
    x = tf.nn.relu(x, name=name)
    return x
