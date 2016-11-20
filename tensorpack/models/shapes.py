#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: shapes.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from ._common import layer_register

__all__ = ['ConcatWith']

@layer_register(use_scope=False, log_shape=False)
def ConcatWith(x, dim, tensor):
    """
    A wrapper around `tf.concat` to support `LinearWrap`
    :param x: the input tensor
    :param dim: the dimension along which to concatenate
    :param tensor: a tensor or list of tensor to concatenate with x. x will be
        at the beginning
    :return: tf.concat(dim, [x] + [tensor])
    """
    if type(tensor) != list:
        tensor = [tensor]
    return tf.concat(dim, [x] + tensor)
