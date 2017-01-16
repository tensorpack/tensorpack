#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: shapes.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from .common import layer_register

__all__ = ['ConcatWith']


@layer_register(use_scope=False, log_shape=False)
def ConcatWith(x, dim, tensor):
    """
    A wrapper around ``tf.concat`` to cooperate with :class:`LinearWrap`.

    Args:
        x (tf.Tensor): input
        dim (int): the dimension along which to concatenate
        tensor (list[tf.Tensor]): a tensor or list of tensors to concatenate with x.
            x will be at the beginning

    Returns:
        tf.Tensor: ``tf.concat_v2([x] + tensor, dim)``
    """
    if type(tensor) != list:
        tensor = [tensor]
    return tf.concat_v2([x] + tensor, dim)
