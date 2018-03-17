#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: shapes.py

import tensorflow as tf
from .common import layer_register

__all__ = ['ConcatWith']


@layer_register(use_scope=None)
def ConcatWith(x, tensor, dim):
    """
    A wrapper around ``tf.concat`` to cooperate with :class:`LinearWrap`.

    Args:
        x (tf.Tensor): input
        tensor (list[tf.Tensor]): a tensor or list of tensors to concatenate with x.
            x will be at the beginning
        dim (int): the dimension along which to concatenate

    Returns:
        tf.Tensor: ``tf.concat([x] + tensor, dim)``
    """
    if type(tensor) != list:
        tensor = [tensor]
    return tf.concat([x] + tensor, dim)
