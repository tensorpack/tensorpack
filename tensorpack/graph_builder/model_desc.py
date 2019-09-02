# -*- coding: utf-8 -*-
# File: model_desc.py


from collections import namedtuple
import tensorflow as tf

from ..utils.develop import log_deprecated
from ..train.model_desc import ModelDesc, ModelDescBase  # kept for BC # noqa


__all__ = ['InputDesc']


class InputDesc(
        namedtuple('InputDescTuple', ['type', 'shape', 'name'])):
    """
    An equivalent of `tf.TensorSpec`.

    History: this concept is used to represent metadata about the inputs,
    which can be later used to build placeholders or other types of input source.
    It is introduced much much earlier than the equivalent concept `tf.TensorSpec`
    was introduced in TensorFlow.
    Therefore, we now switched to use `tf.TensorSpec`, but keep this here for compatibility reasons.
    """

    def __new__(cls, type, shape, name):
        """
        Args:
            type (tf.DType):
            shape (tuple):
            name (str):
        """
        log_deprecated("InputDesc", "Use tf.TensorSpec instead!", "2020-03-01")
        assert isinstance(type, tf.DType), type
        return tf.TensorSpec(shape=shape, dtype=type, name=name)
