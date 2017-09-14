#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: varreplace.py
# Credit: Qinyao He

import tensorflow as tf
from contextlib import contextmanager

from ..utils.develop import deprecated

__all__ = ['custom_getter_scope', 'replace_get_variable',
           'freeze_variables', 'freeze_get_variable', 'remap_get_variable',
           'remap_variables']


@contextmanager
def custom_getter_scope(custom_getter):
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope, custom_getter=custom_getter):
        yield


@deprecated("Use custom_getter_scope instead.", "2017-11-06")
def replace_get_variable(fn):
    """
    Args:
        fn: a function compatible with ``tf.get_variable``.
    Returns:
        a context with a custom getter
    """
    def getter(_, *args, **kwargs):
        return fn(*args, **kwargs)
    return custom_getter_scope(getter)


def remap_variables(fn):
    """
    Use fn to map the output of any variable getter.

    Args:
        fn (tf.Variable -> tf.Tensor)

    Returns:
        a context where all the variables will be mapped by fn.

    Example:
        .. code-block:: python

            with varreplace.remap_variables(lambda var: quantize(var)):
                x = FullyConnected('fc', x, 1000)   # fc/{W,b} will be quantized
    """
    def custom_getter(getter, *args, **kwargs):
        v = getter(*args, **kwargs)
        return fn(v)
    return custom_getter_scope(custom_getter)


def freeze_variables():
    """
    Return a context, where all trainable variables (reused or not) returned by
    ``get_variable`` will have no gradients (they will be wrapped by ``tf.stop_gradient``).
    But they will still be in ``TRAINABLE_VARIABLES`` collections so they will get
    saved correctly. This is useful to fix certain variables for fine-tuning.

    Example:
        .. code-block:: python

            with varreplace.freeze_variable():
                x = FullyConnected('fc', x, 1000)   # fc/* will not be trained
    """
    def custom_getter(getter, *args, **kwargs):
        v = getter(*args, **kwargs)
        if kwargs.pop('trainable', True):
            v = tf.stop_gradient(v)
        return v
    return custom_getter_scope(custom_getter)


@deprecated("Renamed to remap_variables", "2017-11-06")
def remap_get_variable():
    return remap_variables()


@deprecated("Renamed to freeze_variables", "2017-11-06")
def freeze_get_variable():
    return freeze_variables()
