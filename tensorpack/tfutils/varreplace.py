#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: varreplace.py
# Credit: Qinyao He

import tensorflow as tf
from contextlib import contextmanager

from .common import get_tf_version_number

__all__ = ['freeze_variables', 'remap_variables']


@contextmanager
def custom_getter_scope(custom_getter):
    scope = tf.get_variable_scope()
    if get_tf_version_number() >= 1.5:
        with tf.variable_scope(
                scope, custom_getter=custom_getter,
                auxiliary_name_scope=False):
            yield
    else:
        ns = tf.get_default_graph().get_name_scope()
        with tf.variable_scope(
                scope, custom_getter=custom_getter):
            with tf.name_scope(ns + '/' if ns else ''):
                yield


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


def freeze_variables(stop_gradient=True, skip_collection=False):
    """
    Return a context to freeze variables,
    by wrapping ``tf.get_variable`` with a custom getter.
    It works by either applying ``tf.stop_gradient`` on the variables,
    or by keeping them out of the ``TRAINABLE_VARIABLES`` collection, or
    both.

    Example:
        .. code-block:: python

            with varreplace.freeze_variable(stop_gradient=False, skip_collection=True):
                x = FullyConnected('fc', x, 1000)   # fc/* will not be trained

    Args:
        stop_gradient (bool): if True, variables returned from `get_variable`
            will be wrapped with `tf.stop_gradient` and therefore has no
            gradient when used later. Note that the created variables may
            still have gradient when accessed by other approaches (e.g.
            by name, or by collection).
        skip_collection (bool): if True, do not add the variable to
            ``TRAINABLE_VARIABLES`` collection. As a result they will not be
            trained by default.
    """
    def custom_getter(getter, *args, **kwargs):
        trainable = kwargs.get('trainable', True)
        if skip_collection:
            kwargs['trainable'] = False
        v = getter(*args, **kwargs)
        if trainable and stop_gradient:
            v = tf.stop_gradient(v)
        return v
    return custom_getter_scope(custom_getter)
