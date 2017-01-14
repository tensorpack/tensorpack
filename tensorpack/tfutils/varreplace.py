#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: varreplace.py
# Credit: Qinyao He

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from contextlib import contextmanager

__all__ = ['replace_get_variable', 'freeze_get_variable']

_ORIG_GET_VARIABLE = tf.get_variable


@contextmanager
def replace_get_variable(fn):
    """
    Args:
        fn: a function taking the same arguments as ``tf.get_variable``.
    Returns:
        a context where ``tf.get_variable`` and
        ``variable_scope.get_variable`` are replaced with ``fn``.

    Note that originally ``tf.get_variable ==
    tensorflow.python.ops.variable_scope.get_variable``. But some code such as
    some in `rnn_cell/`, uses the latter one to get variable, therefore both
    need to be replaced.
    """
    old_getv = tf.get_variable
    old_vars_getv = variable_scope.get_variable

    tf.get_variable = fn
    variable_scope.get_variable = fn
    yield
    tf.get_variable = old_getv
    variable_scope.get_variable = old_vars_getv


def freeze_get_variable():
    """
    Return a context, where all variables (reused or not) returned by
    ``get_variable`` will have no gradients (surrounded by ``tf.stop_gradient``).
    But they will still be in ``TRAINABLE_VARIABLES`` collections so they will get
    saved correctly. This is useful to fix certain variables for fine-tuning.

    Example:
        .. code-block:: python

            with varreplace.freeze_get_variable():
                x = FullyConnected('fc', x, 1000)   # fc/* will not be trained
    """
    old_get_variable = tf.get_variable

    def fn(name, shape=None, **kwargs):
        v = old_get_variable(name, shape, **kwargs)
        return tf.stop_gradient(v)
    return replace_get_variable(fn)
