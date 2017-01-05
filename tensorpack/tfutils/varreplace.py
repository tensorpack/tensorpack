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
    old_getv = tf.get_variable
    old_vars_getv = variable_scope.get_variable

    tf.get_variable = fn
    variable_scope.get_variable = fn
    yield
    tf.get_variable = old_getv
    variable_scope.get_variable = old_vars_getv


def freeze_get_variable():
    """
    Return a contextmanager, where all variables returned by
    `get_variable` will have no gradients.
    """
    old_get_variable = tf.get_variable

    def fn(name, shape=None, **kwargs):
        v = old_get_variable(name, shape, **kwargs)
        return tf.stop_gradient(v)
    return replace_get_variable(fn)
