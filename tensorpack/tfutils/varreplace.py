#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: varreplace.py
# Credit: Qinyao He

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from contextlib import contextmanager

__all__ = ['replace_get_variable']


@contextmanager
def replace_get_variable(fn):
    old_getv = tf.get_variable
    old_vars_getv = variable_scope.get_variable

    tf.get_variable = fn
    variable_scope.get_variable = fn
    yield
    tf.get_variable = old_getv
    variable_scope.get_variable = old_vars_getv
