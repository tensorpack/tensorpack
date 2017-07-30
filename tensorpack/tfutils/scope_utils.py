#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: scope_utils.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import six
from .common import get_tf_version_number
from ..utils.develop import deprecated
if six.PY2:
    import functools32 as functools
else:
    import functools

__all__ = ['auto_reuse_variable_scope']


@deprecated("Use tf.get_default_graph().get_name_scope() (available since 1.2.1).")
def get_name_scope_name():
    """
    Returns:
        str: the name of the current name scope, without the ending '/'.
    """
    if get_tf_version_number() > 1.2:
        return tf.get_default_graph().get_name_scope()
    else:
        g = tf.get_default_graph()
        s = "RANDOM_STR_ABCDEFG"
        unique = g.unique_name(s)
        scope = unique[:-len(s)].rstrip('/')
        return scope


def auto_reuse_variable_scope(func):
    """
    A decorator which automatically reuse the current variable scope if the
    function has been called with the same variable scope before.
    """
    used_scope = set()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        scope = tf.get_variable_scope()
        h = hash((tf.get_default_graph(), scope.name))
        # print("Entering " + scope.name + " reuse: " + str(h in used_scope))
        if h in used_scope:
            ns = scope.original_name_scope
            with tf.variable_scope(scope, reuse=True):
                with tf.name_scope(ns):
                    return func(*args, **kwargs)
        else:
            used_scope.add(h)
            return func(*args, **kwargs)

    return wrapper
