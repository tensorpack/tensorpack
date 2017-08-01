#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: scope_utils.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import six
if six.PY2:
    import functools32 as functools
else:
    import functools

__all__ = ['auto_reuse_variable_scope']


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
