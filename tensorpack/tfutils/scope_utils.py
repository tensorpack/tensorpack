#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: scope_utils.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import functools
from contextlib import contextmanager

from ..utils.argtools import graph_memoized

__all__ = ['auto_reuse_variable_scope', 'cached_name_scope', 'under_name_scope']


def auto_reuse_variable_scope(func):
    """
    A decorator which automatically reuses the current variable scope if the
    function has been called with the same variable scope before.

    Examples:

    .. code-block:: python

        @auto_reuse_variable_scope
        def myfunc(x):
            return tf.layers.conv2d(x, 128, 3)

        myfunc(x1)  # will inherit parent scope reuse
        myfunc(x2)  # will reuse
        with tf.variable_scope('newscope'):
            myfunc(x3)  # will inherit parent scope reuse
            myfunc(x4)  # will reuse
    """
    used_scope = set()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        scope = tf.get_variable_scope()
        h = hash((tf.get_default_graph(), scope.name))
        # print("Entering " + scope.name + " reuse: " + str(h in used_scope))
        if h in used_scope:
            with tf.variable_scope(scope, reuse=True):
                return func(*args, **kwargs)
        else:
            used_scope.add(h)
            return func(*args, **kwargs)

    return wrapper


def under_name_scope():
    """
    Returns:
        A decorator which makes the function happen under a name scope,
        which is named by the function itself.

    Examples:

    .. code-block:: python

        @under_name_scope()
        def rms(x):
            return tf.sqrt(  # will be under name scope 'rms'
                tf.reduce_mean(tf.square(x)))

    Todo:
        Add a reuse option.
    """

    def _impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__name__
            with tf.name_scope(name):
                return func(*args, **kwargs)
        return wrapper
    return _impl


@graph_memoized
def _get_cached_ns(name):
    with tf.name_scope(None):
        with tf.name_scope(name) as scope:
            return scope


@contextmanager
def cached_name_scope(name):
    """
    Return a context which either opens and caches a new top-level name scope,
    or reenter an existing one.

    Note:
        The name scope will always be top-level. It will not be nested under
        any existing name scope of the caller.
    """
    ns = _get_cached_ns(name)
    with tf.name_scope(ns):
        yield ns
