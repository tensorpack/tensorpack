#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: scope_utils.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import functools
from contextlib import contextmanager

from ..utils.argtools import graph_memoized

__all__ = ['auto_reuse_variable_scope', 'cached_name_scope']


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
            with tf.variable_scope(scope, reuse=True):
                return func(*args, **kwargs)
        else:
            used_scope.add(h)
            return func(*args, **kwargs)

    return wrapper


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
