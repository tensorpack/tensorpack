# -*- coding: utf-8 -*-
# File: scope_utils.py


import functools
from contextlib import contextmanager

from ..compat import tfv1 as tf
from ..utils.argtools import graph_memoized
from ..utils import logger
from .common import get_tf_version_tuple

__all__ = ['auto_reuse_variable_scope', 'cached_name_scope', 'under_name_scope']


def auto_reuse_variable_scope(func):
    """
    A decorator which automatically reuses the current variable scope if the
    function has been called with the same variable scope before.

    Example:

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
            if get_tf_version_tuple() >= (1, 5):
                with tf.variable_scope(scope, reuse=True, auxiliary_name_scope=False):
                    return func(*args, **kwargs)
            else:
                ns = tf.get_default_graph().get_name_scope()
                with tf.variable_scope(scope, reuse=True), \
                        tf.name_scope(ns + '/' if ns else ''):
                    return func(*args, **kwargs)
        else:
            used_scope.add(h)
            return func(*args, **kwargs)

    return wrapper


def under_name_scope(name_scope=None):
    """
    Args:
        name_scope(str): the default scope to use. If None, will use the name of the function.

    Returns:
        A decorator which makes the function run under a name scope.
        The name scope is obtained by the following:
        1. The 'name_scope' keyword argument when the decorated function is called.
        2. The 'name_scope' argument of the decorator.
        3. (default) The name of the decorated function itself.

        If the name is taken and cannot be used, a warning will be
        printed in the first case.

    Example:

    .. code-block:: python

        @under_name_scope()
        def rms(x):
            return tf.sqrt(
                tf.reduce_mean(tf.square(x)))

        rms(tensor)  # will be called under name scope 'rms'
        rms(tensor, name_scope='scope')  # will be called under name scope 'scope'


    Todo:
        Add a reuse option.
    """

    def _impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_incorrect_scope = 'name_scope' in kwargs
            scopename = kwargs.pop('name_scope', name_scope)
            if scopename is None:
                scopename = func.__name__

            if warn_incorrect_scope:
                # cached_name_scope will try to reenter the existing scope
                with cached_name_scope(scopename, top_level=False) as scope:
                    scope = scope.strip('/')
                    # but it can still conflict with an existing tensor
                    if not scope.endswith(scopename):
                        logger.warn(""" \
Calling function {} with name_scope='{}', but actual name scope becomes '{}'.  \
The name '{}' might be taken.""".format(func.__name__, scopename, scope.split('/')[-1], scopename))
                    return func(*args, **kwargs)
            else:
                with tf.name_scope(scopename):
                    return func(*args, **kwargs)
        return wrapper
    return _impl


def under_variable_scope():
    """
    Returns:
        A decorator which makes the function happen under a variable scope,
        which is named by the function itself.

    Example:

    .. code-block:: python

        @under_variable_scope()
        def mid_level(x):
            with argscope(Conv2D, kernel_shape=3, nl=BNReLU):
                x = Conv2D('conv1', x, 512, stride=1)
                x = Conv2D('conv2', x, 256, stride=1)
            return x

    """

    def _impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__name__
            with tf.variable_scope(name):
                return func(*args, **kwargs)
        return wrapper
    return _impl


@graph_memoized
def _get_cached_ns(name):
    with tf.name_scope(None):
        with tf.name_scope(name) as scope:
            return scope


@contextmanager
def cached_name_scope(name, top_level=True):
    """
    Return a context which either opens and caches a new name scope,
    or reenter an existing one.

    Args:
        top_level(bool): if True, the name scope will always be top-level.
            It will not be nested under any existing name scope of the caller.
    """
    if not top_level:
        current_ns = tf.get_default_graph().get_name_scope()
        if current_ns:
            name = current_ns + '/' + name
    ns = _get_cached_ns(name)
    with tf.name_scope(ns):
        yield ns
