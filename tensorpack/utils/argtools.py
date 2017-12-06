#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: argtools.py


import inspect
import six
from . import logger
if six.PY2:
    import functools32 as functools
else:
    import functools

__all__ = ['map_arg', 'memoized', 'graph_memoized', 'shape2d', 'shape4d',
           'memoized_ignoreargs', 'log_once', 'call_only_once']


def map_arg(**maps):
    """
    Apply a mapping on certains argument before calling the original function.

    Args:
        maps (dict): {argument_name: map_func}
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if six.PY2:
                argmap = inspect.getcallargs(func, *args, **kwargs)
            else:
                # getcallargs was deprecated since 3.5
                sig = inspect.signature(func)
                argmap = sig.bind_partial(*args, **kwargs).arguments
            for k, map_func in six.iteritems(maps):
                if k in argmap:
                    argmap[k] = map_func(argmap[k])
            return func(**argmap)
        return wrapper
    return deco


memoized = functools.lru_cache(maxsize=None)
""" Alias to :func:`functools.lru_cache` """


def graph_memoized(func):
    """
    Like memoized, but keep one cache per default graph.
    """
    import tensorflow as tf
    GRAPH_ARG_NAME = '__IMPOSSIBLE_NAME_FOR_YOU__'

    @memoized
    def func_with_graph_arg(*args, **kwargs):
        kwargs.pop(GRAPH_ARG_NAME)
        return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert GRAPH_ARG_NAME not in kwargs, "No Way!!"
        graph = tf.get_default_graph()
        kwargs[GRAPH_ARG_NAME] = graph
        return func_with_graph_arg(*args, **kwargs)
    return wrapper


_MEMOIZED_NOARGS = {}


def memoized_ignoreargs(func):
    """
    A decorator. It performs memoization ignoring the arguments used to call
    the function.
    """
    hash(func)  # make sure it is hashable. TODO is it necessary?

    def wrapper(*args, **kwargs):
        if func not in _MEMOIZED_NOARGS:
            res = func(*args, **kwargs)
            _MEMOIZED_NOARGS[func] = res
            return res
        return _MEMOIZED_NOARGS[func]
    return wrapper

# _GLOBAL_MEMOIZED_CACHE = dict()
# def global_memoized(func):
#     """ Make sure that the same `memoized` object is returned on different
#      calls to global_memoized(func)
#     """
#     ret = _GLOBAL_MEMOIZED_CACHE.get(func, None)
#     if ret is None:
#         ret = _GLOBAL_MEMOIZED_CACHE[func] = memoized(func)
#     return ret


def shape2d(a):
    """
    Ensure a 2D shape.

    Args:
        a: a int or tuple/list of length 2

    Returns:
        list: of length 2. if ``a`` is a int, return ``[a, a]``.
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))


def shape4d(a, data_format='NHWC'):
    """
    Ensuer a 4D shape, to use with 4D symbolic functions.

    Args:
        a: a int or tuple/list of length 2

    Returns:
        list: of length 4. if ``a`` is a int, return ``[1, a, a, 1]``
            or ``[1, 1, a, a]`` depending on data_format.
    """
    s2d = shape2d(a)
    if data_format == 'NHWC':
        return [1] + s2d + [1]
    else:
        return [1, 1] + s2d


@memoized
def log_once(message, func):
    """
    Log certain message only once. Call this function more than one times with
    the same message will result in no-op.

    Args:
        message(str): message to log
        func(str): the name of the logger method. e.g. "info", "warn", "error".
    """
    getattr(logger, func)(message)


_FUNC_CALLED = set()


def call_only_once(func):
    """
    Decorate a method or property of a class, so that this method can only
    be called once for every instance.
    Calling it more than once will result in exception.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # cannot use hasattr here, because hasattr tries to getattr, which
        # fails if func is a property
        assert func.__name__ in dir(self), "call_only_once can only be used on method or property!"

        cls = type(self)
        # cannot use ismethod(), because decorated method becomes a function
        is_method = inspect.isfunction(getattr(cls, func.__name__))
        key = (self, func)
        assert key not in _FUNC_CALLED, \
            "{} {}.{} can only be called once per object!".format(
                'Method' if is_method else 'Property',
                cls.__name__, func.__name__)
        _FUNC_CALLED.add(key)

        return func(*args, **kwargs)

    return wrapper


if __name__ == '__main__':
    class A():
        def __init__(self):
            self._p = 0

        @call_only_once
        def f(self, x):
            print(x)

        @property
        def p(self):
            return self._p

        @p.setter
        @call_only_once
        def p(self, val):
            self._p = val

    a = A()
    a.f(1)

    b = A()
    b.f(2)
    b.f(1)

    print(b.p)
    print(b.p)
    b.p = 2
    print(b.p)
    b.p = 3
    print(b.p)
