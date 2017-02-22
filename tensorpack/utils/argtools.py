#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: argtools.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import inspect
import six
from . import logger
if six.PY2:
    import functools32 as functools
else:
    import functools

__all__ = ['map_arg', 'memoized', 'shape2d', 'shape4d',
           'memoized_ignoreargs', 'log_once']


def map_arg(**maps):
    """
    Apply a mapping on certains argument before calling the original function.

    Args:
        maps (dict): {key: map_func}
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            argmap = inspect.getcallargs(func, *args, **kwargs)
            for k, map_func in six.iteritems(maps):
                if k in argmap:
                    argmap[k] = map_func(argmap[k])
            return func(**argmap)
        return wrapper
    return deco


memoized = functools.lru_cache(maxsize=None)
""" Equivalent to :func:`functools.lru_cache` """


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
