#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: argtools.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import operator
import inspect
import six
import functools
import collections
from . import logger

__all__ = ['map_arg', 'memoized', 'shape2d', 'memoized_ignoreargs', 'log_once']


def map_arg(**maps):
    """
    Apply a mapping on certains argument before calling original function.
    maps: {key: map_func}
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


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        kwlist = tuple(sorted(list(kwargs), key=operator.itemgetter(0)))
        if not isinstance(args, collections.Hashable) or \
                not isinstance(kwlist, collections.Hashable):
            logger.warn("Arguments to memoized call is unhashable!")
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args, **kwargs)
        key = (args, kwlist)
        if key in self.cache:
            return self.cache[key]
        else:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


_MEMOIZED_NOARGS = {}


def memoized_ignoreargs(func):
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
    a: a int or tuple/list of length 2
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))


@memoized
def log_once(message, func):
    getattr(logger, func)(message)
