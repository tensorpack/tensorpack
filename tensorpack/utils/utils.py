# -*- coding: UTF-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os, sys
from contextlib import contextmanager
import inspect, functools
from datetime import datetime
import time
import collections
import numpy as np
import six

__all__ = ['change_env',
        'map_arg',
        'get_rng', 'memoized',
        'get_dataset_path',
        'get_tqdm_kwargs',
        'execute_only_once'
        ]

@contextmanager
def change_env(name, val):
    oldval = os.environ.get(name, None)
    os.environ[name] = val
    yield
    if oldval is None:
        del os.environ[name]
    else:
        os.environ[name] = oldval

class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
           return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


#_GLOBAL_MEMOIZED_CACHE = dict()
#def global_memoized(func):
    #""" Make sure that the same `memoized` object is returned on different
        #calls to global_memoized(func)
    #"""
    #ret = _GLOBAL_MEMOIZED_CACHE.get(func, None)
    #if ret is None:
        #ret = _GLOBAL_MEMOIZED_CACHE[func] = memoized(func)
    #return ret

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

def get_rng(obj=None):
    """ obj: some object to use to generate random seed"""
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)


_EXECUTE_HISTORY = set()
def execute_only_once():
    f = inspect.currentframe().f_back
    ident =  (f.f_code.co_filename, f.f_lineno)
    if ident in _EXECUTE_HISTORY:
        return False
    _EXECUTE_HISTORY.add(ident)
    return True

def get_dataset_path(*args):
    d = os.environ.get('TENSORPACK_DATASET', None)
    if d is None:
        d = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), '..', 'dataflow', 'dataset'))
        if execute_only_once():
            from . import logger
            logger.info("TENSORPACK_DATASET not set, using {} for dataset.".format(d))
    assert os.path.isdir(d), d
    return os.path.join(d, *args)


def get_tqdm_kwargs(**kwargs):
    default = dict(
            smoothing=0.5,
            dynamic_ncols=True,
            ascii=True,
            bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
            )
    f = kwargs.get('file', sys.stderr)
    if f.isatty():
        default['mininterval'] = 0.5
    else:
        default['mininterval'] = 60
    default.update(kwargs)
    return default
