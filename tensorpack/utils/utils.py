# -*- coding: UTF-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os, sys
from contextlib import contextmanager
from datetime import datetime
import time
import collections
import numpy as np

from . import logger

__all__ = ['change_env',
        'get_rng', 'memoized',
        'get_nr_gpu',
        'get_gpus',
        'get_dataset_dir']

#def expand_dim_if_necessary(var, dp):
#    """
#    Args:
#        var: a tensor
#        dp: a numpy array
#    Return a reshaped version of dp, if that makes it match the valid dimension of var
#    """
#    shape = var.get_shape().as_list()
#    valid_shape = [k for k in shape if k]
#    if dp.shape == tuple(valid_shape):
#        new_shape = [k if k else 1 for k in shape]
#        dp = dp.reshape(new_shape)
#    return dp

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

def get_rng(self):
    seed = (id(self) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)

def get_nr_gpu():
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    assert env is not None  # TODO
    return len(env.split(','))

def get_gpus():
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    assert env is not None  # TODO
    return map(int, env.strip().split(','))

def get_dataset_dir(name):
    d = os.environ.get('TENSORPACK_DATASET', None)
    if d:
        assert os.path.isdir(d)
    else:
        d = os.path.join(os.path.dirname(__file__), '..', 'dataflow', 'dataset')
        logger.info("TENSORPACK_DATASET not set, using {} to keep dataset.".format(d))
    return os.path.join(d, name)

