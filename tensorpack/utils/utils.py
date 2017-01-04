# -*- coding: UTF-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import sys
from contextlib import contextmanager
import inspect
from datetime import datetime
from tqdm import tqdm
import numpy as np

__all__ = ['change_env',
           'get_rng',
           'get_dataset_path',
           'get_tqdm_kwargs',
           'get_tqdm',
           'execute_only_once',
           'building_rtfd'
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


def get_rng(obj=None):
    """ obj: some object to use to generate random seed"""
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)


_EXECUTE_HISTORY = set()


def execute_only_once():
    """
    when called with:
        if execute_only_once():
            # do something
    The body is guranteed to be executed only the first time.
    """
    f = inspect.currentframe().f_back
    ident = (f.f_code.co_filename, f.f_lineno)
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


def get_tqdm(**kwargs):
    return tqdm(**get_tqdm_kwargs(**kwargs))


def building_rtfd():
    return os.environ.get('READTHEDOCS') == 'True' \
        or os.environ.get('TENSORPACK_DOC_BUILDING')
