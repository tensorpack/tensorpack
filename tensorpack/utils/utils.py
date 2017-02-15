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
import functools
from . import logger


__all__ = ['change_env',
           'get_rng',
           'get_tqdm_kwargs',
           'get_tqdm',
           'execute_only_once',
           'building_rtfd',
           'log_deprecated',
           'deprecated'
           ]


@contextmanager
def change_env(name, val):
    """
    Args:
        name(str), val(str):

    Returns:
        a context where the environment variable ``name`` being set to
        ``val``. It will be set back after the context exits.
    """
    oldval = os.environ.get(name, None)
    os.environ[name] = val
    yield
    if oldval is None:
        del os.environ[name]
    else:
        os.environ[name] = oldval


def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)


_EXECUTE_HISTORY = set()


def execute_only_once():
    """
    Each called in the code to this function is guranteed to return True the
    first time and False afterwards.

    Returns:
        bool: whether this is the first time this function gets called from this line of code.

    Example:
        .. code-block:: python

            if execute_only_once():
                # do something only once
    """
    f = inspect.currentframe().f_back
    ident = (f.f_code.co_filename, f.f_lineno)
    if ident in _EXECUTE_HISTORY:
        return False
    _EXECUTE_HISTORY.add(ident)
    return True


def get_tqdm_kwargs(**kwargs):
    """
    Return default arguments to be used with tqdm.

    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
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
    """ Similar to :func:`get_tqdm_kwargs`, but returns the tqdm object
    directly. """
    return tqdm(**get_tqdm_kwargs(**kwargs))


def building_rtfd():
    """
    Returns:
        bool: if tensorpack is being imported to generate docs now.
    """
    return os.environ.get('READTHEDOCS') == 'True' \
        or os.environ.get('TENSORPACK_DOC_BUILDING')


def log_deprecated(name="", text="", eos=""):
    """
    Log deprecation warning.

    Args:
        name (str): name of the deprecated item.
        text (str, optional): information about the deprecation.
        eos (str, optional): end of service date such as "YYYY-MM-DD".
    """
    assert name or text
    if eos:
        eos = "after " + datetime(*map(int, eos.split("-"))).strftime("%d %b")
    if name:
        if eos:
            warn_msg = "%s will be deprecated %s. %s" % (name, eos, text)
        else:
            warn_msg = "%s was deprecated. %s" % (name, text)
    else:
        warn_msg = text
        if eos:
            warn_msg += " Legacy period ends %s" % eos
    logger.warn("[Deprecated] " + warn_msg)


def deprecated(text="", eos=""):
    """
    Args:
        text, eos: same as :func:`log_deprecated`.

    Returns:
        a decorator which deprecates the function.

    Example:
        .. code-block:: python

            @deprecated("Explanation of what to do instead.", "2017-11-4")
            def foo(...):
                pass
    """

    def get_location():
        import inspect
        frame = inspect.currentframe()
        if frame:
            callstack = inspect.getouterframes(frame)[-1]
            return '%s:%i' % (callstack[1], callstack[2])
        else:
            stack = inspect.stack(0)
            entry = stack[2]
            return '%s:%i' % (entry[1], entry[2])

    def deprecated_inner(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            name = "{} [{}]".format(func.__name__, get_location())
            log_deprecated(name, text, eos)
            return func(*args, **kwargs)
        return new_func
    return deprecated_inner
