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
import logger


__all__ = ['change_env',
           'get_rng',
           'get_tqdm_kwargs',
           'get_tqdm',
           'execute_only_once',
           'building_rtfd',
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
    Get a good RNG.

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
        bool: whether this is the first time this function gets called from
            this line of code.

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


def deprecated(text, eos=None):
    """Log deprecation warning.
    Args:
        text (str, optional): information
        eos (str, optional): end of service date as tuple "YYYY-MM-DD"

    Example:
        @deprecated("Explanation what to do instead.", "2017-11-4")
        def foo(...):
            pass

        deprecated("deprecated_item")("This is an info about an alternative.", "2017-11-4")
        deprecated("A sentence about a remark.", "2017-11-4")()
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

    def deprecated_inner(func, fix=None):
        end_of_service = ""
        if fix:
            end_of_service = " after " + datetime(*map(int, fix.split("-"))).strftime("%d %b")
        elif eos:
            end_of_service = " after " + datetime(*map(int, eos.split("-"))).strftime("%d %b")
        # function-decorator
        if callable(func):
            @functools.wraps(func)
            def new_func(*args, **kwargs):
                warn_msg = "%s is deprecated%s [%s]."
                warn_msg = warn_msg % (func.__name__, end_of_service, get_location())
                if text:
                    warn_msg += " %s" % text
                logger.warn(warn_msg)
                return func(*args, **kwargs)
            return new_func
        else:
            # now func, text = text, func
            if len(func) > 0:
                warn_msg = "%s is deprecated%s [%s]. %s"
                warn_msg = warn_msg % (text, end_of_service, get_location(), func)
            else:
                warn_msg = "%s Legacy periods ends%s "
                warn_msg = warn_msg % (text, end_of_service)
            logger.warn(warn_msg)

    return deprecated_inner
