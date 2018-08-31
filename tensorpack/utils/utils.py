# -*- coding: utf-8 -*-
# File: utils.py


import os
import sys
from contextlib import contextmanager
import inspect
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np

from . import logger


__all__ = ['change_env',
           'get_rng',
           'fix_rng_seed',
           'get_tqdm',
           'execute_only_once',
           'humanize_time_delta'
           ]


def humanize_time_delta(sec):
    """Humanize timedelta given in seconds

    Args:
        sec (float): time difference in seconds. Must be positive.

    Returns:
        str - time difference as a readable string

    Example:

    .. code-block:: python

        print(humanize_time_delta(1))                                   # 1 second
        print(humanize_time_delta(60 + 1))                              # 1 minute 1 second
        print(humanize_time_delta(87.6))                                # 1 minute 27 seconds
        print(humanize_time_delta(0.01))                                # 0.01 seconds
        print(humanize_time_delta(60 * 60 + 1))                         # 1 hour 1 second
        print(humanize_time_delta(60 * 60 * 24 + 1))                    # 1 day 1 second
        print(humanize_time_delta(60 * 60 * 24 + 60 * 2 + 60*60*9 + 3)) # 1 day 9 hours 2 minutes 3 seconds
    """
    if sec < 0:
        logger.warn("humanize_time_delta() obtains negative seconds!")
        return "{:.3g} seconds".format(sec)
    if sec == 0:
        return "0 second"
    time = datetime(2000, 1, 1) + timedelta(seconds=int(sec))
    units = ['day', 'hour', 'minute', 'second']
    vals = [int(sec // 86400), time.hour, time.minute, time.second]
    if sec < 60:
        vals[-1] = sec

    def _format(v, u):
        return "{:.3g} {}{}".format(v, u, "s" if v > 1 else "")

    ans = []
    for v, u in zip(vals, units):
        if v > 0:
            ans.append(_format(v, u))
    return " ".join(ans)


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


_RNG_SEED = None


def fix_rng_seed(seed):
    """
    Call this function at the beginning of program to fix rng seed within tensorpack.

    Args:
        seed (int):

    Note:
        See https://github.com/tensorpack/tensorpack/issues/196.

    Example:

        Fix random seed in both tensorpack and tensorflow.

    .. code-block:: python

            import tensorpack.utils.utils as utils

            seed = 42
            utils.fix_rng_seed(seed)
            tesnorflow.set_random_seed(seed)
            # run trainer
    """
    global _RNG_SEED
    _RNG_SEED = int(seed)


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
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


_EXECUTE_HISTORY = set()


def execute_only_once():
    """
    Each called in the code to this function is guaranteed to return True the
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


def _pick_tqdm_interval(file):
    # Heuristics to pick a update interval for progress bar that's nice-looking for users.
    isatty = file.isatty()
    # Jupyter notebook should be recognized as tty.
    # Wait for https://github.com/ipython/ipykernel/issues/268
    try:
        from ipykernel import iostream
        if isinstance(file, iostream.OutStream):
            isatty = True
    except ImportError:
        pass

    if isatty:
        return 0.5
    else:
        # When run under mpirun/slurm, isatty is always False.
        # Here we apply some hacky heuristics for slurm.
        if 'SLURM_JOB_ID' in os.environ:
            if int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) > 1:
                # multi-machine job, probably not interactive
                return 60
            else:
                # possibly interactive, so let's be conservative
                return 15

        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            if int(os.environ['OMPI_COMM_WORLD_SIZE']) > 1:
                return 60

        # If not a tty, don't refresh progress bar that often
        return 180


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

    try:
        # Use this env var to override the refresh interval setting
        interval = float(os.environ['TENSORPACK_PROGRESS_REFRESH'])
    except KeyError:
        interval = _pick_tqdm_interval(kwargs.get('file', sys.stderr))

    default['mininterval'] = interval
    default.update(kwargs)
    return default


def get_tqdm(*args, **kwargs):
    """ Similar to :func:`tqdm.tqdm()`,
    but use tensorpack's default options to have consistent style. """
    return tqdm(*args, **get_tqdm_kwargs(**kwargs))
