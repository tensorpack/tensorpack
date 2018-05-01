#  -*- coding: utf-8 -*-
#  File: __init__.py


from pkgutil import iter_modules
import os

"""
Common utils.
These utils should be irrelevant to tensorflow.
"""

__all__ = []


# this two functions for back-compat only
def get_nr_gpu():
    from .gpu import get_nr_gpu as gg
    logger.warn(    # noqa
        "get_nr_gpu will not be automatically imported any more! "
        "Please do `from tensorpack.utils.gpu import get_nr_gpu`")
    return gg()


def change_gpu(val):
    from .gpu import change_gpu as cg
    logger.warn(    # noqa
        "change_gpu will not be automatically imported any more! "
        "Please do `from tensorpack.utils.gpu import change_gpu`")
    return cg(val)


def get_rng(obj=None):
    from .utils import get_rng as gr
    logger.warn(    # noqa
        "get_rng will not be automatically imported any more! "
        "Please do `from tensorpack.utils.utils import get_rng`")
    return gr(obj)

# Import no submodules. they are supposed to be explicitly imported by users.
__all__.extend(['logger', 'get_nr_gpu', 'change_gpu', 'get_rng'])
