#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

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


_CURR_DIR = os.path.dirname(__file__)
for _, module_name, _ in iter_modules(
        [_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
__all__.extend([
    'logger',
    'get_nr_gpu', 'change_gpu', 'get_rng'])
