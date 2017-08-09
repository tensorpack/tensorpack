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


def _global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        globals()[k] = p.__dict__[k]
        __all__.append(k)


_TO_IMPORT = set([
    'naming',
    'utils',
])


# this two functions for back-compat only
def get_nr_gpu():
    from .gpu import get_nr_gpu
    logger.warn(    # noqa
        "get_nr_gpu will not be automatically imported any more! "
        "Please do `from tensorpack.utils.gpu import get_nr_gpu`")
    return get_nr_gpu()


def change_gpu(val):
    from .gpu import change_gpu as cg
    logger.warn(    # noqa
        "change_gpu will not be automatically imported any more! "
        "Please do `from tensorpack.utils.gpu import change_gpu`")
    return cg(val)


_CURR_DIR = os.path.dirname(__file__)
for _, module_name, _ in iter_modules(
        [_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
    if module_name in _TO_IMPORT:
        _global_import(module_name)
__all__.extend([
    'logger',
    'get_nr_gpu', 'change_gpu'])
