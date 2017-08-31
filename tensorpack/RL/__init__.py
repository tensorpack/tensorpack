#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import iter_modules
import os
import os.path

__all__ = []


"""
This module should be removed in the future.
"""


def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]
        __all__.append(k)


for _, module_name, _ in iter_modules(
        [os.path.dirname(__file__)]):
    if not module_name.startswith('_'):
        _global_import(module_name)
